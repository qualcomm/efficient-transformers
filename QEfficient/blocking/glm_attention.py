# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""GLM dense-MLA and DSA attention strategies used by generic blocking."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from QEfficient.blocking.blocked_attention_forwards import blocked_kv_mla_attention_forward
from QEfficient.transformers.cache_utils import glm_dsa_gather_cache
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def blocked_glm_dsa_topk(
    query: torch.Tensor,
    head_weights: torch.Tensor,
    folded_key_cache: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    *,
    scale: float,
    dp: int,
    cp: int,
    num_blocks: int,
    num_cores_per_device: int,
    tokens_per_core: int,
    block_topk: int,
    final_topk: int,
) -> torch.Tensor:
    """Select global DSA indices without materializing a logical ``[B, T, D]`` cache."""
    batch_size, query_length, num_heads, head_dim = query.shape
    batch_local = batch_size // dp
    local_context = folded_key_cache.shape[2]
    block_width = local_context // num_blocks
    keys = folded_key_cache.view(
        batch_local,
        dp,
        cp,
        num_blocks,
        num_cores_per_device,
        tokens_per_core,
        head_dim,
    )
    query = query.view(batch_local, dp, query_length, num_heads, head_dim).float()
    head_weights = head_weights.view(batch_local, dp, query_length, num_heads).float()
    position_ids = position_ids.view(batch_local, dp, query_length)
    if attention_mask is not None:
        attention_mask = attention_mask.view(batch_local, dp, query_length, -1)

    cp_ids = torch.arange(cp, device=query.device, dtype=torch.int64).view(1, 1, 1, cp, 1)
    candidate_scores = []
    candidate_indices = []
    for block_idx in range(num_blocks):
        block_start = block_idx * block_width
        block_keys = keys[:, :, :, block_idx].reshape(batch_local, dp, cp, block_width, head_dim).float()
        head_scores = torch.einsum("bpshd,bpcwd->bpschw", query, block_keys) * scale
        scores = (F.relu(head_scores) * head_weights.unsqueeze(3).unsqueeze(-1)).sum(dim=-2)

        local_ids = torch.arange(
            block_start,
            block_start + block_width,
            device=query.device,
            dtype=torch.int64,
        ).view(1, 1, 1, 1, block_width)
        global_ids = local_ids * cp + cp_ids
        invalid = global_ids > position_ids.unsqueeze(-1).unsqueeze(-1)
        if attention_mask is not None:
            block_mask = torch.index_select(attention_mask, -1, global_ids.reshape(-1)).view(
                batch_local, dp, query_length, cp, block_width
            )
            invalid = invalid | block_mask
        scores = scores.masked_fill(invalid, float("-inf"))

        selected = torch.topk(scores, k=block_topk, dim=-1)
        candidate_scores.append(selected.values.reshape(batch_local, dp, query_length, cp * block_topk))
        candidate_indices.append(
            torch.gather(
                global_ids.expand(batch_local, dp, query_length, cp, block_width),
                -1,
                selected.indices,
            ).reshape(batch_local, dp, query_length, cp * block_topk)
        )

    candidate_scores = torch.cat(candidate_scores, dim=-1)
    candidate_indices = torch.cat(candidate_indices, dim=-1)
    selected = torch.topk(candidate_scores, k=final_topk, dim=-1).indices
    return torch.gather(candidate_indices, -1, selected).reshape(batch_size, query_length, final_topk).to(torch.int32)


def _trim_live_context(
    position_ids: torch.Tensor | None, attention_mask: torch.Tensor | None, *states: torch.Tensor
) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
    if position_ids is None or torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
        return attention_mask, states
    live_context = int(position_ids.max().item()) + 1
    states = tuple(state[..., :live_context, :] for state in states)
    if attention_mask is not None:
        attention_mask = attention_mask[..., :live_context]
    return attention_mask, states


def _expand_kv(module, compressed_kv: torch.Tensor, key_rope: torch.Tensor):
    batch_size, _, seq_len, _ = compressed_kv.shape
    projected = module.kv_b_proj(compressed_kv[:, 0])
    projected = projected.view(batch_size, seq_len, module.num_heads, module.qk_nope_head_dim + module.v_head_dim)
    projected = projected.transpose(1, 2)
    key_nope, values = torch.split(projected, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)
    key_rope = key_rope[:, 0].unsqueeze(1).expand(-1, module.num_heads, -1, -1)
    return torch.cat((key_nope, key_rope), dim=-1), values


def _pad_sequence_to_split(tensor: torch.Tensor, split: int) -> tuple[torch.Tensor, int]:
    sequence_length = tensor.shape[-2]
    padding = (-sequence_length) % split
    if padding:
        tensor = F.pad(tensor, (0, 0, 0, padding))
    return tensor, padding


def _dense_parallel_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    q_len: int,
    n_rep: int,
    group_heads: int,
    split: int,
    t_orig: int,
    t_h: int,
    start_index: int,
    position_ids: torch.Tensor,
    prefill: bool,
) -> torch.Tensor:
    if attention_mask is None:
        block_mask = (
            torch.arange(
                start_index,
                start_index + t_orig,
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).view(1, 1, 1, t_orig)
            > position_ids[:, None, :, None]
        )
    else:
        block_mask = attention_mask[..., start_index : start_index + t_orig].to(torch.bool)
    if t_orig != split * t_h:
        block_mask = F.pad(block_mask, (0, split * t_h - t_orig), value=True)
    block_mask = block_mask.view(batch_size, 1, q_len, split, t_h)
    if prefill:
        return block_mask.permute(0, 1, 3, 2, 4).unsqueeze(3).expand(batch_size, group_heads, split, n_rep, q_len, t_h)
    return (
        block_mask.permute(0, 1, 3, 2, 4)
        .reshape(batch_size, 1, split, q_len, t_h)
        .expand(batch_size, group_heads, split, q_len * n_rep, t_h)
    )


def _prepare_parallel_kv(
    module,
    query: torch.Tensor,
    compressed_kv_block: torch.Tensor,
    k_pe_block: torch.Tensor,
    per_head_k_up_normal: torch.Tensor,
    *,
    absorption: bool,
    split: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    batch_size, num_query_heads, _, query_width = query.shape
    kv_lora_rank = module.config.kv_lora_rank
    physical_kv_heads = compressed_kv_block.shape[1]
    if absorption:
        group_heads = physical_kv_heads
        n_rep = num_query_heads // group_heads
        key_block = torch.cat((compressed_kv_block, k_pe_block), dim=-1)
        value_block = compressed_kv_block
    else:
        group_heads = num_query_heads
        n_rep = 1
        repeat = math.ceil(num_query_heads / physical_kv_heads)
        ckv_heads = (
            compressed_kv_block.unsqueeze(2)
            .expand(-1, -1, repeat, -1, -1)
            .reshape(batch_size, physical_kv_heads * repeat, -1, kv_lora_rank)[:, :num_query_heads]
        )
        rope_heads = (
            k_pe_block.unsqueeze(2)
            .expand(-1, -1, repeat, -1, -1)
            .reshape(batch_size, physical_kv_heads * repeat, -1, module.config.qk_rope_head_dim)[:, :num_query_heads]
        )
        key_nope = torch.matmul(ckv_heads, per_head_k_up_normal)
        key_block = torch.cat((key_nope, rope_heads), dim=-1)
        value_block = ckv_heads

    key_block, _ = _pad_sequence_to_split(key_block, split)
    value_block, _ = _pad_sequence_to_split(value_block, split)
    t_h = key_block.shape[-2] // split
    key_block = key_block.view(batch_size, group_heads, split, t_h, query_width)
    value_block = value_block.view(batch_size, group_heads, split, t_h, kv_lora_rank)
    return key_block, value_block, group_heads, n_rep, t_h


def _merge_block_split_outputs(max_stack: torch.Tensor, sum_stack: torch.Tensor, out_stack: torch.Tensor):
    block_max = max_stack.max(dim=0).values
    block_scale = torch.exp(max_stack - block_max.unsqueeze(0))
    block_sum = (block_scale * sum_stack).sum(dim=0)
    block_out = (block_scale.unsqueeze(-1) * out_stack).sum(dim=0)

    split_max = block_max.max(dim=2).values
    split_scale = torch.exp(block_max - split_max.unsqueeze(2))
    denominator = (split_scale * block_sum).sum(dim=2)
    numerator = (split_scale.unsqueeze(-1) * block_out).sum(dim=2)
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny).unsqueeze(-1)


def _glm_parallel_mla_attention(
    *,
    module,
    query: torch.Tensor,
    per_head_k_up_normal: torch.Tensor,
    per_head_v_up: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_kv_blocks: int,
    par_num_split: int,
    cache_kwargs: dict[str, Any],
    layer_idx: int,
    compressed_kvs,
    absorption: bool,
    prefill: bool,
    online: bool = False,
) -> tuple[torch.Tensor, None]:
    batch_size, num_query_heads, q_len, _ = query.shape
    split = int(par_num_split)
    ctx_len = compressed_kvs.layers[layer_idx].ckv.shape[2]
    kv_block_size = -(-ctx_len // int(num_kv_blocks))
    position_ids = cache_kwargs["position_ids"]
    max_blocks = []
    sum_blocks = []
    out_blocks = []

    running_max = running_sum = running_out = None
    for block_idx in range(int(num_kv_blocks)):
        start_index = block_idx * kv_block_size
        block_len = ctx_len - start_index if block_idx == int(num_kv_blocks) - 1 else kv_block_size
        end_index = start_index + block_len
        compressed_block = compressed_kvs.read_only_blocked_ckv(start_index, end_index, layer_idx, cache_kwargs)
        rope_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, layer_idx, cache_kwargs)
        key_block, value_block, group_heads, n_rep, t_h = _prepare_parallel_kv(
            module,
            query,
            compressed_block,
            rope_block,
            per_head_k_up_normal,
            absorption=absorption,
            split=split,
        )
        if prefill:
            q_fold = query.reshape(batch_size, group_heads, n_rep, q_len, query.shape[-1])
            q_split = q_fold.unsqueeze(2).expand(batch_size, group_heads, split, n_rep, q_len, query.shape[-1])
            scores = torch.matmul(q_split, key_block.unsqueeze(3).transpose(-1, -2)) * scaling
        else:
            q_fold = query.reshape(batch_size, group_heads, q_len * n_rep, query.shape[-1])
            q_split = q_fold.unsqueeze(2).expand(batch_size, group_heads, split, q_len * n_rep, query.shape[-1])
            scores = torch.matmul(q_split, key_block.transpose(-1, -2)) * scaling
        mask = _dense_parallel_mask(
            attention_mask,
            batch_size=batch_size,
            q_len=q_len,
            n_rep=n_rep,
            group_heads=group_heads,
            split=split,
            t_orig=block_len,
            t_h=t_h,
            start_index=start_index,
            position_ids=position_ids,
            prefill=prefill,
        )
        scores = scores.masked_fill(mask, -3.0e4)
        block_max = scores.max(dim=-1).values
        exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
        exp_scores = torch.where(mask, torch.zeros_like(exp_scores), exp_scores)
        block_sum = exp_scores.sum(dim=-1)
        value_view = value_block.unsqueeze(3) if prefill else value_block
        block_out = torch.matmul(exp_scores, value_view)

        if online:
            if running_max is None:
                running_max, running_sum, running_out = block_max, block_sum, block_out
            else:
                new_max = torch.maximum(running_max, block_max)
                old_scale = torch.exp(running_max - new_max)
                new_scale = torch.exp(block_max - new_max)
                running_sum = old_scale * running_sum + new_scale * block_sum
                running_out = old_scale.unsqueeze(-1) * running_out + new_scale.unsqueeze(-1) * block_out
                running_max = new_max
        else:
            max_blocks.append(block_max)
            sum_blocks.append(block_sum)
            out_blocks.append(block_out)

    if online:
        output = _merge_block_split_outputs(
            running_max.unsqueeze(0),
            running_sum.unsqueeze(0),
            running_out.unsqueeze(0),
        )
    else:
        output = _merge_block_split_outputs(torch.stack(max_blocks), torch.stack(sum_blocks), torch.stack(out_blocks))

    if prefill:
        output = output.reshape(batch_size, num_query_heads, q_len, module.config.kv_lora_rank)
    else:
        output = output.view(batch_size, group_heads, n_rep, q_len, module.config.kv_lora_rank).reshape(
            batch_size, num_query_heads, q_len, module.config.kv_lora_rank
        )
    return torch.matmul(output, per_head_v_up).transpose(1, 2).contiguous(), None


def _glm_tiled_sparse_mla_attention(
    *,
    module,
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    sparse_ckv: torch.Tensor,
    sparse_rope: torch.Tensor,
    row_valid: torch.Tensor,
    layer_config,
) -> tuple[torch.Tensor, None]:
    batch_size, num_query_heads, q_len, kv_lora_rank = query_latent.shape
    batch_local = batch_size // layer_config.attn_dp
    rows = layer_config.attn_dp * layer_config.attn_cp
    selected_per_row = sparse_ckv.shape[3]
    cores = layer_config.num_cores_per_device
    if selected_per_row % cores:
        raise ValueError("GLM DSA Top-K rows must be divisible by num_cores_per_device.")
    tokens_per_core = selected_per_row // cores
    rope_dim = query_rope.shape[-1]

    q_latent = query_latent.view(batch_local, layer_config.attn_dp, num_query_heads, q_len, kv_lora_rank).float()
    q_latent = q_latent.unsqueeze(2).expand(
        batch_local, layer_config.attn_dp, layer_config.attn_cp, num_query_heads, q_len, kv_lora_rank
    )
    q_rope = query_rope.view(batch_local, layer_config.attn_dp, num_query_heads, q_len, rope_dim).float()
    q_rope = q_rope.unsqueeze(2).expand(
        batch_local, layer_config.attn_dp, layer_config.attn_cp, num_query_heads, q_len, rope_dim
    )
    query = torch.cat((q_latent, q_rope), dim=-1).reshape(
        batch_local, rows, num_query_heads * q_len, kv_lora_rank + rope_dim
    )
    ckv_core = sparse_ckv.reshape(batch_local, rows, cores, tokens_per_core, kv_lora_rank).float()
    rope_core = sparse_rope.reshape(batch_local, rows, cores, tokens_per_core, rope_dim).float()
    valid_core = row_valid.reshape(batch_local, rows, cores, tokens_per_core)
    key_core = torch.cat((ckv_core, rope_core), dim=-1)

    scores = torch.matmul(query.unsqueeze(2), key_core.transpose(-1, -2)) * module.scaling
    valid_mask = valid_core.unsqueeze(3)
    scores = scores.masked_fill(~valid_mask, -3.0e4)
    local_max = scores.max(dim=-1).values
    exp_scores = torch.exp(scores - local_max.unsqueeze(-1))
    exp_scores = torch.where(valid_mask, exp_scores, torch.zeros_like(exp_scores))
    local_sum = exp_scores.sum(dim=-1)
    local_out = torch.matmul(exp_scores, ckv_core)

    reduce_width = layer_config.attn_cp * cores
    local_max = local_max.reshape(batch_local, layer_config.attn_dp, reduce_width, num_query_heads * q_len)
    local_sum = local_sum.reshape(batch_local, layer_config.attn_dp, reduce_width, num_query_heads * q_len)
    local_out = local_out.reshape(
        batch_local, layer_config.attn_dp, reduce_width, num_query_heads * q_len, kv_lora_rank
    )
    global_max = local_max.max(dim=2).values
    weights = torch.exp(local_max - global_max.unsqueeze(2))
    denominator = (weights * local_sum).sum(dim=2)
    numerator = (weights.unsqueeze(-1) * local_out).sum(dim=2)
    output = numerator / denominator.unsqueeze(-1)
    output = output.reshape(batch_size, num_query_heads, q_len, kv_lora_rank).to(query_latent.dtype)
    return torch.matmul(output, module.per_head_v_up[0]).transpose(1, 2).contiguous(), None


def _project_inputs(module, hidden_states, position_embeddings):
    batch_size, seq_len = hidden_states.shape[:2]
    q_resid = module.q_a_layernorm(module.q_a_proj(hidden_states))
    q_nope = torch.matmul(q_resid, module.q_up)
    q_nope = q_nope.view(batch_size, seq_len, module.num_heads, module.qk_nope_head_dim).transpose(1, 2)
    q_rope = torch.matmul(q_resid, module.q_rope)
    q_rope = q_rope.view(batch_size, seq_len, module.num_heads, module.qk_rope_head_dim).transpose(1, 2)

    projected_kv = module.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, key_rope = torch.split(projected_kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
    compressed_kv = module.kv_a_layernorm(compressed_kv).view(batch_size, 1, seq_len, module.kv_lora_rank)
    key_rope = key_rope.view(batch_size, 1, seq_len, module.qk_rope_head_dim)
    cos, sin = position_embeddings
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave

    q_rope, key_rope = apply_rotary_pos_emb_interleave(q_rope, key_rope, cos, sin)
    return q_resid, q_nope, q_rope, compressed_kv, key_rope


def _dense_mla_attention(
    module,
    q_resid,
    q_nope,
    q_rope,
    compressed_kv,
    key_rope,
    attention_mask,
    position_ids,
    batch_index,
    cache,
    layer_config,
):
    cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
    if cache is not None:
        if layer_config.blocking_mode == "none":
            compressed_kv = cache.update_ckv(compressed_kv, module.layer_idx, cache_kwargs)
            key_rope = cache.update_k_pe(key_rope, module.layer_idx, cache_kwargs)
            attention_mask, (compressed_kv, key_rope) = _trim_live_context(
                position_ids, attention_mask, compressed_kv, key_rope
            )
        else:
            cache.write_only_ckv(compressed_kv, module.layer_idx, cache_kwargs)
            cache.write_only_k_pe(key_rope, module.layer_idx, cache_kwargs)

    if layer_config.absorption:
        if layer_config.online:
            fused_qk = torch.matmul(module.per_head_q_up, module.per_head_k_up)
            query_latent = torch.einsum("bsd,hdk->bhsk", q_resid, fused_qk[0])
        else:
            query_latent = torch.einsum("bsd,hdk->bhsk", q_resid, module.fusedqk[0])
        query = torch.cat((query_latent, q_rope), dim=-1)
    else:
        query = torch.cat((q_nope, q_rope), dim=-1)

    if layer_config.blocking_mode != "none" and cache is not None:
        if layer_config.blocking_mode in {"par", "prefill_par", "prefill_par_online"}:
            output, weights = _glm_parallel_mla_attention(
                module=module,
                query=query,
                per_head_k_up_normal=module.per_head_k_up_normal,
                per_head_v_up=module.per_head_v_up,
                attention_mask=attention_mask,
                scaling=module.scaling,
                num_kv_blocks=layer_config.num_kv_blocks,
                par_num_split=layer_config.par_num_split,
                cache_kwargs=cache_kwargs,
                layer_idx=module.layer_idx,
                compressed_kvs=cache,
                absorption=layer_config.absorption,
                prefill=layer_config.blocking_mode in {"prefill_par", "prefill_par_online"},
                online=layer_config.blocking_mode == "prefill_par_online",
            )
        else:
            output, weights = blocked_kv_mla_attention_forward(
                module=module,
                query=query,
                per_head_k_up_normal=module.per_head_k_up_normal,
                per_head_v_up=module.per_head_v_up,
                attention_mask=attention_mask,
                scaling=module.scaling,
                num_kv_blocks=layer_config.num_kv_blocks,
                cache_kwargs=cache_kwargs,
                layer_idx=module.layer_idx,
                compressed_kvs=cache,
                mla_absorption={"absorption": layer_config.absorption, "online": layer_config.online},
                blocking_config=getattr(module, "attn_blocking_config", None),
                position_ids=position_ids,
            )
        return output, weights

    if layer_config.absorption:
        keys = torch.cat((compressed_kv, key_rope), dim=-1)
        weights = torch.einsum("bhsd,btd->bhst", query, keys[:, 0]) * module.scaling
        if attention_mask is not None:
            weights = torch.where(attention_mask, torch.full_like(weights, MIN_MASKED_ATTENTION_VALUE), weights)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        latent_output = torch.einsum("bhst,btk->bhsk", weights, compressed_kv[:, 0])
        output = torch.einsum("bhsk,hkd->bhsd", latent_output, module.per_head_v_up[0])
    else:
        keys, values = _expand_kv(module, compressed_kv, key_rope)
        weights = torch.einsum("bhsd,bhtd->bhst", query, keys) * module.scaling
        if attention_mask is not None:
            weights = torch.where(attention_mask, torch.full_like(weights, MIN_MASKED_ATTENTION_VALUE), weights)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.einsum("bhst,bhtd->bhsd", weights, values)
    return output.transpose(1, 2).contiguous(), weights


def glm_attention_strategy(
    *,
    module,
    attention_mask=None,
    position_ids=None,
    batch_index=None,
    auxiliary_state=None,
    **kwargs: Any,
):
    """Execute the layer-specific GLM strategy selected by the blocking transform."""
    del kwargs
    hidden_states = auxiliary_state["hidden_states"]
    position_embeddings = auxiliary_state["position_embeddings"]
    compressed_kvs = auxiliary_state.get("compressed_kvs")
    indexer_key_cache = auxiliary_state.get("indexer_key_cache")
    previous_topk = auxiliary_state.get("prev_topk_indices")
    layer_config = module.glm_attention_config

    q_resid, q_nope, q_rope, compressed_kv, key_rope = _project_inputs(module, hidden_states, position_embeddings)
    if layer_config.attention_type == "dense_mla":
        output, weights = _dense_mla_attention(
            module,
            q_resid,
            q_nope,
            q_rope,
            compressed_kv,
            key_rope,
            attention_mask,
            position_ids,
            batch_index,
            compressed_kvs,
            layer_config,
        )
        output = output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return module.o_proj(output), weights, previous_topk

    cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
    folded_dsa = compressed_kvs is not None and compressed_kvs.layers[module.layer_idx].is_folded_dsa
    if compressed_kvs is not None and not folded_dsa:
        compressed_kv = compressed_kvs.update_ckv(compressed_kv, module.layer_idx, cache_kwargs)
        key_rope = compressed_kvs.update_k_pe(key_rope, module.layer_idx, cache_kwargs)
        attention_mask, (compressed_kv, key_rope) = _trim_live_context(
            position_ids, attention_mask, compressed_kv, key_rope
        )
    if layer_config.indexer_type == "full":
        if module.indexer is None:
            raise ValueError(f"GLM DSA layer {module.layer_idx} is configured with a full indexer but has no indexer.")
        topk_indices = module.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            attention_mask[:, 0],
            position_ids,
            indexer_key_cache=indexer_key_cache,
        )
    else:
        if previous_topk is None:
            raise ValueError("Shared DSA layers require Top-K indices from a previous full-indexer layer.")
        topk_indices = previous_topk
    if folded_dsa:
        if topk_indices.shape[1] != 1:
            raise ValueError("Folded GLM DSA attention currently supports single-token decode only.")
        compressed_kvs.update_ckv(compressed_kv, module.layer_idx, cache_kwargs)
        compressed_kvs.update_k_pe(key_rope, module.layer_idx, cache_kwargs)
        valid_topk = topk_indices.to(position_ids.dtype) <= position_ids[:, -1:]
        decode_topk = topk_indices[:, 0]
        decode_valid = valid_topk[:, 0]
        cache_layer = compressed_kvs.layers[module.layer_idx]
        sparse_ckv, row_valid = glm_dsa_gather_cache(
            cache_layer.ckv,
            decode_topk,
            decode_valid,
            dp=layer_config.attn_dp,
            cp=layer_config.attn_cp,
        )
        sparse_rope, _ = glm_dsa_gather_cache(
            cache_layer.k_pe,
            decode_topk,
            decode_valid,
            dp=layer_config.attn_dp,
            cp=layer_config.attn_cp,
        )
        query_latent = torch.einsum("bsd,hdk->bhsk", q_resid, module.fusedqk[0])
        output, weights = _glm_tiled_sparse_mla_attention(
            module=module,
            query_latent=query_latent,
            query_rope=q_rope,
            sparse_ckv=sparse_ckv,
            sparse_rope=sparse_rope,
            row_valid=row_valid,
            layer_config=layer_config,
        )
        output = output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return module.o_proj(output), weights, topk_indices

    query = torch.cat((q_nope, q_rope), dim=-1)
    keys, values = _expand_kv(module, compressed_kv, key_rope)
    sparse_mask = torch.ones(
        (hidden_states.shape[0], hidden_states.shape[1], keys.shape[2]),
        dtype=torch.bool,
        device=keys.device,
    )
    sparse_mask = sparse_mask.scatter(-1, topk_indices.long(), False).unsqueeze(1)
    weights = torch.einsum("bhsd,bhtd->bhst", query, keys) * module.scaling
    weights = torch.where(attention_mask | sparse_mask, torch.full_like(weights, MIN_MASKED_ATTENTION_VALUE), weights)
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    output = (
        torch.einsum("bhst,bhtd->bhsd", weights, values)
        .transpose(1, 2)
        .contiguous()
        .reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
    )
    return module.o_proj(output), weights, topk_indices
