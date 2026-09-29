# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""GLM dense-MLA and DSA attention strategies used by generic blocking."""

from __future__ import annotations

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
        batch_local = hidden_states.shape[0] // layer_config.attn_dp
        sparse_ckv = sparse_ckv.reshape(hidden_states.shape[0], layer_config.attn_cp * topk_indices.shape[-1], -1)
        sparse_rope = sparse_rope.reshape(hidden_states.shape[0], layer_config.attn_cp * topk_indices.shape[-1], -1)
        valid = row_valid.reshape(batch_local, layer_config.attn_dp, -1).reshape(hidden_states.shape[0], -1)
        query_latent = torch.einsum("bsd,hdk->bhsk", q_resid, module.fusedqk[0])
        query = torch.cat((query_latent, q_rope), dim=-1)
        keys = torch.cat((sparse_ckv, sparse_rope), dim=-1).unsqueeze(1)
        weights = torch.einsum("bhsd,bntd->bhst", query, keys) * module.scaling
        weights = weights.masked_fill(~valid[:, None, None, :], MIN_MASKED_ATTENTION_VALUE)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        latent_output = torch.einsum("bhst,btd->bhsd", weights, sparse_ckv)
        output = torch.einsum("bhsk,hkd->bhsd", latent_output, module.per_head_v_up[0])
        output = output.transpose(1, 2).contiguous().reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
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
