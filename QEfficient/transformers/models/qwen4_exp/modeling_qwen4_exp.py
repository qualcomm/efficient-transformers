# -----------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
"""QEfficient wrapper for Transformers 5.18 Qwen4-Exp text models."""

from __future__ import annotations

import math
from functools import partial
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpTextAttention,
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextExperts,
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextModel,
    Qwen4ExpTextPLELayer,
    Qwen4ExpTextRMSNormGated,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextSparseMoeBlock,
    Qwen4ExpTextTopKRouter,
    apply_rotary_pos_emb,
)

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.customop import ctx_gather_rows, ctx_paged_scatter
from QEfficient.transformers.cache_utils import QEffQwen4ExpDynamicCache
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    MoEWeights,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    silu_glu_mlp,
)
from QEfficient.utils import constants
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def _causal_conv1d_update(hidden_states, conv_state, weight):
    """Functional GDN state update adapted from linear_gdn_qwen35."""
    combined = torch.cat((conv_state, hidden_states), dim=-1)
    output = F.conv1d(combined.to(weight.dtype), weight, groups=combined.shape[1])[..., -1:]
    return F.silu(output).to(hidden_states.dtype), combined[..., -conv_state.shape[-1] :]


def _gdn_recurrent_step(query, key, value, decay, beta, state):
    """Decode specialization of Transformers' ``torch_recurrent_gated_delta_rule``."""
    initial_dtype = query.dtype
    query, key, value, beta, decay = [
        tensor.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for tensor in (query, key, value, beta, decay)
    ]
    query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query = query / (query.shape[-1] ** 0.5)
    recurrent_state = state.to(value)
    output = torch.zeros_like(value)
    for token_idx in range(query.shape[2]):
        query_token, key_token, value_token = query[:, :, token_idx], key[:, :, token_idx], value[:, :, token_idx]
        recurrent_state = recurrent_state * decay[:, :, token_idx].exp()[..., None, None]
        beta_token = beta[:, :, token_idx].unsqueeze(-1)
        kv_mem = (recurrent_state * key_token.unsqueeze(-1)).sum(dim=-2)
        delta = (value_token - kv_mem) * beta_token
        recurrent_state = recurrent_state + key_token.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, token_idx] = (recurrent_state * query_token.unsqueeze(-1)).sum(dim=-2)
    return output.transpose(1, 2).contiguous().to(initial_dtype), recurrent_state


class QEffQwen4ExpTextRotaryEmbedding(Qwen4ExpTextRotaryEmbedding):
    """Static base-frequency cache for Qwen4-Exp's MRoPE positions."""

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device)
        self._set_cos_sin_cache(self.original_max_seq_len, self.inv_freq.device, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.float())
        self.register_buffer("cos_cached", frequencies.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", frequencies.sin().to(dtype), persistent=False)


def _qeff_recompose_mrope(frequencies, mrope_section, rotary_width):
    recomposed = frequencies[0].clone()
    for dimension, offset in enumerate((1, 2), start=1):
        # ``partial_rotary_factor`` can make the cached frequency width smaller
        # than the section's nominal interleaved endpoint (e.g. 32 vs 33).
        # Bound the static slice by the model-owned cache width so ONNX never
        # emits an out-of-range Slice endpoint for QAIC.
        end = min(mrope_section[dimension] * 3, rotary_width)
        target = slice(offset, end, 3)
        recomposed[..., target] = frequencies[dimension, ..., target]
    return torch.cat((recomposed, recomposed), dim=-1)


def qeff_prepare_qwen4_exp_mrope(
    cos_cached, sin_cached, position_ids, mrope_section, dtype=None, rotary_width=None
):
    """Index model-owned RoPE caches and apply Qwen4-Exp's MRoPE recomposition."""
    # Avoid integer ``aten::clamp``/ONNX Clip, which QAIC does not accept for
    # int64 tensors. Exporting the equivalent Where keeps MRoPE indexing safe.
    safe_position_ids = torch.where(position_ids < 0, torch.zeros_like(position_ids), position_ids)
    flat_positions = safe_position_ids.reshape(-1)
    cos = cos_cached.index_select(0, flat_positions).reshape(*safe_position_ids.shape, cos_cached.shape[-1])
    sin = sin_cached.index_select(0, flat_positions).reshape(*safe_position_ids.shape, sin_cached.shape[-1])
    if rotary_width is None:
        rotary_width = min(sum(mrope_section), int(cos_cached.shape[-1]))
    cos = _qeff_recompose_mrope(cos, mrope_section, rotary_width)
    sin = _qeff_recompose_mrope(sin, mrope_section, rotary_width)
    if dtype is not None:
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
    return cos, sin


class QEffQwen4ExpTextRMSNormGated(Qwen4ExpTextRMSNormGated):
    def forward(self, hidden_states, gate):
        normed = CustomRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)
        return normed * F.silu(gate.float()).to(normed.dtype)


class QEffQwen4ExpTextTopKRouter(Qwen4ExpTextTopKRouter):
    """ONNX-friendly Qwen4-Exp router with stable HF routing semantics."""

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return router_logits, top_weights.to(router_logits.dtype), top_indices


class QEffQwen4ExpTextExperts(Qwen4ExpTextExperts):
    """Canonical expert weights consumed by the shared QEff MoE dispatch."""

    def __qeff_init__(self):
        self.weights_transformed = False
        self.expert_dim = self.intermediate_dim

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        self.moe_weights = build_canonical_expert_weights(
            gate_up=self.gate_up_proj,
            down=self.down_proj,
            fused=True,
            fused_split_dim=1,
            transpose_gate_up=True,
            transpose_down=True,
            clone=True,
        )
        delete_module_attrs(self, "gate_up_proj", "down_proj")
        self.weights_transformed = True
        return self.moe_weights


class QEffQwen4ExpTextSparseMoeBlock(QEffMoEBlockMixin, Qwen4ExpTextSparseMoeBlock):
    """Qwen4-Exp MoE block using QEff routing and exportable dispatch."""

    supported_moe_flavours = (
        MoEFlavour.SIMPLE_LOOP,
        MoEFlavour.DECODE_BMM,
        MoEFlavour.EXPERT_PARALLEL,
    )

    def __qeff_init__(self):
        super().__qeff_init__()
        self.gate.__class__ = QEffQwen4ExpTextTopKRouter
        self.experts.__class__ = QEffQwen4ExpTextExperts
        self.experts.__qeff_init__()
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        # Keep the transformed model directly executable for PyTorch parity
        # checks; OptimizedMoEWeightsTransform remains idempotent afterwards.
        self.transform_weights()

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        self.moe_weights = self.experts.transform_weights()
        self.weights_transformed = True
        return self.moe_weights

    @property
    def moe_profile(self) -> MoEProfile:
        return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=self.experts.act_fn))

    def route(self, x: torch.Tensor):
        _, top_weights, top_indices = self.gate(x)
        return (top_indices, top_weights.to(x.dtype)), None

    def apply_shared_experts(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        shared = self.shared_expert(residual)
        return out + F.sigmoid(self.shared_expert_gate(residual)) * shared


class QEffQwen4ExpTextGatedDeltaNet(Qwen4ExpTextGatedDeltaNet):
    """GDN decode with separate input/output retained tensors."""

    def forward(self, hidden_states, cache_params=None, position_embeddings=None, **kwargs):
        del position_embeddings, kwargs
        if hidden_states.shape[1] != 1:
            raise ValueError("Qwen4-Exp QEfficient wrapper supports decode (seq_len=1) only")
        conv_state = cache_params.gdn_conv_states[self.layer_idx]
        recurrent_state = cache_params.gdn_recurrent_states[self.layer_idx]
        if conv_state is None or recurrent_state is None:
            raise ValueError(f"Missing GDN retained state for layer {self.layer_idx}")
        batch = hidden_states.shape[0]
        mixed, next_conv = _causal_conv1d_update(
            self.in_proj_qkv(hidden_states).transpose(1, 2), conv_state, self.conv1d.weight
        )
        query, key, value = torch.split(mixed.transpose(1, 2), (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        query = query.reshape(batch, 1, -1, self.head_k_dim)
        key = key.reshape(batch, 1, -1, self.head_k_dim)
        value = value.reshape(batch, 1, -1, self.head_v_dim)
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
        output, next_recurrent = _gdn_recurrent_step(query, key, value, decay, beta, recurrent_state)
        gate = self.in_proj_z(hidden_states).reshape(batch, 1, -1, self.head_v_dim)
        output = self.norm(output.reshape(-1, self.head_v_dim), gate.reshape(-1, self.head_v_dim)).reshape(batch, 1, -1)
        cache_params.update_gdn_state(self.layer_idx, next_conv, next_recurrent)
        return self.out_proj(output)


class QEffQwen4ExpTextPLELayer(Qwen4ExpTextPLELayer):
    """PLE without a local n-gram table; lookup remains host-side by contract."""

    def __qeff_init__(self):
        del self.ple_embedding
        self.register_buffer(
            "_qeff_conv_positions",
            torch.arange(
                0,
                self.short_conv_state_len + 1,
                self.conv1d.dilation[0],
                dtype=torch.long,
                device=self.conv1d.weight.device,
            ),
            persistent=False,
        )

    def forward(self, hidden_states, ngram_embeddings, cache_params):
        if ngram_embeddings is None or ngram_embeddings.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("PLE requires ngram_embeddings shaped [batch, seq_len, ple_embed_dim]")
        embeddings = ngram_embeddings.to(hidden_states.dtype)
        key = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        query = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate_abs = gate.abs()
        gate_abs = torch.where(gate_abs < 1e-6, torch.full_like(gate_abs, 1e-6), gate_abs)
        gate = gate_abs.sqrt() * gate.sign()
        values = (torch.sigmoid(gate) * self.value_proj(embeddings).unsqueeze(-2)).flatten(-2)
        state = cache_params.ple_conv_states[self.layer_idx]
        if state is None:
            raise ValueError(f"Missing PLE retained state for layer {self.layer_idx}")
        combined = torch.cat((state, self.norm_conv(values).transpose(1, 2)), dim=-1)
        cache_params.update_ple_state(self.layer_idx, combined[..., -self.short_conv_state_len :])
        conv_input = combined.to(self.conv1d.weight.dtype).index_select(-1, self._qeff_conv_positions)
        conv_weight = self.conv1d.weight.squeeze(1).unsqueeze(-1)
        conv_output = torch.matmul(conv_input.unsqueeze(-2), conv_weight).squeeze(-1)
        conv_output = F.silu(conv_output).to(values.dtype)
        return values + conv_output[..., -1:].transpose(1, 2)


class QEffQwen4ExpTextAttention(Qwen4ExpTextAttention):
    """QSA attention with the legacy decode ABI and aligned prefill path."""

    def __qeff_init__(self):
        self._qsa_config = None
        self._qsa_use_benchmark_path = False

    def configure_qsa(self, qsa_config=None, *, ctx_len=None, seq_len=None):
        config = self.config
        deployment = dict(qsa_config or {})
        self._qsa_config = {
            "indexer_dp": int(deployment.get("indexer_dp", 1)),
            "indexer_cp": int(deployment.get("indexer_cp", 1)),
            "attn_dp": int(deployment.get("attn_dp", 1)),
            "attn_cp": int(deployment.get("attn_cp", 1)),
            "num_cores": int(deployment.get("num_cores_per_device", deployment.get("num_cores", 16))),
            "indexer_num_blocks": int(deployment.get("indexer_num_blocks", 4)),
            "indexer_ql_chunk": int(deployment.get("indexer_ql_chunk", 512)),
            "indexer_q_block_size": int(deployment.get("indexer_q_block_size", 512)),
            "indexer_topk_blocking": int(deployment.get("indexer_topk_blocking", 4)),
            "sparse_q_block_size": int(deployment.get("sparse_q_block_size", 16)),
            "sparse_kv_num_blocks": int(deployment.get("sparse_kv_num_blocks", 8)),
            "skip_kv": bool(deployment.get("skip_kv", False)),
            "ctx_len": int(deployment.get("ctx_len", ctx_len or config.max_position_embeddings)),
            "seq_len": seq_len,
        }
        self._qsa_use_benchmark_path = bool(qsa_config)

    @property
    def _qsa_deployment(self):
        if self._qsa_config is None:
            self.configure_qsa()
        return self._qsa_config

    def qsa_forward(self, hidden_states, position_embeddings, past_key_values=None, **kwargs):
        if hidden_states.shape[1] != 1:
            return self._qsa_prefill(hidden_states, position_embeddings, past_key_values)
        return self._qsa_decode(hidden_states, position_embeddings, past_key_values, **kwargs)

    def forward(self, hidden_states, position_embeddings, past_key_values=None, **kwargs):
        return self.qsa_forward(hidden_states, position_embeddings, past_key_values, **kwargs)

    def _qsa_decode(self, hidden_states, position_embeddings, past_key_values=None, **kwargs):
        del kwargs
        key_cache = past_key_values.qsa_key_states[self.layer_idx]
        value_cache = past_key_values.qsa_value_states[self.layer_idx]
        index_cache = past_key_values.qsa_index_states[self.layer_idx]
        partial = past_key_values.qsa_partial_states[self.layer_idx]
        if any(value is None for value in (key_cache, value_cache, index_cache, partial)):
            raise ValueError(f"Missing QSA retained state for layer {self.layer_idx}")
        batch = hidden_states.shape[0]
        cos, sin = (tensor[:, -1:, :] for tensor in position_embeddings[:2])
        query, gate = self.q_proj(hidden_states).view(batch, 1, -1, self.head_dim * 2).chunk(2, dim=-1)
        query = apply_rotary_pos_emb(self.q_norm(query), cos=cos, sin=sin, unsqueeze_dim=2).transpose(1, 2)
        key = apply_rotary_pos_emb(
            self.k_norm(self.k_proj(hidden_states).view(batch, 1, -1, self.head_dim)),
            cos=cos,
            sin=sin,
            unsqueeze_dim=2,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, 1, -1, self.head_dim).transpose(1, 2)
        positions = past_key_values.position_ids[0, :, -1].to(torch.long)
        deployment = self._qsa_deployment
        folded = deployment["attn_dp"] > 1 or deployment["attn_cp"] > 1
        if folded:
            next_key = self._qsa_scatter_folded(
                key_cache, positions, key, deployment, rows_per_way=self.config.num_key_value_heads
            )
            next_value = self._qsa_scatter_folded(
                value_cache, positions, value, deployment, rows_per_way=self.config.num_key_value_heads
            )
        else:
            kv_slots = positions[:, None, None, None].expand_as(key)
            next_key = key_cache.scatter(2, kv_slots, key)
            next_value = value_cache.scatter(2, kv_slots, value)
        qk = self.indexer.index_qk_proj(hidden_states)
        index_query, raw_key = torch.split(
            qk, (self.indexer.index_n_heads * self.indexer.index_head_dim, self.indexer.index_head_dim), dim=-1
        )
        index_query = apply_rotary_pos_emb(
            self.indexer.q_layernorm(index_query.view(batch, 1, -1, self.indexer.index_head_dim)),
            cos=cos,
            sin=sin,
            unsqueeze_dim=2,
        ).squeeze(1)
        index_query = index_query.reshape(batch, -1, self.indexer.index_head_dim)
        ratio = self.indexer.compress_ratio
        block = torch.div(positions, ratio, rounding_mode="trunc")
        boundary = (positions + 1).remainder(ratio) == 0
        total = partial + raw_key.float()
        if len(position_embeddings) == 6:
            _, _, cos_cached, sin_cached, rotary_position_ids, mrope_section = position_embeddings
            pooled_position_ids = rotary_position_ids.clone()
            pooled_position_ids[0] = (block * ratio).unsqueeze(-1)
            pooled_cos, pooled_sin = qeff_prepare_qwen4_exp_mrope(
                cos_cached,
                sin_cached,
                pooled_position_ids,
                mrope_section,
                dtype=hidden_states.dtype,
                rotary_width=self._qeff_rotary_width,
            )
        else:
            pooled_cos, pooled_sin = cos, sin
        pooled = apply_rotary_pos_emb(
            self.indexer.k_layernorm((total / ratio).to(hidden_states.dtype)).unsqueeze(1),
            cos=pooled_cos,
            sin=pooled_sin,
            unsqueeze_dim=2,
        ).squeeze(1)
        if index_cache.ndim == 4:
            pooled_update = pooled.reshape(batch, 1, 1, index_cache.shape[-1])
            scattered_index = self._qsa_scatter_folded(
                index_cache,
                block,
                pooled_update,
                deployment,
                rows_per_way=1,
                dp_key="indexer_dp",
                cp_key="indexer_cp",
                valid=boundary,
            )
        else:
            index_slots = block.reshape(batch, 1, 1).expand(-1, 1, index_cache.shape[-1])
            pooled_update = pooled.reshape(batch, 1, index_cache.shape[-1])
            scattered_index = index_cache.scatter(1, index_slots, pooled_update)
        if index_cache.ndim == 4:
            index_dp = deployment["indexer_dp"]
            boundary_shape = boundary.reshape(batch // index_dp, index_dp, 1, 1)
        else:
            boundary_shape = boundary[:, None, None]
        next_index = torch.where(boundary_shape, scattered_index, index_cache)
        next_partial = torch.where(boundary[:, None, None], torch.zeros_like(total), total)
        index_attention = next_index[:, 0] if next_index.ndim == 4 and next_index.shape[1] == 1 else next_index
        complete = torch.div(positions + 1, ratio, rounding_mode="trunc")
        if index_cache.ndim == 4:
            index_cp = deployment["indexer_cp"]
            batch_local = batch // index_dp
            local_context = next_index.shape[2]
            cache_grouped = next_index.reshape(batch_local, index_dp, index_cp, local_context, -1)
            query_grouped = index_query.reshape(batch_local, index_dp, -1, index_query.shape[-1])
            scores = torch.einsum("badh,baclh->bacl", query_grouped.float(), cache_grouped.float())
            ids = (
                torch.arange(local_context, device=positions.device)[None, :] * index_cp
                + torch.arange(index_cp, device=positions.device)[:, None]
            )
            visible = ids[None, None] < complete.reshape(batch_local, index_dp, 1, 1)
            scores = scores.masked_fill(~visible, MIN_MASKED_ATTENTION_VALUE).reshape(batch, -1)
        else:
            scores = F.relu(index_query.float() @ index_attention.float().transpose(-1, -2)).sum(-2)
            visible = torch.arange(index_attention.shape[1], device=positions.device)[None, :] < complete[:, None]
            scores = scores.masked_fill(~visible, MIN_MASKED_ATTENTION_VALUE)
        blocks = scores.topk(min(self.indexer.block_topk, scores.shape[-1]), dim=-1).indices
        selected = (blocks.unsqueeze(-1) * ratio + torch.arange(ratio, device=positions.device)).reshape(batch, -1)
        selected_valid = (blocks < complete[:, None]).unsqueeze(-1).expand(-1, -1, ratio).reshape(batch, -1)
        tail = complete[:, None] * ratio + torch.arange(ratio, device=positions.device)[None, :]
        tail_valid = tail <= positions[:, None]
        selected = torch.cat((selected, tail), dim=-1)
        selected = torch.where(
            selected >= key_cache.shape[2],
            torch.full_like(selected, key_cache.shape[2] - 1),
            selected,
        )
        selected_valid = torch.cat((selected_valid, tail_valid), dim=-1)
        if folded:
            keys = self._qsa_gather_folded(next_key, selected, positions, deployment)
            values = self._qsa_gather_folded(next_value, selected, positions, deployment)
        else:
            gather = selected[:, None, :, None].expand(-1, key_cache.shape[1], -1, key_cache.shape[-1])
            keys = next_key.gather(2, gather).repeat_interleave(self.num_key_value_groups, dim=1)
            values = next_value.gather(2, gather).repeat_interleave(self.num_key_value_groups, dim=1)
        selected_keys, selected_values = keys, values
        logits = (query @ selected_keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = selected_valid[:, None, None, :]
        masked_logits = torch.where(
            mask,
            logits,
            torch.full_like(logits, MIN_MASKED_ATTENTION_VALUE),
        )
        probabilities = masked_logits.float().softmax(-1).to(query.dtype)
        output = (probabilities @ selected_values).reshape(batch, 1, -1)
        output = output * torch.sigmoid(gate.reshape(batch, 1, -1))
        past_key_values.update_qsa_state(self.layer_idx, next_key, next_value, next_index, next_partial)
        return self.o_proj(output), None

    @staticmethod
    def _qsa_scatter_folded(
        cache, positions, updates, deployment, rows_per_way, dp_key="attn_dp", cp_key="attn_cp", valid=None
    ):
        dp, cp = deployment[dp_key], deployment[cp_key]
        batch, _, _, dim = updates.shape
        if batch % dp:
            raise ValueError("QSA batch must be divisible by attn_dp")
        batch_local = batch // dp
        positions_dp = positions.reshape(batch_local, dp)
        slots = torch.div(positions_dp, cp, rounding_mode="trunc")
        ways = positions_dp.remainder(cp)
        rows = dp * cp * rows_per_way
        updates_rows = updates.reshape(batch_local, dp, rows_per_way, 1, dim).unsqueeze(2).expand(
            batch_local, dp, cp, rows_per_way, 1, dim
        ).reshape(batch_local, rows, 1, dim)
        row = torch.arange(rows, device=positions.device).reshape(dp, cp, rows_per_way)
        row = row[None].expand(batch_local, -1, -1, -1)
        owner = ways[:, :, None, None] == torch.arange(cp, device=positions.device).view(1, 1, cp, 1)
        owner = owner.expand(batch_local, dp, cp, rows_per_way)
        if valid is not None:
            owner = owner & valid.reshape(batch_local, dp, 1, 1)
        block_ids = torch.arange(batch_local, device=positions.device).view(batch_local, 1, 1).expand(batch_local, rows, 1)
        block_ids = torch.where(owner.reshape(batch_local, rows, 1), block_ids, torch.full_like(block_ids, torch.iinfo(torch.int32).max))
        addresses = slots[:, :, None, None].expand(batch_local, dp, cp, rows_per_way).reshape(batch_local, rows, 1)
        return ctx_paged_scatter(cache, block_ids.to(torch.int32), addresses.to(torch.int32), updates_rows)

    def _qsa_gather_folded(self, cache, selected, positions, deployment):
        dp, cp = deployment["attn_dp"], deployment["attn_cp"]
        kv_heads = cache.shape[1] // (dp * cp)
        batch, count = selected.shape
        batch_local = batch // dp
        pos = selected[:, None, :].expand(batch, kv_heads, count)
        pos_dp = pos.reshape(batch_local, dp, kv_heads, count)
        slots = torch.div(pos_dp, cp, rounding_mode="trunc")
        ways = pos_dp.remainder(cp)
        row = (
            torch.arange(dp, device=selected.device).view(1, dp, 1, 1) * cp * kv_heads
            + ways * kv_heads
            + torch.arange(kv_heads, device=selected.device).view(1, 1, kv_heads, 1)
        )
        gathered = ctx_gather_rows(cache, row.reshape(batch_local, dp * kv_heads, count).to(torch.int32))
        gathered = gathered.reshape(batch_local, dp, kv_heads, count, cache.shape[-1])
        return gathered.reshape(batch, kv_heads, count, cache.shape[-1]).repeat_interleave(
            self.num_key_value_groups, dim=1
        )

    def _qsa_prefill(self, hidden_states, position_embeddings, past_key_values):
        if past_key_values is None:
            raise ValueError("Qwen4-Exp QSA prefill requires retained cache state")
        deployment = self._qsa_deployment
        if deployment["attn_cp"] != 1 or deployment["indexer_cp"] != 1:
            raise ValueError("QSA aligned prefill requires attention/indexer CP=1")
        batch, query_len, _ = hidden_states.shape
        positions = past_key_values.position_ids
        if positions.ndim == 3:
            positions = positions[0]
        positions = positions[..., :query_len].to(torch.long)
        if positions.shape != (batch, query_len) or query_len % self.indexer.compress_ratio:
            raise ValueError("QSA prefill requires aligned [batch, seq_len] positions")
        cos, sin = position_embeddings[:2]
        query_gate = self.q_proj(hidden_states).view(batch, query_len, -1, self.head_dim * 2)
        query, gate = query_gate.chunk(2, dim=-1)
        query = apply_rotary_pos_emb(self.q_norm(query), cos=cos, sin=sin, unsqueeze_dim=2)
        key = apply_rotary_pos_emb(
            self.k_norm(self.k_proj(hidden_states).view(batch, query_len, -1, self.head_dim)),
            cos=cos,
            sin=sin,
            unsqueeze_dim=2,
        )
        value = self.v_proj(hidden_states).view(batch, query_len, -1, self.head_dim)
        key_cache = past_key_values.qsa_key_states[self.layer_idx]
        value_cache = past_key_values.qsa_value_states[self.layer_idx]
        index_cache = past_key_values.qsa_index_states[self.layer_idx]
        partial = past_key_values.qsa_partial_states[self.layer_idx]
        # Aligned prefill completes pooling groups; retain the same four-state ABI.
        ratio = self.indexer.compress_ratio
        raw = self.indexer.index_qk_proj(hidden_states).split(
            [self.indexer.index_n_heads * self.indexer.index_head_dim, self.indexer.index_head_dim], dim=-1
        )[1]
        pooled = self.indexer.k_layernorm(raw.reshape(batch, -1, ratio, raw.shape[-1]).float().mean(2).to(raw.dtype))
        pooled_pos = positions[:, ::ratio]
        index_cache = index_cache.scatter(1, torch.div(pooled_pos, ratio, rounding_mode="trunc").unsqueeze(-1).expand(-1, -1, index_cache.shape[-1]), pooled)
        key_cache = key_cache.scatter(2, positions[:, :, None, None].expand(batch, query_len, key.shape[2], key.shape[3]).transpose(1, 2), key.transpose(1, 2))
        value_cache = value_cache.scatter(2, positions[:, :, None, None].expand(batch, query_len, value.shape[2], value.shape[3]).transpose(1, 2), value.transpose(1, 2))
        scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) / math.sqrt(self.head_dim)
        causal = positions[:, None, None, :] <= positions[:, None, :, None]
        probs = scores.masked_fill(~causal, MIN_MASKED_ATTENTION_VALUE).softmax(-1).to(query.dtype)
        output = torch.einsum("bhqk,bkhd->bqhd", probs, value).reshape(batch, query_len, -1)
        past_key_values.update_qsa_state(self.layer_idx, key_cache, value_cache, index_cache, partial)
        return self.o_proj(output * gate.reshape_as(output).sigmoid()), None


class QEffQwen4ExpTextDecoderLayer(Qwen4ExpTextDecoderLayer):
    def __qeff_init__(self):
        if self.layer_type != "linear_attention":
            self.layer_type = "qwen_sparse_attention"
            self.self_attn._qeff_rotary_width = self._qeff_rotary_width

    def forward(self, hidden_states, position_embeddings, past_key_values, ngram_embeddings=None):
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, ngram_embeddings, past_key_values)
        hidden_states, hyper_input, weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, cache_params=past_key_values, position_embeddings=position_embeddings
            )
        else:
            hidden_states, _ = self.self_attn(hidden_states, position_embeddings, past_key_values=past_key_values)
        hidden_states = hyper_input + (hidden_states.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)
        hidden_states, hyper_input, weights = self.mlp_hyper_connection(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hyper_input + (hidden_states.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


class QEffQwen4ExpTextModel(Qwen4ExpTextModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffQwen4ExpTextRotaryEmbedding(
            config=self.config,
            device=self.embed_tokens.weight.device,
        )
        partial_rotary_factor = self.config.rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_width = int(self.config.head_dim * partial_rotary_factor) // 2
        self._qeff_rotary_width = rotary_width
        # Keep the bounded, static RoPE tables on the model itself. Exported
        # subfunctions then see the same model-owned constants and signature.
        self.sin_cached = torch.nn.Parameter(
            (self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling).contiguous()
        )
        self.cos_cached = torch.nn.Parameter(
            (self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling).contiguous()
        )
        for layer in self.layers:
            layer._qeff_rotary_width = rotary_width

    def get_weightfree_computed_initializers(self):
        """Return generated constants needed when exporting from ``meta``."""
        rotary = QEffQwen4ExpTextRotaryEmbedding(config=self.config, device="cpu")
        values = {
            "model.sin_cached": (rotary.sin_cached * rotary.attention_scaling).contiguous(),
            "model.cos_cached": (rotary.cos_cached * rotary.attention_scaling).contiguous(),
            "model.rotary_emb.inv_freq": rotary.inv_freq,
            "model.rotary_emb.original_inv_freq": rotary.original_inv_freq,
            "model.rotary_emb.cos_cached": rotary.cos_cached,
            "model.rotary_emb.sin_cached": rotary.sin_cached,
        }
        for index, layer in enumerate(self.layers):
            if layer.ple is not None:
                values[f"model.layers.{index}.ple._qeff_conv_positions"] = torch.arange(
                    0,
                    layer.ple.short_conv_state_len + 1,
                    layer.ple.conv1d.dilation[0],
                    dtype=torch.long,
                )
        return values

    def forward(
        self, input_ids=None, position_ids=None, past_key_values=None, ngram_embeddings=None, use_cache=True, **kwargs
    ):
        del kwargs
        if input_ids is None:
            raise ValueError("Qwen4-Exp QEfficient wrapper requires input_ids")
        if position_ids is None or position_ids.shape[0] != 4:
            raise ValueError("Qwen4-Exp requires explicit position_ids [4, batch, seq_len]")
        return_legacy_cache = isinstance(past_key_values, (list, tuple))
        if return_legacy_cache:
            past_key_values = QEffQwen4ExpDynamicCache.from_legacy_cache(self.config, past_key_values)
        elif not isinstance(past_key_values, QEffQwen4ExpDynamicCache):
            if hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = QEffQwen4ExpDynamicCache.from_legacy_cache(
                    self.config, past_key_values.to_legacy_cache()
                )
            else:
                raise TypeError("Qwen4-Exp requires QEffQwen4ExpDynamicCache or a legacy cache tuple")
        if ngram_embeddings is None:
            raise ValueError("Qwen4-Exp requires host-provided ngram_embeddings")
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)
        past_key_values.position_ids = position_ids[1:]
        rotary_position_ids = position_ids[1:]
        mrope_section = self.config.rope_parameters.get("mrope_section", [11, 11, 10])
        cos, sin = qeff_prepare_qwen4_exp_mrope(
            self.cos_cached,
            self.sin_cached,
            rotary_position_ids,
            mrope_section,
            dtype=hidden_states.dtype,
            rotary_width=self._qeff_rotary_width,
        )
        position_embeddings = (cos, sin, self.cos_cached, self.sin_cached, rotary_position_ids, mrope_section)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, past_key_values, ngram_embeddings)
        return {
            "last_hidden_state": self.hyper_connection_mixer(hidden_states),
            "past_key_values": (past_key_values.to_legacy_cache() if return_legacy_cache else past_key_values)
            if use_cache
            else None,
        }


class QEffQwen4ExpDecodeExportMixin:
    """Explicit retained-state ABI shared by the text causal-LM export wrapper."""

    def _layer_type(self, layer_idx):
        return (
            "qwen_sparse_attention"
            if self.config.layer_types[layer_idx] == "full_attention"
            else self.config.layer_types[layer_idx]
        )

    def _iter_retained_state_names(self):
        names = []
        for index in range(self.config.num_hidden_layers):
            if self._layer_type(index) == "linear_attention":
                names.extend((f"gdn_conv_state.{index}", f"gdn_recurrent_state.{index}"))
                if index + 1 in self.config.ple_layer_ids:
                    names.append(f"ple_conv_state.{index}")
            else:
                names.extend(
                    (
                        f"qsa_key_state.{index}",
                        f"qsa_value_state.{index}",
                        f"qsa_index_state.{index}",
                        f"qsa_partial_state.{index}",
                    )
                )
        return names

    get_retained_state_names = _iter_retained_state_names

    def get_onnx_past_key_value_names(self, layer_idx, layer_state=None):
        del layer_state
        if self._layer_type(layer_idx) == "linear_attention":
            names = [f"gdn_conv_state.{layer_idx}", f"gdn_recurrent_state.{layer_idx}"]
            if layer_idx + 1 in self.config.ple_layer_ids:
                names.append(f"ple_conv_state.{layer_idx}")
            return names
        return [
            f"qsa_key_state.{layer_idx}",
            f"qsa_value_state.{layer_idx}",
            f"qsa_index_state.{layer_idx}",
            f"qsa_partial_state.{layer_idx}",
        ]

    def get_onnx_retained_state_specs(
        self, batch_size, seq_len, kv_cache_shape=None, continuous_batching=False, retain_full_kv=False
    ):
        del seq_len, kv_cache_shape, retain_full_kv
        config, dtype = self.config, self.lm_head.weight.dtype
        batch_axis = "full_batch_size" if continuous_batching else "batch_size"
        specs = {"past_key_values": [], "input_names": [], "output_names": [], "dynamic_axes": {}}
        for index in range(config.num_hidden_layers):
            if self._layer_type(index) == "linear_attention":
                conv_dim = (
                    2 * config.linear_num_key_heads * config.linear_key_head_dim
                    + config.linear_num_value_heads * config.linear_value_head_dim
                )
                names = [f"gdn_conv_state.{index}", f"gdn_recurrent_state.{index}"]
                states = [
                    torch.zeros((batch_size, conv_dim, config.linear_conv_kernel_dim), dtype=dtype),
                    torch.zeros(
                        (
                            batch_size,
                            config.linear_num_value_heads,
                            config.linear_key_head_dim,
                            config.linear_value_head_dim,
                        ),
                        dtype=torch.float32,
                    ),
                ]
                axes = [{0: batch_axis}, {0: batch_axis}]
                if index + 1 in config.ple_layer_ids:
                    names.append(f"ple_conv_state.{index}")
                    states.append(
                        torch.zeros(
                            (
                                batch_size,
                                config.hc_count * config.hidden_size,
                                (config.ple_conv_kernel_size - 1) * config.ngram_size,
                            ),
                            dtype=dtype,
                        )
                    )
                    axes.append({0: batch_axis})
            else:
                context = config.max_position_embeddings
                attention = self.model.layers[index].self_attn
                qsa_config = getattr(attention, "_qsa_config", None)
                attn_dp = int(qsa_config.get("attn_dp", 1)) if qsa_config else 1
                attn_cp = int(qsa_config.get("attn_cp", 1)) if qsa_config else 1
                indexer_dp = int(qsa_config.get("indexer_dp", 1)) if qsa_config else 1
                indexer_cp = int(qsa_config.get("indexer_cp", 1)) if qsa_config else 1
                names = [
                    f"qsa_key_state.{index}",
                    f"qsa_value_state.{index}",
                    f"qsa_index_state.{index}",
                    f"qsa_partial_state.{index}",
                ]
                states = [
                    torch.zeros(
                        (batch_size // attn_dp, attn_dp * attn_cp * config.num_key_value_heads, context // attn_cp, config.head_dim),
                        dtype=dtype,
                    ),
                    torch.zeros(
                        (batch_size // attn_dp, attn_dp * attn_cp * config.num_key_value_heads, context // attn_cp, config.head_dim),
                        dtype=dtype,
                    ),
                    torch.zeros(
                        (
                            batch_size // indexer_dp,
                            indexer_dp * indexer_cp,
                            context // (config.indexer_compress_ratio * indexer_cp),
                            config.indexer_head_dim,
                        ),
                        dtype=dtype,
                    )
                    if qsa_config
                    else torch.zeros(
                        (batch_size, context // config.indexer_compress_ratio, config.indexer_head_dim), dtype=dtype
                    ),
                    torch.zeros((batch_size, 1, config.indexer_head_dim), dtype=torch.float32),
                ]
                axes = [
                    {0: batch_axis, 2: "local_ctx_len"},
                    {0: batch_axis, 2: "local_ctx_len"},
                    {0: batch_axis, 2: "local_pooled_ctx_len"} if qsa_config else {0: batch_axis, 1: "pooled_ctx_len"},
                    {0: batch_axis},
                ]
            specs["past_key_values"].append(states)
            for name, state_axes in zip(names, axes):
                specs["input_names"].append(name)
                specs["output_names"].append(f"{name}_RetainedState")
                specs["dynamic_axes"][name] = state_axes
        return specs

    def get_dummy_inputs(self, continuous_batching=False, **kwargs):
        requested_batch_size = kwargs.get("batch_size", constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE)
        batch_size = min(requested_batch_size, 2)
        # Folded QSA caches are partitioned by DP.  The generic export clamp
        # to batch 2 would otherwise create zero-sized cache dimensions when
        # attn/indexer DP is greater than 2 (for example, the Qwen DP16 path).
        qsa_dp = max(
            (
                int((getattr(layer.self_attn, "_qsa_config", None) or {}).get(key, 1))
                for layer in self.model.layers
                if hasattr(layer, "self_attn")
                for key in ("attn_dp", "indexer_dp")
            ),
            default=1,
        )
        batch_size = max(batch_size, qsa_dp)
        seq_len = int(kwargs.get("seq_len", self.get_export_seq_len()))
        specs = self.get_onnx_retained_state_specs(batch_size, seq_len, continuous_batching=continuous_batching)
        inputs = {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
            "ngram_embeddings": torch.zeros(
                (batch_size, seq_len, self.config.ple_embed_dim), dtype=self.lm_head.weight.dtype
            ),
            "position_ids": torch.zeros((4, batch_size, seq_len), dtype=torch.int64),
            "past_key_values": specs["past_key_values"],
        }
        if continuous_batching:
            inputs["batch_index"] = torch.arange(batch_size, dtype=torch.int64).view(batch_size, 1)
        return inputs

    def get_export_batch_size(self, requested_batch_size: int) -> int:
        """Keep QSA partitioned retained-state dimensions non-empty during export."""
        qsa_dp = max(
            (
                int((getattr(layer.self_attn, "_qsa_config", None) or {}).get(key, 1))
                for layer in self.model.layers
                if hasattr(layer, "self_attn")
                for key in ("attn_dp", "indexer_dp")
            ),
            default=1,
        )
        return max(requested_batch_size, qsa_dp)

    def get_onnx_dynamic_axes(self, continuous_batching=False, **kwargs):
        del kwargs
        batch_axis = "full_batch_size" if continuous_batching else "batch_size"
        axes = {
            "input_ids": {0: batch_axis, 1: "seq_len"},
            "ngram_embeddings": {0: batch_axis, 1: "seq_len"},
            "position_ids": {1: batch_axis, 2: "seq_len"},
            **self.get_onnx_retained_state_specs(1, 1, continuous_batching=continuous_batching)["dynamic_axes"],
        }
        if continuous_batching:
            axes["batch_index"] = {0: batch_axis}
        return axes

    def get_output_names(self, **kwargs):
        del kwargs
        return ["logits", *[f"{name}_RetainedState" for name in self._iter_retained_state_names()]]

    def get_specializations(self, batch_size: int, prefill_seq_len: int, ctx_len: int, **kwargs):
        """Return decode-only specializations for the retained-state graph.

        The exported graph accepts one token at a time. ``prefill_seq_len`` is
        retained as a compile API input for the nominal PL, but it must not be
        used as the graph sequence length because this wrapper has no prefill
        implementation. The QSA index state uses the compressed context axis.
        """
        del prefill_seq_len, kwargs
        batch_size = batch_size or 1
        pooled_ctx_len = -(-ctx_len // self.config.indexer_compress_ratio)
        export_seq_len = self.get_export_seq_len()
        specialization = {
            "batch_size": batch_size,
            "seq_len": export_seq_len,
            "ctx_len": ctx_len,
            "pooled_ctx_len": pooled_ctx_len,
        }
        return [dict(specialization), dict(specialization)]

    def get_custom_io(self, **kwargs):
        """Return custom I/O for every retained state input/output pair."""
        dtype = kwargs.pop("dtype", "float16")
        if kwargs:
            raise TypeError(f"Unexpected custom I/O options: {sorted(kwargs)}")
        return {
            name: dtype
            for state_name in self._iter_retained_state_names()
            for name in (state_name, f"{state_name}_RetainedState")
        }

    def get_export_seq_len(self):
        """Return a configured QSA export length, preserving decode by default."""
        for layer in self.model.layers:
            attention = getattr(layer, "self_attn", None)
            qsa_config = getattr(attention, "_qsa_config", None)
            if qsa_config is not None and qsa_config.get("seq_len") is not None:
                return int(qsa_config["seq_len"])
        return 1

    def prepare_export_inputs(self, example_inputs, dynamic_axes):
        """Add host PLE input and the four-stream MRoPE position input."""
        batch_size = example_inputs["input_ids"].shape[0]
        dtype = self.config.torch_dtype
        seq_len = example_inputs["input_ids"].shape[1]
        example_inputs["ngram_embeddings"] = torch.zeros(
            (batch_size, seq_len, self.config.ple_embed_dim), dtype=dtype
        )
        example_inputs["position_ids"] = torch.zeros((4, batch_size, seq_len), dtype=torch.int64)
        dynamic_axes["ngram_embeddings"] = {0: "batch_size", 1: "seq_len"}
        dynamic_axes["position_ids"] = {1: "batch_size", 2: "seq_len"}
        return example_inputs, dynamic_axes

    def prepare_export_dynamic_axes(self, dynamic_axes):
        """Use fixed DP16 batch/context cache shapes for Dynamo export."""
        static_dims = {
            "batch_size",
            "full_batch_size",
            "ctx_len",
            "local_ctx_len",
            "pooled_ctx_len",
            "local_pooled_ctx_len",
        }
        return {
            name: {axis: dim for axis, dim in axes.items() if dim not in static_dims}
            for name, axes in dynamic_axes.items()
        }


class QEffQwen4ExpForCausalLM(QEffQwen4ExpDecodeExportMixin, Qwen4ExpForCausalLM):
    def __qeff_init__(self):
        # Router logits and the auxiliary load-balancing loss are training-only
        # outputs; keeping them disabled also avoids exporting torch.bincount.
        self.config.output_router_logits = False

    def qeff_cache_from_legacy(self, past_key_values=None):
        return QEffQwen4ExpDynamicCache.from_legacy_cache(self.config, past_key_values)

    def get_weightfree_computed_initializers(self):
        return self.model.get_weightfree_computed_initializers()

    def forward(self, input_ids=None, ngram_embeddings=None, position_ids=None, past_key_values=None, **kwargs):
        del kwargs
        return_legacy_cache = isinstance(past_key_values, (list, tuple))
        if return_legacy_cache:
            past_key_values = self.qeff_cache_from_legacy(past_key_values)
        elif not isinstance(past_key_values, QEffQwen4ExpDynamicCache) and hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = self.qeff_cache_from_legacy(past_key_values.to_legacy_cache())
        outputs = self.model(input_ids, position_ids, past_key_values, ngram_embeddings)
        output_cache = outputs["past_key_values"]
        if return_legacy_cache and output_cache is not None:
            output_cache = output_cache.to_legacy_cache()
        return MoeCausalLMOutputWithPast(
            logits=self.lm_head(outputs["last_hidden_state"]), past_key_values=output_cache
        )
