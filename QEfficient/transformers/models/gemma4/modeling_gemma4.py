# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from collections import defaultdict
from functools import partial
from pathlib import Path

import onnx
import torch
import yaml
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ClippableLinear,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
    Gemma4TextRouter,
    Gemma4VisionAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
    rotate_half,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
)
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffGemma4DynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    MoEWeights,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    silu_glu_mlp,
)
from QEfficient.transformers.spd.dflash import compute_dflash_target_hidden_states
from QEfficient.utils import constants

_FP16_CLAMP_MIN = -65504.0
_FP16_CLAMP_MAX = 65504.0
_DISABLE_EXPORT_FP16_CLAMP = False


def _is_onnx_export() -> bool:
    return torch.onnx.is_in_onnx_export()


def _clamp_to_fp16_range(hidden_states: torch.Tensor) -> torch.Tensor:
    if not _is_onnx_export() or _DISABLE_EXPORT_FP16_CLAMP:
        return hidden_states
    return hidden_states.clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX)


def _saturating_residual_add(residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    if not _is_onnx_export() or _DISABLE_EXPORT_FP16_CLAMP:
        return residual + hidden_states
    return (residual.float() + hidden_states.float()).clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX).to(hidden_states.dtype)


def _attention_mask_min(dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    # During export with -convert-to-fp16, using finfo(float32).min can create
    # extreme constants that destabilize downstream compiler passes.
    if _is_onnx_export():
        return torch.tensor(_FP16_CLAMP_MIN, dtype=dtype, device=device)
    return torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)


def _sanitize_clippable_linear_bounds(module: nn.Module) -> None:
    # Ensure clipping bounds are finite before ONNX export.
    for submodule in module.modules():
        if not getattr(submodule, "use_clipped_linears", False):
            continue
        for name, fallback in (
            ("input_min", -1e30),
            ("input_max", 1e30),
            ("output_min", -1e30),
            ("output_max", 1e30),
        ):
            buf = getattr(submodule, name, None)
            if isinstance(buf, torch.Tensor) and (torch.isinf(buf).any() or torch.isnan(buf).any()):
                with torch.no_grad():
                    buf.fill_(fallback)


def _build_additive_attention_mask(
    position_ids: torch.Tensor,
    target_length,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> torch.Tensor:
    causal_mask = _create_causal_mask(
        position_ids=position_ids,
        target_length=target_length,
        sliding_window=sliding_window,
    )
    return causal_mask


def _build_bidirectional_vision_attention_mask(
    position_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor | None,
    target_length: int,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """
    Export-safe eager attention mask that mirrors Gemma4's HF image-token semantics:
    vision tokens in the same contiguous image block attend bidirectionally, while all
    remaining tokens keep standard causal/sliding attention.
    """
    base_mask = _create_causal_mask(
        position_ids=position_ids,
        target_length=target_length,
        sliding_window=sliding_window,
    )
    if mm_token_type_ids is None:
        return base_mask

    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.to(torch.int64), dim=1) - 1
    vision_group_ids = torch.where(is_vision, vision_group_ids, torch.full_like(vision_group_ids, -1))

    kv_indices = torch.arange(target_length, device=vision_group_ids.device, dtype=torch.int64).view(1, -1)
    seq_len_limit = torch.full_like(kv_indices, vision_group_ids.shape[1] - 1)
    safe_kv_indices = torch.minimum(kv_indices, seq_len_limit)
    kv_group_ids = torch.gather(vision_group_ids, 1, safe_kv_indices.expand(vision_group_ids.shape[0], -1))
    kv_group_ids = torch.where(kv_indices < vision_group_ids.shape[1], kv_group_ids, torch.full_like(kv_group_ids, -1))

    same_group = (vision_group_ids.unsqueeze(-1) == kv_group_ids.unsqueeze(1)) & (vision_group_ids.unsqueeze(-1) >= 0)
    attention_mask = base_mask & ~same_group.unsqueeze(1)
    return attention_mask


def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Applies multidimensional RoPE to inputs.

    For ndim==2 (Gemma4 vision), applies RoPE split-free and reshape-free to
    avoid QAIC issues with non-constant Split inputs and shape-driven reshapes.
    """
    ndim = int(position_ids.shape[-1])
    num_input_channels = int(x.shape[-1])
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))

    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            "Invalid configuration: num_rotated_channels_per_dim must be > 0, got"
            f" {num_rotated_channels_per_dim} (num_input_channels={num_input_channels},"
            f" ndim={ndim})"
        )

    total_rotated_channels = num_rotated_channels_per_dim * ndim
    if total_rotated_channels != num_input_channels:
        raise ValueError(
            "Invalid configuration: expected head dim to be divisible by 2 * ndim."
            f" got num_input_channels={num_input_channels}, ndim={ndim}"
        )

    if ndim == 2:
        c = num_rotated_channels_per_dim
        y0 = apply_rotary_pos_emb(
            x=x[..., :c],
            cos=cos[..., :c],
            sin=sin[..., :c],
            unsqueeze_dim=unsqueeze_dim,
        )
        y1 = apply_rotary_pos_emb(
            x=x[..., c : 2 * c],
            cos=cos[..., c : 2 * c],
            sin=sin[..., c : 2 * c],
            unsqueeze_dim=unsqueeze_dim,
        )
        return torch.cat((y0, y1), dim=-1)

    x_grouped = x.reshape(*x.shape[:-1], ndim, num_rotated_channels_per_dim)
    cos_grouped = cos.reshape(*cos.shape[:-1], ndim, num_rotated_channels_per_dim)
    sin_grouped = sin.reshape(*sin.shape[:-1], ndim, num_rotated_channels_per_dim)
    y_grouped = apply_rotary_pos_emb(
        x=x_grouped,
        cos=cos_grouped,
        sin=sin_grouped,
        unsqueeze_dim=unsqueeze_dim,
    )
    return y_grouped.reshape(*x.shape[:-1], total_rotated_channels)


def eager_attention_forward_text(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask,
            torch.tensor(constants.MIN_MASKED_ATTENTION_VALUE, dtype=module.config.torch_dtype),
            attn_weights,
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffGemma4TextRouter(Gemma4TextRouter):
    def __qeff_init__(self):
        if (
            hasattr(self, "norm")
            and not getattr(self.norm, "with_scale", True)
            and not hasattr(self.norm, "_qeff_unit_weight")
        ):
            self.norm.register_buffer("_qeff_unit_weight", torch.ones(self.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        router_probabilities = nn.functional.softmax(self.proj(hidden_states), dim=-1)
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )

        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index


class QEffGemma4ClippableLinear(Gemma4ClippableLinear):
    def __init__(
        self,
        config,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__(config, in_features, out_features)
        if self.use_clipped_linears:
            with torch.no_grad():
                self.input_min.fill_(-1e30)
                self.input_max.fill_(1e30)
                self.output_min.fill_(-1e30)
                self.output_max.fill_(1e30)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            in_min = self.input_min.to(dtype=hidden_states.dtype, device=hidden_states.device)
            in_max = self.input_max.to(dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = torch.maximum(hidden_states, in_min)
            hidden_states = torch.minimum(hidden_states, in_max)

        hidden_states = self.linear(hidden_states)

        if self.use_clipped_linears:
            out_min = self.output_min.to(dtype=hidden_states.dtype, device=hidden_states.device)
            out_max = self.output_max.to(dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = torch.maximum(hidden_states, out_min)
            hidden_states = torch.minimum(hidden_states, out_max)

        return hidden_states


class QEffGemma4CustomRMSNormAIC(nn.Module):
    """
    Gemma4 RMSNorm replacement that preserves `with_scale=False` behavior while
    still exporting through the compiler-known custom RMSNorm op.
    """

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not _is_onnx_export():
            normed_output = self._norm(hidden_states.float())
            if getattr(self, "with_scale", True):
                normed_output = normed_output * self.weight.float()
            return normed_output.type_as(hidden_states)

        if getattr(self, "with_scale", True):
            weight = self.weight
        else:
            weight = getattr(self, "_qeff_unit_weight", None)
            if weight is None:
                weight = hidden_states.new_ones(hidden_states.shape[-1])
        # Cast weight to match hidden_states dtype so the AIC compiler sees
        # matching Input/Scale dtypes in the CustomRMSNorm op (e.g. bfloat16).
        return CustomRMSNormFunc.apply(hidden_states, weight.to(hidden_states.dtype), self.eps)


class QEffGemma4TextMoeBlock(QEffMoEBlockMixin, nn.Module):
    supported_moe_flavours = (
        MoEFlavour.SIMPLE_LOOP,
        MoEFlavour.DECODE_BMM,
        MoEFlavour.EXPERT_PARALLEL,
    )
    supports_moe_decode_bmm = True

    def __init__(
        self,
        router: nn.Module,
        experts: nn.Module,
        pre_feedforward_layernorm: nn.Module,
        post_feedforward_layernorm: nn.Module,
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = experts
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        self.moe_weights = build_canonical_expert_weights(
            gate_up=self.experts.gate_up_proj,
            down=self.experts.down_proj,
            fused=True,
            fused_split_dim=1,
            transpose_gate_up=True,
            transpose_down=True,
            clone=True,
        )
        delete_module_attrs(self.experts, "gate_up_proj", "down_proj")
        self.weights_transformed = True
        return self.moe_weights

    @property
    def moe_profile(self) -> MoEProfile:
        return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=self.experts.act_fn))

    def route(self, x: torch.Tensor):
        router_probabilities, top_k_weights, top_k_index = self.router(x)
        return (top_k_index, top_k_weights.to(x.dtype)), router_probabilities

    def forward(self, hidden_states: torch.Tensor):
        B, S, H = hidden_states.shape
        x = hidden_states.reshape(B * S, H)
        routing, _ = self.route(x)
        x = self.pre_feedforward_layernorm(x)
        out = self.execute_moe_flavour(x, routing)
        out = out.reshape(B, S, H)
        return self.post_feedforward_layernorm(out)


class QEffGemma4VisionAttention(Gemma4VisionAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope(query_states, cos, sin, position_ids)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(key_states, cos, sin, position_ids)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGemma4TextRotaryEmbedding(Gemma4TextRotaryEmbedding):
    """Gemma4 rotary embeddings with static caches for each text attention type."""

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device)

        for layer_type in sorted(self.layer_types):
            self._set_cos_sin_cache(
                layer_type=layer_type,
                seq_len=self.original_max_seq_len,
                device=getattr(self, f"{layer_type}_inv_freq").device,
                dtype=config.dtype,
            )

    def _set_cos_sin_cache(self, layer_type, seq_len, device, dtype):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        positions = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(positions, inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(f"{layer_type}_cos_cached", embeddings.cos().to(dtype), persistent=False)
        self.register_buffer(f"{layer_type}_sin_cached", embeddings.sin().to(dtype), persistent=False)


def qeff_apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply pre-shaped rotary embeddings and restore the input dtype."""
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.to(x.dtype)


class QEffGemma4TextAttention(Gemma4TextAttention):
    def __qeff_init__(self):
        for norm_name in ("q_norm", "k_norm", "v_norm"):
            norm = getattr(self, norm_name, None)
            if norm is not None and not getattr(norm, "with_scale", True) and not hasattr(norm, "_qeff_unit_weight"):
                norm.register_buffer("_qeff_unit_weight", torch.ones(self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        batch_index: torch.LongTensor | None = None,
        comp_ctx_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        target_dtype = hidden_states.dtype
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        token_key_states = None
        token_value_states = None
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        num_kv_shared_layers = int(getattr(self.config, "num_kv_shared_layers", 0) or 0)
        disable_blocking_for_store_full_kv = self.store_full_length_kv and num_kv_shared_layers > 0
        has_valid_blocking_mode = blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)
        # KV-sharing layers reuse KV from an earlier layer, so they should not run a blocked KV update path.
        is_regular_kv_layer = not self.is_kv_shared_layer
        # When KV sharing is enabled, the "store full KV" source layer must keep the non-blocked cache semantics.
        can_block_when_storing_full_kv = not disable_blocking_for_store_full_kv
        # Sliding-window layers use distinct cache semantics; disable blocked attention variants on
        # those layers to keep behavior consistent and avoid mode-specific cache update issues.
        blocks_supported_for_layer_type = self.sliding_window is None
        use_blocking = (
            has_valid_blocking_mode
            and is_regular_kv_layer
            and can_block_when_storing_full_kv
            and blocks_supported_for_layer_type
        )

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = qeff_apply_rotary_pos_emb(query_states, cos, sin)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer and past_key_values is not None:
            key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device, dtype=target_dtype)
            value_states = value_states.to(query_states.device, dtype=target_dtype)
            if hasattr(past_key_values, "shared_layers_token"):
                token_states = past_key_values.shared_layers_token.get(self.kv_shared_layer_index)
                if token_states is not None:
                    token_key_states, token_value_states = token_states
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = qeff_apply_rotary_pos_emb(key_states, cos, sin)
            key_states = key_states.transpose(1, 2)
            if key_states.dtype != target_dtype:
                key_states = key_states.to(target_dtype)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)
            if value_states.dtype != target_dtype:
                value_states = value_states.to(target_dtype)
            token_key_states, token_value_states = key_states, value_states

        if use_blocking:
            if (
                mm_token_type_ids is not None
                and hidden_states.shape[1] != 1
                and getattr(self.config, "use_bidirectional_attention", None) == "vision"
            ):
                attention_mask = _build_bidirectional_vision_attention_mask(
                    position_ids=position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    target_length=key_states.shape[-2],
                    dtype=query_states.dtype,
                    sliding_window=self.sliding_window,
                )

            past_seen_tokens = (
                int(past_key_values.get_seq_length(self.layer_idx))
                if past_key_values is not None
                else int(key_states.shape[-2])
            )
            attn_output, attn_weights = generic_blocked_attention_interface(
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                layer_idx=self.layer_idx,
                past_key_value=past_key_values,
                blocking_config=blocking_config,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
                past_seen_tokens=past_seen_tokens,
                sliding_window=self.sliding_window,
                prefill_only=blocking_config.mode.is_prefill,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        if past_key_values is not None:
            if comp_ctx_lengths is not None:
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                    cache_kwargs["CCL"] = attention_mask.shape[-1]
                elif self.sliding_window is None:
                    cache_kwargs["CCL"] = comp_ctx_lengths.shape[-1]
            if self.is_kv_shared_layer:
                if token_key_states is not None and token_value_states is not None:
                    key_states, value_states = past_key_values.update(
                        token_key_states,
                        token_value_states,
                        self.layer_idx,
                        cache_kwargs,
                    )
            else:
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                if not hasattr(past_key_values, "shared_layers_token"):
                    past_key_values.shared_layers_token = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states
                if token_key_states is not None and token_value_states is not None:
                    past_key_values.shared_layers_token[self.layer_idx] = token_key_states, token_value_states

        if (
            mm_token_type_ids is not None
            and hidden_states.shape[1] != 1
            and getattr(self.config, "use_bidirectional_attention", None) == "vision"
        ):
            attention_mask = _build_bidirectional_vision_attention_mask(
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
                target_length=key_states.shape[-2],
                dtype=query_states.dtype,
                sliding_window=self.sliding_window,
            )

        attn_output, attn_weights = eager_attention_forward_text(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGemma4TextDecoderLayer(Gemma4TextDecoderLayer):
    def __qeff_init__(self):
        if not getattr(self, "enable_moe_block", False) or hasattr(self, "moe_block"):
            return
        router = self._modules.pop("router")
        experts = self._modules.pop("experts")
        pre_norm = self._modules.pop("pre_feedforward_layernorm_2")
        post_norm = self._modules.pop("post_feedforward_layernorm_2")
        self.moe_block = QEffGemma4TextMoeBlock(router, experts, pre_norm, post_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = _clamp_to_fp16_range(hidden_states)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            hidden_states_2 = self.moe_block(residual)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = _saturating_residual_add(residual, hidden_states)

        hidden_states *= self.layer_scalar
        return hidden_states


class QEffGemma4TextModel(Gemma4TextModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffGemma4TextRotaryEmbedding(config=self.config)
        for layer_type in sorted(self.rotary_emb.layer_types):
            attention_scaling = getattr(self.rotary_emb, f"{layer_type}_attention_scaling")
            sin_cached = getattr(self.rotary_emb, f"{layer_type}_sin_cached") * attention_scaling
            cos_cached = getattr(self.rotary_emb, f"{layer_type}_cos_cached") * attention_scaling
            setattr(self, f"{layer_type}_sin_cached", nn.Parameter(sin_cached))
            setattr(self, f"{layer_type}_cos_cached", nn.Parameter(cos_cached))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        comp_ctx_lengths: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        target_dtype = self.embed_tokens.weight.dtype
        if inputs_embeds is not None and inputs_embeds.dtype != target_dtype:
            inputs_embeds = inputs_embeds.to(target_dtype)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and isinstance(past_key_values, Cache) and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_cache(self.config, past_key_values)
        elif use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffGemma4DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds

        position_embeddings = {}
        for layer_type in sorted(self.unique_layer_types):
            sin_cached = getattr(self, f"{layer_type}_sin_cached")
            cos_cached = getattr(self, f"{layer_type}_cos_cached")
            sin = sin_cached[position_ids].unsqueeze(2)
            cos = cos_cached[position_ids].unsqueeze(2)
            position_embeddings[layer_type] = (cos, sin)

        # DFlash TLM: collect hidden states entering the selected target layers.
        self.target_layer_ids = getattr(self, "target_layer_ids", None)
        target_hidden_list = []

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.target_layer_ids and i in self.target_layer_ids:
                target_hidden_list.append(hidden_states)
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            layer_type = self.config.layer_types[i]
            layer_attention_mask = attention_mask
            use_mm_bidirectional_mask = (
                kwargs.get("mm_token_type_ids") is not None
                and inputs_embeds.shape[1] != 1
                and getattr(self.config, "use_bidirectional_attention", None) == "vision"
            )
            if isinstance(attention_mask, dict):
                layer_attention_mask = attention_mask[layer_type]
            elif use_mm_bidirectional_mask:
                layer_attention_mask = None
            else:
                sliding_window = self.config.sliding_window if layer_type == "sliding_attention" else None
                target_length = (
                    min(self.config.sliding_window, self.config.max_position_embeddings)
                    if sliding_window
                    else inputs_embeds.shape[1]
                )
                if past_key_values is not None and len(past_key_values.layers) > i:
                    layer_keys = past_key_values.layers[i].keys
                    if layer_keys is not None and layer_keys.numel() > 0:
                        target_length = layer_keys.shape[-2]
                layer_attention_mask = _build_additive_attention_mask(
                    position_ids=position_ids,
                    target_length=target_length,
                    dtype=hidden_states.dtype,
                    sliding_window=sliding_window,
                )

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        next_cache = past_key_values.to_legacy_cache() if use_cache else None

        # DFlash TLM: concat collected target-layer hidden states -> fc -> hidden_norm.
        target_hidden = None
        if self.target_layer_ids:
            target_hidden = compute_dflash_target_hidden_states(target_hidden_list, self.fc, self.hidden_norm)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=target_hidden,
        )
        return output if return_dict else output.to_tuple()


class QEffGemma4ForCausalLM(Gemma4ForCausalLM):
    _NPI_FP32_OPS = {"Cast", "Pow", "ReduceMean", "Add", "Mul", "Div", "Softmax", "Tanh", "Clip"}
    _NPI_SEMANTIC_NAMES = ("attn_weights", "top_k_weights", "experts_out")
    _NPI_ATTENTION_NAMES = ("query_states", "key_states", "value_states", "key", "value")
    _NPI_BAD_OUTPUT_TOKENS = ("Shape", "Equal", "Unsqueeze", "Slice", "Gather", "Transpose")
    _NPI_EXCLUDED_OPS = {
        "Constant",
        "ConstantOfShape",
        "Concat",
        "CustomRMSNorm",
        "Equal",
        "Gather",
        "MatMul",
        "Range",
        "Reshape",
        "Shape",
        "Slice",
        "Transpose",
        "Unsqueeze",
    }

    def __qeff_init__(self):
        if hasattr(self.config, "_experts_implementation"):
            self.config._experts_implementation = "eager"

    @staticmethod
    def _matches_semantic_name(output_name: str, semantic_name: str) -> bool:
        return output_name == semantic_name or output_name.startswith(f"{semantic_name}.")

    @classmethod
    def _find_output_name(cls, output_names: list[str], semantic_name: str) -> str | None:
        for output_name in output_names:
            if cls._matches_semantic_name(output_name, semantic_name):
                return output_name
        return None

    @staticmethod
    def _find_consumer(consumers: dict[str, list], input_name: str | None, op_type: str):
        if input_name is None:
            return None
        for node in consumers.get(input_name, []):
            if node.op_type == op_type:
                return node
        return None

    @classmethod
    def _collect_attention_fp32_names(cls, function) -> list[str]:
        consumers = defaultdict(list)
        output_names = []

        def add_output(name: str | None):
            if name is not None:
                output_names.append(name)

        for node in function.node:
            for input_name in node.input:
                consumers[input_name].append(node)

            for semantic_name in cls._NPI_ATTENTION_NAMES:
                add_output(cls._find_output_name(list(node.output), semantic_name))

        attn_weights = None
        for node in function.node:
            attn_weights = cls._find_output_name(list(node.output), "attn_weights")
            if attn_weights is not None:
                add_output(attn_weights)
                break

        if attn_weights is None:
            return output_names

        softmax_node = cls._find_consumer(consumers, attn_weights, "Softmax")
        softmax_output = softmax_node.output[0] if softmax_node is not None else None
        add_output(softmax_output)

        softmax_cast_output = None
        if softmax_output is not None:
            cast_node = cls._find_consumer(consumers, softmax_output, "Cast")
            if cast_node is not None:
                softmax_cast_output = cast_node.output[0]
                add_output(softmax_cast_output)

        attention_probs = softmax_cast_output or softmax_output
        if softmax_cast_output is not None:
            cast_node = cls._find_consumer(consumers, softmax_cast_output, "Cast")
            if cast_node is not None:
                attention_probs = cast_node.output[0]
                add_output(attention_probs)

        query_states = None
        for node in function.node:
            query_states = cls._find_output_name(list(node.output), "query_states")
            if query_states is not None:
                break

        qk_matmul_node = cls._find_consumer(consumers, query_states, "MatMul")
        qk_logits = qk_matmul_node.output[0] if qk_matmul_node is not None else None
        add_output(qk_logits)

        scaled_logits = None
        if qk_logits is not None:
            mul_node = cls._find_consumer(consumers, qk_logits, "Mul")
            if mul_node is not None:
                scaled_logits = mul_node.output[0]
                add_output(scaled_logits)

        for node in function.node:
            if node.op_type == "Cast" and "attention_mask" in node.input:
                add_output(node.output[0])
                break

        context_node = cls._find_consumer(consumers, attention_probs, "MatMul")
        context_output = context_node.output[0] if context_node is not None else None
        add_output(context_output)

        transpose_node = cls._find_consumer(consumers, context_output, "Transpose")
        transposed_context = transpose_node.output[0] if transpose_node is not None else None
        add_output(transposed_context)

        reshape_node = cls._find_consumer(consumers, transposed_context, "Reshape")
        reshaped_context = reshape_node.output[0] if reshape_node is not None else None
        add_output(reshaped_context)

        projected_context_node = cls._find_consumer(consumers, reshaped_context, "MatMul")
        projected_context = projected_context_node.output[0] if projected_context_node is not None else None
        add_output(projected_context)

        return output_names

    def generate_npi_file(self, onnx_path: str | Path, model_name: str | None = None) -> str:
        del model_name
        onnx_path = onnx_path or self.onnx_path
        if onnx_path is None:
            raise ValueError("ONNX path is required to generate Gemma4 NPI file.")
        onnx_path = Path(onnx_path)
        npi_path = onnx_path.with_name(f"{onnx_path.stem}_gemma4_npi.yaml")

        model = onnx.load(str(onnx_path), load_external_data=False)
        fp32_names = []

        for node in model.graph.node:
            if node.op_type in self._NPI_EXCLUDED_OPS:
                continue
            fp32_names.extend(
                out_name for out_name in node.output if out_name and not out_name.endswith("_RetainedState")
            )

        for function in model.functions:
            if "DecoderLayer" not in function.name:
                continue

            for node in function.node:
                if node.op_type in self._NPI_EXCLUDED_OPS:
                    continue
                fp32_names.extend(output_name for output_name in node.output if output_name)

        fp32_names = list(dict.fromkeys(fp32_names))
        fp32_names = [name for name in fp32_names if "MatMul" not in name]

        npi_data = {"FP32NodeInstanceNames": fp32_names}
        with open(npi_path, "w") as fp:
            yaml.safe_dump(npi_data, fp, sort_keys=False)
        return str(npi_path)

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        comp_ctx_lengths_prefill: list[int] | None = None,
        comp_ctx_lengths_decode: list[int] | None = None,
        continuous_batching: bool = False,
        kv_cache_batch_size: int | None = None,
        full_batch_size: int | None = None,
        **kwargs,
    ):
        del kwargs
        batch_size = batch_size if batch_size else 1
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size

        def build_prefill_spec(comp_ctx_lengths: int | None = None):
            spec = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": self.config.sliding_window,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            return spec

        def build_decode_spec(comp_ctx_lengths: int | None = None):
            spec = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "sliding_window": self.config.sliding_window,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            return spec

        if comp_ctx_lengths_prefill or comp_ctx_lengths_decode:
            specializations = [build_prefill_spec(length) for length in (comp_ctx_lengths_prefill or [])]
            specializations.extend(build_decode_spec(length) for length in (comp_ctx_lengths_decode or []))
            return specializations

        return [build_prefill_spec(), build_decode_spec()]

    def get_pkv_dynamic_axes(
        self,
        retain_full_kv: bool | None = False,
        continuous_batching: bool | None = False,
    ):
        del retain_full_kv
        return [
            (
                {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
                if layer_type == "sliding_attention"
                else {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
            )
            for layer_type in self.config.layer_types
        ]

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: list[int] | None = None,
        continuous_batching: bool = False,
    ):
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        if continuous_batching:
            dynamic_axes["batch_index"] = {0: "batch_size"}

        for i, ctx_axis in enumerate(self.get_pkv_dynamic_axes(continuous_batching=continuous_batching)):
            for kv in ("key", "value"):
                dynamic_axes[f"past_{kv}.{i}"] = ctx_axis

        if comp_ctx_lengths is not None:
            dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}
        return dynamic_axes

    def get_submodules_for_export(self) -> type[nn.Module]:
        return {QEffGemma4TextDecoderLayer}

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        past_key_values = []
        for layer_type in config.layer_types:
            if layer_type == "sliding_attention":
                n_heads = config.num_key_value_heads
                d_head = config.head_dim
                layer_seq_len = min(config.sliding_window, seq_len)
            else:
                use_alternative_attention = getattr(config, "attention_k_eq_v", False)
                n_heads = (
                    config.num_global_key_value_heads
                    if use_alternative_attention and getattr(config, "num_global_key_value_heads", None) is not None
                    else config.num_key_value_heads
                )
                d_head = config.global_head_dim if getattr(config, "global_head_dim", None) else config.head_dim
                layer_seq_len = seq_len
            cache_shape = [batch_size, n_heads, layer_seq_len, d_head]
            past_key_values.append(
                (
                    torch.zeros(cache_shape, dtype=config.dtype),
                    torch.zeros(cache_shape, dtype=config.dtype),
                )
            )
        return past_key_values

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del attention_mask, labels, logits_to_keep

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # DFlash TLM emits full per-position logits; the plain path keeps last-token logits.
        if not getattr(self.model, "target_layer_ids", None):
            if position_ids is not None:
                logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
                hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
            else:
                hidden_states = hidden_states[:, -1:, :]

        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        logits = logits.float()
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


class QEffGemma4DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model
        self.config = self.model.config
        self.lm_head = self.model.lm_head

    def get_submodules_for_export(self) -> type[nn.Module]:
        return {QEffGemma4TextDecoderLayer}

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        mm_token_type_ids=None,
        batch_index: torch.LongTensor | None = None,
        comp_ctx_lengths: list[int] | None = None,
        **kwargs,
    ):
        del kwargs
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.language_model.config, past_key_values)

        # Prefer multimodal token type ids when available; this is the most reliable
        # marker for image placeholder span across tokenizer/template variants.
        if mm_token_type_ids is not None and mm_token_type_ids.shape == input_ids.shape:
            special_image_mask = mm_token_type_ids == 1
        else:
            special_image_mask = input_ids == self.config.image_token_id
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = self.config.text_config.pad_token_id
        inputs_embeds = self.model.get_input_embeddings()(llm_input_ids)
        target_dtype = self.language_model.embed_tokens.weight.dtype
        if inputs_embeds.dtype != target_dtype:
            inputs_embeds = inputs_embeds.to(target_dtype)

        next_image_idx = image_idx
        if input_ids.shape[1] != 1 and special_image_mask.any() and vision_embeds is None:
            raise RuntimeError(
                "Image placeholder tokens were found in decoder input, but `vision_embeds` is missing. "
                "This indicates the vision encoder path did not run."
            )
        if vision_embeds is not None and input_ids.shape[1] != 1 and special_image_mask.any():
            if vision_embeds.dim() == 2:
                vision_embeds = vision_embeds.unsqueeze(0)
            if next_image_idx is None:
                next_image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)

            indices1 = special_image_mask.to(torch.int64).cumsum(1) - 1
            indices1 = torch.where(indices1 != -1, indices1 + next_image_idx.to(indices1.device), indices1)
            indices0 = torch.arange(special_image_mask.shape[0], device=special_image_mask.device).view(-1, 1)
            safe_indices1 = torch.where(indices1 < 0, torch.zeros_like(indices1), indices1)
            gathered_vision_embeds = vision_embeds[indices0, safe_indices1]
            if gathered_vision_embeds.dtype != target_dtype:
                gathered_vision_embeds = gathered_vision_embeds.to(target_dtype)
            inputs_embeds = torch.where(special_image_mask.unsqueeze(-1), gathered_vision_embeds, inputs_embeds)
            next_image_idx = (indices1.max() + 1).reshape(1, 1)

        attention_mask = None
        per_layer_inputs = None
        if getattr(self.language_model, "hidden_size_per_layer_input", None):
            per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, None)

        global _DISABLE_EXPORT_FP16_CLAMP
        restore_disable_clamp = _DISABLE_EXPORT_FP16_CLAMP
        if _is_onnx_export():
            _DISABLE_EXPORT_FP16_CLAMP = True
        try:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=True,
                per_layer_inputs=per_layer_inputs,
                mm_token_type_ids=mm_token_type_ids,
            )
        finally:
            _DISABLE_EXPORT_FP16_CLAMP = restore_disable_clamp
        # DFlash TLM emits full per-position logits; the plain path keeps last-token logits.
        if getattr(self.language_model, "target_layer_ids", None):
            hidden_states = outputs[0]
        else:
            logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping
        logits = logits.float()
        if next_image_idx is None:
            next_image_idx = torch.zeros((1, 1), dtype=torch.int64, device=logits.device)

        # DFlash TLM: emit the collected target-layer features as an extra output.
        if getattr(self.language_model, "target_layer_ids", None):
            return logits, vision_embeds, next_image_idx, outputs.past_key_values, outputs.hidden_states

        return logits, vision_embeds, next_image_idx, outputs.past_key_values


class QEffGemma4EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.model.vision_tower
        self.mm_tokens_per_image = getattr(
            self.model.config,
            "mm_tokens_per_image",
            getattr(self.model.config.vision_config, "default_output_length", 280),
        )

    def get_submodules_for_export(self) -> type[nn.Module]:
        return {self.model.model.vision_tower.encoder.layers[0].__class__}

    def forward(self, pixel_values, image_position_ids):
        vision_tower = self.model.model.vision_tower
        padding_positions = (image_position_ids == -1).all(dim=-1)
        inputs_embeds = vision_tower.patch_embedder(pixel_values, image_position_ids, padding_positions)

        valid_tokens = ~padding_positions

        vision_attention_mask = torch.where(
            valid_tokens.unsqueeze(1).unsqueeze(2),
            torch.zeros((), dtype=inputs_embeds.dtype, device=inputs_embeds.device),
            _attention_mask_min(inputs_embeds.dtype, inputs_embeds.device),
        )

        vision_attention_mask = vision_attention_mask.expand(-1, 1, inputs_embeds.shape[1], -1)

        hidden_states = inputs_embeds
        position_embeddings = vision_tower.encoder.rotary_emb(hidden_states, image_position_ids)
        for layer in vision_tower.encoder.layers[: vision_tower.encoder.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=vision_attention_mask,
                position_embeddings=position_embeddings,
                position_ids=image_position_ids,
            )

        output_length = getattr(vision_tower.config, "default_output_length", None)
        if output_length is None:
            output_length = pixel_values.shape[-2] // (
                vision_tower.config.pooling_kernel_size * vision_tower.config.pooling_kernel_size
            )
        hidden_states, pooler_mask = vision_tower.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=image_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        if vision_tower.config.standardize:
            hidden_states = (hidden_states - vision_tower.std_bias) * vision_tower.std_scale

        vision_embeds = self.model.model.embed_vision(inputs_embeds=hidden_states)
        if vision_embeds.dim() == 2:
            vision_embeds = vision_embeds.unsqueeze(0)

        # Keep the encoder output fixed-shape for dual-QPC export/compile.
        # Gemma4 uses vision_config.default_output_length/image_seq_length
        # image placeholders (280), while the vision pooler may emit extra
        # padded bins for the max-patch canvas.
        del pooler_mask
        return vision_embeds[:, : self.mm_tokens_per_image, :]


class QEffGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    _VISION_NPI_FP32_OPS = {"Add", "CustomRMSNorm"}
    _NPI_FP32_OPS = QEffGemma4ForCausalLM._NPI_FP32_OPS
    _NPI_SEMANTIC_NAMES = QEffGemma4ForCausalLM._NPI_SEMANTIC_NAMES
    _NPI_ATTENTION_NAMES = QEffGemma4ForCausalLM._NPI_ATTENTION_NAMES
    _NPI_BAD_OUTPUT_TOKENS = QEffGemma4ForCausalLM._NPI_BAD_OUTPUT_TOKENS
    _NPI_EXCLUDED_OPS = QEffGemma4ForCausalLM._NPI_EXCLUDED_OPS

    def _get_vision_max_patches(self) -> int:
        pooling_kernel_size = getattr(self.config.vision_config, "pooling_kernel_size", 3)
        default_output_length = getattr(self.config.vision_config, "default_output_length", 280)
        return default_output_length * pooling_kernel_size * pooling_kernel_size

    def _get_mm_tokens_per_image(self) -> int:
        return getattr(
            self.config, "mm_tokens_per_image", getattr(self.config.vision_config, "default_output_length", 280)
        )

    def get_qeff_vision_encoder(self):
        return QEffGemma4EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffGemma4DecoderWrapper(self)

    def generate_npi_file(self, onnx_path: str | Path, model_name: str | None = None) -> str:
        return QEffGemma4ForCausalLM.generate_npi_file(self, onnx_path, model_name)

    def generate_vision_npi_file(self, onnx_path: str | Path, model_name: str | None = None) -> str:
        del model_name
        onnx_path = Path(onnx_path)
        npi_path = onnx_path.with_name(f"{onnx_path.stem}_gemma4_vision_npi.yaml")
        model = onnx.load(str(onnx_path), load_external_data=False)
        fp32_names = []
        for node in model.graph.node:
            if node.op_type not in self._VISION_NPI_FP32_OPS:
                continue
            fp32_names.extend(output_name for output_name in node.output if output_name)

        npi_data = {"FP32NodeInstanceNames": list(dict.fromkeys(fp32_names))}
        with open(npi_path, "w") as fp:
            yaml.safe_dump(npi_data, fp, sort_keys=False)
        return str(npi_path)

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        comp_ctx_lengths_prefill: list[int] | None = None,
        comp_ctx_lengths_decode: list[int] | None = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: int | None = None,
        full_batch_size: int | None = None,
        vision_batch_size: int | None = None,
        **compiler_options,
    ):
        # Preserve the legacy shape when callers do not request a separate vision batch.
        vision_batch_size = batch_size if vision_batch_size is None else vision_batch_size
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        max_patches = self._get_vision_max_patches()
        user_vision_size = compiler_options.pop("vision_size", None)
        if user_vision_size:
            if user_vision_size >= ctx_len:
                raise ValueError("vision_size must be less than ctx_len")
            vision_size = user_vision_size
        else:
            vision_size = self._get_mm_tokens_per_image()

        vision = [{"vision_batch_size": vision_batch_size, "max_patches": max_patches}]

        def build_lang_prefill_spec(comp_ctx_lengths: int | None = None):
            spec = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": self.model.language_model.config.sliding_window,
                "vision_batch_size": vision_batch_size,
                "vision_size": vision_size,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size or batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size or batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            return spec

        def build_lang_decode_spec(comp_ctx_lengths: int | None = None):
            spec = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": compiler_options.pop("dflash_block_size", None) or "1",
                "ctx_len": ctx_len,
                "sliding_window": self.model.language_model.config.sliding_window,
                "vision_batch_size": vision_batch_size,
                "vision_size": vision_size,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size or batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size or batch_size
            return spec

        if comp_ctx_lengths_prefill or comp_ctx_lengths_decode:
            lang = [build_lang_prefill_spec(length) for length in (comp_ctx_lengths_prefill or [])]
            lang.extend(build_lang_decode_spec(length) for length in (comp_ctx_lengths_decode or []))
        else:
            lang = [build_lang_prefill_spec(), build_lang_decode_spec()]
        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options
        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: list[int] | None = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        vision_dynamic_axes = {
            "pixel_values": {0: "vision_batch_size", 1: "max_patches"},
            "image_position_ids": {0: "vision_batch_size", 1: "max_patches"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "mm_token_type_ids": {0: "batch_size", 1: "seq_len"},
        }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        for i in range(self.model.language_model.config.num_hidden_layers):
            layer_type = self.model.language_model.config.layer_types[i]
            if layer_type == "sliding_attention":
                ctx_axis = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
            else:
                ctx_axis = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
            for kv in ("key", "value"):
                lang_dynamic_axes[f"past_{kv}.{i}"] = ctx_axis

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}
        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}
        return {**vision_dynamic_axes, **lang_dynamic_axes}

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits", "vision_embeds_RetainedState", "image_idx_output"]
        for i in range(self.model.language_model.config.num_hidden_layers):
            for kv in ("key", "value"):
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")
        # DFlash TLM: extra collected target-layer features output.
        if getattr(self.model.language_model, "target_layer_ids", None):
            lang_output_names.append("target_hidden_states")
        if kv_offload:
            return {"vision": vision_output_names, "lang": lang_output_names}
        return lang_output_names

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        past_key_values = []
        for i, layer_type in enumerate(config.layer_types):
            if layer_type == "sliding_attention":
                n_heads = config.num_key_value_heads
                d_head = config.head_dim
                layer_seq_len = min(config.sliding_window, seq_len)
            else:
                use_alternative_attention = getattr(config, "attention_k_eq_v", False)
                n_heads = (
                    config.num_global_key_value_heads
                    if use_alternative_attention and getattr(config, "num_global_key_value_heads", None) is not None
                    else config.num_key_value_heads
                )
                d_head = config.global_head_dim if getattr(config, "global_head_dim", None) else config.head_dim
                layer_seq_len = seq_len
            cache_shape = [batch_size, n_heads, layer_seq_len, d_head]
            past_key_values.append(
                (
                    torch.zeros(cache_shape, dtype=config.dtype),
                    torch.zeros(cache_shape, dtype=config.dtype),
                )
            )
        return past_key_values

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: list[int] | None = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        seq_len = kwargs.get("prefill_seq_len", None)
        bs = kwargs.get("batch_size", None)
        if seq_len is None:
            seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        seq_len = int(seq_len)
        if bs is None:
            bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        bs = int(bs)
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        max_patches = self._get_vision_max_patches()
        mm_tokens_per_image = self._get_mm_tokens_per_image()
        patch_dim = getattr(self.config.vision_config, "patch_size", 16) ** 2 * 3

        image_position_ids = torch.full((bs, max_patches, 2), -1, dtype=torch.int64)
        pooled_side = int(mm_tokens_per_image**0.5)
        patch_side = pooled_side * getattr(self.config.vision_config, "pooling_kernel_size", 3)
        xs = torch.arange(patch_side, dtype=torch.int64).view(1, -1).expand(patch_side, -1).reshape(-1)
        ys = torch.arange(patch_side, dtype=torch.int64).view(-1, 1).expand(-1, patch_side).reshape(-1)
        valid_positions = torch.stack((xs, ys), dim=-1)
        image_position_ids[:, : valid_positions.shape[0], :] = valid_positions.unsqueeze(0)

        input_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        mm_token_type_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        text_prefix_len = min(5, seq_len)
        image_start = text_prefix_len
        image_end = min(image_start + mm_tokens_per_image, seq_len)
        input_ids[:, image_start:image_end] = self.config.image_token_id
        mm_token_type_ids[:, image_start:image_end] = 1

        vision_inputs = {
            "pixel_values": torch.zeros((bs, max_patches, patch_dim), dtype=self.config.dtype),
            "image_position_ids": image_position_ids,
        }
        lang_inputs = {
            "input_ids": input_ids,
            "vision_embeds": torch.zeros(
                (bs, mm_tokens_per_image, self.model.language_model.config.hidden_size), dtype=self.config.dtype
            ),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "mm_token_type_ids": mm_token_type_ids,
            "past_key_values": self.get_dummy_pkv_cache(
                config=self.model.language_model.config,
                batch_size=fbs if continuous_batching else bs,
                seq_len=seq_len,
            ),
        }
        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)
        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)
        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        return {**vision_inputs, **lang_inputs}
