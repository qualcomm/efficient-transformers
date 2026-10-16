# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from qwen_vl_utils import smart_resize
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    BaseModelOutputWithPooling,
    Qwen3_5Attention,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionAttention,
    Qwen3_5VisionModel,
    apply_rotary_pos_emb_vision,
    create_recurrent_attention_mask,
    repeat_kv,
    rotate_half,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    blocked_gdn_decode_forward,
    generic_blocked_attention_interface,
    get_gdn_num_head_blocks,
    recurrent_gdn_decode_forward,
)
from QEfficient.customop import (
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import (
    QEffDynamicLayer,
)
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.models._layerwise import (
    is_last_layer_window,
    is_layerwise_active,
    resolve_layer_window,
)
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")
QWEN3_5_ROPE_CACHE_EXPORT_CAP = 76800


def _expand_mrope_position_ids(position_ids, cache_position, batch_size):
    if position_ids is None:
        return cache_position.view(1, 1, -1).expand(4, batch_size, -1)
    if position_ids.ndim == 1:
        return position_ids.view(1, 1, -1).expand(4, batch_size, -1)
    if position_ids.ndim == 2:
        return position_ids.unsqueeze(0).expand(4, position_ids.shape[0], -1)
    return position_ids


def _batch_index_scatter(tensor: torch.Tensor, batch_index: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """Place logical request rows into their physical batch slots."""
    batch_first = tensor if batch_dim == 0 else tensor.transpose(0, batch_dim)
    slots = batch_index.reshape(-1).long()
    batch_first = torch.zeros_like(batch_first).index_put((slots,), batch_first, accumulate=False)
    return batch_first if batch_dim == 0 else batch_first.transpose(0, batch_dim)


def _batch_index_gather(tensor: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """Restore physical-slot rows to logical request order."""
    return tensor.index_select(0, batch_index.reshape(-1).long())


class QEffQwen3_5GatedDeltaNetCustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def forward(self, hidden_states, gate):
        normed = CustomRMSNormFunc.apply(
            hidden_states,
            self.weight.to(hidden_states.dtype),
            self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps,
        )
        # silu is computed in float32 for numerical parity, then cast back so the
        # gated output keeps the module dtype (e.g. float16) and matches out_proj.
        return normed * F.silu(gate.to(torch.float32)).to(normed.dtype)


class QEffQwen3_5DynamicCache(Cache):
    """
    Hybrid cache for Qwen3.5 models.

    Full-attention layers retain KV cache, while linear-attention layers retain
    convolution and recurrent states.
    """

    def __init__(self, config):
        super().__init__(layers=[])
        self.config = config
        self.layer_types = list(config.layer_types)
        self.transformer_layers = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]
        self.last_linear_layer = next(
            (i for i in range(len(self.layer_types) - 1, -1, -1) if self.layer_types[i] == "linear_attention"),
            None,
        )
        self.kv_layers = [
            QEffDynamicLayer() if layer_type == "full_attention" else None for layer_type in self.layer_types
        ]
        self.conv_states = [None for _ in self.layer_types]
        self.recurrent_states = [None for _ in self.layer_types]

    @classmethod
    def from_legacy_cache(
        cls,
        config,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,
    ) -> "QEffQwen3_5DynamicCache":
        cache = cls(config)
        if past_key_values is None:
            return cache

        if not is_layerwise_active():
            # Default path: restore every layer, matching pre-layerwise behavior.
            for layer_idx, layer_state in enumerate(past_key_values):
                if cache.layer_types[layer_idx] == "full_attention":
                    key_states, value_states = layer_state
                    layer = QEffDynamicLayer()
                    layer.keys = key_states
                    layer.values = value_states
                    cache.kv_layers[layer_idx] = layer
                else:
                    conv_state, recurrent_state = layer_state
                    cache.conv_states[layer_idx] = conv_state
                    cache.recurrent_states[layer_idx] = recurrent_state
            return cache

        layer_idx = QEffQwen3_5TextModel._start
        layer_state = past_key_values
        if len(past_key_values) == len(cache.layer_types) and isinstance(past_key_values[layer_idx], (tuple, list)):
            layer_state = past_key_values[layer_idx]
        elif len(past_key_values) == 1 and isinstance(past_key_values[0], (tuple, list)):
            layer_state = past_key_values[0]
        if cache.layer_types[layer_idx] == "full_attention":
            key_states, value_states = layer_state
            layer = QEffDynamicLayer()
            layer.keys = key_states
            layer.values = value_states
            cache.kv_layers[layer_idx] = layer
        else:
            conv_state, recurrent_state = layer_state
            cache.conv_states[layer_idx] = conv_state
            cache.recurrent_states[layer_idx] = recurrent_state
        return cache

    def __len__(self):
        return len(self.layer_types)

    def __getitem__(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(layer_idx, slice):
            return tuple(self[idx] for idx in range(*layer_idx.indices(len(self.layer_types))))
        if not isinstance(layer_idx, int):
            raise TypeError(f"Unsupported cache index type: {type(layer_idx)!r}")
        if layer_idx < 0:
            layer_idx += len(self.layer_types)
        if self.layer_types[layer_idx] == "full_attention":
            layer = self.kv_layers[layer_idx]
            return (layer.keys, layer.values)
        return (self.conv_states[layer_idx], self.recurrent_states[layer_idx])

    @property
    def key_cache(self):
        return [None if layer is None else layer.keys for layer in self.kv_layers]

    @property
    def value_cache(self):
        return [None if layer is None else layer.values for layer in self.kv_layers]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position: Optional[torch.LongTensor] = None) -> int:
        del cache_position
        if not self.transformer_layers:
            return 0
        if layer_idx not in self.transformer_layers:
            layer_idx = self.transformer_layers[0]
        layer = self.kv_layers[layer_idx]
        return 0 if layer is None or layer.keys is None else layer.keys.shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        return query_length + past_seen_tokens, kv_offset

    def read_only_blocked_kv(self, start_index: int, end_index: int, layer_idx: int, cache_kwargs: dict):
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.read_only_blocked_kv(start_index, end_index, cache_kwargs)

    def read_only_blocked_K(self, start_index: int, end_index: int, layer_idx: int, cache_kwargs: dict):
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.read_only_blocked_K(start_index, end_index, cache_kwargs)

    def read_only_blocked_V(self, start_index: int, end_index: int, layer_idx: int, cache_kwargs: dict):
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.read_only_blocked_V(start_index, end_index, cache_kwargs)

    def get_batch_folded_kv(self, layer_idx: int):
        return self.kv_layers[layer_idx].get_batch_folded_kv()

    def read_only_blocked_K_batch(
        self, start_index: int, end_index: int, layer_idx: int, cache_kwargs: dict, folded_cache=None
    ):
        return self.kv_layers[layer_idx].read_only_blocked_K_batch(
            start_index, end_index, cache_kwargs, folded_cache=folded_cache
        )

    def read_only_blocked_V_batch(
        self, start_index: int, end_index: int, layer_idx: int, cache_kwargs: dict, folded_cache=None
    ):
        return self.kv_layers[layer_idx].read_only_blocked_V_batch(
            start_index, end_index, cache_kwargs, folded_cache=folded_cache
        )

    def write_only(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: dict):
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.write_only(key_states, value_states, cache_kwargs)

    def write_only_batch(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: dict
    ):
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.write_only_batch(key_states, value_states, cache_kwargs)

    def has_previous_state(self, layer_idx=None) -> bool:
        if layer_idx is not None:
            if layer_idx < 0 or layer_idx >= len(self.layer_types):
                return False
            if self.layer_types[layer_idx] != "linear_attention":
                return False
            return self.conv_states[layer_idx] is not None

        # Layerwise path only materializes the active layer state.
        if is_layerwise_active():
            active_idx = QEffQwen3_5TextModel._start
            if 0 <= active_idx < len(self.layer_types) and self.layer_types[active_idx] == "linear_attention":
                return self.conv_states[active_idx] is not None

        if self.last_linear_layer is None:
            return False
        return self.conv_states[self.last_linear_layer] is not None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                layer = self.kv_layers[layer_idx]
                if layer is not None and layer.keys is not None:
                    device = layer.keys.device
                    beam_idx_device = beam_idx.to(device)
                    layer.keys = layer.keys.index_select(0, beam_idx_device)
                    layer.values = layer.values.index_select(0, beam_idx_device)
            elif self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx_device = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx_device)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx_device)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        legacy_cache = ()
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                layer = self.kv_layers[layer_idx]
                if layer is None or layer.keys is None:
                    legacy_cache += ((torch.empty(0), torch.empty(0)),)
                else:
                    legacy_cache += ((layer.keys, layer.values),)
            else:
                conv_state = self.conv_states[layer_idx]
                recurrent_state = self.recurrent_states[layer_idx]
                legacy_cache += (
                    (
                        torch.empty(0) if conv_state is None else conv_state,
                        torch.empty(0) if recurrent_state is None else recurrent_state,
                    ),
                )
        return legacy_cache


class QEffQwen3_5TextRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    """
    QEff wrapper for Qwen3.5 text RoPE.

    Similar to Qwen3, this precomputes a reusable base cache and then indexes it
    with the current 3D RoPE position ids before applying the Qwen3.5 MRoPE
    interleaving pattern.
    """

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device)
        cached_seq_len = min(int(self.original_max_seq_len), QWEN3_5_ROPE_CACHE_EXPORT_CAP)
        self._set_cos_sin_cache(
            seq_len=cached_seq_len,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
            self.sin_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
        )


def qeff_apply_interleaved_mrope(freqs, mrope_section):
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity.
    args:
        x: (3, bs, seq_len, head_dim // 2)
        mrope_section: (3,)
    returns:
        x_t: (bs, seq_len, head_dim // 2)
    """

    half_shape = freqs[0].shape[-1] // 2
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
        offset += half_shape
        length += half_shape
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def qeff_prepare_mrope_cos_sin(cos, sin, position_ids, mrope_section, dtype=None):
    invalid_pos_mask = position_ids < 0
    safe_position_ids = torch.where(invalid_pos_mask, torch.zeros_like(position_ids), position_ids)
    flat_pos = safe_position_ids.reshape(-1)
    cos = cos.index_select(0, flat_pos).reshape(*safe_position_ids.shape, cos.shape[-1])
    sin = sin.index_select(0, flat_pos).reshape(*safe_position_ids.shape, sin.shape[-1])
    cos = qeff_apply_interleaved_mrope(cos, mrope_section).unsqueeze(1)
    sin = qeff_apply_interleaved_mrope(sin, mrope_section).unsqueeze(1)
    if dtype is not None:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return cos, sin


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, mrope_section=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
        cos = qeff_apply_interleaved_mrope(cos, mrope_section)
        sin = qeff_apply_interleaved_mrope(sin, mrope_section)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[:, :, :, :rotary_dim], q[:, :, :, rotary_dim:]
    k_rot, k_pass = k[:, :, :, :rotary_dim], k[:, :, :, rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=module.config.torch_dtype), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def qeff_torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    position_ids: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, hidden_size, seq_len = hidden_states.shape
    grouped_conv_state = conv_state.ndim == 4

    if grouped_conv_state:
        bsz, num_kv_heads, conv_group_dim, state_len = conv_state.shape
        if hidden_size != num_kv_heads * conv_group_dim:
            raise ValueError(
                "Grouped conv_state shape mismatch: "
                f"hidden_size={hidden_size}, num_kv_heads={num_kv_heads}, conv_group_dim={conv_group_dim}"
            )
        conv_state_flat = conv_state.reshape(bsz, hidden_size, state_len)
    else:
        conv_state_flat = conv_state

    state_len = conv_state_flat.shape[-1]
    pos_ids = position_ids[0]
    hidden_states_new = torch.cat([conv_state_flat, hidden_states], dim=-1).to(weight.dtype)

    is_decode = seq_len == torch.tensor(1)

    # Decode is a fixed shift. Keep this branch tensor-only so the compiler can
    # select it for both BS1 and folded decode graphs.
    shifted_conv_state = torch.cat([conv_state_flat[..., 1:], hidden_states_new[..., -1:]], dim=-1)
    valid_decode = (pos_ids[:, -1:] >= 0).to(shifted_conv_state.dtype).reshape(-1, 1, 1)
    shifted_conv_state = shifted_conv_state * valid_decode + conv_state_flat * (1 - valid_decode)

    # Prefill and padded chunks retain the general last-valid-position gather.
    # Derive zeros from an existing state view; torch.zeros emits a dynamic ConstantOfShape in subfunctions.
    zeros = conv_state_flat[:, 0, :].to(pos_ids.dtype) * 0
    order = torch.argsort(torch.cat([zeros, pos_ids], dim=1), dim=1)
    last_positions = order[:, -state_len:]
    ctx_idx = last_positions.to(torch.long).unsqueeze(1).expand(-1, hidden_size, -1)
    gathered_conv_state = torch.gather(hidden_states_new, dim=2, index=ctx_idx)

    if grouped_conv_state:
        shifted_conv_state = shifted_conv_state.reshape(bsz, num_kv_heads, conv_group_dim, state_len)
        gathered_conv_state = gathered_conv_state.reshape(bsz, num_kv_heads, conv_group_dim, state_len)
    updated_conv_state = torch.where(is_decode, shifted_conv_state, gathered_conv_state)

    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:]).to(hidden_states.dtype)
    return out, updated_conv_state


class QEffQwen3_5Attention(Qwen3_5Attention):
    """
    Full-attention path with QEff cache updates for retained-state export.
    """

    def __qeff_init__(self):
        # RoPE tables are prepared once in the text model and passed down.
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[QEffQwen3_5DynamicCache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_and_gate = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        # Static slicing avoids ONNX SplitToSequence from torch.chunk when
        # layerwise prefill disables canonicalization passes.
        query_states = query_and_gate[..., : self.head_dim]
        gate = query_and_gate[..., self.head_dim :].reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("`position_embeddings` must be provided for QEffQwen3_5Attention.")
        cos, sin = position_embeddings
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        use_blocking = (
            past_key_values is not None and blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)
        )

        if use_blocking:
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
                comp_ctx_length=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids[0],
                past_seen_tokens=past_seen_tokens,
            )
        else:
            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "batch_index": batch_index,
                    "position_ids": position_ids[0],
                }
                if comp_ctx_lengths is not None:
                    attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                    cache_kwargs["CCL"] = attention_mask.shape[-1]
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """
    Linear-attention path with explicit conv/recurrent retained-state updates.
    """

    def __qeff_init__(self):
        self.chunk_gated_delta_rule = self.torch_chunk_gated_delta_rule_qeff
        self.torch_dtype = self.out_proj.weight.dtype
        self.chunk_gated_delta_solver = "tree"
        chunk_size = int(getattr(self, "gdn_chunk_size", 64) or 64)
        if chunk_size <= 0:
            chunk_size = 64
        self.gdn_chunk_size = chunk_size

        # Precompute all constant masks — no triu/tril with diagonal args at runtime
        # mask_causal: upper triangular including diagonal (diagonal=0)
        # = triu(ones, diagonal=0)

        mask_causal = torch.zeros(chunk_size, chunk_size, dtype=torch.bool)
        for i in range(chunk_size):
            for j in range(i, chunk_size):
                mask_causal[i, j] = True
        self.register_buffer("_mask_causal", mask_causal, persistent=False)

        # shape: (C, C), True above diagonal inclusive

        # mask_strict: strict upper triangular (diagonal=1)
        # = triu(ones, diagonal=1)
        mask_strict = torch.zeros(chunk_size, chunk_size, dtype=torch.bool)
        for i in range(chunk_size):
            for j in range(i + 1, chunk_size):
                mask_strict[i, j] = True
        self.register_buffer("_mask_strict", mask_strict, persistent=False)
        # shape: (C, C), True strictly above diagonal

        # ones_lower: lower triangular all-ones for cumsum replacement
        # = tril(ones, diagonal=0)
        ones_lower = torch.zeros(chunk_size, chunk_size)
        for i in range(chunk_size):
            for j in range(i + 1):
                ones_lower[i, j] = 1.0
        self.register_buffer("_ones_lower", ones_lower, persistent=False)
        # shape: (C, C)

        # eye: identity matrix
        self.register_buffer("_eye", torch.eye(chunk_size), persistent=False)

    # TODO: It would be better to use it directly from HF
    def _solve_chunk_attn_original(self, attn: torch.Tensor, mask: torch.Tensor, eye: torch.Tensor, chunk_size: int):
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(dim=-2)
        return attn + eye.to(dtype=attn.dtype)

    def _solve_chunk_attn_tree(self, attn: torch.Tensor, mask: torch.Tensor, eye: torch.Tensor, chunk_size: int):
        if chunk_size <= 0 or (chunk_size & (chunk_size - 1)) != 0:
            raise ValueError(f"tree solver requires power-of-two chunk_size, got {chunk_size}")

        source = attn.masked_fill(mask, 0)
        # Build the inverse of (I - source) bottom-up from 1x1 unit blocks.
        # Starting from full chunk_size x chunk_size blocks causes shape mismatches
        # in the first merge (block_size=1).
        unit = torch.ones_like(source[..., :1, :1])
        blocks = [unit for _ in range(chunk_size)]
        block_size = 1

        while len(blocks) > 1:
            merged = []
            for pair in range(0, len(blocks), 2):
                left = blocks[pair]
                right = blocks[pair + 1]
                row0 = pair * block_size
                row1 = row0 + block_size
                lower_left = source[..., row1 : row1 + block_size, row0 : row0 + block_size]
                solved_lower_left = right @ lower_left @ left
                zeros = torch.zeros_like(left)
                merged.append(
                    torch.cat(
                        [
                            torch.cat([left, zeros], dim=-1),
                            torch.cat([solved_lower_left, right], dim=-1),
                        ],
                        dim=-2,
                    )
                )
            blocks = merged
            block_size *= 2

        return blocks[0].to(attn.dtype)

    def _solve_chunk_attn_recursive_sns(
        self, attn: torch.Tensor, mask: torch.Tensor, eye: torch.Tensor, chunk_size: int
    ):
        strict_lower = (~mask).view(1, 1, 1, chunk_size, chunk_size)
        acc_dtype = attn.dtype
        I64 = eye.to(dtype=acc_dtype).view(1, 1, 1, chunk_size, chunk_size)
        A64 = attn.masked_fill(mask, 0).to(acc_dtype)
        ns_iters = int(math.log2(chunk_size)) + 4

        def _ns_solve(L: torch.Tensor) -> torch.Tensor:
            X = I64.clone()
            for _ in range(ns_iters):
                R = I64 - ((I64 - L) @ X)
                ck = X + (X @ R).masked_fill(~strict_lower, 0)
                X = ck.masked_fill(~strict_lower, 0) + I64
            return X

        default_depth = max(1, (math.ceil(math.log2(chunk_size)) - 1) // 2)
        depth = getattr(self, "recursive_sns_depth", None)
        depth = default_depth if depth is None else int(depth)
        depth = max(1, depth)

        X = _ns_solve(A64 * (0.5**depth))
        for _ in range(depth):
            M = (X - I64).masked_fill(~strict_lower, 0)
            Z = _ns_solve(M)
            X = (Z @ X).masked_fill(~strict_lower, 0) + I64
        return X.to(attn.dtype)

    def _solve_chunk_attn_scaled_newton_schulz(
        self, attn: torch.Tensor, mask: torch.Tensor, eye: torch.Tensor, chunk_size: int
    ):
        strict_lower = (~mask).view(1, 1, 1, chunk_size, chunk_size)
        attn_1 = attn.clone()
        acc_dtype = attn.dtype
        I64 = eye.to(dtype=acc_dtype).view(1, 1, 1, chunk_size, chunk_size)
        A64 = attn_1.masked_fill(mask, 0).to(acc_dtype)
        ns_iters = int(math.log2(chunk_size)) + 4
        As = 0.5 * A64

        Xs = I64.clone()
        for _ in range(ns_iters):
            Rs = I64 - ((I64 - As) @ Xs)
            ck = Xs + (Xs @ Rs).masked_fill(~strict_lower, 0)
            Xs = ck.masked_fill(~strict_lower, 0) + I64

        M = (Xs - I64).masked_fill(~strict_lower, 0)
        Z = I64.clone()
        for _ in range(ns_iters):
            Rz = I64 - ((I64 - M) @ Z)
            ck = Z + (Z @ Rz).masked_fill(~strict_lower, 0)
            Z = ck.masked_fill(~strict_lower, 0) + I64

        X = (Z @ Xs).masked_fill(~strict_lower, 0) + I64
        return X.to(attn.dtype)

    def _solve_chunk_attn_factorized(self, attn: torch.Tensor, eye: torch.Tensor):
        eye = eye.to(dtype=attn.dtype)
        L = eye.clone()
        Apow = attn
        K = 32
        for _ in range(int(math.log2(K))):
            L = L @ (eye + Apow)
            Apow = Apow @ Apow
        return L

    def _solve_chunk_attn_horner(self, attn: torch.Tensor, mask: torch.Tensor, eye: torch.Tensor, chunk_size: int):
        A = attn.masked_fill(mask, 0)
        acc_dtype = torch.float32
        A64 = A.to(acc_dtype)
        I64 = eye.to(dtype=acc_dtype).view(1, 1, 1, chunk_size, chunk_size)
        strict_lower = (~mask).view(1, 1, 1, chunk_size, chunk_size)
        K = chunk_size - 1
        S64 = I64.clone()
        for _ in range(K):
            S64 = I64 + (A64 @ S64).masked_fill(~strict_lower, 0)
        return S64.to(A.dtype)

    def _solve_chunk_attn(self, attn: torch.Tensor, mask: torch.Tensor, eye: torch.Tensor, chunk_size: int):
        solver = getattr(self, "chunk_gated_delta_solver", "recursive_sns")
        if solver == "original":
            return self._solve_chunk_attn_original(attn, mask, eye, chunk_size)
        if solver in ("tree", "forward_sub_tree"):
            return self._solve_chunk_attn_tree(attn, mask, eye, chunk_size)
        if solver == "factorized":
            return self._solve_chunk_attn_factorized(attn, eye)
        if solver == "horner":
            return self._solve_chunk_attn_horner(attn, mask, eye, chunk_size)
        if solver == "scaled_newton_schulz":
            return self._solve_chunk_attn_scaled_newton_schulz(attn, mask, eye, chunk_size)
        return self._solve_chunk_attn_recursive_sns(attn, mask, eye, chunk_size)

    def torch_chunk_gated_delta_rule_qeff(
        self,
        query,
        key,
        value,
        g,
        beta,
        position_ids,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        mask_causal=None,
        mask_strict=None,
        ones_lower=None,
        eye=None,
    ):
        initial_dtype = query.dtype
        # if use_qk_l2norm_in_kernel:
        #     query = l2norm(query, dim=-1, eps=1e-6)
        #     key = l2norm(key, dim=-1, eps=1e-6)
        if use_qk_l2norm_in_kernel:
            query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
            key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(self.torch_dtype) for x in (query, key, value, beta, g)
        ]

        mask = (position_ids[0] != -1).unsqueeze(1)

        zeros = torch.zeros(g.shape, dtype=g.dtype, device=g.device)

        g = torch.where(mask, g, zeros)
        beta = torch.where(mask, beta, zeros)

        qkv_zeros = torch.zeros(key.shape, dtype=key.dtype, device=key.device)
        key = torch.where(mask.unsqueeze(-1), key, qkv_zeros)
        query = torch.where(mask.unsqueeze(-1), query, qkv_zeros)
        value = torch.where(mask.unsqueeze(-1), value, qkv_zeros)

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        # QAIC's LoadPad kernel only supports float32/int32/int64 inputs, so bf16/fp16
        # tensors are padded via torch.cat with a zero tensor instead of F.pad.
        query = torch.cat(
            [query, torch.zeros(*query.shape[:2], pad_size, query.shape[3], dtype=query.dtype)],
            dim=2,
        )
        key = torch.cat([key, torch.zeros(*key.shape[:2], pad_size, key.shape[3], dtype=key.dtype)], dim=2)
        value = torch.cat(
            [value, torch.zeros(*value.shape[:2], pad_size, value.shape[3], dtype=value.dtype)],
            dim=2,
        )
        beta = torch.cat([beta, torch.zeros(*beta.shape[:2], pad_size, dtype=beta.dtype)], dim=2)
        g = torch.cat([g, torch.zeros(*g.shape[:2], pad_size, dtype=g.dtype)], dim=2)
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (self.head_k_dim**0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        # reshape to chunks
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = mask_causal

        # chunk decay
        # g = g.cumsum(dim=-1)

        L = g.size(-1)
        idx = torch.arange(L, device=g.device)
        mask_g = (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(g.dtype)

        g = g @ mask_g.T

        # decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril() # original decay_mask

        diff = g.unsqueeze(-1) - g.unsqueeze(-2)  # (B, H, num_chunks, C, C)
        diff = diff * (~mask_strict).float()  # zero upper triangle (strict)
        decay_mask = diff.exp().float()
        decay_mask = (decay_mask * (~mask_strict).float()).to(self.torch_dtype)  # ensure upper is zero

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        I_base = eye if eye is not None else torch.eye(chunk_size, dtype=attn.dtype)
        attn = self._solve_chunk_attn(attn=attn, mask=mask, eye=I_base, chunk_size=chunk_size)

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = mask_strict

        # for each chunk
        for i in range(total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        if last_recurrent_state is not None:
            last_recurrent_state = last_recurrent_state.to(initial_dtype)
        return core_attn_out, last_recurrent_state

    def _recurrent_step_batched(self, query, key, value, g, beta, recurrent_state, gdn_num_head_blocks: int = 1):
        if query.shape[1] != 1:
            raise ValueError("_recurrent_step_batched is decode-only and requires sequence_length == 1")
        if gdn_num_head_blocks <= 1:
            return recurrent_gdn_decode_forward(query, key, value, g, beta, recurrent_state)
        return blocked_gdn_decode_forward(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            gdn_num_head_blocks=gdn_num_head_blocks,
        )

    def forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        position_ids=None,
        batch_index: torch.LongTensor | None = None,
        batch_fold: bool = False,
        gdn_num_head_blocks: int = 1,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        # ── Projections ──────────────────────────────────────
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        # ── Conv (unified, handles T=1 and T=N) ──────────────
        if cache_params is not None:
            conv_state_all = cache_params.conv_states[self.layer_idx]
            recurrent_state_all = cache_params.recurrent_states[self.layer_idx]

            # Continuous batching path: gather only active rows, then scatter updates back.
            if batch_index is not None:
                conv_state_grouped = conv_state_all.ndim == 4
                if conv_state_grouped:
                    conv_state_all_flat = conv_state_all.reshape(
                        conv_state_all.shape[0],
                        conv_state_all.shape[1] * conv_state_all.shape[2],
                        conv_state_all.shape[3],
                    )
                else:
                    conv_state_all_flat = conv_state_all
                conv_batch_index = batch_index.to(conv_state_all_flat.device)
                conv_ctx_indices = torch.arange(
                    conv_state_all_flat.shape[1], dtype=torch.int64, device=conv_state_all_flat.device
                )[None, :]
                conv_state = CtxGatherFuncCB3D.apply(conv_state_all_flat, conv_batch_index, conv_ctx_indices)
                if conv_state_grouped:
                    conv_state = conv_state.reshape(
                        conv_state.shape[0], conv_state_all.shape[1], conv_state_all.shape[2], conv_state_all.shape[3]
                    )

                recurrent_batch_index = batch_index.to(recurrent_state_all.device)
                recurrent_ctx_indices = torch.arange(
                    recurrent_state_all.shape[2], dtype=torch.int64, device=recurrent_state_all.device
                )[None, None, :]
                recurrent_state = CtxGatherFuncCB.apply(
                    recurrent_state_all, recurrent_batch_index, recurrent_ctx_indices, recurrent_state_all.shape[2]
                )
            else:
                conv_state = conv_state_all
                recurrent_state = recurrent_state_all

            if batch_fold and position_ids is not None:
                text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
                zero_cumsum = torch.cumsum((text_position_ids == 0).to(torch.int32), dim=1)[:, -1:]
                conv_reset_mask = zero_cumsum.to(dtype=torch.bool, device=conv_state.device).reshape(
                    conv_state.shape[0], *([1] * (conv_state.ndim - 1))
                )
                # Blend the small reset mask instead of broadcasting it through
                # a full Select/Where tensor in the folded decode graph.
                conv_keep = (~conv_reset_mask).to(conv_state.dtype)
                conv_state = conv_state * conv_keep

                recurrent_reset_mask = conv_reset_mask.reshape(
                    recurrent_state.shape[0], *([1] * (recurrent_state.ndim - 1))
                ).to(device=recurrent_state.device)
                recurrent_keep = (~recurrent_reset_mask).to(recurrent_state.dtype)
                recurrent_state = recurrent_state * recurrent_keep

            mixed_qkv, new_conv_state = qeff_torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                position_ids,
                self.conv1d.bias,
            )
            if batch_index is not None:
                if conv_state_all.ndim == 4:
                    conv_state_all_flat = conv_state_all.reshape(
                        conv_state_all.shape[0],
                        conv_state_all.shape[1] * conv_state_all.shape[2],
                        conv_state_all.shape[3],
                    )
                    new_conv_state_flat = new_conv_state.reshape(
                        new_conv_state.shape[0],
                        new_conv_state.shape[1] * new_conv_state.shape[2],
                        new_conv_state.shape[3],
                    )
                else:
                    conv_state_all_flat = conv_state_all
                    new_conv_state_flat = new_conv_state
                conv_batch_index = batch_index.to(conv_state_all_flat.device)
                conv_position_ids = torch.arange(
                    conv_state_all_flat.shape[1], dtype=torch.int64, device=conv_state_all_flat.device
                )[None, :]
                cache_params.conv_states[self.layer_idx] = CtxScatterFuncCB3D.apply(
                    conv_state_all_flat, conv_batch_index, conv_position_ids, new_conv_state_flat
                )
                if conv_state_all.ndim == 4:
                    cache_params.conv_states[self.layer_idx] = cache_params.conv_states[self.layer_idx].reshape(
                        conv_state_all.shape[0],
                        conv_state_all.shape[1],
                        conv_state_all.shape[2],
                        conv_state_all.shape[3],
                    )
            else:
                cache_params.conv_states[self.layer_idx] = new_conv_state
        else:
            recurrent_state = None
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        # ── Split Q/K/V ──────────────────────────────────────
        mixed_qkv = mixed_qkv.transpose(1, 2)
        # query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = mixed_qkv[..., : self.key_dim]
        key = mixed_qkv[..., self.key_dim : 2 * self.key_dim]
        value = mixed_qkv[..., 2 * self.key_dim :]
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # ── Recurrent State ───────────────────────────────────
        if cache_params is not None:
            if batch_fold:
                # Folded decode is exported with a static one-token sequence.
                core_attn_out, last_recurrent_state = self._recurrent_step_batched(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    recurrent_state,
                    gdn_num_head_blocks=gdn_num_head_blocks,
                )
            else:
                # General graphs serve both prefill and decode; select the
                # matching branch at runtime after export.
                recurrent_out, recurrent_state_new = recurrent_gdn_decode_forward(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    recurrent_state,
                )
                chunk_out, chunk_state = self.chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    position_ids=position_ids,
                    chunk_size=self.gdn_chunk_size,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    mask_causal=self._mask_causal,
                    mask_strict=self._mask_strict,
                    ones_lower=self._ones_lower,
                    eye=self._eye,
                )
                is_decode = hidden_states.shape[1] == torch.tensor(1)
                core_attn_out = torch.where(is_decode, recurrent_out, chunk_out)
                last_recurrent_state = torch.where(is_decode, recurrent_state_new, chunk_state)

            if batch_index is not None:
                recurrent_batch_index = batch_index.to(recurrent_state_all.device)
                recurrent_position_ids = torch.arange(
                    recurrent_state_all.shape[2], dtype=torch.int64, device=recurrent_state_all.device
                )[None, :].expand(recurrent_batch_index.shape[0], -1)
                cache_params.recurrent_states[self.layer_idx] = CtxScatterFuncCB.apply(
                    recurrent_state_all,
                    recurrent_batch_index,
                    recurrent_position_ids,
                    last_recurrent_state.to(recurrent_state_all.dtype),
                )
            else:
                cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        else:
            # No cache — prefill only, no state needed
            core_attn_out, _ = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=self.gdn_chunk_size,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                mask_causal=self._mask_causal,
                mask_strict=self._mask_strict,
                ones_lower=self._ones_lower,
                eye=self._eye,
            )

        # ── Output ────────────────────────────────────────────
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1).to(self.out_proj.weight.dtype)
        return self.out_proj(core_attn_out)

    @staticmethod
    def apply_mask_to_padding_states(hidden_states, attention_mask):
        if attention_mask is not None and attention_mask.shape[1] > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        return hidden_states


class QEffQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def __qeff_init__(self):
        layer_type = getattr(self, "layer_type", getattr(self, "block_type", None))
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn.__class__ = QEffQwen3_5GatedDeltaNet
            self.linear_attn.__qeff_init__()
        elif layer_type == "full_attention":
            self.self_attn.__class__ = QEffQwen3_5Attention
            self.self_attn.__qeff_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: QEffQwen3_5DynamicCache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        batch_fold: bool = False,
        gdn_num_head_blocks: int = 1,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        del use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
                batch_index=batch_index,
                batch_fold=batch_fold,
                gdn_num_head_blocks=gdn_num_head_blocks,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffQwen3_5TextModel(Qwen3_5TextModel):
    _start = 0
    _end = 0
    _total_layers = None

    def __qeff_init__(self):
        self.rotary_emb = QEffQwen3_5TextRotaryEmbedding(config=self.config)
        rope_rows = min(int(self.rotary_emb.sin_cached.shape[0]), QWEN3_5_ROPE_CACHE_EXPORT_CAP)
        self.sin_cached = torch.nn.Parameter(
            (self.rotary_emb.sin_cached[:rope_rows] * self.rotary_emb.attention_scaling).contiguous()
        )
        self.cos_cached = torch.nn.Parameter(
            (self.rotary_emb.cos_cached[:rope_rows] * self.rotary_emb.attention_scaling).contiguous()
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: QEffQwen3_5DynamicCache | tuple[tuple[torch.FloatTensor, ...], ...] | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        batch_fold: bool = False,
        gdn_num_head_blocks: int = 1,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_legacy_cache = False

        if past_key_values is not None and not isinstance(past_key_values, QEffQwen3_5DynamicCache):
            return_legacy_cache = True
            past_key_values = QEffQwen3_5DynamicCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffQwen3_5DynamicCache(self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        start, end = resolve_layer_window(QEffQwen3_5TextModel, len(self.layers))
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(layer_idx=start) if past_key_values is not None else 0
            active_layer_type = self.config.layer_types[start] if start < len(self.config.layer_types) else None
            if is_layerwise_active() and position_ids is not None and active_layer_type == "linear_attention":
                text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
                cache_position = text_position_ids[0].clamp_min(0)
            else:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
        position_ids = _expand_mrope_position_ids(position_ids, cache_position, inputs_embeds.shape[0])
        text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
        rotary_position_ids = (
            position_ids[1:] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else position_ids
        )

        if isinstance(attention_mask, torch.Tensor):
            target_length = attention_mask.shape[-1]
        else:
            pos_max = 0
            if position_ids is not None:
                text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
                pos_max = int(text_position_ids.max().item()) + 1
            target_length = max(past_seen_tokens, pos_max)
        causal_mask = _create_causal_mask(
            position_ids=text_position_ids, target_length=target_length, sliding_window=None
        )
        linear_attn_mask = create_recurrent_attention_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds

        rope_parameters = getattr(self.config, "rope_parameters", {}) or {}
        mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
        cos, sin = qeff_prepare_mrope_cos_sin(
            self.cos_cached, self.sin_cached, rotary_position_ids, mrope_section, dtype=hidden_states.dtype
        )
        position_embeddings = (cos, sin)
        all_hidden_states = () if output_hidden_states else None
        layer_indices_to_run = kwargs.get("layer_indices_to_run", None)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if layer_idx < start or layer_idx >= end:
                continue
            if layer_indices_to_run is not None and layer_idx not in layer_indices_to_run:
                continue
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                batch_fold=batch_fold,
                gdn_num_head_blocks=gdn_num_head_blocks,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            # break

        if is_last_layer_window(QEffQwen3_5TextModel, len(self.layers)):
            hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        if is_layerwise_active():
            past_key_values = past_key_values[QEffQwen3_5TextModel._start]
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3_5DecoderLayer}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def _iter_retained_state_names(self) -> List[str]:
        names = []
        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                names.extend([f"past_key.{layer_idx}", f"past_value.{layer_idx}"])
            else:
                names.extend([f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"])
        return names

    def get_retained_state_names(self) -> List[str]:
        return self._iter_retained_state_names()

    def get_onnx_past_key_value_names(self, layer_idx: int, layer_state=None) -> List[str]:
        if self.config.layer_types[layer_idx] == "full_attention":
            return [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
        return [f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"]

    def get_onnx_retained_state_specs(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache_shape: List[int],
        continuous_batching: bool = False,
        retain_full_kv: bool = False,
    ) -> dict:
        del seq_len, retain_full_kv
        batch_axis_name = "full_batch_size" if continuous_batching else "batch_size"
        specs = {
            "past_key_values": [],
            "input_names": [],
            "output_names": [],
            "dynamic_axes": {},
        }

        kv_dtype = getattr(self.config, "torch_dtype", torch.float32)
        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                layer_names = [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
                layer_tensors = [
                    torch.zeros(tuple(kv_cache_shape), dtype=kv_dtype),
                    torch.zeros(tuple(kv_cache_shape), dtype=kv_dtype),
                ]
                layer_axes = [
                    {0: batch_axis_name, 2: "ctx_len"},
                    {0: batch_axis_name, 2: "ctx_len"},
                ]
            else:
                layer = self.model.layers[layer_idx].linear_attn
                if layer.conv_dim % layer.num_k_heads != 0:
                    raise ValueError(
                        f"conv_dim ({layer.conv_dim}) must be divisible by num_k_heads ({layer.num_k_heads})"
                    )
                conv_shape = (
                    batch_size,
                    layer.num_k_heads,
                    layer.conv_dim // layer.num_k_heads,
                    layer.conv_kernel_size,
                )
                recurrent_shape = (batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
                layer_names = [f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"]
                layer_tensors = [
                    torch.zeros(conv_shape, dtype=kv_dtype),
                    torch.zeros(recurrent_shape, dtype=kv_dtype),
                ]
                layer_axes = [{0: batch_axis_name}, {0: batch_axis_name}]

            specs["past_key_values"].append(layer_tensors)
            for name, axes in zip(layer_names, layer_axes):
                specs["input_names"].append(name)
                specs["output_names"].append(f"{name}_RetainedState")
                specs["dynamic_axes"][name] = axes

        return specs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[QEffQwen3_5DynamicCache, Tuple[Tuple[torch.FloatTensor, ...], ...]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del logits_to_keep
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        if position_ids is None:
            hidden_states = outputs.last_hidden_state[:, -1:, :]
        else:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            logit_index = text_position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = outputs.last_hidden_state[torch.arange(text_position_ids.shape[0]).view(-1, 1), logit_index]

        logits = self.lm_head(hidden_states).float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QEffQwen3_5Model(Qwen3_5Model):
    _start = 0
    _end = 0
    _total_layers = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | Qwen3_5ModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class QEffQwen3_5VisionModel(Qwen3_5VisionModel):
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = max(grid_thw.shape)
        freq_table = self.rotary_pos_emb(torch.arange(max_hw, device=grid_thw.device))
        device = freq_table.device
        _bs, num_frames, height, width = grid_thw.shape
        grid_thw = (torch.tensor(grid_thw.shape, dtype=torch.int64)).unsqueeze(0)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)

        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        pos_ids = coords
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        bs, t, h, w = grid_thw.shape
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        max_t = torch.tensor(self.num_grid_per_side - 1, device=h_idxs.device)

        h_idxs_ceil = torch.minimum(h_idxs_floor + 1, max_t)
        w_idxs_ceil = torch.minimum(w_idxs_floor + 1, max_t)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        idx_tensor = torch.stack(indices, dim=0).to(dtype=torch.long, device=self.pos_embed.weight.device)

        weight_tensor = torch.stack(weights, dim=0).to(
            dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        pos_embed = patch_pos_embeds[0]
        pos_embed = pos_embed.repeat(t, 1)

        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        x_expanded = patch_pos_embeds.unsqueeze(0)
        x_expanded = x_expanded.expand(bs, -1, -1)
        patch_pos_embeds = x_expanded.reshape(-1, patch_pos_embeds.size(1))
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        bs, t, h, w = grid_thw.shape

        t = torch.arange(t, t + 1).squeeze().expand(bs)
        h = torch.arange(h, h + 1).squeeze().expand(bs)
        w = torch.arange(w, w + 1).squeeze().expand(bs)

        cu_seqlens = (h * w).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype), cu_seqlens])

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.merger(hidden_states)
        return hidden_states


class QEffQwen3_5VisionAttention(Qwen3_5VisionAttention):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.full(
            [1, seq_length, seq_length], MIN_MASKED_ATTENTION_VALUE, device=q.device, dtype=q.dtype
        )
        seq_len = attention_mask.shape[-1]
        rows = torch.arange(seq_len).view(1, -1)
        cols = torch.arange(seq_len).view(-1, 1)

        start = cu_seqlens[:-1].view(-1, 1, 1)
        end = cu_seqlens[1:].view(-1, 1, 1)
        row_mask = (rows >= start) & (rows < end)
        col_mask = (cols >= start) & (cols < end)
        block_mask = row_mask & col_mask

        final_mask = torch.ones((seq_len, seq_len), dtype=self.config.torch_dtype)
        final_mask[block_mask.any(dim=0)] = 0
        final_mask = torch.where(final_mask == 1.0, MIN_MASKED_ATTENTION_VALUE, final_mask)
        attention_mask[0] = final_mask

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class QEffQwen3_5EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        if hasattr(self.model.model, "visual") and hasattr(self.model.model.visual, "blocks"):
            return {self.model.model.visual.blocks[0].__class__}
        if hasattr(self.model.model, "vision_model") and hasattr(self.model.model.vision_model, "blocks"):
            return {self.model.model.vision_model.blocks[0].__class__}
        return set()

    def forward(self, pixel_values, image_grid_thw):
        if hasattr(self.model.model, "visual"):
            image_outputs = self.model.model.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_outputs[0] if isinstance(image_outputs, tuple) else image_outputs
        else:
            image_outputs: BaseModelOutputWithPooling = self.model.model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(pixel_values.device, pixel_values.dtype)
        bs = image_grid_thw.shape[0]
        split_size = torch.floor_divide(torch.tensor(image_embeds.size(0)), bs)
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(1))
        return image_embeds


class QEffQwen3_5DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3_5DecoderLayer}

    def _get_attention_blocking_config(self):
        for layer in getattr(self.language_model, "layers", ()):
            attention = getattr(layer, "self_attn", None)
            blocking_config = getattr(attention, "attn_blocking_config", None)
            if blocking_config is not None:
                return blocking_config
        return None

    def get_onnx_past_key_value_names(self, layer_idx: int, layer_state=None) -> List[str]:
        del layer_state
        if self.config.text_config.layer_types[layer_idx] == "full_attention":
            return [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
        return [f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"]

    def forward(
        self,
        input_ids=None,
        vision_embeds=None,
        position_ids=None,
        image_idx=None,
        past_key_values=None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        # Continuous batching also supplies batch_index for prefill. Folded
        # attention is decode-only, so do not infer it from batch_index alone.
        blocking_config = self._get_attention_blocking_config()
        batch_fold_cb = (
            batch_index is not None
            and input_ids.shape[1] == 1
            and blocking_config is not None
            and bool(blocking_config.batch_fold)
        )
        gdn_num_head_blocks = get_gdn_num_head_blocks(blocking_config, batch_fold_cb)
        if batch_fold_cb:
            # Folded attention consumes physical batch rows; restore logical
            # request order only at the decoder output boundary.
            input_ids = _batch_index_scatter(input_ids, batch_index)
            position_ids = _batch_index_scatter(position_ids, batch_index, batch_dim=1)
            cache_batch_index = None
        else:
            cache_batch_index = batch_index

        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        _, _, channel_size = inputs_embeds.shape
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.reshape(-1, channel_size).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = image_input_embeds
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=cache_batch_index,
            batch_fold=batch_fold_cb,
            gdn_num_head_blocks=gdn_num_head_blocks,
            use_cache=True,
        )
        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids[0].shape[0]).view(-1, 1), logit_index]
        if batch_fold_cb:
            hidden_states = _batch_index_gather(hidden_states, batch_index)
        logits = self.model.lm_head(hidden_states)
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        return logits, vision_embeds, image_idx, outputs.past_key_values[: len(past_key_values)]


class QEffQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffQwen3_5EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffQwen3_5DecoderWrapper(self)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | Qwen3_5CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        >>> model = Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs[0]

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids[0].shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)

        return logits, outputs.past_key_values[: len(past_key_values)]

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        height: int | List[int] = None,
        width: int | List[int] = None,
        img_size=None,
        time: int = 1,
        num_frames: int | List[int] = 1,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        vision_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill", None)
        comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode", None)
        # Preserve the legacy shape when callers do not request a separate vision batch.
        vision_batch_size = batch_size if vision_batch_size is None else vision_batch_size

        language_model = getattr(self.model, "language_model", None)
        blocking_config = None
        for layer in getattr(language_model, "layers", ()):
            attention = getattr(layer, "self_attn", None)
            blocking_config = getattr(attention, "attn_blocking_config", None)
            if blocking_config is not None:
                break
        if (
            continuous_batching
            and blocking_config is not None
            and blocking_config.batch_fold
            and batch_size != full_batch_size
        ):
            raise ValueError("Batch-folded continuous batching requires batch_size == full_batch_size.")

        if height is None or width is None:
            height = constants.QWEN3_VL_HEIGHT
            width = constants.QWEN3_VL_WIDTH
            logger.warning(
                f"Setting height and width to be {height} and {width} respectively, as it was neither passed nor found in vision_config"
            )

        height = [height] if isinstance(height, int) else height
        width = [width] if isinstance(width, int) else width
        num_frames = [num_frames] * len(height) if isinstance(num_frames, int) else num_frames

        prefill_seq_len = prefill_seq_len if prefill_seq_len else 128
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size
        channel = 3
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = getattr(self.config.vision_config, "temporal_patch_size", 1)

        image_factor = constants.IMAGE_FACTOR_QWEN_3
        image_min_token_num = constants.IMAGE_MIN_TOKEN_NUM
        image_max_token_num = constants.IMAGE_MAX_TOKEN_NUM
        min_pixels = image_min_token_num * image_factor**2
        max_pixels = image_max_token_num * image_factor**2
        mm_processor_kwargs = compiler_options.pop("mm_processor_kwargs", None)
        if mm_processor_kwargs:
            min_pixels = mm_processor_kwargs.get("min_pixels", min_pixels)
            max_pixels = mm_processor_kwargs.get("max_pixels", max_pixels)

        vision = []
        max_vision_size = 0
        user_vision_size = compiler_options.pop("vision_size", None)
        if user_vision_size:
            assert user_vision_size < ctx_len, "vision_size must be less than ctx_len"
            max_vision_size = user_vision_size

        for h, w, f in zip(height, width, num_frames):
            resized_height, resized_width = smart_resize(
                height=h, width=w, factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels
            )
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            grid_height = grid_h * grid_w
            grid_width = patch_size * patch_size * temporal_patch_size * channel
            vision_size = (grid_height // 4) * time
            grid_height = grid_height * time * vision_batch_size

            if not user_vision_size:
                max_vision_size = max(max_vision_size, vision_size * f)
                assert max_vision_size < ctx_len, (
                    f"Computed vision_size of {vision_size * f} tokens "
                    f"(vision_size={vision_size}, num_frames={f}) for image resolution "
                    f"(width={w}, height={h}) must be less than ctx_len. Please adjust the image "
                    "resolution."
                )
            else:
                if vision_size * f > user_vision_size:
                    logger.warning_once(
                        f"Computed vision_size of {vision_size * f} tokens "
                        f"(vision_size={vision_size}, num_frames={f}) for image resolution "
                        f"(width={w}, height={h}) exceeds the provided "
                        f"vision_size={user_vision_size}. "
                        f"Vision embedding needs to be chunked during prefill."
                    )

            vision.append(
                {
                    "vision_batch_size": vision_batch_size,
                    "vision_size": vision_size,
                    "grid_height": grid_height,
                    "grid_width": grid_width,
                    "time": time,
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                }
            )

        def _build_lang_spec(seq_len_val, comp_ctx_len=None):
            spec = {
                "batch_size": full_batch_size
                if (continuous_batching and seq_len_val == 1)
                else (1 if continuous_batching else batch_size),
                "seq_len": seq_len_val,
                "ctx_len": ctx_len,
            }
            if kv_offload:
                spec["vision_size"] = max_vision_size
                spec["vision_batch_size"] = vision_batch_size
            if comp_ctx_len is not None:
                spec["comp_ctx_lengths"] = comp_ctx_len
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if full_batch_size and seq_len_val != 1:
                spec["full_batch_exec_size"] = full_batch_size
            return spec

        lang = []
        if comp_ctx_lengths_prefill is not None or comp_ctx_lengths_decode is not None:
            for comp_ctx in comp_ctx_lengths_prefill or []:
                lang.append(_build_lang_spec(prefill_seq_len, comp_ctx_len=comp_ctx))
            for comp_ctx in comp_ctx_lengths_decode or []:
                lang.append(_build_lang_spec(1, comp_ctx_len=comp_ctx))
        else:
            lang.append(_build_lang_spec(prefill_seq_len))
            lang.append(_build_lang_spec(1))

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options

        for spec in lang:
            spec.pop("vision_size", None)
            spec.pop("vision_batch_size", None)
        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: list[int] | None = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        batch_fold: bool = False,
    ):
        num_layers = self.config.text_config.num_hidden_layers
        batch_axis_name = "full_batch_size" if continuous_batching else "batch_size"
        input_batch_axis = "full_batch_size" if continuous_batching and batch_fold else "batch_size"

        vision_dynamic_axes = {
            "pixel_values": {0: "grid_height", 1: "grid_width"},
            "image_grid_thw": {0: "vision_batch_size", 1: "time", 2: "grid_h", 3: "grid_w"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: input_batch_axis, 1: "seq_len"},
            "position_ids": {1: input_batch_axis, 2: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }

        for i in range(num_layers):
            if self.config.text_config.layer_types[i] == "full_attention":
                lang_dynamic_axes[f"past_key.{i}"] = {0: batch_axis_name, 2: "ctx_len"}
                lang_dynamic_axes[f"past_value.{i}"] = {0: batch_axis_name, 2: "ctx_len"}
            else:
                lang_dynamic_axes[f"conv_state.{i}"] = {0: batch_axis_name}
                lang_dynamic_axes[f"recurrent_state.{i}"] = {0: batch_axis_name}

        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: input_batch_axis}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}

        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            lang_dynamic_axes.pop("vision_embeds")
            dynamic_axes = lang_dynamic_axes

        return dynamic_axes

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        inputs_shapes = {}

        dummy_seq_len = int(kwargs.get("prefill_seq_len", 32))
        bs = kwargs.get("batch_size", constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE)
        if bs > 1:
            bs = 2
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        batch_fold = kwargs.pop("batch_fold", False)
        if continuous_batching and batch_fold:
            bs = fbs
        inputs_shapes["input_ids"] = (bs, dummy_seq_len)

        inputs_shapes["position_ids"] = (
            4,
            bs,
            dummy_seq_len,
        )
        inputs_shapes["pixel_values"] = (11008, 1536)
        inputs_shapes["image_grid_thw"] = (
            bs,
            1,
            86,
            128,
        )
        inputs_shapes["vision_embeds"] = (
            bs,
            2752,
            self.model.config.text_config.hidden_size,
        )
        inputs_shapes["image_idx"] = (1, 1)

        vision_inputs = {}
        lang_inputs = {}
        # Float inputs follow the model dtype so float16 export traces cleanly.
        float_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=float_dtype)
        vision_inputs["image_grid_thw"] = torch.zeros((inputs_shapes["image_grid_thw"]), dtype=torch.int64)
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=float_dtype)
        lang_inputs["position_ids"] = (
            (torch.arange(dummy_seq_len, dtype=torch.int64).view(1, dummy_seq_len).repeat(bs, 1))
            .unsqueeze(0)
            .repeat(4, 1, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)

        kv_cache_shape = get_padding_shape_from_config(
            config=self.model.config.text_config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=dummy_seq_len,
        )

        linear_batch_size = fbs if continuous_batching else bs

        lang_inputs["past_key_values"] = [[] for _ in range(self.model.config.text_config.num_hidden_layers)]
        # Default path exports all layers; layerwise exports only the active window's layer.
        if is_layerwise_active():
            window_layers = [QEffQwen3_5TextModel._start]
        else:
            window_layers = range(self.model.config.text_config.num_hidden_layers)
        # KV/state dummy dtype follows the model dtype so the export trace works
        # for float16 as well as float32.
        kv_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        for i in window_layers:
            if self.model.config.text_config.layer_types[i] == "full_attention":
                for kv in ["key", "value"]:
                    lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=kv_dtype))
            else:
                layer = self.model.language_model.layers[i].linear_attn
                if layer.conv_dim % layer.num_k_heads != 0:
                    raise ValueError(
                        f"conv_dim ({layer.conv_dim}) must be divisible by num_k_heads ({layer.num_k_heads})"
                    )
                conv_shape = (
                    linear_batch_size,
                    layer.num_k_heads,
                    layer.conv_dim // layer.num_k_heads,
                    layer.conv_kernel_size,
                )
                recurrent_shape = (linear_batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
                lang_inputs["past_key_values"][i].append(torch.zeros(conv_shape, dtype=kv_dtype))
                lang_inputs["past_key_values"][i].append(torch.zeros(recurrent_shape, dtype=kv_dtype))

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            lang_inputs.pop("image_idx")
            inputs = lang_inputs

        return inputs

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for layer_idx, layer_type in enumerate(self.model.config.text_config.layer_types):
            if layer_idx >= self.model.config.text_config.num_hidden_layers:
                break
            if layer_type == "full_attention":
                lang_output_names.extend(
                    [
                        f"past_key.{layer_idx}_RetainedState",
                        f"past_value.{layer_idx}_RetainedState",
                    ]
                )
            else:
                lang_output_names.extend(
                    [
                        f"conv_state.{layer_idx}_RetainedState",
                        f"recurrent_state.{layer_idx}_RetainedState",
                    ]
                )

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            return lang_output_names
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            # IOInfo(name="pixel_values", datatype=self.config.torch_dtype, shape=("batch_size", 3, "image_size", "image_size")),
        ]

    def prepare_inputs_for_generation(self, inputs, prefill_seq_len=32, batch_size=1):
        input_ids_length = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(input_ids_length).view(1, 1, input_ids_length).expand(-1, batch_size, -1)
        pos_ids, _rope_deltas = self.model.get_rope_index(
            inputs["input_ids"],
            inputs["mm_token_type_ids"],
            inputs.get("image_grid_thw", None),
            video_grid_thw=None,
            attention_mask=inputs["attention_mask"],
        )

        inputs["position_ids"] = torch.cat((inputs["position_ids"], pos_ids), dim=0)

        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        inputs["position_ids"] = F.pad(
            inputs["position_ids"], pad=(0, padded_len - input_ids_length), mode="constant", value=-1
        )

        inputs.pop("mm_token_type_ids")
        return inputs
