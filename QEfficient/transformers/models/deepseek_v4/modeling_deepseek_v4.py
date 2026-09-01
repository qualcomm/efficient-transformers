# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4HashRouter,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4MLP,
    DeepseekV4Model,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseMoeBlock,
    DeepseekV4TopKRouter,
    DeepseekV4UnweightedRMSNorm,
)

from QEfficient.customop.ctx_scatter_gather import CtxGatherFuncBlockedKV as CtxGatherBlockedKVFunc
from QEfficient.customop.ctx_scatter_gather import CtxScatterFunc
from QEfficient.customop.rms_norm import CustomRMSNormAIC, CustomRMSNormFunc
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffSlidingCacheLayer(CacheLayerMixin):
    """Fixed-capacity retained state for one DeepSeek V4 sliding-attention layer."""

    is_compileable = True
    is_sliding = True

    def __init__(
        self,
        config: DeepseekV4Config,
        sliding_window_kv: torch.Tensor,
        *,
        cumulative_length: int = 0,
    ) -> None:
        expected = (
            sliding_window_kv.shape[0],
            config.num_key_value_heads,
            sliding_window_kv.shape[2],
            config.head_dim,
        )
        if sliding_window_kv.ndim != 4 or tuple(sliding_window_kv.shape) != expected:
            raise ValueError(f"sliding_window_kv must have shape {expected}.")
        if not 0 <= cumulative_length <= sliding_window_kv.shape[2]:
            raise ValueError("cumulative_length is outside the cache capacity.")
        self.sliding_window = config.sliding_window
        self.sliding_window_kv = sliding_window_kv
        self.cumulative_length = cumulative_length
        self.max_cache_len = sliding_window_kv.shape[2]
        self.is_initialized = True
        self.device = sliding_window_kv.device
        self.dtype = sliding_window_kv.dtype

    @property
    def max_batch_size(self) -> int:
        return self.sliding_window_kv.shape[0]

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.shape[0] != self.max_batch_size or key_states.shape[-1] != self.sliding_window_kv.shape[-1]:
            raise ValueError("KV initialization shape does not match the allocated QEff sliding cache.")

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        is_full = self.cumulative_length >= self.sliding_window
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            return self.sliding_window - 1 + query_length, kv_offset
        return self.cumulative_length + query_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states is not value_states and not torch.equal(key_states, value_states):
            raise ValueError("QEffSlidingCache requires shared K/V states.")
        position_ids = (cache_kwargs or {}).get("position_ids")
        if position_ids is None:
            raise ValueError("QEffSlidingCache.update requires position_ids in cache_kwargs.")
        if position_ids.shape != key_states.shape[:1] + key_states.shape[2:3]:
            raise ValueError("position_ids must have shape [batch, query_length].")
        if key_states.shape[0] != self.max_batch_size or key_states.shape[1] != self.sliding_window_kv.shape[1]:
            raise ValueError("KV update batch/head dimensions do not match the allocated cache.")
        if key_states.shape[-1] != self.sliding_window_kv.shape[-1]:
            raise ValueError("KV update head_dim does not match the allocated cache.")
        if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            expected = position_ids[:, :1] + torch.arange(
                position_ids.shape[1], device=position_ids.device, dtype=position_ids.dtype
            )
            if not torch.equal(position_ids, expected):
                raise ValueError("QEffSlidingCache requires contiguous position_ids within each batch row.")
            expected_start = torch.full_like(position_ids[:, 0], self.cumulative_length)
            if not torch.equal(position_ids[:, 0], expected_start):
                raise ValueError("position_ids must begin at the cache's cumulative_length.")
            if position_ids.numel() and (
                position_ids.min().item() < 0 or position_ids.max().item() >= self.max_cache_len
            ):
                raise ValueError("position_ids exceed the allocated QEff sliding cache capacity.")
        self.sliding_window_kv = CtxScatterFunc.apply(self.sliding_window_kv, position_ids, key_states)
        self.cumulative_length += key_states.shape[2]

        context_length = (cache_kwargs or {}).get("context_length", self.max_cache_len)
        context_indices = torch.arange(context_length, device=self.device, dtype=torch.int32).view(1, 1, -1)
        context_end = position_ids[:, -1:].to(torch.int32).unsqueeze(1) + 1
        context_indices = context_indices + context_end - context_length
        context_indices = context_indices.expand(-1, self.sliding_window_kv.shape[1], -1)
        valid = context_indices >= 0
        invalid_index = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        gathered = CtxGatherBlockedKVFunc.apply(
            self.sliding_window_kv, torch.where(valid, context_indices, invalid_index)
        )
        gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        return gathered, gathered

    def reset(self) -> None:
        self.sliding_window_kv.zero_()
        self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.sliding_window_kv = self.sliding_window_kv.index_select(0, beam_idx.to(self.device))

    def crop(self, max_length: int) -> None:
        if max_length != self.cumulative_length:
            raise NotImplementedError("QEffSlidingCache does not support cropping fixed retained state.")

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.sliding_window_kv = self.sliding_window_kv.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.sliding_window_kv = self.sliding_window_kv[indices]


class QEffHCACacheLayer(CacheLayerMixin):
    """Fixed-capacity retained state for one DeepSeek V4 HCA layer."""

    is_compileable = True
    is_sliding = True

    def __init__(
        self,
        config: DeepseekV4Config,
        sliding_window_kv: torch.Tensor,
        compressor_kv_buffer: torch.Tensor,
        compressor_gate_buffer: torch.Tensor,
        actual_compressed_kv: torch.Tensor,
        *,
        cumulative_length: int = 0,
        compressor_entry_count: int = 0,
    ) -> None:
        self.sliding_window = config.sliding_window
        self.compression_size = config.compress_rates["heavily_compressed_attention"]
        self.sliding_window_kv = sliding_window_kv
        self.compressor_kv_buffer = compressor_kv_buffer
        self.compressor_gate_buffer = compressor_gate_buffer
        self.actual_compressed_kv = actual_compressed_kv
        self.cumulative_length = cumulative_length
        self.compressor_entry_count = compressor_entry_count
        self.max_cache_len = sliding_window_kv.shape[2]
        self.is_initialized = True
        self.device = sliding_window_kv.device
        self.dtype = sliding_window_kv.dtype
        self._last_position_ids: torch.Tensor | None = None
        self._previous_length = cumulative_length
        self._validate_state(config)

    def _validate_state(self, config: DeepseekV4Config) -> None:
        expected_prefix = (
            self.sliding_window_kv.shape[0],
            1,
        )
        if self.sliding_window_kv.ndim != 4:
            raise ValueError("sliding_window_kv must have shape [batch, heads, ctx_len, head_dim].")
        if self.sliding_window_kv.shape[1] != config.num_key_value_heads:
            raise ValueError("sliding_window_kv head count does not match the model config.")
        if self.sliding_window_kv.shape[-1] != config.head_dim:
            raise ValueError("sliding_window_kv head_dim does not match the model config.")
        expected_buffer_shape = (*expected_prefix, self.compression_size, config.head_dim)
        if tuple(self.compressor_kv_buffer.shape) != expected_buffer_shape:
            raise ValueError(
                f"compressor_kv_buffer must have shape {expected_buffer_shape}, "
                f"got {tuple(self.compressor_kv_buffer.shape)}."
            )
        if tuple(self.compressor_gate_buffer.shape) != expected_buffer_shape:
            raise ValueError(
                f"compressor_gate_buffer must have shape {expected_buffer_shape}, "
                f"got {tuple(self.compressor_gate_buffer.shape)}."
            )
        expected_compressed_shape = (
            self.sliding_window_kv.shape[0],
            1,
            (self.max_cache_len + self.compression_size - 1) // self.compression_size,
            config.head_dim,
        )
        if tuple(self.actual_compressed_kv.shape) != expected_compressed_shape:
            raise ValueError(
                f"actual_compressed_kv must have shape {expected_compressed_shape}, "
                f"got {tuple(self.actual_compressed_kv.shape)}."
            )
        states = (
            self.compressor_kv_buffer,
            self.compressor_gate_buffer,
            self.actual_compressed_kv,
        )
        if any(state.device != self.device or state.dtype != self.dtype for state in states):
            raise ValueError("All QEff HCA cache tensors must have the same device and dtype.")
        if not 0 <= self.cumulative_length <= self.max_cache_len:
            raise ValueError("cumulative_length is outside the cache capacity.")
        if not 0 <= self.compressor_entry_count <= self.actual_compressed_kv.shape[2]:
            raise ValueError("compressor_entry_count is outside the compressed cache capacity.")

    @property
    def max_batch_size(self) -> int:
        return self.sliding_window_kv.shape[0]

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.shape[0] != self.max_batch_size or key_states.shape[-1] != self.sliding_window_kv.shape[-1]:
            raise ValueError("KV initialization shape does not match the allocated QEff HCA cache.")

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        is_full = self.cumulative_length >= self.sliding_window
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            return self.sliding_window - 1 + query_length, kv_offset
        return self.cumulative_length + query_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states is not value_states and not torch.equal(key_states, value_states):
            raise ValueError("QEffHCACache requires shared K/V states.")
        cache_kwargs = cache_kwargs or {}
        position_ids = cache_kwargs.get("position_ids")
        if position_ids is None:
            raise ValueError("QEffHCACache.update requires position_ids in cache_kwargs.")
        if position_ids.shape != key_states.shape[:1] + key_states.shape[2:3]:
            raise ValueError("position_ids must have shape [batch, query_length].")
        if key_states.shape[0] != self.max_batch_size or key_states.shape[1] != self.sliding_window_kv.shape[1]:
            raise ValueError("KV update batch/head dimensions do not match the allocated cache.")
        if key_states.shape[-1] != self.sliding_window_kv.shape[-1]:
            raise ValueError("KV update head_dim does not match the allocated cache.")
        if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            expected = position_ids[:, :1] + torch.arange(
                position_ids.shape[1], device=position_ids.device, dtype=position_ids.dtype
            )
            if not torch.equal(position_ids, expected):
                raise ValueError("QEffHCACache requires contiguous position_ids within each batch row.")
            expected_start = torch.full_like(position_ids[:, 0], self.cumulative_length)
            if not torch.equal(position_ids[:, 0], expected_start):
                raise ValueError("position_ids must begin at the cache's cumulative_length.")
            if position_ids.numel() and (
                position_ids.min().item() < 0 or position_ids.max().item() >= self.max_cache_len
            ):
                raise ValueError("position_ids exceed the allocated QEff HCA cache capacity.")

        self._previous_length = self.cumulative_length
        self._last_position_ids = position_ids
        scatter_positions = position_ids
        self.sliding_window_kv = CtxScatterFunc.apply(self.sliding_window_kv, scatter_positions, key_states)
        self.cumulative_length += key_states.shape[2]

        projected_kv = cache_kwargs.get("compressor_kv")
        projected_gate = cache_kwargs.get("compressor_gate")
        if (projected_kv is None) != (projected_gate is None):
            raise ValueError("compressor_kv and compressor_gate must be provided together.")
        if projected_kv is not None:
            expected_shape = (self.max_batch_size, 1, self.sliding_window_kv.shape[-1])
            if tuple(projected_kv.shape) != expected_shape or projected_gate.shape != projected_kv.shape:
                raise ValueError(f"Decode-only compressor projections must both have shape {expected_shape}.")
            buffer_positions = torch.remainder(position_ids, self.compression_size)
            self.compressor_kv_buffer = CtxScatterFunc.apply(
                self.compressor_kv_buffer,
                buffer_positions,
                projected_kv.unsqueeze(1),
            )
            self.compressor_gate_buffer = CtxScatterFunc.apply(
                self.compressor_gate_buffer,
                buffer_positions,
                projected_gate.unsqueeze(1),
            )

        context_length = cache_kwargs.get("context_length")
        if context_length is None:
            context_indices = torch.arange(self.max_cache_len, device=self.device, dtype=torch.int32).view(1, 1, -1)
            context_indices = context_indices.expand(self.max_batch_size, self.sliding_window_kv.shape[1], -1)
            valid = context_indices <= position_ids.max(dim=1, keepdim=True).values.to(torch.int32).unsqueeze(1)
        else:
            context_indices = torch.arange(context_length, device=self.device, dtype=torch.int32).view(1, 1, -1)
            context_end = position_ids[:, -1:].to(torch.int32).unsqueeze(1) + 1
            context_indices = context_indices + context_end - context_length
            context_indices = context_indices.expand(-1, self.sliding_window_kv.shape[1], -1)
            valid = context_indices >= 0
        invalid_index = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        gathered = CtxGatherBlockedKVFunc.apply(
            self.sliding_window_kv,
            torch.where(valid, context_indices, invalid_index),
        )
        gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        return gathered, gathered

    def store_compression_weights(
        self,
        name: str,
        kv: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if name != "compressor":
            raise ValueError(f"Unsupported HCA compressor state: {name}")
        if self._last_position_ids is None:
            raise RuntimeError("Normal KV must be updated before compressor state.")
        if kv.shape != gate.shape or kv.ndim != 3:
            raise ValueError("Compressor KV and gate inputs must have matching [batch, seq, dim] shapes.")
        if kv.shape[0] != self.max_batch_size or kv.shape[-1] != self.sliding_window_kv.shape[-1]:
            raise ValueError("Compressor update shape does not match the allocated cache.")

        prior_count = self._previous_length % self.compression_size
        prior_kv = self.compressor_kv_buffer[:, 0, :prior_count]
        prior_gate = self.compressor_gate_buffer[:, 0, :prior_count]
        combined_kv = torch.cat([prior_kv, kv], dim=1)
        combined_gate = torch.cat([prior_gate, gate], dim=1)
        usable = (combined_kv.shape[1] // self.compression_size) * self.compression_size
        chunk_kv = combined_kv[:, :usable]
        chunk_gate = combined_gate[:, :usable]
        remainder_kv = combined_kv[:, usable:]
        remainder_gate = combined_gate[:, usable:]
        remainder_length = remainder_kv.shape[1]
        if remainder_length:
            buffer_positions = (
                torch.arange(remainder_length, device=kv.device, dtype=torch.int64)
                .unsqueeze(0)
                .expand(self.max_batch_size, -1)
            )
            self.compressor_kv_buffer = CtxScatterFunc.apply(
                self.compressor_kv_buffer,
                buffer_positions,
                remainder_kv.unsqueeze(1),
            )
            self.compressor_gate_buffer = CtxScatterFunc.apply(
                self.compressor_gate_buffer,
                buffer_positions,
                remainder_gate.unsqueeze(1),
            )
        first_window_position = self.compressor_entry_count * self.compression_size
        return chunk_kv, chunk_gate, first_window_position

    def update_compressor_states(
        self,
        name: str,
        compressed: torch.Tensor,
        *,
        entry_positions: torch.Tensor | None = None,
        write_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if name != "compressor":
            raise ValueError(f"Unsupported HCA compressor state: {name}")
        if compressed.ndim != 3 or compressed.shape[0] != self.max_batch_size:
            raise ValueError("Compressed KV must have shape [batch, entries, head_dim].")
        if entry_positions is not None:
            if compressed.shape[1] != 1 or entry_positions.shape != compressed.shape[:2]:
                raise ValueError("Decode-only compressed updates require one entry position per batch row.")
            if write_mask is None or write_mask.shape != entry_positions.shape:
                raise ValueError("Decode-only compressed updates require a matching write_mask.")
            self.actual_compressed_kv = CtxScatterFunc.apply(
                self.actual_compressed_kv,
                entry_positions,
                compressed.unsqueeze(1),
            )
            if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                completed = entry_positions[write_mask.to(torch.bool)]
                if completed.numel():
                    self.compressor_entry_count = max(
                        self.compressor_entry_count,
                        int(completed.max().item()) + 1,
                    )
            return self.actual_compressed_kv

        new_entries = compressed.shape[1]
        new_count = self.compressor_entry_count + new_entries
        if new_count > self.actual_compressed_kv.shape[2]:
            raise ValueError("Compressed KV update exceeds the allocated cache capacity.")
        if new_entries:
            entry_positions = (
                torch.arange(
                    self.compressor_entry_count,
                    new_count,
                    device=compressed.device,
                    dtype=torch.int64,
                )
                .unsqueeze(0)
                .expand(self.max_batch_size, -1)
            )
            self.actual_compressed_kv = CtxScatterFunc.apply(
                self.actual_compressed_kv,
                entry_positions,
                compressed.unsqueeze(1),
            )
        self.compressor_entry_count = new_count
        return self.actual_compressed_kv[:, 0, :new_count]

    def reset(self) -> None:
        self.sliding_window_kv.zero_()
        self.compressor_kv_buffer.zero_()
        self.compressor_gate_buffer.zero_()
        self.actual_compressed_kv.zero_()
        self.cumulative_length = 0
        self.compressor_entry_count = 0
        self._last_position_ids = None
        self._previous_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.sliding_window_kv = self.sliding_window_kv.index_select(0, beam_idx.to(self.device))
        self.compressor_kv_buffer = self.compressor_kv_buffer.index_select(0, beam_idx.to(self.device))
        self.compressor_gate_buffer = self.compressor_gate_buffer.index_select(0, beam_idx.to(self.device))
        self.actual_compressed_kv = self.actual_compressed_kv.index_select(0, beam_idx.to(self.device))

    def crop(self, max_length: int) -> None:
        if max_length != self.cumulative_length:
            raise NotImplementedError("QEffHCACache does not support cropping fixed retained state.")

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.sliding_window_kv = self.sliding_window_kv.repeat_interleave(repeats, dim=0)
        self.compressor_kv_buffer = self.compressor_kv_buffer.repeat_interleave(repeats, dim=0)
        self.compressor_gate_buffer = self.compressor_gate_buffer.repeat_interleave(repeats, dim=0)
        self.actual_compressed_kv = self.actual_compressed_kv.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.sliding_window_kv = self.sliding_window_kv[indices]
        self.compressor_kv_buffer = self.compressor_kv_buffer[indices]
        self.compressor_gate_buffer = self.compressor_gate_buffer[indices]
        self.actual_compressed_kv = self.actual_compressed_kv[indices]


class QEffCSACacheLayer(CacheLayerMixin):
    """Fixed-capacity retained state with ping-pong buffers for one DeepSeek V4 CSA layer."""

    is_compileable = True
    is_sliding = True

    def __init__(
        self,
        config: DeepseekV4Config,
        sliding_window_kv: torch.Tensor,
        compressor_kv_buffer: torch.Tensor,
        compressor_gate_buffer: torch.Tensor,
        actual_compressed_kv: torch.Tensor,
        indexer_kv_buffer: torch.Tensor,
        indexer_gate_buffer: torch.Tensor,
        actual_indexer_compressed_kv: torch.Tensor,
        *,
        cumulative_length: int = 0,
        compressor_entry_count: int = 0,
        indexer_entry_count: int = 0,
    ) -> None:
        self.sliding_window = config.sliding_window
        self.compression_size = config.compress_rates["compressed_sparse_attention"]
        self.sliding_window_kv = sliding_window_kv
        self.compressor_kv_buffer = compressor_kv_buffer
        self.compressor_gate_buffer = compressor_gate_buffer
        self.actual_compressed_kv = actual_compressed_kv
        self.indexer_kv_buffer = indexer_kv_buffer
        self.indexer_gate_buffer = indexer_gate_buffer
        self.actual_indexer_compressed_kv = actual_indexer_compressed_kv
        self.cumulative_length = cumulative_length
        self.compressor_entry_count = compressor_entry_count
        self.indexer_entry_count = indexer_entry_count
        self.max_cache_len = sliding_window_kv.shape[2]
        self.is_initialized = True
        self.device = sliding_window_kv.device
        self.dtype = sliding_window_kv.dtype
        self._validate_state(config)

    def _validate_state(self, config: DeepseekV4Config) -> None:
        batch = self.sliding_window_kv.shape[0]
        capacity = (self.max_cache_len + self.compression_size - 1) // self.compression_size
        common_prefix = (batch, 1)
        banked = 2 * self.compression_size
        expected = {
            "sliding_window_kv": (batch, config.num_key_value_heads, self.max_cache_len, config.head_dim),
            "compressor_kv_buffer": (*common_prefix, banked, 2 * config.head_dim),
            "compressor_gate_buffer": (*common_prefix, banked, 2 * config.head_dim),
            "actual_compressed_kv": (*common_prefix, capacity, config.head_dim),
            "indexer_kv_buffer": (*common_prefix, banked, 2 * config.index_head_dim),
            "indexer_gate_buffer": (*common_prefix, banked, 2 * config.index_head_dim),
            "actual_indexer_compressed_kv": (*common_prefix, capacity, config.index_head_dim),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}.")
            if tensor.device != self.device or tensor.dtype != self.dtype:
                raise ValueError("All QEff CSA cache tensors must have the same device and dtype.")
        if not 0 <= self.cumulative_length <= self.max_cache_len:
            raise ValueError("cumulative_length is outside the cache capacity.")

    @property
    def max_batch_size(self) -> int:
        return self.sliding_window_kv.shape[0]

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.shape[0] != self.max_batch_size or key_states.shape[-1] != self.sliding_window_kv.shape[-1]:
            raise ValueError("KV initialization shape does not match the allocated QEff CSA cache.")

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        is_full = self.cumulative_length >= self.sliding_window
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            return self.sliding_window - 1 + query_length, kv_offset
        return self.cumulative_length + query_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states is not value_states and not torch.equal(key_states, value_states):
            raise ValueError("QEffCSACache requires shared K/V states.")
        cache_kwargs = cache_kwargs or {}
        position_ids = cache_kwargs.get("position_ids")
        if position_ids is None:
            raise ValueError("QEffCSACache.update requires position_ids in cache_kwargs.")
        if position_ids.shape != key_states.shape[:1] + key_states.shape[2:3]:
            raise ValueError("position_ids must have shape [batch, query_length].")
        if key_states.shape[0] != self.max_batch_size or key_states.shape[1] != self.sliding_window_kv.shape[1]:
            raise ValueError("KV update batch/head dimensions do not match the allocated cache.")
        if key_states.shape[-1] != self.sliding_window_kv.shape[-1]:
            raise ValueError("KV update head_dim does not match the allocated cache.")
        if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            expected = position_ids[:, :1] + torch.arange(
                position_ids.shape[1], device=position_ids.device, dtype=position_ids.dtype
            )
            if not torch.equal(position_ids, expected):
                raise ValueError("QEffCSACache requires contiguous position_ids within each batch row.")
            expected_start = torch.full_like(position_ids[:, 0], self.cumulative_length)
            if not torch.equal(position_ids[:, 0], expected_start):
                raise ValueError("position_ids must begin at the cache's cumulative_length.")
            if position_ids.numel() and (
                position_ids.min().item() < 0 or position_ids.max().item() >= self.max_cache_len
            ):
                raise ValueError("position_ids exceed the allocated QEff CSA cache capacity.")

        scatter_positions = position_ids
        self.sliding_window_kv = CtxScatterFunc.apply(self.sliding_window_kv, scatter_positions, key_states)
        self.cumulative_length += key_states.shape[2]

        entry_positions = torch.div(position_ids, self.compression_size, rounding_mode="floor")
        buffer_positions = torch.remainder(entry_positions, 2) * self.compression_size + torch.remainder(
            position_ids, self.compression_size
        )
        for prefix, expected_dim in (
            ("compressor", self.compressor_kv_buffer.shape[-1]),
            ("indexer", self.indexer_kv_buffer.shape[-1]),
        ):
            projected_kv = cache_kwargs.get(f"{prefix}_kv")
            projected_gate = cache_kwargs.get(f"{prefix}_gate")
            if (projected_kv is None) != (projected_gate is None):
                raise ValueError(f"{prefix}_kv and {prefix}_gate must be provided together.")
            if projected_kv is None:
                continue
            expected_shape = (self.max_batch_size, 1, expected_dim)
            if tuple(projected_kv.shape) != expected_shape or projected_gate.shape != projected_kv.shape:
                raise ValueError(f"Decode-only {prefix} projections must both have shape {expected_shape}.")
            setattr(
                self,
                f"{prefix}_kv_buffer",
                CtxScatterFunc.apply(getattr(self, f"{prefix}_kv_buffer"), buffer_positions, projected_kv.unsqueeze(1)),
            )
            setattr(
                self,
                f"{prefix}_gate_buffer",
                CtxScatterFunc.apply(
                    getattr(self, f"{prefix}_gate_buffer"), buffer_positions, projected_gate.unsqueeze(1)
                ),
            )

        context_length = cache_kwargs.get("context_length")
        if context_length is None:
            context_indices = torch.arange(self.max_cache_len, device=self.device, dtype=torch.int32).view(1, 1, -1)
            context_indices = context_indices.expand(self.max_batch_size, self.sliding_window_kv.shape[1], -1)
            valid = context_indices <= position_ids.max(dim=1, keepdim=True).values.to(torch.int32).unsqueeze(1)
        else:
            context_indices = torch.arange(context_length, device=self.device, dtype=torch.int32).view(1, 1, -1)
            context_end = position_ids[:, -1:].to(torch.int32).unsqueeze(1) + 1
            context_indices = context_indices + context_end - context_length
            context_indices = context_indices.expand(-1, self.sliding_window_kv.shape[1], -1)
            valid = context_indices >= 0
        invalid_index = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        gathered = CtxGatherBlockedKVFunc.apply(
            self.sliding_window_kv, torch.where(valid, context_indices, invalid_index)
        )
        gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        return gathered, gathered

    def update_csa_compressed_state(
        self,
        name: str,
        compressed: torch.Tensor,
        entry_positions: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> torch.Tensor:
        if name == "compressor":
            compressed_attr = "actual_compressed_kv"
            count_attr = "compressor_entry_count"
        elif name == "indexer":
            compressed_attr = "actual_indexer_compressed_kv"
            count_attr = "indexer_entry_count"
        else:
            raise ValueError(f"Unsupported CSA compressor state: {name}")
        setattr(
            self,
            compressed_attr,
            CtxScatterFunc.apply(
                getattr(self, compressed_attr),
                entry_positions,
                compressed.unsqueeze(1).unsqueeze(1),
            ),
        )
        if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            completed = entry_positions[write_mask.to(torch.bool)]
            if completed.numel():
                setattr(self, count_attr, max(getattr(self, count_attr), int(completed.max().item()) + 1))
        return getattr(self, compressed_attr)

    def reset(self) -> None:
        for name in (
            "sliding_window_kv",
            "compressor_kv_buffer",
            "compressor_gate_buffer",
            "actual_compressed_kv",
            "indexer_kv_buffer",
            "indexer_gate_buffer",
            "actual_indexer_compressed_kv",
        ):
            getattr(self, name).zero_()
        self.cumulative_length = 0
        self.compressor_entry_count = 0
        self.indexer_entry_count = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for name in (
            "sliding_window_kv",
            "compressor_kv_buffer",
            "compressor_gate_buffer",
            "actual_compressed_kv",
            "indexer_kv_buffer",
            "indexer_gate_buffer",
            "actual_indexer_compressed_kv",
        ):
            setattr(self, name, getattr(self, name).index_select(0, beam_idx.to(self.device)))

    def crop(self, max_length: int) -> None:
        if max_length != self.cumulative_length:
            raise NotImplementedError("QEffCSACache does not support cropping fixed retained state.")

    def batch_repeat_interleave(self, repeats: int) -> None:
        for name in (
            "sliding_window_kv",
            "compressor_kv_buffer",
            "compressor_gate_buffer",
            "actual_compressed_kv",
            "indexer_kv_buffer",
            "indexer_gate_buffer",
            "actual_indexer_compressed_kv",
        ):
            setattr(self, name, getattr(self, name).repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for name in (
            "sliding_window_kv",
            "compressor_kv_buffer",
            "compressor_gate_buffer",
            "actual_compressed_kv",
            "indexer_kv_buffer",
            "indexer_gate_buffer",
            "actual_indexer_compressed_kv",
        ):
            setattr(self, name, getattr(self, name)[indices])


def qeff_apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply interleaved RoPE using export-friendly, explicitly bounded indexing."""
    cos = cos.unsqueeze(-1).expand(-1, -1, -1, 2).flatten(-2)
    sin = sin.unsqueeze(-1).expand(-1, -1, -1, 2).flatten(-2)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    head_dim = x.shape[-1]
    rope_dim = cos.shape[-1]
    rope_start = head_dim - rope_dim
    nope = torch.narrow(x, dim=x.dim() - 1, start=0, length=rope_start)
    rope = torch.narrow(x, dim=x.dim() - 1, start=rope_start, length=rope_dim)

    rope_pairs = rope.reshape(*rope.shape[:-1], rope_dim // 2, 2)
    rope_even = rope_pairs.select(dim=rope_pairs.dim() - 1, index=0)
    rope_odd = rope_pairs.select(dim=rope_pairs.dim() - 1, index=1)
    rotated_half = torch.stack((-rope_odd, rope_even), dim=-1).flatten(-2)
    rotated = ((rope.float() * cos) + (rotated_half.float() * sin)).to(x.dtype)
    return torch.cat((nope, rotated), dim=x.dim() - 1)


class QEffDeepseekV4Attention(DeepseekV4Attention):
    """DeepSeek V4 sliding/HCA/CSA attention using fixed-capacity QEff cache state."""

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor | None,
        compressed_key: torch.Tensor | None = None,
        compressed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def repeat_key_states(states: torch.Tensor) -> torch.Tensor:
            batch, num_kv_heads, key_length, head_dim = states.shape
            states = states[:, :, None, :, :].expand(
                batch,
                num_kv_heads,
                self.num_key_value_groups,
                key_length,
                head_dim,
            )
            return states.reshape(batch, self.num_heads, key_length, head_dim)

        key_states = repeat_key_states(key)
        attn_logits = torch.matmul(query, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_logits = torch.where(
                attention_mask,
                torch.full_like(attn_logits, MIN_MASKED_ATTENTION_VALUE, dtype=attn_logits.dtype),
                attn_logits,
            )

        sinks = self.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        compressed_states = None
        compressed_logits = None
        logit_parts = [attn_logits, sinks]
        if compressed_key is not None:
            compressed_states = repeat_key_states(compressed_key)
            compressed_logits = torch.matmul(query, compressed_states.transpose(2, 3)) * self.scaling
            if compressed_mask is not None:
                compressed_logits = torch.where(
                    compressed_mask,
                    torch.full_like(compressed_logits, MIN_MASKED_ATTENTION_VALUE, dtype=compressed_logits.dtype),
                    compressed_logits,
                )
            logit_parts.append(compressed_logits)

        combined_logits = torch.cat(logit_parts, dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probabilities = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        probability_parts = probabilities.split(tuple(logits.shape[-1] for logits in logit_parts), dim=-1)
        attn_weights = probability_parts[0]
        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        ).to(key_states.dtype)
        attn_output = torch.matmul(attn_weights, key_states)
        if compressed_states is not None:
            compressed_weights = F.dropout(
                probability_parts[2],
                p=0.0 if not self.training else self.attention_dropout,
                training=self.training,
            ).to(compressed_states.dtype)
            attn_output = attn_output + torch.matmul(compressed_weights, compressed_states)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is not None and not isinstance(past_key_values, QEffDeepseekV4Cache):
            raise TypeError("QEffDeepseekV4Attention requires QEffDeepseekV4Cache.")
        if hidden_states.shape[1] != 1:
            raise ValueError("QEff attention is decode-only and requires query length 1.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states)).to(hidden_states.dtype)
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = qeff_apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).to(hidden_states.dtype).view(*hidden_shape).transpose(1, 2)
        kv = qeff_apply_rotary_pos_emb(kv, cos, sin)

        projected_kv = None
        projected_gate = None
        indexer_kv = None
        indexer_gate = None
        if self.compressor is not None:
            projected_kv = self.compressor.kv_proj(hidden_states)
            projected_gate = self.compressor.gate_proj(hidden_states)
            if self.layer_type == "compressed_sparse_attention":
                indexer_kv = self.compressor.indexer.kv_proj(hidden_states)
                indexer_gate = self.compressor.indexer.gate_proj(hidden_states)

        if past_key_values is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise ValueError("QEff cached attention requires a tensor attention_mask.")
            kv = past_key_values.update(
                kv,
                kv,
                self.layer_idx,
                {
                    "position_ids": position_ids,
                    "context_length": attention_mask.shape[-1],
                    "compressor_kv": projected_kv,
                    "compressor_gate": projected_gate,
                    "indexer_kv": indexer_kv,
                    "indexer_gate": indexer_gate,
                },
            )[0]

        block_bias = None
        compressed_kv = None
        if (
            self.compressor is not None
            and past_key_values is not None
            and self.layer_type == "heavily_compressed_attention"
        ):
            layer = past_key_values.layers[self.layer_idx]
            weighted_gate = layer.compressor_gate_buffer + self.compressor.position_bias.view(
                1, 1, layer.compression_size, -1
            )
            compressed = self.compressor.kv_norm(
                torch.einsum(
                    "bhrd->bhd",
                    layer.compressor_kv_buffer * weighted_gate.softmax(dim=2, dtype=torch.float32).to(layer.dtype),
                )
            ).to(layer.dtype)
            entry_positions = torch.div(position_ids, layer.compression_size, rounding_mode="floor")
            rope_positions = entry_positions * layer.compression_size
            comp_cos, comp_sin = self.compressor.rotary_emb(
                compressed,
                position_ids=rope_positions,
                layer_type=self.compressor.rope_layer_type,
            )
            compressed = qeff_apply_rotary_pos_emb(compressed.unsqueeze(1), comp_cos, comp_sin).squeeze(1)
            write_mask = torch.remainder(position_ids + 1, layer.compression_size) == 0
            compressed_kv = layer.update_compressor_states(
                "compressor",
                compressed,
                entry_positions=entry_positions,
                write_mask=write_mask,
            )
            compressed_capacity = compressed_kv.shape[2]
            completed_entries = torch.div(position_ids + 1, layer.compression_size, rounding_mode="floor")
            entry_indices = torch.arange(
                compressed_capacity, device=compressed_kv.device, dtype=completed_entries.dtype
            )
            block_bias = entry_indices.view(1, 1, 1, -1) >= completed_entries.unsqueeze(1).unsqueeze(-1)
        elif (
            self.compressor is not None
            and past_key_values is not None
            and self.layer_type == "compressed_sparse_attention"
        ):
            layer = past_key_values.layers[self.layer_idx]

            def build_csa_pingpong_compressed(
                name: str,
                kv_buffer: torch.Tensor,
                gate_buffer: torch.Tensor,
                position_bias: torch.Tensor,
                norm: nn.Module,
                rotary: DeepseekV4RotaryEmbedding,
                head_dim: int,
            ) -> torch.Tensor:
                ratio = layer.compression_size
                entry_positions = torch.div(position_ids, ratio, rounding_mode="floor")
                current_bank = torch.remainder(entry_positions, 2).to(torch.int32)
                previous_bank = 1 - current_bank
                slot_offsets = torch.arange(ratio, device=position_ids.device, dtype=torch.int32).unsqueeze(0)
                current_positions = current_bank.to(torch.int32) * ratio + slot_offsets
                previous_positions = previous_bank.to(torch.int32) * ratio + slot_offsets
                current_kv = CtxGatherBlockedKVFunc.apply(kv_buffer, current_positions.unsqueeze(1))[:, 0]
                current_gate = CtxGatherBlockedKVFunc.apply(gate_buffer, current_positions.unsqueeze(1))[:, 0]
                previous_kv = CtxGatherBlockedKVFunc.apply(kv_buffer, previous_positions.unsqueeze(1))[:, 0]
                previous_gate = CtxGatherBlockedKVFunc.apply(gate_buffer, previous_positions.unsqueeze(1))[:, 0]
                bias = position_bias.view(1, ratio, 2 * head_dim)
                current_gate = current_gate + bias
                previous_gate = previous_gate + bias
                new_kv = current_kv.new_zeros((current_kv.shape[0], 2 * ratio, head_dim))
                new_gate = current_gate.new_full((current_gate.shape[0], 2 * ratio, head_dim), float("-inf"))
                new_kv[:, :ratio] = previous_kv[..., :head_dim]
                previous_valid = entry_positions > 0
                new_gate[:, :ratio] = torch.where(
                    previous_valid.unsqueeze(-1),
                    previous_gate[..., :head_dim],
                    new_gate[:, :ratio],
                )
                new_kv[:, ratio:] = current_kv[..., head_dim:]
                new_gate[:, ratio:] = current_gate[..., head_dim:]
                compressed = norm(
                    torch.einsum("brd->bd", new_kv * new_gate.softmax(dim=1, dtype=torch.float32).to(new_kv.dtype))
                ).to(new_kv.dtype)
                rope_positions = entry_positions * ratio
                comp_cos, comp_sin = rotary(
                    compressed,
                    position_ids=rope_positions,
                    layer_type=self.compressor.rope_layer_type,
                )
                compressed = (
                    qeff_apply_rotary_pos_emb(compressed.unsqueeze(1).unsqueeze(1), comp_cos, comp_sin)
                    .squeeze(1)
                    .squeeze(1)
                )
                write_mask = torch.remainder(position_ids + 1, ratio) == 0
                return layer.update_csa_compressed_state(
                    name,
                    compressed,
                    entry_positions,
                    write_mask,
                )

            compressed_kv = build_csa_pingpong_compressed(
                "compressor",
                layer.compressor_kv_buffer,
                layer.compressor_gate_buffer,
                self.compressor.position_bias,
                self.compressor.kv_norm,
                self.compressor.rotary_emb,
                self.compressor.head_dim,
            )
            indexer_compressed = build_csa_pingpong_compressed(
                "indexer",
                layer.indexer_kv_buffer,
                layer.indexer_gate_buffer,
                self.compressor.indexer.position_bias,
                self.compressor.indexer.kv_norm,
                self.compressor.indexer.rotary_emb,
                self.compressor.indexer.head_dim,
            )
            completed_entries = torch.div(position_ids + 1, layer.compression_size, rounding_mode="floor")
            compressed_capacity = compressed_kv.shape[2]
            entry_indices = torch.arange(
                compressed_capacity, device=compressed_kv.device, dtype=completed_entries.dtype
            )

            indexer = self.compressor.indexer
            cos_q, sin_q = indexer.rotary_emb(
                hidden_states, position_ids=position_ids, layer_type=indexer.rope_layer_type
            )
            q_index = (
                indexer.q_b_proj(q_residual)
                .view(hidden_states.shape[0], 1, indexer.num_heads, indexer.head_dim)
                .transpose(1, 2)
            )
            q_index = qeff_apply_rotary_pos_emb(q_index, cos_q, sin_q).transpose(1, 2)
            index_scores = indexer.scorer(q_index, indexer_compressed[:, 0], hidden_states)
            future_mask = entry_indices.view(1, 1, -1) >= completed_entries.unsqueeze(-1)
            index_scores = index_scores.masked_fill(future_mask, float("-inf"))
            top_k = min(indexer.index_topk, compressed_capacity)
            top_k_indices = index_scores.topk(top_k, dim=-1).indices
            valid = top_k_indices < completed_entries.unsqueeze(-1)
            safe_indices = torch.where(valid, top_k_indices, torch.zeros_like(top_k_indices)).to(torch.int32)
            compressed_kv = CtxGatherBlockedKVFunc.apply(compressed_kv, safe_indices[:, 0].unsqueeze(1))
            block_bias = ~valid[:, None, :, :]
        elif self.compressor is not None:
            compressed_kv, block_bias = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )

        attn_output, attn_weights = self._attention_forward(
            q,
            kv,
            attention_mask,
            compressed_kv,
            block_bias,
        )

        attn_output = qeff_apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights


class QEffDeepseekV4Cache(Cache):
    """Multi-layer fixed-capacity cache represented as tensor tuples at model I/O."""

    _HCA_STATE_NAMES = (
        "sliding_window_kv",
        "compressor_kv_buffer",
        "compressor_gate_buffer",
        "actual_compressed_kv",
    )
    _CSA_STATE_NAMES = (
        "sliding_window_kv",
        "compressor_kv_buffer",
        "compressor_gate_buffer",
        "actual_compressed_kv",
        "indexer_kv_buffer",
        "indexer_gate_buffer",
        "actual_indexer_compressed_kv",
    )

    def __init__(self, layers: list[CacheLayerMixin]) -> None:
        super().__init__(layers=layers)

    @staticmethod
    def _position(position_ids: torch.Tensor) -> int:
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            return 0
        return int(position_ids[:, 0].min().item())

    @classmethod
    def from_legacy_cache(
        cls,
        config: DeepseekV4Config,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...],
        position_ids: torch.Tensor,
    ) -> "QEffDeepseekV4Cache":
        if len(past_key_values) != config.num_hidden_layers:
            raise ValueError("DeepSeek V4 cache must contain one state tuple per decoder layer.")
        cumulative_length = cls._position(position_ids)
        layers = []
        for layer_type, states in zip(config.layer_types, past_key_values):
            if layer_type == "sliding_attention":
                if len(states) != 1:
                    raise ValueError("Sliding-attention cache layers require one retained-state tensor.")
                layers.append(QEffSlidingCacheLayer(config, states[0], cumulative_length=cumulative_length))
            elif layer_type == "heavily_compressed_attention":
                if len(states) != len(cls._HCA_STATE_NAMES):
                    raise ValueError("HCA cache layers require four retained-state tensors.")
                ratio = config.compress_rates[layer_type]
                layers.append(
                    QEffHCACacheLayer(
                        config,
                        *states,
                        cumulative_length=cumulative_length,
                        compressor_entry_count=cumulative_length // ratio,
                    )
                )
            elif layer_type == "compressed_sparse_attention":
                if len(states) != len(cls._CSA_STATE_NAMES):
                    raise ValueError("CSA cache layers require seven retained-state tensors.")
                ratio = config.compress_rates[layer_type]
                entry_count = cumulative_length // ratio
                layers.append(
                    QEffCSACacheLayer(
                        config,
                        *states,
                        cumulative_length=cumulative_length,
                        compressor_entry_count=entry_count,
                        indexer_entry_count=entry_count,
                    )
                )
            else:
                raise ValueError(f"Unsupported DeepSeek V4 attention layer type: {layer_type}")
        return cls(layers)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        states = []
        for layer in self.layers:
            if isinstance(layer, QEffSlidingCacheLayer):
                names = ("sliding_window_kv",)
            elif isinstance(layer, QEffHCACacheLayer):
                names = self._HCA_STATE_NAMES
            elif isinstance(layer, QEffCSACacheLayer):
                names = self._CSA_STATE_NAMES
            else:
                raise TypeError(f"Unsupported DeepSeek V4 cache layer: {type(layer).__name__}")
            states.append(tuple(getattr(layer, name) for name in names))
        return tuple(states)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    @classmethod
    def get_dummy_cache(
        cls,
        config: DeepseekV4Config,
        batch_size: int,
        ctx_len: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        common = {"device": device, "dtype": dtype}
        layers = []
        for layer_type in config.layer_types:
            sliding = torch.zeros(batch_size, config.num_key_value_heads, ctx_len, config.head_dim, **common)
            if layer_type == "sliding_attention":
                layers.append((sliding,))
                continue
            ratio = config.compress_rates[layer_type]
            capacity = (ctx_len + ratio - 1) // ratio
            if layer_type == "heavily_compressed_attention":
                layers.append(
                    (
                        sliding,
                        torch.zeros(batch_size, 1, ratio, config.head_dim, **common),
                        torch.zeros(batch_size, 1, ratio, config.head_dim, **common),
                        torch.zeros(batch_size, 1, capacity, config.head_dim, **common),
                    )
                )
                continue
            banked = 2 * ratio
            layers.append(
                (
                    sliding,
                    torch.zeros(batch_size, 1, banked, 2 * config.head_dim, **common),
                    torch.zeros(batch_size, 1, banked, 2 * config.head_dim, **common),
                    torch.zeros(batch_size, 1, capacity, config.head_dim, **common),
                    torch.zeros(batch_size, 1, banked, 2 * config.index_head_dim, **common),
                    torch.zeros(batch_size, 1, banked, 2 * config.index_head_dim, **common),
                    torch.zeros(batch_size, 1, capacity, config.index_head_dim, **common),
                )
            )
        return tuple(layers)


class QEffDeepseekV4Experts(DeepseekV4Experts):
    """Gather routed expert weights and evaluate them as activations with BMM."""

    def __qeff_init__(self):
        self.expert_dim = self.intermediate_dim
        self.gate_proj = nn.Parameter(self.gate_up_proj[:, : self.expert_dim, :].transpose(1, 2).detach().clone())
        self.up_proj = nn.Parameter(self.gate_up_proj[:, self.expert_dim :, :].transpose(1, 2).detach().clone())
        self.down_proj = nn.Parameter(self.down_proj.transpose(1, 2).detach().clone())
        delattr(self, "gate_up_proj")

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        gate_proj = self.gate_proj[top_k_index.flatten()]
        up_proj = self.up_proj[top_k_index.flatten()]
        down_proj = self.down_proj[top_k_index.flatten()]
        expert_in = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).contiguous().view(-1, 1, hidden_dim)
        gate = torch.bmm(expert_in, gate_proj)
        up = torch.bmm(expert_in, up_proj)
        gate = gate.float().clamp(max=self.limit).to(gate.dtype)
        up = up.float().clamp(min=-self.limit, max=self.limit).to(up.dtype)
        activated = self.act_fn(gate) * up
        expert_out = torch.bmm(activated, down_proj).view(num_tokens, self.top_k, hidden_dim)
        expert_out = expert_out * top_k_weights.unsqueeze(-1)
        return torch.einsum("tkh->th", expert_out)


class QEffDeepseekV4MLP(DeepseekV4MLP):
    """Shared expert MLP with compiler-supported SwiGLU bounds."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        gate = gate.float().clamp(max=self.limit).to(gate.dtype)
        up = up.float().clamp(min=-self.limit, max=self.limit).to(up.dtype)
        return self.down_proj(self.act_fn(gate) * up)


class QEffDeepseekV4RMSNorm(CustomRMSNormAIC):
    """DeepSeek V4 FP32 normalization around the QEff RMSNorm custom operation."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = CustomRMSNormFunc.apply(
            hidden_states.float(),
            self.weight.float(),
            self.variance_epsilon,
        )
        return normalized.to(input_dtype)


class QEffDeepseekV4UnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    """DeepSeek V4 unweighted FP32 normalization using the QEff RMSNorm custom operation."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        unit_weight = hidden_states.new_ones(hidden_states.shape[-1], dtype=torch.float32)
        normalized = CustomRMSNormFunc.apply(hidden_states.float(), unit_weight, self.eps)
        return normalized.to(input_dtype)


class QEffDeepseekV4SparseMoeBlock(DeepseekV4SparseMoeBlock):
    def __qeff_init__(self):
        self.experts.top_k = self.gate.top_k

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(hidden_states)


class QEffDeepseekV4TopKRouter(DeepseekV4TopKRouter):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (torch.einsum("tk->t", weights).unsqueeze(-1) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class QEffDeepseekV4HashRouter(DeepseekV4HashRouter):
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].reshape(-1, self.top_k).long()
        weights = scores.gather(1, indices)
        weights = weights / (torch.einsum("tk->t", weights).unsqueeze(-1) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class QEffDeepseekV4HyperConnection(DeepseekV4HyperConnection):
    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (torch.einsum("bsij->bsj", comb).unsqueeze(-2) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (torch.einsum("bsij->bsi", comb).unsqueeze(-1) + self.hc_eps)
            comb = comb / (torch.einsum("bsij->bsj", comb).unsqueeze(-2) + self.hc_eps)
        collapsed = torch.einsum("bsh,bshd->bsd", pre, hidden_streams.float()).to(hidden_streams.dtype)
        return post, comb, collapsed


class QEffDeepseekV4HyperHead(DeepseekV4HyperHead):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return torch.einsum("bsh,bshd->bsd", pre, x.float()).to(x.dtype)


class QEffDeepseekV4DecoderLayer(DeepseekV4DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        normalized = self.input_layernorm(collapsed.to(dtype)).to(dtype)
        attn_output, _ = self.self_attn(normalized, **kwargs)
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        normalized = self.post_attention_layernorm(collapsed.to(dtype)).to(dtype)
        mlp_output = self.mlp(normalized, input_ids=input_ids)
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )


class QEffDeepseekV4Model(DeepseekV4Model):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[1] != 1:
            raise ValueError("DeepSeek V4 QEff integration is decode-only and requires one token per call.")
        if position_ids is None:
            raise ValueError("Decode-only DeepSeek V4 requires explicit position_ids.")
        if past_key_values is None:
            ctx_len = attention_mask.shape[-1] if attention_mask is not None else self.config.max_seq_len_cached
            if ctx_len is None:
                raise ValueError("A cache or attention_mask is required to determine retained-state capacity.")
            past_key_values = QEffDeepseekV4Cache.get_dummy_cache(
                self.config, inputs_embeds.shape[0], ctx_len, inputs_embeds.dtype, inputs_embeds.device
            )
        cache = QEffDeepseekV4Cache.from_legacy_cache(self.config, past_key_values, position_ids)
        ctx_len = cache.layers[0].max_cache_len
        physical_position_ids = torch.full_like(position_ids, ctx_len - 1)
        causal_mask = _create_causal_mask(
            position_ids=physical_position_ids,
            target_length=ctx_len,
            sliding_window=self.config.sliding_window,
        )
        cache_indices = torch.arange(ctx_len, device=position_ids.device).view(1, 1, 1, -1)
        uninitialized_prefix_mask = cache_indices < (ctx_len - position_ids - 1)[:, None, :, None]
        causal_mask = causal_mask | uninitialized_prefix_mask
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=cache,
                **kwargs,
            )
        hidden_states = self.norm(self.hc_head(hidden_states)).to(inputs_embeds.dtype)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=cache.to_legacy_cache() if use_cache is not False else None,
        )


class QEffDeepseekV4ForCausalLM(DeepseekV4ForCausalLM):
    def get_submodules_for_export(self) -> type[nn.Module]:
        return {QEffDeepseekV4DecoderLayer}

    def get_specializations(self, batch_size: int, prefill_seq_len: int, ctx_len: int, **kwargs: Any) -> list[dict]:
        del kwargs
        compressed_capacities = {
            f"compressed_ctx_len_{layer_idx}": (ctx_len + self.config.compress_rates[layer_type] - 1)
            // self.config.compress_rates[layer_type]
            for layer_idx, layer_type in enumerate(self.config.layer_types)
            if layer_type != "sliding_attention"
        }
        return [
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "ctx_len": ctx_len,
                **compressed_capacities,
            }
            for seq_len in (prefill_seq_len, 1)
        ]

    def get_dummy_pkv_cache(self, config: DeepseekV4Config, batch_size: int, ctx_len: int):
        dtype = next(self.parameters()).dtype
        return QEffDeepseekV4Cache.get_dummy_cache(config, batch_size, ctx_len, dtype)

    def get_onnx_past_key_value_names(self, layer_idx: int, layer_state: tuple[torch.Tensor, ...]) -> list[str]:
        layer_type = self.config.layer_types[layer_idx]
        if layer_type == "sliding_attention":
            state_names = ("sliding_window_kv",)
        elif layer_type == "heavily_compressed_attention":
            state_names = QEffDeepseekV4Cache._HCA_STATE_NAMES
        elif layer_type == "compressed_sparse_attention":
            state_names = QEffDeepseekV4Cache._CSA_STATE_NAMES
        else:
            raise ValueError(f"Unsupported DeepSeek V4 attention layer type: {layer_type}")
        if len(layer_state) != len(state_names):
            raise ValueError(f"Layer {layer_idx} cache has {len(layer_state)} tensors; expected {len(state_names)}.")
        return [f"past_{name}.{layer_idx}" for name in state_names]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> MoeCausalLMOutputWithPast:
        if labels is not None:
            raise NotImplementedError("Training is not supported by the decode-only DeepSeek V4 wrapper.")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        logit_index = position_ids.to(torch.int32).argmax(dim=1, keepdim=True)
        batch_index = torch.arange(position_ids.shape[0], device=position_ids.device).view(-1, 1)
        hidden_states = outputs.last_hidden_state[batch_index, logit_index]
        logits = self.lm_head(hidden_states).float()
        return MoeCausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)
