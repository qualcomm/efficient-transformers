# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache, CacheLayerMixin, EncoderDecoderCache

from QEfficient.customop import (
    ctx_chunk_scatter_batch,
    ctx_gather,
    ctx_gather_3d,
    ctx_gather_block_range_kv_dp,
    ctx_gather_blocked_kv,
    ctx_gather_blocked_kv_batch,
    ctx_gather_blocked_kv_cb,
    ctx_gather_blocked_kv_dp,
    ctx_gather_cb,
    ctx_gather_cb_3d,
    ctx_gather_paged_attention,
    ctx_gather_paged_kv_dp,
    ctx_gather_paged_kv_heads,
    ctx_paged_scatter_dp,
    ctx_paged_scatter_page,
    ctx_scatter,
    ctx_scatter_3d,
    ctx_scatter_cb,
    ctx_scatter_cb_3d,
    ctx_scatter_paged_attention,
    m3_ctx_scatter,
)


# HybridCache and HybridChunkedCache were removed from transformers in 5.3+.
# Define lightweight local stubs so downstream QEff wrappers can still inherit from them.
class HybridCache:  # type: ignore[no-redef]
    pass


class HybridChunkedCache:  # type: ignore[no-redef]
    pass


class InvalidIndexProvider:
    SUBFUNC_ENABLED = False

    @classmethod
    def enable_subfunc(cls):
        cls.SUBFUNC_ENABLED = True

    @classmethod
    def _get_invalid_idx_value(cls):
        """
        Get the appropriate invalid index value for CtxGather operations.

        For ONNX export with functions, we use 0 to avoid INT32_MAX constants
        that cause issues when functions are inlined at runtime.

        Returns:
            int: Invalid index value (0 for ONNX functions, INT32_MAX otherwise)
        """
        if torch.onnx.is_in_onnx_export():
            if cls.SUBFUNC_ENABLED or torch._dynamo.is_compiling():
                # TODO: should not return 0 remove this if condition, it can hurt perf
                return 0
            else:
                return torch.iinfo(torch.int32).max
        else:
            return 0


def _match_invalid_mask(invalid_mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if invalid_mask.shape[-1] == target_len:
        return invalid_mask
    return invalid_mask[..., :target_len]


def _remainder_with_symbolic_divisor(value: torch.Tensor, divisor) -> torch.Tensor:
    if torch.is_tensor(divisor):
        divisor_tensor = divisor.to(device=value.device, dtype=value.dtype)
    else:
        divisor_tensor = torch.scalar_tensor(divisor, dtype=value.dtype, device=value.device)
    return torch.remainder(value, divisor_tensor)


def read_kv_cache_with_indices(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    token_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected key/value slices from a flat [batch, heads, ctx_len, head_dim] cache.

    token_indices: int32 tensor of shape [batch, heads, n_tokens] where invalid entries are 0.
    Returns selected_k, selected_v both of shape [batch, heads, n_tokens, head_dim].
    """
    return ctx_gather_blocked_kv(key_cache, token_indices), ctx_gather_blocked_kv(value_cache, token_indices)


def update_and_read_index_key_cache(
    index_key_cache: torch.Tensor,
    position_ids: torch.Tensor,
    idx_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter idx_k into index_key_cache at position_ids, then read all ctx_len index keys.

    Future positions (beyond max position_id) are masked with INT32_MAX (ONNX export) or 0 (eager).
    Returns (gathered_index_keys [batch, 1, ctx_len, head_dim], updated_index_key_cache).
    """
    index_key_cache = m3_ctx_scatter(index_key_cache, position_ids.to(torch.int32), idx_k)
    batch, _, ctx_len, _ = index_key_cache.shape
    ctx_indices = torch.arange(ctx_len, device=index_key_cache.device)[None, None, :]
    gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
    invalid_idx = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
    ctx_indices = torch.where(ctx_indices > gather_limit, invalid_idx, ctx_indices).to(torch.int32)
    ctx_indices = ctx_indices.expand(batch, 1, ctx_len)
    return ctx_gather_blocked_kv(index_key_cache, ctx_indices), index_key_cache


def scatter_kv_into_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_ids: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write key_states and value_states into the flat KV cache at position_ids."""
    pid = position_ids.to(torch.int32)
    return ctx_scatter(key_cache, pid, key_states), ctx_scatter(value_cache, pid, value_states)


class QEffDynamicLayer(CacheLayerMixin):
    is_compileable = False

    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.is_initialized = False
        self.device = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.keys = key_states
        self.values = value_states
        self.is_initialized = True
        self.device = key_states.device

    @classmethod
    def from_tensors(cls, key_states: torch.Tensor, value_states: torch.Tensor) -> "QEffDynamicLayer":
        layer = cls()
        layer.keys = key_states
        layer.values = value_states
        layer.is_initialized = True
        layer.device = key_states.device
        return layer

    def _mark_initialized(self, reference_states: torch.Tensor) -> None:
        if not self.is_initialized:
            self.dtype = reference_states.dtype
            self.device = reference_states.device
            self.is_initialized = True

    def get_mask_sizes(self, cache_position: torch.Tensor | int) -> tuple[int, int]:
        query_length = cache_position if isinstance(cache_position, int) else cache_position.shape[0]
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self.keys.shape[-2] if self.keys is not None else 0

    def get_max_length(self) -> int:
        return -1

    def get_max_cache_shape(self) -> int:
        return -1

    @property
    def max_batch_size(self) -> int:
        return self.keys.shape[0] if self.keys is not None else 0

    @property
    def max_cache_len(self) -> int:
        return self.keys.shape[-2] if self.keys is not None else 0

    def reset(self) -> None:
        if self.keys is not None:
            self.keys.zero_()
        if self.values is not None:
            self.values.zero_()

    def offload(self) -> None:
        if self.keys is not None and self.values is not None:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        if (
            self.keys is not None
            and self.values is not None
            and self.device is not None
            and self.keys.device != self.device
        ):
            self.keys = self.keys.to(self.device, non_blocking=True)
            self.values = self.values.to(self.device, non_blocking=True)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.keys is not None and self.values is not None and self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))

    def crop(self, max_length: int) -> None:
        if self.keys is not None:
            self.keys = self.keys[:, :, :max_length, :]
        if self.values is not None:
            self.values = self.values[:, :, :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.keys is not None:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
        if self.values is not None:
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.keys is not None:
            self.keys = self.keys[indices, ...]
        if self.values is not None:
            self.values = self.values[indices, ...]

    def read_only(self, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Gather
        k_out, v_out = self.keys, self.values
        if k_out is not None:
            self._mark_initialized(k_out)
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        ctx_len = cache_kwargs.get("CCL", k_out.shape[2])

        ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(torch.int32)
        invalid_mask = ctx_indices > gather_limit

        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = ctx_gather_cb(k_out, batch_index, ctx_indices, ctx_len)
            v_out = ctx_gather_cb(v_out, batch_index, ctx_indices, ctx_len)
        else:
            k_out = ctx_gather(k_out, ctx_indices, ctx_len)
            v_out = ctx_gather(v_out, ctx_indices, ctx_len)

        invalid_mask = _match_invalid_mask(invalid_mask, v_out.shape[-2])
        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)

        return k_out, v_out

    def read_only_blocked_K(self, start_index, end_index, cache_kwargs):
        """
        Reads the `key_states` for each KV block.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

            start_index (`int`):
                Start index of the K/V block to read

            end_index (`int`):
                End index of the K/V block to read

        Return:
            the updated key states.
        """
        # Gather
        k_out = self.keys
        if k_out is not None:
            self._mark_initialized(k_out)
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        batch, num_kv_heads, _, _ = k_out.shape
        ctx_indices = torch.arange(
            start=start_index, end=end_index, dtype=position_ids.dtype, device=position_ids.device
        )[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit

        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = ctx_gather_blocked_kv_cb(k_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            k_out = ctx_gather_blocked_kv(k_out, ctx_indices)

        return k_out

    def get_batch_folded_kv(self):
        """Return K/V views with batch and KV-head axes folded together."""
        key_states, value_states = self.keys, self.values
        if key_states is not None:
            self._mark_initialized(key_states)
        full_batch_size, num_kv_heads, ctx_len, head_dim = key_states.shape
        folded_shape = (1, full_batch_size * num_kv_heads, ctx_len, head_dim)
        return key_states.reshape(folded_shape), value_states.reshape(folded_shape)

    def read_only_blocked_K_batch(self, start_index, end_index, cache_kwargs, folded_cache=None):
        """
        Batch-folded counterpart of _read_blocked_k. The retained cache keeps
        the standard [FBS, Hkv, T, D] layout and is viewed as
        [1, FBS*Hkv, T, D] only for the optimized gather. The gathered block
        uses [1, B*Hkv, T_block, D], where B is the execution batch size.

        Merges the B/Hkv axes into BH on the tiny [B]-sized gather_limit
        (via reshape) rather than on the T_block-sized ctx_indices tensor —
        letting Greater/Where broadcast ctx_indices straight to [1, BH,
        T_block] keeps the op chain feeding CtxGatherBlockedKVBatch as
        Where->Cast->Expand (matching _read_blocked_k's pattern), instead of
        ending in an axis-merging Reshape. The compiler's GatherNDRange
        pattern match (an on-chip range-gather fast path) apparently keys off
        that exact tail shape and falls back to a plain DDR GatherND when a
        merge Reshape sits right before the gather.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

            start_index (`int`):
                Start index of the K/V block to read

            end_index (`int`):
                End index of the K/V block to read

        Return:
            the updated key states.
        """
        # Gather
        k_out = self.keys
        if k_out is not None:
            self._mark_initialized(k_out)
        position_ids = cache_kwargs.get("position_ids")
        # batch_index = cache_kwargs.get("batch_index", None)
        full_batch_size, Hkv, ctx_len, head_dim = k_out.shape
        cache_bh = full_batch_size * Hkv
        if folded_cache is None:
            k_out = k_out.reshape(1, cache_bh, ctx_len, head_dim)
        else:
            k_out = folded_cache
        # Batch-folded decode prepares the range/where chain in the owning
        # attention-layer body and passes it through these private kwargs. Do
        # not rebuild it here: doing so allows ONNX export to lift the graph
        # out of the decoder-layer subfunction, creating extra function
        # variants and preventing compiler range optimization.
        prepared_ctx_indices = cache_kwargs.get("_qeff_batch_fold_ctx_indices")
        if prepared_ctx_indices is None:
            # Preserve the original reader-local path for non-batch-folded
            # callers and for any legacy invocation that does not provide the
            # prepared tensors. This fallback is intentionally unchanged in
            # semantics and continues to use the cache's invalid-index value.
            T_block = end_index - start_index
            ctx_indices = torch.arange(start=start_index, end=end_index)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values
            gather_limit = torch.cat([gather_limit] * Hkv, dim=1).reshape(1, -1, 1)
            invalid_mask = ctx_indices > gather_limit

            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            ctx_indices = ctx_indices.expand(1, cache_bh, T_block)
        else:
            # The attention forward already expanded this tensor to the folded
            # [1, FBS * Hkv, T_block] layout, so no additional reshape or range
            # construction is needed in the cache reader.
            ctx_indices = prepared_ctx_indices
        k_out = ctx_gather_blocked_kv_batch(k_out, ctx_indices)

        return k_out

    def read_only_blocked_V(self, start_index, end_index, cache_kwargs):
        """
        Reads the `value_states` for the layer for each KV block.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

            start_index (`int`):
                Start index of the K/V block to read

            end_index (`int`):
                End index of the K/V block to read

        Return:
            the updated value states.
        """
        # Gather
        v_out = self.values
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        batch, num_kv_heads, _, _ = v_out.shape
        ctx_indices = torch.arange(
            start=start_index, end=end_index, dtype=position_ids.dtype, device=position_ids.device
        )[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit

        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            v_out = ctx_gather_blocked_kv_cb(v_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            v_out = ctx_gather_blocked_kv(v_out, ctx_indices)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)
        return v_out

    def read_only_blocked_V_batch(self, start_index, end_index, cache_kwargs, folded_cache=None):
        """
        Reads a value block through a folded view of the standard retained
        cache layout.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

            start_index (`int`):
                Start index of the K/V block to read

            end_index (`int`):
                End index of the K/V block to read

        Return:
            the updated value states.
        """
        # Gather
        v_out = self.values
        position_ids = cache_kwargs.get("position_ids")
        # batch_index = cache_kwargs.get("batch_index", None)
        full_batch_size, Hkv, ctx_len, head_dim = v_out.shape
        cache_bh = full_batch_size * Hkv
        if folded_cache is None:
            v_out = v_out.reshape(1, cache_bh, ctx_len, head_dim)
        else:
            v_out = folded_cache
        # See read_only_blocked_K_batch above. The precomputed tensors are
        # shared by the K and V reads so both sides of the attention operation
        # use the exact same folded-cache selection.
        prepared_ctx_indices = cache_kwargs.get("_qeff_batch_fold_ctx_indices")
        if prepared_ctx_indices is None:
            # Keep the generic fallback for callers outside the optimized
            # batch-folded decode path.
            T_block = end_index - start_index
            ctx_indices = torch.arange(start=start_index, end=end_index)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values
            gather_limit = torch.cat([gather_limit] * Hkv, dim=1).reshape(1, -1, 1)
            invalid_mask = ctx_indices > gather_limit

            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            ctx_indices = ctx_indices.expand(1, cache_bh, T_block)
        else:
            ctx_indices = prepared_ctx_indices
            # Unlike the K read, the V read must retain the mask after the
            # gather so invalid/future positions can be zeroed explicitly.
            invalid_mask = cache_kwargs["_qeff_batch_fold_invalid_mask"]
        v_out = ctx_gather_blocked_kv_batch(v_out, ctx_indices)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)
        return v_out

    def read_only_blocked_kv(self, start_index, end_index, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer for each KV block.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

            start_index (`int`):
                Start index of the K/V block to read

            end_index (`int`):
                End index of the K/V block to read

        Return:
            A tuple containing the updated key and value states.
        """
        # Gather
        k_out, v_out = self.keys, self.values
        if k_out is not None:
            self._mark_initialized(k_out)
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        batch, num_kv_heads, _, _ = k_out.shape
        ctx_indices = torch.arange(
            start=start_index, end=end_index, dtype=position_ids.dtype, device=position_ids.device
        )[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(torch.int32)
        invalid_mask = ctx_indices > gather_limit

        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = ctx_gather_blocked_kv_cb(k_out, batch_index, ctx_indices)
            v_out = ctx_gather_blocked_kv_cb(v_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            k_out = ctx_gather_blocked_kv(k_out, ctx_indices)
            v_out = ctx_gather_blocked_kv(v_out, ctx_indices)

        invalid_mask = _match_invalid_mask(invalid_mask, v_out.shape[-2])
        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)

        return k_out, v_out

    def read_only_paged_attention(self, block_index, updated, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer for each KV block.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

            block_index :
                Block index of the K/V block to read from the block table

            updated:
                Was the current block updated during the current write? If yes, then read position_ids.max % block_size worth of entires from current block

        Return:
            A tuple containing the updated key and value states.
        """
        # Gather
        k_out, v_out = self.keys, self.values
        if k_out is not None:
            self._mark_initialized(k_out)
        position_ids = cache_kwargs.get("position_ids")
        batch, seq_len = position_ids.shape
        num_kv_blocks, num_kv_heads, block_size, dh = k_out.shape
        ctx_indices = torch.arange(block_size)[None, ...]
        block_fill_len = position_ids.max(1, keepdim=True).values % block_size
        gather_limit = torch.where(updated, block_fill_len, block_size)

        block_indices = block_index.unsqueeze(-1)
        gather_limit = torch.where(block_indices < 0, 0, gather_limit)
        invalid_mask = ctx_indices > gather_limit

        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        k_out = ctx_gather_paged_attention(k_out, block_indices, ctx_indices)
        v_out = ctx_gather_paged_attention(v_out, block_indices, ctx_indices)

        v_out = torch.where((invalid_mask.unsqueeze(1)).unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out

    def write_only_paged_attention(self, key_states, value_states, cache_kwargs):
        """
        Write in the cache with the new `key_states` and `value_states` for the layer.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
        else:
            self._mark_initialized(self.keys)
            position_ids = cache_kwargs.get("position_ids")
            block_table = cache_kwargs.get("block_table")  # [BS, num_kv_blocks] -> each entry is block_id value
            slot_id = cache_kwargs.get("slot_id")
            batch, num_kv_heads, seq_len, dh = key_states.shape
            num_kv_blocks, num_kv_heads, block_size, dh = self.keys.shape
            if seq_len > block_size:
                raise NotImplementedError(
                    f"write_only_paged_attention only supports writing within a single KV block per call "
                    f"(seq_len={seq_len} > block_size={block_size}). Prefill/decode chunks larger than "
                    f"block_size must be split by the caller before writing to a paged KV cache."
                )
            if int((slot_id + seq_len).max()) > block_size:
                raise NotImplementedError(
                    f"write_only_paged_attention write would cross a KV block boundary "
                    f"(slot_id + seq_len > block_size={block_size}). Writes must stay within a single "
                    f"physical block; split the write across blocks before calling this method."
                )
            block_index = position_ids.max(1).values // block_size  # Assuming only 1 block is written at max
            invalid_scatter_index = torch.iinfo(torch.int32).max
            ctx_indices = torch.arange(seq_len) + slot_id.unsqueeze(-1)

            rows = torch.arange(batch)
            block_id = block_table[rows, block_index].unsqueeze(-1)
            ctx_indices = torch.where(block_id < 0, invalid_scatter_index, ctx_indices)
            block_id = block_id.unsqueeze(-1)
            self.keys = ctx_scatter_paged_attention(self.keys, block_id, ctx_indices, key_states)
            self.values = ctx_scatter_paged_attention(self.values, block_id, ctx_indices, value_states)

    def write_only_paged_attention_dp(self, key_states, value_states, cache_kwargs):
        """Write GQA K/V into a physical page pool with rows ordered [DP, CP, Hkv]."""
        if self.keys is None or self.values is None:
            raise ValueError("Paged GQA requires preallocated physical key and value pools.")
        self._mark_initialized(self.keys)
        position_ids = cache_kwargs.get("position_ids")
        block_table = cache_kwargs.get("block_table")
        dp = int(cache_kwargs.get("attn_dp", 1))
        cp = int(cache_kwargs.get("attn_cp", 1))
        page_block_size = int(cache_kwargs.get("page_block_size", self.keys.shape[2]))
        batch, num_kv_heads, seq_len, head_dim = key_states.shape
        if batch % dp:
            raise ValueError("Paged GQA batch size must be divisible by attn_dp.")
        batch_local = batch // dp
        rows = dp * cp * num_kv_heads
        if tuple(self.keys.shape[1:]) != (rows, page_block_size, head_dim):
            raise ValueError(
                f"Paged GQA key cache must have tail shape {(rows, page_block_size, head_dim)}, "
                f"got {tuple(self.keys.shape[1:])}."
            )
        if tuple(self.values.shape) != tuple(self.keys.shape):
            raise ValueError("Paged GQA key and value cache shapes must match.")
        if block_table is None or block_table.ndim != 3 or tuple(block_table.shape[:2]) != (dp, batch_local):
            raise ValueError(f"Paged GQA block_table must have shape [{dp}, {batch_local}, pages].")

        position_dp = position_ids.view(dp, batch_local, seq_len).permute(1, 0, 2)
        logical_page = torch.div(position_dp, page_block_size, rounding_mode="trunc")
        logical_page_group = torch.div(logical_page, cp, rounding_mode="trunc")
        owner_cp = logical_page - logical_page_group * cp
        if (
            not torch.onnx.is_in_onnx_export()
            and not torch._dynamo.is_compiling()
            and int(logical_page_group.max()) >= block_table.shape[-1]
        ):
            raise ValueError("Paged GQA block_table is too short for the requested position_ids.")
        table = block_table.permute(1, 0, 2)
        physical_page = torch.gather(table, 2, logical_page_group.to(torch.int64))
        physical_page = physical_page.unsqueeze(2).unsqueeze(3).expand(batch_local, dp, cp, num_kv_heads, seq_len)
        cp_index = torch.arange(cp, device=key_states.device).view(1, 1, cp, 1, 1)
        row_live = (owner_cp.unsqueeze(2).unsqueeze(3) == cp_index).expand_as(physical_page)
        invalid_page = torch.full_like(physical_page, torch.iinfo(torch.int32).max)
        block_id = torch.where(row_live, physical_page, invalid_page).reshape(batch_local, rows, seq_len)
        address = (
            (position_dp - logical_page * page_block_size)
            .to(torch.int32)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(batch_local, dp, cp, num_kv_heads, seq_len)
            .reshape(batch_local, rows, seq_len)
        )
        key_updates = (
            key_states.view(dp, batch_local, num_kv_heads, seq_len, head_dim)
            .permute(1, 0, 2, 3, 4)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, num_kv_heads, seq_len, head_dim)
            .reshape(batch_local, rows, seq_len, head_dim)
        )
        value_updates = (
            value_states.view(dp, batch_local, num_kv_heads, seq_len, head_dim)
            .permute(1, 0, 2, 3, 4)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, num_kv_heads, seq_len, head_dim)
            .reshape(batch_local, rows, seq_len, head_dim)
        )
        is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()
        page_aligned = dp == 1 and cp == 1 and seq_len % page_block_size == 0
        if page_aligned and not is_export:
            page_aligned = bool(
                torch.all(position_ids[:, 0] % page_block_size == 0).item()
                and torch.all(position_ids[:, 1:] == position_ids[:, :-1] + 1).item()
            )
        if page_aligned:
            num_pages = seq_len // page_block_size
            logical_pages = (position_ids[:, ::page_block_size] // page_block_size).to(torch.int64)
            physical_pages = torch.gather(block_table[0].to(torch.int64), 1, logical_pages).to(torch.int32)
            for batch_idx in range(batch):
                page_ids = physical_pages[batch_idx].view(num_pages, 1).expand(num_pages, num_kv_heads)
                key_pages = (
                    key_states[batch_idx]
                    .reshape(num_kv_heads, num_pages, page_block_size, head_dim)
                    .permute(1, 0, 2, 3)
                )
                value_pages = (
                    value_states[batch_idx]
                    .reshape(num_kv_heads, num_pages, page_block_size, head_dim)
                    .permute(1, 0, 2, 3)
                )
                self.keys = ctx_paged_scatter_page(self.keys, page_ids, key_pages)
                self.values = ctx_paged_scatter_page(self.values, page_ids, value_pages)
            return self.keys, self.values
        self.keys = ctx_paged_scatter_dp(self.keys, block_id.to(torch.int32), address, key_updates)
        self.values = ctx_paged_scatter_dp(self.values, block_id.to(torch.int32), address, value_updates)
        return self.keys, self.values

    def write_only(self, key_states, value_states, cache_kwargs):
        """
        Write in the cache with the new `key_states` and `value_states` for the layer.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
        else:
            self._mark_initialized(self.keys)
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.keys = ctx_scatter_cb(self.keys, batch_index, scatter_position_ids, key_states)
                self.values = ctx_scatter_cb(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = ctx_scatter(self.keys, position_ids, key_states)
                self.values = ctx_scatter(self.values, position_ids, value_states)

    def write_only_batch(self, key_states, value_states, cache_kwargs):
        """
        Write through the batch-folded scatter while retaining the standard
        [B, Hkv, T, D] cache layout.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
        else:
            full_batch_size, Hkv, ctx_len, head_dim = self.keys.shape
            cache_bh = full_batch_size * Hkv
            self._mark_initialized(self.keys)
            position_ids = cache_kwargs.get("position_ids")

            # batch_index = cache_kwargs.get("batch_index")
            keys_folded = self.keys.reshape(1, cache_bh, ctx_len, head_dim)
            values_folded = self.values.reshape(1, cache_bh, ctx_len, head_dim)

            keys_folded = ctx_chunk_scatter_batch(keys_folded, position_ids, key_states)
            values_folded = ctx_chunk_scatter_batch(values_folded, position_ids, value_states)

            self.keys = keys_folded.reshape(full_batch_size, Hkv, ctx_len, head_dim)
            self.values = values_folded.reshape(full_batch_size, Hkv, ctx_len, head_dim)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
            k_out, v_out = self.keys, self.values
        else:
            self._mark_initialized(self.keys)
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.keys = ctx_scatter_cb(self.keys, batch_index, scatter_position_ids, key_states)
                self.values = ctx_scatter_cb(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = ctx_scatter(self.keys, position_ids, key_states)
                self.values = ctx_scatter(self.values, position_ids, value_states)

            k_out, v_out = self.keys, self.values

            # Gather
            ctx_len = cache_kwargs.get("CCL", k_out.shape[2])
            ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(torch.int32)
            invalid_mask = ctx_indices > gather_limit

            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            if batch_index is not None:
                k_out = ctx_gather_cb(k_out, batch_index, ctx_indices, ctx_len)
                v_out = ctx_gather_cb(v_out, batch_index, ctx_indices, ctx_len)
            else:
                k_out = ctx_gather(k_out, ctx_indices, ctx_len)
                v_out = ctx_gather(v_out, ctx_indices, ctx_len)

            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)

        return k_out, v_out

    # TODO:This function will be depercated in future.
    def update3D(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
            k_out, v_out = self.keys, self.values
        else:
            self._mark_initialized(self.keys)
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.keys = ctx_scatter_cb_3d(self.keys, batch_index, scatter_position_ids, key_states)
                self.values = ctx_scatter_cb_3d(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = ctx_scatter_3d(self.keys, position_ids, key_states)
                self.values = ctx_scatter_3d(self.values, position_ids, value_states)

            k_out, v_out = self.keys, self.values

            # Gather
            ctx_len = k_out.shape[1]
            ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.to(position_ids.dtype)
            invalid_mask = ctx_indices > gather_limit
            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            if batch_index is not None:
                k_out = ctx_gather_cb_3d(k_out, batch_index, ctx_indices)
                v_out = ctx_gather_cb_3d(v_out, batch_index, ctx_indices)
            else:
                k_out = ctx_gather_3d(k_out, ctx_indices)
                v_out = ctx_gather_3d(v_out, ctx_indices)

            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)

        return k_out, v_out


class QEffDynamicCompressedKVRopeLayer:
    def __init__(self, ckv, k_pe):
        self.ckv = ckv
        self.k_pe = k_pe

    def update_ckv(self, compressed_kv, cache_kwargs):
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

        if batch_index is not None:
            invalid_scatter_index = torch.iinfo(torch.int32).max
            scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
            self.ckv = ctx_scatter_cb(self.ckv, batch_index, scatter_position_ids, compressed_kv)
        else:
            self.ckv = ctx_scatter(self.ckv, position_ids, compressed_kv)

        ckv_out = self.ckv
        ctx_len = ckv_out.shape[-2]
        ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            ckv_out = ctx_gather_cb(ckv_out, batch_index, ctx_indices, ctx_len)
        else:
            ckv_out = ctx_gather(ckv_out, ctx_indices, ctx_len)
        ckv_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), ckv_out)
        return ckv_out

    def update_k_pe(self, k_pe_cache, cache_kwargs):
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

        if batch_index is not None:
            invalid_scatter_index = torch.iinfo(torch.int32).max
            scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
            self.k_pe = ctx_scatter_cb(self.k_pe, batch_index, scatter_position_ids, k_pe_cache)
        else:
            self.k_pe = ctx_scatter(self.k_pe, position_ids, k_pe_cache)

        k_pe_out = self.k_pe
        ctx_len = k_pe_out.shape[-2]
        ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_pe_out = ctx_gather_cb(k_pe_out, batch_index, ctx_indices, ctx_len)
        else:
            k_pe_out = ctx_gather(k_pe_out, ctx_indices, ctx_len)
        k_pe_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), k_pe_out)
        return k_pe_out

    def read_only_blocked_ckv(self, start_index, end_index, cache_kwargs):
        # Gather
        ckv_out = self.ckv
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        batch, num_kv_heads, _, _ = ckv_out.shape
        ctx_indices = torch.arange(
            start=start_index, end=end_index, dtype=position_ids.dtype, device=position_ids.device
        )[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit

        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            ckv_out = ctx_gather_blocked_kv_cb(ckv_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            ckv_out = ctx_gather_blocked_kv(ckv_out, ctx_indices)

        ckv_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(ckv_out, dtype=ckv_out.dtype), ckv_out)
        return ckv_out

    def read_only_blocked_k_pe(self, start_index, end_index, cache_kwargs):
        # Gather
        k_pe_out = self.k_pe
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        batch, num_kv_heads, _, _ = k_pe_out.shape
        ctx_indices = torch.arange(
            start=start_index, end=end_index, dtype=position_ids.dtype, device=position_ids.device
        )[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit

        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_pe_out = ctx_gather_blocked_kv_cb(k_pe_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            k_pe_out = ctx_gather_blocked_kv(k_pe_out, ctx_indices)

        k_pe_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(k_pe_out, dtype=k_pe_out.dtype), k_pe_out)
        return k_pe_out

    def write_only_k_pe(self, k_pe_cache, cache_kwargs):
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

        if batch_index is not None:
            invalid_scatter_index = torch.iinfo(torch.int32).max
            scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
            self.k_pe = ctx_scatter_cb(self.k_pe, batch_index, scatter_position_ids, k_pe_cache)
        else:
            self.k_pe = ctx_scatter(self.k_pe, position_ids, k_pe_cache)
        return self.k_pe

    def write_only_ckv(self, compressed_kv, cache_kwargs):
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

        if batch_index is not None:
            invalid_scatter_index = torch.iinfo(torch.int32).max
            scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
            self.ckv = ctx_scatter_cb(self.ckv, batch_index, scatter_position_ids, compressed_kv)
        else:
            self.ckv = ctx_scatter(self.ckv, position_ids, compressed_kv)
        return self.ckv


class QEffDynamicCompressedKVRopeCache:
    def __init__(
        self,
    ):
        self.layers = []

    def add_new(self, ckv, k_pe, layer_idx):
        self.layers.append(QEffDynamicCompressedKVRopeLayer(ckv, k_pe))

    def update_ckv(self, ckv, layer_idx, cache_kwargs):
        return self.layers[layer_idx].update_ckv(ckv, cache_kwargs)

    def update_k_pe(self, k_pe, layer_idx, cache_kwargs):
        return self.layers[layer_idx].update_k_pe(k_pe, cache_kwargs)

    def read_only_blocked_ckv(self, start_index, end_index, layer_idx, cache_kwargs):
        return self.layers[layer_idx].read_only_blocked_ckv(start_index, end_index, cache_kwargs)

    def read_only_blocked_k_pe(self, start_index, end_index, layer_idx, cache_kwargs):
        return self.layers[layer_idx].read_only_blocked_k_pe(start_index, end_index, cache_kwargs)

    def write_only_ckv(self, ckv, layer_idx, cache_kwargs):
        # self.append_new_layers(layer_idx)
        return self.layers[layer_idx].write_only_ckv(ckv, cache_kwargs)

    def write_only_k_pe(self, k_pe, layer_idx, cache_kwargs):
        # self.append_new_layers(layer_idx)
        return self.layers[layer_idx].write_only_k_pe(k_pe, cache_kwargs)

    @classmethod
    def from_legacy_cache(cls, past_key_values):
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                ckv, k_pe = past_key_values[layer_idx]
                cache.add_new(ckv, k_pe, layer_idx)
        return cache

    def to_legacy_cache(
        self,
    ):
        legacy_cache = ()
        for layer in self.layers:
            x = (layer.ckv, layer.k_pe)
            legacy_cache += (x,)
        return legacy_cache


class QEffDynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    - Optimized implementation for the Cloud AI 100 to reuse KV Cache.
    - get the position_ids input using kwargs.
    - Use custom Onnxscript ops to write optimized version to generate Onnx model.

    """

    def __init__(self, ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None, *args, **kwargs):
        # Remove cache-layer construction args if present to avoid duplicate arguments.
        kwargs.pop("layer_classes", None)
        kwargs.pop("layers", None)
        kwargs.pop("layer_class_to_replicate", None)

        try:
            # transformers>=4.57
            Cache.__init__(self, *args, layer_class_to_replicate=QEffDynamicLayer, **kwargs)
        except TypeError:
            # transformers<=4.56
            Cache.__init__(self, *args, layer_classes=QEffDynamicLayer, **kwargs)
        if ddp_cache_data is not None:
            for key_states, value_states in ddp_cache_data:
                self.layers.append(QEffDynamicLayer.from_tensors(key_states, value_states))

    def append_new_layers(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(QEffDynamicLayer())

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position: Optional[torch.LongTensor] = None) -> int:
        """
        Keep backward-compatible call shape while deferring to upstream implementation.
        """
        return super().get_seq_length(layer_idx)

    def read_only(self, layer_idx, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only(cache_kwargs)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        return (layer.keys, layer.values)

    def __iter__(self):
        for idx in range(len(self.layers)):
            yield self[idx]

    def read_only_blocked_kv(self, start_index, end_index, layer_idx, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            start_index (`int`):
                Start index of the K/V block to read
            end_index (`int`):
                End index of the K/V block to read
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only_blocked_kv(start_index, end_index, cache_kwargs)

    def read_only_blocked_K(self, start_index, end_index, layer_idx, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            start_index (`int`):
                Start index of the K/V block to read
            end_index (`int`):
                End index of the K/V block to read
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only_blocked_K(start_index, end_index, cache_kwargs)

    def read_only_blocked_V(self, start_index, end_index, layer_idx, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            start_index (`int`):
                Start index of the K/V block to read
            end_index (`int`):
                End index of the K/V block to read
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only_blocked_V(start_index, end_index, cache_kwargs)

    def get_batch_folded_kv(self, layer_idx):
        """Return retained K/V cache for one layer in batch-folded compute layout."""
        return self.layers[layer_idx].get_batch_folded_kv()

    def read_only_blocked_K_batch(self, start_index, end_index, layer_idx, cache_kwargs, folded_cache=None):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            start_index (`int`):
                Start index of the K/V block to read
            end_index (`int`):
                End index of the K/V block to read
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only_blocked_K_batch(
            start_index, end_index, cache_kwargs, folded_cache=folded_cache
        )

    def read_only_blocked_V_batch(self, start_index, end_index, layer_idx, cache_kwargs, folded_cache=None):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            start_index (`int`):
                Start index of the K/V block to read
            end_index (`int`):
                End index of the K/V block to read
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only_blocked_V_batch(
            start_index, end_index, cache_kwargs, folded_cache=folded_cache
        )

    def read_only_paged_attention(self, block_index, updated, layer_idx, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            start_index (`int`):
                Start index of the K/V block to read
            end_index (`int`):
                End index of the K/V block to read
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only_paged_attention(block_index, updated, cache_kwargs)
        # return self.layers[layer_idx].read_only_paged_attention(start_index, end_index, cache_kwargs)

    def write_only_paged_attention(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        Write in the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].write_only_paged_attention(key_states, value_states, cache_kwargs)

    def write_only_paged_attention_dp(self, key_states, value_states, layer_idx, cache_kwargs):
        """Write and return one layer's DP/CP-row paged GQA cache."""
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].write_only_paged_attention_dp(key_states, value_states, cache_kwargs)

    def write_only(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        Write in the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].write_only(key_states, value_states, cache_kwargs)

    def write_only_batch(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        Write in the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].write_only_batch(key_states, value_states, cache_kwargs)

    # TODO:This function will be depercated in future.
    def update3D(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].update3D(key_states, value_states, cache_kwargs)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Compatibility helper for wrappers still expecting tuple-based caches.
        """
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    ) -> "QEffDynamicCache":
        """
        Compatibility helper for tuple-based cache inputs used by older call sites.
        """
        cache = cls()
        if past_key_values is not None:
            for key_states, value_states in past_key_values:
                cache.layers.append(QEffDynamicLayer.from_tensors(key_states, value_states))
        return cache


class QEffMiniMaxSparseCache(QEffDynamicCache):
    """
    QEffDynamicCache extended with per-layer index key caches for MiniMax-M3 sparse attention.

    Dense layers use the inherited write_only() / update() interface unchanged.
    Sparse layers additionally use update_index_key_cache() and read_kv_with_block_indices().
    """

    def __init__(self, ddp_cache_data=None, *args, **kwargs):
        super().__init__(ddp_cache_data, *args, **kwargs)
        self.index_keys: dict[int, Optional[torch.Tensor]] = {}

    @staticmethod
    def _use_cache_custom_op() -> bool:
        return torch.onnx.is_in_onnx_export() or torch._dynamo.is_compiling()

    @staticmethod
    def _cache_call(op, args, eager):
        if QEffMiniMaxSparseCache._use_cache_custom_op():
            return op(*args) if not hasattr(op, "apply") else op.apply(*args)
        return eager(*args)

    @staticmethod
    def ctx_scatter(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """data [B, H, T, D][b, h, position_ids[b, s]] = updates."""

        def eager(data, position_ids, updates):
            out = data.clone()
            b = torch.arange(data.shape[0]).view(-1, 1, 1)
            h = torch.arange(data.shape[1]).view(1, -1, 1)
            out[b, h, position_ids.long().unsqueeze(1)] = updates
            return out

        return QEffMiniMaxSparseCache._cache_call(m3_ctx_scatter, (data, position_ids.to(torch.int32), updates), eager)

    @staticmethod
    def paged_scatter(data, block_id, addr, updates):
        """data [P, R, T, D][block_id, r, addr] = updates [B, R, S, D]."""

        def eager(data, block_id, addr, updates):
            out = data.clone()
            rows = torch.arange(updates.shape[1]).view(1, -1, 1).expand_as(block_id)
            valid = block_id != torch.iinfo(torch.int32).max
            out[block_id[valid].long(), rows[valid], addr[valid].long()] = updates[valid]
            return out

        return QEffMiniMaxSparseCache._cache_call(ctx_paged_scatter_dp, (data, block_id, addr, updates), eager)

    @staticmethod
    def paged_scatter_page(data, block_id, updates):
        """data [P, H, page, D][block_id[p, h], h] = updates [pages, H, page, D]."""

        def eager(data, block_id, updates):
            out = data.clone()
            heads = torch.arange(updates.shape[1]).view(1, -1).expand_as(block_id)
            valid = block_id != torch.iinfo(torch.int32).max
            out[block_id[valid].long(), heads[valid]] = updates[valid]
            return out

        return QEffMiniMaxSparseCache._cache_call(ctx_paged_scatter_page, (data, block_id, updates), eager)

    @staticmethod
    def gather_paged_kv_dp(data, block_ids):
        pages, rows = block_ids.shape

        def eager(data, block_ids):
            ids = torch.where(block_ids == torch.iinfo(torch.int32).max, torch.zeros_like(block_ids), block_ids).long()
            out = data[ids, torch.arange(rows).view(1, rows)]
            return out.permute(1, 0, 2, 3).reshape(1, rows, pages * data.shape[2], data.shape[3])

        return QEffMiniMaxSparseCache._cache_call(ctx_gather_paged_kv_dp, (data, block_ids.to(torch.int32)), eager)

    @staticmethod
    def _gather_blocked(op, data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
        def eager(data, ctx_indices):
            idx = torch.where(
                ctx_indices == torch.iinfo(torch.int32).max, torch.zeros_like(ctx_indices), ctx_indices
            ).long()
            b = torch.arange(data.shape[0]).view(-1, 1, 1)
            r = torch.arange(data.shape[1]).view(1, -1, 1)
            return data[b, r, idx]

        return QEffMiniMaxSparseCache._cache_call(op, (data, ctx_indices.to(torch.int32)), eager)

    @staticmethod
    def gather_blocked_kv_dp(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
        """Gather ``[B, rows, T, D]`` using per-row DP/CP indices."""
        return QEffMiniMaxSparseCache._gather_blocked(ctx_gather_blocked_kv_dp, data, ctx_indices)

    @staticmethod
    def gather_blocked_kv(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
        """Gather ``[B, Hkv, T, D]`` using per-head indices."""
        return QEffMiniMaxSparseCache._gather_blocked(ctx_gather_blocked_kv, data, ctx_indices)

    @staticmethod
    def gather_blocked_kv_blocks(data: torch.Tensor, block_ids: torch.Tensor, block_size: int) -> torch.Tensor:
        """Gather complete contiguous blocks using one index per block."""
        if data.shape[2] % block_size:
            raise ValueError(f"cache length {data.shape[2]} must be divisible by block size {block_size}.")
        blocked = data.reshape(data.shape[0], data.shape[1], data.shape[2] // block_size, block_size, data.shape[3])

        def eager(blocked, block_ids):
            ids = torch.where(block_ids == torch.iinfo(torch.int32).max, torch.zeros_like(block_ids), block_ids).long()
            b = torch.arange(blocked.shape[0], device=blocked.device).view(-1, 1, 1)
            r = torch.arange(blocked.shape[1], device=blocked.device).view(1, -1, 1)
            return blocked[b, r, ids]

        return QEffMiniMaxSparseCache._cache_call(
            ctx_gather_block_range_kv_dp, (blocked, block_ids.to(torch.int32)), eager
        )

    @staticmethod
    def gather_paged_kv_heads(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        """Physical pages [P, H, page, D] at block_ids [pages, H] -> [1, H, pages*page, D]."""
        pages, heads = block_ids.shape

        def eager(data, block_ids):
            valid = block_ids != torch.iinfo(torch.int32).max
            ids = torch.where(valid, block_ids, torch.zeros_like(block_ids)).long()
            out = data[ids, torch.arange(heads).view(1, heads)]
            out = torch.where(valid[..., None, None], out, torch.zeros_like(out))
            return out.permute(1, 0, 2, 3).reshape(1, heads, pages * data.shape[2], data.shape[3])

        return QEffMiniMaxSparseCache._cache_call(ctx_gather_paged_kv_heads, (data, block_ids.to(torch.int32)), eager)

    @staticmethod
    def gather_paged_kv_batch_with_heads(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        """block_ids [B, pages] shared by all heads -> [B, H, pages*page, D]."""
        batch, num_pages = block_ids.shape
        num_heads = data.shape[1]
        outputs = []
        for batch_idx in range(batch):
            ids = block_ids[batch_idx].to(torch.int32).view(num_pages, 1).expand(num_pages, num_heads)
            outputs.append(QEffMiniMaxSparseCache.gather_paged_kv_heads(data, ids))
        return torch.cat(outputs, dim=0)

    @staticmethod
    def gather_paged_kv_selected_heads(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        """block_ids [B, H, pages] -> [B, H, pages*page, D]."""
        outputs = []
        for batch_idx in range(block_ids.shape[0]):
            ids = block_ids[batch_idx].transpose(0, 1).contiguous().to(torch.int32)
            outputs.append(QEffMiniMaxSparseCache.gather_paged_kv_heads(data, ids))
        return torch.cat(outputs, dim=0)

    def write_only_sparse(self, key_states, value_states, layer_idx, cache_kwargs):
        """Write sparse-layer KV states using paged block/address metadata.

        When ``block_id`` and ``addr`` are present in ``cache_kwargs``, the
        cache is expected to use the benchmark layout
        ``[physical_blocks, rows, page_size, head_dim]`` and updates are
        scattered with the paged cache custom op. Without paged metadata, the
        inherited standard ``[batch, heads, ctx_len, head_dim]`` path is used.
        """
        self.append_new_layers(layer_idx)
        layer = self.layers[layer_idx]
        if layer.keys is None:
            layer.keys = key_states
            layer.values = value_states
            layer._mark_initialized(layer.keys)
        else:
            layer._mark_initialized(layer.keys)
            batch, nh, query_len, head_dim = key_states.shape

            position_ids = cache_kwargs.get("position_ids")
            dp = cache_kwargs.get("dp", 1)
            batch_local = batch // dp
            hkv = cache_kwargs.get("hkv", nh)
            cp = cache_kwargs.get("cp", 1) or 1
            if cp > 1:
                rows = dp * hkv * cp
                cache_shape = tuple(layer.keys.shape)
                if tuple(layer.values.shape) != cache_shape:
                    raise ValueError("CP sparse key and value cache shapes must match.")
                if (
                    layer.keys.ndim != 4
                    or tuple(layer.keys.shape[:2]) != (batch_local, rows)
                    or layer.keys.shape[3] != head_dim
                ):
                    raise ValueError(
                        f"CP sparse cache shape {cache_shape} does not match "
                        f"({batch_local}, {rows}, local_ctx_len, {head_dim})."
                    )
                local_ctx_len = layer.keys.shape[2]
                key_dp = key_states.view(dp, batch_local, query_len, hkv, head_dim).permute(1, 0, 3, 2, 4)
                value_dp = value_states.view(dp, batch_local, query_len, hkv, head_dim).permute(1, 0, 3, 2, 4)
                key_updates = (
                    key_dp.unsqueeze(2)
                    .expand(batch_local, dp, cp, hkv, query_len, head_dim)
                    .reshape(batch_local, rows, query_len, head_dim)
                )
                value_updates = (
                    value_dp.unsqueeze(2)
                    .expand(batch_local, dp, cp, hkv, query_len, head_dim)
                    .reshape(batch_local, rows, query_len, head_dim)
                )
                position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
                owner_cp = position_ids_dp.remainder(cp)
                owner_valid = (owner_cp >= 0) & (owner_cp < cp)
                local_pos = position_ids_dp // cp
                row_cp = torch.arange(rows, device=layer.keys.device).remainder(hkv * cp) // hkv
                row_live = row_cp.view(1, dp, cp, hkv, 1) == owner_cp.view(batch_local, dp, 1, 1, query_len)
                row_live = row_live & owner_valid.view(batch_local, dp, 1, 1, query_len)
                row_live = row_live.expand(batch_local, dp, cp, hkv, query_len).reshape(batch_local, rows, query_len)
                addr = (
                    local_pos.view(batch_local, dp, 1, 1, query_len)
                    .expand(batch_local, dp, cp, hkv, query_len)
                    .reshape(batch_local, rows, query_len)
                    .to(torch.int32)
                )
                batch_idx = torch.arange(batch_local, device=layer.keys.device).view(batch_local, 1, 1)
                block_id = torch.where(
                    row_live,
                    batch_idx.expand(batch_local, rows, query_len),
                    torch.iinfo(torch.int32).max,
                ).to(torch.int32)

                layer.keys = self.paged_scatter(layer.keys, block_id, addr, key_updates)
                layer.values = self.paged_scatter(layer.values, block_id, addr, value_updates)
                layer._mark_initialized(layer.keys)
                return
            rows = dp * hkv

            cache_shape = tuple(layer.keys.shape)
            if tuple(layer.values.shape) != cache_shape:
                raise ValueError("Sparse key and value cache shapes must match.")
            # key_states = key_states.reshape(batch_local, rows, query_len, head_dim)
            # value_states = value_states.reshape(batch_local, rows, query_len, head_dim)
            # layer.keys = layer.keys.reshape(batch_local, rows, -1, head_dim)
            # layer.values = layer.values.reshape(batch_local, rows, -1, head_dim)

            batch_idx = torch.arange(batch_local, device=key_states.device).view(batch_local, 1, 1)
            block_id = batch_idx.expand(batch_local, rows, query_len).to(torch.int32)
            addr = (
                position_ids.view(dp, batch_local, query_len)
                .permute(1, 0, 2)
                .to(torch.int32)[:, :, None, :]
                .expand(batch_local, dp, hkv, query_len)
                .reshape(batch_local, rows, query_len)
            )

            if layer.keys.ndim != 4 or layer.values.ndim != 4:
                raise ValueError("Paged sparse caches must have rank-4 key and value tensors.")
            if key_states.ndim != 4 or value_states.ndim != 4:
                raise ValueError("Paged sparse-cache updates must have rank-4 key and value tensors.")

            block_id = block_id.to(dtype=torch.int32, device=layer.keys.device)
            addr = addr.to(dtype=torch.int32, device=layer.keys.device)
            layer.keys = self.paged_scatter(layer.keys, block_id, addr, key_states)
            # layer.keys = layer.keys.reshape(cache_shape)
            layer.values = self.paged_scatter(layer.values, block_id, addr, value_states)
            # layer.values = layer.values.reshape(cache_shape)
            layer._mark_initialized(layer.keys)

    def update_index_key_cache(
        self,
        idx_k: torch.Tensor,
        layer_idx: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter idx_k into the index key cache and return all ctx_len gathered index keys."""
        gathered, updated = update_and_read_index_key_cache(self.index_keys.get(layer_idx), position_ids, idx_k)
        self.index_keys[layer_idx] = updated
        return gathered

    def read_kv_with_block_indices(
        self,
        layer_idx: int,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather K/V by block-sparse token indices. Returns (selected_k, selected_v, flat_valid)."""
        flat_indices = token_indices.flatten(-2)
        flat_valid = token_valid.flatten(-2)
        layer = self.layers[layer_idx]
        selected_k, selected_v = read_kv_cache_with_indices(layer.keys, layer.values, flat_indices)
        return selected_k, selected_v, flat_valid

    def read_index_key_block_gp(
        self,
        layer_idx: int,
        position_ids_dp: torch.Tensor,
        start: int,
        end: int,
        cp: int,
        hkv: int,
        index_block_size: int,
        cache_gp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather one contiguous block [start:end] from the GP-layout index key cache.

        The index key cache is expected to have shape
        ``[B_local, rows, cache_slots, D]`` where ``rows = dp * cp * hkv`` and
        ``cache_slots = ctx_len // cp``.

        position_ids_dp:  ``[B_local, dp, QL]``
        cache_gp:         Optional pre-reshaped GP-layout cache ``[B_local, rows, ctx_len, D]``.
                          When provided it is used directly; otherwise ``self.index_keys[layer_idx]``
                          is expected to already be in GP layout.
        Returns:          ``[B_local, rows, end - start, D]``

        For ``cp == 1`` the causal validity limit per row is the DP-lane's
        maximum position_id.  For ``cp > 1`` the compact CP-interleaved layout
        is assumed and the per-way limit is computed accordingly.
        """
        if cache_gp is not None:
            cache = cache_gp
        else:
            cache = self.index_keys.get(layer_idx)
            if cache is None:
                raise ValueError(f"No index key cache for layer {layer_idx}.")
        batch_local, rows, _, _ = cache.shape
        block_len = end - start

        pos_max = position_ids_dp.max(dim=-1).values  # [B_local, dp]

        # Map dp-lane pos_max → rows: row r belongs to dp-lane r // (cp * hkv).
        dp_lane_per_row = torch.arange(rows, device=cache.device) // (cp * hkv)
        pos_max_rows = pos_max[:, dp_lane_per_row]  # [B_local, rows]

        if cp == 1:
            gather_limit = pos_max_rows
        else:
            way = torch.arange(rows, device=cache.device).remainder(cp * hkv) // hkv
            cycle_size = index_block_size * cp
            way_start = way * index_block_size
            offset_in_cycle = _remainder_with_symbolic_divisor(pos_max_rows, cycle_size)
            offset_in_way = offset_in_cycle - way_start.unsqueeze(0)
            offset_in_way = torch.where(
                offset_in_way >= 0,
                offset_in_way.clamp_max(index_block_size - 1),
                torch.full_like(offset_in_way, -1),
            )
            gather_limit = (pos_max_rows // cycle_size) * index_block_size + offset_in_way

        ctx_indices = torch.arange(start, end, device=cache.device).view(1, 1, block_len)
        invalid_mask = ctx_indices > gather_limit.unsqueeze(-1)  # [B_local, rows, block_len]
        invalid_idx = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        ctx_indices = torch.where(invalid_mask, invalid_idx, ctx_indices).to(torch.int32)
        ctx_indices = ctx_indices.expand(batch_local, rows, block_len)
        return ctx_gather_blocked_kv(cache, ctx_indices)

    def read_blocked_k_dp(
        self,
        layer_idx: int,
        cache_gp: torch.Tensor,
        position_ids_dp: torch.Tensor,
        start: int,
        end: int,
        index_block_size: int,
        cp: int = 1,
        hkv: int = 1,
    ) -> torch.Tensor:
        """Gather one contiguous block [start:end] from a GP-layout cache with DP-aware validity.

        Mirrors ``_read_blocked_k_dp`` from the MSA decode benchmark.

        cache_gp:         ``[B_local, rows, ctx_len, D]``  (GP-layout; rows = dp * cp * hkv)
        position_ids_dp:  ``[B_local, dp, QL]``
        Returns:          ``[B_local, rows, end - start, D]``

        For ``cp == 1`` the per-row validity limit equals the maximum position_id of the
        owning DP lane.  For ``cp > 1`` the CP-interleaved layout is assumed and the
        per-way limit is derived accordingly (same logic as read_index_key_block_gp).
        """
        batch_local, rows, _, _ = cache_gp.shape
        block_len = end - start

        pos_max = position_ids_dp.max(dim=-1).values  # [B_local, dp]

        dp_lane_per_row = torch.arange(rows, device=cache_gp.device) // (cp * hkv)
        pos_max_rows = pos_max[:, dp_lane_per_row]  # [B_local, rows]

        if cp == 1:
            gather_limit = pos_max_rows
        else:
            way = torch.arange(rows, device=cache_gp.device).remainder(cp * hkv) // hkv
            cycle_size = index_block_size * cp
            way_start = way * index_block_size
            offset_in_cycle = _remainder_with_symbolic_divisor(pos_max_rows, cycle_size)
            offset_in_way = offset_in_cycle - way_start.unsqueeze(0)
            offset_in_way = torch.where(
                offset_in_way >= 0,
                offset_in_way.clamp_max(index_block_size - 1),
                torch.full_like(offset_in_way, -1),
            )
            gather_limit = (pos_max_rows // cycle_size) * index_block_size + offset_in_way

        ctx_indices = torch.arange(start, end, device=cache_gp.device).view(1, 1, block_len)
        invalid_mask = ctx_indices > gather_limit.unsqueeze(-1)  # [B_local, rows, block_len]
        invalid_idx = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        ctx_indices = torch.where(invalid_mask, invalid_idx, ctx_indices).to(torch.int32)
        ctx_indices = ctx_indices.expand(batch_local, rows, block_len)
        return ctx_gather_blocked_kv_dp(cache_gp, ctx_indices)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple] = None,
        index_keys: Optional[dict] = None,
    ) -> "QEffMiniMaxSparseCache":
        """Build from a per-layer tuple cache.

        Each layer entry is either a 2-tuple ``(key, value)`` for dense layers or a
        3-tuple ``(key, value, index_key)`` for sparse layers.  The legacy
        ``index_keys`` dict is accepted for backward-compat but the inline
        3-tuple form takes precedence.
        """
        cache = cls()
        if past_key_values is not None:
            for i, layer_tuple in enumerate(past_key_values):
                if len(layer_tuple) == 3:
                    key_states, value_states, index_key = layer_tuple
                    cache.index_keys[i] = index_key
                else:
                    key_states, value_states = layer_tuple
                cache.layers.append(QEffDynamicLayer.from_tensors(key_states, value_states))
        if index_keys is not None:
            cache.index_keys.update(index_keys)
        return cache

    def to_legacy_cache(self) -> Tuple:
        """Return per-layer tuples; sparse layers carry a 3-tuple ``(key, value, index_key)``."""
        layers = []
        for i, layer in enumerate(self.layers):
            if i in self.index_keys and self.index_keys[i] is not None:
                layers.append((layer.keys, layer.values, self.index_keys[i]))
            else:
                layers.append((layer.keys, layer.values))
        return tuple(layers)

    def to_kv_only_cache(self) -> Tuple:
        """Return per-layer 2-tuples ``(key, value)`` only, without index keys."""
        return tuple((layer.keys, layer.values) for layer in self.layers)

    def get_index_keys_tuple(self) -> Tuple:
        """Return index keys as a flat tuple ordered by layer index."""
        return tuple(self.index_keys[i] for i in sorted(self.index_keys.keys()))


class QEffEncoderDecoderCache(EncoderDecoderCache):
    """
    Updated the `EncoderDecoderCache` to use the `QEffDynamicCache` for both self-attention and cross-attention caches.
    """

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(QEffDynamicCache(), QEffDynamicCache())
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, QEffDynamicCache)
            and isinstance(self.cross_attention_cache, QEffDynamicCache)
        ):
            raise TypeError(
                f"`{method}` requires QEffDynamicCache objects, got "
                f"{self.self_attention_cache.__class__.__name__} and {self.cross_attention_cache.__class__.__name__}."
            )

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        legacy_cache = ()
        total_layers = max(len(self.self_attention_cache.layers), len(self.cross_attention_cache.layers))
        for layer_idx in range(total_layers):
            self_key = self_value = cross_key = cross_value = None
            if layer_idx < len(self.self_attention_cache.layers):
                self_key, self_value = self.self_attention_cache[layer_idx]
            if layer_idx < len(self.cross_attention_cache.layers):
                cross_key, cross_value = self.cross_attention_cache[layer_idx]
            if cross_key is None or cross_value is None:
                legacy_cache += ((self_key, self_value),)
            else:
                legacy_cache += ((self_key, self_value, cross_key, cross_value),)
        return legacy_cache


# TODO:This function will be depercated in future.
class QEffHybridCache(HybridCache):
    def __init__(self, config, batch_size, max_cache_len):
        super().__init__(config, batch_size, max_cache_len=max_cache_len)
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "HybridCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(config, batch_size=past_key_values[0][0].shape[0], max_cache_len=past_key_values[0][0].shape[2])
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position: Optional[torch.LongTensor] = None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            sliding_window_pattern = cache_kwargs.get("sliding_window_pattern")
            # is_sliding_layer = torch.tensor(bool((layer_idx + 1) % sliding_window_pattern))
            is_sliding_layer = torch.tensor(
                bool(_remainder_with_symbolic_divisor(torch.scalar_tensor(layer_idx + 1), sliding_window_pattern))
            )
            layer_ctx_len = self.key_cache[layer_idx].shape[2]
            kv_position_ids = torch.where(
                (~is_sliding_layer | (position_ids == -1)),
                position_ids,
                _remainder_with_symbolic_divisor(position_ids, layer_ctx_len - 1),
            )

            kv_position_ids = torch.where(
                is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1) * 2),
                _remainder_with_symbolic_divisor(position_ids + 1, layer_ctx_len),
                kv_position_ids,
            )

            valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
            key_states = torch.where(valid_mask == 1, key_states, torch.zeros_like(key_states, dtype=key_states.dtype))
            value_states = torch.where(
                valid_mask == 1, value_states, torch.zeros_like(value_states, dtype=value_states.dtype)
            )

            self.key_cache[layer_idx] = ctx_scatter(self.key_cache[layer_idx], kv_position_ids, key_states)
            self.value_cache[layer_idx] = ctx_scatter(self.value_cache[layer_idx], kv_position_ids, value_states)
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Original Gather
            ctx_len = cache_kwargs.get("CCL", self.key_cache[layer_idx].shape[2])
            ctx_indices = torch.arange(ctx_len, dtype=kv_position_ids.dtype, device=kv_position_ids.device)[
                None, None, ...
            ]
            gather_limit = kv_position_ids.max(1, keepdim=True).values.unsqueeze(1).to(kv_position_ids.dtype)
            invalid_mask = ctx_indices > gather_limit
            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            all_indices = torch.arange(layer_ctx_len, device=kv_position_ids.device) + kv_position_ids.max() + 1
            rolling_indices = torch.where(
                all_indices > layer_ctx_len - 1,
                _remainder_with_symbolic_divisor(all_indices, layer_ctx_len),
                all_indices,
            )
            rolling_indices = rolling_indices[:ctx_len]
            final_indices = torch.where(
                (is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), rolling_indices, ctx_indices
            )

            k_out = ctx_gather(k_out, final_indices, ctx_len)
            v_out = ctx_gather(v_out, final_indices, ctx_len)
            ctx_v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)
            v_out = torch.where((is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), v_out, ctx_v_out)
        return k_out, v_out


# TODO:This function will be depercated in future.
class QEffHybridChunkedCache(HybridChunkedCache):
    def __init__(self, config, max_batch_size: int = 1, max_cache_len: int = 2048):
        self.config = config
        sliding_window_pattern = config.sliding_window_pattern
        num_layers = config.num_hidden_layers
        # self.is_sliding = [bool((i + 1) % sliding_window_pattern) for i in range(num_layers)]
        self.is_sliding = []
        for i in range(num_layers):
            val = torch.scalar_tensor(i + 1)
            rem = _remainder_with_symbolic_divisor(val, sliding_window_pattern)
            self.is_sliding.append(bool(rem))

        self.key_cache: List[torch.Tensor] = [None] * num_layers
        self.value_cache: List[torch.Tensor] = [None] * num_layers

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position: Optional[torch.LongTensor] = None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `HybridChunkedCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "HybridChunkedCache":
        """Converts a cache in the legacy cache format into an equivalent `HybridChunkedCache`. Used for
        backward compatibility."""
        cache = cls(config, max_batch_size=past_key_values[0][0].shape[0], max_cache_len=past_key_values[0][0].shape[2])
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states

        else:
            position_ids = cache_kwargs.get("position_ids")
            is_sliding_layer = torch.tensor(bool(self.is_sliding[layer_idx]))

            # Update the position_ids to handle the sliding window
            layer_ctx_len = self.key_cache[layer_idx].shape[2]
            kv_position_ids = torch.where(
                (~is_sliding_layer | (position_ids == -1)),
                position_ids,
                _remainder_with_symbolic_divisor(position_ids, layer_ctx_len - 1),
            )

            kv_position_ids = torch.where(
                is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1) * 2),
                _remainder_with_symbolic_divisor(position_ids + 1, layer_ctx_len),
                kv_position_ids,
            )

            valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
            key_states = torch.where(valid_mask == 1, key_states, torch.zeros_like(key_states, dtype=key_states.dtype))
            value_states = torch.where(
                valid_mask == 1, value_states, torch.zeros_like(value_states, dtype=value_states.dtype)
            )

            self.key_cache[layer_idx] = ctx_scatter(self.key_cache[layer_idx], kv_position_ids, key_states)
            self.value_cache[layer_idx] = ctx_scatter(self.value_cache[layer_idx], kv_position_ids, value_states)
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Original Gather
            ctx_len = cache_kwargs.get("CCL", k_out.shape[2])
            ctx_len = min(layer_ctx_len, ctx_len)
            ctx_indices = torch.arange(ctx_len, dtype=kv_position_ids.dtype, device=kv_position_ids.device)[
                None, None, ...
            ]
            gather_limit = kv_position_ids.max(1, keepdim=True).values.unsqueeze(1).to(kv_position_ids.dtype)
            invalid_mask = ctx_indices > gather_limit
            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            # Rolling indices for sliding window
            all_indices = torch.arange(layer_ctx_len, device=kv_position_ids.device) + kv_position_ids.max() + 1
            rolling_indices = torch.where(
                all_indices > layer_ctx_len - 1,
                _remainder_with_symbolic_divisor(all_indices, layer_ctx_len),
                all_indices,
            )
            rolling_indices = rolling_indices[:ctx_len]
            final_indices = torch.where(
                (is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), rolling_indices, ctx_indices
            )

            k_out = ctx_gather(k_out, final_indices, ctx_len)
            v_out = ctx_gather(v_out, final_indices, ctx_len)
            ctx_v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)
            v_out = torch.where((is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), v_out, ctx_v_out)
        return k_out, v_out


# This is a hack for now, until we get to merging this code with HybridCache class,
# We don't really need to inherit transformers classes as their cache classes are made to work with pytorch and
# ours are made to work with AIC
class QEffSlidingWindowCache:
    def __init__(self, config, batch_size, max_cache_len, sliding_window_len):
        self.config = config
        self.max_cache_len = max_cache_len
        self.batch_size = batch_size
        self.sliding_window_len = sliding_window_len
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = 0

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "HybridCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""

        # Get the sliding_window_pattern from config
        sliding_window_pattern = getattr(
            config, "_sliding_window_pattern", getattr(config, "sliding_window_pattern", None)
        )

        cache = cls(
            config,
            batch_size=past_key_values[0][0].shape[0],
            max_cache_len=past_key_values[sliding_window_pattern - 1][0].shape[2],
            sliding_window_len=past_key_values[0][0].shape[2],
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        # Legacy tuples are often preallocated to ctx len. Track real progression via update() calls.
        cache.seen_tokens = 0
        return cache

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns seen token length (logical sequence length)."""
        return self.seen_tokens

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        query_length = cache_position.shape[0]
        layer_types = getattr(self.config, "layer_types", None)
        is_sliding_layer = bool(
            layer_types is not None and layer_idx < len(layer_types) and layer_types[layer_idx] == "sliding_attention"
        )

        if is_sliding_layer:
            kv_offset = max(self.seen_tokens - self.sliding_window_len + 1, 0)
            if self.seen_tokens >= self.sliding_window_len:
                kv_length = self.sliding_window_len - 1 + query_length
            else:
                kv_length = self.seen_tokens + query_length
            return kv_length, kv_offset

        return self.seen_tokens + query_length, 0

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = cache_kwargs or {}
        position_ids = cache_kwargs.get("position_ids")
        if position_ids is None and cache_kwargs.get("cache_position") is not None:
            cache_position = cache_kwargs.get("cache_position")
            if cache_position.dim() == 1:
                position_ids = cache_position.unsqueeze(0).repeat(key_states.shape[0], 1)
            else:
                position_ids = cache_position
        if position_ids is not None:
            # Track logical progression independent of preallocated tensor shape.
            self.seen_tokens = max(self.seen_tokens, int(position_ids.max().item()) + 1)

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = position_ids
            layer_types = getattr(self.config, "layer_types", None)
            is_sliding_layer = cache_kwargs.get("is_sliding")
            if is_sliding_layer is None and layer_types is not None and layer_idx < len(layer_types):
                is_sliding_layer = layer_types[layer_idx] == "sliding_attention"
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value from the kwargs

            if is_sliding_layer:
                sliding_window_len = self.key_cache[layer_idx].shape[2]
                kv_position_ids = torch.where(
                    position_ids == -1, position_ids, _remainder_with_symbolic_divisor(position_ids, sliding_window_len)
                )
            else:
                kv_position_ids = position_ids

            if batch_index is not None:
                if torch.onnx.is_in_onnx_export() or torch._dynamo.is_compiling():
                    invalid_scatter_index = torch.iinfo(torch.int32).max
                    scatter_position_ids = torch.where(kv_position_ids < 0, invalid_scatter_index, kv_position_ids)
                else:
                    scatter_position_ids = kv_position_ids
                self.key_cache[layer_idx] = ctx_scatter_cb(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )
                self.value_cache[layer_idx] = ctx_scatter_cb(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )
            else:
                self.key_cache[layer_idx] = ctx_scatter(self.key_cache[layer_idx], kv_position_ids, key_states)
                self.value_cache[layer_idx] = ctx_scatter(self.value_cache[layer_idx], kv_position_ids, value_states)

            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Original Gather
            if is_sliding_layer:
                ctx_len = self.key_cache[layer_idx].shape[2]
            else:
                ctx_len = cache_kwargs.get("CCL", self.key_cache[layer_idx].shape[2])

            ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(position_ids.dtype)
            invalid_mask = ctx_indices > gather_limit
            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            if batch_index is not None:
                k_out = ctx_gather_cb(k_out, batch_index, ctx_indices, ctx_len)
                v_out = ctx_gather_cb(v_out, batch_index, ctx_indices, ctx_len)
            else:
                k_out = ctx_gather(k_out, ctx_indices, ctx_len)
                v_out = ctx_gather(v_out, ctx_indices, ctx_len)

            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)
        return k_out, v_out


class QEffGPTOSSDynamicLayer(QEffDynamicLayer):
    def __init__(self, is_sliding: bool = False):
        super().__init__()
        self.is_sliding = is_sliding

    @property
    def cache_len(self):
        return self.keys.shape[-2] if self.keys is not None else 0

    @classmethod
    def from_tensors(
        cls, key_states: torch.Tensor, value_states: torch.Tensor, is_sliding: bool = False
    ) -> "QEffGPTOSSDynamicLayer":
        layer = cls(is_sliding=is_sliding)
        layer.keys = key_states
        layer.values = value_states
        layer._mark_initialized(key_states)
        return layer

    def write_only(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
            return self.keys, self.values

        position_ids = cache_kwargs.get("position_ids")
        _, _, ctx_len, _ = self.keys.shape
        batch_index = cache_kwargs.get("batch_index", None)

        if self.is_sliding:
            # prefill only mode we slice the passed key states to sliding window len and need to adjust kv_position_ids accordingly
            kv_position_ids = torch.arange(ctx_len, dtype=torch.int64, device=position_ids.device).reshape(1, -1)
        else:
            kv_position_ids = position_ids

        if batch_index is not None:
            invalid_scatter_index = torch.iinfo(torch.int32).max
            scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
            self.keys = ctx_scatter_cb(self.keys, batch_index, scatter_position_ids, key_states)
            self.values = ctx_scatter_cb(self.values, batch_index, scatter_position_ids, value_states)
        else:
            self.keys = ctx_scatter(self.keys, kv_position_ids, key_states)
            self.values = ctx_scatter(self.values, kv_position_ids, value_states)
        return self.keys, self.values

    def read_only_blocked_kv(
        self,
        start_idx: torch.Tensor,
        end_idx: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)

        k_out, v_out = self.keys, self.values

        batch, num_kv_heads, _, _ = k_out.shape

        ctx_indices = torch.arange(start=start_idx, end=end_idx, dtype=position_ids.dtype, device=position_ids.device)[
            None, None, ...
        ]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = ctx_gather_blocked_kv_cb(k_out, batch_index, ctx_indices)
            v_out = ctx_gather_blocked_kv_cb(v_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            k_out = ctx_gather_blocked_kv(k_out, ctx_indices)
            v_out = ctx_gather_blocked_kv(v_out, ctx_indices)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)
        return k_out, v_out

    def read_only_blocked_K(
        self,
        start_idx: torch.Tensor,
        end_idx: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value from the kwargs

        k_out = self.key_cache[layer_idx]

        batch, num_kv_heads, _, _ = k_out.shape

        ctx_indices = torch.arange(start=start_idx, end=end_idx)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = ctx_gather_blocked_kv_cb(k_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            k_out = ctx_gather_blocked_kv(k_out, ctx_indices)

        return k_out

    def read_only_blocked_V(
        self,
        start_idx: torch.Tensor,
        end_idx: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value from the kwargs

        v_out = self.value_cache[layer_idx]

        batch, num_kv_heads, _, _ = v_out.shape

        ctx_indices = torch.arange(start=start_idx, end=end_idx)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            v_out = ctx_gather_blocked_kv_cb(v_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            v_out = ctx_gather_blocked_kv(v_out, ctx_indices)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_sliding or cache_kwargs is None:
            return super().update(key_states, value_states, cache_kwargs)

        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
            return self.keys, self.values

        self._mark_initialized(self.keys)
        key_states = key_states.to(self.keys.dtype)
        value_states = value_states.to(self.values.dtype)
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        layer_ctx_len = self.keys.shape[2]

        kv_position_ids = torch.where(
            position_ids == -1, position_ids, _remainder_with_symbolic_divisor(position_ids, layer_ctx_len)
        )

        if batch_index is not None:
            if torch.onnx.is_in_onnx_export() or torch._dynamo.is_compiling():
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(kv_position_ids < 0, invalid_scatter_index, kv_position_ids)
            else:
                scatter_position_ids = kv_position_ids
            self.keys = ctx_scatter_cb(self.keys, batch_index, scatter_position_ids, key_states)
            self.values = ctx_scatter_cb(self.values, batch_index, scatter_position_ids, value_states)
        else:
            self.keys = ctx_scatter(self.keys, kv_position_ids, key_states)
            self.values = ctx_scatter(self.values, kv_position_ids, value_states)

        k_out, v_out = self.keys, self.values

        # Original Gather
        ctx_len = cache_kwargs.get("CCL", self.keys.shape[2])
        ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = ctx_gather_cb(k_out, batch_index, ctx_indices, ctx_len)
            v_out = ctx_gather_cb(v_out, batch_index, ctx_indices, ctx_len)
        else:
            k_out = ctx_gather(k_out, ctx_indices, ctx_len)
            v_out = ctx_gather(v_out, ctx_indices, ctx_len)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)
        return k_out, v_out

    def full_cache_update_chunked(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index")
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        # Scatter
        if batch_index is not None:
            if torch.onnx.is_in_onnx_export() or torch._dynamo.is_compiling():
                scatter_position_ids = torch.where(position_ids < 0, torch.iinfo(torch.int32).max, position_ids)
            else:
                scatter_position_ids = position_ids
            self.keys = ctx_scatter_cb(self.keys, batch_index, scatter_position_ids, key_states)
            self.values = ctx_scatter_cb(self.values, batch_index, scatter_position_ids, value_states)
        else:
            self.keys = ctx_scatter(self.keys, position_ids, key_states)
            self.values = ctx_scatter(self.values, position_ids, value_states)

        k_out, v_out = self.keys, self.values

        # Gather
        ctx_len = cache_kwargs.get("CCL", k_out.shape[2])
        ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
        if batch_index is not None:
            k_out = ctx_gather_cb(k_out, batch_index, ctx_indices, ctx_len)
            v_out = ctx_gather_cb(v_out, batch_index, ctx_indices, ctx_len)
        else:
            k_out = ctx_gather(k_out, ctx_indices, ctx_len)
            v_out = ctx_gather(v_out, ctx_indices, ctx_len)
        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)

        return k_out, v_out

    def sliding_window_update_chunked(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index")
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        if batch_index is not None:
            if torch.onnx.is_in_onnx_export() or torch._dynamo.is_compiling():
                scatter_position_ids = torch.where(position_ids < 0, torch.iinfo(torch.int32).max, position_ids)
            else:
                scatter_position_ids = position_ids
            self.keys = ctx_scatter_cb(self.keys, batch_index, scatter_position_ids, key_states)
            self.values = ctx_scatter_cb(self.values, batch_index, scatter_position_ids, value_states)
        else:
            self.keys = ctx_scatter(self.keys, position_ids, key_states)
            self.values = ctx_scatter(self.values, position_ids, value_states)

        k_out, v_out = self.keys, self.values
        sliding_window_len = cache_kwargs.get("sliding_window")

        # Gather
        ctx_len = position_ids.shape[1] + sliding_window_len
        ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, None, ...]
        first_pos_idx = position_ids[0][0]
        add_idx = torch.where(first_pos_idx >= sliding_window_len, first_pos_idx - sliding_window_len, 0)
        ctx_indices += add_idx
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(torch.int32)
        invalid_mask = ctx_indices > gather_limit
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
        if batch_index is not None:
            k_out = ctx_gather_cb(k_out, batch_index, ctx_indices, ctx_len)
            v_out = ctx_gather_cb(v_out, batch_index, ctx_indices, ctx_len)
        else:
            k_out = ctx_gather(k_out, ctx_indices, ctx_len)
            v_out = ctx_gather(v_out, ctx_indices, ctx_len)
        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)

        return k_out, v_out


class QEffGPTOSSHybridCache(QEffDynamicCache):
    def __init__(
        self,
        config=None,
        ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
        *args,
        **kwargs,
    ):
        self.config = config
        kwargs.pop("layer_classes", None)
        kwargs.pop("layers", None)
        kwargs.pop("layer_class_to_replicate", None)
        Cache.__init__(self, layers=[], *args, **kwargs)
        if ddp_cache_data is not None:
            for layer_idx, (key_states, value_states) in enumerate(ddp_cache_data):
                self.append_new_layers(layer_idx)
                self.layers[layer_idx] = QEffGPTOSSDynamicLayer.from_tensors(
                    key_states,
                    value_states,
                    is_sliding=self._is_sliding_layer(layer_idx),
                )

    def _is_sliding_layer(self, layer_idx: int) -> bool:
        layer_types = getattr(self.config, "layer_types", None)
        return (
            layer_types is not None and layer_idx < len(layer_types) and layer_types[layer_idx] == "sliding_attention"
        )

    def get_max_cache_len(self, layer_idx: Optional[int] = 1, cache_position: Optional[torch.LongTensor] = None) -> int:
        """
        Keep backward-compatible call shape while deferring to upstream implementation.
        """
        return super().get_seq_length(layer_idx)

    def get_sliding_window_len(
        self, layer_idx: Optional[int] = 0, cache_position: Optional[torch.LongTensor] = None
    ) -> int:
        """
        Keep backward-compatible call shape while deferring to upstream implementation.
        """
        return super().get_seq_length(layer_idx)

    def append_new_layers(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(QEffGPTOSSDynamicLayer(is_sliding=self._is_sliding_layer(len(self.layers))))

    @classmethod
    def from_legacy_cache(
        cls,
        config,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> "QEffGPTOSSHybridCache":
        cache = cls(config=config)
        if past_key_values is not None:
            for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                cache.append_new_layers(layer_idx)
                cache.layers[layer_idx] = QEffGPTOSSDynamicLayer.from_tensors(
                    key_states,
                    value_states,
                    is_sliding=cache._is_sliding_layer(layer_idx),
                )
        return cache

    @classmethod
    def from_cache(cls, config, past_key_values: Cache) -> "QEffGPTOSSHybridCache":
        cache = cls(config=config)
        for layer_idx, layer in enumerate(getattr(past_key_values, "layers", [])):
            key_states = getattr(layer, "keys", None)
            value_states = getattr(layer, "values", None)
            if key_states is None or value_states is None:
                continue
            cache.append_new_layers(layer_idx)
            cache.layers[layer_idx] = QEffGPTOSSDynamicLayer.from_tensors(
                key_states,
                value_states,
                is_sliding=cache._is_sliding_layer(layer_idx),
            )

        return cache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.append_new_layers(layer_idx)
        outputs = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
        return outputs

    def write_only(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.append_new_layers(layer_idx)
        outputs = self.layers[layer_idx].write_only(key_states, value_states, cache_kwargs)
        return outputs

    def read_only_blockedKV(self, start_idx, end_idx, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].read_only_blockedKV(start_idx, end_idx, cache_kwargs)

    def full_cache_update_chunked(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.append_new_layers(layer_idx)
        outputs = self.layers[layer_idx].full_cache_update_chunked(key_states, value_states, cache_kwargs)
        return outputs

    def sliding_window_update_chunked(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.append_new_layers(layer_idx)
        outputs = self.layers[layer_idx].sliding_window_update_chunked(key_states, value_states, cache_kwargs)
        return outputs


class QEffGemma4DynamicCache(QEffDynamicCache):
    def __init__(
        self,
        config=None,
        ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
        *args,
        **kwargs,
    ):
        self.config = config
        kwargs.pop("layer_classes", None)
        kwargs.pop("layers", None)
        kwargs.pop("layer_class_to_replicate", None)
        Cache.__init__(self, layers=[], *args, **kwargs)
        if ddp_cache_data is not None:
            for layer_idx, (key_states, value_states) in enumerate(ddp_cache_data):
                self.append_new_layers(layer_idx)
                self.layers[layer_idx] = QEffGemma4DynamicLayer.from_tensors(
                    key_states,
                    value_states,
                    is_sliding=self._is_sliding_layer(layer_idx),
                    sliding_window=self._sliding_window_len(layer_idx),
                )

    def _is_sliding_layer(self, layer_idx: int) -> bool:
        layer_types = getattr(self.config, "layer_types", None)
        return (
            layer_types is not None and layer_idx < len(layer_types) and layer_types[layer_idx] == "sliding_attention"
        )

    def _sliding_window_len(self, layer_idx: int) -> Optional[int]:
        if not self._is_sliding_layer(layer_idx):
            return None
        return getattr(self.config, "sliding_window", None)

    def append_new_layers(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            new_layer_idx = len(self.layers)
            self.layers.append(
                QEffGemma4DynamicLayer(
                    is_sliding=self._is_sliding_layer(new_layer_idx),
                    sliding_window=self._sliding_window_len(new_layer_idx),
                )
            )

    @classmethod
    def from_legacy_cache(
        cls,
        config,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> "QEffGemma4DynamicCache":
        cache = cls(config=config)
        if past_key_values is not None:
            for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                cache.append_new_layers(layer_idx)
                cache.layers[layer_idx] = QEffGemma4DynamicLayer.from_tensors(
                    key_states,
                    value_states,
                    is_sliding=cache._is_sliding_layer(layer_idx),
                    sliding_window=cache._sliding_window_len(layer_idx),
                )
        return cache

    @classmethod
    def from_cache(cls, config, past_key_values: Cache) -> "QEffGemma4DynamicCache":
        cache = cls(config=config)
        for layer_idx, layer in enumerate(getattr(past_key_values, "layers", [])):
            key_states = getattr(layer, "keys", None)
            value_states = getattr(layer, "values", None)
            if key_states is None or value_states is None:
                continue
            cache.append_new_layers(layer_idx)
            cache.layers[layer_idx] = QEffGemma4DynamicLayer.from_tensors(
                key_states,
                value_states,
                is_sliding=cache._is_sliding_layer(layer_idx),
                sliding_window=cache._sliding_window_len(layer_idx),
            )
        return cache


class QEffGemma4DynamicLayer(QEffDynamicLayer):
    def __init__(self, is_sliding: bool = False, sliding_window: Optional[int] = None):
        super().__init__()
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window

    @classmethod
    def from_tensors(
        cls,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        is_sliding: bool = False,
        sliding_window: Optional[int] = None,
    ) -> "QEffGemma4DynamicLayer":
        layer = cls(is_sliding=is_sliding, sliding_window=sliding_window)
        layer.keys = key_states
        layer.values = value_states
        layer._mark_initialized(key_states)
        return layer

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_sliding or cache_kwargs is None:
            return super().update(key_states, value_states, cache_kwargs)

        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
            return self.keys, self.values

        self._mark_initialized(self.keys)
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        layer_ctx_len = self.sliding_window
        if layer_ctx_len is None:
            layer_ctx_len = self.keys.shape[2]

        kv_position_ids = torch.where(
            position_ids == -1, position_ids, _remainder_with_symbolic_divisor(position_ids, layer_ctx_len)
        )

        valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
        key_states = torch.where(valid_mask, key_states, torch.zeros_like(key_states, dtype=key_states.dtype))
        value_states = torch.where(valid_mask, value_states, torch.zeros_like(value_states, dtype=value_states.dtype))

        if batch_index is not None:
            invalid_scatter_index = torch.iinfo(torch.int32).max
            scatter_position_ids = torch.where(kv_position_ids < 0, invalid_scatter_index, kv_position_ids)
            self.keys = ctx_scatter_cb(self.keys, batch_index, scatter_position_ids, key_states)
            self.values = ctx_scatter_cb(self.values, batch_index, scatter_position_ids, value_states)
        else:
            self.keys = ctx_scatter(self.keys, kv_position_ids, key_states)
            self.values = ctx_scatter(self.values, kv_position_ids, value_states)

        k_out, v_out = self.keys, self.values

        ctx_len = cache_kwargs.get("CCL", k_out.shape[2])
        ctx_len = min(layer_ctx_len, ctx_len)
        ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1).to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = ctx_gather_cb(k_out, batch_index, ctx_indices, ctx_len)
            v_out = ctx_gather_cb(v_out, batch_index, ctx_indices, ctx_len)
        else:
            k_out = ctx_gather(k_out, ctx_indices, ctx_len)
            v_out = ctx_gather(v_out, ctx_indices, ctx_len)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out, dtype=v_out.dtype), v_out)
        return k_out, v_out
