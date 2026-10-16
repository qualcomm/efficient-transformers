# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache

from QEfficient.customop.utils import ctx_gather_paged_kv_dp
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import HEADPAR_MASKED_ATTENTION_VALUE, MIN_MASKED_ATTENTION_VALUE


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep) for GQA.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_kv_states(
    module: nn.Module, key: torch.Tensor, value: torch.Tensor, num_repeat: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_kv_groups = getattr(module, "num_key_value_groups", None) if not num_repeat else num_repeat
    if num_kv_groups is None:
        return key, value
    return repeat_kv(key, num_kv_groups), repeat_kv(value, num_kv_groups)


def _normalize_int(value: Optional[torch.Tensor | int]) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value) if value is not None else 0


def update_running_softmax(
    current_max: torch.Tensor,
    attn_weights_block: torch.Tensor,
    current_denominator: torch.Tensor,
    output: torch.Tensor,
    v_block: torch.Tensor,
    skip_kv: bool = False,
    skip_future: Optional[torch.Tensor] = None,
):
    # Update Running row maximum
    prev_max = current_max
    current_max_updated = torch.max(prev_max, attn_weights_block.max(dim=3).values)
    delta_max = prev_max - current_max_updated

    current_exp = torch.exp(attn_weights_block - current_max_updated.unsqueeze(-1))

    # update running denominator
    prev_denominator = current_denominator
    curr_exp_sum = current_exp.sum(dim=-1)
    current_denominator_updated = prev_denominator * torch.exp(delta_max) + curr_exp_sum

    prob = current_exp / current_denominator_updated.unsqueeze(-1)

    prev_output = output
    # if updating running softmax with attention sinks, we don't have v_block
    output_scale = ((prev_denominator / current_denominator_updated).unsqueeze(-1)) * torch.exp(delta_max.unsqueeze(-1))
    if v_block is not None:
        value_output = torch.matmul(prob.to(v_block.dtype), v_block).to(prev_output.dtype)
        output_updated = output_scale.to(prev_output.dtype) * prev_output + value_output
    else:
        output_updated = output_scale.to(prev_output.dtype) * prev_output

    if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
        current_max = torch.where(skip_future, prev_max, current_max_updated)
        current_denominator = torch.where(skip_future, prev_denominator, current_denominator_updated)
        output = torch.where(skip_future.unsqueeze(-1), prev_output, output_updated)
    else:
        # Eager mode
        current_max = current_max_updated
        current_denominator = current_denominator_updated
        output = output_updated
    return current_max, current_denominator, output


def update_running_softmax_prefill(
    current_max: torch.Tensor,
    attn_weights_block: torch.Tensor,
    current_denominator: torch.Tensor,
    output: torch.Tensor,
    v_block: torch.Tensor,
    skip_kv: bool = False,
    skip_future: Optional[torch.Tensor] = None,
):
    prev_max = current_max
    current_max_updated = torch.max(prev_max, attn_weights_block.max(dim=-1).values)
    delta_max = prev_max - current_max_updated
    current_exp = torch.exp(attn_weights_block - current_max_updated.unsqueeze(-1))
    prev_denominator = current_denominator
    curr_exp_sum = current_exp.sum(dim=-1)
    current_denominator_updated = prev_denominator * torch.exp(delta_max) + curr_exp_sum
    prev_output = output
    output_updated = prev_output * torch.exp(delta_max.unsqueeze(-1)) + torch.matmul(current_exp, v_block)
    if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
        assert skip_future is not None
        current_max = torch.where(skip_future, prev_max, current_max_updated)
        current_denominator = torch.where(skip_future, prev_denominator, current_denominator_updated)
        output = torch.where(skip_future.unsqueeze(-1), prev_output, output_updated)
    else:
        current_max = current_max_updated
        current_denominator = current_denominator_updated
        output = output_updated
    return current_max, current_denominator, output


def _read_kv_block(
    *,
    past_key_value: Cache,
    start_index: int,
    end_index: int,
    layer_idx: int,
    kv_block_size: int,
    paged_attention: bool,
    kv_block_idx: int,
    block_table: Optional[torch.Tensor],
    cache_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read one key/value cache block.

    When ``paged_attention`` is set the block is gathered through the paged-attention
    block table; otherwise it is read as a contiguous slice of the KV cache.
    """
    if paged_attention:
        position_ids = cache_kwargs.get("position_ids")
        block_index = block_table[:, kv_block_idx]
        updated = (position_ids.max(1, keepdim=True).values // kv_block_size) == kv_block_idx
        return past_key_value.read_only_paged_attention(block_index, updated, layer_idx, cache_kwargs)
    return past_key_value.read_only_blocked_kv(start_index, end_index, layer_idx, cache_kwargs)


def _prepare_batch_folded_kv_indices(
    start_index: int,
    end_index: int,
    position_ids: torch.Tensor,
    num_kv_heads: int,
    cache_bh: int,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one masked folded-cache index tensor for a K/V block.

    The range, gather limit, and ``where`` must be traced from the batch-fold
    attention body. Otherwise TorchScript's common-expression elimination can
    lift the identical tensors from the first decoder layer and expose them as
    inputs/outputs of every subsequent ONNX decoder-layer function. That extra
    function plumbing both creates unnecessary subfunction variants and keeps
    the AIC compiler from seeing the range bounds inside the function where it
    performs the gather.

    Returns tensors with shape ``[1, cache_bh, block_size]``. ``ctx_indices``
    uses zero for positions beyond the current decode position; callers use
    ``invalid_mask`` to preserve the causal semantics. In particular, the K
    path is already masked by attention, while the V path explicitly zeros
    invalid gathered values.
    """
    # Add and then subtract layer_idx so the values remain numerically equal to
    # [start_index, end_index), while the traced range is anchored to the
    # current decoder layer. This graph-local dependency is required for ONNX
    # subfunction range analysis; removing it can recreate the regression.
    ctx_indices = torch.arange(
        start=start_index + layer_idx,
        end=end_index + layer_idx,
        dtype=position_ids.dtype,
        device=position_ids.device,
    )[None, None, ...]
    ctx_indices = ctx_indices - layer_idx

    gather_limit = position_ids.max(1, keepdim=True).values
    gather_limit = torch.cat([gather_limit] * num_kv_heads, dim=1).reshape(1, -1, 1)
    invalid_mask = ctx_indices > gather_limit
    ctx_indices = torch.where(invalid_mask, 0, ctx_indices)

    block_size = end_index - start_index
    ctx_indices = ctx_indices.expand(1, cache_bh, block_size)
    invalid_mask = invalid_mask.expand(1, cache_bh, block_size)
    return ctx_indices, invalid_mask


def _div_trunc(value: torch.Tensor, divisor: int) -> torch.Tensor:
    """Divide non-negative integers; avoids the sign correction that AIC cannot compile for floor division."""
    return torch.div(value, divisor, rounding_mode="trunc")


def _gather_paged_gqa_dp(pool: torch.Tensor, block_ids_per_page: torch.Tensor) -> torch.Tensor:
    """Gather page IDs for a physical cache whose rows are ordered [DP, CP, Hkv]."""
    if block_ids_per_page.ndim != 2:
        raise ValueError("Paged GQA block IDs must have shape [DP, pages].")
    dp, num_pages = block_ids_per_page.shape
    rows = pool.shape[1]
    if rows % dp:
        raise ValueError("Paged GQA block IDs must have shape [DP, pages] and divide the cache rows.")
    rows_per_dp = rows // dp
    block_ids_by_row = (
        block_ids_per_page.transpose(0, 1).unsqueeze(2).expand(num_pages, dp, rows_per_dp).reshape(num_pages, rows)
    )
    return ctx_gather_paged_kv_dp(pool, block_ids_by_row.to(torch.int32))


def _gather_paged_gqa_v_dp(
    value_cache: torch.Tensor,
    block_ids: torch.Tensor,
    threshold: torch.Tensor,
    block_len: int,
    num_cores: int,
    head_dim: int,
) -> torch.Tensor:
    """Gather one paged V block and zero positions beyond each CP/core validity limit."""
    invalid = (
        torch.arange(block_len, device=value_cache.device).view(1, 1, 1, block_len) > threshold.unsqueeze(-1)
    ).unsqueeze(-1)
    value_block = _gather_paged_gqa_dp(value_cache, block_ids).view(
        1, value_cache.shape[1], num_cores, block_len, head_dim
    )
    return torch.where(invalid, torch.zeros_like(value_block), value_block)


def _paged_gqa_prefill_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    position_ids: torch.Tensor,
    scaling: float,
    ctx_len: int,
    page_block_size: int,
    num_kv_blocks: int,
    num_cores: int,
    skip_kv: bool,
    num_q_blocks: int,
    q_blocks_per_outer: int,
    head_block_size: int,
) -> torch.Tensor:
    """Run benchmark-equivalent block-streamed paged GQA prefill."""
    batch, num_heads, query_len, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    if block_table.ndim == 3:
        block_table = block_table[0]
    if tuple(block_table.shape) != (batch, ctx_len // page_block_size):
        raise ValueError("Paged GQA prefill block_table must have shape [B, ctx_len / page_block_size].")

    if num_cores % num_kv_heads or num_heads % num_cores:
        raise ValueError("Paged GQA prefill requires KV heads to divide cores and cores to divide query heads.")
    if query_len <= 1 or query_len % page_block_size:
        raise ValueError("Paged GQA prefill query length must be page aligned.")
    if ctx_len % num_kv_blocks:
        raise ValueError("Paged GQA prefill requires ctx_len divisible by num_kv_blocks.")
    kv_block_size = ctx_len // num_kv_blocks
    if kv_block_size % page_block_size:
        raise ValueError("Paged GQA prefill KV blocks must contain complete pages.")
    if num_q_blocks < 1 or query_len % num_q_blocks:
        raise ValueError("Paged GQA prefill requires query length divisible by num_q_blocks.")
    if q_blocks_per_outer < 1 or num_q_blocks % q_blocks_per_outer:
        raise ValueError("Paged GQA prefill requires num_q_blocks divisible by q_blocks_per_outer.")

    kv_repeat = num_cores // num_kv_heads
    heads_per_core = num_heads // num_cores
    if head_block_size < 1 or heads_per_core % head_block_size:
        raise ValueError("Paged GQA prefill head_block_size must divide query heads per core.")
    query_block_size = query_len // num_q_blocks
    query_chunk_size = query_block_size * q_blocks_per_outer
    query_folded = query.reshape(batch, num_cores, heads_per_core, query_len, head_dim)
    head_ranges = [
        (start, min(start + head_block_size, heads_per_core)) for start in range(0, heads_per_core, head_block_size)
    ]
    output_chunks: list[torch.Tensor] = []
    is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()

    for query_chunk_start in range(0, query_len, query_chunk_size):
        query_chunk_end = min(query_chunk_start + query_chunk_size, query_len)
        query_chunk_position = position_ids[:, query_chunk_start:query_chunk_end].max(dim=-1).values
        query_states: dict[int, dict[str, Any]] = {}
        for query_start in range(query_chunk_start, query_chunk_end, query_block_size):
            query_end = min(query_start + query_block_size, query_chunk_end)
            block_len = query_end - query_start
            positions = position_ids[:, query_start:query_end]
            accumulators = []
            for head_start, head_end in head_ranges:
                head_count = head_end - head_start
                accumulators.append(
                    {
                        "head_count": head_count,
                        "query": query_folded[:, :, head_start:head_end, query_start:query_end].reshape(
                            batch, num_cores, head_count * block_len, head_dim
                        ),
                        "maximum": torch.full(
                            (batch, num_cores, head_count * block_len),
                            MIN_MASKED_ATTENTION_VALUE,
                            dtype=query.dtype,
                            device=query.device,
                        ),
                        "denominator": torch.zeros(
                            batch, num_cores, head_count * block_len, dtype=query.dtype, device=query.device
                        ),
                        "output": torch.zeros(
                            batch,
                            num_cores,
                            head_count * block_len,
                            head_dim,
                            dtype=query.dtype,
                            device=query.device,
                        ),
                    }
                )
            query_states[query_start] = {
                "block_len": block_len,
                "positions": positions,
                "current_position": positions.max(dim=-1).values,
                "accumulators": accumulators,
            }

        for kv_block_idx in range(num_kv_blocks):
            start_index = kv_block_idx * kv_block_size
            end_index = start_index + kv_block_size
            if (
                skip_kv
                and not is_export
                and bool((torch.tensor(start_index, device=query.device) > query_chunk_position).all().item())
            ):
                break
            page_start = start_index // page_block_size
            page_end = end_index // page_block_size
            key_batches = []
            value_batches = []
            for batch_idx in range(batch):
                page_ids = block_table[batch_idx, page_start:page_end].view(-1, 1).expand(-1, num_kv_heads)
                key_batches.append(ctx_gather_paged_kv_dp(key_cache, page_ids.to(torch.int32)))
                value_batches.append(ctx_gather_paged_kv_dp(value_cache, page_ids.to(torch.int32)))
            key_block = torch.cat(key_batches, dim=0).repeat_interleave(kv_repeat, dim=1)
            value_block = torch.cat(value_batches, dim=0).repeat_interleave(kv_repeat, dim=1)
            key_positions = torch.arange(start_index, end_index, device=query.device)
            key_block_transposed = key_block.transpose(2, 3)
            invalid_value = key_positions[None, None, :] > position_ids.max(dim=1, keepdim=True).values.unsqueeze(1)
            value_block = torch.where(invalid_value.unsqueeze(-1), torch.zeros_like(value_block), value_block)

            for query_start in range(query_chunk_start, query_chunk_end, query_block_size):
                state = query_states[query_start]
                skip_future = (torch.tensor(start_index, device=query.device) > state["current_position"]).all()
                if skip_kv and not is_export and bool(skip_future.item()):
                    continue
                causal = key_positions[None, None, None, :] > state["positions"][:, None, :, None]
                for accumulator in state["accumulators"]:
                    head_count = accumulator["head_count"]
                    scores = torch.matmul(accumulator["query"], key_block_transposed) * scaling
                    scores = scores.view(batch, num_cores, head_count, state["block_len"], -1)
                    scores = torch.where(
                        causal.unsqueeze(2),
                        torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=scores.dtype, device=scores.device),
                        scores,
                    ).reshape(batch, num_cores, head_count * state["block_len"], -1)
                    accumulator["maximum"], accumulator["denominator"], accumulator["output"] = (
                        update_running_softmax_prefill(
                            accumulator["maximum"],
                            scores,
                            accumulator["denominator"],
                            accumulator["output"],
                            value_block,
                            skip_kv,
                            skip_future,
                        )
                    )

        block_outputs = []
        for query_start in range(query_chunk_start, query_chunk_end, query_block_size):
            state = query_states[query_start]
            head_outputs = []
            for accumulator in state["accumulators"]:
                denominator = torch.where(
                    accumulator["denominator"] > 0,
                    accumulator["denominator"],
                    torch.ones_like(accumulator["denominator"]),
                )
                head_outputs.append(
                    (accumulator["output"] / denominator.unsqueeze(-1)).view(
                        batch,
                        num_cores,
                        accumulator["head_count"],
                        state["block_len"],
                        head_dim,
                    )
                )
            block_outputs.append(torch.cat(head_outputs, dim=2))
        output_chunks.append(torch.cat(block_outputs, dim=3))

    return torch.cat(output_chunks, dim=3).reshape(batch, num_heads, query_len, head_dim).transpose(1, 2).contiguous()


def paged_gqa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    ctx_len: Optional[int],
    skip_kv: bool = False,
    attn_dp: int = 1,
    attn_cp: int = 1,
    page_block_size: Optional[int] = None,
    num_cores_per_device: int = 1,
    num_q_blocks: int = 1,
    q_blocks_per_outer: int = 1,
    head_block_size: int = 1,
) -> tuple[torch.Tensor, None]:
    """Paged dense GQA using the MiniMax benchmark's [DP, CP, Hkv] physical layout."""
    if ctx_len is None or page_block_size is None:
        raise ValueError("Paged GQA requires ctx_len and page_block_size.")
    batch, num_heads, query_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    dp = int(attn_dp)
    cp = int(attn_cp)
    num_cores = int(num_cores_per_device)
    if dp < 1 or cp < 1 or num_cores < 1 or page_block_size < 1 or num_kv_blocks < 1 or batch % dp:
        raise ValueError("Paged GQA requires positive DP/CP/core counts and a DP-divisible batch.")
    if num_heads % num_kv_heads:
        raise ValueError("Paged GQA requires the query-head count to be divisible by the KV-head count.")
    block_table = cache_kwargs.get("block_table")
    position_ids = cache_kwargs.get("position_ids")
    if block_table is None:
        raise ValueError("Paged GQA requires a block_table.")
    if position_ids is None:
        raise ValueError("Paged GQA requires position_ids.")

    key_cache, value_cache = past_key_value.write_only_paged_attention_dp(
        key,
        value,
        layer_idx,
        {
            **cache_kwargs,
            "attn_dp": dp,
            "attn_cp": cp,
            "page_block_size": page_block_size,
        },
    )

    if query_len > 1:
        if dp != 1 or cp != 1:
            raise NotImplementedError("Paged GQA prefill currently requires attn_dp=1 and attn_cp=1.")
        return (
            _paged_gqa_prefill_attention(
                query,
                key_cache,
                value_cache,
                block_table,
                position_ids,
                scaling,
                ctx_len,
                page_block_size,
                num_kv_blocks,
                num_cores,
                skip_kv,
                int(num_q_blocks or 1),
                int(q_blocks_per_outer or 1),
                int(head_block_size or 1),
            ),
            None,
        )

    batch_local = batch // dp
    rows = dp * cp * num_kv_heads
    expected_tail = (rows, page_block_size, head_dim)
    if key_cache.ndim != 4 or tuple(key_cache.shape[1:]) != expected_tail:
        raise ValueError(
            f"Paged GQA cache must have shape [physical_pages, {rows}, {page_block_size}, {head_dim}], "
            f"got {tuple(key_cache.shape)}."
        )
    if tuple(value_cache.shape) != tuple(key_cache.shape):
        raise ValueError("Paged GQA key and value cache shapes must match.")
    if block_table.ndim != 3 or tuple(block_table.shape[:2]) != (dp, batch_local):
        raise ValueError(f"Paged GQA block_table must have shape [{dp}, {batch_local}, pages].")
    if ctx_len % page_block_size or ctx_len % num_kv_blocks:
        raise ValueError("Paged GQA requires ctx_len divisible by page_block_size and num_kv_blocks.")

    logical_block_size = ctx_len // num_kv_blocks
    if logical_block_size % (cp * page_block_size):
        raise ValueError("Each paged GQA KV block must contain complete CP page groups.")
    local_block_size = logical_block_size // cp
    num_page_groups = local_block_size // page_block_size
    if num_page_groups % num_cores:
        raise ValueError("Paged GQA page groups per KV block must be divisible by num_cores_per_device.")
    groups_per_core = num_page_groups // num_cores
    tokens_per_core = groups_per_core * page_block_size
    logical_page_groups = math.ceil(ctx_len / page_block_size / cp)
    if block_table.shape[2] < logical_page_groups:
        raise ValueError("Paged GQA block_table is too short for ctx_len.")

    num_kv_groups = num_heads // num_kv_heads
    query_len_effective = num_kv_groups * query_len
    group_order = (
        torch.arange(num_page_groups, device=query.device).view(groups_per_core, num_cores).transpose(0, 1).reshape(-1)
    )
    query_dp = query.view(dp, batch_local, num_heads, query_len, head_dim).permute(1, 0, 2, 3, 4)
    position_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2).to(torch.int32)
    position_rows = (
        position_dp.view(batch_local, dp, 1, 1, query_len)
        .expand(batch_local, dp, cp, num_kv_heads, query_len)
        .reshape(batch_local, rows, 1, query_len)
    )
    query_block = _div_trunc(position_rows, logical_block_size)
    query_block_offset = position_rows - query_block * logical_block_size
    query_page = _div_trunc(query_block_offset, page_block_size)
    query_remainder = query_block_offset - query_page * page_block_size
    core_index = torch.arange(num_cores, dtype=position_rows.dtype, device=query.device).view(1, 1, num_cores, 1)
    row_group = _div_trunc(torch.arange(rows, dtype=position_rows.dtype, device=query.device), num_kv_heads).view(
        1, rows, 1, 1
    )
    row_way = row_group - _div_trunc(row_group, cp) * cp
    block_indices = torch.arange(num_kv_blocks, dtype=position_rows.dtype, device=query.device).view(
        1, 1, 1, num_kv_blocks
    )
    query_after = (query_block > block_indices).squeeze(2)
    query_in = (query_block == block_indices).squeeze(2)
    query_page_local = _div_trunc(query_page, cp)
    query_page_way = query_page - query_page_local * cp
    last_page = torch.where(
        row_way < query_page_way,
        query_page_local,
        torch.where(row_way == query_page_way, query_page_local, query_page_local - 1),
    )
    valid_page = last_page >= 0
    safe_page = torch.where(valid_page, last_page, torch.zeros_like(last_page))
    page_group = _div_trunc(safe_page, num_cores)
    page_core = safe_page - page_group * num_cores
    group_start = page_group * page_block_size
    threshold_before = group_start + page_block_size - 1
    threshold_current = group_start + query_remainder
    threshold_after = torch.where(page_group > 0, group_start - 1, torch.full_like(group_start, -1))
    current_threshold = torch.where(
        valid_page & (core_index < page_core),
        threshold_before,
        torch.where(
            valid_page & (core_index == page_core),
            torch.where(row_way == query_page_way, threshold_current, threshold_before),
            torch.where(valid_page, threshold_after, torch.full_like(threshold_after, -1)),
        ),
    ).squeeze(-1)
    thresholds = torch.where(
        query_after.unsqueeze(2),
        torch.full_like(current_threshold.unsqueeze(-1), tokens_per_core - 1),
        torch.where(
            query_in.unsqueeze(2),
            current_threshold.unsqueeze(-1),
            torch.full_like(current_threshold.unsqueeze(-1), -1),
        ),
    )

    batch_maxima = []
    batch_sums = []
    batch_outputs = []
    is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()
    for batch_idx in range(batch_local):
        query_local = query_dp[batch_idx].reshape(dp, num_kv_heads, query_len_effective, head_dim)
        query_rows = (
            query_local.unsqueeze(1)
            .expand(dp, cp, num_kv_heads, query_len_effective, head_dim)
            .reshape(1, rows, query_len_effective, head_dim)
        )
        query_core = query_rows.unsqueeze(2).expand(1, rows, num_cores, query_len_effective, head_dim)
        table = block_table[:, batch_idx]
        maxima = []
        sums = []
        outputs = []
        key_block = _gather_paged_gqa_dp(key_cache, table[:, group_order]).view(
            1, rows, num_cores, tokens_per_core, head_dim
        )
        for block_idx in range(num_kv_blocks):
            threshold = thresholds[batch_idx : batch_idx + 1, :, :, block_idx]
            skip_future = (threshold < 0).unsqueeze(-1)
            causal_mask = torch.arange(tokens_per_core, device=query.device).view(1, 1, 1, tokens_per_core)
            causal_mask = causal_mask > threshold.unsqueeze(-1)
            causal_mask = causal_mask.unsqueeze(3).expand(-1, -1, -1, num_kv_groups, -1)
            scores = torch.matmul(query_core.float(), key_block.transpose(-1, -2).float()) * scaling
            scores = scores.masked_fill(causal_mask, -3.0e4)
            block_max = scores.max(dim=-1).values
            block_exp = torch.exp(scores - block_max.unsqueeze(-1))
            block_sum = block_exp.sum(dim=-1)
            if skip_kv:
                block_max = torch.where(skip_future, torch.full_like(block_max, -3.0e4), block_max)
                block_exp = torch.where(skip_future.unsqueeze(-1), torch.zeros_like(block_exp), block_exp)
                block_sum = torch.where(skip_future, torch.zeros_like(block_sum), block_sum)
            page_offset = block_idx * num_page_groups
            page_ids = table[:, page_offset + group_order]
            value_block = _gather_paged_gqa_v_dp(value_cache, page_ids, threshold, tokens_per_core, num_cores, head_dim)
            next_key_block = None
            if block_idx + 1 < num_kv_blocks:
                next_offset = (block_idx + 1) * num_page_groups
                next_key_block = _gather_paged_gqa_dp(key_cache, table[:, next_offset + group_order])
            block_output = torch.matmul(block_exp, value_block.float())
            if skip_kv and is_export:
                block_output = torch.where(skip_future.unsqueeze(-1), torch.zeros_like(block_output), block_output)
            maxima.append(block_max)
            sums.append(block_sum)
            outputs.append(block_output)
            if next_key_block is not None:
                key_block = next_key_block.view(1, rows, num_cores, tokens_per_core, head_dim)
        batch_maxima.append(torch.stack(maxima))
        batch_sums.append(torch.stack(sums))
        batch_outputs.append(torch.stack(outputs))

    maxima = torch.cat(batch_maxima, dim=1)
    sums = torch.cat(batch_sums, dim=1)
    outputs = torch.cat(batch_outputs, dim=1)
    block_max = maxima.max(dim=0).values
    block_weight = torch.exp(maxima - block_max.unsqueeze(0))
    block_sum = (block_weight * sums).sum(dim=0)
    block_output = (block_weight.unsqueeze(-1) * outputs).sum(dim=0)
    core_max = block_max.max(dim=2).values
    core_weight = torch.exp(block_max - core_max.unsqueeze(2))
    core_sum = (core_weight * block_sum).sum(dim=2)
    core_output = (core_weight.unsqueeze(-1) * block_output).sum(dim=2)
    core_max = core_max.view(batch_local, dp, cp, num_kv_heads, query_len_effective)
    core_sum = core_sum.view(batch_local, dp, cp, num_kv_heads, query_len_effective)
    core_output = core_output.view(batch_local, dp, cp, num_kv_heads, query_len_effective, head_dim)
    cp_max = core_max.max(dim=2).values
    cp_weight = torch.exp(core_max - cp_max.unsqueeze(2))
    cp_sum = (cp_weight * core_sum).sum(dim=2)
    cp_output = (cp_weight.unsqueeze(-1) * core_output).sum(dim=2)
    safe_sum = torch.where(cp_sum > 0, cp_sum, torch.ones_like(cp_sum))
    output = (cp_output / safe_sum.unsqueeze(-1)).view(batch_local, dp, num_heads, query_len, head_dim)
    output = output.permute(1, 0, 2, 3, 4).reshape(batch, num_heads, query_len, head_dim)
    return output.to(query.dtype).transpose(1, 2).contiguous(), None


def blocked_kv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    paged_attention: bool = False,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    ctx_len: Optional[int] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute attention by streaming key/value cache blocks through running softmax.

    This reduces peak activation memory for long contexts by splitting the cached
    key/value sequence into ``num_kv_blocks`` chunks while preserving numerically
    stable softmax accumulation across blocks. When ``paged_attention`` is set, cache
    blocks are gathered through the paged-attention block table instead of contiguous
    slices, and the block extent is taken from the physical cache layout.
    """
    # Initialize result tensor
    output = torch.zeros_like(query)

    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, _ = query.shape
    current_max = torch.full(
        (batch_size, num_heads, seq_len),
        float(MIN_MASKED_ATTENTION_VALUE),
        device=query.device,
    )
    current_denominator = torch.zeros(batch_size, num_heads, seq_len, device=query.device)

    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    if ctx_len is None:
        raise ValueError("`ctx_len` is required for blocked KV attention.")
    num_kv_blocks = max(1, num_kv_blocks)
    block_table = None
    if paged_attention:
        block_table = cache_kwargs.get("block_table")  # [BS, num_kv_blocks] -> each entry is block_id value
        kv_block_size = past_key_value.get_seq_length() if past_key_value is not None else 0
    else:
        kv_block_size = -(-ctx_len // num_kv_blocks)

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)
    current_position = position_ids.max(dim=-1).values
    # needed for GPT-OSS
    if sinks is not None:
        sinks = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

    for kv_block_idx in range(num_kv_blocks):
        start_index = kv_block_idx * kv_block_size
        if kv_block_idx == num_kv_blocks - 1:
            kv_len_block = ctx_len - start_index
        else:
            kv_len_block = kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        k_block, v_block = _read_kv_block(
            past_key_value=past_key_value,
            start_index=start_index,
            end_index=end_index,
            layer_idx=layer_idx,
            kv_block_size=kv_block_size,
            paged_attention=paged_attention,
            kv_block_idx=kv_block_idx,
            block_table=block_table,
            cache_kwargs=cache_kwargs,
        )
        k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)

        attn_weights_block = torch.matmul(query, k_block_states.transpose(2, 3)) * scaling
        # position bias needed for mpt model
        if position_bias is not None:
            attn_weights_block = attn_weights_block + position_bias[:, :, start_index:end_index]

        mask_block = None
        if attention_mask is not None:
            mask_block = attention_mask[..., start_index:end_index]
            if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                mask_block = None

        if use_causal_mask or mask_block is None:
            target_length = torch.where(
                torch.tensor(ctx_len, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
                ctx_len,
                end_index,
            )
            causal_mask_block = _create_causal_mask(
                position_ids=position_ids,
                target_length=target_length,
                sliding_window=sliding_window,
                start_index=start_index,
            )
            if mask_block is None:
                mask_block = causal_mask_block
            else:
                mask_block = mask_block.to(torch.bool) | causal_mask_block

        if mask_block is not None:
            attn_weights_block = torch.where(mask_block, masked_tensor, attn_weights_block)

        current_max, current_denominator, output = update_running_softmax(
            current_max, attn_weights_block, current_denominator, output, v_block_states, skip_kv, skip_future
        )

    # If present, apply Attention Sinks, needed for GPT-OSS
    if sinks is not None:
        _, _, output = update_running_softmax(current_max, sinks, current_denominator, output, None)

    attn_output = output.transpose(1, 2).contiguous()
    attn_weights = None

    return attn_output, attn_weights


def blocked_kv_attention_forward_decode_headpar_batch(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    ctx_len: int,
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Batch-folded decode with standard retained cache IO [FBS, Hkv, T, D].
    Used when B > 1 (decode, non-chunk_kv) — one core/device per (batch, kv-head)
    pair, so no inner `split` dimension is needed (unlike blocked_kv_attention_forward_headpar_offline).
    The physical cache is viewed as [1, FBS*Hkv, ...]; gathered blocks and
    attention use [1, B*Hkv, ...], with batch_index mapping B into FBS.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_kv_groups = getattr(module, "num_key_value_groups", None)
    num_kv_heads = num_heads // num_kv_groups
    BH = batch_size * num_kv_heads  # static at compile time
    position_ids = cache_kwargs.get("position_ids")
    num_kv_blocks = max(1, num_kv_blocks)
    kv_block_size = -(-ctx_len // num_kv_blocks)
    current_position = position_ids.max(dim=-1).values

    # Reshape query: [B, NQH, 1, D] -> [1, BH, num_kv_groups, D]
    query_flat = query.reshape(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim).reshape(
        1, BH, num_kv_groups * seq_len, head_dim
    )

    max_blocks: list = []
    sum_blocks: list = []
    out_blocks: list = []
    key_cache_folded, value_cache_folded = past_key_value.get_batch_folded_kv(layer_idx)
    # The retained cache is physically folded to [1, FBS * Hkv, T, D]. The
    # precomputed gather tensors must use that second dimension, not only the
    # active batch size, because batch_index selects rows from the full folded
    # cache during continuous/batched decode.
    cache_bh = key_cache_folded.shape[1]

    for kv_block_idx in range(num_kv_blocks):
        start_index = kv_block_idx * kv_block_size
        kv_len_block = (ctx_len - start_index) if kv_block_idx == num_kv_blocks - 1 else kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        # Construct the range/where chain in this layer's attention body and
        # pass the same tensors to both K and V readers. Keeping this work here
        # is deliberate: the ONNX exporter then records it inside the current
        # decoder-layer subfunction instead of lifting it into shared graph
        # inputs/outputs. The private cache_kwargs entries are consumed only by
        # the batch-folded readers; all other cache paths retain their existing
        # behavior.
        folded_cache_kwargs = dict(cache_kwargs)
        folded_ctx_indices, folded_invalid_mask = _prepare_batch_folded_kv_indices(
            start_index,
            end_index,
            position_ids,
            num_kv_heads,
            cache_bh,
            layer_idx,
        )
        folded_cache_kwargs["_qeff_batch_fold_ctx_indices"] = folded_ctx_indices
        folded_cache_kwargs["_qeff_batch_fold_invalid_mask"] = folded_invalid_mask

        # Read K through the folded [1, BH, T_block, D] view.
        k_block = past_key_value.read_only_blocked_K_batch(
            start_index, end_index, layer_idx, folded_cache_kwargs, folded_cache=key_cache_folded
        )

        attn_weights_block = torch.matmul(query_flat, k_block.transpose(3, 2)) * scaling

        # _create_causal_mask returns [B, 1, seq_len, T_block]; fold to [1, BH, G*seq_len, T_block].
        causal_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=end_index,
            sliding_window=sliding_window,
            start_index=start_index,
        )
        causal_mask = (
            causal_mask.expand(batch_size, num_kv_heads, seq_len, kv_len_block)
            .reshape(1, BH, seq_len, kv_len_block)
            .unsqueeze(3)
            .expand(1, BH, seq_len, num_kv_groups, kv_len_block)
            .reshape(1, BH, seq_len * num_kv_groups, kv_len_block)
        )
        attn_weights_block = attn_weights_block.masked_fill(causal_mask, MIN_MASKED_ATTENTION_VALUE)

        max_block = attn_weights_block.max(dim=3).values
        exp_block = torch.exp(attn_weights_block - max_block.unsqueeze(-1))
        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            max_block = torch.where(skip_future, torch.full_like(max_block, MIN_MASKED_ATTENTION_VALUE), max_block)
            exp_block = torch.where(skip_future, torch.zeros_like(exp_block), exp_block)

        # Read V through the folded [1, BH, T_block, D] view.
        v_block = past_key_value.read_only_blocked_V_batch(
            start_index, end_index, layer_idx, folded_cache_kwargs, folded_cache=value_cache_folded
        )
        sum_block = exp_block.sum(dim=-1)
        out_block = torch.matmul(exp_block, v_block)
        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            sum_block = torch.where(skip_future, torch.zeros_like(sum_block), sum_block)
            out_block = torch.where(skip_future, torch.zeros_like(out_block), out_block)
        max_blocks.append(max_block)
        sum_blocks.append(sum_block)
        out_blocks.append(out_block)

    max_stacked = torch.stack(max_blocks)
    sum_stacked = torch.stack(sum_blocks)
    out_stacked = torch.stack(out_blocks)
    block_max = max_stacked.max(dim=0).values
    block_weight = torch.exp(max_stacked - block_max.unsqueeze(0))
    block_sum = (block_weight * sum_stacked).sum(dim=0)
    block_out = (block_weight.unsqueeze(4) * out_stacked).sum(dim=0)
    output = block_out / block_sum.unsqueeze(-1)  # [1, BH, num_kv_groups*seq_len, D]
    attn_output = output.reshape(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim).reshape(
        batch_size, num_heads, seq_len, head_dim
    )

    return attn_output.transpose(1, 2).contiguous(), None


def blocked_kv_attention_forward_headpar_offline(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    ctx_len: int,
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    configured_split: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Head-parallel block softmax: K is split into `split` chunks along the
    # ctx dimension, computed in parallel as a 5D matmul, then two-stage
    # merged (across kv-blocks, then across splits).
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_kv_groups = getattr(module, "num_key_value_groups", None)
    past_seen_tokens = ctx_len
    position_ids = cache_kwargs.get("position_ids")
    num_kv_heads = num_heads // num_kv_groups
    split = configured_split
    num_kv_blocks = max(1, num_kv_blocks)
    kv_block_size = -(-past_seen_tokens // num_kv_blocks)
    current_position = position_ids.max(dim=-1).values

    query_folded = query.reshape(batch_size, num_kv_heads, seq_len * num_kv_groups, head_dim)
    query_5d = query_folded.unsqueeze(2).expand(batch_size, num_kv_heads, split, seq_len * num_kv_groups, head_dim)
    # -------------------------------------------------------
    # Precompute query positions once.
    # Shape: [B, 1, G*Q, 1]
    # -------------------------------------------------------
    q_pos = (
        position_ids.reshape(batch_size, 1, seq_len)
        .unsqueeze(2)
        .expand(-1, num_kv_groups, -1, -1)
        .reshape(batch_size, 1, num_kv_groups * seq_len, 1)
        .unsqueeze(2)
    )

    # -------------------------------------------------------
    # Split index template
    # Shape: [1,1,split,1,1]
    # -------------------------------------------------------
    split_idx = torch.arange(
        split,
        device=query.device,
        dtype=position_ids.dtype,
    ).view(1, 1, split, 1, 1)

    max_blocks = []
    sum_blocks = []
    out_blocks = []

    for kv_block_idx in range(num_kv_blocks):
        start_index = kv_block_idx * kv_block_size
        if kv_block_idx == num_kv_blocks - 1:
            kv_len_block = past_seen_tokens - start_index
        else:
            kv_len_block = kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (start_index > current_position).all()
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing() and not torch._dynamo.is_compiling():
                if skip_future.item():
                    break

        k_block = past_key_value.read_only_blocked_K(start_index, end_index, layer_idx, cache_kwargs)
        block_len = kv_len_block
        pad_len = 0
        if block_len % split != 0:
            pad_len = split - (block_len % split)
            k_block = nn.functional.pad(k_block, (0, 0, 0, pad_len))
            block_len += pad_len
        split_block_len = block_len // split

        key_5d = k_block.view(batch_size, num_kv_heads, split, split_block_len, head_dim)
        attn_weights_block = torch.matmul(query_5d, key_5d.transpose(-1, -2)) * scaling

        if pad_len > 0:
            chunk_start = torch.arange(split, device=query.device) * split_block_len
            valid_in_chunk = kv_len_block - chunk_start
            key_idx = torch.arange(split_block_len, device=query.device)
            pad_mask = key_idx.unsqueeze(0) >= valid_in_chunk.unsqueeze(1)
            attn_weights_block = attn_weights_block.masked_fill(
                pad_mask.view(1, 1, split, 1, split_block_len), HEADPAR_MASKED_ATTENTION_VALUE
            )

        # Absolute KV positions for this block.
        #
        # Shape:
        #   [1,1,split,1,split_block_len]
        # -------------------------------------------------------
        kv_idx = torch.arange(
            split_block_len,
            device=query.device,
            dtype=position_ids.dtype,
        ).view(1, 1, 1, 1, split_block_len)

        abs_kv_pos = start_index + split_idx * split_block_len + kv_idx

        causal_mask = abs_kv_pos > q_pos

        attn_weights_block = attn_weights_block.masked_fill(
            causal_mask,
            HEADPAR_MASKED_ATTENTION_VALUE,
        )

        max_block = attn_weights_block.max(dim=-1).values
        exp_block = torch.exp(attn_weights_block - max_block.unsqueeze(-1))
        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()):
            max_block = torch.where(skip_future, torch.full_like(max_block, HEADPAR_MASKED_ATTENTION_VALUE), max_block)
            exp_block = torch.where(skip_future, torch.zeros_like(exp_block), exp_block)

        v_block = past_key_value.read_only_blocked_V(start_index, end_index, layer_idx, cache_kwargs)
        if pad_len > 0:
            v_block = nn.functional.pad(v_block, (0, 0, 0, pad_len))
        value_5d = v_block.view(batch_size, num_kv_heads, split, split_block_len, head_dim)
        sum_block = exp_block.sum(dim=-1)
        out_block = torch.matmul(exp_block, value_5d)
        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()):
            sum_block = torch.where(skip_future, torch.zeros_like(sum_block), sum_block)
            out_block = torch.where(skip_future, torch.zeros_like(out_block), out_block)

        max_blocks.append(max_block)
        sum_blocks.append(sum_block)
        out_blocks.append(out_block)

    max_stacked = torch.stack(max_blocks)
    sum_stacked = torch.stack(sum_blocks)
    out_stacked = torch.stack(out_blocks)
    block_max = max_stacked.max(dim=0).values
    block_weight = torch.exp(max_stacked - block_max.unsqueeze(0))
    block_sum = (block_weight * sum_stacked).sum(dim=0)
    block_out = (block_weight.unsqueeze(-1) * out_stacked).sum(dim=0)

    split_max = block_max.max(dim=2).values
    split_weight = torch.exp(block_max - split_max.unsqueeze(2))
    split_sum = (split_weight * block_sum).sum(dim=2)
    split_out = (split_weight.unsqueeze(-1) * block_out).sum(dim=2)

    if sinks is not None:
        sinks_logits = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

        # Fold heads the same way as query: [B, H, QL, 1] -> [B, Hkv, QL*num_kv_groups, 1]
        sinks_folded = sinks_logits.reshape(batch_size, num_kv_heads, seq_len * num_kv_groups, 1)
        sink_logits = sinks_folded.squeeze(-1)  # [B, Hkv, QL*num_kv_groups]

        new_max = torch.maximum(split_max, sink_logits)
        scale_old = torch.exp(split_max - new_max)
        scale_sink = torch.exp(sink_logits - new_max)

        split_sum = split_sum * scale_old + scale_sink
        split_out = split_out * scale_old.unsqueeze(-1)
        split_max = new_max

    output = split_out / split_sum.unsqueeze(-1)
    attn_output = output.view(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim).reshape(
        batch_size, num_heads, seq_len, head_dim
    )
    return attn_output.transpose(1, 2).contiguous(), None


def blocked_qkv_attention_forward_prefill_headpar_offline(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_q_blocks: int,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    ctx_len: int,
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    configured_split: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Prefill: head-parallel split, online softmax per Q chunk.
    K is split into `split` chunks along ctx (same as headpar decode).
    impls=parallel path.

    ql_chunk:     Q tokens processed per iteration (q_block_size).
    n_rep_chunk:  Q head groups processed per KV block iteration.
                    1 = process all n_rep Q heads per KV head at once.
                    >1 = split Q head groups into smaller chunks to reduce memory.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_kv_groups = getattr(module, "num_key_value_groups", None)
    num_kv_heads = num_heads // num_kv_groups
    split = configured_split
    num_kv_blocks = max(1, num_kv_blocks)
    kv_block_size = -(-ctx_len // num_kv_blocks)
    n_rep_chunk = kwargs.get("n_rep_chunk")
    if n_rep_chunk is None:
        n_rep_chunk = num_kv_groups
    ql_chunk = -(-seq_len // num_q_blocks)
    position_ids = cache_kwargs.get("position_ids")

    query_folded = query.reshape(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim)

    t_chunks = []
    for t_start in range(0, seq_len, ql_chunk):
        t_end = min(t_start + ql_chunk, seq_len)
        tc = t_end - t_start
        pos_sub = position_ids[:, t_start:t_end]
        current_position = pos_sub.max(dim=-1).values

        r_ranges = [
            (r_start, min(r_start + n_rep_chunk, num_kv_groups)) for r_start in range(0, num_kv_groups, n_rep_chunk)
        ]

        assert kv_block_size % split == 0, f"kv_block_size ({kv_block_size}) must be divisible by split ({split})"
        T_h_nom = kv_block_size // split
        kv_offsets = (
            torch.arange(split, device=query.device)[:, None] * T_h_nom
            + torch.arange(T_h_nom, device=query.device)[None, :]
        ).repeat(num_kv_heads, 1)  # [num_kv_heads*split, T_h_nom]
        is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=query.dtype, device=query.device)

        accs = []
        for r_start, r_end in r_ranges:
            rc = r_end - r_start
            accs.append(
                {
                    "rc": rc,
                    "query": query_folded[:, :, r_start:r_end, t_start:t_end, :]
                    .reshape(batch_size, num_kv_heads, rc * tc, head_dim)
                    .repeat_interleave(split, dim=1),  # [B, num_kv_heads*split, rc*tc, head_dim]
                    "m_acc": torch.full(
                        (batch_size, num_kv_heads * split, rc * tc),
                        float(MIN_MASKED_ATTENTION_VALUE),
                        device=query.device,
                        dtype=query.dtype,
                    ),
                    "s_acc": torch.zeros(
                        batch_size, num_kv_heads * split, rc * tc, device=query.device, dtype=query.dtype
                    ),
                    "o_acc": torch.zeros(
                        batch_size, num_kv_heads * split, rc * tc, head_dim, device=query.device, dtype=query.dtype
                    ),
                }
            )

        for kv_block_idx in range(num_kv_blocks):
            start_index = kv_block_idx * kv_block_size
            kv_len_block = (ctx_len - start_index) if kv_block_idx == num_kv_blocks - 1 else kv_block_size
            end_index = start_index + kv_len_block
            split_block_len = kv_len_block // split

            skip_future = None
            if skip_kv:
                skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
                if not is_export and skip_future.item():
                    break

            k_block = past_key_value.read_only_blocked_K(start_index, end_index, layer_idx, cache_kwargs)
            v_block = past_key_value.read_only_blocked_V(start_index, end_index, layer_idx, cache_kwargs)
            key_5d = k_block.view(batch_size, num_kv_heads, split, split_block_len, head_dim)
            value_5d = v_block.view(batch_size, num_kv_heads, split, split_block_len, head_dim)
            K_4d = key_5d.reshape(batch_size, num_kv_heads * split, split_block_len, head_dim)
            V_4d = value_5d.reshape(batch_size, num_kv_heads * split, split_block_len, head_dim)

            split_causal_masks = []
            for s in range(split):
                s_start = start_index + s * split_block_len
                mask_s = _create_causal_mask(
                    position_ids=pos_sub,
                    target_length=s_start + split_block_len,
                    sliding_window=sliding_window,
                    start_index=s_start,
                )
                # mask_s: [B, 1, Q, split_block_len]
                split_causal_masks.append(mask_s)
            causal_mask_block = (
                torch.stack(split_causal_masks, dim=2)
                .expand(batch_size, num_kv_heads, split, tc, split_block_len)
                .reshape(batch_size, num_kv_heads * split, tc, split_block_len)
            )
            skip_split = (kv_offsets[:, 0] > (pos_sub.max() - start_index)).view(1, num_kv_heads * split)

            for acc in accs:
                rc = acc["rc"]
                # causal_mask_block: [B, num_kv_heads*split, tc, split_block_len] → expand to [B, num_kv_heads*split, rc*tc, split_block_len]
                causal_rc = causal_mask_block.repeat(1, 1, rc, 1) if rc > 1 else causal_mask_block
                attn_weights_block = torch.matmul(acc["query"], K_4d.transpose(-1, -2)) * scaling
                attn_weights_block = torch.where(causal_rc, masked_tensor, attn_weights_block)
                acc["m_acc"], acc["s_acc"], acc["o_acc"] = update_running_softmax(
                    acc["m_acc"],
                    attn_weights_block,
                    acc["s_acc"],
                    acc["o_acc"],
                    V_4d,
                    skip_kv=True,
                    skip_future=skip_split.unsqueeze(-1),
                )

        r_chunks = []
        for acc in accs:
            rc = acc["rc"]
            m = acc["m_acc"].view(batch_size, num_kv_heads, split, rc * tc)
            s = acc["s_acc"].view(batch_size, num_kv_heads, split, rc * tc)
            o = acc["o_acc"].view(batch_size, num_kv_heads, split, rc * tc, head_dim)
            split_max = m.max(dim=2).values
            split_weight = torch.exp(m - split_max.unsqueeze(2))
            split_sum = (split_weight * s).sum(dim=2)
            split_out = (split_weight.unsqueeze(-1) * s.unsqueeze(-1) * o).sum(dim=2)
            r_chunks.append((split_out / split_sum.unsqueeze(-1)).view(batch_size, num_kv_heads, rc, tc, head_dim))

        t_chunks.append(torch.cat(r_chunks, dim=2))

    output = torch.cat(t_chunks, dim=3)
    attn_output = output.reshape(batch_size, num_heads, seq_len, head_dim)
    return attn_output.transpose(1, 2).contiguous(), None


def blocked_qkv_attention_forward_prefill_online(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_q_blocks: int,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    ctx_len: int,
    n_rep_chunk: Optional[int] = 1,
    num_cores_per_device: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, NQH, QL, D = query.shape
    num_kv_groups = getattr(module, "num_key_value_groups", None)
    Hkv = NQH // num_kv_groups
    num_cores = num_cores_per_device if num_cores_per_device is not None else Hkv
    if num_cores > NQH:
        num_cores = Hkv
    if num_cores <= 0:
        raise ValueError(f"Invalid number of cores {num_cores}; num_cores must be greater than zero")
    if num_cores < Hkv:
        raise ValueError(
            f"Invalid number of cores {num_cores} for {Hkv} KV heads; num_cores must be at least the number of KV heads"
        )
    if num_cores % Hkv != 0:
        raise ValueError(
            f"Invalid number of cores {num_cores} for {Hkv} KV heads; "
            "num_cores must be a multiple of the number of KV heads"
        )
    if NQH % num_cores != 0:
        raise ValueError(
            f"Invalid number of cores {num_cores} for number of query heads {NQH}, "
            "should be able to evenly distribute number of query heads across number of cores"
        )
    kv_repeat = num_cores // Hkv
    n_rep_per_core = NQH // num_cores
    skip_kv = kwargs.get("skip_kv", False)
    num_kv_blocks = max(1, num_kv_blocks)
    kv_block_size = -(-ctx_len // num_kv_blocks)
    ql_chunk = -(-QL // num_q_blocks)
    position_ids = cache_kwargs.get("position_ids")

    assert n_rep_per_core % n_rep_chunk == 0, (
        f"q_head_block_chunk ({n_rep_chunk}) must divide NQH//num_cores ({n_rep_per_core})."
    )

    q_fold = query.reshape(B, num_cores, n_rep_per_core, QL, D)
    is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()
    t_chunks = []
    for t_start in range(0, QL, ql_chunk):
        t_end = min(t_start + ql_chunk, QL)
        tc = t_end - t_start
        pos_sub = position_ids[:, t_start:t_end]
        current_position = pos_sub.max(dim=-1).values

        r_ranges = [
            (r_start, min(r_start + n_rep_chunk, n_rep_per_core)) for r_start in range(0, n_rep_per_core, n_rep_chunk)
        ]

        accs = []
        for r_start, r_end in r_ranges:
            rc = r_end - r_start
            accs.append(
                {
                    "rc": rc,
                    "Q": q_fold[:, :, r_start:r_end, t_start:t_end, :].reshape(B, num_cores, rc * tc, D),
                    "m_acc": torch.full(
                        (B, num_cores, rc * tc),
                        float(MIN_MASKED_ATTENTION_VALUE),
                        device=query.device,
                        dtype=query.dtype,
                    ),
                    "s_acc": torch.zeros(B, num_cores, rc * tc, device=query.device, dtype=query.dtype),
                    "o_acc": torch.zeros(B, num_cores, rc * tc, D, device=query.device, dtype=query.dtype),
                }
            )

        for kv_block_idx in range(num_kv_blocks):
            start_index = kv_block_idx * kv_block_size
            kv_len_block = (ctx_len - start_index) if kv_block_idx == num_kv_blocks - 1 else kv_block_size
            end_index = start_index + kv_len_block

            skip_future = None
            if skip_kv:
                skip_future = (start_index > current_position).all()
                if not is_export and skip_future.item():
                    break

            k_block = past_key_value.read_only_blocked_K(start_index, end_index, layer_idx, cache_kwargs)
            v_block = past_key_value.read_only_blocked_V(start_index, end_index, layer_idx, cache_kwargs)
            k_block, v_block = _get_kv_states(module, k_block, v_block, num_repeat=kv_repeat)

            k_abs = torch.arange(start_index, end_index, device=query.device)
            causal = k_abs[None, None, None, :] > pos_sub[:, None, :, None]

            for acc in accs:
                rc = acc["rc"]
                attn = torch.matmul(acc["Q"], k_block.transpose(2, 3)) * scaling
                attn_m = (
                    attn.view(B, num_cores, rc, tc, -1)
                    .masked_fill(causal.unsqueeze(2), float(MIN_MASKED_ATTENTION_VALUE))
                    .view(B, num_cores, rc * tc, -1)
                )
                acc["m_acc"], acc["s_acc"], acc["o_acc"] = update_running_softmax_prefill(
                    acc["m_acc"],
                    attn_m,
                    acc["s_acc"],
                    acc["o_acc"],
                    v_block,
                    skip_kv,
                    skip_future,
                )

        r_chunks = []
        for acc in accs:
            rc = acc["rc"]
            out = acc["o_acc"] / acc["s_acc"].unsqueeze(-1)
            r_chunks.append(out.view(B, num_cores, rc, tc, D))
        t_chunks.append(torch.cat(r_chunks, dim=2))

    attn_output = torch.cat(t_chunks, dim=3).reshape(B, NQH, QL, D)
    return attn_output.transpose(1, 2).contiguous(), None


def blocked_kv_attention_forward_prefill_headpar_offline(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    configured_split: Optional[int] = None,
    ctx_len: int,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, NQH, QL, D_abs = query.shape
    kv_lora_rank = module.head_dim
    num_kv_groups = getattr(module, "num_key_value_groups", None)
    split = configured_split
    num_kv_blocks = max(1, num_kv_blocks)
    n_rep = num_kv_groups
    Hkv = NQH // num_kv_groups
    n_rep_chunk = n_rep

    position_ids = cache_kwargs.get("position_ids")
    kv_block_size = -(-ctx_len // num_kv_blocks)

    # ── Q 6D: [B, Hkv, split, n_rep, QL, D_abs] ─────────────────────────────
    q_fold = query.reshape(B, Hkv, n_rep, QL, D_abs)
    Q_6d = q_fold.unsqueeze(2).expand(B, Hkv, split, n_rep, QL, D_abs)

    current_position = position_ids.max(dim=-1).values

    max_buf: list = []
    sum_buf: list = []
    out_buf: list = []

    for kv_block_idx in range(num_kv_blocks):
        start_index = kv_block_idx * kv_block_size
        kv_len_block = ctx_len - start_index if kv_block_idx == num_kv_blocks - 1 else kv_block_size
        end_index = start_index + kv_len_block
        T_orig = kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        k_block = past_key_value.read_only_blocked_K(start_index, end_index, layer_idx, cache_kwargs)
        ckv_for_v = past_key_value.read_only_blocked_V(start_index, end_index, layer_idx, cache_kwargs)

        T_blk = T_orig
        pad = 0
        if T_blk % split != 0:
            pad = split - (T_blk % split)
            k_block = nn.functional.pad(k_block, (0, 0, 0, pad))
            ckv_for_v = nn.functional.pad(ckv_for_v, (0, 0, 0, pad))
            T_blk += pad
        T_h = T_blk // split

        # 5D K/V: [B, Hkv, split, T_h, D]
        K_5d = k_block.view(B, Hkv, split, T_h, D_abs)
        V_5d = ckv_for_v.view(B, Hkv, split, T_h, kv_lora_rank)

        split_causal_masks = []
        for s in range(split):
            s_start = start_index + s * T_h
            mask_s = _create_causal_mask(
                position_ids=position_ids,
                target_length=s_start + T_h,
                sliding_window=sliding_window,
                start_index=s_start,
            )
            split_causal_masks.append(mask_s.unsqueeze(2))  # [B, 1, 1, QL, T_h]
        causal_mask = torch.stack(split_causal_masks, dim=2)  # [B, 1, split, 1, QL, T_h]

        rep_max: list = []
        rep_sum: list = []
        rep_out: list = []

        for r_start in range(0, n_rep, n_rep_chunk):
            r_end = min(r_start + n_rep_chunk, n_rep)
            # Q_chunk: [B, Hkv, split, chunk, QL, D_abs]
            Q_chunk = Q_6d[:, :, :, r_start:r_end, :, :]
            # [B, Hkv, split, chunk, QL, D] @ [B, Hkv, split, 1, D, T_h]
            attn_c = torch.matmul(Q_chunk, K_5d.unsqueeze(3).transpose(-1, -2)) * scaling

            if pad > 0:
                chunk_start = torch.arange(split, device=attn_c.device) * T_h
                valid_in_chunk = T_orig - chunk_start
                k_idx = torch.arange(T_h, device=attn_c.device)
                pad_mask = k_idx.unsqueeze(0) >= valid_in_chunk.unsqueeze(1)
                attn_c = attn_c.masked_fill(pad_mask.view(1, 1, split, 1, 1, T_h), MIN_MASKED_ATTENTION_VALUE)

            attn_c = attn_c.masked_fill(causal_mask, MIN_MASKED_ATTENTION_VALUE)

            m_c = attn_c.max(dim=-1).values  # [B, Hkv, split, chunk, QL]
            exp_c = torch.exp(attn_c - m_c.unsqueeze(-1))

            if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                m_c = torch.where(skip_future, torch.full_like(m_c, float(MIN_MASKED_ATTENTION_VALUE)), m_c)
                exp_c = torch.where(skip_future, torch.zeros_like(exp_c), exp_c)

            sum_c = exp_c.sum(dim=-1)
            out_c = torch.matmul(exp_c, V_5d.unsqueeze(3))  # [B, Hkv, split, chunk, QL, kv_lora_rank]

            if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                sum_c = torch.where(skip_future, torch.zeros_like(sum_c), sum_c)
                out_c = torch.where(skip_future, torch.zeros_like(out_c), out_c)

            rep_max.append(m_c)
            rep_sum.append(sum_c)
            rep_out.append(out_c)

        # concat over n_rep chunks → [B, Hkv, split, n_rep, QL] / [..., kv_lora_rank]
        m_blk = torch.cat(rep_max, dim=3)
        sum_blk = torch.cat(rep_sum, dim=3)
        out_blk = torch.cat(rep_out, dim=3)

        max_buf.append(m_blk)
        sum_buf.append(sum_blk)
        out_buf.append(out_blk)

    # ── Stage 1: merge across KV blocks ──────────────────────────────────────
    max_stk = torch.stack(max_buf)  # [nkvb, B, Hkv, split, n_rep, QL]
    sum_stk = torch.stack(sum_buf)
    out_stk = torch.stack(out_buf)  # [nkvb, B, Hkv, split, n_rep, QL, kv_lora_rank]
    m1 = max_stk.max(dim=0).values
    w1 = torch.exp(max_stk - m1.unsqueeze(0))
    s1 = (w1 * sum_stk).sum(dim=0)
    o1 = (w1.unsqueeze(-1) * out_stk).sum(dim=0)

    # ── Stage 2: merge across splits ─────────────────────────────────────────
    m2 = m1.max(dim=2).values  # [B, Hkv, n_rep, QL]
    w2 = torch.exp(m1 - m2.unsqueeze(2))
    s2 = (w2 * s1).sum(dim=2)
    o2 = (w2.unsqueeze(-1) * o1).sum(dim=2)

    if sinks is not None:
        # sinks: [NQH] → per-head logit, same for all query positions
        # sink_logits: [B, Hkv, n_rep, QL]
        sink_logits = sinks.reshape(1, -1, 1, 1).expand(B, -1, QL, -1).reshape(B, Hkv, n_rep, QL, 1).squeeze(-1)
        new_max = torch.maximum(m2, sink_logits)
        scale_old = torch.exp(m2 - new_max)
        scale_sink = torch.exp(sink_logits - new_max)
        s2 = s2 * scale_old + scale_sink
        o2 = o2 * scale_old.unsqueeze(-1)

    output = o2 / s2.unsqueeze(-1)

    # ── Unfold + v_up ─────────────────────────────────────────────────────────
    # [B, Hkv, n_rep, QL, kv_lora_rank] → [B, NQH, QL, kv_lora_rank]
    attn_output = output.reshape(B, NQH, QL, kv_lora_rank)

    return attn_output.transpose(1, 2).contiguous(), None


def blocked_q_attention_forward_prefill(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_q_blocks: int,
    cache_kwargs: Dict[str, Any],
    *,
    sliding_window: Optional[int] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Q-blocked prefill attention.

    Query tokens are sliced into num_q_blocks blocks; each block attends over
    the full K/V using a causal mask derived from position_ids.
    """
    batch_size, num_heads, q_len, _ = query.shape
    num_q_blocks = max(1, _normalize_int(num_q_blocks))
    key_states, value_states = _get_kv_states(module, key, value)
    position_ids = cache_kwargs.get("position_ids")

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    q_block_starts = [-(-i * q_len) // num_q_blocks for i in range(num_q_blocks)]
    q_output_blocks = []
    q_attn_blocks = []

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_starts[q_block_idx]
        q_len_block = q_len - q_start if q_block_idx == num_q_blocks - 1 else q_block_starts[q_block_idx + 1] - q_start

        q_block = query[:, :, q_start : q_start + q_len_block, :]
        position_ids_block = position_ids[:, q_start : q_start + q_len_block]

        attn_weights = torch.matmul(q_block, key_states.transpose(2, 3)) * scaling

        if position_bias is not None:
            attn_weights = attn_weights + position_bias

        causal_mask = _create_causal_mask(
            position_ids=position_ids_block,
            target_length=key_states.shape[2],
            sliding_window=sliding_window,
            start_index=0,
        )
        attn_weights = torch.where(causal_mask, masked_tensor, attn_weights)

        if sinks is not None:
            sinks_g = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, q_len_block, -1)
            combined_logits = torch.cat([attn_weights, sinks_g], dim=3)
            attn_weights = combined_logits - combined_logits.max(dim=3, keepdim=True).values

        attn_weights = torch.softmax(attn_weights, dim=3, dtype=torch.float32).to(query.dtype)

        if sinks is not None:
            attn_weights = attn_weights[..., : key.shape[2]]

        q_output_blocks.append(torch.matmul(attn_weights, value_states))
        q_attn_blocks.append(attn_weights)

    attn_output = torch.cat(q_output_blocks, dim=2).transpose(1, 2).contiguous()
    attn_weights = torch.cat(q_attn_blocks, dim=2)
    return attn_output, attn_weights


def blocked_qkv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    num_q_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    paged_attention: bool = False,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    ctx_len: Optional[int] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute attention by streaming query and key/value blocks.

    Query tokens are split into ``num_q_blocks`` and each query block attends over
    ``num_kv_blocks`` cached key/value chunks using running softmax accumulation.
    """
    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, DH = query.shape

    if ctx_len is None:
        raise ValueError("`ctx_len` is required for blocked QKV attention.")
    past_seen_tokens = ctx_len
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")

    num_q_blocks = max(1, num_q_blocks) if num_q_blocks else 1
    q_block_positions = [-(-i * seq_len) // num_q_blocks for i in range(num_q_blocks)]
    num_kv_blocks = max(1, num_kv_blocks) if num_kv_blocks else 1

    block_table = None
    if paged_attention:
        block_table = cache_kwargs.get("block_table")  # [BS, num_kv_blocks] -> each entry is block_id value
        kv_block_size = past_key_value.get_seq_length() if past_key_value is not None else 0
        past_seen_tokens = kv_block_size * num_kv_blocks
    else:
        kv_block_size = -(-past_seen_tokens // num_kv_blocks)

    q_output_blocks = []
    q_attn_blocks = []
    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)
    current_position = position_ids.max(dim=-1).values
    # needed for GPT-OSS
    if sinks is not None:
        sinks = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

    # Gather each KV block once: block_index/updated/cache_kwargs only depend on `kv_block_idx`,
    # not on q_block_idx, so hoist the read out of the q-block loop below.
    kv_blocks = []
    for kv_block_idx in range(num_kv_blocks):
        start_index = kv_block_idx * kv_block_size
        if kv_block_idx == num_kv_blocks - 1:
            kv_len_block = past_seen_tokens - start_index
        else:
            kv_len_block = kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        k_block, v_block = _read_kv_block(
            past_key_value=past_key_value,
            start_index=start_index,
            end_index=end_index,
            layer_idx=layer_idx,
            kv_block_size=kv_block_size,
            paged_attention=paged_attention,
            kv_block_idx=kv_block_idx,
            block_table=block_table,
            cache_kwargs=cache_kwargs,
        )
        k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)
        kv_blocks.append((start_index, end_index, skip_future, k_block_states, v_block_states))

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_positions[q_block_idx]
        if q_block_idx == num_q_blocks - 1:
            q_len_block = seq_len - q_start
        else:
            q_len_block = q_block_positions[q_block_idx + 1] - q_start

        q_block = query[:, :, q_start : q_start + q_len_block, :]

        current_max = torch.full(
            (batch_size, num_heads, q_len_block),
            float(MIN_MASKED_ATTENTION_VALUE),
            device=query.device,
        )
        current_denominator = torch.zeros(batch_size, num_heads, q_len_block, device=query.device)
        output_blocks = torch.zeros((batch_size, num_heads, q_len_block, DH), device=query.device, dtype=query.dtype)

        for start_index, end_index, skip_future, k_block_states, v_block_states in kv_blocks:
            attn_weights_block = torch.matmul(q_block, k_block_states.transpose(2, 3)) * scaling
            # position bias needed for mpt model
            if position_bias is not None:
                attn_weights_block = attn_weights_block + position_bias[:, :, start_index:end_index]

            mask_block = None
            if attention_mask is not None:
                mask_block = attention_mask[..., start_index:end_index]
                if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                    mask_block = None

            if use_causal_mask or mask_block is None:
                # target_length = min(total_seen_tokens, end_index)
                target_length = torch.where(
                    torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
                    past_seen_tokens,
                    end_index,
                )
                causal_mask_block = _create_causal_mask(
                    position_ids=position_ids,
                    target_length=target_length,
                    sliding_window=sliding_window,
                    start_index=start_index,
                )
                if mask_block is None:
                    mask_block = causal_mask_block
                else:
                    mask_block = mask_block.to(torch.bool) | causal_mask_block

            if mask_block is not None:
                attn_mask_block = mask_block[:, :, q_start : q_start + q_len_block, :]
                attn_weights_block = torch.where(attn_mask_block, masked_tensor, attn_weights_block)

            current_max, current_denominator, output_blocks = update_running_softmax(
                current_max,
                attn_weights_block,
                current_denominator,
                output_blocks,
                v_block_states,
                skip_kv,
                skip_future,
            )

        # If present, apply Attention Sinks, needed for GPT-OSS
        if sinks is not None:
            _, _, output_blocks = update_running_softmax(current_max, sinks, current_denominator, output_blocks, None)
        q_output_blocks.append(output_blocks)
        q_attn_blocks.append(attn_weights_block)

    attn_output = torch.cat(q_output_blocks, dim=2).transpose(1, 2).contiguous()
    attn_weights = torch.cat(q_attn_blocks, dim=2)

    return attn_output, attn_weights


def blocked_hqkv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    num_q_blocks: int,
    head_block_size: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    paged_attention: bool = False,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    ctx_len: Optional[int] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, DH = query.shape

    if ctx_len is None:
        raise ValueError("`ctx_len` is required for blocked HQKV attention.")
    past_seen_tokens = ctx_len
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)
    num_q_blocks = max(1, num_q_blocks) if num_q_blocks else 1
    q_block_positions = [-(-i * seq_len) // num_q_blocks for i in range(num_q_blocks)]
    num_kv_blocks = max(1, num_kv_blocks)
    block_table = None
    if paged_attention:
        block_table = cache_kwargs.get("block_table")  # [BS, num_kv_blocks] -> each entry is block_id value
        kv_block_size = past_key_value.get_seq_length() if past_key_value is not None else 0
        past_seen_tokens = kv_block_size * num_kv_blocks
    else:
        kv_block_size = -(-past_seen_tokens // num_kv_blocks) if num_kv_blocks else 1

    h_output_blocks = []
    h_attn_blocks = []
    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)
    current_position = position_ids.max(dim=-1).values
    # needed for GPT-OSS
    if sinks is not None:
        sinks = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

    # Gather each KV block once: block_index/updated/cache_kwargs only depend on `kv_block_idx`,
    # not on head_block_idx/q_block_idx, so hoist the read out of the loops below.
    kv_blocks = []
    for kv_block_idx in range(num_kv_blocks):
        start_index = kv_block_idx * kv_block_size
        if kv_block_idx == num_kv_blocks - 1:
            kv_len_block = past_seen_tokens - start_index
        else:
            kv_len_block = kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        k_block, v_block = _read_kv_block(
            past_key_value=past_key_value,
            start_index=start_index,
            end_index=end_index,
            layer_idx=layer_idx,
            kv_block_size=kv_block_size,
            paged_attention=paged_attention,
            kv_block_idx=kv_block_idx,
            block_table=block_table,
            cache_kwargs=cache_kwargs,
        )
        k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)
        kv_blocks.append((start_index, end_index, skip_future, k_block_states, v_block_states))

    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        # Extract head blocks
        q_g = query[:, h_start:h_end, :, :]

        q_output_blocks = []
        q_attn_blocks = []

        for q_block_idx in range(num_q_blocks):
            q_start = q_block_positions[q_block_idx]
            if q_block_idx == num_q_blocks - 1:
                q_len_block = seq_len - q_start
            else:
                q_len_block = q_block_positions[q_block_idx + 1] - q_start

            q_block = q_g[:, :, q_start : q_start + q_len_block, :]

            current_max = torch.full(
                (batch_size, h_end - h_start, q_len_block),
                float(MIN_MASKED_ATTENTION_VALUE),
                device=query.device,
            )
            current_denominator = torch.zeros(batch_size, h_end - h_start, q_len_block, device=query.device)
            output_blocks = torch.zeros(
                (batch_size, h_end - h_start, q_len_block, DH), device=query.device, dtype=query.dtype
            )

            for start_index, end_index, skip_future, k_block_states, v_block_states in kv_blocks:
                k_g = k_block_states[:, h_start:h_end, :, :]
                v_g = v_block_states[:, h_start:h_end, :, :]

                attn_weights_block = torch.matmul(q_block, k_g.transpose(2, 3)) * scaling
                # position bias needed for mpt model
                if position_bias is not None:
                    attn_weights_block = attn_weights_block + position_bias[h_start:h_end, :, start_index:end_index]

                mask_block = None
                if attention_mask is not None:
                    mask_block = attention_mask[..., start_index:end_index]
                    if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                        mask_block = None

                if use_causal_mask or mask_block is None:
                    # target_length = min(total_seen_tokens, end_index)
                    target_length = torch.where(
                        torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
                        past_seen_tokens,
                        end_index,
                    )
                    causal_mask_block = _create_causal_mask(
                        position_ids=position_ids,
                        target_length=target_length,
                        sliding_window=sliding_window,
                        start_index=start_index,
                    )
                    if mask_block is None:
                        mask_block = causal_mask_block
                    else:
                        mask_block = mask_block.to(torch.bool) | causal_mask_block

                if mask_block is not None:
                    mask_block_g = mask_block[:, :, q_start : q_start + q_len_block, :]
                    attn_weights_block = torch.where(mask_block_g, masked_tensor, attn_weights_block)

                current_max, current_denominator, output_blocks = update_running_softmax(
                    current_max, attn_weights_block, current_denominator, output_blocks, v_g, skip_kv, skip_future
                )
            # If present, apply Attention Sinks, needed for GPT-OSS
            if sinks is not None:
                _, _, output_blocks = update_running_softmax(
                    current_max, sinks, current_denominator, output_blocks, None
                )
            q_output_blocks.append(output_blocks)
            q_attn_blocks.append(attn_weights_block)

        head_output = torch.cat(q_output_blocks, dim=2)
        head_attn_weights = torch.cat(q_attn_blocks, dim=2)
        h_output_blocks.append(head_output)
        h_attn_blocks.append(head_attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)

    return attn_output, attn_weights


def blocked_bhqkv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    num_q_blocks: int,
    num_batch_blocks: int,
    head_block_size: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    paged_attention: bool = False,
    score_mod: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    ctx_len: Optional[int] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, DH = query.shape

    if ctx_len is None:
        raise ValueError("`ctx_len` is required for blocked BHQKV attention.")
    past_seen_tokens = ctx_len
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)
    num_q_blocks = max(1, _normalize_int(num_q_blocks))
    q_block_positions = [-(-i * seq_len) // num_q_blocks for i in range(num_q_blocks)]
    num_kv_blocks = max(1, num_kv_blocks)

    block_table = None
    if paged_attention:
        block_table = cache_kwargs.get("block_table")  # [BS, num_kv_blocks] -> each entry is block_id value
        kv_block_size = past_key_value.get_seq_length() if past_key_value is not None else 0
        past_seen_tokens = kv_block_size * num_kv_blocks
    else:
        kv_block_size = -(-past_seen_tokens // num_kv_blocks)

    h_output_blocks = []
    h_attn_blocks = []

    num_batch_blocks = max(
        1, min(batch_size, _normalize_int(num_batch_blocks))
    )  # default to batch size for number of batch blocks
    batch_block_positions = [(i * batch_size) // num_batch_blocks for i in range(num_batch_blocks)]

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    current_position = position_ids.max(dim=-1).values
    # needed for GPT-OSS
    if sinks is not None:
        sinks = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

    # Gather each KV block once: block_index/updated/cache_kwargs only depend on `kv_block_idx`,
    # not on head_block_idx/q_block_idx/b_block_idx, so hoist the read out of the loops below.
    kv_blocks = []
    for kv_block_idx in range(num_kv_blocks):
        start_index = kv_block_idx * kv_block_size
        if kv_block_idx == num_kv_blocks - 1:
            kv_len_block = past_seen_tokens - start_index
        else:
            kv_len_block = kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        k_block, v_block = _read_kv_block(
            past_key_value=past_key_value,
            start_index=start_index,
            end_index=end_index,
            layer_idx=layer_idx,
            kv_block_size=kv_block_size,
            paged_attention=paged_attention,
            kv_block_idx=kv_block_idx,
            block_table=block_table,
            cache_kwargs=cache_kwargs,
        )
        k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)
        kv_blocks.append((start_index, end_index, skip_future, k_block_states, v_block_states))

    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        # Extract head blocks
        q_g = query[:, h_start:h_end, :, :]

        q_output_blocks = []
        q_attn_blocks = []

        for q_block_idx in range(num_q_blocks):
            q_start = q_block_positions[q_block_idx]
            if q_block_idx == num_q_blocks - 1:
                q_len_block = seq_len - q_start
            else:
                q_len_block = q_block_positions[q_block_idx + 1] - q_start

            q_block_head = q_g[:, :, q_start : q_start + q_len_block, :]

            batch_output_blocks = []
            batch_attn_blocks = []

            for b_block_idx in range(num_batch_blocks):
                batch_start = batch_block_positions[b_block_idx]
                if b_block_idx == num_batch_blocks - 1:
                    batch_len = batch_size - batch_start
                else:
                    batch_len = batch_block_positions[b_block_idx + 1] - batch_start

                q_block = q_block_head[batch_start : batch_start + batch_len, :, :, :]

                current_max = torch.full(
                    (batch_len, h_end - h_start, q_len_block),
                    float(MIN_MASKED_ATTENTION_VALUE),
                    device=query.device,
                )
                current_denominator = torch.zeros(batch_len, h_end - h_start, q_len_block, device=query.device)
                output_blocks = torch.zeros(
                    (batch_len, h_end - h_start, q_len_block, DH), device=query.device, dtype=query.dtype
                )

                for start_index, end_index, skip_future, k_block_states, v_block_states in kv_blocks:
                    k_g = k_block_states[batch_start : batch_start + batch_len, h_start:h_end, :, :]
                    v_g = v_block_states[batch_start : batch_start + batch_len, h_start:h_end, :, :]

                    attn_weights_block = torch.matmul(q_block, k_g.transpose(2, 3)) * scaling
                    # position bias needed for mpt model
                    if position_bias is not None:
                        attn_weights_block = attn_weights_block + position_bias[h_start:h_end, :, start_index:end_index]

                    mask_block = None
                    if attention_mask is not None:
                        mask_block = attention_mask[..., start_index:end_index]
                        if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                            mask_block = None

                    if use_causal_mask or mask_block is None:
                        # target_length = min(total_seen_tokens, end_index)
                        target_length = torch.where(
                            torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
                            past_seen_tokens,
                            end_index,
                        )
                        causal_mask_block = _create_causal_mask(
                            position_ids=position_ids,
                            target_length=target_length,
                            sliding_window=sliding_window,
                            start_index=start_index,
                        )
                        if mask_block is None:
                            mask_block = causal_mask_block
                        else:
                            mask_block = mask_block.to(torch.bool) | causal_mask_block

                    if mask_block is not None:
                        mask_block_g = mask_block[
                            batch_start : batch_start + batch_len, :, q_start : q_start + q_len_block, :
                        ]
                        attn_weights_block = torch.where(mask_block_g, masked_tensor, attn_weights_block)

                    current_max, current_denominator, output_blocks = update_running_softmax(
                        current_max, attn_weights_block, current_denominator, output_blocks, v_g, skip_kv, skip_future
                    )
                batch_output_blocks.append(output_blocks)
                batch_attn_blocks.append(attn_weights_block)
            # If present, apply Attention Sinks, needed for GPT-OSS
            if sinks is not None:
                _, _, batch_output_blocks = update_running_softmax(
                    current_max, sinks, current_denominator, batch_output_blocks, None
                )
            q_output_blocks.append(torch.cat(batch_output_blocks, dim=0))
            q_attn_blocks.append(torch.cat(batch_attn_blocks, dim=0))

        head_output = torch.cat(q_output_blocks, dim=2)
        head_attn_weights = torch.cat(q_attn_blocks, dim=2)
        h_output_blocks.append(head_output)
        h_attn_blocks.append(head_attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)

    return attn_output, attn_weights


def blocked_h_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    head_block_size: int,
    *,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    H-blocked attention that slices along head dimension to create blocks and processes each block.
    """
    batch_size, num_heads, q_len, _ = query.shape
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)

    key_states, value_states = _get_kv_states(module, key, value)

    h_output_blocks = []
    h_attn_blocks = []

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        # Extract head blocks
        q_g = query[:, h_start:h_end, :, :]
        k_g = key_states[:, h_start:h_end, :, :]
        v_g = value_states[:, h_start:h_end, :, :]

        attn_weights = torch.matmul(q_g, k_g.transpose(2, 3)) * scaling

        # position bias needed for mpt
        if position_bias is not None:
            attn_weights = attn_weights + position_bias[h_start:h_end, :, :]
        if attention_mask is not None:
            attn_weights = torch.where(attention_mask, masked_tensor, attn_weights)
        # attention sinks needed for gpt-oss
        if sinks is not None:
            sinks_g = (
                module.sinks[h_start:h_end]
                .reshape(1, -1, 1, 1)
                .expand(attn_weights.shape[0], -1, attn_weights.shape[2], -1)
            )
            combined_logits = torch.cat([attn_weights, sinks_g], dim=-1)
            attn_weights = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if sinks is not None:
            attn_weights = attn_weights[..., :-1]
        output_block = torch.matmul(attn_weights, v_g)

        h_output_blocks.append(output_block)
        h_attn_blocks.append(attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)

    return attn_output, attn_weights


def blocked_q_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_q_blocks: int,
    *,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Q-blocked attention that slices the query sequence into blocks and processes each block.
    """
    batch_size, num_heads, q_len, _ = query.shape
    num_q_blocks = max(1, _normalize_int(num_q_blocks))
    key_states, value_states = _get_kv_states(module, key, value)

    q_block_positions = [-(-i * q_len) // num_q_blocks for i in range(num_q_blocks)]
    q_output_blocks = []
    q_attn_blocks = []

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_positions[q_block_idx]
        if q_block_idx == num_q_blocks - 1:
            q_len_block = q_len - q_start
        else:
            q_len_block = q_block_positions[q_block_idx + 1] - q_start

        q_block = query[:, :, q_start : q_start + q_len_block, :]
        attn_mask_block = None
        if attention_mask is not None:
            attn_mask_block = attention_mask[:, :, q_start : q_start + q_len_block, :]

        attn_weights = torch.matmul(q_block, key_states.transpose(2, 3)) * scaling
        # position bias needed for mpt model
        if position_bias is not None:
            attn_weights = attn_weights + position_bias
        if attn_mask_block is not None:
            attn_weights = torch.where(attn_mask_block, masked_tensor, attn_weights)
        # attention sinks needed for gpt-oss
        if sinks is not None:
            sinks_g = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, q_len_block, -1)
            combined_logits = torch.cat([attn_weights, sinks_g], dim=3)
            attn_weights = combined_logits - combined_logits.max(dim=3, keepdim=True).values

        attn_weights = torch.softmax(attn_weights, dim=3, dtype=torch.float32).to(query.dtype)
        if sinks is not None:
            attn_weights = attn_weights[..., : key.shape[2]]
        output_block = torch.matmul(attn_weights, value_states)

        q_output_blocks.append(output_block)
        q_attn_blocks.append(attn_weights)

    attn_output = torch.cat(q_output_blocks, dim=2).transpose(1, 2).contiguous()
    attn_weights = torch.cat(q_attn_blocks, dim=2)

    return attn_output, attn_weights


def blocked_kv_mla_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    per_head_k_up_normal: torch.Tensor,
    per_head_v_up: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    compressed_kvs: Optional[torch.Tensor],
    mla_absorption: Dict[str, Any],
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize result tensor
    batch_size, num_heads, seq_len, _ = query.shape
    output = torch.zeros(
        batch_size, num_heads, seq_len, module.config.kv_lora_rank, device=query.device, dtype=query.dtype
    )

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = query.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    # Initialize Running Maximum and Denominator
    current_max = torch.full(
        (batch_size, num_heads, seq_len),
        float(MIN_MASKED_ATTENTION_VALUE),
        device=query.device,
        dtype=query.dtype,
    )
    skip_kv = True
    current_denominator = torch.zeros(batch_size, num_heads, seq_len, device=query.device, dtype=query.dtype)

    ctx_len = compressed_kvs.layers[layer_idx].ckv.shape[2]
    kv_block_size = -(-ctx_len // num_kv_blocks)

    position_ids = cache_kwargs.get("position_ids")
    current_position = position_ids.max(dim=-1).values

    for kv_block_idx in range(num_kv_blocks):
        start_index = kv_block_idx * kv_block_size
        if kv_block_idx == num_kv_blocks - 1:
            kv_len_block = ctx_len - start_index
        else:
            kv_len_block = kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        compressed_kv_block = compressed_kvs.read_only_blocked_ckv(start_index, end_index, layer_idx, cache_kwargs)
        k_pe_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, layer_idx, cache_kwargs)

        causal_mask_block = _create_causal_mask(
            position_ids=position_ids,
            target_length=end_index,
            start_index=start_index,
        )

        if mla_absorption is not None:
            absorption = mla_absorption.get("absorption", False)
        else:
            absorption = False

        k_heads, q_heads = compressed_kv_block.shape[1], query.shape[1]

        if k_heads > 1:
            num_heads_to_repeat = math.ceil(q_heads / k_heads)
            compressed_kv_block = (
                compressed_kv_block.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(batch_size, num_heads_to_repeat * k_heads, -1, module.config.kv_lora_rank)
            )
            compressed_kv_block = compressed_kv_block[:, :q_heads, :, :]

            k_pe_block = (
                k_pe_block.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(batch_size, num_heads_to_repeat * k_heads, -1, module.config.qk_rope_head_dim)
            )
            k_pe_block = k_pe_block[:, :q_heads, :, :]

        if absorption:
            krope_nope = torch.cat((compressed_kv_block, k_pe_block), dim=-1)
            attn_weights_block = torch.matmul(query, krope_nope.transpose(2, 3)) * scaling
            # [1, 64, q_len, 576] X [1, 1, 576, kv_block_size] -> [1, 64, q_len, kv_block_size]
            attn_weights_block = torch.where(causal_mask_block, masked_tensor, attn_weights_block)
            current_max, current_denominator, output = update_running_softmax(
                current_max,
                attn_weights_block,
                current_denominator,
                output,
                compressed_kv_block,
                skip_kv,
                skip_future,
            )  # [1, 64, q_len, kv_block_size] X [1, 1, kv_block_size, 512] -> [1, 64, q_len, 512]
        else:
            knope = torch.matmul(compressed_kv_block, per_head_k_up_normal)
            if k_heads == 1:
                k_pe_block = (
                    k_pe_block.unsqueeze(1)
                    .expand(-1, num_heads, -1, -1, -1)
                    .reshape(batch_size, num_heads, -1, module.config.qk_rope_head_dim)
                )
            krope_nope = torch.cat((knope, k_pe_block), dim=-1)
            attn_weights_block = torch.matmul(query, krope_nope.transpose(2, 3)) * scaling
            attn_weights_block = torch.where(causal_mask_block, masked_tensor, attn_weights_block)
            current_max, current_denominator, output = update_running_softmax(
                current_max,
                attn_weights_block,
                current_denominator,
                output,
                compressed_kv_block,
                skip_kv,
                skip_future,
            )

    attn_output = torch.matmul(output, per_head_v_up)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_weights = None

    return attn_output, attn_weights


def blocked_h_mla_attention_forward(
    module: nn.Module,
    q_a_proj_out: torch.Tensor,
    fusedqk: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kva: torch.Tensor,
    k_pe: torch.Tensor,
    per_head_q_up: torch.Tensor,
    per_head_k_up: torch.Tensor,
    per_head_v_up: torch.Tensor,
    per_head_k_up_normal: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    mla_absorption: Dict[str, Any],
    head_block_size: int,
    *,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    H-blocked attention that slices along head dimension to create blocks and processes each block.
    """
    batch_size, num_heads, q_len, _ = q_pe.shape
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = q_pe.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=q_pe.device)

    if mla_absorption is not None:
        absorption = mla_absorption.get("absorption", False)
        online = mla_absorption.get("online", False)
    else:
        absorption = False

    h_output_blocks = []
    h_attn_blocks = []
    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        if absorption:
            if online:
                qup_kupT = torch.matmul(per_head_q_up[:, h_start:h_end, :, :], per_head_k_up[:, h_start:h_end, :, :])
                dq_qup_kupT = torch.matmul(q_a_proj_out, qup_kupT)
            else:
                dq_qup_kupT = torch.matmul(q_a_proj_out, fusedqk[:, h_start:h_end, :, :])
            qkupTrope_nope = torch.cat((dq_qup_kupT, q_pe[:, h_start:h_end, :, :]), dim=-1)
            krope_nope = torch.cat((kva, k_pe), dim=-1)
            attn_weights = torch.matmul(qkupTrope_nope, krope_nope.transpose(2, 3)) * scaling
        else:
            knope = torch.matmul(kva, per_head_k_up_normal[:, h_start:h_end, :, :])
            krope_nope = torch.cat((knope, k_pe), dim=-1)
            qrope_nope = torch.cat((q_nope[:, h_start:h_end, :, :], q_pe[:, h_start:h_end, :, :]), dim=-1)
            attn_weights = torch.matmul(qrope_nope, krope_nope.transpose(2, 3)) * scaling

        if attention_mask is not None:
            attn_weights = torch.where(attention_mask, masked_tensor, attn_weights)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
        attn_output = torch.matmul(attn_weights, kva)
        attn_output = torch.matmul(attn_output, per_head_v_up[:, h_start:h_end, :, :])
        h_output_blocks.append(attn_output)
        h_attn_blocks.append(attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)
    return attn_output, attn_weights
