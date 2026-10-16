# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch

from QEfficient.customop.ctx_scatter_gather import (  # noqa: E402
    CtxChunkScatterBatch,
    CtxGather,
    CtxGather3D,
    CtxGatherBlockedKV,
    CtxGatherBlockedKVBatch,
    CtxGatherBlockedKVDP,
    CtxGatherBlockRangeKVDP,
    CtxGatherPagedAttention,
    CtxGatherPagedKVDP,
    CtxGatherPagedKVHeads,
    CtxPagedScatterDP,
    CtxPagedScatterPage,
    CtxScatter,
    CtxScatter3D,
    CtxScatter3DInt,
    CtxScatterPagedAttention,
    M3CtxScatter,
)
from QEfficient.customop.ctx_scatter_gather_cb import (  # noqa: E402
    CtxGatherBlockedKVCB,
    CtxGatherCB,
    CtxGatherCB3D,
    CtxScatterCB,
    CtxScatterCB3D,
)
from QEfficient.customop.onnxscript_utils import get_dynamo_onnxscript_func
from QEfficient.customop.rms_norm import CustomRMSNorm  # noqa: E402
from QEfficient.customop.sequence_chunk import CompileLengthSequenceChunk


@torch.library.custom_op("qefficient::rms_norm", mutates_args=())
def rms_norm_op(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Custom RMS Norm operation for QEfficient"""
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
    return weight * hidden_states


@rms_norm_op.register_fake
def _(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Fake implementation for torch.export - just returns tensor with same shape/dtype"""
    return torch.empty_like(hidden_states)


@torch.library.custom_op("qefficient::compile_length_sequence_chunk", mutates_args=())
def compile_length_sequence_chunk_op(
    tensor: torch.Tensor,
    dim: int,
    num_chunks: int,
    chunk_idx: int,
    compile_axis_size: int,
) -> torch.Tensor:
    del compile_axis_size
    return torch.chunk(tensor, num_chunks, dim=dim)[chunk_idx].clone()


@compile_length_sequence_chunk_op.register_fake
def _(
    tensor: torch.Tensor,
    dim: int,
    num_chunks: int,
    chunk_idx: int,
    compile_axis_size: int,
) -> torch.Tensor:
    normalized_dim = dim if dim >= 0 else tensor.dim() + dim
    output_shape = list(tensor.shape)
    output_shape[normalized_dim] = compile_axis_size // num_chunks
    return torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)


@torch.library.custom_op("qefficient::ctx_scatter", mutates_args=())
def ctx_scatter_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Custom context scatter operation"""
    result = data.clone()
    batch_idx = torch.arange(result.shape[0]).view(-1, 1, 1)
    head_idx = torch.arange(result.shape[1]).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    result[batch_idx, head_idx, ctx_idx] = updates
    return result


@ctx_scatter_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export - just returns data tensor with same shape/dtype"""
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_scatter_paged_attention", mutates_args=())
def ctx_scatter_paged_attention_op(
    data: torch.Tensor,
    block_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """Scatter updates into a paged-attention cache."""
    result = data.clone()
    block_index = block_index.view(-1, 1, 1)
    head_idx = torch.arange(result.shape[1], device=result.device).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    result[block_index, head_idx, ctx_idx] = updates
    return result


@ctx_scatter_paged_attention_op.register_fake
def _(
    data: torch.Tensor,
    block_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_scatter_3d", mutates_args=())
def ctx_scatter_3d_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Custom 3D context scatter operation"""
    # Clone the data to avoid aliasing issues with torch.library.custom_op
    result = data.clone()
    batch_idx = torch.arange(result.shape[0]).view(-1, 1)
    ctx_idx = position_ids
    result[batch_idx, ctx_idx] = updates
    return result


@ctx_scatter_3d_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export - just returns data tensor with same shape/dtype"""
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_chunk_scatter_batch", mutates_args=())
def ctx_chunk_scatter_batch_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Scatter batch-folded updates into a cache laid out as [1, B*H, T, D]."""
    batch_size, num_heads, seq_len, _ = updates.shape
    batch_idx = torch.arange(batch_size, device=data.device).view(batch_size, 1, 1)
    batch_idx = batch_idx.expand(batch_size, num_heads, seq_len)
    head_idx = torch.arange(num_heads, device=data.device).view(1, num_heads, 1)
    head_idx = head_idx.expand(batch_size, num_heads, seq_len)
    head_flat_idx = batch_idx * num_heads + head_idx
    position_idx = position_ids.long().unsqueeze(1).expand(batch_size, num_heads, seq_len)
    result = data.clone()
    result[0, head_flat_idx, position_idx] = updates
    return result


@ctx_chunk_scatter_batch_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_gather_3d", mutates_args=())
def ctx_gather_3d_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Custom 3D context gather operation"""
    batch_indices = torch.arange(data.shape[0]).view(-1, 1)
    return data[batch_indices, ctx_indices]


@ctx_gather_3d_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export"""
    batch_size = data.shape[0]
    seq_len = ctx_indices.shape[1]
    feature_shape = data.shape[2:]
    return torch.empty((batch_size, seq_len, *feature_shape), dtype=data.dtype, device=data.device)


@torch.library.custom_op("qefficient::ctx_gather", mutates_args=())
def ctx_gather_op(data: torch.Tensor, ctx_indices: torch.Tensor, comp_ctx_len: int) -> torch.Tensor:
    batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
    return data[batch_indices, head_indices, ctx_indices]


@ctx_gather_op.register_fake
def _(
    data: torch.Tensor,
    ctx_indices: torch.Tensor,
    comp_ctx_len: int,
) -> torch.Tensor:
    """
    Fake kernel for shape inference.

    Use ctx_indices when available (true PyTorch behavior),
    but fall back to comp_ctx_len if needed (ONNX contract).
    """

    # Prefer actual indexing size when known
    gather_dim = ctx_indices.shape[-1] if ctx_indices.shape[-1] is not None else comp_ctx_len
    out_shape = (
        data.shape[0],
        data.shape[1],
        gather_dim,
        *data.shape[3:],
    )

    return torch.empty(
        out_shape,
        dtype=data.dtype,
        device=data.device,
    )


# SCATTER CB (4D with heads, context, etc.)
@torch.library.custom_op("qefficient::ctx_scatter_cb", mutates_args=())
def ctx_scatter_cb_op(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """
    Custom 4D context scatter op with batch_index (CB version).
    Semantics: same as CtxScatterFuncCB.forward, but returns a new tensor.
    """
    # Clone to avoid aliasing issues with custom_op
    result = data.clone()
    batch_idx = batch_index.view(-1, 1, 1)
    head_idx = torch.arange(result.shape[1], device=result.device).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    result[batch_idx, head_idx, ctx_idx] = updates
    return result


@ctx_scatter_cb_op.register_fake
def _(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """
    Fake implementation for torch.export: correct shape/dtype/device, values don't matter.
    Output shape matches data.
    """
    return torch.empty_like(data)


# SCATTER CB 3D
@torch.library.custom_op("qefficient::ctx_scatter_cb_3d", mutates_args=())
def ctx_scatter_cb_3d_op(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """
    Custom 3D context scatter op with batch_index (CB3D version).
    Semantics: same as CtxScatterFuncCB3D.forward but returns new tensor.
    """
    result = data.clone()
    batch_idx = batch_index.view(-1, 1)
    ctx_idx = position_ids
    result[batch_idx, ctx_idx] = updates
    return result


@ctx_scatter_cb_3d_op.register_fake
def _(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """
    Fake implementation for torch.export: same shape/dtype/device as data.
    """
    return torch.empty_like(data)


# GATHER CB
@torch.library.custom_op("qefficient::ctx_gather_cb", mutates_args=())
def ctx_gather_cb_op(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    ctx_indices: torch.Tensor,
    comp_ctx_len: int,
) -> torch.Tensor:
    """
    Custom 4D context gather op with batch_index (CB version).
    Semantics: same as CtxGatherFuncCB.forward.
    """
    batch_indices = batch_index.view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
    return data[batch_indices, head_indices, ctx_indices]


@ctx_gather_cb_op.register_fake
def _(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    ctx_indices: torch.Tensor,
    comp_ctx_len: int,
) -> torch.Tensor:
    """
    Fake implementation for torch.export.

    We derive the output shape from input shapes:
    - batch_size: from batch_index
    - num_heads: from data
    - seq_len: from ctx_indices (dimension 1, typically)
    - hidden dims: from data (starting from dim 3)
    """
    batch_size = batch_index.shape[0]
    num_heads = data.shape[1]
    seq_len = ctx_indices.shape[-1]

    # Remaining feature dimensions (e.g., head_dim or more)
    feature_shape = data.shape[3:]  # could be () if 3D

    out_shape = (batch_size, num_heads, seq_len, *feature_shape)
    return torch.empty(out_shape, dtype=data.dtype, device=data.device)


# GATHER CB 3D
@torch.library.custom_op("qefficient::ctx_gather_cb_3d", mutates_args=())
def ctx_gather_cb_3d_op(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    ctx_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Custom 3D context gather op with batch_index (CB3D version).
    Semantics: similar to CtxGatherFuncCB3D.forward.
    """
    batch_indices = batch_index.view(-1, 1)
    return data[batch_indices, ctx_indices]


@ctx_gather_cb_3d_op.register_fake
def _(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    ctx_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Fake implementation for torch.export.

    Output shape:
    - batch_size: from batch_index
    - seq_len: from ctx_indices (dim 1)
    - any trailing dims from data
    """
    batch_size = batch_index.shape[0]
    seq_len = ctx_indices.shape[1]
    feature_shape = data.shape[2:]  # if data is [B, C], this is ()

    out_shape = (batch_size, seq_len, *feature_shape)
    return torch.empty(out_shape, dtype=data.dtype, device=data.device)


# GATHER BLOCKED KV (no batch_index)
@torch.library.custom_op("qefficient::ctx_gather_blocked_kv", mutates_args=())
def ctx_gather_blocked_kv_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Custom blocked-KV gather op (non-CB). Semantics: same as CtxGatherFuncBlockedKV.forward."""
    batch_indices = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
    ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
    return data[batch_indices, head_indices, ctx_indices]


@ctx_gather_blocked_kv_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    batch_size = data.shape[0]
    num_heads = data.shape[1]
    ctx_len = ctx_indices.shape[2]
    feature_shape = data.shape[3:]
    out_shape = (batch_size, num_heads, ctx_len, *feature_shape)
    return torch.empty(out_shape, dtype=data.dtype, device=data.device)


@torch.library.custom_op("qefficient::ctx_gather_blocked_kv_batch", mutates_args=())
def ctx_gather_blocked_kv_batch_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Gather from a batch-folded cache laid out as [1, B*H, T, D]."""
    num_heads = data.shape[1]
    ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
    head_idx = torch.arange(num_heads, device=data.device).view(num_heads, 1)
    return data[0, head_idx, ctx_indices[0]].unsqueeze(0)


@ctx_gather_blocked_kv_batch_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (data.shape[0], data.shape[1], ctx_indices.shape[-1], *data.shape[3:]),
        dtype=data.dtype,
        device=data.device,
    )


@torch.library.custom_op("qefficient::ctx_gather_paged_attention", mutates_args=())
def ctx_gather_paged_attention_op(
    data: torch.Tensor, block_indices: torch.Tensor, ctx_indices: torch.Tensor
) -> torch.Tensor:
    """Gather valid KV values from a paged-attention cache."""
    block_indices = block_indices.view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
    ctx_indices = ctx_indices.unsqueeze(1)
    return data[block_indices, head_indices, ctx_indices]


@ctx_gather_paged_attention_op.register_fake
def _(data: torch.Tensor, block_indices: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (block_indices.shape[0], data.shape[1], ctx_indices.shape[-1], *data.shape[3:]),
        dtype=data.dtype,
        device=data.device,
    )


# GATHER BLOCKED KV CB (with batch_index)
@torch.library.custom_op("qefficient::ctx_gather_blocked_kv_cb", mutates_args=())
def ctx_gather_blocked_kv_cb_op(
    data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor
) -> torch.Tensor:
    """Custom blocked-KV gather op with batch_index. Semantics: same as CtxGatherFuncBlockedKVCB.forward."""
    batch_indices = batch_index.view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
    ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
    return data[batch_indices, head_indices, ctx_indices]


@ctx_gather_blocked_kv_cb_op.register_fake
def _(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    batch_size = batch_index.shape[0]
    num_heads = data.shape[1]
    ctx_len = ctx_indices.shape[2]
    feature_shape = data.shape[3:]
    out_shape = (batch_size, num_heads, ctx_len, *feature_shape)
    return torch.empty(out_shape, dtype=data.dtype, device=data.device)


# SCATTER 3D INT (builds packed->original index table; outputs INT32)
@torch.library.custom_op("qefficient::ctx_scatter_3d_int", mutates_args=())
def ctx_scatter_3d_int_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Custom 3D INT32 scatter. Semantics: same as CtxScatterFunc3DInt.forward."""
    result = data.clone()
    valid = position_ids != torch.iinfo(torch.int32).max
    batch_idx = torch.arange(result.shape[0], device=result.device).view(-1, 1).expand_as(position_ids)
    result[batch_idx[valid], position_ids[valid].long()] = updates[valid]
    return result


@ctx_scatter_3d_int_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(data)


# GATHER 3D GENERALIZED (tolerates INT32_MAX indices)
@torch.library.custom_op("qefficient::ctx_gather_3d_generalized", mutates_args=())
def ctx_gather_3d_generalized_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Custom 3D gather with INT32_MAX guard. Semantics: same as CtxGatherFunc3DGeneralized.forward."""
    batch_indices = torch.arange(data.shape[0], device=data.device).view(-1, 1)
    ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
    return data[batch_indices, ctx_indices]


@ctx_gather_3d_generalized_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    batch_size = data.shape[0]
    seq_len = ctx_indices.shape[1]
    feature_shape = data.shape[2:]
    return torch.empty((batch_size, seq_len, *feature_shape), dtype=data.dtype, device=data.device)


# SCATTER 3D GENERALIZED (preserves data at INT32_MAX positions)
@torch.library.custom_op("qefficient::ctx_scatter_3d_generalized", mutates_args=())
def ctx_scatter_3d_generalized_op(
    data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor
) -> torch.Tensor:
    """Custom 3D scatter that leaves data untouched at invalid positions. Semantics: same as CtxScatterFunc3DGeneralized.forward."""
    result = data.clone()
    valid = position_ids != torch.iinfo(torch.int32).max
    batch_idx = torch.arange(result.shape[0], device=result.device).view(-1, 1).expand_as(position_ids)
    result[batch_idx[valid], position_ids[valid].long()] = updates[valid]
    return result


@ctx_scatter_3d_generalized_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(data)


# M3 INDEX-KEY SCATTER (CtxScatter with INT64 index cast — MiniMax-M3 indexer)
@torch.library.custom_op("qefficient::m3_ctx_scatter", mutates_args=())
def m3_ctx_scatter_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """MiniMax-M3 index-key cache scatter. Same eager semantics as ctx_scatter but the
    ONNX export uses M3CtxScatter which casts position_ids to INT64 before building
    ScatterND indices, avoiding the INT32/INT64 Concat type mismatch."""
    result = data.clone()
    batch_idx = torch.arange(result.shape[0]).view(-1, 1, 1)
    head_idx = torch.arange(result.shape[1]).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    result[batch_idx, head_idx, ctx_idx] = updates
    return result


@m3_ctx_scatter_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_paged_scatter_dp", mutates_args=())
def ctx_paged_scatter_dp_op(
    data: torch.Tensor,
    block_id: torch.Tensor,
    addr: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """Scatter updates into a DP/CP-layout paged cache."""
    result = data.clone()
    batch, rows, seq_len, _ = updates.shape
    row_idx = torch.arange(rows, device=data.device).view(1, rows, 1).expand(batch, rows, seq_len)
    valid = block_id != torch.iinfo(torch.int32).max
    result[block_id[valid].long(), row_idx[valid], addr[valid].long()] = updates[valid]
    return result


@ctx_paged_scatter_dp_op.register_fake
def _(data: torch.Tensor, block_id: torch.Tensor, addr: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_gather_paged_kv_dp", mutates_args=())
def ctx_gather_paged_kv_dp_op(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    """Assemble contiguous KV values from a DP/CP-layout physical page pool."""
    block_ids = torch.where(block_ids == torch.iinfo(torch.int32).max, 0, block_ids)
    num_pages, rows = block_ids.shape
    _, data_rows, page_size, head_dim = data.shape
    if rows != data_rows:
        raise ValueError("Paged GQA gather block-id rows must match pool rows.")
    row_idx = torch.arange(rows, device=data.device).view(1, rows)
    pages = data[block_ids.long(), row_idx]
    return pages.permute(1, 0, 2, 3).reshape(1, rows, num_pages * page_size, head_dim)


@ctx_gather_paged_kv_dp_op.register_fake
def _(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (1, data.shape[1], block_ids.shape[0] * data.shape[2], *data.shape[3:]),
        dtype=data.dtype,
        device=data.device,
    )


@torch.library.custom_op("qefficient::ctx_paged_scatter_page", mutates_args=())
def ctx_paged_scatter_page_op(data: torch.Tensor, block_id: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Scatter whole pages into a head-major physical page pool."""
    result = data.clone()
    valid = block_id != torch.iinfo(torch.int32).max
    heads = torch.arange(data.shape[1], device=data.device).view(1, -1).expand_as(block_id)
    result[block_id[valid].long(), heads[valid]] = updates[valid]
    return result


@ctx_paged_scatter_page_op.register_fake
def _(data: torch.Tensor, block_id: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_gather_paged_kv_heads", mutates_args=())
def ctx_gather_paged_kv_heads_op(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    """Assemble contiguous KV from a head-major physical page pool."""
    block_ids = torch.where(block_ids == torch.iinfo(torch.int32).max, 0, block_ids)
    num_pages, heads = block_ids.shape
    if heads != data.shape[1]:
        raise ValueError("Paged head gather IDs must match the cache head dimension.")
    head_idx = torch.arange(heads, device=data.device).view(1, heads)
    pages = data[block_ids.long(), head_idx]
    return pages.permute(1, 0, 2, 3).reshape(1, heads, num_pages * data.shape[2], data.shape[3])


@ctx_gather_paged_kv_heads_op.register_fake
def _(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (1, block_ids.shape[1], block_ids.shape[0] * data.shape[2], *data.shape[3:]),
        dtype=data.dtype,
        device=data.device,
    )


@torch.library.custom_op("qefficient::ctx_gather_blocked_kv_dp", mutates_args=())
def ctx_gather_blocked_kv_dp_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Gather a blocked KV range from a DP/CP-layout cache."""
    batch_indices = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
    row_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
    ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
    return data[batch_indices, row_indices, ctx_indices.long()]


@ctx_gather_blocked_kv_dp_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (*data.shape[:2], ctx_indices.shape[-1], *data.shape[3:]),
        dtype=data.dtype,
        device=data.device,
    )


@torch.library.custom_op("qefficient::ctx_gather_block_range_kv_dp", mutates_args=())
def ctx_gather_block_range_kv_dp_op(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    """Gather full KV blocks from a 5-D DP/CP-layout cache."""
    batch_indices = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
    row_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
    block_ids = torch.where(block_ids == torch.iinfo(torch.int32).max, 0, block_ids)
    return data[batch_indices, row_indices, block_ids.long()]


@ctx_gather_block_range_kv_dp_op.register_fake
def _(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (*data.shape[:2], block_ids.shape[-1], data.shape[3], data.shape[4]),
        dtype=data.dtype,
        device=data.device,
    )


# ---------------------------------------------------------------------------
# Translation table: torch.ops.qefficient.* → ONNX export classes.
# Used by _export_via_dynamo via custom_translation_table.
# ---------------------------------------------------------------------------

DYNAMO_CUSTOM_OP_TABLE = {
    torch.ops.qefficient.rms_norm.default: get_dynamo_onnxscript_func(CustomRMSNorm),
    torch.ops.qefficient.compile_length_sequence_chunk.default: get_dynamo_onnxscript_func(CompileLengthSequenceChunk),
    torch.ops.qefficient.ctx_scatter.default: get_dynamo_onnxscript_func(CtxScatter),
    torch.ops.qefficient.ctx_scatter_paged_attention.default: get_dynamo_onnxscript_func(CtxScatterPagedAttention),
    torch.ops.qefficient.ctx_scatter_3d.default: get_dynamo_onnxscript_func(CtxScatter3D),
    torch.ops.qefficient.ctx_chunk_scatter_batch.default: get_dynamo_onnxscript_func(CtxChunkScatterBatch),
    torch.ops.qefficient.ctx_scatter_cb.default: get_dynamo_onnxscript_func(CtxScatterCB),
    torch.ops.qefficient.ctx_scatter_cb_3d.default: get_dynamo_onnxscript_func(CtxScatterCB3D),
    torch.ops.qefficient.ctx_scatter_3d_int.default: get_dynamo_onnxscript_func(CtxScatter3DInt),
    torch.ops.qefficient.ctx_scatter_3d_generalized.default: get_dynamo_onnxscript_func(CtxScatter3D),
    torch.ops.qefficient.m3_ctx_scatter.default: get_dynamo_onnxscript_func(M3CtxScatter),
    torch.ops.qefficient.ctx_gather.default: get_dynamo_onnxscript_func(CtxGather),
    torch.ops.qefficient.ctx_gather_3d.default: get_dynamo_onnxscript_func(CtxGather3D),
    torch.ops.qefficient.ctx_gather_cb.default: get_dynamo_onnxscript_func(CtxGatherCB),
    torch.ops.qefficient.ctx_gather_cb_3d.default: get_dynamo_onnxscript_func(CtxGatherCB3D),
    torch.ops.qefficient.ctx_gather_blocked_kv.default: get_dynamo_onnxscript_func(CtxGatherBlockedKV),
    torch.ops.qefficient.ctx_gather_blocked_kv_batch.default: get_dynamo_onnxscript_func(CtxGatherBlockedKVBatch),
    torch.ops.qefficient.ctx_gather_paged_attention.default: get_dynamo_onnxscript_func(CtxGatherPagedAttention),
    torch.ops.qefficient.ctx_paged_scatter_dp.default: get_dynamo_onnxscript_func(CtxPagedScatterDP),
    torch.ops.qefficient.ctx_gather_paged_kv_dp.default: get_dynamo_onnxscript_func(CtxGatherPagedKVDP),
    torch.ops.qefficient.ctx_paged_scatter_page.default: get_dynamo_onnxscript_func(CtxPagedScatterPage),
    torch.ops.qefficient.ctx_gather_paged_kv_heads.default: get_dynamo_onnxscript_func(CtxGatherPagedKVHeads),
    torch.ops.qefficient.ctx_gather_blocked_kv_dp.default: get_dynamo_onnxscript_func(CtxGatherBlockedKVDP),
    torch.ops.qefficient.ctx_gather_block_range_kv_dp.default: get_dynamo_onnxscript_func(CtxGatherBlockRangeKVDP),
    torch.ops.qefficient.ctx_gather_blocked_kv_cb.default: get_dynamo_onnxscript_func(CtxGatherBlockedKVCB),
    torch.ops.qefficient.ctx_gather_3d_generalized.default: get_dynamo_onnxscript_func(CtxGather3D),
}
