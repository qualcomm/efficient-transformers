# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch

from QEfficient.customop.ctx_scatter_gather import (  # noqa: E402
    CtxGather,
    CtxGather3D,
    CtxGatherBlockedKV,
    CtxScatter,
    CtxScatter3D,
    CtxScatter3DInt,
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


@torch.library.custom_op("qefficient::ctx_gather_3d", mutates_args=())
def ctx_gather_3d_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Custom 3D context gather operation"""
    batch_indices = torch.arange(data.shape[0]).view(-1, 1)
    return data[batch_indices, ctx_indices]


@ctx_gather_3d_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export"""
    # Return tensor with shape [batch_size, seq_len]
    batch_size = data.shape[0]
    seq_len = ctx_indices.shape[1]
    return torch.empty(batch_size, seq_len, dtype=data.dtype, device=data.device)


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


# ---------------------------------------------------------------------------
# Translation table: torch.ops.qefficient.* → ONNX export classes.
# Used by _export_via_dynamo via custom_translation_table.
# ---------------------------------------------------------------------------

DYNAMO_CUSTOM_OP_TABLE = {
    torch.ops.qefficient.rms_norm.default: get_dynamo_onnxscript_func(CustomRMSNorm),
    torch.ops.qefficient.ctx_scatter.default: get_dynamo_onnxscript_func(CtxScatter),
    torch.ops.qefficient.ctx_scatter_3d.default: get_dynamo_onnxscript_func(CtxScatter3D),
    torch.ops.qefficient.ctx_scatter_cb.default: get_dynamo_onnxscript_func(CtxScatterCB),
    torch.ops.qefficient.ctx_scatter_cb_3d.default: get_dynamo_onnxscript_func(CtxScatterCB3D),
    torch.ops.qefficient.ctx_scatter_3d_int.default: get_dynamo_onnxscript_func(CtxScatter3DInt),
    torch.ops.qefficient.ctx_scatter_3d_generalized.default: get_dynamo_onnxscript_func(CtxScatter3D),
    torch.ops.qefficient.ctx_gather.default: get_dynamo_onnxscript_func(CtxGather),
    torch.ops.qefficient.ctx_gather_3d.default: get_dynamo_onnxscript_func(CtxGather3D),
    torch.ops.qefficient.ctx_gather_cb.default: get_dynamo_onnxscript_func(CtxGatherCB),
    torch.ops.qefficient.ctx_gather_cb_3d.default: get_dynamo_onnxscript_func(CtxGatherCB3D),
    torch.ops.qefficient.ctx_gather_blocked_kv.default: get_dynamo_onnxscript_func(CtxGatherBlockedKV),
    torch.ops.qefficient.ctx_gather_blocked_kv_cb.default: get_dynamo_onnxscript_func(CtxGatherBlockedKVCB),
    torch.ops.qefficient.ctx_gather_3d_generalized.default: get_dynamo_onnxscript_func(CtxGather3D),
}
