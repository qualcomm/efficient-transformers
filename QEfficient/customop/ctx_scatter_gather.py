# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch

from QEfficient.customop.onnxscript_utils import qeff_custom_op
from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_LEGACY_EXPORT_OPSET))


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatter(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [1, 3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFunc(torch.autograd.Function):
    """
    Function to scatter the current key values into KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        batch_idx = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_idx = position_ids.unsqueeze(1)
        data[batch_idx, head_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter, data, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatterPagedAttention(
    data: onnxscript.FLOAT, block_index: onnxscript.INT32, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    # Find dims
    num_blocks = ops.Gather(ops.Shape(block_index), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(num_blocks, num_heads, seq_len, one, axis=0)

    # Create indices
    block_idx = ops.Expand(ops.Unsqueeze(block_index, [3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [1, 3]), exp_shape)
    indices = ops.Concat(block_idx, head_idx, ctx_idx, axis=3)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFuncPagedAttention(torch.autograd.Function):
    """
    Function to scatter the current key values into KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, block_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        if not torch.onnx.is_in_onnx_export():
            data = data.clone()
        block_index = block_index.view(-1, 1, 1)
        head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_idx = position_ids.unsqueeze(1)
        data[block_index, head_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph, data: torch.Value, block_index: torch.Value, position_ids: torch.Value, updates: torch.Value
    ) -> torch.Value:
        return g.onnxscript_op(CtxScatterPagedAttention, data, block_index, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatter3D(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2]), exp_shape)

    # keep index tensor types aligned for backend that require exact dtype match
    batch_idx = ops.Cast(batch_idx, to=onnxscript.INT32.dtype)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFunc3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        data = data.clone()
        batch_idx = torch.arange(data.shape[0]).view(-1, 1)
        ctx_idx = torch.where(position_ids == torch.iinfo(torch.int32).max, data.shape[1] - 1, position_ids)
        data[batch_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter3D, data, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatterDP(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    position_ids = ops.Cast(position_ids, to=onnxscript.INT64.dtype)
    batch_local = ops.Gather(ops.Shape(data), [0])
    data_parallel = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [2])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_local, data_parallel, seq_len, one, axis=0)
    batch_idx = ops.Cast(
        ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_local, one), [1, 2, 3]), exp_shape),
        to=onnxscript.INT64.dtype,
    )
    dp_idx = ops.Cast(
        ops.Expand(ops.Unsqueeze(ops.Range(zero, data_parallel, one), [0, 2, 3]), exp_shape),
        to=onnxscript.INT64.dtype,
    )
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [3]), exp_shape)
    indices = ops.Concat(batch_idx, dp_idx, ctx_idx, axis=3)
    return ops.ScatterND(data, indices, updates)


class CtxScatterDPFunc(torch.autograd.Function):
    """Scatter into a ``[batch_local, data_parallel, context, ...]`` cache."""

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        result = data.clone()
        batch_idx = torch.arange(result.shape[0], device=result.device).view(-1, 1, 1)
        dp_idx = torch.arange(result.shape[1], device=result.device).view(1, -1, 1)
        safe_positions = torch.where(position_ids == torch.iinfo(torch.int32).max, 0, position_ids)
        result[batch_idx, dp_idx, safe_positions.long()] = updates
        return result

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatterDP, data, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatterFoldedRows(
    data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    """Scatter into a batch-folded cache shaped ``[1, batch, context, dim]``."""
    batch_size = ops.Gather(ops.Shape(updates), [0])
    query_length = ops.Gather(ops.Shape(updates), [2])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(one, batch_size, query_length, one, axis=0)
    row = ops.Cast(
        ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [0, 2, 3]), exp_shape),
        to=onnxscript.INT64.dtype,
    )
    batch_zero = ops.Mul(row, zero)
    context = ops.Expand(ops.Unsqueeze(ops.Cast(position_ids, to=onnxscript.INT64.dtype), [0, 3]), exp_shape)
    indices = ops.Concat(batch_zero, row, context, axis=3)
    return ops.ScatterND(data, indices, ops.Transpose(updates, perm=[1, 0, 2, 3]))


class CtxScatterFoldedRowsFunc(torch.autograd.Function):
    """Scatter ``[batch, 1, query, dim]`` updates into a folded local-KV cache."""

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        if data.ndim != 4 or data.shape[0] != 1 or updates.ndim != 4 or updates.shape[1] != 1:
            raise ValueError(
                "Folded-row scatter expects data [1, batch, context, dim] and updates [batch, 1, query, dim]."
            )
        if data.shape[1] != updates.shape[0]:
            raise ValueError("Folded-row scatter cache batch dimension does not match updates.")
        result = data.clone()
        batch_idx = torch.arange(data.shape[1], device=data.device).view(-1, 1).expand_as(position_ids)
        valid = position_ids != torch.iinfo(torch.int32).max
        result[0, batch_idx[valid], position_ids[valid].long()] = updates[:, 0][valid]
        return result

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatterFoldedRows, data, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherDP(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    ctx_indices = ops.Cast(ctx_indices, to=onnxscript.INT64.dtype)
    return ops.GatherND(data, ops.Unsqueeze(ctx_indices, [-1]), batch_dims=2)


class CtxGatherDPFunc(torch.autograd.Function):
    """Gather from a ``[batch_local, data_parallel, context, ...]`` cache."""

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
        dp_idx = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
        safe_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_idx, dp_idx, safe_indices.long()]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherDP, data, ctx_indices).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherFoldedRows(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """Gather from a batch-folded cache ``[1, batch, context, dim]``."""
    indices = ops.Unsqueeze(ops.Unsqueeze(ops.Cast(ctx_indices, to=onnxscript.INT64.dtype), [0]), [-1])
    return ops.GatherND(data, indices, batch_dims=2)


class CtxGatherFoldedRowsFunc(torch.autograd.Function):
    """Gather ``[batch, query]`` rows from a folded local-KV cache."""

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        if data.ndim != 4 or data.shape[0] != 1:
            raise ValueError("Folded-row gather expects data [1, batch, context, dim].")
        if ctx_indices.shape[0] != data.shape[1]:
            raise ValueError("Folded-row gather cache batch dimension does not match indices.")
        safe_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        batch_idx = torch.arange(data.shape[1], device=data.device).view(-1, 1)
        return data[0, batch_idx, safe_indices.long()].unsqueeze(0)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherFoldedRows, data, ctx_indices).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatterDPCP(
    data: onnxscript.FLOAT,
    position_ids: onnxscript.INT32,
    updates: onnxscript.FLOAT,
    context_parallel: onnxscript.INT32,
) -> onnxscript.FLOAT:
    position_ids = ops.Cast(position_ids, to=onnxscript.INT64.dtype)
    context_parallel = ops.Cast(context_parallel, to=onnxscript.INT64.dtype)
    batch_local = ops.Gather(ops.Shape(data), [0])
    data_parallel = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [2])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_local, data_parallel, seq_len, one, axis=0)
    batch_idx = ops.Cast(
        ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_local, one), [1, 2, 3]), exp_shape),
        to=onnxscript.INT64.dtype,
    )
    dp_idx = ops.Cast(
        ops.Expand(ops.Unsqueeze(ops.Range(zero, data_parallel, one), [0, 2, 3]), exp_shape),
        to=onnxscript.INT64.dtype,
    )
    context_slots = ops.Gather(ops.Shape(data), [2])
    slots_per_cp = ops.Div(context_slots, context_parallel)
    slot_idx = ops.Expand(
        ops.Unsqueeze(
            ops.Add(
                ops.Mul(ops.Mod(position_ids, context_parallel), slots_per_cp),
                ops.Div(position_ids, context_parallel),
            ),
            [3],
        ),
        exp_shape,
    )
    indices = ops.Concat(batch_idx, dp_idx, slot_idx, axis=3)
    return ops.ScatterND(data, indices, updates)


class CtxScatterDPCPFunc(torch.autograd.Function):
    """Scatter global slots into a DP-major cache with CP-interleaved physical slots."""

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor, context_parallel: int):
        result = data.clone()
        batch_idx = torch.arange(result.shape[0], device=result.device).view(-1, 1, 1)
        dp_idx = torch.arange(result.shape[1], device=result.device).view(1, -1, 1)
        valid = position_ids != torch.iinfo(torch.int32).max
        safe_positions = torch.where(valid, position_ids, torch.zeros_like(position_ids))
        slots_per_cp = result.shape[2] // context_parallel
        address = (
            torch.remainder(safe_positions, context_parallel) * slots_per_cp
            + torch.div(safe_positions, context_parallel, rounding_mode="floor")
        ).long()
        current = result[batch_idx, dp_idx, address]
        result[batch_idx, dp_idx, address] = torch.where(valid.unsqueeze(-1), updates, current)
        return result

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        data: torch.Value,
        position_ids: torch.Value,
        updates: torch.Value,
        context_parallel: int,
    ) -> torch.Value:
        context_parallel = g.op(
            "Constant",
            value_t=torch.tensor(context_parallel, dtype=torch.int32),
        )
        return g.onnxscript_op(CtxScatterDPCP, data, position_ids, updates, context_parallel).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherDPCP(
    data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32, context_parallel: onnxscript.INT32
) -> onnxscript.FLOAT:
    ctx_indices = ops.Cast(ctx_indices, to=onnxscript.INT64.dtype)
    context_parallel = ops.Cast(context_parallel, to=onnxscript.INT64.dtype)
    context_slots = ops.Gather(ops.Shape(data), [2])
    slots_per_cp = ops.Div(context_slots, context_parallel)
    slot_idx = ops.Add(
        ops.Mul(ops.Mod(ctx_indices, context_parallel), slots_per_cp),
        ops.Div(ctx_indices, context_parallel),
    )
    return ops.GatherND(data, ops.Unsqueeze(slot_idx, [-1]), batch_dims=2)


class CtxGatherDPCPFunc(torch.autograd.Function):
    """Gather global slots from a DP-major cache with CP-interleaved physical slots."""

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor, context_parallel: int):
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
        dp_idx = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
        safe_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        slots_per_cp = data.shape[2] // context_parallel
        address = (
            torch.remainder(safe_indices, context_parallel) * slots_per_cp
            + torch.div(safe_indices, context_parallel, rounding_mode="floor")
        ).long()
        return data[batch_idx, dp_idx, address]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value, context_parallel: int) -> torch.Value:
        context_parallel = g.op(
            "Constant",
            value_t=torch.tensor(context_parallel, dtype=torch.int32),
        )
        return g.onnxscript_op(CtxGatherDPCP, data, ctx_indices, context_parallel).setTypeAs(data)


class CtxScatterFunc3DGeneralized(torch.autograd.Function):
    """Scatter variant that preserves ``data`` at invalid (INT32_MAX) positions.

    Unlike :class:`CtxScatterFunc3D`, which writes updates for invalid rows to
    ``data.shape[1]-1`` (potentially clobbering valid content), this version
    masks out invalid rows before scattering so ``data`` is left untouched where
    ``position_ids == INT32_MAX``.
    """

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        data = data.clone()
        valid = position_ids != torch.iinfo(torch.int32).max
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1).expand_as(position_ids)
        data[batch_idx[valid], position_ids[valid].long()] = updates[valid]
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter3D, data, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatter3DInt(
    data: onnxscript.INT32, position_ids: onnxscript.INT32, updates: onnxscript.INT32
) -> onnxscript.INT32:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2]), exp_shape)
    batch_idx = ops.Cast(batch_idx, to=onnxscript.INT32.dtype)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFunc3DInt(torch.autograd.Function):
    """Int32-typed scatter used to build a packed->original index table."""

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        data = data.clone()
        valid = position_ids != torch.iinfo(torch.int32).max
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1).expand_as(position_ids)
        data[batch_idx[valid], position_ids[valid].long()] = updates[valid]
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter3DInt, data, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGather3D(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    batch_size = ops.Slice(ops.Shape(data), starts=[0], ends=[1], axes=[0])
    idx_seq_len = ops.Slice(ops.Shape(ctx_indices), starts=[1], ends=[2], axes=[0])
    expand_shape = ops.Concat(batch_size, idx_seq_len, axis=0)
    ctx_indices = ops.Expand(ctx_indices, expand_shape)
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=1)


class CtxGatherFunc3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0], device=data.device).view(-1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGather3D, data, ctx_indices).setTypeAs(data)


class CtxGatherFunc3DGeneralized(torch.autograd.Function):
    """Gather variant that tolerates INT32_MAX indices (invalid rows read from 0).

    Semantically equivalent to :class:`CtxGatherFunc3D` on the PyTorch side but
    exposed as a separate autograd op so callers using the packed/cumsum scatter
    pipeline can be easily recognized and so the ONNX symbolic omits
    ``setTypeAs`` (needed when the caller already has a matching dtype on
    ``data`` and wants the op signature to flow through without dtype pinning).
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGather3D, data, ctx_indices)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGather(
    data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32, comp_ctx_len: onnxscript.INT64
) -> onnxscript.FLOAT:
    # Create a shape tensor based on comp_ctx_len
    shape_tensor = ops.Concat(ops.Shape(data)[:2], ops.Reshape(comp_ctx_len, [1]), axis=0)

    # Directly use the shape tensor without validation
    ctx_indices = ops.Expand(ctx_indices, shape_tensor)
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=2)


class CtxGatherFunc(torch.autograd.Function):
    """
    Function to gather only the valid key values from KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor, comp_ctx_len: int):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value, comp_ctx_len: int) -> torch.Value:
        if not isinstance(comp_ctx_len, torch.Value):
            comp_ctx_len = g.op("Constant", value_t=torch.tensor(comp_ctx_len, dtype=torch.int64))
        return g.onnxscript_op(CtxGather, data, ctx_indices, comp_ctx_len).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherBlockedKV(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=2)


class CtxGatherFuncBlockedKV(torch.autograd.Function):
    """
    Function to gather only the valid key values from KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherBlockedKV, data, ctx_indices).setTypeAs(data)


# ─────────────────────────────────────────────────────────────────────────────
# Batch-folded variants — cache laid out as [1, B*NKVH, T, D] (BH static at
# compile time) so batch and KV-head are pre-flattened onto one axis that
# matches a B*Hkv physical core/device layout 1:1.
# ─────────────────────────────────────────────────────────────────────────────
@onnxscript.script(onnxscript.values.Opset("com.qti.aisw.onnx", 1))
def CtxChunkScatterBatch(
    data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    # Batch version: data [1, BH, T, D], updates [B, NKVH, QL, D], position_ids [B, QL]
    # (BH = B*NKVH, static at compile time).
    #
    # Fold updates/indices onto the BH axis so data, indices and updates all share
    # the same leading [1, BH, ...] axes. The scatter coords stay [0, head_flat, pos]
    # but head_flat now runs 0..BH-1 in lockstep with the BH axis (index i on the
    # BH axis carries head_flat==i), so the compiler can prove the BH axis splits
    # across devices and each device scatters only its own [1, BH/N, ...] slice —
    # instead of mapping the whole KV$ VA range per layer.
    batch_size = ops.Gather(ops.Shape(updates), [0])
    num_heads = ops.Gather(ops.Shape(updates), [1])
    seq_len = ops.Gather(ops.Shape(updates), [2])
    head_dim = ops.Gather(ops.Shape(updates), [3])

    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    bh = ops.Mul(batch_size, num_heads)  # BH = B*NKVH

    # updates [B, NKVH, QL, D] -> [1, BH, QL, D]
    updates_folded = ops.Reshape(updates, ops.Concat(one, bh, seq_len, head_dim, axis=0))

    # head_flat = 0..BH-1 along the BH axis: [1, BH, QL, 1]
    head_flat = ops.Range(zero, bh, one)  # [BH]
    head_flat_exp = ops.Expand(
        ops.Unsqueeze(head_flat, [0, 2, 3]), ops.Concat(one, bh, seq_len, one, axis=0)
    )  # [1, BH, QL, 1]

    # position_ids [B, QL] -> [1, BH, QL, 1] (each batch's pos tiled over its NKVH heads)
    pos_i64 = ops.Cast(position_ids, to=7)  # [B, QL]
    pos_tiled = ops.Expand(
        ops.Unsqueeze(pos_i64, [1]), ops.Concat(batch_size, num_heads, seq_len, axis=0)
    )  # [B, NKVH, QL]
    pos_exp = ops.Reshape(pos_tiled, ops.Concat(one, bh, seq_len, one, axis=0))  # [1, BH, QL, 1]

    # coords [0, head_flat, pos] -> indices [1, BH, QL, 3]
    batch_zero = ops.Mul(head_flat_exp, zero)  # [1, BH, QL, 1] of int64 zeros
    indices = ops.Concat(batch_zero, head_flat_exp, pos_exp, axis=3)  # [1, BH, QL, 3]

    return ops.ScatterND(data, indices, updates_folded)


class CtxChunkScatterBatchFunc(torch.autograd.Function):
    """Batch version: data [1, BH, T, D], updates [B, NKVH, QL, D], position_ids [B, QL].
    BH = B*NKVH static at compile time. Folds updates/indices onto the BH axis so the
    BH axis stays split-able across devices (mirrors CtxGatherFuncBlockedKVBatch).
    head_flat = b*NKVH + h matches the reshape([B,NKVH,...] -> [1, B*NKVH, ...])
    convention used everywhere else.
    """

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        B, NKVH, QL, D = updates.shape
        pos = position_ids.long()
        batch_idx = torch.arange(B, device=data.device).view(B, 1, 1).expand(B, NKVH, QL)
        head_idx = torch.arange(NKVH, device=data.device).view(1, NKVH, 1).expand(B, NKVH, QL)
        head_flat_idx = batch_idx * NKVH + head_idx
        p_idx = pos.unsqueeze(1).expand(B, NKVH, QL)
        out = data.clone()
        out[0, head_flat_idx, p_idx] = updates
        return out

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        data: torch.Value,
        position_ids: torch.Value,
        updates: torch.Value,
    ) -> torch.Value:
        return g.onnxscript_op(CtxChunkScatterBatch, data, position_ids, updates).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qti.aisw.onnx", 1))
def CtxGatherBlockedKVBatch(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    # data [1, BH, T, D], ctx_indices [1, BH, T_block]  (BH = B*NKVH, static at compile time)
    # batch_dims=2: checks data.shape[0]==indices.shape[0] (1==1) and
    #               data.shape[1]==indices.shape[1] (BH==BH) — both static
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])  # [1, BH, T_block, 1]
    return ops.GatherND(data, ctx_indices, batch_dims=2)


class CtxGatherFuncBlockedKVBatch(torch.autograd.Function):
    """Batch version: data [1, BH, T, D], ctx_indices [1, BH, T_block].
    BH = B*NKVH is static (compile-time fixed). Returns [1, BH, T_block, D].
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        # data [1, BH, T, D], ctx_indices [1, BH, T_block]
        BH = data.shape[1]
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        head_idx = torch.arange(BH, device=data.device).view(BH, 1)  # [BH, 1]
        # data[0, head_idx, ctx_indices[0]]: [BH, T_block, D] -> unsqueeze -> [1, BH, T_block, D]
        return data[0, head_idx, ctx_indices[0]].unsqueeze(0)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherBlockedKVBatch, data, ctx_indices).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGatherPagedAttention(
    data: onnxscript.FLOAT, block_indices: onnxscript.INT32, ctx_indices: onnxscript.INT32
) -> onnxscript.FLOAT:
    num_kv_blocks = ops.Gather(ops.Shape(block_indices), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(ctx_indices), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(num_kv_blocks, num_heads, seq_len, one, axis=0)

    # Create indices
    block_idx = ops.Expand(ops.Unsqueeze(block_indices, [1, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(ctx_indices, [1, 3]), exp_shape)
    indices = ops.Concat(block_idx, head_idx, ctx_idx, axis=3)

    return ops.GatherND(data, indices)


class CtxGatherFuncPagedAttention(torch.autograd.Function):
    """
    Function to gather only the valid key values from KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, block_indices: torch.Tensor, ctx_indices: torch.Tensor):
        block_indices = block_indices.view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_indices = ctx_indices.unsqueeze(1)
        return data[block_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph, data: torch.Value, block_indices: torch.Value, ctx_indices: torch.Value
    ) -> torch.Value:
        return g.onnxscript_op(CtxGatherPagedAttention, data, block_indices, ctx_indices).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def V4CtxScatter1D(
    data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    """Scatter [B, S, D] updates into a [B, T, D] CSA prefill cache."""
    batch_size = ops.Gather(ops.Shape(data), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2]), exp_shape)
    ctx_idx = ops.Cast(ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape), to=onnxscript.INT64.dtype)
    return ops.ScatterND(data, ops.Concat(batch_idx, ctx_idx, axis=2), updates)


class V4CtxScatter1DFunc(torch.autograd.Function):
    """CSA prefill retained-state scatter matching the benchmark ABI."""

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        result = data.clone()
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1)
        result[batch_idx, position_ids.to(torch.long)] = updates
        return result

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(V4CtxScatter1D, data, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGather1D(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """Gather CSA prefill cache rows from a [B, T, D] tensor."""
    return ops.GatherND(data, ops.Unsqueeze(ctx_indices, [-1]), batch_dims=1)


class CtxGather1DFunc(torch.autograd.Function):
    """CSA prefill retained-state gather matching the benchmark ABI."""

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        safe_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1)
        return data[batch_idx, safe_indices.to(torch.long)]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGather1D, data, ctx_indices).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CompressedAttn(
    kv_cache: onnxscript.FLOAT,
    topk_indices: onnxscript.INT32,
    valid: onnxscript.BOOL,
    query: onnxscript.FLOAT,
    scale: onnxscript.FLOAT,
    masked_logit: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """CSA prefill fused gather, QK, softmax, and PV benchmark function."""
    batch_size = ops.Gather(ops.Shape(kv_cache), [0])
    sequence_length = ops.Gather(ops.Shape(topk_indices), [1])
    top_k = ops.Gather(ops.Shape(topk_indices), [2])
    head_dim = ops.Gather(ops.Shape(query), [3])
    gathered = ops.GatherND(
        kv_cache,
        ops.Unsqueeze(ops.Reshape(topk_indices, ops.Concat(batch_size, ops.Mul(sequence_length, top_k), axis=0)), [-1]),
        batch_dims=1,
    )
    gathered = ops.Reshape(gathered, ops.Concat(batch_size, sequence_length, top_k, head_dim, axis=0))
    gathered = ops.Where(ops.Unsqueeze(valid, [-1]), gathered, ops.Constant(value_float=0.0))
    scores = ops.Mul(ops.Einsum(query, gathered, equation="bshd,bskd->bshk"), scale)
    scores = ops.Where(ops.Unsqueeze(valid, [2]), scores, masked_logit)
    probabilities = ops.Softmax(scores, axis=-1)
    probabilities = ops.Where(ops.Unsqueeze(valid, [2]), probabilities, ops.Constant(value_float=0.0))
    return ops.Einsum(probabilities, gathered, equation="bshk,bskd->bshd")


class CompressedAttnFunc(torch.autograd.Function):
    """CSA prefill fused compressed-cache attention matching the benchmark graph."""

    @staticmethod
    def forward(
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        valid: torch.Tensor,
        query: torch.Tensor,
        scale: torch.Tensor,
        masked_logit: torch.Tensor,
    ):
        batch_size, sequence_length, _, _ = query.shape
        top_k = topk_indices.shape[-1]
        batch_idx = torch.arange(batch_size, device=kv_cache.device).view(-1, 1)
        gathered = kv_cache[batch_idx, topk_indices.reshape(batch_size, -1).to(torch.long)].reshape(
            batch_size, sequence_length, top_k, kv_cache.shape[-1]
        )
        gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        scores = torch.matmul(query.float(), gathered.transpose(-1, -2).float()) * scale
        scores = torch.where(valid.unsqueeze(2), scores, masked_logit)
        probabilities = torch.where(valid.unsqueeze(2), torch.softmax(scores, dim=-1), torch.zeros_like(scores))
        return torch.matmul(probabilities.to(gathered.dtype), gathered)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        kv_cache: torch.Value,
        topk_indices: torch.Value,
        valid: torch.Value,
        query: torch.Value,
        scale: torch.Value,
        masked_logit: torch.Value,
    ) -> torch.Value:
        return g.onnxscript_op(CompressedAttn, kv_cache, topk_indices, valid, query, scale, masked_logit).setTypeAs(
            query
        )


@qeff_custom_op("com.qualcomm.cloud", 1)
def CSAJointAttention(
    local_kv: onnxscript.FLOAT,
    local_valid: onnxscript.BOOL,
    compressed_kv: onnxscript.FLOAT,
    compressed_indices: onnxscript.INT32,
    compressed_valid: onnxscript.BOOL,
    query: onnxscript.FLOAT,
    sinks: onnxscript.FLOAT,
    scale: onnxscript.FLOAT,
    masked_logit: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Sequence-major fused cache gather and joint local/sink/compressed attention."""
    batch_size = ops.Gather(ops.Shape(compressed_kv), [0])
    query_length = ops.Gather(ops.Shape(compressed_indices), [1])
    top_k = ops.Gather(ops.Shape(compressed_indices), [2])
    head_dim = ops.Gather(ops.Shape(compressed_kv), [2])
    flat_indices = ops.Reshape(compressed_indices, ops.Concat(batch_size, ops.Mul(query_length, top_k), axis=0))
    gathered_flat = ops.GatherND(compressed_kv, ops.Unsqueeze(flat_indices, [-1]), batch_dims=1)
    compressed_kv = ops.Reshape(
        gathered_flat, ops.Concat(batch_size, query_length, top_k, head_dim, axis=0)
    )
    num_heads = ops.Gather(ops.Shape(query), [2])
    sink_shape = ops.Concat(batch_size, query_length, num_heads, ops.Constant(value_ints=[1]), axis=0)
    sink_logits = ops.Expand(ops.Unsqueeze(sinks, [0, 1, 3]), sink_shape)

    local_scores = ops.Mul(ops.Einsum(query, local_kv, equation="bshd,btd->bsht"), scale)
    local_scores = ops.Where(ops.Unsqueeze(local_valid, [2]), local_scores, masked_logit)
    compressed_scores = ops.Mul(ops.Einsum(query, compressed_kv, equation="bshd,bskd->bshk"), scale)
    compressed_valid_heads = ops.Unsqueeze(compressed_valid, [2])
    compressed_scores = ops.Where(compressed_valid_heads, compressed_scores, masked_logit)
    local_length = ops.Gather(ops.Shape(local_kv), [1])
    local_value_shape = ops.Concat(batch_size, query_length, local_length, head_dim, axis=0)
    local_values = ops.Expand(ops.Unsqueeze(local_kv, [1]), local_value_shape)
    sink_value_shape = ops.Concat(batch_size, query_length, ops.Constant(value_ints=[1]), head_dim, axis=0)
    sink_values = ops.Expand(ops.Constant(value_float=0.0), sink_value_shape)
    all_scores = ops.Concat(local_scores, sink_logits, compressed_scores, axis=3)
    all_values = ops.Concat(local_values, sink_values, compressed_kv, axis=2)
    return ops.Einsum(ops.Softmax(all_scores, axis=3), all_values, equation="bsht,bstd->bshd")


class CSAJointAttentionFunc(torch.autograd.Function):
    """CSA joint attention with a compiler-stable ONNX custom-op boundary."""

    @staticmethod
    def forward(
        local_kv, local_valid, compressed_cache, compressed_indices, compressed_valid, query, sinks, scale, masked_logit
    ):
        batch_size, query_length, top_k = compressed_indices.shape
        batch_indices = torch.arange(batch_size, device=compressed_cache.device).view(-1, 1, 1)
        compressed_kv = compressed_cache[
            batch_indices, compressed_indices.to(torch.long)
        ].reshape(batch_size, query_length, top_k, compressed_cache.shape[-1])
        local_scores = torch.einsum("bshd,btd->bsht", query.float(), local_kv.float()) * scale
        local_scores = torch.where(local_valid.unsqueeze(2), local_scores, masked_logit)
        sink_logits = sinks.float().view(1, 1, -1, 1).expand(query.shape[0], query.shape[1], -1, -1)
        compressed_scores = torch.einsum("bshd,bskd->bshk", query.float(), compressed_kv.float()) * scale
        compressed_scores = torch.where(compressed_valid.unsqueeze(2), compressed_scores, masked_logit)
        all_scores = torch.cat((local_scores, sink_logits, compressed_scores), dim=3)
        local_values = local_kv.float().unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        sink_values = torch.zeros(
            query.shape[0], query.shape[1], 1, local_kv.shape[-1], dtype=torch.float32, device=query.device
        )
        all_values = torch.cat((local_values, sink_values, compressed_kv.float()), dim=2)
        probabilities = torch.softmax(all_scores.float(), dim=3).to(local_kv.dtype)
        output = torch.einsum("bsht,bstd->bshd", probabilities.float(), all_values.float())
        return output.to(local_kv.dtype)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g,
        local_kv,
        local_valid,
        compressed_kv,
        compressed_indices,
        compressed_valid,
        query,
        sinks,
        scale,
        masked_logit,
    ):
        return g.onnxscript_op(
            CSAJointAttention,
            local_kv,
            local_valid,
            compressed_kv,
            compressed_indices,
            compressed_valid,
            query,
            sinks,
            scale,
            masked_logit,
        ).setTypeAs(query)
