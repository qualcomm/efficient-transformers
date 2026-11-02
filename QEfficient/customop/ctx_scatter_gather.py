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
