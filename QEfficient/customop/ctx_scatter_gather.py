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
@qeff_custom_op("com.qualcomm.cloud", 1)
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


@qeff_custom_op("com.qualcomm.cloud", 1)
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


@qeff_custom_op("com.qualcomm.cloud", 1)
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


# ---------------------------------------------------------------------------
# MiniMax-M3 indexer scatter
# Identical semantics to CtxScatter but casts position_ids to INT64 before
# building ScatterND indices so that all three index tensors (batch_idx,
# head_idx, ctx_idx) share the same INT64 element type.  Without this cast
# ctx_idx stays INT32 while batch_idx/head_idx are INT64 (derived from
# Shape → Gather), producing a mixed-type Concat that triggers the QAIC
# compiler assertion "sameSameShapeExceptDim: Different types".
# ---------------------------------------------------------------------------
@qeff_custom_op("com.qualcomm.cloud", 1)
def M3CtxScatter(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    batch_size = ops.Gather(ops.Shape(data), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(ops.Cast(position_ids, to=onnxscript.INT64.dtype), [1, 3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)
    return ops.ScatterND(data, indices, updates)


class M3CtxScatterFunc(torch.autograd.Function):
    """Scatter idx_k into the MiniMax-M3 index-key cache at position_ids."""

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        # Keep eager/export tracing semantics aligned with the MiniMax M3
        # reference implementation.  In particular, the cache must be cloned
        # before the indexed update and position_ids must be expanded across
        # the cache-head dimension.  The latter is important for prefill,
        # where updates have shape [B, H, Q, D].
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
        head_idx = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
        out = data.clone()
        out[batch_idx, head_idx, position_ids.long().unsqueeze(1)] = updates
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(M3CtxScatter, data, position_ids, updates).setTypeAs(data)


# ─────────────────────────────────────────────────────────────────────────────
# DP-layout ops for MiniMax M3 MSA indexer and attention (com.qti.aisw.onnx).
# These mirror the benchmark's DP-path custom ops and use the qti.aisw namespace
# so the QAIC compiler can pattern-match them to efficient on-chip implementations.
# ─────────────────────────────────────────────────────────────────────────────


def _symbolic_sizes(value: "torch.Value"):
    try:
        sizes = value.type().sizes()
    except Exception:
        return None
    return tuple(sizes) if sizes is not None else None


def _set_gather_output_type(output: "torch.Value", data: "torch.Value", indices: "torch.Value") -> "torch.Value":
    data_sizes = _symbolic_sizes(data)
    index_sizes = _symbolic_sizes(indices)
    if data_sizes is None or len(data_sizes) != 4:
        return output
    gathered_len = index_sizes[2] if index_sizes is not None and len(index_sizes) == 3 else None
    try:
        output.setType(data.type().with_sizes((data_sizes[0], data_sizes[1], gathered_len, data_sizes[3])))
    except Exception:
        pass
    return output


def _set_paged_gather_output_type(
    output: "torch.Value", data: "torch.Value", block_ids: "torch.Value"
) -> "torch.Value":
    data_sizes = _symbolic_sizes(data)
    id_sizes = _symbolic_sizes(block_ids)
    if data_sizes is None or len(data_sizes) != 4 or id_sizes is None or len(id_sizes) != 2:
        return output
    num_pages = id_sizes[0]
    page_size = data_sizes[2]
    token_len = num_pages * page_size if isinstance(num_pages, int) and isinstance(page_size, int) else None
    try:
        output.setType(data.type().with_sizes((1, data_sizes[1], token_len, data_sizes[3])))
    except Exception:
        pass
    return output


def _set_block_gather_output_type(
    output: "torch.Value", data: "torch.Value", block_ids: "torch.Value"
) -> "torch.Value":
    data_sizes = _symbolic_sizes(data)
    id_sizes = _symbolic_sizes(block_ids)
    if data_sizes is None or len(data_sizes) != 5 or id_sizes is None or len(id_sizes) != 3:
        return output
    try:
        output.setType(
            data.type().with_sizes((data_sizes[0], data_sizes[1], id_sizes[2], data_sizes[3], data_sizes[4]))
        )
    except Exception:
        pass
    return output


# @qeff_custom_op("com.qti.aisw.onnx", 1)
@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxPagedScatterDP(
    data: onnxscript.FLOAT,
    block_id: onnxscript.INT32,
    addr: onnxscript.INT32,
    updates: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    update_shape = ops.Shape(updates)
    batch_size = ops.Gather(update_shape, [0])
    num_rows = ops.Gather(update_shape, [1])
    seq_len = ops.Gather(update_shape, [2])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_rows, seq_len, one, axis=0)
    row_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_rows, one), [0, 2, 3]), exp_shape)
    indices = ops.Concat(
        ops.Unsqueeze(ops.Cast(block_id, to=7), [-1]),
        ops.Cast(row_idx, to=7),
        ops.Unsqueeze(ops.Cast(addr, to=7), [-1]),
        axis=3,
    )
    return ops.ScatterND(data, indices, updates)


class CtxPagedScatterFuncDP(torch.autograd.Function):
    """Paged scatter for GP-layout KV caches (block_id, addr addressing).

    data:    [physical_blocks, rows, slot_size, D]
    block_id: [B_local, rows, seq_len] – logical block (= batch-local index for non-paged)
    addr:    [B_local, rows, seq_len] – position within the block
    updates: [B_local, rows, seq_len, D]
    """

    @staticmethod
    def forward(data: torch.Tensor, block_id: torch.Tensor, addr: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        batch, rows, seq_len, _ = updates.shape
        row_idx = torch.arange(rows, device=data.device).view(1, rows, 1).expand(batch, rows, seq_len)
        out = data.clone()
        valid = block_id != torch.iinfo(torch.int32).max
        out[block_id[valid].long(), row_idx[valid], addr[valid].long()] = updates[valid]
        return out

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph, data: torch.Value, block_id: torch.Value, addr: torch.Value, updates: torch.Value
    ) -> torch.Value:
        return g.onnxscript_op(CtxPagedScatterDP, data, block_id, addr, updates).setTypeAs(data)


# @qeff_custom_op("com.qti.aisw.onnx", 1)
@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxPagedScatterPage(
    data: onnxscript.FLOAT, block_id: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    """Scatter complete pages into a physical page pool.

    ``data`` is ``[physical_pages, heads, page_size, dim]``;
    ``block_id`` is ``[pages, heads]`` and ``updates`` is
    ``[pages, heads, page_size, dim]``.
    """
    update_shape = ops.Shape(updates)
    num_pages = ops.Gather(update_shape, [0])
    num_heads = ops.Gather(update_shape, [1])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    head_idx = ops.Expand(
        ops.Unsqueeze(ops.Range(zero, num_heads, one), [0]),
        ops.Concat(num_pages, num_heads, axis=0),
    )
    indices = ops.Concat(
        ops.Unsqueeze(ops.Cast(block_id, to=7), [-1]),
        ops.Unsqueeze(ops.Cast(head_idx, to=7), [-1]),
        axis=2,
    )
    return ops.ScatterND(data, indices, updates)


class CtxPagedScatterFuncPage(torch.autograd.Function):
    """Whole-page scatter matching the MSA microbenchmark ABI."""

    @staticmethod
    def forward(data: torch.Tensor, block_id: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        out = data.clone()
        valid = block_id != torch.iinfo(torch.int32).max
        page_idx = block_id.long()
        heads = torch.arange(data.shape[1], device=data.device).view(1, -1).expand_as(block_id)
        out[page_idx[valid], heads[valid]] = updates[valid]
        return out

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, block_id: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxPagedScatterPage, data, block_id, updates).setTypeAs(data)


# @qeff_custom_op("com.qti.aisw.onnx", 1)
@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherBlockedKVDP(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    return ops.GatherND(data, ops.Unsqueeze(ctx_indices, [-1]), batch_dims=2)


class CtxGatherFuncBlockedKVDP(torch.autograd.Function):
    """Blocked KV gather for GP-layout caches with per-row (DP-aware) index validity.

    data:        [B_local, rows, ctx_len, D]
    ctx_indices: [B_local, rows, block_len]  – INT32; INT32_MAX → invalid (reads index 0)
    Returns:     [B_local, rows, block_len, D]
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
        row_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, row_indices, ctx_indices.long()]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        output = g.onnxscript_op(CtxGatherBlockedKVDP, data, ctx_indices)
        return _set_gather_output_type(output, data, ctx_indices)


# @qeff_custom_op("com.qti.aisw.onnx", 1)
@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherPagedKVDP(data: onnxscript.FLOAT, block_ids: onnxscript.INT32) -> onnxscript.FLOAT:
    # data: [physical_blocks, rows, page_size, D]
    # block_ids: [num_pages, rows]
    data_shape = ops.Shape(data)
    ids_shape = ops.Shape(block_ids)
    num_rows = ops.Gather(data_shape, [1])
    page_size = ops.Gather(data_shape, [2])
    head_dim = ops.Gather(data_shape, [3])
    num_pages = ops.Gather(ids_shape, [0])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    row_idx = ops.Expand(
        ops.Unsqueeze(ops.Range(zero, num_rows, one), [0]),
        ops.Concat(num_pages, num_rows, axis=0),
    )
    indices = ops.Concat(
        ops.Unsqueeze(ops.Cast(block_ids, to=7), [-1]),
        ops.Unsqueeze(ops.Cast(row_idx, to=7), [-1]),
        axis=2,
    )
    pages = ops.GatherND(data, indices, batch_dims=0)
    pages = ops.Transpose(pages, perm=[1, 0, 2, 3])
    return ops.Reshape(
        pages,
        ops.Concat(one, num_rows, ops.Mul(num_pages, page_size), head_dim, axis=0),
    )


class CtxGatherFuncPagedKVDP(torch.autograd.Function):
    """Paged gather that assembles contiguous KV from a physical page pool.

    data:      [physical_blocks, rows, page_size, D]
    block_ids: [num_pages, rows]  – physical block IDs; INT32_MAX → reads page 0
    Returns:   [1, rows, num_pages * page_size, D]
    """

    @staticmethod
    def forward(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        block_ids = torch.where(block_ids == torch.iinfo(torch.int32).max, torch.zeros_like(block_ids), block_ids)
        num_pages, rows = block_ids.shape
        _, data_rows, page_size, head_dim = data.shape
        if rows != data_rows:
            raise ValueError("Paged GQA gather block-id rows must match pool rows.")
        row_idx = torch.arange(rows, device=data.device).view(1, rows)
        pages = data[block_ids.long(), row_idx]
        return pages.permute(1, 0, 2, 3).reshape(1, rows, num_pages * page_size, head_dim)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, block_ids: torch.Value) -> torch.Value:
        output = g.onnxscript_op(CtxGatherPagedKVDP, data, block_ids)
        return _set_paged_gather_output_type(output, data, block_ids)


# @qeff_custom_op("com.qti.aisw.onnx", 1)
@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherPagedKVHeads(data: onnxscript.FLOAT, block_ids: onnxscript.INT32) -> onnxscript.FLOAT:
    """Gather pages for head-major IDs: ``[pages, heads] -> [1, heads, tokens, D]``."""
    data_shape = ops.Shape(data)
    ids_shape = ops.Shape(block_ids)
    num_heads = ops.Gather(ids_shape, [1])
    page_size = ops.Gather(data_shape, [2])
    head_dim = ops.Gather(data_shape, [3])
    num_pages = ops.Gather(ids_shape, [0])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0]), ids_shape)
    indices = ops.Concat(
        ops.Unsqueeze(ops.Cast(block_ids, to=7), [-1]),
        ops.Unsqueeze(ops.Cast(head_idx, to=7), [-1]),
        axis=2,
    )
    pages = ops.GatherND(data, indices, batch_dims=0)
    pages = ops.Transpose(pages, perm=[1, 0, 2, 3])
    return ops.Reshape(pages, ops.Concat(one, num_heads, ops.Mul(num_pages, page_size), head_dim, axis=0))


class CtxGatherFuncPagedKVHeads(torch.autograd.Function):
    """Head-aware whole-page gather matching the MSA microbenchmark."""

    @staticmethod
    def forward(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        block_ids = torch.where(block_ids == torch.iinfo(torch.int32).max, torch.zeros_like(block_ids), block_ids)
        num_pages, heads = block_ids.shape
        if heads != data.shape[1]:
            raise ValueError("Paged head gather IDs must match the cache head dimension.")
        head_idx = torch.arange(heads, device=data.device).view(1, heads)
        pages = data[block_ids.long(), head_idx]
        return pages.permute(1, 0, 2, 3).reshape(1, heads, num_pages * data.shape[2], data.shape[3])

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, block_ids: torch.Value) -> torch.Value:
        output = g.onnxscript_op(CtxGatherPagedKVHeads, data, block_ids)
        return _set_paged_gather_output_type(output, data, block_ids)


# @qeff_custom_op("com.qti.aisw.onnx", 1)
@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherBlockRangeKVDP(data: onnxscript.FLOAT, block_ids: onnxscript.INT32) -> onnxscript.FLOAT:
    # data: [B_local, rows, cache_blocks, block_size, D]
    # ids:  [B_local, rows, selected_blocks]
    return ops.GatherND(data, ops.Unsqueeze(block_ids, [-1]), batch_dims=2)


class CtxGatherFuncBlockRangeKVDP(torch.autograd.Function):
    """Block-range gather for 5-D GP caches (cache laid out as blocks).

    data:      [B_local, rows, cache_blocks, block_size, D]
    block_ids: [B_local, rows, selected_blocks]  – INT32_MAX → reads block 0
    Returns:   [B_local, rows, selected_blocks, block_size, D]
    """

    @staticmethod
    def forward(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
        row_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
        block_ids = torch.where(block_ids == torch.iinfo(torch.int32).max, 0, block_ids)
        return data[batch_indices, row_indices, block_ids.long()]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, block_ids: torch.Value) -> torch.Value:
        output = g.onnxscript_op(CtxGatherBlockRangeKVDP, data, block_ids)
        return _set_block_gather_output_type(output, data, block_ids)
