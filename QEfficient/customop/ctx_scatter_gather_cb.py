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
def CtxScatterCB(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [1, 3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFuncCB(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        # Avoid mutating graph inputs in-place during export.
        data = data.clone()
        batch_idx = batch_index.view(-1, 1, 1)
        head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_idx = position_ids.unsqueeze(1)
        data[batch_idx, head_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph, data: torch.Value, batch_index: torch.Value, position_ids: torch.Value, updates: torch.Value
    ) -> torch.Value:
        return g.onnxscript_op(CtxScatterCB, data, batch_index, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatterCB3D(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFuncCB3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        # Avoid mutating graph inputs in-place during export.
        data = data.clone()
        batch_idx = batch_index.view(-1, 1)
        ctx_idx = position_ids
        data[batch_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph, data: torch.Value, batch_index: torch.Value, position_ids: torch.Value, updates: torch.Value
    ) -> torch.Value:
        return g.onnxscript_op(CtxScatterCB3D, data, batch_index, position_ids, updates).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherCB(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, ctx_indices: onnxscript.INT32, comp_ctx_len: onnxscript.INT32
) -> onnxscript.FLOAT:
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    # Derive ctx_len from ctx_indices.shape[2] so the compiler sees a shape-derived
    # constant rather than a dynamic INT32 tensor input (comp_ctx_len).  The two are
    # always equal at every call site; using Shape avoids the "non-constant repeats"
    # error that the QAIC sub-function lowering raises for Expand.
    ctx_len = ops.Gather(ops.Shape(ctx_indices), [2])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(
        ops.Reshape(batch_size, [1]), ops.Reshape(num_heads, [1]), ops.Reshape(ctx_len, [1]), one, axis=0
    )

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(ctx_indices, [3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)

    return ops.GatherND(data, indices)


class CtxGatherFuncCB(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor, comp_ctx_len: int):
        batch_indices = batch_index.view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_indices = torch.where(ctx_indices >= data.shape[2], 0, ctx_indices)
        return data[batch_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph, data: torch.Value, batch_index: torch.Value, ctx_indices: torch.Value, comp_ctx_len: int
    ) -> torch.Value:
        if not isinstance(comp_ctx_len, torch.Value):
            comp_ctx_len = g.op("Constant", value_t=torch.tensor(comp_ctx_len, dtype=torch.int64))
        return g.onnxscript_op(CtxGatherCB, data, batch_index, ctx_indices, comp_ctx_len).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherBlockedKVCB(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, ctx_indices: onnxscript.INT32
) -> onnxscript.FLOAT:
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    ctx_len = ops.Gather(ops.Shape(ctx_indices), [2])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, ctx_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(ctx_indices, [3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)

    return ops.GatherND(data, indices)


class CtxGatherFuncBlockedKVCB(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = batch_index.view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, batch_index: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherBlockedKVCB, data, batch_index, ctx_indices).setTypeAs(data)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxGatherCB3D(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, ctx_indices: onnxscript.INT32
) -> onnxscript.FLOAT:
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    ctx_len = ops.Gather(ops.Shape(data), [1])

    # Expanded shape to create indices
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, ctx_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(ctx_indices, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)

    return ops.GatherND(data, indices)


class CtxGatherFuncCB3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = batch_index.view(-1, 1)
        return data[batch_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, batch_index: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherCB3D, data, batch_index, ctx_indices).setTypeAs(data)
