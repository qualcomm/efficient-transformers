# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Custom cache operators required by folded GLM DSA layouts."""

import onnxscript
import torch

from QEfficient.customop.onnxscript_utils import qeff_custom_op

ops = getattr(onnxscript, "opset17")


@qeff_custom_op("com.qti.aisw.onnx", 1)
def GlmFoldedRowGather(data: onnxscript.FLOAT, indices: onnxscript.INT32) -> onnxscript.FLOAT:
    return ops.GatherND(data, ops.Unsqueeze(ops.Cast(indices, to=7), [-1]), batch_dims=2)


@qeff_custom_op("com.qti.aisw.onnx", 1)
def V4CtxGatherRows(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """Benchmark-compatible direct gather from [B, R, T, D] cache rows."""
    return ops.GatherND(data, ops.Unsqueeze(ctx_indices, [-1]), batch_dims=2)


# Dynamo's custom_translation_table inlines an ONNXScript function's body.
# Keep the translation as an ONNXScript function in the standard domain whose
# body calls the QTI custom op, so the exported graph retains the vendor op node
# and CustomOpTransform can attach its function prototype afterward.
_qti_ops = onnxscript.values.Opset("com.qti.aisw.onnx", 1)


@onnxscript.script()
def V4CtxGatherRowsDynamoTranslation(
    data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32
) -> onnxscript.FLOAT:
    return _qti_ops.V4CtxGatherRows(data, ctx_indices)


@qeff_custom_op("com.qti.aisw.onnx", 1)
def GlmPagedScatter(
    data: onnxscript.FLOAT,
    block_id: onnxscript.INT32,
    address: onnxscript.INT32,
    updates: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    shape = ops.Shape(updates)
    batch = ops.Gather(shape, [0])
    rows = ops.Gather(shape, [1])
    seq_len = ops.Gather(shape, [2])
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    expanded_shape = ops.Concat(batch, rows, seq_len, one, axis=0)
    row = ops.Expand(ops.Unsqueeze(ops.Range(zero, rows, one), [0, 2, 3]), expanded_shape)
    scatter_indices = ops.Concat(
        ops.Unsqueeze(ops.Cast(block_id, to=7), [-1]),
        ops.Cast(row, to=7),
        ops.Unsqueeze(ops.Cast(address, to=7), [-1]),
        axis=3,
    )
    return ops.ScatterND(data, scatter_indices, updates)


@qeff_custom_op("com.qti.aisw.onnx", 1)
def GlmSparseScatter(data: onnxscript.FLOAT, indices: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    return ops.ScatterND(data, indices, updates)


@qeff_custom_op("com.qti.aisw.onnx", 1)
def GlmIntDiv(values: onnxscript.INT32, divisor: int) -> onnxscript.INT32:
    return ops.Div(values, divisor)


@qeff_custom_op("com.qti.aisw.onnx", 1)
def GlmIntMod(values: onnxscript.INT32, divisor: int) -> onnxscript.INT32:
    return ops.Mod(values, divisor)


class GlmFoldedRowGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        safe = torch.where(indices == torch.iinfo(torch.int32).max, 0, indices).long()
        return data.gather(2, safe.unsqueeze(-1).expand(*safe.shape, data.shape[-1]))

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g, data, indices):
        return g.onnxscript_op(GlmFoldedRowGather, data, indices).setTypeAs(data)


class V4CtxGatherRowsFunc(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
        if ctx_indices.ndim < 3:
            raise ValueError("ctx_indices must have shape [B, rows, ...].")
        batch_indices = torch.arange(data.shape[0], device=data.device).view(
            data.shape[0], *(1,) * (ctx_indices.ndim - 1)
        )
        row_indices = torch.arange(data.shape[1], device=data.device).view(
            1, data.shape[1], *(1,) * (ctx_indices.ndim - 2)
        )
        safe_indices = torch.where(
            ctx_indices == torch.iinfo(torch.int32).max,
            torch.zeros_like(ctx_indices),
            ctx_indices,
        )
        return data[batch_indices, row_indices, safe_indices.long()]

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g, data, ctx_indices):
        data_sizes = data.type().sizes()
        index_sizes = ctx_indices.type().sizes()
        output_type = data.type()
        if data_sizes is not None and index_sizes is not None and data_sizes:
            # The gather preserves every index dimension and appends the cache
            # row's feature dimension: [..., indices..., D].  Reusing data's
            # type here incorrectly describes the cache [B, R, T, D] as the
            # gathered result (which is rank-dependent on ctx_indices).
            output_type = data.type().with_sizes((*index_sizes, data_sizes[-1]))
        return g.onnxscript_op(V4CtxGatherRows, data, ctx_indices).setType(output_type)


class GlmPagedScatterFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        data: torch.Tensor, block_id: torch.Tensor, address: torch.Tensor, updates: torch.Tensor
    ) -> torch.Tensor:
        result = data.clone()
        batch, rows, seq_len = updates.shape[:3]
        row = torch.arange(rows, device=data.device).view(1, rows, 1).expand(batch, rows, seq_len)
        valid = block_id != torch.iinfo(torch.int32).max
        result[block_id[valid].long(), row[valid], address[valid].long()] = updates[valid]
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g, data, block_id, address, updates):
        return g.onnxscript_op(GlmPagedScatter, data, block_id, address, updates).setTypeAs(data)


class GlmSparseScatterFunc(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        result = data.clone()
        result[indices[..., 0].long(), indices[..., 1].long(), indices[..., 2].long()] = updates
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g, data, indices, updates):
        return g.op("ScatterND", data, indices, updates).setTypeAs(data)


class GlmIntDivFunc(torch.autograd.Function):
    @staticmethod
    def forward(values: torch.Tensor, divisor: int) -> torch.Tensor:
        return torch.div(values, divisor, rounding_mode="floor")

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g, values, divisor):
        divisor = g.op("Constant", value_t=torch.tensor(divisor, dtype=torch.int32))
        return g.op("Div", values, divisor).setTypeAs(values)


class GlmIntModFunc(torch.autograd.Function):
    @staticmethod
    def forward(values: torch.Tensor, divisor: int) -> torch.Tensor:
        return torch.remainder(values, divisor)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g, values, divisor):
        divisor = g.op("Constant", value_t=torch.tensor(divisor, dtype=torch.int32))
        return g.op("Mod", values, divisor).setTypeAs(values)


def glm_folded_row_gather(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        return torch.ops.qefficient.glm_folded_row_gather(data, indices)
    return GlmFoldedRowGatherFunc.apply(data, indices)


def v4_ctx_gather_rows(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    # In Dynamo export, use the registered custom op so the translation table
    # can emit the QTI node even inside nested subfunction capture. Calling
    # torch.onnx.ops.symbolic directly is not traceable in invoke_subgraph.
    if torch._dynamo.is_compiling():
        return torch.ops.qefficient.v4_ctx_gather_rows(data, ctx_indices)
    if torch.compiler.is_exporting():
        output_shape = (*ctx_indices.shape, data.shape[-1])
        return torch.onnx.ops.symbolic(
            "com.qti.aisw.onnx::V4CtxGatherRows",
            (data, ctx_indices),
            {},
            dtype=data.dtype,
            shape=output_shape,
            version=1,
        )
    return V4CtxGatherRowsFunc.apply(data, ctx_indices)


def glm_paged_scatter(data, block_id, address, updates):
    if torch._dynamo.is_compiling():
        return torch.ops.qefficient.glm_paged_scatter(data, block_id, address, updates)
    return GlmPagedScatterFunc.apply(data, block_id, address, updates)


def glm_sparse_scatter(data, indices, updates):
    if torch._dynamo.is_compiling():
        return torch.ops.qefficient.glm_sparse_scatter(data, indices, updates)
    return GlmSparseScatterFunc.apply(data, indices, updates)


def glm_int_div(values: torch.Tensor, divisor: int) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        return torch.ops.qefficient.glm_int_div(values, divisor)
    return GlmIntDivFunc.apply(values, divisor)


def glm_int_mod(values: torch.Tensor, divisor: int) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        return torch.ops.qefficient.glm_int_mod(values, divisor)
    return GlmIntModFunc.apply(values, divisor)
