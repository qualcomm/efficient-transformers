# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch
from torch.onnx.symbolic_helper import parse_args

from QEfficient.customop.onnxscript_utils import qeff_custom_op
from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_LEGACY_EXPORT_OPSET))


@qeff_custom_op("com.qualcomm.cloud", 1)
def CompileLengthSequenceChunk(
    tensor: onnxscript.FLOAT,
    dim: onnxscript.INT64,
    num_chunks: onnxscript.INT64,
    chunk_idx: onnxscript.INT64,
    compile_axis_size: onnxscript.INT64,
) -> onnxscript.FLOAT:
    start = ops.Div(ops.Mul(compile_axis_size, chunk_idx), num_chunks)
    end = ops.Div(ops.Mul(compile_axis_size, ops.Add(chunk_idx, 1)), num_chunks)
    starts = ops.Reshape(start, [1])
    ends = ops.Reshape(end, [1])
    axes = ops.Reshape(dim, [1])
    steps = ops.Constant(value_ints=[1])
    return ops.Slice(tensor, starts, ends, axes, steps)


class CompileLengthSequenceChunkFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        dim: int,
        num_chunks: int,
        chunk_idx: int,
        compile_axis_size: int,
    ) -> torch.Tensor:
        del compile_axis_size
        return torch.chunk(tensor, num_chunks, dim=dim)[chunk_idx]

    @staticmethod
    @parse_args("v", "i", "i", "i", "i")
    def symbolic(g, tensor, dim: int, num_chunks: int, chunk_idx: int, compile_axis_size: int):
        start = compile_axis_size * chunk_idx // num_chunks
        end = compile_axis_size * (chunk_idx + 1) // num_chunks
        starts = g.op("Constant", value_t=torch.tensor([start], dtype=torch.long))
        ends = g.op("Constant", value_t=torch.tensor([end], dtype=torch.long))
        axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
        steps = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        return g.op("Slice", tensor, starts, ends, axes, steps)
