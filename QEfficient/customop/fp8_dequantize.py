# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
TorchScript symbolic wrappers for the three FP8 DequantizeLinear granularities.

Used via ``select_interface`` in FP8DeQuantLinear and FP8BlockWiseDequantLinear
forward methods so that the legacy TorchScript export path emits standard ONNX
``DequantizeLinear`` nodes instead of ``Cast + Mul``.  Weights remain in FP8
(dtype=17) in the ONNX initializers.

Three granularities:

* ``FP8DequantizePerTensorFunc``  — scalar scale, DequantizeLinear (no axis).
* ``FP8DequantizePerAxisFunc``    — (out_features,) scale, DequantizeLinear(axis=0).
* ``FP8DequantizeBlockedFunc``    — compact (out//R, in//C) scale.
                                    Emits: Unsqueeze + Tile([1,R,1]) + Flatten +
                                    DequantizeLinear(axis=-1, block_size=C).
"""

import torch

from QEfficient.utils import constants

# ── TorchScript-path symbolic wrappers ───────────────────────────────────────


class FP8DequantizePerTensorFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_per_tensor."""

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return weight.to(scale.dtype) * scale

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, weight: torch.Value, scale: torch.Value) -> torch.Value:
        return g.op("DequantizeLinear", weight, scale)


class FP8DequantizePerAxisFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_per_axis."""

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if scale.ndim == 1:
            scale = scale.unsqueeze(-1)
        return weight.to(scale.dtype) * scale

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, weight: torch.Value, scale: torch.Value) -> torch.Value:
        return g.op("DequantizeLinear", weight, scale, axis_i=0)


class FP8DequantizeBlockedFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_blocked.

    Emits Unsqueeze(scale) + Tile([1, row_bs, 1]) + Flatten(axis=2) +
    DequantizeLinear(axis=-1, block_size=col_bs).
    """

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor, row_block_size: int, col_block_size: int) -> torch.Tensor:
        scale_row = scale.repeat_interleave(row_block_size, dim=0)
        scale_full = scale_row.repeat_interleave(col_block_size, dim=-1)
        return weight.to(scale_full.dtype) * scale_full

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        weight: torch.Value,
        scale: torch.Value,
        row_block_size,
        col_block_size,
    ) -> torch.Value:
        import torch.onnx.symbolic_helper as sym_help

        row_bs = sym_help._maybe_get_const(row_block_size, "i")
        col_bs = sym_help._maybe_get_const(col_block_size, "i")
        axes = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
        scale_unsq = g.op("Unsqueeze", scale, axes)
        repeats = g.op("Constant", value_t=torch.tensor([1, row_bs, 1], dtype=torch.int64))
        tiled = g.op("Tile", scale_unsq, repeats)
        scale_row = g.op("Flatten", tiled, axis_i=2)
        return g.op("DequantizeLinear", weight, scale_row, axis_i=-1, block_size_i=col_bs)


# ── Register symbolics for the legacy TorchScript export path ────────────────
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_per_tensor",
    FP8DequantizePerTensorFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_per_axis",
    FP8DequantizePerAxisFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_blocked",
    FP8DequantizeBlockedFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
