# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
onnxscript translation functions and TorchScript symbolics for the three FP8
DequantizeLinear granularities.

Used as entries in ``custom_translation_table`` for ``torch.onnx.export(..., dynamo=True)``
so the dynamo exporter emits standard ONNX ``DequantizeLinear`` nodes instead of
``Cast + Mul``.  Weights remain in FP8 (dtype=17) in the ONNX initializers.

Scale dtype (bf16 / fp16 / fp32) is propagated automatically from the concrete
dtype of the ``scale`` input tensor — no separate function per dtype is needed.

Three granularities:

* ``FP8DequantizePerTensor``      — scalar scale, no axis/block_size.
* ``FP8DequantizePerAxis``        — (out_features,) scale, axis=0.
* ``FP8DequantizeBlocked_R_C``    — compact (out//R, in//C) scale.
                                    Emits: Tile(scale, [R, 1]) then
                                    DequantizeLinear(axis=-1, block_size=C).
                                    Concrete functions are provided for the
                                    block sizes used by real FP8 models.
                                    get_blocked_fn(row_bs, col_bs) returns the
                                    right function or raises for unsupported sizes.

Note on opsets:
  - Per-tensor and per-axis DequantizeLinear are valid from opset 13 onward, so
    they are compiled for both ONNX_LEGACY_EXPORT_OPSET (17) and
    ONNX_DYNAMO_EXPORT_OPSET (21) via the qeff_custom_op decorator.
  - Blocked DequantizeLinear uses the ``block_size`` attribute introduced in
    opset 21, so the blocked functions are compiled for opset 21 only and are
    only used on the dynamo export path.
"""

import onnxscript
import torch

from QEfficient.customop.onnxscript_utils import qeff_custom_op
from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_LEGACY_EXPORT_OPSET))
ops21 = getattr(onnxscript, "opset" + str(constants.ONNX_DYNAMO_EXPORT_OPSET))


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
    The row_block_size and col_block_size are passed as int args from the custom op.
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
        axes = g.op(
            "Constant",
            value_t=torch.tensor([1], dtype=torch.int64),
        )
        scale_unsq = g.op("Unsqueeze", scale, axes)
        repeats = g.op(
            "Constant",
            value_t=torch.tensor([1, row_bs, 1], dtype=torch.int64),
        )
        tiled = g.op("Tile", scale_unsq, repeats)
        scale_row = g.op("Flatten", tiled, axis_i=2)
        return g.op(
            "DequantizeLinear",
            weight,
            scale_row,
            axis_i=-1,
            block_size_i=col_bs,
        )


# ── onnxscript translation functions ─────────────────────────────────────────
# Per-tensor and per-axis: compiled for both opset-17 (legacy) and opset-21
# (dynamo) via the qeff_custom_op decorator.


@qeff_custom_op("", constants.ONNX_LEGACY_EXPORT_OPSET)
def FP8DequantizePerTensor(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Per-tensor: DequantizeLinear(weight, scale) — scalar scale, no axis."""
    return ops.DequantizeLinear(weight, scale)


@qeff_custom_op("", constants.ONNX_LEGACY_EXPORT_OPSET)
def FP8DequantizePerAxis(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Per-axis: DequantizeLinear(weight, scale, axis=0). scale: (out_features,)."""
    return ops.DequantizeLinear(weight, scale, axis=0)


# ── Concrete blocked onnxscript functions (opset-21 only) ────────────────────
# DequantizeLinear with block_size was introduced in opset 21.  These functions
# are compiled directly against opset 21 and are only used on the dynamo path.
# onnxscript cannot use function int parameters as Constant node values, so we
# write one concrete function per (row_bs, col_bs) pair used by real FP8 models.
# get_blocked_fn(row_bs, col_bs) returns the right function for
# custom_translation_table.


@onnxscript.script(onnxscript.values.Opset(domain="", version=constants.ONNX_DYNAMO_EXPORT_OPSET))
def FP8DequantizeBlocked_128_128(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Blocked [128, 128]: Tile(scale,[128,1]) + DQL(axis=-1, block_size=128)."""
    repeats = ops21.Constant(value_ints=[128, 1])
    scale_row = ops21.Tile(scale, repeats)
    return ops21.DequantizeLinear(weight, scale_row, axis=-1, block_size=128)


@onnxscript.script(onnxscript.values.Opset(domain="", version=constants.ONNX_DYNAMO_EXPORT_OPSET))
def FP8DequantizeBlocked_64_64(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Blocked [64, 64]: Tile(scale,[64,1]) + DQL(axis=-1, block_size=64)."""
    repeats = ops21.Constant(value_ints=[64, 1])
    scale_row = ops21.Tile(scale, repeats)
    return ops21.DequantizeLinear(weight, scale_row, axis=-1, block_size=64)


@onnxscript.script(onnxscript.values.Opset(domain="", version=constants.ONNX_DYNAMO_EXPORT_OPSET))
def FP8DequantizeBlocked_32_32(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Blocked [32, 32]: Tile(scale,[32,1]) + DQL(axis=-1, block_size=32)."""
    repeats = ops21.Constant(value_ints=[32, 1])
    scale_row = ops21.Tile(scale, repeats)
    return ops21.DequantizeLinear(weight, scale_row, axis=-1, block_size=32)


_BLOCKED_FN_MAP = {
    (128, 128): FP8DequantizeBlocked_128_128,
    (64, 64): FP8DequantizeBlocked_64_64,
    (32, 32): FP8DequantizeBlocked_32_32,
}


def get_blocked_fn(row_block_size: int, col_block_size: int):
    """
    Return the onnxscript translation function for the given block sizes.
    Raises ValueError for unsupported sizes — add a new concrete function above.
    """
    fn = _BLOCKED_FN_MAP.get((row_block_size, col_block_size))
    if fn is None:
        raise ValueError(
            f"No onnxscript translation for FP8 blocked dequantize with "
            f"block_size=({row_block_size}, {col_block_size}). "
            f"Supported: {sorted(_BLOCKED_FN_MAP.keys())}. "
            f"Add a concrete FP8DequantizeBlocked_{row_block_size}_{col_block_size} "
            f"function to QEfficient/customop/fp8_dequantize.py."
        )
    return fn


# ── Eager TorchScript symbolic registration (legacy export path) ──────────────
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
