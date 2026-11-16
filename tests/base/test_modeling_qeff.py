# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import onnx
import pytest
import torch

from QEfficient.base.modeling_qeff import (
    QEFFBaseModel,
    _build_blocked_translation_table,
    _validate_legacy_fp8_export,
)
from QEfficient.exporter.onnx_exporter import export_via_dynamo
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import FP8BlockWiseDequantLinear


def test_compiler_invalid_file(tmp_path):
    qeff_obj = SimpleNamespace()

    invalid_file = tmp_path / "invalid.onnx"
    with open(invalid_file, "wb") as fp:
        fp.write(chr(0).encode() * 100)

    with pytest.raises(RuntimeError):
        QEFFBaseModel._compile(qeff_obj, invalid_file, tmp_path)


def test_compiler_invalid_flag(tmp_path):
    qeff_obj = SimpleNamespace()

    onnx_model = onnx.parser.parse_model("""
    <
        ir_version: 8,
        opset_import: ["": 17]
    >
    test_compiler(float x) => (float y)
    {
        y = Identity(x)
    }
    """)
    valid_file = tmp_path / "valid.onnx"
    onnx.save(onnx_model, valid_file)

    with pytest.raises(RuntimeError):
        QEFFBaseModel._compile(qeff_obj, valid_file, tmp_path, convert_tofp16=True, aic_binary_dir=tmp_path)


def test_dynamo_export_saves_standard_onnx(tmp_path, mocker):
    onnx_program = mocker.MagicMock()
    mocker.patch("QEfficient.exporter.onnx_exporter.torch.onnx.export", return_value=onnx_program)
    mocker.patch("QEfficient.exporter.onnx_exporter.PruneFakeInitializersTransform.apply")

    onnx_path = tmp_path / "model.onnx"
    result = export_via_dynamo(
        SimpleNamespace(model=torch.nn.Identity().eval()),
        onnx_path,
        {"input": torch.ones(1)},
        ["input"],
        ["output"],
        None,
        {},
    )

    onnx_program.save.assert_called_once_with(str(onnx_path))
    assert result.onnx_path == onnx_path


def _make_blocked_fp8_layer(block_size):
    return FP8BlockWiseDequantLinear.for_fp8_layer_with_blocksize(
        128,
        128,
        block_size,
        "e4m3",
        False,
        scale_dtype=torch.float16,
    )


def test_blocked_fp8_translation_rejects_unsupported_block_size():
    model = torch.nn.Sequential(_make_blocked_fp8_layer([64, 64]))

    with pytest.raises(ValueError, match=r"Unsupported blocked FP8 block size \(64, 64\)"):
        _build_blocked_translation_table(model)


def test_blocked_fp8_translation_accepts_uniform_supported_block_size():
    model = torch.nn.Sequential(_make_blocked_fp8_layer([128, 128]), _make_blocked_fp8_layer([128, 128]))

    table = _build_blocked_translation_table(model)

    assert len(table) == 1
    assert table[torch.ops.qefficient.fp8_dequantize_blocked.default].__name__ == ("FP8DequantizeBlocked128x128Dynamo")


def test_blocked_fp8_translation_rejects_mixed_block_sizes():
    model = torch.nn.Sequential(_make_blocked_fp8_layer([1, 128]), _make_blocked_fp8_layer([128, 128]))

    with pytest.raises(ValueError, match="Mixed FP8 block sizes"):
        _build_blocked_translation_table(model)


def test_legacy_export_rejects_retained_fp8_weights():
    model = torch.nn.Sequential(_make_blocked_fp8_layer([128, 128]))

    with pytest.raises(ValueError, match="use dynamo=True"):
        _validate_legacy_fp8_export(model, dynamo=False)

    _validate_legacy_fp8_export(model, dynamo=True)
