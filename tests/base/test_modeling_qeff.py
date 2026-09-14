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

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.exporter.onnx_exporter import export_via_dynamo


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


def test_dynamo_export_forces_external_data(tmp_path, mocker):
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

    onnx_program.save.assert_called_once_with(str(onnx_path), external_data=True)
    assert result.onnx_path == onnx_path
