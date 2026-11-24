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
    _copy_onnx_external_data_files,
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


def _external_initializer(name, location):
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = onnx.TensorProto.FLOAT
    tensor.dims.append(1)
    tensor.data_location = onnx.TensorProto.EXTERNAL
    external_data = tensor.external_data.add()
    external_data.key = "location"
    external_data.value = str(location)
    return tensor


def _save_model_with_external_initializers(onnx_path, initializers):
    graph = onnx.helper.make_graph(
        [],
        "external_data_graph",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
        initializer=initializers,
    )
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)]), onnx_path)


def test_copy_onnx_external_data_files_deduplicates_shared_file(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    artifact_dir = tmp_path / "artifact"
    source_dir.mkdir()
    artifact_dir.mkdir()
    source_onnx_path = source_dir / "model.onnx"
    artifact_onnx_path = artifact_dir / "model.onnx"
    shared_data_path = source_dir / "shared.data"
    shared_data_path.write_bytes(b"shared-weights")
    _save_model_with_external_initializers(
        source_onnx_path,
        [
            _external_initializer("relative_weight", shared_data_path.name),
            _external_initializer("absolute_weight", shared_data_path),
        ],
    )
    copy_calls = []

    def copy2(source, destination):
        copy_calls.append((source, destination))
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr("QEfficient.base.modeling_qeff.shutil.copy2", copy2)

    _copy_onnx_external_data_files(source_onnx_path, artifact_onnx_path)

    assert copy_calls == [(shared_data_path.resolve(), (artifact_dir / shared_data_path.name).resolve())]
    assert (artifact_dir / shared_data_path.name).read_bytes() == shared_data_path.read_bytes()


def test_copy_onnx_external_data_files_skips_same_file_destination(tmp_path, monkeypatch):
    source_onnx_path = tmp_path / "model.onnx"
    shared_data_path = tmp_path / "shared.data"
    shared_data_path.write_bytes(b"shared-weights")
    _save_model_with_external_initializers(
        source_onnx_path,
        [
            _external_initializer("first_weight", shared_data_path.name),
            _external_initializer("second_weight", shared_data_path.name),
        ],
    )
    copy_calls = []
    monkeypatch.setattr("QEfficient.base.modeling_qeff.shutil.copy2", lambda source, destination: copy_calls.append(1))

    _copy_onnx_external_data_files(source_onnx_path, source_onnx_path)

    assert copy_calls == []
