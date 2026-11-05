import sys

import ml_dtypes
import numpy as np
import onnx
import pytest
import torch

from examples.glm.glm53_four_layer_prefill_compile import resolve_prefill_torch_dtype
from examples.glm.glm53_four_layer_prefill_trace import _onnx_numpy_dtype


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_resolve_prefill_torch_dtype(name, expected):
    assert resolve_prefill_torch_dtype(name) is expected


def test_resolve_prefill_torch_dtype_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_prefill_torch_dtype("float8")


def test_prefill_compile_accepts_weight_free_subfunctions(monkeypatch):
    from examples.glm.glm53_four_layer_prefill_compile import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["glm53_four_layer_prefill_compile.py", "--weight-free", "--use-onnx-subfunctions"],
    )

    args = parse_args()

    assert args.weight_free is True
    assert args.use_onnx_subfunctions is True


def test_onnx_bfloat16_uses_two_byte_numpy_dtype():
    value_info = onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.BFLOAT16, [1])

    dtype = _onnx_numpy_dtype(value_info)

    assert dtype == np.dtype(ml_dtypes.bfloat16)
    assert dtype.itemsize == 2
