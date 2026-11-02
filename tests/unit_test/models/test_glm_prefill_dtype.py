import pytest
import torch

from examples.glm.glm53_four_layer_prefill_compile import resolve_prefill_torch_dtype


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
