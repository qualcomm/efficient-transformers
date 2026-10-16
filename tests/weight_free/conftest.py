# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Shared fixtures and configuration for the weight-free test suite (tests/weight_free/).

Weight-free export builds a meta-device model from config, exports an ONNX with no
embedded weights, and stores the weight mapping in a weight_spec.json alongside the
ONNX.  These tests verify the exported ONNX structure and (where weights are available)
HF PyTorch == ORT token parity after injecting real weights into ORT inputs.

Run with: pytest tests/weight_free/ -m "not on_qaic" -n auto -v
"""

from __future__ import annotations

import pytest
import torch

from QEfficient.utils.device_utils import get_available_device_id


def _parse_torch_version() -> tuple:
    parts = torch.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (0, 0)


def pytest_configure(config):
    config.addinivalue_line("markers", "weight_free: mark a test as part of the weight-free export test suite")
    config.addinivalue_line("markers", "weight_free_export: CPU-only weight-free export smoke and parity tests")


_XFAIL_MODELS = {}
_XFAIL_REASON = (
    "gpt_oss: dynamo=True + use_onnx_subfunctions=True triggers SerdeError "
    "(ir_version=10, serialize_model_into); export must use use_onnx_subfunctions=False"
)


def pytest_collection_modifyitems(config, items):
    torch_version = _parse_torch_version()
    xfail_models = pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
    for item in items:
        if not (item.fspath.parts and "weight_free" in str(item.fspath)):
            continue
        if torch_version < (2, 13):
            item.add_marker(
                pytest.mark.skip(reason=f"Weight-free tests require torch >= 2.13; running {torch.__version__}")
            )
        for model in _XFAIL_MODELS:
            if f"[{model}]" in item.nodeid or f"[{model}@" in item.nodeid:
                item.add_marker(xfail_models)


@pytest.fixture(autouse=True)
def set_cpu_threads():
    """Limit CPU threads per worker to avoid contention in parallel runs."""
    original = torch.get_num_threads()
    torch.set_num_threads(min(4, original))
    yield
    torch.set_num_threads(original)


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    torch.manual_seed(42)


@pytest.fixture
def tmp_export_dir(tmp_path):
    """Provide a temporary directory for ONNX exports (unique per test)."""
    export_dir = tmp_path / "qeff_weightfree_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


@pytest.fixture(autouse=True)
def skip_if_no_qaic_device(request):
    """Auto-skip any on_qaic test when no QAIC device is ready."""
    if request.node.get_closest_marker("on_qaic"):
        if get_available_device_id() is None:
            pytest.skip("No available QAIC device")
