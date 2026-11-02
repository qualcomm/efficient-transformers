# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Shared fixtures and configuration for the dynamo test suite (tests/dynamo/).

All dynamo tests require torch >= 2.13 (the minimum version that supports
torch.compiler.nested_compile_region and the dynamo export path).

Run with: pytest tests/dynamo/ -m "not on_qaic" -n 4 -v
"""

from __future__ import annotations

import pytest
import torch

from QEfficient.utils.device_utils import is_multi_qranium_setup_available


def _parse_torch_version() -> tuple:
    parts = torch.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (0, 0)


def pytest_configure(config):
    config.addinivalue_line("markers", "dynamo: mark a test as part of the dynamo test suite")
    config.addinivalue_line("markers", "dynamo_export: CPU-only dynamo export smoke and parity tests")
    config.addinivalue_line(
        "markers",
        "dynamo_multi_device: dynamo multi-device (MDP) compile tests — requires MDP-capable QAIC setup",
    )


def pytest_collection_modifyitems(config, items):
    torch_version = _parse_torch_version()
    if torch_version < (2, 13):
        skip = pytest.mark.skip(reason=f"Dynamo tests require torch >= 2.13; running {torch.__version__}")
        for item in items:
            if item.fspath.parts and "dynamo" in str(item.fspath):
                item.add_marker(skip)


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
    export_dir = tmp_path / "qeff_dynamo_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


@pytest.fixture(autouse=True)
def skip_if_no_mdp_setup(request):
    """Auto-skip multi-device tests when the hardware doesn't have MDP-capable devices."""
    if request.node.get_closest_marker("dynamo_multi_device"):
        if not is_multi_qranium_setup_available():
            pytest.skip("No MDP-capable QAIC device setup available (requires HybridBoot+ MDP+)")
