# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo nightly fixtures and artifact storage."""

from __future__ import annotations

import pytest
import torch


def _parse_torch_version():
    parts = torch.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (0, 0)


def pytest_collection_modifyitems(config, items):
    torch_version = _parse_torch_version()
    if torch_version < (2, 13):
        skip = pytest.mark.skip(reason=f"Dynamo nightly tests require torch >= 2.13; running {torch.__version__}")
        for item in items:
            if "dynamo/nightly" in str(item.fspath):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def dynamo_causal_model_artifacts_file(artifacts_dir):
    """JSON file for Dynamo nightly artifacts."""
    return artifacts_dir / "dynamo_causal_model_artifacts.json"


@pytest.fixture
def dynamo_causal_model_artifacts(dynamo_causal_model_artifacts_file):
    """Session-scoped dict for storing dynamo nightly per-model results."""
    from tests.nightly_pipeline.conftest import load_artifacts, save_artifacts

    artifacts = load_artifacts(dynamo_causal_model_artifacts_file)
    yield artifacts
    save_artifacts(dynamo_causal_model_artifacts_file, artifacts)
