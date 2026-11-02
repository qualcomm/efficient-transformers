# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Pytest fixtures and hooks for the dynamo nightly test suite (tests/dynamo/nightly/).

Provides a separate artifacts fixture (dynamo_causal_model_artifacts) so dynamo
nightly results are stored independently from regular nightly results.

Reuses all session-scoped fixtures from tests/nightly_pipeline/conftest.py
(artifacts_dir, get_pipeline_config, save_artifacts, load_artifacts) — those
are discovered automatically by pytest when this suite runs from the repo root.
"""

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
    """Separate JSON file for dynamo nightly artifacts — does not overwrite regular nightly results."""
    return artifacts_dir / "dynamo_causal_model_artifacts.json"


@pytest.fixture
def dynamo_causal_model_artifacts(dynamo_causal_model_artifacts_file):
    """Session-scoped dict for storing dynamo nightly per-model results."""
    from tests.nightly_pipeline.conftest import load_artifacts, save_artifacts

    artifacts = load_artifacts(dynamo_causal_model_artifacts_file)
    yield artifacts
    save_artifacts(dynamo_causal_model_artifacts_file, artifacts)
