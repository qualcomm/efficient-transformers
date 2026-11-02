# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo nightly export + compile tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from QEfficient import QEFFAutoModelForCausalLM
from tests.dynamo._helpers import DYNAMO
from tests.nightly_pipeline.nightly_utils import pre_export_compile_utils

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "nightly_pipeline" / "configs"
_VALIDATED_MODELS_PATH = _CONFIGS_DIR / "validated_models.json"

with open(_VALIDATED_MODELS_PATH, "r") as _f:
    _config = json.load(_f)

test_models = _config["causal_lm_models"]


@pytest.mark.nightly
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_dynamo_export_compile_causal_lm(model_name, dynamo_causal_model_artifacts, get_pipeline_config):
    """Export with Dynamo and compile the resulting ONNX."""
    export_params, compile_params = pre_export_compile_utils(model_name, "causal_pipeline_configs", get_pipeline_config)

    # Force dynamo export path with ONNX subfunctions.
    export_params = {**export_params, "dynamo": DYNAMO, "use_onnx_subfunctions": True}
    compile_params = {**compile_params, "use_onnx_subfunctions": True}

    if model_name not in dynamo_causal_model_artifacts:
        dynamo_causal_model_artifacts[model_name] = {}

    print(f"\nLoading model for dynamo export: {model_name}")
    load_start = time.time()
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
    loading_time = time.time() - load_start

    print(f"\nExporting with dynamo={DYNAMO}: {model_name}")
    export_start = time.time()
    onnx_path = qeff_model.export(**export_params)
    export_time = time.time() - export_start

    print(f"\nCompiling: {model_name}")
    compile_start = time.time()
    qpc_path = qeff_model.compile(onnx_path=onnx_path, **compile_params)
    compile_time = time.time() - compile_start

    dynamo_causal_model_artifacts[model_name].update(
        {
            "onnx_path": onnx_path,
            "qpc_path": qpc_path,
            "export_loading_time": loading_time,
            "export_time": export_time,
            "compile_time": compile_time,
        }
    )
