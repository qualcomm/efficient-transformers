# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo nightly generate tests.

Mirrors tests/nightly_pipeline/causal_lm_models/test_generate.py but reads
artifacts from dynamo_causal_model_artifacts (produced by test_export_compile.py
in this directory) rather than causal_model_artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from tests.nightly_pipeline.nightly_utils import get_onnx_and_qpc_size, pre_generate_utils

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "nightly_pipeline" / "configs"
_VALIDATED_MODELS_PATH = _CONFIGS_DIR / "validated_models.json"

with open(_VALIDATED_MODELS_PATH, "r") as _f:
    _config = json.load(_f)

test_models = _config["causal_lm_models"]


@pytest.mark.nightly
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_dynamo_generate_causal_lm(model_name, dynamo_causal_model_artifacts, get_pipeline_config):
    """Generate with QAIC using the QPC compiled from a dynamo-exported ONNX."""
    compile_params, generate_params = pre_generate_utils(
        model_name, "causal_pipeline_configs", get_pipeline_config, dynamo_causal_model_artifacts
    )

    onnx_path = dynamo_causal_model_artifacts[model_name].get("onnx_path")

    print(f"\nLoading model for dynamo generation: {model_name}")
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    print(f"\nGenerating for model (dynamo path): {model_name}")
    exec_info = qeff_model.generate(tokenizer=tokenizer, **generate_params)

    onnx_and_qpc_dir = os.path.dirname(onnx_path)
    size = get_onnx_and_qpc_size(onnx_and_qpc_dir)

    dynamo_causal_model_artifacts[model_name].update(
        {
            "batch_size": exec_info.batch_size,
            "generated_texts": exec_info.generated_texts,
            "generated_ids": exec_info.generated_ids[0][0][:20],
            "onnx_and_qpc_dir": onnx_and_qpc_dir,
            "size": size,
            "perf_metrics": {
                "prefill_time": exec_info.perf_metrics.prefill_time,
                "decode_perf": exec_info.perf_metrics.decode_perf,
                "total_perf": exec_info.perf_metrics.total_perf,
                "total_time": exec_info.perf_metrics.total_time,
            },
        }
    )
