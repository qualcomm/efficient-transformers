# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free continuous-batching export tests.

All tests run with weight_free=True (set on the QEff model) and use_onnx_subfunctions=True.
CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import pytest
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    exported_onnx_path,
    skip_on_model_fetch_error,
)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    [
        *sorted(
            (model_type, model_id)
            for model_type, model_id in WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()
            if model_type != "gpt_oss"
        ),
        # ("gpt_oss", WEIGHT_FREE_CAUSAL_LM_MODEL_IDS["gpt_oss"]),
    ],
    ids=sorted(model_type for model_type in WEIGHT_FREE_CAUSAL_LM_MODEL_IDS if model_type != "gpt_oss"),
)
def test_weight_free_cb_export(model_type, model_id, tmp_export_dir):
    """continuous_batching=True + weight_free=True + use_onnx_subfunctions=True.

    Validates ONNX structure only (no ORT parity — CB ORT inference requires
    batch_index routing which adds significant test complexity).
    """
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, config=config, weight_free=True, continuous_batching=True
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    num_layers = qeff_model.model.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)
