# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo continuous-batching export tests.

All tests run with dynamo=True and use_onnx_subfunctions=True.
CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    exported_onnx_path,
    load_hf_model,
    skip_on_model_fetch_error,
)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_cb_export(model_type, model_id, tmp_export_dir):
    """continuous_batching=True + dynamo=True and use_onnx_subfunctions=True."""

    try:
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=True)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    num_layers = model_hf.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)
