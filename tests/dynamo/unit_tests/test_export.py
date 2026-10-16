# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CPU-only Dynamo export, ONNX structure, and ORT parity tests."""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from .._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    DTYPE,
    DYNAMO,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    PROMPT_LEN,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    assert_subfunction_names_match_decoder_class,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    skip_on_hf_model_load_error,
)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_export_and_ort_parity(model_type, model_id, tmp_export_dir):
    """Export with dynamo=True and use_onnx_subfunctions=True, then validate
    ONNX structure and HF PT == ORT token parity in a single export pass."""

    try:
        model_hf = load_hf_model(model_id, torch_dtype=DTYPE)
        tokenizer = load_tokenizer(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    # Run HF PT before QEff transforms mutate the model.
    from QEfficient.utils.run_utils import ApiRunner

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
        dtype=DTYPE,
    )
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=DYNAMO,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    num_layers = model_hf.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)
    assert_subfunction_names_match_decoder_class(onnx_path, qeff_model)

    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))
    assert hf_tokens is not None and ort_tokens is not None
    assert hf_tokens.flatten().tolist() == ort_tokens.flatten().tolist(), (
        f"HF PT vs ORT parity failed for {model_hf.__class__.__name__}: "
        f"HF={hf_tokens.flatten().tolist()}, ORT={ort_tokens.flatten().tolist()}"
    )
