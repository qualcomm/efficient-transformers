# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo export and ORT parity tests.

All tests run with dynamo=True and use_onnx_subfunctions=True.
Each test exports once then validates both:
  - ONNX graph structure: _RetainedState outputs, subfunctions, naming
  - HF PT == ORT token parity

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    PROMPT_LEN,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    assert_subfunction_names_match_decoder_class,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)


def _flatten_qwen4_exp_inputs(qeff_model, inputs):
    retained = [state for layer in inputs["past_key_values"] for state in layer]
    values = [inputs["input_ids"], inputs["ngram_embeddings"], inputs["position_ids"], *retained]
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(qeff_model.get_onnx_dynamic_axes(), values, strict=True)
    }


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_export_and_ort_parity(model_type, model_id, tmp_export_dir):
    """Export with dynamo=True and use_onnx_subfunctions=True, then validate
    ONNX structure and HF PT == ORT token parity in a single export pass."""

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    # Run HF PT first, before QEff transforms mutate the model.
    from QEfficient.utils.run_utils import ApiRunner

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    # --- Export ---
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    # --- Structure checks ---
    num_layers = model_hf.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)
    assert_subfunction_names_match_decoder_class(onnx_path, qeff_model)

    # --- ORT parity ---
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))
    assert hf_tokens is not None and ort_tokens is not None
    assert hf_tokens.flatten().tolist() == ort_tokens.flatten().tolist(), (
        f"HF PT vs ORT parity failed for {model_hf.__class__.__name__}: "
        f"HF={hf_tokens.flatten().tolist()}, ORT={ort_tokens.flatten().tolist()}"
    )


@pytest.mark.dynamo
@pytest.mark.dynamo_export
def test_qwen4_exp_dynamo_export_and_ort_retained_state_parity(tmp_export_dir):
    """Export the heterogeneous four-layer Qwen4-Exp decode graph through Dynamo.

    Qwen4-Exp layers have different retained-state signatures (GDN, PLE, and
    QSA), so repeated-subgraph extraction is intentionally disabled. Dynamo
    export itself must still preserve the complete flattened ONNX ABI.
    """
    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform
    from tests.unit_test.models.test_new_arch_accuracy import make_tiny_qwen4_exp

    ort = pytest.importorskip("onnxruntime")

    torch.manual_seed(42)
    model, _ = make_tiny_qwen4_exp()
    model, applied = KVCacheTransform.apply(model)
    assert applied
    qeff_model = QEFFAutoModelForCausalLM(model)
    inputs = qeff_model.model.get_dummy_inputs(batch_size=2)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=True,
            use_onnx_subfunctions=False,
            offload_pt_weights=False,
        )
    )

    assert onnx_path.is_file()
    assert_retained_state_outputs(onnx_path, expected_count=11)
    assert [item.name for item in ort.InferenceSession(str(onnx_path)).get_inputs()] == list(
        qeff_model.model.get_onnx_dynamic_axes()
    )

    with torch.no_grad():
        pytorch_output = qeff_model.model(**inputs)
    ort_output = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, _flatten_qwen4_exp_inputs(qeff_model.model, inputs)
    )
    expected_outputs = [
        pytorch_output.logits.detach().cpu().numpy(),
        *[state.detach().cpu().numpy() for layer in pytorch_output.past_key_values for state in layer],
    ]
    for expected, actual in zip(expected_outputs, ort_output, strict=True):
        np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-5)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_dynamo_compile_export_uses_requested_batch_size(batch_size, tmp_export_dir, monkeypatch):
    """Compile-driven Dynamo export must bind the batch selected by compile()."""
    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform
    from tests.unit_test.models.test_new_arch_accuracy import make_tiny_qwen4_exp

    model, _ = make_tiny_qwen4_exp()
    model, applied = KVCacheTransform.apply(model)
    assert applied
    qeff_model = QEFFAutoModelForCausalLM(model)
    captured = {}

    def capture_export(example_inputs, output_names, dynamic_axes, **kwargs):
        captured["batch_shape"] = tuple(example_inputs["input_ids"].shape)
        captured["dynamic_axes"] = dynamic_axes
        qeff_model.onnx_path = tmp_export_dir / "captured.onnx"
        return str(qeff_model.onnx_path)

    monkeypatch.setattr(qeff_model, "transform", lambda **kwargs: None)
    monkeypatch.setattr(qeff_model, "_export", capture_export)
    qeff_model.get_onnx_path(
        prefill_only=False,
        specializations=[{"batch_size": batch_size, "seq_len": 8, "ctx_len": 16}],
        dynamo=True,
        offload_pt_weights=False,
    )

    assert captured["batch_shape"][0] == batch_size
    assert 0 not in captured["dynamic_axes"]["input_ids"]
