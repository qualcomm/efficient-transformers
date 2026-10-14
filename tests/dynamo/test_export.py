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

import onnx
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    MODEL_KWARGS,
    PROMPT_LEN,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    assert_subfunction_names_match_decoder_class,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
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
def test_dynamo_export_reuses_equivalent_mixed_layer_subfunctions(tmp_export_dir):
    """A 4-layer sliding/full-attention model must export one subfunction per distinct layer
    type, call each one for every matching layer, and keep HF PT == ORT parity."""
    model_id = DYNAMO_CAUSAL_LM_MODEL_IDS["gpt_oss"]
    layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = len(layer_types)
        config.layer_types = layer_types
        config.dtype = MODEL_KWARGS["torch_dtype"]
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)
    torch.manual_seed(0)
    model_hf = AutoModelForCausalLM.from_config(
        config, attn_implementation=MODEL_KWARGS["attn_implementation"], dtype=MODEL_KWARGS["torch_dtype"]
    ).eval()

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

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(tmp_export_dir, dynamo=True, use_onnx_subfunctions=True, offload_pt_weights=False)
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    layer_functions = {fn.name for fn in onnx_model.functions if "DecoderLayer" in fn.name}
    layer_calls = [node for node in onnx_model.graph.node if node.op_type in layer_functions]
    assert len(layer_calls) == len(layer_types), f"Expected one call per layer, got {len(layer_calls)}"
    assert len({node.op_type for node in layer_calls}) == 2, (
        f"Expected sliding and full attention layers to map to two distinct subfunctions, "
        f"got {sorted({node.op_type for node in layer_calls})}"
    )
    for sliding_call, full_call in ((0, 1), (2, 3)):
        assert layer_calls[sliding_call].op_type != layer_calls[full_call].op_type
    assert layer_calls[0].op_type == layer_calls[2].op_type
    assert layer_calls[1].op_type == layer_calls[3].op_type

    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))
    assert hf_tokens.flatten().tolist() == ort_tokens.flatten().tolist()
