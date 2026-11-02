# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo on-QAIC tests.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
All tests run with dynamo=True and use_onnx_subfunctions=True.

Covers:
  - FP16 compile
  - FP32 compile (torch_dtype=float32 model)
  - Multi-device compile (num_devices=4)
  - Generate FP16
  - HF AIC HW parity (HF PT tokens == QAIC FP16 top-1 token)
  - Continuous-batching generate
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
    PROMPT_LEN,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_fp16_compile(model_type, model_id, tmp_export_dir):
    """Export with dynamo=True and use_onnx_subfunctions=True, compile to FP16 QPC."""

    try:
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "fp16_export",
            dynamo=True,
            use_onnx_subfunctions=True,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "fp16_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "fp16_compile").is_dir()


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_fp32_compile(model_type, model_id, tmp_export_dir):
    """Export with dynamo=True and use_onnx_subfunctions=True, compile with fp32 model."""

    try:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
        )
        model_hf.eval()
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "fp32_export",
            dynamo=True,
            use_onnx_subfunctions=True,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "fp32_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "fp32_compile").is_dir()


@pytest.mark.dynamo
@pytest.mark.dynamo_multi_device
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_multi_device_compile(model_type, model_id, tmp_export_dir):
    """Export with dynamo=True and use_onnx_subfunctions=True, compile for 4 devices."""

    try:
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "mdp_export",
            dynamo=True,
            use_onnx_subfunctions=True,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "mdp_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=4,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "mdp_compile").is_dir()


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_generate_fp16(model_type, model_id, tmp_export_dir):
    """End-to-end export → compile → generate with dynamo=True and use_onnx_subfunctions=True."""

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "gen_export",
            dynamo=True,
            use_onnx_subfunctions=True,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert output is not None
    assert output.generated_texts is not None


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_hw_hf_parity(model_type, model_id, tmp_export_dir):
    """HF PT tokens == QAIC FP16 tokens (exact equality)."""
    from QEfficient.utils.run_utils import ApiRunner

    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

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
    assert hf_tokens is not None, "HF PT inference returned None"

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "hw_parity_export",
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "hw_parity_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )

    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )

    assert qaic_output is not None, "QAIC generate returned None"
    if hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None:
        gen_len = CTX_LEN - PROMPT_LEN
        qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
        assert np.array_equal(hf_tokens, qaic_tokens), (
            f"HF AIC HW parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
        )


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_cb_generate(model_type, model_id, tmp_export_dir):
    """Continuous-batching export → compile → generate with dynamo=True and use_onnx_subfunctions=True."""

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=True)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "cb_gen_export",
            dynamo=True,
            use_onnx_subfunctions=True,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "cb_gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    prompts = ["hello world"] * FULL_BATCH_SIZE
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
    )
    assert output is not None
    assert output.generated_texts is not None
    assert len(output.generated_texts) == FULL_BATCH_SIZE
