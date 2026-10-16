# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo on-QAIC compile and generation tests."""

from __future__ import annotations

import pytest
import torch

from ._helpers import (
    BATCH_SIZE,
    CB_PROMPTS,
    CTX_LEN,
    DTYPE,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
    PROMPT_LEN,
    assert_hf_hw_parity,
    get_dynamo_export,
    get_hf_tokens,
    load_hf_model,
    load_tokenizer,
    skip_on_hf_model_load_error,
)

GRANITE_HW_PARITY_TOKENS = 10


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_fp16_weights_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Export a torch.float16-weights model, compile to QPC."""

    onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=DTYPE)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "fp16_weights_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "fp16_weights_compile").is_dir()


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.skip(reason="Disabled while testing -- fp32-weights compile still converts to fp16 on-device anyway.")
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_fp32_weights_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Export a torch.float32-weights model, compile to QPC (compiler still converts to fp16)."""

    onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=torch.float32)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "fp32_weights_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        use_onnx_subfunctions=True,
    )
    assert (tmp_export_dir / "fp32_weights_compile").is_dir()


@pytest.mark.dynamo
@pytest.mark.dynamo_multi_device
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_multi_device_compile(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Compile the shared fp16-weights basic export for 4 devices."""

    onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=DTYPE)

    qeff_model.compile(
        onnx_path=onnx_path,
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
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_hw_hf_parity(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Validate exact HF PT vs QAIC FP16 token parity."""
    try:
        tokenizer = load_tokenizer(model_id, torch_dtype=DTYPE)
        model_hf = load_hf_model(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=DTYPE)

    hf_tokens = get_hf_tokens(
        tokenizer,
        model_hf,
        ["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
    )

    qeff_model.compile(
        onnx_path=onnx_path,
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

    gen_len = CTX_LEN - PROMPT_LEN
    if model_type == "granite":
        gen_len = GRANITE_HW_PARITY_TOKENS

    assert_hf_hw_parity(
        model_id,
        hf_tokens,
        qaic_output,
        gen_len=gen_len,
    )


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()),
    ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS),
)
def test_dynamo_cb_generate(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Continuous-batching export → compile → generate with dynamo=True and use_onnx_subfunctions=True."""

    try:
        tokenizer = load_tokenizer(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=DTYPE, continuous_batching=True)

    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "cb_gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    prompts = CB_PROMPTS
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
    )
    assert output is not None
    assert output.generated_texts is not None
    assert len(output.generated_texts) == FULL_BATCH_SIZE
