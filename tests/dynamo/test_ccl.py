# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo + Compute-Context-Length (CCL) compile/generate tests."""

from __future__ import annotations

import pytest

from ._helpers import (
    BATCH_SIZE,
    CB_PROMPTS,
    DTYPE,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
    assert_hf_hw_parity,
    get_dynamo_export,
    get_hf_tokens,
    load_hf_model,
    load_tokenizer,
    skip_on_hf_model_load_error,
)

# CCL lengths need a larger ctx_len than the tiny default used by basic tests.
CCL_PREFILL_SEQ_LEN = 16
CCL_CTX_LEN = 64
CCL_LENGTHS = [1024, 2048]
PROMPT = "hello world"
GRANITE_CCL_PARITY_TOKENS = 10


def _generate(qeff_model, tokenizer, prompts, model_type, model_id, hf_tokens, full_batch_size=None):
    """Generate from the QPC produced by the preceding compile() call."""
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
    )
    assert output.generated_texts is not None
    gen_len = CCL_CTX_LEN - CCL_PREFILL_SEQ_LEN
    if model_type == "granite":
        gen_len = GRANITE_CCL_PARITY_TOKENS
    assert_hf_hw_parity(
        model_id,
        hf_tokens,
        output,
        gen_len=gen_len,
        full_batch_size=full_batch_size,
    )
    return output


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_ccl_compile_and_generate(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Compile and generate with default and explicit CCL lengths."""
    try:
        tokenizer = load_tokenizer(model_id, torch_dtype=DTYPE)
        model_hf = load_hf_model(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    onnx_path, qeff_model = get_dynamo_export(model_id, tmp_path_factory, torch_dtype=DTYPE, ccl_enabled=True)

    prompts = [PROMPT]
    hf_tokens = get_hf_tokens(tokenizer, model_hf, prompts, prompt_len=CCL_PREFILL_SEQ_LEN, ctx_len=CCL_CTX_LEN)

    # Compile with auto-generated CCL lists.
    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "normal_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    _generate(qeff_model, tokenizer, prompts, model_type, model_id, hf_tokens)

    # Compile with explicit CCL lists.
    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    _generate(qeff_model, tokenizer, prompts, model_type, model_id, hf_tokens)


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_cb_ccl_compile_and_generate(model_type, model_id, tmp_export_dir, tmp_path_factory):
    """Compile and generate with continuous batching and CCL."""
    if model_type == "gpt_oss":
        pytest.skip("gpt_oss CB scatter op has shape mismatch with dynamo subfunctions; pending fix")

    try:
        tokenizer = load_tokenizer(model_id, torch_dtype=DTYPE)
        model_hf = load_hf_model(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    onnx_path, qeff_model = get_dynamo_export(
        model_id, tmp_path_factory, torch_dtype=DTYPE, continuous_batching=True, ccl_enabled=True
    )

    prompts = CB_PROMPTS
    hf_tokens = get_hf_tokens(
        tokenizer,
        model_hf,
        prompts,
        prompt_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        full_batch_size=FULL_BATCH_SIZE,
    )

    # Compile with auto-generated CCL lists.
    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "cb_normal_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    _generate(qeff_model, tokenizer, prompts, model_type, model_id, hf_tokens, full_batch_size=FULL_BATCH_SIZE)

    # Compile with explicit CCL lists.
    qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(tmp_export_dir / "cb_ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    _generate(qeff_model, tokenizer, prompts, model_type, model_id, hf_tokens, full_batch_size=FULL_BATCH_SIZE)
