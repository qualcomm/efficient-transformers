# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free + Compute-Context-Length (CCL) compile/generate tests.
"""

from __future__ import annotations

import pytest
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    WEIGHT_FREE_QAIC_MODEL_PARAMS,
    exported_onnx_path,
    load_tokenizer,
    skip_on_model_fetch_error,
)

# Mirrors tests/dynamo/test_ccl.py: CCL's specialization validation floors small
# comp_ctx_lengths values up to CCL_MIN_CTX_LEN (1024) then clamps back down to
# ctx_len; with a tiny CTX_LEN, both prefill and decode collapse to the same value
# and the collision-repair step lands on 0, which the compiler rejects. Use
# ctx_len/prefill_seq_len large enough to avoid that collision.
CCL_PREFILL_SEQ_LEN = 32
CCL_CTX_LEN = 128
CCL_LENGTHS = [1024, 2048]


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_ccl_compile_and_generate(model_type, model_id, tmp_export_dir):
    """Export once, compile twice (normal and with explicit CCL), generate on each QPC."""
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, config=config, weight_free=True, qaic_config={"ccl_enabled": True}
        )
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_ccl_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    # --- Compile normally (auto-generated CCL lists; no explicit comp_ctx_lengths) ---
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_normal_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
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

    # --- Compile with explicit CCL specializations, reusing the same export ---
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
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
