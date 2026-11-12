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

from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    WEIGHT_FREE_PREFILL_BLOCKING_MODE_CASES,
    assert_blocked_kv_ops_for_mode,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    exported_onnx_path,
    load_causal_lm_config,
    skip_on_model_fetch_error,
)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
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


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.weight_free_blocking
def test_weight_free_gpt_oss_cb_prefill_online_export(tmp_export_dir):
    """Check GPT-OSS prefill_online blocking with continuous-batching inputs."""
    model_id = WEIGHT_FREE_CAUSAL_LM_MODEL_IDS["gpt_oss"]
    qaic_config = dict(WEIGHT_FREE_PREFILL_BLOCKING_MODE_CASES["prefill_online"])
    qaic_config["ctx_len"] = CTX_LEN
    qaic_config["moe_config"] = {"expert_parallel_chunk_size": 16}

    try:
        config = load_causal_lm_config(model_id)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            weight_free=True,
            continuous_batching=True,
            qaic_config=qaic_config,
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model.transform(
        ctx_len=CTX_LEN,
        seq_len=PROMPT_LEN,
        bs=1,
        num_devices=1,
        qaic_config=qaic_config,
        prefill_only=True,
        prefill_seq_len=PROMPT_LEN,
        enable_chunking=True,
    )
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_gpt_oss_cb_prefill_online",
            prefill_only=True,
            enable_chunking=True,
            prefill_seq_len=PROMPT_LEN,
            ctx_len=CTX_LEN,
            qaic_config=qaic_config,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    blocking_config = qeff_model.hash_params.get("blocking_kwargs")
    assert blocking_config is not None
    assert blocking_config.mode == BlockingMode.PREFILL_ONLINE
    assert_blocked_kv_ops_for_mode(onnx_path, qeff_model, "prefill_online", continuous_batching=True)
