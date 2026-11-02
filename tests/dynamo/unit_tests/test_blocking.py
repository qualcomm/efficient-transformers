# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CPU-only Dynamo blocking export structure tests."""

from __future__ import annotations

import onnx
import pytest

from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from .._helpers import (
    CTX_LEN,
    DTYPE,
    DYNAMO,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    PROMPT_LEN,
    assert_blocked_kv_ops_for_mode,
    exported_onnx_path,
    skip_on_hf_model_load_error,
)

# Same supported model set used by tests/dynamo/test_blocking.py.
_BLOCKING_SUPPORTED_TYPES = {
    "gemma",
    "gemma2",
    "glm4_moe",
    "gpt_oss",
    "granite",
    "granitemoe",
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "starcoder2",
}
BLOCKING_MODEL_IDS = {k: v for k, v in DYNAMO_CAUSAL_LM_MODEL_IDS.items() if k in _BLOCKING_SUPPORTED_TYPES}

HEAD_BLOCK_SIZE = 2
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2

# Modes with reliable mode-specific blocked-KV export markers.
BLOCKING_MODE_CASES = {
    "kv": dict(blocking_mode="kv", num_kv_blocks=NUM_KV_BLOCKS),
    "qkv": dict(blocking_mode="qkv", num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS),
    "hkv": dict(blocking_mode="hkv", head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS),
    "hqkv": (
        dict(
            blocking_mode="hqkv",
            head_block_size=HEAD_BLOCK_SIZE,
            num_kv_blocks=NUM_KV_BLOCKS,
            num_q_blocks=NUM_Q_BLOCKS,
        )
    ),
}


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize("blocking_key", list(BLOCKING_MODE_CASES))
@pytest.mark.parametrize("model_type,model_id", sorted(BLOCKING_MODEL_IDS.items()), ids=sorted(BLOCKING_MODEL_IDS))
def test_dynamo_blocking_export_gather_ops(model_type, model_id, blocking_key, tmp_export_dir):
    """Verify KV-blocking modes emit the expected gather marker in decoder subfunctions."""
    if blocking_key == "hkv":
        pytest.xfail(f"blocked_hqkv_attention_forward crashes on mode='{blocking_key}' (None block count)")

    # transform() mutates qaic_config.
    qaic_config = dict(BLOCKING_MODE_CASES[blocking_key])

    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE)
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)
    qeff_model.transform(
        ctx_len=CTX_LEN,
        seq_len=PROMPT_LEN,
        bs=1,
        qaic_config=qaic_config,
    )
    expected_mode = BlockingMode(BLOCKING_MODE_CASES[blocking_key]["blocking_mode"])
    attached_configs = [
        module.attn_blocking_config for module in qeff_model.model.modules() if hasattr(module, "attn_blocking_config")
    ]
    assert attached_configs, "Expected blocking config on at least one attention module"
    assert all(config.mode == expected_mode for config in attached_configs)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            dynamo=DYNAMO,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    decoder_names = {cls.__name__ for cls in get_submodules()} if callable(get_submodules) else set()
    decoder_functions = [fn for fn in onnx_model.functions if any(name in fn.name for name in decoder_names)]
    assert decoder_functions, (
        f"Expected decoder-block subfunctions ({decoder_names}) but found none. "
        f"Functions present: {[fn.name for fn in onnx_model.functions]}"
    )

    assert_blocked_kv_ops_for_mode(onnx_path, qeff_model, blocking_key)
