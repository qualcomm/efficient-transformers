# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free export ONNX structure and ORT parity tests.

All tests export with weight_free=True (set on the QEff model) and use_onnx_subfunctions=True.
Each test validates:
  - ONNX graph structure: _RetainedState outputs, subfunctions, naming
  - Weight-free-specific: unique graph input names, no int64 KV cache inputs
  - HF PT == ORT token parity (weights injected from HF cache via load_weight_free_ort_inputs)

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.exporter.weight_free import load_weight_free_ort_inputs, resolve_weight_spec_path
from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP
from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForImageTextToText,
)
from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils.test_utils import load_vlm_model_from_config

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_BLOCKING_CAUSAL_LM_MODEL_IDS,
    WEIGHT_FREE_BLOCKING_MODE_CASES,
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    WEIGHT_FREE_EMBEDDING_MODEL_PARAMS,
    WEIGHT_FREE_PREFILL_BLOCKING_MODE_CASES,
    assert_blocked_kv_ops_for_mode,
    assert_has_subfunctions,
    assert_no_int64_kv_cache_inputs,
    assert_retained_state_outputs,
    assert_subfunction_names_match_decoder_class,
    assert_unique_graph_input_names,
    exported_onnx_path,
    load_causal_lm_config,
    load_hf_embedding_model,
    load_hf_model,
    load_tokenizer,
    run_weight_free_ort,
    skip_on_model_fetch_error,
)

_BLOCKING_EXPORT_CASES = [
    pytest.param("llama", blocking_key, id=f"llama-{blocking_key}") for blocking_key in WEIGHT_FREE_BLOCKING_MODE_CASES
]

_BLOCKING_GATHER_EXPORT_MODE_CASES = {
    "kv": {"blocking_mode": "kv", "num_kv_blocks": 2},
    "qkv": {"blocking_mode": "qkv", "num_kv_blocks": 2, "num_q_blocks": 2},
    "hkv": {"blocking_mode": "hkv", "head_block_size": 2, "num_kv_blocks": 2},
    "hqkv": {"blocking_mode": "hqkv", "head_block_size": 2, "num_kv_blocks": 2, "num_q_blocks": 2},
}

_PREFILL_BLOCKING_EXPORT_MODEL_CASES = (
    ("gpt_oss", "tiny-random/gpt-oss-mxfp4", False),
    ("gemma4", "tiny-random/gemma-4-dense", True),
    ("qwen3_5_moe", "tiny-random/qwen3.5-moe", True),
    ("qwen3_vl_moe", "tiny-random/qwen3-vl-moe", True),
)


def _load_prefill_blocking_config(model_id: str, is_vlm: bool):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_config = config.text_config if is_vlm else config
    text_config.num_hidden_layers = 2
    if getattr(config, "model_type", None) in {"gemma4", "qwen3_5_moe"}:
        text_config.layer_types = ["full_attention"] * text_config.num_hidden_layers
    return config


def _build_prefill_blocking_model(
    model_type: str,
    model_id: str,
    is_vlm: bool,
    qaic_config: dict,
    tmp_export_dir,
):
    config = _load_prefill_blocking_config(model_id, is_vlm)
    if is_vlm:
        for nested_config in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
            if nested_config is not None:
                nested_config.dtype = torch.float32
                nested_config.torch_dtype = torch.float32
        checkpoint_dir = tmp_export_dir / f"{model_type}_prefill_blocking_checkpoint"
        torch.manual_seed(42)
        load_vlm_model_from_config(config).save_pretrained(checkpoint_dir, safe_serialization=True)
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            str(checkpoint_dir),
            kv_offload=True,
            weight_free=True,
            qaic_config=qaic_config,
        )
        return qeff_model

    return QEFFAutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        weight_free=True,
        qaic_config=qaic_config,
    )


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize("model_type,model_id,pooling", WEIGHT_FREE_EMBEDDING_MODEL_PARAMS)
def test_weight_free_embedding_export_ort_parity(model_type, model_id, pooling, tmp_export_dir):
    """Export an embedding model without weights and validate HF-to-ORT parity."""
    try:
        model_hf = load_hf_embedding_model(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if model_type in {"jina", "nomic"}:
            raise
        skip_on_model_fetch_error(exc, model_id)

    inputs = tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        hf_hidden_states = model_hf(**inputs).last_hidden_state
        hf_output = POOLING_MAP[pooling](hf_hidden_states, inputs["attention_mask"])

    qeff_model = QEFFAutoModel.from_pretrained(model_id, pooling=pooling, weight_free=True)
    onnx_path = exported_onnx_path(qeff_model.export(tmp_export_dir, dynamo=True, offload_pt_weights=False))
    assert onnx_path.is_file(), f"Expected ONNX file at {onnx_path}"

    weight_spec_path = resolve_weight_spec_path(onnx_path)
    assert weight_spec_path.is_file(), f"Expected weight spec at {weight_spec_path}"

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    graph_input_names = [value.name for value in onnx_model.graph.input]
    assert len(graph_input_names) == len(set(graph_input_names))
    assert {"input_ids", "attention_mask"}.issubset(graph_input_names)

    session = onnxruntime.InferenceSession(str(onnx_path))
    session_input_names = {value.name for value in session.get_inputs()}
    ort_inputs = {name: value.detach().cpu().numpy() for name, value in inputs.items() if name in session_input_names}
    ort_inputs = load_weight_free_ort_inputs(weight_spec_path, ort_inputs)
    ort_output = session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(
        hf_output.detach().cpu().numpy(),
        ort_output,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"HF PT vs weight-free ORT parity failed for {model_type}",
    )


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_export_onnx_structure(model_type, model_id, tmp_export_dir):
    """Export with weight_free=True and use_onnx_subfunctions=True, then validate:
    - Correct _RetainedState output count (2 x num_layers)
    - Subfunctions renamed to decoder class names
    - No duplicate graph input names (guards position_ids aliasing regression)
    - No int64 tensor named past_key.X / past_value.X (same regression, dtype check)

    CPU-only. No QAIC hardware or weight injection required.
    """
    try:
        # Build meta-device model — no weights loaded, only shapes.
        # pretrained_model_name_or_path is carried in the QEff model so the export
        # can write weight_spec.json pointing at the HF cache checkpoint.
        config = load_causal_lm_config(model_id)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    # Shared structure checks
    num_layers = qeff_model.model.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)
    assert_subfunction_names_match_decoder_class(onnx_path, qeff_model)

    # Weight-free-specific regression guards
    assert_unique_graph_input_names(onnx_path)
    assert_no_int64_kv_cache_inputs(onnx_path)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize("model_type,blocking_key", _BLOCKING_EXPORT_CASES)
def test_weight_free_export_applies_causal_lm_blocking_from_qaic_config(model_type, blocking_key, tmp_export_dir):
    """Direct CausalLM weight-free export honors qaic_config blocking modes."""
    use_onnx_subfunctions = True
    model_id = WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.get(model_type, model_type)
    qaic_config = dict(WEIGHT_FREE_BLOCKING_MODE_CASES[blocking_key])
    try:
        config = load_causal_lm_config(model_id)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            weight_free=True,
            qaic_config=qaic_config,
        )
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=False,
        )
    )

    blocking_config = qeff_model.hash_params.get("blocking_kwargs")
    assert blocking_config is not None
    assert blocking_config.mode == BlockingMode(qaic_config["blocking_mode"])
    if use_onnx_subfunctions:
        assert_has_subfunctions(onnx_path, qeff_model)
    assert_blocked_kv_ops_for_mode(onnx_path, qeff_model, blocking_key)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.weight_free_blocking
@pytest.mark.parametrize("blocking_key", list(_BLOCKING_GATHER_EXPORT_MODE_CASES))
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_BLOCKING_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_BLOCKING_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_blocking_export_gather_ops(model_type, model_id, blocking_key, tmp_export_dir):
    """Mirror tests/dynamo/unit_tests/test_blocking.py for the weight-free export path."""
    if blocking_key == "hkv" and model_type != "gpt_oss":
        pytest.xfail(f"blocked_hqkv_attention_forward crashes on mode='{blocking_key}' (None block count)")

    qaic_config = dict(_BLOCKING_GATHER_EXPORT_MODE_CASES[blocking_key])
    try:
        config = load_causal_lm_config(model_id)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            weight_free=True,
            qaic_config=qaic_config,
        )
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    expected_mode = BlockingMode(_BLOCKING_GATHER_EXPORT_MODE_CASES[blocking_key]["blocking_mode"])
    attached_configs = [
        module.attn_blocking_config for module in qeff_model.model.modules() if hasattr(module, "attn_blocking_config")
    ]
    assert attached_configs, "Expected blocking config on at least one attention module"
    assert all(config.mode == expected_mode for config in attached_configs)

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    decoder_names = {cls.__name__ for cls in get_submodules()} if callable(get_submodules) else set()
    decoder_functions = [fn for fn in onnx_model.functions if any(name in fn.name for name in decoder_names)]
    assert decoder_functions, (
        f"Expected decoder-block subfunctions ({decoder_names}) but found none. "
        f"Functions present: {[fn.name for fn in onnx_model.functions]}"
    )

    assert_blocked_kv_ops_for_mode(onnx_path, qeff_model, blocking_key)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.weight_free_blocking
@pytest.mark.parametrize("blocking_key", list(WEIGHT_FREE_PREFILL_BLOCKING_MODE_CASES))
@pytest.mark.parametrize(
    ("model_type", "model_id", "is_vlm"),
    _PREFILL_BLOCKING_EXPORT_MODEL_CASES,
    ids=[model_type for model_type, _, _ in _PREFILL_BLOCKING_EXPORT_MODEL_CASES],
)
def test_weight_free_prefill_blocking_export_gather_ops(
    model_type,
    model_id,
    is_vlm,
    blocking_key,
    tmp_export_dir,
):
    """Weight-free prefill-only export uses each requested offline blocking mode."""
    qaic_config = {**WEIGHT_FREE_PREFILL_BLOCKING_MODE_CASES[blocking_key], "ctx_len": 16}
    try:
        qeff_model = _build_prefill_blocking_model(
            model_type,
            model_id,
            is_vlm,
            qaic_config,
            tmp_export_dir,
        )
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    export_dir = tmp_export_dir / f"wf_{model_type}_{blocking_key}_prefill_export"
    export_kwargs = {
        "prefill_only": True,
        "enable_chunking": True,
        "prefill_seq_len": 8,
        "ctx_len": 16,
        "qaic_config": qaic_config,
        "use_onnx_subfunctions": False,
        "offload_pt_weights": False,
    }
    if is_vlm:
        qeff_model.transform(
            ctx_len=16,
            seq_len=8,
            qaic_config=qaic_config,
            prefill_only=True,
            prefill_seq_len=8,
        )
        onnx_path = Path(qeff_model.export(export_dir, skip_vision=True, **export_kwargs)[1])
        qeff_model = qeff_model.lang_model
    else:
        onnx_path = exported_onnx_path(qeff_model.export(export_dir, **export_kwargs))

    expected_mode = BlockingMode(qaic_config["blocking_mode"])
    blocking_config = qeff_model.hash_params.get("blocking_kwargs")
    assert blocking_config is not None
    assert blocking_config.mode == expected_mode
    assert_blocked_kv_ops_for_mode(onnx_path, qeff_model, blocking_key)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_export_ort_parity(model_type, model_id, tmp_export_dir):
    """Export with weight_free=True, inject real weights into ORT, then validate
    HF PyTorch == ORT token parity.

    Unlike regular ONNX (which embeds weights), weight-free ONNX stores weights externally
    in a weight_spec.json. load_weight_free_ort_inputs reads the checkpoint from HF cache
    and injects the tensors as ORT inputs at each generation step.

    CPU-only. No QAIC hardware required.
    """
    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:  # noqa: BLE001
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

    # HF PyTorch reference tokens (real weights)
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    # Weight-free export uses a meta-device model — no weights in the ONNX.
    # The real weights are read from the HF cache at ORT inference time via
    # load_weight_free_ort_inputs(weight_spec_path, inputs).
    # The exported model must have the same layer count as model_hf, or the
    # HF PT and ORT token streams come from architecturally different models.
    try:
        config = load_causal_lm_config(model_id)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    weight_spec_path = resolve_weight_spec_path(onnx_path)

    # ORT inference with weights injected from HF cache
    ort_tokens = run_weight_free_ort(api_runner, onnx_path, weight_spec_path)

    assert hf_tokens is not None and ort_tokens is not None
    assert hf_tokens.flatten().tolist() == ort_tokens.flatten().tolist(), (
        f"HF PT vs weight-free ORT parity failed for {model_hf.__class__.__name__}: "
        f"HF={hf_tokens.flatten().tolist()}, ORT={ort_tokens.flatten().tolist()}"
    )
