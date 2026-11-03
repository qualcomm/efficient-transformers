# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.utils.test_utils import load_vlm_model_from_config
from tests.two_phase import resolve_two_phase_cleanup

from .test_image_text_to_text_models import model_config_dict

MODEL_NAME = "tiny-random/gemma-4-moe"
ALL_BLOCKING_MODES = ["h", "q", "kv", "qkv", "hqkv", "kv_headpar"]
HEAD_BLOCK_SIZE = 8
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
GEMMA4_TEST_SLIDING_WINDOW = 16
PREFILL_MDP_NUM_DEVICES = 2
PREFILL_MDP_NUM_PARTITIONS = 2
PREFILL_ONLINE_QL_CHUNK = 32
PREFILL_ONLINE_N_REP_CHUNK = 1


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_lang_only_compile(qeff_model, qpc_paths: dict, qpc_keys: tuple[str, ...]):
    assert any(qpc_paths.get(key) for key in qpc_keys), f"Compile did not return any of: {qpc_keys}"
    assert not qpc_paths.get("vision_qpc_path"), "Vision compile should be skipped"
    assert getattr(qeff_model.vision_model, "onnx_path", None) is None, "Vision export should be skipped"


def _assert_qaic_config_in_export_hash(onnx_path: Path, qaic_config: dict):
    hash_params_path = onnx_path.parent / "hashed_export_params.json"
    assert hash_params_path.is_file(), f"Export hash metadata does not exist: {hash_params_path}"
    hash_params = json.loads(hash_params_path.read_text())
    assert hash_params.get("qaic_config") == qaic_config, (
        f"ONNX export hash does not capture the QAIC config: {hash_params_path}"
    )


def _build_qaic_blocking_config(blocking_mode: str) -> dict:
    cfg = {"blocking_mode": blocking_mode}
    if blocking_mode in ("h", "hqkv"):
        cfg["head_block_size"] = HEAD_BLOCK_SIZE
    if blocking_mode in ("kv", "kv_headpar", "qkv", "hqkv"):
        cfg["num_kv_blocks"] = NUM_KV_BLOCKS
    if blocking_mode in ("q", "qkv", "hqkv"):
        cfg["num_q_blocks"] = NUM_Q_BLOCKS
    return cfg


def _build_gemma4_prefill_mdp_qaic_config(blocking_mode: str, prefill_seq_len: int, ctx_len: int) -> dict:
    qaic_config = {
        "blocking_mode": blocking_mode,
        "ctx_len": ctx_len,
    }
    if blocking_mode == "prefill_online":
        qaic_config.update(
            {
                "num_kv_blocks": NUM_KV_BLOCKS,
                "num_q_blocks": -(-prefill_seq_len // PREFILL_ONLINE_QL_CHUNK),
                "n_rep_chunk": PREFILL_ONLINE_N_REP_CHUNK,
            }
        )
    else:
        qaic_config.update(
            {
                "num_kv_blocks": NUM_KV_BLOCKS,
                "num_q_blocks": NUM_Q_BLOCKS,
            }
        )
    return qaic_config


def _build_gemma4_blocking_test_config(model_name: str) -> AutoConfig:
    model_type = model_config_dict[model_name].get("model_type")
    custom_config = model_config_dict[model_name].get("additional_params", {})
    hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
    hf_config.name_or_path = model_name

    # Ensure full-attention path (where blocking applies) plus sliding-window path.
    if hasattr(hf_config, "text_config"):
        hf_config.text_config.sliding_window = GEMMA4_TEST_SLIDING_WINDOW
        hf_config.text_config.num_kv_shared_layers = 0
        hf_config.text_config.num_hidden_layers = 2
        hf_config.text_config.layer_types = ["sliding_attention", "full_attention"]
    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.num_hidden_layers = 1
    return hf_config


def _load_gemma4_qeff_model_for_compile_only(
    num_hidden_layers: int = 2, include_sliding_attention: bool = False
) -> QEFFAutoModelForImageTextToText:
    hf_config = _build_gemma4_blocking_test_config(MODEL_NAME)
    if hasattr(hf_config, "text_config"):
        hf_config.text_config.num_hidden_layers = num_hidden_layers
        hf_config.text_config.dtype = "float32"
        hf_config.text_config.torch_dtype = torch.float32
        if not include_sliding_attention:
            hf_config.text_config.layer_types = ["full_attention"] * num_hidden_layers
            hf_config.text_config.per_layer_config = None
    hf_config.dtype = "float32"
    hf_config.torch_dtype = torch.float32

    hf_model = load_vlm_model_from_config(hf_config)
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32

    return QEFFAutoModelForImageTextToText(
        copy.deepcopy(hf_model),
        kv_offload=True,
        config=hf_model.config,
        torch_dtype=torch.float32,
        layerwise=False,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("blocking_mode", ["prefill_qkv", "prefill_online"])
def test_gemma4_blocked_prefill_mdp_intersection_compile_and_distinct_prefill_onnx(blocking_mode, manual_cleanup):
    torch.manual_seed(42)
    manual_cleanup, _ = resolve_two_phase_cleanup(manual_cleanup, compile_only=True)
    prefill_seq_len = model_config_dict[MODEL_NAME]["prompt_len"]
    ctx_len = model_config_dict[MODEL_NAME]["ctx_len"]
    qeff_model = _load_gemma4_qeff_model_for_compile_only(num_hidden_layers=2, include_sliding_attention=False)
    qaic_config = _build_gemma4_prefill_mdp_qaic_config(blocking_mode, prefill_seq_len, ctx_len)
    onnx_path = None

    try:
        prefill_qpc_paths = qeff_model.compile(
            batch_size=1,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_cores=16,
            num_devices=PREFILL_MDP_NUM_DEVICES,
            mdp_num_partitions=PREFILL_MDP_NUM_PARTITIONS,
            mdp_strategy="intersection",
            retain_full_kv=True,
            split_model_io=True,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
            use_onnx_subfunctions=False,
            layerwise=False,
            qaic_config=qaic_config,
        )
        onnx_path = _assert_onnx_path(qeff_model.lang_model.onnx_path, blocking_mode)
        _assert_lang_only_compile(qeff_model, prefill_qpc_paths, ("lang_prefill_qpc_path", "lang_qpc_path"))
        _assert_qaic_config_in_export_hash(onnx_path, qaic_config)
    finally:
        if onnx_path is not None:
            manual_cleanup(onnx_path)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("blocking_mode", ALL_BLOCKING_MODES)
def test_gemma4_blocked_decode_compile_and_distinct_decode_onnx(blocking_mode, manual_cleanup):
    torch.manual_seed(42)
    manual_cleanup, _ = resolve_two_phase_cleanup(manual_cleanup, compile_only=True)
    ctx_len = model_config_dict[MODEL_NAME]["ctx_len"]
    qeff_model = _load_gemma4_qeff_model_for_compile_only(num_hidden_layers=2, include_sliding_attention=False)
    qaic_config = _build_qaic_blocking_config(blocking_mode)
    qaic_config["ctx_len"] = ctx_len
    onnx_path = None

    try:
        decode_qpc_paths = qeff_model.compile(
            batch_size=1,
            prefill_seq_len=1,
            ctx_len=ctx_len,
            num_cores=16,
            num_devices=1,
            retain_full_kv=True,
            split_model_io=True,
            prefill_only=False,
            skip_vision=True,
            use_onnx_subfunctions=False,
            layerwise=False,
            qaic_config=qaic_config,
        )
        onnx_path = _assert_onnx_path(qeff_model.lang_model.onnx_path, blocking_mode)
        _assert_lang_only_compile(qeff_model, decode_qpc_paths, ("lang_decode_qpc_path", "lang_qpc_path"))
        _assert_qaic_config_in_export_hash(onnx_path, qaic_config)
    finally:
        if onnx_path is not None:
            manual_cleanup(onnx_path)
