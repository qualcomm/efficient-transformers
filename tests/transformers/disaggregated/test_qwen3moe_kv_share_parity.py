# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""parity: KV-DMA-share disagg path vs HF generate.
pytest -m 'on_qaic and disagg_dma'   tests/transformers/disaggregated/test_qwen3moe_kv_share_parity.py

"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from tests.transformers.disaggregated._disagg_dma_config import disagg_dma_configs

MODEL_ID = "yujiepan/qwen3-moe-tiny-random"
TINY_RANDOM_MODEL_NAMES = {"yujiepan/qwen3-moe-tiny-random"}
PROMPT = "Explain quantum computing in simple terms."
PREFILL_SEQ_LEN = 256
CTX_LEN = PREFILL_SEQ_LEN * 3
NUM_TOKEN_MATCH = 40
STAGES = 2
PREFILL_NUM_DEVICES = 2
DECODE_NUM_DEVICES = 1
FULL_BATCH_SIZE = 1

NUM_CORES = DEFAULT_AIC_NUM_CORES
MOE_PREFILL_PACKED_CHUNK_SIZE = 128

HF_COMPARE_TOKENS = int(os.environ.get("QEFF_QWEN3MOE_HF_COMPARE_TOKENS", NUM_TOKEN_MATCH))
HF_MIN_LEADING_MATCH = int(os.environ.get("QEFF_QWEN3MOE_HF_MIN_MATCH", 20))
NUM_HIDDEN_LAYERS = int(os.environ.get("QEFF_QWEN3MOE_NUM_HIDDEN_LAYERS", 4))


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _build_config(model_id: str, num_hidden_layers: int | None):
    if num_hidden_layers is None or model_id in TINY_RANDOM_MODEL_NAMES:
        return None
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = num_hidden_layers
    return config


def _run_hf_greedy_reference(model_id: str, compare_tokens: int) -> list:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = _build_config(model_id, NUM_HIDDEN_LAYERS)
    from_pretrained_kwargs = {"config": config} if config is not None else {}

    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", torch_dtype=torch.float32, **from_pretrained_kwargs
    ).eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            max_new_tokens=compare_tokens,
            min_new_tokens=compare_tokens,
            do_sample=False,
        )
    return sequences[0, prompt_len:].tolist()


def _compile_sessions(
    qeff_model,
    onnx_paths: dict,
    *,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
    stages: int = STAGES,
) -> tuple[QAICInferenceSession, QAICInferenceSession]:
    """Compile the DMA KV-share prefill/decode QPCs (split_retained_state_io + retain_full_kv)."""
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        full_batch_size=FULL_BATCH_SIZE,
        num_cores=NUM_CORES,
        num_devices=decode_num_devices,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,
        split_retained_state_io=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
    )
    onnx_paths["decode"] = _assert_onnx_path(qeff_model.onnx_path, "decode")

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=FULL_BATCH_SIZE,
        num_cores=NUM_CORES,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
        num_devices=prefill_num_devices,
        mdp_num_partitions=stages,
        split_retained_state_io=True,
        retain_full_kv=True,
        mos=1,
        user_tiled=True,
        aic_enable_depth_first=False,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
        use_onnx_subfunctions=True,
    )
    onnx_paths["prefill"] = _assert_onnx_path(qeff_model.onnx_path, "prefill")

    prefill_session = QAICInferenceSession(
        prefill_qpc_path, kv_dma_share=True, full_batch_size=FULL_BATCH_SIZE, cluster_id="prefill"
    )
    decode_session = QAICInferenceSession(
        decode_qpc_path, kv_dma_share=True, full_batch_size=FULL_BATCH_SIZE, cluster_id="decode"
    )
    return prefill_session, decode_session


def _prepare_prompt(tokenizer, prompt: str):
    enc = tokenizer(prompt, return_tensors="np", padding=True)
    prompt_len = enc["input_ids"].shape[1]
    num_chunks = -(prompt_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN

    enc = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    lang_inputs = {"input_ids": enc["input_ids"]}
    lang_inputs["position_ids"] = np.where(enc["attention_mask"], np.arange(padded_len), -1)
    return lang_inputs, num_chunks


def _prefill_slot(prefill_session, kv_caches: list, lang_inputs: dict, num_chunks: int, slot: int):
    """Chunked prefill of one prompt into KV ``slot``. Returns (first_token, next_pos)."""
    chunk_inputs = {"batch_index": np.array([[slot]], dtype=np.int64)}
    slot_kv_view = [kv[slot : slot + 1] for kv in kv_caches]
    exec_idx = None
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        last_chunk = i == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=slot_kv_view if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    first_token = int(np.argmax(prefill_out["logits"]))
    next_pos = int(np.max(lang_inputs["position_ids"])) + 1
    return first_token, next_pos


def _run_disagg_kv_share_qaic_generation(
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
    tokenizer,
    compare_tokens: int,
) -> list:
    """Chunked prefill + single-slot DMA-share decode loop for ``PROMPT``."""
    assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"

    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    assert kv_caches and kv_caches[0].shape[0] == FULL_BATCH_SIZE, (
        f"decode KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != full_batch_size {FULL_BATCH_SIZE}"
    )
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    lang_inputs, num_chunks = _prepare_prompt(tokenizer, PROMPT)
    first_token, next_pos = _prefill_slot(prefill_session, kv_caches, lang_inputs, num_chunks, slot=0)

    tokens = [first_token]
    last_token = first_token
    pos = next_pos
    batch_index = np.array([[0]], dtype=np.int64)
    for _ in range(compare_tokens - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = {
            "input_ids": np.array([[last_token]], dtype=np.int64),
            "position_ids": np.array([[pos]], dtype=np.int64),
            "batch_index": batch_index,
        }
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        logits = out["logits"].reshape(FULL_BATCH_SIZE, -1, out["logits"].shape[-1])[:, -1, :]
        last_token = int(np.argmax(logits, axis=-1)[0])
        tokens.append(last_token)
        pos += 1

    return tokens


@pytest.mark.skip("local test")
@pytest.mark.on_qaic
def test_qwen3moe_kv_share_kv_handoff_correctness(manual_cleanup):
    """KV-handoff correctness for the qwen3moe DMA path: the shared host ``kv_caches``
    arrays are inspected right after the last chunked-prefill DMA write and right after
    the first decode step, to prove the last prefill chunk's KV is exactly what decode
    reads as its input KV -- decode's write-back must only append the new position
    without disturbing the prefill-written prefix.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = _build_config(MODEL_ID, NUM_HIDDEN_LAYERS)
    from_pretrained_kwargs = {"config": config} if config is not None else {}
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(MODEL_ID, continuous_batching=True, **from_pretrained_kwargs)

    sessions = []
    compiled_onnx_paths = {}
    try:
        prefill_session, decode_session = _compile_sessions(qeff_model, compiled_onnx_paths)
        sessions.extend([prefill_session, decode_session])
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"

        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
        assert kv_caches and kv_caches[0].shape[0] == FULL_BATCH_SIZE, (
            f"decode KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != full_batch_size {FULL_BATCH_SIZE}"
        )
        assert all(np.all(kv == 0) for kv in kv_caches), "KV caches are not zero-initialised before prefill"

        lang_inputs, num_chunks = _prepare_prompt(tokenizer, PROMPT)
        prompt_len = int(np.sum(lang_inputs["position_ids"] != -1))

        # -------------------- Chunked prefill --------------------
        first_token, next_pos = _prefill_slot(prefill_session, kv_caches, lang_inputs, num_chunks, slot=0)

        # Post-condition: the last chunk's DMA write landed the real prompt prefix.
        written = [kv[:, :, :prompt_len, :] for kv in kv_caches]
        assert all(np.any(w != 0) for w in written), (
            "KV caches are still zero after the last prefill chunk -- DMA handoff did not write them"
        )

        pre_decode_kv = [kv.copy() for kv in kv_caches]
        # -------------------- First decode step --------------------
        decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = {
            "input_ids": np.array([[first_token]], dtype=np.int64),
            "position_ids": np.array([[next_pos]], dtype=np.int64),
            "batch_index": np.array([[0]], dtype=np.int64),
        }
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_session.get_outputs(index=exec_idx)

        for kv_before, kv_after in zip(pre_decode_kv, kv_caches):
            prefix_before = kv_before[:, :, :prompt_len, :]
            prefix_after = kv_after[:, :, :prompt_len, :]
            assert np.array_equal(prefix_before, prefix_after), (
                "decode step overwrote the prefill-written KV prefix -- the last prefill "
                "chunk's KV no longer matches the input KV the first decode step read"
            )
            new_pos_after = kv_after[:, :, next_pos, :]
            assert np.any(new_pos_after != 0), (
                f"decode step did not write KV at the new position {next_pos} -- "
                "write-back side of the handoff is not wired"
            )
    finally:
        for session in sessions:
            session.deactivate()
        manual_cleanup([path for path in compiled_onnx_paths.values() if path is not None])


@pytest.mark.on_qaic
@pytest.mark.disagg_dma
@pytest.mark.parametrize("dma_config", disagg_dma_configs("qwen3moe_tiny"))
def test_kv_share_matches_hf_generate_leading_tokens(manual_cleanup, dma_config):
    compare_tokens = HF_COMPARE_TOKENS
    model_id = dma_config["model_id"]
    skip_hf_reference = dma_config.get("skip_hf_reference", False)

    hf_tokens = None if skip_hf_reference else _run_hf_greedy_reference(model_id, compare_tokens)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = _build_config(model_id, NUM_HIDDEN_LAYERS)
    from_pretrained_kwargs = {"config": config} if config is not None else {}
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, continuous_batching=True, **from_pretrained_kwargs)

    sessions = []
    compiled_onnx_paths = {}
    try:
        prefill_session, decode_session = _compile_sessions(
            qeff_model,
            compiled_onnx_paths,
            prefill_num_devices=dma_config["prefill_num_devices"],
            decode_num_devices=dma_config["decode_num_devices"],
            stages=dma_config["stages"],
        )
        sessions.extend([prefill_session, decode_session])
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        qaic_tokens = _run_disagg_kv_share_qaic_generation(prefill_session, decode_session, tokenizer, compare_tokens)
    finally:
        for session in sessions:
            session.deactivate()
        manual_cleanup([path for path in compiled_onnx_paths.values() if path is not None])

    print(f"Disagg QAIC DMA tokens : {qaic_tokens}")

    if skip_hf_reference:
        return

    n = min(compare_tokens, len(hf_tokens), len(qaic_tokens))
    assert n > 0, "no tokens to compare"

    # Length of the leading run that matches token-for-token.
    matched = 0
    for hf_tok, qaic_tok in zip(hf_tokens[:n], qaic_tokens[:n]):
        if hf_tok != qaic_tok:
            break
        matched += 1

    print(f"HF Torch fp32 tokens   : {hf_tokens}")
    print(f"Matched leading tokens : {matched}/{n}")

    assert qaic_tokens[0] == hf_tokens[0], f"first token mismatch: kv_share={qaic_tokens[0]} hf={hf_tokens[0]}"
    assert matched >= min(HF_MIN_LEADING_MATCH, n), (
        f"disagg DMA output diverged from HF generate after only {matched} tokens "
        f"(required >= {min(HF_MIN_LEADING_MATCH, n)} of {n}):\n"
        f"  hf   ={hf_tokens[:n]}\n  qaic ={qaic_tokens[:n]}"
    )
