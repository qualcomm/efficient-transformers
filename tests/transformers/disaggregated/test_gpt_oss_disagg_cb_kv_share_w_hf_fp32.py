# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Continuous-batching parity + KV-handoff check for the gpt-oss disaggregated

Two checks, in one on-device session:
  1. KV-handoff correctness: the shared host ``kv_caches`` arrays are inspected
     directly right after each slot's chunked prefill (DMA write) and right after
     the first decode step for both slots (DMA read + write-back), to prove the
     prefill-written rows land in the right slot with no cross-slot contamination
     and that decode only appends the new position without disturbing prior KV.
  2. Token-level parity: full decoded output per CB slot vs HF PyTorch fp32
     ``generate(do_sample=False)`` on the same prompt.

pytest -m "on_qaic and disagg_dma" tests/transformers/disaggregated/test_gpt_oss_disagg_cb_kv_share_w_hf_fp32.py

"""

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from tests.transformers.disaggregated._disagg_dma_config import disagg_dma_configs
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    get_next_token_ids as _get_next_token_ids,
)

MODEL_NAME = "openai/gpt-oss-20b"
NUM_HIDDEN_LAYERS = 2
PREFILL_SEQ_LEN = 32
CTX_LEN = 256
BATCH_SIZE = 1
GENERATION_LEN = 40
FULL_BATCH_SIZE = 2
TEXT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
]

NUM_CORES = DEFAULT_AIC_NUM_CORES
MOE_PREFILL_PACKED_CHUNK_SIZE = 16


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _build_config(dtype: str = "float32", full_model: bool = False, model_name: str = MODEL_NAME):
    """Load the real config; optionally truncate depth for the regular CB test."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not full_model:
        config.num_hidden_layers = NUM_HIDDEN_LAYERS
        config.layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(NUM_HIDDEN_LAYERS)]
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    return config


def _load_hf_model(config, model_name: str = MODEL_NAME) -> AutoModelForCausalLM:
    torch.manual_seed(42)
    if getattr(config, "num_hidden_layers", None) != NUM_HIDDEN_LAYERS:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            attn_implementation="eager",
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
        ).eval()

    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    # Scale weights down so fp32 activations stay small; keeps HF and QAIC numerics close.
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model.eval()


def _run_hf_torch_fp32(model, tokenizer, prompt: str) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    prompt_len = input_ids.shape[-1]
    return outputs[0, prompt_len:].detach().cpu().numpy()


def _prepare_prompt(tokenizer, prompt: str):
    """Tokenise + pad to a multiple of PREFILL_SEQ_LEN; -1 position_ids at pad positions."""
    enc = tokenizer(prompt, return_tensors="np", padding=True)
    prompt_len = enc["input_ids"].shape[1]
    num_chunks = -(prompt_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN

    enc = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    input_ids = enc["input_ids"]
    position_ids = np.where(enc["attention_mask"], np.arange(padded_len), -1)
    return input_ids, position_ids.astype(np.int64), num_chunks, prompt_len


def _next_token_ids_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 2:
        return logits.argmax(axis=-1).astype(np.int64)
    return _get_next_token_ids(logits)


def _prefill_slot(prefill_session, input_ids, position_ids, num_chunks, slot: int, slot_kv_view):
    """Chunked prefill of one prompt into KV ``slot``. Returns (first_token, next_pos)."""
    chunk_inputs = {"batch_index": np.array([[slot]], dtype=np.int64)}
    exec_idx = None
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = input_ids[:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = position_ids[:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        last_chunk = i == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=slot_kv_view if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    first_token = int(_next_token_ids_from_logits(prefill_out["logits"])[0])
    next_pos = int(np.max(position_ids)) + 1
    return first_token, next_pos


@pytest.mark.on_qaic
@pytest.mark.disagg_dma
@pytest.mark.parametrize("dma_config", disagg_dma_configs("gpt_oss_tiny"))
def test_gpt_oss_disagg_cb_kv_handoff_and_hf_parity(manual_cleanup, dma_config):
    torch.manual_seed(42)

    model_id = dma_config["model_id"]
    use_onnx_subfunctions = dma_config.get("use_onnx_subfunctions", True)

    config = _build_config(dtype="float32", model_name=model_id)
    hf_model = _load_hf_model(config, model_name=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_tokens = [_run_hf_torch_fp32(hf_model, tokenizer, prompt) for prompt in TEXT_PROMPTS]

    qeff_model = QEFFAutoModelForCausalLM(hf_model, continuous_batching=True)

    sessions = []
    compiled_onnx_paths = {}
    try:
        decode_qpc_path = qeff_model.compile(
            prefill_seq_len=1,
            ctx_len=CTX_LEN,
            full_batch_size=FULL_BATCH_SIZE,
            num_cores=NUM_CORES,
            num_devices=dma_config["decode_num_devices"],
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            offload_pt_weights=False,
            split_retained_state_io=True,
            retain_full_kv=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
        compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.onnx_path, "decode")

        prefill_qpc_path = qeff_model.compile(
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            full_batch_size=FULL_BATCH_SIZE,
            num_cores=NUM_CORES,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
            num_devices=dma_config["prefill_num_devices"],
            mdp_num_partitions=dma_config["stages"],
            split_retained_state_io=True,
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
        compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.onnx_path, "prefill")
        print(f"Disagg CB ONNX paths: {compiled_onnx_paths}")

        prefill_session = QAICInferenceSession(
            prefill_qpc_path, kv_dma_share=True, full_batch_size=FULL_BATCH_SIZE, cluster_id="prefill"
        )
        decode_session = QAICInferenceSession(
            decode_qpc_path, kv_dma_share=True, full_batch_size=FULL_BATCH_SIZE, cluster_id="decode"
        )
        sessions.extend([prefill_session, decode_session])

        assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"

        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
        assert kv_caches[0].shape[0] == FULL_BATCH_SIZE, (
            f"decode KV batch dim {kv_caches[0].shape[0]} != full_batch_size {FULL_BATCH_SIZE}"
        )
        decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

        # -------------------- Chunked prefill into both slots --------------------
        first_tokens = [None] * FULL_BATCH_SIZE
        next_pos = [None] * FULL_BATCH_SIZE
        prompt_len = [None] * FULL_BATCH_SIZE
        for slot, prompt in enumerate(TEXT_PROMPTS):
            input_ids, position_ids, num_chunks, plen = _prepare_prompt(tokenizer, prompt)
            prompt_len[slot] = plen

            # Pre-condition: this slot's row is still all-zero before its prefill runs.
            assert all(np.all(kv[slot] == 0) for kv in kv_caches), f"slot {slot} KV row is not zero before prefill"

            slot_kv_view = [kv[slot : slot + 1] for kv in kv_caches]
            ft, npos = _prefill_slot(prefill_session, input_ids, position_ids, num_chunks, slot, slot_kv_view)
            first_tokens[slot] = ft
            next_pos[slot] = npos

            written = [kv[slot, :, : prompt_len[slot], :] for kv in kv_caches]
            assert all(np.any(w != 0) for w in written), (
                f"slot {slot} KV row is still zero after prefill -- DMA handoff did not write it"
            )
            for other in range(FULL_BATCH_SIZE):
                if other == slot or first_tokens[other] is None:
                    continue
                assert np.any(kv_caches[0][other] != 0), (
                    f"slot {other} KV row went to zero after prefilling slot {slot} -- cross-slot corruption"
                )

        pre_decode_kv = [kv.copy() for kv in kv_caches]

        # -------------------- First decode step, both slots --------------------
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        input_ids = np.array([[first_tokens[s]] for s in range(FULL_BATCH_SIZE)], dtype=np.int64)
        position_ids = np.array([[next_pos[s]] for s in range(FULL_BATCH_SIZE)], dtype=np.int64)
        batch_index = np.array([[s] for s in range(FULL_BATCH_SIZE)], dtype=np.int64)
        decode_inputs = {"input_ids": input_ids, "position_ids": position_ids, "batch_index": batch_index}
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_out = decode_session.get_outputs(index=exec_idx)
        decode_logits = decode_out["logits"].reshape(FULL_BATCH_SIZE, -1, decode_out["logits"].shape[-1])[:, -1, :]
        second_tokens = np.argmax(decode_logits, axis=-1)

        for slot in range(FULL_BATCH_SIZE):
            for kv_before, kv_after in zip(pre_decode_kv, kv_caches):
                prefix_before = kv_before[slot, :, : prompt_len[slot], :]
                prefix_after = kv_after[slot, :, : prompt_len[slot], :]
                assert np.array_equal(prefix_before, prefix_after), (
                    f"slot {slot}: decode step overwrote prefill-written KV prefix "
                    f"(positions 0..{prompt_len[slot]}) -- handoff/write-back is wiring the wrong offset"
                )
                new_pos_after = kv_after[slot, :, next_pos[slot], :]
                assert np.any(new_pos_after != 0), (
                    f"slot {slot}: decode step did not write KV at the new position {next_pos[slot]} "
                    "-- write-back side of the handoff is not wired"
                )

        # -------------------- Continue decoding to full length --------------------
        gen_tokens = [[first_tokens[s], int(second_tokens[s])] for s in range(FULL_BATCH_SIZE)]
        pos = position_ids + 1
        last_token = second_tokens.reshape(FULL_BATCH_SIZE, 1)
        for _ in range(GENERATION_LEN - 2):
            decode_session.set_data_for_kv_handoff(
                kv_caches + kv_caches,
                [("batch_index", 0), ("ctx_start", 0)],
                index=decode_session.decode_execObj_idx,
                buff_map=decode_kv_map,
            )
            decode_inputs = {
                "input_ids": last_token.astype(np.int64),
                "position_ids": pos.astype(np.int64),
                "batch_index": batch_index,
            }
            exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
            decode_session.complete_inf(exec_idx, is_prefill=False)
            out = decode_session.get_outputs(index=exec_idx)
            logits = out["logits"].reshape(FULL_BATCH_SIZE, -1, out["logits"].shape[-1])[:, -1, :]
            next_tokens = np.argmax(logits, axis=-1)
            for s in range(FULL_BATCH_SIZE):
                gen_tokens[s].append(int(next_tokens[s]))
            last_token = next_tokens.reshape(FULL_BATCH_SIZE, 1)
            pos = pos + 1
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [getattr(qeff_model, "onnx_path", None)]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    for slot in range(FULL_BATCH_SIZE):
        qaic_tokens = np.array(gen_tokens[slot], dtype=np.int64)
        ref_tokens = hf_tokens[slot]
        matches = ref_tokens == qaic_tokens
        num_matched = int(np.cumprod(matches).sum())
        print(f"\nslot[{slot}] prompt: {TEXT_PROMPTS[slot]}")
        print(f"HF Torch fp32 tokens   : {ref_tokens.tolist()}")
        print(f"Disagg CB QAIC tokens  : {qaic_tokens.tolist()}")
        print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")
        if not matches.all():
            first_mismatch = int(np.argmin(matches))
            raise AssertionError(
                f"slot {slot}: tokens don't match HF Torch fp32 output; "
                f"first mismatch at token index {first_mismatch} "
                f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
                f"HF={ref_tokens[first_mismatch]} vs QAIC={qaic_tokens[first_mismatch]}"
            )
