# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Continuous-batching disaggregated prefill/decode for gpt-oss — DMA KV handoff."""

import argparse
from collections import deque
from time import perf_counter

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

# NOTE: 120b-requires NPI yaml
DEFAULT_SUBFUNC_NPI = None
DEFAULT_NON_SUBFUNC_NPI = None
DEFAULT_MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Name three primary colors.",
]
DEFAULT_PREFILL_SEQ_LEN = 512
DEFAULT_CTX_LEN = DEFAULT_PREFILL_SEQ_LEN * 2
DEFAULT_GENERATION_LEN = 200
DEFAULT_FULL_BATCH_SIZE = 4

NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
STAGES = 4
PREFILL_NUM_DEVICES = 8
DECODE_NUM_DEVICES = 4


def _build_config(model_id: str, num_hidden_layers: int = None):
    """Load the model config, optionally reducing ``num_hidden_layers``."""
    if num_hidden_layers is None:
        return None
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = num_hidden_layers
    return config


def _compile_sessions(
    qeff_model,
    prefill_seq_len,
    ctx_len,
    full_batch_size,
    stages,
    prefill_num_devices,
    decode_num_devices,
    subfunc_npi,
    non_subfunc_npi,
):
    """Compile decode/prefill QPCs (CB flags) and open kv_dma_share sessions."""
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_cores=NUM_CORES,
        num_devices=decode_num_devices,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation
        split_retained_state_io=True,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        use_onnx_subfunctions=True,
    )
    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_cores=NUM_CORES,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
        num_devices=prefill_num_devices,
        mdp_num_partitions=stages,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
    )
    prefill_session = QAICInferenceSession(
        prefill_qpc_path, kv_dma_share=True, stages=stages, full_batch_size=full_batch_size, cluster_id="prefill"
    )
    decode_session = QAICInferenceSession(
        decode_qpc_path, kv_dma_share=True, full_batch_size=full_batch_size, cluster_id="decode"
    )
    return prefill_session, decode_session


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompts=None,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    generation_len: int = DEFAULT_GENERATION_LEN,
    full_batch_size: int = DEFAULT_FULL_BATCH_SIZE,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
    num_hidden_layers: int = None,
    subfunc_npi: str = DEFAULT_SUBFUNC_NPI,
    non_subfunc_npi: str = DEFAULT_NON_SUBFUNC_NPI,
):
    """Run CB (chunked-prefill + batched decode) over ``prompts`` with the DMA KV handoff."""
    prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = _build_config(model_id, num_hidden_layers)
    from_pretrained_kwargs = {"config": config} if config is not None else {}
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, continuous_batching=True, **from_pretrained_kwargs)
    prefill_session, decode_session = _compile_sessions(
        qeff_model,
        prefill_seq_len,
        ctx_len,
        full_batch_size,
        stages,
        prefill_num_devices,
        decode_num_devices,
        subfunc_npi,
        non_subfunc_npi,
    )

    assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"

    # Shared host KV arrays, allocated once in decode-map order. Under CB the leading batch
    # dim is full_batch_size, so each family is [N, ...]: prefill writes one row, decode
    # reads/writes all N. gpt-oss is hybrid, so families carry a mix of shapes.
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    assert kv_caches and kv_caches[0].shape[0] == full_batch_size, (
        f"decode KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != full_batch_size {full_batch_size}"
    )
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    def _prepare_prompt(prompt: str):
        """Tokenise one prompt, padded to a multiple of ``prefill_seq_len``.

        Returns ``(lang_inputs, num_chunks)`` where ``lang_inputs`` carries ``input_ids`` /
        ``position_ids`` (``-1`` at pad positions).
        """
        enc = tokenizer(prompt, return_tensors="np", padding=True)
        prompt_len = enc["input_ids"].shape[1]
        num_chunks = -(prompt_len // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        enc = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        lang_inputs = {"input_ids": enc["input_ids"]}
        lang_inputs["position_ids"] = np.where(enc["attention_mask"], np.arange(padded_len), -1)
        return lang_inputs, num_chunks

    def _prefill_slot(lang_inputs, num_chunks, slot: int):
        """Chunked prefill of one prompt into KV ``slot``.

        Every chunk carries ``batch_index=slot`` so the on-device scatter accumulates into
        row ``slot``; the last chunk wires the DMA handoff of that single row into
        ``kv_caches[*][slot]``. Returns ``(first_token, next_pos)``.
        """
        chunk_inputs = {"batch_index": np.array([[slot]], dtype=np.int64)}
        slot_kv_view = [kv[slot : slot + 1] for kv in kv_caches]
        exec_idx = None
        for i in range(num_chunks):
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                :, i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
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

    ongoing = [False] * full_batch_size
    last_token = [0] * full_batch_size
    pos = [0] * full_batch_size
    gen_count = [0] * full_batch_size
    slot_prompt_idx = [-1] * full_batch_size
    slot_tokens = [None] * full_batch_size
    results = [None] * len(prompts)

    def _seed_slot(slot, prompt_idx, first_token, next_pos):
        slot_prompt_idx[slot] = prompt_idx
        slot_tokens[slot] = [first_token]
        gen_count[slot] = 1
        last_token[slot] = first_token
        pos[slot] = next_pos
        ongoing[slot] = True

    # Prompt queue: each entry is (prompt_idx, prompt). Everything beyond the first N slots
    # waits here and refills on completion.
    prompt_queue = deque(enumerate(prompts))

    prefill_start = perf_counter()
    for slot in range(full_batch_size):
        if not prompt_queue:
            break
        prompt_idx, prompt = prompt_queue.popleft()
        lang_inputs, num_chunks = _prepare_prompt(prompt)
        ft, next_pos = _prefill_slot(lang_inputs, num_chunks, slot)
        _seed_slot(slot, prompt_idx, ft, next_pos)
    print(f"Initial prefill time : {perf_counter() - prefill_start:.2f} secs")

    def _build_decode_inputs():
        input_ids = np.full((full_batch_size, 1), -1, dtype=np.int64)
        position_ids = np.full((full_batch_size, 1), -1, dtype=np.int64)
        batch_index = np.full((full_batch_size, 1), -1, dtype=np.int64)
        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            input_ids[slot, 0] = last_token[slot]
            position_ids[slot, 0] = pos[slot]
            batch_index[slot, 0] = slot
        return {"input_ids": input_ids, "position_ids": position_ids, "batch_index": batch_index}

    st = perf_counter()
    decode_steps = 0
    while any(ongoing):
        # Wire the full [N, ...] KV buffers once (identity: device row i <-> host row i);
        # per-slot addressing is carried by the decode batch_index input above.
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = _build_decode_inputs()
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        decode_steps += 1

        logits = out["logits"]
        logits = logits.reshape(full_batch_size, -1, logits.shape[-1])[:, -1, :]
        next_tokens = np.argmax(logits, axis=-1)

        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            tok = int(next_tokens[slot])
            if tok == tokenizer.eos_token_id or gen_count[slot] >= generation_len:
                # Slot finished: record its output, then refill from the queue or retire.
                results[slot_prompt_idx[slot]] = slot_tokens[slot]
                if prompt_queue:
                    prompt_idx, prompt = prompt_queue.popleft()
                    lang_inputs, num_chunks = _prepare_prompt(prompt)
                    ft, next_pos = _prefill_slot(lang_inputs, num_chunks, slot)
                    _seed_slot(slot, prompt_idx, ft, next_pos)
                else:
                    ongoing[slot] = False
            else:
                slot_tokens[slot].append(tok)
                gen_count[slot] += 1
                last_token[slot] = tok
                pos[slot] += 1
    ft = perf_counter()

    total_tokens = sum(len(t) for t in results if t)
    print(f"decode steps={decode_steps} tok/sec={total_tokens / (ft - st):.2f}")
    first_tokens = []
    for idx, prompt in enumerate(prompts):
        toks = results[idx] or []
        first_tokens.append(toks[0] if toks else None)
        print(f"\ninput [{idx}]\n{prompt}\noutput\n{tokenizer.decode(toks)}")

    return {"first_tokens": first_tokens, "tokens": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--prompt", action="append", dest="prompts", help="prompt (repeatable); defaults to a set of 4")
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument("--full-batch-size", type=int, default=DEFAULT_FULL_BATCH_SIZE, help="CB decode width (N)")
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="reduce model depth for a fast compile (testing only; outputs not meaningful)",
    )
    parser.add_argument("--subfunc-npi", default=DEFAULT_SUBFUNC_NPI, help="prefill (subfunction) NPI yaml path")
    parser.add_argument(
        "--non-subfunc-npi", default=DEFAULT_NON_SUBFUNC_NPI, help="decode (non-subfunction) NPI yaml path"
    )
    args = parser.parse_args()

    run(
        model_id=args.model_id,
        prompts=args.prompts,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        full_batch_size=args.full_batch_size,
        stages=args.stages,
        prefill_num_devices=args.prefill_num_devices,
        decode_num_devices=args.decode_num_devices,
        num_hidden_layers=args.num_hidden_layers,
        subfunc_npi=args.subfunc_npi,
        non_subfunc_npi=args.non_subfunc_npi,
    )
