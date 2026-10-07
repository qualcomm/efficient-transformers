# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import math
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def _expand_batch(inputs, batch_size: int):
    """Repeat single-prompt tokenizer tensors for the compiled execution batch."""
    expanded = {}
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            expanded[name] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            expanded[name] = value
    return expanded


def _execution_batch_size(batch_size: int, msa_indexer_dp: int, msa_attn_dp: int) -> int:
    return batch_size * math.lcm(msa_indexer_dp, msa_attn_dp)


def _build_qaic_config(
    *,
    msa_indexer_dp: int,
    msa_indexer_cp: int,
    msa_attn_dp: int,
    msa_attn_cp: int,
    indexer_n_head: int,
    indexer_prefill_parallel: bool,
    indexer_q_chunk: int | None,
    indexer_q_size: int | None,
    num_cores: int,
    msa_q_chunk: int,
    expert_parallel_chunk_size: int,
    cores_per_expert: int,
    tree_reduce: bool,
) -> dict:
    """Build the MiniMax blocking config for one phase (prefill or decode)."""
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": 2,
        "msa_indexer_dp": msa_indexer_dp,
        "msa_indexer_cp": msa_indexer_cp,
        "msa_attn_dp": msa_attn_dp,
        "msa_attn_cp": msa_attn_cp,
        "indexer_n_head": indexer_n_head,
        "indexer_prefill_parallel": indexer_prefill_parallel,
        "num_cores_per_device": num_cores,
        "msa_q_chunk": msa_q_chunk,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": expert_parallel_chunk_size,
            "cores_per_expert": cores_per_expert,
            "tree_reduce": tree_reduce,
        },
    }
    if indexer_q_chunk is not None:
        qaic_config["indexer_q_chunk"] = indexer_q_chunk
    if indexer_q_size is not None:
        qaic_config["indexer_q_size"] = indexer_q_size
    return qaic_config


def _update_retained_cache_inputs(
    inputs: dict,
    outputs: dict,
    num_layers: int,
    target_shapes: dict[str, tuple[int, ...]] | None = None,
) -> None:
    """Carry retained KV and sparse-indexer states between QPC invocations."""
    for layer_idx in range(num_layers):
        for cache_name in ("past_key", "past_value", "index_key"):
            retained_name = f"{cache_name}.{layer_idx}_RetainedState"
            if retained_name in outputs:
                input_name = f"{cache_name}.{layer_idx}"
                retained = outputs[retained_name]
                if target_shapes is not None and input_name in target_shapes:
                    target_shape = target_shapes[input_name]
                    if retained.size != math.prod(target_shape):
                        raise ValueError(
                            f"Cannot hand off {retained_name} with shape {retained.shape} to "
                            f"decode input shape {target_shape}."
                        )
                    retained = retained.reshape(target_shape)
                inputs[input_name] = retained


def main():
    parser = argparse.ArgumentParser(
        description="Export and compile separate MiniMax-M3 prefill and decode QPCs for disaggregated serving."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument(
        "--decode-num-devices",
        type=int,
        default=None,
        help="Number of devices used by the decode QPC; defaults to --num-devices.",
    )
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument(
        "--mdp-num-partitions",
        "--prefill-mdp-num-partitions",
        dest="prefill_mdp_num_partitions",
        type=int,
        default=16,
        help=(
            "Number of pipeline-parallel MDP partitions for the prefill QPC only. "
            "Tensor-slice devices per stage are num_devices / mdp_num_partitions."
        ),
    )
    parser.add_argument(
        "--prefill-seq-len",
        type=int,
        default=128,
        help="Prompt-token specialization length; must be greater than 1 for MSA prefill.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Logical prompt batch size; QEfficient expands it by the DP LCM for execution.",
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument(
        "--expert-parallel-chunk-size",
        type=int,
        default=256,
        help="MoE expert-parallel chunk size (expert_parallel_chunk_size in moe_config).",
    )
    parser.add_argument(
        "--cores-per-expert",
        type=int,
        default=2,
        help="Number of NSP cores assigned to each expert during decode.",
    )
    parser.add_argument(
        "--no-tree-reduce",
        dest="tree_reduce",
        action="store_false",
        default=True,
        help="Disable tree-reduce for MoE expert-parallel dispatch.",
    )
    parser.add_argument("--prefill-msa-indexer-dp", type=int, default=1, help="Prefill MSA indexer DP factor.")
    parser.add_argument("--prefill-msa-indexer-cp", type=int, default=1, help="Prefill MSA indexer CP factor.")
    parser.add_argument("--prefill-msa-attn-dp", type=int, default=1, help="Prefill MSA attention DP factor.")
    parser.add_argument("--prefill-msa-attn-cp", type=int, default=1, help="Prefill MSA attention CP factor.")
    parser.add_argument(
        "--msa-indexer-dp",
        "--decode-msa-indexer-dp",
        dest="decode_msa_indexer_dp",
        type=int,
        default=1,
        help="Decode MSA indexer DP factor (legacy --msa-indexer-dp alias).",
    )
    parser.add_argument(
        "--msa-indexer-cp",
        "--decode-msa-indexer-cp",
        dest="decode_msa_indexer_cp",
        type=int,
        default=1,
        help="Decode MSA indexer CP factor (legacy --msa-indexer-cp alias).",
    )
    parser.add_argument(
        "--msa-attn-dp",
        "--decode-msa-attn-dp",
        dest="decode_msa_attn_dp",
        type=int,
        default=1,
        help="Decode MSA attention DP factor (legacy --msa-attn-dp alias).",
    )
    parser.add_argument(
        "--msa-attn-cp",
        "--decode-msa-attn-cp",
        dest="decode_msa_attn_cp",
        type=int,
        default=1,
        help="Decode MSA attention CP factor (legacy --msa-attn-cp alias).",
    )
    parser.add_argument(
        "--indexer-n-head",
        type=int,
        default=1,
        help="Number of KV heads used by the MSA indexer in the DP path.",
    )
    parser.add_argument(
        "--indexer-prefill-parallel",
        dest="indexer_prefill_parallel",
        action="store_true",
        default=False,
        help="Use the parallel indexer prefill selector.",
    )
    parser.add_argument(
        "--indexer-q-chunk",
        type=int,
        default=None,
        help="Outer query chunk size for MSA indexer prefill; defaults to the indexer query block size.",
    )
    parser.add_argument(
        "--indexer-q-size",
        type=int,
        default=None,
        help="Query micro-block size for MSA indexer prefill; defaults to the model index block size.",
    )
    parser.add_argument(
        "--msa-q-chunk",
        type=int,
        default=64,
        help="MSA prefill attention query chunk size.",
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=16,
        help="Number of prefill KV blocks.",
    )
    parser.add_argument(
        "--q-block-size",
        type=int,
        default=256,
        help="Prefill query block size within the outer query chunk.",
    )
    parser.add_argument(
        "--q-block-chunk",
        type=int,
        default=1024,
        help="Outer prefill query chunk size used to derive n_rep_chunk.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.decode_num_devices is not None and args.decode_num_devices < 1:
        parser.error("--decode-num-devices must be positive")
    if args.prefill_seq_len <= 1:
        parser.error("--prefill-seq-len must be greater than 1")
    if args.num_kv_blocks < 1:
        parser.error("--num-kv-blocks must be positive")
    if args.q_block_size < 1:
        parser.error("--q-block-size must be positive")
    if args.q_block_chunk < 1:
        parser.error("--q-block-chunk must be positive")
    if args.q_block_chunk < args.q_block_size or args.q_block_chunk % args.q_block_size:
        parser.error("--q-block-chunk must be at least and divisible by --q-block-size")
    if args.prefill_mdp_num_partitions < 1:
        parser.error("--mdp-num-partitions must be positive")
    if args.prefill_mdp_num_partitions > args.num_devices:
        parser.error("--mdp-num-partitions must not exceed --num-devices")
    if args.num_devices % args.prefill_mdp_num_partitions:
        parser.error("--num-devices must be divisible by --mdp-num-partitions")
    if args.indexer_q_chunk is not None and args.indexer_q_chunk < 1:
        parser.error("--indexer-q-chunk must be positive")
    if args.indexer_q_size is not None and args.indexer_q_size < 1:
        parser.error("--indexer-q-size must be positive")
    if (
        args.indexer_q_chunk is not None
        and args.indexer_q_size is not None
        and (args.indexer_q_chunk < args.indexer_q_size or args.indexer_q_chunk % args.indexer_q_size)
    ):
        parser.error("--indexer-q-chunk must be at least and divisible by --indexer-q-size")
    phase_values = (
        args.prefill_msa_indexer_dp,
        args.prefill_msa_indexer_cp,
        args.prefill_msa_attn_dp,
        args.prefill_msa_attn_cp,
        args.decode_msa_indexer_dp,
        args.decode_msa_indexer_cp,
        args.decode_msa_attn_dp,
        args.decode_msa_attn_cp,
    )
    if any(value < 1 for value in phase_values):
        parser.error("All MSA DP/CP factors must be positive")
    if args.prefill_msa_indexer_dp != 1 or args.prefill_msa_indexer_cp != 1:
        parser.error("MiniMax MSA prefill currently requires prefill indexer DP=1 and CP=1")
    if args.prefill_msa_attn_dp != 1:
        parser.error("MiniMax MSA prefill currently requires prefill attention DP=1")
    if args.prefill_msa_attn_cp != 1:
        parser.error("MiniMax MSA prefill currently requires prefill attention CP=1")

    # Both QPCs must use the same physical execution batch for the explicit KV handoff.
    execution_batch_size = args.batch_size * math.lcm(
        args.prefill_msa_indexer_dp,
        args.prefill_msa_attn_dp,
        args.decode_msa_indexer_dp,
        args.decode_msa_attn_dp,
    )
    decode_num_devices = args.num_devices if args.decode_num_devices is None else args.decode_num_devices
    num_q_blocks = max(1, math.ceil(args.prefill_seq_len / args.q_block_size))
    n_rep_chunk = args.q_block_chunk // args.q_block_size

    factory_kwargs = dict(kv_offload=True, dtype=torch.float16)
    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers
    factory_kwargs["config"] = config

    t0 = time.perf_counter()
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_id, **factory_kwargs)
    print(f"[timing] model load:          {time.perf_counter() - t0:.2f}s")

    common_compile_kwargs = dict(
        batch_size=execution_batch_size,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_onnx_subfunctions=True,
        skip_vision=True,
        offload_pt_weights=False,
        node_precision_info=True,
        log_times=True,
        retain_full_kv=True,
        split_model_io=True,
    )

    prefill_qaic_config = _build_qaic_config(
        msa_indexer_dp=args.prefill_msa_indexer_dp,
        msa_indexer_cp=args.prefill_msa_indexer_cp,
        msa_attn_dp=args.prefill_msa_attn_dp,
        msa_attn_cp=args.prefill_msa_attn_cp,
        indexer_n_head=args.indexer_n_head,
        indexer_prefill_parallel=args.indexer_prefill_parallel,
        indexer_q_chunk=args.indexer_q_chunk,
        indexer_q_size=args.indexer_q_size,
        num_cores=args.num_cores,
        msa_q_chunk=args.msa_q_chunk,
        expert_parallel_chunk_size=args.expert_parallel_chunk_size,
        cores_per_expert=args.cores_per_expert,
        tree_reduce=args.tree_reduce,
    )
    prefill_qaic_config.update(
        blocking_mode="prefill_online",
        num_kv_blocks=args.num_kv_blocks,
        num_q_blocks=num_q_blocks,
        n_rep_chunk=n_rep_chunk,
        ctx_len=args.ctx_len,
    )
    prefill_compile_kwargs = dict(
        **common_compile_kwargs,
        num_devices=args.num_devices,
        qaic_config=prefill_qaic_config,
    )
    decode_compile_kwargs = dict(
        **common_compile_kwargs,
        num_devices=decode_num_devices,
        qaic_config=_build_qaic_config(
            msa_indexer_dp=args.decode_msa_indexer_dp,
            msa_indexer_cp=args.decode_msa_indexer_cp,
            msa_attn_dp=args.decode_msa_attn_dp,
            msa_attn_cp=args.decode_msa_attn_cp,
            indexer_n_head=args.indexer_n_head,
            indexer_prefill_parallel=False,
            indexer_q_chunk=None,
            indexer_q_size=None,
            num_cores=args.num_cores,
            msa_q_chunk=args.msa_q_chunk,
            expert_parallel_chunk_size=args.expert_parallel_chunk_size,
            cores_per_expert=args.cores_per_expert,
            tree_reduce=args.tree_reduce,
        ),
    )

    t0 = time.perf_counter()
    prefill_qpc_paths = qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        prefill_only=True,
        enable_chunking=True,
        mdp_num_partitions=args.prefill_mdp_num_partitions,
        **prefill_compile_kwargs,
    )
    prefill_qpc_path = prefill_qpc_paths["lang_prefill_qpc_path"]
    print(f"[timing] prefill export + compile: {time.perf_counter() - t0:.2f}s")
    print(f"Prefill QPC path: {prefill_qpc_path}")

    t0 = time.perf_counter()
    decode_qpc_paths = qeff_model.compile(
        prefill_seq_len=1,
        prefill_only=False,
        **decode_compile_kwargs,
    )
    decode_qpc_path = decode_qpc_paths["lang_decode_qpc_path"]
    print(f"[timing] decode export + compile:  {time.perf_counter() - t0:.2f}s")
    print(f"Decode QPC path: {decode_qpc_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    messages = [[{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]]
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    model_inputs = _expand_batch(model_inputs, execution_batch_size)
    model_inputs = qeff_model.model.prepare_inputs_for_generation(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs.get("attention_mask"),
        position_ids=model_inputs.get("position_ids"),
        prefill_seq_len=args.prefill_seq_len,
        batch_size=execution_batch_size,
    )
    input_len = model_inputs["input_ids"].shape[1]
    num_chunks = (input_len + args.prefill_seq_len - 1) // args.prefill_seq_len
    padded_len = num_chunks * args.prefill_seq_len
    pad_len = padded_len - input_len
    model_inputs["input_ids"] = torch.nn.functional.pad(model_inputs["input_ids"], (0, pad_len), value=0)
    if "attention_mask" in model_inputs:
        model_inputs["attention_mask"] = torch.nn.functional.pad(model_inputs["attention_mask"], (0, pad_len), value=0)
    if "attention_mask" in model_inputs:
        model_inputs["position_ids"] = torch.where(
            model_inputs["attention_mask"].bool(),
            torch.arange(padded_len).unsqueeze(0),
            torch.full((execution_batch_size, padded_len), -1),
        )
    elif "position_ids" not in model_inputs:
        model_inputs["position_ids"] = torch.arange(padded_len).unsqueeze(0).expand(execution_batch_size, -1)
    elif model_inputs["position_ids"].shape[-1] != padded_len:
        model_inputs["position_ids"] = torch.nn.functional.pad(model_inputs["position_ids"], (0, pad_len), value=-1)
    np_inputs = {key: value.detach().cpu().numpy() for key, value in model_inputs.items() if torch.is_tensor(value)}
    np_inputs.pop("attention_mask", None)
    prefill_session = QAICInferenceSession(prefill_qpc_path)
    prefill_state = np_inputs.copy()
    prefill_start = time.perf_counter()
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * args.prefill_seq_len
        chunk_end = chunk_start + args.prefill_seq_len
        prefill_state["input_ids"] = np_inputs["input_ids"][:, chunk_start:chunk_end]
        prefill_state["position_ids"] = np_inputs["position_ids"][:, chunk_start:chunk_end]
        prefill_output = prefill_session.run(prefill_state)
        _update_retained_cache_inputs(prefill_state, prefill_output, config.text_config.num_hidden_layers)
    ttft_seconds = time.perf_counter() - prefill_start
    prefill_session.deactivate()
    decode_session = QAICInferenceSession(decode_qpc_path)
    decode_session.activate()
    decode_input_shapes = {
        name: tuple(decode_session.bindings[decode_session.binding_index_map[name]].dims)
        for name in decode_session.input_names
    }
    last_position = np.max(np_inputs["position_ids"], axis=-1, keepdims=True)
    # MiniMax exports prefill logits for the last valid position as [batch, 1, vocab].
    next_tokens = np.argmax(prefill_output["logits"], axis=-1).astype(np_inputs["input_ids"].dtype)
    decode_inputs = {"input_ids": next_tokens, "position_ids": last_position + 1}
    _update_retained_cache_inputs(
        decode_inputs,
        prefill_output,
        config.text_config.num_hidden_layers,
        target_shapes=decode_input_shapes,
    )
    generated = [next_tokens]
    decode_start = time.perf_counter()
    for _ in range(max(0, args.generation_len - 1)):
        decode_output = decode_session.run(decode_inputs)
        next_tokens = np.argmax(decode_output["logits"], axis=-1).astype(np_inputs["input_ids"].dtype)
        generated.append(next_tokens)
        decode_inputs["input_ids"] = next_tokens
        decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
        _update_retained_cache_inputs(decode_inputs, decode_output, config.text_config.num_hidden_layers)
    decode_seconds = time.perf_counter() - decode_start
    generated_ids = np.concatenate(generated, axis=1)
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    decode_token_count = generated_ids.shape[1] - 1
    decode_tok_per_sec = decode_token_count / decode_seconds if decode_seconds > 0 else 0.0
    print(f"[performance] TTFT: {ttft_seconds * 1000.0:.2f} ms")
    if decode_token_count:
        print(f"[performance] decode tok/sec: {decode_tok_per_sec:.2f} ({decode_token_count} tokens)")
    else:
        print("[performance] decode tok/sec: N/A (no decode steps)")


if __name__ == "__main__":
    main()
