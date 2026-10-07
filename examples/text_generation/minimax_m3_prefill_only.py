# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import math
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value is not None else None


Q_HEAD_BLOCK_CHUNK = int(os.environ.get("Q_HEAD_BLOCK_CHUNK", "1"))
Q_BLOCK_SIZE = int(os.environ.get("Q_BLOCK_SIZE", "256"))
Q_BLOCK_CHUNK = int(os.environ.get("Q_BLOCK_CHUNK", "1024"))
NUM_KV_BLOCKS = int(os.environ.get("NUM_KV_BLOCKS", "16"))
INDEXER_NUM_BLOCKS = _optional_int_env("INDEXER_NUM_BLOCKS")
MSA_NUM_KV_BLOCKS = _optional_int_env("MSA_NUM_KV_BLOCKS")


def _expand_batch(inputs, batch_size: int):
    """Repeat single-prompt tokenizer tensors for the compiled execution batch."""
    expanded = {}
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            expanded[name] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            expanded[name] = value
    return expanded


def _parse_device_group(device_ids: str) -> list[int]:
    return [int(device_id) for device_id in device_ids.strip("[]").split(",") if device_id.strip()]


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
    indexer_num_blocks: int | None,
    msa_num_kv_blocks: int | None,
    num_cores: int,
    msa_q_chunk: int,
    expert_parallel_chunk_size: int,
    cores_per_expert: int,
    tree_reduce: bool,
    head_block_size: int,
    num_kv_blocks: int,
    num_q_blocks: int,
    n_rep_chunk: int,
    ctx_len: int,
    static_export_dims: bool = False,
) -> dict:
    """Build the MiniMax blocking config for prefill."""
    qaic_config = {
        "blocking_mode": "prefill_online",
        "head_block_size": head_block_size,
        "num_kv_blocks": num_kv_blocks,
        "num_q_blocks": num_q_blocks,
        "n_rep_chunk": n_rep_chunk,
        "ctx_len": ctx_len,
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
    if static_export_dims:
        qaic_config["static_export_dims"] = True
    if indexer_q_chunk is not None:
        qaic_config["indexer_q_chunk"] = indexer_q_chunk
    if indexer_q_size is not None:
        qaic_config["indexer_q_size"] = indexer_q_size
    if indexer_num_blocks is not None:
        qaic_config["indexer_num_blocks"] = indexer_num_blocks
    if msa_num_kv_blocks is not None:
        qaic_config["msa_num_kv_blocks"] = msa_num_kv_blocks
    return qaic_config


def _update_retained_cache_inputs(inputs: dict, outputs: dict, num_layers: int) -> None:
    """Carry retained KV and sparse-indexer states between prefill chunks."""
    for layer_idx in range(num_layers):
        for cache_name in ("past_key", "past_value", "index_key"):
            retained_name = f"{cache_name}.{layer_idx}_RetainedState"
            if retained_name in outputs:
                inputs[f"{cache_name}.{layer_idx}"] = outputs[retained_name]


def main():
    parser = argparse.ArgumentParser(
        description="Export, compile, and run a MiniMax-M3 prefill-only QPC, then report TTFT."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument(
        "--device-group",
        type=_parse_device_group,
        default=None,
        help="Physical QAIC device IDs to use at runtime, for example '[0,1,2,3]'.",
    )
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument(
        "--compile-threads",
        type=int,
        default=None,
        help="Forward -compile-threads to qaic-compile when set.",
    )
    parser.add_argument(
        "--time-passes",
        action="store_true",
        help="Forward -time-passes to qaic-compile.",
    )
    parser.add_argument(
        "--profile-export",
        choices=("off", "stages", "cprofile"),
        default="off",
        help=(
            "Export profiling (sets QEFF_EXPORT_PROFILE). 'stages': per-stage timings, FX/ONNX node counts, "
            "loop-unroll budget, torch.export strategy attempts. 'cprofile': also a cProfile of torch.onnx.export."
        ),
    )
    parser.add_argument(
        "--static-export-dims",
        action="store_true",
        help="Export with fixed batch and ctx_len dims (much faster tracing; ONNX has no symbolic batch/ctx_len).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Stop after ONNX export + ONNX transforms; skip qaic-compile and inference.",
    )
    onnx_subfunctions_group = parser.add_mutually_exclusive_group()
    onnx_subfunctions_group.add_argument(
        "--use-onnx-subfunctions",
        dest="use_onnx_subfunctions",
        action="store_true",
        default=True,
        help="Enable ONNX subfunction export.",
    )
    onnx_subfunctions_group.add_argument(
        "--no-onnx-subfunctions",
        dest="use_onnx_subfunctions",
        action="store_false",
        help="Disable ONNX subfunction export.",
    )
    parser.add_argument(
        "--mdp-num-partitions",
        type=int,
        default=16,
        help="Number of pipeline-parallel MDP partitions.",
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
        help="Logical prompt batch size; execution expands it by the prefill DP LCM.",
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument(
        "--expert-parallel-chunk-size",
        type=int,
        default=256,
        help="MoE expert-parallel chunk size.",
    )
    parser.add_argument(
        "--cores-per-expert",
        type=int,
        default=2,
        help="Number of NSP cores assigned to each expert.",
    )
    parser.add_argument(
        "--no-tree-reduce",
        dest="tree_reduce",
        action="store_false",
        default=True,
        help="Disable tree-reduce for MoE expert-parallel dispatch.",
    )
    parser.add_argument("--msa-indexer-dp", type=int, default=1, help="Prefill MSA indexer DP factor.")
    parser.add_argument("--msa-indexer-cp", type=int, default=1, help="Prefill MSA indexer CP factor.")
    parser.add_argument("--msa-attn-dp", type=int, default=1, help="Prefill MSA attention DP factor.")
    parser.add_argument("--msa-attn-cp", type=int, default=1, help="Prefill MSA attention CP factor.")
    parser.add_argument(
        "--indexer-n-head",
        type=int,
        default=1,
        help="Number of KV heads used by the MSA indexer in the DP path.",
    )
    parser.add_argument(
        "--indexer-prefill-parallel",
        action="store_true",
        default=False,
        help="Use the parallel indexer prefill selector.",
    )
    parser.add_argument(
        "--indexer-q-chunk",
        type=int,
        default=None,
        help="Outer query chunk size; defaults to the indexer query block size.",
    )
    parser.add_argument(
        "--indexer-q-size",
        type=int,
        default=None,
        help="Query micro-block size; defaults to the model index block size.",
    )
    parser.add_argument("--msa-q-chunk", type=int, default=64, help="MSA prefill attention query chunk size.")
    parser.add_argument(
        "--q-head-block-chunk",
        type=int,
        default=Q_HEAD_BLOCK_CHUNK,
        help="Prefill attention head block size (default: Q_HEAD_BLOCK_CHUNK or 1).",
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=NUM_KV_BLOCKS,
        help="Number of prefill KV blocks (default: NUM_KV_BLOCKS or 16).",
    )
    parser.add_argument(
        "--indexer-num-blocks",
        type=int,
        default=INDEXER_NUM_BLOCKS,
        help="MiniMax MSA indexer KV blocking factor; defaults to --num-kv-blocks when omitted.",
    )
    parser.add_argument(
        "--msa-num-kv-blocks",
        type=int,
        default=MSA_NUM_KV_BLOCKS,
        help="MiniMax MSA attention KV blocking factor; defaults to --num-kv-blocks when omitted.",
    )
    parser.add_argument(
        "--q-block-size",
        type=int,
        default=Q_BLOCK_SIZE,
        help="Prefill query block size within the outer query chunk (default: Q_BLOCK_SIZE or 256).",
    )
    parser.add_argument(
        "--q-block-chunk",
        type=int,
        default=Q_BLOCK_CHUNK,
        help="Outer prefill query chunk size used to derive n_rep_chunk (default: Q_BLOCK_CHUNK or 1024).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.prefill_seq_len <= 1:
        parser.error("--prefill-seq-len must be greater than 1")
    if args.q_head_block_chunk < 1:
        parser.error("--q-head-block-chunk must be positive")
    if args.num_kv_blocks < 1:
        parser.error("--num-kv-blocks must be positive")
    if args.q_block_size < 1:
        parser.error("--q-block-size must be positive")
    if args.q_block_chunk < 1:
        parser.error("--q-block-chunk must be positive")
    if args.q_block_chunk < args.q_block_size or args.q_block_chunk % args.q_block_size:
        parser.error("--q-block-chunk must be at least and divisible by --q-block-size")
    if args.mdp_num_partitions < 1:
        parser.error("--mdp-num-partitions must be positive")
    if args.mdp_num_partitions > args.num_devices:
        parser.error("--mdp-num-partitions must not exceed --num-devices")
    if args.num_devices % args.mdp_num_partitions:
        parser.error("--num-devices must be divisible by --mdp-num-partitions")
    if args.device_group is not None and len(args.device_group) != args.num_devices:
        parser.error("--device-group length must match --num-devices")
    if args.compile_threads is not None and args.compile_threads < 1:
        parser.error("--compile-threads must be positive")
    if args.indexer_num_blocks is not None and args.indexer_num_blocks < 1:
        parser.error("--indexer-num-blocks must be positive")
    if args.msa_num_kv_blocks is not None and args.msa_num_kv_blocks < 1:
        parser.error("--msa-num-kv-blocks must be positive")
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
    if any(value < 1 for value in (args.msa_indexer_dp, args.msa_indexer_cp, args.msa_attn_dp, args.msa_attn_cp)):
        parser.error("All MSA DP/CP factors must be positive")
    if args.msa_indexer_dp != 1 or args.msa_indexer_cp != 1:
        parser.error("MiniMax MSA prefill currently requires indexer DP=1 and CP=1")
    if args.msa_attn_dp != 1 or args.msa_attn_cp != 1:
        parser.error("MiniMax MSA prefill currently requires attention DP=1 and CP=1")

    if args.profile_export != "off":
        os.environ["QEFF_EXPORT_PROFILE"] = "cprofile" if args.profile_export == "cprofile" else "1"

    execution_batch_size = args.batch_size * math.lcm(args.msa_indexer_dp, args.msa_attn_dp)
    num_q_blocks = max(1, math.ceil(args.prefill_seq_len / args.q_block_size))
    n_rep_chunk = args.q_block_chunk // args.q_block_size
    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers

    load_start = time.perf_counter()
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        config=config,
        kv_offload=True,
        weight_free=True,
        dtype=torch.float16,
    )
    print(f"[timing] model load: {time.perf_counter() - load_start:.2f}s")

    compile_start = time.perf_counter()
    compiler_options = {}
    if args.compile_threads is not None:
        compiler_options["compile_threads"] = args.compile_threads
    if args.time_passes:
        compiler_options["time_passes"] = True
    if args.export_only:
        compiler_options["export_only"] = True

    qpc_paths = qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        prefill_only=True,
        enable_chunking=True,
        mdp_num_partitions=args.mdp_num_partitions,
        batch_size=execution_batch_size,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        skip_vision=True,
        offload_pt_weights=False,
        node_precision_info=True,
        log_times=True,
        retain_full_kv=True,
        split_model_io=True,
        qaic_config=_build_qaic_config(
            msa_indexer_dp=args.msa_indexer_dp,
            msa_indexer_cp=args.msa_indexer_cp,
            msa_attn_dp=args.msa_attn_dp,
            msa_attn_cp=args.msa_attn_cp,
            indexer_n_head=args.indexer_n_head,
            indexer_prefill_parallel=args.indexer_prefill_parallel,
            indexer_q_chunk=args.indexer_q_chunk,
            indexer_q_size=args.indexer_q_size,
            indexer_num_blocks=args.indexer_num_blocks,
            msa_num_kv_blocks=args.msa_num_kv_blocks,
            num_cores=args.num_cores,
            msa_q_chunk=args.msa_q_chunk,
            expert_parallel_chunk_size=args.expert_parallel_chunk_size,
            cores_per_expert=args.cores_per_expert,
            tree_reduce=args.tree_reduce,
            head_block_size=args.q_head_block_chunk,
            num_kv_blocks=args.num_kv_blocks,
            num_q_blocks=num_q_blocks,
            n_rep_chunk=n_rep_chunk,
            ctx_len=args.ctx_len,
            static_export_dims=args.static_export_dims,
        ),
        **compiler_options,
    )
    if args.export_only:
        print(f"[timing] prefill export only (incl. ONNX transforms): {time.perf_counter() - compile_start:.2f}s")
        print(f"Prefill ONNX path: {qpc_paths.get('lang_onnx_path')}")
        return
    prefill_qpc_path = qpc_paths["lang_prefill_qpc_path"]
    print(f"[timing] prefill export + compile: {time.perf_counter() - compile_start:.2f}s")
    print(f"Prefill QPC path: {prefill_qpc_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    messages = [[{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
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
    if padded_len > args.ctx_len:
        parser.error(f"Padded prompt length ({padded_len}) exceeds --ctx-len ({args.ctx_len})")

    pad_len = padded_len - input_len
    model_inputs["input_ids"] = torch.nn.functional.pad(model_inputs["input_ids"], (0, pad_len), value=0)
    if "attention_mask" in model_inputs:
        model_inputs["attention_mask"] = torch.nn.functional.pad(model_inputs["attention_mask"], (0, pad_len), value=0)
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
    prefill_state = np_inputs.copy()
    prefill_session = QAICInferenceSession(prefill_qpc_path, device_ids=args.device_group)

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

    first_token_ids = np.argmax(prefill_output["logits"], axis=-1).astype(np_inputs["input_ids"].dtype)
    print(f"First token: {tokenizer.batch_decode(first_token_ids, skip_special_tokens=True)}")
    print(f"[performance] TTFT: {ttft_seconds * 1000.0:.3f} ms")


if __name__ == "__main__":
    main()
