# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import math
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "MiniMaxAI/MiniMax-M3"
# MODEL_ID = "/home/smedhe/MINIMAX/tiny-random-minimax-m3/model.safetensors"


def _expand_batch(inputs, batch_size: int):
    """Repeat single-prompt tokenizer tensors for the compiled execution batch."""
    expanded = {}
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            expanded[name] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            expanded[name] = value
    return expanded


def _execution_batch_size(batch_size: int, msa_indexer_dp: int, msa_attn_dp: int, attn_dp: int) -> int:
    return batch_size * math.lcm(msa_indexer_dp, msa_attn_dp, attn_dp)


def parse_device_ids(value: str) -> list[int]:
    device_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not device_ids or any(device_id < 0 for device_id in device_ids):
        raise argparse.ArgumentTypeError("device IDs must be a non-empty list of non-negative integers")
    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("device IDs must be unique")
    return device_ids


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M3 text-only decode (PL=1) with DP and GP enabled.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Logical prompt batch size; QEfficient expands it by the DP LCM for execution.",
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--skip-generate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--static-export-dims",
        action="store_true",
        help="Export with fixed batch and context sizes instead of symbolic ones.",
    )
    parser.add_argument(
        "--skip-kv",
        action="store_true",
        help="Skip KV blocks that are entirely in the future (disabled by default).",
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=2,
        help="Number of KV cache blocks used by blocked attention.",
    )
    parser.add_argument(
        "--indexer-num-blocks",
        type=int,
        default=None,
        help="Override the number of KV cache blocks used by the MSA indexer.",
    )
    parser.add_argument(
        "--msa-num-kv-blocks",
        type=int,
        default=None,
        help="Override the number of KV cache blocks used by MSA attention.",
    )
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
    parser.add_argument(
        "--msa-indexer-dp",
        type=int,
        default=2,
        help="DP factor for the MSA sparse-attention indexer (_select_blocks_dp path).",
    )
    parser.add_argument(
        "--msa-indexer-cp",
        type=int,
        default=2,
        help="CP factor for the MSA sparse-attention indexer compact cache layout.",
    )
    parser.add_argument(
        "--msa-attn-dp",
        type=int,
        default=2,
        help="DP factor for GP attention (_baseline_attention_gp path). Must divide batch_size.",
    )
    parser.add_argument(
        "--msa-attn-cp",
        type=int,
        default=1,
        help="CP factor for GP attention cache layout.",
    )
    parser.add_argument(
        "--attn-dp",
        type=int,
        default=1,
        help="DP factor for dense GQA attention and its cache layout.",
    )
    parser.add_argument(
        "--attn-cp",
        type=int,
        default=1,
        help="CP factor for dense GQA attention and its cache layout.",
    )
    parser.add_argument(
        "--indexer-n-head",
        type=int,
        default=1,
        help="Number of KV heads used by the MSA indexer in the DP path.",
    )
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=None,
        help="Comma-separated QAIC device IDs used for generation (default: 0..num-devices-1).",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_kv_blocks < 1:
        parser.error("--num-kv-blocks must be positive")
    for name in ("indexer_num_blocks", "msa_num_kv_blocks"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    device_ids = args.device_ids if args.device_ids is not None else list(range(args.num_devices))
    if len(device_ids) != args.num_devices:
        parser.error("--device-ids must contain exactly --num-devices entries")
    for name in ("msa_indexer_dp", "msa_indexer_cp", "msa_attn_dp", "msa_attn_cp", "attn_dp", "attn_cp"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    execution_batch_size = _execution_batch_size(args.batch_size, args.msa_indexer_dp, args.msa_attn_dp, args.attn_dp)

    factory_kwargs = dict(kv_offload=True, dtype=torch.float16)
    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers
    factory_kwargs["config"] = config

    t0 = time.perf_counter()
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_id, weight_free=True, **factory_kwargs)
    print(f"[timing] model load:          {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    qpc_paths = qeff_model.compile(
        batch_size=execution_batch_size,
        prefill_seq_len=1,
        prefill_only=False,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_onnx_subfunctions=True,
        dynamo=True,
        skip_vision=True,
        node_precision_info=True,
        offload_pt_weights=False,
        log_times=True,
        qaic_config={
            "blocking_mode": "kv_minimax_dedicated",
            "num_kv_blocks": args.num_kv_blocks,
            "indexer_num_blocks": args.indexer_num_blocks,
            "msa_num_kv_blocks": args.msa_num_kv_blocks,
            "skip_kv": args.skip_kv,
            "attn_dp": args.attn_dp,
            "attn_cp": args.attn_cp,
            "msa_indexer_dp": args.msa_indexer_dp,
            "msa_indexer_cp": args.msa_indexer_cp,
            "msa_attn_dp": args.msa_attn_dp,
            "msa_attn_cp": args.msa_attn_cp,
            "indexer_n_head": args.indexer_n_head,
            "num_cores_per_device": args.num_cores,
            "static_export_dims": args.static_export_dims,
            "moe_config": {
                "flavour": "expert_parallel",
                "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
                "cores_per_expert": args.cores_per_expert,
                "tree_reduce": args.tree_reduce,
            },
        },
    )
    print(f"[timing] compile total:       {time.perf_counter() - t0:.2f}s")
    print(f"QPC paths: {qpc_paths}")

    if args.skip_generate:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            }
        ]
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _expand_batch(inputs, execution_batch_size)
    t0 = time.perf_counter()
    output = qeff_model.generate(inputs=inputs, generation_len=args.generation_len, device_ids=device_ids)
    generate_time = time.perf_counter() - t0

    num_generated = output.generated_ids.shape[-1]
    toks_per_sec = num_generated / float(generate_time)
    print(f"[timing] generation:          {generate_time:.2f}s  ({num_generated} tokens, {toks_per_sec:.02f} tok/s)")

    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))


if __name__ == "__main__":
    main()
