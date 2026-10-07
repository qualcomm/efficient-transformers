# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MiniMax-M3 paged MSA decode example with DP/CP enabled.

The paged runtime contract uses two block tables with shape
``[dp, batch // dp, ceil(ctx_len / page_block_size)]``:

* ``msa_indexer_block_table`` addresses the physical index-key pool.
* ``msa_attn_block_table`` addresses the physical attention KV pool.

This example uses the same physical page numbering for both pools. A serving
system can replace :func:`build_block_table` with its allocator's tables.
"""

import argparse
import math
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def build_block_table(dp: int, batch_size: int, ctx_len: int, page_block_size: int) -> torch.Tensor:
    """Build a DP-major ``[DP, B_local, num_pages]`` page table."""
    if batch_size % dp:
        raise ValueError("batch_size must be divisible by dp")
    pages = math.ceil(ctx_len / page_block_size)
    batch_local = batch_size // dp
    physical_pages_per_dp = batch_local * pages
    table = torch.empty((dp, batch_local, pages), dtype=torch.int32)
    for dp_idx in range(dp):
        table[dp_idx] = torch.randperm(physical_pages_per_dp, dtype=torch.int32).view(batch_local, pages)
    return table


def expand_batch(inputs, batch_size: int):
    expanded = {}
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            expanded[name] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            expanded[name] = value
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax-M3 paged MSA decode with DP/CP.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--page-block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--msa-indexer-dp", type=int, default=2)
    parser.add_argument("--msa-indexer-cp", type=int, default=1)
    parser.add_argument("--msa-attn-dp", type=int, default=2)
    parser.add_argument("--msa-attn-cp", type=int, default=1)
    parser.add_argument("--num-kv-blocks", type=int, default=2)
    parser.add_argument("--indexer-num-blocks", type=int, default=None)
    parser.add_argument("--msa-num-kv-blocks", type=int, default=None)
    parser.add_argument("--indexer-n-head", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--skip-generate", action=argparse.BooleanOptionalAction, default=False)
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
        "--enable-proxy", action="store_true", help="Enable QEff proxy transforms during model loading."
    )
    args = parser.parse_args()

    if any(value < 1 for value in (args.msa_indexer_dp, args.msa_indexer_cp, args.msa_attn_dp, args.msa_attn_cp)):
        parser.error("All MSA DP/CP factors must be positive")
    if args.num_kv_blocks < 1:
        parser.error("--num-kv-blocks must be positive")
    if args.indexer_num_blocks is not None and args.indexer_num_blocks < 1:
        parser.error("--indexer-num-blocks must be positive")
    if args.msa_num_kv_blocks is not None and args.msa_num_kv_blocks < 1:
        parser.error("--msa-num-kv-blocks must be positive")
    if args.ctx_len % args.page_block_size:
        parser.error("--ctx-len must be divisible by --page-block-size")
    if (args.ctx_len // args.page_block_size) % args.msa_attn_cp:
        parser.error("The number of attention pages must be divisible by --msa-attn-cp")
    dp_lcm = math.lcm(args.msa_indexer_dp, args.msa_attn_dp)
    execution_batch_size = args.batch_size * dp_lcm
    if execution_batch_size % args.msa_indexer_dp or execution_batch_size % args.msa_attn_dp:
        parser.error("DP factors must divide the execution batch size")

    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        config=config,
        kv_offload=True,
        dtype=torch.float16,
        weight_free=True,
        **({"enable_proxy": True} if args.enable_proxy else {}),
    )
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": args.num_kv_blocks,
        "indexer_num_blocks": args.indexer_num_blocks,
        "msa_num_kv_blocks": args.msa_num_kv_blocks,
        "msa_indexer_dp": args.msa_indexer_dp,
        "msa_indexer_cp": args.msa_indexer_cp,
        "msa_attn_dp": args.msa_attn_dp,
        "msa_attn_cp": args.msa_attn_cp,
        "indexer_n_head": args.indexer_n_head,
        "num_cores_per_device": args.num_cores,
        "paged_kv": True,
        "page_block_size": args.page_block_size,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
            "cores_per_expert": args.cores_per_expert,
            "tree_reduce": args.tree_reduce,
        },
    }

    t0 = time.perf_counter()
    qpc_paths = qeff_model.compile(
        batch_size=execution_batch_size,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        skip_vision=True,
        node_precision_info=True,
        use_onnx_subfunctions=True,
        mxint8_kv_cache=True,
        qaic_config=qaic_config,
    )
    print(f"[timing] compile: {time.perf_counter() - t0:.2f}s")
    print(f"QPC paths: {qpc_paths}")

    if args.skip_generate:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    messages = [[{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = expand_batch(inputs, execution_batch_size)

    # Tables are DP-major even though input_ids are flattened as [DP * B_local, ...].
    indexer_table = build_block_table(args.msa_indexer_dp, execution_batch_size, args.ctx_len, args.page_block_size)
    attention_table = build_block_table(args.msa_attn_dp, execution_batch_size, args.ctx_len, args.page_block_size)
    inputs["msa_indexer_block_table"] = indexer_table
    inputs["msa_attn_block_table"] = attention_table

    t0 = time.perf_counter()
    output = qeff_model.generate(inputs=inputs, generation_len=args.generation_len)
    elapsed = time.perf_counter() - t0
    generated = output.generated_ids.shape[-1]
    print(f"[timing] generation: {elapsed:.2f}s ({generated / elapsed:.02f} tok/s)")
    print(tokenizer.batch_decode(output.generated_ids))


if __name__ == "__main__":
    main()
