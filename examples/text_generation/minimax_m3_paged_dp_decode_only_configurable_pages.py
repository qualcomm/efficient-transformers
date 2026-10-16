# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MiniMax-M3 dedicated paged GQA/MSA decode with configurable pages.

The GQA, selector/indexer, and sparse-attention caches can use different page
sizes and logical-page counts. Each pair controls its corresponding physical
cache shape during export and compilation through ``qaic_config``. Use
``--layout-only`` to inspect the resulting grouped tables without loading the
model or using QAIC.
"""

import argparse
import math
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def build_block_table(
    dp: int,
    cp: int,
    batch_size: int,
    num_logical_pages: int,
    generator: torch.Generator,
    physical_pages: int | None = None,
) -> torch.Tensor:
    """Build a DP-major logical-to-physical page table."""
    if batch_size % dp:
        raise ValueError("batch_size must be divisible by dp")
    batch_local = batch_size // dp
    num_page_groups = math.ceil(num_logical_pages / cp)
    required_pages = batch_local * num_page_groups
    physical_pages = required_pages if physical_pages is None else int(physical_pages)
    if physical_pages < required_pages:
        raise ValueError(
            f"physical page capacity ({physical_pages}) is smaller than the required logical entries "
            f"({required_pages})."
        )
    table = torch.empty((dp, batch_local, num_page_groups), dtype=torch.int32)
    for dp_idx in range(dp):
        table[dp_idx] = torch.randperm(
            physical_pages,
            dtype=torch.int32,
            generator=generator,
        )[:required_pages].view(batch_local, num_page_groups)
    return table


def print_page_layout(
    label: str,
    block_table: torch.Tensor,
    cp: int,
    num_logical_pages: int,
    page_block_size: int,
) -> None:
    """Print the logical-page mapping used by CP-grouped paged decode."""
    print(
        f"[{label}] page_size={page_block_size}, logical_pages={num_logical_pages}, cp={cp}, "
        f"page_groups={math.ceil(num_logical_pages / cp)}, table_shape={tuple(block_table.shape)}"
    )
    preview_pages = min(num_logical_pages, max(cp + 1, 4))
    for logical_page in range(preview_pages):
        page_group = logical_page // cp
        owner_cp = logical_page % cp
        physical_page = int(block_table[0, 0, page_group])
        print(f"  logical_page={logical_page}: group={page_group}, owner_cp={owner_cp}, physical_page={physical_page}")


def tokenize_prompt(tokenizer, prompt: str, min_prompt_tokens: int):
    """Repeat the prompt until decode starts at or beyond the requested token position."""
    repetitions = 1
    while True:
        repeated_prompt = " ".join([prompt] * repetitions)
        messages = [[{"role": "user", "content": [{"type": "text", "text": repeated_prompt}]}]]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[1] >= min_prompt_tokens:
            return inputs
        repetitions *= 2


def expand_batch(inputs, batch_size: int):
    expanded = {}
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            expanded[name] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            expanded[name] = value
    return expanded


def parse_device_ids(value: str) -> list[int]:
    """Parse a comma-separated QAIC device list, for example ``0,2,3``."""
    try:
        device_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("device IDs must be comma-separated integers") from error
    if not device_ids or any(device_id < 0 for device_id in device_ids):
        raise ValueError("device IDs must contain at least one non-negative integer")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("device IDs must be unique")
    return device_ids


def _build_qaic_config(
    args: argparse.Namespace,
    gqa_page_block_size: int,
    indexer_page_block_size: int,
    attn_page_block_size: int,
) -> dict:
    return {
        "blocking_mode": "kv_minimax_dedicated",
        "num_kv_blocks": args.num_kv_blocks,
        "attn_dp": args.attn_dp,
        "attn_cp": args.attn_cp,
        "indexer_num_blocks": args.indexer_num_blocks,
        "msa_num_kv_blocks": args.msa_num_kv_blocks,
        "skip_kv": args.skip_kv,
        "non_dynamic_cache": args.non_dynamic_cache,
        "msa_indexer_dp": args.msa_indexer_dp,
        "msa_indexer_cp": args.msa_indexer_cp,
        "msa_attn_dp": args.msa_attn_dp,
        "msa_attn_cp": args.msa_attn_cp,
        "indexer_n_head": args.indexer_n_head,
        "num_cores_per_device": args.num_cores,
        "paged_kv": True,
        "static_export_dims": args.static_export_dims,
        "gqa_page_block_size": gqa_page_block_size,
        "msa_indexer_page_block_size": indexer_page_block_size,
        "msa_attn_page_block_size": attn_page_block_size,
        "gqa_num_logical_pages": args.gqa_num_pages,
        "msa_indexer_num_logical_pages": args.msa_indexer_num_logical_pages,
        "msa_attn_num_logical_pages": args.msa_attn_num_logical_pages,
        "gqa_physical_pages": args.gqa_physical_pages,
        "msa_indexer_physical_pages": args.msa_indexer_physical_pages,
        "msa_attn_physical_pages": args.msa_attn_physical_pages,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
            "cores_per_expert": args.cores_per_expert,
            "tree_reduce": args.tree_reduce,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniMax-M3 paged decode with independently configurable cache page sizes and counts."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument(
        "--page-block-size",
        type=int,
        default=None,
        help="Legacy shared page size used when a cache-specific page size is omitted.",
    )
    parser.add_argument(
        "--gqa-page-block-size",
        type=int,
        default=None,
        help="Physical page width for dense GQA K/V caches.",
    )
    parser.add_argument(
        "--selector-page-block-size",
        "--msa-indexer-page-block-size",
        dest="msa_indexer_page_block_size",
        type=int,
        default=None,
        help="Physical page width for the sparse selector/index-key cache.",
    )
    parser.add_argument(
        "--msa-page-block-size",
        "--msa-attn-page-block-size",
        dest="msa_attn_page_block_size",
        type=int,
        default=None,
        help="Physical page width for sparse MSA K/V caches.",
    )
    parser.add_argument(
        "--gqa-num-pages",
        type=int,
        default=64,
        help="Logical pages per sequence in the dense GQA K/V cache.",
    )
    parser.add_argument(
        "--msa-indexer-num-logical-pages",
        type=int,
        default=128,
        help="Logical pages per sequence in the index-key cache.",
    )
    parser.add_argument(
        "--msa-attn-num-logical-pages",
        type=int,
        default=64,
        help="Logical pages per sequence in the full attention K/V cache.",
    )
    parser.add_argument(
        "--gqa-physical-pages",
        type=int,
        default=None,
        help="Physical page capacity of the dense GQA KV pool.",
    )
    parser.add_argument(
        "--msa-attn-physical-pages",
        type=int,
        default=None,
        help="Physical page capacity of the sparse MSA attention KV pool.",
    )
    parser.add_argument(
        "--msa-indexer-physical-pages",
        type=int,
        default=None,
        help="Physical page capacity of the sparse indexer KV pool.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attn-dp", type=int, default=1, help="Dense GQA decode data-parallel factor.")
    parser.add_argument("--attn-cp", type=int, default=1, help="Dense GQA decode context-parallel factor.")
    parser.add_argument("--msa-indexer-dp", type=int, default=2)
    parser.add_argument("--msa-indexer-cp", type=int, default=4)
    parser.add_argument("--msa-attn-dp", type=int, default=2)
    parser.add_argument("--msa-attn-cp", type=int, default=4)
    parser.add_argument("--num-kv-blocks", type=int, default=2)
    parser.add_argument("--indexer-num-blocks", type=int, default=None)
    parser.add_argument("--msa-num-kv-blocks", type=int, default=None)
    parser.add_argument(
        "--skip-kv",
        action="store_true",
        help="Skip KV blocks that are entirely in the future (disabled by default).",
    )
    parser.add_argument(
        "--non-dynamic-cache",
        action="store_true",
        help="Export MiniMax KV-cache inputs and outputs without dynamic axes.",
    )
    parser.add_argument("--indexer-n-head", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=None,
        help="Comma-separated QAIC device IDs used for generation (default: 0..num-devices-1).",
    )
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=129,
        help="Repeat the prompt until decode starts after at least this many tokens; 129 exercises CP owner row 1.",
    )
    parser.add_argument("--block-table-seed", type=int, default=17)
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Print grouped table shapes and logical-page mappings, then exit before loading the model.",
    )
    parser.add_argument("--skip-generate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--static-export-dims",
        action="store_true",
        help="Export with every decode dimension fixed instead of symbolic.",
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
        "--enable-proxy", action="store_true", help="Enable QEff proxy transforms during model loading."
    )
    args = parser.parse_args()

    device_ids = args.device_ids if args.device_ids is not None else list(range(args.num_devices))
    if len(device_ids) != args.num_devices:
        parser.error("--device-ids must contain exactly --num-devices entries")

    if any(
        value < 1
        for value in (
            args.attn_dp,
            args.attn_cp,
            args.msa_indexer_dp,
            args.msa_indexer_cp,
            args.msa_attn_dp,
            args.msa_attn_cp,
        )
    ):
        parser.error("All GQA and MSA DP/CP factors must be positive")
    shared_page_block_size = 128 if args.page_block_size is None else args.page_block_size
    gqa_page_block_size = shared_page_block_size if args.gqa_page_block_size is None else args.gqa_page_block_size
    indexer_page_block_size = (
        shared_page_block_size if args.msa_indexer_page_block_size is None else args.msa_indexer_page_block_size
    )
    attn_page_block_size = (
        shared_page_block_size if args.msa_attn_page_block_size is None else args.msa_attn_page_block_size
    )
    if any(size < 1 for size in (gqa_page_block_size, indexer_page_block_size, attn_page_block_size)):
        parser.error("All page block sizes must be positive")
    if args.min_prompt_tokens < 1:
        parser.error("--min-prompt-tokens must be positive")
    if any(
        value < 1
        for value in (
            args.gqa_num_pages,
            args.msa_indexer_num_logical_pages,
            args.msa_attn_num_logical_pages,
        )
    ):
        parser.error("All logical-page counts must be positive")
    if any(
        value is not None and value < 1
        for value in (
            args.gqa_physical_pages,
            args.msa_indexer_physical_pages,
            args.msa_attn_physical_pages,
        )
    ):
        parser.error("All physical-page capacities must be positive")
    if args.gqa_num_pages * gqa_page_block_size < args.ctx_len:
        parser.error("GQA logical-page capacity must be at least --ctx-len")
    if args.msa_indexer_num_logical_pages * indexer_page_block_size < args.ctx_len:
        parser.error("Indexer logical-page capacity must be at least --ctx-len")
    if args.msa_attn_num_logical_pages * attn_page_block_size < args.ctx_len:
        parser.error("Attention logical-page capacity must be at least --ctx-len")
    if args.num_kv_blocks < 1:
        parser.error("--num-kv-blocks must be positive")
    if args.indexer_num_blocks is not None and args.indexer_num_blocks < 1:
        parser.error("--indexer-num-blocks must be positive")
    if args.msa_num_kv_blocks is not None and args.msa_num_kv_blocks < 1:
        parser.error("--msa-num-kv-blocks must be positive")
    if any(args.ctx_len % size for size in (gqa_page_block_size, indexer_page_block_size, attn_page_block_size)):
        parser.error("--ctx-len must be divisible by every page block size")
    indexer_num_blocks = args.indexer_num_blocks or args.num_kv_blocks
    indexer_page_divisor = indexer_num_blocks * args.msa_indexer_cp
    if args.msa_indexer_num_logical_pages % indexer_page_divisor:
        parser.error("Indexer logical pages must be divisible by the indexer block count times --msa-indexer-cp")
    if (args.msa_indexer_num_logical_pages // indexer_page_divisor) % args.num_cores:
        parser.error("Indexer page groups per block must be divisible by --num-cores")
    gqa_page_divisor = args.num_kv_blocks * args.attn_cp
    if args.gqa_num_pages % gqa_page_divisor:
        parser.error("GQA logical pages must be divisible by --num-kv-blocks times --attn-cp")
    if (args.gqa_num_pages // gqa_page_divisor) % args.num_cores:
        parser.error("GQA page groups per block must be divisible by --num-cores")

    dp_lcm = math.lcm(args.attn_dp, args.msa_indexer_dp, args.msa_attn_dp)
    execution_batch_size = args.batch_size * dp_lcm
    if (
        execution_batch_size % args.attn_dp
        or execution_batch_size % args.msa_indexer_dp
        or execution_batch_size % args.msa_attn_dp
    ):
        parser.error("DP factors must divide the execution batch size")

    table_generator = torch.Generator(device="cpu").manual_seed(args.block_table_seed)
    gqa_table = build_block_table(
        args.attn_dp,
        args.attn_cp,
        execution_batch_size,
        args.gqa_num_pages,
        table_generator,
        args.gqa_physical_pages,
    )
    indexer_table = build_block_table(
        args.msa_indexer_dp,
        args.msa_indexer_cp,
        execution_batch_size,
        args.msa_indexer_num_logical_pages,
        table_generator,
        args.msa_indexer_physical_pages,
    )
    attention_table = build_block_table(
        args.msa_attn_dp,
        args.msa_attn_cp,
        execution_batch_size,
        args.msa_attn_num_logical_pages,
        table_generator,
        args.msa_attn_physical_pages,
    )
    print_page_layout("gqa", gqa_table, args.attn_cp, args.gqa_num_pages, gqa_page_block_size)
    print_page_layout(
        "indexer",
        indexer_table,
        args.msa_indexer_cp,
        args.msa_indexer_num_logical_pages,
        indexer_page_block_size,
    )
    print_page_layout(
        "attention",
        attention_table,
        args.msa_attn_cp,
        args.msa_attn_num_logical_pages,
        attn_page_block_size,
    )
    if args.layout_only:
        return

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
    qaic_config = _build_qaic_config(args, gqa_page_block_size, indexer_page_block_size, attn_page_block_size)

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
        mxfp6_matmul=True,
        retain_full_kv=True,
        split_model_io=True,
        user_tiled=True,
        qaic_config=qaic_config,
    )
    print(f"[timing] compile: {time.perf_counter() - t0:.2f}s")
    print(f"QPC paths: {qpc_paths}")

    if args.skip_generate:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    inputs = tokenize_prompt(tokenizer, args.prompt, args.min_prompt_tokens)
    prompt_tokens = inputs["input_ids"].shape[1]
    if prompt_tokens + args.generation_len > args.ctx_len:
        parser.error(
            f"Tokenized prompt ({prompt_tokens}) plus generation length ({args.generation_len}) "
            f"exceeds --ctx-len ({args.ctx_len})"
        )
    first_gqa_page = prompt_tokens // gqa_page_block_size
    first_indexer_page = prompt_tokens // indexer_page_block_size
    first_attn_page = prompt_tokens // attn_page_block_size
    if (
        first_gqa_page >= args.gqa_num_pages
        or first_indexer_page >= args.msa_indexer_num_logical_pages
        or first_attn_page >= args.msa_attn_num_logical_pages
    ):
        parser.error("The first decode position exceeds one of the configured logical-page capacities")
    print(
        f"[decode] prompt_tokens={prompt_tokens}, "
        f"gqa=(page={first_gqa_page}, group={first_gqa_page // args.attn_cp}, "
        f"owner_cp={first_gqa_page % args.attn_cp}), "
        f"indexer=(page={first_indexer_page}, group={first_indexer_page // args.msa_indexer_cp}, "
        f"owner_cp={first_indexer_page % args.msa_indexer_cp}), "
        f"attention=(page={first_attn_page}, group={first_attn_page // args.msa_attn_cp}, "
        f"owner_cp={first_attn_page % args.msa_attn_cp})"
    )
    inputs = expand_batch(inputs, execution_batch_size)

    inputs["gqa_block_table"] = gqa_table
    inputs["msa_indexer_block_table"] = indexer_table
    inputs["msa_attn_block_table"] = attention_table
    t0 = time.perf_counter()
    output = qeff_model.generate(
        inputs=inputs,
        generation_len=args.generation_len,
        device_ids=device_ids,
    )
    elapsed = time.perf_counter() - t0
    generated = output.generated_ids.shape[-1]
    print(f"[timing] generation: {elapsed:.2f}s ({generated / elapsed:.02f} tok/s)")
    print(tokenizer.batch_decode(output.generated_ids))


if __name__ == "__main__":
    main()
