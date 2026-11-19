# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from argparse import Namespace
from typing import Iterable, Optional

from QEfficient.utils.constants import ATTENTION_HEAD_CONFIG_KEYS, HIDDEN_SIZE_CONFIG_KEYS, KV_HEAD_CONFIG_KEYS

DEEPSEEKV4_HW_CORES_PER_DEVICE = {"ai100": 16, "ai200": 4}


def _build_minimax_qaic_config(
    args: Namespace,
    gqa_page_block_size: int | None = None,
    indexer_page_block_size: int | None = None,
    attn_page_block_size: int | None = None,
    *,
    prefill: bool = False,
) -> dict:
    """Build MiniMax decode or prefill QAIC compiler settings."""
    if prefill:
        qaic_config = {
            "blocking_mode": "prefill_online",
            "head_block_size": args.q_head_block_chunk,
            "num_kv_blocks": args.num_kv_blocks,
            "num_q_blocks": max(1, math.ceil(args.prefill_seq_len / args.q_block_size)),
            "n_rep_chunk": args.q_block_chunk // args.q_block_size,
            "ctx_len": args.ctx_len,
            "msa_indexer_dp": args.msa_indexer_dp,
            "msa_indexer_cp": args.msa_indexer_cp,
            "msa_attn_dp": args.msa_attn_dp,
            "msa_attn_cp": args.msa_attn_cp,
            "indexer_n_head": args.indexer_n_head,
            "indexer_prefill_parallel": args.indexer_prefill_parallel,
            "num_cores_per_device": args.num_cores,
            "msa_q_chunk": args.msa_q_chunk,
            "moe_config": {
                "flavour": "expert_parallel",
                "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
                "cores_per_expert": args.cores_per_expert,
                "tree_reduce": args.tree_reduce,
            },
        }
        if args.static_export_dims:
            qaic_config["static_export_dims"] = True
        if args.indexer_q_chunk is not None:
            qaic_config["indexer_q_chunk"] = args.indexer_q_chunk
        if args.indexer_q_size is not None:
            qaic_config["indexer_q_size"] = args.indexer_q_size
        if args.indexer_num_blocks is not None:
            qaic_config["indexer_num_blocks"] = args.indexer_num_blocks
        if args.msa_num_kv_blocks is not None:
            qaic_config["msa_num_kv_blocks"] = args.msa_num_kv_blocks
        return qaic_config

    if any(page_size is None for page_size in (gqa_page_block_size, indexer_page_block_size, attn_page_block_size)):
        raise ValueError("GQA, indexer, and attention page sizes are required for MiniMax decode QAIC config.")

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


def build_deepseekv4_ffn_blocking_config(args: Namespace) -> dict[str, int | str | None]:
    """Build the compile-time routed-expert FFN blocking configuration for DeepSeek V4."""
    return {
        "ffn_blocking_mode": args.ffn_blocking_mode,
        "ffn_token_block_size": args.ffn_token_block_size,
        "ffn_weight_block_size": args.ffn_weight_block_size,
    }


def configure_qeff_parallel_layout(config, args: Namespace) -> None:
    """Apply DeepSeek V4 folded decode controls to QEff cache layouts."""
    attention_dp = args.attn_dp or len(args.device_group)
    indexer_cp = args.indexer_cp
    num_kv_blocks = args.num_kv_blocks
    attention_cores = DEEPSEEKV4_HW_CORES_PER_DEVICE[args.hw_version]
    hca_compressed_kv_cp = args.hca_compressed_kv_cp
    hca_attn_blocks = args.hca_attn_blocks or attention_cores
    if attention_dp < 1:
        raise ValueError("attn_dp must be at least 1.")
    if indexer_cp < 1:
        raise ValueError("indexer_cp must be at least 1.")
    if num_kv_blocks < 1:
        raise ValueError("num_kv_blocks must be at least 1.")
    if hca_compressed_kv_cp < 1:
        raise ValueError("hca_compressed_kv_cp must be at least 1.")
    if hca_attn_blocks < 1:
        raise ValueError("hca_attn_blocks must be at least 1.")
    if args.batch_size % attention_dp:
        raise ValueError("batch_size must be divisible by attn_dp.")

    csa_capacity = None
    hca_capacity = None
    for layer_type in config.layer_types:
        if layer_type == "compressed_sparse_attention":
            csa_capacity = (args.ctx_len + config.compress_rates[layer_type] - 1) // config.compress_rates[layer_type]
            if csa_capacity % indexer_cp:
                raise ValueError("CSA compressed-cache capacity must be divisible by indexer_cp.")
            csa_slots_per_cp = csa_capacity // indexer_cp
            if csa_slots_per_cp % num_kv_blocks:
                raise ValueError("CSA CP-way slot count must be divisible by num_kv_blocks.")
            if (csa_slots_per_cp // num_kv_blocks) % attention_cores:
                raise ValueError("CSA KV-block width must be divisible by the hardware attention-core count.")
        if layer_type == "heavily_compressed_attention":
            hca_capacity = (args.ctx_len + config.compress_rates[layer_type] - 1) // config.compress_rates[layer_type]
            if hca_capacity % hca_compressed_kv_cp:
                raise ValueError("HCA compressed-cache capacity must be divisible by hca_compressed_kv_cp.")
            if (hca_capacity // hca_compressed_kv_cp) % hca_attn_blocks:
                raise ValueError("HCA CP-way slot count must be divisible by hca_attn_blocks.")
    config.qeff_csa_attention_dp = attention_dp
    config.qeff_csa_indexer_cp = indexer_cp
    config.qeff_csa_num_kv_blocks = num_kv_blocks
    config.qeff_csa_indexer_attention_cores = attention_cores
    config.qeff_csa_folded_row_cache = True
    csa_prefill_attn_gather_sub_qc = getattr(args, "csa_prefill_attn_gather_sub_qc", None)
    if csa_prefill_attn_gather_sub_qc is not None:
        config.qeff_csa_prefill_attn_gather_sub_qc = csa_prefill_attn_gather_sub_qc
    csa_prefill_indexer_kv_block_size = getattr(args, "csa_prefill_indexer_kv_block_size", None)
    if csa_prefill_indexer_kv_block_size is not None:
        config.qeff_csa_prefill_indexer_kv_block_size = csa_prefill_indexer_kv_block_size
    csa_prefill_indexer_topk_per_block = getattr(args, "csa_prefill_indexer_topk_per_block", None)
    if csa_prefill_indexer_topk_per_block is not None:
        config.qeff_csa_prefill_indexer_topk_per_block = csa_prefill_indexer_topk_per_block
    config.qeff_hca_attention_dp = attention_dp
    config.qeff_hca_compressed_kv_cp = hca_compressed_kv_cp
    config.qeff_hca_attn_blocks = hca_attn_blocks
    config.qeff_hca_folded_row_cache = True
    print(
        "Folded decode layout: "
        f"batch_size={args.batch_size}, attn_dp={attention_dp}, indexer_cp={indexer_cp}, "
        f"num_kv_blocks={num_kv_blocks}, "
        f"hw_version={args.hw_version}, hca_compressed_kv_cp={hca_compressed_kv_cp}, "
        f"hca_attn_blocks={hca_attn_blocks}, csa_capacity={csa_capacity}, hca_capacity={hca_capacity}"
    )


def get_first_config_value(config, names: Iterable[str], default=None, cast_int: bool = False):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value) if cast_int else value
    return default


def resolve_attention_heads(config) -> Optional[int]:
    return get_first_config_value(config, ATTENTION_HEAD_CONFIG_KEYS, cast_int=True)


def resolve_kv_heads(config) -> Optional[int]:
    value = get_first_config_value(config, KV_HEAD_CONFIG_KEYS, cast_int=True)
    if value is None:
        value = resolve_attention_heads(config)
    return value


def resolve_hidden_size(config) -> Optional[int]:
    return get_first_config_value(config, HIDDEN_SIZE_CONFIG_KEYS, cast_int=True)


def set_kv_head_aliases(config, value: int):
    setattr(config, "num_key_value_heads", value)
    for key in KV_HEAD_CONFIG_KEYS:
        if hasattr(config, key):
            setattr(config, key, value)


def calculate_num_replicate_kv_heads(num_devices: int, text_model_config) -> int:
    """
    Choose a KV-repeat value from model config and device count.

    Primary criteria:
    1. MQA models do not request replication; the compiler handles them internally.
    2. GQA models reach one replicated KV head per device.
    3. num_attention_heads is divisible by the replicated KV-head count.

    Returns 1 if no valid repeat exists (replication not applicable or not achievable).
    """
    num_attention_heads = resolve_attention_heads(text_model_config)
    num_kv_heads = resolve_kv_heads(text_model_config)

    if num_attention_heads is None or num_kv_heads is None or num_attention_heads < 1 or num_kv_heads < 1:
        return None

    # MQA already has a single shared KV head.  Its expansion is a compiler
    # concern, so keep the PyTorch model and its checkpoint tensors untouched.
    if num_kv_heads == 1:
        return 1

    num_devices = max(1, int(num_devices))
    max_repeat = max(1, num_attention_heads // num_kv_heads)
    if num_devices <= num_kv_heads or num_devices % num_kv_heads != 0:
        # There is no GQA replication to perform when the existing KV layout
        # already covers the device group, or when the device count cannot be
        # reached by an integral repeat of the original KV-head count.
        return None
    for repeat in range(2, max_repeat + 1):
        new_kv_heads = num_kv_heads * repeat
        # The device-count divisibility check alone also accepts a smaller
        # KV-head count (for example, 2 heads across 4 devices).  For GQA we
        # need the repeated layout to provide one effective KV head per device:
        # together, ``num_devices % new_kv_heads == 0`` and
        # ``num_devices <= new_kv_heads`` require ``new_kv_heads`` to equal
        # ``num_devices``.  The attention-head check preserves a valid GQA
        # grouping after the projection weights are expanded.
        if (
            (num_devices % new_kv_heads == 0)
            and (num_devices <= new_kv_heads)
            and (num_attention_heads % new_kv_heads == 0)
        ):
            return repeat

    return 1
