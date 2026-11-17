# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run DeepSeek-V4-Flash microbenchmarks with decode or prefill defaults."""

import argparse
import sys

if __package__:
    from .deepseek_v4_flash_decode import main
else:
    from deepseek_v4_flash_decode import main

DECODE_MICROBENCH_DEFAULTS = {
    "batch_size": 16,
    "ctx_len": 262144,
    "num_hidden_layers": 4,
    "num_cores": 16,
    "device_group": list(range(16)),
    "attn_dp": 16,
    "indexer_cp": 16,
    "num_kv_blocks": 1,
    "hca_compressed_kv_cp": 2,
    "hca_attn_blocks": 8,
    "hw_version": "ai100",
    "ffn_blocking_mode": "token",
    "ffn_token_block_size": 4,
}

PREFILL_MICROBENCH_DEFAULTS = {
    "prefill_only": True,
    "prefill_seq_len": 1024,
    "batch_size": 1,
    "ctx_len": 1048576,
    "num_hidden_layers": 4,
    "num_cores": 16,
    "device_group": [0],
    "attn_dp": 1,
    "indexer_cp": 1,
    "num_kv_blocks": 1,
    "hca_compressed_kv_cp": 1,
    "hca_attn_blocks": 1,
    "hca_prefill_q_block_size": 128,
    "csa_prefill_attn_gather_sub_qc": 4,
    "csa_prefill_indexer_kv_block_size": 512,
    "csa_prefill_indexer_topk_per_block": 4,
    "hw_version": "ai100",
    "mdp_num_partitions": 1,
    "mdp_strategy": "onnx",
}

MICROBENCH_DEFAULTS = DECODE_MICROBENCH_DEFAULTS
MICROBENCH_DEFAULTS_BY_MODE = {
    "decode": DECODE_MICROBENCH_DEFAULTS,
    "prefill": PREFILL_MICROBENCH_DEFAULTS,
}


def parse_microbench_defaults() -> dict[str, object]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=MICROBENCH_DEFAULTS_BY_MODE, default="decode")
    parser.add_argument("--pp-size", type=int, default=None, help="Pipeline-parallel MDP partitions.")
    parser.add_argument("--hca-prefill-q-block-size", type=int, default=None, help="HCA prefill query block size.")
    parser.add_argument(
        "--csa-prefill-q-block-size",
        type=int,
        default=None,
        help="CSA prefill compressed-attention query block size.",
    )
    parser.add_argument(
        "--csa-block-size",
        type=int,
        default=None,
        help="CSA prefill indexer KV block size.",
    )
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    defaults = dict(MICROBENCH_DEFAULTS_BY_MODE[args.mode])
    overrides = {
        "mdp_num_partitions": args.pp_size,
        "hca_prefill_q_block_size": args.hca_prefill_q_block_size,
        "csa_prefill_attn_gather_sub_qc": args.csa_prefill_q_block_size,
        "csa_prefill_indexer_kv_block_size": args.csa_block_size,
    }
    defaults.update({name: value for name, value in overrides.items() if value is not None})
    return defaults


if __name__ == "__main__":
    main(parse_microbench_defaults())
