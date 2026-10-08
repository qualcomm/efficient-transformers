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
    "hw_version": "ai100",
}

MICROBENCH_DEFAULTS = DECODE_MICROBENCH_DEFAULTS
MICROBENCH_DEFAULTS_BY_MODE = {
    "decode": DECODE_MICROBENCH_DEFAULTS,
    "prefill": PREFILL_MICROBENCH_DEFAULTS,
}


def parse_microbench_mode() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=MICROBENCH_DEFAULTS_BY_MODE, default="decode")
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return args.mode


if __name__ == "__main__":
    main(MICROBENCH_DEFAULTS_BY_MODE[parse_microbench_mode()])
