# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run DeepSeek-V4-Flash with the benchmark's folded DP/CP decode layout."""

if __package__:
    from .deepseek_v4_flash_decode import main
else:
    from deepseek_v4_flash_decode import main

MICROBENCH_DEFAULTS = {
    "batch_size": 16,
    "ctx_len": 262144,
    "num_hidden_layers": 4,
    "num_cores": 16,
    "device_group": list(range(16)),
    "attn_dp": 16,
    "indexer_cp": 16,
    "num_kv_blocks": 4,
    "hca_compressed_kv_cp": 1,
    "hca_attn_blocks": 16,
    "hw_version": "ai100",
}


if __name__ == "__main__":
    main(MICROBENCH_DEFAULTS)
