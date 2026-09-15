# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run DeepSeek-V4-Flash with the CSA DP16/CP16 decode microbenchmark layout."""

if __package__:
    from .deepseek_v4_flash_decode import main
else:
    from deepseek_v4_flash_decode import main

MICROBENCH_DEFAULTS = {
    "batch_size": 16,
    "ctx_len": 262144,
    "num_cores": 16,
    "device_group": list(range(16)),
    "csa_attention_dp": 16,
    "csa_indexer_cp": 16,
    "csa_folded_row_cache": True,
}


if __name__ == "__main__":
    main(MICROBENCH_DEFAULTS)
