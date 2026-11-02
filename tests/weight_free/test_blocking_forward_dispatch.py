# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Weight-free suite mirror of the CPU-only blocking forward dispatch tests.

The implementation under test is shared by dynamo and weight-free exports. Keep
these wrappers in tests/weight_free so blocking dispatch coverage still runs in
CI jobs that disable the tests/dynamo node.
"""

from __future__ import annotations

import pytest

from tests.dynamo.unit_tests import test_blocking_forward_dispatch as dynamo_dispatch

_ENABLED_BLOCKING_MODES = {"h", "q", "kv", "qkv", "hq", "hkv", "hqkv", "bhqkv", "kv_headpar"}
_ENABLED_DISPATCH_CASES = [
    case
    for case in dynamo_dispatch.DISPATCH_CASES
    if case.qaic_config["blocking_mode"] in _ENABLED_BLOCKING_MODES
    and case.expected_mode.value in _ENABLED_BLOCKING_MODES
]


def _enabled_case_id(case) -> str:
    return dynamo_dispatch._case_id(case)


@pytest.mark.weight_free
@pytest.mark.weight_free_blocking
@pytest.mark.parametrize("case", _ENABLED_DISPATCH_CASES, ids=_enabled_case_id)
def test_weight_free_blocking_forward_dispatch(case):
    dynamo_dispatch.test_dynamo_blocking_forward_dispatch(case)


@pytest.mark.weight_free
@pytest.mark.weight_free_blocking
@pytest.mark.parametrize("case", dynamo_dispatch._LLAMA_CPU_PARITY_CASES, ids=dynamo_dispatch._case_id)
def test_weight_free_llama_blocking_cpu_logits_parity(case):
    dynamo_dispatch.test_dynamo_llama_blocking_cpu_logits_parity(case)


@pytest.mark.weight_free
@pytest.mark.weight_free_blocking
@pytest.mark.parametrize(
    ("attention_cfg", "expected_mode"),
    [
        (
            {"head_blocking_enabled": False, "head_block_size": 1, "num_q_blocks": 2, "num_kv_blocks": 1},
            dynamo_dispatch.BlockingMode.Q,
        ),
        (
            {"head_blocking_enabled": False, "head_block_size": 1, "num_q_blocks": 1, "num_kv_blocks": 2},
            dynamo_dispatch.BlockingMode.KV,
        ),
        (
            {"head_blocking_enabled": True, "head_block_size": 2, "num_q_blocks": 2, "num_kv_blocks": 2},
            dynamo_dispatch.BlockingMode.HQKV,
        ),
    ],
    ids=["hqkv_to_q", "hqkv_to_kv", "hqkv_stays_hqkv"],
)
def test_weight_free_blocking_auto_hqkv_reduction_is_explicit(attention_cfg, expected_mode):
    dynamo_dispatch.test_dynamo_blocking_auto_hqkv_reduction_is_explicit(attention_cfg, expected_mode)
