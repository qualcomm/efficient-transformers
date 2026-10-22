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
import torch

from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.blocking.blocked_attention_forwards import blocked_qkv_attention_forward_prefill_online
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
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


@pytest.mark.weight_free
@pytest.mark.weight_free_blocking
def test_weight_free_blocking_transform_clears_stale_prefill_config():
    """A prefill-blocked export must not poison a later unblocked decode export."""
    qeff_model = QEFFAutoModelForCausalLM(dynamo_dispatch._make_tiny_llama(), weight_free=True)
    qeff_model.transform(
        ctx_len=dynamo_dispatch.CTX_LEN,
        seq_len=dynamo_dispatch.SEQ_LEN,
        qaic_config={
            "blocking_mode": "prefill_online",
            "num_kv_blocks": dynamo_dispatch.NUM_KV_BLOCKS,
            "num_q_blocks": dynamo_dispatch.NUM_Q_BLOCKS,
            "n_rep_chunk": 1,
        },
        aic_num_cores=4,
        prefill_only=True,
        prefill_seq_len=dynamo_dispatch.SEQ_LEN,
    )
    blocked_modules = [module for module in qeff_model.model.modules() if hasattr(module, "attn_blocking_config")]
    assert blocked_modules
    assert all(module.attn_blocking_config.mode == BlockingMode.PREFILL_ONLINE for module in blocked_modules)

    qeff_model.transform(ctx_len=dynamo_dispatch.CTX_LEN, seq_len=1, qaic_config=None)

    assert all(module.attn_blocking_config.mode == BlockingMode.NONE for module in blocked_modules)


@pytest.mark.weight_free
@pytest.mark.weight_free_blocking
def test_weight_free_prefill_online_honors_attention_mask():
    class _Module:
        num_key_value_groups = 1

    class _Cache:
        def __init__(self, key, value):
            self.key = key
            self.value = value

        def read_only_blocked_K(self, start_index, end_index, layer_idx, cache_kwargs):
            return self.key[:, :, start_index:end_index, :]

        def read_only_blocked_V(self, start_index, end_index, layer_idx, cache_kwargs):
            return self.value[:, :, start_index:end_index, :]

    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]]])
    value = torch.tensor([[[[10.0, 0.0], [0.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]])
    position_ids = torch.tensor([[0, 1]])
    attention_mask = torch.zeros((1, 1, 2, 4), dtype=torch.bool)
    attention_mask[:, :, :, 2:] = True
    attention_mask[:, :, 1, 0] = True

    output, _ = blocked_qkv_attention_forward_prefill_online(
        module=_Module(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        scaling=1.0,
        num_q_blocks=1,
        num_kv_blocks=2,
        cache_kwargs={"position_ids": position_ids},
        layer_idx=0,
        past_key_value=_Cache(key, value),
        ctx_len=4,
    )

    scores = torch.matmul(query, key.transpose(-1, -2)).masked_fill(attention_mask, float("-inf"))
    expected = torch.matmul(torch.softmax(scores, dim=-1), value).transpose(1, 2).contiguous()
    torch.testing.assert_close(output, expected)
