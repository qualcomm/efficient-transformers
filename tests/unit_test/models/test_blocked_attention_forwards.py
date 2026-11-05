# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for the KV-block gather inside QEfficient.blocking.blocked_attention_forwards.

Covers, for `blocked_qkv_attention_forward`, `blocked_hqkv_attention_forward` and
`blocked_bhqkv_attention_forward` with `paged_attention=True`:
  1. The per-KV-block gather (`_read_kv_block`) runs exactly once per KV block per
     forward call, not once per (KV block x head/q/batch block) combination.
  2. The gathered output is still numerically correct (matches plain causal
     attention over the same paged cache).
  3. `skip_kv`'s early break out of the KV-block loop still stops the gather as
     soon as a block lies entirely in the future.

All tests run on CPU only, using hand-built paged KV caches (no full model needed).
"""

from unittest.mock import patch

import pytest
import torch

from QEfficient.blocking.blocked_attention_forwards import (
    _read_kv_block,
    blocked_bhqkv_attention_forward,
    blocked_hqkv_attention_forward,
    blocked_qkv_attention_forward,
    blocked_qkv_attention_forward_prefill_online,
    repeat_kv,
    update_running_softmax,
    update_running_softmax_prefill,
)
from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffDynamicLayer

HEAD_DIM = 4
BLOCK_SIZE = 2
NUM_PHYS_BLOCKS = 4
CTX_LEN = BLOCK_SIZE * NUM_PHYS_BLOCKS  # 8
SCALING = 1.0 / (HEAD_DIM**0.5)


class _DummyModule:
    """Minimal attention-module double: no `config` attribute (forces the
    value.dtype fallback for the mask) and an explicit `num_key_value_groups`
    so `_get_kv_states` takes its no-op path (num_kv_heads == num_heads)."""

    def __init__(self, num_key_value_groups=1):
        self.num_key_value_groups = num_key_value_groups


class _PrefillCache:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def read_only_blocked_K(self, start_index, end_index, layer_idx, cache_kwargs):
        return self.key[:, :, start_index:end_index, :]

    def read_only_blocked_V(self, start_index, end_index, layer_idx, cache_kwargs):
        return self.value[:, :, start_index:end_index, :]


class TestRunningSoftmaxMaskedRows:
    @staticmethod
    def _masked_inputs():
        current_max = torch.full((1, 1, 1), float("-inf"))
        attn_weights_block = torch.full((1, 1, 1, 2), float("-inf"))
        current_denominator = torch.zeros(1, 1, 1)
        output = torch.zeros(1, 1, 1, HEAD_DIM)
        v_block = torch.ones(1, 1, 2, HEAD_DIM)
        return current_max, attn_weights_block, current_denominator, output, v_block

    def test_decoder_running_softmax_keeps_fully_masked_rows_finite(self):
        inputs = self._masked_inputs()
        current_max, denominator, output = update_running_softmax(*inputs)

        assert torch.equal(current_max, inputs[0])
        assert torch.equal(denominator, inputs[2])
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output) == 0

    def test_prefill_running_softmax_keeps_fully_masked_rows_finite(self):
        inputs = self._masked_inputs()
        current_max, denominator, output = update_running_softmax_prefill(*inputs)

        assert torch.equal(current_max, inputs[0])
        assert torch.equal(denominator, inputs[2])
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output) == 0


class TestPrefillOnlineMaskedRows:
    def test_masked_rows_and_invalid_values_stay_finite(self):
        key = torch.zeros(1, 1, 2, HEAD_DIM)
        value = torch.cat([torch.ones(1, 1, 1, HEAD_DIM), torch.full((1, 1, 1, HEAD_DIM), float("nan"))], dim=2)
        query = torch.zeros(1, 1, 2, HEAD_DIM)
        position_ids = torch.tensor([[0, -1]])

        output, _ = blocked_qkv_attention_forward_prefill_online(
            module=_DummyModule(),
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            scaling=SCALING,
            num_q_blocks=1,
            num_kv_blocks=1,
            cache_kwargs={"position_ids": position_ids},
            layer_idx=0,
            past_key_value=_PrefillCache(key, value),
            ctx_len=2,
        )

        expected = torch.cat([torch.ones(1, 1, 1, HEAD_DIM), torch.zeros(1, 1, 1, HEAD_DIM)], dim=2).transpose(1, 2)
        assert torch.isfinite(output).all()
        assert torch.allclose(output, expected)


def _make_paged_cache(heads, block_size=BLOCK_SIZE, num_phys_blocks=NUM_PHYS_BLOCKS, head_dim=HEAD_DIM):
    cache = QEffDynamicCache()
    layer = QEffDynamicLayer()
    layer.keys = torch.randn(num_phys_blocks, heads, block_size, head_dim)
    layer.values = torch.randn(num_phys_blocks, heads, block_size, head_dim)
    layer.is_initialized = True
    cache.layers.append(layer)
    return cache


def _full_kv(cache, layer_idx=0):
    """Concatenate all physical blocks into one contiguous (heads, CTX_LEN, head_dim)
    tensor, matching what an identity block_table exposes to every batch row."""
    layer = cache.layers[layer_idx]
    k_full = torch.cat(list(layer.keys), dim=1)
    v_full = torch.cat(list(layer.values), dim=1)
    return k_full, v_full


def _manual_causal_attention(query, k_full, v_full, query_positions, scaling):
    """Reference: plain (unblocked) causal attention over `k_full`/`v_full`."""
    batch, _heads, q_len, _ = query.shape
    ctx_len = k_full.shape[1]
    k = k_full.unsqueeze(0).expand(batch, -1, -1, -1)
    v = v_full.unsqueeze(0).expand(batch, -1, -1, -1)
    attn_weights = torch.matmul(query, k.transpose(2, 3)) * scaling
    kv_indices = torch.arange(ctx_len).view(1, 1, 1, -1)
    q_pos = query_positions.view(batch, 1, q_len, 1)
    causal_mask = kv_indices > q_pos
    attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def _identity_block_table(batch, num_kv_blocks):
    return torch.arange(num_kv_blocks).unsqueeze(0).expand(batch, -1).clone()


def _cache_kwargs(position_ids, block_table):
    return {"position_ids": position_ids, "block_table": block_table}


# ---------------------------------------------------------------------------
# Tests: hoisted gather is called once per KV block, not once per KV block per
# head/q/batch block
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestHoistedGatherCallCount:
    """After hoisting, `_read_kv_block` must be called exactly once per KV
    block per forward call, regardless of how many head/q/batch blocks wrap
    around the KV-block loop."""

    def test_qkv_gathers_each_kv_block_once(self):
        heads = 2
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        query = torch.randn(1, heads, 2, HEAD_DIM)
        cache_kwargs = _cache_kwargs(torch.tensor([[6, 7]]), _identity_block_table(1, NUM_PHYS_BLOCKS))

        with patch("QEfficient.blocking.blocked_attention_forwards._read_kv_block", wraps=_read_kv_block) as mocked:
            blocked_qkv_attention_forward(
                module=module,
                query=query,
                key=None,
                value=torch.zeros(1, heads, 1, HEAD_DIM),
                attention_mask=None,
                scaling=SCALING,
                num_kv_blocks=NUM_PHYS_BLOCKS,
                num_q_blocks=2,
                cache_kwargs=cache_kwargs,
                layer_idx=0,
                past_key_value=cache,
                paged_attention=True,
                use_causal_mask=True,
                skip_kv=False,
                ctx_len=CTX_LEN,
            )
        assert mocked.call_count == NUM_PHYS_BLOCKS, (
            f"Expected the KV gather to run once per KV block ({NUM_PHYS_BLOCKS}), got "
            f"{mocked.call_count} calls -- the per-q-block gather was not hoisted."
        )

    def test_hqkv_gathers_each_kv_block_once(self):
        heads = 4
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        query = torch.randn(1, heads, 2, HEAD_DIM)
        cache_kwargs = _cache_kwargs(torch.tensor([[6, 7]]), _identity_block_table(1, NUM_PHYS_BLOCKS))

        with patch("QEfficient.blocking.blocked_attention_forwards._read_kv_block", wraps=_read_kv_block) as mocked:
            blocked_hqkv_attention_forward(
                module=module,
                query=query,
                key=None,
                value=torch.zeros(1, heads, 1, HEAD_DIM),
                attention_mask=None,
                scaling=SCALING,
                num_kv_blocks=NUM_PHYS_BLOCKS,
                num_q_blocks=2,
                head_block_size=2,
                cache_kwargs=cache_kwargs,
                layer_idx=0,
                past_key_value=cache,
                paged_attention=True,
                use_causal_mask=True,
                skip_kv=False,
                ctx_len=CTX_LEN,
            )
        assert mocked.call_count == NUM_PHYS_BLOCKS, (
            f"Expected the KV gather to run once per KV block ({NUM_PHYS_BLOCKS}) regardless of "
            f"head/q blocking, got {mocked.call_count} calls."
        )

    def test_bhqkv_gathers_each_kv_block_once(self):
        heads = 4
        batch = 2
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        query = torch.randn(batch, heads, 2, HEAD_DIM)
        cache_kwargs = _cache_kwargs(torch.tensor([[6, 7], [6, 7]]), _identity_block_table(batch, NUM_PHYS_BLOCKS))

        with patch("QEfficient.blocking.blocked_attention_forwards._read_kv_block", wraps=_read_kv_block) as mocked:
            blocked_bhqkv_attention_forward(
                module=module,
                query=query,
                key=None,
                value=torch.zeros(batch, heads, 1, HEAD_DIM),
                attention_mask=None,
                scaling=SCALING,
                num_kv_blocks=NUM_PHYS_BLOCKS,
                num_q_blocks=2,
                num_batch_blocks=2,
                head_block_size=2,
                cache_kwargs=cache_kwargs,
                layer_idx=0,
                past_key_value=cache,
                paged_attention=True,
                use_causal_mask=True,
                skip_kv=False,
                ctx_len=CTX_LEN,
            )
        assert mocked.call_count == NUM_PHYS_BLOCKS, (
            f"Expected the KV gather to run once per KV block ({NUM_PHYS_BLOCKS}) regardless of "
            f"head/q/batch blocking, got {mocked.call_count} calls."
        )


# ---------------------------------------------------------------------------
# Tests: hoisted gather output is still numerically correct
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestHoistedGatherCorrectness:
    """The hoisted gather must still produce numerically identical output to a
    plain (unblocked) causal-attention computation over the same paged cache."""

    def test_qkv_paged_matches_manual_causal_attention(self):
        heads = 2
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        k_full, v_full = _full_kv(cache)

        query = torch.randn(1, heads, 2, HEAD_DIM)
        position_ids = torch.tensor([[6, 7]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(1, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_qkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(1, heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=False,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full, v_full, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather qkv output diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )

    def test_hqkv_paged_matches_manual_causal_attention(self):
        """Exercises the per-head slicing (k_block_states[:, h_start:h_end])
        that now reads from the hoisted/cached kv_blocks tuple instead of a
        freshly gathered k_block -- this is the line most likely to break if
        hoisting introduced a stale-block or index bug."""
        heads = 4
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        k_full, v_full = _full_kv(cache)

        query = torch.randn(1, heads, 3, HEAD_DIM)
        position_ids = torch.tensor([[5, 6, 7]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(1, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_hqkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(1, heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=2,
            head_block_size=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=False,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full, v_full, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather hqkv output diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )

    def test_bhqkv_paged_matches_manual_causal_attention_per_row(self):
        """Exercises both the per-head and per-batch slicing of the cached
        kv_blocks tuple, with two batch rows at *different* positions so a
        row/index mixup in the hoisted cache would show up as cross-row
        contamination rather than cancelling out."""
        heads = 4
        batch = 2
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        k_full, v_full = _full_kv(cache)

        query = torch.randn(batch, heads, 2, HEAD_DIM)
        position_ids = torch.tensor([[5, 6], [3, 4]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(batch, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_bhqkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(batch, heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=2,
            num_batch_blocks=2,
            head_block_size=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=False,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full, v_full, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather bhqkv output diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )

    def test_hqkv_paged_gqa_matches_manual_causal_attention(self):
        """GQA case (num_key_value_groups > 1): the hoisted pre-pass calls
        _get_kv_states (repeat_kv expansion) once per KV block, same as
        before hoisting -- verify that expansion is still correct when read
        back out of the cached kv_blocks tuple."""
        num_kv_heads = 2
        num_q_heads = 4
        cache = _make_paged_cache(heads=num_kv_heads)
        module = _DummyModule(num_key_value_groups=num_q_heads // num_kv_heads)
        k_full, v_full = _full_kv(cache)
        k_full_rep = repeat_kv(k_full.unsqueeze(0), module.num_key_value_groups).squeeze(0)
        v_full_rep = repeat_kv(v_full.unsqueeze(0), module.num_key_value_groups).squeeze(0)

        query = torch.randn(1, num_q_heads, 3, HEAD_DIM)
        position_ids = torch.tensor([[5, 6, 7]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(1, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_hqkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(1, num_q_heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=1,
            head_block_size=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=False,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full_rep, v_full_rep, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather GQA hqkv output diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )


# ---------------------------------------------------------------------------
# Tests: skip_kv early break is preserved by the hoisted pre-pass
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestSkipKvBreakPreserved:
    """The hoisted pre-pass must still break out of the KV-block loop as soon
    as a block lies entirely in the future, exactly as the un-hoisted loop did."""

    def test_hqkv_break_stops_gather_early(self):
        heads = 4
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        query = torch.randn(1, heads, 1, HEAD_DIM)
        # current position 3 falls in logical block 1 (start_index=2); block 2
        # (start_index=4) and block 3 (start_index=6) are entirely in the future.
        cache_kwargs = _cache_kwargs(torch.tensor([[3]]), _identity_block_table(1, NUM_PHYS_BLOCKS))

        with patch("QEfficient.blocking.blocked_attention_forwards._read_kv_block", wraps=_read_kv_block) as mocked:
            blocked_hqkv_attention_forward(
                module=module,
                query=query,
                key=None,
                value=torch.zeros(1, heads, 1, HEAD_DIM),
                attention_mask=None,
                scaling=SCALING,
                num_kv_blocks=NUM_PHYS_BLOCKS,
                num_q_blocks=1,
                head_block_size=2,
                cache_kwargs=cache_kwargs,
                layer_idx=0,
                past_key_value=cache,
                paged_attention=True,
                use_causal_mask=True,
                skip_kv=True,
                ctx_len=CTX_LEN,
            )
        expected_calls = 2  # blocks j=0,1 cover position 3; j=2 (start_index=4) is entirely future
        assert mocked.call_count == expected_calls, (
            f"Expected skip_kv to stop gathering after {expected_calls} KV blocks once a block "
            f"lies entirely in the future, got {mocked.call_count} calls."
        )

    def test_hqkv_skip_kv_matches_manual_causal_attention(self):
        """The call-count test above only proves the pre-pass gathers the right
        *number* of blocks once skip_kv breaks early -- it never checks that the
        surviving kv_blocks still produce numerically correct output. Verify
        that separately here: with skip_kv=True the blocks lying entirely in
        the future are never gathered at all (rather than gathered and then
        masked to -inf), so this must still match plain causal attention over
        the full cache."""
        heads = 4
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        k_full, v_full = _full_kv(cache)

        query = torch.randn(1, heads, 1, HEAD_DIM)
        position_ids = torch.tensor([[3]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(1, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_hqkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(1, heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=1,
            head_block_size=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=True,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full, v_full, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather hqkv output with skip_kv=True diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )
