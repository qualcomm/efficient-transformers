# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Correctness tests for QEfficient cache utilities.

Tests verify numerical correctness of:
  - QEffDynamicLayer: scatter/gather round-trip
  - QEffDynamicCache: multi-layer update, write/read, prefill+decode
  - QEffEncoderDecoderCache: from_legacy_cache
  - InvalidIndexProvider: value logic

All tests run on CPU only.
"""

import pytest
import torch

from QEfficient.transformers.cache_utils import (
    InvalidIndexProvider,
    QEffDynamicCache,
    QEffDynamicLayer,
    QEffEncoderDecoderCache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kv(batch=1, heads=2, seq=8, head_dim=16):
    k = torch.randn(batch, heads, seq, head_dim)
    v = torch.randn(batch, heads, seq, head_dim)
    return k, v


def pos_ids(batch=1, seq=8, start=0):
    return torch.arange(start, start + seq).unsqueeze(0).expand(batch, -1)


# ---------------------------------------------------------------------------
# Tests: InvalidIndexProvider
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestInvalidIndexProvider:
    """InvalidIndexProvider must return 0 outside ONNX export."""

    def test_returns_zero_outside_onnx_export(self):
        val = InvalidIndexProvider._get_invalid_idx_value()
        assert val == 0, f"Expected 0 outside ONNX export, got {val}"

    def test_subfunc_disabled_by_default(self):
        assert InvalidIndexProvider.SUBFUNC_ENABLED is False

    def test_enable_subfunc_sets_flag(self):
        original = InvalidIndexProvider.SUBFUNC_ENABLED
        try:
            InvalidIndexProvider.enable_subfunc()
            assert InvalidIndexProvider.SUBFUNC_ENABLED is True
        finally:
            InvalidIndexProvider.SUBFUNC_ENABLED = original


# ---------------------------------------------------------------------------
# Tests: QEffDynamicLayer
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffDynamicLayerCorrectness:
    """QEffDynamicLayer scatter/gather must be numerically correct."""

    def test_initial_state_is_none(self):
        layer = QEffDynamicLayer()
        assert layer.keys is None
        assert layer.values is None

    def test_first_update_stores_tensors(self):
        layer = QEffDynamicLayer()
        k, v = make_kv(seq=8)
        k_out, v_out = layer.update(k, v, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert layer.keys is not None
        assert layer.values is not None
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_write_then_read_returns_same_values(self):
        """write_only then read_only must return the exact same tensors."""
        layer = QEffDynamicLayer()
        k, v = make_kv(batch=1, heads=2, seq=8, head_dim=16)
        pids = pos_ids(seq=8)

        layer.write_only(k, v, cache_kwargs={"position_ids": pids})
        k_out, v_out = layer.read_only(cache_kwargs={"position_ids": pids})

        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
        assert torch.allclose(k_out, k), "read_only must return the same keys as written"
        assert torch.allclose(v_out, v), "read_only must return the same values as written"

    def test_update_output_has_ctx_len_dimension(self):
        """After update, output must have the context length dimension."""
        layer = QEffDynamicLayer()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8
        k = torch.zeros(batch, heads, ctx_len, head_dim)
        v = torch.zeros(batch, heads, ctx_len, head_dim)
        pids = pos_ids(seq=ctx_len)

        k_out, v_out = layer.update(k, v, cache_kwargs={"position_ids": pids})
        assert k_out.shape == (batch, heads, ctx_len, head_dim)
        assert v_out.shape == (batch, heads, ctx_len, head_dim)

    def test_decode_step_scatter_at_correct_position(self):
        """Decode step must scatter the new token at the correct position."""
        layer = QEffDynamicLayer()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8

        # Initialize with zeros
        k_init = torch.zeros(batch, heads, ctx_len, head_dim)
        v_init = torch.zeros(batch, heads, ctx_len, head_dim)
        layer.update(k_init, v_init, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode: write a known value at position 5
        k_new = torch.ones(batch, heads, 1, head_dim) * 7.0
        v_new = torch.ones(batch, heads, 1, head_dim) * 7.0
        pos_decode = torch.tensor([[5]])

        k_out, v_out = layer.update(k_new, v_new, cache_kwargs={"position_ids": pos_decode})

        assert k_out.shape[2] == ctx_len
        assert k_out[0, 0, 5, 0].item() == pytest.approx(7.0, abs=1e-5), (
            f"Expected 7.0 at position 5, got {k_out[0, 0, 5, 0].item()}"
        )

    def test_update_output_is_finite(self):
        layer = QEffDynamicLayer()
        k, v = make_kv(seq=8)
        k_out, v_out = layer.update(k, v, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_batch_fold_continuous_batching_uses_preordered_physical_slots(self):
        batch_size, heads, ctx_len, head_dim = 3, 2, 8, 4
        layer = QEffDynamicLayer.from_tensors(
            torch.zeros(batch_size, heads, ctx_len, head_dim),
            torch.zeros(batch_size, heads, ctx_len, head_dim),
        )

        batch_index = torch.tensor([[2], [0], [1]], dtype=torch.int64)
        logical_positions = torch.tensor([[3], [5], [1]], dtype=torch.int32)
        logical_keys = torch.stack(
            [
                torch.full((heads, 1, head_dim), 2.0),
                torch.full((heads, 1, head_dim), 7.0),
                torch.full((heads, 1, head_dim), 11.0),
            ]
        )
        logical_values = logical_keys + 10.0

        slots = batch_index.flatten()
        physical_positions = torch.empty_like(logical_positions).index_copy(0, slots, logical_positions)
        physical_keys = torch.empty_like(logical_keys).index_copy(0, slots, logical_keys)
        physical_values = torch.empty_like(logical_values).index_copy(0, slots, logical_values)
        layer.write_only_batch(
            physical_keys,
            physical_values,
            cache_kwargs={"position_ids": physical_positions},
        )

        assert layer.keys.shape == (batch_size, heads, ctx_len, head_dim)
        assert layer.values.shape == (batch_size, heads, ctx_len, head_dim)
        for logical_row, physical_slot in enumerate(slots):
            position = logical_positions[logical_row, 0]
            assert torch.equal(layer.keys[physical_slot, :, position], logical_keys[logical_row, :, 0])
            assert torch.equal(layer.values[physical_slot, :, position], logical_values[logical_row, :, 0])

        read_kwargs = {"position_ids": torch.full((batch_size, 1), ctx_len - 1)}
        key_out = layer.read_only_blocked_K_batch(0, 6, read_kwargs).reshape(batch_size, heads, 6, head_dim)
        value_out = layer.read_only_blocked_V_batch(0, 6, read_kwargs).reshape(batch_size, heads, 6, head_dim)
        assert torch.equal(key_out, layer.keys[:, :, :6])
        assert torch.equal(value_out, layer.values[:, :, :6])

    def test_batch_fold_write_preserves_standard_cache_layout(self):
        batch, heads, ctx_len, head_dim = 3, 2, 8, 4
        layer = QEffDynamicLayer.from_tensors(
            torch.zeros(batch, heads, ctx_len, head_dim),
            torch.zeros(batch, heads, ctx_len, head_dim),
        )
        positions = torch.tensor([[1], [3], [5]])
        keys = torch.arange(batch * heads * head_dim, dtype=torch.float32).reshape(batch, heads, 1, head_dim)
        values = keys + 100

        layer.write_only_batch(keys, values, cache_kwargs={"position_ids": positions})

        assert layer.keys.shape == (batch, heads, ctx_len, head_dim)
        assert layer.values.shape == (batch, heads, ctx_len, head_dim)
        for batch_idx, position in enumerate(positions.flatten()):
            assert torch.equal(layer.keys[batch_idx, :, position], keys[batch_idx, :, 0])
            assert torch.equal(layer.values[batch_idx, :, position], values[batch_idx, :, 0])

    def test_batch_fold_read_uses_folded_compute_view(self):
        batch, heads, ctx_len, head_dim = 3, 2, 8, 4
        keys = torch.arange(batch * heads * ctx_len * head_dim, dtype=torch.float32).reshape(
            batch, heads, ctx_len, head_dim
        )
        values = keys + 100
        layer = QEffDynamicLayer.from_tensors(keys, values)
        positions = torch.full((batch, 1), ctx_len - 1)
        cache_kwargs = {"position_ids": positions, "num_kv_heads": heads}

        folded_keys, folded_values = layer.get_batch_folded_kv()
        key_block = layer.read_only_blocked_K_batch(2, 6, cache_kwargs, folded_cache=folded_keys)
        value_block = layer.read_only_blocked_V_batch(2, 6, cache_kwargs, folded_cache=folded_values)

        expected_keys = keys[:, :, 2:6].reshape(1, batch * heads, 4, head_dim)
        expected_values = values[:, :, 2:6].reshape(1, batch * heads, 4, head_dim)
        assert folded_keys.shape == (1, batch * heads, ctx_len, head_dim)
        assert folded_values.shape == (1, batch * heads, ctx_len, head_dim)
        assert key_block.shape == (1, batch * heads, 4, head_dim)
        assert value_block.shape == (1, batch * heads, 4, head_dim)
        assert torch.equal(key_block, expected_keys)
        assert torch.equal(value_block, expected_values)


# ---------------------------------------------------------------------------
# Tests: QEffDynamicCache
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffDynamicCacheCorrectness:
    """QEffDynamicCache must correctly manage multiple layers."""

    def test_empty_cache_creation(self):
        cache = QEffDynamicCache()
        assert cache is not None

    def test_update_adds_layer(self):
        cache = QEffDynamicCache()
        k, v = make_kv(seq=8)
        k_out, v_out = cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert k_out is not None
        assert v_out is not None

    def test_update_multiple_layers_creates_correct_count(self):
        cache = QEffDynamicCache()
        for i in range(4):
            k, v = make_kv(seq=8)
            cache.update(k, v, layer_idx=i, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert len(cache.layers) == 4

    def test_layers_are_qeff_dynamic_layer_instances(self):
        cache = QEffDynamicCache()
        k, v = make_kv(seq=8)
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert isinstance(cache.layers[0], QEffDynamicLayer)

    def test_write_only_then_read_only_returns_same_values(self):
        """write_only + read_only round-trip must return identical tensors."""
        cache = QEffDynamicCache()
        k, v = make_kv(batch=1, heads=2, seq=8, head_dim=16)
        pids = pos_ids(seq=8)

        cache.write_only(k, v, layer_idx=0, cache_kwargs={"position_ids": pids})
        k_out, v_out = cache.read_only(layer_idx=0, cache_kwargs={"position_ids": pids})

        assert torch.allclose(k_out, k), "read_only must return the same keys as written"
        assert torch.allclose(v_out, v), "read_only must return the same values as written"

    def test_prefill_then_decode_produces_finite_outputs(self):
        """Prefill + decode must produce finite key/value tensors."""
        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8

        k_prefill = torch.randn(batch, heads, ctx_len, head_dim)
        v_prefill = torch.randn(batch, heads, ctx_len, head_dim)
        cache.update(k_prefill, v_prefill, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        k_decode = torch.randn(batch, heads, 1, head_dim)
        v_decode = torch.randn(batch, heads, 1, head_dim)
        pos_decode = torch.tensor([[ctx_len - 1]])

        k_out, v_out = cache.update(k_decode, v_decode, layer_idx=0, cache_kwargs={"position_ids": pos_decode})

        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()
        assert k_out.shape[2] == ctx_len

    def test_decode_scatter_at_correct_position(self):
        """Decode must scatter the new token at the correct position in the cache."""
        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8

        k_prefill = torch.zeros(batch, heads, ctx_len, head_dim)
        v_prefill = torch.zeros(batch, heads, ctx_len, head_dim)
        cache.update(k_prefill, v_prefill, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        k_decode = torch.ones(batch, heads, 1, head_dim) * 42.0
        v_decode = torch.ones(batch, heads, 1, head_dim) * 42.0
        pos_decode = torch.tensor([[3]])

        k_out, v_out = cache.update(k_decode, v_decode, layer_idx=0, cache_kwargs={"position_ids": pos_decode})

        assert k_out[0, 0, 3, 0].item() == pytest.approx(42.0, abs=1e-5), (
            f"Expected 42.0 at position 3, got {k_out[0, 0, 3, 0].item()}"
        )

    def test_ddp_cache_data_populates_layers(self):
        """QEffDynamicCache with ddp_cache_data must populate layers."""
        k, v = make_kv(seq=8)
        ddp_data = [(k, v), (k.clone(), v.clone())]
        cache = QEffDynamicCache(ddp_cache_data=ddp_data)
        assert len(cache.layers) >= 2

    def test_batch_index_continuous_batching_mode(self):
        """Cache update with batch_index (continuous batching) must work."""
        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 2, 2, 8, 4

        k = torch.zeros(batch, heads, ctx_len, head_dim)
        v = torch.zeros(batch, heads, ctx_len, head_dim)
        pids = pos_ids(batch=batch, seq=ctx_len)
        batch_index = torch.arange(batch).view(-1, 1)

        k_out, v_out = cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pids, "batch_index": batch_index})
        assert k_out is not None
        assert v_out is not None
        assert torch.isfinite(k_out).all()


# ---------------------------------------------------------------------------
# Tests: QEffEncoderDecoderCache
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffEncoderDecoderCacheCorrectness:
    """QEffEncoderDecoderCache must correctly initialize from legacy cache."""

    def test_from_legacy_cache_none_creates_empty_cache(self):
        cache = QEffEncoderDecoderCache.from_legacy_cache(past_key_values=None)
        assert cache is not None
        assert isinstance(cache.self_attention_cache, QEffDynamicCache)
        assert isinstance(cache.cross_attention_cache, QEffDynamicCache)

    def test_from_legacy_cache_with_2tuple_populates_self_attention(self):
        k, v = make_kv(seq=8)
        past = [(k, v), (k.clone(), v.clone())]
        cache = QEffEncoderDecoderCache.from_legacy_cache(past_key_values=past)
        assert cache is not None

    def test_from_legacy_cache_with_4tuple_populates_cross_attention(self):
        k, v = make_kv(seq=8)
        past = [(k, v, k.clone(), v.clone())]
        cache = QEffEncoderDecoderCache.from_legacy_cache(past_key_values=past)
        assert cache is not None


# ---------------------------------------------------------------------------
# Tests: Cache numerical correctness (scatter/gather round-trip)
# ---------------------------------------------------------------------------


@pytest.mark.cache
@pytest.mark.accuracy
class TestCacheScatterGatherNumericalCorrectness:
    """
    Scatter/gather operations must be numerically correct.
    These tests verify that the cache correctly stores and retrieves values.
    """

    def test_prefill_values_preserved_in_cache(self):
        """After prefill, the cache must contain the exact prefill values."""
        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8

        k = torch.arange(batch * heads * ctx_len * head_dim, dtype=torch.float32).reshape(
            batch, heads, ctx_len, head_dim
        )
        v = k * 2.0
        pids = pos_ids(seq=ctx_len)

        cache.write_only(k, v, layer_idx=0, cache_kwargs={"position_ids": pids})
        k_out, v_out = cache.read_only(layer_idx=0, cache_kwargs={"position_ids": pids})

        assert torch.allclose(k_out, k), "Cache must preserve exact prefill key values"
        assert torch.allclose(v_out, v), "Cache must preserve exact prefill value values"

    def test_decode_overwrites_correct_position(self):
        """Decode step must overwrite exactly the specified position."""
        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 4

        k_prefill = torch.zeros(batch, heads, ctx_len, head_dim)
        v_prefill = torch.zeros(batch, heads, ctx_len, head_dim)
        cache.update(k_prefill, v_prefill, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        k_decode = torch.ones(batch, heads, 1, head_dim) * 99.0
        v_decode = torch.ones(batch, heads, 1, head_dim) * 99.0
        pos_decode = torch.tensor([[7]])

        k_out, v_out = cache.update(k_decode, v_decode, layer_idx=0, cache_kwargs={"position_ids": pos_decode})

        # Position 7 must have 99.0
        assert k_out[0, 0, 7, 0].item() == pytest.approx(99.0, abs=1e-5)
        assert v_out[0, 0, 7, 0].item() == pytest.approx(99.0, abs=1e-5)

        # Other positions must still be 0.0
        assert k_out[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert k_out[0, 0, 6, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert k_out[0, 0, 8, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_multiple_decode_steps_overwrite_correct_positions(self):
        """Multiple decode steps must each overwrite the correct position."""
        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 4

        k_prefill = torch.zeros(batch, heads, ctx_len, head_dim)
        v_prefill = torch.zeros(batch, heads, ctx_len, head_dim)
        cache.update(k_prefill, v_prefill, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        for pos, val in [(2, 10.0), (5, 20.0), (10, 30.0)]:
            k_d = torch.ones(batch, heads, 1, head_dim) * val
            v_d = torch.ones(batch, heads, 1, head_dim) * val
            k_out, v_out = cache.update(k_d, v_d, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[pos]])})

        # Final state: position 10 should have 30.0
        assert k_out[0, 0, 10, 0].item() == pytest.approx(30.0, abs=1e-5)

    def test_multi_layer_cache_independence(self):
        """Different layers must not interfere with each other."""
        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 8, 4

        for layer_idx in range(3):
            k = torch.ones(batch, heads, ctx_len, head_dim) * float(layer_idx + 1)
            v = torch.ones(batch, heads, ctx_len, head_dim) * float(layer_idx + 1)
            cache.write_only(k, v, layer_idx=layer_idx, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        for layer_idx in range(3):
            k_out, v_out = cache.read_only(layer_idx=layer_idx, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})
            expected_val = float(layer_idx + 1)
            assert k_out[0, 0, 0, 0].item() == pytest.approx(expected_val, abs=1e-5), (
                f"Layer {layer_idx} key value mismatch: expected {expected_val}, got {k_out[0, 0, 0, 0].item()}"
            )

    def test_decode_does_not_corrupt_prior_positions(self):
        """A decode write at position N must not corrupt positions 0..N-1.

        Note: QEfficient's CtxScatter zeros out positions > decode_position
        (they are not yet valid tokens). Only positions <= decode_position
        are guaranteed to be preserved.
        """
        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 1, 8, 4

        # Prefill with known sequential values
        k_prefill = (
            torch.arange(ctx_len, dtype=torch.float32)
            .reshape(1, 1, ctx_len, 1)
            .expand(batch, heads, ctx_len, head_dim)
            .clone()
        )
        v_prefill = k_prefill.clone()
        cache.update(k_prefill, v_prefill, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode: overwrite position 4 with 999.0
        k_decode = torch.ones(batch, heads, 1, head_dim) * 999.0
        v_decode = torch.ones(batch, heads, 1, head_dim) * 999.0
        k_out, v_out = cache.update(k_decode, v_decode, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[4]])})

        # Position 4 must be 999.0
        assert k_out[0, 0, 4, 0].item() == pytest.approx(999.0, abs=1e-5)
        # Positions before the decode position must be preserved
        assert k_out[0, 0, 3, 0].item() == pytest.approx(3.0, abs=1e-5)
        assert k_out[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert k_out[0, 0, 1, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert k_out[0, 0, 2, 0].item() == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: write_only_paged_attention must reject writes larger than one block
# ---------------------------------------------------------------------------


def make_paged_cache(num_phys_blocks=16, heads=1, block_size=4, head_dim=2):
    layer = QEffDynamicLayer()
    layer.keys = torch.zeros(num_phys_blocks, heads, block_size, head_dim)
    layer.values = torch.zeros(num_phys_blocks, heads, block_size, head_dim)
    layer.is_initialized = True
    return layer


@pytest.mark.cache
class TestPagedAttentionSingleBlockWriteLimit:
    """write_only_paged_attention assumes a write never spans more than one
    physical block; PL > kv_block_size must raise, not silently mis-scatter."""

    def test_write_within_block_size_succeeds(self):
        layer = make_paged_cache(num_phys_blocks=4, heads=1, block_size=4, head_dim=2)
        k = torch.ones(1, 1, 4, 2)
        v = torch.ones(1, 1, 4, 2)
        layer.write_only_paged_attention(
            k,
            v,
            cache_kwargs={
                "position_ids": torch.arange(4).unsqueeze(0),
                "block_table": torch.tensor([[0]]),
                "slot_id": torch.tensor([0]),
            },
        )
        assert torch.allclose(layer.keys[0], k[0])

    def test_write_exceeding_block_size_raises_not_implemented(self):
        layer = make_paged_cache(num_phys_blocks=4, heads=1, block_size=4, head_dim=2)
        k = torch.ones(1, 1, 5, 2)  # seq_len=5 > block_size=4
        v = torch.ones(1, 1, 5, 2)
        with pytest.raises(NotImplementedError):
            layer.write_only_paged_attention(
                k,
                v,
                cache_kwargs={
                    "position_ids": torch.arange(5).unsqueeze(0),
                    "block_table": torch.tensor([[0]]),
                    "slot_id": torch.tensor([0]),
                },
            )

    def test_slot_id_plus_seq_len_crossing_block_boundary_raises_not_implemented(self):
        """seq_len alone fits within block_size, but slot_id offset pushes the
        write past the block boundary."""
        layer = make_paged_cache(num_phys_blocks=4, heads=1, block_size=4, head_dim=2)
        k = torch.ones(1, 1, 2, 2)  # seq_len=2, fits within block_size=4 on its own
        v = torch.ones(1, 1, 2, 2)
        with pytest.raises(NotImplementedError):
            layer.write_only_paged_attention(
                k,
                v,
                cache_kwargs={
                    "position_ids": torch.tensor([[3, 4]]),
                    "block_table": torch.tensor([[0]]),
                    "slot_id": torch.tensor([3]),  # 3 + 2 = 5 > block_size=4
                },
            )

    def test_slot_id_plus_seq_len_exactly_at_boundary_succeeds(self):
        """slot_id + seq_len == block_size is the boundary case and must be allowed."""
        layer = make_paged_cache(num_phys_blocks=4, heads=1, block_size=4, head_dim=2)
        k = torch.ones(1, 1, 2, 2)
        v = torch.ones(1, 1, 2, 2)
        layer.write_only_paged_attention(
            k,
            v,
            cache_kwargs={
                "position_ids": torch.tensor([[2, 3]]),
                "block_table": torch.tensor([[0]]),
                "slot_id": torch.tensor([2]),  # 2 + 2 = 4 == block_size
            },
        )
        assert torch.allclose(layer.keys[0, 0, 2:4], k[0, 0])


# ---------------------------------------------------------------------------
# Tests: paged-attention block_table handling for caller-supplied block
# tables where two rows alias the same physical block (e.g. an external
# scheduler doing prefix-sharing across requests) and later diverge.
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestPagedAttentionBlockTableRegression:
    """These tests construct a block_table by hand (kv_block_size=4, block_table=[[3,7],[3,12]],
    position_ids=[[4],[4]], slot_id=[0,0]) to verify the per-row block_table
    lookup (block_table[rows, block_index]) handles that correctly."""

    BLOCK_SIZE = 4
    HEAD_DIM = 2
    # Row 0's logical blocks map to physical blocks 3 (shared prefix) and 7.
    # Row 1's logical blocks map to physical blocks 3 (shared prefix) and 12.
    BLOCK_TABLE = torch.tensor([[3, 7], [3, 12]])

    def _write_shared_prefix(self, layer):
        """Both rows prefill the same prompt and are handed a block_table
        that maps them both to physical block 3 (as a prefix-sharing caller
        would do) -> both writes target physical block 3."""
        shared_prefix = (
            torch.arange(self.BLOCK_SIZE * self.HEAD_DIM, dtype=torch.float32)
            .reshape(1, 1, self.BLOCK_SIZE, self.HEAD_DIM)
            .expand(2, 1, self.BLOCK_SIZE, self.HEAD_DIM)
        )
        layer.write_only_paged_attention(
            shared_prefix.clone(),
            shared_prefix.clone(),
            cache_kwargs={
                "position_ids": torch.arange(self.BLOCK_SIZE).unsqueeze(0).expand(2, -1),
                "block_table": self.BLOCK_TABLE,
                "slot_id": torch.tensor([0, 0]),
            },
        )
        return shared_prefix

    def test_shared_prefix_written_to_common_physical_block(self):
        layer = make_paged_cache(block_size=self.BLOCK_SIZE, head_dim=self.HEAD_DIM)
        shared_prefix = self._write_shared_prefix(layer)

        assert torch.allclose(layer.keys[3], shared_prefix[0]), "Shared prefix must land in physical block 3"
        assert torch.allclose(layer.values[3], shared_prefix[0]), "Shared prefix must land in physical block 3"

    def test_divergent_decode_writes_go_to_separate_physical_blocks(self):
        layer = make_paged_cache(block_size=self.BLOCK_SIZE, head_dim=self.HEAD_DIM)
        self._write_shared_prefix(layer)

        k_decode = torch.tensor([[[[100.0, 100.0]]], [[[200.0, 200.0]]]])  # (batch=2, heads=1, seq=1, dh=2)
        v_decode = torch.tensor([[[[101.0, 101.0]]], [[[201.0, 201.0]]]])
        layer.write_only_paged_attention(
            k_decode,
            v_decode,
            cache_kwargs={
                "position_ids": torch.tensor([[4], [4]]),
                "block_table": self.BLOCK_TABLE,
                "slot_id": torch.tensor([0, 0]),
            },
        )

        # Row 0's new token must land in physical block 7, at slot 0.
        assert torch.allclose(layer.keys[7, 0, 0], torch.tensor([100.0, 100.0]))
        assert torch.allclose(layer.values[7, 0, 0], torch.tensor([101.0, 101.0]))

        # Row 1's new token must land in physical block 12, at slot 0.
        assert torch.allclose(layer.keys[12, 0, 0], torch.tensor([200.0, 200.0]))
        assert torch.allclose(layer.values[12, 0, 0], torch.tensor([201.0, 201.0]))

        # The two rows' decode writes must not have bled into each other's block.
        assert not torch.allclose(layer.keys[7], layer.keys[12])

        # The shared prefix block must be untouched by either decode write.
        assert layer.keys[3, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)

        # Blocks that were never targeted must remain at their initial value.
        for untouched_block in (0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15):
            assert torch.allclose(layer.keys[untouched_block], torch.zeros(1, self.BLOCK_SIZE, self.HEAD_DIM)), (
                f"Physical block {untouched_block} should not have been written to"
            )

    def test_read_back_gathers_shared_prefix_and_divergent_tail_per_row(self):
        layer = make_paged_cache(block_size=self.BLOCK_SIZE, head_dim=self.HEAD_DIM)
        self._write_shared_prefix(layer)

        k_decode = torch.tensor([[[[100.0, 100.0]]], [[[200.0, 200.0]]]])
        v_decode = torch.tensor([[[[101.0, 101.0]]], [[[201.0, 201.0]]]])
        layer.write_only_paged_attention(
            k_decode,
            v_decode,
            cache_kwargs={
                "position_ids": torch.tensor([[4], [4]]),
                "block_table": self.BLOCK_TABLE,
                "slot_id": torch.tensor([0, 0]),
            },
        )

        # Gather logical block 0 (shared prefix, physical block 3) for both rows.
        k_block0, v_block0 = layer.read_only_paged_attention(
            block_index=self.BLOCK_TABLE[:, 0],
            updated=torch.tensor([[False], [False]]),
            cache_kwargs={"position_ids": torch.arange(self.BLOCK_SIZE).unsqueeze(0).expand(2, -1)},
        )
        assert torch.allclose(k_block0[0], k_block0[1]), "Both rows must read the identical shared prefix"

        # Gather logical block 1 (divergent tail, physical blocks 7/12) for both rows.
        k_block1, v_block1 = layer.read_only_paged_attention(
            block_index=self.BLOCK_TABLE[:, 1],
            updated=torch.tensor([[True], [True]]),
            cache_kwargs={"position_ids": torch.tensor([[4], [4]])},
        )
        assert k_block1[0, 0, 0, 0].item() == pytest.approx(100.0, abs=1e-5)
        assert k_block1[1, 0, 0, 0].item() == pytest.approx(200.0, abs=1e-5)

    def test_replacement_prefill_uses_request_specific_block_table_row(self):
        """A replacement prefill must key off each request's own block_table
        row, not row 0's for every row. Uses two rows whose logical block 0
        maps to two *different* physical blocks."""
        layer = make_paged_cache(block_size=self.BLOCK_SIZE, head_dim=self.HEAD_DIM)

        # Row 0's logical block 0 -> physical block 3; row 1's logical block 0
        # (a freshly assigned block, distinct from row 0's) -> physical block 9.
        block_table = torch.tensor([[3, 7], [9, 12]])
        k_prefill = torch.tensor([[400.0, 400.0]] * self.BLOCK_SIZE + [[500.0, 500.0]] * self.BLOCK_SIZE).reshape(
            2, 1, self.BLOCK_SIZE, self.HEAD_DIM
        )
        v_prefill = k_prefill + 1.0

        layer.write_only_paged_attention(
            k_prefill,
            v_prefill,
            cache_kwargs={
                "position_ids": torch.arange(self.BLOCK_SIZE).unsqueeze(0).expand(2, -1),
                "block_table": block_table,
                "slot_id": torch.tensor([0, 0]),
            },
        )

        # Row 0 must land in its own physical block (3), not row 1's (9).
        assert torch.allclose(layer.keys[3], torch.full((1, self.BLOCK_SIZE, self.HEAD_DIM), 400.0))
        # Row 1 must land in its own physical block (9), not row 0's (3).
        assert torch.allclose(layer.keys[9], torch.full((1, self.BLOCK_SIZE, self.HEAD_DIM), 500.0))
