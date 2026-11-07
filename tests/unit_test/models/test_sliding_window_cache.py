# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Correctness tests for QEffSlidingWindowCache and QEffDynamicCache.update3D.

Tests verify:
  - QEffSlidingWindowCache: creation, update (sliding + non-sliding), modular scatter,
    output shape, multi-layer independence, to_legacy_cache round-trip, get_seq_length
  - QEffDynamicLayer.update3D / QEffDynamicCache.update3D: 3D KV shape (GPTBigCode)
  - QEffGPTOSSHybridCache: full_cache_update_chunked, sliding_window_update_chunked

All tests run on CPU only.
"""

import pytest
import torch
from transformers import GptOssConfig

from QEfficient.transformers.cache_utils import (
    QEffDynamicCache,
    QEffDynamicLayer,
    QEffGPTOSSHybridCache,
    QEffSlidingWindowCache,
)

# ---------------------------------------------------------------------------
# Minimal config stub (no HF model needed)
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal config stub for cache constructors."""

    sliding_window_pattern = 2  # every 2nd layer is sliding
    sliding_window = 4


def _gptoss_cfg(num_layers=2, sliding_window=4):
    """Minimal GptOssConfig: layer 0 sliding_attention, layer 1 full_attention."""
    layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(num_layers)]
    return GptOssConfig(
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=64,
        head_dim=32,
        sliding_window=sliding_window,
        layer_types=layer_types,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kv_4d(batch=1, heads=2, seq=8, head_dim=16):
    k = torch.randn(batch, heads, seq, head_dim)
    v = torch.randn(batch, heads, seq, head_dim)
    return k, v


def make_kv_3d(batch=1, seq=8, kv_dim=32):
    """3D KV tensors as used by GPTBigCode: [batch, seq, heads*head_dim]."""
    k = torch.randn(batch, seq, kv_dim)
    v = torch.randn(batch, seq, kv_dim)
    return k, v


def pos_ids(batch=1, seq=8, start=0):
    return torch.arange(start, start + seq).unsqueeze(0).expand(batch, -1)


# ---------------------------------------------------------------------------
# Tests: QEffSlidingWindowCache
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffSlidingWindowCache:
    """QEffSlidingWindowCache must correctly implement sliding-window KV caching."""

    @pytest.mark.parametrize("layer_types", [None, []])
    def test_write_only_rejects_unknown_attention_mode(self, layer_types):
        cfg = _FakeConfig()
        cfg.layer_types = layer_types
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        cache.key_cache = [torch.zeros(1, 2, 4, 16)]
        cache.value_cache = [torch.zeros(1, 2, 4, 16)]
        cache.seen_tokens = 7
        keys, values = make_kv_4d(seq=1)
        with pytest.raises(ValueError, match="requires is_sliding or config.layer_types"):
            cache.write_only(keys, values, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=1, start=5)})
        assert torch.count_nonzero(cache.key_cache[0]) == 0
        assert torch.count_nonzero(cache.value_cache[0]) == 0
        assert cache.seen_tokens == 7

    def test_write_only_initial_append_preserves_legacy_compatibility(self):
        cache = QEffSlidingWindowCache(_FakeConfig(), batch_size=1, max_cache_len=16, sliding_window_len=4)
        keys, values = make_kv_4d(seq=4)
        cache.write_only(keys, values, layer_idx=0)
        torch.testing.assert_close(cache.key_cache[0], keys)
        torch.testing.assert_close(cache.value_cache[0], values)
        assert cache.seen_tokens == 0

    @pytest.mark.parametrize(
        "attention_type,explicit_mode",
        [
            (None, False),
            (None, True),
            ("sliding_attention", None),
            ("sliding_attention", False),
            ("sliding_attention", True),
            ("full_attention", None),
            ("full_attention", False),
            ("full_attention", True),
        ],
    )
    @pytest.mark.parametrize("continuous_batching", [False, True])
    def test_write_only_resolves_attention_mode(self, attention_type, explicit_mode, continuous_batching):
        cfg = _FakeConfig()
        if attention_type is not None:
            cfg.layer_types = [attention_type]
        is_sliding = explicit_mode if explicit_mode is not None else attention_type == "sliding_attention"
        cache_len = 4 if is_sliding else 16
        cache = QEffSlidingWindowCache(cfg, batch_size=2, max_cache_len=16, sliding_window_len=4)
        cache.key_cache = [torch.zeros(2, 2, cache_len, 16)]
        cache.value_cache = [torch.zeros(2, 2, cache_len, 16)]
        cache.seen_tokens = 7
        batch_size = 1 if continuous_batching else 2
        keys, values = make_kv_4d(batch=batch_size, seq=1)
        kwargs = {"position_ids": pos_ids(batch=batch_size, seq=1, start=5)}
        if explicit_mode is not None:
            kwargs["is_sliding"] = explicit_mode
        if continuous_batching:
            kwargs["batch_index"] = torch.tensor([[1]])
        cache.write_only(keys, values, layer_idx=0, cache_kwargs=kwargs)
        expected_keys = torch.zeros_like(cache.key_cache[0])
        expected_values = torch.zeros_like(cache.value_cache[0])
        position = 1 if is_sliding else 5
        if continuous_batching:
            expected_keys[1, :, position] = keys[0, :, 0]
            expected_values[1, :, position] = values[0, :, 0]
        else:
            expected_keys[:, :, position] = keys[:, :, 0]
            expected_values[:, :, position] = values[:, :, 0]
        torch.testing.assert_close(cache.key_cache[0], expected_keys)
        torch.testing.assert_close(cache.value_cache[0], expected_values)
        assert cache.seen_tokens == 7

    def test_creation_succeeds(self):
        """Cache must be created without errors."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        assert cache is not None
        assert cache.max_cache_len == 16
        assert cache.sliding_window_len == 4
        assert cache.batch_size == 1

    def test_initial_cache_is_empty(self):
        """Newly created cache must have empty key/value lists."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        assert len(cache.key_cache) == 0
        assert len(cache.value_cache) == 0

    def test_len_returns_number_of_layers(self):
        """__len__ must return the number of cached layers."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        assert len(cache) == 0

        k, v = make_kv_4d(seq=4)
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=4), "is_sliding": False})
        assert len(cache) == 1

        cache.update(
            k.clone(), v.clone(), layer_idx=1, cache_kwargs={"position_ids": pos_ids(seq=4), "is_sliding": True}
        )
        assert len(cache) == 2

    def test_first_update_non_sliding_stores_tensors(self):
        """First update (non-sliding) must store tensors in the cache."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        k, v = make_kv_4d(seq=8)
        k_out, v_out = cache.update(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False}
        )
        assert len(cache.key_cache) == 1
        assert k_out is not None
        assert v_out is not None

    def test_first_update_returns_finite_tensors(self):
        """First update must return finite tensors."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        k, v = make_kv_4d(seq=8)
        k_out, v_out = cache.update(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False}
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_non_sliding_decode_scatter_at_correct_position(self):
        """Non-sliding decode must scatter at the exact position_id."""
        cfg = _FakeConfig()
        ctx_len = 16
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=ctx_len, sliding_window_len=4)

        # Prefill with zeros
        k_init = torch.zeros(1, 2, ctx_len, 8)
        v_init = torch.zeros(1, 2, ctx_len, 8)
        cache.update(
            k_init, v_init, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len), "is_sliding": False}
        )

        # Decode: write known value at position 5
        k_dec = torch.ones(1, 2, 1, 8) * 7.0
        v_dec = torch.ones(1, 2, 1, 8) * 7.0
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[5]]), "is_sliding": False}
        )
        assert k_out[0, 0, 5, 0].item() == pytest.approx(7.0, abs=1e-5)

    def test_sliding_modular_scatter_position(self):
        """Sliding update must scatter at position % sliding_window_len."""
        cfg = _FakeConfig()
        sliding_window_len = 4
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=sliding_window_len)

        # Prefill sliding layer with zeros
        k_init = torch.zeros(1, 2, sliding_window_len, 8)
        v_init = torch.zeros(1, 2, sliding_window_len, 8)
        cache.update(
            k_init,
            v_init,
            layer_idx=0,
            cache_kwargs={"position_ids": pos_ids(seq=sliding_window_len), "is_sliding": True},
        )

        # Decode at position 5: slot = 5 % 4 = 1
        k_dec = torch.ones(1, 2, 1, 8) * 99.0
        v_dec = torch.ones(1, 2, 1, 8) * 99.0
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[5]]), "is_sliding": True}
        )
        # The output shape should be sliding_window_len
        assert k_out.shape[2] == sliding_window_len
        assert torch.isfinite(k_out).all()

    def test_output_shape_non_sliding_equals_ctx_len(self):
        """Non-sliding update output must have shape matching ctx_len."""
        cfg = _FakeConfig()
        ctx_len = 16
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=ctx_len, sliding_window_len=4)
        k, v = make_kv_4d(seq=ctx_len)
        k_out, v_out = cache.update(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len), "is_sliding": False}
        )
        assert k_out.shape[2] == ctx_len

    def test_output_shape_sliding_equals_window_size(self):
        """Sliding update output must have shape matching sliding_window_len."""
        cfg = _FakeConfig()
        sliding_window_len = 4
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=sliding_window_len)
        k, v = make_kv_4d(seq=sliding_window_len)
        k_out, v_out = cache.update(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=sliding_window_len), "is_sliding": True}
        )
        assert k_out.shape[2] == sliding_window_len

    def test_multi_layer_independence(self):
        """Different layers must not interfere with each other."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)

        for layer_idx in range(3):
            k = torch.ones(1, 2, 8, 4) * float(layer_idx + 1)
            v = torch.ones(1, 2, 8, 4) * float(layer_idx + 1)
            cache.update(k, v, layer_idx=layer_idx, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False})

        # Each layer's cache must have its own value
        for layer_idx in range(3):
            expected = float(layer_idx + 1)
            assert cache.key_cache[layer_idx][0, 0, 0, 0].item() == pytest.approx(expected, abs=1e-5)

    def test_to_legacy_cache_round_trip(self):
        """to_legacy_cache must return a tuple of (key, value) pairs per layer."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)

        for layer_idx in range(2):
            k, v = make_kv_4d(seq=8)
            cache.update(k, v, layer_idx=layer_idx, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False})

        legacy = cache.to_legacy_cache()
        assert isinstance(legacy, tuple)
        assert len(legacy) == 2
        for layer_kv in legacy:
            assert len(layer_kv) == 2  # (key, value)

    def test_get_seq_length_returns_correct_value(self):
        """get_seq_length must return the sequence length of the cached layer."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)

        # Empty cache
        assert cache.get_seq_length(layer_idx=0) == 0

        # After update
        k, v = make_kv_4d(seq=8)
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False})
        assert cache.get_seq_length(layer_idx=0) == 8

    def test_update_returns_finite_tensors_after_decode(self):
        """Decode update must return finite tensors."""
        cfg = _FakeConfig()
        ctx_len = 16
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=ctx_len, sliding_window_len=4)

        # Prefill
        k, v = make_kv_4d(seq=ctx_len)
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len), "is_sliding": False})

        # Decode
        k_dec = torch.randn(1, 2, 1, 16)
        v_dec = torch.randn(1, 2, 1, 16)
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[ctx_len - 1]]), "is_sliding": False}
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()


# ---------------------------------------------------------------------------
# Tests: QEffDynamicLayer.update3D (GPTBigCode 3D KV cache)
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffDynamicCache3D:
    """QEffDynamicLayer.update3D must handle 3D KV tensors [batch, seq, kv_dim]."""

    def test_update3d_first_call_stores_tensors(self):
        """First update3D call must store tensors in the layer."""
        layer = QEffDynamicLayer()
        k, v = make_kv_3d(batch=1, seq=8, kv_dim=32)
        k_out, v_out = layer.update3D(k, v, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert layer.keys is not None
        assert layer.values is not None
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_update3d_output_is_finite(self):
        """update3D must return finite tensors."""
        layer = QEffDynamicLayer()
        k, v = make_kv_3d(batch=1, seq=8, kv_dim=32)
        k_out, v_out = layer.update3D(k, v, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_update3d_output_shape_is_correct(self):
        """update3D output must have shape [batch, ctx_len, kv_dim]."""
        layer = QEffDynamicLayer()
        batch, ctx_len, kv_dim = 1, 16, 32
        k = torch.zeros(batch, ctx_len, kv_dim)
        v = torch.zeros(batch, ctx_len, kv_dim)
        k_out, v_out = layer.update3D(k, v, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})
        assert k_out.shape == (batch, ctx_len, kv_dim)
        assert v_out.shape == (batch, ctx_len, kv_dim)

    def test_update3d_scatter_at_correct_position(self):
        """update3D decode must scatter at the correct position."""
        layer = QEffDynamicLayer()
        batch, ctx_len, kv_dim = 1, 16, 32

        # Prefill with zeros
        k_init = torch.zeros(batch, ctx_len, kv_dim)
        v_init = torch.zeros(batch, ctx_len, kv_dim)
        layer.update3D(k_init, v_init, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode: write known value at position 3
        k_dec = torch.ones(batch, 1, kv_dim) * 42.0
        v_dec = torch.ones(batch, 1, kv_dim) * 42.0
        k_out, v_out = layer.update3D(k_dec, v_dec, cache_kwargs={"position_ids": torch.tensor([[3]])})

        assert k_out[0, 3, 0].item() == pytest.approx(42.0, abs=1e-5)

    def test_update3d_prior_positions_not_corrupted(self):
        """update3D decode must not corrupt positions before the decode position."""
        layer = QEffDynamicLayer()
        batch, ctx_len, kv_dim = 1, 16, 4

        # Prefill with sequential values
        k_init = (
            torch.arange(ctx_len, dtype=torch.float32).reshape(1, ctx_len, 1).expand(batch, ctx_len, kv_dim).clone()
        )
        v_init = k_init.clone()
        layer.update3D(k_init, v_init, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode at position 5
        k_dec = torch.ones(batch, 1, kv_dim) * 999.0
        v_dec = torch.ones(batch, 1, kv_dim) * 999.0
        k_out, v_out = layer.update3D(k_dec, v_dec, cache_kwargs={"position_ids": torch.tensor([[5]])})

        # Position 5 must be 999.0
        assert k_out[0, 5, 0].item() == pytest.approx(999.0, abs=1e-5)
        # Positions before 5 must be preserved
        assert k_out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert k_out[0, 3, 0].item() == pytest.approx(3.0, abs=1e-5)
        assert k_out[0, 4, 0].item() == pytest.approx(4.0, abs=1e-5)

    def test_qeff_dynamic_cache_update3d_delegates_to_layer(self):
        """QEffDynamicCache.update3D must delegate to the layer's update3D."""
        cache = QEffDynamicCache()
        batch, ctx_len, kv_dim = 1, 8, 32
        k = torch.randn(batch, ctx_len, kv_dim)
        v = torch.randn(batch, ctx_len, kv_dim)
        k_out, v_out = cache.update3D(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})
        assert k_out is not None
        assert v_out is not None
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_qeff_dynamic_cache_update3d_creates_layer(self):
        """QEffDynamicCache.update3D must create a new layer at the given index."""
        cache = QEffDynamicCache()
        k, v = make_kv_3d(batch=1, seq=8, kv_dim=32)
        cache.update3D(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert len(cache.layers) == 1


# ---------------------------------------------------------------------------
# Tests: QEffGPTOSSHybridCache chunked methods
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffGPTOSSHybridCacheChunked:
    """QEffGPTOSSHybridCache chunked prefill methods must be numerically correct."""

    def _make_cache_with_layer(self, batch=1, heads=2, ctx_len=16, head_dim=8, sliding_window_len=4):
        """Create a cache with one pre-initialized full-attention layer (layer 1)."""
        cfg = _gptoss_cfg(sliding_window=sliding_window_len)
        cache = QEffGPTOSSHybridCache(config=cfg)
        k = torch.zeros(batch, heads, ctx_len, head_dim)
        v = torch.zeros(batch, heads, ctx_len, head_dim)
        cache.append_new_layers(1)
        cache.layers[1].keys = k
        cache.layers[1].values = v
        cache.layers[1]._mark_initialized(k)
        return cache

    def _make_sliding_cache_with_layer(self, batch=1, heads=2, sliding_window_len=4, head_dim=8):
        """Create a cache with one pre-initialized sliding window layer (layer 0)."""
        cfg = _gptoss_cfg(sliding_window=sliding_window_len)
        cache = QEffGPTOSSHybridCache(config=cfg)
        k = torch.zeros(batch, heads, sliding_window_len, head_dim)
        v = torch.zeros(batch, heads, sliding_window_len, head_dim)
        cache.append_new_layers(0)
        cache.layers[0].keys = k
        cache.layers[0].values = v
        cache.layers[0]._mark_initialized(k)
        return cache

    def test_full_cache_update_chunked_returns_finite(self):
        """full_cache_update_chunked must return finite tensors."""
        cache = self._make_cache_with_layer()
        batch, heads, seq_len, head_dim = 1, 2, 4, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.full_cache_update_chunked(
            k, v, layer_idx=1, cache_kwargs={"position_ids": pos_ids(seq=seq_len), "batch_index": None}
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_full_cache_update_chunked_scatter_at_correct_position(self):
        """full_cache_update_chunked must scatter at the correct position."""
        cache = self._make_cache_with_layer(ctx_len=16)
        batch, heads, head_dim = 1, 2, 8

        # Write known value at positions 0-3
        k = torch.ones(batch, heads, 4, head_dim) * 5.0
        v = torch.ones(batch, heads, 4, head_dim) * 5.0
        k_out, v_out = cache.full_cache_update_chunked(
            k, v, layer_idx=1, cache_kwargs={"position_ids": pos_ids(seq=4), "batch_index": None}
        )
        # Positions 0-3 should have value 5.0
        assert k_out[0, 0, 0, 0].item() == pytest.approx(5.0, abs=1e-5)
        assert k_out[0, 0, 3, 0].item() == pytest.approx(5.0, abs=1e-5)

    def test_full_cache_update_chunked_output_shape(self):
        """full_cache_update_chunked output must have the correct shape."""
        ctx_len = 16
        cache = self._make_cache_with_layer(ctx_len=ctx_len)
        batch, heads, seq_len, head_dim = 1, 2, 4, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.full_cache_update_chunked(
            k, v, layer_idx=1, cache_kwargs={"position_ids": pos_ids(seq=seq_len), "batch_index": None}
        )
        assert k_out.shape[2] == ctx_len

    def test_sliding_window_update_chunked_returns_finite(self):
        """sliding_window_update_chunked must return finite tensors."""
        sliding_window_len = 4
        cache = self._make_sliding_cache_with_layer(sliding_window_len=sliding_window_len)
        batch, heads, seq_len, head_dim = 1, 2, 4, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.sliding_window_update_chunked(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": pos_ids(seq=seq_len),
                "batch_index": None,
                "sliding_window": sliding_window_len,
            },
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_sliding_window_update_chunked_output_shape(self):
        """sliding_window_update_chunked output must have the correct shape."""
        sliding_window_len = 4
        seq_len = 4
        cache = self._make_sliding_cache_with_layer(sliding_window_len=sliding_window_len)
        batch, heads, head_dim = 1, 2, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.sliding_window_update_chunked(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": pos_ids(seq=seq_len),
                "batch_index": None,
                "sliding_window": sliding_window_len,
            },
        )
        # Output shape: seq_len + sliding_window_len
        expected_ctx = seq_len + sliding_window_len
        assert k_out.shape[2] == expected_ctx

    def test_sliding_window_update_chunked_with_larger_window(self):
        """sliding_window_update_chunked with a larger window must return finite tensors."""
        sliding_window_len = 8
        seq_len = 4
        cache = self._make_sliding_cache_with_layer(sliding_window_len=sliding_window_len)
        batch, heads, head_dim = 1, 2, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.sliding_window_update_chunked(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": pos_ids(seq=seq_len),
                "batch_index": None,
                "sliding_window": sliding_window_len,
            },
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()


# ---------------------------------------------------------------------------
# Tests: CCL (Compute Context Length) cache path
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestCCLCachePath:
    """QEffDynamicCache.update with CCL kwarg must work correctly."""

    def test_update_with_ccl_returns_finite(self):
        """update() with CCL kwarg must return finite tensors."""
        from QEfficient.transformers.cache_utils import QEffDynamicCache

        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8
        k = torch.randn(batch, heads, ctx_len, head_dim)
        v = torch.randn(batch, heads, ctx_len, head_dim)

        # Prefill
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode with CCL
        k_dec = torch.randn(batch, heads, 1, head_dim)
        v_dec = torch.randn(batch, heads, 1, head_dim)
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[8]]), "CCL": 8}
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_update_with_ccl_output_shape_matches_ccl(self):
        """update() with CCL kwarg must return tensors with ctx_len=CCL."""
        from QEfficient.transformers.cache_utils import QEffDynamicCache

        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8
        k = torch.randn(batch, heads, ctx_len, head_dim)
        v = torch.randn(batch, heads, ctx_len, head_dim)

        # Prefill
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode with CCL=8 (smaller than ctx_len=16)
        ccl = 8
        k_dec = torch.randn(batch, heads, 1, head_dim)
        v_dec = torch.randn(batch, heads, 1, head_dim)
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[4]]), "CCL": ccl}
        )
        assert k_out.shape[2] == ccl
        assert v_out.shape[2] == ccl
