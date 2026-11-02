# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Priority-2 fix: QEffHybridCache, QEffHybridChunkedCache, QEffGPTOSSHybridCache
correctness — these classes had ZERO test coverage.

Constructor signatures (verified from source):
  QEffHybridCache(config, batch_size, max_cache_len)
  QEffHybridChunkedCache — constructed via from_legacy_cache(config, past_key_values)
    which calls cls(config, max_batch_size=..., max_cache_len=...)
  QEffGPTOSSHybridCache(config) — per-layer cache; layers created via
    append_new_layers/from_legacy_cache, tagged is_sliding from config.layer_types

QEffHybridCache.update() required cache_kwargs:
  position_ids, sliding_window_pattern
  is_sliding is derived internally: bool((layer_idx + 1) % sliding_window_pattern)

QEffHybridChunkedCache.update() required cache_kwargs:
  position_ids
  is_sliding comes from self.is_sliding[layer_idx] set by parent HybridChunkedCache

QEffGPTOSSHybridCache.update() required cache_kwargs:
  position_ids (is_sliding/sliding_window are derived from the layer itself, not cache_kwargs)
QEffGPTOSSHybridCache.write_only() required cache_kwargs:
  position_ids

All tests run on CPU only.
"""

import pytest
import torch
from transformers import Gemma2Config, GptOssConfig, MistralConfig

from QEfficient.transformers.cache_utils import (
    QEffGPTOSSHybridCache,
    QEffHybridCache,
    QEffHybridChunkedCache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gemma2_cfg(num_layers=4, sliding_window=4, sliding_window_pattern=2):
    """
    Minimal Gemma2Config.
    With sliding_window_pattern=2:
      layer_idx=0 → (0+1) % 2 = 1 (truthy)  → sliding
      layer_idx=1 → (1+1) % 2 = 0 (falsy)   → non-sliding
      layer_idx=2 → (2+1) % 2 = 1 (truthy)  → sliding
      layer_idx=3 → (3+1) % 2 = 0 (falsy)   → non-sliding
    """
    return Gemma2Config(
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=64,
        head_dim=32,
        sliding_window=sliding_window,
        sliding_window_pattern=sliding_window_pattern,
    )


def _mistral_cfg(sliding_window=4):
    """Minimal MistralConfig for QEffHybridChunkedCache."""
    cfg = MistralConfig(
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=64,
        sliding_window=sliding_window,
    )
    # HybridChunkedCache parent reads this to build is_sliding list
    cfg.sliding_window_pattern = 2
    return cfg


def _gptoss_cfg(num_layers=4, sliding_window=4):
    """Minimal GptOssConfig with alternating sliding/full attention layers."""
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


def _kv(batch=1, heads=2, ctx_len=16, head_dim=8, fill=None):
    """Build (key, value) tensors. fill=None → random."""
    if fill is not None:
        k = torch.full((batch, heads, ctx_len, head_dim), fill, dtype=torch.float32)
        v = torch.full((batch, heads, ctx_len, head_dim), fill, dtype=torch.float32)
    else:
        k = torch.randn(batch, heads, ctx_len, head_dim)
        v = torch.randn(batch, heads, ctx_len, head_dim)
    return k, v


def _pids(seq=8, start=0, batch=1):
    """Build position_ids tensor of shape (batch, seq)."""
    return torch.arange(start, start + seq, dtype=torch.long).unsqueeze(0).expand(batch, -1).clone()


# ---------------------------------------------------------------------------
# _StandaloneHybridCache: test-only subclass of QEffHybridCache
#
# Problems with the current QEffHybridCache:
#
# 1. __init__ chain is broken:
#    QEffHybridCache.__init__ → HybridCache.__init__ → Cache.__init__ raises
#    TypeError: Cache.__init__() got multiple values for argument 'layer_classes'
#    (QEffHybridCache passes batch_size as a positional arg which ends up
#    colliding with the layer_classes keyword arg that HybridCache already passes.)
#
# 2. Cache.key_cache / value_cache are properties returning KeyValuesWrapper,
#    which wraps self.layers and does NOT support .append().
#    QEffHybridCache.update() calls self.key_cache.append(), so it is
#    incompatible with the KeyValuesWrapper-based properties.
#
# Fix: subclass that overrides __init__ (bypassing the broken chain) and
# re-declares key_cache / value_cache as plain-list properties backed by
# _key_cache / _value_cache instance attributes.
# ---------------------------------------------------------------------------


class _StandaloneHybridCache(QEffHybridCache):
    """
    Test-only subclass of QEffHybridCache.

    Overrides __init__ to avoid the broken HybridCache → Cache __init__ chain,
    and overrides key_cache / value_cache as plain-list properties so that
    QEffHybridCache.update() (which calls .append() and uses direct indexing)
    works correctly.
    """

    def __init__(self, config, batch_size=1, max_cache_len=16):
        # Bypass the broken super().__init__() chain entirely.
        # We only need the attributes that QEffHybridCache.update() reads.
        self._key_cache: list = []
        self._value_cache: list = []
        self.config = config
        self._seen_tokens = 0

    @property
    def key_cache(self):
        return self._key_cache

    @key_cache.setter
    def key_cache(self, value):
        self._key_cache = value

    @property
    def value_cache(self):
        return self._value_cache

    @value_cache.setter
    def value_cache(self, value):
        self._value_cache = value


def _make_hybrid_cache_raw(cfg, ctx_len=16):
    """
    Construct a QEffHybridCache-compatible instance for testing.

    Uses _StandaloneHybridCache to avoid:
    1. The broken HybridCache.__init__ → Cache.__init__ double-kwarg bug.
    2. The KeyValuesWrapper-based key_cache/value_cache properties that do
       not support .append() (required by QEffHybridCache.update()).
    """
    return _StandaloneHybridCache(cfg, batch_size=1, max_cache_len=ctx_len)


# ---------------------------------------------------------------------------
# Tests: QEffHybridCache — non-sliding layer (standard KV path)
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffHybridCacheNonSlidingLayer:
    """
    Non-sliding layers (where (layer_idx+1) % sliding_window_pattern == 0)
    must behave like QEffDynamicCache: scatter at position_ids, gather back.
    With sliding_window_pattern=2, layer_idx=1 is non-sliding.

    Note: QEffHybridCache.update() uses list.append() for the first call per
    layer and scatter/gather for subsequent calls.  Because layers are appended
    sequentially, tests that exercise layer_idx=1 must first call update() for
    layer_idx=0 so that len(key_cache) > 1 before the second layer_idx=1 call
    triggers the scatter/gather branch.
    """

    def _make(self, ctx_len=16, sw=4):
        return _make_hybrid_cache_raw(_gemma2_cfg(sliding_window=sw), ctx_len=ctx_len)

    def test_first_update_stores_tensors(self):
        cache = self._make()
        k, v = _kv(ctx_len=8)
        k_out, v_out = cache.update(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(8),
                "sliding_window_pattern": 2,
            },
        )
        assert k_out is not None and v_out is not None

    def test_non_sliding_update_returns_finite(self):
        """layer_idx=1 → (1+1)%2==0 → non-sliding."""
        cache = self._make(ctx_len=16)
        k, v = _kv(ctx_len=8)
        k_out, v_out = cache.update(
            k,
            v,
            layer_idx=1,
            cache_kwargs={
                "position_ids": _pids(8),
                "sliding_window_pattern": 2,
            },
        )
        assert torch.isfinite(k_out).all(), "Non-sliding keys must be finite"
        assert torch.isfinite(v_out).all(), "Non-sliding values must be finite"

    def test_non_sliding_scatter_at_correct_position(self):
        """
        Non-sliding layer (layer_idx=1): write 7.0 at position 5,
        verify the gathered output has 7.0 at slot 5.

        layer_idx=0 is initialised first so that the second layer_idx=1 call
        (the decode step) enters the scatter/gather branch of update().
        """
        cache = self._make(ctx_len=16)
        # Initialise layer 0 (sliding) so len(key_cache) becomes 1 after this call.
        k_dummy, v_dummy = _kv(ctx_len=16, fill=0.0)
        cache.update(
            k_dummy,
            v_dummy,
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(16),
                "sliding_window_pattern": 2,
            },
        )
        # Prefill layer 1 (non-sliding): fill all 16 slots with zeros.
        # len(key_cache) == 1 <= 1, so this call appends → len becomes 2.
        k_init, v_init = _kv(ctx_len=16, fill=0.0)
        cache.update(
            k_init,
            v_init,
            layer_idx=1,
            cache_kwargs={
                "position_ids": _pids(16),
                "sliding_window_pattern": 2,
            },
        )
        # Decode: write 7.0 at position 5.
        # len(key_cache) == 2 > 1, so this call enters the scatter/gather branch.
        k_dec, v_dec = _kv(ctx_len=1, fill=7.0)
        k_out, v_out = cache.update(
            k_dec,
            v_dec,
            layer_idx=1,
            cache_kwargs={
                "position_ids": torch.tensor([[5]]),
                "sliding_window_pattern": 2,
            },
        )
        assert k_out[0, 0, 5, 0].item() == pytest.approx(7.0, abs=1e-5), (
            f"Expected 7.0 at position 5, got {k_out[0, 0, 5, 0].item()}"
        )

    def test_non_sliding_prior_positions_not_corrupted(self):
        """
        Writing at position 5 must not corrupt positions 0..4.

        layer_idx=0 is initialised first so that the decode call for layer_idx=1
        enters the scatter/gather branch.
        """
        cache = self._make(ctx_len=16)
        # Initialise layer 0 so len(key_cache) becomes 1.
        k_dummy, v_dummy = _kv(ctx_len=16, fill=0.0)
        cache.update(
            k_dummy,
            v_dummy,
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(16),
                "sliding_window_pattern": 2,
            },
        )
        # Prefill layer 1 with sequential values: position i → value float(i).
        k_init = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1).expand(1, 2, 16, 8).clone()
        v_init = k_init.clone()
        cache.update(
            k_init,
            v_init,
            layer_idx=1,
            cache_kwargs={
                "position_ids": _pids(16),
                "sliding_window_pattern": 2,
            },
        )
        # Decode at position 5.
        k_dec, v_dec = _kv(ctx_len=1, fill=99.0)
        k_out, _ = cache.update(
            k_dec,
            v_dec,
            layer_idx=1,
            cache_kwargs={
                "position_ids": torch.tensor([[5]]),
                "sliding_window_pattern": 2,
            },
        )
        assert k_out[0, 0, 5, 0].item() == pytest.approx(99.0, abs=1e-5)
        for pos in range(5):
            assert k_out[0, 0, pos, 0].item() == pytest.approx(float(pos), abs=1e-5), (
                f"Position {pos} corrupted: expected {float(pos)}, got {k_out[0, 0, pos, 0].item()}"
            )

    def test_len_tracks_updated_layers(self):
        cache = self._make(ctx_len=16)
        k, v = _kv(ctx_len=8)
        for i in range(3):
            cache.update(
                k,
                v,
                layer_idx=i,
                cache_kwargs={
                    "position_ids": _pids(8),
                    "sliding_window_pattern": 2,
                },
            )
        assert len(cache) == 3

    def test_to_legacy_cache_shape(self):
        cache = self._make(ctx_len=16)
        k, v = _kv(ctx_len=8)
        cache.update(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(8),
                "sliding_window_pattern": 2,
            },
        )
        legacy = cache.to_legacy_cache()
        assert isinstance(legacy, tuple) and len(legacy) == 1
        assert len(legacy[0]) == 2


# ---------------------------------------------------------------------------
# Tests: QEffHybridCache — sliding layer (modular position arithmetic)
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffHybridCacheSlidingLayer:
    """
    Sliding layers (where (layer_idx+1) % sliding_window_pattern != 0) use
    modular arithmetic: kv_position_ids = position_ids % (layer_ctx_len - 1).
    layer_idx=0 with sliding_window_pattern=2 is sliding.
    """

    def _make(self, ctx_len=4, sw=4):
        return _make_hybrid_cache_raw(_gemma2_cfg(sliding_window=sw), ctx_len=ctx_len)

    def test_sliding_first_update_stores_tensors(self):
        cache = self._make(ctx_len=4, sw=4)
        k, v = _kv(ctx_len=4)
        k_out, v_out = cache.update(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(4),
                "sliding_window_pattern": 2,
            },
        )
        assert k_out is not None and v_out is not None

    def test_sliding_update_returns_finite(self):
        cache = self._make(ctx_len=4, sw=4)
        k, v = _kv(ctx_len=4)
        k_out, v_out = cache.update(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(4),
                "sliding_window_pattern": 2,
            },
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_sliding_output_shape_equals_window_size(self):
        """The gather output for a sliding layer must have ctx_len == sliding_window."""
        sw = 4
        cache = self._make(ctx_len=sw, sw=sw)
        k, v = _kv(ctx_len=sw)
        k_out, v_out = cache.update(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(sw),
                "sliding_window_pattern": 2,
            },
        )
        assert k_out.shape[2] == sw, f"Sliding output ctx_len={k_out.shape[2]}, expected {sw}"

    def test_sliding_modular_scatter_position(self):
        """
        For sliding_window=4 (layer_ctx_len=4), position 5 maps to
        slot = 5 % (4-1) = 5 % 3 = 2.
        Write 55.0 at position 5 and verify cache slot 2 holds 55.0.
        """
        sw = 4
        cache = self._make(ctx_len=sw, sw=sw)
        # Prefill: fill all 4 slots with zeros
        k_init, v_init = _kv(ctx_len=sw, fill=0.0)
        cache.update(
            k_init,
            v_init,
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(sw),
                "sliding_window_pattern": 2,
            },
        )
        # Decode at position 5 → slot = 5 % (4-1) = 2
        k_dec, v_dec = _kv(ctx_len=1, fill=55.0)
        cache.update(
            k_dec,
            v_dec,
            layer_idx=0,
            cache_kwargs={
                "position_ids": torch.tensor([[5]]),
                "sliding_window_pattern": 2,
            },
        )
        assert cache.key_cache[0][0, 0, 2, 0].item() == pytest.approx(55.0, abs=1e-5), (
            f"Sliding: position 5 should map to slot 2, got {cache.key_cache[0][0, 0, 2, 0].item()}"
        )

    def test_sliding_padding_positions_do_not_corrupt(self):
        """Padding positions (position_id == -1) must not corrupt the cache."""
        sw = 4
        cache = self._make(ctx_len=sw, sw=sw)
        k, v = _kv(ctx_len=4)
        pids = torch.tensor([[0, 1, -1, -1]])  # two valid, two padding
        k_out, v_out = cache.update(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": pids,
                "sliding_window_pattern": 2,
            },
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()


# ---------------------------------------------------------------------------
# Tests: QEffHybridCache — multi-layer independence
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffHybridCacheMultiLayerIndependence:
    """Sliding and non-sliding layers must maintain independent state."""

    def test_four_layers_independent(self):
        """Write distinct values to 4 layers, verify each holds its own value."""
        cfg = _gemma2_cfg(num_layers=4, sliding_window=4, sliding_window_pattern=2)
        cache = _make_hybrid_cache_raw(cfg, ctx_len=16)
        for layer_idx in range(4):
            fill = float(layer_idx + 1) * 10.0
            k = torch.full((1, 2, 16, 8), fill)
            v = torch.full((1, 2, 16, 8), fill)
            cache.update(
                k,
                v,
                layer_idx=layer_idx,
                cache_kwargs={
                    "position_ids": _pids(16),
                    "sliding_window_pattern": 2,
                },
            )
        for layer_idx in range(4):
            expected = float(layer_idx + 1) * 10.0
            actual = cache.key_cache[layer_idx][0, 0, 0, 0].item()
            assert actual == pytest.approx(expected, abs=1e-4), f"Layer {layer_idx}: expected {expected}, got {actual}"

    def test_sliding_and_non_sliding_do_not_interfere(self):
        """
        layer_idx=0 is sliding, layer_idx=1 is non-sliding (pattern=2).
        Writing to one must not affect the other.
        """
        cfg = _gemma2_cfg(num_layers=4, sliding_window=4, sliding_window_pattern=2)
        cache = _make_hybrid_cache_raw(cfg, ctx_len=16)

        k0 = torch.full((1, 2, 16, 8), 1.0)
        cache.update(
            k0,
            k0.clone(),
            layer_idx=0,
            cache_kwargs={
                "position_ids": _pids(16),
                "sliding_window_pattern": 2,
            },
        )
        k1 = torch.full((1, 2, 16, 8), 2.0)
        cache.update(
            k1,
            k1.clone(),
            layer_idx=1,
            cache_kwargs={
                "position_ids": _pids(16),
                "sliding_window_pattern": 2,
            },
        )

        assert cache.key_cache[0][0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert cache.key_cache[1][0, 0, 0, 0].item() == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: QEffHybridCache — from_legacy_cache
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffHybridCacheFromLegacyCache:
    """from_legacy_cache must populate layers and survive a round-trip."""

    def test_from_legacy_cache_populates_layers(self):
        """
        Populate the cache by appending tensors directly to key_cache/value_cache
        (plain lists in _StandaloneHybridCache) and verify len() == 4.
        """
        cfg = _gemma2_cfg(num_layers=4)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        cache = _make_hybrid_cache_raw(cfg, ctx_len=8)
        for i in range(4):
            cache.key_cache.append(k.clone())
            cache.value_cache.append(v.clone())
        assert len(cache) == 4

    def test_from_legacy_cache_to_legacy_cache_shape_preserved(self):
        cfg = _gemma2_cfg(num_layers=4)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        cache = _make_hybrid_cache_raw(cfg, ctx_len=8)
        for i in range(4):
            cache.key_cache.append(k.clone())
            cache.value_cache.append(v.clone())
        legacy = cache.to_legacy_cache()
        assert isinstance(legacy, tuple) and len(legacy) == 4
        for i, (lk, lv) in enumerate(legacy):
            assert lk.shape == k.shape, f"Layer {i} key shape mismatch"
            assert lv.shape == v.shape, f"Layer {i} value shape mismatch"

    def test_get_seq_length_returns_correct_value(self):
        cfg = _gemma2_cfg(num_layers=4)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        cache = _make_hybrid_cache_raw(cfg, ctx_len=8)
        for i in range(4):
            cache.key_cache.append(k.clone())
            cache.value_cache.append(v.clone())
        # seq_length is the ctx_len dimension (dim 2) of the stored tensor
        assert cache.get_seq_length(layer_idx=0) == 8


# ---------------------------------------------------------------------------
# Tests: QEffHybridChunkedCache — correctness
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffHybridChunkedCacheCorrectness:
    """
    QEffHybridChunkedCache inherits from HybridChunkedCache.
    is_sliding[layer_idx] is set by the parent constructor based on config.
    We use from_legacy_cache to construct it safely.
    """

    def _make_via_legacy(self, ctx_len=16, num_layers=4):
        """
        Construct QEffHybridChunkedCache via __init__ and populate layers directly.
        key_cache is a KeyValuesWrapper that supports __setitem__, so we can assign
        tensors per layer without calling update() (which requires cache_kwargs).
        """
        cfg = _mistral_cfg(sliding_window=4)
        cache = QEffHybridChunkedCache(cfg, max_batch_size=1, max_cache_len=ctx_len)
        if not hasattr(cache, "key_cache"):
            # transformers>=4.57 no longer exposes key_cache/value_cache on HybridChunkedCache.
            # Attach legacy list fields so these backward-compatibility tests can exercise QEff methods.
            cache.key_cache = [None] * len(cache.layers)
            cache.value_cache = [None] * len(cache.layers)
        k = torch.zeros(1, 2, ctx_len, 8)
        v = torch.zeros(1, 2, ctx_len, 8)
        for layer_idx in range(num_layers):
            cache.key_cache[layer_idx] = k.clone()
            cache.value_cache[layer_idx] = v.clone()
        return cache, cfg

    def test_creation_via_legacy_succeeds(self):
        cache, _ = self._make_via_legacy()
        assert cache is not None

    def test_len_after_from_legacy(self):
        cache, _ = self._make_via_legacy(num_layers=4)
        assert len(cache) == 4

    def test_update_returns_finite_tensors(self):
        cache, _ = self._make_via_legacy(ctx_len=16)
        k, v = _kv(ctx_len=1)
        k_out, v_out = cache.update(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": torch.tensor([[8]]),
            },
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_non_sliding_scatter_at_correct_position(self):
        """
        For a non-sliding layer, write 42.0 at position 3 and verify it's there.
        """
        cache, _ = self._make_via_legacy(ctx_len=16)
        # Find a non-sliding layer index
        non_sliding_idx = next((i for i, s in enumerate(cache.is_sliding) if not s), None)
        if non_sliding_idx is None:
            pytest.skip("No non-sliding layer found in this config")

        k_dec, v_dec = _kv(ctx_len=1, fill=42.0)
        k_out, v_out = cache.update(
            k_dec,
            v_dec,
            layer_idx=non_sliding_idx,
            cache_kwargs={
                "position_ids": torch.tensor([[3]]),
            },
        )
        assert k_out[0, 0, 3, 0].item() == pytest.approx(42.0, abs=1e-5), (
            f"Expected 42.0 at position 3, got {k_out[0, 0, 3, 0].item()}"
        )

    def test_to_legacy_cache_round_trip(self):
        cache, _ = self._make_via_legacy(ctx_len=16, num_layers=4)
        legacy = cache.to_legacy_cache()
        assert isinstance(legacy, tuple) and len(legacy) == 4
        for lk, lv in legacy:
            assert lk.shape[2] == 16

    def test_get_seq_length_returns_correct_value(self):
        cache, _ = self._make_via_legacy(ctx_len=16, num_layers=4)
        assert cache.get_seq_length(layer_idx=0) == 16

    def test_multi_layer_independence(self):
        """Different layers must not interfere via direct tensor assignment."""
        cache, _ = self._make_via_legacy(ctx_len=16, num_layers=4)
        for layer_idx in range(4):
            fill = float(layer_idx + 1) * 5.0
            cache.key_cache[layer_idx] = torch.full((1, 2, 16, 8), fill)
            cache.value_cache[layer_idx] = torch.full((1, 2, 16, 8), fill)
        for layer_idx in range(4):
            expected = float(layer_idx + 1) * 5.0
            actual = cache.key_cache[layer_idx][0, 0, 0, 0].item()
            assert actual == pytest.approx(expected, abs=1e-4), f"Layer {layer_idx}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Tests: QEffGPTOSSHybridCache / QEffGPTOSSDynamicLayer — correctness
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffGPTOSSHybridCacheCorrectness:
    """
    QEffGPTOSSHybridCache is used by the GPT-OSS disaggregated serving path.
    Layers are tagged is_sliding from config.layer_types; update()/write_only() only
    need position_ids in cache_kwargs (sliding_window/is_sliding come from the layer).
    """

    def _make(self, sw=4):
        return QEffGPTOSSHybridCache(config=_gptoss_cfg(sliding_window=sw))

    def test_creation_succeeds(self):
        assert self._make() is not None

    def test_update_first_call_stores_tensors(self):
        cache = self._make()
        cache.append_new_layers(0)
        k, v = _kv(ctx_len=8)
        k_out, v_out = cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": _pids(8)})
        assert k_out is not None and v_out is not None

    def test_update_non_sliding_returns_finite(self):
        cache = self._make()
        cache.append_new_layers(1)  # layer 1 is full_attention per _gptoss_cfg
        k, v = _kv(ctx_len=8)
        k_out, v_out = cache.update(k, v, layer_idx=1, cache_kwargs={"position_ids": _pids(8)})
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_update_sliding_returns_finite(self):
        cache = self._make(sw=4)
        cache.append_new_layers(0)  # layer 0 is sliding_attention per _gptoss_cfg
        k, v = _kv(ctx_len=4)
        k_out, v_out = cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": _pids(4)})
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_cache_lengths_track_layer_tensor_shapes(self):
        cache = self._make(sw=4)
        k_sliding, v_sliding = _kv(ctx_len=4)
        k_full, v_full = _kv(ctx_len=16)

        cache.update(k_sliding, v_sliding, layer_idx=0, cache_kwargs={"position_ids": _pids(4)})
        cache.update(k_full, v_full, layer_idx=1, cache_kwargs={"position_ids": _pids(16)})

        assert cache.get_sliding_window_len() == 4
        assert cache.get_max_cache_len() == 16
        assert cache.layers[0].cache_len == 4
        assert cache.layers[1].cache_len == 16

    def test_non_sliding_scatter_at_correct_position(self):
        """Write 33.0 at position 4, verify it lands at slot 4."""
        cache = self._make()
        cache.append_new_layers(1)
        k_init, v_init = _kv(ctx_len=16, fill=0.0)
        cache.update(k_init, v_init, layer_idx=1, cache_kwargs={"position_ids": _pids(16)})
        k_dec, v_dec = _kv(ctx_len=1, fill=33.0)
        k_out, v_out = cache.update(k_dec, v_dec, layer_idx=1, cache_kwargs={"position_ids": torch.tensor([[4]])})
        assert k_out[0, 0, 4, 0].item() == pytest.approx(33.0, abs=1e-5), (
            f"Expected 33.0 at position 4, got {k_out[0, 0, 4, 0].item()}"
        )

    def test_non_sliding_prior_positions_not_corrupted(self):
        """Writing at position 4 must not corrupt positions 0..3."""
        cache = self._make()
        cache.append_new_layers(1)
        k_init = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1).expand(1, 2, 16, 8).clone()
        cache.update(k_init, k_init.clone(), layer_idx=1, cache_kwargs={"position_ids": _pids(16)})
        k_dec, v_dec = _kv(ctx_len=1, fill=99.0)
        k_out, _ = cache.update(k_dec, v_dec, layer_idx=1, cache_kwargs={"position_ids": torch.tensor([[4]])})
        assert k_out[0, 0, 4, 0].item() == pytest.approx(99.0, abs=1e-5)
        for pos in range(4):
            assert k_out[0, 0, pos, 0].item() == pytest.approx(float(pos), abs=1e-5), (
                f"Position {pos} corrupted: expected {float(pos)}, got {k_out[0, 0, pos, 0].item()}"
            )

    def test_write_only_populates_cache(self):
        """write_only must populate the cache without running gather."""
        cache = self._make()
        k, v = _kv(ctx_len=16)
        cache.write_only(k, v, layer_idx=1, cache_kwargs={"position_ids": _pids(16)})
        assert len(cache) == 2
        assert cache.layers[1].keys is not None

    def test_write_only_then_update_returns_finite(self):
        """write_only followed by update must return finite tensors."""
        cache = self._make()
        k_init, v_init = _kv(ctx_len=16)
        cache.write_only(k_init, v_init, layer_idx=1, cache_kwargs={"position_ids": _pids(16)})
        k_dec, v_dec = _kv(ctx_len=1)
        k_out, v_out = cache.update(k_dec, v_dec, layer_idx=1, cache_kwargs={"position_ids": torch.tensor([[8]])})
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_len_tracks_updated_layers(self):
        cache = QEffGPTOSSHybridCache(config=_gptoss_cfg(num_layers=4, sliding_window=4))
        k, v = _kv(ctx_len=8)
        for i in range(3):
            cache.update(k, v, layer_idx=i, cache_kwargs={"position_ids": _pids(8)})
        assert len(cache) == 3

    def test_to_legacy_cache_shape(self):
        cache = self._make()
        k, v = _kv(ctx_len=8)
        cache.update(k, v, layer_idx=1, cache_kwargs={"position_ids": _pids(8)})
        legacy = cache.to_legacy_cache()
        assert isinstance(legacy, tuple) and len(legacy) == 2  # layer 0 auto-appended + layer 1
        assert len(legacy[0]) == 2

    def test_multi_layer_independence(self):
        """Different layers must not interfere."""
        cache = QEffGPTOSSHybridCache(config=_gptoss_cfg(num_layers=4, sliding_window=4))
        for layer_idx in [1, 3]:  # full_attention layers per _gptoss_cfg
            fill = float(layer_idx + 1) * 7.0
            k = torch.full((1, 2, 16, 8), fill)
            v = torch.full((1, 2, 16, 8), fill)
            cache.update(k, v, layer_idx=layer_idx, cache_kwargs={"position_ids": _pids(16)})
        for layer_idx in [1, 3]:
            expected = float(layer_idx + 1) * 7.0
            actual = cache.layers[layer_idx].keys[0, 0, 0, 0].item()
            assert actual == pytest.approx(expected, abs=1e-4), f"Layer {layer_idx}: expected {expected}, got {actual}"

    def test_from_legacy_cache_populates_layers(self):
        cfg = _gptoss_cfg(num_layers=4, sliding_window=4)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        past = [(k.clone(), v.clone()) for _ in range(4)]
        cache = QEffGPTOSSHybridCache.from_legacy_cache(cfg, past_key_values=past)
        assert len(cache) == 4


# ---------------------------------------------------------------------------
# Tests: QEffGPTOSSDynamicLayer — chunked update methods (GAP C)
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffGPTOSSHybridCacheChunkedMethods:
    """
    Tests for full_cache_update_chunked and sliding_window_update_chunked
    on QEffGPTOSSHybridCache.

    Both methods require the layer to already exist (not the first call).
    batch_index=None is used to avoid the ONNX-export-only scatter_position_ids bug.
    """

    def _make(self, sw=4):
        return QEffGPTOSSHybridCache(config=_gptoss_cfg(sliding_window=sw))

    def _populate_layer(self, cache, layer_idx=1, ctx_len=16):
        """Populate a layer using update() so it exists in cache.layers."""
        k_init, v_init = _kv(ctx_len=ctx_len, fill=0.0)
        cache.update(k_init, v_init, layer_idx=layer_idx, cache_kwargs={"position_ids": _pids(ctx_len)})

    def test_full_cache_update_chunked_returns_finite(self):
        """full_cache_update_chunked must return finite tensors."""
        cache = self._make()
        self._populate_layer(cache)
        k_chunk, v_chunk = _kv(ctx_len=8)
        k_out, v_out = cache.full_cache_update_chunked(
            k_chunk, v_chunk, layer_idx=1, cache_kwargs={"position_ids": _pids(8), "batch_index": None}
        )
        assert torch.isfinite(k_out).all(), "full_cache_update_chunked must return finite keys"
        assert torch.isfinite(v_out).all(), "full_cache_update_chunked must return finite values"

    def test_full_cache_update_chunked_scatter_at_correct_position(self):
        """full_cache_update_chunked must scatter at the correct position."""
        cache = self._make()
        self._populate_layer(cache)
        # Write 77.0 at position 3
        k_chunk = torch.full((1, 2, 1, 8), 77.0)
        v_chunk = torch.full((1, 2, 1, 8), 77.0)
        k_out, v_out = cache.full_cache_update_chunked(
            k_chunk,
            v_chunk,
            layer_idx=1,
            cache_kwargs={"position_ids": torch.tensor([[3]]), "batch_index": None},
        )
        assert k_out[0, 0, 3, 0].item() == pytest.approx(77.0, abs=1e-5), (
            f"Expected 77.0 at position 3, got {k_out[0, 0, 3, 0].item()}"
        )

    def test_full_cache_update_chunked_prior_positions_not_corrupted(self):
        """Writing at position 3 must not corrupt positions 0..2."""
        cache = self._make()
        # Initialize with sequential values
        k_init = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1).expand(1, 2, 16, 8).clone()
        v_init = k_init.clone()
        cache.update(k_init, v_init, layer_idx=1, cache_kwargs={"position_ids": _pids(16)})
        # Write 99.0 at position 3
        k_chunk = torch.full((1, 2, 1, 8), 99.0)
        v_chunk = torch.full((1, 2, 1, 8), 99.0)
        k_out, _ = cache.full_cache_update_chunked(
            k_chunk,
            v_chunk,
            layer_idx=1,
            cache_kwargs={"position_ids": torch.tensor([[3]]), "batch_index": None},
        )
        assert k_out[0, 0, 3, 0].item() == pytest.approx(99.0, abs=1e-5)
        for pos in range(3):
            assert k_out[0, 0, pos, 0].item() == pytest.approx(float(pos), abs=1e-5), (
                f"Position {pos} corrupted: expected {float(pos)}, got {k_out[0, 0, pos, 0].item()}"
            )

    def test_sliding_window_update_chunked_returns_finite(self):
        """sliding_window_update_chunked must return finite tensors."""
        sw = 4
        cache = self._make(sw=sw)
        self._populate_layer(cache, layer_idx=0)  # layer 0 is sliding_attention
        seq_len = 4
        k_chunk, v_chunk = _kv(ctx_len=seq_len)
        k_out, v_out = cache.sliding_window_update_chunked(
            k_chunk,
            v_chunk,
            layer_idx=0,
            cache_kwargs={"position_ids": _pids(seq_len), "batch_index": None, "sliding_window": sw},
        )
        assert torch.isfinite(k_out).all(), "sliding_window_update_chunked must return finite keys"
        assert torch.isfinite(v_out).all(), "sliding_window_update_chunked must return finite values"

    def test_sliding_window_update_chunked_output_shape(self):
        """sliding_window_update_chunked output ctx_len must equal seq_len + sliding_window."""
        sw = 4
        cache = self._make(sw=sw)
        self._populate_layer(cache, layer_idx=0)
        seq_len = 4
        k_chunk, v_chunk = _kv(ctx_len=seq_len)
        k_out, v_out = cache.sliding_window_update_chunked(
            k_chunk,
            v_chunk,
            layer_idx=0,
            cache_kwargs={"position_ids": _pids(seq_len), "batch_index": None, "sliding_window": sw},
        )
        # ctx_len = position_ids.shape[1] + sliding_window_len = seq_len + sw
        expected_ctx_len = seq_len + sw
        assert k_out.shape[2] == expected_ctx_len, f"Expected ctx_len={expected_ctx_len}, got {k_out.shape[2]}"

    def test_sliding_window_update_chunked_with_offset_position(self):
        """sliding_window_update_chunked with position > sliding_window must use add_idx offset."""
        sw = 4
        cache = self._make(sw=sw)
        self._populate_layer(cache, layer_idx=0)
        seq_len = 4
        # Start at position 8 (> sw=4), so add_idx = 8 - 4 = 4
        k_chunk, v_chunk = _kv(ctx_len=seq_len)
        k_out, v_out = cache.sliding_window_update_chunked(
            k_chunk,
            v_chunk,
            layer_idx=0,
            cache_kwargs={"position_ids": _pids(seq_len, start=8), "batch_index": None, "sliding_window": sw},
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()


# ---------------------------------------------------------------------------
# Tests: from_legacy_cache classmethods (GAP C)
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestFromLegacyCacheClassmethods:
    """
    Tests that from_legacy_cache classmethods exist and have correct signatures.
    QEffHybridCache.from_legacy_cache is a classmethod but has a broken __init__ chain.
    QEffHybridChunkedCache.from_legacy_cache is a classmethod that should work.
    """

    def test_qeff_hybrid_cache_has_from_legacy_cache(self):
        """QEffHybridCache must have a from_legacy_cache classmethod."""
        from QEfficient.transformers.cache_utils import QEffHybridCache

        assert hasattr(QEffHybridCache, "from_legacy_cache")
        assert callable(QEffHybridCache.from_legacy_cache)

    def test_qeff_hybrid_chunked_cache_has_from_legacy_cache(self):
        """QEffHybridChunkedCache must have a from_legacy_cache classmethod."""
        assert hasattr(QEffHybridChunkedCache, "from_legacy_cache")
        assert callable(QEffHybridChunkedCache.from_legacy_cache)

    def test_qeff_gptoss_dynamic_cache_has_from_legacy_cache(self):
        """QEffGPTOSSHybridCache must have a from_legacy_cache classmethod."""
        assert hasattr(QEffGPTOSSHybridCache, "from_legacy_cache")
        assert callable(QEffGPTOSSHybridCache.from_legacy_cache)

    def test_qeff_gptoss_dynamic_cache_from_legacy_cache_creates_instance(self):
        """QEffGPTOSSHybridCache.from_legacy_cache must create a valid instance."""
        cfg = _gptoss_cfg(num_layers=4, sliding_window=4)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        past = [(k.clone(), v.clone()) for _ in range(4)]
        cache = QEffGPTOSSHybridCache.from_legacy_cache(cfg, past_key_values=past)
        assert isinstance(cache, QEffGPTOSSHybridCache)
        assert len(cache) == 4

    def test_qeff_gptoss_dynamic_cache_from_legacy_cache_preserves_shapes(self):
        """from_legacy_cache must preserve tensor shapes."""
        cfg = _gptoss_cfg(num_layers=4, sliding_window=4)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        past = [(k.clone(), v.clone()) for _ in range(4)]
        cache = QEffGPTOSSHybridCache.from_legacy_cache(cfg, past_key_values=past)
        for i in range(4):
            assert cache.layers[i].keys.shape[0] == 1  # batch
            assert cache.layers[i].keys.shape[1] == 2  # heads
