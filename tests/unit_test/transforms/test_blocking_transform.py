# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for BlockingAttentionTransform in QEfficient.transformers.models.pytorch_transforms.

Verifies that:
  1. BlockingAttentionTransform attaches attn_blocking_config to QEff attention modules
  2. Works correctly for supported model families
  3. Preserves blocking mode/config values
  4. Re-applying overrides the previous config
  5. Handles wrapper config fallback and preserves fast CPU parity

All tests run on CPU only, using tiny in-memory models.
KVCacheTransform must be applied before BlockingAttentionTransform because the
blocking transform matches against QEff attention class types (the *values* of
KVCacheTransform._module_mapping), not the raw HF attention class types (the keys).
"""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from QEfficient.blocking import attention_blocking
from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode

VOCAB_SIZE = 500
CTX_LEN = 32


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval()


def make_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return Qwen3ForCausalLM(cfg).eval()


def make_tiny_qwen3_vl():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    text_cfg = Qwen3VLTextConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    vision_cfg = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)
    return Qwen3VLForConditionalGeneration(cfg).eval()


def make_tiny_gpt_oss():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    cfg = GptOssConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=64,
        head_dim=32,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=CTX_LEN,
        rope_parameters={"rope_type": "default"},
    )
    return GptOssForCausalLM(cfg).eval()


def make_tiny_gemma():
    from transformers import GemmaConfig, GemmaForCausalLM

    cfg = GemmaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return GemmaForCausalLM(cfg).eval()


def make_tiny_gemma2():
    from transformers import Gemma2Config, Gemma2ForCausalLM

    cfg = Gemma2Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
        sliding_window=CTX_LEN,
    )
    return Gemma2ForCausalLM(cfg).eval()


def make_tiny_mistral():
    from transformers import MistralConfig, MistralForCausalLM

    cfg = MistralConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return MistralForCausalLM(cfg).eval()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_FACTORIES = [
    (make_tiny_llama, "llama"),
    (make_tiny_qwen3, "qwen3"),
    (make_tiny_qwen3_vl, "qwen3_vl"),
    (make_tiny_gpt_oss, "gpt_oss"),
    (make_tiny_gemma, "gemma"),
    (make_tiny_gemma2, "gemma2"),
    (make_tiny_mistral, "mistral"),
]
_MODEL_IDS = [label for _, label in _MODEL_FACTORIES]


def _qeff_attention_modules(model):
    """Return all modules whose type is in KVCacheTransform supported attention classes."""
    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

    supported = {
        qeff_cls for qeff_cls in KVCacheTransform._module_mapping.values() if qeff_cls.__name__.endswith("Attention")
    }
    return [m for m in model.modules() if type(m) in supported]


def _blocking_cfg(**kwargs):
    return AttentionBlockingConfig(**kwargs)


def _make_qeff_inputs(input_ids, config, ctx_len=CTX_LEN):
    batch, seq = input_ids.shape
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    n_layers = config.num_hidden_layers
    n_attn = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_attn)
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_attn)
    past_key_values = tuple(
        (
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    )
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


# ---------------------------------------------------------------------------
# Tests: basic application per model family
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingTransformApplied:
    """BlockingAttentionTransform must attach attn_blocking_config to all QEff attention modules."""

    @pytest.mark.parametrize("make_model,label", _MODEL_FACTORIES, ids=_MODEL_IDS)
    def test_config_attached_to_all_attn_modules(self, make_model, label):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_model()
        model, _ = KVCacheTransform.apply(model)
        config = _blocking_cfg(mode=BlockingMode.KV, num_kv_blocks=2)
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        attn_mods = _qeff_attention_modules(model)
        assert attn_mods, f"[{label}] no QEff attention modules found after KVCacheTransform"

        for m in attn_mods:
            assert hasattr(m, "attn_blocking_config"), f"[{label}] {type(m).__name__} missing attn_blocking_config"
            assert m.attn_blocking_config is config, (
                f"[{label}] attn_blocking_config must be the same object that was passed in"
            )


# ---------------------------------------------------------------------------
# Tests: blocking modes
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingModes:
    """BlockingAttentionTransform must preserve the BlockingMode in the attached config."""

    @pytest.mark.parametrize(
        "mode",
        [BlockingMode.NONE, BlockingMode.KV, BlockingMode.Q, BlockingMode.H, BlockingMode.QKV],
    )
    def test_blocking_mode_preserved_on_llama(self, mode):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)
        config = AttentionBlockingConfig(mode=mode, head_block_size=8, num_kv_blocks=2, num_q_blocks=2, ctx_len=128)
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        for m in _qeff_attention_modules(model):
            assert m.attn_blocking_config.mode == mode, f"Expected mode={mode}, got {m.attn_blocking_config.mode}"

    def test_all_config_fields_preserved(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)
        config = AttentionBlockingConfig(
            mode=BlockingMode.KV,
            num_kv_blocks=4,
            num_q_blocks=2,
            head_block_size=8,
            skip_kv=False,
            num_batch_blocks=1,
            ctx_len=128,
        )
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        for m in _qeff_attention_modules(model):
            c = m.attn_blocking_config
            assert c.mode == BlockingMode.KV
            assert c.num_kv_blocks == 4
            assert c.num_q_blocks == 2
            assert c.head_block_size == 8
            assert c.skip_kv is False
            assert c.num_batch_blocks == 1


@pytest.mark.transforms
def test_generic_blocked_attention_infers_prefill_only_from_mode(monkeypatch):
    class Cache:
        def __init__(self):
            self.write_only_calls = []

        def write_only(self, key, value, layer_idx, cache_kwargs):
            self.write_only_calls.append((key, value, layer_idx, cache_kwargs))

    cache = Cache()
    query = torch.ones(1, 1, 1, 1)
    key = torch.ones(1, 1, 1, 1)
    value = torch.ones(1, 1, 1, 1)
    strategy_calls = []

    def prefill_strategy(**kwargs):
        strategy_calls.append(kwargs)
        return kwargs["query"], None

    monkeypatch.setitem(attention_blocking._STRATEGIES, BlockingMode.PREFILL_Q, prefill_strategy)

    output, weights = attention_blocking.generic_blocked_attention_interface(
        module=type("Attention", (), {"layer_idx": 0})(),
        query=query,
        key=key,
        value=value,
        past_key_value=cache,
        blocking_config=AttentionBlockingConfig(mode=BlockingMode.PREFILL_Q, num_q_blocks=1),
    )

    assert torch.equal(output, query)
    assert weights is None
    assert len(cache.write_only_calls) == 1
    assert len(strategy_calls) == 1


@pytest.mark.transforms
def test_kv_batch_fold_allows_uninitialized_cache_layers(monkeypatch):
    class Cache:
        layers = []

        def write_only_batch(self, key, value, layer_idx, cache_kwargs):
            pass

    def batch_strategy(**kwargs):
        return kwargs["query"], None

    monkeypatch.setitem(attention_blocking._STRATEGIES, BlockingMode.KV_BATCH_FOLD, batch_strategy)
    query = torch.ones(1, 1, 1, 1)

    output, weights = attention_blocking.generic_blocked_attention_interface(
        module=type("Attention", (), {"layer_idx": 0})(),
        query=query,
        key=query,
        value=query,
        position_ids=torch.zeros(1, 1, dtype=torch.long),
        layer_idx=0,
        past_key_value=Cache(),
        blocking_config=AttentionBlockingConfig(mode=BlockingMode.KV_BATCH_FOLD, num_kv_blocks=1),
    )

    assert torch.equal(output, query)
    assert weights is None


@pytest.mark.transforms
def test_kv_batch_fold_preserves_optional_gdn_num_head_blocks():
    from QEfficient.blocking.blocking_configurator import build_transformer_blocking_config_for_transform

    config = build_transformer_blocking_config_for_transform(
        model_config=object(),
        ctx_len=1024,
        seq_len=1,
        bs=512,
        num_devices=4,
        qaic_config={
            "blocking_mode": "kv_batch_fold",
            "num_kv_blocks": 16,
            "gdn_num_head_blocks": 8,
        },
    )

    assert config.mode == BlockingMode.KV_BATCH_FOLD
    assert config.batch_fold is True
    assert config.gdn_num_head_blocks == 8


# ---------------------------------------------------------------------------
# Tests: re-application overrides the previous config
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingTransformIdempotent:
    """Applying BlockingAttentionTransform twice must replace the first config with the second."""

    def test_second_apply(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)

        config1 = AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=2, ctx_len=1024)
        config2 = AttentionBlockingConfig(mode=BlockingMode.Q, num_q_blocks=4)

        model, _ = BlockingAttentionTransform.apply(model, config1)
        model, transformed = BlockingAttentionTransform.apply(model, config2)

        assert transformed, "Reapplication of BlockingAttentionTransform did not succeed"

        for m in _qeff_attention_modules(model):
            assert m.attn_blocking_config is config2, (
                "Second BlockingAttentionTransform.apply must override the first config"
            )
            assert m.attn_blocking_config.mode == BlockingMode.Q


# ---------------------------------------------------------------------------
# Tests: wrapper config fallback + CPU parity
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingWrapperFallbackAndParity:
    """Regression guards for wrapper config lookup and CPU parity checks."""

    def test_wrapper_without_config_uses_nested_model_config(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        class _DummyAttention(nn.Module):
            pass

        class _DeepseekContainer(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {"architectures": ["DeepseekV3ForCausalLM"]})()
                self.attn = _DummyAttention()

        class _WrapperWithoutConfig(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        cfg = AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=2, ctx_len=128)
        wrapped = _WrapperWithoutConfig(_DeepseekContainer())
        wrapped, transformed = BlockingAttentionTransform.apply(wrapped, cfg)

        assert transformed, "BlockingAttentionTransform must use nested wrapper model config"
        assert wrapped.model.attn.attn_blocking_config is cfg

    @pytest.mark.parametrize(
        "blocking_cfg",
        [
            AttentionBlockingConfig(mode=BlockingMode.NONE),
            AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=2, ctx_len=128),
        ],
        ids=["mode_none", "mode_kv"],
    )
    def test_cpu_parity_original_vs_transformed_with_same_input(self, blocking_cfg):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        torch.manual_seed(7)
        base = make_tiny_llama()
        original = deepcopy(base).eval()
        transformed = deepcopy(base).eval()
        transformed, _ = KVCacheTransform.apply(transformed)
        transformed, applied = BlockingAttentionTransform.apply(transformed, blocking_cfg)
        assert applied

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        qeff_inputs = _make_qeff_inputs(input_ids, transformed.config)

        with torch.no_grad():
            original_token = original(input_ids=input_ids).logits[:, -1, :].argmax(-1)
            transformed_token = transformed(**qeff_inputs).logits[:, -1, :].argmax(-1)

        assert torch.equal(original_token, transformed_token), (
            "Original and transformed model outputs diverged for same CPU input"
        )

    @pytest.mark.parametrize(
        "blocking_cfg",
        [
            AttentionBlockingConfig(mode=BlockingMode.NONE),
            AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=2, ctx_len=128),
            AttentionBlockingConfig(mode=BlockingMode.QKV, num_kv_blocks=2, num_q_blocks=2, ctx_len=128),
            AttentionBlockingConfig(
                mode=BlockingMode.HQKV, num_kv_blocks=2, num_q_blocks=2, head_block_size=1, ctx_len=128
            ),
            AttentionBlockingConfig(
                mode=BlockingMode.BHQKV,
                num_kv_blocks=2,
                num_q_blocks=2,
                head_block_size=1,
                num_batch_blocks=2,
                ctx_len=128,
            ),
        ],
        ids=["mode_none", "mode_kv", "mode_qkv", "mode_hqkv", "mode_bhqkv"],
    )
    def test_cpu_logits_allclose_original_vs_blocked(self, blocking_cfg):
        """Blocked attention (kv/qkv/hqkv, paged_attention=False) must reproduce the
        unblocked model's logits to numerical precision, not just the same argmax
        token - splitting attention into blocks with running-softmax accumulation
        is a reformulation of the same computation, so it should match tightly.
        Also covers "non-paged generation unchanged when num_kv_blocks is absent"
        via the mode_none case, which never sets num_kv_blocks.

        QEffLlamaForCausalLM.forward always gathers only the last token's hidden
        state (via position_ids.argmax) before the lm_head projection, so it never
        returns per-position logits for the full prompt - compare the last position
        only, but with allclose instead of just argmax to make this a real numeric
        tightening over test_cpu_parity_original_vs_transformed_with_same_input."""
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        torch.manual_seed(7)
        base = make_tiny_llama()
        original = deepcopy(base).eval()
        transformed = deepcopy(base).eval()
        transformed, _ = KVCacheTransform.apply(transformed)
        transformed, applied = BlockingAttentionTransform.apply(transformed, blocking_cfg)
        assert applied

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        qeff_inputs = _make_qeff_inputs(input_ids, transformed.config)

        with torch.no_grad():
            original_logits = original(input_ids=input_ids).logits[:, -1, :]
            transformed_logits = transformed(**qeff_inputs).logits[:, -1, :]

        assert torch.allclose(original_logits, transformed_logits, atol=1e-4, rtol=1e-4), (
            f"Logits diverged beyond tolerance for blocking_cfg.mode={blocking_cfg.mode} "
            f"(max abs diff={(original_logits - transformed_logits).abs().max().item()})"
        )
