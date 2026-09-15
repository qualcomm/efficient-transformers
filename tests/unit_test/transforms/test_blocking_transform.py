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
from torch import nn

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

    def test_kv_headpar_uses_blocked_kv_cache(self):
        from QEfficient.blocking.attention_blocking import _uses_blocked_kv_cache

        assert _uses_blocked_kv_cache(BlockingMode.KV_HEADPAR)

    def test_decode_kv_blocking_does_not_pad_export_query_axis(self):
        from QEfficient.transformers.models.modeling_auto import _blocking_requires_query_axis_export_padding

        assert not _blocking_requires_query_axis_export_padding(
            AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=8)
        )
        assert not _blocking_requires_query_axis_export_padding(
            AttentionBlockingConfig(mode=BlockingMode.KV_HEADPAR, num_kv_blocks=8, headpar_split=4)
        )
        assert _blocking_requires_query_axis_export_padding(
            AttentionBlockingConfig(mode=BlockingMode.Q, num_q_blocks=8)
        )
        assert _blocking_requires_query_axis_export_padding(
            AttentionBlockingConfig(mode=BlockingMode.PREFILL_KV, num_kv_blocks=8)
        )

    def test_qwen3_5_moe_hybrid_cache_supports_kv_headpar_split_reads(self):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

        from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeDynamicCache

        cfg = Qwen3_5MoeTextConfig(
            layer_types=["linear_attention", "full_attention"],
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            hidden_size=128,
            head_dim=32,
        )
        cache = QEffQwen3_5MoeDynamicCache(cfg)
        key_states = torch.arange(4 * 32, dtype=torch.float32).reshape(1, 1, 4, 32)
        value_states = key_states + 1000
        cache.kv_layers[1].keys = key_states
        cache.kv_layers[1].values = value_states
        cache_kwargs = {"position_ids": torch.tensor([[3]])}

        expected_key, expected_value = cache.read_only_blockedKV(1, 3, 1, cache_kwargs)

        torch.testing.assert_close(cache.read_only_blocked_K(1, 3, 1, cache_kwargs), expected_key)
        torch.testing.assert_close(cache.read_only_blocked_V(1, 3, 1, cache_kwargs), expected_value)


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
def test_past_key_value_update_uses_comp_ctx_length_by_default():
    class Cache:
        def update(self, key, value, layer_idx, cache_kwargs):
            self.cache_kwargs = cache_kwargs
            return key, value

    attention_mask = torch.zeros(1, 1, 1, 8)
    cache = Cache()

    _, _, updated_attention_mask, _ = attention_blocking.past_key_value_update(
        module=type("Attention", (), {"layer_idx": 0})(),
        key=torch.ones(1, 1, 1, 1),
        value=torch.ones(1, 1, 1, 1),
        attention_mask=attention_mask,
        past_key_value=cache,
        comp_ctx_lengths=torch.zeros(4, dtype=torch.long),
    )

    assert updated_attention_mask.shape[-1] == 4
    assert cache.cache_kwargs["CCL"] == 4


@pytest.mark.transforms
def test_past_key_value_update_preserves_pre_aligned_attention_mask():
    class Cache:
        def update(self, key, value, layer_idx, cache_kwargs):
            self.cache_kwargs = cache_kwargs
            return key, value

    attention_mask = torch.zeros(1, 1, 1, 4)
    cache = Cache()

    _, _, updated_attention_mask, _ = attention_blocking.past_key_value_update(
        module=type("Attention", (), {"layer_idx": 0})(),
        key=torch.ones(1, 1, 1, 1),
        value=torch.ones(1, 1, 1, 1),
        attention_mask=attention_mask,
        past_key_value=cache,
        comp_ctx_lengths=torch.zeros(4, dtype=torch.long),
        ccl_length=attention_mask.shape[-1],
    )

    assert updated_attention_mask is attention_mask
    assert cache.cache_kwargs["CCL"] == 4


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


@pytest.mark.transforms
@pytest.mark.parametrize(
    "gdn_class",
    [
        pytest.param(
            "QEffQwen3_5GatedDeltaNet",
            id="qwen3_5",
        ),
        pytest.param(
            "QEffQwen3_5MoeGatedDeltaNet",
            id="qwen3_5_moe",
        ),
    ],
)
def test_blocking_transform_binds_qwen3_5_gated_delta_forward(gdn_class):
    from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform
    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import QEffQwen3_5GatedDeltaNet
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeGatedDeltaNet

    classes = {
        "QEffQwen3_5GatedDeltaNet": QEffQwen3_5GatedDeltaNet,
        "QEffQwen3_5MoeGatedDeltaNet": QEffQwen3_5MoeGatedDeltaNet,
    }
    module_class = classes[gdn_class]
    module = object.__new__(module_class)
    torch.nn.Module.__init__(module)
    model = torch.nn.Module()
    model.gated_delta = module
    config = AttentionBlockingConfig(mode=BlockingMode.KV_BATCH_FOLD, num_kv_blocks=2, batch_fold=True)

    _, transformed = BlockingAttentionTransform.apply(model, config)

    assert transformed
    assert module.attn_blocking_config is config
    assert module.forward.__func__ is module_class.forward


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

def test_glm_attention_config_resolves_dense_and_dsa_layers():
    from types import SimpleNamespace

    from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        resolve_glm_attention_layer_configs,
    )

    config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["full_attention", "indexed_attention"],
        indexer_types=["full", "shared"],
        index_topk=32,
    )
    resolved = resolve_glm_attention_layer_configs(
        config,
        {
            "blocking_mode": "par",
            "num_kv_blocks": 2,
            "par_num_split": 16,
            "dsa_topk": 16,
            "mla_absorption": {"absorption": False, "online": False, "cache_compressed": True},
            "indexer_dp": 1,
            "indexer_cp": 2,
            "indexer_kvp": 1,
            "attn_dp": 2,
            "attn_cp": 1,
            "attn_kvp": 1,
            "indexer_num_blocks": 1,
            "num_cores_per_device": 16,
        },
        batch_size=2,
        context_length=512,
        num_devices=2,
        num_cores=16,
    )

    assert resolved[0].attention_type == "dense_mla"
    assert resolved[0].blocking_mode == "par"
    assert resolved[0].absorption is False
    assert resolved[1].attention_type == "dsa"
    assert resolved[1].blocking_mode == "dsa"
    assert resolved[1].absorption is True
    assert resolved[1].indexer_type == "shared"
    assert resolved[1].dsa_topk == config.index_topk
    assert resolved[1].attention_tokens_per_core == config.index_topk // 16


@pytest.mark.parametrize(
    ("blocking_mode", "seq_len", "prefill_only", "error"),
    [
        ("par", 4, False, "decode-only"),
        ("prefill_par", 1, False, "prefill_only=True"),
        ("prefill_par_online", 1, False, "prefill_only=True"),
    ],
)
def test_glm_dense_parallel_modes_validate_decode_prefill_contract(blocking_mode, seq_len, prefill_only, error):
    from types import SimpleNamespace

    from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        resolve_glm_attention_layer_configs,
    )

    config = SimpleNamespace(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        indexer_types=["full"],
        index_topk=32,
    )
    qaic_config = {
        "blocking_mode": blocking_mode,
        "num_kv_blocks": 2,
        "par_num_split": 4,
        "mla_absorption": {
            "absorption": True,
            "online": blocking_mode == "prefill_par_online",
            "cache_compressed": True,
        },
    }
    with pytest.raises(ValueError, match=error):
        resolve_glm_attention_layer_configs(
            config,
            qaic_config,
            batch_size=1,
            context_length=32,
            num_devices=1,
            num_cores=4,
            seq_len=seq_len,
            prefill_only=prefill_only,
        )


@pytest.mark.parametrize("blocking_mode", ["none", "par", "prefill_par", "prefill_par_online"])
def test_glm_specific_blocking_modes_skip_generic_blocking_config(blocking_mode):
    from types import SimpleNamespace

    from QEfficient.blocking.blocking_configurator import build_transformer_blocking_config_for_transform

    config = SimpleNamespace(model_type="glm_moe_dsa")
    assert (
        build_transformer_blocking_config_for_transform(
            config,
            ctx_len=128,
            seq_len=1,
            qaic_config={"blocking_mode": blocking_mode},
        )
        is None
    )


def test_glm_attention_config_rejects_non_unit_kvp():
    from types import SimpleNamespace

    from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        resolve_glm_attention_layer_configs,
    )

    config = SimpleNamespace(
        num_hidden_layers=1,
        layer_types=["indexed_attention"],
        indexer_types=["full"],
        index_topk=16,
    )
    with pytest.raises(ValueError, match="only indexer_kvp=1"):
        resolve_glm_attention_layer_configs(
            config,
            {"indexer_kvp": 2},
            batch_size=1,
            context_length=256,
            num_devices=1,
            num_cores=16,
        )


@pytest.mark.parametrize(
    ("batch_size", "dp", "cp", "context_length", "num_blocks", "topk"),
    [
        (2, 1, 1, 16, 2, 4),
        (2, 1, 2, 32, 2, 4),
        (16, 1, 16, 4096, 16, 32),
    ],
    ids=["cp1", "cp2", "ts16"],
)
def test_glm_blocked_topk_matches_monolithic_after_cache_updates(batch_size, dp, cp, context_length, num_blocks, topk):
    from QEfficient.blocking.glm_attention import blocked_glm_dsa_topk
    from QEfficient.transformers.cache_utils import glm_dsa_scatter_cache

    torch.manual_seed(11)
    num_heads = 2
    head_dim = 4
    batch_local = batch_size // dp
    local_context = context_length // cp
    folded_cache = torch.zeros(batch_local, dp * cp, local_context, head_dim)
    logical_keys = torch.rand(batch_size, context_length, head_dim)
    logical_keys = logical_keys + torch.arange(context_length).view(1, -1, 1) * 1e-3

    update_points = (context_length // 2, context_length)
    previous = 0
    for stop in update_points:
        positions = torch.arange(previous, stop, dtype=torch.int64).view(1, -1).expand(batch_size, -1)
        folded_cache = glm_dsa_scatter_cache(
            folded_cache,
            positions,
            logical_keys[:, previous:stop],
            dp=dp,
            cp=cp,
        )
        previous = stop

        query = torch.rand(batch_size, 1, num_heads, head_dim)
        head_weights = torch.rand(batch_size, 1, num_heads)
        position_ids = torch.full((batch_size, 1), stop - 1, dtype=torch.int64)
        attention_mask = torch.arange(context_length).view(1, 1, -1) >= stop
        attention_mask = attention_mask.expand(batch_size, -1, -1).clone()
        attention_mask[:, :, 1] = True

        actual = blocked_glm_dsa_topk(
            query,
            head_weights,
            folded_cache,
            attention_mask,
            position_ids,
            scale=head_dim**-0.5,
            dp=dp,
            cp=cp,
            num_blocks=num_blocks,
            num_cores_per_device=1,
            tokens_per_core=local_context // num_blocks,
            block_topk=min(topk, local_context // num_blocks),
            final_topk=topk,
        )

        reconstructed = (
            folded_cache.view(batch_local, dp, cp, local_context, head_dim)
            .transpose(2, 3)
            .reshape(batch_size, context_length, head_dim)
        )
        scores = torch.matmul(query.float(), reconstructed.transpose(-1, -2).float().unsqueeze(1))
        scores = torch.relu(scores * (head_dim**-0.5))
        scores = torch.matmul(head_weights.float().unsqueeze(-2), scores).squeeze(-2)
        scores = scores.masked_fill(attention_mask, float("-inf"))
        scores = scores.to(torch.float16).view(batch_local, dp, 1, local_context, cp).transpose(-1, -2)
        local_topk = min(topk, local_context)
        local_scores, local_order = scores.topk(local_topk, dim=-1, sorted=False)

        local_ids = torch.arange(local_context, device=scores.device, dtype=torch.int64).view(1, 1, 1, 1, -1)
        cp_ids = torch.arange(cp, device=scores.device, dtype=torch.int64).view(1, 1, 1, cp, 1)
        logical_ids = (local_ids * cp + cp_ids).expand_as(scores)
        local_indices = torch.gather(logical_ids, -1, local_order)

        merged_scores = local_scores.reshape(batch_local, dp, 1, cp * local_topk)
        merged_indices = local_indices.reshape(batch_local, dp, 1, cp * local_topk)
        _, merged_order = merged_scores.topk(topk, dim=-1, sorted=False)
        expected = torch.gather(merged_indices, -1, merged_order).reshape(batch_size, 1, topk).to(torch.int32)

        torch.testing.assert_close(
            actual.sort(dim=-1).values,
            expected.sort(dim=-1).values,
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_glm_blocked_prefill_topk_matches_monolithic(parallel):
    from QEfficient.blocking.glm_attention import blocked_glm_dsa_prefill_topk
    from QEfficient.transformers.cache_utils import glm_dsa_scatter_cache

    torch.manual_seed(23)
    batch_size, query_length, context_length = 2, 4, 16
    dp, cp, num_heads, head_dim = 1, 2, 2, 4
    folded_cache = torch.zeros(batch_size, dp * cp, context_length // cp, head_dim)
    logical_keys = torch.rand(batch_size, context_length, head_dim)
    cache_positions = torch.arange(context_length).view(1, -1).expand(batch_size, -1)
    folded_cache = glm_dsa_scatter_cache(folded_cache, cache_positions, logical_keys, dp=dp, cp=cp)
    query = torch.rand(batch_size, query_length, num_heads, head_dim)
    head_weights = torch.rand(batch_size, query_length, num_heads)
    position_ids = torch.arange(8, 12).view(1, -1).expand(batch_size, -1)
    attention_mask = torch.arange(context_length).view(1, 1, -1) > position_ids.unsqueeze(-1)
    attention_mask = attention_mask.clone()
    attention_mask[:, :, 1] = True

    actual = blocked_glm_dsa_prefill_topk(
        query,
        head_weights,
        folded_cache,
        attention_mask,
        position_ids,
        scale=head_dim**-0.5,
        dp=dp,
        cp=cp,
        num_blocks=2,
        num_cores_per_device=2,
        query_chunk_size=4,
        query_block_size=4,
        topk_blocking=2,
        parallel=parallel,
        final_topk=4,
    )

    scores = torch.einsum("bqhd,btd->bqht", query.float(), logical_keys.float()) * (head_dim**-0.5)
    scores = (torch.relu(scores) * head_weights.unsqueeze(-1)).sum(dim=2)
    scores = scores.masked_fill(attention_mask, float("-inf"))
    expected = scores.topk(4, dim=-1).indices.to(torch.int32)
    torch.testing.assert_close(torch.sort(actual, dim=-1).values, torch.sort(expected, dim=-1).values)


@pytest.mark.parametrize(
    ("batch_size", "context_length", "num_devices", "error"),
    [
        (1, 4096, 16, "batch_size must be divisible by attention_dp"),
        (16, 2176, 16, "indexer local context"),
        (16, 4096, 8, "indexer_dp \* indexer_cp"),
        (16, 4100, 16, "context_length must be divisible by indexer_cp"),
    ],
)
def test_glm_ts16_rejects_invalid_runtime_dimensions(batch_size, context_length, num_devices, error):
    from types import SimpleNamespace

    from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        resolve_glm_attention_layer_configs,
    )

    config = SimpleNamespace(
        num_hidden_layers=1,
        layer_types=["indexed_attention"],
        indexer_types=["full"],
        index_topk=2048,
    )
    qaic_config = {
        "indexer_dp": 1,
        "indexer_cp": 16,
        "indexer_kvp": 1,
        "attn_dp": 16,
        "attn_cp": 1,
        "attn_kvp": 1,
        "indexer_num_blocks": 16,
        "num_cores_per_device": 16,
    }
    with pytest.raises(ValueError, match=error):
        resolve_glm_attention_layer_configs(
            config,
            qaic_config,
            batch_size=batch_size,
            context_length=context_length,
            num_devices=num_devices,
            num_cores=16,
        )


def test_glm_ts16_resolves_topology_values_for_hashing():
    from types import SimpleNamespace

    from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        resolve_glm_attention_layer_configs,
    )

    config = SimpleNamespace(
        num_hidden_layers=1,
        layer_types=["indexed_attention"],
        indexer_types=["full"],
        index_topk=2048,
    )
    resolved = resolve_glm_attention_layer_configs(
        config,
        {
            "indexer_dp": 1,
            "indexer_cp": 16,
            "attn_dp": 16,
            "attn_cp": 1,
            "indexer_num_blocks": 16,
            "num_cores_per_device": 16,
        },
        batch_size=16,
        context_length=262144,
        num_devices=16,
        num_cores=16,
    )[0]

    assert resolved.indexer_local_context == 16384
    assert resolved.indexer_block_width == 1024
    assert resolved.indexer_tokens_per_core == 64
    assert resolved.indexer_block_topk == 1024
    assert resolved.attention_tokens_per_core == 128
    assert resolved.to_hash_dict()["indexer_tokens_per_core"] == 64


def test_glm_tiled_sparse_attention_matches_flat_selected_softmax():
    from types import SimpleNamespace

    from QEfficient.blocking.glm_attention import _glm_tiled_sparse_mla_attention

    torch.manual_seed(13)
    batch_size = 2
    num_heads = 4
    kv_lora_rank = 3
    rope_dim = 2
    topk = 4
    layer_config = SimpleNamespace(attn_dp=1, attn_cp=2, num_cores_per_device=2)
    module = SimpleNamespace(
        config=SimpleNamespace(kv_lora_rank=kv_lora_rank),
        scaling=(kv_lora_rank + rope_dim) ** -0.5,
        per_head_v_up=torch.randn(1, num_heads, kv_lora_rank, kv_lora_rank),
    )
    query_latent = torch.randn(batch_size, num_heads, 1, kv_lora_rank)
    query_rope = torch.randn(batch_size, num_heads, 1, rope_dim)
    sparse_ckv = torch.randn(batch_size, layer_config.attn_dp, layer_config.attn_cp, topk, kv_lora_rank)
    sparse_rope = torch.randn(batch_size, layer_config.attn_dp, layer_config.attn_cp, topk, rope_dim)
    row_valid = torch.ones(batch_size, layer_config.attn_dp, layer_config.attn_cp, topk, dtype=torch.bool)
    row_valid[:, :, 1, -1] = False

    actual, weights = _glm_tiled_sparse_mla_attention(
        module=module,
        query_latent=query_latent,
        query_rope=query_rope,
        sparse_ckv=sparse_ckv,
        sparse_rope=sparse_rope,
        row_valid=row_valid,
        layer_config=layer_config,
    )
    assert weights is None

    flat_ckv = sparse_ckv.reshape(batch_size, layer_config.attn_cp * topk, kv_lora_rank)
    flat_rope = sparse_rope.reshape(batch_size, layer_config.attn_cp * topk, rope_dim)
    valid = row_valid.reshape(batch_size, layer_config.attn_cp * topk)
    query = torch.cat((query_latent, query_rope), dim=-1)
    keys = torch.cat((flat_ckv, flat_rope), dim=-1)
    scores = torch.einsum("bhsd,btd->bhst", query, keys) * module.scaling
    scores = scores.masked_fill(~valid[:, None, None], -3.0e4)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_latent.dtype)
    latent = torch.einsum("bhst,btd->bhsd", probs, flat_ckv)
    expected = torch.matmul(latent, module.per_head_v_up[0]).transpose(1, 2).contiguous()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_glm_tiled_sparse_prefill_attention_matches_selected_softmax():
    from types import SimpleNamespace

    from QEfficient.blocking.glm_attention import _glm_tiled_sparse_mla_attention_prefill
    from QEfficient.transformers.cache_utils import glm_dsa_scatter_cache

    torch.manual_seed(29)
    batch_size, query_length, context_length = 2, 4, 8
    num_heads, kv_lora_rank, rope_dim = 4, 3, 2
    layer_config = SimpleNamespace(
        attn_dp=1,
        attn_cp=2,
        num_cores_per_device=2,
        sparse_q_block_size=4,
        sparse_kv_num_blocks=2,
    )
    module = SimpleNamespace(
        scaling=(kv_lora_rank + rope_dim) ** -0.5,
        per_head_v_up=torch.randn(1, num_heads, kv_lora_rank, kv_lora_rank),
    )
    logical_ckv = torch.randn(batch_size, context_length, kv_lora_rank)
    logical_rope = torch.randn(batch_size, context_length, rope_dim)
    positions = torch.arange(context_length).view(1, -1).expand(batch_size, -1)
    ckv_cache = torch.zeros(batch_size, 2, context_length // 2, kv_lora_rank)
    rope_cache = torch.zeros(batch_size, 2, context_length // 2, rope_dim)
    ckv_cache = glm_dsa_scatter_cache(ckv_cache, positions, logical_ckv, dp=1, cp=2)
    rope_cache = glm_dsa_scatter_cache(rope_cache, positions, logical_rope, dp=1, cp=2)
    query_latent = torch.randn(batch_size, num_heads, query_length, kv_lora_rank)
    query_rope = torch.randn(batch_size, num_heads, query_length, rope_dim)
    topk_indices = torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]], dtype=torch.int32).expand(
        batch_size, -1, -1
    )
    valid_topk = torch.ones_like(topk_indices, dtype=torch.bool)

    actual, weights = _glm_tiled_sparse_mla_attention_prefill(
        module=module,
        query_latent=query_latent,
        query_rope=query_rope,
        ckv_cache=ckv_cache,
        k_pe_cache=rope_cache,
        topk_indices=topk_indices,
        valid_topk=valid_topk,
        layer_config=layer_config,
    )
    assert weights is None

    selected_ckv = (
        logical_ckv.unsqueeze(1)
        .expand(-1, query_length, -1, -1)
        .gather(2, topk_indices.long().unsqueeze(-1).expand(-1, -1, -1, kv_lora_rank))
    )
    selected_rope = (
        logical_rope.unsqueeze(1)
        .expand(-1, query_length, -1, -1)
        .gather(2, topk_indices.long().unsqueeze(-1).expand(-1, -1, -1, rope_dim))
    )
    query = torch.cat((query_latent, query_rope), dim=-1).permute(0, 2, 1, 3)
    keys = torch.cat((selected_ckv, selected_rope), dim=-1)
    scores = torch.einsum("bqhd,bqkd->bqhk", query, keys) * module.scaling
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    latent = torch.einsum("bqhk,bqkr->bqhr", probs, selected_ckv)
    expected = torch.einsum("bqhr,hrv->bqhv", latent, module.per_head_v_up[0]).reshape(batch_size, query_length, -1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("prefill", "online"),
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_glm_parallel_mla_attention_matches_flat_absorbed_attention(prefill, online):
    from types import SimpleNamespace

    from QEfficient.blocking.glm_attention import _glm_parallel_mla_attention

    class FakeCompressedCache:
        def __init__(self, ckv, k_pe):
            self.layers = [SimpleNamespace(ckv=ckv, k_pe=k_pe)]

        def read_only_blocked_ckv(self, start_index, end_index, layer_idx, cache_kwargs):
            return self.layers[layer_idx].ckv[:, :, start_index:end_index]

        def read_only_blocked_k_pe(self, start_index, end_index, layer_idx, cache_kwargs):
            return self.layers[layer_idx].k_pe[:, :, start_index:end_index]

    torch.manual_seed(19)
    batch_size = 2
    num_heads = 4
    q_len = 3 if prefill else 1
    ctx_len = 7
    kv_lora_rank = 3
    rope_dim = 2
    query_width = kv_lora_rank + rope_dim
    module = SimpleNamespace(config=SimpleNamespace(kv_lora_rank=kv_lora_rank, qk_rope_head_dim=rope_dim))
    query = torch.randn(batch_size, num_heads, q_len, query_width)
    ckv = torch.randn(batch_size, 1, ctx_len, kv_lora_rank)
    k_pe = torch.randn(batch_size, 1, ctx_len, rope_dim)
    per_head_v_up = torch.randn(1, num_heads, kv_lora_rank, kv_lora_rank)
    position_ids = torch.arange(q_len, dtype=torch.long).view(1, q_len).expand(batch_size, -1)
    attention_mask = torch.arange(ctx_len).view(1, 1, 1, ctx_len) > position_ids[:, None, :, None]

    actual, weights = _glm_parallel_mla_attention(
        module=module,
        query=query,
        per_head_k_up_normal=torch.empty(0),
        per_head_v_up=per_head_v_up,
        attention_mask=attention_mask,
        scaling=query_width**-0.5,
        num_kv_blocks=2,
        par_num_split=2,
        cache_kwargs={"position_ids": position_ids},
        layer_idx=0,
        compressed_kvs=FakeCompressedCache(ckv, k_pe),
        absorption=True,
        prefill=prefill,
        online=online,
    )
    assert weights is None

    keys = torch.cat((ckv[:, 0], k_pe[:, 0]), dim=-1)
    scores = torch.einsum("bhsd,btd->bhst", query, keys) * (query_width**-0.5)
    scores = scores.masked_fill(attention_mask, -3.0e4)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    latent = torch.einsum("bhst,btd->bhsd", probs, ckv[:, 0])
    expected = torch.matmul(latent, per_head_v_up[0]).transpose(1, 2).contiguous()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
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
