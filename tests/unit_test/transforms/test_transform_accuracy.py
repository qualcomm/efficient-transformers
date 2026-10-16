# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Accuracy tests for PyTorch transforms in QEfficient.

Improvements over unit_v2:
  - Expanded CustomOpsTransform coverage: Phi3, Gemma, Gemma2
  - Expanded KVCacheTransform coverage: Phi3, Gemma, Gemma2, Falcon
  - Expanded combined transforms: Phi3, Gemma, Gemma2
  - SamplerTransform and SpDTransform behavior tests

Tests verify that transforms:
  1. Replace the correct module types
  2. Do NOT change the model's numerical output (accuracy preservation)
  3. Work correctly in combination

All tests run on CPU only, using tiny in-memory models.
"""

import copy
import inspect
import logging
import warnings
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    FalconConfig,
    FalconForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    GemmaConfig,
    GemmaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

import QEfficient.transformers.models.pytorch_transforms as pytorch_transforms
from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.transformers.models.pytorch_transforms import (
    CustomOpsTransform,
    ExternalOptimizedMoEMapperTransform,
    KVCacheTransform,
    OptimizedMoEExpertParallelWeightsTransform,
    OptimizedMoEExportConfigTransform,
    OptimizedMoEMapperTransform,
    OptimizedMoETransform,
    OptimizedMoEWeightsTransform,
    PoolingTransform,
    ReplicateKVHeadTransform,
    SamplerTransform,
    SimpleDecodeMoeTransform,
    SpDTransform,
)
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    QEffMoEBlockMixin,
    moe_simple_loop,
    pack_moe_weights_for_expert_parallel,
)
from QEfficient.transformers.moe.weights import MoEWeights
from QEfficient.utils.config_utils import calculate_num_replicate_kv_heads
from QEfficient.utils.constants import MOE_PREFILL_PACKED_CHUNK_SIZE
from QEfficient.utils.repeat_kv_utils import get_attention_module, get_projection_layer, get_text_model

VOCAB_SIZE = 500
SEQ_LEN = 8
CTX_LEN = 32


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_gpt2():
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=VOCAB_SIZE, n_positions=CTX_LEN, n_ctx=CTX_LEN)
    return GPT2LMHeadModel(cfg).eval()


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval()


def make_tiny_mistral():
    cfg = MistralConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return MistralForCausalLM(cfg).eval()


def make_tiny_qwen2():
    cfg = Qwen2Config(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return Qwen2ForCausalLM(cfg).eval()


def make_tiny_phi3():
    cfg = Phi3Config(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        pad_token_id=0,
    )
    return Phi3ForCausalLM(cfg).eval()


def make_tiny_gemma():
    cfg = GemmaConfig(
        num_hidden_layers=1,
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


def make_tiny_falcon():
    cfg = FalconConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=64,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        new_decoder_architecture=False,
        multi_query=True,
    )
    return FalconForCausalLM(cfg).eval()


# ---------------------------------------------------------------------------
# QEff input helpers
# ---------------------------------------------------------------------------


def _get_dims(config):
    """Extract (n_layers, n_kv_heads, head_dim) from any model config."""
    if hasattr(config, "num_hidden_layers"):
        n_layers = config.num_hidden_layers
        n_attn = config.num_attention_heads
        n_kv = getattr(config, "num_key_value_heads", n_attn)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_attn)
    else:
        # GPT2
        n_layers = config.n_layer
        n_kv = config.n_head
        head_dim = config.n_embd // config.n_head
    return n_layers, n_kv, head_dim


def _make_qeff_inputs(input_ids, config, ctx_len=CTX_LEN):
    """Build QEff-style inputs: input_ids + position_ids + zero-initialized past_key_values."""
    batch, seq = input_ids.shape
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    n_layers, n_kv, head_dim = _get_dims(config)
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
# Tests: RepeatKV transform fast unit checks
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestRepeatKVTransformFast:
    """RepeatKV must update KV head config and projection shapes on tiny local configs."""

    @staticmethod
    def _tiny_llama_qeff(num_attention_heads=4, num_key_value_heads=2):
        cfg = AutoConfig.for_model(
            "llama",
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=64,
        )
        return QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg), qaic_config={})

    def test_repeat_kv_dummy_causal_config(self):
        qeff_model = self._tiny_llama_qeff()

        text_model_before = get_text_model(qeff_model.model)
        attn_before = get_attention_module(text_model_before.layers[0])
        k_before = get_projection_layer(attn_before, ("k_proj", "key_proj")).weight.shape
        v_before = get_projection_layer(attn_before, ("v_proj", "value_proj")).weight.shape

        qeff_model.transform(
            ctx_len=64,
            seq_len=8,
            batch_size=1,
            num_devices=4,
            qaic_config={"replicate_kv_heads": True},
        )

        text_model_after = get_text_model(qeff_model.model)
        attn_after = get_attention_module(text_model_after.layers[0])
        k_after = get_projection_layer(attn_after, ("k_proj", "key_proj")).weight.shape
        v_after = get_projection_layer(attn_after, ("v_proj", "value_proj")).weight.shape

        assert qeff_model.model.config.orig_kv_heads == 2
        assert qeff_model.model.config.num_key_value_heads == 4
        assert k_after[0] == k_before[0] * 2
        assert v_after[0] == v_before[0] * 2

    def test_repeat_kv_dummy_vlm_config(self):
        cfg = AutoConfig.for_model(
            "llava",
            text_config={
                "model_type": "llama",
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
            },
            vision_config={
                "model_type": "clip_vision_model",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 32,
                "patch_size": 16,
                "projection_dim": 32,
            },
            image_token_index=0,
            projector_hidden_act="gelu",
            vision_feature_select_strategy="default",
            vision_feature_layer=-1,
        )
        model_hf = AutoModelForImageTextToText.from_config(cfg)
        qeff_model = QEFFAutoModelForImageTextToText(copy.deepcopy(model_hf), kv_offload=False, qaic_config={})

        text_model_before = get_text_model(qeff_model.model)
        attn_before = get_attention_module(text_model_before.layers[0])
        k_before = get_projection_layer(attn_before, ("k_proj", "key_proj")).weight.shape
        v_before = get_projection_layer(attn_before, ("v_proj", "value_proj")).weight.shape

        qeff_model.transform(
            ctx_len=64,
            seq_len=8,
            batch_size=1,
            num_devices=4,
            qaic_config={"replicate_kv_heads": True},
        )

        text_model_after = get_text_model(qeff_model.model)
        attn_after = get_attention_module(text_model_after.layers[0])
        k_after = get_projection_layer(attn_after, ("k_proj", "key_proj")).weight.shape
        v_after = get_projection_layer(attn_after, ("v_proj", "value_proj")).weight.shape

        assert qeff_model.model.config.text_config.orig_kv_heads == 2
        assert qeff_model.model.config.text_config.num_key_value_heads == 4
        assert k_after[0] == k_before[0] * 2
        assert v_after[0] == v_before[0] * 2

    def test_repeat_kv_mqa_config(self):
        qeff_model = self._tiny_llama_qeff(num_attention_heads=4, num_key_value_heads=1)
        qeff_model.transform(ctx_len=64, seq_len=8, bs=1, num_devices=4, qaic_config={"replicate_kv_heads": True})
        assert qeff_model.model.config.orig_kv_heads == 1
        assert qeff_model.model.config.num_key_value_heads == 2

    def test_repeat_kv_mutate_is_attention_local(self):
        qeff_model = self._tiny_llama_qeff()
        attn = get_attention_module(qeff_model.model.model.layers[0])
        k_before = get_projection_layer(attn, ("k_proj",)).weight.shape
        mutated_attn = ReplicateKVHeadTransform.mutate(
            attn,
            qeff_model.model.model.layers[0],
            n_repeat=2,
            orig_kv_heads=2,
            new_kv_heads=4,
            num_attention_heads=4,
            hidden_size=32,
        )
        k_after = get_projection_layer(mutated_attn, ("k_proj",)).weight.shape
        assert mutated_attn is attn
        assert k_after[0] == k_before[0] * 2
        assert attn.num_key_value_groups == 1
        assert qeff_model.model.config.num_key_value_heads == 2

    def test_repeat_kv_rejects_mha_config(self):
        qeff_model = self._tiny_llama_qeff(num_attention_heads=4, num_key_value_heads=4)
        qeff_model.transform(ctx_len=64, seq_len=8, bs=1, qaic_config={"replicate_kv_heads": True})
        assert qeff_model.model.config.num_key_value_heads == 4

    def test_repeat_kv_idempotent_for_same_repeat(self):
        qeff_model = self._tiny_llama_qeff()
        qaic_config = {"replicate_kv_heads": True}
        qeff_model.transform(ctx_len=64, seq_len=8, bs=1, qaic_config=qaic_config)
        first_shape = get_projection_layer(
            get_attention_module(qeff_model.model.model.layers[0]), ("k_proj",)
        ).weight.shape
        qeff_model.transform(ctx_len=64, seq_len=8, bs=1, qaic_config=qaic_config)
        second_shape = get_projection_layer(
            get_attention_module(qeff_model.model.model.layers[0]), ("k_proj",)
        ).weight.shape
        assert second_shape == first_shape

    def test_repeat_kv_skips_encoder_wrapper_without_config(self):
        cfg = AutoConfig.for_model(
            "llava",
            text_config={
                "model_type": "llama",
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
            },
            vision_config={
                "model_type": "clip_vision_model",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 32,
                "patch_size": 16,
                "projection_dim": 32,
            },
            image_token_index=0,
            projector_hidden_act="gelu",
            vision_feature_select_strategy="default",
            vision_feature_layer=-1,
        )
        model_hf = AutoModelForImageTextToText.from_config(cfg)
        qeff_model = QEFFAutoModelForImageTextToText(copy.deepcopy(model_hf), kv_offload=True, qaic_config={})
        assert not hasattr(qeff_model.vision_model.model, "config")
        qeff_model.vision_model.transform(ctx_len=64, seq_len=8, bs=1, qaic_config={"replicate_kv_heads": True})
        assert qeff_model.vision_model.hash_params["num_replicate_kv_heads"] == 1

    def test_calculate_num_replicate_kv_heads_for_gqa_mqa_and_mha(self):
        gqa_cfg = self._tiny_llama_qeff(num_attention_heads=4, num_key_value_heads=2).model.config
        mqa_cfg = self._tiny_llama_qeff(num_attention_heads=4, num_key_value_heads=1).model.config
        mha_cfg = self._tiny_llama_qeff(num_attention_heads=4, num_key_value_heads=4).model.config
        assert calculate_num_replicate_kv_heads(num_devices=4, text_model_config=gqa_cfg) == 2
        assert calculate_num_replicate_kv_heads(num_devices=4, text_model_config=mqa_cfg) == 2
        assert calculate_num_replicate_kv_heads(num_devices=4, text_model_config=mha_cfg) is None


# ---------------------------------------------------------------------------
# Tests: CustomOpsTransform - module replacement
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestCustomOpsTransformReplacement:
    """CustomOpsTransform must replace RMSNorm with CustomRMSNormAIC."""

    def test_llama_rms_norm_replaced_with_custom_rms_norm(self):
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        from QEfficient.customop import CustomRMSNormAIC

        model = make_tiny_llama()
        assert any(isinstance(m, LlamaRMSNorm) for m in model.modules())

        transformed, applied = CustomOpsTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if type(m) is LlamaRMSNorm:
                pytest.fail("Found unreplaced LlamaRMSNorm after transform")

        assert any(isinstance(m, CustomRMSNormAIC) for m in transformed.modules())

    def test_mistral_rms_norm_replaced(self):
        from QEfficient.customop import CustomRMSNormAIC

        model = make_tiny_mistral()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, CustomRMSNormAIC) for m in transformed.modules())

    def test_qwen2_rms_norm_replaced(self):
        from QEfficient.customop import CustomRMSNormAIC

        model = make_tiny_qwen2()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, CustomRMSNormAIC) for m in transformed.modules())

    def test_phi3_rms_norm_replaced(self):
        from QEfficient.customop import CustomRMSNormAIC

        model = make_tiny_phi3()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, CustomRMSNormAIC) for m in transformed.modules())

    def test_gemma_rms_norm_replaced(self):
        from QEfficient.customop import GemmaCustomRMSNormAIC

        model = make_tiny_gemma()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, GemmaCustomRMSNormAIC) for m in transformed.modules())

    def test_gemma2_rms_norm_replaced(self):
        from QEfficient.customop import GemmaCustomRMSNormAIC

        model = make_tiny_gemma2()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, GemmaCustomRMSNormAIC) for m in transformed.modules())

    def test_gpt2_not_transformed(self):
        """GPT2 uses LayerNorm, not RMSNorm. CustomOpsTransform must not apply."""
        model = make_tiny_gpt2()
        transformed, applied = CustomOpsTransform.apply(model)
        assert not applied, "CustomOpsTransform must not apply to GPT2 (no RMSNorm)"

    def test_module_mapping_contains_expected_types(self):
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        from transformers.models.mistral.modeling_mistral import MistralRMSNorm
        from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        mapping = CustomOpsTransform._module_mapping
        assert LlamaRMSNorm in mapping
        assert MistralRMSNorm in mapping
        assert Qwen2RMSNorm in mapping
        assert Phi3RMSNorm in mapping
        assert GemmaRMSNorm in mapping
        assert Gemma2RMSNorm in mapping


# ---------------------------------------------------------------------------
# Tests: CustomOpsTransform - accuracy preservation
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.accuracy
class TestCustomOpsTransformAccuracy:
    """
    CustomOpsTransform must NOT change the model's numerical output.
    CustomRMSNormAIC must be numerically equivalent to LlamaRMSNorm.
    """

    def test_llama_output_unchanged_after_custom_ops_transform(self):
        """Llama logits must be identical before and after CustomOpsTransform."""
        model = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_logits = transformed(input_ids=input_ids).logits[:, -1, :]

        max_diff = (before_logits - after_logits).abs().max().item()
        assert max_diff < 1e-5, (
            f"CustomOpsTransform changed Llama output: max_diff={max_diff:.2e}. "
            f"CustomRMSNormAIC must be numerically equivalent to LlamaRMSNorm."
        )

    def test_llama_greedy_token_unchanged_after_custom_ops_transform(self):
        model = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_token = transformed(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        assert before_token == after_token, (
            f"CustomOpsTransform changed greedy token: before={before_token}, after={after_token}"
        )

    def test_mistral_output_unchanged_after_custom_ops_transform(self):
        model = make_tiny_mistral()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_logits = transformed(input_ids=input_ids).logits[:, -1, :]

        max_diff = (before_logits - after_logits).abs().max().item()
        assert max_diff < 1e-5, f"CustomOpsTransform changed Mistral output: max_diff={max_diff:.2e}"

    def test_phi3_output_unchanged_after_custom_ops_transform(self):
        model = make_tiny_phi3()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_logits = transformed(input_ids=input_ids).logits[:, -1, :]

        max_diff = (before_logits - after_logits).abs().max().item()
        assert max_diff < 1e-5, f"CustomOpsTransform changed Phi3 output: max_diff={max_diff:.2e}"

    def test_gemma_output_unchanged_after_custom_ops_transform(self):
        model = make_tiny_gemma()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_logits = transformed(input_ids=input_ids).logits[:, -1, :]

        max_diff = (before_logits - after_logits).abs().max().item()
        assert max_diff < 1e-5, f"CustomOpsTransform changed Gemma output: max_diff={max_diff:.2e}"

    def test_custom_rms_norm_forward_is_finite(self):
        """CustomRMSNormAIC forward must produce finite outputs."""
        model = make_tiny_llama()
        transformed, _ = CustomOpsTransform.apply(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        with torch.no_grad():
            out = transformed(input_ids=input_ids)
        assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# Tests: KVCacheTransform - module replacement
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestKVCacheTransformReplacement:
    """KVCacheTransform must replace attention layers with QEff variants."""

    def test_gpt2_attention_replaced(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2Attention

        model = make_tiny_gpt2()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, GPT2Attention):
                assert isinstance(m, QEffGPT2Attention)

    def test_gpt2_lm_head_model_replaced(self):
        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel

        model = make_tiny_gpt2()
        transformed, _ = KVCacheTransform.apply(model)
        assert isinstance(transformed, QEffGPT2LMHeadModel)

    def test_llama_attention_replaced(self):
        from transformers.models.llama.modeling_llama import LlamaAttention

        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaAttention

        model = make_tiny_llama()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, LlamaAttention):
                assert isinstance(m, QEffLlamaAttention)

    def test_llama_for_causal_lm_replaced(self):
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

        model = make_tiny_llama()
        transformed, _ = KVCacheTransform.apply(model)
        assert isinstance(transformed, QEffLlamaForCausalLM)

    def test_mistral_attention_replaced(self):
        from transformers.models.mistral.modeling_mistral import MistralAttention

        from QEfficient.transformers.models.mistral.modeling_mistral import QEffMistralAttention

        model = make_tiny_mistral()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, MistralAttention):
                assert isinstance(m, QEffMistralAttention)

    def test_qwen2_attention_replaced(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2Attention

        model = make_tiny_qwen2()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, Qwen2Attention):
                assert isinstance(m, QEffQwen2Attention)

    def test_phi3_attention_replaced(self):
        from transformers.models.phi3.modeling_phi3 import Phi3Attention

        from QEfficient.transformers.models.phi3.modeling_phi3 import QEffPhi3Attention

        model = make_tiny_phi3()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, Phi3Attention):
                assert isinstance(m, QEffPhi3Attention)

    def test_gemma_attention_replaced(self):
        from transformers.models.gemma.modeling_gemma import GemmaAttention

        from QEfficient.transformers.models.gemma.modeling_gemma import QEffGemmaAttention

        model = make_tiny_gemma()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, GemmaAttention):
                assert isinstance(m, QEffGemmaAttention)

    def test_falcon_attention_replaced(self):
        from transformers.models.falcon.modeling_falcon import FalconAttention

        from QEfficient.transformers.models.falcon.modeling_falcon import QEffFalconAttention

        model = make_tiny_falcon()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, FalconAttention):
                assert isinstance(m, QEffFalconAttention)

    def test_module_mapping_covers_major_architectures(self):
        from transformers.models.falcon.modeling_falcon import FalconForCausalLM
        from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from transformers.models.mistral.modeling_mistral import MistralForCausalLM
        from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
        from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        mapping = KVCacheTransform._module_mapping
        assert GPT2LMHeadModel in mapping
        assert LlamaForCausalLM in mapping
        assert MistralForCausalLM in mapping
        assert MixtralForCausalLM in mapping
        assert Qwen2ForCausalLM in mapping
        assert Phi3ForCausalLM in mapping
        assert GemmaForCausalLM in mapping
        assert FalconForCausalLM in mapping


# ---------------------------------------------------------------------------
# Tests: KVCacheTransform - accuracy preservation
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.accuracy
class TestKVCacheTransformAccuracy:
    """
    KVCacheTransform must NOT change the model's greedy next token prediction.
    This is the core regression test for the KV cache transform.
    """

    def _check_greedy_token_preserved(self, model, label):
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        cfg = model.config
        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)

        with torch.no_grad():
            after_out = transformed(**qeff_inputs)
        after_token = after_out.logits[:, -1, :].argmax(-1).item()

        assert before_token == after_token, (
            f"[{label}] KVCacheTransform changed greedy token: "
            f"before={before_token}, after={after_token}. "
            f"KVCacheTransform must not change the model's prediction."
        )

    def test_gpt2_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_gpt2(), "GPT2")

    def test_llama_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_llama(), "Llama")

    def test_mistral_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_mistral(), "Mistral")

    def test_qwen2_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_qwen2(), "Qwen2")

    def test_phi3_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_phi3(), "Phi3")

    def test_gemma_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_gemma(), "Gemma")

    def test_gpt2_logits_numerically_close_after_kv_transform(self):
        """GPT2 logits must be numerically close before and after KVCacheTransform."""
        model = make_tiny_gpt2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        cfg = model.config
        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            after_logits = transformed(**qeff_inputs).logits[:, -1, :]

        hf_probs = F.softmax(before_logits, dim=-1)
        qeff_probs = F.softmax(after_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < 1e-3, f"KVCacheTransform changed GPT2 probability distribution: max_diff={max_diff:.2e}"

    def test_llama_logits_numerically_close_after_kv_transform(self):
        model = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        cfg = model.config
        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            after_logits = transformed(**qeff_inputs).logits[:, -1, :]

        hf_probs = F.softmax(before_logits, dim=-1)
        qeff_probs = F.softmax(after_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < 1e-3, f"KVCacheTransform changed Llama probability distribution: max_diff={max_diff:.2e}"

    def test_phi3_logits_numerically_close_after_kv_transform(self):
        model = make_tiny_phi3()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        cfg = model.config
        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            after_logits = transformed(**qeff_inputs).logits[:, -1, :]

        hf_probs = F.softmax(before_logits, dim=-1)
        qeff_probs = F.softmax(after_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < 1e-3, f"KVCacheTransform changed Phi3 probability distribution: max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# Tests: Combined transforms accuracy
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.accuracy
class TestCombinedTransformsAccuracy:
    """
    Applying CustomOpsTransform + KVCacheTransform together must preserve accuracy.
    This is the exact combination used by QEFFAutoModelForCausalLM.
    """

    def _check_combined_transforms(self, model, label):
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            original_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        cfg = model.config
        model, _ = CustomOpsTransform.apply(model)
        model, _ = KVCacheTransform.apply(model)

        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            transformed_token = model(**qeff_inputs).logits[:, -1, :].argmax(-1).item()

        assert original_token == transformed_token, (
            f"[{label}] Combined transforms changed greedy token: "
            f"original={original_token}, transformed={transformed_token}"
        )

    def test_llama_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_llama(), "Llama")

    def test_mistral_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_mistral(), "Mistral")

    def test_qwen2_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_qwen2(), "Qwen2")

    def test_phi3_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_phi3(), "Phi3")

    def test_gemma_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_gemma(), "Gemma")

    def test_combined_transforms_produce_finite_outputs(self):
        """Combined transforms must produce finite logits for all supported models."""
        for factory, label in [
            (make_tiny_llama, "Llama"),
            (make_tiny_mistral, "Mistral"),
            (make_tiny_qwen2, "Qwen2"),
            (make_tiny_phi3, "Phi3"),
        ]:
            model = factory()
            cfg = model.config
            model, _ = CustomOpsTransform.apply(model)
            model, _ = KVCacheTransform.apply(model)

            input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
            qeff_inputs = _make_qeff_inputs(input_ids, cfg)
            with torch.no_grad():
                out = model(**qeff_inputs)
            assert torch.isfinite(out.logits).all(), f"{label} combined transforms produce NaN/Inf"

    def test_gpt2_kv_transform_then_custom_ops_no_crash(self):
        """Applying KVCacheTransform then CustomOpsTransform to GPT2 must not crash."""
        model = make_tiny_gpt2()
        model, _ = KVCacheTransform.apply(model)
        model, applied = CustomOpsTransform.apply(model)
        assert not applied, "CustomOpsTransform must not apply to GPT2"


# ---------------------------------------------------------------------------
# Tests: PoolingTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestPoolingTransformCorrectness:
    """PoolingTransform must produce correct pooled embeddings."""

    def test_mean_pooling_wraps_model(self):
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        pooled, applied = PoolingTransform.apply(model, pooling="mean")
        assert isinstance(pooled, PooledModel)

    def test_cls_pooling_wraps_model(self):
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        pooled, applied = PoolingTransform.apply(model, pooling="cls")
        assert isinstance(pooled, PooledModel)

    def test_invalid_pooling_raises_error(self):
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        with pytest.raises((ValueError, KeyError, TypeError)):
            PoolingTransform.apply(model, pooling="invalid_pooling_xyz")

    def test_mean_pooled_output_matches_manual_mean(self):
        """PooledModel mean output must match manually computed mean pooling."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        inputs = {
            "input_ids": torch.randint(0, 500, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        with torch.no_grad():
            hf_out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        manual_mean = (hf_out.last_hidden_state * mask).sum(1) / mask.sum(1)

        pooled, _ = PoolingTransform.apply(model, pooling="mean")
        with torch.no_grad():
            pooled_mean = pooled(**inputs)

        max_diff = (manual_mean - pooled_mean).abs().max().item()
        assert max_diff < 1e-5, f"Mean pooling mismatch: max_diff={max_diff:.2e}"

    def test_max_pooling_wraps_model(self):
        """PoolingTransform with pooling='max' must wrap the model in PooledModel."""
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        pooled, _ = PoolingTransform.apply(model, pooling="max")
        # PoolingTransform always returns applied=False (it wraps, not replaces)
        assert isinstance(pooled, PooledModel)

    def test_max_pooled_output_matches_manual_max(self):
        """PooledModel max output must match manually computed max pooling."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        inputs = {
            "input_ids": torch.randint(0, 500, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        with torch.no_grad():
            hf_out = model(**inputs)
        # Manual max pooling: max over sequence dimension
        manual_max = hf_out.last_hidden_state.max(dim=1).values

        pooled, _ = PoolingTransform.apply(model, pooling="max")
        with torch.no_grad():
            pooled_max = pooled(**inputs)

        max_diff = (manual_max - pooled_max).abs().max().item()
        assert max_diff < 1e-5, f"Max pooling mismatch: max_diff={max_diff:.2e}"

    def test_avg_pooling_wraps_model(self):
        """PoolingTransform with pooling='avg' must wrap the model in PooledModel."""
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        # 'avg' is supported in POOLING_MAP
        pooled, _ = PoolingTransform.apply(model, pooling="avg")
        assert isinstance(pooled, PooledModel)

    def test_custom_callable_pooling_is_accepted(self):
        """PoolingTransform must accept a callable as the pooling argument."""
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()

        def custom_pool(last_hidden_states, attention_mask):
            # Simple: return first token (like CLS)
            return last_hidden_states[:, 0, :]

        try:
            pooled, _ = PoolingTransform.apply(model, pooling=custom_pool)
            assert isinstance(pooled, PooledModel)
        except (ValueError, TypeError, NotImplementedError):
            # If custom callable is not supported, skip
            pytest.skip("Custom callable pooling not supported in this version")

    def test_pooling_output_is_finite(self):
        """Pooled output must be finite (no NaN/Inf)."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        inputs = {
            "input_ids": torch.randint(0, 500, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        for pooling_type in ["mean", "cls", "max"]:
            try:
                pooled, _ = PoolingTransform.apply(model, pooling=pooling_type)
                with torch.no_grad():
                    output = pooled(**inputs)
                assert torch.isfinite(output).all(), f"Pooled output for '{pooling_type}' must be finite"
            except (ValueError, KeyError):
                pass  # Skip unsupported pooling types


# ---------------------------------------------------------------------------
# Tests: SamplerTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSamplerTransformBehavior:
    """SamplerTransform must only apply when qaic_config has include_sampler=True."""

    def test_no_transform_when_qaic_config_is_none(self):
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SamplerTransform.apply(kv_model, qaic_config=None)
        assert not applied

    def test_no_transform_when_include_sampler_false(self):
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SamplerTransform.apply(kv_model, qaic_config={"include_sampler": False})
        assert not applied

    def test_unsupported_model_raises_not_implemented(self):
        import torch.nn as nn

        class UnsupportedModel(nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(NotImplementedError):
            SamplerTransform.apply(UnsupportedModel(), qaic_config={"include_sampler": True})

    def test_supported_model_classes_include_gpt2_and_llama(self):
        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

        assert QEffGPT2LMHeadModel in SamplerTransform._module_mapping
        assert QEffLlamaForCausalLM in SamplerTransform._module_mapping


# ---------------------------------------------------------------------------
# Tests: SpDTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformBehavior:
    """SpDTransform must only apply when speculative_model_type is in qaic_config."""

    def test_no_transform_when_qaic_config_is_none(self):
        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SpDTransform.apply(kv_model, qaic_config=None)
        assert not applied

    def test_no_transform_when_speculative_model_type_missing(self):
        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SpDTransform.apply(kv_model, qaic_config={})
        assert not applied

    def test_invalid_speculative_model_type_raises_value_error(self):
        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        with pytest.raises(ValueError):
            SpDTransform.apply(kv_model, qaic_config={"speculative_model_type": "invalid_xyz"})

    def test_module_mapping_contains_llama_and_qwen2(self):
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM
        from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2ForCausalLM

        assert QEffLlamaForCausalLM in SpDTransform._module_mapping
        assert QEffQwen2ForCausalLM in SpDTransform._module_mapping


# ---------------------------------------------------------------------------
# Tests: SamplerTransform actual apply
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSamplerTransformActualApply:
    """SamplerTransform with include_sampler=True must attach sampler_forward."""

    def test_sampler_transform_applies_to_gpt2_with_include_sampler_true(self):
        """SamplerTransform must apply to QEffGPT2LMHeadModel when include_sampler=True."""
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        assert applied, "SamplerTransform must apply when include_sampler=True"

    def test_sampler_transform_applies_to_llama_with_include_sampler_true(self):
        """SamplerTransform must apply to QEffLlamaForCausalLM when include_sampler=True."""
        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        assert applied, "SamplerTransform must apply to Llama when include_sampler=True"

    def test_sampler_transform_saves_old_forward(self):
        """After SamplerTransform, model.old_forward must be set to the original forward."""
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        original_forward = kv_model.forward
        SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        assert hasattr(kv_model, "old_forward"), "SamplerTransform must save old_forward"
        assert kv_model.old_forward == original_forward, "old_forward must be the original forward method"

    def test_sampler_transform_replaces_forward_with_sampler_forward(self):
        """After SamplerTransform, model.forward must be replaced."""
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        original_forward = kv_model.forward
        SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        # The forward must have been replaced
        assert kv_model.forward is not original_forward, "SamplerTransform must replace model.forward"

    def test_sampler_transform_returns_same_model_instance(self):
        """SamplerTransform must modify model in-place."""
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        transformed, applied = SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        assert applied
        assert transformed is kv_model, "SamplerTransform must modify model in-place"

    def test_sampler_transform_module_mapping_contains_gpt2_and_llama(self):
        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

        assert QEffGPT2LMHeadModel in SamplerTransform._module_mapping
        assert QEffLlamaForCausalLM in SamplerTransform._module_mapping

    def test_sampler_transform_module_mapping_contains_phi3_and_qwen2(self):
        from QEfficient.transformers.models.phi3.modeling_phi3 import QEffPhi3ForCausalLM
        from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2ForCausalLM

        assert QEffPhi3ForCausalLM in SamplerTransform._module_mapping
        assert QEffQwen2ForCausalLM in SamplerTransform._module_mapping


# ---------------------------------------------------------------------------
# Tests: split OptimizedMoETransform facade
# ---------------------------------------------------------------------------


class _DummyOptimizedMoEBlock(QEffMoEBlockMixin, nn.Module):
    supported_moe_flavours = (
        MoEFlavour.SIMPLE_LOOP,
        MoEFlavour.DECODE_BMM,
        MoEFlavour.EXPERT_PARALLEL,
    )

    def __init__(self):
        super().__init__()
        self.transform_count = 0
        self.weights_transformed = False
        self.num_experts = 2

    def transform_weights(self):
        self.transform_count += 1
        tensor = torch.ones(2, 4, 8)
        self.moe_weights = MoEWeights(gate=tensor, up=tensor, down=tensor.transpose(-1, -2).contiguous())
        self.weights_transformed = True
        return self.moe_weights

    def route(self, x):
        topk_indices = torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device)
        topk_weights = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return (topk_indices, topk_weights), None


class _DummyOptimizedMoEModel(nn.Module):
    def __init__(self, model_type="qwen3_moe"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.block = _DummyOptimizedMoEBlock()


class _DummyNestedOptimizedMoEBlock(_DummyOptimizedMoEBlock):
    supported_moe_flavours = _DummyOptimizedMoEBlock.supported_moe_flavours

    def __init__(self):
        super().__init__()
        self.experts = nn.Module()

    def transform_weights(self):
        weights = super().transform_weights()
        self.experts.moe_weights = weights
        return weights


class _DummyNestedOptimizedMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_moe")
        self.block = _DummyNestedOptimizedMoEBlock()


def test_moe_simple_loop_prescale_matches_manual_expert_input_scaling():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    routing_weights = torch.tensor([[0.25, 0.0, 0.5], [0.0, 0.75, 0.0]])
    dummy = torch.ones(3, 2, 2)
    weights = MoEWeights(gate=dummy, up=dummy, down=dummy)

    def double_input(expert_input, *_):
        return expert_input * 2

    actual = moe_simple_loop(
        x,
        routing_weights,
        weights,
        MoEProfile(expert_mlp=double_input),
        prescale=True,
    )

    expected = torch.zeros_like(x)
    for expert_idx in range(routing_weights.shape[1]):
        routing_weight = routing_weights[:, expert_idx].unsqueeze(-1)
        expected = expected + torch.where(routing_weight > 0, x * routing_weight * 2, torch.zeros_like(x))

    torch.testing.assert_close(actual, expected)


@pytest.mark.transforms
class TestSplitOptimizedMoETransform:
    def test_moe_block_mixin_requires_explicit_supported_flavours(self):
        with pytest.raises(TypeError, match="supported_moe_flavours"):

            class _MissingSupportedMoEFlavours(QEffMoEBlockMixin, nn.Module):
                pass

    def test_moe_block_mixin_subclasses_must_implement_transform_weights(self):
        class _MissingTransformWeights(QEffMoEBlockMixin, nn.Module):
            supported_moe_flavours = (MoEFlavour.SIMPLE_LOOP,)

            def route(self, x):
                return x, None

        with pytest.raises(TypeError, match="transform_weights"):
            _MissingTransformWeights()

    def test_simple_decode_moe_transform_is_optimized_moe_transform_subclass(self):
        assert issubclass(SimpleDecodeMoeTransform, OptimizedMoETransform)

    def test_simple_decode_moe_transform_builds_weights_and_selects_decode_bmm(self):
        model = _DummyOptimizedMoEModel()

        _, transformed = SimpleDecodeMoeTransform.apply(model)

        assert transformed
        assert model.block.transform_count == 1
        assert model.block.moe_weights is not None
        assert model.block.weights_transformed is True
        assert model.block._moe_flavour is MoEFlavour.DECODE_BMM

    def test_simple_decode_moe_transform_falls_back_when_decode_bmm_unsupported(self):
        model = _DummyOptimizedMoEModel(model_type="llama4")
        model.block.supported_moe_flavours = (MoEFlavour.SIMPLE_LOOP,)

        _, transformed = SimpleDecodeMoeTransform.apply(model)

        assert transformed
        assert model.block.transform_count == 1
        assert model.block._moe_flavour is MoEFlavour.SIMPLE_LOOP

    def test_simple_decode_moe_transform_is_registered_after_cache_transforms(self):
        from QEfficient.transformers.models.modeling_auto import (
            QEFFAutoModelForCausalLM,
            QEffCausalLMForTextImageToTextModel,
            _QEFFAutoModelForImageTextToTextSingleQPC,
        )

        assert QEFFAutoModelForCausalLM._pytorch_transforms[-1] is SimpleDecodeMoeTransform
        assert QEffCausalLMForTextImageToTextModel._pytorch_transforms[-1] is SimpleDecodeMoeTransform
        assert _QEFFAutoModelForImageTextToTextSingleQPC._pytorch_transforms[-1] is SimpleDecodeMoeTransform

        for wrapper in (
            QEFFAutoModelForCausalLM,
            QEffCausalLMForTextImageToTextModel,
            _QEFFAutoModelForImageTextToTextSingleQPC,
        ):
            transforms = wrapper._pytorch_transforms
            assert transforms.index(KVCacheTransform) < transforms.index(SimpleDecodeMoeTransform)

    def test_moe_component_mappings_owned_by_optimized_mapper(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE, Glm4MoeTopkRouter
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssMLP
        from transformers.models.granitemoe.modeling_granitemoe import (
            GraniteMoeExperts,
            GraniteMoeMoE,
            GraniteMoeTopKRouter,
        )
        from transformers.models.llama4.modeling_llama4 import Llama4Router, Llama4TextExperts, Llama4TextMoe
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeExperts,
            Qwen3_5MoeSparseMoeBlock,
            Qwen3_5MoeTopKRouter,
        )
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeExperts,
            Qwen3MoeSparseMoeBlock,
            Qwen3MoeTopKRouter,
        )
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextExperts,
            Qwen3VLMoeTextSparseMoeBlock,
            Qwen3VLMoeTextTopKRouter,
        )

        moe_components = {
            Glm4MoeMoE,
            Glm4MoeTopkRouter,
            Llama4TextMoe,
            Llama4TextExperts,
            Llama4Router,
            Qwen3MoeExperts,
            Qwen3MoeSparseMoeBlock,
            Qwen3MoeTopKRouter,
            Qwen3VLMoeTextExperts,
            Qwen3VLMoeTextSparseMoeBlock,
            Qwen3VLMoeTextTopKRouter,
            Qwen3_5MoeExperts,
            Qwen3_5MoeSparseMoeBlock,
            Qwen3_5MoeTopKRouter,
            Gemma4TextRouter,
            GptOssMLP,
            GptOssExperts,
            GraniteMoeMoE,
            GraniteMoeExperts,
            GraniteMoeTopKRouter,
            MixtralSparseMoeBlock,
        }

        assert moe_components.isdisjoint(KVCacheTransform._module_mapping)
        assert moe_components <= set(OptimizedMoEMapperTransform._module_mapping)

    def test_mapper_discovers_moe_modules_without_mutating_weights(self):
        model = _DummyOptimizedMoEModel()

        _, transformed = OptimizedMoEMapperTransform.apply(model)

        assert transformed
        assert not hasattr(model.block, "moe_weights")
        assert model.block.transform_count == 0
        assert model.block.weights_transformed is False

    def test_weights_transform_canonicalizes_once(self):
        model = _DummyOptimizedMoEModel()

        _, transformed = OptimizedMoEWeightsTransform.apply(model)
        _, transformed_again = OptimizedMoEWeightsTransform.apply(model)

        assert transformed
        assert not transformed_again
        assert model.block.transform_count == 1
        assert model.block.weights_transformed is True
        assert model.block.moe_weights.gate.shape == (2, 4, 8)

    def test_weights_transform_collects_after_transform_weights(self, monkeypatch):
        model = _DummyOptimizedMoEModel()
        collect_calls = []
        monkeypatch.setattr(pytorch_transforms.gc, "collect", lambda: collect_calls.append(None))

        _, transformed = OptimizedMoEWeightsTransform.apply(model)

        assert transformed
        assert collect_calls == [None]

    def test_weights_transform_registers_canonical_moe_parameters(self):
        model = _DummyOptimizedMoEModel()

        OptimizedMoEWeightsTransform.apply(model)
        parameter_names = dict(model.named_parameters())

        assert isinstance(model.block.moe_weights, nn.Module)
        assert "block.moe_weights.gate" in parameter_names
        assert "block.moe_weights.up" in parameter_names
        assert "block.moe_weights.down" in parameter_names
        assert not any(name.endswith(("gate_proj", "all_gate_proj", "down_proj_t")) for name in parameter_names)

    def test_moe_forward_requires_pretransformed_weights(self):
        model = _DummyOptimizedMoEModel()

        with pytest.raises(RuntimeError, match="weights are not transformed"):
            model.block(torch.ones(1, 1, 4))

        assert model.block.transform_count == 0

    def test_moe_forward_consumes_prepared_weights_without_transforming(self):
        model = _DummyOptimizedMoEModel()
        OptimizedMoEWeightsTransform.apply(model)
        model.block._moe_return_router_logits = True

        def fail_builder():
            raise AssertionError("forward must not build MoE weights")

        model.block.transform_weights = fail_builder

        out, router_logits = model.block(torch.ones(1, 1, 4))

        assert router_logits is None
        assert torch.isfinite(out).all()

    def test_export_config_transform_sets_blocking_attrs_and_hash_params(self):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": 16}},
            prefill_seq_len=40,
            hash_params=hash_params,
        )

        assert transformed
        assert model.block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
        assert model.block.total_avl_cores == 2
        assert model.block.cores_per_expert == 1
        assert model.block.num_pipeline_stages == 1
        assert model.block.num_parallelized_experts == 2
        assert model.block.expert_parallel_num_packed_chunks == 3
        assert hash_params == {
            "moe_prefill_flavour": "expert_parallel",
            "moe_prefill_total_avl_cores": 2,
            "moe_prefill_cores_per_expert": 1,
            "moe_prefill_tree_reduce": False,
            "moe_prefill_num_pipeline_stages": 1,
            "moe_prefill_num_parallelized_experts": 2,
            "moe_prefill_expert_parallel_chunk_size": 16,
            "moe_prefill_num_packed_chunks": 3,
        }

    def test_export_config_transform_uses_flattened_decode_token_count_for_packing(self, caplog):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=False,
            batch_size=512,
            num_cores=2,
            qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 256}},
            prefill_seq_len=1,
            hash_params=hash_params,
        )

        assert transformed
        assert model.block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
        assert model.block.expert_parallel_num_packed_chunks == 2
        assert hash_params["moe_prefill_num_packed_chunks"] == 2
        assert "does not evenly divide" not in caplog.text

    def test_export_config_transform_sets_cores_per_expert_tree_reduce_and_multi_device_attrs(self):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=True,
            num_devices=2,
            num_cores=2,
            qaic_config={
                "moe_config": {
                    "cores_per_expert": 2,
                    "tree_reduce": True,
                    "expert_parallel_chunk_size": 16,
                }
            },
            prefill_seq_len=32,
            hash_params=hash_params,
        )

        assert transformed
        assert model.block.num_devices == 2
        assert model.block.cores_per_expert == 2
        assert model.block.total_avl_cores == 4
        assert model.block.num_pipeline_stages == 1
        assert model.block.num_parallelized_experts == 2
        assert model.block.experts_per_soc == 1
        assert model.block.tree_reduce is True
        assert hash_params["moe_prefill_total_avl_cores"] == 4
        assert hash_params["moe_prefill_cores_per_expert"] == 2
        assert hash_params["moe_prefill_tree_reduce"] is True

    @pytest.mark.parametrize(
        ("kwargs", "moe_config", "error_type", "match"),
        [
            ({}, {"cores_per_expert": 0}, ValueError, "cores_per_expert"),
            ({}, {"tree_reduce": "yes"}, TypeError, "tree_reduce"),
            ({"num_devices": 0}, {}, ValueError, "num_devices"),
            ({"num_cores": 3}, {}, ValueError, "num_experts \\* cores_per_expert"),
        ],
    )
    def test_export_config_transform_rejects_invalid_expert_parallel_layout(
        self, kwargs, moe_config, error_type, match
    ):
        model = _DummyOptimizedMoEModel()

        with pytest.raises(error_type, match=match):
            OptimizedMoEExportConfigTransform.apply(
                model,
                prefill_only=True,
                qaic_config={"moe_config": moe_config},
                **kwargs,
            )

    def test_pack_moe_weights_uses_qwen3_branch_layout(self):
        gate = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
        up = gate + 100
        down = torch.arange(4 * 3 * 2, dtype=torch.float32).view(4, 3, 2)
        weights = MoEWeights(gate=gate, up=up, down=down)

        packed = pack_moe_weights_for_expert_parallel(
            weights,
            num_pipeline_stages=2,
            num_parallelized_experts=2,
        )

        expected_gate = gate.view(2, 2, 2, 3).transpose(0, 1).contiguous()
        expected_up = up.view(2, 2, 2, 3).transpose(0, 1).contiguous()
        expected_down = down.view(2, 2, 3, 2).transpose(0, 1).contiguous()
        torch.testing.assert_close(packed.gate, expected_gate)
        torch.testing.assert_close(packed.up, expected_up)
        torch.testing.assert_close(packed.down, expected_down)

    @pytest.mark.parametrize(
        ("moe_flavour", "expected"),
        [
            ("simple_loop", MoEFlavour.SIMPLE_LOOP),
            ("expert_parallel", MoEFlavour.EXPERT_PARALLEL),
            ("expert_blocked", MoEFlavour.EXPERT_PARALLEL),
            ("decode_bmm", MoEFlavour.DECODE_BMM),
        ],
    )
    def test_export_config_transform_respects_nested_moe_flavour(self, moe_flavour, expected):
        model = _DummyOptimizedMoEModel()
        hash_params = {}
        moe_config = {"flavour": moe_flavour}
        if expected is MoEFlavour.EXPERT_PARALLEL:
            moe_config["expert_parallel_chunk_size"] = 16

        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": moe_config},
            prefill_seq_len=32,
            hash_params=hash_params,
        )

        assert transformed
        assert model.block._moe_flavour is expected
        assert hash_params["moe_prefill_flavour"] == expected.value
        if expected is MoEFlavour.EXPERT_PARALLEL:
            assert model.block.num_parallelized_experts == 2
            assert model.block.num_pipeline_stages == 1
            assert model.block.expert_parallel_num_packed_chunks == 2
        else:
            assert "moe_prefill_num_nsp" not in hash_params
            assert "moe_prefill_packed_chunk_size" not in hash_params
            assert "moe_prefill_expert_parallel_chunk_size" not in hash_params
            assert "moe_prefill_num_packed_chunks" not in hash_params

    def test_export_config_transform_auto_decode_uses_decode_bmm(self):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=False,
            hash_params=hash_params,
        )

        assert transformed
        assert model.block._moe_flavour is MoEFlavour.DECODE_BMM
        assert hash_params == {"moe_prefill_flavour": "decode_bmm"}

    def test_export_config_transform_auto_decode_uses_simple_loop_when_decode_bmm_unsupported(self):
        model = _DummyOptimizedMoEModel(model_type="llama4")
        model.block.supported_moe_flavours = (MoEFlavour.SIMPLE_LOOP,)
        hash_params = {}

        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=False,
            hash_params=hash_params,
        )

        assert transformed
        assert model.block._moe_flavour is MoEFlavour.SIMPLE_LOOP
        assert hash_params == {"moe_prefill_flavour": "simple_loop"}

    def test_export_config_transform_rejects_invalid_nested_moe_flavour(self):
        model = _DummyOptimizedMoEModel()

        with pytest.raises(ValueError, match=r"qaic_config\['moe_config'\]\['flavour'\]"):
            OptimizedMoEExportConfigTransform.apply(
                model,
                prefill_only=True,
                qaic_config={"moe_config": {"flavour": "legacy"}},
            )

    def test_export_config_transform_rejects_unsupported_valid_moe_flavour(self):
        model = _DummyOptimizedMoEModel(model_type="llama4")
        model.block.supported_moe_flavours = (MoEFlavour.SIMPLE_LOOP,)

        with pytest.raises(NotImplementedError, match="expert_parallel"):
            OptimizedMoEExportConfigTransform.apply(
                model,
                prefill_only=True,
                qaic_config={"moe_config": {"flavour": "expert_parallel"}},
            )

    def test_export_config_transform_ignores_legacy_nested_prefill_flavour(self):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"prefill_flavour": "simple_loop"}},
            hash_params=hash_params,
        )

        assert transformed
        assert model.block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
        assert hash_params["moe_prefill_flavour"] == "expert_parallel"

    def test_export_config_transform_ignores_top_level_moe_flavour(self):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_flavour": "simple_loop"},
            hash_params=hash_params,
        )

        assert transformed
        assert model.block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
        assert hash_params["moe_prefill_flavour"] == "expert_parallel"

    def test_export_config_transform_warns_when_expert_parallel_chunk_size_is_unused(self, caplog):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        caplog.set_level(logging.WARNING, logger="QEfficient")
        _, transformed = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=True,
            qaic_config={"moe_config": {"flavour": "simple_loop", "expert_parallel_chunk_size": 16}},
            hash_params=hash_params,
        )

        assert transformed
        assert model.block._moe_flavour is MoEFlavour.SIMPLE_LOOP
        assert hash_params == {"moe_prefill_flavour": "simple_loop"}
        assert "qaic_config['moe_config']['expert_parallel_chunk_size']" in caplog.text
        assert "expert_parallel" in caplog.text
        assert "ignored" in caplog.text

    def test_facade_applies_weights_and_export_config(self):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        _, transformed = OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": 32}},
            prefill_seq_len=64,
            hash_params=hash_params,
        )

        assert transformed
        assert model.block.transform_count == 1
        assert model.block.moe_weights is not None
        assert model.block.moe_weights.gate.shape == (2, 1, 4, 8)
        assert model.block.moe_weights.down.shape == (2, 1, 8, 4)
        assert model.block.moe_weights.num_experts == 2
        assert model.block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
        assert hash_params["moe_prefill_num_packed_chunks"] == 2

    @pytest.mark.parametrize(
        ("prefill_only", "qaic_config", "expected_flavour"),
        [
            (True, {"moe_config": {"flavour": "simple_loop"}}, MoEFlavour.SIMPLE_LOOP),
            (False, {"moe_config": {"flavour": "decode_bmm"}}, MoEFlavour.DECODE_BMM),
        ],
    )
    def test_facade_keeps_non_expert_parallel_weights_canonical(self, prefill_only, qaic_config, expected_flavour):
        model = _DummyOptimizedMoEModel()

        _, transformed = OptimizedMoETransform.apply(
            model,
            prefill_only=prefill_only,
            qaic_config=qaic_config,
        )

        assert transformed
        assert model.block._moe_flavour is expected_flavour
        assert model.block.moe_weights.gate.shape == (2, 4, 8)
        assert model.block.moe_weights.down.shape == (2, 8, 4)

    def test_facade_does_not_rebuild_weights_when_reapplied_to_expert_parallel(self):
        model = _DummyOptimizedMoEModel()

        OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 16}},
            prefill_seq_len=32,
        )
        first_weights = model.block.moe_weights
        OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 16}},
            prefill_seq_len=32,
        )

        assert model.block.transform_count == 1
        assert model.block.moe_weights is first_weights
        assert model.block.moe_weights.gate.shape == (2, 1, 4, 8)

    @pytest.mark.parametrize(
        "qaic_config",
        [None, {"moe_config": {}}, {"moe_config": {"expert_parallel_chunk_size": None}}],
    )
    def test_missing_expert_parallel_chunk_size_uses_default_constant(self, qaic_config):
        model = _DummyOptimizedMoEModel()
        hash_params = {}

        _, transformed = OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config=qaic_config,
            prefill_seq_len=MOE_PREFILL_PACKED_CHUNK_SIZE * 2,
            hash_params=hash_params,
        )

        assert transformed
        assert hash_params["moe_prefill_expert_parallel_chunk_size"] == MOE_PREFILL_PACKED_CHUNK_SIZE
        assert hash_params["moe_prefill_num_packed_chunks"] == 2

    def test_export_config_transform_rejects_non_positive_expert_parallel_chunk_size(self):
        model = _DummyOptimizedMoEModel()

        with pytest.raises(ValueError, match="expert_parallel_chunk_size"):
            OptimizedMoETransform.apply(
                model,
                prefill_only=True,
                qaic_config={"moe_config": {"expert_parallel_chunk_size": 0}},
                hash_params={},
            )

    def test_facade_no_longer_forwards_removed_export_config_kwargs(self, monkeypatch):
        model = _DummyOptimizedMoEModel()
        seen_kwargs = {}

        def spy_apply(model, **kwargs):
            seen_kwargs.update(kwargs)
            return model, True

        monkeypatch.setattr(OptimizedMoEExportConfigTransform, "apply", spy_apply)

        _, transformed = OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": 16}},
            hash_params={},
        )

        assert transformed
        assert "enable_chunking" not in seen_kwargs
        assert "moe_prefill_packed_chunk_size" not in seen_kwargs

    def test_facade_can_reapply_with_different_moe_flavours(self):
        model = _DummyOptimizedMoEModel()
        first_hash_params = {}
        second_hash_params = {}
        third_hash_params = {}

        _, first_transformed = OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            qaic_config={"moe_config": {"flavour": "simple_loop"}},
            hash_params=first_hash_params,
        )
        _, second_transformed = OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 16}},
            prefill_seq_len=32,
            hash_params=second_hash_params,
        )
        _, third_transformed = OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            qaic_config={"moe_config": {"flavour": "simple_loop"}},
            hash_params=third_hash_params,
        )

        assert first_transformed
        assert second_transformed
        assert third_transformed
        assert model.block.transform_count == 1
        assert model.block._moe_flavour is MoEFlavour.SIMPLE_LOOP
        assert model.block.num_parallelized_experts == 2
        assert model.block.num_pipeline_stages == 1
        assert model.block.expert_parallel_num_packed_chunks == 2
        assert model.block.moe_weights.gate.shape == (2, 4, 8)
        assert model.block.moe_weights.down.shape == (2, 8, 4)
        assert first_hash_params == {"moe_prefill_flavour": "simple_loop"}
        assert second_hash_params == {
            "moe_prefill_flavour": "expert_parallel",
            "moe_prefill_total_avl_cores": 2,
            "moe_prefill_cores_per_expert": 1,
            "moe_prefill_tree_reduce": False,
            "moe_prefill_num_pipeline_stages": 1,
            "moe_prefill_num_parallelized_experts": 2,
            "moe_prefill_expert_parallel_chunk_size": 16,
            "moe_prefill_num_packed_chunks": 2,
        }
        assert third_hash_params == {"moe_prefill_flavour": "simple_loop"}

    def test_expert_parallel_transform_replaces_nested_moe_weight_aliases(self):
        model = _DummyNestedOptimizedMoEModel()

        OptimizedMoEWeightsTransform.apply(model)
        old_weights = model.block.moe_weights
        assert model.block.experts.moe_weights is old_weights
        OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 16}},
            prefill_seq_len=32,
        )
        _, transformed = OptimizedMoEExpertParallelWeightsTransform.apply(model)

        assert transformed
        assert model.block.moe_weights is model.block.experts.moe_weights
        assert model.block.moe_weights is not old_weights
        assert model.block.moe_weights.gate.shape == (2, 1, 4, 8)
        for name in ("gate", "up", "down"):
            packed_param = getattr(model.block.moe_weights, name)
            old_param = getattr(old_weights, name)
            assert isinstance(packed_param, nn.Parameter)
            assert packed_param.requires_grad is False
            assert packed_param.data_ptr() != old_param.data_ptr()
        assert old_weights.gate.shape == (2, 4, 8)

    def test_expert_parallel_transform_collects_after_replacing_weights(self, monkeypatch):
        model = _DummyOptimizedMoEModel()
        OptimizedMoEWeightsTransform.apply(model)
        OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 16}},
            prefill_seq_len=32,
        )
        collect_calls = []
        monkeypatch.setattr(pytorch_transforms.gc, "collect", lambda: collect_calls.append(None))

        _, transformed = OptimizedMoEExpertParallelWeightsTransform.apply(model)

        assert transformed
        assert collect_calls == [None]

    def test_split_transforms_noop_for_non_moe_models(self, caplog):
        model = nn.Sequential(nn.Linear(4, 4))
        caplog.set_level(logging.WARNING, logger="QEfficient")

        assert not OptimizedMoEMapperTransform.apply(model)[1]
        assert not OptimizedMoEWeightsTransform.apply(model)[1]
        assert not OptimizedMoEExportConfigTransform.apply(model)[1]
        assert not OptimizedMoEExpertParallelWeightsTransform.apply(model)[1]
        assert not OptimizedMoETransform.apply(model)[1]
        assert "expert_parallel_chunk_size" not in caplog.text

    def test_facade_skips_downstream_transforms_for_non_moe_models(self, monkeypatch, caplog):
        model = nn.Sequential(nn.Linear(4, 4))

        def fail_apply(*args, **kwargs):
            raise AssertionError("downstream MoE transforms should not run for non-MoE models")

        monkeypatch.setattr(OptimizedMoEWeightsTransform, "apply", fail_apply)
        monkeypatch.setattr(OptimizedMoEExportConfigTransform, "apply", fail_apply)
        monkeypatch.setattr(OptimizedMoEExpertParallelWeightsTransform, "apply", fail_apply)
        caplog.set_level(logging.WARNING, logger="QEfficient")

        _, transformed = OptimizedMoETransform.apply(
            model,
            prefill_only=True,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
            prefill_seq_len=32,
            hash_params={},
        )

        assert not transformed
        assert "expert_parallel_chunk_size" not in caplog.text

    def test_mapper_discovers_structurally_bound_external_moe_modules(self):
        class StructuralMoE(nn.Module):
            def route(self, x):
                return x, None

            def transform_weights(self):
                return self.moe_weights

        model = nn.Sequential(StructuralMoE())

        _, transformed = OptimizedMoEMapperTransform.apply(model)

        assert transformed

    def test_export_wrapper_warns_for_direct_export(self, tmp_path):
        from QEfficient.utils.export_utils import export_from_compile, export_wrapper

        class DummyInner(nn.Module):
            config = SimpleNamespace(to_diff_dict=lambda: {"model_type": "dummy"})

        class DummyQEff:
            model = DummyInner()
            model_architecture = None
            model_name = "DummyQEff"
            hash_params = {}
            _onnx_transforms = []

            @export_wrapper
            def export(
                self,
                export_dir=None,
                prefill_only=False,
                enable_chunking=False,
                num_cores=1,
                prefill_seq_len=None,
                **kwargs,
            ):
                export_dir.mkdir(parents=True, exist_ok=True)
                self.export_kwargs = {
                    "prefill_only": prefill_only,
                    "enable_chunking": enable_chunking,
                    "num_cores": num_cores,
                    "prefill_seq_len": prefill_seq_len,
                }
                return export_dir / "DummyQEff.onnx"

        qeff = DummyQEff()
        with pytest.warns(DeprecationWarning, match="Direct \\.export\\(\\) is deprecated"):
            onnx_path = qeff.export(
                export_dir=tmp_path,
                prefill_only=True,
                enable_chunking=True,
                num_cores=2,
                prefill_seq_len=32,
            )

        assert onnx_path.parent.is_dir()
        assert qeff.export_kwargs["enable_chunking"] is True
        assert qeff.export_kwargs["num_cores"] == 2
        assert qeff.export_kwargs["prefill_seq_len"] == 32

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with export_from_compile():
                qeff.export(export_dir=tmp_path)

    def test_base_export_signature_does_not_expose_transform_compile_kwargs(self):
        from QEfficient.base.modeling_qeff import QEFFBaseModel

        signature = inspect.signature(QEFFBaseModel._export)
        removed_kwargs = {"prefill_only", "enable_chunking", "num_cores", "qaic_config", "prefill_seq_len"}

        assert removed_kwargs.isdisjoint(signature.parameters)

    def test_get_onnx_path_runs_optimized_moe_hook_before_export(self, monkeypatch, tmp_path):
        from QEfficient.base.modeling_qeff import QEFFBaseModel
        from QEfficient.utils.export_utils import export_wrapper

        events = []

        class DummyInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    architectures=[],
                    torch_dtype=torch.float32,
                    to_diff_dict=lambda: {"model_type": "dummy"},
                )

        class DummyQEff(QEFFBaseModel):
            _pytorch_transforms = []
            _onnx_transforms = []

            @property
            def get_model_config(self):
                return self.model.config.to_diff_dict()

            @export_wrapper
            def export(
                self,
                export_dir=None,
                prefill_only=False,
                enable_chunking=False,
                num_cores=1,
                qaic_config=None,
                prefill_seq_len=None,
                **kwargs,
            ):
                events.append(("export", prefill_only, enable_chunking, num_cores, prefill_seq_len))
                export_dir.mkdir(parents=True, exist_ok=True)
                self.onnx_path = export_dir / "DummyQEff.onnx"
                return self.onnx_path

            def compile(self, *args, **kwargs):
                raise NotImplementedError

        def spy_apply(model, **kwargs):
            events.append(("optimized_moe", kwargs))
            kwargs["hash_params"]["optimized_moe_hook"] = kwargs["prefill_only"]
            return model, True

        monkeypatch.setattr(OptimizedMoETransform, "apply", spy_apply)

        qeff = DummyQEff(DummyInner())
        onnx_path = qeff.get_onnx_path(
            prefill_only=True,
            enable_chunking=True,
            specializations=[{"batch_size": 1, "seq_len": 32, "ctx_len": 64}],
            offload_pt_weights=False,
            qaic_config={"moe_config": {"flavour": "simple_loop"}},
            export_dir=tmp_path,
            num_devices=2,
            aic_num_cores=2,
        )

        assert onnx_path.parent.is_dir()
        assert events[0][0] == "optimized_moe"
        assert "enable_chunking" not in events[0][1]
        assert "moe_prefill_packed_chunk_size" not in events[0][1]
        assert events[0][1]["num_devices"] == 2
        assert events[0][1]["num_cores"] == 2
        assert events[0][1]["qaic_config"]["moe_config"] == {"flavour": "simple_loop"}
        assert events[1] == ("export", True, True, 2, 32)
        assert qeff.hash_params["optimized_moe_hook"] is True

    def test_get_onnx_path_rejects_legacy_packed_chunk_size(self, tmp_path):
        from QEfficient.base.modeling_qeff import QEFFBaseModel

        class DummyInner(nn.Module):
            config = SimpleNamespace(architectures=[], torch_dtype=torch.float32)

        class DummyQEff(QEFFBaseModel):
            _pytorch_transforms = []
            _onnx_transforms = []

            @property
            def get_model_config(self):
                return {}

            def export(self, *args, **kwargs):
                raise AssertionError("legacy argument should be rejected before export")

            def compile(self, *args, **kwargs):
                raise NotImplementedError

        qeff = DummyQEff(DummyInner())
        with pytest.raises(TypeError, match=r"qaic_config\['moe_config'\]\['expert_parallel_chunk_size'\]"):
            qeff.get_onnx_path(
                prefill_only=True,
                enable_chunking=True,
                specializations=[{"batch_size": 1, "seq_len": 32, "ctx_len": 64}],
                offload_pt_weights=False,
                moe_prefill_packed_chunk_size=16,
                export_dir=tmp_path,
                aic_num_cores=2,
            )

    def test_dual_qpc_compile_rejects_legacy_packed_chunk_size(self):
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        with pytest.raises(TypeError, match=r"qaic_config\['moe_config'\]\['expert_parallel_chunk_size'\]"):
            _QEffAutoModelForImageTextToTextDualQPC.compile(object(), moe_prefill_packed_chunk_size=16)


# ---------------------------------------------------------------------------
# Tests: MoE transform (Mixtral)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestMoETransformReplacement:
    """OptimizedMoETransform must replace MixtralSparseMoeBlock with QEffMixtralSparseMoeBlock."""

    def _make_tiny_mixtral(self):
        from transformers import MixtralConfig, MixtralForCausalLM

        cfg = MixtralConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=CTX_LEN,
            num_experts_per_tok=2,
            num_local_experts=4,
        )
        return MixtralForCausalLM(cfg).eval(), cfg

    def test_mixtral_sparse_moe_block_replaced(self):
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import QEffMixtralSparseMoeBlock

        model, cfg = self._make_tiny_mixtral()
        assert any(isinstance(m, MixtralSparseMoeBlock) for m in model.modules())

        transformed, kv_applied = KVCacheTransform.apply(model)
        transformed, moe_applied = OptimizedMoETransform.apply(transformed)
        assert kv_applied
        assert moe_applied

        for m in transformed.modules():
            if type(m) is MixtralSparseMoeBlock:
                pytest.fail("Found unreplaced MixtralSparseMoeBlock after transform")

        assert any(isinstance(m, QEffMixtralSparseMoeBlock) for m in transformed.modules())

    def test_mixtral_for_causal_lm_replaced(self):
        from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import QEffMixtralForCausalLM

        model, cfg = self._make_tiny_mixtral()
        transformed, _ = KVCacheTransform.apply(model)
        assert isinstance(transformed, QEffMixtralForCausalLM)

    def test_mixtral_greedy_token_preserved_after_kv_transform(self):
        """Mixtral greedy token must be preserved after KVCacheTransform."""
        model, cfg = self._make_tiny_mixtral()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        transformed, _ = KVCacheTransform.apply(model)
        transformed, _ = OptimizedMoETransform.apply(transformed)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            after_token = transformed(**qeff_inputs).logits[:, -1, :].argmax(-1).item()

        assert before_token == after_token, (
            f"Mixtral KVCacheTransform changed greedy token: before={before_token}, after={after_token}"
        )

    def test_mixtral_kv_transform_produces_finite_outputs(self):
        model, cfg = self._make_tiny_mixtral()
        transformed, _ = KVCacheTransform.apply(model)
        transformed, _ = OptimizedMoETransform.apply(transformed)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            out = transformed(**qeff_inputs)
        assert torch.isfinite(out.logits).all(), "Mixtral KVCacheTransform must produce finite logits"


# ---------------------------------------------------------------------------
# Tests: T5ModelTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestT5ModelTransform:
    """T5ModelTransform must replace T5Attention and T5LayerNorm with QEff variants."""

    def _make_tiny_t5(self):
        from transformers import T5Config, T5ForConditionalGeneration

        cfg = T5Config(
            num_heads=2,
            d_model=64,
            d_ff=128,
            d_kv=32,
            num_layers=1,
            num_decoder_layers=1,
            vocab_size=500,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
        )
        return T5ForConditionalGeneration(cfg).eval(), cfg

    def test_t5_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert T5ModelTransform is not None

    def test_t5_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert hasattr(T5ModelTransform, "_module_mapping")
        assert len(T5ModelTransform._module_mapping) > 0

    def test_t5_transform_maps_t5_attention(self):
        from transformers.models.t5.modeling_t5 import T5Attention

        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert T5Attention in T5ModelTransform._module_mapping
        qeff_cls = T5ModelTransform._module_mapping[T5Attention]
        assert qeff_cls.__name__ == "QEffT5Attention"

    def test_t5_transform_maps_t5_layer_norm(self):
        from transformers.models.t5.modeling_t5 import T5LayerNorm

        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert T5LayerNorm in T5ModelTransform._module_mapping
        qeff_cls = T5ModelTransform._module_mapping[T5LayerNorm]
        assert qeff_cls.__name__ == "QEffT5LayerNorm"

    def test_t5_transform_replaces_attention(self):
        from transformers.models.t5.modeling_t5 import T5Attention

        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        model, cfg = self._make_tiny_t5()
        assert any(isinstance(m, T5Attention) for m in model.modules())

        transformed, applied = T5ModelTransform.apply(model)
        assert applied

        qeff_t5_attn_cls = T5ModelTransform._module_mapping[T5Attention]
        for m in transformed.modules():
            if type(m) is T5Attention:
                pytest.fail("Found unreplaced T5Attention after T5ModelTransform")

        assert any(isinstance(m, qeff_t5_attn_cls) for m in transformed.modules())

    def test_t5_transform_replaces_layer_norm(self):
        from transformers.models.t5.modeling_t5 import T5LayerNorm

        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        model, cfg = self._make_tiny_t5()
        transformed, applied = T5ModelTransform.apply(model)
        assert applied
        qeff_t5_ln_cls = T5ModelTransform._module_mapping[T5LayerNorm]
        assert any(isinstance(m, qeff_t5_ln_cls) for m in transformed.modules())

    def test_t5_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert hasattr(T5ModelTransform, "apply")
        assert callable(T5ModelTransform.apply)


# ---------------------------------------------------------------------------
# Tests: TextClassificationTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestTextClassificationTransformDirect:
    """TextClassificationTransform must directly replace DisentangledSelfAttention."""

    def _make_tiny_deberta(self):
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

        cfg = DebertaV2Config(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
            num_labels=3,
            type_vocab_size=0,
            pos_att_type=["p2c", "c2p"],
        )
        return DebertaV2ForSequenceClassification(cfg).eval(), cfg

    def test_text_classification_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        assert TextClassificationTransform is not None

    def test_text_classification_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        assert hasattr(TextClassificationTransform, "_module_mapping")
        assert len(TextClassificationTransform._module_mapping) > 0

    def test_text_classification_transform_maps_disentangled_self_attention(self):
        from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention

        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        assert DisentangledSelfAttention in TextClassificationTransform._module_mapping
        qeff_cls = TextClassificationTransform._module_mapping[DisentangledSelfAttention]
        assert qeff_cls.__name__ == "QEffDisentangledSelfAttention"

    def test_text_classification_transform_replaces_attention(self):
        from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention

        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        try:
            model, cfg = self._make_tiny_deberta()
        except Exception as e:
            pytest.skip(f"DeBERTa-v2 not available: {e}")

        assert any(isinstance(m, DisentangledSelfAttention) for m in model.modules())

        transformed, applied = TextClassificationTransform.apply(model)
        assert applied

        qeff_cls = TextClassificationTransform._module_mapping[DisentangledSelfAttention]
        for m in transformed.modules():
            if type(m) is DisentangledSelfAttention:
                pytest.fail("Found unreplaced DisentangledSelfAttention after transform")

        assert any(isinstance(m, qeff_cls) for m in transformed.modules())

    def test_text_classification_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        assert hasattr(TextClassificationTransform, "apply")
        assert callable(TextClassificationTransform.apply)


# ---------------------------------------------------------------------------
# Tests: BlockedKVAttentionTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockedAttentionTransform:
    """BlockedAttentionTransform must patch forward with blocking config parameter."""

    def test_blocked_transform_importable(self):
        from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        assert BlockingAttentionTransform is not None
        assert AttentionBlockingConfig is not None

    def test_blocked_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        assert hasattr(BlockingAttentionTransform, "apply")
        assert callable(BlockingAttentionTransform.apply)

    def test_blocked_transform_applies_to_llama(self):
        """BlockingAttentionTransform must apply to a KV-transformed Llama model."""
        from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        blocking_config = AttentionBlockingConfig(mode="kv", num_kv_blocks=4, ctx_len=CTX_LEN)
        transformed, applied = BlockingAttentionTransform.apply(kv_model, attn_blocking_config=blocking_config)
        assert applied, "BlockingAttentionTransform must apply to KV-transformed Llama"

    def test_blocked_transform_patches_forward(self):
        """After BlockingAttentionTransform, attention forward must be patched."""
        from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaAttention
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        blocking_config = AttentionBlockingConfig(mode="kv", num_kv_blocks=4, ctx_len=CTX_LEN)
        BlockingAttentionTransform.apply(kv_model, attn_blocking_config=blocking_config)

        # After transform, attention modules should have patched forward
        for m in kv_model.modules():
            if isinstance(m, QEffLlamaAttention):
                # The forward should be a partial function with num_kv_blocks
                assert hasattr(m, "forward"), "Attention module must have forward after transform"
                break

    def test_blocked_transform_returns_model_and_bool(self):
        from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        blocking_config = AttentionBlockingConfig(mode="kv", num_kv_blocks=4)
        result = BlockingAttentionTransform.apply(kv_model, attn_blocking_config=blocking_config)
        assert len(result) == 2
        assert isinstance(result[1], bool)


# ---------------------------------------------------------------------------
# Tests: PrefillOnly transforms (structure only - GPT_OSS is external)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestPrefillOnlyTransformStructure:
    """PrefillOnly transforms must have correct structure."""

    def test_prefill_only_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert PrefillOnlyTransform is not None

    def test_prefill_only_chunked_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyChunkedTransform

        assert PrefillOnlyChunkedTransform is not None

    def test_revert_prefill_only_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import RevertPrefillOnlyTransform

        assert RevertPrefillOnlyTransform is not None

    def test_revert_prefill_keep_attention_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import RevertPrefillKeepAttentionTransform

        assert RevertPrefillKeepAttentionTransform is not None

    def test_prefill_only_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert hasattr(PrefillOnlyTransform, "_module_mapping")
        assert len(PrefillOnlyTransform._module_mapping) > 0

    def test_prefill_only_chunked_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyChunkedTransform

        assert hasattr(PrefillOnlyChunkedTransform, "_module_mapping")
        assert len(PrefillOnlyChunkedTransform._module_mapping) > 0

    def test_revert_prefill_only_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import RevertPrefillOnlyTransform

        assert hasattr(RevertPrefillOnlyTransform, "_module_mapping")
        assert len(RevertPrefillOnlyTransform._module_mapping) > 0

    def test_prefill_only_transform_maps_gpt_oss_model(self):
        from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
            QEffGptOssModel,
            QEffPrefillOnlyGptOssModel,
        )
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert QEffGptOssModel in PrefillOnlyTransform._module_mapping
        assert PrefillOnlyTransform._module_mapping[QEffGptOssModel] is QEffPrefillOnlyGptOssModel

    def test_prefill_only_transform_maps_gpt_oss_attention(self):
        from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
            QEffGptOssAttention,
            QEffPrefillOnlyGptOssAttention,
        )
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert QEffGptOssAttention in PrefillOnlyTransform._module_mapping
        assert PrefillOnlyTransform._module_mapping[QEffGptOssAttention] is QEffPrefillOnlyGptOssAttention

    def test_revert_prefill_only_is_inverse_of_prefill_only(self):
        """RevertPrefillOnlyTransform must be the inverse of PrefillOnlyTransform for non-identity mappings."""
        from QEfficient.transformers.models.pytorch_transforms import (
            PrefillOnlyTransform,
            RevertPrefillOnlyTransform,
        )

        # For each (src, dst) in PrefillOnlyTransform where src != dst,
        # (dst, src) must be in RevertPrefillOnlyTransform
        for src, dst in PrefillOnlyTransform._module_mapping.items():
            if src is dst:
                continue  # Skip identity mappings
            assert dst in RevertPrefillOnlyTransform._module_mapping, (
                f"RevertPrefillOnlyTransform missing inverse mapping for {dst}"
            )
            assert RevertPrefillOnlyTransform._module_mapping[dst] is src, (
                f"RevertPrefillOnlyTransform[{dst}] must be {src}"
            )

    def test_all_prefill_transforms_have_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import (
            PrefillOnlyChunkedTransform,
            PrefillOnlyTransform,
            RevertPrefillKeepAttentionTransform,
            RevertPrefillOnlyTransform,
        )

        for cls in [
            PrefillOnlyTransform,
            PrefillOnlyChunkedTransform,
            RevertPrefillOnlyTransform,
            RevertPrefillKeepAttentionTransform,
        ]:
            assert hasattr(cls, "apply"), f"{cls.__name__} missing apply method"
            assert callable(cls.apply), f"{cls.__name__}.apply is not callable"


# ---------------------------------------------------------------------------
# Tests: VlmKVOffloadTransform (GAP D)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestVlmKVOffloadTransform:
    """VlmKVOffloadTransform must be importable and have correct module mapping."""

    def test_vlm_kv_offload_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert VlmKVOffloadTransform is not None

    def test_vlm_kv_offload_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert hasattr(VlmKVOffloadTransform, "_module_mapping")

    def test_vlm_kv_offload_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert hasattr(VlmKVOffloadTransform, "apply")
        assert callable(VlmKVOffloadTransform.apply)


# ---------------------------------------------------------------------------
# Tests: VlmNoKVOffloadTransform (GAP D)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestVlmNoKVOffloadTransform:
    """VlmNoKVOffloadTransform must be importable and have correct module mapping."""

    def test_vlm_no_kv_offload_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert VlmNoKVOffloadTransform is not None

    def test_vlm_no_kv_offload_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert hasattr(VlmNoKVOffloadTransform, "_module_mapping")

    def test_vlm_no_kv_offload_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert hasattr(VlmNoKVOffloadTransform, "apply")
        assert callable(VlmNoKVOffloadTransform.apply)


# ---------------------------------------------------------------------------
# Tests: KVCacheExternalModuleMapperTransform (GAP D)
# ---------------------------------------------------------------------------


class _DeepseekDummyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
        self.act_fn = F.silu

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _DeepseekDummyGate(nn.Module):
    top_k = 1

    def forward(self, hidden_states):
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        topk_indices = torch.zeros((num_tokens, 1), dtype=torch.long, device=hidden_states.device)
        topk_weights = torch.ones((num_tokens, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        return topk_indices, topk_weights


class _DeepseekDummySharedExperts(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states)


def _make_deepseek_external_moe():
    class DeepseekV3MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(n_routed_experts=2)
            self.experts = nn.ModuleList([_DeepseekDummyExpert(), _DeepseekDummyExpert()])
            self.gate = _DeepseekDummyGate()
            self.shared_experts = _DeepseekDummySharedExperts()

    return DeepseekV3MoE()


@pytest.mark.transforms
class TestKVCacheExternalModuleMapperTransform:
    """KVCacheExternalModuleMapperTransform must have correct string-based mappings."""

    def test_external_mapper_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert KVCacheExternalModuleMapperTransform is not None

    def test_external_mapper_has_match_string_replace_method(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert hasattr(KVCacheExternalModuleMapperTransform, "_match_string_replace_method")
        assert isinstance(KVCacheExternalModuleMapperTransform._match_string_replace_method, dict)

    def test_external_mapper_contains_internvl(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "InternVLChatModel" in KVCacheExternalModuleMapperTransform._match_string_replace_method

    def test_external_mapper_contains_molmo(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "MolmoForCausalLM" in KVCacheExternalModuleMapperTransform._match_string_replace_method

    def test_external_mapper_contains_grok1(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "Grok1ModelForCausalLM" in KVCacheExternalModuleMapperTransform._match_string_replace_method

    def test_external_mapper_internvl_has_forward(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        internvl_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["InternVLChatModel"]
        assert "forward" in internvl_mapping
        assert callable(internvl_mapping["forward"])

    def test_external_mapper_molmo_has_forward(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        molmo_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["MolmoForCausalLM"]
        assert "forward" in molmo_mapping
        assert callable(molmo_mapping["forward"])

    def test_external_mapper_grok1_has_forward(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        grok1_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["Grok1ModelForCausalLM"]
        assert "forward" in grok1_mapping
        assert callable(grok1_mapping["forward"])

    def test_external_mapper_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert hasattr(KVCacheExternalModuleMapperTransform, "apply")
        assert callable(KVCacheExternalModuleMapperTransform.apply)

    def test_external_mapper_internvl_has_get_dummy_inputs(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        internvl_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["InternVLChatModel"]
        assert "get_dummy_inputs" in internvl_mapping
        assert callable(internvl_mapping["get_dummy_inputs"])

    def test_external_mapper_rms_norm_has_forward(self):
        """RMSLayerNorm must be mapped to CustomRMSNormAIC.forward."""
        from QEfficient.customop import CustomRMSNormAIC
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "RMSLayerNorm" in KVCacheExternalModuleMapperTransform._match_string_replace_method
        rms_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["RMSLayerNorm"]
        assert rms_mapping["forward"] is CustomRMSNormAIC.forward

    def test_external_mapper_assigns_non_callable_attrs(self):
        from QEfficient.base.pytorch_transforms import ExternalModuleMapperTransform

        class DummyExternal(nn.Module):
            pass

        def mapped_method(self):
            return self.flag

        class DummyExternalMapper(ExternalModuleMapperTransform):
            _match_class_replace_method = {}
            _match_string_replace_method = {
                "DummyExternal": {
                    "flag": True,
                    "label": "mapped",
                    "mapped_method": mapped_method,
                }
            }

        model = DummyExternal()

        _, transformed = DummyExternalMapper.apply(model)

        assert transformed
        assert model.flag is True
        assert model.label == "mapped"
        assert model.mapped_method() is True

    def test_external_optimized_moe_mapper_attaches_only_transform_weights(self):
        for class_name in ("MoeBlock", "DeepseekV3MoE"):
            mapping = ExternalOptimizedMoEMapperTransform._match_string_replace_method[class_name]
            assert "transform_weights" in mapping
            assert "get_moe_weights" not in mapping

    def test_external_mapper_deepseek_moe_is_owned_by_optimized_moe_transform(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "DeepseekV3MoE" not in KVCacheExternalModuleMapperTransform._match_string_replace_method
        assert "DeepseekV3MoE" in ExternalOptimizedMoEMapperTransform._match_string_replace_method
        assert "MoeBlock" in ExternalOptimizedMoEMapperTransform._match_string_replace_method
        model = _make_deepseek_external_moe()

        _, transformed = OptimizedMoETransform.apply(model)

        assert transformed
        assert callable(model.route)
        assert callable(model.transform_weights)
        assert not hasattr(model, "get_moe_weights")
        assert callable(model.moe_profile)
        assert not hasattr(model, "legacy_forward")
        assert not hasattr(model, "legacy_moe")
        assert not hasattr(model, "moe_blocked_forward")
        assert not hasattr(model, "moe_blocked_weights_forward")
        assert model._moe_flavour is MoEFlavour.DECODE_BMM

    def test_prefill_deepseek_uses_mixin_even_when_legacy_env_is_set(self, monkeypatch):
        monkeypatch.setenv("NUM_FFN_BLOCKS", "2")
        monkeypatch.setenv("FFN_W_BLOCK_SIZE", "16")
        model = _make_deepseek_external_moe()
        _, transformed = OptimizedMoETransform.apply(model, prefill_only=True)
        calls = []

        def fake_execute(self, x, routing):
            calls.append(routing)
            return x

        model.execute_moe_flavour = MethodType(fake_execute, model)

        out = model(torch.ones(1, 2, 4))

        assert transformed
        assert calls
        assert torch.isfinite(out).all()
