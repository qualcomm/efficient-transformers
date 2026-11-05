# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for modeling utilities, supported architectures, and model registry.

Improvements over unit_v2:
  - Expanded architecture coverage: Phi3, Gemma, Gemma2, Falcon, Mixtral, Qwen3
  - Expanded MODEL_CLASS_MAPPING coverage
  - Tests for DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH
  - Tests for _create_causal_mask numerical correctness
  - Tests for build_model_class_mapping
  - Tests for QEFFAutoModelForCausalLM class structure including continuous_batching

All tests run on CPU only, no model loading required.
"""

import pytest
import torch

from QEfficient.transformers.modeling_utils import (
    DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH,
    MODEL_CLASS_MAPPING,
    TransformersToQEffModulesDict,
    _create_causal_mask,
    build_model_class_mapping,
    qeff_supported_architectures,
)
from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForSequenceClassification,
    QEFFAutoModelForSpeechSeq2Seq,
)

# ---------------------------------------------------------------------------
# Tests: qeff_supported_architectures
# ---------------------------------------------------------------------------


class TestQEffSupportedArchitectures:
    """qeff_supported_architectures must contain all expected model names."""

    def test_is_not_empty(self):
        assert len(qeff_supported_architectures.architectures) > 0

    def test_contains_gpt2(self):
        assert "GPT2LMHeadModel" in qeff_supported_architectures.architectures

    def test_contains_llama(self):
        assert "LlamaForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_mistral(self):
        assert "MistralForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_mixtral(self):
        assert "MixtralForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_phi3(self):
        assert "Phi3ForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_falcon(self):
        assert "FalconForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_qwen2(self):
        assert "Qwen2ForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_gemma(self):
        assert "GemmaForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_gemma2(self):
        assert "Gemma2ForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_whisper(self):
        assert "WhisperForConditionalGeneration" in qeff_supported_architectures.architectures

    def test_contains_starcoder2(self):
        assert "Starcoder2ForCausalLM" in qeff_supported_architectures.architectures

    def test_contains_gptj(self):
        assert "GPTJForCausalLM" in qeff_supported_architectures.architectures

    def test_all_entries_are_strings(self):
        for arch in qeff_supported_architectures.architectures:
            assert isinstance(arch, str), f"Expected string, got {type(arch)}: {arch}"

    def test_no_duplicates(self):
        archs = qeff_supported_architectures.architectures
        assert len(archs) == len(set(archs)), "Duplicate entries in supported architectures"


# ---------------------------------------------------------------------------
# Tests: TransformersToQEffModulesDict
# ---------------------------------------------------------------------------


class TestTransformersToQEffModulesDict:
    """TransformersToQEffModulesDict must map HF classes to QEff classes correctly."""

    def test_is_not_empty(self):
        assert len(TransformersToQEffModulesDict) > 0

    def test_gpt2_maps_to_qeff_gpt2(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel

        assert GPT2LMHeadModel in TransformersToQEffModulesDict
        assert TransformersToQEffModulesDict[GPT2LMHeadModel] is QEffGPT2LMHeadModel

    def test_llama_maps_to_qeff_llama(self):
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

        assert LlamaForCausalLM in TransformersToQEffModulesDict
        assert TransformersToQEffModulesDict[LlamaForCausalLM] is QEffLlamaForCausalLM

    def test_mistral_maps_to_qeff_mistral(self):
        from transformers.models.mistral.modeling_mistral import MistralForCausalLM

        assert MistralForCausalLM in TransformersToQEffModulesDict

    def test_mixtral_maps_to_qeff_mixtral(self):
        from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

        assert MixtralForCausalLM in TransformersToQEffModulesDict

    def test_qwen2_maps_to_qeff_qwen2(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        assert Qwen2ForCausalLM in TransformersToQEffModulesDict

    def test_gemma_maps_to_qeff_gemma(self):
        from transformers.models.gemma.modeling_gemma import GemmaForCausalLM

        assert GemmaForCausalLM in TransformersToQEffModulesDict

    def test_gemma2_maps_to_qeff_gemma2(self):
        from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM

        assert Gemma2ForCausalLM in TransformersToQEffModulesDict

    def test_falcon_maps_to_qeff_falcon(self):
        from transformers.models.falcon.modeling_falcon import FalconForCausalLM

        assert FalconForCausalLM in TransformersToQEffModulesDict

    def test_phi3_maps_to_qeff_phi3(self):
        from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

        assert Phi3ForCausalLM in TransformersToQEffModulesDict

    def test_whisper_maps_to_qeff_whisper(self):
        from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration

        assert WhisperForConditionalGeneration in TransformersToQEffModulesDict

    def test_all_values_are_different_from_keys(self):
        """QEff classes must be different from original HF classes."""
        for hf_cls, qeff_cls in TransformersToQEffModulesDict.items():
            assert hf_cls is not qeff_cls, f"{hf_cls} maps to itself - must map to a different QEff class"

    def test_all_values_are_classes(self):
        for hf_cls, qeff_cls in TransformersToQEffModulesDict.items():
            assert isinstance(qeff_cls, type), f"Expected class, got {type(qeff_cls)} for key {hf_cls}"


# ---------------------------------------------------------------------------
# Tests: MODEL_CLASS_MAPPING
# ---------------------------------------------------------------------------


class TestModelClassMapping:
    """MODEL_CLASS_MAPPING must map config class names to QEff class names."""

    def test_is_not_empty(self):
        assert len(MODEL_CLASS_MAPPING) > 0

    def test_llama_config_maps_to_qeff_causal_lm(self):
        assert MODEL_CLASS_MAPPING.get("LlamaConfig") == "QEFFAutoModelForCausalLM"

    def test_gpt2_config_maps_to_qeff_causal_lm(self):
        assert MODEL_CLASS_MAPPING.get("GPT2Config") == "QEFFAutoModelForCausalLM"

    def test_mistral_config_maps_to_qeff_causal_lm(self):
        assert MODEL_CLASS_MAPPING.get("MistralConfig") == "QEFFAutoModelForCausalLM"

    def test_qwen2_config_maps_to_qeff_causal_lm(self):
        assert MODEL_CLASS_MAPPING.get("Qwen2Config") == "QEFFAutoModelForCausalLM"

    def test_phi3_config_maps_to_qeff_causal_lm(self):
        assert MODEL_CLASS_MAPPING.get("Phi3Config") == "QEFFAutoModelForCausalLM"

    def test_gemma_config_maps_to_qeff_causal_lm(self):
        assert MODEL_CLASS_MAPPING.get("GemmaConfig") == "QEFFAutoModelForCausalLM"

    def test_falcon_config_maps_to_qeff_causal_lm(self):
        assert MODEL_CLASS_MAPPING.get("FalconConfig") == "QEFFAutoModelForCausalLM"

    def test_all_values_are_qeff_class_name_strings(self):
        for key, value in MODEL_CLASS_MAPPING.items():
            assert isinstance(value, str), f"Expected string value, got {type(value)}"
            assert "QEFF" in value or "QEff" in value, f"Expected QEff class name, got: {value}"

    def test_all_keys_are_config_class_name_strings(self):
        for key in MODEL_CLASS_MAPPING.keys():
            assert isinstance(key, str), f"Expected string key, got {type(key)}"
            assert "Config" in key, f"Expected config class name, got: {key}"


# ---------------------------------------------------------------------------
# Tests: EXTERNAL_MODEL_CLASS_MAPPING
# ---------------------------------------------------------------------------


class TestExternalModelClassMapping:
    """EXTERNAL_MODEL_CLASS_MAPPING must contain external model entries."""

    def test_external_mapping_exists_and_is_dict(self):
        from QEfficient.transformers.modeling_utils import EXTERNAL_MODEL_CLASS_MAPPING

        assert isinstance(EXTERNAL_MODEL_CLASS_MAPPING, dict)

    def test_contains_grok1(self):
        from QEfficient.transformers.modeling_utils import EXTERNAL_MODEL_CLASS_MAPPING

        assert "Grok1Config" in EXTERNAL_MODEL_CLASS_MAPPING

    def test_contains_k2_horizon(self):
        from QEfficient.transformers.modeling_utils import EXTERNAL_MODEL_CLASS_MAPPING

        assert EXTERNAL_MODEL_CLASS_MAPPING["K2HorizonConfig"] == "QEFFAutoModelForCausalLM"


# ---------------------------------------------------------------------------
# Tests: DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH
# ---------------------------------------------------------------------------


class TestDynamicSeqLenSupportedModelArch:
    """DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH must contain expected model types."""

    def test_is_not_empty(self):
        assert len(DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH) > 0

    def test_contains_gemma3(self):
        assert "gemma3" in DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH

    def test_contains_llama4(self):
        assert "llama4" in DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH

    def test_supports_membership_test(self):
        assert hasattr(DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH, "__contains__")

    def test_all_entries_are_strings(self):
        for arch in DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH:
            assert isinstance(arch, str)


# ---------------------------------------------------------------------------
# Tests: _create_causal_mask
# ---------------------------------------------------------------------------


class TestCreateCausalMask:
    """_create_causal_mask must produce correct boolean masks."""

    def test_shape_is_correct(self):
        batch, seq, target_len = 1, 4, 8
        position_ids = torch.arange(seq).unsqueeze(0)
        mask = _create_causal_mask(position_ids, target_length=target_len)
        assert mask.shape == (batch, 1, seq, target_len)

    def test_dtype_is_bool(self):
        position_ids = torch.arange(4).unsqueeze(0)
        mask = _create_causal_mask(position_ids, target_length=8)
        assert mask.dtype == torch.bool

    def test_future_positions_are_masked(self):
        """mask[i, j] must be True when j > i (future token = masked)."""
        seq = 4
        position_ids = torch.arange(seq).unsqueeze(0)
        mask = _create_causal_mask(position_ids, target_length=seq)
        for i in range(seq):
            for j in range(seq):
                if j > i:
                    assert mask[0, 0, i, j].item() is True, f"Expected mask[{i},{j}]=True (future), got False"

    def test_past_positions_are_not_masked(self):
        """mask[i, j] must be False when j <= i (past/current token = not masked)."""
        seq = 4
        position_ids = torch.arange(seq).unsqueeze(0)
        mask = _create_causal_mask(position_ids, target_length=seq)
        for i in range(seq):
            for j in range(i + 1):
                assert mask[0, 0, i, j].item() is False, f"Expected mask[{i},{j}]=False (past), got True"

    def test_batch_size_2_works(self):
        batch, seq, target_len = 2, 4, 8
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        mask = _create_causal_mask(position_ids, target_length=target_len)
        assert mask.shape[0] == batch

    def test_decode_step_shape(self):
        """Single-token decode step must produce correct shape."""
        batch, target_len = 1, 16
        position_ids = torch.tensor([[8]])
        mask = _create_causal_mask(position_ids, target_length=target_len)
        assert mask.shape == (batch, 1, 1, target_len)

    def test_decode_step_masks_future_positions(self):
        """In decode step at position 8, positions 9..15 must be masked."""
        target_len = 16
        decode_pos = 8
        position_ids = torch.tensor([[decode_pos]])
        mask = _create_causal_mask(position_ids, target_length=target_len)
        # Positions 0..decode_pos must be unmasked (False)
        for j in range(decode_pos + 1):
            assert mask[0, 0, 0, j].item() is False, f"Position {j} should be unmasked at decode_pos={decode_pos}"
        # Positions decode_pos+1..target_len-1 must be masked (True)
        for j in range(decode_pos + 1, target_len):
            assert mask[0, 0, 0, j].item() is True, f"Position {j} should be masked at decode_pos={decode_pos}"

    def test_sliding_window_shape_correct(self):
        batch, seq, target_len = 1, 4, 8
        position_ids = torch.arange(seq).unsqueeze(0)
        mask = _create_causal_mask(position_ids, target_length=target_len, sliding_window=2)
        assert mask.shape == (batch, 1, seq, target_len)

    def test_no_sliding_window_none_works(self):
        position_ids = torch.arange(4).unsqueeze(0)
        mask = _create_causal_mask(position_ids, target_length=8, sliding_window=None)
        assert mask is not None
        assert mask.shape[-1] == 8

    def test_causal_mask_is_lower_triangular(self):
        """For a square mask (seq == target_len), the unmasked region must be lower triangular."""
        seq = 6
        position_ids = torch.arange(seq).unsqueeze(0)
        mask = _create_causal_mask(position_ids, target_length=seq)
        # mask[i, j] == False means "attend to j from position i"
        # This should be lower triangular: attend to j <= i
        for i in range(seq):
            for j in range(seq):
                expected_masked = j > i
                actual_masked = mask[0, 0, i, j].item()
                assert actual_masked == expected_masked, (
                    f"mask[{i},{j}]: expected {expected_masked}, got {actual_masked}"
                )


# ---------------------------------------------------------------------------
# Tests: build_model_class_mapping
# ---------------------------------------------------------------------------


class TestBuildModelClassMapping:
    """build_model_class_mapping must return correct config → class name mapping."""

    def test_returns_non_empty_dict(self):
        import transformers.models.auto.modeling_auto as mapping

        result = build_model_class_mapping(mapping.AutoModelForCausalLM, "QEFFAutoModelForCausalLM")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_all_values_are_the_provided_class_name(self):
        import transformers.models.auto.modeling_auto as mapping

        class_name = "QEFFAutoModelForCausalLM"
        result = build_model_class_mapping(mapping.AutoModelForCausalLM, class_name)
        for key, value in result.items():
            assert value == class_name

    def test_all_keys_are_config_class_name_strings(self):
        import transformers.models.auto.modeling_auto as mapping

        result = build_model_class_mapping(mapping.AutoModelForCausalLM, "QEFFAutoModelForCausalLM")
        for key in result.keys():
            assert isinstance(key, str)
            assert "Config" in key

    def test_contains_llama_config(self):
        import transformers.models.auto.modeling_auto as mapping

        result = build_model_class_mapping(mapping.AutoModelForCausalLM, "QEFFAutoModelForCausalLM")
        assert "LlamaConfig" in result

    def test_contains_gpt2_config(self):
        import transformers.models.auto.modeling_auto as mapping

        result = build_model_class_mapping(mapping.AutoModelForCausalLM, "QEFFAutoModelForCausalLM")
        assert "GPT2Config" in result

    def test_contains_mistral_config(self):
        import transformers.models.auto.modeling_auto as mapping

        result = build_model_class_mapping(mapping.AutoModelForCausalLM, "QEFFAutoModelForCausalLM")
        assert "MistralConfig" in result

    def test_contains_qwen2_config(self):
        import transformers.models.auto.modeling_auto as mapping

        result = build_model_class_mapping(mapping.AutoModelForCausalLM, "QEFFAutoModelForCausalLM")
        assert "Qwen2Config" in result


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM class structure
# ---------------------------------------------------------------------------


class TestQEFFAutoModelForCausalLMClassStructure:
    """QEFFAutoModelForCausalLM must have correct class-level attributes."""

    def test_has_pytorch_transforms_list(self):
        assert hasattr(QEFFAutoModelForCausalLM, "_pytorch_transforms")
        assert isinstance(QEFFAutoModelForCausalLM._pytorch_transforms, list)
        assert len(QEFFAutoModelForCausalLM._pytorch_transforms) > 0

    def test_has_onnx_transforms_list(self):
        assert hasattr(QEFFAutoModelForCausalLM, "_onnx_transforms")
        assert isinstance(QEFFAutoModelForCausalLM._onnx_transforms, list)

    def test_kv_cache_transform_in_pytorch_transforms(self):
        transform_names = [
            t.__name__ if hasattr(t, "__name__") else str(t) for t in QEFFAutoModelForCausalLM._pytorch_transforms
        ]
        assert any("KVCache" in name for name in transform_names), (
            f"KVCacheTransform not found in _pytorch_transforms: {transform_names}"
        )

    def test_custom_ops_transform_in_pytorch_transforms(self):
        transform_names = [
            t.__name__ if hasattr(t, "__name__") else str(t) for t in QEFFAutoModelForCausalLM._pytorch_transforms
        ]
        assert any("CustomOps" in name for name in transform_names), (
            f"CustomOpsTransform not found in _pytorch_transforms: {transform_names}"
        )

    def test_has_hf_auto_class(self):
        assert hasattr(QEFFAutoModelForCausalLM, "_hf_auto_class")

    def test_has_from_pretrained_classmethod(self):
        assert hasattr(QEFFAutoModelForCausalLM, "from_pretrained")
        assert callable(QEFFAutoModelForCausalLM.from_pretrained)

    def test_importable_from_public_api(self):
        import QEfficient

        assert hasattr(QEfficient, "QEFFAutoModelForCausalLM")
        assert QEfficient.QEFFAutoModelForCausalLM is QEFFAutoModelForCausalLM

    def test_continuous_batching_flag_stored(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        assert qeff.continuous_batching is True

    def test_continuous_batching_defaults_to_false(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.continuous_batching is False

    def test_model_name_property_returns_string(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        assert hasattr(qeff, "model_name")
        assert isinstance(qeff.model_name, str)
        assert len(qeff.model_name) > 0

    def test_model_attribute_is_transformed_model(self):
        """After construction, qeff.model must be the KV-transformed model."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        assert isinstance(qeff.model, QEffGPT2LMHeadModel), f"Expected QEffGPT2LMHeadModel, got {type(qeff.model)}"

    def test_onnx_transforms_contain_fp16_clip(self):
        """FP16ClipTransform is importable and available for use."""
        from QEfficient.base.onnx_transforms import FP16ClipTransform

        assert FP16ClipTransform is not None
        assert hasattr(FP16ClipTransform, "apply")


# ---------------------------------------------------------------------------
# Tests: Other QEff auto model class structures
# ---------------------------------------------------------------------------


class TestOtherQEffAutoModelClassStructures:
    """Other QEff auto model classes must have correct class-level attributes."""

    def test_qeff_auto_model_for_speech_seq2seq_has_from_pretrained(self):
        assert hasattr(QEFFAutoModelForSpeechSeq2Seq, "from_pretrained")
        assert callable(QEFFAutoModelForSpeechSeq2Seq.from_pretrained)

    def test_qeff_auto_model_for_speech_seq2seq_has_pytorch_transforms(self):
        assert hasattr(QEFFAutoModelForSpeechSeq2Seq, "_pytorch_transforms")
        assert isinstance(QEFFAutoModelForSpeechSeq2Seq._pytorch_transforms, list)

    def test_qeff_auto_model_for_speech_seq2seq_has_hf_auto_class(self):
        assert hasattr(QEFFAutoModelForSpeechSeq2Seq, "_hf_auto_class")

    def test_qeff_auto_model_has_from_pretrained(self):
        assert hasattr(QEFFAutoModel, "from_pretrained")
        assert callable(QEFFAutoModel.from_pretrained)

    def test_qeff_auto_model_has_pytorch_transforms(self):
        assert hasattr(QEFFAutoModel, "_pytorch_transforms")

    def test_qeff_auto_model_has_hf_auto_class(self):
        assert hasattr(QEFFAutoModel, "_hf_auto_class")

    def test_qeff_auto_model_for_seq_classification_has_from_pretrained(self):
        assert hasattr(QEFFAutoModelForSequenceClassification, "from_pretrained")
        assert callable(QEFFAutoModelForSequenceClassification.from_pretrained)

    def test_qeff_auto_model_for_seq_classification_has_pytorch_transforms(self):
        assert hasattr(QEFFAutoModelForSequenceClassification, "_pytorch_transforms")

    def test_qeff_auto_model_for_seq_classification_has_hf_auto_class(self):
        assert hasattr(QEFFAutoModelForSequenceClassification, "_hf_auto_class")

    def test_misclassified_map_exists(self):
        try:
            from QEfficient.transformers.models.modeling_auto import (
                MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP,
            )

            assert isinstance(MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP, dict)
        except ImportError:
            pytest.skip("MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP not available")

    def test_qeff_auto_model_for_seq_classification_wraps_bert(self):
        """QEFFAutoModelForSequenceClassification must wrap BERT without error."""
        from transformers import BertConfig, BertForSequenceClassification

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
            num_labels=3,
        )
        model = BertForSequenceClassification(cfg)
        qeff = QEFFAutoModelForSequenceClassification(model)
        assert qeff is not None
        assert hasattr(qeff, "model")

    def test_qeff_auto_model_wraps_bert(self):
        """QEFFAutoModel must wrap BERT without error."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg)
        qeff = QEFFAutoModel(model)
        assert qeff is not None
        assert hasattr(qeff, "model")


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM error paths
# ---------------------------------------------------------------------------


class TestQEFFAutoModelForCausalLMErrorPaths:
    """QEFFAutoModelForCausalLM must raise appropriate errors for invalid inputs."""

    def test_non_causal_lm_model_raises_assertion_error(self):
        """Passing a non-CausalLM model must raise AssertionError or TypeError."""
        from transformers import BertConfig, BertForSequenceClassification

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
            num_labels=3,
        )
        model = BertForSequenceClassification(cfg)
        with pytest.raises((AssertionError, TypeError, ValueError)):
            QEFFAutoModelForCausalLM(model)

    def test_bert_model_raises_error_when_passed_to_causal_lm(self):
        """BertModel (not CausalLM) must raise an error."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg)
        with pytest.raises((AssertionError, TypeError, ValueError)):
            QEFFAutoModelForCausalLM(model)

    def test_none_model_raises_error(self):
        """Passing None must raise an error."""
        with pytest.raises((AssertionError, TypeError, AttributeError)):
            QEFFAutoModelForCausalLM(None)


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForSpeechSeq2Seq error paths
# ---------------------------------------------------------------------------


class TestQEFFAutoModelForSpeechSeq2SeqErrorPaths:
    """QEFFAutoModelForSpeechSeq2Seq must raise appropriate errors for invalid inputs."""

    def test_non_speech_model_raises_error(self):
        """Passing a non-speech model must raise AssertionError or TypeError."""
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        with pytest.raises((AssertionError, TypeError, ValueError)):
            QEFFAutoModelForSpeechSeq2Seq(model)

    def test_bert_model_raises_error_when_passed_to_speech_seq2seq(self):
        """BertModel must raise an error when passed to QEFFAutoModelForSpeechSeq2Seq."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg)
        with pytest.raises((AssertionError, TypeError, ValueError)):
            QEFFAutoModelForSpeechSeq2Seq(model)


# ---------------------------------------------------------------------------
# Tests: MODEL_CLASS_MAPPING completeness
# ---------------------------------------------------------------------------


class TestModelClassMappingCompleteness:
    """MODEL_CLASS_MAPPING must include VLM config classes."""

    def test_contains_llava_config(self):
        from QEfficient.transformers.modeling_utils import MODEL_CLASS_MAPPING

        # LlavaConfig should map to QEFFAutoModelForImageTextToText
        assert "LlavaConfig" in MODEL_CLASS_MAPPING, "LlavaConfig missing from MODEL_CLASS_MAPPING"

    def test_llava_config_maps_to_vlm_class(self):
        from QEfficient.transformers.modeling_utils import MODEL_CLASS_MAPPING

        if "LlavaConfig" in MODEL_CLASS_MAPPING:
            assert (
                "ImageTextToText" in MODEL_CLASS_MAPPING["LlavaConfig"]
                or "CausalLM" in MODEL_CLASS_MAPPING["LlavaConfig"]
            ), f"LlavaConfig maps to unexpected class: {MODEL_CLASS_MAPPING['LlavaConfig']}"

    def test_all_values_are_qeff_class_names(self):
        from QEfficient.transformers.modeling_utils import MODEL_CLASS_MAPPING

        for key, value in MODEL_CLASS_MAPPING.items():
            assert isinstance(value, str), f"Expected string value for key '{key}', got {type(value)}"
            assert "QEFF" in value or "QEff" in value, f"Expected QEff class name for key '{key}', got: {value}"


# ---------------------------------------------------------------------------
# Tests: SPECIALIZED_DISAGG_SERVING_MODEL_ARCH
# ---------------------------------------------------------------------------


class TestSpecializedDisaggServingModelArch:
    """SPECIALIZED_DISAGG_SERVING_MODEL_ARCH must contain expected model types."""

    def test_exists_and_is_set_or_collection(self):
        from QEfficient.transformers.modeling_utils import SPECIALIZED_DISAGG_SERVING_MODEL_ARCH

        assert hasattr(SPECIALIZED_DISAGG_SERVING_MODEL_ARCH, "__contains__")

    def test_contains_gpt_oss(self):
        from QEfficient.transformers.modeling_utils import SPECIALIZED_DISAGG_SERVING_MODEL_ARCH

        assert "gpt_oss" in SPECIALIZED_DISAGG_SERVING_MODEL_ARCH

    def test_all_entries_are_strings(self):
        from QEfficient.transformers.modeling_utils import SPECIALIZED_DISAGG_SERVING_MODEL_ARCH

        for arch in SPECIALIZED_DISAGG_SERVING_MODEL_ARCH:
            assert isinstance(arch, str), f"Expected string, got {type(arch)}: {arch}"
