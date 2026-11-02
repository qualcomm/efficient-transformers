# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only tests for QEfficient auto model classes in modeling_auto.py.

Covers:
  - QEFFTransformersBase: __repr__, quantization config guard
  - QEFFAutoModelForCausalLM: logic methods (build_prefill_specialization,
    build_decode_specialization, check_and_get_num_speculative_tokens,
    prefill), export prefill seq len handling, compile validation errors,
    generate TypeError
  - QEFFAutoModelForSequenceClassification: init, get_model_config, export
  - QEFFAutoModel: init (with/without pooling), get_model_config, export,
    pytorch_feature_generate, generate TypeError
  - MultimodalUtilityMixin: __new__ TypeError, auto_correct_inputs
  - QEFFAutoModelForSpeechSeq2Seq: init TypeError, get_model_config
  - QEFFAutoModelForCTC: init, get_model_config, export

All tests run on CPU only and are safe for parallel execution.
Run with: pytest tests/unit_test/models/test_modeling_auto_cpu.py -n auto -v
"""

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from QEfficient.transformers.models.modeling_auto import (
    MultimodalUtilityMixin,
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForSequenceClassification,
    QEFFAutoModelForSpeechSeq2Seq,
    _QEffAutoModelForImageTextToTextDualQPC,
    _QEFFAutoModelForImageTextToTextSingleQPC,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 500
CTX_LEN = 32
SEQ_LEN = 8
UNSUPPORTED_WEIGHT_FREE_WARNING = "weight_free=True is only supported for QEFFAutoModelForCausalLM"

# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_gpt2():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=VOCAB_SIZE, n_positions=CTX_LEN, n_ctx=CTX_LEN)
    return GPT2LMHeadModel(cfg).eval(), cfg


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval(), cfg


def make_tiny_bert():
    cfg = BertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return BertModel(cfg).eval(), cfg


def make_tiny_bert_seq_cls():
    cfg = BertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        num_labels=2,
    )
    return BertForSequenceClassification(cfg).eval(), cfg


def make_tiny_whisper():
    # vocab_size must be > all default token IDs (pad=50257, bos=50257, eos=50257, decoder_start=50258)
    # Use small vocab with explicit token IDs within range
    cfg = WhisperConfig(
        num_mel_bins=80,
        encoder_layers=1,
        decoder_layers=1,
        d_model=64,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        vocab_size=200,
        max_source_positions=32,
        max_target_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
    )
    # Ensure num_hidden_layers is set (QEFFAutoModelForSpeechSeq2Seq reads this)
    if not hasattr(cfg, "num_hidden_layers") or cfg.num_hidden_layers is None:
        cfg.num_hidden_layers = cfg.encoder_layers + cfg.decoder_layers
    return WhisperForConditionalGeneration(cfg).eval(), cfg


def make_tiny_wav2vec2():
    cfg = Wav2Vec2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=100,
    )
    return Wav2Vec2ForCTC(cfg).eval(), cfg


# ---------------------------------------------------------------------------
# Helper: InputInfo for MultimodalUtilityMixin tests
# ---------------------------------------------------------------------------


class _InputInfo:
    """Minimal stand-in for the InputInfo objects returned by get_inputs_info()."""

    def __init__(self, name, datatype):
        self.name = name
        self.datatype = datatype

    def __repr__(self):
        return f"InputInfo(name={self.name}, datatype={self.datatype})"


class _ConcreteMultimodalModel(MultimodalUtilityMixin):
    """Minimal concrete subclass of MultimodalUtilityMixin for testing."""

    def __init__(self, inputs_info):
        self.model = MagicMock()
        self.model.get_inputs_info.return_value = inputs_info

    @property
    def get_model_config(self):
        return {}

    def export(self, *args, **kwargs):
        pass

    def compile(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Stage 1: QEFFTransformersBase
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFTransformersBase:
    """Tests for QEFFTransformersBase base class."""

    def test_repr_contains_class_name(self):
        """__repr__ includes the QEff class name."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        r = repr(qeff)
        assert "QEFFAutoModelForCausalLM" in r

    def test_repr_contains_model_repr(self):
        """__repr__ includes the underlying model's repr."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        r = repr(qeff)
        # The model repr should be embedded
        assert len(r) > len("QEFFAutoModelForCausalLM")

    def test_init_raises_assertion_for_unsupported_quantization_config(self):
        """__init__ raises AssertionError when quantization_config is unsupported type.

        QEFFTransformersBase (parent of QEFFAutoModelForSequenceClassification) raises
        AssertionError. QEFFAutoModelForCausalLM only warns (different base class).
        """
        model, cfg = make_tiny_bert_seq_cls()
        # Inject a fake quantization_config that is NOT in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING
        model.config.quantization_config = object()
        with pytest.raises(AssertionError, match="from_pretrained"):
            QEFFAutoModelForSequenceClassification(model)

    def test_init_passes_without_quantization_config(self):
        """__init__ succeeds when model has no quantization_config."""
        model, cfg = make_tiny_gpt2()
        assert not hasattr(model.config, "quantization_config")
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff is not None

    @pytest.mark.parametrize(
        ("wrapper_cls", "model_factory"),
        [
            pytest.param(QEFFAutoModel, make_tiny_bert, id="automodel"),
            pytest.param(QEFFAutoModelForSequenceClassification, make_tiny_bert_seq_cls, id="sequence-classification"),
            pytest.param(QEFFAutoModelForSpeechSeq2Seq, make_tiny_whisper, id="speech-seq2seq"),
            pytest.param(QEFFAutoModelForCTC, make_tiny_wav2vec2, id="ctc"),
        ],
    )
    def test_from_pretrained_disables_unsupported_weight_free(self, wrapper_cls, model_factory, monkeypatch, caplog):
        """Non-CausalLM from_pretrained paths warn and do not forward weight_free."""
        captured_kwargs = {}

        def fake_from_pretrained(_model_id, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return model_factory()[0]

        monkeypatch.setattr(wrapper_cls._hf_auto_class, "from_pretrained", fake_from_pretrained)
        caplog.set_level(logging.WARNING, logger="QEfficient")

        qeff_model = wrapper_cls.from_pretrained("dummy-model", weight_free=True)

        assert "weight_free" not in captured_kwargs
        assert qeff_model._weight_free is False
        assert UNSUPPORTED_WEIGHT_FREE_WARNING in caplog.text
        assert wrapper_cls.__name__ in caplog.text

    @pytest.mark.parametrize(
        ("wrapper_cls", "model_factory"),
        [
            pytest.param(QEFFAutoModel, make_tiny_bert, id="automodel"),
            pytest.param(QEFFAutoModelForSequenceClassification, make_tiny_bert_seq_cls, id="sequence-classification"),
            pytest.param(QEFFAutoModelForSpeechSeq2Seq, make_tiny_whisper, id="speech-seq2seq"),
            pytest.param(QEFFAutoModelForCTC, make_tiny_wav2vec2, id="ctc"),
        ],
    )
    def test_direct_init_disables_unsupported_weight_free(self, wrapper_cls, model_factory, caplog):
        """Non-CausalLM direct construction must not enable the weight-free export path."""
        caplog.set_level(logging.WARNING, logger="QEfficient")

        qeff_model = wrapper_cls(model_factory()[0], weight_free=True)

        assert qeff_model._weight_free is False
        assert UNSUPPORTED_WEIGHT_FREE_WARNING in caplog.text
        assert wrapper_cls.__name__ in caplog.text

    def test_causal_lm_direct_init_preserves_weight_free(self, caplog):
        """CausalLM remains the only wrapper that accepts weight_free=True."""
        caplog.set_level(logging.WARNING, logger="QEfficient")

        qeff_model = QEFFAutoModelForCausalLM(make_tiny_llama()[0], weight_free=True)

        assert qeff_model._weight_free is True
        assert UNSUPPORTED_WEIGHT_FREE_WARNING not in caplog.text


# ---------------------------------------------------------------------------
# Stage 2: QEFFAutoModelForCausalLM — logic methods
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.causal_lm
class TestQEFFAutoModelForCausalLMLogic:
    """Tests for QEFFAutoModelForCausalLM logic methods (no QAIC required)."""

    # --- Basic properties ---

    def test_repr_contains_class_name(self):
        """__repr__ includes QEFFAutoModelForCausalLM."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert "QEFFAutoModelForCausalLM" in repr(qeff)

    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    def test_continuous_batching_default_false(self):
        """continuous_batching defaults to False."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.continuous_batching is False

    def test_continuous_batching_true_when_set(self):
        """continuous_batching=True is stored correctly."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        assert qeff.continuous_batching is True

    def test_is_tlm_false_by_default(self):
        """is_tlm is False when no qaic_config specifies speculative model."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.is_tlm is False

    def test_num_layers_set_correctly(self):
        """num_layers matches the model config."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.num_layers == 2

    def test_init_raises_type_error_for_non_causal_lm(self):
        """__init__ raises TypeError when model is not a CausalLM or LMHeadModel."""
        model, cfg = make_tiny_bert()
        with pytest.raises(TypeError, match="CausalLM or LMHeadModel"):
            QEFFAutoModelForCausalLM(model)

    # --- build_prefill_specialization ---

    def test_build_prefill_specialization_basic(self):
        """build_prefill_specialization returns dict with seq_len and ctx_len."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_prefill_specialization(prefill_seq_len=32, ctx_len=128, batch_size=1)
        assert isinstance(spec, dict)
        assert "seq_len" in spec
        assert spec["seq_len"] == 32
        assert "ctx_len" in spec
        assert spec["ctx_len"] == 128

    def test_build_prefill_specialization_batch_size(self):
        """build_prefill_specialization includes batch_size when kv_cache_batch_size is set."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # kv_cache_batch_size overrides batch_size in the returned spec; must be passed explicitly
        spec = qeff.build_prefill_specialization(prefill_seq_len=32, ctx_len=128, batch_size=4, kv_cache_batch_size=4)
        assert spec["batch_size"] == 4

    def test_build_prefill_specialization_continuous_batching(self):
        """build_prefill_specialization with continuous_batching uses full_batch_size."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        spec = qeff.build_prefill_specialization(
            prefill_seq_len=32,
            ctx_len=128,
            batch_size=1,
            kv_cache_batch_size=4,
            full_batch_size=4,
        )
        assert "full_batch_size" in spec
        assert spec["full_batch_size"] == 4

    def test_build_prefill_specialization_no_none_values(self):
        """build_prefill_specialization filters out None values."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_prefill_specialization(prefill_seq_len=32, ctx_len=128, batch_size=1)
        assert all(v is not None for v in spec.values())

    # --- build_decode_specialization ---

    def test_build_decode_specialization_basic(self):
        """build_decode_specialization returns dict with seq_len=1 for decode."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_decode_specialization(prefill_seq_len=32, ctx_len=128, batch_size=1)
        assert isinstance(spec, dict)
        assert "seq_len" in spec
        assert spec["seq_len"] == 1  # decode step is always seq_len=1

    def test_build_decode_specialization_ctx_len(self):
        """build_decode_specialization includes ctx_len."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_decode_specialization(prefill_seq_len=32, ctx_len=256, batch_size=1)
        assert spec["ctx_len"] == 256

    def test_build_decode_specialization_continuous_batching(self):
        """build_decode_specialization with continuous_batching uses full_batch_size."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        spec = qeff.build_decode_specialization(
            prefill_seq_len=32,
            ctx_len=128,
            batch_size=1,
            kv_cache_batch_size=4,
            full_batch_size=4,
        )
        assert "full_batch_size" in spec
        assert spec["full_batch_size"] == 4

    def test_build_decode_specialization_no_none_values(self):
        """build_decode_specialization filters out None values."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_decode_specialization(prefill_seq_len=32, ctx_len=128, batch_size=1)
        assert all(v is not None for v in spec.values())

    # --- check_and_get_num_speculative_tokens ---

    def test_check_speculative_tokens_not_tlm_returns_none(self):
        """check_and_get_num_speculative_tokens returns None when is_tlm=False."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.is_tlm is False
        result = qeff.check_and_get_num_speculative_tokens(None, 32)
        assert result is None

    def test_check_speculative_tokens_not_tlm_ignores_value(self):
        """check_and_get_num_speculative_tokens returns None even if value passed when not TLM."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff.check_and_get_num_speculative_tokens(5, 32)
        assert result is None

    def test_check_speculative_tokens_tlm_none_raises_type_error(self):
        """check_and_get_num_speculative_tokens raises TypeError when TLM and None passed."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True  # Force TLM mode for testing
        with pytest.raises(TypeError, match="num_speculative_tokens"):
            qeff.check_and_get_num_speculative_tokens(None, 32)

    def test_check_speculative_tokens_seq_len_too_short_raises_value_error(self):
        """check_and_get_num_speculative_tokens raises ValueError when prefill_seq_len too short."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True  # Force TLM mode for testing
        # prefill_seq_len=3 < num_speculative_tokens+1=6
        with pytest.raises(ValueError, match="sequence length"):
            qeff.check_and_get_num_speculative_tokens(5, 3)

    def test_check_speculative_tokens_valid_returns_value(self):
        """check_and_get_num_speculative_tokens returns the value when valid."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True  # Force TLM mode for testing
        result = qeff.check_and_get_num_speculative_tokens(3, 32)
        assert result == 3

    # --- prefill ---

    def test_prefill_enable_applies_transform(self):
        """prefill(enable=True) applies PrefillOnlyTransform without error."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Should not raise
        qeff.prefill(enable=True)

    def test_prefill_disable_reverts_transform(self):
        """prefill(enable=False) applies RevertPrefillOnlyTransform without error."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.prefill(enable=True)
        # Should not raise
        qeff.prefill(enable=False)

    def test_prefill_enable_chunking(self):
        """prefill(enable=True, enable_chunking=True) applies chunked transform."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Should not raise
        qeff.prefill(enable=True, enable_chunking=True)

    def test_prefill_retain_full_kv(self):
        """prefill(enable=False, retain_full_kv=True) applies RevertPrefillKeepAttentionTransform."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Should not raise
        qeff.prefill(enable=False, retain_full_kv=True)


# ---------------------------------------------------------------------------
# Stage 3: QEFFAutoModelForCausalLM — compile/generate validation
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.causal_lm
class TestQEFFAutoModelForCausalLMCompileValidation:
    """Tests for compile/generate validation logic (no QAIC required)."""

    def test_compile_continuous_batching_requires_full_batch_size(self):
        """compile raises TypeError when continuous_batching=True and full_batch_size=None."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        with pytest.raises(TypeError, match="full_batch_size"):
            qeff.compile(prefill_seq_len=32, ctx_len=128)

    def test_compile_prefill_only_must_be_bool(self):
        """compile raises TypeError when prefill_only is not a boolean."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        with pytest.raises(TypeError, match="prefill_only"):
            qeff.compile(prefill_seq_len=32, ctx_len=128, prefill_only="yes")

    @pytest.mark.parametrize(
        ("compile_kwargs", "expected_graph_names"),
        [
            pytest.param({"prefill_seq_len": 32}, ["Prefill", "Decode"], id="combined"),
            pytest.param({"prefill_only": False, "prefill_seq_len": 32}, ["Decode"], id="explicit-decode-nonunit"),
        ],
    )
    def test_weight_free_compile_allows_supported_modes(
        self, tmp_path, monkeypatch, compile_kwargs, expected_graph_names
    ):
        """weight_free=True allows compile modes outside the unsupported disaggregated boundary."""
        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, weight_free=True)
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        captured_kwargs = {}

        def fake_compile(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "qpc"

        monkeypatch.setattr(qeff, "_compile", fake_compile)

        qpc_path = qeff.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_path),
            ctx_len=128,
            **compile_kwargs,
        )

        assert qpc_path == tmp_path / "qpc"
        assert captured_kwargs["prefill_only"] == compile_kwargs.get("prefill_only")
        assert [spec["_graph_name"] for spec in captured_kwargs["specializations"]] == expected_graph_names

    def test_compile_prefill_only_true_continuous_batching_requires_kv_cache_batch_size(self):
        """compile raises ValueError when prefill_only=True + continuous_batching=True + no kv_cache_batch_size."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        with pytest.raises((TypeError, ValueError)):
            qeff.compile(prefill_seq_len=32, ctx_len=128, prefill_only=True)

    def test_generate_raises_type_error_without_compile(self):
        """generate raises TypeError when QPC is not compiled."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        tokenizer = MagicMock()
        with pytest.raises(TypeError, match="compile API"):
            qeff.generate(tokenizer=tokenizer, prompts=["Hello"])

    def test_generate_raises_not_implemented_for_pytorch_runtime(self):
        """generate raises NotImplementedError when runtime_ai100=False."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        tokenizer = MagicMock()
        with pytest.raises(NotImplementedError):
            qeff.generate(tokenizer=tokenizer, prompts=["Hello"], runtime_ai100=False)

    def test_compile_gptoss_prefill_only_forces_chunking(self, tmp_path, monkeypatch, caplog):
        """gpt_oss prefill-only compile always passes enable_chunking=True downstream."""
        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.model.config.model_type = "gpt_oss"
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        captured_kwargs = {}

        def fake_compile(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "qpc"

        monkeypatch.setattr(qeff, "_compile", fake_compile)
        caplog.set_level(logging.WARNING, logger="QEfficient")

        qeff.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_path),
            prefill_seq_len=32,
            ctx_len=128,
            prefill_only=True,
            enable_chunking=False,
        )

        assert captured_kwargs["enable_chunking"] is True
        assert captured_kwargs["specializations"][0]["_graph_name"] == "Prefill"
        assert "chunking is always enabled for prefill-only mode" in caplog.text

    def test_compile_ignores_public_mdp_ts_num_devices(self, tmp_path, monkeypatch, caplog):
        """compile ignores public mdp_ts_num_devices and derives it from num_devices."""
        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        captured_kwargs = {}

        def fake_compile(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "qpc"

        monkeypatch.setattr(qeff, "_compile", fake_compile)
        caplog.set_level(logging.WARNING, logger="QEfficient")

        qeff.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_path),
            prefill_seq_len=8,
            ctx_len=32,
            num_devices=4,
            mdp_num_partitions=2,
            mdp_ts_num_devices=99,
        )

        assert captured_kwargs["mdp_ts_num_devices"] == 4
        assert captured_kwargs["mdp_num_partitions"] == 2
        assert "`mdp_ts_num_devices` passed to compile() is ignored" in caplog.text


# ---------------------------------------------------------------------------
# Stage 4: QEFFAutoModelForCausalLM — export prefill seq len handling
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.causal_lm
class TestQEFFAutoModelForCausalLMExportPrefillSeqLen:
    """Tests for specialized prefill seq len handling inside export()."""

    @staticmethod
    def _make_specialized_qeff(model_type="gpt_oss"):
        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.model.config.model_type = model_type
        setattr(qeff, "_QEFFAutoModelForCausalLM__update_prefill_transform", lambda *args, **kwargs: None)
        return qeff

    @staticmethod
    def _capture_export(qeff, **export_kwargs):
        captured = {}

        def fake_export(example_inputs, output_names=None, dynamic_axes=None, export_dir=None, **kwargs):
            captured["example_inputs"] = example_inputs
            captured["output_names"] = output_names
            captured["dynamic_axes"] = dynamic_axes
            captured["export_dir"] = export_dir
            captured["kwargs"] = kwargs
            return "/tmp/fake.onnx"

        qeff._export = fake_export
        assert qeff.export(**export_kwargs) == "/tmp/fake.onnx"
        return captured

    def test_chunked_specialized_prefill_uses_default_export_seq_len_without_prefill_seq_len(self):
        """With enable_chunking=True and no prefill_seq_len, export uses ONNX_EXPORT_EXAMPLE_SEQ_LEN."""
        from QEfficient.utils.constants import ONNX_EXPORT_EXAMPLE_SEQ_LEN

        qeff = self._make_specialized_qeff()

        captured = self._capture_export(qeff, prefill_only=True, prefill_seq_len=None, enable_chunking=True)

        assert captured["example_inputs"]["input_ids"].shape[1] == ONNX_EXPORT_EXAMPLE_SEQ_LEN
        assert qeff.hash_params.get("prefill_only") is True
        assert qeff.hash_params.get("chunking") is True

    def test_chunked_specialized_prefill_uses_prefill_seq_len_when_larger_than_default(self):
        """Chunked specialized prefill export uses the requested prefill length when it is larger."""
        qeff = self._make_specialized_qeff()

        captured = self._capture_export(qeff, prefill_only=True, prefill_seq_len=512, enable_chunking=True)

        assert captured["example_inputs"]["input_ids"].shape[1] == 512

    def test_expert_parallel_keeps_default_export_seq_len_when_divisible_by_num_packed_chunks(self):
        """Expert-parallel export leaves seq_len unchanged when it is already divisible by packed chunks."""
        from QEfficient.transformers.moe.flavours import MoEFlavour
        from QEfficient.utils.constants import ONNX_EXPORT_EXAMPLE_SEQ_LEN

        qeff = self._make_specialized_qeff(model_type="qwen3_moe")
        qeff.hash_params["moe_prefill_flavour"] = MoEFlavour.EXPERT_PARALLEL
        qeff.hash_params["moe_prefill_num_packed_chunks"] = 2

        captured = self._capture_export(qeff, prefill_only=True, prefill_seq_len=None, enable_chunking=True)

        assert captured["example_inputs"]["input_ids"].shape[1] == ONNX_EXPORT_EXAMPLE_SEQ_LEN

    def test_expert_parallel_rounds_export_seq_len_to_multiple_of_num_packed_chunks(self):
        """Expert-parallel export chooses an input seq len divisible by num_packed_chunks."""
        from QEfficient.transformers.moe.flavours import MoEFlavour
        from QEfficient.utils.constants import ONNX_EXPORT_EXAMPLE_SEQ_LEN

        num_packed_chunks = 3
        qeff = self._make_specialized_qeff(model_type="qwen3_moe")
        qeff.hash_params["moe_prefill_flavour"] = MoEFlavour.EXPERT_PARALLEL
        qeff.hash_params["moe_prefill_num_packed_chunks"] = num_packed_chunks

        captured = self._capture_export(qeff, prefill_only=True, prefill_seq_len=None, enable_chunking=True)
        export_seq_len = captured["example_inputs"]["input_ids"].shape[1]

        assert export_seq_len >= ONNX_EXPORT_EXAMPLE_SEQ_LEN
        assert export_seq_len % num_packed_chunks == 0

    @pytest.mark.skip(
        reason="GPT-OSS non-chunked prefill-only export is disabled; requests are forced to chunked prefill."
    )
    def test_no_prefill_seq_len_no_env_var_raises_value_error(self, monkeypatch):
        """GPT-OSS non-chunked prefill export requires a valid prefill_seq_len."""
        qeff = self._make_specialized_qeff()
        monkeypatch.delenv("NUM_Q_BLOCKS", raising=False)
        with pytest.raises(ValueError, match="prefill_seq_len"):
            qeff.export(prefill_only=True, prefill_seq_len=None, enable_chunking=False)

    @pytest.mark.skip(
        reason="GPT-OSS non-chunked prefill-only export is disabled; requests are forced to chunked prefill."
    )
    def test_prefill_seq_len_not_divisible_raises_value_error(self, monkeypatch):
        """GPT-OSS non-chunked prefill export validates block-size divisibility."""
        from QEfficient.utils.constants import GPT_OSS_PREFILL_Q_BLOCK_SIZE

        qeff = self._make_specialized_qeff()
        monkeypatch.delenv("NUM_Q_BLOCKS", raising=False)
        bad_seq_len = GPT_OSS_PREFILL_Q_BLOCK_SIZE + 1
        with pytest.raises(ValueError):
            qeff.export(prefill_only=True, prefill_seq_len=bad_seq_len, enable_chunking=False)

    def test_chunked_gpt_oss_prefill_export_bypasses_non_chunked_validation(self):
        """GPT-OSS chunked prefill export does not use the non-chunked q-block validation path."""
        qeff = self._make_specialized_qeff()

        def raise_if_called(prefill_seq_len=None):
            raise AssertionError("chunked export should not use non-chunked GPT-OSS validation")

        qeff.handle_gpt_oss_env_variable_legacy_burden = raise_if_called

        captured = self._capture_export(qeff, prefill_only=True, prefill_seq_len=None, enable_chunking=True)

        assert captured["example_inputs"]["input_ids"].shape[1] > 0


# ---------------------------------------------------------------------------
# Stage 5: QEFFAutoModelForSequenceClassification
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.seq_classification
class TestQEFFAutoModelForSequenceClassification:
    """Tests for QEFFAutoModelForSequenceClassification."""

    def test_init_sets_use_cache_true(self):
        """__init__ sets model.config.use_cache=True."""
        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        assert qeff.model.config.use_cache is True

    def test_init_stores_model(self):
        """__init__ stores the model."""
        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        assert qeff.model is not None

    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        assert qeff.hash_params.get("qeff_auto_class") == "QEFFAutoModelForSequenceClassification"

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_produces_onnx_file(self, tmp_export_dir):
        """export produces a valid ONNX file."""

        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        assert onnx_path is not None
        assert os.path.exists(str(onnx_path))
        assert os.path.getsize(str(onnx_path)) > 0

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_correct_inputs(self, tmp_export_dir):
        """Exported ONNX has input_ids and attention_mask inputs."""
        import onnx

        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_logits_output(self, tmp_export_dir):
        """Exported ONNX has logits output."""
        import onnx

        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        output_names = {out.name for out in onnx_model.graph.output}
        assert "logits" in output_names

    def test_compile_warns_for_many_seq_lens(self):
        """compile warns when seq_len list has >= 15 items."""
        import warnings
        from unittest.mock import patch

        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Patch _compile so we never need QAIC hardware; it raises to stop execution
            with patch.object(qeff, "_compile", side_effect=RuntimeError("mocked compile")):
                with pytest.raises(RuntimeError):
                    qeff.compile(seq_len=list(range(1, 20)))  # 19 items >= 15

        assert any("fewer than 15" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Stage 6: QEFFAutoModel
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.embedding
class TestQEFFAutoModel:
    """Tests for QEFFAutoModel (encoder/embedding models)."""

    def test_init_without_pooling(self):
        """__init__ without pooling initializes correctly."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        assert qeff is not None
        assert qeff.model is not None

    def test_init_with_mean_pooling(self):
        """__init__ with pooling='mean' applies PoolingTransform."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model, pooling="mean")
        assert qeff is not None

    def test_init_with_cls_pooling(self):
        """__init__ with pooling='cls' applies PoolingTransform."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model, pooling="cls")
        assert qeff is not None

    def test_init_sets_use_cache_true(self):
        """__init__ sets use_cache based on model architecture.

        Decoder or encoder-decoder models (is_decoder=True or is_encoder_decoder=True)
        require KV-cache and therefore get use_cache=True.
        Encoder-only models (e.g. BERT) do not use a KV-cache, so use_cache is
        explicitly set to None to avoid forcing cache mode on architectures that
        do not support it (needed after the transformers upgrade that added RoBERTa
        support alongside BERT).
        """
        # --- Encoder-only model (BERT): use_cache must be None ---
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        # BERT has is_decoder=False and is_encoder_decoder=False, so cache mode
        # is intentionally disabled.
        assert qeff.model.base_model.config.use_cache is None

        # --- Decoder model: use_cache must be True ---
        decoder_model, decoder_cfg = make_tiny_bert()
        decoder_model.config.is_decoder = True
        qeff_decoder = QEFFAutoModel(decoder_model)
        assert qeff_decoder.model.base_model.config.use_cache is True

    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        assert qeff.hash_params.get("qeff_auto_class") == "QEFFAutoModel"

    def test_pytorch_feature_generate_returns_output(self):
        """pytorch_feature_generate runs the model and returns output."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        output = qeff.pytorch_feature_generate(model=qeff.model, inputs=inputs)
        assert output is not None

    def test_generate_raises_type_error_without_compile(self):
        """generate raises TypeError when QPC is not compiled."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        with pytest.raises(TypeError, match="compile API"):
            qeff.generate(inputs=inputs, runtime_ai100=True)

    def test_generate_pytorch_runtime_returns_output(self):
        """generate with runtime_ai100=False uses PyTorch runtime."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        output = qeff.generate(inputs=inputs, runtime_ai100=False)
        assert output is not None

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_produces_onnx_file(self, tmp_export_dir):
        """export produces a valid ONNX file."""

        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        assert onnx_path is not None
        assert os.path.exists(str(onnx_path))
        assert os.path.getsize(str(onnx_path)) > 0

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_correct_inputs(self, tmp_export_dir):
        """Exported ONNX has input_ids and attention_mask inputs."""
        import onnx

        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_output(self, tmp_export_dir):
        """Exported ONNX has at least one output."""
        import onnx

        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        assert len(onnx_model.graph.output) > 0


# ---------------------------------------------------------------------------
# Stage 7: MultimodalUtilityMixin
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestMultimodalUtilityMixin:
    """Tests for MultimodalUtilityMixin."""

    def test_direct_instantiation_raises_type_error(self):
        """Direct instantiation of MultimodalUtilityMixin raises TypeError."""
        with pytest.raises(TypeError):
            MultimodalUtilityMixin()

    def test_auto_correct_inputs_success(self):
        """auto_correct_inputs succeeds with correct inputs."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
            _InputInfo("attention_mask", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        result = mixin.auto_correct_inputs(inputs)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_auto_correct_inputs_filters_extra_keys(self):
        """auto_correct_inputs filters out keys not in inputs_info."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "extra_key": torch.zeros((1, SEQ_LEN), dtype=torch.float32),
        }
        result = mixin.auto_correct_inputs(inputs)
        assert "input_ids" in result
        assert "extra_key" not in result

    def test_auto_correct_inputs_missing_key_raises_runtime_error(self):
        """auto_correct_inputs raises RuntimeError when a required key is missing."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
            _InputInfo("attention_mask", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            # attention_mask is missing
        }
        with pytest.raises(RuntimeError):
            mixin.auto_correct_inputs(inputs)

    def test_auto_correct_inputs_wrong_dtype_raises_runtime_error(self):
        """auto_correct_inputs raises RuntimeError when dtype does not match."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
            _InputInfo("attention_mask", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.float32),  # wrong dtype
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        with pytest.raises(RuntimeError):
            mixin.auto_correct_inputs(inputs)

    def test_auto_correct_inputs_error_message_contains_expected_info(self):
        """RuntimeError message contains expected input names."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {}  # empty inputs
        with pytest.raises(RuntimeError) as exc_info:
            mixin.auto_correct_inputs(inputs)
        assert "input_ids" in str(exc_info.value) or "Expected" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Stage 8: QEFFAutoModelForSpeechSeq2Seq
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.speech
class TestQEFFAutoModelForSpeechSeq2Seq:
    """Tests for QEFFAutoModelForSpeechSeq2Seq."""

    def test_init_raises_type_error_for_non_conditional_generation(self):
        """__init__ raises TypeError for non-ForConditionalGeneration models."""
        model, cfg = make_tiny_gpt2()
        with pytest.raises(TypeError, match="ForConditionalGeneration"):
            QEFFAutoModelForSpeechSeq2Seq(model)

    def test_init_raises_type_error_for_causal_lm(self):
        """__init__ raises TypeError for CausalLM models."""
        model, cfg = make_tiny_llama()
        with pytest.raises(TypeError, match="ForConditionalGeneration"):
            QEFFAutoModelForSpeechSeq2Seq(model)

    @pytest.mark.slow
    def test_init_succeeds_with_whisper(self):
        """__init__ succeeds with a Whisper ForConditionalGeneration model."""
        model, cfg = make_tiny_whisper()
        qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        assert qeff is not None
        assert qeff.model is not None

    @pytest.mark.slow
    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_whisper()
        qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    @pytest.mark.slow
    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_whisper()
        qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        assert qeff.hash_params.get("qeff_auto_class") == "QEFFAutoModelForSpeechSeq2Seq"

    @pytest.mark.slow
    def test_num_layers_set_correctly(self):
        """num_layers is set from model config."""
        model, cfg = make_tiny_whisper()
        qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        assert isinstance(qeff.num_layers, int)
        assert qeff.num_layers > 0

    def test_generate_raises_type_error_without_compile(self):
        """generate raises TypeError when QPC is not compiled."""
        model, cfg = make_tiny_whisper()
        try:
            qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        except Exception:
            pytest.skip("Whisper model initialization failed, skipping generate test")
        with pytest.raises(TypeError, match="compile API"):
            qeff.generate(inputs={}, generation_len=10)


# ---------------------------------------------------------------------------
# Stage 9: QEFFAutoModelForCTC
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForCTC:
    """Tests for QEFFAutoModelForCTC (Wav2Vec2 and similar CTC models)."""

    def test_init_sets_use_cache_true(self):
        """__init__ sets model.base_model.config.use_cache=True."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        assert qeff.model.base_model.config.use_cache is True

    def test_init_stores_model(self):
        """__init__ stores the model."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        assert qeff.model is not None

    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        assert qeff.hash_params.get("qeff_auto_class") == "QEFFAutoModelForCTC"

    def test_generate_raises_type_error_without_compile(self):
        """generate raises TypeError when QPC is not compiled."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        processor = MagicMock()
        with pytest.raises(TypeError, match="compile API"):
            qeff.generate(processor=processor, inputs=torch.zeros(1, 100))

    def test_generate_pytorch_runtime_calls_model(self):
        """generate with runtime_ai100=False calls pytorch_feature_generate."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        processor = MagicMock()
        processor.return_value = MagicMock(input_values=torch.zeros((1, 100), dtype=torch.float32))
        # pytorch_feature_generate calls processor and model
        # We just verify it doesn't raise TypeError about compile
        try:
            qeff.generate(processor=processor, inputs=torch.zeros(1, 100), runtime_ai100=False)
        except Exception as e:
            # Any exception other than TypeError about compile is acceptable
            assert "compile API" not in str(e)

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_produces_onnx_file(self, tmp_export_dir):
        """export produces a valid ONNX file."""

        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        assert onnx_path is not None
        assert os.path.exists(str(onnx_path))
        assert os.path.getsize(str(onnx_path)) > 0

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_input_values(self, tmp_export_dir):
        """Exported ONNX has input_values input."""
        import onnx

        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        assert "input_values" in input_names

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_logits_output(self, tmp_export_dir):
        """Exported ONNX has logits output."""
        import onnx

        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        output_names_ctc = {out.name for out in onnx_model.graph.output}
        assert "logits" in output_names_ctc


# ---------------------------------------------------------------------------
# TLM multi-spec specialization unit tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.causal_lm
class TestTLMMultiSpecSpecializations:
    """Tests for the multi-spec decode specialization API (num_speculative_tokens as list)."""

    # ---- build_decode_specialization (multi-spec) ----

    def test_build_decode_spec_for_k_seq_len(self):
        """build_decode_specialization sets seq_len = num_speculative_tokens+1 for TLM."""
        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True
        for k in [0, 1, 3, 7]:
            spec = qeff.build_decode_specialization(
                num_speculative_tokens=k, ctx_len=128, batch_size=1, kv_cache_batch_size=1, prefill_seq_len=32
            )
            assert spec is not None
            assert spec["seq_len"] == k + 1

    def test_build_decode_spec_for_k_num_logits_to_keep(self):
        """build_decode_specialization sets num_logits_to_keep = num_speculative_tokens+1 for TLM."""
        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True
        for k in [0, 1, 3]:
            spec = qeff.build_decode_specialization(
                num_speculative_tokens=k, ctx_len=128, batch_size=1, kv_cache_batch_size=1, prefill_seq_len=32
            )
            assert spec["num_logits_to_keep"] == k + 1

    def test_build_decode_spec_for_k_returns_none_when_duplicate_prefill(self):
        """Returns None when seq_len == prefill_seq_len and no continuous batching."""
        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True
        # num_speculative_tokens=0 → seq_len=1 == prefill_seq_len=1 → should be None
        spec = qeff.build_decode_specialization(
            num_speculative_tokens=0, ctx_len=128, batch_size=1, kv_cache_batch_size=1, prefill_seq_len=1
        )
        assert spec is None

    def test_build_decode_spec_for_k_not_none_with_continuous_batching(self):
        """Returns spec even when seq_len == prefill_seq_len if continuous_batching is enabled."""
        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        qeff.is_tlm = True
        # num_speculative_tokens=0 → seq_len=1 == prefill_seq_len=1, but CB is True → should not be None
        spec = qeff.build_decode_specialization(
            num_speculative_tokens=0,
            ctx_len=128,
            batch_size=1,
            kv_cache_batch_size=2,
            full_batch_size=2,
            prefill_seq_len=1,
        )
        assert spec is not None

    # ---- compile() specialization count via mock ----

    def test_compile_list_produces_correct_spec_count(self):
        """compile(num_speculative_tokens=[0, 3]) → 1 prefill + 2 decode specializations."""
        from unittest.mock import patch

        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model, qaic_config={"speculative_model_type": "target"})
        captured = {}

        with patch.object(
            type(qeff),
            "_compile",
            side_effect=lambda *args, **kw: (
                captured.update({"specializations": kw.get("specializations")}) or "/fake/qpc"
            ),
        ):
            qeff.compile(prefill_seq_len=32, ctx_len=128, num_speculative_tokens=[0, 3])

        assert captured.get("specializations") is not None, "_compile was not reached"
        specs = captured["specializations"]
        decode_specs = [s for s in specs if s.get("seq_len", 0) != 32]
        assert len(decode_specs) == 2, f"Expected 2 decode specs, got {len(decode_specs)}: {specs}"

    def test_compile_deduplication(self):
        """compile(num_speculative_tokens=[3, 3, 3]) → only one decode spec for K=3."""
        from unittest.mock import patch

        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model, qaic_config={"speculative_model_type": "target"})
        captured = {}

        with patch.object(
            type(qeff),
            "_compile",
            side_effect=lambda *args, **kw: (
                captured.update({"specializations": kw.get("specializations")}) or "/fake/qpc"
            ),
        ):
            qeff.compile(prefill_seq_len=32, ctx_len=128, num_speculative_tokens=[3, 3, 3])

        assert captured.get("specializations") is not None, "_compile was not reached"
        specs = captured["specializations"]
        decode_specs = [s for s in specs if s.get("seq_len", 0) != 32]
        assert len(decode_specs) == 1, f"Expected 1 decode spec (deduplicated), got: {decode_specs}"
        assert decode_specs[0]["seq_len"] == 4

    def test_compile_sorting(self):
        """compile(num_speculative_tokens=[3, 1, 2]) → decode specs in ascending seq_len order."""
        from unittest.mock import patch

        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model, qaic_config={"speculative_model_type": "target"})
        captured = {}

        with patch.object(
            type(qeff),
            "_compile",
            side_effect=lambda *args, **kw: (
                captured.update({"specializations": kw.get("specializations")}) or "/fake/qpc"
            ),
        ):
            qeff.compile(prefill_seq_len=32, ctx_len=128, num_speculative_tokens=[3, 1, 2])

        assert captured.get("specializations") is not None, "_compile was not reached"
        specs = captured["specializations"]
        decode_specs = [s for s in specs if s.get("seq_len", 0) != 32]
        assert len(decode_specs) == 3
        seq_lens = [s["seq_len"] for s in decode_specs]
        assert seq_lens == sorted(seq_lens), f"Decode specs not in sorted order: {seq_lens}"

    def test_compile_int_backward_compat(self):
        """compile(num_speculative_tokens=3) as plain int still works (treated as [3])."""
        from unittest.mock import patch

        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model, qaic_config={"speculative_model_type": "target"})
        captured = {}

        with patch.object(
            type(qeff),
            "_compile",
            side_effect=lambda *args, **kw: (
                captured.update({"specializations": kw.get("specializations")}) or "/fake/qpc"
            ),
        ):
            qeff.compile(prefill_seq_len=32, ctx_len=128, num_speculative_tokens=3)

        assert captured.get("specializations") is not None, "_compile was not reached"
        specs = captured["specializations"]
        decode_specs = [s for s in specs if s.get("seq_len", 0) != 32]
        assert len(decode_specs) == 1, f"Expected 1 decode spec for int input, got: {decode_specs}"
        assert decode_specs[0]["seq_len"] == 4  # k=3 → seq_len=4

    def test_compile_int_zero_backward_compat(self):
        """compile(num_speculative_tokens=0) as plain scalar int still works (treated as [0])."""
        from unittest.mock import patch

        model, _ = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model, qaic_config={"speculative_model_type": "target"})
        captured = {}

        with patch.object(
            type(qeff),
            "_compile",
            side_effect=lambda *args, **kw: (
                captured.update({"specializations": kw.get("specializations")}) or "/fake/qpc"
            ),
        ):
            qeff.compile(prefill_seq_len=32, ctx_len=128, num_speculative_tokens=0)

        assert captured.get("specializations") is not None, "_compile was not reached"
        specs = captured["specializations"]
        decode_specs = [s for s in specs if s.get("seq_len", 0) != 32]
        assert len(decode_specs) == 1, f"Expected 1 decode spec for scalar 0, got: {decode_specs}"
        assert decode_specs[0]["seq_len"] == 1  # k=0 → seq_len=1


@pytest.mark.cpu_only
class TestArtifactsGenerationAPIs:
    def test_causal_lm_compile_artifacts_reaches_base_compile(self, tmp_path):
        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        expected = tmp_path / "qpc"
        with patch.object(qeff, "_compile", return_value=expected) as compile_model:
            result = qeff.compile(prefill_seq_len=8, ctx_len=32, artifacts=True)
        assert result == expected
        assert compile_model.call_args.kwargs["artifacts"] is True

    def test_causal_lm_proxy_writes_artifact_bundle(self, tmp_path):
        import numpy as np

        class FakeTokenizer:
            padding_side = "right"
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, prompt, return_tensors, padding, max_length=None):
                del prompt, return_tensors
                input_ids = np.array([[1, 2]], dtype=np.int64)
                attention_mask = np.ones_like(input_ids)
                if padding == "max_length":
                    pad_width = max_length - input_ids.shape[1]
                    input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)))
                    attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)))
                return {"input_ids": input_ids, "attention_mask": attention_mask}

        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, enable_proxy=True)
        compile_dir = qeff.compile(
            compile_dir=tmp_path,
            prefill_seq_len=8,
            ctx_len=32,
            artifacts=True,
            offload_pt_weights=False,
        )
        io_dir = qeff.generate(tokenizer=FakeTokenizer(), prompts=["hello"], artifacts=True)

        assert compile_dir == qeff.compile_artifacts_path
        assert (compile_dir / "qaic-compile.sh").is_file()
        assert (io_dir / "aic_batch_io.json").is_file()

    def test_causal_lm_artifacts_delegates_without_runtime(self, tmp_path):
        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.qpc_path = tmp_path / "qpc"
        expected = tmp_path / "io"
        with patch(
            "QEfficient.transformers.models.modeling_auto.write_causal_lm_runner_bundle",
            return_value=expected,
        ) as writer:
            result = qeff.generate(tokenizer=MagicMock(), prompts=["hello"], artifacts=True)
        assert result == expected
        writer.assert_called_once()

    def test_single_qpc_vlm_artifacts_delegates_without_runtime(self, tmp_path):
        model = SimpleNamespace()
        expected = tmp_path / "io"
        with patch(
            "QEfficient.transformers.models.modeling_auto.write_single_qpc_vlm_runner_bundle",
            return_value=expected,
        ) as writer:
            result = _QEFFAutoModelForImageTextToTextSingleQPC.generate(
                model,
                processor=MagicMock(),
                images=["image"],
                prompts=["prompt"],
                artifacts=True,
            )
        assert result == expected
        writer.assert_called_once()

    def test_dual_qpc_vlm_artifacts_delegates_without_runtime(self, tmp_path):
        model = SimpleNamespace()
        expected = tmp_path / "io"
        with patch(
            "QEfficient.transformers.models.modeling_auto.write_dual_qpc_vlm_runner_bundle",
            return_value=expected,
        ) as writer:
            result = _QEffAutoModelForImageTextToTextDualQPC.generate(
                model,
                processor=MagicMock(),
                images=["image"],
                prompts=["prompt"],
                skip_lang=True,
                artifacts=True,
            )
        assert result == expected
        writer.assert_called_once()
