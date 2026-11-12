# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    CohereAsrConfig,
    CohereAsrForConditionalGeneration,
)
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.models.cohere_asr.processing_cohere_asr import CohereAsrProcessor
from transformers.models.parakeet.modeling_parakeet import ParakeetEncoder

from QEfficient.transformers.models.cohere_asr.modeling_cohere_asr import (
    QEffParakeetEncoderAttention,
    QEffParakeetEncoderSubsamplingConv2D,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSpeechSeq2Seq
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import get_padding_shape_from_config, hf_download
from QEfficient.utils._utils import load_hf_processor
from QEfficient.utils.constants import Constants

from ..check_model_results import dump_and_compare_results

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/audio_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    test_models = config_data["speech_seq2seq_models"]


def load_seq2seq_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    kwargs = {
        "attn_implementation": "eager",
        "low_cpu_mem_usage": False,
        "dtype": torch.float32,
    }
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "use_cache"):
        kwargs["use_cache"] = True
    n_layer = model_config.get("n_layer", -1)
    if n_layer != -1:
        if hasattr(config, "encoder_config") or hasattr(config, "decoder_config"):
            for layer_attr in ("num_hidden_layers", "decoder_layers", "encoder_layers"):
                if hasattr(config, layer_attr):
                    setattr(config, layer_attr, n_layer)
            if hasattr(config, "encoder_config") and hasattr(config.encoder_config, "num_hidden_layers"):
                config.encoder_config.num_hidden_layers = n_layer
            if hasattr(config, "decoder_config") and hasattr(config.decoder_config, "num_hidden_layers"):
                config.decoder_config.num_hidden_layers = n_layer
            kwargs["config"] = config
        else:
            for layer_kwarg in ("num_hidden_layers", "decoder_layers", "encoder_layers"):
                if hasattr(config, layer_kwarg):
                    kwargs[layer_kwarg] = n_layer

    model_hf = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        **kwargs,
    )
    if model_hf.config.decoder_start_token_id is None:
        model_hf.config.decoder_start_token_id = model_hf.generation_config.decoder_start_token_id
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def get_feature_lengths(processed_inputs):
    """Return encoder feature lengths across speech processor output contracts."""
    feature_lengths = getattr(processed_inputs, "length", None)
    if feature_lengths is not None:
        return feature_lengths.to(dtype=torch.int64)

    attention_mask = getattr(processed_inputs, "attention_mask", None)
    if attention_mask is not None:
        return attention_mask.sum(dim=-1, dtype=torch.int64)

    raise ValueError("Speech processor must return attention_mask or length")


def prepare_qeff_input_features(input_features, config):
    """Match the runtime feature layout expected by each speech architecture."""
    if not hasattr(config, "encoder_config"):
        return input_features.transpose(1, 2)
    return input_features


@pytest.fixture(scope="module")
def cohere_asr_qeff_model():
    encoder_config = {
        "model_type": "parakeet_encoder",
        "num_mel_bins": 8,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "subsampling_conv_channels": 2,
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "subsampling_factor": 8,
        "conv_kernel_size": 3,
    }
    config = CohereAsrConfig(
        encoder_config=encoder_config,
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        decoder_start_token_id=4,
    )
    return QEFFAutoModelForSpeechSeq2Seq(CohereAsrForConditionalGeneration(config)).model.eval()


def test_cohere_asr_subsampling_lengths_match_parakeet(cohere_asr_qeff_model):
    feature_lengths = torch.tensor([1, 7, 8, 9, 15, 16, 17, 2499, 2500, 2501, 3499, 3500])
    native_lengths = ParakeetEncoder._get_subsampling_output_length(
        cohere_asr_qeff_model.model.encoder, feature_lengths
    )

    assert torch.equal(cohere_asr_qeff_model._get_encoder_output_lengths(feature_lengths), native_lengths)


def test_cohere_asr_feature_lengths_build_per_batch_encoder_mask(cohere_asr_qeff_model, monkeypatch):
    encoder = cohere_asr_qeff_model.model.encoder
    original_forward = encoder.forward
    captured = {}

    def forward_with_mask(input_features, attention_mask=None, **kwargs):
        captured["attention_mask"] = attention_mask.clone()
        return original_forward(input_features, attention_mask=attention_mask, **kwargs)

    class Decoder(torch.nn.Module):
        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(
                last_hidden_state=torch.zeros((*input_ids.shape, cohere_asr_qeff_model.config.hidden_size)),
                past_key_values=None,
            )

    monkeypatch.setattr(encoder, "forward", forward_with_mask)
    monkeypatch.setattr(cohere_asr_qeff_model.model, "decoder", Decoder())
    cohere_asr_qeff_model(
        input_features=torch.randn(2, 8, 16),
        input_ids=torch.tensor([[4], [4]]),
        position_ids=torch.zeros((2, 1), dtype=torch.long),
        feature_lengths=torch.tensor([8, 13]),
    )

    expected = torch.tensor([[True] * 9 + [False] * 7, [True] * 14 + [False] * 2])
    assert torch.equal(captured["attention_mask"], expected)


def test_cohere_asr_decode_keeps_cross_attention_length_from_feature_lengths(cohere_asr_qeff_model, monkeypatch):
    encoder = cohere_asr_qeff_model.model.encoder
    hidden_size = cohere_asr_qeff_model.config.encoder_config.hidden_size
    captured = []

    def encoder_forward(_input_features, **_kwargs):
        return SimpleNamespace(last_hidden_state=torch.zeros((1, 2, hidden_size)))

    class Decoder(torch.nn.Module):
        def forward(self, input_ids, encoder_attention_mask, **_kwargs):
            captured.append(encoder_attention_mask.clone())
            return SimpleNamespace(
                last_hidden_state=torch.zeros((*input_ids.shape, cohere_asr_qeff_model.config.hidden_size)),
                past_key_values=None,
            )

    monkeypatch.setattr(encoder, "forward", encoder_forward)
    monkeypatch.setattr(cohere_asr_qeff_model.model, "decoder", Decoder())
    feature_lengths = torch.tensor([8])
    for input_features in (torch.randn(1, 8, 16), torch.zeros(1, 8, 1)):
        cohere_asr_qeff_model(
            input_features=input_features,
            input_ids=torch.tensor([[4]]),
            position_ids=torch.zeros((1, 1), dtype=torch.long),
            feature_lengths=feature_lengths,
        )

    assert len(captured) == 2
    assert torch.equal(captured[0], captured[1])


def test_cohere_asr_padding_preserves_valid_encoder_outputs_and_logits(cohere_asr_qeff_model):
    valid_features = torch.randn(1, 8, 9)
    padded_features = torch.zeros(1, 8, 16)
    padded_features[:, :, :9] = valid_features
    feature_lengths = torch.tensor([9])

    encoder = cohere_asr_qeff_model.model.encoder
    valid_encoder_outputs = encoder(
        valid_features.transpose(1, 2), attention_mask=torch.ones((1, 9), dtype=torch.bool)
    ).last_hidden_state
    padded_encoder_outputs = encoder(
        padded_features.transpose(1, 2),
        attention_mask=torch.arange(16).unsqueeze(0) < feature_lengths.unsqueeze(1),
    ).last_hidden_state
    assert torch.allclose(valid_encoder_outputs, padded_encoder_outputs, atol=1e-5, rtol=1e-5)

    model_inputs = {"input_ids": torch.tensor([[4]]), "position_ids": torch.zeros((1, 1), dtype=torch.long)}
    valid_logits = cohere_asr_qeff_model(
        input_features=valid_features,
        feature_lengths=feature_lengths,
        past_key_values=[[torch.zeros((1, 2, 4, 4)) for _ in range(2)] + [torch.zeros((1, 2, 2, 4)) for _ in range(2)]],
        **model_inputs,
    ).logits
    padded_logits = cohere_asr_qeff_model(
        input_features=padded_features,
        feature_lengths=feature_lengths,
        past_key_values=[[torch.zeros((1, 2, 4, 4)) for _ in range(2)] + [torch.zeros((1, 2, 2, 4)) for _ in range(2)]],
        **model_inputs,
    ).logits
    assert torch.allclose(valid_logits, padded_logits, atol=1e-5, rtol=1e-5)


def test_cohere_asr_large_padding_keeps_parakeet_attention_and_logits_finite():
    encoder_config = {
        "model_type": "parakeet_encoder",
        "num_mel_bins": 8,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 3504,
        "subsampling_conv_channels": 2,
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "subsampling_factor": 8,
        "conv_kernel_size": 3,
    }
    config = CohereAsrConfig(
        encoder_config=encoder_config,
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        decoder_start_token_id=4,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    native_model = CohereAsrForConditionalGeneration(config).eval()
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(deepcopy(native_model)).model.eval()

    valid_length = 80
    padded_length = 3504
    valid_features = torch.randn(1, valid_length, config.encoder_config.num_mel_bins)
    padded_features = torch.zeros(1, padded_length, config.encoder_config.num_mel_bins)
    padded_features[:, :valid_length] = valid_features
    padded_attention_mask = torch.arange(padded_length).unsqueeze(0) < valid_length
    encoder_output_length = ParakeetEncoder._get_subsampling_output_length(
        native_model.model.encoder, torch.tensor([padded_length])
    ).item()
    valid_encoder_output_length = ParakeetEncoder._get_subsampling_output_length(
        native_model.model.encoder, torch.tensor([valid_length])
    ).item()

    captured_attention_outputs = []
    hooks = []
    assert isinstance(qeff_model.model.encoder.subsampling, QEffParakeetEncoderSubsamplingConv2D)
    for encoder_layer in qeff_model.model.encoder.layers:
        assert isinstance(encoder_layer.self_attn, QEffParakeetEncoderAttention)
        hooks.append(
            encoder_layer.self_attn.register_forward_hook(
                lambda _module, _inputs, output: captured_attention_outputs.append(output[0].detach())
            )
        )

    try:
        with torch.no_grad():
            native_subsampled_states = native_model.model.encoder.subsampling(valid_features.repeat(2, 1, 1))
            qeff_subsampled_states = qeff_model.model.encoder.subsampling(
                padded_features.repeat(2, 1, 1), padded_attention_mask.repeat(2, 1)
            )
            native_encoder_states = native_model.model.encoder(
                valid_features.repeat(2, 1, 1), attention_mask=torch.ones((2, valid_length), dtype=torch.bool)
            ).last_hidden_state
            qeff_encoder_states = qeff_model.model.encoder(
                padded_features.repeat(2, 1, 1), attention_mask=padded_attention_mask.repeat(2, 1)
            ).last_hidden_state
    finally:
        for hook in hooks:
            hook.remove()

    assert qeff_encoder_states.shape[1] == encoder_output_length
    assert torch.isfinite(qeff_encoder_states).all()
    assert torch.allclose(
        native_subsampled_states,
        qeff_subsampled_states[:, :valid_encoder_output_length],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.equal(
        qeff_subsampled_states[:, valid_encoder_output_length:],
        torch.zeros_like(qeff_subsampled_states[:, valid_encoder_output_length:]),
    )
    assert torch.allclose(
        native_encoder_states,
        qeff_encoder_states[:, :valid_encoder_output_length],
        atol=1e-5,
        rtol=1e-5,
    )
    assert len(captured_attention_outputs) == len(qeff_model.model.encoder.layers)
    for attention_output in captured_attention_outputs:
        assert torch.allclose(
            attention_output[:, valid_encoder_output_length:],
            torch.zeros_like(attention_output[:, valid_encoder_output_length:]),
        )

    class Tokenizer:
        def __init__(self):
            self.token_ids = {}

        def convert_tokens_to_ids(self, tokens):
            return [self.token_ids.setdefault(token, len(self.token_ids) + 1) for token in tokens]

    processor = SimpleNamespace(tokenizer=Tokenizer())
    english_prompt = CohereAsrProcessor.get_decoder_prompt_ids(processor, "en")
    arabic_prompt = CohereAsrProcessor.get_decoder_prompt_ids(processor, "ar", punctuation=False)
    input_ids = torch.tensor([english_prompt, arabic_prompt])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)

    with torch.no_grad():
        native_logits = native_model(
            input_features=valid_features.repeat(2, 1, 1),
            attention_mask=torch.ones((2, valid_length), dtype=torch.bool),
            decoder_input_ids=input_ids,
            decoder_position_ids=position_ids,
        ).logits
        qeff_cache = [
            [
                torch.zeros((2, 2, config.max_position_embeddings, 4)),
                torch.zeros((2, 2, config.max_position_embeddings, 4)),
                torch.zeros((2, 2, encoder_output_length, 4)),
                torch.zeros((2, 2, encoder_output_length, 4)),
            ]
        ]
        qeff_logits = qeff_model(
            input_features=padded_features.transpose(1, 2).repeat(2, 1, 1),
            feature_lengths=torch.full((2,), valid_length),
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=qeff_cache,
        ).logits

    assert torch.isfinite(native_logits).all()
    assert torch.isfinite(qeff_logits).all()
    assert torch.allclose(native_logits, qeff_logits, atol=1e-5, rtol=1e-5)


def test_cohere_asr_processor_prompts_include_requested_languages():
    class Tokenizer:
        def convert_tokens_to_ids(self, tokens):
            self.tokens = tokens
            return list(range(len(tokens)))

    tokenizer = Tokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer)

    assert CohereAsrProcessor.get_decoder_prompt_ids(processor, "en") == list(range(10))
    assert tokenizer.tokens[4:7] == ["<|en|>", "<|en|>", "<|pnc|>"]
    assert CohereAsrProcessor.get_decoder_prompt_ids(processor, "ar", punctuation=False) == list(range(10))
    assert tokenizer.tokens[4:7] == ["<|ar|>", "<|ar|>", "<|nopnc|>"]


def test_cohere_asr_decode_reuses_cross_attention_cache(cohere_asr_qeff_model):
    cache = [[]]
    for cache_type in ["self", "cross"]:
        for _ in ["key", "value"]:
            cache[0].append(torch.zeros((1, 2, 4 if cache_type == "self" else 2, 4)))

    prefill_outputs = cohere_asr_qeff_model(
        input_features=torch.randn(1, 8, 16),
        feature_lengths=torch.tensor([9]),
        input_ids=torch.tensor([[4]]),
        position_ids=torch.tensor([[0]]),
        past_key_values=cache,
    )
    prefill_cross_key = prefill_outputs.past_key_values[0][2].clone()
    prefill_cross_value = prefill_outputs.past_key_values[0][3].clone()
    decode_outputs = cohere_asr_qeff_model(
        input_features=torch.zeros(1, 8, 1),
        feature_lengths=torch.tensor([9]),
        input_ids=torch.tensor([[5]]),
        position_ids=torch.tensor([[1]]),
        past_key_values=prefill_outputs.past_key_values,
    )

    assert decode_outputs.logits.shape == (1, 1, 32)
    assert torch.equal(prefill_cross_key, decode_outputs.past_key_values[0][2])
    assert torch.equal(prefill_cross_value, decode_outputs.past_key_values[0][3])


def test_speech_static_export_preserves_zero_argument_model_hooks(cohere_asr_qeff_model, monkeypatch, tmp_path):
    wrapper = QEFFAutoModelForSpeechSeq2Seq(deepcopy(cohere_asr_qeff_model))
    calls = []

    monkeypatch.setattr(wrapper.model, "get_dummy_inputs", lambda **kwargs: calls.append(("dummy", kwargs)) or {})
    monkeypatch.setattr(wrapper.model, "get_onnx_dynamic_axes", lambda: calls.append(("axes", {})) or {})
    monkeypatch.setattr(wrapper.model, "get_output_names", lambda: [])
    monkeypatch.setattr(wrapper, "_export", lambda *_args, **_kwargs: tmp_path / "model.onnx")

    wrapper.export(export_dir=tmp_path)

    assert calls == [("dummy", {}), ("axes", {})]


def test_speech_static_compile_does_not_forward_none_full_batch_size(cohere_asr_qeff_model, monkeypatch):
    wrapper = QEFFAutoModelForSpeechSeq2Seq(deepcopy(cohere_asr_qeff_model))
    specialization_kwargs = {}

    def get_specializations(batch_size, encoder_ctx_len, ctx_len, **kwargs):
        specialization_kwargs.update(kwargs)
        return [], {}

    monkeypatch.setattr(wrapper.model, "get_specializations", get_specializations)
    monkeypatch.setattr(wrapper.model, "get_export_hash_params", None)
    monkeypatch.setattr(wrapper.model, "get_output_names", lambda: [])
    monkeypatch.setattr(wrapper, "_compile", lambda **_kwargs: "qpc")

    assert wrapper.compile(onnx_path="model.onnx") == "qpc"
    assert "full_batch_size" not in specialization_kwargs


def test_cohere_asr_export_hash_params_track_graph_shapes(cohere_asr_qeff_model, monkeypatch, tmp_path):
    wrapper = QEFFAutoModelForSpeechSeq2Seq(deepcopy(cohere_asr_qeff_model))
    monkeypatch.setattr(wrapper, "_export", lambda *_args, **_kwargs: tmp_path / "model.onnx")

    wrapper.export(export_dir=tmp_path, batch_size=1, encoder_ctx_len=2)
    first_export_params = deepcopy(wrapper.hash_params["speech_export"])
    wrapper.export(export_dir=tmp_path, batch_size=2, encoder_ctx_len=3)

    assert first_export_params != wrapper.hash_params["speech_export"]
    assert wrapper.hash_params["speech_export"] == {"batch_size": 2, "encoder_ctx_len": 3}


def test_cohere_asr_rejects_encoder_output_larger_than_cross_cache(cohere_asr_qeff_model):
    cache = [
        [
            torch.zeros((1, 2, 4, 4)),
            torch.zeros((1, 2, 4, 4)),
            torch.zeros((1, 2, 1, 4)),
            torch.zeros((1, 2, 1, 4)),
        ]
    ]

    with pytest.raises(ValueError, match="exceeds the configured encoder context length"):
        cohere_asr_qeff_model(
            input_features=torch.randn(1, 8, 16),
            feature_lengths=torch.tensor([16]),
            input_ids=torch.tensor([[4]]),
            position_ids=torch.zeros((1, 1), dtype=torch.int64),
            past_key_values=cache,
            use_cache=True,
        )


def test_cohere_asr_native_and_qeff_logits_match_with_cached_decode():
    encoder_config = {
        "model_type": "parakeet_encoder",
        "num_mel_bins": 8,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "subsampling_conv_channels": 2,
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "subsampling_factor": 8,
        "conv_kernel_size": 3,
    }
    config = CohereAsrConfig(
        encoder_config=encoder_config,
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        decoder_start_token_id=4,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    native_model = CohereAsrForConditionalGeneration(config).eval()
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(deepcopy(native_model)).model.eval()
    native_model.config._attn_implementation = "eager"
    qeff_model.config._attn_implementation = "eager"

    padded_features = torch.randn(2, 8, 24)
    feature_lengths = torch.tensor([16, 17])
    frame_mask = torch.arange(padded_features.shape[-1]).unsqueeze(0) < feature_lengths.unsqueeze(1)
    input_ids = torch.tensor([[4], [4]])
    position_ids = torch.zeros_like(input_ids)
    native_subsampled_lengths = ParakeetEncoder._get_subsampling_output_length(
        native_model.model.encoder, feature_lengths
    )
    assert torch.equal(native_subsampled_lengths, torch.tensor([2, 3]))
    encoder_output_length = ParakeetEncoder._get_subsampling_output_length(
        native_model.model.encoder, torch.tensor([padded_features.shape[-1]])
    ).item()
    qeff_cache = [
        [
            torch.zeros((2, 2, 4, 4)),
            torch.zeros((2, 2, 4, 4)),
            torch.zeros((2, 2, encoder_output_length, 4)),
            torch.zeros((2, 2, encoder_output_length, 4)),
        ]
    ]
    native_encoder_outputs = native_model.model.encoder(padded_features.transpose(1, 2), attention_mask=frame_mask)
    raw_encoder_hidden_states = native_encoder_outputs.last_hidden_state
    encoder_attention_mask = (
        torch.arange(raw_encoder_hidden_states.shape[1]).unsqueeze(0) < native_subsampled_lengths.unsqueeze(1)
    ).long()
    valid_encoder_output_mask = encoder_attention_mask.bool()
    valid_encoder_state_mask = valid_encoder_output_mask.unsqueeze(-1).expand_as(raw_encoder_hidden_states)
    raw_nan_mask = torch.isnan(raw_encoder_hidden_states)
    assert raw_nan_mask.any()
    assert torch.all(~raw_nan_mask | ~valid_encoder_state_mask)

    sanitized_encoder_hidden_states = torch.where(
        valid_encoder_output_mask.unsqueeze(-1), raw_encoder_hidden_states, torch.zeros_like(raw_encoder_hidden_states)
    )
    assert torch.equal(
        sanitized_encoder_hidden_states[valid_encoder_state_mask], raw_encoder_hidden_states[valid_encoder_state_mask]
    )
    assert torch.equal(
        sanitized_encoder_hidden_states[~valid_encoder_state_mask],
        torch.zeros_like(sanitized_encoder_hidden_states[~valid_encoder_state_mask]),
    )

    native_decoder_outputs = native_model.model.decoder(
        input_ids=input_ids,
        position_ids=position_ids,
        encoder_hidden_states=sanitized_encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_values=EncoderDecoderCache(DynamicCache(), DynamicCache()),
        use_cache=True,
    )
    native_prefill_logits = native_model.proj_out(native_decoder_outputs.last_hidden_state)
    qeff_prefill = qeff_model(
        input_features=padded_features,
        feature_lengths=feature_lengths,
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=qeff_cache,
        use_cache=True,
    )

    assert torch.isfinite(native_prefill_logits).all()
    assert torch.isfinite(qeff_prefill.logits).all()
    prefill_max_abs_diff = (native_prefill_logits - qeff_prefill.logits).abs().max().item()
    print(f"prefill max absolute difference: {prefill_max_abs_diff:.3e}")
    assert prefill_max_abs_diff <= 1e-5, f"prefill max absolute difference: {prefill_max_abs_diff:.3e}"

    decode_input_ids = torch.tensor([[5], [5]])
    decode_position_ids = torch.ones_like(decode_input_ids)
    native_decode_outputs = native_model.model.decoder(
        input_ids=decode_input_ids,
        position_ids=decode_position_ids,
        encoder_hidden_states=sanitized_encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_values=native_decoder_outputs.past_key_values,
        use_cache=True,
    )
    native_decode_logits = native_model.proj_out(native_decode_outputs.last_hidden_state)
    qeff_decode = qeff_model(
        input_features=torch.zeros(2, 8, 1),
        feature_lengths=feature_lengths,
        input_ids=decode_input_ids,
        position_ids=decode_position_ids,
        past_key_values=qeff_prefill.past_key_values,
        use_cache=True,
    )

    assert torch.isfinite(native_decode_logits).all()
    assert torch.isfinite(qeff_decode.logits).all()
    decode_max_abs_diff = (native_decode_logits - qeff_decode.logits).abs().max().item()
    print(f"cached decode max absolute difference: {decode_max_abs_diff:.3e}")
    assert decode_max_abs_diff <= 1e-5, f"cached decode max absolute difference: {decode_max_abs_diff:.3e}"


def test_cohere_asr_qeff_pytorch_and_ort_match_with_cached_decode(tmp_path, cohere_asr_qeff_model):
    qeff_wrapper = QEFFAutoModelForSpeechSeq2Seq(deepcopy(cohere_asr_qeff_model))
    qeff_model = qeff_wrapper.model.eval()
    padded_features = torch.randn(2, 8, 24)
    feature_lengths = torch.tensor([16, 17])
    input_ids = torch.tensor([[4], [4]])
    position_ids = torch.zeros_like(input_ids)
    output_names = qeff_model.get_output_names()
    encoder_output_length = ParakeetEncoder._get_subsampling_output_length(
        qeff_model.model.encoder, torch.tensor([padded_features.shape[-1]])
    ).item()

    def make_zero_cache():
        return [
            [
                torch.zeros((2, 2, 4, 4)),
                torch.zeros((2, 2, 4, 4)),
                torch.zeros((2, 2, encoder_output_length, 4)),
                torch.zeros((2, 2, encoder_output_length, 4)),
            ]
        ]

    with torch.no_grad():
        qeff_prefill = qeff_model(
            input_features=padded_features,
            feature_lengths=feature_lengths,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=make_zero_cache(),
            use_cache=True,
        )
        qeff_prefill_outputs = [qeff_prefill.logits.clone()] + [
            cache_tensor.clone() for layer_cache in qeff_prefill.past_key_values for cache_tensor in layer_cache
        ]
        qeff_decode = qeff_model(
            input_features=torch.zeros(2, 8, 1),
            feature_lengths=feature_lengths,
            input_ids=torch.tensor([[5], [5]]),
            position_ids=torch.ones_like(input_ids),
            past_key_values=qeff_prefill.past_key_values,
            use_cache=True,
        )
        qeff_decode_outputs = [qeff_decode.logits.clone()] + [
            cache_tensor.clone() for layer_cache in qeff_decode.past_key_values for cache_tensor in layer_cache
        ]
    assert len(qeff_prefill_outputs) == len(output_names)
    assert len(qeff_decode_outputs) == len(output_names)
    assert all(torch.isfinite(output).all() for output in qeff_prefill_outputs + qeff_decode_outputs)
    assert torch.count_nonzero(qeff_prefill_outputs[1][:, :, 0, :]).item() > 0
    assert not torch.equal(qeff_prefill_outputs[1][:, :, 1, :], qeff_decode_outputs[1][:, :, 1, :])
    assert torch.equal(qeff_prefill_outputs[3], qeff_decode_outputs[3])
    assert torch.equal(qeff_prefill_outputs[4], qeff_decode_outputs[4])

    onnx_path = qeff_wrapper.export(export_dir=tmp_path)
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    feature_lengths_input = next(value for value in onnx_model.graph.input if value.name == "feature_lengths")
    assert feature_lengths_input.type.tensor_type.elem_type == onnx.TensorProto.INT64
    assert feature_lengths_input.type.tensor_type.shape.dim[0].dim_param == "batch_size"

    added_initializers = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            constant_value = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(onnx_path))
            if len(constant_value.shape) == 0 and constant_value.item() == 2147483647:
                added_initializers[node.output[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                    np.array(0, constant_value.dtype)
                )

    session_options = onnxruntime.SessionOptions()
    for name, value in added_initializers.items():
        session_options.add_initializer(name, value)
    session = onnxruntime.InferenceSession(onnx_path, session_options)

    def run_ort(input_features, input_ids, position_ids, past_key_values):
        ort_inputs = {
            "input_features": input_features.numpy(),
            "feature_lengths": feature_lengths.numpy(),
            "input_ids": input_ids.numpy(),
            "position_ids": position_ids.numpy(),
        }
        for output_name, cache_tensor in zip(
            output_names[1:], [cache for layer_cache in past_key_values for cache in layer_cache]
        ):
            ort_inputs[output_name.removesuffix("_RetainedState")] = cache_tensor.numpy()
        return session.run(output_names, ort_inputs)

    ort_initial_cache = make_zero_cache()
    assert all(
        torch.count_nonzero(cache_tensor).item() == 0
        for layer_cache in ort_initial_cache
        for cache_tensor in layer_cache
    )
    ort_prefill_outputs = [
        output.copy() for output in run_ort(padded_features, input_ids, position_ids, ort_initial_cache)
    ]
    ort_decode_outputs = [
        output.copy()
        for output in run_ort(
            torch.zeros(2, 8, 1),
            torch.tensor([[5], [5]]),
            torch.ones_like(input_ids),
            [[torch.from_numpy(output) for output in ort_prefill_outputs[1:]]],
        )
    ]

    assert np.count_nonzero(ort_prefill_outputs[1][:, :, 0, :]) > 0
    assert not np.array_equal(ort_prefill_outputs[1][:, :, 1, :], ort_decode_outputs[1][:, :, 1, :])
    assert np.array_equal(ort_prefill_outputs[3], ort_decode_outputs[3])
    assert np.array_equal(ort_prefill_outputs[4], ort_decode_outputs[4])

    for output_kind, qeff_outputs, ort_outputs in [
        ("prefill", qeff_prefill_outputs, ort_prefill_outputs),
        ("decode", qeff_decode_outputs, ort_decode_outputs),
    ]:
        for output_name, qeff_output, ort_output in zip(output_names, qeff_outputs, ort_outputs):
            assert np.isfinite(ort_output).all()
            max_abs_diff = np.max(np.abs(qeff_output.numpy() - ort_output))
            print(f"{output_kind} {output_name} max absolute difference: {max_abs_diff:.3e}")
            assert max_abs_diff <= 1e-5, f"{output_kind} {output_name} max absolute difference: {max_abs_diff:.3e}"


def test_speech_generate_consumes_decoder_prompt_and_retains_feature_lengths(
    cohere_asr_qeff_model, monkeypatch, tmp_path
):
    batch_size = 2
    vocab_size = cohere_asr_qeff_model.config.vocab_size
    eos_token_id = cohere_asr_qeff_model.config.eos_token_id

    class Session:
        input_names = ["input_features", "feature_lengths", "input_ids", "position_ids"]
        output_names = ["logits"]
        bindings = [SimpleNamespace(dims=(batch_size, 8, 16))]
        binding_index_map = {"input_features": 0}

        def __init__(self, *_args, **_kwargs):
            self.calls = []
            self.output_shapes = []

        def skip_buffers(self, _names):
            pass

        def set_buffers(self, _outputs):
            pass

        def run(self, inputs):
            self.calls.append({name: value.copy() for name, value in inputs.items()})
            logits = np.zeros(
                (inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], vocab_size), dtype=np.float32
            )
            if len(self.calls) == 4:
                logits[0, ..., eos_token_id] = 1.0
                logits[1, ..., 6] = 1.0
            elif len(self.calls) == 5:
                logits[0, ..., 7] = 1.0
                logits[1, ..., eos_token_id] = 1.0
            else:
                logits[..., 5] = 1.0
            self.output_shapes.append(logits.shape)
            return {"logits": logits}

    wrapper = object.__new__(QEFFAutoModelForSpeechSeq2Seq)
    wrapper.model = cohere_asr_qeff_model
    wrapper.qpc_path = tmp_path
    wrapper.onnx_path = tmp_path / "model.onnx"
    wrapper.qpc_session = None
    monkeypatch.setattr("QEfficient.transformers.models.modeling_auto.QAICInferenceSession", Session)

    attention_mask = torch.tensor([[1] * 8 + [0] * 8, [1] * 13 + [0] * 3])
    input_features = torch.randn(2, 13, 8)
    decoder_prompt = torch.tensor([[1, 2, 3], [1, 4, 3]], dtype=torch.int64)
    result = wrapper.generate(
        inputs={
            "input_features": input_features,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_prompt,
        },
        generation_len=3,
    )

    assert len(wrapper.qpc_session.calls) == 5
    assert wrapper.qpc_session.output_shapes == [(batch_size, 1, vocab_size)] * 5
    assert [call["input_ids"].tolist() for call in wrapper.qpc_session.calls] == [
        [[1], [1]],
        [[2], [4]],
        [[3], [3]],
        [[5], [5]],
        [[eos_token_id], [6]],
    ]
    assert [call["position_ids"].tolist() for call in wrapper.qpc_session.calls] == [
        [[0], [0]],
        [[1], [1]],
        [[2], [2]],
        [[3], [3]],
        [[4], [4]],
    ]
    expected_features = torch.nn.functional.pad(input_features.transpose(1, 2), (0, 3)).numpy().astype(np.float16)
    assert np.array_equal(wrapper.qpc_session.calls[0]["input_features"], expected_features)
    for call in wrapper.qpc_session.calls[1:]:
        assert np.count_nonzero(call["input_features"]) == 0
    for call in wrapper.qpc_session.calls:
        assert np.array_equal(call["feature_lengths"], np.array([8, 13], dtype=np.int64))
    assert np.array_equal(
        result.generated_ids,
        np.array([[1, 2, 3, 5, eos_token_id, eos_token_id], [1, 4, 3, 5, 6, eos_token_id]], dtype=np.int64),
    )


def test_speech_generate_defaults_to_single_decoder_start_token(cohere_asr_qeff_model, monkeypatch, tmp_path):
    batch_size = 2
    vocab_size = cohere_asr_qeff_model.config.vocab_size
    start_token_id = cohere_asr_qeff_model.config.decoder_start_token_id

    class Session:
        input_names = ["input_features", "input_ids", "position_ids"]
        output_names = ["logits"]
        bindings = [SimpleNamespace(dims=(batch_size, 8, 16))]
        binding_index_map = {"input_features": 0}

        def __init__(self, *_args, **_kwargs):
            self.calls = []

        def skip_buffers(self, _names):
            pass

        def set_buffers(self, _outputs):
            pass

        def run(self, inputs):
            self.calls.append({name: value.copy() for name, value in inputs.items()})
            logits = np.zeros((batch_size, 1, vocab_size), dtype=np.float32)
            logits[..., 5] = 1.0
            return {"logits": logits}

    wrapper = object.__new__(QEFFAutoModelForSpeechSeq2Seq)
    wrapper.model = cohere_asr_qeff_model
    wrapper.qpc_path = tmp_path
    wrapper.onnx_path = tmp_path / "model.onnx"
    wrapper.qpc_session = None
    monkeypatch.setattr("QEfficient.transformers.models.modeling_auto.QAICInferenceSession", Session)

    result = wrapper.generate(
        inputs={
            "input_features": torch.randn(2, 8, 16),
            "attention_mask": torch.ones((2, 16), dtype=torch.int64),
        },
        generation_len=1,
    )

    assert len(wrapper.qpc_session.calls) == 1
    assert wrapper.qpc_session.calls[0]["input_ids"].tolist() == [[start_token_id], [start_token_id]]
    assert wrapper.qpc_session.calls[0]["position_ids"].tolist() == [[0], [0]]
    assert result.generated_ids.tolist() == [[start_token_id, 5], [start_token_id, 5]]


def run_seq2seq_pytorch_hf(
    model, processor: AutoProcessor, inputs: np.ndarray, sample_rate: int, generation_len: int
) -> List[str]:
    """
    Run pytorch inference on model

    ``Mandatory`` Args:
        :model: The transformed PyTorch model used for generating transcripts
        :processor: autoprocessor to process inputs and decode logits
        :inputs (np.ndarray): inputs to run the execution.
        :sample_rate (int): sampling rate at which input audio is stored in inputs (needed for processor)
        :generation_len (int): length upto which to generate

    Returns:
        torch.Tensor: A list of output features generated by the model for each prompt.
    """
    seq_len = 1
    batch_size = 1

    # prepare inputs
    input_features = processor(inputs, sampling_rate=sample_rate, return_tensors="pt").input_features
    if hasattr(model.config, "encoder_config"):
        # Composite (encoder_config + decoder_config) architectures such as CohereAsr expose their
        # plain HF forward()/generate() in the encoder's native (batch, seq_len, num_mel_bins) layout,
        # unlike Whisper-style configs whose processor output is already mel-bins-major.
        input_features = input_features.transpose(1, 2)
    decoder_input_ids = torch.ones((batch_size, seq_len), dtype=torch.int64) * model.config.decoder_start_token_id
    decoder_position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(batch_size, 1)

    model_inputs = dict(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        decoder_position_ids=decoder_position_ids,
    )

    # TODO: temporary hack to nullify effect of KVCacheTransform add this as setup_module in pytest
    # encoder run
    outputs = model(**model_inputs)

    # array to hold generated tokens
    generated_ids = np.full((batch_size, generation_len + 1), processor.tokenizer.eos_token_id)
    generated_ids[:, 0] = [model.config.decoder_start_token_id]
    logits = outputs["logits"]
    next_token = logits.argmax(-1)
    generated_ids[:, 1] = next_token.squeeze(1)

    model_inputs["encoder_outputs"] = SimpleNamespace(
        last_hidden_state=outputs["encoder_last_hidden_state"],
        attention_mask=None,
        hidden_states=None,
        attentions=None,
    )
    pkv = outputs.get("past_key_values")
    if pkv is not None:
        model_inputs["past_key_values"] = pkv

    # Track full decoded sequence for models that return no KV cache (e.g. CohereAsr plain HF).
    # When past_key_values is absent, we must pass the full growing sequence each step so the
    # decoder has its own token history; Whisper-style models with KV cache only need next_token.
    accumulated_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.int64)

    if next_token[0][0] == processor.tokenizer.eos_token_id:
        return generated_ids[0]

    if pkv is not None:
        model_inputs["decoder_input_ids"] = next_token
        model_inputs["decoder_position_ids"] = model_inputs["decoder_position_ids"][:, -1:] + 1
    else:
        accumulated_ids = torch.cat([accumulated_ids, next_token], dim=1)
        model_inputs["decoder_input_ids"] = accumulated_ids
        model_inputs["decoder_position_ids"] = torch.arange(accumulated_ids.shape[1], dtype=torch.int64).unsqueeze(0)

    for num_tokens in range(1, generation_len):
        outputs = model(**model_inputs)
        logits = outputs["logits"]
        next_token = logits.argmax(-1)[:, -1:]
        generated_ids[:, num_tokens + 1] = next_token.squeeze(1)

        if next_token[0][0] == processor.tokenizer.eos_token_id:
            break

        pkv = outputs.get("past_key_values")
        if pkv is not None:
            model_inputs["past_key_values"] = pkv
            model_inputs["decoder_input_ids"] = next_token
            model_inputs["decoder_position_ids"] = model_inputs["decoder_position_ids"][:, -1:] + 1
        else:
            # No KV cache: accumulate tokens and pass the full growing sequence
            accumulated_ids = torch.cat([accumulated_ids, next_token], dim=1)
            model_inputs["decoder_input_ids"] = accumulated_ids
            model_inputs["decoder_position_ids"] = torch.arange(accumulated_ids.shape[1], dtype=torch.int64).unsqueeze(
                0
            )

    return generated_ids[0]


def run_seq2seq_pytorch_with_kv(
    model,
    processor: AutoProcessor,
    inputs: np.ndarray,
    sample_rate: int,
    generation_len: int,
    cross_ctx_len: Optional[int] = None,
) -> List[str]:
    """
    Run pytorch inference on model

    ``Mandatory`` Args:
        :model: The transformed PyTorch model used for generating transcripts
        :processor: autoprocessor to process inputs and decode logits
        :inputs (np.ndarray): inputs to run the execution.
        :sample_rate (int): sampling rate at which input audio is stored in inputs (needed for processor)
        :generation_len (int): length upto which to generate

    ``Optional`` Args:
        :cross_ctx_len (int): cross-attention KV cache context length. Defaults to
            ``config.max_source_positions`` when the config exposes it (Whisper-style);
            composite (encoder_config + decoder_config) architectures such as CohereAsr have
            no such static field and must pass the actual encoder output length instead.

    Returns:
        torch.Tensor: A list of output features generated by the model for each prompt.
    """
    seq_len = 1
    batch_size = 1
    config = model.model.config

    # prepare inputs
    processed_inputs = processor(inputs, sampling_rate=sample_rate, return_tensors="pt")
    input_features = processed_inputs.input_features
    decoder_input_ids = torch.ones((batch_size, seq_len), dtype=torch.int64) * config.decoder_start_token_id
    decoder_position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(batch_size, 1)

    model_inputs = dict(
        input_features=input_features,
        input_ids=decoder_input_ids,
        position_ids=decoder_position_ids,
        past_key_values=[[] for _ in range(config.num_hidden_layers)],
    )
    qpc_input_names = {input_info.name for input_info in model.model.get_inputs_info()}
    if "feature_lengths" in qpc_input_names:
        model_inputs["feature_lengths"] = get_feature_lengths(processed_inputs)
        input_features = prepare_qeff_input_features(input_features, config)
        model_inputs["input_features"] = input_features

    # prepare dummy past kvs and cross kvs
    kv_cache_shape = get_padding_shape_from_config(config, batch_size, generation_len)
    if cross_ctx_len is None:
        cross_ctx_len = config.max_source_positions
    kv_cross_cache_shape = get_padding_shape_from_config(config, batch_size, cross_ctx_len)

    for i in range(config.num_hidden_layers):
        for self_cross in ["self", "cross"]:
            for kv in ["key", "value"]:
                model_inputs["past_key_values"][i].append(
                    torch.zeros(kv_cache_shape if self_cross == "self" else kv_cross_cache_shape, dtype=torch.float32)
                )

    # encoder run
    outputs = model.model(**model_inputs)

    # array to hold generated tokens
    generated_ids = np.full((batch_size, generation_len + 1), config.eos_token_id)
    generated_ids[:, 0] = [config.decoder_start_token_id]
    logits = outputs["logits"]
    next_token = logits.argmax(-1)
    generated_ids[:, 1] = next_token.squeeze(1)

    model_inputs["input_features"] = torch.zeros((batch_size, input_features.shape[1], 1), dtype=torch.float32)
    model_inputs["past_key_values"] = outputs["past_key_values"]

    if next_token[0][0] == processor.tokenizer.eos_token_id:
        return generated_ids[0]

    model_inputs["input_ids"] = next_token
    model_inputs["position_ids"] += 1

    for num_tokens in range(1, generation_len):
        outputs = model.model(**model_inputs)
        logits = outputs["logits"]
        next_token = logits.argmax(-1)
        generated_ids[:, num_tokens + 1] = next_token.squeeze(1)

        if next_token[0][0] == processor.tokenizer.eos_token_id:
            break

        model_inputs["input_ids"] = next_token
        model_inputs["position_ids"] += 1
        model_inputs["past_key_values"] = outputs["past_key_values"]

    return generated_ids[0]


def run_seq2seq_ort(
    onnx_path,
    config,
    processor: AutoProcessor,
    inputs: np.ndarray,
    sample_rate: int,
    generation_len: int,
    cross_ctx_len: Optional[int] = None,
) -> List[str]:
    """
    Run onnxruntime inference on model

    ``Mandatory`` Args:
        :model: The transformed PyTorch model used for generating transcripts
        :processor: autoprocessor to process inputs and decode logits
        :inputs (np.ndarray): inputs to run the execution.
        :sample_rate (int): sampling rate at which input audio is stored in inputs (needed for processor)
        :generation_len (int): length upto which to generate

    ``Optional`` Args:
        :cross_ctx_len (int): cross-attention KV cache context length. Defaults to
            ``config.max_source_positions`` when the config exposes it (Whisper-style);
            composite (encoder_config + decoder_config) architectures such as CohereAsr have
            no such static field and must pass the actual encoder output length instead.

    Returns:
        torch.Tensor: A list of output features generated by the model for each prompt.
    """
    seq_len = 1
    batch_size = 1

    # Replace invalid index value for INT32 max to 0 using add_initializer
    m = onnx.load(onnx_path, load_external_data=False)
    # NOTE: OrtValue objects should be kept around until the session is run, hence this dict is required
    added_initializers = {}
    for node in m.graph.node:
        if node.op_type == "Constant":
            np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(onnx_path))
            if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                added_initializers[node.output[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                    np.array(0, np_tensor.dtype)
                )

    session_options = onnxruntime.SessionOptions()
    for name, value in added_initializers.items():
        session_options.add_initializer(name, value)

    session = onnxruntime.InferenceSession(onnx_path, session_options)

    # prepare inputs
    processed_inputs = processor(inputs, sampling_rate=sample_rate, return_tensors="pt")
    input_features = processed_inputs.input_features
    decoder_input_ids = torch.ones((batch_size, seq_len), dtype=torch.int64) * config.decoder_start_token_id
    decoder_position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(batch_size, 1)

    model_inputs = dict(
        input_features=input_features,
        input_ids=decoder_input_ids,
        position_ids=decoder_position_ids,
    )
    session_input_names = {session_input.name for session_input in session.get_inputs()}
    if "feature_lengths" in session_input_names:
        model_inputs["feature_lengths"] = get_feature_lengths(processed_inputs)
        input_features = prepare_qeff_input_features(input_features, config)
        model_inputs["input_features"] = input_features

    # prepare dummy past kvs and cross kvs
    kv_cache_shape = get_padding_shape_from_config(config, batch_size, generation_len)
    if cross_ctx_len is None:
        cross_ctx_len = config.max_source_positions
    kv_cross_cache_shape = get_padding_shape_from_config(config, batch_size, cross_ctx_len)

    pkv_names = []
    for i in range(config.num_hidden_layers):
        for self_cross in ["self", "cross"]:
            for kv in ["key", "value"]:
                pkv_names.append(f"past_{kv}_{self_cross}.{i}_RetainedState")
                model_inputs[f"past_{kv}_{self_cross}.{i}"] = torch.zeros(
                    kv_cache_shape if self_cross == "self" else kv_cross_cache_shape, dtype=torch.float32
                )

    output_names = ["logits"] + pkv_names

    # encoder run
    outputs = session.run(output_names, {k: v.detach().numpy() for k, v in model_inputs.items()})

    # array to hold generated tokens
    generated_ids = np.full((batch_size, generation_len + 1), config.eos_token_id)
    generated_ids[:, 0] = [config.decoder_start_token_id]
    logits = outputs[0]
    next_token = logits.argmax(-1)
    generated_ids[:, 1] = next_token.squeeze(1)

    model_inputs["input_features"] = torch.zeros((batch_size, input_features.shape[1], 1), dtype=torch.float32)
    for i, name in enumerate(pkv_names):
        model_inputs[name.split("_RetainedState")[0]] = outputs[1 + i]

    if next_token[0][0] == processor.tokenizer.eos_token_id:
        return generated_ids[0]

    model_inputs["input_ids"] = next_token
    model_inputs["position_ids"] += 1

    for num_tokens in range(1, generation_len):
        outputs = session.run(
            output_names, {k: (v.detach().numpy() if type(v) is torch.Tensor else v) for k, v in model_inputs.items()}
        )
        logits = outputs[0]
        next_token = logits.argmax(-1)
        generated_ids[:, num_tokens + 1] = next_token.squeeze(1)

        if next_token[0][0] == processor.tokenizer.eos_token_id:
            break

        model_inputs["input_ids"] = next_token
        model_inputs["position_ids"] += 1
        for i, name in enumerate(pkv_names):
            model_inputs[name.split("_RetainedState")[0]] = outputs[1 + i]

    return generated_ids[0]


def check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    ctx_len: int = Constants.CTX_LEN,
    n_layer: int = -1,
    num_devices: int = 1,
    compare_results: Optional[bool] = False,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, ONNX model and the Cloud AI 100 model
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``whisper``
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
    """
    replace_transformers_quantizers()
    model_config = {"model_name": model_name}
    model_config["n_layer"] = n_layer

    model_hf, _ = load_seq2seq_model(model_config)

    processor = load_hf_processor(pretrained_model_name_or_path=model_name)
    batch_size = 1

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    data = ds[0]["audio"]["array"]
    data = data.reshape(-1)
    sample_rate = ds[0]["audio"]["sampling_rate"]
    pytorch_hf_tokens = run_seq2seq_pytorch_hf(model_hf, processor, data, sample_rate, ctx_len)

    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model_hf, pretrained_model_name_or_path=model_name)

    cross_ctx_len = None
    if not hasattr(qeff_model.model.config, "max_source_positions"):
        # Composite (encoder_config + decoder_config) architectures such as CohereAsr expose no
        # static cross-attention context length; derive it from an actual encoder forward pass
        # (a decoder-side max_position_embeddings-like field produces the wrong length here).
        encoder_input_features = processor(
            data, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.transpose(1, 2)
        with torch.no_grad():
            cross_ctx_len = qeff_model.model.model.encoder(encoder_input_features).last_hidden_state.shape[1]

    pytorch_kv_tokens = run_seq2seq_pytorch_with_kv(
        qeff_model, processor, data, sample_rate, ctx_len, cross_ctx_len=cross_ctx_len
    )
    assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
        "Tokens don't match for HF PyTorch model output and KV PyTorch model output"
    )

    qeff_model.export()
    ort_tokens = run_seq2seq_ort(
        qeff_model.onnx_path,
        qeff_model.model.config,
        processor,
        data,
        sample_rate,
        ctx_len,
        cross_ctx_len=cross_ctx_len,
    )
    assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for pytorch output and ort output"

    qeff_model.compile(
        ctx_len=ctx_len,
        encoder_ctx_len=cross_ctx_len,
        num_devices=num_devices,
        batch_size=batch_size,
    )

    exec_info = qeff_model.generate(
        inputs=processor(data, sampling_rate=sample_rate, return_tensors="pt"), generation_len=ctx_len
    )
    cloud_ai_100_tokens = exec_info.generated_ids[0]  # Because we always run for single input and single batch size
    assert (pytorch_kv_tokens == cloud_ai_100_tokens).all(), (
        "Tokens don't match for pytorch output and Cloud AI 100 output."
    )
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))

    manual_cleanup(qeff_model.onnx_path)
    if compare_results is False:
        return

    compile_params = {"seq_len": ctx_len, "n_layer": n_layer}
    assert dump_and_compare_results(
        model_name,
        compile_params,
        "speech_seq2seq_model_results.json",
        cloud_ai_100_tokens,
        exec_info=exec_info,
        pytorch_hf_tokens=pytorch_hf_tokens,
        pytorch_kv_tokens=pytorch_kv_tokens,
        ort_tokens=ort_tokens,
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model_name", test_models)
def test_full_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    torch.manual_seed(42)
    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, compare_results=True, manual_cleanup=manual_cleanup, num_devices=4
    )


@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model_name", test_models)
def test_few_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    torch.manual_seed(42)
    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=4, manual_cleanup=manual_cleanup)
