# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests verify:
  - HF Gemma4 baseline: model runs and returns finite logits/tokens
  - QEff Gemma4 wraps into QEffGemma4ForConditionalGeneration
  - QEff Gemma4 vision encoder path runs and returns valid vision embeddings
  - QEff Gemma4 returns (batch, 1, vocab) for prefill and decode
  - QEff Gemma4 prefill token and probabilities are close to HF
  - QEff Gemma4 writes non-zero KV cache on prefill
  - Real prefill cache is consumed correctly during decode handoff
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

CTX_LEN = 32
PREFILL_LEN = 8
VOCAB_SIZE = 500


def _load_gemma4_classes():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    from QEfficient.transformers.models.gemma4.modeling_gemma4 import QEffGemma4ForConditionalGeneration

    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    except ImportError:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config as Gemma4TextConfig

    return (
        Gemma4Config,
        Gemma4TextConfig,
        Gemma4VisionConfig,
        Gemma4ForConditionalGeneration,
        QEffGemma4ForConditionalGeneration,
    )


# ---------------------------------------------------------------------------
# Tiny Gemma4 factory
# ---------------------------------------------------------------------------


def make_tiny_gemma4():
    """
    Minimal Gemma4 vision+text config with both sliding and full attention text layers.
    Softcapping is disabled so HF and QEff logits are directly comparable.
    """
    Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig, Gemma4ForConditionalGeneration, _ = _load_gemma4_classes()

    text_cfg = Gemma4TextConfig()
    text_cfg.num_hidden_layers = 2
    text_cfg.num_attention_heads = 2
    text_cfg.num_key_value_heads = 2
    text_cfg.hidden_size = 64
    text_cfg.intermediate_size = 128
    text_cfg.vocab_size = VOCAB_SIZE
    text_cfg.max_position_embeddings = CTX_LEN

    if hasattr(text_cfg, "head_dim"):
        text_cfg.head_dim = 32
    if hasattr(text_cfg, "global_head_dim"):
        text_cfg.global_head_dim = 32
    if hasattr(text_cfg, "num_global_key_value_heads"):
        text_cfg.num_global_key_value_heads = 2
    if hasattr(text_cfg, "attention_k_eq_v"):
        text_cfg.attention_k_eq_v = False

    if hasattr(text_cfg, "sliding_window"):
        text_cfg.sliding_window = 8
    if hasattr(text_cfg, "layer_types"):
        text_cfg.layer_types = ["sliding_attention", "full_attention"]
    if hasattr(text_cfg, "_sliding_window_pattern"):
        text_cfg._sliding_window_pattern = 2
    if hasattr(text_cfg, "sliding_window_pattern"):
        text_cfg.sliding_window_pattern = 2

    if hasattr(text_cfg, "final_logit_softcapping"):
        text_cfg.final_logit_softcapping = None
    if hasattr(text_cfg, "attn_logit_softcapping"):
        text_cfg.attn_logit_softcapping = None
    if hasattr(text_cfg, "rope_scaling") and text_cfg.rope_scaling is None:
        text_cfg.rope_scaling = {"rope_type": "default"}

    if hasattr(text_cfg, "num_experts"):
        text_cfg.num_experts = 4
    if hasattr(text_cfg, "num_local_experts"):
        text_cfg.num_local_experts = 4
    if hasattr(text_cfg, "num_experts_per_tok"):
        text_cfg.num_experts_per_tok = 2
    if hasattr(text_cfg, "top_k_experts"):
        text_cfg.top_k_experts = 2
    if hasattr(text_cfg, "moe_intermediate_size"):
        text_cfg.moe_intermediate_size = 64
    if hasattr(text_cfg, "dtype"):
        text_cfg.dtype = torch.float32

    vision_cfg = Gemma4VisionConfig()
    if hasattr(vision_cfg, "hidden_size"):
        vision_cfg.hidden_size = 64
    if hasattr(vision_cfg, "intermediate_size"):
        vision_cfg.intermediate_size = 128
    if hasattr(vision_cfg, "num_hidden_layers"):
        vision_cfg.num_hidden_layers = 2
    if hasattr(vision_cfg, "num_attention_heads"):
        vision_cfg.num_attention_heads = 2
    if hasattr(vision_cfg, "num_key_value_heads"):
        vision_cfg.num_key_value_heads = 2
    if hasattr(vision_cfg, "head_dim"):
        vision_cfg.head_dim = 32
    if hasattr(vision_cfg, "default_output_length"):
        vision_cfg.default_output_length = 16
    if hasattr(vision_cfg, "pooling_kernel_size"):
        vision_cfg.pooling_kernel_size = 2
    if hasattr(vision_cfg, "patch_size"):
        vision_cfg.patch_size = 2
    if hasattr(vision_cfg, "dtype"):
        vision_cfg.dtype = torch.float32

    try:
        mm_cfg = Gemma4Config(text_config=text_cfg, vision_config=vision_cfg)
    except TypeError:
        mm_cfg = Gemma4Config()
        if hasattr(mm_cfg, "text_config"):
            mm_cfg.text_config = text_cfg
        if hasattr(mm_cfg, "vision_config"):
            mm_cfg.vision_config = vision_cfg

    if hasattr(mm_cfg, "vocab_size"):
        mm_cfg.vocab_size = VOCAB_SIZE
    if hasattr(mm_cfg, "image_token_id"):
        mm_cfg.image_token_id = VOCAB_SIZE + 5
    if hasattr(mm_cfg, "mm_tokens_per_image"):
        mm_cfg.mm_tokens_per_image = 16
    if getattr(mm_cfg, "text_config", None) is None:
        mm_cfg.text_config = text_cfg
    if getattr(mm_cfg, "vision_config", None) is None:
        mm_cfg.vision_config = vision_cfg
    if hasattr(mm_cfg, "dtype"):
        mm_cfg.dtype = torch.float32
    if hasattr(mm_cfg.text_config, "dtype"):
        mm_cfg.text_config.dtype = torch.float32
    if hasattr(mm_cfg.vision_config, "dtype"):
        mm_cfg.vision_config.dtype = torch.float32

    return Gemma4ForConditionalGeneration(mm_cfg).eval(), text_cfg


@pytest.mark.parametrize("moe", [False, True])
def test_gemma4_dflash_example_subfunction_policy(monkeypatch, moe):
    from examples.performance.dflash import utils

    config = SimpleNamespace(model_type="gemma4", text_config=SimpleNamespace(enable_moe_block=moe))
    text_model = SimpleNamespace(fc=torch.nn.Linear(4, 4, bias=False), hidden_norm=torch.nn.LayerNorm(4))
    options = {}

    def compile_model(**kwargs):
        options.update(kwargs)
        return {"lang_qpc_path": "language-qpc", "vision_qpc_path": "vision-qpc"}

    model = SimpleNamespace(
        lang_model=SimpleNamespace(model=SimpleNamespace(language_model=text_model)), compile=compile_model
    )
    monkeypatch.setattr(utils.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(utils.QEFFAutoModelForImageTextToText, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        utils,
        "read_dlm_meta",
        lambda *args: ({"fc.weight": torch.eye(4), "hidden_norm.weight": torch.ones(4)}, [1], 16),
    )
    result = utils.compile_gemma_vlm_qpcs(
        "target", "draft", prefill_seq_len=128, ctx_len=4096, num_cores=8, num_devices=4
    )
    assert result == ("language-qpc", "vision-qpc")
    assert options["use_onnx_subfunctions"] is True
    assert options["node_precision_info"] is True
    assert options["dflash_block_size"] == 16


@pytest.mark.parametrize("prompt_length", [2, 10, 22])
def test_gemma4_dflash_verification_matches_sequential(prompt_length):
    torch.manual_seed(7)
    model, _ = make_tiny_gemma4()
    qeff = QEFFAutoModelForImageTextToText(
        model, kv_offload=True, qaic_config={"target_layer_ids": [1], "speculative_model_type": "target"}
    )
    wrapper = qeff.lang_model.model
    config = model.config.text_config
    initial_cache = qeff.model.get_dummy_pkv_cache(config, batch_size=1, seq_len=CTX_LEN)
    specs, _ = qeff.model.get_specializations(
        batch_size=1, prefill_seq_len=4, ctx_len=CTX_LEN, img_size=16, kv_offload=True, dflash_block_size=4
    )
    capacity = config.sliding_window + 3
    assert all(spec["sliding_window"] == capacity for spec in specs["lang"])
    for index, kind in enumerate(config.layer_types):
        if kind == "sliding_attention":
            initial_cache[index] = tuple(t[:, :, :capacity].clone() for t in initial_cache[index])

    def run(ids, position, cache):
        count = len(ids)
        return wrapper(
            input_ids=torch.tensor([ids]),
            position_ids=torch.arange(position, position + count).view(1, -1),
            vision_embeds=torch.zeros(1, 16, config.hidden_size),
            image_idx=torch.zeros((1, 1), dtype=torch.int64),
            mm_token_type_ids=torch.zeros((1, count), dtype=torch.int64),
            past_key_values=cache,
            num_logits_to_keep=torch.zeros((count, 1), dtype=torch.int64),
        )

    with torch.no_grad():
        cache = initial_cache
        for position in range(prompt_length):
            cache = run([20 + position], position, cache)[3]
        reference_cache = copy.deepcopy(cache)
        block_ids = [40, 41, 42, 43]
        block = run(block_ids, prompt_length, cache)
        rejected_cache = copy.deepcopy(block[3])
        sequential = []
        seed_cache = None
        for offset, token in enumerate(block_ids):
            output = run([token], prompt_length + offset, reference_cache)
            reference_cache = output[3]
            sequential.append(output[0])
            if offset == 0:
                seed_cache = copy.deepcopy(reference_cache)
        # Reject all drafts, then verify a new block after the committed seed.
        next_block = run([60, 61, 62, 63], prompt_length + 1, rejected_cache)
        expected_next = run([60, 61, 62, 63], prompt_length + 1, seed_cache)
    assert block[0].shape == (1, 4, VOCAB_SIZE)
    assert block[4].shape == (1, 4, config.hidden_size)
    torch.testing.assert_close(block[0], torch.cat(sequential, dim=1), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(next_block[0], expected_next[0], atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("dflash", [False, True])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("prefill_only", [False, True, None])
def test_gemma4_dflash_expert_parallel_policy(dflash, moe, prefill_only):
    from QEfficient.transformers.moe import MoEFlavour

    model, _ = make_tiny_gemma4()
    config = copy.deepcopy(model.config)
    config.text_config.enable_moe_block = moe
    model = type(model)(config).eval()
    qaic_config = {"moe_config": {"cores_per_expert": 2, "tree_reduce": True, "expert_parallel_chunk_size": 16}}
    if dflash:
        qaic_config["target_layer_ids"] = [1]
    qeff = QEFFAutoModelForImageTextToText(model, kv_offload=True, qaic_config=qaic_config)
    original_config = copy.deepcopy(qaic_config)
    for _ in range(2):
        qeff.transform(
            ctx_len=32,
            seq_len=16,
            bs=1,
            num_devices=4,
            aic_num_cores=8,
            qaic_config=qaic_config,
            prefill_only=prefill_only,
            prefill_seq_len=16,
        )
        assert qaic_config == original_config
        blocks = [
            layer.moe_block for layer in qeff.lang_model.model.language_model.layers if hasattr(layer, "moe_block")
        ]
        if not moe:
            assert not blocks
            assert "moe_prefill_flavour" not in qeff.lang_model.hash_params
            continue
        assert blocks
        expected = MoEFlavour.EXPERT_PARALLEL if dflash or prefill_only else MoEFlavour.DECODE_BMM
        assert qeff.lang_model.hash_params["moe_prefill_flavour"] == expected.value
        for block in blocks:
            assert block._moe_flavour is expected
            if expected is MoEFlavour.EXPERT_PARALLEL:
                assert block.num_devices == 4
                assert block.cores_per_expert == 2
                assert block.tree_reduce is True


def _make_qeff_gemma4(model):
    """
    Use the ImageTextToText auto-wrapper for this Gemma4 test model.
    """
    return QEFFAutoModelForImageTextToText(model)


def _qeff_forward(qeff, **inputs):
    """
    Run the language decoder path for image-text Gemma4 wrappers.
    Returns an object with `.logits` and `.past_key_values` for parity with CausalLM tests.
    """
    if hasattr(qeff, "lang_model"):
        lang_inputs = dict(inputs)
        model_cfg = qeff.model.config
        text_cfg = getattr(model_cfg, "text_config", model_cfg)
        batch_size = int(lang_inputs["input_ids"].shape[0])
        mm_tokens = int(getattr(model_cfg, "mm_tokens_per_image", 256))
        lang_model_cfg = getattr(getattr(qeff.lang_model.model, "language_model", None), "config", None)
        hidden_size = int(
            getattr(
                text_cfg,
                "hidden_size",
                getattr(lang_model_cfg, "hidden_size", 0),
            )
        )
        if hidden_size <= 0:
            hidden_size = int(getattr(qeff.lang_model.model.lm_head, "in_features", 64))
        try:
            dtype = next(qeff.lang_model.model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

        if "vision_embeds" not in lang_inputs or lang_inputs["vision_embeds"] is None:
            lang_inputs["vision_embeds"] = torch.zeros(
                (batch_size, mm_tokens, hidden_size),
                dtype=dtype,
            )
        if "image_idx" not in lang_inputs or lang_inputs["image_idx"] is None:
            lang_inputs["image_idx"] = torch.zeros((1, 1), dtype=torch.int64)

        logits, vision_embeds, image_idx, past_key_values = qeff.lang_model.model(**lang_inputs)
        return SimpleNamespace(
            logits=logits,
            past_key_values=past_key_values,
            vision_embeds=vision_embeds,
            image_idx=image_idx,
        )

    return qeff.model(**inputs)


def _qeff_vision_inputs(qeff):
    """
    Build vision-only inputs using the model's own dummy-input helper.
    """
    dummy_inputs = qeff.model.get_dummy_inputs(kv_offload=True)
    assert isinstance(dummy_inputs, dict) and "vision" in dummy_inputs, "Missing vision dummy inputs from model"
    return dummy_inputs["vision"]


def _qeff_vision_forward(qeff):
    """
    Run the vision encoder path and return vision embeddings.
    """
    vision_inputs = _qeff_vision_inputs(qeff)
    return qeff.vision_model.model(**vision_inputs)


def _layer_type(config, layer_idx):
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and layer_idx < len(layer_types):
        return layer_types[layer_idx]
    return "full_attention"


def _kv_heads_for_layer(config, layer_idx):
    layer_t = _layer_type(config, layer_idx)
    use_alt = bool(getattr(config, "attention_k_eq_v", False))
    if layer_t != "sliding_attention" and use_alt and getattr(config, "num_global_key_value_heads", None) is not None:
        return int(config.num_global_key_value_heads)
    return int(getattr(config, "num_key_value_heads", config.num_attention_heads))


def _head_dim_for_layer(config, layer_idx):
    layer_t = _layer_type(config, layer_idx)
    global_dim = getattr(config, "global_head_dim", None)
    if layer_t != "sliding_attention" and global_dim is not None:
        return int(global_dim)
    return int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))


def _ctx_len_for_layer(config, layer_idx, ctx_len):
    if _layer_type(config, layer_idx) != "sliding_attention":
        return int(ctx_len)
    sliding_window = int(getattr(config, "sliding_window", ctx_len))
    return int(min(sliding_window, ctx_len))


def _zero_kv_cache(config, ctx_len=CTX_LEN):
    """Build a zero-initialised past_key_values tuple for Gemma4."""
    n_layers = int(config.num_hidden_layers)
    cache = []
    for i in range(n_layers):
        n_kv = _kv_heads_for_layer(config, i)
        head_dim = _head_dim_for_layer(config, i)
        layer_ctx = _ctx_len_for_layer(config, i, ctx_len)
        cache.append(
            (
                torch.zeros(1, n_kv, layer_ctx, head_dim, dtype=torch.float32),
                torch.zeros(1, n_kv, layer_ctx, head_dim, dtype=torch.float32),
            )
        )
    return tuple(cache)


def _prefill_inputs(input_ids, config, ctx_len=CTX_LEN):
    """Build QEff-style prefill inputs for Gemma4."""
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": _zero_kv_cache(config, ctx_len),
    }


def _decode_inputs(next_token, decode_position, past_key_values):
    """Build a single-token decode input using the REAL past_key_values."""
    return {
        "input_ids": torch.tensor([[next_token]], dtype=torch.long),
        "position_ids": torch.tensor([[decode_position]], dtype=torch.long),
        "past_key_values": past_key_values,
    }


def _extract_next_token(logits):
    """
    Extract greedy next token. The QEff decoder path returns (batch, 1, vocab),
    so logits[0, -1, :] works for both (batch, seq, vocab) and (batch, 1, vocab).
    """
    return logits[0, -1, :].argmax(-1).item()


# ---------------------------------------------------------------------------
# Tests: HF Gemma4 baseline
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
class TestHFGemma4Baseline:
    """HF Gemma4 model runs correctly on CPU and produces valid logits."""

    def test_forward_returns_logits_with_expected_shape(self):
        model, cfg = make_tiny_gemma4()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape[0] == 1 and out.logits.shape[-1] == VOCAB_SIZE, (
            f"Unexpected HF Gemma4 logits shape {out.logits.shape}; expected batch=1 and vocab={VOCAB_SIZE}"
        )
        assert out.logits.shape[1] in (1, PREFILL_LEN), (
            f"Unexpected HF Gemma4 sequence logits length {out.logits.shape[1]}; expected 1 or {PREFILL_LEN}"
        )

    def test_logits_are_finite(self):
        model, cfg = make_tiny_gemma4()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# Tests: QEff Gemma4 architecture
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
class TestQEffGemma4Architecture:
    """QEff Gemma4 must use QEffGemma4ForConditionalGeneration after KVCacheTransform."""

    def test_qeff_wraps_without_error(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        assert qeff is not None
        assert hasattr(qeff, "model")

    def test_qeff_model_class_is_qeff_gemma4(self):
        model, cfg = make_tiny_gemma4()
        from QEfficient.transformers.models.gemma4.modeling_gemma4 import (
            QEffGemma4DecoderWrapper,
            QEffGemma4EncoderWrapper,
        )

        qeff = _make_qeff_gemma4(model)
        assert isinstance(qeff.lang_model.model, QEffGemma4DecoderWrapper), (
            f"Expected QEffGemma4DecoderWrapper, got {type(qeff.lang_model.model)}"
        )
        assert isinstance(qeff.vision_model.model, QEffGemma4EncoderWrapper), (
            f"Expected QEffGemma4EncoderWrapper, got {type(qeff.vision_model.model)}"
        )

    def test_qeff_model_is_eval_mode(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        assert not qeff.model.training

    def test_text_rotary_caches_cover_supported_context_and_match_hf(self):
        model, cfg = make_tiny_gemma4()
        hf_text_model = model.model.language_model
        position_ids = torch.tensor([[0, 1, 7, CTX_LEN - 1]], dtype=torch.long)
        hidden_states = torch.zeros(1, position_ids.shape[1], cfg.hidden_size)
        with torch.no_grad():
            expected_embeddings = {
                layer_type: hf_text_model.rotary_emb(hidden_states, position_ids, layer_type)
                for layer_type in ("full_attention", "sliding_attention")
            }

        qeff = _make_qeff_gemma4(model)
        text_model = qeff.lang_model.model.language_model

        for layer_type in ("full_attention", "sliding_attention"):
            expected_head_dim = cfg.global_head_dim if layer_type == "full_attention" else cfg.head_dim
            for embedding_index, embedding_name in enumerate(("cos", "sin")):
                cached_embedding = getattr(text_model, f"{layer_type}_{embedding_name}_cached")
                assert cached_embedding.shape == (text_model.rotary_emb.original_max_seq_len, expected_head_dim)
                torch.testing.assert_close(
                    cached_embedding[position_ids],
                    expected_embeddings[layer_type][embedding_index],
                )

    def test_text_forward_reuses_rotary_caches(self, mocker):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        text_model = qeff.lang_model.model.language_model
        rotary_forward = mocker.spy(text_model.rotary_emb, "forward")
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))

        rotary_forward.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: QEff Gemma4 logit shape (argmax-based extraction)
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffGemma4LogitShape:
    """
    QEff Gemma4 language decoder uses position_ids.argmax to extract a single logit
    per batch item, returning (batch, 1, vocab).
    """

    def test_prefill_logits_shape_is_batch_1_vocab(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
        assert out.logits.shape == (1, 1, VOCAB_SIZE), (
            f"QEffGemma4 prefill logits shape: expected (1, 1, {VOCAB_SIZE}), got {out.logits.shape}"
        )

    def test_decode_logits_shape_is_batch_1_vocab(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            prefill_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
        prefill_token = _extract_next_token(prefill_out.logits)
        with torch.no_grad():
            decode_out = _qeff_forward(qeff, **_decode_inputs(prefill_token, PREFILL_LEN, prefill_out.past_key_values))
        assert decode_out.logits.shape == (1, 1, VOCAB_SIZE), (
            f"QEffGemma4 decode logits shape: expected (1, 1, {VOCAB_SIZE}), got {decode_out.logits.shape}"
        )


# ---------------------------------------------------------------------------
# Tests: QEff Gemma4 vision encoder
# ---------------------------------------------------------------------------


@pytest.mark.multimodal
@pytest.mark.accuracy
class TestQEffGemma4VisionEncoder:
    """QEff Gemma4 vision encoder path must produce stable, finite embeddings."""

    def test_encoder_forward_returns_3d_embeddings(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        with torch.no_grad():
            vision_embeds = _qeff_vision_forward(qeff)

        assert isinstance(vision_embeds, torch.Tensor)
        assert vision_embeds.ndim == 3, f"Expected vision_embeds rank=3, got shape={vision_embeds.shape}"
        assert vision_embeds.shape[2] == cfg.hidden_size, (
            f"Expected hidden dim {cfg.hidden_size}, got {vision_embeds.shape[2]}"
        )
        expected_mm_tokens = int(getattr(qeff.model.config, "mm_tokens_per_image", vision_embeds.shape[1]))
        assert vision_embeds.shape[1] == expected_mm_tokens, (
            f"Expected mm_tokens_per_image {expected_mm_tokens}, got {vision_embeds.shape[1]}"
        )

    def test_encoder_embeddings_are_finite(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        with torch.no_grad():
            vision_embeds = _qeff_vision_forward(qeff)
        assert torch.isfinite(vision_embeds).all(), "Gemma4 vision embeddings contain NaN/Inf"


# ---------------------------------------------------------------------------
# Tests: QEff Gemma4 accuracy vs HF
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffGemma4AccuracyVsHF:
    """QEff Gemma4 must preserve HF greedy token and stay numerically close."""

    def test_prefill_token_matches_hf(self):
        model, cfg = make_tiny_gemma4()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            hf_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        qeff = _make_qeff_gemma4(model)
        with torch.no_grad():
            qeff_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
        qeff_token = _extract_next_token(qeff_out.logits)

        assert hf_token == qeff_token, (
            f"Gemma4 prefill token mismatch: HF={hf_token}, QEff={qeff_token}. "
            "KVCacheTransform must not change the greedy prediction."
        )

    def test_prefill_logits_numerically_close_to_hf(self):
        model, cfg = make_tiny_gemma4()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            hf_logits = model(input_ids=input_ids).logits[:, -1, :]

        qeff = _make_qeff_gemma4(model)
        with torch.no_grad():
            qeff_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
        qeff_logits = qeff_out.logits[:, -1, :]

        hf_probs = F.softmax(hf_logits, dim=-1)
        qeff_probs = F.softmax(qeff_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < 1e-3, f"Gemma4 probability distribution mismatch: max_diff={max_diff:.6f} > 1e-3"


# ---------------------------------------------------------------------------
# Tests: QEff Gemma4 KV cache is written during prefill
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffGemma4CacheWritten:
    """After prefill, the Gemma4 KV cache must contain non-zero values."""

    def test_past_key_values_not_none_after_prefill(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
        assert out.past_key_values is not None, "Gemma4 past_key_values is None after prefill"

    def test_cache_is_non_zero_after_prefill(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))

        pkv = out.past_key_values
        assert isinstance(pkv, (list, tuple)), f"Gemma4 expected tuple/list past_key_values, got {type(pkv)}"
        assert len(pkv) == cfg.num_hidden_layers, (
            f"Gemma4 expected {cfg.num_hidden_layers} cache layers, got {len(pkv)}"
        )
        layer0_keys = pkv[0][0]
        assert layer0_keys is not None, "Gemma4 layer-0 keys are None after prefill"
        prefill_span = min(PREFILL_LEN, layer0_keys.shape[-2])
        prefill_slice = layer0_keys[0, :, :prefill_span, :]
        assert not torch.all(prefill_slice == 0.0), "Gemma4 KV cache is all-zeros after prefill"


# ---------------------------------------------------------------------------
# Tests: QEff Gemma4 prefill -> decode handoff with REAL cache
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffGemma4PrefillDecodeHandoff:
    """Gemma4 prefill -> decode handoff with the REAL cache."""

    def test_decode_with_real_cache_produces_valid_token(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
        prefill_token = _extract_next_token(prefill_out.logits)

        with torch.no_grad():
            decode_out = _qeff_forward(qeff, **_decode_inputs(prefill_token, PREFILL_LEN, prefill_out.past_key_values))

        dec_token = _extract_next_token(decode_out.logits)
        assert 0 <= dec_token < VOCAB_SIZE, f"Gemma4 decode token {dec_token} out of range [0, {VOCAB_SIZE})"

    def test_decode_with_real_cache_returns_finite_logits(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
        prefill_token = _extract_next_token(prefill_out.logits)

        with torch.no_grad():
            decode_out = _qeff_forward(qeff, **_decode_inputs(prefill_token, PREFILL_LEN, prefill_out.past_key_values))

        assert torch.isfinite(decode_out.logits).all(), "Gemma4 decode logits contain NaN/Inf after real-cache handoff"

    def test_three_decode_steps_all_valid(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))

        token = _extract_next_token(prefill_out.logits)
        current_past = prefill_out.past_key_values
        decode_pos = PREFILL_LEN
        decode_tokens = []

        for _ in range(3):
            with torch.no_grad():
                out = _qeff_forward(qeff, **_decode_inputs(token, decode_pos, current_past))
            token = _extract_next_token(out.logits)
            decode_tokens.append(token)
            current_past = out.past_key_values
            decode_pos += 1

        assert len(decode_tokens) == 3
        for i, tok in enumerate(decode_tokens):
            assert 0 <= tok < VOCAB_SIZE, f"Gemma4 decode step {i}: token {tok} out of range"

    def test_three_decode_steps_all_finite(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))

        token = _extract_next_token(prefill_out.logits)
        current_past = prefill_out.past_key_values
        decode_pos = PREFILL_LEN

        for step in range(3):
            with torch.no_grad():
                out = _qeff_forward(qeff, **_decode_inputs(token, decode_pos, current_past))
            assert torch.isfinite(out.logits).all(), f"Gemma4 decode step {step}: logits contain NaN/Inf"
            token = _extract_next_token(out.logits)
            current_past = out.past_key_values
            decode_pos += 1

    def test_decode_is_deterministic(self):
        import copy

        model, cfg = make_tiny_gemma4()
        model_copy = copy.deepcopy(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        def _run(m):
            qeff = _make_qeff_gemma4(m)
            with torch.no_grad():
                prefill_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
            token = _extract_next_token(prefill_out.logits)
            current_past = prefill_out.past_key_values
            tokens = []
            for pos in range(PREFILL_LEN, PREFILL_LEN + 3):
                with torch.no_grad():
                    out = _qeff_forward(qeff, **_decode_inputs(token, pos, current_past))
                token = _extract_next_token(out.logits)
                tokens.append(token)
                current_past = out.past_key_values
            return tokens

        tokens1 = _run(model)
        tokens2 = _run(model_copy)
        assert tokens1 == tokens2, f"Gemma4 decode is not deterministic: {tokens1} vs {tokens2}"

    def test_real_cache_differs_from_zero_cache(self):
        """
        Decode token using the REAL prefill cache should differ from decode token
        using ZERO cache for at least one seed.
        """
        model, cfg = make_tiny_gemma4()
        found_difference = False

        for seed in range(8):
            torch.manual_seed(seed)
            qeff = _make_qeff_gemma4(model)
            input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

            with torch.no_grad():
                prefill_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))
            prefill_token = _extract_next_token(prefill_out.logits)
            real_cache = prefill_out.past_key_values

            with torch.no_grad():
                out_real = _qeff_forward(qeff, **_decode_inputs(prefill_token, PREFILL_LEN, real_cache))
            real_token = _extract_next_token(out_real.logits)

            with torch.no_grad():
                out_zero = _qeff_forward(qeff, **_decode_inputs(prefill_token, PREFILL_LEN, _zero_kv_cache(cfg)))
            zero_token = _extract_next_token(out_zero.logits)

            if real_token != zero_token:
                found_difference = True
                break

        assert found_difference, (
            "Gemma4 real-cache decode always matched zero-cache decode across 8 seeds. "
            "The KV cache may not be influencing output."
        )

    def test_decode_position_advances_strictly(self):
        model, cfg = make_tiny_gemma4()
        qeff = _make_qeff_gemma4(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = _qeff_forward(qeff, **_prefill_inputs(input_ids, cfg))

        token = _extract_next_token(prefill_out.logits)
        current_past = prefill_out.past_key_values
        positions_used = [PREFILL_LEN - 1]

        for _ in range(4):
            next_pos = positions_used[-1] + 1
            decode_in = _decode_inputs(token, next_pos, current_past)
            assert decode_in["position_ids"].item() == next_pos
            positions_used.append(next_pos)

            with torch.no_grad():
                out = _qeff_forward(qeff, **decode_in)
            token = _extract_next_token(out.logits)
            current_past = out.past_key_values

        for i in range(1, len(positions_used)):
            assert positions_used[i] > positions_used[i - 1], (
                f"Gemma4 positions not strictly increasing: {positions_used}"
            )
