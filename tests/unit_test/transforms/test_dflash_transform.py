# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for the DFlash transforms in QEfficient.transformers.models.pytorch_transforms
(DFlashTransform, DFlashDLMTransform, DFlashTLMTransform) and for
QEfficient.transformers.spd.dflash.compute_dflash_target_hidden_states.

Verifies that:
  1. DFlashTransform swaps QEff Qwen3 modules for their DFlash draft-model (QEffDFlash*) counterparts.
  2. DFlashDLMTransform cleanly drops fc/hidden_norm on the draft side, with no checkpoint required.
  3. DFlashTLMTransform attaches fc/hidden_norm/target_layer_ids on the target side, for multiple
     model families, with no checkpoint required.
  4. The DFlash draft model's forward pass (two-stream rotary embedding, two-stream attention,
     custom causal mask) runs correctly on a tiny in-memory model.
  5. The target-model side's target_layer_ids collection + compute_dflash_target_hidden_states
     wiring produces the expected hidden_states shape end to end.
  6. compute_dflash_target_hidden_states itself matches a manual reference computation.

All tests run on CPU only, using tiny in-memory models -- no checkpoints, no Hugging Face Hub access.
"""

import pytest
import torch
from torch import nn

VOCAB_SIZE = 500
CTX_LEN = 32
HIDDEN_SIZE = 64


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return Qwen3ForCausalLM(cfg).eval()


def make_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval()


_TLM_MODEL_FACTORIES = [
    (make_tiny_qwen3, "qwen3"),
    (make_tiny_llama, "llama"),
]
_TLM_MODEL_IDS = [label for _, label in _TLM_MODEL_FACTORIES]


def _make_legacy_past_key_values(config, batch, ctx_len):
    """Zero-tensor legacy KV cache tuple, sized to ctx_len (mirrors test_blocking_transform.py)."""
    n_layers = config.num_hidden_layers
    n_attn = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_attn)
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_attn)
    return tuple(
        (
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    )


# ---------------------------------------------------------------------------
# Tests: transform ordering
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.parametrize(
    ("qaic_config", "expected_order"),
    [
        ({"dflash_dlm": True}, ["dflash", "dflash_dlm", "spd", "sampler"]),
        ({"target_layer_ids": [1]}, ["dflash_tlm", "spd", "sampler"]),
    ],
)
def test_dflash_transforms_run_before_spd_and_sampler(monkeypatch, qaic_config, expected_order):
    from QEfficient import QEFFAutoModelForCausalLM
    from QEfficient.transformers.models.pytorch_transforms import (
        DFlashDLMTransform,
        DFlashTLMTransform,
        DFlashTransform,
        SamplerTransform,
        SpDTransform,
    )

    applied = []

    def record_apply(name):
        def apply(cls, model, config=None, **kwargs):
            applied.append(name)
            return model, False

        return classmethod(apply)

    monkeypatch.setattr(DFlashTransform, "apply", record_apply("dflash"))
    monkeypatch.setattr(DFlashDLMTransform, "apply", record_apply("dflash_dlm"))
    monkeypatch.setattr(DFlashTLMTransform, "apply", record_apply("dflash_tlm"))
    monkeypatch.setattr(SpDTransform, "apply", record_apply("spd"))
    monkeypatch.setattr(SamplerTransform, "apply", record_apply("sampler"))

    QEFFAutoModelForCausalLM(make_tiny_qwen3(), qaic_config=qaic_config)

    assert applied == expected_order


def test_dflash2_target_transform_sets_shared_dflash_chunk_metadata():
    from QEfficient.transformers.models.pytorch_transforms import DFlash2TLMTransform

    model = make_tiny_qwen3()
    transformed, applied = DFlash2TLMTransform.apply(
        model,
        {"target_layer_ids": [1], "dflash_block_size": 8},
    )

    assert applied is True
    assert transformed.model.target_layer_ids == [1]
    assert transformed.model.dflash_chunk_size == 8


def test_dflash2_state_dict_remap_and_export_submodule():
    from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import (
        DFlash2DecoderLayer,
        DFlash2ForCausalLM,
        remap_dflash2_state_dict,
    )

    remapped = remap_dflash2_state_dict(
        {
            "layers.0.input_layernorm.weight": torch.ones(4),
            "candidate_selector.predecessor_codebook": torch.ones(8, 2),
            "fc.weight": torch.ones(4, 4),
        }
    )

    assert "model.layers.0.input_layernorm.weight" in remapped
    assert "model.candidate_selector.predecessor_codebook.weight" in remapped
    assert "fc.weight" not in remapped
    assert DFlash2ForCausalLM.__new__(DFlash2ForCausalLM).get_submodules_for_export() == {DFlash2DecoderLayer}


def test_dflash2_dlm_transform_scales_fp16_residual_stream():
    from transformers import Qwen3Config

    from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM
    from QEfficient.transformers.models.pytorch_transforms import DFlash2DLMTransform

    config = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
        layer_types=["sliding_attention", "sliding_attention"],
        dflash_config={
            "block_size": 4,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": VOCAB_SIZE - 1,
            "selector_rank": 8,
            "selector_top_k": 1,
        },
    )
    model = DFlash2ForCausalLM(config)
    epsilon = model.model.norm.variance_epsilon
    finish_weights = [
        layer.attention_conv.kernel_projection.weight[layer.attention_conv.kernel_size * layer.attention_conv.groups :]
        .detach()
        .clone()
        for layer in model.model.layers
    ]

    transformed, applied = DFlash2DLMTransform.apply(
        model,
        {"dflash_dlm": True, "dflash_residual_scale": 4.0},
    )

    assert applied is True
    assert transformed.model.residual_scale == 4.0
    assert transformed.model.layers[0].scale_input_residual is True
    assert transformed.model.layers[1].scale_input_residual is False
    assert transformed.model.layers[0].input_layernorm.variance_epsilon == epsilon
    assert transformed.model.layers[0].post_attention_layernorm.variance_epsilon == epsilon / 16
    assert transformed.model.layers[1].input_layernorm.variance_epsilon == epsilon / 16
    assert transformed.model.norm.variance_epsilon == epsilon / 16
    for layer, original_finish_weight in zip(transformed.model.layers, finish_weights):
        finish_offset = layer.attention_conv.kernel_size * layer.attention_conv.groups
        assert torch.equal(
            layer.attention_conv.kernel_projection.weight[finish_offset:],
            original_finish_weight / 4,
        )


# ---------------------------------------------------------------------------
# Tests: draft (DLM) side -- DFlashTransform + DFlashDLMTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestDFlashDraftTransform:
    """DFlashTransform + DFlashDLMTransform on the draft (DLM) side -- Qwen3 only."""

    def _build_dlm(self):
        from QEfficient import QEFFAutoModelForCausalLM

        hf_model = make_tiny_qwen3()
        return QEFFAutoModelForCausalLM(hf_model, qaic_config={"dflash_dlm": True})

    def test_unsupported_dlm_architecture_raises(self):
        from QEfficient import QEFFAutoModelForCausalLM

        with pytest.raises(NotImplementedError, match="QEffLlamaForCausalLM"):
            QEFFAutoModelForCausalLM(make_tiny_llama(), qaic_config={"dflash_dlm": True})

    def test_dflash_transform_swaps_to_dflash_classes(self):
        qeff_model = self._build_dlm()

        assert qeff_model.dflash_dlm is True
        assert type(qeff_model.model.model).__name__ == "QEffDFlashModel"
        assert type(qeff_model.model).__name__ == "QEffDFlashForCausalLM"
        for layer in qeff_model.model.model.layers:
            assert type(layer).__name__ == "QEffDFlashDecoderLayer"
            assert type(layer.self_attn).__name__ == "QEffDFlashAttention"

    def test_dflash_dlm_transform_clean_without_tlm_repo(self):
        qeff_model = self._build_dlm()

        assert not hasattr(qeff_model.model.model, "fc")
        assert not hasattr(qeff_model.model.model, "hidden_norm")

    def test_dflash_draft_forward_smoke(self):
        from QEfficient.transformers.cache_utils import QEffDynamicCache

        qeff_model = self._build_dlm()
        model = qeff_model.model  # QEffDFlashForCausalLM

        block_size = 4
        batch = 1
        input_ids = torch.randint(0, VOCAB_SIZE, (batch, block_size))
        target_hidden = torch.randn(batch, block_size, HIDDEN_SIZE)
        position_ids_target = torch.arange(block_size).unsqueeze(0)
        position_ids = position_ids_target + block_size
        attention_mask = torch.zeros(batch, 2 * block_size)  # only .shape[-1] is read by _create_mask
        # KV cache buffer must be pre-allocated to cover both the context block (positions
        # 0..block_size-1) and the noise block (positions block_size..2*block_size-1) that
        # get written into it -- an empty/lazy cache has nothing to index into at position 4+.
        legacy_pkv = _make_legacy_past_key_values(model.config, batch, ctx_len=2 * block_size)
        past_key_values = QEffDynamicCache.from_legacy_cache(legacy_pkv)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                target_hidden=target_hidden,
                position_ids=position_ids,
                position_ids_target=position_ids_target,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        assert outputs.logits.shape == (batch, block_size, VOCAB_SIZE)
        assert torch.isfinite(outputs.logits).all()


# ---------------------------------------------------------------------------
# Tests: target (TLM) side -- DFlashTLMTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestDFlashTLMTransform:
    """DFlashTLMTransform on the target (TLM) side -- multiple model families."""

    @pytest.mark.parametrize("make_model,label", _TLM_MODEL_FACTORIES, ids=_TLM_MODEL_IDS)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_dflash_tlm_projection_matches_model_precision(self, make_model, label, dtype):
        from QEfficient import QEFFAutoModelForCausalLM
        from QEfficient.transformers.spd.dflash import compute_dflash_target_hidden_states

        qeff_model = QEFFAutoModelForCausalLM(make_model().to(dtype=dtype), qaic_config={"target_layer_ids": [1]})
        inner = qeff_model.model.model
        target_hidden = torch.randn(1, 2, HIDDEN_SIZE, dtype=dtype)

        assert inner.fc.weight.dtype is dtype, f"[{label}] fc dtype does not match model dtype"
        assert inner.hidden_norm.weight.dtype is dtype, f"[{label}] hidden_norm dtype does not match model dtype"
        projected = compute_dflash_target_hidden_states([target_hidden], inner.fc, inner.hidden_norm)
        assert projected.dtype is dtype
        assert torch.isfinite(projected).all()

    @pytest.mark.parametrize("make_model,label", _TLM_MODEL_FACTORIES, ids=_TLM_MODEL_IDS)
    def test_dflash_tlm_transform_attaches_fc_and_hidden_norm(self, make_model, label):
        from QEfficient import QEFFAutoModelForCausalLM

        hf_model = make_model()
        target_layer_ids = [1]
        qeff_model = QEFFAutoModelForCausalLM(hf_model, qaic_config={"target_layer_ids": target_layer_ids})

        assert qeff_model.dflash_tlm is True, f"[{label}] dflash_tlm flag not set"
        inner = qeff_model.model.model
        assert isinstance(inner.fc, nn.Linear), f"[{label}] fc not attached"
        assert inner.fc.in_features == HIDDEN_SIZE * len(target_layer_ids)
        assert inner.fc.out_features == HIDDEN_SIZE
        assert hasattr(inner, "hidden_norm"), f"[{label}] hidden_norm not attached"
        assert inner.target_layer_ids == target_layer_ids

    @pytest.mark.parametrize("make_model,label", _TLM_MODEL_FACTORIES, ids=_TLM_MODEL_IDS)
    def test_dflash_tlm_forward_produces_target_hidden_states(self, make_model, label):
        from QEfficient import QEFFAutoModelForCausalLM

        hf_model = make_model()
        target_layer_ids = [1]
        qeff_model = QEFFAutoModelForCausalLM(hf_model, qaic_config={"target_layer_ids": target_layer_ids})

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
        qeff_inputs = {
            "input_ids": input_ids,
            "position_ids": torch.arange(6).unsqueeze(0),
            "past_key_values": _make_legacy_past_key_values(hf_model.config, batch=1, ctx_len=CTX_LEN),
        }
        with torch.no_grad():
            outputs = qeff_model.model(**qeff_inputs)

        assert outputs.hidden_states is not None, f"[{label}] no target hidden_states returned"
        assert outputs.hidden_states.shape == (1, 6, HIDDEN_SIZE), f"[{label}] unexpected target hidden_states shape"


# ---------------------------------------------------------------------------
# Tests: compute_dflash_target_hidden_states (pure function)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestComputeDFlashTargetHiddenStates:
    """Pure-function correctness of compute_dflash_target_hidden_states."""

    def test_matches_manual_reference(self):
        from QEfficient.transformers.spd.dflash import compute_dflash_target_hidden_states

        n_layers, hidden = 3, HIDDEN_SIZE
        target_hidden_list = [torch.randn(2, 5, hidden) for _ in range(n_layers)]
        fc = nn.Linear(n_layers * hidden, hidden, bias=False)
        hidden_norm = nn.RMSNorm(hidden)

        result = compute_dflash_target_hidden_states(target_hidden_list, fc, hidden_norm)

        expected = hidden_norm(fc(torch.cat(target_hidden_list, dim=-1)))
        assert result.shape == (2, 5, hidden)
        assert torch.equal(result, expected)
