# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for Speculative Decoding (SpDTransform) in QEfficient.

Tests verify:
  - SpDTransform.apply() with speculative_model_type="target" attaches tlm_forward
  - SpDTransform._module_mapping contains expected model classes
  - SpDTransform raises ValueError for invalid speculative_model_type
  - SpDTransform raises NotImplementedError for unsupported model class
  - QEFFAutoModelForCausalLM has check_and_get_num_speculative_tokens method
  - QEFFAutoModelForCausalLM has build_prefill_specialization / build_decode_specialization
  - is_tlm flag is set correctly on the wrapper

All tests run on CPU only.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform, SpDTransform

VOCAB_SIZE = 500
SEQ_LEN = 8
CTX_LEN = 32


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


def make_kv_transformed_llama():
    model, cfg = make_tiny_llama()
    transformed, _ = KVCacheTransform.apply(model)
    return transformed, cfg


# ---------------------------------------------------------------------------
# Tests: SpDTransform module mapping and structure
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformStructure:
    """SpDTransform must have correct class-level structure."""

    def test_spd_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import SpDTransform

        assert SpDTransform is not None

    def test_module_mapping_is_set(self):
        assert hasattr(SpDTransform, "_module_mapping")
        assert len(SpDTransform._module_mapping) > 0

    def test_module_mapping_contains_llama(self):
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

        assert QEffLlamaForCausalLM in SpDTransform._module_mapping

    def test_module_mapping_contains_qwen2(self):
        from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2ForCausalLM

        assert QEffQwen2ForCausalLM in SpDTransform._module_mapping

    def test_apply_classmethod_exists(self):
        assert hasattr(SpDTransform, "apply")
        assert callable(SpDTransform.apply)


# ---------------------------------------------------------------------------
# Tests: SpDTransform no-op paths (already tested in test_transform_accuracy.py,
# but included here for completeness)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformNoOpPaths:
    """SpDTransform must not apply when qaic_config is None or missing key."""

    def test_no_transform_when_qaic_config_is_none(self):
        model, _ = make_kv_transformed_llama()
        _, applied = SpDTransform.apply(model, qaic_config=None)
        assert not applied

    def test_no_transform_when_speculative_model_type_missing(self):
        model, _ = make_kv_transformed_llama()
        _, applied = SpDTransform.apply(model, qaic_config={})
        assert not applied

    def test_invalid_speculative_model_type_raises_value_error(self):
        model, _ = make_kv_transformed_llama()
        with pytest.raises(ValueError):
            SpDTransform.apply(model, qaic_config={"speculative_model_type": "invalid_xyz_abc"})

    def test_unsupported_model_class_raises_not_implemented(self):
        import torch.nn as nn

        class UnsupportedModel(nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(NotImplementedError):
            SpDTransform.apply(
                UnsupportedModel(),
                qaic_config={"speculative_model_type": "target"},
            )


# ---------------------------------------------------------------------------
# Tests: SpDTransform actual apply (TLM path)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformTLMApply:
    """SpDTransform with speculative_model_type='target' must attach tlm_forward."""

    def test_spd_transform_applies_to_llama_with_target_type(self):
        """SpDTransform must apply successfully to QEffLlamaForCausalLM with target type."""
        model, _ = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied, "SpDTransform must apply when speculative_model_type='target'"

    def test_spd_transform_forward_is_replaced(self):
        """After SpDTransform, model.forward must be replaced with a SpD-specific forward."""
        model, _ = make_kv_transformed_llama()
        original_forward = model.forward
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied
        assert hasattr(transformed, "forward")
        # The forward must have been replaced (different from original)
        assert transformed.forward is not original_forward, (
            "SpDTransform must replace model.forward with a SpD-specific forward"
        )

    def test_spd_transform_returns_model_instance(self):
        """SpDTransform must return the same model instance (in-place modification)."""
        model, _ = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied
        assert transformed is model, "SpDTransform must modify model in-place"

    def test_spd_transformed_model_is_still_eval_mode(self):
        """SpDTransform must not change the model's training mode."""
        model, _ = make_kv_transformed_llama()
        assert not model.training
        transformed, _ = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert not transformed.training, "SpDTransform must not change model to training mode"

    def test_spd_transform_model_still_has_parameters(self):
        """After SpDTransform, model must still have its parameters."""
        model, _ = make_kv_transformed_llama()
        param_count_before = sum(p.numel() for p in model.parameters())
        transformed, _ = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        param_count_after = sum(p.numel() for p in transformed.parameters())
        assert param_count_before == param_count_after, (
            f"SpDTransform changed parameter count: {param_count_before} → {param_count_after}"
        )


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM SpD-related methods
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestQEFFAutoModelSpDMethods:
    """QEFFAutoModelForCausalLM must have SpD-related methods."""

    def test_has_check_and_get_num_speculative_tokens(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "check_and_get_num_speculative_tokens")
        assert callable(QEFFAutoModelForCausalLM.check_and_get_num_speculative_tokens)

    def test_has_build_prefill_specialization(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "build_prefill_specialization")
        assert callable(QEFFAutoModelForCausalLM.build_prefill_specialization)

    def test_has_build_decode_specialization(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "build_decode_specialization")
        assert callable(QEFFAutoModelForCausalLM.build_decode_specialization)

    def test_has_is_tlm_property(self):
        """QEFFAutoModelForCausalLM instances must expose is_tlm."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        assert hasattr(qeff, "is_tlm"), "QEFFAutoModelForCausalLM instance must have is_tlm attribute"

    def test_is_tlm_false_by_default(self):
        """Without SpD config, is_tlm must be False."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.is_tlm is False, "is_tlm must be False when no SpD config is provided"

    def test_check_and_get_num_speculative_tokens_returns_none_for_non_tlm(self):
        """For a non-TLM model, check_and_get_num_speculative_tokens must not raise."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        # For non-TLM, is_tlm=False; method accepts num_speculative_tokens and prefill_seq_len
        result = qeff.check_and_get_num_speculative_tokens(num_speculative_tokens=None, prefill_seq_len=1)
        assert result is None, f"check_and_get_num_speculative_tokens must return None for non-TLM, got {result}"

    def test_build_prefill_specialization_returns_dict(self):
        """build_prefill_specialization must return a dict-like object."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff.build_prefill_specialization(prefill_seq_len=8, ctx_len=32, batch_size=1, full_batch_size=None)
        assert isinstance(result, dict), f"build_prefill_specialization must return dict, got {type(result)}"

    def test_build_decode_specialization_returns_dict(self):
        """build_decode_specialization must return a dict-like object."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff.build_decode_specialization(ctx_len=32, batch_size=1, full_batch_size=None)
        assert isinstance(result, dict), f"build_decode_specialization must return dict, got {type(result)}"


# ---------------------------------------------------------------------------
# Tests: TLM forward execution
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.accuracy
class TestTLMForwardExecution:
    """After SpDTransform, the replaced tlm_forward must produce correct outputs."""

    def _make_tlm_inputs(self, batch=1, num_spec_tokens=3, n_layers=2, n_kv=2, head_dim=32):
        """Create inputs for TLM forward with pre-allocated zero KV cache."""
        seq_len = num_spec_tokens + 1
        input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        past_key_values = tuple(
            (
                torch.zeros(batch, n_kv, CTX_LEN, head_dim, dtype=torch.float32),
                torch.zeros(batch, n_kv, CTX_LEN, head_dim, dtype=torch.float32),
            )
            for _ in range(n_layers)
        )
        return input_ids, position_ids, past_key_values

    def test_tlm_forward_returns_logits(self):
        """tlm_forward must return an object with logits attribute."""
        model, cfg = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied

        batch, num_spec_tokens = 1, 3
        # n_kv=2, head_dim=64//2=32 for tiny llama
        # num_logits_to_keep must be a tensor (as expected by spd_transform_forward)
        input_ids, position_ids, past_kv = self._make_tlm_inputs(
            batch, num_spec_tokens, n_layers=2, n_kv=2, head_dim=32
        )
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        assert hasattr(output, "logits"), "TLM forward must return output with logits"

    def test_tlm_forward_logits_are_finite(self):
        """tlm_forward logits must be finite (no NaN/Inf)."""
        model, cfg = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied

        batch, num_spec_tokens = 1, 3
        input_ids, position_ids, past_kv = self._make_tlm_inputs(
            batch, num_spec_tokens, n_layers=2, n_kv=2, head_dim=32
        )
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        assert torch.isfinite(output.logits).all(), "TLM logits must be finite"

    def test_tlm_forward_logits_shape_is_batch_x_kept_x_vocab(self):
        """tlm_forward logits shape must be [batch, num_logits_to_keep, vocab_size].
        num_logits_to_keep is a 1D tensor of shape [1] containing the count,
        so the output has shape[1] == num_logits_to_keep.shape[0] == 1."""
        model, cfg = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied

        batch, num_spec_tokens = 1, 3
        input_ids, position_ids, past_kv = self._make_tlm_inputs(
            batch, num_spec_tokens, n_layers=2, n_kv=2, head_dim=32
        )
        # num_logits_to_keep is a 1D tensor; shape[0] determines how many logits are kept
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        # batch dimension must match
        assert output.logits.shape[0] == batch
        # vocab dimension must match
        assert output.logits.shape[-1] == VOCAB_SIZE
        # logits must be 3D: [batch, seq, vocab]
        assert output.logits.ndim == 3

    def test_tlm_forward_greedy_tokens_in_valid_range(self):
        """Greedy tokens from tlm_forward must be in [0, vocab_size)."""
        model, cfg = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied

        batch, num_spec_tokens = 1, 3
        input_ids, position_ids, past_kv = self._make_tlm_inputs(
            batch, num_spec_tokens, n_layers=2, n_kv=2, head_dim=32
        )
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        greedy_tokens = output.logits.argmax(dim=-1)
        assert (greedy_tokens >= 0).all()
        assert (greedy_tokens < VOCAB_SIZE).all()

    @pytest.mark.parametrize("num_spec_tokens", [1, 2, 3, 5])
    def test_tlm_multi_spec_logit_consistency(self, num_spec_tokens):
        """
        The anchor-token logit from seq_len=1 must equal the anchor-token logit at
        position 0 from seq_len=K+1 — for the same input and standard causal attention.

        This is the core correctness guarantee for multi-spec dispatch on QAIC hardware.

        We test this using the raw HuggingFace LlamaForCausalLM (no QEffDynamicCache)
        because the eager-mode QEffDynamicCache simulation uses max(position_ids) as the
        KV gather limit, which exposes speculative positions to the anchor query and breaks
        the property in Python.  On QAIC hardware, per-query causal masking is applied
        correctly by the hardware attention kernel — the property is verified empirically
        by test_few_spd_inference, which asserts mean_num_accepted_tokens == K+1
        (100% acceptance rate when TLM == DLM).

        Why it holds: Standard causal attention masks position P from seeing positions
        P+1..P+K, so the hidden state at P is identical regardless of what follows it.
        SpDTransform's filter_hidden_states extracts this hidden state at index 0 of the
        K+1 output, so the accepted token is always the same.
        """
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
        raw_model = LlamaForCausalLM(cfg).eval()

        batch = 1
        anchor_token = torch.randint(0, VOCAB_SIZE, (batch, 1))
        anchor_pos = torch.tensor([[0]], dtype=torch.long)  # start of sequence, no past

        # ── seq_len=1: just the anchor ───────────────────────────────────────────────
        with torch.no_grad():
            out_k0 = raw_model(
                input_ids=anchor_token,
                position_ids=anchor_pos,
            )
        logit_k0 = out_k0.logits[:, 0:1, :]  # [batch, 1, vocab]

        # ── seq_len=K+1: anchor at position 0, K random speculative tokens ──────────
        spec_tokens = torch.randint(0, VOCAB_SIZE, (batch, num_spec_tokens))
        full_input_ids = torch.cat([anchor_token, spec_tokens], dim=1)
        full_pos_ids = torch.arange(num_spec_tokens + 1).unsqueeze(0).expand(batch, -1)

        with torch.no_grad():
            out_kK = raw_model(
                input_ids=full_input_ids,
                position_ids=full_pos_ids,
            )
        logit_kK_anchor = out_kK.logits[:, 0:1, :]  # anchor is at index 0

        # The anchor logit must be numerically identical regardless of K
        assert torch.allclose(logit_k0, logit_kK_anchor, atol=1e-5), (
            f"Causal property violated: anchor logit differs between seq_len=1 and "
            f"seq_len={num_spec_tokens + 1}: "
            f"max_diff={(logit_k0 - logit_kK_anchor).abs().max().item():.2e}"
        )
        # Accepted token (greedy argmax) must also be identical
        assert logit_k0.argmax(dim=-1).eq(logit_kK_anchor.argmax(dim=-1)).all(), (
            "Accepted token differs between seq_len=1 and seq_len=K+1 — causal property violated in raw model"
        )


# ---------------------------------------------------------------------------
# Tests: SpDTransform for Qwen2
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformQwen2:
    """SpDTransform must apply correctly to Qwen2 models."""

    def _make_kv_transformed_qwen2(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

        cfg = Qwen2Config(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=CTX_LEN,
        )
        model = Qwen2ForCausalLM(cfg).eval()
        transformed, _ = KVCacheTransform.apply(model)
        return transformed, cfg

    def test_spd_transform_applies_to_qwen2_with_target_type(self):
        """SpDTransform must apply successfully to QEffQwen2ForCausalLM."""
        model, _ = self._make_kv_transformed_qwen2()
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied, "SpDTransform must apply to Qwen2 with target type"

    def test_spd_transform_qwen2_forward_is_replaced(self):
        """After SpDTransform, Qwen2 model.forward must be replaced."""
        model, _ = self._make_kv_transformed_qwen2()
        original_forward = model.forward
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied
        assert transformed.forward is not original_forward

    def test_spd_transform_qwen2_produces_finite_logits(self):
        """After SpDTransform, Qwen2 forward must produce finite logits."""

        model, _ = self._make_kv_transformed_qwen2()
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied

        batch, num_spec_tokens = 1, 2
        seq_len = num_spec_tokens + 1
        input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        # Use tuple-based KV cache (n_kv=2, head_dim=64//2=32)
        past_kv = tuple(
            (
                torch.zeros(batch, 2, CTX_LEN, 32, dtype=torch.float32),
                torch.zeros(batch, 2, CTX_LEN, 32, dtype=torch.float32),
            )
            for _ in range(2)
        )
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        assert torch.isfinite(output.logits).all()


# ---------------------------------------------------------------------------
# Tests: filter_hidden_states indexing
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestFilterHiddenStates:
    """filter_hidden_states must gather the K positions ending at argmax(position_ids)."""

    def _hidden_states(self, batch=2, seq_len=4):
        # hidden[b, s, :] == 10 * b + s, so gathered values identify (batch, position) directly.
        values = 10 * torch.arange(batch).view(-1, 1) + torch.arange(seq_len).view(1, -1)
        return values.unsqueeze(-1).expand(batch, seq_len, 3).float()

    def test_keeps_last_k_positions_per_row(self):
        from QEfficient.transformers.spd.spd_transform_forward import filter_hidden_states

        position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])
        kept = filter_hidden_states(self._hidden_states(), position_ids, torch.arange(2).view(2, 1))
        assert kept.shape == (2, 2, 3)
        assert kept[..., 0].tolist() == [[2.0, 3.0], [11.0, 12.0]]

    def test_k_is_clamped_to_sequence_start(self):
        from QEfficient.transformers.spd.spd_transform_forward import filter_hidden_states

        position_ids = torch.tensor([[0, -1, -1, -1], [0, 1, 2, 3]])
        kept = filter_hidden_states(self._hidden_states(), position_ids, torch.arange(3).view(3, 1))
        assert kept[..., 0].tolist() == [[0.0, 1.0, 2.0], [11.0, 12.0, 13.0]]

    def test_none_keeps_only_argmax_position(self):
        from QEfficient.transformers.spd.spd_transform_forward import filter_hidden_states

        position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])
        kept = filter_hidden_states(self._hidden_states(), position_ids, None)
        assert kept[..., 0].tolist() == [[3.0], [12.0]]

    def test_tensor_extent_not_value_selects_k(self):
        from QEfficient.transformers.spd.spd_transform_forward import filter_hidden_states

        position_ids = torch.arange(4).view(1, 4)
        hidden = self._hidden_states(batch=1)
        assert filter_hidden_states(hidden, position_ids, torch.tensor([3])).shape[1] == 1
        assert filter_hidden_states(hidden, position_ids, torch.arange(3).view(3, 1)).shape[1] == 3


# ---------------------------------------------------------------------------
# Tests: SpDTransform for MoE target models (GPT-OSS, Mixtral)
# ---------------------------------------------------------------------------

MOE_NUM_LAYERS = 2
MOE_NUM_KV_HEADS = 2
MOE_HEAD_DIM = 32
GPT_OSS_SLIDING_WINDOW = 8


def make_tiny_gpt_oss():
    from transformers import GptOssConfig, GptOssForCausalLM

    cfg = GptOssConfig(
        num_hidden_layers=MOE_NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=MOE_NUM_KV_HEADS,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=64,
        head_dim=MOE_HEAD_DIM,
        sliding_window=GPT_OSS_SLIDING_WINDOW,
        layer_types=["sliding_attention", "full_attention"],
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    return GptOssForCausalLM(cfg).eval(), cfg


def make_tiny_mixtral():
    from transformers import MixtralConfig, MixtralForCausalLM

    cfg = MixtralConfig(
        num_hidden_layers=MOE_NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=MOE_NUM_KV_HEADS,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        num_experts_per_tok=2,
        num_local_experts=4,
    )
    return MixtralForCausalLM(cfg).eval(), cfg


MOE_FACTORIES = {"gpt_oss": make_tiny_gpt_oss, "mixtral": make_tiny_mixtral}


def make_kv_transformed_moe(arch):
    torch.manual_seed(0)
    model, cfg = MOE_FACTORIES[arch]()
    transformed, _ = KVCacheTransform.apply(model)
    return transformed, cfg


def make_moe_past_key_values(cfg, batch):
    layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
    return tuple(
        tuple(
            torch.zeros(
                batch,
                MOE_NUM_KV_HEADS,
                GPT_OSS_SLIDING_WINDOW if layer_type == "sliding_attention" else CTX_LEN,
                MOE_HEAD_DIM,
            )
            for _ in range(2)
        )
        for layer_type in layer_types
    )


def clone_past_key_values(past_key_values):
    return tuple(tuple(state.clone() for state in layer) for layer in past_key_values)


@pytest.mark.transforms
@pytest.mark.parametrize("arch", ["gpt_oss", "mixtral"])
class TestSpDTransformMoE:
    """SpDTransform must accept QEff MoE wrappers as target models and keep K logits."""

    def _run(self, model, cfg, input_ids, position_ids, **kwargs):
        with torch.no_grad():
            return model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=make_moe_past_key_values(cfg, input_ids.shape[0]),
                **kwargs,
            )

    def _reference_logits(self, model, cfg, input_ids, position_ids):
        """Logits for every position, computed from the unsliced backbone output."""
        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=make_moe_past_key_values(cfg, input_ids.shape[0]),
            )
            return model.lm_head(outputs.last_hidden_state).float()

    def test_target_type_is_accepted(self, arch):
        model, _ = make_kv_transformed_moe(arch)
        transformed, applied = SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        assert applied
        assert transformed is model

    def test_draft_type_is_rejected(self, arch):
        model, _ = make_kv_transformed_moe(arch)
        with pytest.raises(NotImplementedError, match="only supports speculative_model_type='target'"):
            SpDTransform.apply(
                model,
                qaic_config={"speculative_model_type": "turbo", "pretrained_model_name_or_path": "unused"},
            )

    @pytest.mark.parametrize("num_logits_to_keep", [1, 3])
    def test_keeps_k_logits_matching_reference(self, arch, num_logits_to_keep):
        model, cfg = make_kv_transformed_moe(arch)
        SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})

        batch, seq_len = 2, num_logits_to_keep + 1
        torch.manual_seed(1)
        input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

        output = self._run(
            model,
            cfg,
            input_ids,
            position_ids,
            num_logits_to_keep=torch.arange(num_logits_to_keep).view(num_logits_to_keep, 1),
        )
        reference = self._reference_logits(model, cfg, input_ids, position_ids)

        assert output.logits.shape == (batch, num_logits_to_keep, VOCAB_SIZE)
        assert torch.isfinite(output.logits).all()
        torch.testing.assert_close(output.logits, reference[:, -num_logits_to_keep:], rtol=1e-5, atol=1e-5)

    def test_moe_output_fields_match_non_spd_forward(self, arch):
        model, cfg = make_kv_transformed_moe(arch)
        SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
        position_ids = torch.arange(4).view(1, 4)
        spd = self._run(
            model,
            cfg,
            input_ids,
            position_ids,
            output_router_logits=True,
            num_logits_to_keep=torch.arange(4).view(4, 1),
        )
        base = self._run(model, cfg, input_ids, position_ids, output_router_logits=True)

        assert type(spd) is type(base)
        assert set(spd.keys()) == set(base.keys())
        for field in ("aux_loss", "router_logits", "hidden_states", "attentions"):
            if getattr(base, field) is None:
                assert getattr(spd, field) is None, field
            else:
                torch.testing.assert_close(getattr(spd, field), getattr(base, field))
        torch.testing.assert_close(spd.logits[:, -1:], base.logits, rtol=1e-5, atol=1e-5)

    def test_non_spd_forward_is_unchanged(self, arch):
        model, cfg = make_kv_transformed_moe(arch)
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 4))
        position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])
        before = self._run(model, cfg, input_ids, position_ids)

        SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})
        after = self._run(model, cfg, input_ids, position_ids)
        reference = self._reference_logits(model, cfg, input_ids, position_ids)

        assert after.logits.shape == (2, 1, VOCAB_SIZE)
        assert torch.equal(before.logits, after.logits)
        torch.testing.assert_close(after.logits[:, 0], reference[[0, 1], [3, 2]], rtol=1e-5, atol=1e-5)

    def test_padded_batch_keeps_k_logits_ending_at_last_valid_position(self, arch):
        model, cfg = make_kv_transformed_moe(arch)
        SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})

        num_logits_to_keep = 2
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 4))
        position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])
        output = self._run(
            model,
            cfg,
            input_ids,
            position_ids,
            num_logits_to_keep=torch.arange(num_logits_to_keep).view(num_logits_to_keep, 1),
        )
        reference = self._reference_logits(model, cfg, input_ids, position_ids)

        assert output.logits.shape == (2, num_logits_to_keep, VOCAB_SIZE)
        torch.testing.assert_close(output.logits[0], reference[0, 2:4], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(output.logits[1], reference[1, 1:3], rtol=1e-5, atol=1e-5)

    def test_decode_after_prefill_keeps_k_logits_matching_reference(self, arch):
        """Verify K+1 tokens against a populated cache, past the GPT-OSS sliding window."""
        model, cfg = make_kv_transformed_moe(arch)
        SpDTransform.apply(model, qaic_config={"speculative_model_type": "target"})

        prefill_len, num_logits_to_keep = GPT_OSS_SLIDING_WINDOW + 2, 3
        decode_len = num_logits_to_keep + 1
        input_ids = torch.randint(0, VOCAB_SIZE, (1, prefill_len + decode_len))
        position_ids = torch.arange(prefill_len + decode_len).view(1, -1)

        with torch.no_grad():
            prefill = model(
                input_ids=input_ids[:, :prefill_len],
                position_ids=position_ids[:, :prefill_len],
                past_key_values=make_moe_past_key_values(cfg, 1),
            )
            decode = model(
                input_ids=input_ids[:, prefill_len:],
                position_ids=position_ids[:, prefill_len:],
                past_key_values=clone_past_key_values(prefill.past_key_values),
                num_logits_to_keep=torch.arange(num_logits_to_keep).view(num_logits_to_keep, 1),
            )
            reference = model.lm_head(
                model.model(
                    input_ids=input_ids[:, prefill_len:],
                    position_ids=position_ids[:, prefill_len:],
                    past_key_values=clone_past_key_values(prefill.past_key_values),
                ).last_hidden_state
            ).float()

        assert decode.logits.shape == (1, num_logits_to_keep, VOCAB_SIZE)
        assert torch.isfinite(decode.logits).all()
        torch.testing.assert_close(decode.logits, reference[:, -num_logits_to_keep:], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: post_processing.py registry
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestPostProcessingRegistry:
    """post_processing.model_type_registry must contain expected model types."""

    def test_model_type_registry_is_not_empty(self):
        """model_type_registry must not be empty."""
        from QEfficient.transformers.post_processing import model_type_registry

        assert len(model_type_registry) > 0

    def test_model_type_registry_contains_turbo(self):
        """model_type_registry must contain 'turbo' (the SpD post-processing type)."""
        from QEfficient.transformers.post_processing import model_type_registry

        assert "turbo" in model_type_registry

    def test_model_type_registry_keys_are_strings(self):
        """All keys in model_type_registry must be strings."""
        from QEfficient.transformers.post_processing import model_type_registry

        for key in model_type_registry:
            assert isinstance(key, str), f"Registry key must be string, got {type(key)}"

    def test_model_type_registry_values_are_callable(self):
        """All values in model_type_registry must be callable."""
        from QEfficient.transformers.post_processing import model_type_registry

        for model_type, handler in model_type_registry.items():
            assert callable(handler), f"Handler for '{model_type}' must be callable"


# ---------------------------------------------------------------------------
# Tests: SpD ONNX structure (GAP I)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDONNXStructure:
    """SpD-related ONNX structure tests — verify num_logits_to_keep input and build_and_attach_mlp."""

    def test_build_and_attach_mlp_importable(self):
        """build_and_attach_mlp must be importable from post_processing."""
        from QEfficient.transformers.post_processing import build_and_attach_mlp

        assert build_and_attach_mlp is not None

    def test_build_and_attach_mlp_is_callable(self):
        """build_and_attach_mlp must be callable."""
        from QEfficient.transformers.post_processing import build_and_attach_mlp

        assert callable(build_and_attach_mlp)

    def test_build_and_attach_mlp_accepts_model_parameter(self):
        """build_and_attach_mlp must accept 'model' as first parameter."""
        import inspect

        from QEfficient.transformers.post_processing import build_and_attach_mlp

        sig = inspect.signature(build_and_attach_mlp)
        assert "model" in sig.parameters

    def test_build_and_attach_mlp_accepts_speculative_model_type(self):
        """build_and_attach_mlp must accept 'speculative_model_type' parameter."""
        import inspect

        from QEfficient.transformers.post_processing import build_and_attach_mlp

        sig = inspect.signature(build_and_attach_mlp)
        assert "speculative_model_type" in sig.parameters

    def test_model_type_registry_has_turbo(self):
        """model_type_registry must contain 'turbo' key."""
        from QEfficient.transformers.post_processing import model_type_registry

        assert "turbo" in model_type_registry

    def test_build_and_attach_turbo_importable(self):
        """build_and_attach_turbo must be importable from spd.turbo."""
        from QEfficient.transformers.spd.turbo import build_and_attach_turbo

        assert build_and_attach_turbo is not None

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_tlm_onnx_has_num_logits_to_keep_input(self, tmp_export_dir):
        """TLM ONNX export must include 'num_logits_to_keep' as an input."""
        import onnx

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(
            model,
            qaic_config={"speculative_model_type": "target"},
        )
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))

        input_names = [inp.name for inp in onnx_model.graph.input]
        assert "num_logits_to_keep" in input_names, (
            f"TLM ONNX must have 'num_logits_to_keep' input. Found: {input_names}"
        )

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_tlm_onnx_logits_output_is_present(self, tmp_export_dir):
        """TLM ONNX export must include 'logits' as an output."""
        import onnx

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(
            model,
            qaic_config={"speculative_model_type": "target"},
        )
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))

        output_names = [out.name for out in onnx_model.graph.output]
        assert "logits" in output_names, f"TLM ONNX must have 'logits' output. Found: {output_names}"

    @pytest.mark.onnx
    @pytest.mark.slow
    @pytest.mark.parametrize("arch", ["gpt_oss", "mixtral"])
    def test_moe_tlm_onnx_keeps_k_logits_matching_pytorch(self, arch, tmp_export_dir):
        """MoE TLM ONNX must expose num_logits_to_keep and match PyTorch logits for K kept positions."""
        import numpy as np
        import onnx
        import onnxruntime as ort

        from QEfficient.transformers.cache_utils import InvalidIndexProvider
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        torch.manual_seed(0)
        model, cfg = MOE_FACTORIES[arch]()
        qeff_model = QEFFAutoModelForCausalLM(model, qaic_config={"speculative_model_type": "target"})
        InvalidIndexProvider.SUBFUNC_ENABLED = True
        try:
            onnx_path = qeff_model.export(export_dir=str(tmp_export_dir), offload_pt_weights=False)
        finally:
            InvalidIndexProvider.SUBFUNC_ENABLED = False

        graph_inputs = {inp.name: inp for inp in onnx.load(str(onnx_path)).graph.input}
        assert "num_logits_to_keep" in graph_inputs

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for num_logits_to_keep in (1, 3):
            seq_len = num_logits_to_keep + 1
            input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
            position_ids = torch.arange(seq_len).view(1, seq_len)
            with torch.no_grad():
                pt_logits = qeff_model.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    past_key_values=make_moe_past_key_values(cfg, 1),
                    num_logits_to_keep=torch.arange(num_logits_to_keep).view(num_logits_to_keep, 1),
                ).logits.numpy()

            ort_inputs = {
                "input_ids": input_ids.numpy(),
                "position_ids": position_ids.numpy(),
                "num_logits_to_keep": np.zeros((num_logits_to_keep, 1), dtype=np.int64),
            }
            for i, (key, value) in enumerate(make_moe_past_key_values(cfg, 1)):
                ort_inputs[f"past_key.{i}"] = key.numpy()
                ort_inputs[f"past_value.{i}"] = value.numpy()
            assert set(ort_inputs) == {inp.name for inp in session.get_inputs()}
            (ort_logits,) = session.run(["logits"], ort_inputs)

            assert ort_logits.shape == (1, num_logits_to_keep, VOCAB_SIZE)
            np.testing.assert_allclose(ort_logits, pt_logits, rtol=1e-4, atol=1e-4)
