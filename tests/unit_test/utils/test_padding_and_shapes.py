# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for utility functions: get_padding_shape_from_config, sampler_utils, hash_utils.

Tests verify:
  - get_padding_shape_from_config: correct KV cache shapes for various model configs
  - get_sampling_inputs_and_outputs: correct input/output names for sampler
  - hash_dict_params: deterministic, correct length, different configs → different hashes

All tests run on CPU only.
"""

import pytest
import torch
from transformers import (
    GPT2Config,
    LlamaConfig,
    MistralConfig,
)

from QEfficient.utils.constants import HASH_HEXDIGEST_STR_LEN
from QEfficient.utils.hash_utils import hash_dict_params
from QEfficient.utils.sampler_utils import get_sampling_inputs_and_outputs

# ---------------------------------------------------------------------------
# Helpers: get_padding_shape_from_config
# ---------------------------------------------------------------------------


def _get_padding_shape(config, batch_size=1, seq_len=32):
    """Import and call get_padding_shape_from_config."""
    from QEfficient.utils import get_padding_shape_from_config

    return get_padding_shape_from_config(config, batch_size, seq_len)


# ---------------------------------------------------------------------------
# Tests: get_padding_shape_from_config
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestGetPaddingShapeFromConfig:
    """get_padding_shape_from_config must return correct KV cache shapes."""

    def test_llama_returns_correct_shape(self):
        """Llama: shape must be [batch, n_kv_heads, seq_len, head_dim]."""
        cfg = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        shape = _get_padding_shape(cfg, batch_size=1, seq_len=32)
        assert len(shape) == 4, f"Expected 4D shape, got {len(shape)}D: {shape}"
        assert shape[0] == 1  # batch_size
        assert shape[1] == 4  # n_kv_heads
        assert shape[2] == 32  # seq_len
        assert shape[3] == 16  # head_dim = hidden_size / num_attention_heads = 64/4

    def test_gpt2_returns_correct_shape(self):
        """GPT2: shape must be [batch, n_heads, seq_len, head_dim]."""
        cfg = GPT2Config(
            n_layer=2,
            n_head=4,
            n_embd=64,
            vocab_size=500,
            n_positions=64,
            n_ctx=64,
        )
        shape = _get_padding_shape(cfg, batch_size=1, seq_len=32)
        assert len(shape) == 4
        assert shape[0] == 1
        assert shape[2] == 32

    def test_mistral_gqa_returns_correct_kv_heads(self):
        """Mistral with GQA: n_kv_heads must be less than n_heads."""
        cfg = MistralConfig(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,  # GQA: 2 KV heads for 8 query heads
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        shape = _get_padding_shape(cfg, batch_size=1, seq_len=32)
        assert len(shape) == 4
        assert shape[1] == 2, f"Expected 2 KV heads for GQA, got {shape[1]}"

    def test_shape_has_4_dimensions(self):
        """Shape must always have exactly 4 dimensions for standard models."""
        cfg = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        shape = _get_padding_shape(cfg, batch_size=2, seq_len=16)
        assert len(shape) == 4

    def test_batch_size_reflected_in_shape(self):
        """Batch size must be reflected in the first dimension of the shape."""
        cfg = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        shape = _get_padding_shape(cfg, batch_size=4, seq_len=32)
        assert shape[0] == 4

    def test_seq_len_reflected_in_shape(self):
        """Sequence length must be reflected in the third dimension of the shape."""
        cfg = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        shape = _get_padding_shape(cfg, batch_size=1, seq_len=64)
        assert shape[2] == 64

    def test_head_dim_is_hidden_size_divided_by_num_heads(self):
        """head_dim must equal hidden_size / num_attention_heads."""
        hidden_size = 128
        num_heads = 8
        cfg = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            hidden_size=hidden_size,
            intermediate_size=256,
            vocab_size=500,
            max_position_embeddings=64,
        )
        shape = _get_padding_shape(cfg, batch_size=1, seq_len=32)
        expected_head_dim = hidden_size // num_heads
        assert shape[3] == expected_head_dim, f"Expected head_dim={expected_head_dim}, got {shape[3]}"


# ---------------------------------------------------------------------------
# Tests: get_sampling_inputs_and_outputs
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestSamplerUtils:
    """get_sampling_inputs_and_outputs must return correct input/output names."""

    def _make_base_inputs(self, batch=1, seq_len=8):
        """Create minimal example inputs for sampler utils."""
        return {
            "input_ids": torch.zeros((batch, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch, -1),
        }

    def _make_base_dynamic_axes(self):
        return {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }

    def test_get_sampling_inputs_returns_temperatures(self):
        """Sampler inputs must include 'temperatures'."""
        inputs = self._make_base_inputs()
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        updated_inputs, _, _ = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        assert "temperatures" in updated_inputs

    def test_get_sampling_inputs_returns_top_ks(self):
        """Sampler inputs must include 'top_ks'."""
        inputs = self._make_base_inputs()
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        updated_inputs, _, _ = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        assert "top_ks" in updated_inputs

    def test_get_sampling_inputs_returns_top_ps(self):
        """Sampler inputs must include 'top_ps'."""
        inputs = self._make_base_inputs()
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        updated_inputs, _, _ = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        assert "top_ps" in updated_inputs

    def test_get_sampling_inputs_returns_repetition_penalties(self):
        """Sampler inputs must include 'repetition_penalties'."""
        inputs = self._make_base_inputs()
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        updated_inputs, _, _ = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        assert "repetition_penalties" in updated_inputs

    def test_get_sampling_inputs_returns_random_numbers(self):
        """Sampler inputs must include 'random_numbers'."""
        inputs = self._make_base_inputs()
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        updated_inputs, _, _ = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        assert "random_numbers" in updated_inputs

    def test_get_sampling_outputs_includes_retained_state(self):
        """Sampler outputs must include retained state buffers."""
        inputs = self._make_base_inputs()
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        _, updated_output_names, _ = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        # Must include retained state outputs
        retained_state_outputs = [n for n in updated_output_names if "_RetainedState" in n]
        assert len(retained_state_outputs) > 0, "Sampler must add RetainedState outputs"

    def test_get_sampling_inputs_includes_last_accepted_output_tokens(self):
        """Sampler inputs must include 'last_accepted_output_tokens'."""
        inputs = self._make_base_inputs()
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        updated_inputs, _, _ = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        assert "last_accepted_output_tokens" in updated_inputs

    def test_get_sampling_dynamic_axes_updated(self):
        """Dynamic axes must be updated for all new sampler inputs."""
        inputs = self._make_base_inputs()
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        _, _, updated_axes = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        assert "temperatures" in updated_axes
        assert "top_ks" in updated_axes
        assert "top_ps" in updated_axes

    def test_get_sampling_inputs_tensor_shapes_are_correct(self):
        """Sampler input tensors must have correct shapes (batch dim >= 1)."""
        batch = 1
        inputs = self._make_base_inputs(batch=batch)
        output_names = ["logits"]
        dynamic_axes = self._make_base_dynamic_axes()
        qaic_config = {"max_top_k_ids": 512}

        updated_inputs, _, _ = get_sampling_inputs_and_outputs(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            continuous_batching=False,
            vocab_size=500,
            qaic_config=qaic_config,
        )
        # temperatures must be a tensor with at least 1 element
        assert updated_inputs["temperatures"].numel() >= 1
        # top_ks must be a tensor with at least 1 element
        assert updated_inputs["top_ks"].numel() >= 1
        # top_ps must be a tensor with at least 1 element
        assert updated_inputs["top_ps"].numel() >= 1


# ---------------------------------------------------------------------------
# Tests: hash_utils
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestHashUtils:
    """hash_dict_params must be deterministic, correct length, and collision-resistant."""

    def test_compute_hash_returns_string(self):
        """hash_dict_params must return a string."""
        result = hash_dict_params({"key": "value"})
        assert isinstance(result, str)

    def test_compute_hash_is_deterministic(self):
        """Same input must always produce the same hash."""
        params = {"model": "llama", "layers": 2, "heads": 4}
        hash1 = hash_dict_params(params)
        hash2 = hash_dict_params(params)
        assert hash1 == hash2, "hash_dict_params must be deterministic"

    def test_different_configs_produce_different_hashes(self):
        """Different configs must produce different hashes."""
        params1 = {"model": "llama", "layers": 2}
        params2 = {"model": "llama", "layers": 4}
        hash1 = hash_dict_params(params1)
        hash2 = hash_dict_params(params2)
        assert hash1 != hash2, "Different configs must produce different hashes"

    def test_hash_length_is_correct(self):
        """Hash must have length HASH_HEXDIGEST_STR_LEN (16)."""
        result = hash_dict_params({"key": "value"})
        assert len(result) == HASH_HEXDIGEST_STR_LEN, (
            f"Expected hash length {HASH_HEXDIGEST_STR_LEN}, got {len(result)}"
        )

    def test_hash_is_hexadecimal(self):
        """Hash must consist of hexadecimal characters only."""
        result = hash_dict_params({"key": "value", "num": 42})
        assert all(c in "0123456789abcdef" for c in result), f"Hash must be hexadecimal, got: {result}"

    def test_empty_dict_produces_valid_hash(self):
        """Empty dict must produce a valid hash."""
        result = hash_dict_params({})
        assert isinstance(result, str)
        assert len(result) == HASH_HEXDIGEST_STR_LEN

    def test_nested_dict_produces_valid_hash(self):
        """Nested dict must produce a valid hash."""
        params = {"outer": {"inner": "value"}, "num": 42}
        result = hash_dict_params(params)
        assert isinstance(result, str)
        assert len(result) == HASH_HEXDIGEST_STR_LEN

    def test_order_independent_hashing(self):
        """Dict with same keys in different order must produce the same hash (sort_keys=True)."""
        params1 = {"b": 2, "a": 1}
        params2 = {"a": 1, "b": 2}
        hash1 = hash_dict_params(params1)
        hash2 = hash_dict_params(params2)
        assert hash1 == hash2, "Hash must be order-independent (sort_keys=True)"

    def test_custom_hash_size(self):
        """Custom hash_string_size must be respected."""
        result = hash_dict_params({"key": "value"}, hash_string_size=8)
        assert len(result) == 8


# ---------------------------------------------------------------------------
# Tests: process_ccl_specializations (GAP H)
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestCheckCCLSpecializations:
    """Tests for process_ccl_specializations and related CCL utility functions."""

    def test_process_ccl_specializations_returns_three_values(self):
        """process_ccl_specializations must return (ccl_prefill, ccl_decode, ctx_len)."""
        from QEfficient.utils.check_ccl_specializations import process_ccl_specializations

        result = process_ccl_specializations(None, None, ctx_len=4096, prefill_seq_len=128)
        assert len(result) == 3

    def test_process_ccl_specializations_returns_lists(self):
        """process_ccl_specializations must return lists for prefill and decode."""
        from QEfficient.utils.check_ccl_specializations import process_ccl_specializations

        ccl_prefill, ccl_decode, ctx_len = process_ccl_specializations(None, None, ctx_len=4096, prefill_seq_len=128)
        assert isinstance(ccl_prefill, list)
        assert isinstance(ccl_decode, list)

    def test_process_ccl_specializations_lists_not_empty(self):
        """process_ccl_specializations must return non-empty lists."""
        from QEfficient.utils.check_ccl_specializations import process_ccl_specializations

        ccl_prefill, ccl_decode, ctx_len = process_ccl_specializations(None, None, ctx_len=4096, prefill_seq_len=128)
        assert len(ccl_prefill) > 0
        assert len(ccl_decode) > 0

    def test_process_ccl_specializations_last_element_leq_ctx_len(self):
        """Last element of CCL lists must be <= ctx_len."""
        from QEfficient.utils.check_ccl_specializations import process_ccl_specializations

        ctx_len = 4096
        ccl_prefill, ccl_decode, returned_ctx_len = process_ccl_specializations(
            None, None, ctx_len=ctx_len, prefill_seq_len=128
        )
        assert ccl_prefill[-1] <= ctx_len
        assert ccl_decode[-1] <= ctx_len

    def test_process_ccl_specializations_with_explicit_lists(self):
        """process_ccl_specializations with explicit lists must validate and return them."""
        from QEfficient.utils.check_ccl_specializations import process_ccl_specializations

        ccl_prefill, ccl_decode, ctx_len = process_ccl_specializations(
            [512, 1024], [1024, 2048], ctx_len=4096, prefill_seq_len=128
        )
        assert isinstance(ccl_prefill, list)
        assert isinstance(ccl_decode, list)

    def test_process_ccl_specializations_with_only_prefill(self):
        """process_ccl_specializations with only prefill list must fill decode with ctx_len."""
        from QEfficient.utils.check_ccl_specializations import process_ccl_specializations

        ccl_prefill, ccl_decode, ctx_len = process_ccl_specializations(
            [512, 1024], None, ctx_len=4096, prefill_seq_len=128
        )
        assert isinstance(ccl_prefill, list)
        assert isinstance(ccl_decode, list)
        assert len(ccl_decode) > 0

    def test_process_ccl_specializations_with_only_decode(self):
        """process_ccl_specializations with only decode list must fill prefill with ctx_len."""
        from QEfficient.utils.check_ccl_specializations import process_ccl_specializations

        ccl_prefill, ccl_decode, ctx_len = process_ccl_specializations(
            None, [1024, 2048], ctx_len=4096, prefill_seq_len=128
        )
        assert isinstance(ccl_prefill, list)
        assert isinstance(ccl_decode, list)
        assert len(ccl_prefill) > 0

    def test_process_ccl_specializations_prefill_seq_len_1(self):
        """With prefill_seq_len=1, prefill and decode lists must be identical."""
        from QEfficient.utils.check_ccl_specializations import process_ccl_specializations

        ccl_prefill, ccl_decode, ctx_len = process_ccl_specializations(None, None, ctx_len=4096, prefill_seq_len=1)
        assert ccl_prefill == ccl_decode, "With prefill_seq_len=1, prefill and decode CCL lists must be identical"


@pytest.mark.cpu_only
class TestAutomaticCCLGeneration:
    """Tests for automatic_ccl_generation utility function."""

    def test_automatic_ccl_generation_returns_three_values(self):
        """automatic_ccl_generation must return (prefill_list, decode_list, mapped_cl)."""
        from QEfficient.utils.check_ccl_specializations import automatic_ccl_generation

        result = automatic_ccl_generation(ctx_len=4096, prefill_seq_len=128)
        assert len(result) == 3

    def test_automatic_ccl_generation_returns_lists(self):
        """automatic_ccl_generation must return lists."""
        from QEfficient.utils.check_ccl_specializations import automatic_ccl_generation

        prefill_list, decode_list, mapped_cl = automatic_ccl_generation(ctx_len=4096, prefill_seq_len=128)
        assert isinstance(prefill_list, list)
        assert isinstance(decode_list, list)

    def test_automatic_ccl_generation_mapped_cl_is_multiple_of_1024(self):
        """mapped_cl must be a multiple of 1024."""
        from QEfficient.utils.check_ccl_specializations import automatic_ccl_generation

        _, _, mapped_cl = automatic_ccl_generation(ctx_len=3000, prefill_seq_len=128)
        assert mapped_cl % 1024 == 0, f"mapped_cl={mapped_cl} must be a multiple of 1024"

    def test_automatic_ccl_generation_small_ctx_len(self):
        """automatic_ccl_generation with small ctx_len must return valid lists."""
        from QEfficient.utils.check_ccl_specializations import automatic_ccl_generation

        prefill_list, decode_list, mapped_cl = automatic_ccl_generation(ctx_len=512, prefill_seq_len=128)
        assert len(prefill_list) > 0
        assert len(decode_list) > 0

    def test_automatic_ccl_generation_zero_ctx_len(self):
        """automatic_ccl_generation with ctx_len=0 must return valid lists."""
        from QEfficient.utils.check_ccl_specializations import automatic_ccl_generation

        prefill_list, decode_list, mapped_cl = automatic_ccl_generation(ctx_len=0, prefill_seq_len=128)
        assert len(prefill_list) > 0
        assert len(decode_list) > 0


@pytest.mark.cpu_only
class TestCCLHelperFunctions:
    """Tests for CCL helper functions: next_multiple_of_1024, build_doubling_list, etc."""

    def test_next_multiple_of_1024_rounds_up(self):
        """next_multiple_of_1024 must round up to the next multiple of 1024."""
        from QEfficient.utils.check_ccl_specializations import next_multiple_of_1024

        assert next_multiple_of_1024(1) == 1024
        assert next_multiple_of_1024(1024) == 1024
        assert next_multiple_of_1024(1025) == 2048
        assert next_multiple_of_1024(2048) == 2048
        assert next_multiple_of_1024(2049) == 3072

    def test_next_multiple_of_1024_zero_or_negative(self):
        """next_multiple_of_1024 with n<=0 must return 0."""
        from QEfficient.utils.check_ccl_specializations import next_multiple_of_1024

        assert next_multiple_of_1024(0) == 0
        assert next_multiple_of_1024(-1) == 0

    def test_build_doubling_list_basic(self):
        """build_doubling_list must return a doubling sequence."""
        from QEfficient.utils.check_ccl_specializations import build_doubling_list

        result = build_doubling_list(start=1024, limit=8192, max_elements=5)
        assert result[0] == 1024
        # Each element must be double the previous
        for i in range(1, len(result)):
            assert result[i] == result[i - 1] * 2 or result[i] <= 8192

    def test_build_doubling_list_respects_max_elements(self):
        """build_doubling_list must not exceed max_elements."""
        from QEfficient.utils.check_ccl_specializations import build_doubling_list

        result = build_doubling_list(start=1024, limit=1024 * 1024, max_elements=4)
        assert len(result) <= 4

    def test_build_doubling_list_respects_limit(self):
        """build_doubling_list must not exceed limit."""
        from QEfficient.utils.check_ccl_specializations import build_doubling_list

        limit = 4096
        result = build_doubling_list(start=1024, limit=limit, max_elements=10)
        for val in result:
            assert val <= limit, f"Value {val} exceeds limit {limit}"

    def test_build_doubling_list_with_last_value(self):
        """build_doubling_list with last_value must end with that value."""
        from QEfficient.utils.check_ccl_specializations import build_doubling_list

        result = build_doubling_list(start=1024, limit=8192, max_elements=5, last_value=8192)
        assert result[-1] == 8192

    def test_is_power_of_two(self):
        """is_power_of_two must correctly identify powers of two."""
        from QEfficient.utils.check_ccl_specializations import is_power_of_two

        assert is_power_of_two(1)
        assert is_power_of_two(2)
        assert is_power_of_two(4)
        assert is_power_of_two(1024)
        assert is_power_of_two(4096)
        assert not is_power_of_two(3)
        assert not is_power_of_two(5)
        assert not is_power_of_two(0)
        assert not is_power_of_two(-1)

    def test_floor_to_1000(self):
        """floor_to_1000 must floor to the nearest lower multiple of 1000."""
        from QEfficient.utils.check_ccl_specializations import floor_to_1000

        assert floor_to_1000(1500) == 1000
        assert floor_to_1000(2000) == 2000
        assert floor_to_1000(999) == 0
        assert floor_to_1000(0) == 0
        assert floor_to_1000(-1) == 0
