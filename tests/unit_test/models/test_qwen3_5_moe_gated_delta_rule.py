# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import QEffQwen3_5GatedDeltaNet
from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeGatedDeltaNet

MAX_ABS_DEV_RECURSIVE_VS_ORIGINAL = 1e-4
CB_LAYERS = {
    "dense": (Qwen3_5TextConfig, Qwen3_5GatedDeltaNet, QEffQwen3_5GatedDeltaNet),
    "moe": (Qwen3_5MoeTextConfig, Qwen3_5MoeGatedDeltaNet, QEffQwen3_5MoeGatedDeltaNet),
}
NUM_SLOTS = 2


def _build_chunk_masks(chunk_size: int, device: torch.device):
    mask_causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=0)
    mask_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=1)
    eye = torch.eye(chunk_size, dtype=torch.float32, device=device)
    return mask_causal, mask_strict, eye


def _run_chunk_rule(solver: str, chunk_size: int = 8):
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch_size, seq_len, num_heads, k_head_dim, v_head_dim = 2, 11, 3, 4, 4

    # Method only depends on these attrs/helpers; full HF model init is not required for this unit test.
    layer = object.__new__(QEffQwen3_5MoeGatedDeltaNet)
    layer.chunk_gated_delta_solver = solver
    layer.torch_dtype = torch.float32
    layer.head_k_dim = k_head_dim

    query = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32, device=device)
    key = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32, device=device)
    value = torch.randn(batch_size, seq_len, num_heads, v_head_dim, dtype=torch.float32, device=device)
    g = torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32, device=device))

    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len).repeat(1, batch_size, 1)
    position_ids[:, 1, -2:] = -1

    mask_causal, mask_strict, eye = _build_chunk_masks(chunk_size, device)

    output, final_state = layer.torch_chunk_gated_delta_rule_qeff(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        position_ids=position_ids,
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        mask_causal=mask_causal,
        mask_strict=mask_strict,
        ones_lower=None,
        eye=eye,
    )
    return (
        output,
        final_state,
        (batch_size, seq_len, num_heads, v_head_dim),
        (batch_size, num_heads, k_head_dim, v_head_dim),
    )


def test_torch_chunk_gated_delta_rule_qeff_output_and_state_shapes():
    output, final_state, out_shape, state_shape = _run_chunk_rule("recursive_sns")

    assert output.shape == out_shape
    assert final_state is not None
    assert final_state.shape == state_shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


def test_torch_chunk_gated_delta_rule_qeff_supports_all_solver_modes():
    for solver in ("forward_sub", "recursive_sns", "scaled_newton_schulz", "original", "factorized", "horner"):
        output, final_state, _, _ = _run_chunk_rule(solver)
        assert torch.isfinite(output).all(), f"non-finite output for solver={solver}"
        assert torch.isfinite(final_state).all(), f"non-finite final_state for solver={solver}"


def test_torch_chunk_gated_delta_rule_qeff_recursive_sns_matches_original_max_abs_dev():
    output_recursive, _, _, _ = _run_chunk_rule("recursive_sns")
    # TODO: It would be better to test directly between original HF vs QEFF instead of making copy of original code snippet within qeff.
    output_original, _, _, _ = _run_chunk_rule("original")

    max_abs_dev = (output_recursive - output_original).abs().max().item()
    assert max_abs_dev <= MAX_ABS_DEV_RECURSIVE_VS_ORIGINAL, (
        f"max abs deviation {max_abs_dev:.6e} exceeded threshold "
        f"{MAX_ABS_DEV_RECURSIVE_VS_ORIGINAL:.6e} for recursive_sns vs original"
    )


def test_torch_chunk_gated_delta_rule_qeff_forward_sub_matches_original():
    output_forward_sub, state_forward_sub, _, _ = _run_chunk_rule("forward_sub")
    output_original, state_original, _, _ = _run_chunk_rule("original")

    torch.testing.assert_close(output_forward_sub, output_original)
    torch.testing.assert_close(state_forward_sub, state_original)


@pytest.mark.parametrize("layer_type", [QEffQwen3_5GatedDeltaNet, QEffQwen3_5MoeGatedDeltaNet])
def test_recurrent_step_head_blocks_match_reference(layer_type):
    torch.manual_seed(1)
    batch_size, seq_len, num_heads, k_head_dim, v_head_dim = 3, 1, 8, 4, 5
    layer = object.__new__(layer_type)

    query = torch.randn(batch_size, seq_len, num_heads, k_head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, k_head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, v_head_dim)
    g = torch.randn(batch_size, seq_len, num_heads) * 0.1
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads))
    recurrent_state = torch.randn(batch_size, num_heads, k_head_dim, v_head_dim)

    query_reference = query.float().clone()
    key_reference = key.float().clone()
    query_reference *= torch.rsqrt((query_reference * query_reference).sum(dim=-1, keepdim=True) + 1e-6)
    key_reference *= torch.rsqrt((key_reference * key_reference).sum(dim=-1, keepdim=True) + 1e-6)
    query_reference *= 1.0 / (k_head_dim**0.5)
    decayed_state = recurrent_state.float() * g[:, 0].float().exp().unsqueeze(-1).unsqueeze(-1)
    kv_memory = (decayed_state * key_reference[:, 0].unsqueeze(-1)).sum(dim=-2)
    delta = (value[:, 0].float() - kv_memory) * beta[:, 0].float().unsqueeze(-1)
    expected_state = decayed_state + key_reference[:, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    expected_output = (expected_state * query_reference[:, 0].unsqueeze(-1)).sum(dim=-2).unsqueeze(1)

    for gdn_num_head_blocks in (1, 2, 3, 8, 16):
        output, state = layer._recurrent_step_batched(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            gdn_num_head_blocks=gdn_num_head_blocks,
        )
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(state, expected_state)


def test_recurrent_step_rejects_non_decode_sequence():
    layer = object.__new__(QEffQwen3_5MoeGatedDeltaNet)
    query = torch.zeros(1, 2, 1, 2)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    g = torch.zeros(1, 2, 1)
    beta = torch.ones_like(g)
    recurrent_state = torch.zeros(1, 1, 2, 2)

    try:
        layer._recurrent_step_batched(query, key, value, g, beta, recurrent_state)
    except ValueError as error:
        assert "sequence_length == 1" in str(error)
    else:
        raise AssertionError("Expected non-decode recurrence to raise ValueError")


def _make_cb_layer(kind):
    config_cls, hf_cls, qeff_cls = CB_LAYERS[kind]
    config = config_cls(
        hidden_size=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    torch.manual_seed(0)
    layer = hf_cls(config, layer_idx=0)
    layer.__class__ = qeff_cls
    layer.__qeff_init__()
    return layer.eval()


def _random_cb_states(layer, kind):
    conv_shape = (
        (NUM_SLOTS, layer.conv_kernel_size, layer.conv_dim)
        if kind == "moe"
        else (NUM_SLOTS, layer.conv_dim, layer.conv_kernel_size)
    )
    conv = torch.randn(*conv_shape)
    recurrent = torch.randn(NUM_SLOTS, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
    return conv, recurrent


def _run_cb(layer, hidden_states, position_ids, batch_index, conv, recurrent):
    cache = SimpleNamespace(conv_states=[conv.clone()], recurrent_states=[recurrent.clone()])
    # The text model hands the layers mrope position_ids of shape (4, batch, seq).
    mrope_position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)
    with torch.no_grad():
        return layer(
            hidden_states,
            cache_params=cache,
            position_ids=mrope_position_ids,
            batch_index=batch_index,
            return_cache_states=True,
        )


@pytest.mark.parametrize("kind", CB_LAYERS)
def test_padded_batch_slot_does_not_overwrite_live_state(kind):
    """A padded row (batch_index -1) must not write into any slot, including the last one."""
    layer = _make_cb_layer(kind)
    conv, recurrent = _random_cb_states(layer, kind)
    hidden_states = torch.randn(NUM_SLOTS, 1, 32)
    position_ids = torch.tensor([[5], [-1]])
    batch_index = torch.tensor([[0], [-1]])

    _, new_conv, new_recurrent = _run_cb(layer, hidden_states, position_ids, batch_index, conv, recurrent)

    assert not torch.equal(new_conv[0], conv[0])
    assert not torch.equal(new_recurrent[0], recurrent[0])
    assert torch.equal(new_conv[NUM_SLOTS - 1], conv[NUM_SLOTS - 1])
    assert torch.equal(new_recurrent[NUM_SLOTS - 1], recurrent[NUM_SLOTS - 1])


@pytest.mark.parametrize("kind", CB_LAYERS)
def test_new_request_ignores_previous_state_of_its_slot(kind):
    """A request starting at position 0 must not see the state left in its slot by the previous request."""
    layer = _make_cb_layer(kind)
    stale_conv, stale_recurrent = _random_cb_states(layer, kind)
    hidden_states = torch.randn(1, 4, 32)
    position_ids = torch.arange(4)[None, :]
    batch_index = torch.tensor([[1]])

    stale = _run_cb(layer, hidden_states, position_ids, batch_index, stale_conv, stale_recurrent)
    clean = _run_cb(
        layer, hidden_states, position_ids, batch_index, torch.zeros_like(stale_conv), torch.zeros_like(stale_recurrent)
    )

    stale_output, stale_conv_out, stale_recurrent_out = stale
    clean_output, clean_conv_out, clean_recurrent_out = clean
    torch.testing.assert_close(stale_output, clean_output)
    torch.testing.assert_close(stale_conv_out[1], clean_conv_out[1])
    torch.testing.assert_close(stale_recurrent_out[1], clean_recurrent_out[1])
