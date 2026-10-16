# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest
import torch

from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import QEffQwen3_5GatedDeltaNet
from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeGatedDeltaNet

MAX_ABS_DEV_RECURSIVE_VS_ORIGINAL = 1e-4


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
    for solver in ("recursive_sns", "scaled_newton_schulz", "original", "factorized", "horner"):
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
