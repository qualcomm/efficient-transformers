# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch


@torch.jit.script
def dflash_chunk_gated_delta_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cumdecay: torch.Tensor,
    g: torch.Tensor,
    decay_mask: torch.Tensor,
    mask_strict: torch.Tensor,
    state: torch.Tensor,
    commit_final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Commit all prefill chunks, or retain the state before the verification chunk."""
    output = torch.zeros_like(value)
    batch, heads, chunk = query.shape[0], query.shape[1], query.shape[3]
    prefix_state = state
    for i in range(query.shape[2]):
        prefix_state = state
        q_i = query[:, :, i : i + 1].reshape(batch, heads, chunk, query.shape[4])
        k_i = key[:, :, i : i + 1].reshape(batch, heads, chunk, key.shape[4])
        v_i = value[:, :, i : i + 1].reshape(batch, heads, chunk, value.shape[4])
        k_decay_i = k_cumdecay[:, :, i : i + 1].reshape(batch, heads, chunk, key.shape[4])
        decay_i = decay_mask[:, :, i : i + 1].reshape(batch, heads, chunk, chunk)
        g_i = g[:, :, i : i + 1].reshape(batch, heads, chunk)
        last_g = g_i[:, :, -1:]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_i).masked_fill(mask_strict, 0)
        v_prime = k_decay_i @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_i.exp().unsqueeze(-1)) @ state
        output[:, :, i : i + 1] = (attn_inter + attn @ v_new).unsqueeze(2)
        state = (
            state * last_g.exp().unsqueeze(-1) + (k_i * (last_g - g_i).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
        )
    return output, torch.where(commit_final.to(torch.bool), state, prefix_state)


def compute_dflash_target_hidden_states(
    target_hidden_list: list[torch.Tensor],
    fc: torch.nn.Module,
    hidden_norm: torch.nn.Module,
) -> torch.Tensor:
    """
    Build the DFlash TLM target hidden state from the hidden states collected at
    ``target_layer_ids``: concatenate them along the last dim, project with ``fc``,
    then normalize with ``hidden_norm``.

    ``Mandatory`` Args:
        :target_hidden_list (list[torch.Tensor]): Hidden states collected at each target layer.
        :fc (torch.nn.Module): Linear projection from concatenated target hidden states to hidden_size.
        :hidden_norm (torch.nn.Module): Norm applied after the fc projection.

    Returns:
        :torch.Tensor: The projected and normalized target hidden state.
    """
    target_hidden = torch.cat(target_hidden_list, dim=-1)
    return hidden_norm(fc(target_hidden))
