# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Per-expert MLP activation profiles.

A :class:`MoEProfile` packages the per-expert projection math so the shared MoE
kernels and flavour functions stay model-agnostic. ``expert_mlp`` operates on the
token axis at ``dim=-2`` and therefore serves all three flavours unchanged:

    expert-parallel : x [num_parallelized_experts, rows, H]
    decode bmm     : x [T*K, 1, H]
    simple loop    : x [T, H]   (single-expert 2-D weights)
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MoEProfile:
    """Captures the per-expert MLP math (activation + optional biases)."""

    expert_mlp: Callable
    has_bias: bool = False
    # "post": weight the expert output after the MLP (default for all current models).
    # "pre": input is pre-scaled by the routing weight and routing is a boolean mask (llama4).
    scale_mode: str = "post"
    # Optional split operations used by decode BMM weight blocking. Arbitrary
    # expert MLP profiles cannot be decomposed unless both are provided.
    expert_mlp_gate_up: Callable | None = None
    expert_mlp_down: Callable | None = None


def silu_glu_mlp(
    x: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    b_g: Optional[torch.Tensor] = None,
    b_u: Optional[torch.Tensor] = None,
    b_d: Optional[torch.Tensor] = None,
    *,
    act_fn: Callable = F.silu,
) -> torch.Tensor:
    """Plain gated-GLU expert: ``(up * act_fn(gate)) @ W_d`` (no biases)."""
    gate = x @ W_g
    up = x @ W_u
    return (up * act_fn(gate)) @ W_d


def silu_glu_gate_up(
    x: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    b_g: torch.Tensor | None = None,
    b_u: torch.Tensor | None = None,
    *,
    act_fn: Callable = F.silu,
) -> torch.Tensor:
    """Compute the intermediate activation for a gated GLU expert."""
    gate = x @ W_g
    up = x @ W_u
    if b_g is not None:
        gate = gate + b_g.unsqueeze(-2)
    if b_u is not None:
        up = up + b_u.unsqueeze(-2)
    return up * act_fn(gate)


def silu_glu_down(
    intermediate: torch.Tensor,
    W_d: torch.Tensor,
    b_d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the down projection to a gated GLU activation."""
    down = intermediate @ W_d
    if b_d is not None:
        down = down + b_d.unsqueeze(-2)
    return down


def gptoss_clamped_glu_mlp(
    x: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    b_g: torch.Tensor,
    b_u: torch.Tensor,
    b_d: torch.Tensor,
    *,
    limit: float,
    alpha: float,
) -> torch.Tensor:
    """GPT-OSS clamped GLU with per-expert biases: ``(up + 1) * gate * sigmoid(gate * alpha)``."""
    gate = (x @ W_g) + b_g.unsqueeze(-2)
    up = (x @ W_u) + b_u.unsqueeze(-2)
    gate = gate.clamp(min=torch.finfo(torch.float16).min, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    intermediate = (up + 1) * glu
    return (intermediate @ W_d) + b_d.unsqueeze(-2)


SILU_GLU_PROFILE = MoEProfile(expert_mlp=silu_glu_mlp, has_bias=False)
