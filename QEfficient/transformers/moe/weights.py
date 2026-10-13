# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MoE expert-weight container and builders.

Every MoE model stores its expert weights differently (fused gate_up vs separate
gate/up, transposed or not, interleaved halves, module-list, compressed). These
helpers canonicalize all of them into a single orientation so the shared flavour
kernels can be model-agnostic:

    gate : [E, H, I]
    up   : [E, H, I]
    down : [E, I, H]

(optional per-expert biases: gate_bias/up_bias [E, I], down_bias [E, H]).

Expert-parallel export packs the same data as:

    gate : [num_parallelized_experts, num_pipeline_stages, H, I]
    up   : [num_parallelized_experts, num_pipeline_stages, H, I]
    down : [num_parallelized_experts, num_pipeline_stages, I, H]
"""

from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn


def _as_frozen_parameter(tensor: Optional[torch.Tensor]) -> Optional[nn.Parameter]:
    if tensor is None:
        return None
    if isinstance(tensor, nn.Parameter):
        tensor.requires_grad_(False)
        return tensor
    return nn.Parameter(tensor.detach(), requires_grad=False)


class MoEWeights(nn.Module):
    """Expert weights in canonical or expert-parallel-packed orientation."""

    def __init__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        gate_bias: Optional[torch.Tensor] = None,
        up_bias: Optional[torch.Tensor] = None,
        down_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gate = _as_frozen_parameter(gate)
        self.up = _as_frozen_parameter(up)
        self.down = _as_frozen_parameter(down)
        self.gate_bias = _as_frozen_parameter(gate_bias)
        self.up_bias = _as_frozen_parameter(up_bias)
        self.down_bias = _as_frozen_parameter(down_bias)

    @property
    def num_experts(self) -> int:
        if self.gate.ndim == 4:
            return self.gate.shape[0] * self.gate.shape[1]
        return self.gate.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.gate.shape[-2]

    @property
    def intermediate_size(self) -> int:
        return self.gate.shape[-1]

    @property
    def has_bias(self) -> bool:
        return self.gate_bias is not None


def _maybe_clone(t: Optional[torch.Tensor], clone: bool) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.detach().clone() if clone else t.detach()


def _validate_matching_optional_shape(
    *,
    name: str,
    tensor: Optional[torch.Tensor],
    expected: tuple[int, ...],
) -> None:
    if tensor is not None and tuple(tensor.shape) != expected:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match expected shape {expected}")


def validate_canonical_moe_weights(weights: MoEWeights) -> None:
    """Validate canonical MoE weight layout.

    Canonical layout is gate/up ``[E,H,I]`` and down ``[E,I,H]``. Biases, when
    present, are gate/up ``[E,I]`` and down ``[E,H]``.
    """
    if weights.gate.ndim != 3:
        raise ValueError(f"canonical MoE gate weights must be 3-D, got shape {tuple(weights.gate.shape)}")
    expected_gate_shape = tuple(weights.gate.shape)
    expected_down_shape = (weights.gate.shape[0], weights.gate.shape[2], weights.gate.shape[1])
    if tuple(weights.up.shape) != expected_gate_shape:
        raise ValueError(
            f"canonical MoE up weights must have shape {expected_gate_shape}, got {tuple(weights.up.shape)}"
        )
    if tuple(weights.down.shape) != expected_down_shape:
        raise ValueError(
            f"canonical MoE down weights must have shape {expected_down_shape}, got {tuple(weights.down.shape)}"
        )
    _validate_matching_optional_shape(
        name="canonical MoE gate bias",
        tensor=weights.gate_bias,
        expected=(weights.gate.shape[0], weights.gate.shape[2]),
    )
    _validate_matching_optional_shape(
        name="canonical MoE up bias",
        tensor=weights.up_bias,
        expected=(weights.gate.shape[0], weights.gate.shape[2]),
    )
    _validate_matching_optional_shape(
        name="canonical MoE down bias",
        tensor=weights.down_bias,
        expected=(weights.gate.shape[0], weights.gate.shape[1]),
    )


def validate_expert_parallel_moe_weights(
    weights: MoEWeights,
    *,
    num_parallelized_experts: Optional[int] = None,
    num_pipeline_stages: Optional[int] = None,
) -> None:
    """Validate expert-parallel packed MoE weight layout."""
    if weights.gate.ndim != 4:
        raise ValueError(f"expert-parallel MoE gate weights must be 4-D, got shape {tuple(weights.gate.shape)}")
    expected_gate_shape = tuple(weights.gate.shape)
    expected_down_shape = (weights.gate.shape[0], weights.gate.shape[1], weights.gate.shape[3], weights.gate.shape[2])
    if num_parallelized_experts is not None and weights.gate.shape[0] != num_parallelized_experts:
        raise ValueError(
            "expert-parallel weights were packed for "
            f"num_parallelized_experts={weights.gate.shape[0]}, "
            f"but expected {num_parallelized_experts}"
        )
    if num_pipeline_stages is not None and weights.gate.shape[1] != num_pipeline_stages:
        raise ValueError(
            f"expert-parallel weights were packed for num_pipeline_stages={weights.gate.shape[1]}, "
            f"but expected {num_pipeline_stages}"
        )
    if tuple(weights.up.shape) != expected_gate_shape:
        raise ValueError(
            f"expert-parallel MoE up weights must have shape {expected_gate_shape}, got {tuple(weights.up.shape)}"
        )
    if tuple(weights.down.shape) != expected_down_shape:
        raise ValueError(
            f"expert-parallel MoE down weights must have shape {expected_down_shape}, got {tuple(weights.down.shape)}"
        )
    _validate_matching_optional_shape(
        name="expert-parallel MoE gate bias",
        tensor=weights.gate_bias,
        expected=(weights.gate.shape[0], weights.gate.shape[1], weights.gate.shape[3]),
    )
    _validate_matching_optional_shape(
        name="expert-parallel MoE up bias",
        tensor=weights.up_bias,
        expected=(weights.gate.shape[0], weights.gate.shape[1], weights.gate.shape[3]),
    )
    _validate_matching_optional_shape(
        name="expert-parallel MoE down bias",
        tensor=weights.down_bias,
        expected=(weights.gate.shape[0], weights.gate.shape[1], weights.gate.shape[2]),
    )


def _pack_expert_parallel_tensor(
    tensor: Optional[torch.Tensor],
    *,
    num_pipeline_stages: int,
    num_parallelized_experts: int,
) -> Optional[nn.Parameter]:
    if tensor is None:
        return None
    packed = (
        tensor.view(num_pipeline_stages, num_parallelized_experts, *tensor.shape[1:])
        .transpose(0, 1)
        .contiguous()
        .detach()
        .clone()
    )
    return nn.Parameter(packed, requires_grad=False)


def _unpack_expert_parallel_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.transpose(0, 1).contiguous().view(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def pack_moe_weights_for_expert_parallel(
    weights: MoEWeights,
    *,
    num_pipeline_stages: int,
    num_parallelized_experts: int,
) -> MoEWeights:
    """Return weights packed for expert-parallel execution.

    The layout matches the Qwen3-VL-MoE expert-parallel path:
    ``view(num_pipeline_stages, num_parallelized_experts, ...)`` followed by a
    transpose to ``[num_parallelized_experts, num_pipeline_stages, ...]``.
    """
    if num_pipeline_stages <= 0:
        raise ValueError("num_pipeline_stages must be greater than zero")
    if num_parallelized_experts <= 0:
        raise ValueError("num_parallelized_experts must be greater than zero")
    if weights.gate.ndim == 4:
        validate_expert_parallel_moe_weights(weights)
        if weights.gate.shape[:2] == (num_parallelized_experts, num_pipeline_stages):
            return weights
        weights = unpack_moe_weights_from_expert_parallel(weights)
    validate_canonical_moe_weights(weights)
    num_experts = weights.num_experts
    if num_experts != num_pipeline_stages * num_parallelized_experts:
        raise ValueError(
            "num_experts must equal num_pipeline_stages * num_parallelized_experts "
            f"({num_pipeline_stages} * {num_parallelized_experts}), got {num_experts}"
        )
    return MoEWeights(
        gate=_pack_expert_parallel_tensor(
            weights.gate,
            num_pipeline_stages=num_pipeline_stages,
            num_parallelized_experts=num_parallelized_experts,
        ),
        up=_pack_expert_parallel_tensor(
            weights.up,
            num_pipeline_stages=num_pipeline_stages,
            num_parallelized_experts=num_parallelized_experts,
        ),
        down=_pack_expert_parallel_tensor(
            weights.down,
            num_pipeline_stages=num_pipeline_stages,
            num_parallelized_experts=num_parallelized_experts,
        ),
        gate_bias=_pack_expert_parallel_tensor(
            weights.gate_bias,
            num_pipeline_stages=num_pipeline_stages,
            num_parallelized_experts=num_parallelized_experts,
        ),
        up_bias=_pack_expert_parallel_tensor(
            weights.up_bias,
            num_pipeline_stages=num_pipeline_stages,
            num_parallelized_experts=num_parallelized_experts,
        ),
        down_bias=_pack_expert_parallel_tensor(
            weights.down_bias,
            num_pipeline_stages=num_pipeline_stages,
            num_parallelized_experts=num_parallelized_experts,
        ),
    )


def unpack_moe_weights_from_expert_parallel(weights: MoEWeights) -> MoEWeights:
    """Return weights in canonical layout, unpacking expert-parallel layout if needed."""
    if weights.gate.ndim == 3:
        validate_canonical_moe_weights(weights)
        return weights
    validate_expert_parallel_moe_weights(weights)
    return MoEWeights(
        gate=_unpack_expert_parallel_tensor(weights.gate),
        up=_unpack_expert_parallel_tensor(weights.up),
        down=_unpack_expert_parallel_tensor(weights.down),
        gate_bias=_unpack_expert_parallel_tensor(weights.gate_bias),
        up_bias=_unpack_expert_parallel_tensor(weights.up_bias),
        down_bias=_unpack_expert_parallel_tensor(weights.down_bias),
    )


def build_canonical_expert_weights(
    *,
    down: torch.Tensor,
    gate_up: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    up: Optional[torch.Tensor] = None,
    gate_up_bias: Optional[torch.Tensor] = None,
    down_bias: Optional[torch.Tensor] = None,
    fused: bool = True,
    fused_split_dim: int = 1,
    interleaved: bool = False,
    transpose_gate_up: bool = True,
    transpose_down: bool = True,
    clone: bool = False,
) -> MoEWeights:
    """Canonicalize tensor-level expert weights into :class:`MoEWeights`.

    Args:
        down: source down-projection weight (``[E,H,I]`` if ``transpose_down`` else ``[E,I,H]``).
        gate_up: fused gate+up source (used when ``fused=True``).
        gate, up: pre-split sources (used when ``fused=False``).
        gate_up_bias: fused bias source split the same way as ``gate_up``.
        down_bias: down-projection bias ``[E,H]``.
        fused: whether ``gate_up`` carries both halves and must be split.
        fused_split_dim: dim along which the fused halves are concatenated
            (1 for HF ``[E,2I,H]``; 2 for HF ``[E,H,2I]`` as in GPT-OSS).
        interleaved: GPT-OSS stores ``[gate0, up0, gate1, up1, ...]`` along the
            split dim; when True the halves are taken as ``[..., ::2]`` / ``[..., 1::2]``.
        transpose_gate_up: transpose the last two dims of gate/up to reach ``[E,H,I]``.
        transpose_down: transpose the last two dims of down to reach ``[E,I,H]``.
        clone: detach+clone the slices (set False to alias existing parameters).
    """
    if fused:
        if gate_up is None:
            raise ValueError("build_canonical_expert_weights(fused=True) requires gate_up")
        half = gate_up.shape[fused_split_dim] // 2
        if interleaved:
            even = [slice(None)] * gate_up.ndim
            odd = [slice(None)] * gate_up.ndim
            even[fused_split_dim] = slice(0, None, 2)
            odd[fused_split_dim] = slice(1, None, 2)
            gate = gate_up[tuple(even)]
            up = gate_up[tuple(odd)]
        else:
            gate = gate_up.narrow(fused_split_dim, 0, half)
            up = gate_up.narrow(fused_split_dim, half, half)
    else:
        if gate is None or up is None:
            raise ValueError("build_canonical_expert_weights(fused=False) requires gate and up")

    gate_bias = up_bias = None
    if gate_up_bias is not None:
        # bias fused along the last dim ([E,2I]); same interleave rule as the weights.
        half_b = gate_up_bias.shape[-1] // 2
        if interleaved:
            gate_bias = gate_up_bias[..., ::2]
            up_bias = gate_up_bias[..., 1::2]
        else:
            gate_bias = gate_up_bias[..., :half_b]
            up_bias = gate_up_bias[..., half_b:]

    if transpose_gate_up:
        gate = gate.transpose(-1, -2)
        up = up.transpose(-1, -2)
    if transpose_down:
        down = down.transpose(-1, -2)

    return MoEWeights(
        gate=_maybe_clone(gate.contiguous() if clone else gate, clone),
        up=_maybe_clone(up.contiguous() if clone else up, clone),
        down=_maybe_clone(down.contiguous() if clone else down, clone),
        gate_bias=_maybe_clone(gate_bias.contiguous() if (clone and gate_bias is not None) else gate_bias, clone),
        up_bias=_maybe_clone(up_bias.contiguous() if (clone and up_bias is not None) else up_bias, clone),
        down_bias=_maybe_clone(down_bias, clone),
    )


def stack_expert_linears(
    experts: Iterable[Any],
    weight_getter: Callable[[Any], torch.Tensor],
    *,
    transpose: bool = True,
) -> torch.Tensor:
    """Stack per-expert weights from a module list into a single ``[E, *, *]`` tensor.

    ``weight_getter`` returns one expert's 2-D weight (e.g. ``lambda e: e.gate_proj.weight``);
    with ``transpose=True`` a ``nn.Linear`` weight ``[out, in]`` becomes ``[in, out]``.
    """
    slices = []
    for expert in experts:
        w = weight_getter(expert).detach()
        if transpose:
            w = w.t()
        slices.append(w.unsqueeze(0))
    return torch.cat(slices, dim=0).contiguous()


def as_parameters(weights: MoEWeights) -> MoEWeights:
    """Backward-compatible no-op; :class:`MoEWeights` already owns frozen parameters."""
    return weights


def delete_module_attrs(module: nn.Module, *names: str) -> None:
    """Delete original weight attributes after moving them into ``module.moe_weights``."""
    for name in names:
        if hasattr(module, name):
            delattr(module, name)
