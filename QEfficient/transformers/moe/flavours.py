# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""The three MoE forward flavours and the flavour selector.

All flavours consume canonical :class:`MoEWeights` plus a :class:`MoEProfile` and
return the routed expert output ``[T, H]``:

* ``moe_expert_parallel`` / ``moe_simple_loop`` consume a dense ``routing_weights [T, E]``.
* ``moe_decode_bmm`` consumes ``(topk_indices [T, K], topk_weights [T, K])``.
"""

from enum import Enum
from typing import Callable, Iterable, Optional, Tuple, Union

import torch

from QEfficient.customop import (
    ctx_gather_3d_generalized,
    ctx_scatter_3d_generalized,
    ctx_scatter_3d_int,
)
from QEfficient.transformers.moe.profiles import MoEProfile
from QEfficient.transformers.moe.weights import MoEWeights, validate_expert_parallel_moe_weights
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


class MoEFlavour(str, Enum):
    EXPERT_PARALLEL = "expert_parallel"
    # Backward-compatible alias for one transition period.
    EXPERT_BLOCKED = "expert_parallel"
    SIMPLE_LOOP = "simple_loop"
    DECODE_BMM = "decode_bmm"

    @classmethod
    def _missing_(cls, value):
        if value == "expert_blocked":
            return cls.EXPERT_PARALLEL
        return None


def densify_topk(topk_indices: torch.Tensor, topk_weights: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Scatter ``(topk_indices, topk_weights)`` into a dense ``[T, E]`` routing matrix."""
    T = topk_indices.shape[0]
    routing_weights = topk_weights.new_zeros((T, num_experts))
    routing_weights.scatter_(1, topk_indices, topk_weights)
    return routing_weights


def supports_moe_flavour(module, flavour: Union[MoEFlavour, str]) -> bool:
    """Return whether a module advertises a shared MoE flavour."""
    flavour = MoEFlavour(flavour)
    get_supported_moe_flavours = getattr(module, "get_supported_moe_flavours", None)
    if callable(get_supported_moe_flavours):
        supported_flavours = get_supported_moe_flavours()
    else:
        supported_flavours = getattr(module, "supported_moe_flavours", ())
    try:
        return flavour in tuple(MoEFlavour(supported_flavour) for supported_flavour in supported_flavours)
    except ValueError:
        return False


def reduce_nsp_tree(values: torch.Tensor, num_lanes: int) -> torch.Tensor:
    """Reduce lane outputs with the tree pattern used by Qwen3-VL-MoE expert parallelism."""
    logger.warning(f"Reducing {values.shape} across {num_lanes} lanes using tree reduction")
    current = values
    width = num_lanes
    while width > 1:
        pair_count = width // 2
        reduced = current[0 : 2 * pair_count : 2] + current[1 : 2 * pair_count : 2]
        if width % 2 == 1:
            reduced = torch.cat((reduced, current[width - 1 : width]), dim=0)
        current = reduced
        width = pair_count + (width % 2)
    return current[0]


def build_matched_idx_from_cumsum(T2Ei: torch.Tensor) -> torch.Tensor:
    """Build packed->original token index from a per-token expert-match mask."""
    batch_size, seq_len = T2Ei.shape
    int32_max = torch.iinfo(torch.int32).max
    token_idx = torch.arange(seq_len, dtype=torch.int32, device=T2Ei.device).unsqueeze(0).expand(batch_size, -1)
    valid_prefix = torch.cumsum(T2Ei.to(torch.int32), dim=1, dtype=torch.int32)
    valid_dest = valid_prefix - 1
    int32_max_scalar = torch.full_like(valid_dest, int32_max)
    scatter_pos = torch.where(T2Ei, valid_dest, int32_max_scalar)
    # NOTE: expand_as(...) instead of torch.full_like(...) is the compiler-preferred
    # workaround for ConstantOfShape(INT32_MAX); both produce identical traced Ctx ops.
    matched_idx = int32_max_scalar.expand_as(token_idx)
    matched_idx = ctx_scatter_3d_int(
        matched_idx.unsqueeze(-1),
        scatter_pos,
        token_idx.unsqueeze(-1),
    ).squeeze(-1)
    return matched_idx


def cumsum_scatter_gather_update_expert_blocked(
    x: torch.Tensor,
    T2Ei: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    routing_weight: torch.Tensor,
    expert_out: torch.Tensor,
    expert_mlp: Callable,
    *,
    b_g: Optional[torch.Tensor] = None,
    b_u: Optional[torch.Tensor] = None,
    b_d: Optional[torch.Tensor] = None,
    num_packed_chunks: int = 1,
) -> torch.Tensor:
    """Run one local expert slot over statically traced packed chunks."""
    batch_size, seq_len = T2Ei.shape
    num_packed_chunks = max(1, int(num_packed_chunks))
    assert seq_len % num_packed_chunks == 0, (
        f"seq_len={seq_len} must be divisible by num_packed_chunks={num_packed_chunks}"
    )
    packed_chunk_size = seq_len // num_packed_chunks
    matched_idx = build_matched_idx_from_cumsum(T2Ei)
    valid_rows = T2Ei.to(torch.int32).sum(dim=-1, keepdim=True)
    x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)
    for chunk_idx in range(num_packed_chunks):
        packed_start = chunk_idx * packed_chunk_size
        if chunk_idx == num_packed_chunks - 1:
            packed_stop = seq_len
        else:
            packed_stop = packed_start + packed_chunk_size
        chunk_rows = packed_stop - packed_start
        row_range = torch.arange(chunk_rows, dtype=torch.int32, device=x.device).unsqueeze(0)
        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

        x_chunk = ctx_gather_3d_generalized(x_expanded, chunk_matched_idx)
        down_chunk = expert_mlp(x_chunk, W_g, W_u, W_d, b_g, b_u, b_d)

        rw_chunk = ctx_gather_3d_generalized(routing_weight, chunk_matched_idx)
        down_chunk = down_chunk * rw_chunk
        expert_out_chunk = ctx_gather_3d_generalized(expert_out, chunk_matched_idx)
        updated_chunk = expert_out_chunk + down_chunk

        chunk_valid_rows = torch.clamp(
            valid_rows - packed_start,
            min=torch.zeros_like(valid_rows),
            max=torch.full_like(valid_rows, chunk_rows),
        )
        updated_chunk = torch.where(
            (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
        )
        expert_out = ctx_scatter_3d_generalized(expert_out, chunk_matched_idx, updated_chunk)

    return expert_out


def moe_expert_parallel(
    x: torch.Tensor,
    routing_weights: torch.Tensor,
    weights: MoEWeights,
    profile: MoEProfile,
    *,
    num_pipeline_stages: int,
    num_parallelized_experts: int,
    num_devices: int = 1,
    experts_per_soc: Optional[int] = None,
    tree_reduce: bool = False,
    num_packed_chunks: int = 1,
) -> torch.Tensor:
    """Prefill expert-parallel flavour with branch-style expert reshaping."""
    T, H = x.shape
    validate_expert_parallel_moe_weights(
        weights,
        num_parallelized_experts=num_parallelized_experts,
        num_pipeline_stages=num_pipeline_stages,
    )
    num_experts = weights.num_experts
    if routing_weights.shape != (T, num_experts):
        raise ValueError(
            f"expert-parallel routing_weights must have shape {(T, num_experts)}, got {tuple(routing_weights.shape)}"
        )
    if weights.hidden_size != H:
        raise ValueError(f"expert-parallel hidden size mismatch: input H={H}, weights H={weights.hidden_size}")
    if num_devices <= 0:
        raise ValueError(f"num_devices ({num_devices}) must be greater than 0")

    rw = (
        routing_weights.transpose(0, 1)
        .contiguous()
        .view(num_pipeline_stages, num_parallelized_experts, T)
        .transpose(0, 1)
        .contiguous()
    )
    W_g = weights.gate
    W_u = weights.up
    W_d = weights.down
    b_g = weights.gate_bias
    b_u = weights.up_bias
    b_d = weights.down_bias

    expert_out = x.new_zeros((num_parallelized_experts, T, H))
    routing_weights_unsqueezed = rw.unsqueeze(-1)
    for slot in range(num_pipeline_stages):
        T2Ei = rw[:, slot, :] > 0
        expert_out = cumsum_scatter_gather_update_expert_blocked(
            x=x,
            T2Ei=T2Ei,
            W_g=W_g[:, slot],
            W_u=W_u[:, slot],
            W_d=W_d[:, slot],
            routing_weight=routing_weights_unsqueezed[:, slot],
            expert_out=expert_out,
            expert_mlp=profile.expert_mlp,
            b_g=b_g[:, slot] if b_g is not None else None,
            b_u=b_u[:, slot] if b_u is not None else None,
            b_d=b_d[:, slot] if b_d is not None else None,
            num_packed_chunks=num_packed_chunks,
        )

    if experts_per_soc is not None:
        if num_parallelized_experts % num_devices != 0:
            raise ValueError(
                f"num_parallelized_experts ({num_parallelized_experts}) must be divisible by num_devices "
                f"({num_devices})"
            )
        parallelized_experts_per_soc = num_parallelized_experts // num_devices
        expert_out = expert_out.view(num_devices, parallelized_experts_per_soc, T, H).sum(dim=1)
        if tree_reduce:
            return reduce_nsp_tree(expert_out, num_devices)
        return expert_out.sum(dim=0)
    return expert_out.sum(dim=0)


# Backward-compatible helper name for one transition period.
moe_expert_blocked = moe_expert_parallel


def moe_simple_loop(
    x: torch.Tensor,
    routing_weights: torch.Tensor,
    weights: MoEWeights,
    profile: MoEProfile,
    *,
    prescale: bool = False,
) -> torch.Tensor:
    """Prefill simple-loop flavour: one masked pass per expert."""
    T, H = x.shape
    expert_out = x.new_zeros((T, H))
    for e in range(weights.num_experts):
        routing_weight = routing_weights[:, e].unsqueeze(-1)
        expert_input = x * routing_weight if prescale else x
        down = profile.expert_mlp(
            expert_input,
            weights.gate[e],
            weights.up[e],
            weights.down[e],
            weights.gate_bias[e] if weights.gate_bias is not None else None,
            weights.up_bias[e] if weights.up_bias is not None else None,
            weights.down_bias[e] if weights.down_bias is not None else None,
        )
        if prescale or profile.scale_mode == "pre":
            expert_out = expert_out + torch.where(routing_weight > 0, down, torch.zeros_like(down))
        else:
            expert_out = expert_out + down * routing_weight
    return expert_out


def moe_decode_bmm(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MoEWeights,
    profile: MoEProfile,
    *,
    top_k: int,
) -> torch.Tensor:
    """Decode flavour: gather selected expert weights by index and batch-matmul."""
    T, H = x.shape
    idx = topk_indices.reshape(-1)
    gate_proj = weights.gate[idx]
    up_proj = weights.up[idx]
    down_proj = weights.down[idx]
    b_g = weights.gate_bias[idx] if weights.gate_bias is not None else None
    b_u = weights.up_bias[idx] if weights.up_bias is not None else None
    b_d = weights.down_bias[idx] if weights.down_bias is not None else None

    expert_in = x.unsqueeze(1).expand(-1, top_k, -1).contiguous().view(-1, 1, H)
    down = profile.expert_mlp(expert_in, gate_proj, up_proj, down_proj, b_g, b_u, b_d)
    experts_out = down.view(T, top_k, H)
    experts_out = experts_out * topk_weights.unsqueeze(-1)
    return experts_out.sum(dim=1)


def select_moe_flavour(
    *,
    supported_flavours: Iterable[MoEFlavour],
    is_prefill: bool,
    requested_flavour: Optional[Union[MoEFlavour, str]] = None,
) -> MoEFlavour:
    """Resolve the MoE forward flavour for a module.

    Explicit overrides are parsed as :class:`MoEFlavour`. When no override is
    requested, prefill prefers expert-parallel where the module supports it, and
    decode prefers gather+bmm where available.
    """
    supported_flavours = tuple(MoEFlavour(flavour) for flavour in supported_flavours)
    if requested_flavour is None:
        preferred = MoEFlavour.EXPERT_PARALLEL if is_prefill else MoEFlavour.DECODE_BMM
        if preferred in supported_flavours:
            return preferred
        return MoEFlavour.SIMPLE_LOOP

    if isinstance(requested_flavour, MoEFlavour):
        flavour = requested_flavour
    else:
        try:
            flavour = MoEFlavour(requested_flavour)
        except ValueError as exc:
            raise ValueError(f"Unsupported qaic_config['moe_config']['flavour']={requested_flavour!r}") from exc

    if flavour not in supported_flavours:
        supported = ", ".join(sorted({supported_flavour.value for supported_flavour in supported_flavours}))
        raise NotImplementedError(
            f"moe flavour {flavour.value!r} is not implemented by this module; supported flavours: {supported}"
        )
    return flavour


def resolve_routing(
    routing,
    num_experts: int,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Normalize a router output into (dense_routing_weights, optional topk pair).

    Accepts either a dense ``[T, E]`` tensor or a ``(topk_indices, topk_weights)``
    tuple and returns both representations (densifying the tuple when needed).
    """
    if torch.is_tensor(routing):
        return routing, None
    topk_indices, topk_weights = routing
    return densify_topk(topk_indices, topk_weights, num_experts), (topk_indices, topk_weights)
