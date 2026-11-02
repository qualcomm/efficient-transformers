# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""UInt4 MoE kernels shared by quantized MoE model wrappers."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from QEfficient.customop import (
    ctx_gather_3d_generalized,
    ctx_scatter_3d_generalized,
    ctx_scatter_3d_int,
)
from QEfficient.customop.quantization_ops import CastToUInt4Func, DequantizeLinearFunc


@dataclass(frozen=True)
class QuantizedMoEWeights:
    """Stacked UInt4 MoE projections prepared by the owning model wrapper."""

    gate_qweight: torch.Tensor
    gate_scales: torch.Tensor
    gate_qzeros: torch.Tensor
    up_qweight: torch.Tensor
    up_scales: torch.Tensor
    up_qzeros: torch.Tensor
    down_qweight: torch.Tensor
    down_scales: torch.Tensor
    down_qzeros: torch.Tensor
    group_size: int

    @classmethod
    def from_module(cls, module) -> "QuantizedMoEWeights":
        return cls(
            gate_qweight=module.all_gate_qweight,
            gate_scales=module.all_gate_scales,
            gate_qzeros=module.all_gate_qzeros,
            up_qweight=module.all_up_qweight,
            up_scales=module.all_up_scales,
            up_qzeros=module.all_up_qzeros,
            down_qweight=module.all_down_qweight,
            down_scales=module.all_down_scales,
            down_qzeros=module.all_down_qzeros,
            group_size=module.group_size,
        )

    @property
    def num_experts(self) -> int:
        return self.gate_qweight.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.gate_qweight.shape[-1] * 2


def _build_matched_idx_from_cumsum(token_to_expert: torch.Tensor) -> torch.Tensor:
    """Build packed-to-original token indices for cumsum expert dispatch."""
    batch_size, seq_len = token_to_expert.shape
    int32_max = torch.iinfo(torch.int32).max
    invalid_index = torch.tensor(int32_max, dtype=torch.int32, device=token_to_expert.device)
    token_indices = torch.arange(seq_len, dtype=torch.int32, device=token_to_expert.device).expand(batch_size, -1)
    packed_indices = torch.cumsum(token_to_expert.to(torch.int32), dim=1) - 1
    scatter_indices = torch.where(token_to_expert, packed_indices, invalid_index)
    matched_indices = torch.full_like(token_indices, int32_max)
    return ctx_scatter_3d_int(matched_indices.unsqueeze(-1), scatter_indices, token_indices.unsqueeze(-1)).squeeze(-1)


def _dequantize_projection(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int):
    return DequantizeLinearFunc.apply(CastToUInt4Func.apply(qweight), scales, CastToUInt4Func.apply(qzeros), group_size)


def moe_quantized_decode_bmm(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: QuantizedMoEWeights,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Decode quantized MoE experts by dequantizing stacked UInt4 projections."""
    gate_proj = _dequantize_projection(
        weights.gate_qweight, weights.gate_scales, weights.gate_qzeros, weights.group_size
    )
    up_proj = _dequantize_projection(weights.up_qweight, weights.up_scales, weights.up_qzeros, weights.group_size)
    down_proj = _dequantize_projection(
        weights.down_qweight, weights.down_scales, weights.down_qzeros, weights.group_size
    )

    expert_in = x.unsqueeze(0).expand(weights.num_experts, -1, -1)
    gate_out = torch.bmm(expert_in, gate_proj.transpose(1, 2).to(expert_in.dtype))
    up_out = torch.bmm(expert_in, up_proj.transpose(1, 2).to(expert_in.dtype))
    down_out = torch.bmm(act_fn(gate_out) * up_out, down_proj.transpose(1, 2).to(expert_in.dtype))

    routed_out = down_out.transpose(0, 1)
    selected_out = torch.gather(
        routed_out,
        1,
        topk_indices.unsqueeze(-1).expand(-1, topk_indices.shape[1], down_out.shape[-1]),
    )
    return (selected_out * topk_weights.unsqueeze(-1)).sum(dim=1)


def _cumsum_scatter_gather_update_quantized_expert(
    x: torch.Tensor,
    token_to_expert: torch.Tensor,
    gate_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    gate_qzeros: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
    up_qzeros: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    down_qzeros: torch.Tensor,
    routing_weight: torch.Tensor,
    expert_out: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    group_size: int,
    num_packed_chunks: int,
) -> torch.Tensor:
    """Cumsum-scatter-gather-update expert helper for NSP-blocked dispatch.

    Accumulates one local expert's contribution in-place onto ``expert_out``.
    Uses a packed/cumsum layout so the MLP runs only over active rows, then
    scatters the weighted output back to original token positions.

    Shapes:
        x               : [T, H]
        T2Ei            : [num_nsp, T]            (bool)
        W_g, W_u        : [num_nsp, H, I]
        W_d             : [num_nsp, I, H]
        routing_weight  : [num_nsp, T]
        expert_out      : [num_nsp, T, H]         (accumulator, in-out)
    """
    batch_size, seq_len = token_to_expert.shape
    if num_packed_chunks <= 0:
        raise ValueError("num_packed_chunks must be greater than zero")
    if seq_len % num_packed_chunks:
        raise ValueError(
            "Quantized MoE expert parallelism requires the sequence length to be divisible by num_packed_chunks."
        )

    packed_chunk_size = seq_len // num_packed_chunks

    matched_idx = _build_matched_idx_from_cumsum(token_to_expert)
    valid_rows = token_to_expert.to(torch.int32).sum(dim=-1, keepdim=True)
    row_range = torch.arange(packed_chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
    x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)

    for chunk_idx in range(num_packed_chunks):
        packed_start = chunk_idx * packed_chunk_size
        if chunk_idx == num_packed_chunks - 1:
            packed_stop = seq_len
        else:
            packed_stop = packed_start + packed_chunk_size

        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

        x_chunk = ctx_gather_3d_generalized(x_expanded, chunk_matched_idx)

        gate_proj_unpacked = CastToUInt4Func.apply(gate_qweight)
        gate_zeros_unpacked = CastToUInt4Func.apply(gate_qzeros)
        gate_proj_dq = DequantizeLinearFunc.apply(gate_proj_unpacked, gate_scales, gate_zeros_unpacked, group_size)

        up_proj_unpacked = CastToUInt4Func.apply(up_qweight)
        up_zeros_unpacked = CastToUInt4Func.apply(up_qzeros)
        up_proj_dq = DequantizeLinearFunc.apply(up_proj_unpacked, up_scales, up_zeros_unpacked, group_size)

        down_proj_unpacked = CastToUInt4Func.apply(down_qweight)
        down_zeros_unpacked = CastToUInt4Func.apply(down_qzeros)

        down_proj_dq = DequantizeLinearFunc.apply(down_proj_unpacked, down_scales, down_zeros_unpacked, group_size)

        gate_out = torch.bmm(x_chunk, gate_proj_dq.transpose(1, 2).to(x_chunk.dtype))
        up_out = torch.bmm(x_chunk, up_proj_dq.transpose(1, 2).to(x_chunk.dtype))
        hidden = act_fn(gate_out) * up_out
        down_out = torch.bmm(hidden, down_proj_dq.transpose(1, 2).to(x_chunk.dtype))

        rw_chunk = ctx_gather_3d_generalized(routing_weight, chunk_matched_idx)
        old_expert_out = ctx_gather_3d_generalized(expert_out, chunk_matched_idx)
        valid_rows_delta = valid_rows - packed_start
        chunk_valid_rows = torch.where(
            valid_rows_delta < 0,
            torch.zeros_like(valid_rows_delta),
            torch.where(
                valid_rows_delta > packed_chunk_size,
                torch.full_like(valid_rows_delta, packed_chunk_size),
                valid_rows_delta,
            ),
        )
        current_expert_out = (
            torch.where(
                (row_range < chunk_valid_rows).unsqueeze(-1),
                down_out,
                torch.zeros_like(down_out),
            )
            * rw_chunk
        )
        updated_chunk = old_expert_out + current_expert_out
        expert_out = ctx_scatter_3d_generalized(expert_out, chunk_matched_idx, updated_chunk)

    return expert_out


def moe_quantized_expert_parallel(
    x: torch.Tensor,
    routing_weights: torch.Tensor,
    weights: QuantizedMoEWeights,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    num_pipeline_stages: int,
    num_parallelized_experts: int,
    num_packed_chunks: int,
) -> torch.Tensor:
    """Prefill quantized MoE experts using the shared expert-parallel layout."""
    if routing_weights.shape != (x.shape[0], weights.num_experts):
        raise ValueError(
            "expert-parallel routing_weights must have shape "
            f"{(x.shape[0], weights.num_experts)}, got {tuple(routing_weights.shape)}"
        )
    if x.shape[1] != weights.hidden_size:
        raise ValueError(f"expert-parallel hidden size mismatch: input H={x.shape[1]}, weights H={weights.hidden_size}")
    if weights.num_experts != num_pipeline_stages * num_parallelized_experts:
        raise ValueError(
            "num_experts must equal num_pipeline_stages * num_parallelized_experts "
            f"({num_pipeline_stages} * {num_parallelized_experts}), got {weights.num_experts}"
        )

    token_to_expert = (
        routing_weights.transpose(0, 1)
        .contiguous()
        .view(num_pipeline_stages, num_parallelized_experts, x.shape[0])
        .transpose(0, 1)
        .contiguous()
    )
    expert_out = x.new_zeros((num_parallelized_experts, x.shape[0], x.shape[1]))
    routing_weights = token_to_expert.unsqueeze(-1)

    def reshape_projection(projection: torch.Tensor) -> torch.Tensor:
        return (
            projection.view(num_pipeline_stages, num_parallelized_experts, *projection.shape[1:])
            .transpose(0, 1)
            .contiguous()
        )

    # TODO: Move this to QuantizedMoEWeights and precompute in the model wrapper, so we don't have to reshape every forward pass.
    gate_qweight = reshape_projection(weights.gate_qweight)
    gate_scales = reshape_projection(weights.gate_scales)
    gate_qzeros = reshape_projection(weights.gate_qzeros)
    up_qweight = reshape_projection(weights.up_qweight)
    up_scales = reshape_projection(weights.up_scales)
    up_qzeros = reshape_projection(weights.up_qzeros)
    down_qweight = reshape_projection(weights.down_qweight)
    down_scales = reshape_projection(weights.down_scales)
    down_qzeros = reshape_projection(weights.down_qzeros)

    for slot in range(num_pipeline_stages):
        expert_out = _cumsum_scatter_gather_update_quantized_expert(
            x=x,
            token_to_expert=token_to_expert[:, slot] > 0,
            gate_qweight=gate_qweight[:, slot],
            gate_scales=gate_scales[:, slot],
            gate_qzeros=gate_qzeros[:, slot],
            up_qweight=up_qweight[:, slot],
            up_scales=up_scales[:, slot],
            up_qzeros=up_qzeros[:, slot],
            down_qweight=down_qweight[:, slot],
            down_scales=down_scales[:, slot],
            down_qzeros=down_qzeros[:, slot],
            routing_weight=routing_weights[:, slot],
            expert_out=expert_out,
            act_fn=act_fn,
            group_size=weights.group_size,
            num_packed_chunks=num_packed_chunks,
        )
    return expert_out.sum(dim=0)
