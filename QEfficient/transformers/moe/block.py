# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Shared MoE block orchestration.

``QEffMoEBlockMixin`` provides the ``forward`` that routes tokens, dispatches to the
configured flavour, and adds shared experts. Per-model blocks subclass it and
override only the variation points:

* ``routing_input(x, hidden_states)`` -> input accepted by the model gate (default: ``x``).
* ``route(x)``               -> (routing, router_logits); routing is dense ``[T,E]``
                                 or ``(topk_indices, topk_weights)``.
* ``transform_weights()``    -> transform original weights into canonical
                                 :class:`MoEWeights` during the MoE weights transform.
* ``moe_profile``            -> :class:`MoEProfile` (default: plain SiLU-GLU).
* ``apply_shared_experts``   -> add a shared-expert term (default: no-op).

The flavour is assigned at export time by ``OptimizedMoETransform`` via
``self._moe_flavour`` and the ``expert_parallel_*`` attributes.
"""

from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import torch

from QEfficient.transformers.moe.flavours import (
    MoEFlavour,
    moe_decode_bmm,
    moe_expert_parallel,
    moe_simple_loop,
    resolve_routing,
)
from QEfficient.transformers.moe.profiles import SILU_GLU_PROFILE, MoEProfile
from QEfficient.transformers.moe.qflavours import (
    QuantizedMoEWeights,
    moe_quantized_decode_bmm,
    moe_quantized_expert_parallel,
)
from QEfficient.transformers.moe.weights import MoEWeights


class QEffMoEBlockMixin(metaclass=ABCMeta):
    # Default profile; models with non-standard activations override this.
    moe_profile: MoEProfile = SILU_GLU_PROFILE
    # Set by OptimizedMoETransform at export time; decode is the safe default.
    _moe_flavour: MoEFlavour = MoEFlavour.DECODE_BMM
    # Whether forward returns (out, router_logits) to match the HF MoE convention.
    _moe_return_router_logits: bool = False
    # Expert-parallel knobs, set by OptimizedMoETransform when flavour is EXPERT_PARALLEL.
    num_devices: int = 1
    cores_per_expert: int = 1
    total_avl_cores: Optional[int] = None
    num_pipeline_stages: Optional[int] = None
    num_parallelized_experts: Optional[int] = None
    experts_per_soc: Optional[int] = None
    tree_reduce: bool = False
    expert_parallel_num_packed_chunks: int = 1
    expert_intermediate_block_size: Optional[int] = None
    # Set by OptimizedMoEWeightsTransform after model-local canonicalization.
    weights_transformed: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        supported_moe_flavours = cls.__dict__.get("supported_moe_flavours")
        if supported_moe_flavours is None:
            inherited_supported_moe_flavours = next(
                (
                    base.__dict__["supported_moe_flavours"]
                    for base in cls.__mro__[1:]
                    if "supported_moe_flavours" in base.__dict__
                ),
                None,
            )
            is_fx_attr_proxy = cls.__module__ == "abc" and any(
                base.__name__ == cls.__name__ for base in cls.__mro__[1:]
            )
            if inherited_supported_moe_flavours is None or not is_fx_attr_proxy:
                raise TypeError(f"{cls.__name__} must explicitly define supported_moe_flavours")
            supported_moe_flavours = inherited_supported_moe_flavours
        supported_moe_flavours = tuple(MoEFlavour(flavour) for flavour in supported_moe_flavours)
        if not supported_moe_flavours:
            raise TypeError(f"{cls.__name__}.supported_moe_flavours must not be empty")
        cls.supported_moe_flavours = supported_moe_flavours

    def __qeff_init__(self) -> None:
        if "weights_transformed" not in self.__dict__:
            self.weights_transformed = False

    # ---- variation points (override per model) --------------------------------
    def route(self, x: torch.Tensor):
        raise NotImplementedError

    def routing_input(self, x: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return x

    @abstractmethod
    def transform_weights(self) -> MoEWeights:
        raise NotImplementedError

    def apply_shared_experts(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return out

    def get_supported_moe_flavours(self) -> Tuple[MoEFlavour, ...]:
        if not hasattr(self, "supported_moe_flavours"):
            raise TypeError(f"{type(self).__name__} must explicitly define supported_moe_flavours")
        return tuple(self.supported_moe_flavours)

    # ---- orchestration (shared) ----------------------------------------------
    def execute_moe_flavour(self, x: torch.Tensor, routing) -> torch.Tensor:
        if not getattr(self, "weights_transformed", False):
            raise RuntimeError(
                f"{type(self).__name__} weights are not transformed; run OptimizedMoEWeightsTransform before forward"
            )

        flavour = getattr(self, "_moe_flavour", MoEFlavour.DECODE_BMM)
        if not isinstance(flavour, MoEFlavour):
            flavour = MoEFlavour(flavour)

        quantized_experts = getattr(self, "_qeff_quantized_experts", False)
        if quantized_experts:
            quantized_weights = QuantizedMoEWeights.from_module(self)
            num_experts = quantized_weights.num_experts
        else:
            weights = self.moe_weights
            num_experts = weights.num_experts
            profile = self.moe_profile
            if callable(profile):
                profile = profile()

        if flavour is MoEFlavour.DECODE_BMM:
            _, topk = resolve_routing(routing, num_experts)
            if topk is None:
                raise ValueError("decode_bmm flavour requires route() to return (topk_indices, topk_weights)")
            topk_indices, topk_weights = topk
            if quantized_experts:
                return moe_quantized_decode_bmm(x, topk_indices, topk_weights, quantized_weights, self.act_fn)
            return moe_decode_bmm(x, topk_indices, topk_weights, weights, profile, top_k=topk_indices.shape[1])

        dense, _ = resolve_routing(routing, num_experts)
        if flavour is MoEFlavour.EXPERT_PARALLEL:
            num_pipeline_stages = getattr(self, "num_pipeline_stages", None) or (
                1 if quantized_experts else weights.gate.shape[1]
            )
            num_parallelized_experts = getattr(self, "num_parallelized_experts", None) or (
                num_experts if quantized_experts else weights.gate.shape[0]
            )
            num_packed_chunks = self.expert_parallel_num_packed_chunks
            if quantized_experts:
                return moe_quantized_expert_parallel(
                    x,
                    dense,
                    quantized_weights,
                    self.act_fn,
                    num_pipeline_stages=num_pipeline_stages,
                    num_parallelized_experts=num_parallelized_experts,
                    num_packed_chunks=num_packed_chunks,
                )
            return moe_expert_parallel(
                x,
                dense,
                weights,
                profile,
                num_pipeline_stages=num_pipeline_stages,
                num_parallelized_experts=num_parallelized_experts,
                num_devices=getattr(self, "num_devices", 1),
                experts_per_soc=getattr(self, "experts_per_soc", None),
                tree_reduce=getattr(self, "tree_reduce", False),
                num_packed_chunks=num_packed_chunks,
                expert_intermediate_block_size=getattr(self, "expert_intermediate_block_size", None),
            )
        if quantized_experts:
            raise NotImplementedError(f"moe flavour {flavour.value!r} is not supported for quantized experts")
        return moe_simple_loop(x, dense, weights, profile, prescale=profile.scale_mode == "pre")

    def moe_dispatch(self, x: torch.Tensor, routing) -> torch.Tensor:
        return self.execute_moe_flavour(x, routing)

    def forward(self, hidden_states: torch.Tensor):
        B, S, H = hidden_states.shape
        x = hidden_states.view(B * S, H)
        routing, router_logits = self.route(self.routing_input(x, hidden_states))
        out = self.execute_moe_flavour(x, routing)
        out = self.apply_shared_experts(out, x)
        out = out.view(B, S, H)
        if self._moe_return_router_logits:
            return out, router_logits
        return out
