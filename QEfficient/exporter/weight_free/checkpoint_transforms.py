# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Checkpoint preparation transforms for weight-free ONNX export.

Each transform declares a stable TRANSFORM_ID and plans grouped tasks. The
pipeline schedules those tasks under one RAM budget, while each task loads its
raw tensors, applies all compatible in-memory stages, and writes one final
output shard. Only the pipeline owns checkpoint execution and finalization.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open

from QEfficient.base.checkpoint_transforms import (
    BaseCheckpointTransform,
    CheckpointPlanningContext,
    CheckpointStage,
    CheckpointTask,
    CheckpointTaskPlan,
    TaskParams,
    TensorRef,
)
from QEfficient.transformers.quantizers.quantizer_utils import convert_moe_packed_tensors
from QEfficient.utils.checkpoint_utils import safetensors_dtype_to_torch

# ---------------------------------------------------------------------------
# Canonical key mapping helpers
# ---------------------------------------------------------------------------


def build_canonical_maps(
    weight_map: Dict[str, str],
    key_remap: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build canonical index and key translation from raw weight_map + KEY_REMAP.

    Parameters
    ----------
    weight_map
        Raw ``{actual_key: shard_file}`` from model.safetensors.index.json.
    key_remap
        ``{regex_pattern: canonical_suffix_replacement}`` declared by the
        transform class.  Empty dict means no remapping (Mixtral-style
        checkpoints already use canonical names).

    Returns
    -------
    canonical_index
        ``{canonical_key: shard_file}`` — WHERE to find each tensor.
    key_translation
        ``{canonical_key: actual_key_in_shard}`` — WHAT to ask the shard for.
        Only contains entries where the key was remapped; absent means
        canonical_key == actual_key.
    """
    canonical_index: Dict[str, str] = {}
    key_translation: Dict[str, str] = {}
    for actual_key, shard_file in weight_map.items():
        canonical_key = actual_key
        for pattern, replacement in key_remap.items():
            remapped = re.sub(pattern, replacement, actual_key)
            if remapped != actual_key:
                canonical_key = remapped
                key_translation[canonical_key] = actual_key
                break
        canonical_index[canonical_key] = shard_file
    return canonical_index, key_translation


# ---------------------------------------------------------------------------
# MoE-specific memory estimation — tied to _LayerStacker's tensor layout below,
# so it stays here rather than in the generic checkpoint_utils helpers.
# ---------------------------------------------------------------------------


def _moe_weights_prefix_from_experts_prefix(prefix: str) -> str:
    """Return the parent ``moe_weights`` prefix for a checkpoint expert prefix."""
    if prefix.endswith(".experts"):
        return prefix[: -len(".experts")] + ".moe_weights"
    return f"{prefix}.moe_weights"


def _infer_fused_gate_up_split_dim(
    gate_up_shape: Tuple[int, ...],
    down_shape: Optional[Tuple[int, ...]],
    *,
    preferred_split_dim: Optional[int] = None,
) -> int:
    """Infer whether fused gate/up uses [E,2I,H] or [E,H,2I]."""
    if len(gate_up_shape) != 3 or down_shape is None or len(down_shape) != 3:
        return preferred_split_dim if preferred_split_dim in (1, 2) else 1

    split_dim_1_match = gate_up_shape[1] == 2 * down_shape[2] and gate_up_shape[2] == down_shape[1]
    split_dim_2_match = gate_up_shape[2] == 2 * down_shape[1] and gate_up_shape[1] == down_shape[2]
    if split_dim_1_match and not split_dim_2_match:
        return 1
    if split_dim_2_match and not split_dim_1_match:
        return 2
    return preferred_split_dim if preferred_split_dim in (1, 2) else 1


def _split_fused_gate_up_to_canonical(
    gate_up: torch.Tensor,
    down_shape: Optional[Tuple[int, ...]] = None,
    *,
    interleaved: bool = False,
    preferred_split_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fused gate/up tensor into canonical gate/up [E,H,I] tensors."""
    split_dim = _infer_fused_gate_up_split_dim(
        tuple(gate_up.shape),
        down_shape,
        preferred_split_dim=preferred_split_dim,
    )
    if split_dim == 2:
        if interleaved:
            return gate_up[..., 0::2].contiguous(), gate_up[..., 1::2].contiguous()
        ffn_dim = gate_up.shape[2] // 2
        return gate_up[..., :ffn_dim].contiguous(), gate_up[..., ffn_dim:].contiguous()

    ffn_dim = gate_up.shape[1] // 2
    gate = gate_up[:, :ffn_dim, :].transpose(1, 2).contiguous()
    up = gate_up[:, ffn_dim:, :].transpose(1, 2).contiguous()
    return gate, up


def _down_to_canonical(
    down: torch.Tensor,
    gate_up_shape: Optional[Tuple[int, ...]] = None,
    *,
    preferred_split_dim: Optional[int] = None,
) -> torch.Tensor:
    """Return canonical down [E,I,H] for the fused source layout."""
    split_dim = _infer_fused_gate_up_split_dim(
        gate_up_shape or (),
        tuple(down.shape),
        preferred_split_dim=preferred_split_dim,
    )
    if split_dim == 2:
        return down.contiguous().clone()
    return down.transpose(1, 2).contiguous()


def _split_gate_up_bias(gate_up_bias: torch.Tensor, *, interleaved: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fused gate/up bias into canonical gate/up bias [E,I] tensors."""
    if interleaved:
        return gate_up_bias[..., 0::2].contiguous(), gate_up_bias[..., 1::2].contiguous()
    ffn_dim = gate_up_bias.shape[-1] // 2
    return gate_up_bias[..., :ffn_dim].contiguous(), gate_up_bias[..., ffn_dim:].contiguous()


# ---------------------------------------------------------------------------
# Transform 1: dtype conversion only — dense model path
# ---------------------------------------------------------------------------


class DtypeConversionCheckpointTransform(BaseCheckpointTransform):
    """Plan dense shard reuse or dtype conversion into final checkpoint shards."""

    TRANSFORM_ID = "dtype_conversion_v1"

    @classmethod
    def plan_tasks(cls, context: CheckpointPlanningContext) -> None:
        _plan_dtype_stages(cls, context)


# ---------------------------------------------------------------------------
# Internal stacker helper for MoE layers
# ---------------------------------------------------------------------------


class _LayerStacker:
    """Accumulates per-expert tensors for one MoE layer and produces batched output."""

    def __init__(self, prefix: str, num_experts: int):
        """Create an accumulator for all experts in one MoE layer."""
        self.prefix = prefix
        self.num_experts = num_experts
        self._gate: Optional[torch.Tensor] = None
        self._up: Optional[torch.Tensor] = None
        self._down: Optional[torch.Tensor] = None

    def add(self, expert_idx: int, kind: str, tensor: torch.Tensor) -> None:
        """Add one expert projection tensor to the layer accumulator."""
        # Accept qwen3-moe names (gate_proj/up_proj/down_proj),
        # grok-1 names (linear/linear_v/linear_1),
        # and Mixtral names (w1=gate, w3=up, w2=down) — map to the same accumulators.
        if kind in ("gate_proj", "linear", "w1"):
            ffn_dim, hidden_dim = tensor.shape
            if self._gate is None:
                self._gate = torch.empty(self.num_experts, ffn_dim, hidden_dim, dtype=tensor.dtype)
            self._gate[expert_idx] = tensor
        elif kind in ("up_proj", "linear_v", "w3"):
            ffn_dim, hidden_dim = tensor.shape
            if self._up is None:
                self._up = torch.empty(self.num_experts, ffn_dim, hidden_dim, dtype=tensor.dtype)
            self._up[expert_idx] = tensor
        else:  # down_proj / linear_1 / w2 — shape is [hidden_dim, ffn_dim]
            hidden_dim, ffn_dim = tensor.shape
            if self._down is None:
                self._down = torch.empty(self.num_experts, hidden_dim, ffn_dim, dtype=tensor.dtype)
            self._down[expert_idx] = tensor

    def stack(self) -> Dict[str, torch.Tensor]:
        """Return stacked expert tensors in the derived QEff checkpoint layout."""
        # Output in the canonical layout that OptimizedMoETransform creates so
        # promote_initializers_and_build_spec finds an exact checkpoint key match.
        #   _gate [E, I, H] -> transpose(1,2) -> moe_weights.gate [E, H, I]
        #   _up   [E, I, H] -> transpose(1,2) -> moe_weights.up   [E, H, I]
        #   _down [E, H, I] -> transpose(1,2) -> moe_weights.down [E, I, H]
        moe_prefix = _moe_weights_prefix_from_experts_prefix(self.prefix)
        gate = self._gate.transpose(1, 2).contiguous()
        up = self._up.transpose(1, 2).contiguous()
        down = self._down.transpose(1, 2).contiguous()
        return {
            f"{moe_prefix}.gate": gate,  # [E, H, I]
            f"{moe_prefix}.up": up,  # [E, H, I]
            f"{moe_prefix}.down": down,  # [E, I, H]
        }


# ---------------------------------------------------------------------------
# Transform 2: MoE expert stacking
# ---------------------------------------------------------------------------


class MoEExpertStackingCheckpointTransform(BaseCheckpointTransform):
    """Stack per-expert checkpoint keys into canonical batched tensors.

    Detects the HuggingFace per-expert layout::

        *.experts.{E}.gate_proj.weight  [I, H]  x  num_experts
        *.experts.{E}.up_proj.weight    [I, H]  x  num_experts
        *.experts.{E}.down_proj.weight  [H, I]  x  num_experts

    and produces::

        *.moe_weights.gate [E, H, I]   (gate weights, transposed)
        *.moe_weights.up   [E, H, I]   (up weights, transposed)
        *.moe_weights.down [E, I, H]   (down weights, transposed)

    matching the derived parameter layout that OptimizedMoETransform creates.
    Expert-parallel packing and dtype conversion are independent later stages
    attached by the shared planning context.
    """

    TRANSFORM_ID = "moe_expert_stacking_v1"
    EXPERT_RE = re.compile(
        r"^(.+\.layers\.(\d+)\..+?\.experts)\.(\d+)\."
        r"(gate_proj|up_proj|down_proj|linear|linear_v|linear_1|w1|w2|w3)\.weight$"
    )

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when the checkpoint uses per-expert MoE tensor keys."""
        return any(cls.EXPERT_RE.match(k) for k in weight_map)

    @classmethod
    def resolve_onnx_key(cls, onnx_key: str, checkpoint_index: Dict[str, str]) -> Optional[str]:
        """Explicit ONNX → checkpoint key mapping for per-expert MoE models.

        Handles the Mixtral/Qwen3-MoE convention where the ONNX graph names
        the MoE block as ``.mlp.`` but the checkpoint stores it as
        ``.block_sparse_moe.``.
        """
        if onnx_key in checkpoint_index:
            return onnx_key
        candidate = onnx_key.replace(".mlp.", ".block_sparse_moe.")
        if candidate in checkpoint_index:
            return candidate
        return None

    @classmethod
    def plan_tasks(cls, context: CheckpointPlanningContext) -> None:
        _plan_expert_stages(cls, context)


# ---------------------------------------------------------------------------
# Transform 3: GptOss MXFP4 dequantize + split fused projections
# ---------------------------------------------------------------------------


class GptOssMxfp4ExpertDequantSplitCheckpointTransform(BaseCheckpointTransform):
    """Dequantize MXFP4-packed stacked expert tensors and split fused gate_up_proj.

    Detects the GptOss MXFP4 checkpoint layout::

        *.experts.gate_up_proj_blocks  [E, 2*I, G, B]   U8
        *.experts.gate_up_proj_scales  [E, 2*I, G]       U8
        *.experts.gate_up_proj_bias    [E, 2*I]           BF16
        *.experts.down_proj_blocks     [E, I,   G, B]   U8
        *.experts.down_proj_scales     [E, I,   G]       U8
        *.experts.down_proj_bias       [E, H]             BF16

    and produces::

        *.moe_weights.gate      [E, H, I]   (dequant gate_up_proj, first half)
        *.moe_weights.up        [E, H, I]   (dequant gate_up_proj, second half)
        *.moe_weights.gate_bias [E, I]       (gate_up_proj_bias, first half)
        *.moe_weights.up_bias   [E, I]       (gate_up_proj_bias, second half)
        *.moe_weights.down      [E, I, H]   (dequant down_proj)
        *.moe_weights.down_bias [E, H]       (dtype-converted)

    matching the derived parameter layout that OptimizedMoETransform creates.
    Expert-parallel packing and dtype conversion are independent later stages
    attached by the shared planning context.
    """

    TRANSFORM_ID = "gptoss_mxfp4_dequant_v1"
    _BLOCKS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_blocks$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when the checkpoint contains GPT-OSS MXFP4 expert blocks."""
        return any(cls._BLOCKS_RE.match(k) for k in weight_map)

    @classmethod
    def resolve_onnx_key(cls, onnx_key: str, checkpoint_index: Dict[str, str]) -> Optional[str]:
        """Direct lookup only — GptOss checkpoint keys match ONNX names directly."""
        return onnx_key if onnx_key in checkpoint_index else None

    @classmethod
    def plan_tasks(cls, context: CheckpointPlanningContext) -> None:
        _plan_gptoss_stages(cls, context)


class FusedExpertSplitCheckpointTransform(BaseCheckpointTransform):
    """Split pre-stacked fused expert tensors into canonical moe_weights layout.

    Handles checkpoints where all experts are stored as one stacked tensor:

        *.experts.gate_up_proj  [E, 2*I, H]   → moe_weights.gate + up
        *.experts.down_proj     [E, H, I]      → moe_weights.down

    Architecture-specific key names are translated to the canonical form
    before grouped tasks are planned.
    """

    TRANSFORM_ID = "fused_expert_split_v1"

    # Canonical key patterns — all architectures map to these names.
    _FUSED_GATE_UP_RE = re.compile(r"^(.+\.experts)\.gate_up_proj$")
    _FUSED_DOWN_RE = re.compile(r"^(.+\.experts)\.down_proj$")
    _FUSED_GATE_UP_BIAS_RE = re.compile(r"^(.+\.experts)\.gate_up_proj_bias$")
    _FUSED_DOWN_BIAS_RE = re.compile(r"^(.+\.experts)\.down_proj_bias$")

    @classmethod
    def _get_key_remap(cls, weight_map: Dict[str, str]) -> Dict[str, str]:
        """Detect architecture from weight_map keys and return the right remap.

        Uses weight_map key patterns — consistent with the rest of the detection
        design and requires no config or model_type.

        GraniteMoE always uses input_linear/output_linear key names.
        Mixtral fused and others already use canonical experts.gate_up_proj names.
        """
        if any("input_linear.weight" in k for k in weight_map):
            return {
                r"\.input_linear\.weight$": ".experts.gate_up_proj",
                r"\.output_linear\.weight$": ".experts.down_proj",
            }
        return {}

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        canonical_index, _ = build_canonical_maps(weight_map, cls._get_key_remap(weight_map))
        return any(cls._FUSED_GATE_UP_RE.match(k) for k in canonical_index)

    @classmethod
    def resolve_onnx_key(cls, onnx_key: str, checkpoint_index: Dict[str, str]) -> Optional[str]:
        """Explicit ONNX → checkpoint key mapping for fused MoE models.

        Handles the Mixtral convention where the ONNX graph names the MoE
        block as ``.mlp.`` but the checkpoint stores it as ``.block_sparse_moe.``.
        Also handles GraniteMoE which uses canonical names after key remapping.
        """
        if onnx_key in checkpoint_index:
            return onnx_key
        candidate = onnx_key.replace(".mlp.", ".block_sparse_moe.")
        if candidate in checkpoint_index:
            return candidate
        return None

    @classmethod
    def plan_tasks(cls, context: CheckpointPlanningContext) -> None:
        _plan_fused_stages(cls, context)

    @classmethod
    def _resolve_split_dim(cls, prefix: str, canonical_index: Dict[str, str]) -> int:
        """Return split dimension from canonical_index key presence.

        GptOss-MXFP4 has its own transform so FusedExpertSplitCheckpointTransform
        only sees two cases:
          bias present → GptOss dense interleaved → dim=2
          bias absent  → Mixtral or GraniteMoE    → dim=1
        No shape reads needed — canonical_index (from index.json) is sufficient.
        """
        return 2 if f"{prefix}.gate_up_proj_bias" in canonical_index else 1


# ---------------------------------------------------------------------------
# Independent stage transforms and task planning
# ---------------------------------------------------------------------------


class ExpertParallelPackingCheckpointTransform(BaseCheckpointTransform):
    """Append expert-parallel packing to canonical MoE tensor groups."""

    TRANSFORM_ID = "expert_parallel_pack_v1"

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        return kwargs.get("hash_params", {}).get("moe_prefill_flavour") == "expert_parallel"

    @classmethod
    def plan_tasks(cls, context: CheckpointPlanningContext) -> None:
        _plan_expert_parallel_stages(cls, context)


def _estimate_task_bytes(
    src: Optional[Path],
    weight_map: Dict[str, str],
    keys,
    target_dtype,
    *,
    output_copies: int = 1,
) -> int:
    """Estimate source tensors plus live target-sized output copies."""
    if src is None or not keys:
        return 1
    source_bytes = 0
    output_bytes = 0
    seen = set()
    target_element_size = torch.empty((), dtype=target_dtype).element_size()
    for key in keys:
        shard = weight_map[key]
        if (shard, key) in seen:
            continue
        seen.add((shard, key))
        with safe_open(str(src / shard), framework="pt") as handle:
            tensor_slice = handle.get_slice(key)
            elements = 1
            for dimension in tensor_slice.get_shape():
                elements *= dimension
            dtype_name = tensor_slice.get_dtype()
            source_element_size = {"F32": 4, "F16": 2, "BF16": 2}.get(dtype_name, 1)
            source_bytes += elements * source_element_size
            output_bytes += elements * target_element_size
    return max(1, source_bytes + output_bytes * output_copies)


def _estimate_gptoss_task_bytes(
    src: Optional[Path],
    weight_map: Dict[str, str],
    keys,
    target_dtype: torch.dtype,
    *,
    packed: bool,
) -> int:
    """Estimate MXFP4 inputs, dequantized temporaries, final outputs, and packing."""
    if src is None or not keys:
        return 1

    source_bytes = 0
    final_output_bytes = 0
    target_element_size = torch.empty((), dtype=target_dtype).element_size()
    for key in keys:
        with safe_open(str(src / weight_map[key]), framework="pt") as handle:
            tensor_slice = handle.get_slice(key)
            elements = 1
            for dimension in tensor_slice.get_shape():
                elements *= dimension
            dtype_name = tensor_slice.get_dtype()
            source_element_size = {"F32": 4, "F16": 2, "BF16": 2}.get(dtype_name, 1)
            source_bytes += elements * source_element_size
            if key.endswith("_blocks"):
                final_output_bytes += elements * 2 * target_element_size
            elif key.endswith("_bias"):
                final_output_bytes += elements * target_element_size

    output_copies = 3 if packed else 2
    return max(1, source_bytes + final_output_bytes * output_copies)


def _expert_parallel_params(hash_params: Dict) -> Optional[Tuple[int, int]]:
    if hash_params.get("moe_prefill_flavour") != "expert_parallel":
        return None

    pipeline_stages = hash_params.get("moe_prefill_num_pipeline_stages")
    parallelized_experts = hash_params.get("moe_prefill_num_parallelized_experts")
    if pipeline_stages is None or parallelized_experts is None:
        raise ValueError(
            "expert_parallel flavour requires moe_prefill_num_pipeline_stages "
            "and moe_prefill_num_parallelized_experts in hash_params."
        )

    pipeline_stages = int(pipeline_stages)
    parallelized_experts = int(parallelized_experts)
    if pipeline_stages <= 0 or parallelized_experts <= 0:
        raise ValueError("expert_parallel pipeline stages and parallelized experts must be positive.")

    return pipeline_stages, parallelized_experts


def _task_refs(keys, stage="raw") -> tuple[TensorRef, ...]:
    return tuple(TensorRef(key, stage) for key in sorted(keys))


def _task_values(**values) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(values.items(), key=lambda item: item[0]))


def _declared_num_experts(config) -> Optional[int]:
    if config is None:
        return None
    return getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)


def _source_num_experts(context: CheckpointPlanningContext, key: str) -> int:
    with safe_open(str(context.source_dir / context.weight_map[key]), framework="pt") as handle:
        return int(handle.get_slice(key).get_shape()[0])


def _can_reuse_dense_bf16_shard(
    source_dir: Optional[Path],
    shard_name: str,
    keys: tuple[str, ...],
    target_dtype: torch.dtype,
) -> bool:
    if source_dir is None or target_dtype != torch.bfloat16:
        return False

    with safe_open(str(source_dir / shard_name), framework="pt") as handle:
        if set(handle.keys()) != set(keys):
            return False

        for key in keys:
            safetensors_dtype = handle.get_slice(key).get_dtype()
            torch_dtype = safetensors_dtype_to_torch(safetensors_dtype)
            if torch_dtype is not None and torch_dtype != target_dtype:
                return False
            if torch_dtype is None and safetensors_dtype.startswith(("F", "BF")):
                return False

    return True


def _run_reuse_shard_task(shard_name, keys, output_file):
    def runner(src: Path, out: Path, target_dtype: torch.dtype) -> Dict[str, str]:
        source = (src / shard_name).resolve()
        destination = out / output_file
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)

        try:
            os.link(source, temporary)
        except OSError:
            try:
                temporary.symlink_to(source)
            except OSError:
                shutil.copy2(source, temporary)

        temporary.replace(destination)
        return {key: output_file for key in keys}

    return runner


def _make_dtype_stage(cls, input_refs: tuple[TensorRef, ...], target_dtype: torch.dtype) -> CheckpointStage:
    output_refs = tuple(TensorRef(ref.key, "final") for ref in input_refs)

    def runner(get_tensor, requested_dtype):
        outputs = {}
        for input_ref, output_ref in zip(input_refs, output_refs):
            tensor = get_tensor(input_ref)
            if tensor.is_floating_point() and tensor.dtype != requested_dtype:
                tensor = tensor.to(requested_dtype)
            outputs[output_ref] = tensor.contiguous()
        return outputs

    return CheckpointStage(
        stage_id="dtype",
        input_refs=input_refs,
        output_refs=output_refs,
        params=TaskParams(cls.TRANSFORM_ID, _task_values(target_dtype=str(target_dtype))),
        runner=runner,
        labels=("dtype",),
    )


def _plan_dtype_stages(cls, context: CheckpointPlanningContext) -> None:
    for task_plan in context.task_plans:
        if any(ref.stage == "final" for ref in task_plan.current_refs):
            continue
        task_plan.append_stage(_make_dtype_stage(cls, task_plan.current_refs, context.target_dtype))

    remaining_weight_map = context.remaining_weight_map()
    by_shard: Dict[str, List[str]] = {}
    for key, shard in remaining_weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    allow_shard_reuse = not context.active_layout_ids
    for index, (shard_name, keys) in enumerate(sorted(by_shard.items())):
        keys = tuple(sorted(keys))
        output_file = f"base-{index:04d}.safetensors"
        if allow_shard_reuse and _can_reuse_dense_bf16_shard(
            context.source_dir,
            shard_name,
            keys,
            context.target_dtype,
        ):
            context.add_direct_task(
                CheckpointTask(
                    task_id=f"reuse:{shard_name}",
                    input_refs=_task_refs(keys),
                    output_refs=_task_refs(keys, "final"),
                    source_files=(shard_name,),
                    output_file=output_file,
                    estimated_peak_bytes=1,
                    params=TaskParams(
                        "dense_bf16_shard_reuse_v1",
                        _task_values(
                            keys=keys,
                            output_file=output_file,
                            source_file=shard_name,
                            stages=("reuse", "final"),
                            target_dtype=str(context.target_dtype),
                        ),
                    ),
                    runner=_run_reuse_shard_task(shard_name, keys, output_file),
                )
            )
            continue

        input_refs = _task_refs(keys)
        task_plan = CheckpointTaskPlan(
            task_id=f"dtype:{shard_name}",
            input_refs=input_refs,
            source_files=(shard_name,),
            output_file=output_file,
            estimated_peak_bytes=_estimate_task_bytes(
                context.source_dir,
                context.weight_map,
                keys,
                context.target_dtype,
            ),
            params=TaskParams(
                cls.TRANSFORM_ID,
                _task_values(keys=keys, output_file=output_file, target_dtype=str(context.target_dtype)),
            ),
        )
        task_plan.append_stage(_make_dtype_stage(cls, input_refs, context.target_dtype))
        context.add_task_plan(task_plan)


def _plan_expert_parallel_stages(cls, context: CheckpointPlanningContext) -> None:
    pack_params = _expert_parallel_params(context.hash_params)
    if pack_params is None:
        return

    from QEfficient.transformers.moe.weights import _pack_expert_parallel_tensor

    for task_plan in context.task_plans:
        if task_plan.params.transform_id not in {
            MoEExpertStackingCheckpointTransform.TRANSFORM_ID,
            GptOssMxfp4ExpertDequantSplitCheckpointTransform.TRANSFORM_ID,
            FusedExpertSplitCheckpointTransform.TRANSFORM_ID,
        }:
            continue

        num_experts = task_plan.params.as_dict().get("num_experts")
        if num_experts is not None and num_experts != pack_params[0] * pack_params[1]:
            raise ValueError(
                f"Checkpoint task {task_plan.task_id} has {num_experts} experts, but expert_parallel layout "
                f"requires P * E/P = {pack_params[0] * pack_params[1]}."
            )

        input_refs = task_plan.current_refs
        output_refs = tuple(TensorRef(ref.key, "packed") for ref in input_refs)

        def runner(get_tensor, target_dtype, input_refs=input_refs, output_refs=output_refs):
            outputs = {}
            for input_ref, output_ref in zip(input_refs, output_refs):
                tensor = get_tensor(input_ref)
                outputs[output_ref] = (
                    _pack_expert_parallel_tensor(
                        tensor,
                        num_pipeline_stages=pack_params[0],
                        num_parallelized_experts=pack_params[1],
                    ).data
                    if tensor.is_floating_point()
                    else tensor
                )
            return outputs

        task_plan.append_stage(
            CheckpointStage(
                stage_id="expert_parallel_pack",
                input_refs=input_refs,
                output_refs=output_refs,
                params=TaskParams(
                    cls.TRANSFORM_ID,
                    _task_values(
                        parallelized_experts=pack_params[1],
                        pipeline_stages=pack_params[0],
                    ),
                ),
                runner=runner,
                labels=("expert_parallel_pack",),
            )
        )


def _expert_entries_for_plan(cls, weight_map):
    entries: Dict[int, Dict[Tuple[int, str], Tuple[str, str]]] = {}
    prefixes: Dict[int, str] = {}
    for key, shard in weight_map.items():
        match = cls.EXPERT_RE.match(key)
        if match:
            layer_index = int(match.group(2))
            expert_index = int(match.group(3))
            kind = match.group(4)
            entries.setdefault(layer_index, {})[(expert_index, kind)] = (shard, key)
            prefixes[layer_index] = match.group(1)

    for layer_index, layer_entries in entries.items():
        expert_indices = sorted({expert for expert, _ in layer_entries})
        groups = [
            {"gate_proj", "linear", "w1"},
            {"up_proj", "linear_v", "w3"},
            {"down_proj", "linear_1", "w2"},
        ]
        for expert_index in expert_indices:
            for group in groups:
                if not any((expert_index, kind) in layer_entries for kind in group):
                    raise ValueError(
                        f"Layer {layer_index}, expert {expert_index} is missing one of {sorted(group)} projection keys."
                    )
    return entries, prefixes


def _plan_expert_stages(cls, context: CheckpointPlanningContext) -> None:
    entries_by_layer, prefixes = _expert_entries_for_plan(cls, context.weight_map)
    declared_num_experts = _declared_num_experts(context.config)
    packed = _expert_parallel_params(context.hash_params) is not None

    for layer_index in sorted(entries_by_layer):
        layer_entries = entries_by_layer[layer_index]
        expert_indices = {expert for expert, _ in layer_entries}
        num_experts = len(expert_indices)
        if declared_num_experts is not None and expert_indices != set(range(declared_num_experts)):
            raise ValueError(
                f"Layer {layer_index}: config declares {declared_num_experts} experts "
                f"but checkpoint contains indices {sorted(expert_indices)}."
            )

        input_keys = [key for _, key in layer_entries.values()]
        input_refs = _task_refs(input_keys)
        output_prefix = _moe_weights_prefix_from_experts_prefix(prefixes[layer_index])
        output_refs = _task_refs(
            [f"{output_prefix}.gate", f"{output_prefix}.up", f"{output_prefix}.down"],
            "stacked",
        )
        output_file = f"experts-layer-{layer_index:05d}.safetensors"

        def runner(
            get_tensor,
            target_dtype,
            layer_entries=layer_entries,
            prefix=prefixes[layer_index],
            num_experts=num_experts,
            output_refs=output_refs,
        ):
            stacker = _LayerStacker(prefix, num_experts)
            for (expert_index, kind), (_, key) in sorted(layer_entries.items()):
                stacker.add(expert_index, kind, get_tensor(TensorRef(key, "raw")))
            stacked = stacker.stack()
            return {output_ref: stacked[output_ref.key] for output_ref in output_refs}

        task_plan = CheckpointTaskPlan(
            task_id=f"{cls.TRANSFORM_ID}:layer:{layer_index}",
            input_refs=input_refs,
            source_files=tuple(sorted({shard for shard, _ in layer_entries.values()})),
            output_file=output_file,
            estimated_peak_bytes=_estimate_task_bytes(
                context.source_dir,
                context.weight_map,
                input_keys,
                context.target_dtype,
                output_copies=2 if packed else 1,
            ),
            params=TaskParams(
                cls.TRANSFORM_ID,
                _task_values(layer=layer_index, num_experts=num_experts, output_file=output_file),
            ),
        )
        task_plan.append_stage(
            CheckpointStage(
                stage_id="stack",
                input_refs=input_refs,
                output_refs=output_refs,
                params=TaskParams(cls.TRANSFORM_ID, _task_values(layer=layer_index, num_experts=num_experts)),
                runner=runner,
                labels=("stack",),
            )
        )
        context.add_task_plan(task_plan, layout_transform_id=cls.TRANSFORM_ID)


def _gptoss_locations_for_plan(cls, weight_map):
    scales_re = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_scales$")
    bias_re = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_bias$")
    locations = {}
    biases = {}
    prefixes = {}
    for key, shard in weight_map.items():
        match = cls._BLOCKS_RE.match(key)
        if match:
            locations.setdefault(int(match.group(2)), {}).setdefault(match.group(3), {})["blocks"] = (shard, key)
            prefixes[int(match.group(2))] = match.group(1)
            continue
        match = scales_re.match(key)
        if match:
            locations.setdefault(int(match.group(2)), {}).setdefault(match.group(3), {})["scales"] = (shard, key)
            prefixes[int(match.group(2))] = match.group(1)
            continue
        match = bias_re.match(key)
        if match:
            biases[(int(match.group(2)), match.group(3))] = (shard, key)
            prefixes[int(match.group(2))] = match.group(1)

    for layer_index, kinds in locations.items():
        for kind in ("gate_up_proj", "down_proj"):
            if "blocks" not in kinds.get(kind, {}) or "scales" not in kinds.get(kind, {}):
                raise ValueError(f"GPT-OSS layer {layer_index} is missing {kind} blocks or scales.")
    return locations, biases, prefixes


def _plan_gptoss_stages(cls, context: CheckpointPlanningContext) -> None:
    locations_by_layer, biases, prefixes = _gptoss_locations_for_plan(cls, context.weight_map)
    packed = _expert_parallel_params(context.hash_params) is not None

    for layer_index in sorted(locations_by_layer):
        locations = locations_by_layer[layer_index]
        input_keys = [
            key
            for kind in ("gate_up_proj", "down_proj")
            for part in ("blocks", "scales")
            for _, key in [locations[kind][part]]
        ]
        for kind in ("gate_up_proj", "down_proj"):
            if (layer_index, kind) in biases:
                input_keys.append(biases[(layer_index, kind)][1])

        input_refs = _task_refs(input_keys)
        output_prefix = _moe_weights_prefix_from_experts_prefix(prefixes[layer_index])
        output_keys = [f"{output_prefix}.gate", f"{output_prefix}.up", f"{output_prefix}.down"]
        if (layer_index, "gate_up_proj") in biases:
            output_keys.extend([f"{output_prefix}.gate_bias", f"{output_prefix}.up_bias"])
        if (layer_index, "down_proj") in biases:
            output_keys.append(f"{output_prefix}.down_bias")
        output_refs = _task_refs(output_keys, "canonical")
        output_file = f"experts-layer-{layer_index:05d}.safetensors"
        num_experts = _declared_num_experts(context.config) or _source_num_experts(
            context, locations["gate_up_proj"]["blocks"][1]
        )

        def runner(
            get_tensor,
            target_dtype,
            locations=locations,
            biases=biases,
            layer_index=layer_index,
            output_prefix=output_prefix,
            output_refs=output_refs,
        ):
            def load(location):
                return get_tensor(TensorRef(location[1], "raw"))

            tensors = {}
            gate_up = convert_moe_packed_tensors(
                load(locations["gate_up_proj"]["blocks"]),
                load(locations["gate_up_proj"]["scales"]),
                dtype=target_dtype,
            )
            tensors[f"{output_prefix}.gate"] = gate_up[..., 0::2].contiguous()
            tensors[f"{output_prefix}.up"] = gate_up[..., 1::2].contiguous()
            if (layer_index, "gate_up_proj") in biases:
                bias = load(biases[(layer_index, "gate_up_proj")])
                tensors[f"{output_prefix}.gate_bias"] = bias[..., 0::2].contiguous()
                tensors[f"{output_prefix}.up_bias"] = bias[..., 1::2].contiguous()

            tensors[f"{output_prefix}.down"] = convert_moe_packed_tensors(
                load(locations["down_proj"]["blocks"]),
                load(locations["down_proj"]["scales"]),
                dtype=target_dtype,
            )
            if (layer_index, "down_proj") in biases:
                tensors[f"{output_prefix}.down_bias"] = load(biases[(layer_index, "down_proj")])
            return {output_ref: tensors[output_ref.key] for output_ref in output_refs}

        task_plan = CheckpointTaskPlan(
            task_id=f"{cls.TRANSFORM_ID}:layer:{layer_index}",
            input_refs=input_refs,
            source_files=tuple(sorted({context.weight_map[key] for key in input_keys})),
            output_file=output_file,
            estimated_peak_bytes=_estimate_gptoss_task_bytes(
                context.source_dir,
                context.weight_map,
                input_keys,
                context.target_dtype,
                packed=packed,
            ),
            params=TaskParams(
                cls.TRANSFORM_ID,
                _task_values(layer=layer_index, num_experts=num_experts, output_file=output_file),
            ),
        )
        task_plan.append_stage(
            CheckpointStage(
                stage_id="gptoss_dequant_split",
                input_refs=input_refs,
                output_refs=output_refs,
                params=TaskParams(cls.TRANSFORM_ID, _task_values(layer=layer_index)),
                runner=runner,
                labels=("dequant", "split"),
            )
        )
        context.add_task_plan(task_plan, layout_transform_id=cls.TRANSFORM_ID)


def _plan_fused_stages(cls, context: CheckpointPlanningContext) -> None:
    key_remap = cls._get_key_remap(context.weight_map)
    canonical_index, key_translation = build_canonical_maps(context.weight_map, key_remap)
    keys_by_prefix: Dict[str, List[str]] = {}
    for canonical_key in canonical_index:
        match = (
            cls._FUSED_GATE_UP_RE.match(canonical_key)
            or cls._FUSED_DOWN_RE.match(canonical_key)
            or cls._FUSED_GATE_UP_BIAS_RE.match(canonical_key)
            or cls._FUSED_DOWN_BIAS_RE.match(canonical_key)
        )
        if match:
            keys_by_prefix.setdefault(match.group(1), []).append(canonical_key)

    packed = _expert_parallel_params(context.hash_params) is not None
    for group_index, prefix in enumerate(sorted(keys_by_prefix)):
        canonical_keys = tuple(sorted(keys_by_prefix[prefix]))
        required_keys = {f"{prefix}.gate_up_proj", f"{prefix}.down_proj"}
        missing_keys = sorted(required_keys - set(canonical_keys))
        if missing_keys:
            raise ValueError(f"Fused expert group {prefix} is missing required keys: {missing_keys}")

        raw_keys = [key_translation.get(key, key) for key in canonical_keys]
        input_refs = _task_refs(raw_keys)
        output_file = f"fused-group-{group_index:05d}.safetensors"
        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
        output_keys = [f"{moe_prefix}.gate", f"{moe_prefix}.up", f"{moe_prefix}.down"]
        if f"{prefix}.gate_up_proj_bias" in canonical_keys:
            output_keys.extend([f"{moe_prefix}.gate_bias", f"{moe_prefix}.up_bias"])
        if f"{prefix}.down_proj_bias" in canonical_keys:
            output_keys.append(f"{moe_prefix}.down_bias")
        output_refs = _task_refs(output_keys, "canonical")
        gate_up_raw_key = key_translation.get(f"{prefix}.gate_up_proj", f"{prefix}.gate_up_proj")
        num_experts = _declared_num_experts(context.config) or _source_num_experts(context, gate_up_raw_key)

        def runner(
            get_tensor,
            target_dtype,
            canonical_keys=canonical_keys,
            key_translation=key_translation,
            canonical_index=canonical_index,
            output_refs=output_refs,
        ):
            out_tensors = {}
            for canonical_key in canonical_keys:
                raw_key = key_translation.get(canonical_key, canonical_key)
                tensor = get_tensor(TensorRef(raw_key, "raw"))
                gate_up_match = cls._FUSED_GATE_UP_RE.match(canonical_key)
                down_match = cls._FUSED_DOWN_RE.match(canonical_key)
                gate_bias_match = cls._FUSED_GATE_UP_BIAS_RE.match(canonical_key)
                down_bias_match = cls._FUSED_DOWN_BIAS_RE.match(canonical_key)

                if gate_up_match:
                    group_prefix = gate_up_match.group(1)
                    group_moe_prefix = _moe_weights_prefix_from_experts_prefix(group_prefix)
                    split_dim = cls._resolve_split_dim(group_prefix, canonical_index)
                    gate, up = _split_fused_gate_up_to_canonical(
                        tensor,
                        None,
                        interleaved=split_dim == 2,
                        preferred_split_dim=split_dim,
                    )
                    out_tensors[f"{group_moe_prefix}.gate"] = gate
                    out_tensors[f"{group_moe_prefix}.up"] = up
                elif down_match:
                    group_prefix = down_match.group(1)
                    group_moe_prefix = _moe_weights_prefix_from_experts_prefix(group_prefix)
                    out_tensors[f"{group_moe_prefix}.down"] = _down_to_canonical(
                        tensor,
                        None,
                        preferred_split_dim=cls._resolve_split_dim(group_prefix, canonical_index),
                    )
                elif gate_bias_match:
                    group_prefix = gate_bias_match.group(1)
                    group_moe_prefix = _moe_weights_prefix_from_experts_prefix(group_prefix)
                    gate_bias, up_bias = _split_gate_up_bias(tensor, interleaved=True)
                    out_tensors[f"{group_moe_prefix}.gate_bias"] = gate_bias
                    out_tensors[f"{group_moe_prefix}.up_bias"] = up_bias
                elif down_bias_match:
                    group_prefix = down_bias_match.group(1)
                    group_moe_prefix = _moe_weights_prefix_from_experts_prefix(group_prefix)
                    out_tensors[f"{group_moe_prefix}.down_bias"] = tensor.contiguous()
            return {output_ref: out_tensors[output_ref.key] for output_ref in output_refs}

        task_plan = CheckpointTaskPlan(
            task_id=f"{cls.TRANSFORM_ID}:{prefix}",
            input_refs=input_refs,
            source_files=tuple(sorted({canonical_index[key] for key in canonical_keys})),
            output_file=output_file,
            estimated_peak_bytes=_estimate_task_bytes(
                context.source_dir,
                context.weight_map,
                raw_keys,
                context.target_dtype,
                output_copies=2 if packed else 1,
            ),
            params=TaskParams(
                cls.TRANSFORM_ID,
                _task_values(
                    keys=canonical_keys,
                    num_experts=num_experts,
                    output_file=output_file,
                    prefix=prefix,
                ),
            ),
        )
        task_plan.append_stage(
            CheckpointStage(
                stage_id="fused_split",
                input_refs=input_refs,
                output_refs=output_refs,
                params=TaskParams(cls.TRANSFORM_ID, _task_values(keys=canonical_keys, prefix=prefix)),
                runner=runner,
                labels=("split",),
            )
        )
        context.add_task_plan(task_plan, layout_transform_id=cls.TRANSFORM_ID)


# Backward-compatible aliases — existing callers keep working.
GraniteMoeFusedExpertSplitCheckpointTransform = FusedExpertSplitCheckpointTransform
MoEFusedExpertSplitCheckpointTransform = FusedExpertSplitCheckpointTransform
