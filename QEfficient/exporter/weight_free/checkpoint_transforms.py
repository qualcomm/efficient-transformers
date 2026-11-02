# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Checkpoint preparation transforms for weight-free ONNX export.

Concrete transforms below are picked in priority order by CheckpointTransformPipeline
(QEfficient/base/checkpoint_transforms.py) — the first whose is_applicable() returns
True runs and the pipeline stops. Layout transforms rewrite HF checkpoint keys to
match QEff-derived parameters; DtypeConversionCheckpointTransform is only used when
the source floating-point dtype does not already match the exported ONNX input dtype.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from QEfficient.base.checkpoint_transforms import CHECKPOINT_PREPARED_SENTINEL, BaseCheckpointTransform
from QEfficient.transformers.quantizers.quantizer_utils import blockwise_dequantize, convert_moe_packed_tensors
from QEfficient.utils.checkpoint_utils import (
    atomic_save,
    available_ram_gb,
    copy_checkpoint_aux_files,
    cpu_count,
    read_weight_map,
    requires_dtype_conversion,
    write_index,
)
from QEfficient.utils.logging_utils import logger

# ---------------------------------------------------------------------------
# MoE-specific memory estimation — tied to _LayerStacker's tensor layout below,
# so it stays here rather than in the generic checkpoint_utils helpers.
# ---------------------------------------------------------------------------


def _estimate_layer_stack_gb(
    expert_entries: Dict[Tuple[int, int, str], Tuple[str, str]],
    layer_idx: int,
    num_experts: int,
    src: Path,
    target_dtype: torch.dtype = torch.float32,
) -> float:
    """Estimate peak RAM (GB) required to stack one MoE layer's experts.

    At the moment stacker.stack(target_dtype) runs, five tensors exist in RAM:

        Inputs  (checkpoint dtype, e.g. BF16):
          gate  [E, I, H]
          up    [E, I, H]
          down  [E, H, I]

        Outputs (target_dtype, e.g. FP32 — twice as large when converting BF16->FP32):
          gate [E, H, I]
          up   [E, H, I]
          down [E, I, H]

    Using source dtype bytes for the outputs underestimates by ~45% when
    converting BF16→FP32, causing too many parallel workers and OOM.
    Returns 1.0 GB as a safe fallback if the shape cannot be read.
    """
    sample = next(
        (v for (li, ei, k), v in expert_entries.items() if li == layer_idx and k in ("gate_proj", "linear", "w1")),
        None,
    )
    if sample is None:
        return 1.0

    shard_name, orig_key = sample
    try:
        with safe_open(str(src / shard_name), framework="pt") as f:
            sl = f.get_slice(orig_key)
            shape = sl.get_shape()  # [I, H]
            dtype_str = sl.get_dtype()
    except Exception:
        return 1.0

    src_bytes = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1}.get(dtype_str, 2)
    tgt_bytes = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(target_dtype, 4)
    ffn_dim, hidden_dim = shape

    # Three input accumulators in source dtype + three output tensors in target dtype
    input_elements = num_experts * (
        ffn_dim * hidden_dim + ffn_dim * hidden_dim + hidden_dim * ffn_dim
    )  # gate + up + down
    output_elements = num_experts * (2 * ffn_dim * hidden_dim + hidden_dim * ffn_dim)  # gate + up + down
    return (input_elements * src_bytes + output_elements * tgt_bytes) / 1024**3


# ---------------------------------------------------------------------------
# Sentinel marking a fully-prepared checkpoint directory
# ---------------------------------------------------------------------------
_SENTINEL = CHECKPOINT_PREPARED_SENTINEL


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
    """Convert all floating-point tensors to ``target_dtype``.

    One pass per shard, shards processed in parallel via ThreadPoolExecutor.
    Used as the dense-model fallback; for MoE checkpoints,
    MoEExpertStackingCheckpointTransform handles dtype conversion as part
    of its own single pass and this transform is never reached.
    """

    @classmethod
    def is_applicable(
        cls,
        weight_map: Dict[str, str],
        src: Optional[Path] = None,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> bool:
        """Return True when dtype conversion is required for this checkpoint."""
        return src is None or requires_dtype_conversion(Path(src), weight_map, target_dtype)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Convert checkpoint shards to ``target_dtype`` in a prepared output directory."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("DtypeConversionCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        # TODO(wf): Move sidecar copying out of dtype conversion into a separate preparation step.
        # This is not a transform
        copy_checkpoint_aux_files(src, out)

        weight_map = read_weight_map(src)
        shard_names = sorted(set(weight_map.values()))
        new_name_for = {
            shard: (f"model_{idx:04d}.safetensors" if len(shard_names) > 1 else "model.safetensors")
            for idx, shard in enumerate(shard_names)
        }

        # I/O-bound: one thread per shard, capped at 4× CPU count and hard-capped
        # at 256 — beyond that OS scheduling overhead outweighs I/O parallelism gains.
        n_workers = max_workers if max_workers is not None else min(len(shard_names), cpu_count() * 4, 256)

        def _process_shard(shard_name: str) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            atomic_save(tensors, out / new_name_for[shard_name])

        logger.info(
            f"DtypeConversionCheckpointTransform: converting {len(shard_names)} shards "
            f"→ {target_dtype} | workers={n_workers} (cpus={cpu_count()})"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_shard, s) for s in shard_names]
            for fut in as_completed(futures):
                fut.result()

        new_weight_map = {k: new_name_for[v] for k, v in weight_map.items()}
        write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"DtypeConversionCheckpointTransform: done → {out}")
        return True


# ---------------------------------------------------------------------------
# Transform 2: reduced GLM-MoE-DSA FP8 checkpoint preparation
# ---------------------------------------------------------------------------


class GlmMoeDsaReducedCheckpointTransform(BaseCheckpointTransform):
    """Prepare an active-layer GLM-MoE-DSA checkpoint.

    The source checkpoint contains blockwise FP8 weights and per-expert MLP
    tensors. This transform reads the configured active layers, dequantizes
    their FP8 weights, and materializes QEff-only attention and MoE parameters.
    """

    _LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
    _EXPERT_RE = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    _Q_B_PROJ_SUFFIX = ".self_attn.q_b_proj.weight"
    _SCALE_SUFFIX = "_scale_inv"

    @classmethod
    def is_applicable(cls, weight_map: dict[str, str], **kwargs) -> bool:
        """Return True for GLM-MoE-DSA FP8 checkpoints."""
        model_config = kwargs.get("model_config")
        is_glm_config = getattr(model_config, "model_type", None) == "glm_moe_dsa"
        quantization_config = getattr(model_config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method") or quantization_config.get("quant_type")
        else:
            quant_method = getattr(quantization_config, "quant_method", None) or getattr(
                quantization_config, "quant_type", None
            )
        quant_method = quant_method.value if hasattr(quant_method, "value") else quant_method
        has_preserved_fp8_metadata = getattr(model_config, "_glm_moe_dsa_weight_block_size", None) is not None
        is_fp8_checkpoint = quant_method == "fp8" or (quant_method is None and has_preserved_fp8_metadata)
        has_q_b_proj = any(
            key.endswith(cls._Q_B_PROJ_SUFFIX) and f"{key}{cls._SCALE_SUFFIX}" in weight_map for key in weight_map
        )
        dense_mlp_suffixes = (".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")
        has_dense_mlp = all(any(key.endswith(suffix) for key in weight_map) for suffix in dense_mlp_suffixes)
        has_per_expert_mlp = any(cls._EXPERT_RE.match(key) for key in weight_map)
        # Weight-free GLM export clears config.quantization_config before
        # tracing, but the source checkpoint still carries the FP8 tensors and
        # scale-inverse entries.  Select the reduced-layer transform from the
        # checkpoint structure in that case as well; otherwise the generic
        # expert stacker scans every layer in the full checkpoint.
        return is_glm_config and is_fp8_checkpoint and has_q_b_proj and has_dense_mlp and has_per_expert_mlp

    @staticmethod
    def _weight_block_size(model_config) -> tuple[int, int]:
        quantization_config = getattr(model_config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            block_size = quantization_config.get("weight_block_size")
        else:
            block_size = getattr(quantization_config, "weight_block_size", None)
        # The weight-free GLM tracing flow clears quantization_config before
        # constructing the QEff model.  Preserve/use the source FP8 metadata
        # when that happens instead of losing the dequantization block shape.
        if block_size is None:
            block_size = getattr(model_config, "_glm_moe_dsa_weight_block_size", None)
        if not isinstance(block_size, (list, tuple)) or len(block_size) != 2:
            raise ValueError("GLM-MoE-DSA weight-free export requires a two-dimensional FP8 weight_block_size.")
        return tuple(int(size) for size in block_size)

    @staticmethod
    def _dequantize_fp8_weight(
        tensor: torch.Tensor,
        scale: torch.Tensor,
        block_size: tuple[int, int],
    ) -> torch.Tensor:
        """Dequantize a blockwise FP8 tensor, including a partial final block."""
        rows, cols = tensor.shape[-2:]
        block_rows, block_cols = block_size
        padded_rows = scale.shape[-2] * block_rows
        padded_cols = scale.shape[-1] * block_cols
        if padded_rows == rows and padded_cols == cols:
            return blockwise_dequantize(tensor, scale, block_size)
        if padded_rows < rows or padded_cols < cols:
            raise ValueError(
                f"FP8 scale grid {tuple(scale.shape[-2:])} cannot cover weight shape {(rows, cols)} "
                f"with block size {block_size}."
            )
        padded = torch.zeros((*tensor.shape[:-2], padded_rows, padded_cols), dtype=tensor.dtype, device=tensor.device)
        padded[..., :rows, :cols] = tensor
        return blockwise_dequantize(padded, scale, block_size)[..., :rows, :cols]

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        model_config=None,
        **kwargs,
    ) -> bool:
        """Write a dequantized active-layer GLM-MoE-DSA checkpoint."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("GlmMoeDsaReducedCheckpointTransform: prepared checkpoint exists, skipping.")
            return False
        num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
        if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
            raise ValueError("GLM-MoE-DSA weight-free export requires a positive num_hidden_layers value.")

        block_size = cls._weight_block_size(model_config)
        weight_map = read_weight_map(src)
        if not cls.is_applicable(weight_map, model_config=model_config):
            raise ValueError("Checkpoint does not use the supported GLM-5.3 FP8 per-expert layout.")

        base_entries: dict[str, str] = {}
        expert_entries: dict[tuple[int, int, str], tuple[str, str]] = {}
        for key, shard_name in weight_map.items():
            layer_match = cls._LAYER_RE.match(key)
            if layer_match is not None and int(layer_match.group(1)) >= num_hidden_layers:
                continue
            if key.startswith("model.mtp"):
                continue
            expert_match = cls._EXPERT_RE.match(key)
            if expert_match is not None:
                layer_idx = int(expert_match.group(2))
                expert_idx = int(expert_match.group(3))
                expert_entries[(layer_idx, expert_idx, expert_match.group(4))] = (shard_name, key)
                continue
            if key.endswith(cls._SCALE_SUFFIX):
                continue
            base_entries[key] = shard_name

        num_experts = getattr(model_config, "n_routed_experts", None) or getattr(
            model_config, "num_local_experts", None
        )
        expert_layers = sorted({layer_idx for layer_idx, _, _ in expert_entries})
        mlp_layer_types = getattr(model_config, "mlp_layer_types", None)
        if mlp_layer_types is not None:
            expected_expert_layers = [
                layer_idx
                for layer_idx, layer_type in enumerate(mlp_layer_types[:num_hidden_layers])
                if layer_type == "sparse"
            ]
            missing_expert_layers = sorted(set(expected_expert_layers).difference(expert_layers))
            if missing_expert_layers:
                raise ValueError(
                    f"GLM-5.3 is missing expert weights for active sparse layers: {missing_expert_layers[:3]}."
                )
        if num_experts is None:
            num_experts = len({expert_idx for _, expert_idx, _ in expert_entries})
        expected_expert_entries = {
            (layer_idx, expert_idx, kind)
            for layer_idx in expert_layers
            for expert_idx in range(num_experts)
            for kind in ("gate_proj", "up_proj", "down_proj")
        }
        missing_expert_entries = expected_expert_entries.difference(expert_entries)
        if missing_expert_entries:
            sample = ", ".join(
                f"layer {layer_idx} expert {expert_idx} {kind}"
                for layer_idx, expert_idx, kind in sorted(missing_expert_entries)[:3]
            )
            raise ValueError(f"GLM-5.3 is missing required expert weights: {sample}.")

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)
        new_weight_map: dict[str, str] = {}
        entries_by_shard: dict[str, list[str]] = {}
        for key, shard_name in base_entries.items():
            entries_by_shard.setdefault(shard_name, []).append(key)

        from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            derive_glm_mla_parameters,
            split_glm_moe_dsa_q_b_proj,
        )

        def _load_dequantized(key: str, checkpoint=None) -> torch.Tensor:
            shard_name = weight_map[key]
            if checkpoint is not None:
                tensor = checkpoint.get_tensor(key)
            else:
                with safe_open(str(src / shard_name), framework="pt") as source_checkpoint:
                    tensor = source_checkpoint.get_tensor(key)
            scale_key = f"{key}{cls._SCALE_SUFFIX}"
            if scale_key in weight_map:
                scale_shard = weight_map[scale_key]
                if checkpoint is not None and scale_shard == shard_name:
                    scale = checkpoint.get_tensor(scale_key)
                else:
                    with safe_open(str(src / scale_shard), framework="pt") as scale_checkpoint:
                        scale = scale_checkpoint.get_tensor(scale_key)
                return cls._dequantize_fp8_weight(tensor, scale, block_size).to(target_dtype)
            return tensor.to(target_dtype) if tensor.is_floating_point() else tensor

        for shard_index, (shard_name, keys) in enumerate(sorted(entries_by_shard.items())):
            tensors: dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as checkpoint:
                for key in keys:
                    tensor = _load_dequantized(key, checkpoint)
                    tensors[key] = tensor
                    if key.endswith(cls._Q_B_PROJ_SUFFIX):
                        prefix = key[: -len(".q_b_proj.weight")]
                        kv_b_key = f"{prefix}.kv_b_proj.weight"
                        derived_config_fields = ("q_lora_rank", "kv_lora_rank", "v_head_dim")
                        if kv_b_key in weight_map and all(
                            hasattr(model_config, name) for name in derived_config_fields
                        ):
                            derived = derive_glm_mla_parameters(
                                tensor,
                                _load_dequantized(kv_b_key),
                                num_heads=model_config.num_attention_heads,
                                q_lora_rank=model_config.q_lora_rank,
                                kv_lora_rank=model_config.kv_lora_rank,
                                qk_nope_head_dim=model_config.qk_nope_head_dim,
                                qk_rope_head_dim=model_config.qk_rope_head_dim,
                                v_head_dim=model_config.v_head_dim,
                            )
                            tensors.update({f"{prefix}.{name}": value for name, value in derived.items()})
                        else:
                            q_up, q_rope = split_glm_moe_dsa_q_b_proj(
                                tensor,
                                model_config.num_attention_heads,
                                model_config.qk_nope_head_dim,
                                model_config.qk_rope_head_dim,
                            )
                            tensors[f"{prefix}.q_up"] = q_up
                            tensors[f"{prefix}.q_rope"] = q_rope

            out_name = f"model_{shard_index:04d}.safetensors"
            atomic_save(tensors, out / out_name)
            new_weight_map.update({key: out_name for key in tensors})

        for layer_idx in expert_layers:
            layer_start = time.perf_counter()
            logger.info("Preparing GLM expert layer %d/%d", layer_idx, num_hidden_layers - 1)
            for kind_index, kind in enumerate(("gate_proj", "up_proj", "down_proj")):
                entries_for_kind: dict[str, list[tuple[int, str]]] = {}
                for expert_idx in range(num_experts):
                    shard_name, expert_key = expert_entries[(layer_idx, expert_idx, kind)]
                    entries_for_kind.setdefault(shard_name, []).append((expert_idx, expert_key))

                first_key = expert_entries[(layer_idx, 0, kind)][1]
                with safe_open(str(src / weight_map[first_key]), framework="pt") as checkpoint:
                    first_shape = checkpoint.get_slice(first_key).get_shape()
                stacked = torch.empty((num_experts, first_shape[1], first_shape[0]), dtype=target_dtype)
                for shard_name, shard_entries in sorted(entries_for_kind.items()):
                    with safe_open(str(src / shard_name), framework="pt") as checkpoint:
                        for expert_idx, expert_key in shard_entries:
                            stacked[expert_idx] = _load_dequantized(expert_key, checkpoint).transpose(0, 1)

                canonical_kind = kind.removesuffix("_proj")
                canonical_key = f"model.layers.{layer_idx}.mlp.moe_weights.{canonical_kind}"
                out_name = f"model_glm_experts_layer_{layer_idx:05d}_{kind_index:02d}.safetensors"
                expert_parallel_layout = kwargs.get("expert_parallel_layout")
                if expert_parallel_layout is not None:
                    from QEfficient.transformers.moe.weights import pack_expert_parallel_tensor

                    num_pipeline_stages = int(expert_parallel_layout["num_pipeline_stages"])
                    num_parallelized_experts = int(expert_parallel_layout["num_parallelized_experts"])
                    if num_experts != num_pipeline_stages * num_parallelized_experts:
                        raise ValueError(
                            "GLM EP checkpoint layout is inconsistent with the checkpoint: "
                            f"{num_experts} experts != {num_pipeline_stages} stages * "
                            f"{num_parallelized_experts} parallel experts"
                        )
                    stacked = pack_expert_parallel_tensor(
                        stacked,
                        num_pipeline_stages=num_pipeline_stages,
                        num_parallelized_experts=num_parallelized_experts,
                    )
                atomic_save({canonical_key: stacked}, out / out_name)
                new_weight_map[canonical_key] = out_name
                del stacked
            logger.info(
                "Prepared GLM expert layer %d/%d in %.2fs",
                layer_idx,
                num_hidden_layers - 1,
                time.perf_counter() - layer_start,
            )

        write_index(out, new_weight_map)
        sentinel.touch()
        logger.info("GlmMoeDsaReducedCheckpointTransform: done -> %s", out)
        return True


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

    def stack(self, target_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Return stacked expert tensors in the derived QEff checkpoint layout."""
        # Output in the canonical layout that OptimizedMoETransform creates so
        # promote_initializers_and_build_spec finds an exact checkpoint key match.
        #   _gate [E, I, H] -> transpose(1,2) -> moe_weights.gate [E, H, I]
        #   _up   [E, I, H] -> transpose(1,2) -> moe_weights.up   [E, H, I]
        #   _down [E, H, I] -> transpose(1,2) -> moe_weights.down [E, I, H]
        moe_prefix = _moe_weights_prefix_from_experts_prefix(self.prefix)
        gate = self._gate.to(target_dtype).transpose(1, 2).contiguous()
        up = self._up.to(target_dtype).transpose(1, 2).contiguous()
        down = self._down.to(target_dtype).transpose(1, 2).contiguous()
        return {
            f"{moe_prefix}.gate": gate,  # [E, H, I]
            f"{moe_prefix}.up": up,  # [E, H, I]
            f"{moe_prefix}.down": down,  # [E, I, H]
        }


# ---------------------------------------------------------------------------
# Transform 2: MoE expert stacking + dtype conversion — single pass
# ---------------------------------------------------------------------------


class MoEExpertStackingCheckpointTransform(BaseCheckpointTransform):
    """Stack per-expert checkpoint keys into batched tensors AND convert dtype.

    Detects the HuggingFace per-expert layout::

        *.experts.{E}.gate_proj.weight  [I, H]  x  num_experts
        *.experts.{E}.up_proj.weight    [I, H]  x  num_experts
        *.experts.{E}.down_proj.weight  [H, I]  x  num_experts

    and produces::

        *.moe_weights.gate [E, H, I]   (gate weights, transposed)
        *.moe_weights.up   [E, H, I]   (up weights, transposed)
        *.moe_weights.down [E, I, H]   (down weights, transposed)

    matching the derived parameter layout that OptimizedMoETransform
    creates, so promote_initializers_and_build_spec finds an exact key match.
    Non-expert keys receive dtype conversion in the same pass.

    Parallelism:

    - Phase 1 (scan):  one thread per shard, reads keys only (I/O bound, cheap).
    - Phase 2 (stack): one thread per layer, loads and stacks its experts.
    - Phase 3 (base):  one thread per shard, converts non-expert keys.

    Phases 2 and 3 run concurrently once phase 1 completes.
    """

    EXPERT_RE = re.compile(
        r"^(.+\.layers\.(\d+)\..+?\.experts)\.(\d+)\."
        r"(gate_proj|up_proj|down_proj|linear|linear_v|linear_1|w1|w2|w3)\.weight$"
    )

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when the checkpoint uses per-expert MoE tensor keys."""
        return any(cls.EXPERT_RE.match(k) for k in weight_map)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        max_workers_scan: Optional[int] = None,
        max_workers_layers: Optional[int] = None,
        max_workers_base: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Stack per-expert MoE tensors and convert remaining tensors to ``target_dtype``."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("MoEExpertStackingCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)

        weight_map = read_weight_map(src)
        shard_names = sorted(set(weight_map.values()))

        # Phase 1: parallel key scan - no tensor data loaded.
        #
        # expert_entries[(layer_idx, expert_idx, kind)] = (shard_name, orig_key)
        # layer_prefix[layer_idx]                        = prefix up to .experts
        # base_entries[orig_key]                         = shard_name
        expert_entries: Dict[Tuple[int, int, str], Tuple[str, str]] = {}
        layer_prefix: Dict[int, str] = {}
        base_entries: Dict[str, str] = {}

        def _scan(shard_name: str) -> Tuple[Dict, Dict, Dict]:
            loc_e: Dict[Tuple[int, int, str], Tuple[str, str]] = {}
            loc_p: Dict[int, str] = {}
            loc_b: Dict[str, str] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in f.keys():
                    m = cls.EXPERT_RE.match(key)
                    if m:
                        loc_e[(int(m.group(2)), int(m.group(3)), m.group(4))] = (shard_name, key)
                        loc_p[int(m.group(2))] = m.group(1)
                    else:
                        loc_b[key] = shard_name
            return loc_e, loc_p, loc_b

        # Phase 1: I/O-bound — cap at 4× logical CPUs, no point exceeding shard count.
        # Hard cap at 256: beyond that, OS scheduling overhead outweighs I/O gains.
        n_workers_scan = (
            max_workers_scan if max_workers_scan is not None else min(len(shard_names), cpu_count() * 4, 256)
        )
        logger.info(
            f"MoEExpertStackingCheckpointTransform: scanning {len(shard_names)} shards "
            f"(workers={n_workers_scan}, cpus={cpu_count()}, ram_avail={available_ram_gb():.1f} GB)..."
        )
        with ThreadPoolExecutor(max_workers=n_workers_scan) as ex:
            for loc_e, loc_p, loc_b in ex.map(_scan, shard_names):
                expert_entries.update(loc_e)
                layer_prefix.update(loc_p)
                base_entries.update(loc_b)

        experts_per_layer: Dict[int, set] = {}
        for layer_idx, expert_idx, _ in expert_entries:
            experts_per_layer.setdefault(layer_idx, set()).add(expert_idx)
        layer_indices = sorted(experts_per_layer.keys())
        sample_n = len(next(iter(experts_per_layer.values()))) if experts_per_layer else 0
        logger.info(f"  {len(layer_indices)} MoE layers × {sample_n} experts each.")

        new_weight_map: Dict[str, str] = {}

        # Phase 2: parallel layer stacking.
        # Each layer thread loads its own experts (grouped by shard to open each
        # shard at most once per layer), stacks, converts dtype, writes atomically.
        def _stack_layer(layer_idx: int) -> Tuple[str, List[str]]:
            num_exp = len(experts_per_layer[layer_idx])
            stacker = _LayerStacker(layer_prefix[layer_idx], num_exp)

            # Detect which kind names are present (qwen3-moe: gate_proj/up_proj/down_proj;
            # grok-1: linear/linear_v/linear_1).
            kinds_present = {k for (li, _, k) in expert_entries if li == layer_idx}

            by_shard: Dict[str, List[Tuple[int, str, str]]] = {}
            for exp_idx in range(num_exp):
                for kind in kinds_present:
                    shard_name, orig_key = expert_entries[(layer_idx, exp_idx, kind)]
                    by_shard.setdefault(shard_name, []).append((exp_idx, kind, orig_key))

            for shard_name, entries in by_shard.items():
                with safe_open(str(src / shard_name), framework="pt") as f:
                    for exp_idx, kind, orig_key in entries:
                        stacker.add(exp_idx, kind, f.get_tensor(orig_key))

            stacked = stacker.stack(target_dtype)
            out_name = f"experts-layer-{layer_idx:05d}.safetensors"
            atomic_save(stacked, out / out_name)
            return out_name, list(stacked.keys())

        # Phase 2: memory-bound — each layer holds all E×3 expert tensors + the
        # stacked output in RAM simultaneously. Derive the worker count from
        # available RAM so we never OOM: keep 20% headroom, compute RAM per layer
        # from the actual tensor shapes in the checkpoint.
        if max_workers_layers is not None:
            n_workers_layers = max_workers_layers
            layer_gb = 0.0
        elif layer_indices:
            sample_layer = layer_indices[0]
            layer_gb = _estimate_layer_stack_gb(
                expert_entries, sample_layer, len(experts_per_layer[sample_layer]), src, target_dtype
            )
            available_gb = available_ram_gb()
            usable_gb = available_gb * 0.8
            n_workers_layers = max(1, min(len(layer_indices), int(usable_gb / layer_gb)))
        else:
            n_workers_layers = 1
            layer_gb = 0.0

        logger.info(
            f"  Stacking {len(layer_indices)} layers → {target_dtype} | "
            f"workers={n_workers_layers} (~{layer_gb:.2f} GB/layer, "
            f"{available_ram_gb():.1f} GB available)..."
        )
        with ThreadPoolExecutor(max_workers=n_workers_layers) as ex:
            futures = {ex.submit(_stack_layer, li): li for li in layer_indices}
            for fut in as_completed(futures):
                li = futures[fut]
                out_name, out_keys = fut.result()
                for key in out_keys:
                    new_weight_map[key] = out_name
                logger.info(f"    layer {li:5d} → {out_name}")

        # Phase 3: parallel base shard conversion.
        by_shard_base: Dict[str, List[str]] = {}
        for key, shard_name in base_entries.items():
            by_shard_base.setdefault(shard_name, []).append(key)

        base_shard_list = sorted(by_shard_base)
        new_base_name_for = {shard: f"base-{idx:04d}.safetensors" for idx, shard in enumerate(base_shard_list)}

        def _convert_base(shard_name: str, keys: List[str]) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in keys:
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            atomic_save(tensors, out / new_base_name_for[shard_name])

        # Phase 3: mixed I/O + memory — one thread per shard, capped at CPU count.
        n_workers_base = (
            max_workers_base if max_workers_base is not None else max(1, min(len(base_shard_list), cpu_count()))
        )
        logger.info(f"  Converting {len(base_shard_list)} base shards → {target_dtype} | workers={n_workers_base}...")
        if base_shard_list:
            with ThreadPoolExecutor(max_workers=n_workers_base) as ex:
                futures_base = [ex.submit(_convert_base, s, keys) for s, keys in by_shard_base.items()]
                for fut in as_completed(futures_base):
                    fut.result()

        for key, shard_name in base_entries.items():
            new_weight_map[key] = new_base_name_for[shard_name]

        write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"MoEExpertStackingCheckpointTransform: done → {out}")
        return True


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

    matching the derived parameter layout that OptimizedMoETransform
    creates, so promote_initializers_and_build_spec finds an exact key match.
    Non-expert keys receive dtype conversion in the same pass.

    Parallelism mirrors MoEExpertStackingCheckpointTransform:
    - Phase 1 (scan):    one thread per shard — collect expert tensor locations.
    - Phase 2 (dequant): one thread per layer — dequant, split, write.
    - Phase 3 (base):    one thread per shard — dtype-convert non-expert keys.
    """

    _BLOCKS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_blocks$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when the checkpoint contains GPT-OSS MXFP4 expert blocks."""
        return any(cls._BLOCKS_RE.match(k) for k in weight_map)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        max_workers_scan: Optional[int] = None,
        max_workers_layers: Optional[int] = None,
        max_workers_base: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Dequantize GPT-OSS MXFP4 experts and split fused expert projections."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("GptOssMxfp4ExpertDequantSplitCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)

        weight_map = read_weight_map(src)
        shard_names = sorted(set(weight_map.values()))

        # Phase 1: scan - collect expert tensor locations.
        # expert_locs[(layer_idx, kind)] = (blocks_shard, blocks_key, scales_shard, scales_key)
        # bias_locs[(layer_idx, kind)]   = (shard, key)   for gate_up_proj_bias / down_proj_bias
        # layer_prefix[layer_idx]        = prefix up to .experts
        # base_entries[orig_key]         = shard_name
        _SCALES_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_scales$")
        _BIAS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_bias$")

        expert_locs: Dict[Tuple[int, str], Dict] = {}  # {(layer, kind): {blocks/scales: (shard, key)}}
        bias_locs: Dict[Tuple[int, str], Tuple[str, str]] = {}
        layer_prefix: Dict[int, str] = {}
        base_entries: Dict[str, str] = {}

        def _scan(shard_name: str):
            loc_e: Dict[Tuple[int, str], Dict] = {}
            loc_b: Dict[Tuple[int, str], Tuple[str, str]] = {}
            loc_p: Dict[int, str] = {}
            loc_base: Dict[str, str] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in f.keys():
                    m = cls._BLOCKS_RE.match(key)
                    if m:
                        li, kind = int(m.group(2)), m.group(3)
                        loc_e.setdefault((li, kind), {})["blocks"] = (shard_name, key)
                        loc_p[li] = m.group(1)
                        continue
                    m = _SCALES_RE.match(key)
                    if m:
                        li, kind = int(m.group(2)), m.group(3)
                        loc_e.setdefault((li, kind), {})["scales"] = (shard_name, key)
                        loc_p[li] = m.group(1)
                        continue
                    m = _BIAS_RE.match(key)
                    if m:
                        li, kind = int(m.group(2)), m.group(3)
                        loc_b[(li, kind)] = (shard_name, key)
                        loc_p[li] = m.group(1)
                        continue
                    loc_base[key] = shard_name
            return loc_e, loc_b, loc_p, loc_base

        n_scan = max_workers_scan if max_workers_scan is not None else min(len(shard_names), cpu_count() * 4, 256)
        logger.info(
            f"GptOssMxfp4ExpertDequantSplitCheckpointTransform: scanning {len(shard_names)} shards "
            f"(workers={n_scan})..."
        )
        with ThreadPoolExecutor(max_workers=n_scan) as ex:
            for loc_e, loc_b, loc_p, loc_base in ex.map(_scan, shard_names):
                for k, v in loc_e.items():
                    expert_locs.setdefault(k, {}).update(v)
                bias_locs.update(loc_b)
                layer_prefix.update(loc_p)
                base_entries.update(loc_base)

        layer_indices = sorted({li for li, _ in expert_locs})
        logger.info(f"  Found {len(layer_indices)} MoE layers.")

        new_weight_map: Dict[str, str] = {}

        # Phase 2: per-layer dequant + split.
        def _process_layer(layer_idx: int) -> Tuple[str, List[str]]:
            prefix = layer_prefix[layer_idx]
            moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
            tensors: Dict[str, torch.Tensor] = {}

            def _load(shard: str, key: str) -> torch.Tensor:
                with safe_open(str(src / shard), framework="pt") as f:
                    return f.get_tensor(key)

            # gate_up_proj: dequant → [E, H, 2*I], then split interleaved (gate=even cols, up=odd cols)
            # HF _apply_gate uses gate_up[..., ::2] for gate and gate_up[..., 1::2] for up,
            # so columns are interleaved: col 0=gate0, col 1=up0, col 2=gate1, col 3=up1, ...
            gu_blocks_shard, gu_blocks_key = expert_locs[(layer_idx, "gate_up_proj")]["blocks"]
            gu_scales_shard, gu_scales_key = expert_locs[(layer_idx, "gate_up_proj")]["scales"]
            gu_blocks = _load(gu_blocks_shard, gu_blocks_key)
            gu_scales = _load(gu_scales_shard, gu_scales_key)
            gate_up = convert_moe_packed_tensors(gu_blocks, gu_scales, dtype=target_dtype)
            tensors[f"{moe_prefix}.gate"] = gate_up[..., 0::2].contiguous()
            tensors[f"{moe_prefix}.up"] = gate_up[..., 1::2].contiguous()

            # gate_up_proj_bias: split [E, 2*I] → [E, I] + [E, I] (same interleaved convention)
            if (layer_idx, "gate_up_proj") in bias_locs:
                bias_shard, bias_key = bias_locs[(layer_idx, "gate_up_proj")]
                gu_bias = _load(bias_shard, bias_key).to(target_dtype)
                tensors[f"{moe_prefix}.gate_bias"] = gu_bias[..., 0::2].contiguous()
                tensors[f"{moe_prefix}.up_bias"] = gu_bias[..., 1::2].contiguous()

            # down_proj: dequant -> [E, I, H]
            dp_blocks_shard, dp_blocks_key = expert_locs[(layer_idx, "down_proj")]["blocks"]
            dp_scales_shard, dp_scales_key = expert_locs[(layer_idx, "down_proj")]["scales"]
            dp_blocks = _load(dp_blocks_shard, dp_blocks_key)
            dp_scales = _load(dp_scales_shard, dp_scales_key)
            tensors[f"{moe_prefix}.down"] = convert_moe_packed_tensors(dp_blocks, dp_scales, dtype=target_dtype)

            # down_proj_bias: pass through with dtype conversion
            if (layer_idx, "down_proj") in bias_locs:
                dp_bias_shard, dp_bias_key = bias_locs[(layer_idx, "down_proj")]
                tensors[f"{moe_prefix}.down_bias"] = _load(dp_bias_shard, dp_bias_key).to(target_dtype)

            out_name = f"experts-layer-{layer_idx:05d}.safetensors"
            atomic_save(tensors, out / out_name)
            return out_name, list(tensors.keys())

        n_layers = (
            max_workers_layers if max_workers_layers is not None else max(1, min(len(layer_indices), cpu_count()))
        )
        logger.info(f"  Dequantizing {len(layer_indices)} layers | workers={n_layers}...")
        with ThreadPoolExecutor(max_workers=n_layers) as ex:
            futures = {ex.submit(_process_layer, li): li for li in layer_indices}
            for fut in as_completed(futures):
                li = futures[fut]
                out_name, out_keys = fut.result()
                for key in out_keys:
                    new_weight_map[key] = out_name
                logger.info(f"    layer {li:5d} → {out_name}")

        # Phase 3: base shard dtype conversion.
        by_shard_base: Dict[str, List[str]] = {}
        for key, shard_name in base_entries.items():
            by_shard_base.setdefault(shard_name, []).append(key)

        base_shard_list = sorted(by_shard_base)
        new_base_name_for = {shard: f"base-{idx:04d}.safetensors" for idx, shard in enumerate(base_shard_list)}

        def _convert_base(shard_name: str, keys: List[str]) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in keys:
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            atomic_save(tensors, out / new_base_name_for[shard_name])

        n_base = max_workers_base if max_workers_base is not None else max(1, min(len(base_shard_list), cpu_count()))
        logger.info(f"  Converting {len(base_shard_list)} base shards | workers={n_base}...")
        if base_shard_list:
            with ThreadPoolExecutor(max_workers=n_base) as ex:
                futures_base = [ex.submit(_convert_base, s, keys) for s, keys in by_shard_base.items()]
                for fut in as_completed(futures_base):
                    fut.result()

        for key, shard_name in base_entries.items():
            new_weight_map[key] = new_base_name_for[shard_name]

        write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"GptOssMxfp4ExpertDequantSplitCheckpointTransform: done → {out}")
        return True


# ---------------------------------------------------------------------------
# Transform 4: split already-stacked fused MoE experts (Mixtral v5+ layout)
# ---------------------------------------------------------------------------


class MoEFusedExpertSplitCheckpointTransform(BaseCheckpointTransform):
    """Split already-stacked MoE expert weights into the derived layout.

    Some MoE checkpoints (e.g. Mixtral transformers >= 5.x) store experts
    as per-layer fused tensors rather than per-expert individual weights:

        *.experts.gate_up_proj  [E, 2*I, H]   (gate and up concatenated)
        *.experts.down_proj     [E, H, I]

    The QEff model wrappers create derived parameters that the ONNX
    initializer names refer to:

        *.moe_weights.gate [E, H, I]
        *.moe_weights.up   [E, H, I]
        *.moe_weights.down [E, I, H]

    is_applicable returns True only when the fused format is detected.
    Old-format checkpoints with per-expert keys (e.g. experts.0.gate_proj.weight)
    are handled by MoEExpertStackingCheckpointTransform instead.
    Also handles dtype conversion in the same pass.
    """

    _FUSED_GATE_UP_RE = re.compile(r"^(.+\.experts)\.gate_up_proj$")
    _FUSED_DOWN_RE = re.compile(r"^(.+\.experts)\.down_proj$")
    _FUSED_GATE_UP_BIAS_RE = re.compile(r"^(.+\.experts)\.gate_up_proj_bias$")
    _FUSED_DOWN_BIAS_RE = re.compile(r"^(.+\.experts)\.down_proj_bias$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when already-stacked fused MoE expert tensors are present."""
        return any(cls._FUSED_GATE_UP_RE.match(k) for k in weight_map)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> bool:
        """Split fused MoE expert tensors and write a prepared checkpoint."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("MoEFusedExpertSplitCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        index_path = src / "model.safetensors.index.json"
        if index_path.exists():
            weight_map: Dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        else:
            # TODO(wf): Reuse read_weight_map() here instead of duplicating safetensors map resolution
            # It does not make sense to rebuild weight_map from shards, error out if its not present.
            shards = sorted(src.glob("*.safetensors"))
            if not shards:
                return False
            weight_map = {}
            for shard in shards:
                with safe_open(str(shard), framework="pt") as f:
                    for k in f.keys():
                        weight_map[k] = shard.name

        if not cls.is_applicable(weight_map):
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)

        new_weight_map: Dict[str, str] = {}
        source_shapes: Dict[str, Tuple[int, ...]] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_src = src / shard_name
            if not shard_src.exists():
                continue
            with safe_open(str(shard_src), framework="pt") as f:
                for key in f.keys():
                    source_shapes[key] = tuple(f.get_slice(key).get_shape())

        for shard_name in sorted(set(weight_map.values())):
            shard_src = src / shard_name
            if not shard_src.exists():
                continue

            out_tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(shard_src), framework="pt") as f:
                for key in f.keys():
                    raw_tensor = f.get_tensor(key)
                    tensor = raw_tensor.to(target_dtype) if raw_tensor.is_floating_point() else raw_tensor
                    gate_up_m = cls._FUSED_GATE_UP_RE.match(key)
                    down_m = cls._FUSED_DOWN_RE.match(key)
                    gate_up_bias_m = cls._FUSED_GATE_UP_BIAS_RE.match(key)
                    down_bias_m = cls._FUSED_DOWN_BIAS_RE.match(key)

                    if gate_up_m:
                        prefix = gate_up_m.group(1)
                        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
                        down_shape = source_shapes.get(f"{prefix}.down_proj")
                        preferred_split_dim = 2 if f"{prefix}.gate_up_proj_bias" in weight_map else None
                        split_dim = _infer_fused_gate_up_split_dim(
                            tuple(tensor.shape),
                            down_shape,
                            preferred_split_dim=preferred_split_dim,
                        )
                        gate, up = _split_fused_gate_up_to_canonical(
                            tensor,
                            down_shape,
                            interleaved=split_dim == 2 and preferred_split_dim == 2,
                            preferred_split_dim=split_dim,
                        )
                        out_tensors[f"{moe_prefix}.gate"] = gate
                        out_tensors[f"{moe_prefix}.up"] = up
                        new_weight_map[f"{moe_prefix}.gate"] = shard_name
                        new_weight_map[f"{moe_prefix}.up"] = shard_name
                        # Keep original for completeness
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name
                    elif down_m:
                        prefix = down_m.group(1)
                        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
                        gate_up_shape = source_shapes.get(f"{prefix}.gate_up_proj")
                        preferred_split_dim = 2 if f"{prefix}.gate_up_proj_bias" in weight_map else None
                        split_dim = _infer_fused_gate_up_split_dim(
                            gate_up_shape or (),
                            tuple(tensor.shape),
                            preferred_split_dim=preferred_split_dim,
                        )
                        out_tensors[f"{moe_prefix}.down"] = _down_to_canonical(
                            tensor,
                            gate_up_shape,
                            preferred_split_dim=split_dim,
                        )
                        new_weight_map[f"{moe_prefix}.down"] = shard_name
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name
                    elif gate_up_bias_m:
                        prefix = gate_up_bias_m.group(1)
                        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
                        split_dim = _infer_fused_gate_up_split_dim(
                            source_shapes.get(f"{prefix}.gate_up_proj", ()),
                            source_shapes.get(f"{prefix}.down_proj"),
                            preferred_split_dim=2,
                        )
                        gate_bias, up_bias = _split_gate_up_bias(tensor, interleaved=split_dim == 2)
                        out_tensors[f"{moe_prefix}.gate_bias"] = gate_bias
                        out_tensors[f"{moe_prefix}.up_bias"] = up_bias
                        new_weight_map[f"{moe_prefix}.gate_bias"] = shard_name
                        new_weight_map[f"{moe_prefix}.up_bias"] = shard_name
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name
                    elif down_bias_m:
                        prefix = down_bias_m.group(1)
                        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
                        out_tensors[f"{moe_prefix}.down_bias"] = tensor.clone()
                        new_weight_map[f"{moe_prefix}.down_bias"] = shard_name
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name
                    else:
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name

            save_file({k: v.contiguous() for k, v in out_tensors.items()}, str(out / shard_name))

        write_index(out, new_weight_map)
        sentinel.touch()
        return True


# ---------------------------------------------------------------------------
# Transform 5: split GraniteMoE fused parallel experts
# ---------------------------------------------------------------------------


class GraniteMoeFusedExpertSplitCheckpointTransform(BaseCheckpointTransform):
    """Split GraniteMoE fused parallel expert tensors into MoEWeights keys."""

    _FUSED_GATE_UP_RE = re.compile(r"^(.+)\.input_linear\.weight$")
    _FUSED_DOWN_RE = re.compile(r"^(.+)\.output_linear\.weight$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when GraniteMoE fused expert tensors are present."""
        return any(cls._FUSED_GATE_UP_RE.match(k) for k in weight_map)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> bool:
        """Split GraniteMoE fused expert tensors and write a prepared checkpoint."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("GraniteMoeFusedExpertSplitCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        weight_map = read_weight_map(src)
        if not cls.is_applicable(weight_map):
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)

        new_weight_map: Dict[str, str] = {}
        source_shapes: Dict[str, Tuple[int, ...]] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_src = src / shard_name
            if not shard_src.exists():
                continue
            with safe_open(str(shard_src), framework="pt") as f:
                for key in f.keys():
                    source_shapes[key] = tuple(f.get_slice(key).get_shape())

        for shard_name in sorted(set(weight_map.values())):
            shard_src = src / shard_name
            if not shard_src.exists():
                continue

            out_tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(shard_src), framework="pt") as f:
                for key in f.keys():
                    raw_tensor = f.get_tensor(key)
                    tensor = raw_tensor.to(target_dtype) if raw_tensor.is_floating_point() else raw_tensor
                    gate_up_m = cls._FUSED_GATE_UP_RE.match(key)
                    down_m = cls._FUSED_DOWN_RE.match(key)

                    if gate_up_m:
                        prefix = gate_up_m.group(1)
                        moe_prefix = f"{prefix}.moe_weights"
                        down_shape = source_shapes.get(f"{prefix}.output_linear.weight")
                        gate, up = _split_fused_gate_up_to_canonical(tensor, down_shape, preferred_split_dim=1)
                        out_tensors[f"{moe_prefix}.gate"] = gate
                        out_tensors[f"{moe_prefix}.up"] = up
                        new_weight_map[f"{moe_prefix}.gate"] = shard_name
                        new_weight_map[f"{moe_prefix}.up"] = shard_name
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name
                    elif down_m:
                        prefix = down_m.group(1)
                        moe_prefix = f"{prefix}.moe_weights"
                        gate_up_shape = source_shapes.get(f"{prefix}.input_linear.weight")
                        out_tensors[f"{moe_prefix}.down"] = _down_to_canonical(
                            tensor,
                            gate_up_shape,
                            preferred_split_dim=1,
                        )
                        new_weight_map[f"{moe_prefix}.down"] = shard_name
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name
                    else:
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name

            save_file({k: v.contiguous() for k, v in out_tensors.items()}, str(out / shard_name))

        write_index(out, new_weight_map)
        sentinel.touch()
        return True
