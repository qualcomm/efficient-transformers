# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Base mechanism for checkpoint file transforms.

Unlike PytorchTransform (base/pytorch_transforms.py), which mutates a live
nn.Module already in memory, a checkpoint transform operates on files on disk
before any model object exists - rewriting a source checkpoint directory into
a prepared one (dtype conversion, MoE expert stacking, etc.) for weight-free
ONNX export. Concrete transforms live alongside the feature that needs them,
e.g. QEfficient/exporter/weight_free/checkpoint_transforms.py.
"""

import json
import os
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import psutil
import torch

from QEfficient.utils.checkpoint_utils import (
    copy_checkpoint_aux_files,
    read_weight_map,
    write_index,
)
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


@dataclass(frozen=True)
class TensorRef:
    """Reference to a tensor at a specific planning stage."""

    key: str
    stage: str = "raw"


@dataclass(frozen=True)
class TaskParams:
    """Stable transform parameters used for planning and cache identity."""

    transform_id: str
    values: tuple[tuple[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return dict(self.values)


StageTensorGetter = Callable[[TensorRef], torch.Tensor]
StageRunner = Callable[[StageTensorGetter, torch.dtype], dict[TensorRef, torch.Tensor]]


@dataclass
class CheckpointStage:
    """One composable in-memory transformation between staged tensor refs."""

    stage_id: str
    input_refs: tuple[TensorRef, ...]
    output_refs: tuple[TensorRef, ...]
    params: TaskParams
    runner: StageRunner = field(repr=False, compare=False)
    labels: tuple[str, ...] = ()

    def run(self, get_tensor: StageTensorGetter, target_dtype: torch.dtype) -> dict[TensorRef, torch.Tensor]:
        return self.runner(get_tensor, target_dtype)

    def fingerprint(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "inputs": [asdict(ref) for ref in self.input_refs],
            "outputs": [asdict(ref) for ref in self.output_refs],
            "labels": list(self.labels),
            "params": {
                "transform_id": self.params.transform_id,
                "values": [[key, _stable_json_value(value)] for key, value in self.params.values],
            },
        }


@dataclass
class CheckpointTask:
    """One independently schedulable checkpoint transformation."""

    task_id: str
    input_refs: tuple[TensorRef, ...]
    output_refs: tuple[TensorRef, ...]
    source_files: tuple[str, ...]
    output_file: str
    estimated_peak_bytes: int
    params: TaskParams
    runner: Callable[[Path, Path, torch.dtype], dict[str, str]] = field(repr=False, compare=False)
    stages: tuple[CheckpointStage, ...] = ()

    def run(self, src: Path, out: Path, target_dtype: torch.dtype) -> dict[str, str]:
        return self.runner(src, out, target_dtype)

    def fingerprint(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "inputs": [asdict(ref) for ref in self.input_refs],
            "outputs": [asdict(ref) for ref in self.output_refs],
            "source_files": list(self.source_files),
            "output_file": self.output_file,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "params": {
                "transform_id": self.params.transform_id,
                "values": [[key, _stable_json_value(value)] for key, value in self.params.values],
            },
            "stages": [stage.fingerprint() for stage in self.stages],
        }


def _stable_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value) if isinstance(value, torch.dtype) else value


@dataclass
class CheckpointTaskPlan:
    """A staged tensor group that will be fused into one executable task."""

    task_id: str
    input_refs: tuple[TensorRef, ...]
    source_files: tuple[str, ...]
    output_file: str
    estimated_peak_bytes: int
    params: TaskParams
    stages: list[CheckpointStage] = field(default_factory=list)

    @property
    def current_refs(self) -> tuple[TensorRef, ...]:
        return self.stages[-1].output_refs if self.stages else self.input_refs

    def append_stage(self, stage: CheckpointStage) -> None:
        if set(stage.input_refs) != set(self.current_refs):
            raise ValueError(f"Checkpoint stage {stage.stage_id} inputs do not match task {self.task_id} current refs.")
        self.stages.append(stage)


@dataclass
class CheckpointPlanningContext:
    """Mutable planning state shared by independent checkpoint transforms."""

    weight_map: dict[str, str]
    config: object
    hash_params: dict
    target_dtype: torch.dtype
    source_dir: Path
    task_plans: list[CheckpointTaskPlan] = field(default_factory=list)
    direct_tasks: list[CheckpointTask] = field(default_factory=list)
    claimed_raw_refs: set[TensorRef] = field(default_factory=set)
    active_layout_ids: list[str] = field(default_factory=list)

    def add_task_plan(self, task_plan: CheckpointTaskPlan, *, layout_transform_id: Optional[str] = None) -> None:
        raw_refs = {ref for ref in task_plan.input_refs if ref.stage == "raw"}
        unknown_refs = {ref for ref in raw_refs if ref.key not in self.weight_map}
        if unknown_refs:
            raise ValueError(
                f"Checkpoint task {task_plan.task_id} references unknown source keys: "
                f"{sorted(ref.key for ref in unknown_refs)}"
            )
        overlap = raw_refs & self.claimed_raw_refs
        if overlap:
            raise ValueError(
                f"Checkpoint task {task_plan.task_id} reclaims raw tensor keys: {sorted(ref.key for ref in overlap)}"
            )
        self.claimed_raw_refs.update(raw_refs)
        self.task_plans.append(task_plan)
        if layout_transform_id and layout_transform_id not in self.active_layout_ids:
            self.active_layout_ids.append(layout_transform_id)

    def add_direct_task(self, task: CheckpointTask) -> None:
        raw_refs = {ref for ref in task.input_refs if ref.stage == "raw"}
        overlap = raw_refs & self.claimed_raw_refs
        if overlap:
            raise ValueError(
                f"Checkpoint task {task.task_id} reclaims raw tensor keys: {sorted(ref.key for ref in overlap)}"
            )
        self.claimed_raw_refs.update(raw_refs)
        self.direct_tasks.append(task)

    def remaining_weight_map(self) -> dict[str, str]:
        claimed_keys = {ref.key for ref in self.claimed_raw_refs}
        return {key: shard for key, shard in self.weight_map.items() if key not in claimed_keys}

    def materialize_tasks(self) -> list[CheckpointTask]:
        return [_materialize_staged_task(task_plan, self.weight_map) for task_plan in self.task_plans] + list(
            self.direct_tasks
        )


def _materialize_staged_task(task_plan: CheckpointTaskPlan, weight_map: dict[str, str]) -> CheckpointTask:
    if not task_plan.stages:
        raise ValueError(f"Checkpoint task plan {task_plan.task_id} has no transformation stages.")

    available_refs = set(task_plan.input_refs)
    for stage in task_plan.stages:
        missing_refs = set(stage.input_refs) - available_refs
        if missing_refs:
            raise ValueError(
                f"Checkpoint stage {stage.stage_id} has unsatisfied inputs: "
                f"{sorted((ref.key, ref.stage) for ref in missing_refs)}"
            )
        duplicate_refs = set(stage.output_refs) & available_refs
        if duplicate_refs:
            raise ValueError(
                f"Checkpoint stage {stage.stage_id} reproduces existing refs: "
                f"{sorted((ref.key, ref.stage) for ref in duplicate_refs)}"
            )
        available_refs.update(stage.output_refs)

    final_refs = task_plan.current_refs
    if any(ref.stage != "final" for ref in final_refs):
        raise ValueError(f"Checkpoint task plan {task_plan.task_id} does not terminate in final tensor refs.")

    stage_labels = tuple(label for stage in task_plan.stages for label in (stage.labels or (stage.stage_id,)))
    task_values = dict(task_plan.params.values)
    task_values["stages"] = stage_labels + ("final",)
    task_values["stage_transform_ids"] = tuple(stage.params.transform_id for stage in task_plan.stages)
    task_params = TaskParams(
        task_plan.params.transform_id, tuple(sorted(task_values.items(), key=lambda item: item[0]))
    )
    stages = tuple(task_plan.stages)
    raw_refs = tuple(task_plan.input_refs)
    output_file = task_plan.output_file

    def runner(src: Path, out: Path, target_dtype: torch.dtype) -> dict[str, str]:
        from safetensors import safe_open

        from QEfficient.utils.checkpoint_utils import atomic_save

        remaining_reads = Counter(ref for stage in stages for ref in stage.input_refs)
        values: dict[TensorRef, torch.Tensor] = {}
        with ExitStack() as stack:
            handles = {
                shard_name: stack.enter_context(safe_open(str(src / shard_name), framework="pt"))
                for shard_name in task_plan.source_files
            }

            def get_tensor(ref: TensorRef) -> torch.Tensor:
                if ref in values:
                    return values[ref]
                if ref.stage != "raw":
                    raise KeyError(f"Tensor ref is unavailable: {ref}")
                return handles[weight_map[ref.key]].get_tensor(ref.key)

            for stage in stages:
                outputs = stage.run(get_tensor, target_dtype)
                expected_refs = set(stage.output_refs)
                if set(outputs) != expected_refs:
                    missing = sorted((ref.key, ref.stage) for ref in expected_refs - set(outputs))
                    unexpected = sorted((ref.key, ref.stage) for ref in set(outputs) - expected_refs)
                    raise ValueError(
                        f"Checkpoint stage {stage.stage_id} returned invalid refs; "
                        f"missing={missing}, unexpected={unexpected}."
                    )
                for ref in stage.input_refs:
                    remaining_reads[ref] -= 1
                    if remaining_reads[ref] == 0 and ref.stage != "raw":
                        values.pop(ref, None)
                values.update(outputs)

            final_tensors = {ref.key: values[ref] for ref in final_refs}
            atomic_save(final_tensors, out / output_file)

        return {ref.key: output_file for ref in final_refs}

    return CheckpointTask(
        task_id=task_plan.task_id,
        input_refs=raw_refs,
        output_refs=final_refs,
        source_files=task_plan.source_files,
        output_file=output_file,
        estimated_peak_bytes=task_plan.estimated_peak_bytes,
        params=task_params,
        runner=runner,
        stages=stages,
    )


@dataclass
class CheckpointPlan:
    """Complete description of a prepared checkpoint execution."""

    tasks: list[CheckpointTask]
    raw_refs: set[TensorRef]
    target_dtype: torch.dtype
    source_fingerprint: list[dict]
    transform_ids: tuple[str, ...]
    active_group_id: str = "none"

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "target_dtype": str(self.target_dtype),
            "source_fingerprint": self.source_fingerprint,
            "active_group_id": self.active_group_id,
            "transform_ids": list(self.transform_ids),
            "tasks": [task.fingerprint() for task in self.tasks],
        }


def execute_checkpoint_plan(
    plan: CheckpointPlan,
    src: Path,
    out: Path,
    target_dtype: torch.dtype,
    *,
    max_ram_bytes: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> dict[str, str]:
    """Execute ready checkpoint tasks under one shared RAM budget."""

    from QEfficient.utils.checkpoint_utils import available_ram_gb, cpu_count

    if max_ram_bytes is None:
        max_ram_bytes = max(1, int(available_ram_gb() * 0.8 * 1024**3))
    if max_workers is None:
        max_workers = max(1, min(len(plan.tasks), cpu_count())) if plan.tasks else 1

    for task in plan.tasks:
        if task.estimated_peak_bytes > max_ram_bytes:
            required_gb = task.estimated_peak_bytes / 1024**3
            limit_gb = max_ram_bytes / 1024**3
            raise ValueError(
                f"Checkpoint transform task {task.task_id} requires at least {required_gb:.2f} GB, "
                f"but max checkpoint transform RAM is {limit_gb:.2f} GB."
            )

    pending = list(plan.tasks)
    available_refs = set(plan.raw_refs)
    running = {}
    active_bytes = 0
    final_weight_map: dict[str, str] = {}

    _proc = psutil.Process(os.getpid())
    _peak_rss = _proc.memory_info().rss
    _peak_vms = _proc.memory_info().vms

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or running:
            submitted = True
            while submitted:
                submitted = False
                for task in list(pending):
                    if not set(task.input_refs).issubset(available_refs):
                        continue
                    if active_bytes + task.estimated_peak_bytes > max_ram_bytes:
                        continue
                    future = executor.submit(task.run, src, out, target_dtype)
                    running[future] = task
                    active_bytes += task.estimated_peak_bytes
                    pending.remove(task)
                    submitted = True

            if not running:
                waiting = ", ".join(task.task_id for task in pending)
                raise ValueError(f"Checkpoint transform plan has unsatisfied dependencies: {waiting}")

            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                task = running.pop(future)
                active_bytes -= task.estimated_peak_bytes
                result = future.result()
                _mi = _proc.memory_info()
                if _mi.rss > _peak_rss:
                    _peak_rss = _mi.rss
                if _mi.vms > _peak_vms:
                    _peak_vms = _mi.vms
                expected_final_keys = {ref.key for ref in task.output_refs if ref.stage == "final"}
                result_keys = set(result)
                if result_keys != expected_final_keys:
                    missing = sorted(expected_final_keys - result_keys)
                    unexpected = sorted(result_keys - expected_final_keys)
                    raise ValueError(
                        f"Checkpoint task {task.task_id} returned an invalid final weight map; "
                        f"missing={missing}, unexpected={unexpected}."
                    )
                for key, shard_name in result.items():
                    if key in final_weight_map:
                        raise ValueError(f"Checkpoint task produced duplicate final tensor key: {key}")
                    final_weight_map[key] = shard_name
                available_refs.update(task.output_refs)

    logger.info(
        "Checkpoint transform complete | peak RSS: %.2f GB | peak VMS: %.2f GB",
        _peak_rss / 1024**3,
        _peak_vms / 1024**3,
    )
    return final_weight_map


# Marks a prepared checkpoint directory as complete, so re-runs can skip work.
CHECKPOINT_PREPARED_SENTINEL = ".checkpoint_prepared"
CHECKPOINT_PREPARED_MANIFEST = ".checkpoint_prepared.json"


def _checkpoint_files(root: Path) -> List[Path]:
    patterns = ("*.safetensors", "*.bin", "*.json")
    files = set()
    for pattern in patterns:
        files.update(root.glob(pattern))
    return sorted(files)


def _checkpoint_file_fingerprint(root: Path) -> List[dict]:
    return [
        {"path": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in _checkpoint_files(root)
    ]


def _checkpoint_manifest(
    src: Path,
    target_dtype: torch.dtype,
    transforms: List[Type["BaseCheckpointTransform"]],
    active_group_id: str = "none",
    plan_payload: Optional[dict] = None,
) -> dict:
    payload = plan_payload or {}
    return {
        "version": 3,
        "source": str(src.resolve()),
        "target_dtype": str(target_dtype),
        "active_group": active_group_id,
        "transforms": [f"{t.__module__}.{t.__name__}" for t in transforms],
        "files": _checkpoint_file_fingerprint(src),
        "plan": payload,
        "output_files": sorted({task["output_file"] for task in payload.get("tasks", [])}),
    }


def _manifest_matches(out: Path, expected: dict) -> bool:
    manifest_path = out / CHECKPOINT_PREPARED_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        if json.loads(manifest_path.read_text()) != expected:
            return False
        if not (out / "model.safetensors.index.json").is_file():
            return False
        return all((out / file_name).is_file() for file_name in expected.get("output_files", []))
    except (OSError, json.JSONDecodeError):
        return False


def _prepared_checkpoint_is_readable(out: Path) -> bool:
    """Return True when every indexed prepared checkpoint shard is readable."""
    try:
        weight_map = read_weight_map(out)
    except (OSError, json.JSONDecodeError, KeyError):
        return False

    if not weight_map:
        return False

    for shard_name in set(weight_map.values()):
        shard_path = out / shard_name
        if not shard_path.is_file() or not os.access(shard_path, os.R_OK):
            return False
    return True


def _write_manifest(out: Path, manifest: dict) -> None:
    (out / CHECKPOINT_PREPARED_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _validate_final_weight_map(out: Path, weight_map: dict[str, str]) -> None:
    """Verify every final index entry names a tensor in an existing output shard."""
    from safetensors import safe_open

    keys_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard_name in weight_map.items():
        keys_by_shard[shard_name].add(key)

    for shard_name, expected_keys in keys_by_shard.items():
        shard_path = out / shard_name
        if not shard_path.is_file():
            raise ValueError(f"Prepared checkpoint references missing output shard: {shard_name}")
        with safe_open(str(shard_path), framework="pt") as handle:
            missing_keys = sorted(expected_keys - set(handle.keys()))
        if missing_keys:
            raise ValueError(f"Prepared checkpoint shard {shard_name} is missing indexed tensor keys: {missing_keys}")


def _clear_stale_prepared_dir(out: Path, src: Path) -> None:
    if not out.exists() or out == src:
        return
    if out.is_dir():
        shutil.rmtree(out)
    else:
        out.unlink()


def detect_group_transform(
    config,
    weight_map: Dict[str, str],
    hash_params: Optional[Dict] = None,
    transforms: Optional[List] = None,
) -> Optional[Type["BaseCheckpointTransform"]]:
    """Return the active layout transform class, or None for dense models.

    Scans weight_map key patterns (gated by config.num_experts) to identify
    which layout transform applies.  DtypeConversionCheckpointTransform is
    excluded — it always runs unconditionally and is not a layout transform.

    Parameters
    ----------
    config
        HuggingFace model config.  ``num_local_experts`` / ``num_experts``
        gates all MoE detection — absent means dense model.
    weight_map
        ``{tensor_key: shard_filename}`` from ``model.safetensors.index.json``.
    hash_params
        Model hash parameters (from ``qeff_model.hash_params``).
    transforms
        Registered transforms list — used to look up the class by TRANSFORM_ID.
        Falls back to direct imports when not provided.

    Returns
    -------
    Type[BaseCheckpointTransform] or None
        The active layout transform class, or ``None`` for dense models.
    """
    if hash_params is None:
        hash_params = {}

    num_experts = None
    if config is not None:
        num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
    if not num_experts:
        return None

    # Per-expert format: validate expert indices match config declaration.
    expert_indices_per_layer: Dict[int, set] = defaultdict(set)
    for k in weight_map:
        m = re.search(r"\.layers\.(\d+)\..*\.experts\.(\d+)\.", k)
        if m:
            expert_indices_per_layer[int(m.group(1))].add(int(m.group(2)))

    if expert_indices_per_layer:
        expected = set(range(num_experts))
        for layer_idx, found in expert_indices_per_layer.items():
            if found != expected:
                raise ValueError(
                    f"Layer {layer_idx}: config declares {num_experts} experts "
                    f"but checkpoint contains indices {sorted(found)}. "
                    "The checkpoint may be incomplete or corrupted."
                )
        return _find_transform_by_id("moe_expert_stacking_v1", transforms)

    # GptOss is identified by model_type — always uses MXFP4 dequant transform.
    model_type = getattr(config, "model_type", None) if config else None
    if model_type == "gpt_oss":
        return _find_transform_by_id("gptoss_mxfp4_dequant_v1", transforms)

    # Pre-stacked formats — delegate detection to each transform's is_applicable().
    # FusedExpertSplitCheckpointTransform handles both Mixtral fused and GraniteMoE
    # internally via _get_key_remap() — no hardcoded patterns needed here.
    fused_cls = _find_transform_by_id("fused_expert_split_v1", transforms)
    if fused_cls is not None and fused_cls.is_applicable(weight_map):
        return fused_cls

    return None


def _find_transform_by_id(
    transform_id: str,
    transforms: Optional[List],
) -> Optional[Type["BaseCheckpointTransform"]]:
    """Return the transform class with matching TRANSFORM_ID from the list."""
    if transforms:
        for t in transforms:
            if getattr(t, "TRANSFORM_ID", None) == transform_id:
                return t
    # Fallback: import directly when transforms list not provided
    from QEfficient.exporter.weight_free.checkpoint_transforms import (  # noqa: PLC0415
        DtypeConversionCheckpointTransform,
        FusedExpertSplitCheckpointTransform,
        GptOssMxfp4ExpertDequantSplitCheckpointTransform,
        MoEExpertStackingCheckpointTransform,
    )

    # Legacy expert-parallel transform IDs resolve to the corresponding planner.
    # _find_transform_by_id is used by promote_initializers_and_build_spec()
    # to recover the active_transform from the manifest TRANSFORM_ID so that
    # resolve_onnx_key() can map ONNX names (e.g. .mlp.) to checkpoint keys
    # (.block_sparse_moe.).  That mapping doesn't need P or E/P, so the
    # base transform is sufficient for key-lookup purposes.
    _ID_MAP = {
        "moe_expert_stacking_v1": MoEExpertStackingCheckpointTransform,
        "moe_expert_parallel_stacking_v1": MoEExpertStackingCheckpointTransform,
        "gptoss_mxfp4_dequant_v1": GptOssMxfp4ExpertDequantSplitCheckpointTransform,
        "gptoss_mxfp4_dequant_expert_parallel_v1": GptOssMxfp4ExpertDequantSplitCheckpointTransform,
        "fused_expert_split_v1": FusedExpertSplitCheckpointTransform,
        "moe_fused_expert_split_v1": FusedExpertSplitCheckpointTransform,
        "granite_moe_fused_split_v1": FusedExpertSplitCheckpointTransform,
        "dtype_conversion_v1": DtypeConversionCheckpointTransform,
    }
    return _ID_MAP.get(transform_id)


class BaseCheckpointTransform:
    """Base class for transforms that contribute stages to a checkpoint plan."""

    TRANSFORM_ID: str = ""

    def __init__(self):
        """Prevent direct instantiation of transform marker classes."""
        raise TypeError("Checkpoint transform classes are not to be instantiated.")

    @classmethod
    def plan_tasks(cls, context: CheckpointPlanningContext) -> None:
        """Add task plans or in-memory stages to the shared planning context."""

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True if this transform should contribute to the plan."""
        return True


class CheckpointTransformPipeline:
    """Plan, execute, and finalize a prepared checkpoint."""

    def __init__(self, transforms: List[Type[BaseCheckpointTransform]]):
        self.transforms = transforms

    def build_plan(
        self,
        src: Path,
        target_dtype: torch.dtype,
        *,
        config=None,
        hash_params: Optional[Dict] = None,
    ) -> tuple[CheckpointPlan, str]:
        """Collect transform stages, fuse each tensor group, and return the execution plan."""
        src = Path(src)
        weight_map = read_weight_map(src)
        hash_params = hash_params or {}

        from QEfficient.exporter.weight_free.checkpoint_transforms import DtypeConversionCheckpointTransform

        context = CheckpointPlanningContext(
            weight_map=weight_map,
            config=config,
            hash_params=hash_params,
            target_dtype=target_dtype,
            source_dir=src,
        )

        # Source-layout transforms inspect the complete raw checkpoint. Stage transforms
        # run afterwards and consume the staged refs produced by those layout transforms.
        seen_transforms = set()
        for transform in self.transforms:
            if transform in seen_transforms or transform is DtypeConversionCheckpointTransform:
                continue
            seen_transforms.add(transform)
            if transform.is_applicable(
                weight_map,
                config=config,
                hash_params=hash_params,
                target_dtype=target_dtype,
                source_dir=src,
            ):
                transform.plan_tasks(context)

        DtypeConversionCheckpointTransform.plan_tasks(context)

        if len(context.active_layout_ids) > 1:
            raise ValueError(
                "Checkpoint planning selected multiple source layout transforms: "
                f"{context.active_layout_ids}. A checkpoint must use one source expert layout."
            )
        active_group_id = context.active_layout_ids[0] if context.active_layout_ids else "none"
        tasks = context.materialize_tasks()

        output_refs = [ref for task in tasks for ref in task.output_refs]
        if len(output_refs) != len(set(output_refs)):
            raise ValueError("Checkpoint plan contains duplicate output tensor references.")

        planned_raw_keys = {ref.key for task in tasks for ref in task.input_refs if ref.stage == "raw"}
        missing_raw_keys = set(weight_map) - planned_raw_keys
        if missing_raw_keys:
            raise ValueError(f"Checkpoint plan does not cover source keys: {sorted(missing_raw_keys)}")

        transform_ids = []
        for task in tasks:
            stage_ids = [stage.params.transform_id for stage in task.stages]
            for transform_id in stage_ids or [task.params.transform_id]:
                if transform_id not in transform_ids:
                    transform_ids.append(transform_id)

        plan = CheckpointPlan(
            tasks=tasks,
            raw_refs={TensorRef(key, "raw") for key in weight_map},
            target_dtype=target_dtype,
            source_fingerprint=_checkpoint_file_fingerprint(src),
            transform_ids=tuple(transform_ids),
            active_group_id=active_group_id,
        )
        return plan, active_group_id

    def apply(
        self,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> Path:
        """Prepare checkpoint at src into out and return the usable directory."""
        src, out = Path(src), Path(out)

        if list(src.glob("*.bin")) and not list(src.glob("*.safetensors")):
            raise ValueError(
                f"Checkpoint at {src} contains .bin files but no safetensors files. "
                "Weight-free export requires safetensors format. "
                "Convert the checkpoint to safetensors before exporting."
            )

        config = kwargs.pop("config", None)
        hash_params = kwargs.pop("hash_params", None) or {}
        max_ram_bytes = kwargs.pop("max_ram_bytes", None)
        max_workers = kwargs.pop("max_workers", None)
        plan = kwargs.pop("plan", None)
        if plan is None:
            plan, active_group_id = self.build_plan(
                src,
                target_dtype,
                config=config,
                hash_params=hash_params,
            )
        else:
            active_group_id = plan.active_group_id
            if plan.target_dtype != target_dtype:
                raise ValueError(
                    f"Checkpoint plan target dtype {plan.target_dtype} does not match requested {target_dtype}."
                )
            current_source_fingerprint = _checkpoint_file_fingerprint(src)
            if plan.source_fingerprint != current_source_fingerprint:
                raise ValueError("Source checkpoint changed after the checkpoint transform plan was built.")
        plan_payload = plan.fingerprint_payload()
        expected_manifest = _checkpoint_manifest(
            src,
            target_dtype,
            self.transforms,
            active_group_id,
            plan_payload,
        )
        if (out / CHECKPOINT_PREPARED_SENTINEL).exists() and _manifest_matches(out, expected_manifest):
            if _prepared_checkpoint_is_readable(out):
                return out
            logger.warning("Prepared checkpoint at %s is incomplete or unreadable; rebuilding.", out)

        _clear_stale_prepared_dir(out, src)
        out.mkdir(parents=True, exist_ok=True)

        try:
            new_weight_map = execute_checkpoint_plan(
                plan,
                src,
                out,
                target_dtype,
                max_ram_bytes=max_ram_bytes,
                max_workers=max_workers,
            )
            _validate_final_weight_map(out, new_weight_map)
            copy_checkpoint_aux_files(src, out)
            write_index(out, new_weight_map)
            _write_manifest(out, expected_manifest)
            (out / CHECKPOINT_PREPARED_SENTINEL).touch()
        except Exception:
            _clear_stale_prepared_dir(out, src)
            raise
        return out
