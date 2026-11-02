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
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Type

import torch

from QEfficient.utils.checkpoint_utils import checkpoint_files_complete, convert_bin_to_safetensors, read_weight_map

# Marks a prepared checkpoint directory as complete, so re-runs can skip work.
CHECKPOINT_PREPARED_SENTINEL = ".checkpoint_prepared"
CHECKPOINT_PREPARED_MANIFEST = ".checkpoint_prepared.json"


def _checkpoint_files(root: Path) -> List[Path]:
    patterns = ("*.safetensors", "*.bin", "*.json")
    files = set()
    for pattern in patterns:
        files.update(root.glob(pattern))
    return sorted(files)


def _checkpoint_file_fingerprint(root: Path, label: str) -> List[dict]:
    fingerprint = []
    for path in _checkpoint_files(root):
        stat = path.stat()
        fingerprint.append(
            {
                "label": label,
                "path": path.name,
                "size": stat.st_size,
            }
        )
    return fingerprint


def _checkpoint_manifest(
    src: Path,
    source_dir: Path,
    target_dtype: torch.dtype,
    transforms: List[Type["BaseCheckpointTransform"]],
    transform_context: Dict[str, Any] | None = None,
) -> dict:
    files = _checkpoint_file_fingerprint(source_dir, "source")
    if source_dir != src:
        files.extend(_checkpoint_file_fingerprint(src, "original"))
    manifest = {
        "version": 1,
        "source": str(source_dir.resolve()),
        "original_source": str(src.resolve()),
        "target_dtype": str(target_dtype),
        "transforms": [f"{transform.__module__}.{transform.__name__}" for transform in transforms],
        "files": files,
    }
    if transform_context:
        manifest["transform_context"] = transform_context
    return manifest


def _manifest_matches(out: Path, expected: dict) -> bool:
    manifest_path = out / CHECKPOINT_PREPARED_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text()) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _write_manifest(out: Path, manifest: dict) -> None:
    (out / CHECKPOINT_PREPARED_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _clear_stale_prepared_dir(out: Path, src: Path, source_dir: Path) -> None:
    if not out.exists() or out in {src, source_dir}:
        return
    if out.is_dir():
        shutil.rmtree(out)
    else:
        out.unlink()


class _CheckpointPreparationLock:
    """Small cross-process lock for shared checkpoint preparation directories."""

    def __init__(self, out: Path):
        self.path = out.with_name(out.name + ".lock")
        self._lock_dir = out.with_name(out.name + ".lockdir")
        self._handle = None
        self._using_lock_dir = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl

            self._handle = self.path.open("w")
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            while True:
                try:
                    self._lock_dir.mkdir()
                    self._using_lock_dir = True
                    break
                except FileExistsError:
                    time.sleep(0.1)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            import fcntl

            if self._handle is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            if self._using_lock_dir:
                try:
                    self._lock_dir.rmdir()
                except OSError:
                    pass
        if self._handle is not None:
            self._handle.close()


class BaseCheckpointTransform:
    """Base class for checkpoint file transforms. Not to be instantiated.

    Each subclass produces a *complete* prepared checkpoint directory in ``out``.
    The pipeline picks the first applicable transform and stops — no chaining.
    """

    def __init__(self):
        """Prevent direct instantiation of transform marker classes."""
        raise TypeError("Checkpoint transform classes are not to be instantiated.")

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> bool:
        """Transform checkpoint at ``src``, write result to ``out``.
        Returns True if the checkpoint was prepared, False if skipped (idempotent)."""
        raise NotImplementedError

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True if this transform should run for the given checkpoint."""
        return True


class CheckpointTransformPipeline:
    """Selects and runs the first applicable checkpoint transform.

    Transforms are priority-ordered. The first one whose ``is_applicable()``
    returns True is executed and the pipeline stops. Each transform produces a
    complete prepared checkpoint — there is no chaining between transforms.

    TODO(wf): Current implementation is a selector but is named as pipeline.
    Correct design is to apply multiple transforms sequentially on the tensors that it applies to
    and we parallelize this processing across all tensors.
    Each transform should have single transformation responsibility.
    Currently all transforms copy multiple responsibitlies from each-other. This is not scalable.

    Example::

        pipeline = CheckpointTransformPipeline([
            MoEExpertStackingCheckpointTransform,   # MoE models: stacks + converts
            DtypeConversionCheckpointTransform,     # dense models: converts only
        ])
        prepared_dir = pipeline.apply(src, out, target_dtype=torch.float32)
    """

    def __init__(self, transforms: List[Type[BaseCheckpointTransform]]):
        """Create a priority-ordered checkpoint transform pipeline."""
        self.transforms = transforms

    def apply(
        self,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> Path:
        """Apply the first matching transform and return the usable checkpoint directory."""
        src, out = Path(src), Path(out)

        with _CheckpointPreparationLock(out):
            return self._apply_locked(src, out, target_dtype=target_dtype, **kwargs)

    def _apply_locked(
        self,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> Path:
        """Apply checkpoint preparation while holding the output lock."""

        source_dir = src
        has_safetensors = bool(list(src.glob("*.safetensors"))) or (src / "model.safetensors.index.json").exists()
        if not has_safetensors and list(src.glob("*.bin")):
            # TODO(wf): rewriting bin into safetensors is not good idea,
            # we better error out saying we don't support bin format or handle without the rewrite.
            source_dir = out.with_name(out.name + "-source-safetensors")
            convert_bin_to_safetensors(src, source_dir)

        model_config = kwargs.get("model_config")
        manifest_context = None
        if model_config is not None:
            manifest_context = {
                "model_type": getattr(model_config, "model_type", None),
                "num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
            }
        checkpoint_transform_context = kwargs.get("checkpoint_transform_context")
        if checkpoint_transform_context:
            if manifest_context is None:
                manifest_context = {}
            manifest_context["checkpoint_transform_context"] = checkpoint_transform_context
        expected_manifest = _checkpoint_manifest(
            src,
            source_dir,
            target_dtype,
            self.transforms,
            transform_context=manifest_context,
        )
        if (
            (out / CHECKPOINT_PREPARED_SENTINEL).exists()
            and _manifest_matches(out, expected_manifest)
            and checkpoint_files_complete(out)
        ):
            return out
        _clear_stale_prepared_dir(out, src, source_dir)

        weight_map = read_weight_map(source_dir)
        for transform in self.transforms:
            if transform.is_applicable(weight_map, src=source_dir, target_dtype=target_dtype, **kwargs):
                transform.apply(source_dir, out, target_dtype=target_dtype, **kwargs)
                if (out / CHECKPOINT_PREPARED_SENTINEL).exists():
                    _write_manifest(out, expected_manifest)
                return out
        return source_dir  # no transform applicable - source is already usable as-is
