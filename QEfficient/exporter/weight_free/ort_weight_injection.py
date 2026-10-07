# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from safetensors import safe_open

from QEfficient.exporter.weight_free.weight_spec import ExternalDataFile, WeightSpecLocation, load_weight_spec
from QEfficient.utils.checkpoint_utils import checkpoint_root, resolve_checkpoint_dir


def _load_checkpoint_tensor(checkpoint_file: str, key: str) -> np.ndarray:
    """Load one tensor from a safetensors checkpoint as a NumPy array."""
    with safe_open(checkpoint_file, framework="pt") as handle:
        tensor = handle.get_tensor(key).detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy()


def _default_weights_roots(weight_spec_path: Path, spec) -> List[Path]:
    """Return candidate roots for resolving relative checkpoint paths."""
    roots = []
    ext_root = os.environ.get("AIC_EXTERNAL_DATA_ROOT")
    if ext_root:
        roots.append(Path(ext_root).expanduser())
    roots.append(weight_spec_path.parent)

    candidate = Path(spec.model_id).expanduser()
    if candidate.exists():
        roots.append(candidate.parent)
    else:
        checkpoint_dir = resolve_checkpoint_dir(spec.model_id)
        root = checkpoint_root(spec.model_id, [str(path) for path in checkpoint_dir.glob("*.safetensors")])
        if root is not None:
            roots.append(root)

    deduped_roots: List[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_roots.append(resolved)
    return deduped_roots


def _resolve_location_file(
    location: WeightSpecLocation,
    files: Sequence[ExternalDataFile],
    candidate_roots: Sequence[Path],
) -> Path:
    """Resolve a weight-spec location to an absolute or best-effort file path."""
    location_path = Path(files[location.file].path) if isinstance(location.file, int) else Path(location.file)
    if location_path.is_absolute():
        return location_path

    for root in candidate_roots:
        candidate = root / location_path
        if candidate.exists():
            return candidate

    return candidate_roots[0] / location_path if candidate_roots else location_path


def load_weight_free_ort_inputs(
    weight_spec_path: Path,
    runtime_inputs: Dict[str, np.ndarray],
    weights_root: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Load external checkpoint weights and merge them with ONNX Runtime inputs.

    Parameters
    ----------
    weight_spec_path : Path
        Path to ``weight_spec.json`` emitted during weight-free export.
    runtime_inputs : Dict[str, np.ndarray]
        Dynamic inputs already prepared for ONNX Runtime inference.
    weights_root : Optional[Path], optional
        Explicit root directory used to resolve relative checkpoint file paths.

    Returns
    -------
    Dict[str, np.ndarray]
        ONNX Runtime input mapping containing dynamic inputs and external weights.
    """
    weight_spec_path = Path(weight_spec_path)
    spec = load_weight_spec(weight_spec_path)
    candidate_roots = []
    if weights_root is not None:
        candidate_roots.append(Path(weights_root).expanduser().resolve())
    candidate_roots.extend(_default_weights_roots(weight_spec_path, spec))

    ort_inputs = dict(runtime_inputs)
    for spec_input in spec.inputs:
        if spec_input.name in ort_inputs:
            continue
        checkpoint_file = _resolve_location_file(spec_input.location, spec.files, candidate_roots)
        ort_inputs[spec_input.name] = _load_checkpoint_tensor(str(checkpoint_file), spec_input.location.key)

    return ort_inputs
