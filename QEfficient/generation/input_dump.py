# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Runtime input dumping utilities for qaic-runner replay.

The dumper is intentionally session-local: each QPC gets its own directory with
an ``aic_batch_io.json`` descriptor containing one entry per runtime invocation.
Inputs are serialized as raw host buffers, while output entries describe the
buffers qaic-runner should allocate for the same invocation sequence.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

_DUMP_ENV_VAR = "QEFFICIENT_DUMP_INPUTS"
_DEFAULT_DUMP_DIR = "qeff_input_dumps"
_CAPTURE_COUNTER = count()
_SESSION_COUNTER = count()
_DUAL_QPC_MULTIMODAL_DEPENDENCIES = {"vision": [], "text": ["vision"]}
_DUAL_QPC_MULTIMODAL_ORDER = ["vision", "text"]
_SINGLE_QPC_TEXT_ORDER = ["text"]
_SINGLE_QPC_MULTIMODAL_ORDER = ["vision_text"]


@dataclass(frozen=True)
class RuntimeInputDumpCapture:
    """Prepared root for one logical runtime-input capture."""

    root: Path


class InputDumpValidationError(ValueError):
    """Raised when a runtime input dump is internally inconsistent."""


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": value.min().item() if value.size else None,
            "max": value.max().item() if value.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe or "buffer"


def _dims_nbytes(dims: Sequence[int], elem_size: int) -> int:
    return int(np.prod([int(dim) for dim in dims], dtype=np.int64)) * int(elem_size)


def _resolve_dump_base(path: Optional[Union[str, Path, bool, RuntimeInputDumpCapture]]) -> Optional[Path]:
    if isinstance(path, RuntimeInputDumpCapture):
        return path.root
    if path is False:
        return None
    if path is None:
        env_path = os.environ.get(_DUMP_ENV_VAR)
        if not env_path:
            return None
        path = env_path
    if path is True:
        path = _DEFAULT_DUMP_DIR
    return Path(path).expanduser().resolve()


def _make_unique_capture_root(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    capture_id = next(_CAPTURE_COUNTER)
    capture_root = base / f"capture_{capture_id:02d}"
    suffix = 1
    while capture_root.exists():
        capture_root = base / f"capture_{capture_id:02d}_{suffix}"
        suffix += 1
    capture_root.mkdir(parents=True)
    return capture_root


def _resolve_dump_root(path: Optional[Union[str, Path, bool, RuntimeInputDumpCapture]]) -> Optional[Path]:
    base = _resolve_dump_base(path)
    if base is None:
        return None
    if isinstance(path, RuntimeInputDumpCapture):
        return base
    return _make_unique_capture_root(base)


def resolve_runtime_dump_inputs_path(
    artifacts: Optional[Union[str, Path, bool]],
    dump_inputs_path: Optional[Union[str, Path, bool, RuntimeInputDumpCapture]] = None,
) -> Optional[Union[str, Path, bool, RuntimeInputDumpCapture]]:
    """Resolve runtime input dumping from path-valued artifacts mode.

    ``artifacts=True`` keeps the existing dry-run runner-bundle behavior in
    public ``generate`` APIs. A string or ``Path`` value enables executed runtime
    input dumps under a fresh capture directory at that location. ``dump_inputs_path``
    is retained for internal plumbing and backward compatibility.
    """
    path = dump_inputs_path
    if path is None and isinstance(artifacts, (str, Path)):
        path = artifacts
    if path is None:
        return None
    if isinstance(path, RuntimeInputDumpCapture):
        return path
    base = _resolve_dump_base(path)
    if base is None:
        return None
    return RuntimeInputDumpCapture(_make_unique_capture_root(base))


def _make_unique_dir(root: Path, component_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    session_id = next(_SESSION_COUNTER)
    base = f"{session_id:02d}_{_safe_filename(component_name)}"
    session_dir = root / base
    suffix = 1
    while session_dir.exists():
        session_dir = root / f"{base}_{suffix}"
        suffix += 1
    session_dir.mkdir(parents=True)
    return session_dir


def _component_contract(component_names: set[str]) -> dict[str, Any]:
    if component_names == {"vision", "text"}:
        return {
            "capture_kind": "dual_qpc_multimodal",
            "component_order": _DUAL_QPC_MULTIMODAL_ORDER,
            "dependencies": _DUAL_QPC_MULTIMODAL_DEPENDENCIES,
            "required_component_counts": {"vision": 1, "text": 1},
            "replay_component": "text",
        }
    if component_names == {"vision_text"}:
        return {
            "capture_kind": "single_qpc_multimodal",
            "component_order": _SINGLE_QPC_MULTIMODAL_ORDER,
            "dependencies": {"vision_text": []},
            "required_component_counts": {"vision_text": 1},
            "replay_component": "vision_text",
        }
    if component_names == {"text"}:
        return {
            "capture_kind": "single_qpc_text",
            "component_order": _SINGLE_QPC_TEXT_ORDER,
            "dependencies": {"text": []},
            "required_component_counts": {"text": 1},
            "replay_component": "text",
        }

    component_order = sorted(component_names)
    return {
        "capture_kind": "custom_or_incomplete",
        "component_order": component_order,
        "dependencies": {component: [] for component in component_order},
        "required_component_counts": {component: 1 for component in component_order},
        "replay_component": component_order[-1] if component_order else None,
    }


def _write_aggregate_manifest(root: Path) -> None:
    component_manifests = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        component = str(manifest.get("component", manifest_path.parent.name))
        component_manifests.append((manifest_path.parent, component, manifest))

    component_names = {component for _, component, _ in component_manifests}
    contract = _component_contract(component_names)
    component_order = contract["component_order"]
    order_index = {component: index for index, component in enumerate(component_order)}
    required_component_counts = contract["required_component_counts"]

    component_counts = {component: 0 for component in sorted(component_names)}
    for _, component, _ in component_manifests:
        component_counts[component] = component_counts.get(component, 0) + 1

    missing_components = [
        component
        for component, count in required_component_counts.items()
        if component_counts.get(component, 0) < count
    ]
    unexpected_components = sorted(set(component_counts) - set(required_component_counts))
    all_required_components_captured = (
        not missing_components
        and not unexpected_components
        and all(component_counts.get(component, 0) == count for component, count in required_component_counts.items())
    )

    components = []
    for session_dir, component, manifest in sorted(
        component_manifests, key=lambda item: (order_index.get(item[1], len(order_index)), item[0].name)
    ):
        components.append(
            {
                "id": session_dir.name,
                "component": component,
                "path": session_dir.name,
                "manifest": f"{session_dir.name}/manifest.json",
                "aic_batch_io": f"{session_dir.name}/{manifest.get('aic_batch_io', 'aic_batch_io.json')}",
                "order": order_index.get(component),
                "depends_on": contract["dependencies"].get(component, []),
                "required": component in required_component_counts,
                "replay": component == contract["replay_component"],
                "invocation_count": len(manifest.get("invocations", [])),
            }
        )

    aggregate_manifest = {
        "version": 1,
        "format": "qefficient-runtime-input-dump",
        "capture_kind": contract["capture_kind"],
        "component_order": component_order,
        "dependencies": contract["dependencies"],
        "required_component_counts": required_component_counts,
        "component_counts": component_counts,
        "captured_components": sorted(component_counts),
        "missing_components": missing_components,
        "unexpected_components": unexpected_components,
        "all_required_components_captured": all_required_components_captured,
        "replay_component": contract["replay_component"],
        "components": components,
    }
    (root / "manifest.json").write_text(json.dumps(aggregate_manifest, indent=2))


class QAICInputDumper:
    """Collect qaic-runner inputs for every invocation of one QPC session."""

    def __init__(self, *, session, qpc_path: Union[str, Path], root: Union[str, Path], component_name: str):
        self.session = session
        self.qpc_path = Path(qpc_path)
        self.component_name = component_name
        self.root = Path(root)
        self.session_dir = _make_unique_dir(self.root, component_name)
        self.data_dir = self.session_dir / "data"
        self.data_dir.mkdir()
        self.io_batches: list[list[dict[str, Any]]] = []
        self.invocations: list[dict[str, Any]] = []
        self.runtime_schedule_events: list[dict[str, Any]] = []
        self.pending_slice_configs: dict[int, list[dict[str, Any]]] = {}
        self._write_manifest()

    @classmethod
    def from_config(
        cls,
        *,
        session,
        qpc_path: Union[str, Path],
        dump_inputs_path: Optional[Union[str, Path, bool, RuntimeInputDumpCapture]],
        component_name: Optional[str],
    ) -> Optional["QAICInputDumper"]:
        root = _resolve_dump_root(dump_inputs_path)
        if root is None:
            return None
        resolved_component = component_name or Path(qpc_path).name or "qpc"
        return cls(session=session, qpc_path=qpc_path, root=root, component_name=resolved_component)

    def _binding_item_size(self, name: str) -> int:
        binding = self.session.bindings[self.session.binding_index_map[name]]
        return int(self.session.aic_to_np_dtype_mapping[binding.type].itemsize)

    def _binding_dtype(self, name: str) -> np.dtype:
        binding = self.session.bindings[self.session.binding_index_map[name]]
        return np.dtype(self.session.aic_to_np_dtype_mapping[binding.type])

    def _canonical_binding_name(self, name: str) -> str:
        return self.session.bindings[self.session.binding_index_map[name]].name

    def _binding_dims(self, name: str) -> list[int]:
        binding = self.session.bindings[self.session.binding_index_map[name]]
        return [int(dim) for dim in binding.dims]

    @staticmethod
    def _dims_match(expected_dims: Sequence[int], actual_dims: Sequence[int]) -> bool:
        if len(expected_dims) != len(actual_dims):
            return False
        return all(
            int(expected) <= 0 or int(expected) == int(actual) for expected, actual in zip(expected_dims, actual_dims)
        )

    def _required_input_names(self, exec_obj_index: Optional[int]) -> list[str]:
        required_names = getattr(self.session, "required_input_names", None)
        if required_names is None:
            if exec_obj_index is not None or not hasattr(self.session, "input_names"):
                return []
            required_names = self.session.input_names

        required_inputs = []
        seen = set()
        for name in required_names:
            if name not in self.session.binding_index_map:
                raise InputDumpValidationError(f"Unknown required input binding: {name}")
            canonical_name = self._canonical_binding_name(name)
            if canonical_name in seen:
                raise InputDumpValidationError(f"Duplicate required input binding: {canonical_name}")
            if self._is_required_buffer_present(name, exec_obj_index):
                required_inputs.append(canonical_name)
                seen.add(canonical_name)
        return required_inputs

    def _is_required_buffer_present(self, name: str, exec_obj_index: Optional[int]) -> bool:
        if getattr(self.session, "required_input_names", None) is not None:
            return True
        qbuffers = getattr(self.session, "qbuffers", None)
        if qbuffers is None:
            return True
        binding_index = self.session.binding_index_map[name]
        if exec_obj_index is not None and qbuffers and isinstance(qbuffers[0], list):
            qbuffer = qbuffers[exec_obj_index][binding_index]
        else:
            qbuffer = qbuffers[binding_index]
        return int(getattr(qbuffer, "size", 1)) != 0

    @staticmethod
    def _array_dims(value: np.ndarray) -> list[int]:
        return list(value.shape) if value.shape else [1]

    def _write_input(self, invocation_dir: Path, name: str, value: np.ndarray) -> dict[str, Any]:
        array = np.ascontiguousarray(np.asarray(value))
        relative_path = Path("data") / invocation_dir.name / f"{_safe_filename(name)}.raw"
        array.tofile(self.session_dir / relative_path)
        return {
            "path": relative_path.as_posix(),
            "io-direction": "in",
            "elem-size": int(array.itemsize),
            "map-to": name,
            "dims": self._array_dims(array),
        }

    def _input_detail(self, entry: Mapping[str, Any], value: np.ndarray) -> dict[str, Any]:
        array = np.ascontiguousarray(np.asarray(value))
        return {
            "name": entry["map-to"],
            "path": entry["path"],
            "dims": self._array_dims(array),
            "dtype": str(array.dtype),
            "elem-size": int(array.itemsize),
            "byte-size": int(array.nbytes),
        }

    def _validate_array_for_binding(self, name: str, value: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(value))
        expected_dtype = self._binding_dtype(name)
        if array.dtype != expected_dtype:
            raise InputDumpValidationError(
                f"Input {name!r} has dtype {array.dtype}, expected {expected_dtype} for the QPC binding"
            )
        expected_elem_size = self._binding_item_size(name)
        if int(array.itemsize) != expected_elem_size:
            raise InputDumpValidationError(
                f"Input {name!r} has element size {array.itemsize}, expected {expected_elem_size}"
            )
        expected_dims = self._binding_dims(name)
        actual_dims = self._array_dims(array)
        if not self._dims_match(expected_dims, actual_dims):
            raise InputDumpValidationError(
                f"Input {name!r} has shape {actual_dims}, expected {expected_dims} for the QPC binding"
            )
        return array

    def _validate_invocation_inputs(
        self,
        entries: Sequence[Mapping[str, Any]],
        required_inputs: Sequence[str],
        input_details: Mapping[str, Mapping[str, Any]],
    ) -> None:
        input_entries = [entry for entry in entries if entry.get("io-direction") == "in"]
        descriptor_inputs = [str(entry.get("map-to")) for entry in input_entries]
        duplicate_inputs = sorted({name for name in descriptor_inputs if descriptor_inputs.count(name) > 1})
        if duplicate_inputs:
            raise InputDumpValidationError(f"Duplicate input bindings in descriptor: {duplicate_inputs}")

        unknown_descriptor_inputs = [name for name in descriptor_inputs if name not in self.session.binding_index_map]
        if unknown_descriptor_inputs:
            raise InputDumpValidationError(f"Unknown descriptor input bindings: {unknown_descriptor_inputs}")

        missing_required = sorted(set(required_inputs) - set(descriptor_inputs))
        if missing_required:
            raise InputDumpValidationError(f"Missing required input bindings: {missing_required}")

        descriptor_paths = [str(entry.get("path")) for entry in input_entries]
        duplicate_paths = sorted({path for path in descriptor_paths if descriptor_paths.count(path) > 1})
        if duplicate_paths:
            raise InputDumpValidationError(f"Duplicate input file paths in descriptor: {duplicate_paths}")

        for entry in input_entries:
            name = str(entry["map-to"])
            path = str(entry["path"])
            dims = [int(dim) for dim in entry["dims"]]
            elem_size = int(entry["elem-size"])
            file_path = self.session_dir / path
            if not file_path.is_file():
                raise InputDumpValidationError(f"Descriptor input {name!r} references missing file: {path}")
            expected_elem_size = self._binding_item_size(name)
            if elem_size != expected_elem_size:
                raise InputDumpValidationError(
                    f"Descriptor input {name!r} has element size {elem_size}, expected {expected_elem_size}"
                )
            detail = input_details.get(name)
            if detail is not None:
                if dims != [int(dim) for dim in detail["dims"]]:
                    raise InputDumpValidationError(
                        f"Descriptor input {name!r} has dims {dims}, expected {detail['dims']}"
                    )
                if elem_size != int(detail["elem-size"]):
                    raise InputDumpValidationError(
                        f"Descriptor input {name!r} has element size {elem_size}, expected {detail['elem-size']}"
                    )
                expected_nbytes = int(detail["byte-size"])
            else:
                expected_nbytes = _dims_nbytes(dims, elem_size)
            actual_nbytes = file_path.stat().st_size
            if actual_nbytes != expected_nbytes:
                raise InputDumpValidationError(
                    f"Descriptor input {name!r} file {path} has {actual_nbytes} bytes, expected {expected_nbytes}"
                )

    def validate(self) -> None:
        """Validate descriptor input bindings and raw files for this component dump."""
        descriptor_path = self.session_dir / "aic_batch_io.json"
        if not descriptor_path.exists():
            if self.io_batches:
                raise InputDumpValidationError("Missing aic_batch_io.json descriptor")
            return
        descriptor = json.loads(descriptor_path.read_text())
        io_batches = descriptor.get("IO-files", [])
        if len(io_batches) != len(self.invocations):
            raise InputDumpValidationError(
                f"Descriptor has {len(io_batches)} invocation batches, expected {len(self.invocations)}"
            )
        for invocation_index, entries in enumerate(io_batches):
            invocation = self.invocations[invocation_index]
            input_details = {detail["name"]: detail for detail in invocation.get("input_details", [])}
            self._validate_invocation_inputs(entries, invocation.get("required_inputs", []), input_details)

            descriptor_input_paths = {
                str(entry["path"]) for entry in entries if entry.get("io-direction") == "in" and "path" in entry
            }
            invocation_dir = self.data_dir / f"{invocation_index:05d}"
            actual_input_paths = {
                path.relative_to(self.session_dir).as_posix() for path in invocation_dir.glob("*.raw")
            }
            unreferenced_paths = sorted(actual_input_paths - descriptor_input_paths)
            if unreferenced_paths:
                raise InputDumpValidationError(f"Input files not referenced by descriptor: {unreferenced_paths}")

    def record_schedule_event(self, event: Mapping[str, Any]) -> None:
        runtime_event = _as_jsonable(dict(event))
        runtime_event.setdefault("index", len(self.runtime_schedule_events))
        self.runtime_schedule_events.append(runtime_event)
        self._write_manifest()

    def _output_entry(self, invocation_dir: Path, name: str, shape: Optional[Sequence[int]] = None) -> dict[str, Any]:
        relative_path = Path("data") / invocation_dir.name / f"{_safe_filename(name)}.raw"
        return {
            "path": relative_path.as_posix(),
            "io-direction": "out",
            "elem-size": self._binding_item_size(name),
            "map-to": name,
            "dims": [int(dim) for dim in (shape if shape is not None else self._binding_dims(name))],
        }

    def record_invocation(
        self,
        *,
        kind: str,
        inputs: Mapping[str, np.ndarray],
        output_names: Sequence[str],
        output_shapes: Optional[Mapping[str, Sequence[int]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        exec_obj_index: Optional[int] = None,
    ) -> None:
        invocation_index = len(self.io_batches)
        invocation_dir = self.data_dir / f"{invocation_index:05d}"
        invocation_dir.mkdir()

        entries: list[dict[str, Any]] = []
        recorded_inputs = []
        skipped_inputs = []
        input_details = []
        required_inputs = self._required_input_names(exec_obj_index)
        seen_input_bindings = set()
        for name, value in inputs.items():
            if value is None:
                skipped_inputs.append({"name": str(name), "reason": "value_is_none"})
                continue
            if name not in self.session.binding_index_map:
                skipped_inputs.append({"name": str(name), "reason": "unknown_binding"})
                continue
            name = self._canonical_binding_name(name)
            if name in seen_input_bindings:
                raise InputDumpValidationError(f"Duplicate input binding: {name}")
            seen_input_bindings.add(name)
            array = self._validate_array_for_binding(name, value)
            entry = self._write_input(invocation_dir, name, array)
            entries.append(entry)
            recorded_inputs.append(name)
            input_details.append(self._input_detail(entry, array))

        self._validate_invocation_inputs(entries, required_inputs, {detail["name"]: detail for detail in input_details})

        output_shapes = output_shapes or {}
        recorded_outputs = []
        for name in output_names:
            if name not in self.session.binding_index_map:
                continue
            canonical_name = self._canonical_binding_name(name)
            entries.append(
                self._output_entry(
                    invocation_dir,
                    canonical_name,
                    output_shapes.get(name, output_shapes.get(canonical_name)),
                )
            )
            name = canonical_name
            recorded_outputs.append(name)

        self.io_batches.append(entries)
        invocation_metadata = {
            "index": invocation_index,
            "kind": kind,
            "exec_obj_index": exec_obj_index,
            "inputs": recorded_inputs,
            "required_inputs": required_inputs,
            "outputs": recorded_outputs,
            "input_details": input_details,
        }
        if skipped_inputs:
            invocation_metadata["skipped_inputs"] = skipped_inputs
        if metadata:
            invocation_metadata.update(_as_jsonable(metadata))
        if exec_obj_index in self.pending_slice_configs:
            invocation_metadata["slice_configs"] = self.pending_slice_configs.pop(exec_obj_index)
        self.invocations.append(invocation_metadata)
        self._write_batch_io()
        self.validate()
        self._write_manifest()

    def record_slice_config(
        self,
        *,
        index: int,
        kv_cache_buffers: Sequence[np.ndarray],
        slicing_parameters: Any,
        buff_map: Sequence[tuple[str, int]],
    ) -> None:
        slice_index = sum(len(items) for items in self.pending_slice_configs.values())
        slice_dir = self.data_dir / f"slice_{slice_index:05d}"
        slice_dir.mkdir()
        buffers = []
        for buffer_index, ((name, binding_index), value) in enumerate(zip(buff_map, kv_cache_buffers)):
            array = np.ascontiguousarray(np.asarray(value))
            relative_path = Path("data") / slice_dir.name / f"{buffer_index:05d}_{_safe_filename(name)}.raw"
            array.tofile(self.session_dir / relative_path)
            buffers.append(
                {
                    "name": name,
                    "binding-index": int(binding_index),
                    "path": relative_path.as_posix(),
                    "elem-size": int(array.itemsize),
                    "dims": self._array_dims(array),
                    "dtype": str(array.dtype),
                }
            )
        self.pending_slice_configs.setdefault(index, []).append(
            {
                "slicing_parameters": _as_jsonable(slicing_parameters),
                "buffers": buffers,
            }
        )
        self._write_manifest()

    def _write_batch_io(self) -> None:
        (self.session_dir / "aic_batch_io.json").write_text(json.dumps({"IO-files": self.io_batches}, indent=2))

    def _write_manifest(self) -> None:
        manifest = {
            "version": 1,
            "component": self.component_name,
            "aic_batch_io": "aic_batch_io.json",
            "env_var": _DUMP_ENV_VAR,
            "invocations": self.invocations,
        }
        if self.pending_slice_configs:
            manifest["pending_slice_configs"] = _as_jsonable(self.pending_slice_configs)
        if self.runtime_schedule_events:
            manifest["runtime_schedule"] = {
                "version": 1,
                "events": self.runtime_schedule_events,
            }
        (self.session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _write_aggregate_manifest(self.root)
