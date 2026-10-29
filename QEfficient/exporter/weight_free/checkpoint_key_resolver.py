# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional

import onnx_ir as ir
from torch import nn

from QEfficient.exporter.weight_free.weight_spec import (
    ExternalDataFile,
    TiedWeightAlias,
    WeightSpec,
    WeightSpecInput,
    WeightSpecLocation,
)
from QEfficient.transformers.embeddings.embedding_utils import PooledModel
from QEfficient.utils.checkpoint_utils import checkpoint_root, load_checkpoint_index, resolve_checkpoint_files

_MOE_WEIGHT_LEGACY_SUFFIXES = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj_t",
    "gate_bias": "gate_proj_bias",
    "up_bias": "up_proj_bias",
    "down_bias": "down_proj_bias",
}


_COMPUTED_INITIALIZER_NAMES = {
    "cos_cached",
    "sin_cached",
    "inv_freq",
    "full_attention_sin_cached",
    "full_attention_cos_cached",
    "sliding_attention_sin_cached",
    "sliding_attention_cos_cached",
    "original_inv_freq",
    "embed_positions",
    "embed_scale",
    "position_ids",
    "token_type_ids",
    "_qeff_unit_weight",
    "_mask_causal",
    "_mask_strict",
    "_ones_lower",
    "_eye",
}


def _collect_tied_weights(model: nn.Module, prefix: str = "") -> list[TiedWeightAlias]:
    """Return aliases for tied weights, keyed by the model's own tied-weights contract.

    Uses ``get_expanded_tied_weights_keys`` instead of comparing live module identity
    (``get_input_embeddings()``/``get_output_embeddings()`` against ``named_modules()``)
    so this stays correct even if a module was rebuilt/replaced since the tie was
    established — the mapping comes from ``model._tied_weights_keys``, not from
    whatever object graph happens to exist at export time.
    """
    aliases = []
    get_expanded_tied_weights_keys = getattr(model, "get_expanded_tied_weights_keys", None)
    if get_expanded_tied_weights_keys is not None:
        tied_mapping = get_expanded_tied_weights_keys(all_submodels=True)
        aliases.extend(
            TiedWeightAlias(alias=f"{prefix}{alias}", canonical=canonical) for alias, canonical in tied_mapping.items()
        )

    for child_name, child in model.named_children():
        aliases.extend(_collect_tied_weights(child, prefix=f"{prefix}{child_name}."))

    unique_aliases = {entry.alias: entry for entry in aliases}
    return list(unique_aliases.values())


def _moe_weight_aliases(name: str) -> List[str]:
    """Return equivalent checkpoint aliases for shared MoEWeights parameters."""
    aliases = []
    canonical = name
    if ".moe_block.experts." in name:
        aliases.append(name.replace(".moe_block.experts.", ".experts.", 1))
    for onnx_suffix, checkpoint_suffix in (
        (".moe_block.router.", ".router."),
        (".moe_block.pre_feedforward_layernorm.", ".pre_feedforward_layernorm_2."),
        (".moe_block.post_feedforward_layernorm.", ".post_feedforward_layernorm_2."),
    ):
        if onnx_suffix in name:
            aliases.append(name.replace(onnx_suffix, checkpoint_suffix, 1))
    if ".moe_block.moe_weights." in canonical:
        canonical = canonical.replace(".moe_block.moe_weights.", ".moe_weights.", 1)
        aliases.append(canonical)

    if ".experts.moe_weights." in name:
        canonical = name.replace(".experts.moe_weights.", ".moe_weights.", 1)
        aliases.append(canonical)
    elif ".moe_weights." in name:
        aliases.append(name.replace(".moe_weights.", ".experts.moe_weights.", 1))

    prefix, separator, suffix = canonical.rpartition(".moe_weights.")
    if separator and suffix in _MOE_WEIGHT_LEGACY_SUFFIXES:
        aliases.append(f"{prefix}.experts.{_MOE_WEIGHT_LEGACY_SUFFIXES[suffix]}")

    return aliases


def _router_gate_aliases(name: str) -> List[str]:
    """Return the legacy router/gate spelling for sparse-MoE router weights."""
    if name.endswith(".mlp.gate.weight"):
        return [name[: -len(".mlp.gate.weight")] + ".mlp.router.weight"]
    if name.endswith(".mlp.router.weight"):
        return [name[: -len(".mlp.router.weight")] + ".mlp.gate.weight"]
    return []


def _vision_model_aliases(name: str) -> List[str]:
    """Return equivalent namespaces for VLM vision tower parameters.

    Qwen VL wrappers expose the HF ``visual`` module as ``vision_model`` so
    the component can use the common VLM interface. That changes the
    exported module name, while the checkpoint retains the HF ``visual``
    namespace.
    """
    if name.startswith("model.vision_model."):
        return [name.replace("model.vision_model.", "model.visual.", 1)]
    if name.startswith("model.visual."):
        return [name.replace("model.visual.", "model.vision_model.", 1)]
    return []


def _find_checkpoint_key(candidates: List[str], checkpoint_index: Dict[str, str], onnx_name: str) -> Optional[str]:
    """Return the unique matching checkpoint key, or fail on ambiguous matches."""
    seen: set = set()
    matches = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in checkpoint_index:
            matches.append(candidate)
        for alias in _moe_weight_aliases(candidate):
            if alias in seen:
                continue
            seen.add(alias)
            if alias in checkpoint_index:
                matches.append(alias)
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous checkpoint key for ONNX initializer '{onnx_name}': matched {matches}. "
            "Checkpoint transforms must produce unambiguous keys."
        )
    return matches[0] if matches else None


def _is_computed_initializer(name: str) -> bool:
    """Return True for generated constants that are not stored in HF checkpoints."""
    return name.rsplit(".", 1)[-1] in _COMPUTED_INITIALIZER_NAMES


def _is_non_persistent_buffer(model: nn.Module, name: str) -> bool:
    """Return True when a model buffer is explicitly excluded from checkpoints."""
    parent_name, separator, buffer_name = name.rpartition(".")
    if not separator:
        parent = model
    else:
        try:
            parent = model.get_submodule(parent_name)
        except AttributeError:
            return False
    return buffer_name in getattr(parent, "_non_persistent_buffers_set", set())


def find_checkpoint_key(
    onnx_name: str,
    checkpoint_index: Dict[str, str],
    backbone: nn.Module,
    active_transform=None,
) -> Optional[str]:
    """Resolve an ONNX initializer name to its safetensors checkpoint key.

    Resolution order:
    1. Universal HF prefix rules (base_model., base_model_prefix).
    2. Legacy sparse-MoE router/gate spelling fallback.
    3. Transform-specific explicit mapping via resolve_onnx_key().
    4. Legacy MoE weight aliases fallback for old checkpoints.
    """
    # 1. Universal HF prefix rules
    candidates = [onnx_name, *_vision_model_aliases(onnx_name)]
    stripped = onnx_name.removeprefix("base_model.")
    candidates.append(stripped)
    # Component wrappers can add a leading ``model.`` module while the HF
    # checkpoint keeps top-level parameters (for example ``lm_head.weight``)
    # at the root.
    if stripped.startswith("model."):
        candidates.append(stripped.removeprefix("model."))
    else:
        # Some component wrappers expose the HF backbone below ``model`` in
        # the exported module, while the resolver receives names relative to
        # that component (for example ``language_model.embed_tokens``).
        candidates.append(f"model.{stripped}")

    collapsed = stripped
    while collapsed.startswith("model.model."):
        collapsed = collapsed.removeprefix("model.")
        candidates.append(collapsed)

    collapsed = stripped
    while collapsed.startswith("model."):
        collapsed = collapsed.removeprefix("model.")
        candidates.append(collapsed)

    for candidate in list(candidates):
        if ".language_model." in candidate:
            candidates.append(candidate.replace(".language_model.", ".", 1))
        elif candidate.startswith("language_model."):
            language_model_key = candidate.removeprefix("language_model.")
            candidates.append(language_model_key)
            candidates.append(f"model.{language_model_key}")
        if ".vision_model." in candidate:
            candidates.append(candidate.replace(".vision_model.", ".visual.", 1))
            candidates.append(candidate.replace(".vision_model.", ".vision_tower.", 1))
        elif candidate.startswith("vision_model."):
            candidates.append(candidate.replace("vision_model.", "visual.", 1))
            candidates.append(candidate.replace(".vision_model.", ".vision_tower.", 1))
        if ".visual." in candidate:
            candidates.append(candidate.replace(".visual.", ".vision_model.", 1))
            candidates.append(candidate.replace(".visual.", ".vision_tower.", 1))
        elif candidate.startswith("visual."):
            candidates.append(candidate.replace("visual.", "vision_model.", 1))
            candidates.append(candidate.replace(".visual.", ".vision_tower.", 1))
        if candidate.startswith("model.visual."):
            candidates.append(candidate.replace("model.visual.", "model.language_model.visual.", 1))
        elif candidate.startswith("visual."):
            candidates.append(f"model.language_model.{candidate}")

    if stripped.startswith("model.lm_head."):
        candidates.append(stripped.removeprefix("model."))
    if stripped.startswith("language_model."):
        candidates.append(f"model.{stripped}")

    prefix = getattr(backbone, "base_model_prefix", "")
    if prefix:
        candidates.append(f"{prefix}.{stripped}")

    if prefix and stripped.startswith(f"{prefix}."):
        unprefixed = stripped[len(f"{prefix}.") :]
        candidates.append(unprefixed)
        candidates.append(f"{prefix}.language_model.{unprefixed}")

    key = _find_checkpoint_key(candidates, checkpoint_index, onnx_name)
    if key is not None:
        return key

    # 2. Keep the historic sparse-MoE router/gate compatibility after exact lookup.
    router_gate_candidates = [alias for candidate in candidates for alias in _router_gate_aliases(candidate)]
    key = _find_checkpoint_key(router_gate_candidates, checkpoint_index, onnx_name)
    if key is not None:
        return key

    # 3. Transform-specific explicit mapping
    if active_transform is not None and hasattr(active_transform, "resolve_onnx_key"):
        for candidate in [onnx_name, *_router_gate_aliases(onnx_name)]:
            key = active_transform.resolve_onnx_key(candidate, checkpoint_index)
            if key is not None:
                return key

    # 4. Legacy MoE weight aliases (kept for old prepared checkpoints)
    return _find_checkpoint_key(
        [alias for c in candidates for alias in _moe_weight_aliases(c)],
        checkpoint_index,
        onnx_name,
    )


def _named_state_keys(module: nn.Module, iterator_name: str) -> set[str]:
    iterator = getattr(module, iterator_name)
    try:
        return {name for name, _ in iterator(remove_duplicate=False)}
    except TypeError:
        return {name for name, _ in iterator()}


def _safetensors_shapes(checkpoint_file: str) -> Dict[str, List[int]]:
    """Return ``{key: shape}`` from a safetensors header without reading tensor data."""
    return {key: shape for key, (shape, _) in _safetensors_tensor_info(checkpoint_file).items()}


def _safetensors_tensor_info(checkpoint_file: str) -> Dict[str, tuple[List[int], str]]:
    """Return ``{key: (shape, dtype)}`` from a safetensors header without reading tensor data."""
    with open(checkpoint_file, "rb") as handle:
        (header_size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_size))
    return {
        key: ([int(dim) for dim in entry["shape"]], entry["dtype"])
        for key, entry in header.items()
        if key != "__metadata__"
    }


_SAFETENSORS_TO_ONNX_DTYPE = {
    "BOOL": ir.DataType.BOOL,
    "U8": ir.DataType.UINT8,
    "I8": ir.DataType.INT8,
    "U16": ir.DataType.UINT16,
    "I16": ir.DataType.INT16,
    "U32": ir.DataType.UINT32,
    "I32": ir.DataType.INT32,
    "U64": ir.DataType.UINT64,
    "I64": ir.DataType.INT64,
    "F16": ir.DataType.FLOAT16,
    "BF16": ir.DataType.BFLOAT16,
    "F32": ir.DataType.FLOAT,
    "F64": ir.DataType.DOUBLE,
}


def _safetensors_onnx_dtype(dtype: str) -> ir.DataType:
    try:
        return _SAFETENSORS_TO_ONNX_DTYPE[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported safetensors dtype '{dtype}' in weight-free export") from exc


def _promote_initializer(
    model_ir,
    name: str,
    init_value,
    source_dtype: ir.DataType,
) -> None:
    """Promote one initializer, inserting a source-to-graph dtype cast when needed."""
    graph = model_ir.graph
    target_dtype = init_value.dtype
    graph_input = ir.Value(
        name=name,
        shape=init_value.shape,
        type=ir.TensorType(source_dtype),
    )

    if source_dtype == target_dtype:
        if hasattr(init_value, "replace_all_uses_with"):
            init_value.replace_all_uses_with(graph_input)
        graph.inputs.append(graph_input)
    else:
        cast_name = f"{name}__qeff_cast_{source_dtype.name.lower()}_to_{target_dtype.name.lower()}"
        cast_output = ir.Value(
            name=cast_name,
            shape=init_value.shape,
            type=ir.TensorType(target_dtype),
        )
        cast_node = ir.Node(
            domain="",
            op_type="Cast",
            inputs=[graph_input],
            attributes={"to": ir.AttrInt64("to", int(target_dtype))},
            outputs=[cast_output],
            version=13,
            name=cast_name,
        )
        consumers = init_value.consumers() if hasattr(init_value, "consumers") else ()
        if hasattr(init_value, "replace_all_uses_with"):
            init_value.replace_all_uses_with(cast_output)
        graph.inputs.append(graph_input)
        if consumers:
            graph.insert_before(min(consumers, key=graph.index), cast_node)
        else:
            graph.append(cast_node)

    del graph.initializers[name]


def _check_stored_shape(name: str, graph_shape, checkpoint_key: str, stored_shape: List[int]) -> None:
    """Fail the export when a graph weight input cannot bind to its stored tensor."""
    try:
        dims = [int(dim) for dim in graph_shape]
    except (TypeError, ValueError):
        return
    if dims != stored_shape:
        raise ValueError(
            f"Weight input '{name}' expects shape {dims}, but checkpoint tensor '{checkpoint_key}' has shape "
            f"{stored_shape}. Weight-free export requires the graph to consume weights in their stored layout; "
            "express any layout change as graph ops instead of rewriting the checkpoint."
        )


def promote_initializers_and_build_spec(onnx_program, model_ref: str, model_name: str, qeff_model) -> WeightSpec:
    """Promote ONNX initializers to graph inputs and create the weight spec.

    Parameters
    ----------
    onnx_program
        Dynamo ONNX export program whose graph initializers should be promoted.
    model_ref : str
        Checkpoint directory or model reference used to resolve external weights.
    model_name : str
        Name stored in the emitted weight spec.
    qeff_model
        QEfficient model wrapper whose parameters and buffers define promotable weights.

    Returns
    -------
    WeightSpec
        Specification mapping promoted ONNX inputs to checkpoint tensor locations.
    """
    model_ir = onnx_program.model
    parameter_names = {name for name, _ in qeff_model.model.named_parameters(remove_duplicate=False)}
    buffer_names = {name for name, _ in qeff_model.model.named_buffers(remove_duplicate=False)}
    model_names = parameter_names | buffer_names
    tied_weight_map = {entry.alias: entry.canonical for entry in _collect_tied_weights(qeff_model.model)}
    # named_parameters()/named_buffers() dedup tied tensors by identity, so a tied alias
    # (e.g. lm_head.weight when tie_word_embeddings=True) is absent from model_names even
    # though torch.export still emits a distinct ONNX initializer for it. Add tied aliases
    # explicitly so they aren't skipped below and reach the tied_weight_map redirect.
    model_names.update(tied_weight_map.keys())
    checkpoint_files = resolve_checkpoint_files(model_ref)
    root = checkpoint_root(model_ref, checkpoint_files)
    checkpoint_index = load_checkpoint_index(checkpoint_files)
    relative_checkpoint_files = [
        ExternalDataFile(
            path=str(Path(checkpoint_file).relative_to(root)) if root is not None else Path(checkpoint_file).name,
            format="safetensors",
        )
        for checkpoint_file in checkpoint_files
    ]
    backbone = qeff_model.model.base_model if isinstance(qeff_model.model, PooledModel) else qeff_model.model

    # Identify the active layout transform from the prepared checkpoint manifest.
    # The manifest stores the active layout transform ID during centralized finalization.
    # Reading from the manifest avoids re-running detection on the prepared checkpoint
    # (which would fail — the prepared checkpoint has canonical output keys like
    # moe_weights.gate, not the original per-expert keys that trigger detection).

    from QEfficient.base.checkpoint_transforms import (
        CHECKPOINT_PREPARED_MANIFEST,
        _find_transform_by_id,
    )

    active_transform = None
    manifest_path = Path(model_ref) / CHECKPOINT_PREPARED_MANIFEST
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            transform_id = manifest.get("active_group", "none")
            if transform_id and transform_id != "none":
                active_transform = _find_transform_by_id(
                    transform_id,
                    getattr(qeff_model, "_checkpoint_transforms", []),
                )
        except (OSError, json.JSONDecodeError):
            pass  # no manifest → active_transform stays None, fallback to legacy aliases

    promoted_inputs: List[WeightSpecInput] = []
    stored_tensor_info: Dict[str, Dict[str, tuple[List[int], str]]] = {}

    for name, init_value in list(model_ir.graph.initializers.items()):
        if name not in model_names:
            continue

        onnx_name = tied_weight_map.get(name, name)
        checkpoint_key = find_checkpoint_key(onnx_name, checkpoint_index, backbone, active_transform)
        if checkpoint_key is None:
            if _is_computed_initializer(onnx_name) or _is_non_persistent_buffer(qeff_model.model, name):
                continue
            raise ValueError(
                f"Could not resolve model initializer '{name}' to a safetensors checkpoint key "
                f"(resolved name: '{onnx_name}', model: '{model_ref}'). "
                "Only explicitly classified computed initializers may remain embedded in the ONNX model."
            )

        checkpoint_file = checkpoint_index[checkpoint_key]
        if checkpoint_file not in stored_tensor_info:
            stored_tensor_info[checkpoint_file] = _safetensors_tensor_info(checkpoint_file)
        try:
            stored_shape, stored_dtype = stored_tensor_info[checkpoint_file][checkpoint_key]
        except KeyError as exc:
            raise ValueError(
                f"Checkpoint file '{checkpoint_file}' does not contain tensor '{checkpoint_key}' referenced by "
                f"ONNX initializer '{name}'."
            ) from exc
        _check_stored_shape(name, init_value.shape, checkpoint_key, stored_shape)
        _promote_initializer(model_ir, name, init_value, _safetensors_onnx_dtype(stored_dtype))
        promoted_inputs.append(
            WeightSpecInput(
                name=name,
                location=WeightSpecLocation(file=checkpoint_files.index(checkpoint_file), key=checkpoint_key),
            )
        )

    return WeightSpec(
        model_name=model_name,
        model_id=model_ref,
        external_data_root=str(root) if root is not None else None,
        files=relative_checkpoint_files,
        inputs=promoted_inputs,
    )
