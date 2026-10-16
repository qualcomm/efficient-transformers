# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import inspect
import re
import warnings
from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
import torch.nn as nn
from torch.export import Dim

from QEfficient.base.onnx_transforms import (
    CustomOpTransform,
    LocalizeFunctionReduceSumAxesTransform,
    PreserveNestedCacheRetainedStateTransform,
    RenameFunctionOutputsTransform,
    RenameRepeatedSubgraphTransform,
    RenameWsubNodesTransform,
)
from QEfficient.transformers.cache_utils import InvalidIndexProvider
from QEfficient.utils.cache import QEFF_HOME
from QEfficient.utils.constants import (
    _KNOWN_DECODER_LAYER_ATTR_PATHS,
    _KNOWN_DECODER_LAYER_SUFFIXES,
    DYNAMO_DIM_MAX_BATCH_SIZE,
    DYNAMO_DIM_MIN_COMP_CTX_LENGTHS,
)
from QEfficient.utils.hash_utils import create_export_hash
from QEfficient.utils.logging_utils import QEFFLogger, log_api_arguments
from QEfficient.utils.runtime_requirements import validate_dynamo_export_requirements
from QEfficient.utils.torch_patches import (
    apply_torch_patches,
    temporarily_enable_nested_compile_regions,
    undo_torch_patches,
)

logger = QEFFLogger.get_logger("INFRA")
_EXPORT_FROM_COMPILE = ContextVar("export_from_compile", default=False)


@contextmanager
def export_from_compile():
    """Suppress the direct-export deprecation warning during compilation."""
    token = _EXPORT_FROM_COMPILE.set(True)
    try:
        yield
    finally:
        _EXPORT_FROM_COMPILE.reset(token)


def reorder_inputs_by_signature(model, example_inputs, dynamic_shapes=None):
    """Reorder example_inputs (and optional dynamic_shapes) to match model.forward signature.

    torch.export requires inputs and dynamic_shapes to follow the forward parameter order
    so that each shape constraint binds to the correct input tensor.
    """
    sig_keys = list(inspect.signature(model.forward).parameters.keys())
    sig_key_set = set(sig_keys)
    ordered_inputs, ordered_shapes = {}, {}
    for k in sig_keys:
        if k in example_inputs:
            ordered_inputs[k] = example_inputs[k]
        if dynamic_shapes is not None and k in dynamic_shapes:
            ordered_shapes[k] = dynamic_shapes[k]
    reordered_inputs = {**ordered_inputs, **{k: v for k, v in example_inputs.items() if k not in sig_key_set}}
    if dynamic_shapes is not None:
        reordered_shapes = {**ordered_shapes, **{k: v for k, v in dynamic_shapes.items() if k not in sig_key_set}}
        return reordered_inputs, reordered_shapes
    return reordered_inputs, None


def build_dynamo_export_kwargs(export_kwargs):
    """Prepare export kwargs for dynamo (torch.export) path.

    Sets dynamo=True, default report/optimize flags, and injects DYNAMO_CUSTOM_OP_TABLE
    into custom_translation_table. Returns a new dict; does not mutate the input.
    """
    from QEfficient.customop.dynamo_ops import DYNAMO_CUSTOM_OP_TABLE
    from QEfficient.utils import constants

    kwargs = dict(export_kwargs)
    kwargs.setdefault("report", False)
    kwargs.setdefault("optimize", False)
    kwargs["dynamo"] = True
    kwargs["opset_version"] = constants.ONNX_DYNAMO_EXPORT_OPSET
    kwargs["custom_translation_table"] = {
        **(kwargs.pop("custom_translation_table", None) or {}),
        **DYNAMO_CUSTOM_OP_TABLE,
    }
    return kwargs


def convert_dynamic_axes_to_dynamic_shapes(
    dynamic_axes: Dict[str, Dict[int, str]],
    model_config=None,
) -> Dict[str, Any]:
    """
    Convert ONNX dynamic_axes format to torch.export dynamic_shapes format.

    Parameters
    ----------
    dynamic_axes : Dict[str, Dict[int, str]]
        ONNX-style axes map, e.g.
        {"input_ids": {0: "batch_size", 1: "seq_len"},
         "past_key.0": {0: "batch_size", 2: "ctx_len"}, ...}
    model_config : optional
        The HuggingFace model config object (model.config). Used to read:
          - max_position_embeddings  -> upper bound for seq_len / ctx_len dims
          - sliding_window           -> upper bound for sliding_window dim
          - model_type               -> controls batch_min and whether
                                       compressed_kvs reconstruction runs
        When None, safe defaults are used for all bounds and model-type-specific
        passes are skipped.

    Returns
    -------
    Dict[str, Any]
        torch.export dynamic_shapes dict with Dim objects, suitable for
        torch.onnx.export(dynamic_shapes=...).
    """
    max_seq_len = getattr(model_config, "max_position_embeddings", 1024)
    model_type = getattr(model_config, "model_type", None)
    batch_min = 1 if model_type == "gpt_oss" else 2

    dim_registry: Dict[str, Any] = {}

    def resolve_dim(dim_name: str):
        if dim_name not in dim_registry:
            if dim_name == "batch_size":
                dim_registry[dim_name] = Dim("batch_size", min=batch_min, max=DYNAMO_DIM_MAX_BATCH_SIZE)
            elif dim_name == "full_batch_size":
                # CB pool capacity; different min prevents torch.export collapsing it with batch_size.
                dim_registry[dim_name] = Dim("full_batch_size", min=batch_min + 1, max=DYNAMO_DIM_MAX_BATCH_SIZE)
            elif "seq_len" in dim_name:
                dim_registry[dim_name] = Dim("seq_len", min=2, max=max_seq_len)
            elif "comp_ctx_lengths" in dim_name:
                dim_registry[dim_name] = Dim("comp_ctx_lengths", min=DYNAMO_DIM_MIN_COMP_CTX_LENGTHS, max=max_seq_len)
            elif "ctx_len" in dim_name:
                dim_registry[dim_name] = Dim("ctx_len", min=2, max=max_seq_len)
            elif "sliding_window" in dim_name:
                dim_registry[dim_name] = Dim(
                    "sliding_window",
                    min=2,
                    max=getattr(model_config, "sliding_window", max_seq_len),
                )
            else:
                dim_registry[dim_name] = Dim.DYNAMIC
        return dim_registry[dim_name]

    dynamic_shapes: Dict[str, Any] = {}
    past_keys: Dict[int, Any] = {}
    past_values: Dict[int, Any] = {}
    compressed_kv_layers: Dict[int, Any] = {}
    k_pe_layers: Dict[int, Any] = {}

    for input_name, axes_map in dynamic_axes.items():
        resolved = {axis_idx: resolve_dim(dim_name) for axis_idx, dim_name in axes_map.items()}
        if input_name.startswith("past_key."):
            past_keys[int(input_name.split(".")[1])] = resolved
        elif input_name.startswith("past_value."):
            past_values[int(input_name.split(".")[1])] = resolved
        elif input_name.startswith("compressed_kv."):
            compressed_kv_layers[int(input_name.split(".")[1])] = resolved
        elif input_name.startswith("k_pe."):
            k_pe_layers[int(input_name.split(".")[1])] = resolved
        else:
            dynamic_shapes[input_name] = resolved

    if past_keys or past_values:
        max_layer = max(list(past_keys.keys()) + list(past_values.keys()))
        dynamic_shapes["past_key_values"] = [
            [past_keys.get(i, {}), past_values.get(i, {})] for i in range(max_layer + 1)
        ]

    if compressed_kv_layers or k_pe_layers:
        max_layer = max(list(compressed_kv_layers.keys()) + list(k_pe_layers.keys()))
        dynamic_shapes["compressed_kvs"] = [
            (compressed_kv_layers.get(i, {}), k_pe_layers.get(i, {})) for i in range(max_layer + 1)
        ]

    return dynamic_shapes


def _resolve_attr_path(root, attr_path):
    current = root
    for attr in attr_path.split("."):
        if current is None or not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def _extract_repeated_block_class(candidate):
    if isinstance(candidate, (nn.ModuleList, nn.Sequential, list, tuple)):
        modules = [item for item in candidate if isinstance(item, nn.Module)]
        if len(modules) < 2:
            return None
        first_cls = modules[0].__class__
        if all(isinstance(item, first_cls) for item in modules):
            return first_cls
    return None


def _discover_submodule_classes_for_export(model):
    discovered = set()

    for attr_path in _KNOWN_DECODER_LAYER_ATTR_PATHS:
        candidate = _resolve_attr_path(model, attr_path)
        repeated_cls = _extract_repeated_block_class(candidate)
        if repeated_cls is not None:
            discovered.add(repeated_cls)

    if discovered:
        return discovered

    if not hasattr(model, "named_modules"):
        return discovered

    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not module_name.endswith(_KNOWN_DECODER_LAYER_SUFFIXES):
            continue
        repeated_cls = _extract_repeated_block_class(module)
        if repeated_cls is None:
            continue
        cls_name = repeated_cls.__name__.lower()
        if any(token in cls_name for token in ("layer", "block", "decoder", "encoder")):
            discovered.add(repeated_cls)

    return discovered


def get_decoder_layer_classes_for_export(model):
    get_submodules_for_export = getattr(model, "get_submodules_for_export", None)
    if get_submodules_for_export is not None:
        try:
            submodule_classes = get_submodules_for_export()
            if submodule_classes:
                return {cls for cls in submodule_classes if inspect.isclass(cls)}
        except Exception as exc:
            logger.warning(
                f"get_submodules_for_export failed for {model.__class__.__name__}: "
                f"{type(exc).__name__}: {exc}. Falling back to auto-discovery."
            )

    discovered = _discover_submodule_classes_for_export(model)
    if discovered:
        logger.info(
            "Auto-discovered repeated submodule classes for export: "
            + ", ".join(sorted(cls.__name__ for cls in discovered))
        )
        return discovered
    return []


def _iter_onnx_nodes(nodes):
    """Yield graph nodes recursively, including nodes inside nested ONNX subgraphs."""
    for node in nodes:
        yield node
        for attr in node.attribute:
            if attr.HasField("g"):
                yield from _iter_onnx_nodes(attr.g.node)


def _function_call_counts(onnx_model, function_names: set[str]) -> Counter:
    """Count call sites for the selected local function names in an ONNX model."""
    nodes = list(_iter_onnx_nodes(onnx_model.graph.node))
    for function_proto in onnx_model.functions:
        nodes.extend(_iter_onnx_nodes(function_proto.node))
    return Counter(node.op_type for node in nodes if node.op_type in function_names)


def _proxy_subfunction_validation_classes(qeff_model) -> set[type[nn.Module]]:
    """Return language-layer classes that need repeated proxy subfunction calls."""
    if qeff_model.__class__.__name__ == "QEffVisionEncoderForTextImageToTextModel":
        return set()

    model = qeff_model.model
    get_language_decoder = getattr(model, "get_qeff_language_decoder", None)
    if callable(get_language_decoder):
        try:
            language_decoder = get_language_decoder()
            language_classes = get_decoder_layer_classes_for_export(language_decoder)
            if language_classes:
                return set(language_classes)
        except Exception as exc:
            logger.warning(
                f"get_qeff_language_decoder failed for {model.__class__.__name__}: "
                f"{type(exc).__name__}: {exc}. Falling back to model-level subfunction validation."
            )

    submodule_classes = get_decoder_layer_classes_for_export(model)
    return {cls for cls in submodule_classes if "vision" not in cls.__name__.lower()}


def _validate_proxy_subfunction_calls(qeff_model, onnx_path) -> None:
    """Ensure proxy decoder subfunctions are repeated enough to be useful performance proxies."""
    if not getattr(qeff_model, "_enable_proxy", False):
        return

    decoder_layer_classes = _proxy_subfunction_validation_classes(qeff_model)
    if not decoder_layer_classes:
        return

    target_classnames = {cls.__name__ for cls in decoder_layer_classes}
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    function_names = {
        function_proto.name
        for function_proto in onnx_model.functions
        if any(classname in function_proto.name for classname in target_classnames)
    }
    if not function_names:
        raise RuntimeError(
            "Proxy ONNX subfunction validation failed: no decoder-layer subfunctions were found "
            f"for expected classes {sorted(target_classnames)}."
        )

    call_counts = _function_call_counts(onnx_model, function_names)
    single_call_functions = {
        name: call_counts.get(name, 0) for name in sorted(function_names) if call_counts.get(name, 0) <= 1
    }
    if single_call_functions:
        raise RuntimeError(
            "Proxy ONNX subfunction validation failed: each decoder-layer subfunction must be invoked more "
            "than once for a faithful compiler performance proxy. "
            f"Observed call counts: {single_call_functions}."
        )


def export_wrapper(func):
    """
    Decorator for export methods that orchestrates the complete export lifecycle.

    Responsibilities:
    1. Prepare export directory structure
    2. Generate reproducible hash for export configuration
    3. Setup ONNX subfunction environment (if enabled)
    4. Execute the wrapped export function
    5. Cleanup subfunction environment (if enabled)
    6. Save export metadata

    Args:
        func: The export method to wrap (typically _export)

    Returns:
        Wrapped function with complete export lifecycle management
    """

    def wrapper(self, *args, **kwargs):
        if not _EXPORT_FROM_COMPILE.get():
            warnings.warn(
                "Direct .export() is deprecated. Use .compile() to export and compile with the complete configuration.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Extract flags
        dynamo = kwargs.get("dynamo", False)
        if dynamo:
            kwargs["dynamo"] = True
        use_onnx_subfunctions = kwargs.pop("use_onnx_subfunctions", False)

        if dynamo:
            validate_dynamo_export_requirements("dynamo=True")
            # Resolve dynamic_shapes from dynamic_axes before the hash so the hash captures
            # the actual shape constraints.
            dynamic_axes = kwargs.get("dynamic_axes")
            if dynamic_axes is not None:
                model_config = getattr(self.model, "config", None)
                kwargs["dynamic_shapes"] = convert_dynamic_axes_to_dynamic_shapes(dynamic_axes, model_config)

        # Cache probe flag (used for layerwise inspection runs)
        cache_probe = kwargs.pop("_layerwise_cache_probe", False)

        # Default context managers and state trackers
        export_context = nullcontext()
        subfunction_state = None

        # 1. Setup the requested export mode
        if use_onnx_subfunctions:
            # Handles both Path 4 (dynamo + subfunctions) and Path 2 (TorchScript + subfunctions).
            args, kwargs, subfunction_state = _setup_onnx_subfunctions(self, args, kwargs, dynamo=dynamo)
            if dynamo:
                # Wrap only decoder layers (not top-level model) — wrapping top-level breaks dynamic_shapes validation.
                target_classes = subfunction_state.get("decoder_layer_classes") or None
                export_context = temporarily_enable_nested_compile_regions(self.model, target_classes=target_classes)

        # 2. Prepare export directory
        export_dir = _prepare_export_directory(self, kwargs)

        # 3. Generate hash and finalize export directory path
        export_hash, filtered_hash_params = _generate_export_hash(self, args, kwargs, func)
        log_api_arguments(
            "export",
            self.__class__.__name__,
            {"export_hash": export_hash, "hash_params": filtered_hash_params},
        )
        export_dir = export_dir.with_name(export_dir.name + "-" + export_hash)
        kwargs["export_dir"] = export_dir
        self.export_hash = export_hash
        self._export_cache_hit = False

        # Re-inject cache probe flag if needed
        if cache_probe:
            kwargs["_layerwise_cache_probe"] = True

        try:
            # 4. Execute the actual export
            # Keep each decoder layer as a repeated_subgraph; inline_single_use_invoke_subgraph=True
            # would inline single-use layers before ONNX translation, losing _RetainedState wiring.
            dynamo_patch = (
                torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
                if use_onnx_subfunctions and dynamo
                else nullcontext()
            )
            try:
                with export_context:
                    with dynamo_patch:
                        onnx_path = func(self, *args, **kwargs)
            except Exception as export_exc:
                QEFFLogger.log_api_failure("export", self.__class__.__name__, export_exc)
                if use_onnx_subfunctions and dynamo:
                    raise RuntimeError(
                        "Export failed with dynamo=True and use_onnx_subfunctions=True "
                        "while nested compile regions were enabled for repeated-subgraph "
                        f"extraction ({type(export_exc).__name__}: {export_exc}). "
                        "Retry export with use_onnx_subfunctions=False for this model/runtime."
                    ) from export_exc
                raise

            if use_onnx_subfunctions:
                _validate_proxy_subfunction_calls(self, onnx_path)

            # 5. Save export metadata (skip when running cache probe)
            if not cache_probe:
                _save_export_metadata(export_dir, filtered_hash_params)

        finally:
            # 6. Always cleanup subfunctions if they were setup
            if use_onnx_subfunctions:
                _cleanup_onnx_subfunctions(self, subfunction_state)

        if getattr(self, "_export_cache_hit", False):
            QEFFLogger.log_event(
                "milestone",
                self.__class__.__name__,
                "ONNX export skipped (cached ONNX).",
                api="export",
                milestone=QEFFLogger.MILESTONE_EXPORT_SKIPPED,
            )
        else:
            QEFFLogger.log_event(
                "milestone",
                self.__class__.__name__,
                "ONNX export completed.",
                api="export",
                milestone=QEFFLogger.MILESTONE_EXPORT_COMPLETE,
            )
        return onnx_path

    return wrapper


def _prepare_export_directory(qeff_model, kwargs) -> Path:
    """
    Prepare and return the base export directory path.

    Args:
        qeff_model: The QEff model instance
        kwargs: Keyword arguments containing optional export_dir

    Returns:
        Path object for the base export directory
    """
    export_dir = kwargs.get("export_dir", None)
    parent_dir = qeff_model.model_architecture or qeff_model.model_name
    return Path(export_dir or (QEFF_HOME / parent_dir / qeff_model.model_name))


def _generate_export_hash(qeff_model, args, kwargs, func):
    """
    Generate export hash from model parameters and export arguments.

    The hash ensures reproducibility and prevents conflicts between
    different export configurations.

    Args:
        qeff_model: The QEff model instance
        args: Positional arguments to the export function
        kwargs: Keyword arguments to the export function
        func: The export function being wrapped

    Returns:
        Tuple of (export_hash: str, filtered_hash_params: dict)
    """
    original_sig = inspect.signature(func)
    params = list(original_sig.parameters.values())[1:]  # Skip 'self'
    new_sig = inspect.Signature(params)
    bound_args = new_sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    all_args = bound_args.arguments
    if func.__name__ == "_export_layerwise":
        export_kwargs = dict(all_args.get("export_kwargs") or {})
        export_kwargs["_qeff_layerwise_export"] = True
        all_args["export_kwargs"] = export_kwargs

    # Use the model's current configuration for hashing to ensure any post-load modifications are captured
    # TODO: Replace with get_model_config property of modeling classes and remove the if-else
    # Determine the config dict to use, preferring .to_diff_dict() if available
    if hasattr(qeff_model.model, "config") and hasattr(qeff_model.model.config, "to_diff_dict"):
        config_val = qeff_model.model.config.to_diff_dict()
    elif hasattr(qeff_model.model, "model") and hasattr(qeff_model.model.model.config, "to_diff_dict"):
        config_val = qeff_model.model.model.config.to_diff_dict()
    else:
        config_val = qeff_model.model.config

    copy_of_hash_params = copy.deepcopy(qeff_model.hash_params)
    copy_of_hash_params.update(
        {
            "config": config_val,
            "use_onnx_subfunctions": getattr(qeff_model, "_use_onnx_subfunctions", False),
            "onnx_transform_version": 1,
            "dynamo": all_args.get("dynamo", False),
        }
    )
    if getattr(qeff_model, "_weight_free", False):
        copy_of_hash_params["weight_free"] = True
    if getattr(qeff_model, "_use_onnx_subfunctions", False):
        copy_of_hash_params["onnx_subfunction_version"] = 3
    # Generate hash from relevant parameters
    export_hash, filtered_hash_params = create_export_hash(
        model_params=copy_of_hash_params,
        output_names=all_args.get("output_names"),
        dynamic_axes=all_args.get("dynamic_axes"),
        blocking_kwargs=all_args.get("blocking_kwargs", None),
        dynamic_shapes=all_args.get("dynamic_shapes"),
        export_kwargs=all_args.get("export_kwargs", None),
        onnx_transform_kwargs=all_args.get("onnx_transform_kwargs", None),
    )

    return export_hash, filtered_hash_params


def _setup_onnx_subfunctions(qeff_model, args, kwargs, dynamo=False):
    """
    Setup ONNX subfunction export environment.

    This function prepares the model and environment for exporting with
    ONNX subfunctions enabled. It:
    - Applies necessary torch patches (TorchScript path only)
    - Modifies output names for subfunction compatibility
    - Adds subfunction-specific ONNX transforms
    - Updates export kwargs with module classes

    Args:
        qeff_model: The QEff model instance.
        args: Positional export arguments.
        kwargs: Export keyword arguments (modified in-place).
        dynamo: Whether the dynamo exporter is active. Passed explicitly by the
            caller so this function does not need to read it from kwargs.

    Returns:
        Updated args/kwargs plus cleanup state used to restore the original
        ONNX transform list after export.
    """
    warnings.warn(
        "The subfunction feature is experimental. Please note that using compile "
        "consecutively with and without subfunction may produce inconsistent results."
    )
    orig_use_onnx_subfunctions = getattr(qeff_model, "_use_onnx_subfunctions", False)
    orig_hash_use_subfunctions = qeff_model.hash_params.get("use_onnx_subfunctions", False)
    orig_hash_subfunction_version = qeff_model.hash_params.get("onnx_subfunction_version")
    qeff_model._use_onnx_subfunctions = True
    qeff_model.hash_params["use_onnx_subfunctions"] = True
    qeff_model.hash_params["onnx_subfunction_version"] = 3
    # TorchScript patches are irrelevant on the dynamo path.
    if not dynamo:
        apply_torch_patches()
    InvalidIndexProvider.SUBFUNC_ENABLED = True

    # TorchScript renames _RetainedState → _InternalRetainedState; dynamo keeps _RetainedState for PreserveNestedCacheRetainedStateTransform.
    if not dynamo:
        if "output_names" in kwargs:
            kwargs["output_names"] = [
                re.sub("_RetainedState", "_InternalRetainedState", name)
                if name.endswith("_RetainedState")
                and (
                    "key" in name
                    or "value" in name
                    or "compressed_kv" in name
                    or "k_pe" in name
                    or "conv" in name
                    or "recurrent" in name
                )
                else name
                for name in kwargs["output_names"]
            ]
        else:
            warnings.warn(
                "ONNX subfunctions are enabled, but no retained-state output names were found to rewrite. "
                "Ensure `output_names` includes key/value retained states if subfunction compatibility is required."
            )

    # Work on an instance-local copy and restore it in cleanup. Without this,
    # failed or repeated subfunction exports can leak transforms into later exports.
    original_transforms = list(qeff_model._onnx_transforms)
    qeff_model._onnx_transforms = list(original_transforms)

    # Add subfunction-specific ONNX transforms based on export path
    if dynamo:
        # Dynamo: PreserveNestedCacheRetainedStateTransform + RenameRepeatedSubgraphTransform.
        if PreserveNestedCacheRetainedStateTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(PreserveNestedCacheRetainedStateTransform)
        if RenameRepeatedSubgraphTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(RenameRepeatedSubgraphTransform)
    else:
        # TorchScript: RenameFunctionOutputsTransform + CustomOpTransform.
        if RenameFunctionOutputsTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(RenameFunctionOutputsTransform)
        if CustomOpTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(CustomOpTransform)
        if RenameWsubNodesTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(RenameWsubNodesTransform)

    if LocalizeFunctionReduceSumAxesTransform not in qeff_model._onnx_transforms:
        qeff_model._onnx_transforms.append(LocalizeFunctionReduceSumAxesTransform)

    # TODO: Handle this in the modelling class QEFFTransformersBase, remove from here.
    decoder_layer_classes = get_decoder_layer_classes_for_export(qeff_model.model)
    if decoder_layer_classes:
        if dynamo:
            # Pass resolved classnames to RenameRepeatedSubgraphTransform; single resolution point to avoid double-call.
            resolved_classnames = sorted(cls.__name__ for cls in decoder_layer_classes)
            qeff_model._subfunction_target_classnames = resolved_classnames
            onnx_transform_kwargs = dict(kwargs.get("onnx_transform_kwargs") or {})
            onnx_transform_kwargs["target_classnames"] = resolved_classnames
            kwargs["onnx_transform_kwargs"] = onnx_transform_kwargs
        else:
            # TorchScript path: pass class objects for export_modules_as_functions
            kwargs["export_modules_as_functions"] = decoder_layer_classes

    return (
        args,
        kwargs,
        {
            "onnx_transforms": original_transforms,
            "dynamo": dynamo,
            "use_onnx_subfunctions": orig_use_onnx_subfunctions,
            "hash_use_subfunctions": orig_hash_use_subfunctions,
            "hash_subfunction_version": orig_hash_subfunction_version,
            "decoder_layer_classes": decoder_layer_classes if dynamo else None,
        },
    )


def _cleanup_onnx_subfunctions(qeff_model, state=None):
    """
    Cleanup ONNX subfunction export environment.

    Restores the model and environment to pre-subfunction state by:
    - Undoing torch patches (TorchScript path only)
    - Resetting InvalidIndexProvider flag
    - Restoring original ONNX transforms list

    Args:
        qeff_model: The QEff model instance

    Note:
        This function is called in a finally block to ensure cleanup
        even if export fails. Errors during cleanup are logged but
        not re-raised to avoid masking the original exception.
    """
    # TorchScript patches are only applied on the non-dynamo path, so only undo them there.
    if state is None or not state.get("dynamo", False):
        undo_torch_patches()
    InvalidIndexProvider.SUBFUNC_ENABLED = False
    if state is not None and "onnx_transforms" in state:
        qeff_model._onnx_transforms = state["onnx_transforms"]
    if state is not None:
        qeff_model._use_onnx_subfunctions = state.get("use_onnx_subfunctions", False)
        qeff_model.hash_params["use_onnx_subfunctions"] = state.get("hash_use_subfunctions", False)
        if state.get("hash_subfunction_version") is None:
            qeff_model.hash_params.pop("onnx_subfunction_version", None)
        else:
            qeff_model.hash_params["onnx_subfunction_version"] = state["hash_subfunction_version"]


def _save_export_metadata(export_dir: Path, filtered_hash_params: Dict):
    """
    Save export metadata to JSON file for reproducibility.

    Args:
        export_dir: Directory where the export was saved
        filtered_hash_params: Dictionary of parameters used for hashing
    """
    # Import here to avoid circular dependency
    from QEfficient.utils._utils import create_json

    hashed_params_path = export_dir / "hashed_export_params.json"
    create_json(hashed_params_path, filtered_hash_params)
    logger.info("Hashed parameters exported successfully.")
