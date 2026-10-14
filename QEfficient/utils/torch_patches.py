# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Runtime monkey patches for ONNX export compatibility.

Patches kept here:
  - TorchScript ONNX exporter (_setup_trace_module_map, _get_module_attributes,
    _jit_pass_onnx_track_scope_attributes): fix attribute-type mismatches in the
    legacy trace-based exporter (dynamo=False path).
  - Layerwise safe export pass patches: disable expensive ONNX exporter passes
    for layerwise prefill export (TorchScript path).
  - temporarily_enable_nested_compile_regions / temporarily_disable_nested_compile_regions:
    context managers for dynamo export path subgraph boundary management.
  - preserve_mixed_export_subfunctions: scoped PyTorch 2.13 non-strict export
    patch that preserves nested regions and reuses only equivalent FX bodies,
    plus a post-decomposition hook that shares equivalent region bodies
    before ONNX translation (QEfficient/utils/export_subfunctions.py).
  - preserve_subfunction_source_lines: preserve FX source metadata while Dynamo
    retraces GraphModule subfunctions.

Patches removed (upstreamed to PyTorch):
  - FunctionalTensorMode.__torch_dispatch__ tracker-entry KeyError
  - _verify_exported_program_signature repeated_subgraph buffer handling
  - ExportedProgram.named_buffers constants fallback
  - invoke_subgraph_placeholder kwargs forwarding
  - materialize_as_graph FunctionalTensor mode handling
  - InvokeSubgraphHOP.gen_schema GraphModule reuse
  - _translate_fx_graph / _convert_fx_arg_to_onnx_arg nested tensor constants
"""

import importlib
import inspect
import os
import threading
from contextlib import contextmanager

import torch
import torch.onnx.utils as onnx_utils
import torch.utils._pytree as pytree
from torch import _C

from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")

try:
    from torch.onnx._internal.torchscript_exporter import utils as ts_utils

    _ts_utils_available = True
except ModuleNotFoundError:
    ts_utils = None
    _ts_utils_available = False

# Store original references before patching
_original_setup_trace_module_map = onnx_utils._setup_trace_module_map
_original_get_module_attributes = getattr(onnx_utils, "_get_module_attributes", None)
_original_model_to_graph = onnx_utils._model_to_graph
_original_track_scope_attrs = getattr(_C, "_jit_pass_onnx_track_scope_attributes", None)
_original_ts_setup_trace_module_map = ts_utils._setup_trace_module_map if _ts_utils_available else None
_original_ts_get_module_attributes = getattr(ts_utils, "_get_module_attributes", None) if _ts_utils_available else None

_PATCHES_ACTIVE = False
_MISSING_INSTANCE_ATTR = object()
_safe_export_patch_depth = 0
_safe_export_original_passes = {}
_SAFE_EXPORT_REQUIRED_PASSES = {
    "_jit_pass_dce",
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects",
    "_jit_pass_constant_propagation",
    "_jit_pass_cse",
    # Keep ONNX constant fold enabled to reduce topology drift between
    # layerwise prefill exports and regular (non-layerwise) prefill exports.
    "_jit_pass_onnx_constant_fold",
}


def _noop(*args, **kwargs):
    return None


def _return_false(*args, **kwargs):
    return False


def _return_graph(graph, *args, **kwargs):
    return graph


def _return_params(_graph, params_dict, *args, **kwargs):
    return params_dict


_SAFE_EXPORT_PASS_REPLACEMENTS = {
    "_jit_pass_constant_propagation": _noop,
    "_jit_pass_dce": _noop,
    "_jit_pass_cse": _return_false,
    "_jit_pass_canonicalize_graph_fuser_ops": _noop,
    "_jit_pass_peephole": _noop,
    "_jit_pass_fuse_addmm": _noop,
    "_jit_pass_onnx_eval_peephole": _return_params,
    "_jit_pass_onnx_constant_fold": _return_params,
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects": _noop,
    "_jit_pass_canonicalize": _return_graph,
    "_jit_pass_onnx_graph_shape_type_inference": _noop,
    "_jit_pass_onnx_deduplicate_initializers": _return_params,
}


def _model_to_graph_patched(model, *args, **kwargs):
    """Preserve model parameter names through TorchScript ONNX export."""
    graph, params_dict, torch_out = _original_model_to_graph(model, *args, **kwargs)

    parameter_names = {}
    for name, parameter in model.named_parameters():
        if parameter.numel():
            parameter_names.setdefault(parameter.data_ptr(), name)

    graph_inputs = {value.debugName(): value for value in graph.inputs()}
    renamed_params = {}
    for old_name, parameter in params_dict.items():
        new_name = parameter_names.get(parameter.data_ptr())
        graph_input = graph_inputs.get(old_name)
        if (
            old_name.startswith("onnx::")
            and new_name
            and new_name not in params_dict
            and new_name not in renamed_params
            and graph_input is not None
        ):
            graph_input.setDebugName(new_name)
        else:
            new_name = old_name
        renamed_params[new_name] = parameter

    return graph, renamed_params, torch_out


def _setup_trace_module_map_patched(
    model,
    export_modules_as_functions,
):
    """Patched version of _setup_trace_module_map that fixes onnx_attrs type mismatch."""

    def __register_attribute_hook():
        attr_name = "_onnx_attrs"

        def _track_module_attributes_forward_pre_hook(module, input):
            setattr(module, attr_name, _get_module_attributes(module))

        def _track_module_attributes_forward_hook(module, input, output):
            tracing_state = _C._get_tracing_state()
            if not tracing_state:
                return
            graph = tracing_state.graph()
            onnx_attrs = {}
            if hasattr(module, attr_name):
                onnx_attrs = getattr(module, attr_name)
                delattr(module, attr_name)
            try:
                onnx_attrs = {}  # HACK: to reduce export time # TODO: study behaviour across models
                _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)
            except Exception:
                # Silently skip: scope-attribute tracking is best-effort and not required for export.
                pass

        for m in model.modules():
            m.register_forward_hook(_track_module_attributes_forward_hook)
            m.register_forward_pre_hook(_track_module_attributes_forward_pre_hook)

    def _unqualified_variable_name(qualified_name: str) -> str:
        name_atoms = qualified_name.split(".")
        for i, atom in reversed(list(enumerate(name_atoms))):
            if not atom.isnumeric():
                return ".".join(name_atoms[i:])
        return qualified_name

    trace_module_map = {
        _m: torch._C._jit_onnx_create_full_scope_name(torch.typename(type(_m)), _unqualified_variable_name(_n))
        for _n, _m in model.named_modules()
    }
    torch.jit._trace._trace_module_map = trace_module_map

    if isinstance(export_modules_as_functions, bool) and export_modules_as_functions:
        module_typenames = {torch.typename(type(module)) for module in trace_module_map}
    elif isinstance(export_modules_as_functions, set) and export_modules_as_functions:

        def _find_typename(v):
            if isinstance(v, type):
                return torch.typename(v)
            else:
                raise RuntimeError(
                    "Only type of the `nn.Module` should be passed in the set for argument `export_modules_as_functions`. "
                    f"Got `{type(v).__name__}`."
                )

        module_typenames = {_find_typename(v) for v in export_modules_as_functions}
    else:
        module_typenames = set()

    if module_typenames:
        __register_attribute_hook()

    return module_typenames


def _get_module_attributes(module):
    """Helper function to get module attributes safely."""
    import typing

    import torch.nn

    # added _is_safe_value guard to prevent IValue-incompatible
    # types from being passed into the ONNX scope-attribute tracker.
    def _is_safe_value(value):
        if isinstance(value, (int, float, bool, str, torch.Tensor)) or value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(_is_safe_value(item) for item in value)
        return False

    annotations = typing.get_type_hints(type(module))
    base_m_annotations = typing.get_type_hints(torch.nn.Module)
    [annotations.pop(k, None) for k in base_m_annotations]

    attrs = {}
    for k in annotations:
        try:
            value = getattr(module, k)
            # Only include IValue-compatible attribute types
            if _is_safe_value(value):
                attrs[k] = value
        except AttributeError:
            _C._jit_onnx_log(f"Skipping module attribute '{k}'")
            continue
    return attrs


def _track_scope_attributes_patched(graph, attrs):
    """Ensure scope attributes passed to ONNX are IValue-compatible."""
    safe_attrs = {}
    for key, value in attrs.items():
        if isinstance(value, (int, float, bool, str, torch.Tensor)) or value is None:
            safe_attrs[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (int, float, bool, str, torch.Tensor)) or item is None for item in value
        ):
            safe_attrs[key] = value
    return _original_track_scope_attrs(graph, safe_attrs)


def _enable_safe_export_pass_patches(keep_passes=None):
    global _safe_export_patch_depth

    keep_passes = _SAFE_EXPORT_REQUIRED_PASSES | set(keep_passes or ())
    if _safe_export_patch_depth == 0:
        _safe_export_original_passes.clear()
        for name, replacement in _SAFE_EXPORT_PASS_REPLACEMENTS.items():
            if name in keep_passes:
                continue
            if hasattr(_C, name):
                _safe_export_original_passes[name] = getattr(_C, name)
                setattr(_C, name, replacement)
    _safe_export_patch_depth += 1


def _disable_safe_export_pass_patches():
    global _safe_export_patch_depth

    if _safe_export_patch_depth == 0:
        return

    _safe_export_patch_depth -= 1
    if _safe_export_patch_depth == 0:
        for name, original in _safe_export_original_passes.items():
            setattr(_C, name, original)
        _safe_export_original_passes.clear()


@contextmanager
def layerwise_safe_onnx_export_patches(enabled: bool = True, keep_passes=None):
    """Temporarily disable expensive ONNX exporter passes for layerwise prefill.

    This is a no-op unless the caller explicitly enables it and the process is
    inside the layerwise export context. Regular/non-layerwise export therefore
    keeps the original PyTorch ONNX exporter behavior. DCE stays enabled by
    default because some exported graphs need it to remove aten/prim nodes before
    PyTorch serializes ONNX. ``keep_passes`` can retain additional passes.
    """
    if not enabled:
        yield
        return

    _enable_safe_export_pass_patches(keep_passes=keep_passes)
    try:
        yield
    finally:
        _disable_safe_export_pass_patches()


def apply_torch_patches():
    """Apply monkey patches for ONNX export (TorchScript path)."""
    global _PATCHES_ACTIVE
    if _PATCHES_ACTIVE:
        return

    # Patch onnx_utils (used by both TorchScript and as fallback)
    onnx_utils._setup_trace_module_map = _setup_trace_module_map_patched
    onnx_utils._model_to_graph = _model_to_graph_patched
    if hasattr(onnx_utils, "_get_module_attributes"):
        onnx_utils._get_module_attributes = _get_module_attributes

    # Patch ts_utils (TorchScript-specific exporter utilities, torch >= 2.13 only)
    if _ts_utils_available:
        ts_utils._setup_trace_module_map = _setup_trace_module_map_patched
        if hasattr(ts_utils, "_get_module_attributes"):
            ts_utils._get_module_attributes = _get_module_attributes

    # Patch _C scope-attribute tracker to filter out IValue-incompatible types
    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _track_scope_attributes_patched

    _PATCHES_ACTIVE = True


def undo_torch_patches():
    """Undo monkey patches and restore original functions."""
    global _PATCHES_ACTIVE
    if not _PATCHES_ACTIVE:
        return

    onnx_utils._setup_trace_module_map = _original_setup_trace_module_map
    onnx_utils._model_to_graph = _original_model_to_graph
    if _original_get_module_attributes:
        onnx_utils._get_module_attributes = _original_get_module_attributes

    if _ts_utils_available:
        ts_utils._setup_trace_module_map = _original_ts_setup_trace_module_map
        if _original_ts_get_module_attributes:
            ts_utils._get_module_attributes = _original_ts_get_module_attributes

    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _original_track_scope_attrs

    _PATCHES_ACTIVE = False


@contextmanager
def temporarily_enable_nested_compile_regions(model, target_classes=None):
    """
    Wrap selected module ``forward`` methods with ``nested_compile_region``
    during export so repeated block functions are materialized by dynamo.

    Used when dynamo=True and use_onnx_subfunctions=True. Requires torch >= 2.13.
    """
    target_classes = tuple(target_classes) if target_classes else None
    patched_modules = []

    # nested_compile_region invokes PyTorch's non-strict HOP compiler. Scope
    # the mixed-layer fix over the same lifetime as the decoder wrappers so
    # ordinary eager execution and unrelated exports remain untouched.
    with preserve_mixed_export_subfunctions():
        try:
            for module in model.modules():
                if target_classes and not isinstance(module, target_classes):
                    continue

                bound_forward = getattr(module, "forward", None)
                if bound_forward is None:
                    continue

                wrapped_forward = getattr(bound_forward, "__func__", bound_forward)
                if getattr(wrapped_forward, "__qualname__", "") == "mark_compile_region.<locals>.wrap.<locals>.inner":
                    continue

                previous_forward = module.__dict__.get("forward", _MISSING_INSTANCE_ATTR)
                nested_forward = torch.compiler.nested_compile_region(wrapped_forward)
                setattr(module, "forward", nested_forward.__get__(module, type(module)))
                patched_modules.append((module, previous_forward))

            yield
        finally:
            for module, previous_forward in reversed(patched_modules):
                if previous_forward is _MISSING_INSTANCE_ATTR:
                    delattr(module, "forward")
                else:
                    setattr(module, "forward", previous_forward)


@contextmanager
def temporarily_disable_nested_compile_regions(model, target_classes=None):
    """
    Replace nested_compile_region-wrapped ``forward`` methods with their original
    underlying functions for the duration of plain dynamo export (flat graph path).

    Used during weight-free export with use_onnx_subfunctions=False so that
    @nested_compile_region boundaries on decoder layer forward() methods do not
    create unwanted subgraph splits during tracing.
    """
    target_classes = tuple(target_classes) if target_classes else None
    patched_modules = []

    try:
        for module in model.modules():
            if target_classes and not isinstance(module, target_classes):
                continue

            bound_forward = getattr(module, "forward", None)
            if bound_forward is None:
                continue

            wrapped_forward = getattr(bound_forward, "__func__", bound_forward)
            if getattr(wrapped_forward, "__qualname__", "") != "mark_compile_region.<locals>.wrap.<locals>.inner":
                continue

            closure = getattr(wrapped_forward, "__closure__", None) or ()
            original_forward = next(
                (cell.cell_contents for cell in closure if inspect.isfunction(cell.cell_contents)),
                None,
            )
            if original_forward is None:
                continue

            previous_forward = module.__dict__.get("forward", _MISSING_INSTANCE_ATTR)
            setattr(module, "forward", original_forward.__get__(module, type(module)))
            patched_modules.append((module, previous_forward))

        yield
    finally:
        for module, previous_forward in reversed(patched_modules):
            if previous_forward is _MISSING_INSTANCE_ATTR:
                delattr(module, "forward")
            else:
                setattr(module, "forward", previous_forward)


_DYNAMO_ENV_LOCK = threading.RLock()
_SUBFUNCTION_SOURCE_PATCH_LOCK = threading.RLock()
_MIXED_EXPORT_PATCH_LOCK = threading.RLock()
_MISSING_META = object()


def _same_export_region(a, b, fake_mode, a_operands=None, b_operands=None):
    """Conservatively identify equivalent flat, lifted-input export regions.

    Tensor attributes remain unsupported, while nested GraphModule attributes
    are compared recursively by PyTorch's graph comparator.

    Actual call operands can temporarily replace each body's placeholder
    metadata. Nested compiled regions have independent ShapeEnvs, so their
    stored symbols cannot be compared reliably across layer calls.
    """
    from torch._dynamo.variables.higher_order_ops import are_same_graph_modules
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

    if not isinstance(a, torch.fx.GraphModule) or not isinstance(b, torch.fx.GraphModule):
        return False
    if a.meta.get("nested_region_config") != b.meta.get("nested_region_config"):
        return False
    if len(a.graph.nodes) != len(b.graph.nodes):
        return False

    for left, right in zip(a.graph.nodes, b.graph.nodes):
        if left.op != right.op or left.op not in {
            "placeholder",
            "call_function",
            "call_method",
            "get_attr",
            "output",
        }:
            return False
        if left.op == "get_attr" and not all(
            isinstance(getattr(module, node.target, None), torch.fx.GraphModule)
            for module, node in ((a, left), (b, right))
        ):
            return False
        if left.op == "placeholder" and not all(
            isinstance(node.meta.get("example_value"), (torch.Tensor, torch.SymInt)) for node in (left, right)
        ):
            return False
        if pytree.tree_flatten((left.args, left.kwargs))[1] != pytree.tree_flatten((right.args, right.kwargs))[1]:
            return False

    placeholders = [tuple(node for node in gm.graph.nodes if node.op == "placeholder") for gm in (a, b)]
    operands = (a_operands, b_operands)
    if any(args is not None and len(nodes) != len(args) for nodes, args in zip(placeholders, operands)):
        return False
    replacements = [
        (node, value)
        for nodes, args in zip(placeholders, operands)
        if args is not None
        for node, value in zip(nodes, args)
    ]
    if any(not isinstance(value, (torch.Tensor, torch.SymInt)) for _, value in replacements):
        return False
    saved = [(node, node.meta.get("example_value", _MISSING_META)) for node, _ in replacements]
    try:
        for node, value in replacements:
            node.meta["example_value"] = value
        # Comparing symbolic tensor metadata can otherwise record SymInt
        # comparisons in the active export proxy trace and fail on symbols
        # belonging to the separately compiled region.
        with disable_proxy_modes_tracing():
            return are_same_graph_modules("non_strict_export", a, b, fake_mode)
    except (KeyError, NotImplementedError):
        return False
    finally:
        for node, value in saved:
            if value is _MISSING_META:
                node.meta.pop("example_value", None)
            else:
                node.meta["example_value"] = value


def _invoke_subgraph_for_export(subgraph, *operands):
    """Invoke an export region in the current proxy trace."""
    from torch._guards import detect_fake_mode
    from torch._higher_order_ops.invoke_subgraph import invoke_subgraph_infer
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    mode = get_proxy_mode()
    fake_mode = detect_fake_mode(operands)
    cache = getattr(mode, "_invoke_subgraph_cache", None) if mode is not None else None
    if mode is not None and fake_mode is not None and cache is not None:
        inputs = getattr(mode, "_qeff_export_region_inputs", None)
        if inputs is None:
            inputs = mode._qeff_export_region_inputs = {}
        for previous in cache:
            previous_inputs = inputs.get(previous)
            if previous_inputs is not None and _same_export_region(
                previous, subgraph, fake_mode, previous_inputs, operands
            ):
                subgraph = previous
                break
        inputs.setdefault(subgraph, operands)

    # The inference entrypoint allocates identifiers against the enclosing
    # proxy trace, avoiding collisions between independently compiled regions.
    return invoke_subgraph_infer(subgraph, *operands)


def _export_region_backend(backend):
    """Preserve nested regions and deduplicate only equivalent region bodies."""

    def wrapped(gm, example_inputs):
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.ops.higher_order.invoke_subgraph:
                node.target = _invoke_subgraph_for_export
                node.args = (node.args[0], *node.args[2:])
        gm.recompile()
        return backend(gm, example_inputs)

    return wrapped


def _mixed_export_hop_compile_and_call(fn, args, kwargs):
    """Compile one nested region without losing its enclosing export identity."""
    from torch._dynamo.eval_frame import set_fullgraph_compiled_frame_count
    from torch._higher_order_ops.utils import setup_compilation_env

    # PyTorch's default inner compile may inline a single-use region. Keeping
    # this disabled preserves the boundary long enough for the outer export
    # trace to see all decoder layers and reuse equivalent bodies safely.
    region_context = torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    with region_context, setup_compilation_env() as backend:
        old_count = set_fullgraph_compiled_frame_count(0)
        try:
            return torch.compile(fn, backend=_export_region_backend(backend), fullgraph=True)(*args, **(kwargs or {}))
        finally:
            set_fullgraph_compiled_frame_count(old_count)


@contextmanager
def preserve_mixed_export_subfunctions():
    """Apply the mixed-layer non-strict export candidate when PyTorch supports it.

    This is intentionally a scoped monkey patch of the private nested compile
    hook and, when available, the ONNX preparation hook. The latter runs after
    decomposition and before ONNX translation, when equivalent region bodies
    can be compared using the enclosing ShapeEnv.
    """
    with _MIXED_EXPORT_PATCH_LOCK:
        try:
            utils = importlib.import_module("torch._higher_order_ops.utils")
            original = utils._hop_compile_and_call
        except (AttributeError, ModuleNotFoundError):
            # PyTorch versions without this hook have a different
            # invoke_subgraph implementation and need no QEff patch.
            yield
            return

        def patched(fn, args, kwargs=None):
            return _mixed_export_hop_compile_and_call(fn, args, kwargs)

        try:
            onnx_core = importlib.import_module("torch.onnx._internal.exporter._core")
            original_prepare = onnx_core._prepare_exported_program_for_export
        except (AttributeError, ModuleNotFoundError):
            onnx_core = original_prepare = None
            logger.warning(
                "torch.onnx exporter hook '_prepare_exported_program_for_export' not found; "
                "ONNX subfunction reuse is disabled and repeated layers may be exported as duplicates."
            )

        utils._hop_compile_and_call = patched
        if onnx_core is not None:
            from QEfficient.utils.export_subfunctions import reuse_exported_subfunctions

            def prepare_with_reuse(*args, **kwargs):
                return reuse_exported_subfunctions(original_prepare(*args, **kwargs))

            onnx_core._prepare_exported_program_for_export = prepare_with_reuse
        try:
            yield
        finally:
            if onnx_core is not None:
                onnx_core._prepare_exported_program_for_export = original_prepare
            utils._hop_compile_and_call = original


@contextmanager
def preserve_subfunction_source_lines():
    """Preserve original FX source metadata while retracing Dynamo subfunctions.

    PyTorch's generated GraphModule code does not preserve the original source
    locations when an ``invoke_subgraph`` GraphModule is retraced. Running the
    GraphModule through ``torch.fx.Interpreter`` under ``preserve_node_meta``
    keeps the metadata available to the ONNX exporter.

    This patches a private PyTorch API for the duration of one export. The
    patch is process-global, so the lock serializes callers and the original
    function is always restored on exit.
    """
    with _SUBFUNCTION_SOURCE_PATCH_LOCK:
        module = importlib.import_module("torch._higher_order_ops.invoke_subgraph")
        original = module.reenter_make_fx

        def retrace(fn, *args, **kwargs):
            if not isinstance(fn, torch.fx.GraphModule):
                return original(fn, *args, **kwargs)

            def interpreted(*operands):
                with torch.fx.traceback.preserve_node_meta():
                    return torch.fx.Interpreter(fn).run(*operands)

            return original(interpreted, *args, **kwargs)

        module.reenter_make_fx = retrace
        try:
            yield
        finally:
            module.reenter_make_fx = original


@contextmanager
def dynamo_invoke_subgraph_fallback_env():
    """Temporarily set TORCH_INVOKE_ALLOW_CREATE_FALLBACK=1 for dynamo export.

    torch.onnx.export's dynamo path (dynamo=True) needs this env var set to
    allow invoke_subgraph placeholders to fall back correctly during tracing.
    Saves and restores whatever value (or absence) the caller's environment
    already had, rather than assuming it was previously unset.
    Uses an RLock so concurrent exports don't race on the env var.
    """
    with _DYNAMO_ENV_LOCK:
        previous = os.environ.get("TORCH_INVOKE_ALLOW_CREATE_FALLBACK")
        os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("TORCH_INVOKE_ALLOW_CREATE_FALLBACK", None)
            else:
                os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = previous
