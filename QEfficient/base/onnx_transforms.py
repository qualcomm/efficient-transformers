# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple

import numpy as np
import onnx
import torch
from onnx import AttributeProto, ModelProto, TensorProto, external_data_helper, numpy_helper

from QEfficient.customop.ctx_scatter_gather import (
    CtxChunkScatterBatch,
    CtxChunkScatterBatchFunc,
    CtxGather,
    CtxGather3D,
    CtxGatherBlockedKV,
    CtxGatherBlockedKVBatch,
    CtxGatherDP,
    CtxGatherDPCP,
    CtxGatherDPCPFunc,
    CtxGatherDPFunc,
    CtxGatherFoldedRows,
    CtxGatherFoldedRowsFunc,
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFunc3DGeneralized,
    CtxGatherFuncBlockedKV,
    CtxGatherFuncBlockedKVBatch,
    CtxScatter,
    CtxScatter3D,
    CtxScatter3DInt,
    CtxScatterDP,
    CtxScatterDPCP,
    CtxScatterDPCPFunc,
    CtxScatterDPFunc,
    CtxScatterFoldedRows,
    CtxScatterFoldedRowsFunc,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.customop.ctx_scatter_gather_cb import (
    CtxGatherBlockedKVCB,
    CtxGatherCB,
    CtxGatherCB3D,
    CtxGatherFuncBlockedKVCB,
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterCB,
    CtxScatterCB3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)

# from QEfficient.customop.quantization_ops import CastToUInt4, CastToUInt4Func
from QEfficient.customop.onnxscript_utils import get_onnxscript_func
from QEfficient.customop.quantization_ops import CastToUInt4, CastToUInt4Func
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.utils import apply_kv_cache_prefix, constants
from QEfficient.utils.constants import FILE_CHUNK_SIZE_DEFAULT, SIZE_THRESHOLD_DEFAULT

logger = logging.getLogger(__name__)


class BaseOnnxTransform:
    """Base class for ONNX graph modifications. Should NOT be instantiated."""

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` method directly.")

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> bool:
        raise NotImplementedError("Use subclasses for ONNX transform")


class FP16ClipTransform(BaseOnnxTransform):
    """Clip FP32 tensors to FP16 range to avoid overflow during conversion."""

    @classmethod
    def apply(cls, tensor: TensorProto, onnx_base_dir: str, fp16_max: float, fp16_min: float) -> bool:
        nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
        if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
            neg_inf_mask = np.isinf(nptensor) & (nptensor < 0)
            clipped_tensor = np.clip(nptensor, fp16_min, fp16_max)

            if neg_inf_mask.any():
                clipped_tensor = np.where(neg_inf_mask, np.float32("-inf"), clipped_tensor)

            tensor.CopyFrom(numpy_helper.from_array(clipped_tensor, tensor.name))
            return True
        return False


class SplitTensorsTransform(BaseOnnxTransform):
    """Split large tensors into external data files for efficient storage."""

    @classmethod
    def apply(
        cls, tensor: TensorProto, model_name: str, file_num: int, mapping: dict[str, tuple[TensorProto, str]]
    ) -> None:
        file_name = f"{model_name}_{file_num}.onnx.data"
        mapping[tensor.name] = (tensor, file_name)


class CustomOpTransform(BaseOnnxTransform):
    """Register custom ONNX ops and append their function prototypes to the model."""

    _custom_ops: Dict[str, Tuple[Any, Any]] = {
        "CustomRMSNormFunc": (CustomRMSNormFunc, CustomRMSNorm),
        "CtxScatterFunc": (CtxScatterFunc, CtxScatter),
        "CtxScatterFunc3D": (CtxScatterFunc3D, CtxScatter3D),
        "CtxScatterFunc3DInt": (CtxScatterFunc3DInt, CtxScatter3DInt),
        "CtxScatterFunc3DGeneralized": (CtxScatterFunc3DGeneralized, CtxScatter3D),
        "CtxScatterDPFunc": (CtxScatterDPFunc, CtxScatterDP),
        "CtxScatterFoldedRowsFunc": (CtxScatterFoldedRowsFunc, CtxScatterFoldedRows),
        "CtxScatterDPCPFunc": (CtxScatterDPCPFunc, CtxScatterDPCP),
        "CtxGatherFunc": (CtxGatherFunc, CtxGather),
        "CtxGatherFunc3D": (CtxGatherFunc3D, CtxGather3D),
        "CtxGatherFunc3DGeneralized": (CtxGatherFunc3DGeneralized, CtxGather3D),
        "CtxGatherDPFunc": (CtxGatherDPFunc, CtxGatherDP),
        "CtxGatherFoldedRowsFunc": (CtxGatherFoldedRowsFunc, CtxGatherFoldedRows),
        "CtxGatherDPCPFunc": (CtxGatherDPCPFunc, CtxGatherDPCP),
        "CtxScatterFuncCB3D": (CtxScatterFuncCB3D, CtxScatterCB3D),
        "CtxGatherFuncCB3D": (CtxGatherFuncCB3D, CtxGatherCB3D),
        "CtxGatherFuncBlockedKV": (CtxGatherFuncBlockedKV, CtxGatherBlockedKV),
        "CtxGatherFuncBlockedKVBatch": (CtxGatherFuncBlockedKVBatch, CtxGatherBlockedKVBatch),
        "CtxChunkScatterBatchFunc": (CtxChunkScatterBatchFunc, CtxChunkScatterBatch),
        "CtxGatherFuncBlockedKVCB": (CtxGatherFuncBlockedKVCB, CtxGatherBlockedKVCB),
        "CtxScatterFuncCB": (CtxScatterFuncCB, CtxScatterCB),
        "CtxGatherFuncCB": (CtxGatherFuncCB, CtxGatherCB),
        "CastToUInt4": (CastToUInt4Func, CastToUInt4),
    }

    @classmethod
    def apply(cls, model: ModelProto, onnx_export_opset: int = constants.ONNX_LEGACY_EXPORT_OPSET) -> bool:
        op_applied = False

        # Register with PyTorch ONNX exporter (for export time)
        for op_name, (func_class, _) in cls._custom_ops.items():
            if hasattr(func_class, "symbolic"):
                torch.onnx.register_custom_op_symbolic(f"::{op_name}", func_class.symbolic, onnx_export_opset)

        used_op_types = {node.op_type for node in model.graph.node}
        for function_proto in model.functions:
            used_op_types.update(node.op_type for node in function_proto.node)

        # Add function prototypes to model
        existing = {f.name for f in model.functions}

        for func_name, onnxscript_func in cls._custom_ops.values():
            proto = get_onnxscript_func(onnxscript_func, onnx_export_opset).to_function_proto()
            if proto.name not in used_op_types:
                continue
            if proto.name not in existing:
                model.functions.append(proto)
                op_applied = True
                cls._ensure_opset_imports(model, proto.domain, 1)
        cls._propagate_function_opset_imports(model)
        return op_applied

    @staticmethod
    def _ensure_opset_imports(container: ModelProto | onnx.FunctionProto, domain: str, version: int) -> None:
        if any(opset.domain == domain for opset in container.opset_import):
            return
        container.opset_import.append(onnx.helper.make_opsetid(domain, version))

    @classmethod
    def _propagate_function_opset_imports(cls, model: ModelProto) -> None:
        for fn in model.functions:
            for node in fn.node:
                if node.domain:
                    cls._ensure_opset_imports(fn, node.domain, 1)
                    cls._ensure_opset_imports(model, node.domain, 1)


class RemovePrefix(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        graph = model.graph
        renamed = False

        def strip_prefix(name: str) -> str:
            parts = name.rsplit("/", 1)
            return parts[1] if len(parts) == 2 else parts[0]

        input_names = []
        for i, inputs in enumerate(graph.input):
            original = inputs.name
            new = strip_prefix(original)
            if new != original:
                renamed = True
            inputs.name = new
            graph.input[i].name = new
            input_names.append(new)

        input_name_set = set(input_names)
        output_rename_map = {}

        # Rename model graph outputs and keep mapping so producer/consumer edges can be fixed.
        for out in graph.output:
            original = out.name
            new = strip_prefix(original)
            if new != original:
                out.name = new
                output_rename_map[original] = new
                renamed = True

        for node in graph.node:
            for i, out in enumerate(node.output):
                if out in output_rename_map and output_rename_map[out] != out:
                    node.output[i] = output_rename_map[out]
                    renamed = True

            new_inputs = []
            for s in node.input:
                # Keep node inputs in sync for renamed model outputs.
                if s in output_rename_map:
                    new_inputs.append(output_rename_map[s])
                    continue

                if s in input_name_set:
                    new_inputs.append(s)
                    continue

                replaced = s
                if "/" in s:
                    tail = s.rsplit("/", 1)[1]
                    if tail in input_name_set:
                        replaced = tail
                new_inputs.append(replaced)

            for idx in range(len(node.input)):
                if node.input[idx] != new_inputs[idx]:
                    node.input[idx] = new_inputs[idx]
                    renamed = True

        return renamed


class RenameFunctionOutputsTransform(BaseOnnxTransform):
    """Rename decoder function retained-state outputs to public graph names.

    When subfunction export emits ``*_InternalRetainedState``, this transform rewrites
    them to ``*_RetainedState`` while preserving any optional KV prefix infix
    (for example ``past_key.0_vllmKvCache``).
    """

    @classmethod
    def apply(cls, model: ModelProto, layer_idx=0) -> bool:
        graph = model.graph
        op_type_to_func = {f.name: f for f in model.functions}
        decoder_patterns = ["DecoderLayer", "Block", "Layer"]
        renamed = False
        model_out_map = {v.name: i for i, v in enumerate(graph.output)}

        for node in graph.node:
            if any(p in node.name or p in node.op_type for p in decoder_patterns):
                func = op_type_to_func.get(node.op_type)
                if not func:
                    continue
                for i, out_name in enumerate(func.output):
                    if "_InternalRetainedState" in out_name:
                        renamed = True
                        orig = node.output[i]
                        if orig.endswith("_InternalRetainedState"):
                            new = orig[: -len("_InternalRetainedState")] + "_RetainedState"
                        else:
                            base = out_name[: -len("_InternalRetainedState")]
                            new = orig
                            for token in (
                                "past_key.",
                                "past_value.",
                                "compressed_kv.",
                                "k_pe.",
                                "recurrent_state.",
                                "conv_state.",
                            ):
                                if not base.startswith(token):
                                    continue
                                tail = base[len(token) :]
                                _, _, infix = tail.partition("_")
                                infix = f"_{infix}" if infix else ""
                                new = f"{token}{layer_idx}{infix}_RetainedState"
                                break
                        node.output[i] = new
                        if orig in model_out_map:
                            graph.output[model_out_map[orig]].name = new
                layer_idx += 1
        return renamed


class PreserveNestedCacheRetainedStateTransform(BaseOnnxTransform):
    """Expose nested decoder cache side effects as explicit ONNX values.

    Must run BEFORE RenameRepeatedSubgraphTransform: this transform looks up
    functions by the dynamo-assigned name (repeated_subgraphN) via fn_by_name.
    Renaming them first would break that lookup.
    """

    _STATE_INPUT_RE = re.compile(r"^(past_key|past_value|conv_state|recurrent_state)\.(\d+)")
    _STATE_PAIRS = (
        frozenset({"past_key", "past_value"}),
        frozenset({"conv_state", "recurrent_state"}),
    )

    # All scatter op_type names that write back a KV cache tensor.
    # Keep in sync with CustomOpTransform._custom_ops and the dynamo
    # custom_translation_table in base/modeling_qeff.py.
    _SCATTER_OP_TYPES = frozenset(
        {
            "CtxScatter",
            "CtxScatterCB",
            "CtxScatter3D",
            "CtxScatter3DInt",
            "CtxScatterCB3D",
            "CtxScatterDP",
            "CtxScatterFoldedRows",
            "CtxScatterDPCP",
        }
    )

    @staticmethod
    def _retained_state_name(state_kind: str, layer_idx: str, kv_cache_prefix: str | None = None) -> str:
        name = f"{state_kind}.{layer_idx}_RetainedState"
        if kv_cache_prefix:
            return apply_kv_cache_prefix([name], kv_cache_prefix)[0]
        return name

    @classmethod
    def _resolve_state_scatter_outputs(cls, fn, node, layer_idx: str, state_kinds: set[str]) -> dict[str, str] | None:
        """Resolve nested cache-state scatter outputs from function data flow.

        ONNX function-call inputs bind positionally. Each cache scatter consumes
        the state buffer as its first input, so the matching scatter is identified
        from that buffer's function argument rather than output names or node order.
        """
        function_state_inputs: dict[str, str] = {}
        for input_index, input_name in enumerate(node.input):
            match = cls._STATE_INPUT_RE.match(input_name)
            if match is None:
                continue
            state_kind, input_layer_idx = match.groups()
            if input_layer_idx != layer_idx or state_kind not in state_kinds:
                continue
            if input_index >= len(fn.input):
                raise ValueError(
                    f"Nested function '{fn.name}' has no input at position {input_index} for "
                    f"call-node state input '{input_name}'."
                )
            if state_kind in function_state_inputs:
                raise ValueError(
                    f"Nested function call '{node.name or node.op_type}' has multiple "
                    f"'{state_kind}.{layer_idx}' inputs."
                )
            function_state_inputs[state_kind] = fn.input[input_index]

        if set(function_state_inputs) != state_kinds:
            return None

        scatter_outputs: dict[str, str] = {}
        for state_kind, function_input in function_state_inputs.items():
            writers = [
                fn_node
                for fn_node in fn.node
                if fn_node.op_type in cls._SCATTER_OP_TYPES
                and fn_node.input
                and fn_node.input[0] == function_input
                and fn_node.output
            ]
            if len(writers) != 1:
                writer_names = [
                    f"{writer.op_type}({writer.output[0] if writer.output else '<no-output>'})" for writer in writers
                ]
                raise ValueError(
                    f"Could not uniquely resolve the nested {state_kind}.{layer_idx} state writer in function "
                    f"'{fn.name}': expected one CtxScatter* with data input '{function_input}', "
                    f"found {len(writers)} ({writer_names})."
                )
            scatter_outputs[state_kind] = writers[0].output[0]

        return scatter_outputs

    @classmethod
    def apply(cls, model: ModelProto, kv_cache_prefix: str | None = None) -> bool:
        graph = model.graph
        produced_names = {name for node in graph.node for name in node.output}
        dangling_retained_outputs = {
            out.name for out in graph.output if out.name.endswith("_RetainedState") and out.name not in produced_names
        }
        if not dangling_retained_outputs:
            return False

        fn_by_name = {fn.name: fn for fn in model.functions}
        changed = False
        state_rename_map: dict[str, str] = {}
        remaining_dangling_outputs = set(dangling_retained_outputs)

        for node in graph.node:
            fn = fn_by_name.get(node.op_type)
            if fn is None:
                continue

            # Identify cache-state inputs grouped by layer. Dynamo can carry a
            # prior layer's cache tensor through a later repeated subgraph, so
            # collecting one layer at a time avoids discarding valid state pairs.
            state_inputs_by_layer: dict[str, dict[str, str]] = {}
            for inp_name in node.input:
                match = cls._STATE_INPUT_RE.match(inp_name)
                if match is None:
                    continue
                state_kind, layer_idx = match.groups()
                layer_state_inputs = state_inputs_by_layer.setdefault(layer_idx, {})
                if state_kind in layer_state_inputs:
                    raise ValueError(
                        f"Function call '{node.name or node.op_type}' has multiple "
                        f"inputs for '{state_kind}.{layer_idx}'."
                    )
                layer_state_inputs[state_kind] = inp_name

            for layer_idx, state_inputs in state_inputs_by_layer.items():
                present_states = frozenset(state_inputs)
                desired_outputs = {
                    state_kind: cls._retained_state_name(state_kind, layer_idx, kv_cache_prefix)
                    for state_kind in state_inputs
                }
                dangling_outputs = {
                    state_kind: output_name
                    for state_kind, output_name in desired_outputs.items()
                    if output_name in remaining_dangling_outputs
                }

                # Skip unrelated state-like inputs and layers whose retained
                # outputs are already produced by the graph.
                if not dangling_outputs:
                    continue
                if present_states not in cls._STATE_PAIRS:
                    raise ValueError(
                        f"Nested function '{fn.name}' has incomplete cache inputs for layer {layer_idx}: "
                        f"found {sorted(present_states)}."
                    )

                if set(dangling_outputs) != set(desired_outputs):
                    raise ValueError(
                        f"Nested function '{fn.name}' has partially dangling retained-state outputs for layer "
                        f"{layer_idx}: dangling={sorted(dangling_outputs.values())}, "
                        f"expected={sorted(desired_outputs.values())}."
                    )

                scatter_outputs = cls._resolve_state_scatter_outputs(fn, node, layer_idx, set(state_inputs))
                if scatter_outputs is None:
                    continue

                # Expose scatter outputs in the function's output list, rename
                # cache inputs and append retained-state output names to the
                # call node. Both writers are resolved before this block, so
                # graph rewiring is atomic.
                for state_kind, desired_output in desired_outputs.items():
                    scatter_output = scatter_outputs[state_kind]
                    if scatter_output not in fn.output:
                        fn.output.append(scatter_output)
                        changed = True

                    retained_input = state_inputs[state_kind]
                    plain_input = f"{state_kind}.{layer_idx}"
                    if retained_input.endswith("_RetainedState"):
                        state_rename_map[retained_input] = plain_input

                    if desired_output not in node.output:
                        node.output.append(desired_output)
                        changed = True
                    remaining_dangling_outputs.discard(desired_output)
        retained_output_names = {output.name for output in graph.output if output.name.endswith("_RetainedState")}
        for graph_input in graph.input:
            match = cls._STATE_INPUT_RE.match(graph_input.name)
            if match is None or not graph_input.name.endswith("_RetainedState"):
                continue
            if graph_input.name in retained_output_names:
                state_kind, layer_idx = match.groups()
                state_rename_map[graph_input.name] = f"{state_kind}.{layer_idx}"

        if state_rename_map:
            changed |= cls._rename_graph_inputs_bulk(graph, state_rename_map)
        return changed

    @staticmethod
    def _rename_graph_inputs_bulk(graph: onnx.GraphProto, rename_map: dict[str, str]) -> bool:
        if not rename_map:
            return False
        changed = False
        for value in graph.input:
            if value.name in rename_map:
                value.name = rename_map[value.name]
                changed = True
        for value in graph.value_info:
            if value.name in rename_map:
                value.name = rename_map[value.name]
                changed = True
        for node in graph.node:
            new_inputs = [rename_map.get(n, n) for n in node.input]
            if new_inputs != list(node.input):
                node.input[:] = new_inputs
                changed = True
        return changed


class RenameRepeatedSubgraphTransform(BaseOnnxTransform):
    """Rename dynamo repeated_subgraph function names to model-specific layer class names.

    Must run AFTER PreserveNestedCacheRetainedStateTransform: that transform
    looks up functions by the dynamo-assigned name.  Renaming them first would
    break that lookup.
    """

    # Primary pattern emitted by torch.export repeated-subgraph canonicalization.
    # Extended to also match alternative patterns seen across PyTorch 2.x releases
    # so a dynamo-internal rename does not silently produce a no-op transform.
    _REPEATED_SUBGRAPH_PATTERNS = [
        re.compile(r"^repeated_subgraph(\d+)$"),  # torch >= 2.5 canonical name
        re.compile(r"^subgraph_(\d+)$"),  # alternative seen in some 2.x nightlies
        re.compile(r"^invoke_subgraph_(\d+)$"),  # earlier 2.x name
    ]

    @classmethod
    def _iter_all_nodes(cls, nodes):
        """Yield every NodeProto reachable from `nodes`, including nodes nested
        inside If/Loop subgraph attributes."""
        for node in nodes:
            yield node
            for attr in node.attribute:
                if attr.HasField("g"):
                    yield from cls._iter_all_nodes(attr.g.node)
                for graph in attr.graphs:
                    yield from cls._iter_all_nodes(graph.node)

    @staticmethod
    def _rename_op_types(nodes, old_to_new: dict[str, str]) -> None:
        for node in RenameRepeatedSubgraphTransform._iter_all_nodes(nodes):
            if node.op_type in old_to_new:
                node.op_type = old_to_new[node.op_type]

    @classmethod
    def apply(cls, model: ModelProto, target_classnames: list[str] | None = None, **kwargs) -> bool:
        target_classnames = [name for name in (target_classnames or []) if name]
        if not target_classnames:
            logger.warning(
                "RenameRepeatedSubgraphTransform: target_classnames is empty — transform is a no-op. "
                "Check that get_submodules_for_export() returns the decoder layer classes."
            )
            return False

        repeated_functions = []
        for fn in model.functions:
            for pattern in cls._REPEATED_SUBGRAPH_PATTERNS:
                match = pattern.match(fn.name)
                if match:
                    repeated_functions.append((int(match.group(1)), fn))
                    break

        if not repeated_functions:
            logger.warning(
                "RenameRepeatedSubgraphTransform: no repeated_subgraph functions found in the ONNX model. "
                "This may indicate that dynamo changed its internal subgraph naming convention. "
                "The transform is a no-op; function names remain as emitted by torch.export."
            )
            return False

        repeated_functions.sort(key=lambda item: item[0])
        old_to_new = {}
        used_names = {fn.name for fn in model.functions}

        for idx, (_, fn) in enumerate(repeated_functions):
            if idx >= len(target_classnames):
                logger.warning(
                    f"RenameRepeatedSubgraphTransform: more repeated subgraph functions ({len(repeated_functions)}) "
                    f"than target class names ({len(target_classnames)}). "
                    f"Function '{fn.name}' (index {idx}) will be assigned the last available name "
                    f"'{target_classnames[-1]}' with a numeric suffix — verify get_submodules_for_export() "
                    "returns all repeated block classes for this model."
                )
            base_name = target_classnames[min(idx, len(target_classnames) - 1)]
            candidate = base_name
            suffix = 1
            while candidate in used_names and candidate != fn.name:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            used_names.discard(fn.name)
            used_names.add(candidate)
            old_to_new[fn.name] = candidate

        if not old_to_new:
            return False

        for fn in model.functions:
            if fn.name in old_to_new:
                fn.name = old_to_new[fn.name]

        cls._rename_op_types(model.graph.node, old_to_new)
        for fn in model.functions:
            cls._rename_op_types(fn.node, old_to_new)

        return True


class DeduplicateRepeatedSubgraphTransform(BaseOnnxTransform):
    """Collapse structurally identical repeated decoder local functions.

    Dynamo subfunction export can emit one FunctionProto per decoder layer, even
    when those function bodies differ only by local SSA names or layer-indexed
    formal parameter names. ONNX local-function calls bind inputs and outputs by
    position, so duplicate call nodes can safely target the first structurally
    equivalent function without changing graph-level tensor names.
    """

    _NUMERIC_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)$")

    @classmethod
    def apply(cls, model: ModelProto, target_classnames: Optional[List[str]] = None, **kwargs) -> bool:
        target_classnames = [name for name in (target_classnames or []) if name]
        candidates = cls._collect_candidate_functions(model, target_classnames)
        if len(candidates) < 2:
            return False

        input_dim_signatures = cls._function_input_dim_signatures(model, candidates)
        active_input_indices = {fn.name: cls._active_input_indices(fn) for _, fn in candidates}
        fingerprint_to_canonical = {}
        duplicate_to_canonical = {}
        functions_to_remove = set()

        for _, fn in candidates:
            fingerprint = cls._function_fingerprint(
                fn, input_dim_signatures.get(fn.name), active_input_indices.get(fn.name)
            )
            canonical = fingerprint_to_canonical.get(fingerprint)
            if canonical is None:
                fingerprint_to_canonical[fingerprint] = fn
                continue
            duplicate_to_canonical[fn.name] = (canonical.name, active_input_indices[fn.name])
            functions_to_remove.add(fn.name)

        if not duplicate_to_canonical:
            return False

        cls._rewrite_function_calls(model.graph.node, duplicate_to_canonical)
        for fn in model.functions:
            cls._rewrite_function_calls(fn.node, duplicate_to_canonical)

        kept_functions = [fn for fn in model.functions if fn.name not in functions_to_remove]
        del model.functions[:]
        model.functions.extend(kept_functions)
        return True

    @classmethod
    def _collect_candidate_functions(cls, model: ModelProto, target_classnames: List[str]):
        called_function_names = cls._called_function_names(model)
        candidates = []
        for fn in model.functions:
            if fn.name not in called_function_names:
                continue
            order = cls._candidate_order(fn.name, target_classnames)
            if order is None:
                continue
            candidates.append((order, fn))
        candidates.sort(key=lambda item: item[0])
        return candidates

    @classmethod
    def _candidate_order(cls, name: str, target_classnames: List[str]) -> Optional[Tuple[int, int, str]]:
        for pattern_index, pattern in enumerate(RenameRepeatedSubgraphTransform._REPEATED_SUBGRAPH_PATTERNS):
            match = pattern.match(name)
            if match:
                return pattern_index, int(match.group(1)), name

        for class_index, class_name in enumerate(target_classnames):
            if name == class_name:
                return 100 + class_index, 0, name
            match = cls._NUMERIC_SUFFIX_RE.match(name)
            if match and match.group("base") == class_name:
                return 100 + class_index, int(match.group("idx")), name

        return None

    @classmethod
    def _called_function_names(cls, model: ModelProto) -> set[str]:
        function_names = {fn.name for fn in model.functions}
        called = set()
        for node in cls._iter_all_nodes(model.graph.node):
            if node.op_type in function_names:
                called.add(node.op_type)
        for fn in model.functions:
            for node in cls._iter_all_nodes(fn.node):
                if node.op_type in function_names:
                    called.add(node.op_type)
        return called

    @classmethod
    def _active_input_indices(cls, fn: onnx.FunctionProto) -> List[int]:
        used_names = set(fn.output)
        for node in cls._iter_all_nodes(fn.node):
            used_names.update(name for name in node.input if name)
        return [idx for idx, name in enumerate(fn.input) if name in used_names]

    @staticmethod
    def _rewrite_function_calls(nodes, old_to_new) -> None:
        for node in DeduplicateRepeatedSubgraphTransform._iter_all_nodes(nodes):
            rewrite = old_to_new.get(node.op_type)
            if rewrite is None:
                continue
            new_op_type, kept_input_indices = rewrite
            original_inputs = list(node.input)
            node.op_type = new_op_type
            node.input[:] = [original_inputs[idx] for idx in kept_input_indices if idx < len(original_inputs)]

    @staticmethod
    def _iter_all_nodes(nodes):
        yield from RenameRepeatedSubgraphTransform._iter_all_nodes(nodes)

    @classmethod
    def _function_fingerprint(
        cls,
        fn: onnx.FunctionProto,
        input_dim_signatures: Optional[Dict[int, tuple]] = None,
        active_input_indices: Optional[List[int]] = None,
    ) -> str:
        active_input_indices = active_input_indices or list(range(len(fn.input)))
        active_inputs = [fn.input[idx] for idx in active_input_indices]
        state = cls._new_value_state(active_inputs, active_input_indices)
        payload = {
            "domain": fn.domain,
            "inputs": [state["value"](name) for name in active_inputs],
            "outputs": [state["value"](name) for name in fn.output],
            "opsets": sorted((opset.domain, opset.version) for opset in fn.opset_import),
            "nodes": [
                node_key
                for node in fn.node
                if (node_key := cls._node_key(node, state, input_dim_signatures)) is not None
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _new_value_state(inputs, original_indices=None):
        original_indices = original_indices or list(range(len(inputs)))
        input_index = {name: original_idx for name, original_idx in zip(inputs, original_indices)}
        value_map = {name: f"arg{idx}" for idx, name in enumerate(inputs)}
        counter = {"value": 0}

        def value(name: str) -> str:
            if not name:
                return ""
            if name not in value_map:
                value_map[name] = f"tmp{counter['value']}"
                counter["value"] += 1
            return value_map[name]

        def assign(name: str, canonical_value: str) -> None:
            if name:
                value_map[name] = canonical_value

        return {"value": value, "assign": assign, "input_index": input_index}

    @classmethod
    def _node_key(
        cls, node: onnx.NodeProto, state, input_dim_signatures: Optional[Dict[int, tuple]] = None
    ) -> Optional[tuple]:
        shape_dim_key = cls._shape_dim_key(node, state, input_dim_signatures)
        if shape_dim_key is not None:
            canonical_output = cls._shape_dim_value_name(shape_dim_key)
            state["assign"](node.output[0], canonical_output)
            return None

        squeezed_shape_dim = cls._squeezed_shape_dim_value(node, state)
        if squeezed_shape_dim is not None:
            state["assign"](node.output[0], squeezed_shape_dim)
            return None

        return (
            node.domain,
            node.op_type,
            tuple(state["value"](name) for name in node.input),
            tuple(state["value"](name) for name in node.output),
            tuple(cls._attribute_key(attr, state) for attr in sorted(node.attribute, key=lambda attr: attr.name)),
        )

    @classmethod
    def _shape_dim_key(
        cls, node: onnx.NodeProto, state, input_dim_signatures: Optional[Dict[int, tuple]]
    ) -> Optional[tuple]:
        if node.op_type != "Shape" or len(node.input) != 1 or len(node.output) != 1 or input_dim_signatures is None:
            return None

        input_index = state["input_index"].get(node.input[0])
        if input_index is None:
            return None

        dims = input_dim_signatures.get(input_index)
        if not dims:
            return None

        attrs = {attr.name: attr.i for attr in node.attribute if attr.name in {"start", "end"}}
        start = attrs.get("start", 0)
        end = attrs.get("end", len(dims))
        if start < 0:
            start += len(dims)
        if end < 0:
            end += len(dims)
        if start < 0 or end > len(dims) or end - start != 1:
            return None

        dim = dims[start]
        if dim is None:
            return None
        return "shape_dim", dim

    @staticmethod
    def _shape_dim_value_name(shape_dim_key: tuple) -> str:
        _, dim = shape_dim_key
        return "shape_dim_" + "_".join(str(part) for part in dim)

    @classmethod
    def _squeezed_shape_dim_value(cls, node: onnx.NodeProto, state) -> Optional[str]:
        if node.op_type != "Squeeze" or len(node.input) != 1 or len(node.output) != 1:
            return None
        input_value = state["value"](node.input[0])
        if not input_value.startswith("shape_dim_"):
            return None
        return f"squeezed_{input_value}"

    @classmethod
    def _attribute_key(cls, attr: onnx.AttributeProto, state) -> tuple:
        attr_type = attr.type
        if attr_type == AttributeProto.FLOAT:
            return attr.name, "f", attr.f
        if attr_type == AttributeProto.INT:
            return attr.name, "i", attr.i
        if attr_type == AttributeProto.STRING:
            return attr.name, "s", attr.s.decode("utf-8", errors="replace")
        if attr_type == AttributeProto.FLOATS:
            return attr.name, "floats", tuple(attr.floats)
        if attr_type == AttributeProto.INTS:
            return attr.name, "ints", tuple(attr.ints)
        if attr_type == AttributeProto.STRINGS:
            return attr.name, "strings", tuple(s.decode("utf-8", errors="replace") for s in attr.strings)
        if attr_type == AttributeProto.TENSOR:
            return attr.name, "t", cls._tensor_key(attr.t)
        if attr_type == AttributeProto.TENSORS:
            return attr.name, "tensors", tuple(cls._tensor_key(tensor) for tensor in attr.tensors)
        if attr_type == AttributeProto.GRAPH:
            return attr.name, "g", cls._graph_key(attr.g)
        if attr_type == AttributeProto.GRAPHS:
            return attr.name, "graphs", tuple(cls._graph_key(graph) for graph in attr.graphs)
        return attr.name, "raw", attr.SerializeToString().hex()

    @classmethod
    def _graph_key(cls, graph: onnx.GraphProto) -> tuple:
        inputs = [value.name for value in graph.input]
        state = cls._new_value_state(inputs)
        return (
            tuple(state["value"](value.name) for value in graph.input),
            tuple(state["value"](value.name) for value in graph.output),
            tuple(cls._tensor_key(tensor) for tensor in graph.initializer),
            tuple(cls._node_key(node, state) for node in graph.node),
        )

    @classmethod
    def _function_input_dim_signatures(cls, model: ModelProto, candidates) -> Dict[str, Dict[int, tuple]]:
        function_by_name = {fn.name: fn for _, fn in candidates}
        value_shapes = cls._graph_value_shape_signatures(model.graph)
        observed_shapes: Dict[str, Dict[int, set]] = {name: {} for name in function_by_name}

        for node in cls._iter_all_nodes(model.graph.node):
            fn = function_by_name.get(node.op_type)
            if fn is None:
                continue
            for idx, input_name in enumerate(node.input[: len(fn.input)]):
                shape = value_shapes.get(input_name)
                if shape is None:
                    continue
                observed_shapes[node.op_type].setdefault(idx, set()).add(shape)

        input_dim_signatures = {}
        for function_name, shapes_by_input in observed_shapes.items():
            stable_shapes = {idx: next(iter(shapes)) for idx, shapes in shapes_by_input.items() if len(shapes) == 1}
            input_dim_signatures[function_name] = stable_shapes
        return input_dim_signatures

    @staticmethod
    def _graph_value_shape_signatures(graph: onnx.GraphProto) -> Dict[str, tuple]:
        value_shapes = {}

        def dim_key(dim, index):
            if dim.dim_param:
                return "sym", dim.dim_param
            if dim.HasField("dim_value"):
                return "value", dim.dim_value
            return "unknown", index

        for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
            tensor_type = value_info.type.tensor_type
            if not tensor_type.HasField("shape"):
                continue
            value_shapes[value_info.name] = tuple(dim_key(dim, idx) for idx, dim in enumerate(tensor_type.shape.dim))

        for initializer in graph.initializer:
            value_shapes[initializer.name] = tuple(("value", dim) for dim in initializer.dims)

        return value_shapes

    @staticmethod
    def _tensor_key(tensor: onnx.TensorProto) -> tuple:
        tensor = copy.deepcopy(tensor)
        tensor.name = ""
        return tensor.data_type, tuple(tensor.dims), tensor.SerializeToString().hex()


class AdapterWeightsToInputsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto, *, adapter_name: str, **kwargs) -> tuple[onnx.ModelProto, bool]:
        transformed = False
        removed_initializers = []

        # Find nodes with lora weights as inputs
        weight_suffix = f".{adapter_name}.weight"
        lora_weight_nodes = {
            inp: node for node in model.graph.node for inp in node.input if inp.endswith(weight_suffix)
        }

        for i, weight in enumerate(model.graph.initializer):
            if weight.name.endswith(weight_suffix):
                transformed = True

                # Create input/output for lora weights
                new_weight_name = weight.name[: -len(weight_suffix)] + ".weight"
                type_proto = onnx.helper.make_tensor_type_proto(weight.data_type, shape=list(weight.dims))
                inp = onnx.ValueInfoProto(name=new_weight_name, type=type_proto)
                out = onnx.ValueInfoProto(name=new_weight_name + "_RetainedState", type=type_proto)
                model.graph.input.append(inp)
                model.graph.output.append(out)

                # Create a node that connects input -> output
                node = onnx.helper.make_node("Identity", [inp.name], [out.name], new_weight_name + "_identity")
                model.graph.node.append(node)

                # Rename weight input
                lora_weight_node = lora_weight_nodes[weight.name]
                for j, inp in enumerate(lora_weight_node.input):
                    if inp == weight.name:
                        lora_weight_node.input[j] = new_weight_name

                # Remove weight initializers
                removed_initializers.append(i)

        if transformed:
            for i in sorted(removed_initializers, reverse=True):
                model.graph.initializer.pop(i)

        return model, transformed


class PruneFakeInitializersTransform(BaseOnnxTransform):
    """Remove initializers backed by FakeTensors from a dynamo onnx_program before serialisation.

    Operates on the torch.onnx.ONNXProgram object returned by dynamo export, not on a
    ModelProto, because FakeTensors cannot survive onnx.load round-trips.
    """

    @classmethod
    def apply(cls, onnx_program) -> bool:
        from torch._subclasses.fake_tensor import FakeTensor

        initializers = onnx_program.model.graph.initializers
        used_names = {name for node in onnx_program.model.graph for name in node.inputs}
        used_names.update(output.name for output in onnx_program.model.graph.outputs)

        pruned = False
        for name in list(initializers):
            const_value = getattr(initializers[name], "const_value", None)
            raw_value = getattr(const_value, "raw", None)
            if isinstance(raw_value, FakeTensor) and name not in used_names:
                del initializers[name]
                pruned = True
        return pruned


class RenameWsubNodesTransform(BaseOnnxTransform):
    """Name local-function operators from their semantic layer parameters."""

    _LAYER_PARAMETER_RE = re.compile(r"(?:^|\.)layers\.\d+\.(?P<role>.+)\.weight$")
    _WEIGHTED_OPS = {"MatMul", "Gemm", "CustomRMSNorm"}

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        transformed = False
        for function in model.functions:
            function_inputs = set(function.input)
            for node in function.node:
                if node.op_type not in cls._WEIGHTED_OPS:
                    continue
                for input_name in node.input[1:]:
                    if input_name not in function_inputs:
                        continue
                    match = cls._LAYER_PARAMETER_RE.search(input_name)
                    if match is None:
                        continue
                    role = match.group("role").replace(".", "/")
                    semantic_name = f"{role}/{node.op_type}"
                    if node.name != semantic_name:
                        node.name = semantic_name
                        transformed = True
                    break
        return transformed


class LocalizeFunctionReduceSumAxesTransform(BaseOnnxTransform):
    """Move constant ReduceSum axes from ONNX function arguments into the function body.

    ONNX subfunction export can lift a constant axes tensor out of a FunctionProto
    and make it a formal argument. That is semantically valid ONNX, but downstream
    compilation then sees ReduceSum axes as a runtime parameter. This transform is
    intentionally narrow: it only rewrites ReduceSum input-1 when that value is a
    function input and every top-level call site passes the same integer constant.
    """

    _INTEGER_TENSOR_TYPES = {
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    }

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        """Localize eligible ReduceSum axes constants for all functions in ``model``.

        Returns ``True`` only when a function signature and its call nodes were
        rewritten. Unsupported cases, including dynamic axes, mismatched constants,
        nested function call sites, and non-ReduceSum uses of the same formal input,
        are left unchanged.
        """
        transformed = False
        graph_constants = cls._collect_graph_constants(model.graph)

        for function in model.functions:
            function_inputs = list(function.input)
            if not function_inputs or cls._has_nested_call_site(model, function):
                continue

            call_sites = cls._find_graph_call_sites(model, function)
            if not call_sites:
                continue

            axes_inputs = cls._find_reduce_sum_axes_inputs(function, function_inputs)
            for axes_name, reduce_nodes in sorted(
                axes_inputs.items(), key=lambda item: function_inputs.index(item[0]), reverse=True
            ):
                formal_index = function_inputs.index(axes_name)
                axes_tensor = cls._resolve_shared_axes_tensor(call_sites, graph_constants, formal_index)
                if axes_tensor is None:
                    continue

                local_axes_name = cls._insert_axes_constant(function, axes_name, axes_tensor)
                for reduce_node in reduce_nodes:
                    reduce_node.input[1] = local_axes_name

                del function.input[formal_index]
                for call_node in call_sites:
                    del call_node.input[formal_index]
                transformed = True

        return transformed

    @classmethod
    def _find_reduce_sum_axes_inputs(
        cls, function: onnx.FunctionProto, function_inputs: list[str]
    ) -> dict[str, list[onnx.NodeProto]]:
        """Return formal inputs used only as ReduceSum axes inside ``function``.

        If a formal input is consumed by any other node or by any ReduceSum input
        other than input 1, removing it from the function signature could change
        graph semantics. Such inputs are excluded.
        """
        formal_inputs = set(function_inputs)
        axes_inputs: dict[str, list[onnx.NodeProto]] = {}
        unsafe_inputs = set()

        for node in function.node:
            for input_index, input_name in enumerate(node.input):
                if input_name not in formal_inputs:
                    continue
                if node.op_type == "ReduceSum" and input_index == 1:
                    axes_inputs.setdefault(input_name, []).append(node)
                else:
                    unsafe_inputs.add(input_name)

        for input_name in unsafe_inputs:
            axes_inputs.pop(input_name, None)
        return axes_inputs

    @staticmethod
    def _find_graph_call_sites(model: ModelProto, function: onnx.FunctionProto) -> list[onnx.NodeProto]:
        """Find top-level graph nodes that call ``function``."""
        return [node for node in model.graph.node if node.op_type == function.name and node.domain == function.domain]

    @staticmethod
    def _has_nested_call_site(model: ModelProto, function: onnx.FunctionProto) -> bool:
        """Return whether ``function`` is called from another FunctionProto.

        The pass intentionally resolves constants only in the main graph. If a
        function is also called from another function, all call sites cannot be
        proven to pass the same graph constant, so the function is skipped.
        """
        for caller_function in model.functions:
            for node in caller_function.node:
                if node.op_type == function.name and node.domain == function.domain:
                    return True
        return False

    @classmethod
    def _collect_graph_constants(cls, graph: onnx.GraphProto) -> dict[str, TensorProto]:
        """Collect graph-scope constants addressable by value name.

        This supports the common exported ``Constant -> FunctionCall`` form and
        small initializer-backed axes. It does not load external tensor data or
        fold producer subgraphs.
        """
        constants_by_name = {initializer.name: initializer for initializer in graph.initializer}
        for node in graph.node:
            if node.op_type != "Constant" or len(node.output) != 1:
                continue
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    constants_by_name[node.output[0]] = attr.t
                    break
        return constants_by_name

    @classmethod
    def _resolve_shared_axes_tensor(
        cls, call_sites: list[onnx.NodeProto], constants_by_name: dict[str, TensorProto], formal_index: int
    ) -> TensorProto | None:
        """Return the shared integer axes tensor passed by every call site.

        ``None`` means at least one call site is dynamic, missing the argument,
        passes a non-integer/non-vector tensor, or passes a different value.
        """
        shared_value = None
        shared_tensor = None

        for call_node in call_sites:
            if len(call_node.input) <= formal_index:
                return None

            axes_tensor = constants_by_name.get(call_node.input[formal_index])
            axes_value = cls._to_integer_axes_value(axes_tensor) if axes_tensor is not None else None
            if axes_value is None:
                return None

            if shared_value is None:
                shared_value = axes_value
                shared_tensor = axes_tensor
            elif not np.array_equal(shared_value, axes_value):
                return None

        return shared_tensor

    @classmethod
    def _to_integer_axes_value(cls, tensor: TensorProto) -> np.ndarray | None:
        """Convert a TensorProto to a valid ReduceSum axes value, or ``None``.

        Valid axes are scalar or 1-D integer tensors. Empty tensors and non-integer
        tensors are rejected because they do not represent an explicit static axes
        list for this transform.
        """
        if tensor.data_type not in cls._INTEGER_TENSOR_TYPES:
            return None
        try:
            value = numpy_helper.to_array(tensor)
        except Exception:
            return None
        if value.ndim > 1 or value.size == 0 or not np.issubdtype(value.dtype, np.integer):
            return None
        return value.astype(np.int64, copy=False)

    @classmethod
    def _insert_axes_constant(cls, function: onnx.FunctionProto, axes_name: str, axes_tensor: TensorProto) -> str:
        """Insert a local Constant node and return its output name."""
        local_axes_name = cls._make_unique_value_name(function, f"{axes_name}_local")
        constant_tensor = TensorProto()
        constant_tensor.CopyFrom(axes_tensor)
        constant_tensor.name = local_axes_name

        function.node.insert(
            0,
            onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=[local_axes_name],
                name=cls._make_unique_node_name(function, f"{local_axes_name}_Constant"),
                value=constant_tensor,
            ),
        )
        return local_axes_name

    @staticmethod
    def _make_unique_value_name(function: onnx.FunctionProto, base_name: str) -> str:
        """Return a value name that does not collide with values in ``function``."""
        used_names = set(function.input) | set(function.output)
        for node in function.node:
            used_names.update(node.input)
            used_names.update(node.output)

        candidate = base_name
        suffix = 0
        while candidate in used_names:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        return candidate

    @staticmethod
    def _make_unique_node_name(function: onnx.FunctionProto, base_name: str) -> str:
        """Return a node name that does not collide with named nodes in ``function``."""
        used_names = {node.name for node in function.node if node.name}
        candidate = base_name
        suffix = 0
        while candidate in used_names:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        return candidate


class OnnxTransformPipeline(BaseOnnxTransform):
    """Pipeline to apply multiple ONNX transformations in sequence."""

    def __init__(self, transforms: list[type[BaseOnnxTransform]]):
        self.transforms = transforms

    def apply(
        self,
        model: ModelProto,
        *,
        model_name: str = "",
        onnx_base_dir: str | None = None,
        file_chunk_size: int = FILE_CHUNK_SIZE_DEFAULT,
        size_threshold: int = SIZE_THRESHOLD_DEFAULT,
        **kwargs,
    ) -> tuple[ModelProto, bool]:
        if not self.transforms:
            return model, False

        # Same logic as before, but replace `transforms` with `self.transforms`
        mapping: dict[str, tuple[TensorProto, str]] = {}
        requested = set(self.transforms)
        applied = {t: False for t in requested}
        f16_applied = False
        do_fp16 = FP16ClipTransform in requested
        do_split = SplitTensorsTransform in requested
        fp16_min, fp16_max = np.finfo(np.float16).min, np.finfo(np.float16).max
        file_num_tracker = {"num": 0, "size": 0}
        if onnx_base_dir is not None:
            external_data_helper.load_external_data_for_model(model, onnx_base_dir)

        if do_fp16 or do_split:
            for tensor in external_data_helper._get_all_tensors(model):
                if do_fp16 and FP16ClipTransform.apply(tensor, onnx_base_dir, fp16_max, fp16_min):
                    f16_applied = True
                applied[FP16ClipTransform] = f16_applied

                if do_split and tensor.HasField("raw_data"):
                    tsize = len(tensor.raw_data)
                    if tsize > size_threshold:
                        if file_num_tracker["size"] + tsize > file_chunk_size:
                            file_num_tracker["num"] += 1
                            file_num_tracker["size"] = tsize
                        else:
                            file_num_tracker["size"] += tsize
                        applied[SplitTensorsTransform] = True
                        SplitTensorsTransform.apply(tensor, model_name, file_num_tracker["num"], mapping)

        def _set_external_data(tensor, file_name):
            external_data_helper.set_external_data(tensor, file_name)

        max_workers = min(32, (os.cpu_count() or 1) * 4)
        logger.info(f"Applying external data mapping with {max_workers} threads")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_set_external_data, tensor, file_name) for tensor, file_name in mapping.values()]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to set external data: {e}")

        # Non-looping transforms
        if CustomOpTransform in requested:
            applied[CustomOpTransform] = CustomOpTransform.apply(
                model, onnx_export_opset=kwargs.get("onnx_export_opset", constants.ONNX_LEGACY_EXPORT_OPSET)
            )

        if RenameFunctionOutputsTransform in requested:
            applied[RenameFunctionOutputsTransform] = RenameFunctionOutputsTransform.apply(
                model, layer_idx=kwargs.get("layer_idx", 0)
            )

        if RenameWsubNodesTransform in requested:
            applied[RenameWsubNodesTransform] = RenameWsubNodesTransform.apply(model)

        if LocalizeFunctionReduceSumAxesTransform in requested:
            applied[LocalizeFunctionReduceSumAxesTransform] = LocalizeFunctionReduceSumAxesTransform.apply(model)

        if PreserveNestedCacheRetainedStateTransform in requested:
            applied[PreserveNestedCacheRetainedStateTransform] = PreserveNestedCacheRetainedStateTransform.apply(
                model, kv_cache_prefix=kwargs.get("kv_cache_prefix")
            )

        if DeduplicateRepeatedSubgraphTransform in requested:
            applied[DeduplicateRepeatedSubgraphTransform] = DeduplicateRepeatedSubgraphTransform.apply(model, **kwargs)

        if RenameRepeatedSubgraphTransform in requested:
            applied[RenameRepeatedSubgraphTransform] = RenameRepeatedSubgraphTransform.apply(model, **kwargs)

        if AdapterWeightsToInputsTransform in requested:
            applied[AdapterWeightsToInputsTransform] = AdapterWeightsToInputsTransform.apply(model, **kwargs)

        for t, done in applied.items():
            logger.info(f"Transform '{t.__name__}' applied={done}")

        return model, any(applied.values())
