# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import onnx
import torch
from onnx import ModelProto, TensorProto, external_data_helper, numpy_helper

from QEfficient.customop.ctx_scatter_gather import (
    CtxChunkScatterBatch,
    CtxChunkScatterBatchFunc,
    CtxGather,
    CtxGather3D,
    CtxGatherBlockedKV,
    CtxGatherBlockedKVBatch,
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFunc3DGeneralized,
    CtxGatherFuncBlockedKV,
    CtxGatherFuncBlockedKVBatch,
    CtxScatter,
    CtxScatter3D,
    CtxScatter3DInt,
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
from QEfficient.customop.onnxscript_utils import get_onnxscript_func
from QEfficient.customop.quantization_ops import CastToUInt4, CastToUInt4Func
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.utils import constants
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
        cls, tensor: TensorProto, model_name: str, file_num: int, mapping: Dict[str, Tuple[TensorProto, str]]
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
        "CtxGatherFunc": (CtxGatherFunc, CtxGather),
        "CtxGatherFunc3D": (CtxGatherFunc3D, CtxGather3D),
        "CtxGatherFunc3DGeneralized": (CtxGatherFunc3DGeneralized, CtxGather3D),
        "CtxScatterFuncCB3D": (CtxScatterFuncCB3D, CtxScatterCB3D),
        "CtxGatherFuncCB3D": (CtxGatherFuncCB3D, CtxGatherCB3D),
        "CtxGatherFuncBlockedKV": (CtxGatherFuncBlockedKV, CtxGatherBlockedKV),
        "CtxGatherFuncBlockedKVCB": (CtxGatherFuncBlockedKVCB, CtxGatherBlockedKVCB),
        "CtxScatterFuncCB": (CtxScatterFuncCB, CtxScatterCB),
        "CtxGatherFuncCB": (CtxGatherFuncCB, CtxGatherCB),
        "CastToUInt4": (CastToUInt4Func, CastToUInt4),
        "CtxChunkScatterBatchFunc": (CtxChunkScatterBatchFunc, CtxChunkScatterBatch),
        "CtxGatherFuncBlockedKVBatch": (CtxGatherFuncBlockedKVBatch, CtxGatherBlockedKVBatch),
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
    def _ensure_opset_imports(container: Union[ModelProto, onnx.FunctionProto], domain: str, version: int) -> None:
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

    # Match past_key.N / past_value.N regardless of any suffix that follows
    # (plain, _RetainedState, or _<prefix>_RetainedState for kv_cache_prefix).
    _KV_INPUT_RE = re.compile(r"^past_(key|value)\.(\d+)")

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
        }
    )

    @classmethod
    def _resolve_kv_scatter_outputs(cls, fn, node, layer_idx: str) -> dict[str, str] | None:
        """Resolve nested KV scatter outputs from the function-call data flow.

        ONNX function-call inputs bind positionally. Each cache scatter consumes
        the cache buffer as its first input, so the matching scatter is identified
        from that buffer's function argument rather than output names or node order.
        """
        function_cache_inputs: dict[str, str] = {}
        for input_index, input_name in enumerate(node.input):
            match = cls._KV_INPUT_RE.match(input_name)
            if match is None:
                continue
            kind, input_layer_idx = match.groups()
            if input_layer_idx != layer_idx:
                continue
            if input_index >= len(fn.input):
                raise ValueError(
                    f"Nested function '{fn.name}' has no input at position {input_index} for "
                    f"call-node cache input '{input_name}'."
                )
            function_cache_inputs[kind] = fn.input[input_index]

        if set(function_cache_inputs) != {"key", "value"}:
            return None

        scatter_outputs = {}
        for kind, function_input in function_cache_inputs.items():
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
                    f"Could not uniquely resolve the nested past_{kind}.{layer_idx} cache writer in function "
                    f"'{fn.name}': expected one CtxScatter* with data input '{function_input}', "
                    f"found {len(writers)} ({writer_names})."
                )
            scatter_outputs[kind] = writers[0].output[0]

        return scatter_outputs

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        graph = model.graph
        produced_names = {name for node in graph.node for name in node.output}
        dangling_retained_outputs = {
            out.name for out in graph.output if out.name.endswith("_RetainedState") and out.name not in produced_names
        }
        if not dangling_retained_outputs:
            return False

        fn_by_name = {fn.name: fn for fn in model.functions}
        changed = False
        kv_rename_map: Dict[str, str] = {}

        for node in graph.node:
            fn = fn_by_name.get(node.op_type)
            if fn is None:
                continue

            # Identify layer index from KV inputs on this call node.
            layer_idx = None
            kv_inputs = {}
            for inp_name in node.input:
                match = cls._KV_INPUT_RE.match(inp_name)
                if match is None:
                    continue
                kind, idx = match.groups()
                layer_idx = idx if layer_idx is None else layer_idx
                if layer_idx != idx:
                    kv_inputs = {}
                    break
                kv_inputs[kind] = inp_name

            if layer_idx is None or set(kv_inputs) != {"key", "value"}:
                continue

            desired_outputs = [
                f"past_key.{layer_idx}_RetainedState",
                f"past_value.{layer_idx}_RetainedState",
            ]
            # Skip layers whose retained-state outputs are not dangling —
            # either the graph is already correctly wired or a previous call
            # to this transform already fixed them.
            if not any(name in dangling_retained_outputs for name in desired_outputs):
                continue

            if not all(name in dangling_retained_outputs for name in desired_outputs):
                raise ValueError(
                    f"Nested function '{fn.name}' has partially dangling KV retained-state outputs for layer "
                    f"{layer_idx}: expected both {desired_outputs}."
                )

            scatter_outputs = cls._resolve_kv_scatter_outputs(fn, node, layer_idx)
            if scatter_outputs is None:
                continue

            # Expose scatter outputs in the function's output list, rename KV
            # inputs and append retained-state output names to the call node.
            # Both writers are resolved before this block, so graph rewiring is atomic.
            for kind, desired_output in zip(("key", "value"), desired_outputs):
                scatter_output = scatter_outputs[kind]
                if scatter_output not in fn.output:
                    fn.output.append(scatter_output)
                    changed = True

                retained_input = kv_inputs[kind]
                plain_input = f"past_{kind}.{layer_idx}"
                if retained_input.endswith("_RetainedState"):
                    kv_rename_map[retained_input] = plain_input

                if desired_output not in node.output:
                    node.output.append(desired_output)
                    changed = True

        if kv_rename_map:
            changed |= cls._rename_graph_inputs_bulk(graph, kv_rename_map)
        return changed

    @staticmethod
    def _rename_graph_inputs_bulk(graph: onnx.GraphProto, rename_map: Dict[str, str]) -> bool:
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

    @staticmethod
    def _rename_op_types(nodes, old_to_new: Dict[str, str]) -> None:
        for node in RenameRepeatedSubgraphTransform._iter_all_nodes(nodes):
            if node.op_type in old_to_new:
                node.op_type = old_to_new[node.op_type]

    @classmethod
    def apply(cls, model: ModelProto, target_classnames: Optional[List[str]] = None, **kwargs) -> bool:
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


class AdapterWeightsToInputsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto, *, adapter_name: str, **kwargs) -> Tuple[onnx.ModelProto, bool]:
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
        cls, function: onnx.FunctionProto, function_inputs: List[str]
    ) -> Dict[str, List[onnx.NodeProto]]:
        """Return formal inputs used only as ReduceSum axes inside ``function``.

        If a formal input is consumed by any other node or by any ReduceSum input
        other than input 1, removing it from the function signature could change
        graph semantics. Such inputs are excluded.
        """
        formal_inputs = set(function_inputs)
        axes_inputs: Dict[str, List[onnx.NodeProto]] = {}
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
    def _find_graph_call_sites(model: ModelProto, function: onnx.FunctionProto) -> List[onnx.NodeProto]:
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
    def _collect_graph_constants(cls, graph: onnx.GraphProto) -> Dict[str, TensorProto]:
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
        cls, call_sites: List[onnx.NodeProto], constants_by_name: Dict[str, TensorProto], formal_index: int
    ) -> Optional[TensorProto]:
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
    def _to_integer_axes_value(cls, tensor: TensorProto) -> Optional[np.ndarray]:
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

    def __init__(self, transforms: List[Type[BaseOnnxTransform]]):
        self.transforms = transforms

    def apply(
        self,
        model: ModelProto,
        *,
        model_name: str = "",
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = FILE_CHUNK_SIZE_DEFAULT,
        size_threshold: int = SIZE_THRESHOLD_DEFAULT,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        if not self.transforms:
            return model, False

        # Same logic as before, but replace `transforms` with `self.transforms`
        mapping: Dict[str, Tuple[TensorProto, str]] = {}
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
            applied[PreserveNestedCacheRetainedStateTransform] = PreserveNestedCacheRetainedStateTransform.apply(model)

        if RenameRepeatedSubgraphTransform in requested:
            applied[RenameRepeatedSubgraphTransform] = RenameRepeatedSubgraphTransform.apply(model, **kwargs)

        if AdapterWeightsToInputsTransform in requested:
            applied[AdapterWeightsToInputsTransform] = AdapterWeightsToInputsTransform.apply(model, **kwargs)

        for t, done in applied.items():
            logger.info(f"Transform '{t.__name__}' applied={done}")

        return model, any(applied.values())
