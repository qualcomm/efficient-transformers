# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Share equivalent ``invoke_subgraph`` bodies after export, before ONNX lowering.

With ``dynamo=True`` and ONNX subfunctions enabled, every decoder layer is
traced as a separate ``torch.ops.higher_order.invoke_subgraph`` call. The ONNX
exporter turns every GraphModule attribute into an ONNX FunctionProto. This
pass runs on the decomposed ExportedProgram, after export has solved shared
``Dim`` constraints, and points equivalent root-level calls at one body.

The executable region is never edited during comparison. Each merged call
keeps its own operands, including layer weights and cache values; only the
GraphModule body and identifier are shared.
"""

import dataclasses
import operator

import sympy
import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import extract_tensor_metadata
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")

_DEAD_SCALAR_OPS = frozenset(
    {
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.mod,
        operator.neg,
        operator.and_,
        operator.or_,
        operator.not_,
        torch.sym_not,
        torch.sym_max,
        torch.sym_min,
        torch.ops.aten.sym_size.int,
    }
)


@dataclasses.dataclass
class _RegionCall:
    """First call seen for a distinct region body; later matches reuse it."""

    comparison_copy: torch.fx.GraphModule
    attr_node: torch.fx.Node
    identifier: str
    operands: tuple


def _symbolic_key(value):
    """Return a hashable, ShapeEnv-scoped identity for an int or SymInt.

    ``SymInt`` values are deliberately not used directly because they are
    unhashable. Two symbolic sizes share a key only when they belong to the
    same ShapeEnv and expand to the same sympy expression.
    """
    if isinstance(value, torch.SymInt):
        return ("sym", id(value.node.shape_env), sympy.expand(value.node.expr))
    return value


def _operand_key(value):
    """Return a hashable key containing comparator-relevant tensor metadata.

    Every TensorMetadata field is included, including dtype, shape, stride,
    device, layout, and gradient state. Tuple fields are mapped element-wise so
    symbolic dimensions use the same ShapeEnv-aware identity as standalone
    SymInt operands.
    """
    if isinstance(value, torch.Tensor):
        metadata = extract_tensor_metadata(value)
        fields = []
        for field in dataclasses.fields(metadata):
            item = getattr(metadata, field.name)
            if isinstance(item, tuple):
                item = tuple(_symbolic_key(element) for element in item)
            else:
                item = _symbolic_key(item)
            fields.append((field.name, item))
        return ("tensor", tuple(fields))
    return _symbolic_key(value)


def _is_kept_in_comparison(node):
    """Keep every node except unused scalar and shape operations.

    Assertion removal can leave scalar comparisons and arithmetic in different
    positions in otherwise equivalent layer graphs. Tensor operations are
    always retained because they can affect outputs or side effects.
    """
    if node.is_impure():
        return True
    if node.target not in _DEAD_SCALAR_OPS:
        return True
    return isinstance(node.meta.get("val"), torch.Tensor)


def _canonical_region(gm, operands):
    """Build a normalized, non-executable comparison copy of a region body.

    The copy replaces each input size read with one canonical read of the first
    input slot having that symbolic size, and then removes dead scalar/shape
    nodes. Placeholders receive the actual call operands so the comparator sees
    the enclosing export ShapeEnv. The original GraphModule and its metadata
    remain unchanged. Unsupported operand types or arity mismatches return
    ``None`` and prevent sharing that region.
    """
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(operands) or any(
        not isinstance(value, (torch.Tensor, torch.SymInt)) for value in operands
    ):
        return None

    graph = torch.fx.Graph()
    env = {}
    first_slot_by_size = {}
    tensor_inputs = {}

    # Copy placeholders first and record the first input slot for every size.
    # This makes equivalent reads independent of which layer input supplied it.
    for index, (node, value) in enumerate(zip(placeholders, operands)):
        env[node] = graph.node_copy(node, lambda arg: env[arg])
        env[node].meta["example_value"] = value
        if isinstance(value, torch.Tensor):
            tensor_inputs[node] = value
            for dim, size in enumerate(value.shape):
                first_slot_by_size.setdefault(_symbolic_key(size), (index, dim))

    # Map each original size read to its canonical input slot.
    slot_by_size_node = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.sym_size.int and node.args[0] in tensor_inputs:
            size = tensor_inputs[node.args[0]].shape[node.args[1]]
            slot_by_size_node[node] = first_slot_by_size[_symbolic_key(size)]

    # Emit one read per slot in a stable order before copying the body.
    canonical_read_by_slot = {}
    for node, slot in sorted(slot_by_size_node.items(), key=lambda item: item[1]):
        if slot not in canonical_read_by_slot:
            input_index, dim = slot
            read = graph.call_function(torch.ops.aten.sym_size.int, (env[placeholders[input_index]], dim))
            read.meta = node.meta.copy()
            canonical_read_by_slot[slot] = read
        env[node] = canonical_read_by_slot[slot]

    # Copy all remaining nodes, preserving the executable body's operations.
    for node in gm.graph.nodes:
        if node not in env:
            env[node] = graph.node_copy(node, lambda arg: env[arg])

    # DCE applies only to this comparison copy; the executable body is intact.
    graph.eliminate_dead_code(is_impure_node=_is_kept_in_comparison)
    copy = torch.fx.GraphModule(gm, graph)
    copy.meta = gm.meta.copy()
    return copy


def _region_calls(root):
    """Yield root-level invoke-subgraph calls and their GraphModule bodies."""
    for call in root.graph.nodes:
        if call.op != "call_function" or call.target is not torch.ops.higher_order.invoke_subgraph:
            continue
        attr = call.args[0]
        if not isinstance(attr, torch.fx.Node) or attr.op != "get_attr":
            continue
        yield call, attr, root.get_submodule(attr.target)


def _plan_region_merges(root):
    """Plan equivalent region merges without mutating the exported program.

    Calls are bucketed by operand metadata and normalized node count before the
    expensive PyTorch graph comparison. Planning completes before any root
    attribute is changed, so a comparator failure can safely skip reuse.
    """
    from QEfficient.utils.torch_patches import _same_export_region

    representatives = {}
    merges = []
    for call, attr, body in _region_calls(root):
        operands = tuple(arg.meta.get("val") if isinstance(arg, torch.fx.Node) else arg for arg in call.args[2:])
        comparison_copy = _canonical_region(body, operands)
        if comparison_copy is None:
            logger.debug("Subfunction reuse: %s has unsupported operands; kept separate.", attr.target)
            continue
        fake_mode = detect_fake_mode(operands)
        if fake_mode is None and any(isinstance(value, torch.Tensor) for value in operands):
            logger.debug("Subfunction reuse: %s has no fake mode; kept separate.", attr.target)
            continue

        # Metadata bucketing avoids comparing obviously different layer bodies.
        key = (tuple(_operand_key(value) for value in operands), len(comparison_copy.graph.nodes))
        bucket = representatives.setdefault(key, [])
        for representative in bucket:
            if _same_export_region(
                representative.comparison_copy,
                comparison_copy,
                fake_mode,
                representative.operands,
                representative.operands,
            ):
                merges.append((call, representative))
                break
        else:
            bucket.append(_RegionCall(comparison_copy, attr, call.args[1], operands))
    return merges


def _apply_region_merges(root, merges):
    """Point merged calls at representative bodies and remove orphaned attrs.

    Only root-level GraphModule attributes orphaned by this pass are deleted;
    nested attributes referenced from a retained region are left untouched.
    """
    orphaned_targets = set()
    for call, representative in merges:
        old_attr = call.args[0]
        call.args = (representative.attr_node, representative.identifier, *call.args[2:])
        if not old_attr.users:
            orphaned_targets.add(old_attr.target)
            root.graph.erase_node(old_attr)

    still_referenced = {node.target for node in root.graph.nodes if node.op == "get_attr"}
    for target in orphaned_targets - still_referenced:
        root.delete_submodule(target)

    root.graph.lint()
    root.recompile()


def reuse_exported_subfunctions(exported_program):
    """Share equivalent exported region bodies while preserving call operands.

    This is an optimization. Any planning failure returns the original
    ExportedProgram unchanged so export can continue with duplicate functions.
    """
    root = exported_program.graph_module
    try:
        with disable_proxy_modes_tracing():
            merges = _plan_region_merges(root)
    except Exception:  # noqa: BLE001 - reuse is an optimization; export must continue.
        logger.warning(
            "ONNX subfunction reuse was skipped because planning failed; "
            "repeated layers may be exported as duplicate functions.",
            exc_info=True,
        )
        return exported_program

    if not merges:
        return exported_program

    _apply_region_merges(root, merges)
    logger.debug("Subfunction reuse: %d region call(s) now share an earlier body.", len(merges))
    exported_program.validate()
    return exported_program
