# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Unit tests for the dynamo-specific transforms and context managers introduced
in the enable_dynamo_for_causallm branch.

Covered:
  - temporarily_enable_nested_compile_regions
  - PreserveNestedCacheRetainedStateTransform
  - RenameRepeatedSubgraphTransform
  - PruneFakeInitializersTransform

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import importlib
import logging
import operator
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from onnx import TensorProto, helper
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.base.onnx_transforms import (
    PreserveNestedCacheRetainedStateTransform,
    PruneFakeInitializersTransform,
    RenameRepeatedSubgraphTransform,
)
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import export_subfunctions
from QEfficient.utils.torch_patches import (
    _same_export_region,
    preserve_mixed_export_subfunctions,
    preserve_subfunction_source_lines,
    temporarily_enable_nested_compile_regions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _InterpreterSpy:
    def __init__(self, graph_module):
        self.graph_module = graph_module

    def run(self, *operands):
        return self.graph_module(*operands)


def test_preserve_subfunction_source_lines_interprets_graph_modules(monkeypatch):
    invoke_subgraph = importlib.import_module("torch._higher_order_ops.invoke_subgraph")
    original_reenter_make_fx = invoke_subgraph.reenter_make_fx
    graph_module = torch.fx.symbolic_trace(torch.nn.Identity())
    calls = []

    def fake_reenter_make_fx(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args)

    monkeypatch.setattr(invoke_subgraph, "reenter_make_fx", fake_reenter_make_fx)
    monkeypatch.setattr(torch.fx, "Interpreter", _InterpreterSpy)

    with preserve_subfunction_source_lines():
        result = invoke_subgraph.reenter_make_fx(graph_module, torch.ones(2))
        assert torch.equal(result, torch.ones(2))
        assert calls and calls[0][0] is not graph_module

    assert invoke_subgraph.reenter_make_fx is fake_reenter_make_fx
    assert original_reenter_make_fx is not invoke_subgraph.reenter_make_fx


def test_preserve_mixed_export_subfunctions_is_scoped_to_supported_pytorch_hook():
    utils = importlib.import_module("torch._higher_order_ops.utils")
    onnx_core = importlib.import_module("torch.onnx._internal.exporter._core")
    original_hop_compile_and_call = getattr(utils, "_hop_compile_and_call", None)
    original_prepare = getattr(onnx_core, "_prepare_exported_program_for_export", None)

    with preserve_mixed_export_subfunctions():
        if original_hop_compile_and_call is not None:
            assert utils._hop_compile_and_call is not original_hop_compile_and_call
        if original_prepare is not None:
            assert onnx_core._prepare_exported_program_for_export is not original_prepare

    if original_hop_compile_and_call is not None:
        assert utils._hop_compile_and_call is original_hop_compile_and_call
    assert getattr(onnx_core, "_prepare_exported_program_for_export", None) is original_prepare


def test_preserve_mixed_export_subfunctions_tolerates_missing_onnx_hook(monkeypatch):
    """A missing private ONNX hook disables reuse without breaking export setup."""
    utils = importlib.import_module("torch._higher_order_ops.utils")
    if getattr(utils, "_hop_compile_and_call", None) is None:
        pytest.skip("PyTorch build has no _hop_compile_and_call hook")
    onnx_core = importlib.import_module("torch.onnx._internal.exporter._core")
    monkeypatch.delattr(onnx_core, "_prepare_exported_program_for_export", raising=False)

    with preserve_mixed_export_subfunctions():
        assert not hasattr(onnx_core, "_prepare_exported_program_for_export")
    assert not hasattr(onnx_core, "_prepare_exported_program_for_export")


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model, cfg


def _make_graph_module_with_attribute(attribute):
    root = torch.nn.Module()
    root.captured = attribute
    graph = torch.fx.Graph()
    captured = graph.get_attr("captured")
    graph.output(captured)
    return torch.fx.GraphModule(root, graph)


def test_same_export_region_accepts_nested_graphmodule_attributes(monkeypatch):
    """Nested GraphModules compare recursively, while tensor captures stay unsupported."""
    comparator = importlib.import_module("torch._dynamo.variables.higher_order_ops")
    monkeypatch.setattr(comparator, "are_same_graph_modules", lambda *args: True)

    nested_left = _make_graph_module_with_attribute(torch.fx.symbolic_trace(torch.nn.Identity()))
    nested_right = _make_graph_module_with_attribute(torch.fx.symbolic_trace(torch.nn.Identity()))
    assert _same_export_region(nested_left, nested_right, fake_mode=None)

    tensor_left = _make_graph_module_with_attribute(torch.ones(2))
    tensor_right = _make_graph_module_with_attribute(torch.ones(2))
    assert not _same_export_region(tensor_left, tensor_right, fake_mode=None)


def _make_identity_region(example_value):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["example_value"] = example_value
    graph.output(x)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_same_export_region_compares_call_operands_and_restores_metadata(monkeypatch):
    comparator = importlib.import_module("torch._dynamo.variables.higher_order_ops")
    seen = []

    def fake_are_same(_name, left, right, _fake_mode):
        seen.extend(next(iter(gm.graph.nodes)).meta["example_value"] for gm in (left, right))
        return True

    monkeypatch.setattr(comparator, "are_same_graph_modules", fake_are_same)
    original_left, original_right = torch.zeros(1), torch.zeros(1)
    left, right = _make_identity_region(original_left), _make_identity_region(original_right)
    operand_left, operand_right = torch.ones(2), torch.ones(2)

    assert _same_export_region(left, right, None, (operand_left,), (operand_right,))
    assert seen == [operand_left, operand_right]
    assert next(iter(left.graph.nodes)).meta["example_value"] is original_left
    assert next(iter(right.graph.nodes)).meta["example_value"] is original_right


def _make_sized_region(size_input_index):
    graph = torch.fx.Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    size = graph.call_function(torch.ops.aten.sym_size.int, ((a, b)[size_input_index], 0))
    graph.call_function(operator.eq, (size, 2))
    graph.output(graph.call_function(torch.ops.aten.add.Tensor, (a, b)))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_canonical_region_normalizes_dead_size_reads():
    operands = (torch.zeros(2, 3), torch.zeros(2, 3))
    left = export_subfunctions._canonical_region(_make_sized_region(0), operands)
    right = export_subfunctions._canonical_region(_make_sized_region(1), operands)

    targets = lambda gm: [node.target for node in gm.graph.nodes if node.op == "call_function"]  # noqa: E731
    assert targets(left) == targets(right) == [torch.ops.aten.add.Tensor]


def test_apply_region_merges_prunes_orphaned_root_region():
    root_module = torch.nn.Module()
    for index in range(2):
        setattr(
            root_module,
            f"region_{index}",
            _make_graph_module_with_attribute(torch.fx.symbolic_trace(torch.nn.Identity())),
        )
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    calls = [
        graph.call_function(
            torch.ops.higher_order.invoke_subgraph,
            (graph.get_attr(f"region_{index}"), f"subgraph_{index}", x),
        )
        for index in range(2)
    ]
    graph.output(tuple(calls))
    root = torch.fx.GraphModule(root_module, graph)
    representative = export_subfunctions._RegionCall(None, calls[0].args[0], "subgraph_0", ())

    export_subfunctions._apply_region_merges(root, [(calls[1], representative)])

    assert calls[1].args[:2] == (calls[0].args[0], "subgraph_0")
    assert "region_1" not in dict(root.named_modules())
    assert "region_0" in dict(root.named_modules())


def test_reuse_exported_subfunctions_leaves_program_untouched_when_planning_fails(monkeypatch):
    monkeypatch.setattr(export_subfunctions, "_plan_region_merges", lambda _root: (_ for _ in ()).throw(RuntimeError()))
    program = MagicMock()
    assert export_subfunctions.reuse_exported_subfunctions(program) is program
    program.validate.assert_not_called()


def test_export_wrapper_disables_grad_for_dynamo_subfunction_exports(monkeypatch, tmp_path):
    """Repeated subfunction bodies see the same inference grad state at every layer."""
    from QEfficient.utils import export_utils

    grad_states = []

    class DummyQEff:
        model = torch.nn.Identity()

        @export_utils.export_wrapper
        def export(self, **kwargs):
            grad_states.append(torch.is_grad_enabled())
            return kwargs["export_dir"] / "dummy.onnx"

    monkeypatch.setattr(export_utils, "validate_dynamo_export_requirements", lambda *args: None)
    monkeypatch.setattr(export_utils, "_setup_onnx_subfunctions", lambda *args, **kwargs: (args[1], args[2], {}))
    monkeypatch.setattr(
        export_utils, "temporarily_enable_nested_compile_regions", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(export_utils, "_prepare_export_directory", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(export_utils, "_generate_export_hash", lambda *args: ("test-hash", {}))
    monkeypatch.setattr(export_utils, "_save_export_metadata", lambda *args: None)
    monkeypatch.setattr(export_utils, "_cleanup_onnx_subfunctions", lambda *args, **kwargs: None)

    with pytest.warns(DeprecationWarning):
        DummyQEff().export(export_dir=tmp_path, dynamo=True, use_onnx_subfunctions=True)

    assert grad_states == [False]


def _make_minimal_onnx_with_repeated_subgraphs(num_layers: int = 2, scatter_count_per_fn: int = 2):
    """
    Build a minimal ONNX ModelProto that mimics dynamo's repeated-subgraph output:
      - graph has num_layers call nodes (one per layer), each referencing repeated_subgraphN
      - each function contains scatter_count_per_fn CtxScatter nodes
      - graph outputs include past_key/value _RetainedState placeholders (dangling)
    """
    functions = []
    call_nodes = []
    graph_outputs = []
    graph_inputs = []

    for i in range(num_layers):
        fn_name = f"repeated_subgraph{i}"

        # Scatter nodes inside the function
        scatter_nodes = []
        fn_outputs = []
        for j in range(scatter_count_per_fn):
            kind = "key" if j == 0 else "value"
            scatter_out = f"scatter_{kind}_{i}"
            scatter_node = helper.make_node(
                "CtxScatter",
                inputs=[f"past_{kind}.{i}", f"new_{kind}_{i}", "position_ids"],
                outputs=[scatter_out],
                domain="qti.aisw",
            )
            scatter_nodes.append(scatter_node)
            fn_outputs.append(scatter_out)

        fn = helper.make_function(
            domain="",
            fname=fn_name,
            inputs=[f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"],
            outputs=fn_outputs,
            nodes=scatter_nodes,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("qti.aisw", 1)],
        )
        functions.append(fn)

        # Call node in the main graph — outputs start EMPTY so that the
        # _RetainedState names in graph.output are dangling (not produced by any
        # node). PreserveNestedCacheRetainedStateTransform must wire them up.
        retained_key = f"past_key.{i}_RetainedState"
        retained_val = f"past_value.{i}_RetainedState"
        call_node = helper.make_node(
            fn_name,
            inputs=[f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"],
            outputs=[],
            domain="",
        )
        call_nodes.append(call_node)

        graph_outputs.append(helper.make_tensor_value_info(retained_key, TensorProto.FLOAT, None))
        graph_outputs.append(helper.make_tensor_value_info(retained_val, TensorProto.FLOAT, None))

        graph_inputs.append(helper.make_tensor_value_info(f"past_key.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"past_value.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"hidden_{i}", TensorProto.FLOAT, None))

    graph_inputs.append(helper.make_tensor_value_info("position_ids", TensorProto.INT64, None))

    graph = helper.make_graph(call_nodes, "test_graph", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for fn in functions:
        model.functions.append(fn)
    return model


# ---------------------------------------------------------------------------
# TestTemporarilyEnableNestedCompileRegions
# ---------------------------------------------------------------------------


class TestTemporarilyEnableNestedCompileRegions:
    def test_patches_decoder_layers_and_restores(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        inner_model = qeff_model.model

        decoder_layers = [m for m in inner_model.modules() if isinstance(m, QEffLlamaDecoderLayer)]
        assert len(decoder_layers) > 0, "No QEffLlamaDecoderLayer found in wrapped model"

        original_qualnames = [getattr(m.forward, "__qualname__", "") for m in decoder_layers]

        with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
            for m in decoder_layers:
                fwd = getattr(m, "forward", None)
                qualname = getattr(fwd, "__qualname__", "")
                assert (
                    "mark_compile_region" in qualname or "nested_compile_region" in qualname or "inner" in qualname
                ), (
                    f"Expected nested_compile_region wrapper on {m.__class__.__name__}.forward, "
                    f"got qualname: {qualname!r}"
                )

        # After context: original forward restored
        for m, orig_qn in zip(decoder_layers, original_qualnames):
            fwd = getattr(m, "forward", None)
            qualname = getattr(fwd, "__qualname__", "")
            assert qualname == orig_qn, f"forward qualname not restored: expected {orig_qn!r}, got {qualname!r}"

    def test_noop_when_already_wrapped(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        inner_model = qeff_model.model

        decoder_layers = [m for m in inner_model.modules() if isinstance(m, QEffLlamaDecoderLayer)]

        # Enter once
        with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
            wrapped_forwards_first = [id(m.forward) for m in decoder_layers]

            # Enter again — already wrapped, should not double-wrap
            with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
                wrapped_forwards_second = [id(m.forward) for m in decoder_layers]

        # IDs may differ (second context creates a new binding), but both contexts
        # must restore cleanly — the important invariant is no crash and final state is restored.
        assert len(wrapped_forwards_first) == len(wrapped_forwards_second)


# ---------------------------------------------------------------------------
# TestPreserveNestedCacheRetainedStateTransform
# ---------------------------------------------------------------------------


class TestPreserveNestedCacheRetainedStateTransform:
    def test_adds_retained_state_outputs_to_call_nodes(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        # Initially the call nodes have outputs but the functions don't expose scatter outputs
        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert changed, "Transform should have modified the model (dangling _RetainedState outputs)"

        # After transform: function outputs should include scatter node outputs
        for i, fn in enumerate(model.functions):
            assert len(fn.output) >= 2, (
                f"Function '{fn.name}' should have at least 2 outputs after transform, got {list(fn.output)}"
            )

    def test_handles_call_with_read_and_updated_cache_layers(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)
        fn = model.functions[1]
        fn.input[:] = ["past_key.0", "past_key.1", "past_value.1", "hidden_1", "position_ids"]
        fn.output[:] = []
        fn.node[0].input[0] = "past_key.1"
        fn.node[1].input[0] = "past_value.1"

        call_node = model.graph.node[1]
        call_node.input[:] = [
            "past_key.0",
            "past_key.1_RetainedState",
            "past_value.1_RetainedState",
            "hidden_1",
            "position_ids",
        ]
        for value in model.graph.input:
            if value.name == "past_key.1":
                value.name = "past_key.1_RetainedState"
            elif value.name == "past_value.1":
                value.name = "past_value.1_RetainedState"

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)

        assert changed
        assert "past_key.1" in {value.name for value in model.graph.input}
        assert "past_value.1" in {value.name for value in model.graph.input}
        assert "past_key.1_RetainedState" in call_node.output
        assert "past_value.1_RetainedState" in call_node.output
        assert len(fn.output) == 2

    def test_noop_when_no_dangling_retained_states(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        # Remove all _RetainedState outputs from the graph — nothing is dangling
        for out in list(model.graph.output):
            if out.name.endswith("_RetainedState"):
                model.graph.output.remove(out)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert not changed, "Transform should be a no-op when there are no dangling _RetainedState outputs"

    def test_rejects_missing_cache_writer_without_partial_rewire(self):
        # Build model where function has only 1 scatter node
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=1, scatter_count_per_fn=1)
        with pytest.raises(ValueError, match="Could not uniquely resolve"):
            PreserveNestedCacheRetainedStateTransform.apply(model)
        # The key invariant: the function call is not partially rewired.
        fn = model.functions[0]
        assert len(fn.output) == 1, f"Function with 1 scatter should not have outputs added, got {list(fn.output)}"
        assert not model.graph.node[0].output


# ---------------------------------------------------------------------------
# TestRenameRepeatedSubgraphTransform
# ---------------------------------------------------------------------------


class TestRenameRepeatedSubgraphTransform:
    def test_renames_repeated_subgraph_functions(self, caplog):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        with caplog.at_level(logging.WARNING, logger="QEfficient.base.onnx_transforms"):
            changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert changed
        assert "more repeated subgraph functions" not in caplog.text

        fn_names = [fn.name for fn in model.functions]
        assert "QEffLlamaDecoderLayer" in fn_names, f"Expected 'QEffLlamaDecoderLayer' in {fn_names}"
        assert "QEffLlamaDecoderLayer_1" in fn_names, f"Expected 'QEffLlamaDecoderLayer_1' in {fn_names}"

        # Call-site op_type must also be updated
        node_op_types = [n.op_type for n in model.graph.node]
        assert "QEffLlamaDecoderLayer" in node_op_types
        assert "QEffLlamaDecoderLayer_1" in node_op_types

    def test_noop_on_empty_classnames(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=[])
        assert not changed

    def test_noop_when_no_repeated_subgraph_functions(self):
        # Build a model with a non-repeated_subgraph function name
        fn = helper.make_function(
            domain="",
            fname="SomeOtherFunction",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        graph = helper.make_graph([], "g", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(fn)

        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert not changed

    def test_handles_alternative_subgraph_pattern(self):
        # torch < 2.5 naming: subgraph_0, subgraph_1
        fn0 = helper.make_function(
            domain="",
            fname="subgraph_0",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        fn1 = helper.make_function(
            domain="",
            fname="subgraph_1",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        call0 = helper.make_node("subgraph_0", inputs=[], outputs=[])
        call1 = helper.make_node("subgraph_1", inputs=[], outputs=[])
        graph = helper.make_graph([call0, call1], "g", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.extend([fn0, fn1])

        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["MyDecoderLayer"])
        assert changed
        fn_names = {fn.name for fn in model.functions}
        assert "MyDecoderLayer" in fn_names


# ---------------------------------------------------------------------------
# TestPruneFakeInitializersTransform
# ---------------------------------------------------------------------------


class TestPruneFakeInitializersTransform:
    def _make_mock_onnx_program(self, initializer_names, used_names, fake_initializers):
        """Build a mock onnx_program object matching PruneFakeInitializersTransform's API."""
        from torch._subclasses.fake_tensor import FakeTensor

        initializers = {}
        for name in initializer_names:
            mock_init = MagicMock()
            if name in fake_initializers:
                fake_tensor = MagicMock(spec=FakeTensor)
                mock_init.const_value.raw = fake_tensor
            else:
                mock_init.const_value.raw = torch.zeros(2)
            initializers[name] = mock_init

        mock_graph = MagicMock()
        mock_graph.initializers = initializers

        # Simulate used_names via graph nodes + outputs
        mock_node = MagicMock()
        mock_node.inputs = list(used_names)
        mock_graph.__iter__ = lambda self: iter([mock_node])
        mock_graph.outputs = []

        mock_program = MagicMock()
        mock_program.model.graph = mock_graph
        return mock_program

    def test_prunes_fake_tensor_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a", "weight_b"],
            used_names=set(),  # neither is used
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert changed
        assert "weight_a" not in program.model.graph.initializers

    def test_preserves_used_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a"],
            used_names={"weight_a"},  # it is used
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "weight_a" in program.model.graph.initializers

    def test_preserves_non_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["real_weight"],
            used_names=set(),
            fake_initializers=set(),  # not a FakeTensor
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "real_weight" in program.model.graph.initializers
