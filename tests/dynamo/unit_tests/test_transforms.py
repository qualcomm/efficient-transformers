# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CPU-only tests for Dynamo-specific transform helpers."""

from __future__ import annotations

import importlib
import operator
from contextlib import nullcontext
from types import SimpleNamespace
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
from QEfficient.customop.dynamo_ops import DYNAMO_CUSTOM_OP_TABLE
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import export_subfunctions
from QEfficient.utils.torch_patches import (
    _same_export_region,
    preserve_mixed_export_subfunctions,
    preserve_subfunction_source_lines,
    temporarily_enable_nested_compile_regions,
)


def _make_export_region(operation):
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    output = graph.call_function(operation, (value, 2))
    graph.output(output)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _make_exported_region_program(second_operation=torch.ops.aten.add.Tensor):
    class RegionHolder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = _make_export_region(torch.ops.aten.add.Tensor)
            self.second = _make_export_region(second_operation)

    holder = RegionHolder()
    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    fake_value = fake_mode.from_tensor(torch.ones(2))
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    value.meta["val"] = fake_value
    first_attr = graph.get_attr("first")
    first_call = graph.call_function(
        torch.ops.higher_order.invoke_subgraph,
        (first_attr, "first", value),
    )
    second_attr = graph.get_attr("second")
    second_call = graph.call_function(
        torch.ops.higher_order.invoke_subgraph,
        (second_attr, "second", value),
    )
    graph.output((first_call, second_call))
    root = torch.fx.GraphModule(holder, graph)
    for region_name in ("first", "second"):
        for node in getattr(root, region_name).graph.nodes:
            if node.op == "placeholder":
                node.meta["example_value"] = fake_value
    return SimpleNamespace(graph_module=root, validate=lambda: root.graph.lint()), first_call, second_call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _InterpreterSpy:
    def __init__(self, graph_module):
        self.graph_module = graph_module

    def run(self, *operands):
        return self.graph_module(*operands)


def test_minimax_ctx_scatter_gather_ops_have_fake_kernels_and_translations():
    def float_tensor(*shape):
        return torch.empty(shape, dtype=torch.float32, device="meta")

    def int_tensor(*shape):
        return torch.empty(shape, dtype=torch.int32, device="meta")

    cases = (
        (
            torch.ops.qefficient.ctx_scatter_paged_attention.default,
            (float_tensor(4, 2, 8, 3), int_tensor(2, 1), int_tensor(2, 2), float_tensor(2, 2, 2, 3)),
            (4, 2, 8, 3),
        ),
        (
            torch.ops.qefficient.ctx_chunk_scatter_batch.default,
            (float_tensor(1, 4, 8, 3), int_tensor(2, 2), float_tensor(2, 2, 2, 3)),
            (1, 4, 8, 3),
        ),
        (
            torch.ops.qefficient.ctx_gather_blocked_kv_batch.default,
            (float_tensor(1, 4, 8, 3), int_tensor(1, 4, 5)),
            (1, 4, 5, 3),
        ),
        (
            torch.ops.qefficient.ctx_gather_paged_attention.default,
            (float_tensor(4, 2, 8, 3), int_tensor(2, 1), int_tensor(2, 5)),
            (2, 2, 5, 3),
        ),
        (
            torch.ops.qefficient.ctx_gather_paged_kv_dp.default,
            (float_tensor(8, 4, 16, 3), int_tensor(2, 4)),
            (1, 4, 32, 3),
        ),
        (
            torch.ops.qefficient.m3_ctx_scatter.default,
            (float_tensor(2, 1, 8, 3), int_tensor(2, 2), float_tensor(2, 1, 2, 3)),
            (2, 1, 8, 3),
        ),
        (
            torch.ops.qefficient.ctx_paged_scatter_dp.default,
            (float_tensor(8, 4, 16, 3), int_tensor(1, 4, 2), int_tensor(1, 4, 2), float_tensor(1, 4, 2, 3)),
            (8, 4, 16, 3),
        ),
        (
            torch.ops.qefficient.ctx_gather_blocked_kv_dp.default,
            (float_tensor(2, 4, 8, 3), int_tensor(2, 4, 5)),
            (2, 4, 5, 3),
        ),
        (
            torch.ops.qefficient.ctx_gather_block_range_kv_dp.default,
            (float_tensor(2, 4, 6, 8, 3), int_tensor(2, 4, 2)),
            (2, 4, 2, 8, 3),
        ),
    )

    for op, args, expected_shape in cases:
        assert op in DYNAMO_CUSTOM_OP_TABLE
        output = op(*args)
        assert output.shape == expected_shape
        assert output.dtype == torch.float32
        assert output.device.type == "meta"


def test_minimax_sequence_chunk_is_safe_inside_nested_compile_region():
    from QEfficient.transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import _dynamic_sequence_chunks

    op = torch.ops.qefficient.compile_length_sequence_chunk.default
    fake_result = op(torch.empty((2, 3, 16), dtype=torch.bool, device="meta"), -1, 4, 2, 16)

    assert fake_result.shape == (2, 3, 4)
    assert fake_result.dtype == torch.bool
    assert op in DYNAMO_CUSTOM_OP_TABLE

    class ChunkModule(torch.nn.Module):
        @torch.compiler.nested_compile_region
        def forward(self, tensor):
            return _dynamic_sequence_chunks(tensor, num_chunks=4, dim=1, compile_axis_size=16)[1]

    source = torch.arange(128).view(1, 16, 8)
    sequence_length = torch.export.Dim("sequence_length", min=4, max=16)
    exported = torch.export.export(
        ChunkModule(),
        (source,),
        dynamic_shapes={"args": ({1: sequence_length},)},
    )
    result = exported.module()(source)

    assert torch.equal(result, source[:, 4:8])


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


def test_reuse_exported_subfunctions_merges_equivalent_regions():
    from QEfficient.utils.export_subfunctions import reuse_exported_subfunctions

    exported_program, first_call, second_call = _make_exported_region_program()

    result = reuse_exported_subfunctions(exported_program)

    assert result is exported_program
    assert first_call.args[0] is second_call.args[0]
    assert not hasattr(exported_program.graph_module, "second")


def test_reuse_exported_subfunctions_keeps_non_equivalent_regions_separate():
    from QEfficient.utils.export_subfunctions import reuse_exported_subfunctions

    exported_program, first_call, second_call = _make_exported_region_program(torch.ops.aten.mul.Tensor)

    reuse_exported_subfunctions(exported_program)

    assert first_call.args[0] is not second_call.args[0]
    assert hasattr(exported_program.graph_module, "first")
    assert hasattr(exported_program.graph_module, "second")


def test_reuse_exported_subfunctions_preserves_program_on_planning_failure(monkeypatch):
    import QEfficient.utils.export_subfunctions as export_subfunctions

    exported_program, _, _ = _make_exported_region_program()
    original_graph = str(exported_program.graph_module.graph)

    def fail_planning(_root):
        raise RuntimeError("planning failure")

    monkeypatch.setattr(export_subfunctions, "_plan_region_merges", fail_planning)

    result = export_subfunctions.reuse_exported_subfunctions(exported_program)

    assert result is exported_program
    assert str(exported_program.graph_module.graph) == original_graph


def test_preserve_mixed_export_subfunctions_restores_private_hooks():
    import QEfficient.utils.torch_patches as torch_patches

    utils = importlib.import_module("torch._higher_order_ops.utils")
    original_hop_compile = utils._hop_compile_and_call
    onnx_core = importlib.import_module("torch.onnx._internal.exporter._core")
    original_prepare = getattr(onnx_core, "_prepare_exported_program_for_export", None)
    if original_prepare is None:
        pytest.skip("PyTorch ONNX exporter has no subfunction preparation hook")

    with torch_patches.preserve_mixed_export_subfunctions():
        assert utils._hop_compile_and_call is not original_hop_compile
        assert onnx_core._prepare_exported_program_for_export is not original_prepare

    assert utils._hop_compile_and_call is original_hop_compile
    assert onnx_core._prepare_exported_program_for_export is original_prepare


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
    """Build a minimal ONNX model that mimics Dynamo repeated subgraphs."""
    functions = []
    call_nodes = []
    graph_outputs = []
    graph_inputs = []

    for i in range(num_layers):
        fn_name = f"repeated_subgraph{i}"

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

        # Empty call outputs leave graph _RetainedState outputs dangling.
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


def _make_minimal_glm_onnx_with_repeated_subgraph():
    """Build a GLM function with three cache writes and one unrelated MoE scatter."""
    cache_names = ["compressed_kv.0", "k_pe.0", "indexer_key.0"]
    formal_names = ["arg10_1", "arg12_1", "arg18_1"]
    nodes = [
        helper.make_node(
            "GlmPagedScatter",
            [formal, "updates", "position_ids"],
            [f"{cache_name}_updated"],
            domain="qti.aisw",
        )
        for cache_name, formal in zip(cache_names, formal_names)
    ]
    nodes.append(
        helper.make_node(
            "CtxScatter3D",
            ["moe_buffer", "updates", "position_ids"],
            ["moe_scatter_output"],
            domain="qti.aisw",
        )
    )
    fn = helper.make_function(
        "",
        "repeated_subgraph0",
        [*formal_names, "moe_buffer", "updates", "position_ids"],
        [],
        nodes,
        [helper.make_opsetid("", 17), helper.make_opsetid("qti.aisw", 1)],
    )
    retained_names = [f"{name}_RetainedState" for name in cache_names]
    call = helper.make_node(
        "repeated_subgraph0",
        [*retained_names, "moe_buffer", "updates", "position_ids"],
        [],
    )
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in retained_names]
    inputs.extend(
        helper.make_tensor_value_info(name, dtype, None)
        for name, dtype in (
            ("moe_buffer", TensorProto.FLOAT),
            ("updates", TensorProto.FLOAT),
            ("position_ids", TensorProto.INT64),
        )
    )
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in retained_names]
    model = helper.make_model(helper.make_graph([call], "glm_test_graph", inputs, outputs))
    model.functions.append(fn)
    return model


def _make_function_model(functions):
    call_nodes = [
        helper.make_node(fn.name, inputs=[f"x{i}", f"w{i}"], outputs=[f"y{i}"]) for i, fn in enumerate(functions)
    ]
    graph_inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for node in call_nodes for name in node.input
    ]
    graph_outputs = [helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, None) for node in call_nodes]
    graph = helper.make_graph(call_nodes, "g", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.functions.extend(functions)
    return model


def _make_linear_function(name, input_name="hidden", weight_name="weight", output_name="out", node_name="matmul"):
    node = helper.make_node("MatMul", inputs=[input_name, weight_name], outputs=[output_name], name=node_name)
    return helper.make_function(
        domain="",
        fname=name,
        inputs=[input_name, weight_name],
        outputs=[output_name],
        nodes=[node],
        opset_imports=[helper.make_opsetid("", 17)],
    )


def _make_shape_dim_function(name, *, shape_input, start, end):
    shape_node = helper.make_node("Shape", inputs=[shape_input], outputs=["dim"], start=start, end=end)
    reshape_node = helper.make_node("Reshape", inputs=["data", "dim"], outputs=["out"])
    return helper.make_function(
        domain="",
        fname=name,
        inputs=["data", "cache", "mask"],
        outputs=["out"],
        nodes=[shape_node, reshape_node],
        opset_imports=[helper.make_opsetid("", 17)],
    )


def _make_shape_dim_model(fn0, fn1, *, second_cache_dim="ctx_len"):
    call0 = helper.make_node(fn0.name, inputs=["data0", "cache0", "mask0"], outputs=["out0"])
    call1 = helper.make_node(fn1.name, inputs=["data1", "cache1", "mask1"], outputs=["out1"])
    graph_inputs = [
        helper.make_tensor_value_info("data0", TensorProto.FLOAT, ["ctx_len"]),
        helper.make_tensor_value_info("cache0", TensorProto.FLOAT, ["batch_size", 2, "ctx_len", 256]),
        helper.make_tensor_value_info("mask0", TensorProto.BOOL, ["batch_size", 1, 32, "ctx_len"]),
        helper.make_tensor_value_info("data1", TensorProto.FLOAT, [second_cache_dim]),
        helper.make_tensor_value_info("cache1", TensorProto.FLOAT, ["batch_size", 2, second_cache_dim, 256]),
        helper.make_tensor_value_info("mask1", TensorProto.BOOL, ["batch_size", 1, 32, "ctx_len"]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("out0", TensorProto.FLOAT, ["ctx_len"]),
        helper.make_tensor_value_info("out1", TensorProto.FLOAT, [second_cache_dim]),
    ]
    graph = helper.make_graph([call0, call1], "g", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.functions.extend([fn0, fn1])
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

        for m, orig_qn in zip(decoder_layers, original_qualnames):
            fwd = getattr(m, "forward", None)
            qualname = getattr(fwd, "__qualname__", "")
            assert qualname == orig_qn, f"forward qualname not restored: expected {orig_qn!r}, got {qualname!r}"

    def test_noop_when_already_wrapped(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        inner_model = qeff_model.model

        decoder_layers = [m for m in inner_model.modules() if isinstance(m, QEffLlamaDecoderLayer)]

        with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
            wrapped_forwards_first = [id(m.forward) for m in decoder_layers]

            with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
                wrapped_forwards_second = [id(m.forward) for m in decoder_layers]

        assert len(wrapped_forwards_first) == len(wrapped_forwards_second)


class TestPreserveNestedCacheRetainedStateTransform:
    def test_adds_retained_state_outputs_to_call_nodes(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert changed, "Transform should have modified the model (dangling _RetainedState outputs)"

        for i, fn in enumerate(model.functions):
            assert len(fn.output) >= 2, (
                f"Function '{fn.name}' should have at least 2 outputs after transform, got {list(fn.output)}"
            )

    def test_noop_when_no_dangling_retained_states(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        for out in list(model.graph.output):
            if out.name.endswith("_RetainedState"):
                model.graph.output.remove(out)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert not changed, "Transform should be a no-op when there are no dangling _RetainedState outputs"

    def test_exposes_index_key_retained_state_for_sparse_layers(self):
        layer_idx = 3
        fn = helper.make_function(
            domain="",
            fname="repeated_subgraph0",
            inputs=["index_key_arg", "past_key_arg", "past_value_arg", "update"],
            outputs=["hidden_out"],
            nodes=[
                helper.make_node(
                    "M3CtxScatter", ["index_key_arg", "pos", "update"], ["index_scatter"], domain="qti.aisw"
                ),
                helper.make_node("CtxScatter", ["past_key_arg", "pos", "update"], ["key_scatter"], domain="qti.aisw"),
                helper.make_node(
                    "CtxScatter", ["past_value_arg", "pos", "update"], ["value_scatter"], domain="qti.aisw"
                ),
                helper.make_node("Identity", ["update"], ["hidden_out"]),
            ],
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("qti.aisw", 1)],
        )
        call_inputs = [
            f"index_key.{layer_idx}_RetainedState",
            f"past_key.{layer_idx}",
            f"past_value.{layer_idx}",
            "update",
        ]
        graph = helper.make_graph(
            [helper.make_node("repeated_subgraph0", call_inputs, ["hidden_out"])],
            "sparse_graph",
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in call_inputs],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                for name in (
                    "hidden_out",
                    f"past_key.{layer_idx}_RetainedState",
                    f"past_value.{layer_idx}_RetainedState",
                    f"index_key.{layer_idx}_RetainedState",
                )
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(fn)

        assert not PreserveNestedCacheRetainedStateTransform.apply(model)
        assert PreserveNestedCacheRetainedStateTransform.apply(model, model_type="minimax_m3_vl")

        assert list(model.functions[0].output) == ["hidden_out", "key_scatter", "value_scatter", "index_scatter"]
        call_node = model.graph.node[0]
        assert list(call_node.output) == [
            "hidden_out",
            f"past_key.{layer_idx}_RetainedState",
            f"past_value.{layer_idx}_RetainedState",
            f"index_key.{layer_idx}_RetainedState",
        ]
        input_names = [value.name for value in model.graph.input]
        assert f"index_key.{layer_idx}" in input_names
        assert f"index_key.{layer_idx}_RetainedState" not in input_names

    def test_rejects_missing_cache_writer_without_partial_rewire(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=1, scatter_count_per_fn=1)
        with pytest.raises(ValueError, match="Could not uniquely resolve"):
            PreserveNestedCacheRetainedStateTransform.apply(model)
        fn = model.functions[0]
        assert len(fn.output) == 1, f"Function with 1 scatter should not have outputs added, got {list(fn.output)}"
        assert not model.graph.node[0].output, "Function call must not be partially rewired"

    def test_wires_glm_cache_outputs_and_ignores_moe_scatter(self):
        model = _make_minimal_glm_onnx_with_repeated_subgraph()

        assert PreserveNestedCacheRetainedStateTransform.apply(model)

        expected_inputs = ["compressed_kv.0", "k_pe.0", "indexer_key.0"]
        expected_outputs = [f"{name}_RetainedState" for name in expected_inputs]
        assert list(model.graph.node[0].input[:3]) == expected_inputs
        assert list(model.graph.node[0].output) == expected_outputs
        assert list(model.functions[0].output) == [
            "compressed_kv.0_updated",
            "k_pe.0_updated",
            "indexer_key.0_updated",
        ]
        assert "moe_scatter_output" not in model.functions[0].output
        assert [value.name for value in model.graph.input[:3]] == expected_inputs

    def test_glm_transform_is_idempotent(self):
        model = _make_minimal_glm_onnx_with_repeated_subgraph()
        assert PreserveNestedCacheRetainedStateTransform.apply(model)
        assert not PreserveNestedCacheRetainedStateTransform.apply(model)

    def test_preserves_v4_retained_state_with_nested_subfunction(self):
        retained_state = "past_local_kv_cache.0_InternalRetainedState"
        state_input = "past_local_kv_cache.0"
        function = helper.make_function(
            domain="",
            fname="repeated_subgraph0",
            inputs=[retained_state, "position_ids", "updates"],
            outputs=[],
            nodes=[
                helper.make_node(
                    "V4CtxScatter1D",
                    inputs=[retained_state, "position_ids", "updates"],
                    outputs=["updated_local_kv_cache"],
                    domain="com.qualcomm.cloud",
                )
            ],
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.qualcomm.cloud", 1)],
        )
        call_node = helper.make_node(
            "repeated_subgraph0",
            inputs=[retained_state, "position_ids", "updates"],
            outputs=[],
        )
        graph = helper.make_graph(
            [call_node],
            "test_graph",
            [
                helper.make_tensor_value_info(retained_state, TensorProto.FLOAT, None),
                helper.make_tensor_value_info("position_ids", TensorProto.INT32, None),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, None),
            ],
            [helper.make_tensor_value_info(retained_state, TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(function)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)

        assert changed
        assert list(model.functions[0].output) == ["updated_local_kv_cache"]
        assert list(model.graph.node[0].input) == [state_input, "position_ids", "updates"]
        assert list(model.graph.node[0].output) == [retained_state]

    def test_preserves_v4_retained_state_from_plain_cache_input(self):
        cache_input = "past_local_kv_cache.0"
        retained_state = f"{cache_input}_RetainedState"
        function = helper.make_function(
            domain="",
            fname="repeated_subgraph0",
            inputs=[cache_input, "position_ids", "updates"],
            outputs=["hidden_states"],
            nodes=[
                helper.make_node(
                    "V4CtxScatter1D",
                    inputs=[cache_input, "position_ids", "updates"],
                    outputs=["updated_local_kv_cache"],
                    domain="com.qualcomm.cloud",
                )
            ],
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.qualcomm.cloud", 1)],
        )
        call_node = helper.make_node(
            "repeated_subgraph0",
            inputs=[cache_input, "position_ids", "updates"],
            outputs=["hidden_states"],
        )
        graph = helper.make_graph(
            [call_node],
            "test_graph",
            [
                helper.make_tensor_value_info(cache_input, TensorProto.FLOAT, None),
                helper.make_tensor_value_info("position_ids", TensorProto.INT32, None),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, None),
            ],
            [
                helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, None),
                helper.make_tensor_value_info(retained_state, TensorProto.FLOAT, None),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(function)

        assert PreserveNestedCacheRetainedStateTransform.apply(model)
        assert list(model.functions[0].output) == ["hidden_states", "updated_local_kv_cache"]
        assert list(model.graph.node[0].input) == [cache_input, "position_ids", "updates"]
        assert list(model.graph.node[0].output) == ["hidden_states", retained_state]

    def test_aligns_prefill_retained_state_function_outputs_with_call_outputs(self):
        retained_states = [
            "past_local_kv_cache.0_RetainedState",
            "past_compressor_kv_state.0_RetainedState",
            "past_compressor_score_state.0_RetainedState",
            "past_compressed_kv_cache.0_RetainedState",
            "past_indexer_kv_state.0_RetainedState",
            "past_indexer_score_state.0_RetainedState",
            "past_indexer_kv_cache.0_RetainedState",
        ]
        state_inputs = [name.removesuffix("_RetainedState") for name in retained_states]
        function_outputs = [
            "hidden_states",
            "local_kv",
            state_inputs[1],
            state_inputs[2],
            state_inputs[4],
            state_inputs[5],
        ]
        function = helper.make_function(
            domain="",
            fname="repeated_subgraph0",
            inputs=["hidden_states", *retained_states],
            outputs=function_outputs,
            nodes=[
                helper.make_node("Identity", ["hidden_states"], ["hidden_states"]),
                helper.make_node("Identity", [retained_states[0]], ["local_kv"]),
                helper.make_node("Identity", [retained_states[1]], [state_inputs[1]]),
                helper.make_node("Identity", [retained_states[2]], [state_inputs[2]]),
                helper.make_node("Identity", [retained_states[3]], ["compressed_kv"]),
                helper.make_node("Identity", [retained_states[4]], [state_inputs[4]]),
                helper.make_node("Identity", [retained_states[5]], [state_inputs[5]]),
                helper.make_node("Identity", [retained_states[6]], ["indexer_kv"]),
            ],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        call_node = helper.make_node(
            "repeated_subgraph0",
            inputs=["hidden_states", *retained_states],
            outputs=["hidden_states_out"],
        )
        graph = helper.make_graph(
            [call_node],
            "test_graph",
            [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, None)]
            + [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in retained_states],
            [helper.make_tensor_value_info("hidden_states_out", TensorProto.FLOAT, None)]
            + [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in retained_states],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(function)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)

        assert changed
        assert list(model.graph.node[0].output) == ["hidden_states_out", *retained_states]
        assert list(model.functions[0].output) == [
            "hidden_states",
            "local_kv",
            state_inputs[1],
            state_inputs[2],
            "compressed_kv",
            state_inputs[4],
            state_inputs[5],
            "indexer_kv",
        ]
        assert len(model.functions[0].output) == len(model.graph.node[0].output)

    def test_preserves_final_scatternd_cache_state_from_prefill_update_chain(self):
        retained_state = "past_local_kv_cache.0_RetainedState"
        state_input = "past_local_kv_cache.0"
        function = helper.make_function(
            domain="",
            fname="repeated_subgraph0",
            inputs=[retained_state, "indices", "updates"],
            outputs=[],
            nodes=[
                helper.make_node("Identity", inputs=[retained_state], outputs=["cache_view"]),
                helper.make_node("ScatterND", inputs=["cache_view", "indices", "updates"], outputs=["cache_step_0"]),
                helper.make_node("Identity", inputs=["cache_step_0"], outputs=["next_cache_view"]),
                helper.make_node(
                    "ScatterND", inputs=["next_cache_view", "indices", "updates"], outputs=["cache_step_1"]
                ),
            ],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        call_node = helper.make_node(
            "repeated_subgraph0",
            inputs=[retained_state, "indices", "updates"],
            outputs=[],
        )
        graph = helper.make_graph(
            [call_node],
            "test_graph",
            [
                helper.make_tensor_value_info(retained_state, TensorProto.FLOAT, None),
                helper.make_tensor_value_info("indices", TensorProto.INT64, None),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, None),
            ],
            [helper.make_tensor_value_info(retained_state, TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(function)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)

        assert changed
        assert list(model.functions[0].output) == ["cache_step_1"]
        assert list(model.graph.node[0].input) == [state_input, "indices", "updates"]
        assert list(model.graph.node[0].output) == [retained_state]

    def test_ignores_scatter_update_dependency_when_resolving_cache_state(self):
        retained_state = "past_compressed_kv_cache.0_RetainedState"
        function = helper.make_function(
            domain="",
            fname="repeated_subgraph0",
            inputs=[retained_state, "indices", "updates"],
            outputs=[],
            nodes=[
                helper.make_node("Identity", [retained_state], ["cache_view"]),
                helper.make_node("ScatterND", ["cache_view", "indices", "updates"], ["cache_result"]),
                helper.make_node("ScatterND", ["updates", "indices", "cache_result"], ["unrelated_result"]),
            ],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        call_node = helper.make_node(
            "repeated_subgraph0",
            inputs=[retained_state, "indices", "updates"],
            outputs=[],
        )
        graph = helper.make_graph(
            [call_node],
            "test_graph",
            [
                helper.make_tensor_value_info(retained_state, TensorProto.FLOAT, None),
                helper.make_tensor_value_info("indices", TensorProto.INT64, None),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, None),
            ],
            [helper.make_tensor_value_info(retained_state, TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(function)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)

        assert changed
        assert list(model.functions[0].output) == ["cache_result"]
        assert list(model.graph.node[0].output) == [retained_state]

    def test_drops_unchanged_prefill_retained_state(self):
        retained_state = "past_compressor_kv_state.0_RetainedState"
        state_input = "past_compressor_kv_state.0"
        function = helper.make_function(
            domain="",
            fname="repeated_subgraph0",
            inputs=[retained_state, "hidden_states"],
            outputs=["output"],
            nodes=[
                helper.make_node("Add", inputs=[retained_state, "hidden_states"], outputs=["output"]),
            ],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        call_node = helper.make_node(
            "repeated_subgraph0",
            inputs=[retained_state, "hidden_states"],
            outputs=["output"],
        )
        graph = helper.make_graph(
            [call_node],
            "test_graph",
            [
                helper.make_tensor_value_info(retained_state, TensorProto.FLOAT, None),
                helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, None),
            ],
            [
                helper.make_tensor_value_info("output", TensorProto.FLOAT, None),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(function)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)

        assert changed
        assert list(model.functions[0].output) == ["output"]
        assert list(model.graph.node[0].input) == [state_input, "hidden_states"]
        assert list(model.graph.node[0].output) == ["output"]
        assert [output.name for output in model.graph.output] == ["output"]


# ---------------------------------------------------------------------------
# TestRenameRepeatedSubgraphTransform
# ---------------------------------------------------------------------------


class TestRenameRepeatedSubgraphTransform:
    def test_renames_repeated_subgraph_functions(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert changed

        fn_names = [fn.name for fn in model.functions]
        assert "QEffLlamaDecoderLayer" in fn_names, f"Expected 'QEffLlamaDecoderLayer' in {fn_names}"
        assert "QEffLlamaDecoderLayer_1" in fn_names, f"Expected 'QEffLlamaDecoderLayer_1' in {fn_names}"

        node_op_types = [n.op_type for n in model.graph.node]
        assert "QEffLlamaDecoderLayer" in node_op_types
        assert "QEffLlamaDecoderLayer_1" in node_op_types

        # Call nodes get readable names instead of dynamo's invoke_subgraph names
        assert [n.name for n in model.graph.node] == ["QEffLlamaDecoderLayer.0", "QEffLlamaDecoderLayer_1.1"]

    def test_renames_stale_subgraph_names_inside_loop_bodies(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=1)
        body = helper.make_function(
            domain="",
            fname="repeated_subgraph0__while_loop_cond_graph_0",
            inputs=["x"],
            outputs=["y"],
            nodes=[
                helper.make_node("Identity", ["x"], ["y"], name="node_repeated_subgraph0__while_loop_cond_graph_0_7")
            ],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.functions.append(body)

        RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])

        body_fn = next(fn for fn in model.functions if fn.name.endswith("__while_loop_cond_graph_0"))
        assert body_fn.name == "QEffLlamaDecoderLayer__while_loop_cond_graph_0"
        assert body_fn.node[0].name == "node_QEffLlamaDecoderLayer__while_loop_cond_graph_0_7"

    def test_noop_on_empty_classnames(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=[])
        assert not changed

    def test_noop_when_no_repeated_subgraph_functions(self):
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


def test_ctx_scatter_gather_dynamo_translation_table_complete():
    from QEfficient.customop.dynamo_ops import DYNAMO_CUSTOM_OP_TABLE

    required_ctx_ops = {
        "ctx_scatter": torch.ops.qefficient.ctx_scatter.default,
        "ctx_scatter_3d": torch.ops.qefficient.ctx_scatter_3d.default,
        "ctx_scatter_3d_generalized": torch.ops.qefficient.ctx_scatter_3d_generalized.default,
        "ctx_scatter_3d_int": torch.ops.qefficient.ctx_scatter_3d_int.default,
        "ctx_scatter_cb": torch.ops.qefficient.ctx_scatter_cb.default,
        "ctx_scatter_cb_3d": torch.ops.qefficient.ctx_scatter_cb_3d.default,
        "ctx_chunk_scatter_batch": torch.ops.qefficient.ctx_chunk_scatter_batch.default,
        "ctx_gather": torch.ops.qefficient.ctx_gather.default,
        "ctx_gather_3d": torch.ops.qefficient.ctx_gather_3d.default,
        "ctx_gather_3d_generalized": torch.ops.qefficient.ctx_gather_3d_generalized.default,
        "ctx_gather_cb": torch.ops.qefficient.ctx_gather_cb.default,
        "ctx_gather_cb_3d": torch.ops.qefficient.ctx_gather_cb_3d.default,
        "ctx_gather_blocked_kv": torch.ops.qefficient.ctx_gather_blocked_kv.default,
        "ctx_gather_blocked_kv_cb": torch.ops.qefficient.ctx_gather_blocked_kv_cb.default,
        "ctx_gather_blocked_kv_batch": torch.ops.qefficient.ctx_gather_blocked_kv_batch.default,
    }

    missing = [name for name, op in required_ctx_ops.items() if op not in DYNAMO_CUSTOM_OP_TABLE]
    assert not missing
