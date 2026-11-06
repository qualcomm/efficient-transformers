# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import numpy as np
import onnx
import pytest

from QEfficient.base.onnx_transforms import (
    DynamoSemanticNodeNameTransform,
    FP16ClipTransform,
    LocalizeFunctionReduceSumAxesTransform,
    OnnxTransformPipeline,
    RenameWsubNodesTransform,
    SplitTensorsTransform,
)


def _add_name_scopes(node, scopes):
    metadata = node.metadata_props.add()
    metadata.key = "pkg.torch.onnx.name_scopes"
    metadata.value = repr(scopes)


def test_dynamo_semantic_node_name_transform_uses_module_metadata():
    graph_node = onnx.helper.make_node("MatMul", ["x", "weight"], ["y"], name="node_linear")
    _add_name_scopes(
        graph_node,
        ["", "model", "model.layers.12", "model.layers.12.self_attn.q_proj", "linear"],
    )
    function_node = onnx.helper.make_node("MatMul", ["hidden", "weight"], ["output"], name="node_linear")
    _add_name_scopes(
        function_node,
        ["", "model", "model.layers.12", "model.layers.12.self_attn.q_proj", "linear"],
    )
    function = onnx.helper.make_function(
        "test",
        "DecoderLayer",
        ["hidden", "weight"],
        ["output"],
        [function_node],
        [onnx.helper.make_opsetid("", 17)],
    )
    graph = onnx.helper.make_graph(
        [graph_node],
        "test",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])],
        initializer=[onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, [4, 4], [1.0] * 16)],
    )
    model = onnx.helper.make_model(
        graph,
        functions=[function],
        opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("test", 1)],
    )

    assert DynamoSemanticNodeNameTransform.apply(model)
    assert model.graph.node[0].name == "/model/layers.12/self_attn/q_proj/linear__node_linear"
    assert model.functions[0].node[0].name == "/DecoderLayer/self_attn/q_proj/linear__node_linear"
    assert list(model.graph.node[0].input) == ["x", "weight"]
    assert list(model.graph.node[0].output) == ["y"]
    assert not DynamoSemanticNodeNameTransform.apply(model)


def test_dynamo_semantic_node_name_transform_names_local_function_calls():
    call_node = onnx.helper.make_node("DecoderLayer", ["x"], ["y"], domain="test", name="node_invoke_subgraph")
    _add_name_scopes(call_node, ["", "model", "model.layers.12", "invoke_subgraph"])
    function = onnx.helper.make_function(
        "test",
        "DecoderLayer",
        ["x"],
        ["y"],
        [onnx.helper.make_node("Identity", ["x"], ["y"], name="node_identity")],
        [onnx.helper.make_opsetid("", 17)],
    )
    graph = onnx.helper.make_graph(
        [call_node],
        "test",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph,
        functions=[function],
        opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("test", 1)],
    )

    assert DynamoSemanticNodeNameTransform.apply(model)
    assert model.graph.node[0].name == "/model/layers.12/DecoderLayer__node_invoke_subgraph"


def test_fp16clip_transform():
    test_onnx = onnx.parser.parse_model("""
    <
        ir_version: 8,
        opset_import: ["" : 17]
    >
    test_fp16clip (float [n, 32] x) => (float [n, 32] y)
    <
        float val1 = {65505.0},
        int64[1] slice_ends = {2147483647},
        float zero = {0.0}
    >
    {
        mask = Greater(x, zero)
        val2 = Constant<value = float {-1e7}>()
        masked = Where(mask, val1, val2)
        slice_starts = Constant<value = int64[1] {0}>()
        y = Slice(masked, slice_starts, slice_ends)
    }
    """)
    onnx.checker.check_model(test_onnx, True, True, True)

    onnx_transforms = OnnxTransformPipeline(transforms=[FP16ClipTransform])
    transformed_onnx, transformed = onnx_transforms.apply(test_onnx, model_name="")
    assert transformed
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[0]) == 65504.0
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[1]) == 2147483647
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.node[1].attribute[0].t) == -65504.0


def test_fp16clip_transform_external(tmp_path):
    external_tensors_file = "fp32_min.raw"
    test_onnx = onnx.parser.parse_model(f"""
    <
        ir_version: 8,
        opset_import: ["" : 17]
    >
    test_fp16clip (float [n, 32] x) => (float [n, 32] y)
    <
        float min_val = [ "location": "{external_tensors_file}" ],
        float zero = {{0.0}}
    >
    {{
        mask = Greater(x, zero)
        y = Where(mask, x, min_val)
    }}
    """)

    # Write onnx and external_data
    onnx_path = tmp_path / "test_fp16_clip_external.onnx"
    onnx.save(test_onnx, onnx_path)
    np.array(-1e10, dtype="float32").tofile(tmp_path / external_tensors_file)
    onnx.checker.check_model(onnx_path, True, True, True)

    onnx_transforms = OnnxTransformPipeline(transforms=[FP16ClipTransform])
    transformed_onnx, transformed = onnx_transforms.apply(test_onnx, model_name="", onnx_base_dir=str(tmp_path))
    assert transformed
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[0]) == -65504.0


def test_rename_wsub_nodes_transform():
    weighted_ops = [
        ("self_attn.q_proj", "MatMul"),
        ("self_attn.k_proj", "MatMul"),
        ("self_attn.v_proj", "MatMul"),
        ("self_attn.o_proj", "MatMul"),
        ("mlp.gate_proj", "MatMul"),
        ("mlp.up_proj", "MatMul"),
        ("mlp.down_proj", "MatMul"),
        ("input_layernorm", "CustomRMSNorm"),
    ]
    weight_names = [f"model.layers.1.{role}.weight" for role, _ in weighted_ops]
    nodes = [
        onnx.helper.make_node(op_type, ["hidden", weight_name], [f"output_{index}"], name=f"{op_type}_{index}")
        for index, ((_, op_type), weight_name) in enumerate(zip(weighted_ops, weight_names))
    ]
    function = onnx.helper.make_function(
        "test",
        "DecoderLayer",
        ["hidden", *weight_names],
        [nodes[-1].output[0]],
        nodes,
        [onnx.helper.make_opsetid("", 17)],
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "test", [], []),
        functions=[function],
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )

    assert RenameWsubNodesTransform.apply(model)
    transformed_function = model.functions[0]
    assert [node.name for node in transformed_function.node] == [
        role.replace(".", "/") + "/" + op_type for role, op_type in weighted_ops
    ]
    assert not RenameWsubNodesTransform.apply(model)


def _make_constant_node(output_name, values, dtype=np.int64):
    tensor = onnx.numpy_helper.from_array(np.asarray(values, dtype=dtype), name=output_name)
    return onnx.helper.make_node("Constant", inputs=[], outputs=[output_name], value=tensor)


def _make_reduce_sum_function(function_name="QKNorm"):
    reduce_node = onnx.helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=1)
    return onnx.helper.make_function(
        "qeff.test",
        function_name,
        ["x", "axes"],
        ["y"],
        [reduce_node],
        [onnx.helper.make_opsetid("", 17)],
    )


def _reduced_shape(axes_value):
    shape = [2, 3, 4, 5]
    for axis in axes_value:
        shape[axis] = 1
    return shape


def _make_reduce_sum_function_model(axes_values, function_name="QKNorm"):
    constants = []
    calls = []
    inputs = []
    outputs = []
    for idx, axes_value in enumerate(axes_values):
        x_name = f"x{idx}"
        axes_name = f"axes{idx}"
        y_name = f"y{idx}"
        constants.append(_make_constant_node(axes_name, axes_value))
        calls.append(onnx.helper.make_node(function_name, [x_name, axes_name], [y_name], domain="qeff.test"))
        inputs.append(onnx.helper.make_tensor_value_info(x_name, onnx.TensorProto.FLOAT, [2, 3, 4, 5]))
        outputs.append(onnx.helper.make_tensor_value_info(y_name, onnx.TensorProto.FLOAT, _reduced_shape(axes_value)))

    graph = onnx.helper.make_graph(constants + calls, "g", inputs, outputs)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("qeff.test", 1)],
        functions=[_make_reduce_sum_function(function_name)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _reduce_sum_axes_input(function):
    reduce_sum = next(node for node in function.node if node.op_type == "ReduceSum")
    return reduce_sum.input[1]


def _constant_value_by_output(nodes, output_name):
    for node in nodes:
        if node.op_type != "Constant" or list(node.output) != [output_name]:
            continue
        value_attr = next(attr for attr in node.attribute if attr.name == "value")
        return onnx.numpy_helper.to_array(value_attr.t)
    raise AssertionError(f"Constant node producing {output_name!r} not found")


class TestLocalizeFunctionReduceSumAxesTransform:
    @pytest.mark.parametrize("axes", [[-1], [2], [2, 3]])
    def test_localizes_constant_reduce_sum_axes(self, axes):
        model = _make_reduce_sum_function_model([axes])

        transformed = LocalizeFunctionReduceSumAxesTransform.apply(model)

        assert transformed
        function = model.functions[0]
        call_node = next(node for node in model.graph.node if node.op_type == "QKNorm")
        local_axes_name = _reduce_sum_axes_input(function)

        assert list(function.input) == ["x"]
        assert list(call_node.input) == ["x0"]
        np.testing.assert_array_equal(_constant_value_by_output(function.node, local_axes_name), np.asarray(axes))
        onnx.checker.check_model(model)

    def test_noop_for_dynamic_reduce_sum_axes(self):
        function = _make_reduce_sum_function()
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("QKNorm", ["x", "axes"], ["y"], domain="qeff.test")],
            "g",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, 4]),
                onnx.helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1]),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, 4])],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("qeff.test", 1)],
            functions=[function],
        )
        model.ir_version = 10

        transformed = LocalizeFunctionReduceSumAxesTransform.apply(model)

        assert not transformed
        assert list(model.functions[0].input) == ["x", "axes"]
        assert list(model.graph.node[0].input) == ["x", "axes"]
        onnx.checker.check_model(model)

    def test_noop_for_different_call_site_constants(self):
        model = _make_reduce_sum_function_model([[-1], [2]])

        transformed = LocalizeFunctionReduceSumAxesTransform.apply(model)

        assert not transformed
        assert list(model.functions[0].input) == ["x", "axes"]
        assert [list(node.input) for node in model.graph.node if node.op_type == "QKNorm"] == [
            ["x0", "axes0"],
            ["x1", "axes1"],
        ]
        onnx.checker.check_model(model)

    def test_noop_for_already_local_constant_axes(self):
        local_axes = _make_constant_node("axes_local", [-1])
        reduce_node = onnx.helper.make_node("ReduceSum", ["x", "axes_local"], ["y"], keepdims=1)
        function = onnx.helper.make_function(
            "qeff.test",
            "QKNorm",
            ["x"],
            ["y"],
            [local_axes, reduce_node],
            [onnx.helper.make_opsetid("", 17)],
        )
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("QKNorm", ["x"], ["y"], domain="qeff.test")],
            "g",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, 4])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, 4])],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("qeff.test", 1)],
            functions=[function],
        )
        model.ir_version = 10

        transformed = LocalizeFunctionReduceSumAxesTransform.apply(model)

        assert not transformed
        assert list(model.functions[0].input) == ["x"]
        assert _reduce_sum_axes_input(model.functions[0]) == "axes_local"
        onnx.checker.check_model(model)

    def test_noop_for_non_reduce_sum_constant_function_input(self):
        function = onnx.helper.make_function(
            "qeff.test",
            "AddBias",
            ["x", "bias"],
            ["y"],
            [onnx.helper.make_node("Add", ["x", "bias"], ["y"])],
            [onnx.helper.make_opsetid("", 17)],
        )
        graph = onnx.helper.make_graph(
            [
                _make_constant_node("bias", [1.0], dtype=np.float32),
                onnx.helper.make_node("AddBias", ["x", "bias"], ["y"], domain="qeff.test"),
            ],
            "g",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("qeff.test", 1)],
            functions=[function],
        )
        model.ir_version = 10

        transformed = LocalizeFunctionReduceSumAxesTransform.apply(model)

        assert not transformed
        assert list(model.functions[0].input) == ["x", "bias"]
        assert list(model.graph.node[1].input) == ["x", "bias"]
        onnx.checker.check_model(model)

    def test_localized_model_matches_ort_output_when_available(self):
        ort = pytest.importorskip("onnxruntime")
        model = _make_reduce_sum_function_model([[-1]])
        reference = onnx.ModelProto()
        reference.CopyFrom(model)

        transformed = LocalizeFunctionReduceSumAxesTransform.apply(model)

        assert transformed
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        reference_session = ort.InferenceSession(reference.SerializeToString(), session_options)
        transformed_session = ort.InferenceSession(model.SerializeToString(), session_options)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        expected = reference_session.run(None, {"x0": x})[0]
        actual = transformed_session.run(None, {"x0": x})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_split_tensors_transform(tmp_path):
    external_tensors_file = "tensors.raw"
    test_onnx = onnx.parser.parse_model(f"""
    <
        ir_version: 8,
        opset_import: ["": 17]
    >
    test_split () => ()
    <
        float[1, 32] tensor0 = [ "location": "{external_tensors_file}", "offset": "0", "length": "{32 * 4}" ],
        float[1, 32] tensor1 = [ "location": "{external_tensors_file}", "offset": "{32 * 4}", "length": "{32 * 4}" ],
        float[1, 16] tensor2 = [ "location": "{external_tensors_file}", "offset": "{64 * 4}", "length": "{16 * 4}" ]
    >
    {{
    }}
    """)

    # Write onnx and external_data
    onnx_path = tmp_path / "test_split_pre.onnx"
    onnx.save(test_onnx, onnx_path)
    tensors = np.random.rand(32 + 32 + 16).astype("float32")
    tensors.tofile(tmp_path / external_tensors_file)
    onnx.checker.check_model(onnx_path, True, True, True)

    onnx_transforms = OnnxTransformPipeline(transforms=[SplitTensorsTransform])
    trans_onnx, transformed = onnx_transforms.apply(
        test_onnx,
        model_name="test_split",
        onnx_base_dir=str(tmp_path),
        file_chunk_size=32 * 4,
        size_threshold=16 * 4,
    )

    tensor0_ext_data = onnx.external_data_helper.ExternalDataInfo(trans_onnx.graph.initializer[0])
    assert tensor0_ext_data.location == "test_split_0.onnx.data"

    tensor1_ext_data = onnx.external_data_helper.ExternalDataInfo(trans_onnx.graph.initializer[1])
    assert tensor1_ext_data.location == "test_split_1.onnx.data"

    tensor2 = trans_onnx.graph.initializer[2]
    assert tensor2.data_location == onnx.TensorProto.DataLocation.Value("DEFAULT")
    assert np.all(onnx.numpy_helper.to_array(tensor2) == tensors[-16:])

    # Save and test if all files are saved
    onnx_path = tmp_path / "test_split.onnx"
    onnx.save(trans_onnx, onnx_path)
    assert onnx_path.is_file()
    assert onnx_path.with_name(onnx_path.name.replace(".onnx", "_0.onnx.data")).is_file()
    assert onnx_path.with_name(onnx_path.name.replace(".onnx", "_1.onnx.data")).is_file()
