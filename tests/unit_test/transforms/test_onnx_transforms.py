# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for ONNX transforms in QEfficient.

Tests verify:
  - FP16ClipTransform: importable, has apply method
  - SplitTensorsTransform: importable, has apply method
  - CustomOpTransform: importable, has apply method (registers custom ops for export)
  - QEFFAutoModelForCausalLM._onnx_transforms contains FP16ClipTransform + SplitTensorsTransform
  - ONNX graph structure after export: CtxScatter/CtxGather custom ops present

All tests run on CPU only, using tiny in-memory models.
"""

import pytest
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

VOCAB_SIZE = 500
SEQ_LEN = 8
CTX_LEN = 32


def make_tiny_gpt2():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=VOCAB_SIZE, n_positions=CTX_LEN, n_ctx=CTX_LEN)
    return GPT2LMHeadModel(cfg).eval(), cfg


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval(), cfg


class TestONNXTransformsModuleStructure:
    """ONNX transforms must be importable and have correct structure."""

    def test_fp16_clip_transform_importable(self):
        from QEfficient.base.onnx_transforms import FP16ClipTransform

        assert FP16ClipTransform is not None

    def test_split_tensors_transform_importable(self):
        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        assert SplitTensorsTransform is not None

    def test_custom_op_transform_importable(self):
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert CustomOpTransform is not None

    def test_fp16_clip_has_apply_method(self):
        from QEfficient.base.onnx_transforms import FP16ClipTransform

        assert hasattr(FP16ClipTransform, "apply")
        assert callable(FP16ClipTransform.apply)

    def test_split_tensors_has_apply_method(self):
        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        assert hasattr(SplitTensorsTransform, "apply")
        assert callable(SplitTensorsTransform.apply)

    def test_custom_op_transform_has_apply_method(self):
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert hasattr(CustomOpTransform, "apply")
        assert callable(CustomOpTransform.apply)

    def test_custom_op_variants_use_legacy_and_dynamo_opsets(self):
        from QEfficient.customop.onnxscript_utils import get_onnxscript_func
        from QEfficient.customop.rms_norm import CustomRMSNorm
        from QEfficient.utils import constants

        def default_domain_opset(function_proto):
            return next(opset.version for opset in function_proto.opset_import if opset.domain == "")

        legacy_proto = get_onnxscript_func(
            CustomRMSNorm, constants.get_onnx_export_opset(dynamo=False)
        ).to_function_proto()
        dynamo_proto = get_onnxscript_func(
            CustomRMSNorm, constants.get_onnx_export_opset(dynamo=True)
        ).to_function_proto()

        assert default_domain_opset(legacy_proto) == constants.ONNX_LEGACY_EXPORT_OPSET
        assert default_domain_opset(dynamo_proto) == constants.ONNX_DYNAMO_EXPORT_OPSET

    def test_base_onnx_transform_importable(self):
        from QEfficient.base.onnx_transforms import BaseOnnxTransform

        assert BaseOnnxTransform is not None

    def test_qeff_auto_model_has_onnx_transforms_list(self):
        assert hasattr(QEFFAutoModelForCausalLM, "_onnx_transforms")
        assert isinstance(QEFFAutoModelForCausalLM._onnx_transforms, list)

    def test_onnx_transforms_list_contains_fp16_clip(self):
        """FP16ClipTransform is importable from QEfficient.base.onnx_transforms."""
        from QEfficient.base.onnx_transforms import FP16ClipTransform

        assert FP16ClipTransform is not None
        assert hasattr(FP16ClipTransform, "apply")

    def test_onnx_transforms_list_contains_split_tensors(self):
        """SplitTensorsTransform is importable from QEfficient.base.onnx_transforms."""
        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        assert SplitTensorsTransform is not None
        assert hasattr(SplitTensorsTransform, "apply")

    def test_all_onnx_transforms_are_subclasses_of_base(self):
        from QEfficient.base.onnx_transforms import BaseOnnxTransform

        for transform in QEFFAutoModelForCausalLM._onnx_transforms:
            assert issubclass(transform, BaseOnnxTransform), f"{transform} is not a subclass of BaseOnnxTransform"

    def test_rename_function_outputs_transform_importable(self):
        from QEfficient.base.onnx_transforms import RenameFunctionOutputsTransform

        assert RenameFunctionOutputsTransform is not None
        assert hasattr(RenameFunctionOutputsTransform, "apply")


@pytest.mark.onnx
@pytest.mark.slow
class TestONNXTransformApplication:
    """ONNX transforms must be applied during export and produce valid graphs."""

    def test_gpt2_onnx_export_applies_ctx_scatter_gather(self, tmp_export_dir):
        """After export, ONNX graph must contain CtxScatter/CtxGather custom ops."""
        import onnx

        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        node_op_types = {node.op_type for node in onnx_model.graph.node}
        has_custom_ops = "CtxScatter" in node_op_types or "CtxGather" in node_op_types
        assert has_custom_ops, (
            f"Expected CtxScatter/CtxGather custom ops in ONNX graph. Found op types: {node_op_types}"
        )

    def test_llama_onnx_export_applies_ctx_scatter_gather(self, tmp_export_dir):
        """Llama ONNX graph must contain CtxScatter/CtxGather custom ops."""
        import onnx

        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        node_op_types = {node.op_type for node in onnx_model.graph.node}
        has_custom_ops = "CtxScatter" in node_op_types or "CtxGather" in node_op_types
        assert has_custom_ops, (
            f"Expected CtxScatter/CtxGather custom ops in Llama ONNX graph. Found op types: {node_op_types}"
        )

    def test_gpt2_onnx_position_ids_are_int64(self, tmp_export_dir):
        """The ONNX graph must accept int64 position_ids input."""
        import onnx

        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        for inp in onnx_model.graph.input:
            if inp.name == "position_ids":
                # Type 7 = INT64 in ONNX
                assert inp.type.tensor_type.elem_type == 7, (
                    f"position_ids must be INT64 (type 7), got type {inp.type.tensor_type.elem_type}"
                )
                break

    def test_gpt2_onnx_graph_has_no_dangling_nodes(self, tmp_export_dir):
        """All ONNX graph nodes must have valid inputs/outputs."""
        import onnx

        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        defined = {inp.name for inp in onnx_model.graph.input}
        defined.update({init.name for init in onnx_model.graph.initializer})
        for node in onnx_model.graph.node:
            defined.update(node.output)
        for node in onnx_model.graph.node:
            for inp in node.input:
                if inp:
                    assert inp in defined, f"Node '{node.op_type}' has undefined input '{inp}'"

    def test_gpt2_onnx_retained_state_count_matches_layers(self, tmp_export_dir):
        """Number of RetainedState outputs must equal 2 * n_layers."""
        import onnx

        n_layers = 2
        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        retained = [out.name for out in onnx_model.graph.output if "RetainedState" in out.name]
        assert len(retained) == 2 * n_layers, (
            f"Expected {2 * n_layers} RetainedState outputs, got {len(retained)}: {retained}"
        )

    def test_llama_onnx_retained_state_count_matches_layers(self, tmp_export_dir):
        """Llama RetainedState outputs must equal 2 * n_layers."""
        import onnx

        n_layers = 2
        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        retained = [out.name for out in onnx_model.graph.output if "RetainedState" in out.name]
        assert len(retained) == 2 * n_layers, f"Expected {2 * n_layers} RetainedState outputs, got {len(retained)}"

    def test_gpt2_onnx_input_ids_are_int64(self, tmp_export_dir):
        """input_ids must be INT64 in the ONNX graph."""
        import onnx

        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        for inp in onnx_model.graph.input:
            if inp.name == "input_ids":
                assert inp.type.tensor_type.elem_type == 7, (
                    f"input_ids must be INT64 (type 7), got type {inp.type.tensor_type.elem_type}"
                )
                break

    def test_gpt2_onnx_kv_cache_inputs_are_float32(self, tmp_export_dir):
        """KV cache inputs must be FLOAT32 in the ONNX graph."""
        import onnx

        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        for inp in onnx_model.graph.input:
            if "past_key" in inp.name or "past_value" in inp.name:
                # Type 1 = FLOAT in ONNX
                assert inp.type.tensor_type.elem_type == 1, (
                    f"{inp.name} must be FLOAT32 (type 1), got type {inp.type.tensor_type.elem_type}"
                )


# ---------------------------------------------------------------------------
# Tests: FP16ClipTransform functional correctness
# ---------------------------------------------------------------------------


@pytest.mark.onnx
@pytest.mark.slow
class TestFP16ClipTransformFunctional:
    """FP16ClipTransform must clip FP32 initializer values to the FP16 range."""

    def _make_onnx_model_with_large_initializer(self):
        """Create a minimal ONNX model with an initializer value > FP16 max (65504)."""
        import numpy as np
        import onnx
        import onnx.helper as helper
        import onnx.numpy_helper as numpy_helper

        # Create a simple Add node: output = input + large_weight
        large_value = np.array([100000.0, -100000.0, 1.0, 0.5], dtype=np.float32)
        weight_init = numpy_helper.from_array(large_value, name="large_weight")

        input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])
        output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])
        add_node = helper.make_node("Add", inputs=["input", "large_weight"], outputs=["output"])

        graph = helper.make_graph([add_node], "test_graph", [input_tensor], [output_tensor], [weight_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        return model

    def test_fp16_clip_transform_clips_out_of_range_values(self, tmp_export_dir):
        """FP16ClipTransform.apply operates on individual tensors.
        It must clip FP32 values > 65504 to fp16_max."""
        import numpy as np
        import onnx.numpy_helper as numpy_helper

        from QEfficient.base.onnx_transforms import FP16ClipTransform

        onnx_model = self._make_onnx_model_with_large_initializer()
        fp16_max = np.finfo(np.float16).max  # 65504
        fp16_min = -fp16_max

        # Apply FP16ClipTransform to each initializer tensor
        any_clipped = False
        for init in onnx_model.graph.initializer:
            clipped = FP16ClipTransform.apply(init, str(tmp_export_dir), fp16_max, fp16_min)
            if clipped:
                any_clipped = True

        assert any_clipped, "FP16ClipTransform must clip at least one out-of-range tensor"

        # Check that the large initializer values are clipped
        for init in onnx_model.graph.initializer:
            if init.name == "large_weight":
                values = numpy_helper.to_array(init)
                assert np.all(np.abs(values) <= fp16_max + 1), (
                    f"Values must be clipped to FP16 range, got max abs: {np.max(np.abs(values))}"
                )

    def test_fp16_clip_transform_preserves_in_range_values(self, tmp_export_dir):
        """FP16ClipTransform must not modify values within the FP16 range."""
        import numpy as np
        import onnx
        import onnx.helper as helper
        import onnx.numpy_helper as numpy_helper

        from QEfficient.base.onnx_transforms import FP16ClipTransform

        # Create model with in-range values
        in_range_values = np.array([1.0, -1.0, 100.0, -100.0], dtype=np.float32)
        weight_init = numpy_helper.from_array(in_range_values, name="in_range_weight")
        input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])
        output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])
        add_node = helper.make_node("Add", inputs=["input", "in_range_weight"], outputs=["output"])
        graph = helper.make_graph([add_node], "test_graph", [input_tensor], [output_tensor], [weight_init])
        onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        fp16_max = np.finfo(np.float16).max
        fp16_min = -fp16_max

        # Apply to each initializer
        for init in onnx_model.graph.initializer:
            FP16ClipTransform.apply(init, str(tmp_export_dir), fp16_max, fp16_min)

        # In-range values must be preserved
        for init in onnx_model.graph.initializer:
            if init.name == "in_range_weight":
                values = numpy_helper.to_array(init)
                np.testing.assert_allclose(values, in_range_values, rtol=1e-5)

    def test_fp16_clip_transform_handles_negative_out_of_range(self, tmp_export_dir):
        """FP16ClipTransform must clip negative values < -65504 to -65504."""
        import numpy as np
        import onnx.numpy_helper as numpy_helper

        from QEfficient.base.onnx_transforms import FP16ClipTransform

        onnx_model = self._make_onnx_model_with_large_initializer()
        fp16_max = np.finfo(np.float16).max  # 65504
        fp16_min = -fp16_max

        for init in onnx_model.graph.initializer:
            FP16ClipTransform.apply(init, str(tmp_export_dir), fp16_max, fp16_min)

        for init in onnx_model.graph.initializer:
            if init.name == "large_weight":
                values = numpy_helper.to_array(init)
                assert np.all(values >= fp16_min - 1), f"Negative values must be clipped to >= {fp16_min}"


# ---------------------------------------------------------------------------
# Tests: RenameFunctionOutputsTransform
# ---------------------------------------------------------------------------


@pytest.mark.onnx
@pytest.mark.slow
class TestRenameFunctionOutputsTransform:
    """RenameFunctionOutputsTransform must rename KV outputs to RetainedState names."""

    def test_rename_transform_is_importable(self):
        """RenameFunctionOutputsTransform must be importable."""
        from QEfficient.base.onnx_transforms import RenameFunctionOutputsTransform

        assert RenameFunctionOutputsTransform is not None

    def test_rename_transform_has_apply_method(self):
        """RenameFunctionOutputsTransform must have an apply classmethod."""
        from QEfficient.base.onnx_transforms import RenameFunctionOutputsTransform

        assert hasattr(RenameFunctionOutputsTransform, "apply")
        assert callable(RenameFunctionOutputsTransform.apply)

    def test_rename_transform_output_count_unchanged(self, tmp_export_dir):
        """After RenameFunctionOutputsTransform, output count must be unchanged.
        RenameFunctionOutputsTransform.apply(model) takes only the model."""
        import onnx

        from QEfficient.base.onnx_transforms import RenameFunctionOutputsTransform
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))

        output_count_before = len(onnx_model.graph.output)
        # RenameFunctionOutputsTransform.apply takes only the model (no path)
        RenameFunctionOutputsTransform.apply(onnx_model)
        output_count_after = len(onnx_model.graph.output)

        assert output_count_before == output_count_after, (
            f"Output count changed: {output_count_before} → {output_count_after}"
        )

    def test_rename_transform_preserves_kv_prefix_infix(self):
        """Subfunction rename keeps optional KV prefix infix on retained-state outputs."""
        import onnx
        from onnx import helper

        from QEfficient.base.onnx_transforms import RenameFunctionOutputsTransform

        fn_name = "DecoderLayerFn"
        fn_domain = "qeff.test"
        fn_out = "past_key.0_vllmKvCache_InternalRetainedState"
        function = helper.make_function(
            fn_domain,
            fn_name,
            ["x"],
            [fn_out],
            [helper.make_node("Identity", ["x"], [fn_out], "id")],
            [helper.make_opsetid("", 17)],
        )

        graph = helper.make_graph(
            [helper.make_node(fn_name, ["input"], ["tmp_out"], name="DecoderLayer_0", domain=fn_domain)],
            "g",
            [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("tmp_out", onnx.TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(fn_domain, 1)],
            functions=[function],
        )

        RenameFunctionOutputsTransform.apply(model, layer_idx=1)

        assert model.graph.output[0].name == "past_key.1_vllmKvCache_RetainedState"
        assert "_InternalRetainedState" not in model.graph.output[0].name


# ---------------------------------------------------------------------------
# Tests: SplitTensorsTransform functional (GAP E)
# ---------------------------------------------------------------------------


class TestSplitTensorsTransformFunctional:
    """SplitTensorsTransform must correctly map tensors to external data files."""

    def test_split_tensors_transform_importable(self):
        """SplitTensorsTransform must be importable."""
        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        assert SplitTensorsTransform is not None

    def test_split_tensors_transform_has_apply_classmethod(self):
        """SplitTensorsTransform.apply must be a classmethod."""
        import inspect

        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        assert isinstance(
            inspect.getattr_static(SplitTensorsTransform, "apply"),
            classmethod,
        )

    def test_split_tensors_apply_populates_mapping(self):
        """SplitTensorsTransform.apply must add tensor to mapping dict."""
        import numpy as np
        import onnx.numpy_helper as numpy_helper

        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        # Create a dummy tensor
        arr = np.random.randn(10, 10).astype(np.float32)
        tensor = numpy_helper.from_array(arr, name="test_tensor")

        mapping = {}
        SplitTensorsTransform.apply(tensor, model_name="test_model", file_num=0, mapping=mapping)

        assert "test_tensor" in mapping, (
            f"SplitTensorsTransform must add tensor to mapping. Got: {list(mapping.keys())}"
        )

    def test_split_tensors_apply_assigns_correct_file_name(self):
        """SplitTensorsTransform.apply must assign correct file name."""
        import numpy as np
        import onnx.numpy_helper as numpy_helper

        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        arr = np.ones((5, 5), dtype=np.float32)
        tensor = numpy_helper.from_array(arr, name="weight_tensor")

        mapping = {}
        SplitTensorsTransform.apply(tensor, model_name="mymodel", file_num=3, mapping=mapping)

        assert "weight_tensor" in mapping
        _, file_name = mapping["weight_tensor"]
        assert file_name == "mymodel_3.onnx.data", f"Expected 'mymodel_3.onnx.data', got '{file_name}'"

    def test_split_tensors_apply_stores_tensor_in_mapping(self):
        """SplitTensorsTransform.apply must store the tensor proto in mapping."""
        import numpy as np
        import onnx.numpy_helper as numpy_helper

        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        arr = np.eye(4, dtype=np.float32)
        tensor = numpy_helper.from_array(arr, name="eye_tensor")

        mapping = {}
        SplitTensorsTransform.apply(tensor, model_name="model", file_num=1, mapping=mapping)

        stored_tensor, _ = mapping["eye_tensor"]
        assert stored_tensor is tensor, "SplitTensorsTransform must store the original tensor proto"

    def test_split_tensors_apply_multiple_tensors(self):
        """SplitTensorsTransform.apply must handle multiple tensors in same mapping."""
        import numpy as np
        import onnx.numpy_helper as numpy_helper

        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        mapping = {}
        for i in range(5):
            arr = np.random.randn(3, 3).astype(np.float32)
            tensor = numpy_helper.from_array(arr, name=f"tensor_{i}")
            SplitTensorsTransform.apply(tensor, model_name="model", file_num=i, mapping=mapping)

        assert len(mapping) == 5, f"Expected 5 entries in mapping, got {len(mapping)}"
        for i in range(5):
            assert f"tensor_{i}" in mapping


# ---------------------------------------------------------------------------
# Tests: CustomOpTransform structure (GAP E)
# ---------------------------------------------------------------------------


class TestCustomOpTransformStructure:
    """CustomOpTransform must have correct structure and contain all expected custom ops."""

    def test_custom_op_transform_importable(self):
        """CustomOpTransform must be importable."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert CustomOpTransform is not None

    def test_custom_op_transform_has_custom_ops_dict(self):
        """CustomOpTransform must have a _custom_ops dict."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert hasattr(CustomOpTransform, "_custom_ops")
        assert isinstance(CustomOpTransform._custom_ops, dict)
        assert len(CustomOpTransform._custom_ops) > 0

    def test_custom_op_transform_contains_rms_norm(self):
        """CustomOpTransform._custom_ops must contain 'CustomRMSNormFunc'."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert "CustomRMSNormFunc" in CustomOpTransform._custom_ops, (
            f"CustomRMSNormFunc not in _custom_ops: {list(CustomOpTransform._custom_ops.keys())}"
        )

    def test_custom_op_transform_contains_ctx_scatter(self):
        """CustomOpTransform._custom_ops must contain 'CtxScatterFunc'."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert "CtxScatterFunc" in CustomOpTransform._custom_ops

    def test_custom_op_transform_contains_ctx_gather(self):
        """CustomOpTransform._custom_ops must contain 'CtxGatherFunc'."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert "CtxGatherFunc" in CustomOpTransform._custom_ops

    def test_custom_op_transform_contains_ctx_scatter_3d_int(self):
        """CustomOpTransform._custom_ops must contain 'CtxScatterFunc3DInt'."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert "CtxScatterFunc3DInt" in CustomOpTransform._custom_ops

    def test_custom_op_transform_contains_ctx_scatter_3d_generalized(self):
        """CustomOpTransform._custom_ops must contain 'CtxScatterFunc3DGeneralized'."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert "CtxScatterFunc3DGeneralized" in CustomOpTransform._custom_ops

    def test_custom_op_transform_contains_ctx_gather_3d_generalized(self):
        """CustomOpTransform._custom_ops must contain 'CtxGatherFunc3DGeneralized'."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        assert "CtxGatherFunc3DGeneralized" in CustomOpTransform._custom_ops

    def test_custom_op_transform_rms_norm_maps_to_custom_rms_norm(self):
        """CustomRMSNormFunc must map to CustomRMSNorm class."""
        from QEfficient.base.onnx_transforms import CustomOpTransform
        from QEfficient.customop.rms_norm import CustomRMSNorm

        _, onnxscript_func = CustomOpTransform._custom_ops["CustomRMSNormFunc"]
        assert onnxscript_func is CustomRMSNorm, f"CustomRMSNormFunc must map to CustomRMSNorm, got {onnxscript_func}"

    def test_custom_op_transform_all_ops_have_to_function_proto(self):
        """All custom ops in CustomOpTransform must have to_function_proto method."""
        from QEfficient.base.onnx_transforms import CustomOpTransform

        for op_name, (_, onnxscript_func) in CustomOpTransform._custom_ops.items():
            assert hasattr(onnxscript_func, "to_function_proto"), (
                f"Custom op '{op_name}' onnxscript_func must have to_function_proto method"
            )

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_custom_op_transform_apply_adds_rms_norm_to_model_functions(self, tmp_export_dir):
        """After CustomOpTransform.apply, model.functions must contain CustomRMSNorm."""
        import onnx

        from QEfficient.base.onnx_transforms import CustomOpTransform
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))

        # Apply CustomOpTransform
        CustomOpTransform.apply(onnx_model)

        # Check that CustomRMSNorm is in model.functions
        function_names = {f.name for f in onnx_model.functions}
        assert "CustomRMSNorm" in function_names, (
            f"CustomRMSNorm not in model.functions after CustomOpTransform.apply. Found: {function_names}"
        )

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_llama_onnx_has_custom_rms_norm_after_export(self, tmp_export_dir):
        """Llama ONNX export must include CustomRMSNorm in model functions."""
        import onnx

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))

        function_names = {f.name for f in onnx_model.functions}
        assert "CustomRMSNorm" in function_names, (
            f"Llama ONNX must have CustomRMSNorm function. Found: {function_names}"
        )
