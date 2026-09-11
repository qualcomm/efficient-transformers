# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for quantization transforms and quantizer auto-detection in QEfficient.

Tests verify:
  - AwqToMatmulNbitsTransform: importable, has _match_class, has mutate method
  - GPTQToMatmulNbitsTransform: importable, has _match_class, has mutate method
  - FP8DeQuantLinearToLinearTransform: importable, has _match_class, has mutate method
  - Mxfp4GptOssExpertDequantizeTransform: importable, has _match_class, has mutate method
  - QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING: contains all expected quantization types
  - QEFF_AUTO_QUANTIZER_MAPPING: contains all expected quantizer types
  - with_replaced_quantizers: replaces and restores transformers quantizers correctly
  - QEFFAutoModelForCausalLM._pytorch_transforms includes quantization transforms

All tests run on CPU only, no quantized model downloads required.
"""


# ---------------------------------------------------------------------------
# Tests: Quantization Transform Importability and Structure
# ---------------------------------------------------------------------------


class TestQuantizationTransformImportability:
    """All quantization transforms must be importable and have correct structure."""

    def test_awq_transform_importable(self):
        from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform

        assert AwqToMatmulNbitsTransform is not None

    def test_gptq_transform_importable(self):
        from QEfficient.transformers.quantizers.quant_transforms import GPTQToMatmulNbitsTransform

        assert GPTQToMatmulNbitsTransform is not None

    def test_fp8_transform_importable(self):
        from QEfficient.transformers.quantizers.quant_transforms import FP8DeQuantLinearToLinearTransform

        assert FP8DeQuantLinearToLinearTransform is not None

    def test_mxfp4_transform_importable(self):
        from QEfficient.transformers.quantizers.quant_transforms import Mxfp4GptOssExpertDequantizeTransform

        assert Mxfp4GptOssExpertDequantizeTransform is not None

    def test_awq_transform_has_match_class(self):
        from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform

        assert hasattr(AwqToMatmulNbitsTransform, "_match_class")

    def test_gptq_transform_has_match_class(self):
        from QEfficient.transformers.quantizers.quant_transforms import GPTQToMatmulNbitsTransform

        assert hasattr(GPTQToMatmulNbitsTransform, "_match_class")

    def test_fp8_transform_has_match_class(self):
        from QEfficient.transformers.quantizers.quant_transforms import FP8DeQuantLinearToLinearTransform

        assert hasattr(FP8DeQuantLinearToLinearTransform, "_match_class")

    def test_mxfp4_transform_has_match_class(self):
        from QEfficient.transformers.quantizers.quant_transforms import Mxfp4GptOssExpertDequantizeTransform

        assert hasattr(Mxfp4GptOssExpertDequantizeTransform, "_match_class")

    def test_awq_match_class_is_wqlinear_gemm(self):
        from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
        from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform

        assert AwqToMatmulNbitsTransform._match_class is WQLinear_GEMM

    def test_gptq_match_class_is_quantlinear_gptq(self):
        from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
        from QEfficient.transformers.quantizers.quant_transforms import GPTQToMatmulNbitsTransform

        assert GPTQToMatmulNbitsTransform._match_class is QuantLinearGPTQ

    def test_fp8_match_class_is_fp8_dequant_linear(self):
        from QEfficient.transformers.quantizers.quant_transforms import FP8DeQuantLinearToLinearTransform
        from QEfficient.transformers.quantizers.quantizer_compressed_tensors import FP8DeQuantLinear

        assert FP8DeQuantLinearToLinearTransform._match_class is FP8DeQuantLinear

    def test_all_transforms_have_mutate_classmethod(self):
        from QEfficient.transformers.quantizers.quant_transforms import (
            AwqToMatmulNbitsTransform,
            FP8DeQuantLinearToLinearTransform,
            GPTQToMatmulNbitsTransform,
            Mxfp4GptOssExpertDequantizeTransform,
            PackQuantizedInt4ToMatMulNBitsTransform,
        )

        for cls in [
            AwqToMatmulNbitsTransform,
            GPTQToMatmulNbitsTransform,
            FP8DeQuantLinearToLinearTransform,
            Mxfp4GptOssExpertDequantizeTransform,
            PackQuantizedInt4ToMatMulNBitsTransform,
        ]:
            assert hasattr(cls, "mutate"), f"{cls.__name__} missing mutate method"
            assert callable(cls.mutate), f"{cls.__name__}.mutate is not callable"

    def test_all_transforms_are_subclasses_of_module_mutator(self):
        from QEfficient.base.pytorch_transforms import ModuleMutatorTransform
        from QEfficient.transformers.quantizers.quant_transforms import (
            AwqToMatmulNbitsTransform,
            FP8DeQuantLinearToLinearTransform,
            GPTQToMatmulNbitsTransform,
            Mxfp4GptOssExpertDequantizeTransform,
            PackQuantizedInt4ToMatMulNBitsTransform,
        )

        for cls in [
            AwqToMatmulNbitsTransform,
            GPTQToMatmulNbitsTransform,
            FP8DeQuantLinearToLinearTransform,
            Mxfp4GptOssExpertDequantizeTransform,
            PackQuantizedInt4ToMatMulNBitsTransform,
        ]:
            assert issubclass(cls, ModuleMutatorTransform), (
                f"{cls.__name__} must be a subclass of ModuleMutatorTransform"
            )


# ---------------------------------------------------------------------------
# Tests: QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING
# ---------------------------------------------------------------------------


class TestQEffAutoQuantizationConfigMapping:
    """QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING must contain all expected quantization types."""

    def test_mapping_exists_and_is_dict(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING

        assert isinstance(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, dict)

    def test_contains_awq(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING

        assert "awq" in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING

    def test_contains_gptq(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING

        assert "gptq" in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING

    def test_contains_compressed_tensors(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING

        assert "compressed-tensors" in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING

    def test_awq_config_is_qeff_awq_config(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING
        from QEfficient.transformers.quantizers.quantizer_awq import QEffAwqConfig

        assert QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING["awq"] is QEffAwqConfig

    def test_gptq_config_is_qeff_gptq_config(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING
        from QEfficient.transformers.quantizers.quantizer_gptq import QEffGPTQConfig

        assert QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING["gptq"] is QEffGPTQConfig

    def test_all_values_are_classes(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING

        for key, val in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.items():
            assert isinstance(val, type), f"Expected class for key '{key}', got {type(val)}"


# ---------------------------------------------------------------------------
# Tests: QEFF_AUTO_QUANTIZER_MAPPING
# ---------------------------------------------------------------------------


class TestQEffAutoQuantizerMapping:
    """QEFF_AUTO_QUANTIZER_MAPPING must contain all expected quantizer types."""

    def test_mapping_exists_and_is_dict(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZER_MAPPING

        assert isinstance(QEFF_AUTO_QUANTIZER_MAPPING, dict)

    def test_contains_awq(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZER_MAPPING

        assert "awq" in QEFF_AUTO_QUANTIZER_MAPPING

    def test_contains_gptq(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZER_MAPPING

        assert "gptq" in QEFF_AUTO_QUANTIZER_MAPPING

    def test_awq_quantizer_is_qeff_awq_quantizer(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZER_MAPPING
        from QEfficient.transformers.quantizers.quantizer_awq import QEffAwqQuantizer

        assert QEFF_AUTO_QUANTIZER_MAPPING["awq"] is QEffAwqQuantizer

    def test_gptq_quantizer_is_qeff_gptq_quantizer(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZER_MAPPING
        from QEfficient.transformers.quantizers.quantizer_gptq import QEffGPTQQuantizer

        assert QEFF_AUTO_QUANTIZER_MAPPING["gptq"] is QEffGPTQQuantizer

    def test_all_values_are_classes(self):
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZER_MAPPING

        for key, val in QEFF_AUTO_QUANTIZER_MAPPING.items():
            assert isinstance(val, type), f"Expected class for key '{key}', got {type(val)}"


# ---------------------------------------------------------------------------
# Tests: with_replaced_quantizers decorator
# ---------------------------------------------------------------------------


class TestWithReplacedQuantizers:
    """with_replaced_quantizers must replace and restore transformers quantizers correctly."""

    def test_with_replaced_quantizers_is_callable(self):
        from QEfficient.transformers.quantizers.auto import with_replaced_quantizers

        assert callable(with_replaced_quantizers)

    def test_with_replaced_quantizers_wraps_function(self):
        """Inside the wrapper, AUTO_QUANTIZATION_CONFIG_MAPPING must have QEff configs."""
        from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING

        from QEfficient.transformers.quantizers.auto import (
            QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING,
            with_replaced_quantizers,
        )

        call_log = []

        @with_replaced_quantizers
        def dummy_func():
            for k, v in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.items():
                assert AUTO_QUANTIZATION_CONFIG_MAPPING.get(k) is v, (
                    f"Key '{k}' not replaced: expected {v}, got {AUTO_QUANTIZATION_CONFIG_MAPPING.get(k)}"
                )
            call_log.append("called")
            return "result"

        result = dummy_func()
        assert result == "result"
        assert call_log == ["called"]

    def test_with_replaced_quantizers_restores_after_call(self):
        """After the wrapped function returns, original quantizers must be restored."""
        from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING

        from QEfficient.transformers.quantizers.auto import with_replaced_quantizers

        # Capture original values before wrapping
        original_awq = AUTO_QUANTIZATION_CONFIG_MAPPING.get("awq")

        @with_replaced_quantizers
        def dummy_func():
            pass

        dummy_func()

        # After call, original must be restored
        assert AUTO_QUANTIZATION_CONFIG_MAPPING.get("awq") is original_awq, (
            "with_replaced_quantizers must restore original 'awq' config after call"
        )

    def test_with_replaced_quantizers_preserves_return_value(self):
        from QEfficient.transformers.quantizers.auto import with_replaced_quantizers

        @with_replaced_quantizers
        def func_with_return():
            return {"key": "value", "num": 42}

        result = func_with_return()
        assert result == {"key": "value", "num": 42}


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM quantization transform integration
# ---------------------------------------------------------------------------


class TestQEFFAutoModelQuantizationIntegration:
    """QEFFAutoModelForCausalLM must include quantization transforms in its pipeline."""

    def test_pytorch_transforms_include_awq_transform(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform

        assert AwqToMatmulNbitsTransform in QEFFAutoModelForCausalLM._pytorch_transforms, (
            "AwqToMatmulNbitsTransform not in QEFFAutoModelForCausalLM._pytorch_transforms"
        )

    def test_pytorch_transforms_include_gptq_transform(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        from QEfficient.transformers.quantizers.quant_transforms import GPTQToMatmulNbitsTransform

        assert GPTQToMatmulNbitsTransform in QEFFAutoModelForCausalLM._pytorch_transforms, (
            "GPTQToMatmulNbitsTransform not in QEFFAutoModelForCausalLM._pytorch_transforms"
        )

    def test_pytorch_transforms_include_fp8_transform(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        from QEfficient.transformers.quantizers.quant_transforms import FP8DeQuantLinearToLinearTransform

        assert FP8DeQuantLinearToLinearTransform in QEFFAutoModelForCausalLM._pytorch_transforms, (
            "FP8DeQuantLinearToLinearTransform not in QEFFAutoModelForCausalLM._pytorch_transforms"
        )

    def test_quantization_transforms_come_before_kv_cache_transform(self):
        """Quantization transforms must be applied before KVCacheTransform."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform
        from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform

        transforms = QEFFAutoModelForCausalLM._pytorch_transforms
        awq_idx = next((i for i, t in enumerate(transforms) if t is AwqToMatmulNbitsTransform), None)
        kv_idx = next((i for i, t in enumerate(transforms) if t is KVCacheTransform), None)
        assert awq_idx is not None, "AwqToMatmulNbitsTransform not found in _pytorch_transforms"
        assert kv_idx is not None, "KVCacheTransform not found in _pytorch_transforms"
        assert awq_idx < kv_idx, (
            f"AwqToMatmulNbitsTransform (idx={awq_idx}) must come before KVCacheTransform (idx={kv_idx})"
        )

    def test_image_text_wrappers_include_pack_quantized_int4_transform(self):
        from QEfficient.transformers.models.modeling_auto import (
            QEffCausalLMForTextImageToTextModel,
            QEffVisionEncoderForTextImageToTextModel,
            _QEFFAutoModelForImageTextToTextSingleQPC,
        )
        from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform
        from QEfficient.transformers.quantizers.quant_transforms import PackQuantizedInt4ToMatMulNBitsTransform

        for wrapper_cls in [
            QEffVisionEncoderForTextImageToTextModel,
            QEffCausalLMForTextImageToTextModel,
            _QEFFAutoModelForImageTextToTextSingleQPC,
        ]:
            transforms = wrapper_cls._pytorch_transforms
            pack_idx = transforms.index(PackQuantizedInt4ToMatMulNBitsTransform)
            custom_ops_idx = transforms.index(CustomOpsTransform)
            assert pack_idx < custom_ops_idx

    def test_image_text_wrappers_include_fp8_dequant_transforms(self):
        from QEfficient.transformers.models.modeling_auto import (
            QEffCausalLMForTextImageToTextModel,
            QEffVisionEncoderForTextImageToTextModel,
            _QEFFAutoModelForImageTextToTextSingleQPC,
        )
        from QEfficient.transformers.quantizers.quant_transforms import (
            FP8BlockWiseDequantLinearToLinearTransform,
            FP8BlockWiseDequantQwen3VLMoeTextExpertsToQwen3VLMoeTextExpertsTransform,
            FP8DeQuantLinearToLinearTransform,
        )

        for wrapper_cls in [
            QEffVisionEncoderForTextImageToTextModel,
            QEffCausalLMForTextImageToTextModel,
            _QEFFAutoModelForImageTextToTextSingleQPC,
        ]:
            transforms = wrapper_cls._pytorch_transforms
            assert FP8DeQuantLinearToLinearTransform in transforms
            assert FP8BlockWiseDequantLinearToLinearTransform in transforms
            assert FP8BlockWiseDequantQwen3VLMoeTextExpertsToQwen3VLMoeTextExpertsTransform in transforms

    def test_pack_quantized_int4_transform_matches_compressed_linear_metadata(self, monkeypatch):
        import importlib
        from types import SimpleNamespace

        import torch

        from QEfficient.customop.matmulnbits import QuantLinearORT
        from QEfficient.transformers.quantizers.quant_transforms import PackQuantizedInt4ToMatMulNBitsTransform

        linear = torch.nn.Linear(8, 8, bias=False)
        del linear._parameters["weight"]
        linear.register_parameter(
            "weight_packed", torch.nn.Parameter(torch.zeros((8, 4), dtype=torch.int32), requires_grad=False)
        )
        linear.register_parameter(
            "weight_scale", torch.nn.Parameter(torch.ones((8, 2), dtype=torch.float32), requires_grad=False)
        )
        linear.quantization_scheme = SimpleNamespace(
            weights=SimpleNamespace(num_bits=4, type="int", strategy="group", group_size=4)
        )

        def fake_decompress_module(module):
            module.weight = torch.nn.Parameter(
                torch.zeros(module.out_features, module.in_features), requires_grad=False
            )

        compressor_base = importlib.import_module("compressed_tensors.compressors.base")
        monkeypatch.setattr(compressor_base, "decompress_module", fake_decompress_module)

        model = torch.nn.Sequential(linear)
        transformed_model, transformed = PackQuantizedInt4ToMatMulNBitsTransform.apply(model)

        assert transformed
        assert isinstance(transformed_model[0], QuantLinearORT)
        assert transformed_model[0].bits == 4
        assert transformed_model[0].group_size == 4

    def test_non_quantized_model_not_affected_by_quant_transforms(self):
        """Applying quantization transforms to a non-quantized model must not change it."""
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.quantizers.quant_transforms import (
            AwqToMatmulNbitsTransform,
            GPTQToMatmulNbitsTransform,
            PackQuantizedInt4ToMatMulNBitsTransform,
        )

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg).eval()

        # Apply AWQ transform - should not change a non-quantized model
        model_awq, applied_awq = AwqToMatmulNbitsTransform.apply(model)
        assert not applied_awq, "AwqToMatmulNbitsTransform must not apply to non-quantized model"

        # Apply GPTQ transform - should not change a non-quantized model
        model_gptq, applied_gptq = GPTQToMatmulNbitsTransform.apply(model)
        assert not applied_gptq, "GPTQToMatmulNbitsTransform must not apply to non-quantized model"

        model_pack, applied_pack = PackQuantizedInt4ToMatMulNBitsTransform.apply(model)
        assert not applied_pack, "PackQuantizedInt4ToMatMulNBitsTransform must not apply to non-quantized model"

        # Model output must be unchanged
        input_ids = torch.randint(0, 500, (1, 8))
        with torch.no_grad():
            original_logits = model(input_ids=input_ids).logits
            awq_logits = model_awq(input_ids=input_ids).logits
            pack_logits = model_pack(input_ids=input_ids).logits
        assert torch.allclose(original_logits, awq_logits), "AWQ transform must not change non-quantized model output"
        assert torch.allclose(original_logits, pack_logits), (
            "Packed-int4 transform must not change non-quantized model output"
        )

    def test_fp8_weight_scales_are_normalized_for_onnx_rank_requirements(self, tmp_path):
        import onnx
        import torch

        from QEfficient.transformers.quantizers.quantizer_compressed_tensors import (
            FP8DeQuantLinear,
            _squeeze_fp8_per_channel_scales,
        )

        tensorwise = FP8DeQuantLinear(4, 4)
        tensorwise.register_buffer("weight_scale", torch.ones(1))
        channelwise = FP8DeQuantLinear(4, 4)
        channelwise.register_buffer("weight_scale", torch.ones(4, 1))
        model = torch.nn.Sequential(tensorwise, channelwise)

        _squeeze_fp8_per_channel_scales(model)

        assert tensorwise.weight_scale.ndim == 0
        assert channelwise.weight_scale.shape == (4,)

        tensorwise.weight.copy_(torch.ones((4, 4), dtype=torch.int8).to(torch.float8_e4m3fn))
        onnx_path = tmp_path / "fp8_tensorwise.onnx"
        torch.onnx.export(
            tensorwise.eval(),
            (torch.ones((1, 4), dtype=torch.float16),),
            str(onnx_path),
            dynamo=False,
            opset_version=20,
        )
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        dequantize_node = next(node for node in onnx_model.graph.node if node.op_type == "DequantizeLinear")
        scale = next(
            initializer for initializer in onnx_model.graph.initializer if initializer.name == dequantize_node.input[1]
        )
        assert list(scale.dims) == []
