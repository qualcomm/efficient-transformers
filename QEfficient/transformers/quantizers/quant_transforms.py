# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from compressed_tensors.compressors import PackedQuantizationCompressor
from compressed_tensors.linear.compressed_linear import CompressedLinear
from torch import nn
from transformers import AutoConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

from QEfficient.base.pytorch_transforms import ModuleMutatorTransform
from QEfficient.customop.matmulnbits import QuantLinearORT
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import (
    FP8BlockWiseDequantLinear,
    FP8BlockWiseDequantQwen3VLMoeTextExperts,
    FP8DeQuantLinear,
)
from QEfficient.transformers.quantizers.quantizer_mxfp4 import QEffMxfp4GptOssExperts
from QEfficient.transformers.quantizers.quantizer_utils import (
    blockwise_dequantize,
    convert_moe_packed_tensors,
    dequantize_gptq,
    unpack_weights_and_zeros,
)
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


class AwqToMatmulNbitsTransform(ModuleMutatorTransform):
    _match_class = WQLinear_GEMM

    @classmethod
    def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
        # Unpack AWQ packed int4 weights and zeros to logical integer tensors,
        # then hand them directly to pack_on_device — bypassing quant_weight() so
        # we never go through a dequant->requant cycle (which introduced large
        # numerical errors because re-quantising the dequantised floats loses
        # precision and changes the packed values).
        bits = original_module.bits

        # int_weight: [in, out]  int_zeros: [in/group, out]  (AWQ column order restored)
        int_weight, int_zeros = unpack_weights_and_zeros(original_module.qweight, original_module.qzeros, bits, "awq")

        # Validate unpacked values are within expected range
        max_val = (2**bits) - 1
        if torch.any(int_weight > max_val) or torch.any(int_zeros > max_val):
            logger.warning(f"AWQ unpacked values exceed {bits}-bit range, applying mask to correct")

        int_weight = torch.bitwise_and(int_weight, max_val)
        int_zeros = torch.bitwise_and(int_zeros, max_val)

        new_module = QuantLinearORT(
            original_module.bits,
            original_module.group_size,
            original_module.in_features,
            original_module.out_features,
            original_module.bias is not None,
        )
        new_module.bias = original_module.bias if original_module.bias is not None else None
        # Set scales before calling pack_on_device (it reads self.scales internally).
        # AWQ scales are [in/group, out]; pack_on_device expects self.scales = [in/group, out]
        # and transposes it internally, so assign directly without transposing.
        new_module.scales = original_module.scales.float()
        new_module.pack_on_device(int_weight, int_zeros)
        return new_module


class PackQuantizedInt4ToMatMulNBitsTransform(ModuleMutatorTransform):
    """
    This transform is used to pack the quantized int4 weights into a format that can be used by the MatMulNBits kernel.
    It is used for the ONNX export of the quantized model.
    """

    _match_class = CompressedLinear

    @classmethod
    def _is_pack_quantized_int4_linear(cls, module):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        quantization_args = getattr(quantization_scheme, "weights", None)
        return (
            isinstance(module, nn.Linear)
            and hasattr(module, "weight_packed")
            and hasattr(module, "weight_scale")
            and quantization_args is not None
            and getattr(quantization_args, "num_bits", None) == 4
            and getattr(quantization_args, "type", None) == "int"
            and getattr(quantization_args, "strategy", None) == "group"
        )

    @classmethod
    def _matches(cls, module):
        return isinstance(module, cls._match_class) or cls._is_pack_quantized_int4_linear(module)

    @classmethod
    def apply(cls, model: nn.Module):
        transformed = False
        for name, module in model.named_children():
            if cls._matches(module):
                setattr(model, name, cls.mutate(module, model))
                transformed = True
            else:
                _, child_transformed = cls.apply(module)
                transformed = transformed or child_transformed

        if cls._matches(model):
            model = cls.mutate(model, None)
            transformed = True

        return model, transformed

    @classmethod
    def mutate(cls, original_module, parent_module):
        # add compressor.decompress to get the decompressed weight
        # and then package into matmulnbit
        if isinstance(original_module, CompressedLinear):
            assert isinstance(original_module.compressor, PackedQuantizationCompressor), (
                f"Only {PackedQuantizationCompressor} supported for now"
            )
            fp_weight = original_module.compressor.decompress_module(original_module)
        else:
            from compressed_tensors.compressors.base import decompress_module as ct_decompress_module

            ct_decompress_module(original_module)
            fp_weight = original_module.weight
        scales = original_module.weight_scale
        # assuming symmetric quantization
        quantization_args = original_module.quantization_scheme.weights
        zeros = (torch.zeros_like(scales) + pow(2, (quantization_args.num_bits - 1))).to(torch.uint8)
        g_idx = torch.arange(original_module.in_features // quantization_args.group_size).repeat_interleave(
            quantization_args.group_size
        )
        original_module.weight = torch.nn.Parameter(fp_weight)
        assert quantization_args.type == "int", "uint is not tested yet"
        new_module = QuantLinearORT(
            quantization_args.num_bits,
            quantization_args.group_size,
            original_module.in_features,
            original_module.out_features,
            original_module.bias is not None,
        )
        new_module.bias = original_module.bias if original_module.bias is not None else None
        new_module.pack(original_module, scales, zeros, g_idx)
        return new_module


class GPTQToMatmulNbitsTransform(ModuleMutatorTransform):
    """
    A transformation class that mutates a ``QuantLinearGPTQ`` module to a ``QuantLinearORT``
    module by unpacking and dequantizing the quantized weights.
    """

    _match_class = QuantLinearGPTQ

    @staticmethod
    def unpack_and_dequantize_gptq(qweight, qzeros, scales, bits, g_idx):
        # Unpack the qweight and qzeros tensors
        int_weight, scales, int_zeros = dequantize_gptq(qweight.T, qzeros, scales, bits, g_idx)
        return int_weight, scales, int_zeros.to(torch.int32)

    @classmethod
    def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
        """
        ``Mutates`` the original ``QuantLinearGPTQ`` module to a ``QuantLinearORT`` module.

        Args:
            original_module (nn.Module): The original ``QuantLinearGPTQ`` module.
            parent_module (nn.Module): The parent module containing the original module.

        Returns:
            :nn.Module: The new ``QuantLinearORT`` module with unpacked and de-quantized weights.
        """

        fp16_weight, scales, zeros = cls.unpack_and_dequantize_gptq(
            original_module.qweight,
            original_module.qzeros,
            original_module.scales,
            original_module.bits,
            original_module.g_idx,
        )
        original_module.weight = fp16_weight.T
        new_module = QuantLinearORT(
            original_module.bits,
            original_module.group_size,
            original_module.in_features,
            original_module.out_features,
            original_module.bias is not None,
        )
        new_module.bias = original_module.bias if original_module.bias is not None else None
        new_module.pack(original_module, scales.T, zeros.T, original_module.g_idx)
        return new_module


class FP8DeQuantLinearToLinearTransform(ModuleMutatorTransform):
    _match_class = FP8DeQuantLinear

    @classmethod
    def mutate(cls, original_module, parent_module):
        #  -- de-quantizing the weights --
        dequant_weights = original_module.weight.to(torch.float32) * original_module.weight_scale
        dequant_linear_layer = nn.Linear(
            original_module.in_features, original_module.out_features, bias=original_module.bias is not None
        )
        dequant_linear_layer.weight = torch.nn.Parameter(dequant_weights)
        if original_module.bias is not None:
            dequant_linear_layer.bias = torch.nn.Parameter(original_module.bias.float())
        return dequant_linear_layer


class Mxfp4GptOssExpertDequantizeTransform(ModuleMutatorTransform):
    """
    Used to dequantize the weights of an Mxfp4GptOssExpert module and replace with transformers GptOssExperts with dequantized weights
    """

    _match_class = QEffMxfp4GptOssExperts

    @classmethod
    def mutate(cls, original_module, parent_module):
        dequant_module = GptOssExperts(original_module.config)
        dequant_module.gate_up_proj = torch.nn.Parameter(
            convert_moe_packed_tensors(
                original_module.gate_up_proj_blocks, original_module.gate_up_proj_scales, dtype=torch.float32
            )
        )
        dequant_module.down_proj = torch.nn.Parameter(
            convert_moe_packed_tensors(
                original_module.down_proj_blocks, original_module.down_proj_scales, dtype=torch.float32
            )
        )
        dequant_module.gate_up_proj_bias = original_module.gate_up_proj_bias
        dequant_module.down_proj_bias = original_module.down_proj_bias
        return dequant_module


class FP8BlockWiseDequantLinearToLinearTransform(ModuleMutatorTransform):
    """
    Used to dequantize the weights of an FP8BlockWiseDequantLinear module and replace with a regular Linear layer
    """

    _match_class = FP8BlockWiseDequantLinear

    @classmethod
    def mutate(cls, original_module, parent_module):
        #  -- de-quantizing the weights --
        dequant_weights = blockwise_dequantize(
            original_module.weight, original_module.weight_scale_inv, original_module.weight_block_size
        )
        dequant_linear_layer = nn.Linear(
            original_module.in_features, original_module.out_features, bias=original_module.bias is not None
        )
        dequant_linear_layer.weight = torch.nn.Parameter(dequant_weights)
        if original_module.bias is not None:
            dequant_linear_layer.bias = torch.nn.Parameter(original_module.bias.float())
        return dequant_linear_layer


class FP8BlockWiseDequantQwen3VLMoeTextExpertsToQwen3VLMoeTextExpertsTransform(ModuleMutatorTransform):
    _match_class = FP8BlockWiseDequantQwen3VLMoeTextExperts
    _model_type = "qwen3_vl_moe"

    @classmethod
    def mutate(cls, original_module, parent_module):
        config = AutoConfig.for_model(cls._model_type).text_config
        config.num_experts = original_module.num_experts
        config.intermediate_size = original_module.intermediate_size
        config.hidden_size = original_module.hidden_size
        assert original_module.act_fn.__class__.__name__ == "SiLUActivation", (
            "Only SiLU activation is supported for now."
        )
        assert config.hidden_act == "silu", "expected silu act fn, something changed in transformers code"
        dequant_module = Qwen3VLMoeTextExperts(config)
        dequant_module.gate_up_proj = torch.nn.Parameter(
            blockwise_dequantize(
                original_module.gate_up_proj,
                original_module.gate_up_proj_scale_inv,
                original_module.weights_block_size,
            )
        )
        dequant_module.down_proj = torch.nn.Parameter(
            blockwise_dequantize(
                original_module.down_proj,
                original_module.down_proj_scale_inv,
                original_module.weights_block_size,
            )
        )
        return dequant_module
