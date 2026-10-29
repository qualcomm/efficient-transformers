# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import math
from typing import List

import torch
from torch import nn
from transformers.integrations.awq import AWQ_SCALES_MAPPINGS


class ScaledActivation(nn.Module):
    """
    A wrapper class for activation modules that scales the output by a specified factor.

    Args:
        module (nn.Module): The activation module to wrap.
        scales (torch.Tensor): The scaling factors.

    Attributes:
        act (nn.Module): The activation module.
        scales (nn.Parameter): The scaling factors.
    """

    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


def get_keys_to_not_convert(model):
    """
    Identifies and returns the names of parameters that should not be converted to a different precision.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        :list: A list of parameter names that should remain in full precision.
    """
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = copy.deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # If there is not tied weights, we want to keep the lm_head（output_embedding) in full precision
    if not has_tied_params:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            return list_last_module

    # otherwise, no tied weights, no output embedding defined, simply keep the last module in full precision
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names


def find_tied_parameters(model: nn.Module, **kwargs):
    """
    Recursively finds and returns tied parameters within a given model.

    Args:
        model (nn.Module): The model to search within.
        **kwargs: Additional keyword arguments for internal use.

    Returns:
        :list: A list of lists, where each sublist contains the names of tied parameters.
    """
    # Initialize result and named_parameters before recursing.
    named_parameters = kwargs.get("named_parameters", None)
    prefix = kwargs.get("prefix", "")
    result = kwargs.get("result", {})

    if named_parameters is None:
        named_parameters = {n: p for n, p in model.named_parameters()}
    else:
        # A tied parameter will not be in the full `named_parameters` seen above but will be in the `named_parameters`
        # of the submodule it belongs to. So while recursing we track the names that are not in the initial
        # `named_parameters`.
        for name, parameter in model.named_parameters():
            full_name = name if prefix == "" else f"{prefix}.{name}"
            if full_name not in named_parameters:
                # When we find one, it has to be one of the existing parameters.
                for new_name, new_param in named_parameters.items():
                    if new_param is parameter:
                        if new_name not in result:
                            result[new_name] = []
                        result[new_name].append(full_name)

    # Once we have treated direct parameters, we move to the child modules.
    for name, child in model.named_children():
        child_name = name if prefix == "" else f"{prefix}.{name}"
        find_tied_parameters(child, named_parameters=named_parameters, prefix=child_name, result=result)

    return [sorted([weight] + list(set(tied))) for weight, tied in result.items()]


def replace_linear_layer_with_target_layer(
    model: torch.nn.Module,
    target_cls,
    quantization_config=None,
    modules_to_not_convert=None,
    current_key_name=None,
    has_been_replaced=False,
):
    """
    Replaces all nn.Linear layers in the model with a specified target class, except for specified modules.

    Args:
        model (torch.nn.Module): The model containing the layers to be replaced.
        target_cls (type): The target class to replace nn.Linear layers with.
        quantization_config (object, optional): Configuration object for quantization.
        modules_to_not_convert (list, optional): List of module names to exclude from replacement.
        current_key_name (list, optional): List of current key names for recursion.
        has_been_replaced (bool, optional): Flag indicating if any layer has been replaced.

    Returns:
        :tuple: The modified model and a flag indicating if any layer has been replaced.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    # target_cls = WQLinear_GEMM

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = target_cls(
                    bits=quantization_config.bits,
                    group_size=quantization_config.group_size,
                    in_features=in_features,
                    out_features=out_features,
                    bias=module.bias is not None,
                    # dev=module.weight.device,
                )
                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_linear_layer_with_target_layer(
                module,
                target_cls,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_quantization_scales(model, model_type):
    """
    Replaces the quantization scales in the model based on the specified model type.

    Args:
        model (torch.nn.Module): The model containing the layers to be modified.
        model_type (str): The type of the model to determine the scale mappings.

    Returns:
        :torch.nn.Module: The modified model with updated quantization scales.
    """
    if model_type not in AWQ_SCALES_MAPPINGS:
        return model
    for name, module in model.named_children():
        act_name = AWQ_SCALES_MAPPINGS[model_type]["act"]
        layer_before_act_name = AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"]
        if name == act_name and hasattr(model, layer_before_act_name):
            layer_before_act = getattr(model, AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"])
            size = layer_before_act.out_features
            scale_like = torch.ones(size)
            model._modules[name] = ScaledActivation(module, scale_like)
            replace_quantization_scales(module, model_type)
    return model


def reverse_awq_order(int_weights: torch.Tensor, int_zeros: torch.Tensor, bits: int):
    """
    Reverses the order of the AWQ (Adaptive Weight Quantization) tensors.

    Args:
        int_weights (torch.Tensor): The integer weight tensor.
        int_zeros (torch.Tensor): The integer zeros tensor.
        bits (int): The number of bits used for quantization.

    Returns:
        :tuple: The reversed integer weight and zeros tensors.
    """
    reverse_order_tensor = torch.arange(
        int_weights.shape[-1],
        dtype=torch.int32,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    int_zeros = int_zeros[:, reverse_order_tensor]
    int_weights = int_weights[:, reverse_order_tensor]

    return int_weights, int_zeros


def unpack_weights_and_zeros(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int, quant: str):
    """
    Unpacks the quantized weights and zeros tensors based on the specified bit width and quantization type.

    Args:
        qweight (torch.Tensor): The quantized weight tensor.
        qzeros (torch.Tensor): The quantized zeros tensor.
        bits (int): The number of bits used for quantization.
        quant (str): The quantization type ("awq" or other).

    Returns:
        :tuple: A tuple containing the unpacked integer weight and zeros tensors.
    """

    shifts = torch.arange(0, 32, bits)

    # unpacking weights column-wise
    int_weights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    int_weights = int_weights.reshape(int_weights.shape[0], -1)

    # unpacking zeros column-wise
    int_zeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    int_zeros = int_zeros.reshape(int_zeros.shape[0], -1)

    if quant == "awq":
        return reverse_awq_order(int_weights, int_zeros, bits)

    return int_weights, int_zeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    """
    Dequantizes the GEMM (General Matrix Multiply) quantized weights and zeros.

    Args:
        qweight (torch.Tensor): The quantized weight tensor.
        qzeros (torch.Tensor): The quantized zeros tensor.
        scales (torch.Tensor): The scales tensor.
        bits (int): The number of bits used for quantization.
        group_size (int): The group size for quantization.

    Returns:
        :torch.Tensor: The dequantized weight tensor.
    """
    # Unpack the qweight and qzeros tensors
    scales, int_weight, int_zeros = unpack_weights(qweight, qzeros, scales, bits, "awq")

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    int_zeros = int_zeros.repeat_interleave(group_size, dim=0)

    int_weight = (int_weight - int_zeros) * scales

    return int_weight


def dequantize_gptq(qweight, qzeros, scales, bits, g_idx):
    """
    Dequantizes the ```GPTQ (Generalized Post-Training Quantization)``` quantized weights and zeros.

    Args:
        qweight (torch.Tensor): The quantized weight tensor.
        qzeros (torch.Tensor): The quantized zeros tensor.
        scales (torch.Tensor): The scales tensor.
        bits (int): The number of bits used for quantization.
        g_idx (torch.Tensor): The group index tensor.

    Returns:
        :tuple: A tuple containing the dequantized weight tensor, scales tensor, and zeros tensor.
    """
    scales, int_weight, int_zeros = unpack_weights(qweight, qzeros, scales, bits, "gptq")
    scales = scales.view(-1, 1, scales.size(-1))
    scales = scales.view(scales.shape[0], -1)
    scale_zeros = int_zeros * scales
    scale_mat = scales[g_idx]
    scale_zeros_mat = scale_zeros[g_idx]
    int_weight = int_weight.T * scale_mat - scale_zeros_mat.float()

    return int_weight, scales, int_zeros


def unpack_weights(qweight, qzeros, scales, bits, quant):
    """
    Unpacks the quantized weights and zeros tensors and performs overflow checks.

    Args:
        qweight (torch.Tensor): The quantized weight tensor.
        qzeros (torch.Tensor): The quantized zeros tensor.
        scales (torch.Tensor): The scales tensor.
        bits (int): The number of bits used for quantization.
        quant (str): The quantization type ("awq" or "gptq").

    Returns:
        :tuple: A tuple containing the scales tensor, unpacked integer weight tensor, and unpacked integer zeros tensor.
    """
    int_weight, int_zeros = unpack_weights_and_zeros(qweight, qzeros, bits, quant)

    # overflow checks
    int_weight = torch.bitwise_and(int_weight, (2**bits) - 1)
    int_zeros = torch.bitwise_and(int_zeros, (2**bits) - 1)

    return scales, int_weight, int_zeros


def repack_zeros(qzeros, bits):
    """
    Unpacks the quantized zeros tensor.

    Args:
        qzeros (torch.Tensor): The quantized zeros tensor.
        bits (int): The number of bits used for quantization.

    Returns:
        :torch.Tensor: The unpacked integer zeros tensor.
    """

    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qzeros.device).unsqueeze(0)
    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int32  # smallest dtype available
    )
    izeros = torch.bitwise_and(izeros[0], (2**bits) - 1).view(-1, 1, izeros[0].size(1) * izeros[0].size(2))
    izeros = izeros.view(izeros.shape[0], -1)
    izeros += 1
    qzeros.mul_(0)
    if qzeros.shape[0] == izeros.shape[0]:
        qzeros = qzeros.T
        izeros = izeros.T
    compress_ratio = 32 // bits
    i = 0
    row = 0
    while row < qzeros.shape[0]:
        for j in range(i, i + compress_ratio):
            qzeros[row:] |= izeros[j::compress_ratio] << (bits * (j - i))
        break
    qzeros = qzeros.T
    return qzeros


FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """
    reference for this function is taken from: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt_oss#L98
    """

    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    out = out.to(dtype).permute(0, 2, 1).contiguous()
    return out


def blockwise_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: List[int] = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    rows, cols = quantized.shape[-2:]
    if block_size is None:
        block_size = (quantized.shape[-2], quantized.shape[-1])

    block_m, block_n = block_size

    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n}).")
    quantized = quantized.to(torch.float32)
    reshaped = quantized.reshape(-1, rows // block_m, block_m, cols // block_n, block_n)
    expanded_scales = scales.to(torch.float32).reshape(-1, rows // block_m, cols // block_n)
    expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(2)
    dequantized = reshaped * expanded_scales

    return dequantized.reshape(quantized.shape)
