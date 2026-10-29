# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts
from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
from transformers.utils import is_compressed_tensors_available
from transformers.utils.quantization_config import CompressedTensorsConfig, QuantizationConfigMixin, QuantizationMethod

# Importing the custom ops registers them with torch.ops.qefficient.*
import QEfficient.customop.dynamo_ops  # noqa: F401
from QEfficient.customop.fp8_dequantize import (
    FP8DequantizeBlockedFunc,
    FP8DequantizePerAxisFunc,
    FP8DequantizePerTensorFunc,
)
from QEfficient.customop.utils import select_interface
from QEfficient.transformers.quantizers.quantizer_utils import blockwise_dequantize, get_keys_to_not_convert
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


FP8_DTYPE = torch.float8_e4m3fn

# Scale dtypes supported for FP8 dequantization.
_SUPPORTED_SCALE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class QEffExtendedQuantizationMethod(str, Enum):
    FP8 = "fp8"


@dataclass
class FP8QuantizationScheme:
    dynamic: bool
    num_bits: int
    strategy: str
    symmetric: bool
    type: str

    def __post_init__(self):
        if self.num_bits != 8 or self.type != "float" or self.strategy not in ["tensor", "channel", "token"]:
            raise NotImplementedError(
                f"Only FP8 compressed-tensors supported, got num_bits={self.num_bits}, type={self.type}, strategy={self.strategy}"
            )


class FP8DeQuantLinear(torch.nn.Module):
    """
    Linear layer that stores weights in FP8 (float8_e4m3fn) and dequantizes
    on the fly during the forward pass.

    Supports per-tensor (scalar scale) and per-axis/channel (1-D scale) weight
    quantization, as well as optional static activation scales.
    The scale dtype is configurable: bfloat16, float16, or float32.

    During ONNX export (dynamo=True) the forward dispatches through
    ``torch.ops.qefficient.fp8_dequantize_per_tensor`` or
    ``torch.ops.qefficient.fp8_dequantize_per_axis``, which are translated to
    standard ONNX ``DequantizeLinear`` nodes via the custom_translation_table.
    Weights are preserved in FP8 (dtype=17) in the ONNX initializers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_dtype = scale_dtype

        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=FP8_DTYPE))

        if bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=scale_dtype))
        else:
            self.bias = None

    @classmethod
    def for_compressed_tensors_fp8_layer(
        cls,
        in_features: int,
        out_features: int,
        weights_quant_scheme: FP8QuantizationScheme,
        input_activations_quant_scheme: FP8QuantizationScheme,
        bias: bool = False,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        if scale_dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(f"scale_dtype must be one of {_SUPPORTED_SCALE_DTYPES}, got {scale_dtype}")
        fp8_dequant_layer = cls(in_features, out_features, bias, scale_dtype)
        fp8_dequant_layer.weights_quantization_scheme = weights_quant_scheme
        fp8_dequant_layer.input_activations_quantization_scheme = input_activations_quant_scheme

        if fp8_dequant_layer.weights_quantization_scheme.dynamic:
            raise NotImplementedError(
                f"Expected statically quantized weights but got weights quantization scheme "
                f"dynamic = {fp8_dequant_layer.weights_quantization_scheme.dynamic}"
            )

        strategy = fp8_dequant_layer.weights_quantization_scheme.strategy
        if strategy == "tensor":
            # Per-tensor: scalar (0-D) scale.
            fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((), dtype=scale_dtype))
        elif strategy == "channel":
            # Per-axis: one scale per output channel, shape (out_features,).
            fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((out_features,), dtype=scale_dtype))
        else:
            raise NotImplementedError(
                f"Unknown weights quantization strategy {strategy!r}; "
                "supported: 'tensor' (per-tensor) or 'channel' (per-axis)."
            )

        act_scheme = fp8_dequant_layer.input_activations_quantization_scheme
        if not act_scheme.dynamic:
            if act_scheme.strategy == "tensor":
                fp8_dequant_layer.register_buffer("input_scale", torch.zeros((), dtype=scale_dtype))
            elif act_scheme.strategy == "token":
                fp8_dequant_layer.register_buffer("input_scale", torch.zeros((1, in_features), dtype=scale_dtype))
            else:
                raise NotImplementedError(
                    f"Unknown input activations quantization strategy {act_scheme.strategy!r}; "
                    "supported: 'tensor' or 'token'."
                )

        return fp8_dequant_layer

    @classmethod
    def for_fp8_layer(
        cls,
        in_features: int,
        out_features: int,
        activation_quantization_strategy: Optional[str],
        bias: bool,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        if scale_dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(f"scale_dtype must be one of {_SUPPORTED_SCALE_DTYPES}, got {scale_dtype}")
        fp8_dequant_layer = cls(in_features, out_features, bias, scale_dtype)
        # Per-tensor quantization: scalar (0-D) scale.
        fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((), dtype=scale_dtype))
        if activation_quantization_strategy == "static":
            fp8_dequant_layer.register_buffer("input_scale", torch.zeros((), dtype=scale_dtype))
        return fp8_dequant_layer

    def forward(self, x):
        scale = self.weight_scale
        if scale.ndim == 2 and scale.shape[1] == 1:
            scale = scale.squeeze(-1)
        if scale.ndim != 0 and scale.numel() == 1:
            scale = scale.reshape(())
        if scale.ndim == 0 or scale.numel() == 1:
            # Per-tensor: scalar scale — dispatches to DequantizeLinear (no axis) in ONNX.
            dequantized_weights = select_interface(
                FP8DequantizePerTensorFunc.apply,
                torch.ops.qefficient.fp8_dequantize_per_tensor,
            )(self.weight, scale)
        else:
            # Per-axis: (out_features,) scale — dispatches to DequantizeLinear(axis=0) in ONNX.
            dequantized_weights = select_interface(
                FP8DequantizePerAxisFunc.apply,
                torch.ops.qefficient.fp8_dequantize_per_axis,
            )(self.weight, scale)
        with torch.no_grad():
            out = torch.matmul(x.to(scale.dtype), dequantized_weights.T)
            out = out + self.bias if self.bias is not None else out
        return out


class FP8BlockWiseDequantLinear(torch.nn.Module):
    """
    Linear layer with 2-D blocked FP8 weight quantization.

    The weight tensor is stored in FP8.  During the forward pass the compact
    scale (out_features//row_bs, in_features//col_bs) is expanded to the full
    weight shape (out_features, in_features) via repeat_interleave, then
    ``torch.ops.qefficient.fp8_dequantize_blocked`` is called.

    During ONNX export the custom op is translated to a standard
    ``DequantizeLinear`` node with the pre-expanded scale (element-wise form),
    keeping the weight initializer in FP8 (dtype=17).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_block_size: List[int],
        bias: bool = False,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_block_size = weight_block_size
        self.scale_dtype = scale_dtype

        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=FP8_DTYPE))

        if bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=scale_dtype))
        else:
            self.bias = None

    @classmethod
    def for_fp8_layer_with_blocksize(
        cls,
        in_features: int,
        out_features: int,
        weight_block_size: List[int],
        fmt: str,
        bias: bool,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        if fmt != "e4m3":
            raise NotImplementedError(f"Only e4m3 FP8 format is supported, got fmt={fmt!r}")
        if scale_dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(f"scale_dtype must be one of {_SUPPORTED_SCALE_DTYPES}, got {scale_dtype}")
        row_bs, col_bs = weight_block_size
        if out_features % row_bs != 0 or in_features % col_bs != 0:
            raise ValueError(
                f"Weight shape ({out_features}, {in_features}) is not divisible by "
                f"weight_block_size ({row_bs}, {col_bs})"
            )
        fp8_dequant_layer = cls(in_features, out_features, weight_block_size, bias, scale_dtype)
        fp8_dequant_layer.register_buffer(
            "weight_scale_inv",
            torch.empty((out_features // row_bs, in_features // col_bs), dtype=scale_dtype),
        )
        return fp8_dequant_layer

    def __repr__(self):
        return (
            f"FP8BlockWiseDequantLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )

    def forward(self, x):
        row_bs, col_bs = self.weight_block_size
        # Pass the compact scale directly. The custom op expands it eagerly for
        # correctness; the ONNX symbolic emits Tile(scale,[row_bs,1]) followed by
        # DequantizeLinear(axis=-1, block_size=col_bs) so the ONNX graph stays
        # clean — no 3D intermediates from repeat_interleave lowering.
        dequantized_weights = select_interface(
            FP8DequantizeBlockedFunc.apply,
            torch.ops.qefficient.fp8_dequantize_blocked,
        )(self.weight, self.weight_scale_inv, row_bs, col_bs)
        with torch.no_grad():
            out = torch.matmul(x.to(self.weight_scale_inv.dtype), dequantized_weights.T)
            out = out + self.bias if self.bias is not None else out
        return out


class FP8BlockWiseDequantQwen3VLMoeTextExperts(torch.nn.Module):
    """MoE expert block with 2-D blocked FP8 weight quantization for Qwen3-VL-MoE."""

    def __init__(
        self,
        num_experts,
        moe_intermediate_size,
        hidden_size,
        act_fn,
        weights_block_size,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = moe_intermediate_size
        self.hidden_size = hidden_size
        self.expert_dim = self.intermediate_size
        self.weights_block_size = weights_block_size
        self.scale_dtype = scale_dtype
        r, c = weights_block_size
        self.register_buffer(
            "gate_up_proj", torch.empty((self.num_experts, self.hidden_size, 2 * self.expert_dim), dtype=FP8_DTYPE)
        )
        self.register_buffer(
            "down_proj", torch.empty((self.num_experts, self.expert_dim, self.hidden_size), dtype=FP8_DTYPE)
        )
        self.register_buffer(
            "gate_up_proj_scale_inv",
            torch.empty((self.num_experts, self.hidden_size // r, (2 * self.expert_dim) // c), dtype=scale_dtype),
        )
        self.register_buffer(
            "down_proj_scale_inv",
            torch.empty((self.num_experts, self.expert_dim // r, self.hidden_size // c), dtype=scale_dtype),
        )
        self.act_fn = act_fn

    @classmethod
    def for_fp8_layer_with_blocksize(
        cls,
        old_module,
        weight_block_size: List[int],
        fmt: str,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        if fmt != "e4m3":
            raise NotImplementedError(f"Only e4m3 FP8 format is supported, got fmt={fmt!r}")
        if scale_dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(f"scale_dtype must be one of {_SUPPORTED_SCALE_DTYPES}, got {scale_dtype}")
        return cls(
            num_experts=old_module.num_experts,
            moe_intermediate_size=old_module.intermediate_size,
            hidden_size=old_module.hidden_size,
            act_fn=old_module.act_fn,
            weights_block_size=weight_block_size,
            scale_dtype=scale_dtype,
        )

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        hidden_states = hidden_states.repeat(self.num_experts, 1)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up_proj = blockwise_dequantize(self.gate_up_proj, self.gate_up_proj_scale_inv, self.weights_block_size)
        down_proj = blockwise_dequantize(self.down_proj, self.down_proj_scale_inv, self.weights_block_size)
        gate_up = torch.bmm(hidden_states, gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        next_states = torch.bmm((up * self.act_fn(gate)), down_proj)
        next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
        next_states = next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)
        return next_states


class QEffFP8Config(QuantizationConfigMixin):
    def __init__(
        self,
        quant_method: str,
        activation_scheme: str,
        ignored_layers: List[str] = None,
        kv_cache_scheme: str = None,
        run_compressed: bool = False,
        fmt: str = None,
        weight_block_size: List[int] = None,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        self.quant_method = quant_method
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers
        self.kv_cache_scheme = kv_cache_scheme
        self.run_compressed = run_compressed
        self.quantization_config = None
        self.sparsity_config = None
        if scale_dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(f"scale_dtype must be one of {_SUPPORTED_SCALE_DTYPES}, got {scale_dtype}")
        self.scale_dtype = scale_dtype
        if kv_cache_scheme:
            logger.warning(
                f"kv_cache_scheme={kv_cache_scheme} will be ignored please use `mxint8_kv_cache=True` during compile call if you want to keep kv cache in int8 at runtime on Cloud AI 100"
            )

        if quant_method != "fp8" or activation_scheme not in ["static", "dynamic", None]:
            raise NotImplementedError(
                f"Expected FP8 quantization with static/dynamic/None activation quantization, go quant_method={quant_method}, activation_scheme={activation_scheme}"
            )

        self.quant_method = QEffExtendedQuantizationMethod.FP8
        self.fmt = fmt
        self.weight_block_size = weight_block_size


def _replace_with_fp8_dequant_linear_and_experts(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    scale_dtype=None,
):
    current_key_name = [] if current_key_name is None else current_key_name
    if scale_dtype is None:
        scale_dtype = getattr(quantization_config, "scale_dtype", torch.bfloat16)

    for name, child_module in model.named_children():
        current_key_name.append(name)

        if isinstance(child_module, torch.nn.Linear) and name not in (modules_to_not_convert or []):
            current_key_name_str = ".".join(current_key_name)
            if not any(key in current_key_name_str for key in (modules_to_not_convert or [])):
                model._modules[name] = FP8BlockWiseDequantLinear.for_fp8_layer_with_blocksize(
                    child_module.in_features,
                    child_module.out_features,
                    quantization_config.weight_block_size,
                    quantization_config.fmt,
                    child_module.bias is not None,
                    scale_dtype,
                )
                has_been_replaced = True

        if isinstance(child_module, Qwen3VLMoeTextExperts) and name not in (modules_to_not_convert or []):
            current_key_name_str = ".".join(current_key_name)
            if not any(key in current_key_name_str for key in (modules_to_not_convert or [])):
                model._modules[name] = FP8BlockWiseDequantQwen3VLMoeTextExperts.for_fp8_layer_with_blocksize(
                    child_module,
                    quantization_config.weight_block_size,
                    quantization_config.fmt,
                    scale_dtype,
                )
                has_been_replaced = True

        if len(list(child_module.children())) > 0:
            _, has_been_replaced = _replace_with_fp8_dequant_linear_and_experts(
                child_module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                scale_dtype=scale_dtype,
            )

        current_key_name.pop(-1)
    return model, has_been_replaced


def _squeeze_fp8_per_channel_scales(model: "torch.nn.Module") -> None:
    """Normalize FP8 weight scale ranks after HF weight loading.

    HF's weight-loading path uses set_module_tensor_to_device which bypasses
    _load_from_state_dict, so checkpoint scale shapes arrive in the buffer
    unchanged. ONNX DequantizeLinear requires a 1-D scale for per-axis
    dequantization (axis=0) and a 0-D scale for tensorwise dequantization, so
    normalize both forms here.
    """
    for module in model.modules():
        if isinstance(module, FP8DeQuantLinear):
            scale = module.weight_scale
            normalized_scale = scale
            if normalized_scale.ndim == 2 and normalized_scale.shape[1] == 1:
                normalized_scale = normalized_scale.squeeze(-1)
            if normalized_scale.ndim != 0 and normalized_scale.numel() == 1:
                normalized_scale = normalized_scale.reshape(())

            if normalized_scale.shape != scale.shape:
                module.weight_scale = (
                    torch.nn.Parameter(normalized_scale, requires_grad=False)
                    if isinstance(scale, torch.nn.Parameter)
                    else normalized_scale.clone()
                )
                # Re-register as buffer so it stays a buffer, not a parameter
                module.register_buffer("weight_scale", module.weight_scale)


def _cast_fp8_scale_dtypes(model: "torch.nn.Module", scale_dtype: torch.dtype) -> None:
    """Restore FP8 scale buffers to the requested model dtype after checkpoint loading."""
    scale_buffer_names = {
        "weight_scale",
        "input_scale",
        "weight_scale_inv",
        "gate_up_proj_scale_inv",
        "down_proj_scale_inv",
    }
    for module in model.modules():
        found_scale = False
        for name in scale_buffer_names:
            scale = getattr(module, name, None)
            if scale is not None and scale.dtype != scale_dtype:
                module._buffers[name] = scale.to(dtype=scale_dtype)
            found_scale = found_scale or scale is not None
        if found_scale:
            module.scale_dtype = scale_dtype


class QEffFP8Quantizer(CompressedTensorsHfQuantizer):
    def __init__(self, quantization_config, **kwargs):
        if not isinstance(quantization_config, QEffFP8Config):
            raise TypeError(f"Only {QEffFP8Config} is supported for initialization got {type(quantization_config)}")

        self.quantization_config = quantization_config
        self.run_compressed = quantization_config.run_compressed
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.modules_to_not_convert = list(
            set(self.modules_to_not_convert if self.modules_to_not_convert else [])
            | set(self.quantization_config.ignored_layers if self.quantization_config.ignored_layers else [])
        )
        self.pre_quantized = kwargs.pop("pre_quantized", True)
        self.scale_dtype = getattr(quantization_config, "scale_dtype", torch.bfloat16)

        if not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                f"pass `pre_quantized=True` while knowing what you are doing."
            )

    def validate_environment(self, *args, **kwargs):
        return True

    def update_dtype(self, dtype):
        if dtype is None:
            return dtype
        if dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(
                f"FP8 models require a supported floating-point dtype, got {dtype}. "
                f"Supported dtypes are {_SUPPORTED_SCALE_DTYPES}."
            )
        self.scale_dtype = dtype
        return dtype

    def update_torch_dtype(self, torch_dtype):
        """Keep compatibility with Transformers versions using the old hook name."""
        return self.update_dtype(torch_dtype)

    def _process_model_before_weight_loading(self, model, **kwargs):
        if not self.modules_to_not_convert or "lm_head" not in self.modules_to_not_convert:
            self.modules_to_not_convert.extend(get_keys_to_not_convert(model))

        logger.warning(
            f"activations quantization strategy = {self.quantization_config.activation_scheme}, will be ignored and the layers will be run with de-quantized weights"
        )
        if self.quantization_config.weight_block_size is not None:
            model, has_been_replaced = _replace_with_fp8_dequant_linear_and_experts(
                model,
                self.modules_to_not_convert,
                quantization_config=self.quantization_config,
                scale_dtype=self.scale_dtype,
            )
            return

        scale_dtype = self.scale_dtype

        def replace_linear_with_fp8_dequant_layer(module):
            for name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear) and name not in self.modules_to_not_convert:
                    compressed_fp8_layer = FP8DeQuantLinear.for_fp8_layer(
                        child_module.in_features,
                        child_module.out_features,
                        self.quantization_config.activation_scheme,
                        child_module.bias is not None,
                        scale_dtype,
                    )
                    setattr(module, name, compressed_fp8_layer)
                else:
                    replace_linear_with_fp8_dequant_layer(child_module)

        replace_linear_with_fp8_dequant_layer(model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        _cast_fp8_scale_dtypes(model, self.scale_dtype)
        _squeeze_fp8_per_channel_scales(model)

    def update_missing_keys_after_loading(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        return missing_keys

    def update_unexpected_keys(self, model, unexpected_keys: List[str], prefix: str = None) -> List[str]:
        return unexpected_keys


class QEffCompressedTensorsConfig(CompressedTensorsConfig):
    def handle_pack_quantized_init(
        self,
        config_groups=None,
        format="dense",
        quantization_status="initialized",
        kv_cache_scheme=None,
        global_compression_ratio=None,
        ignore=None,
        sparsity_config=None,
        quant_method="compressed-tensors",
        run_compressed: bool = True,
        **kwargs,
    ):
        if is_compressed_tensors_available():
            from compressed_tensors.config import SparsityCompressionConfig
            from compressed_tensors.quantization import QuantizationConfig
        else:
            raise ImportError(
                "compressed_tensors is not installed and is required for compressed-tensors quantization. Please install it with `pip install compressed-tensors`."
            )
        self.quantization_config = None
        self.sparsity_config = None

        self.run_compressed = run_compressed
        assert self.run_compressed, "pack-quantized needs to have run_compressed set to True"

        # parse from dict to load nested QuantizationScheme objects
        if config_groups or kv_cache_scheme:
            self.quantization_config = QuantizationConfig.model_validate(
                {
                    "config_groups": config_groups,
                    "quant_method": quant_method,
                    "format": format,
                    "quantization_status": quantization_status,
                    "kv_cache_scheme": kv_cache_scheme,
                    "global_compression_ratio": global_compression_ratio,
                    "ignore": ignore,
                    **kwargs,
                }
            )

        if sparsity_config:
            self.sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config.get("format"), **sparsity_config
            )

        self.quant_method = QuantizationMethod.COMPRESSED_TENSORS

    def handle_fp8_init(
        self,
        config_groups=None,
        format="dense",
        quantization_status="initialized",
        kv_cache_scheme=None,
        global_compression_ratio=None,
        ignore=None,
        sparsity_config=None,
        quant_method="compressed-tensors",
        run_compressed: bool = False,
        scale_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        self.config_groups = config_groups
        self.quant_method = quant_method
        self.kv_cache_scheme = kv_cache_scheme
        self.format = format
        self.quantization_status = quantization_status
        self.global_compression_ratio = global_compression_ratio
        self.ignore = ignore

        self.quantization_config = None
        self.sparsity_config = None

        self.run_compressed = run_compressed
        if scale_dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(f"scale_dtype must be one of {_SUPPORTED_SCALE_DTYPES}, got {scale_dtype}")
        self.scale_dtype = scale_dtype

        if len(self.config_groups) != 1:
            raise NotImplementedError(
                "Currently only single quantization group is supported, please raise an issue with model details for support!"
            )

        if quantization_status not in {"compressed", "frozen"}:
            raise NotImplementedError(f"expected quantization_status=`frozen`, got {quantization_status}")

        if kv_cache_scheme:
            raise NotImplementedError(f"Expected kv_cache_scheme=None, got {kv_cache_scheme}")

        if format not in {"naive-quantized", "float-quantized"}:
            raise NotImplementedError(
                f"Expected quantization format in ['naive_quantized', 'float-quantized']  got {format}"
            )

        if sparsity_config:
            raise NotImplementedError(f"Expected sparsity_config to be None, got {sparsity_config}")

        if quant_method != "compressed-tensors":
            raise NotImplementedError("Only compressed-tensors quant_method is supported for now!")

        if "lm_head" not in self.ignore:
            raise AttributeError(f"Expected `lm_head` to be present in non-quantized layers got ignore={self.ignore}")

        group_0 = self.config_groups.get("group_0")
        activations_quantization_config = group_0.get("input_activations")
        weights_quantization_config = group_0.get("weights")
        output_activation_quantization_config = group_0.get("output_activations")
        self.targets = group_0.get("targets")

        if self.targets != ["Linear"]:
            raise NotImplementedError(f"Only linear targets are supported, got {self.targets}")

        if output_activation_quantization_config:
            raise NotImplementedError(
                f"output_activations quantization is not supported got {output_activation_quantization_config}"
            )

        if (
            activations_quantization_config.get("block_structure")
            or activations_quantization_config.get("group_size")
            or weights_quantization_config.get("block_structure")
            or weights_quantization_config.get("group_size")
        ):
            raise NotImplementedError(f"group_size and block_structure not supported got {group_0}")

        self.weights_quantization_scheme = FP8QuantizationScheme(
            weights_quantization_config.get("dynamic"),
            weights_quantization_config.get("num_bits"),
            weights_quantization_config.get("strategy"),
            weights_quantization_config.get("symmetric"),
            weights_quantization_config.get("type"),
        )
        self.input_activations_quantization_scheme = FP8QuantizationScheme(
            activations_quantization_config.get("dynamic"),
            activations_quantization_config.get("num_bits"),
            activations_quantization_config.get("strategy"),
            activations_quantization_config.get("symmetric"),
            activations_quantization_config.get("type"),
        )

        self.quant_method = QuantizationMethod.COMPRESSED_TENSORS

    def __init__(
        self,
        config_groups=None,
        format="dense",
        quantization_status="initialized",
        kv_cache_scheme=None,
        global_compression_ratio=None,
        ignore=None,
        sparsity_config=None,
        quant_method="compressed-tensors",
        run_compressed: bool = None,
        **kwargs,
    ):
        if format == "pack-quantized":
            self.handle_pack_quantized_init(
                config_groups=config_groups,
                format=format,
                quantization_status=quantization_status,
                kv_cache_scheme=kv_cache_scheme,
                global_compression_ratio=global_compression_ratio,
                ignore=ignore,
                sparsity_config=sparsity_config,
                quant_method=quant_method,
                run_compressed=True if run_compressed is None else run_compressed,
                **kwargs,
            )
        else:
            self.handle_fp8_init(
                config_groups=config_groups,
                format=format,
                quantization_status=quantization_status,
                kv_cache_scheme=kv_cache_scheme,
                global_compression_ratio=global_compression_ratio,
                ignore=ignore,
                sparsity_config=sparsity_config,
                quant_method=quant_method,
                run_compressed=False if run_compressed is None else run_compressed,
                **kwargs,
            )

    def to_dict(self):
        if getattr(self.quantization_config, "format", None) == "pack-quantized":
            return super().to_dict()

        return {
            "quantization_config": {
                "config_groups": self.config_groups,
                "weights_quantization_scheme": self.weights_quantization_scheme.__dict__,
                "activations_quantization_scheme": self.input_activations_quantization_scheme.__dict__,
                "quant_method": self.quant_method,
                "kv_cache_scheme": self.kv_cache_scheme,
                "format": self.format,
                "quantization_status": self.quantization_status,
                "global_compression_ratio": self.global_compression_ratio,
                "ignore": self.ignore,
                "targets": self.targets,
            },
            "sparsity_config": None,
        }


class QEffCompressedTensorsFP8Quantizer(CompressedTensorsHfQuantizer):
    requires_calibration = False

    @staticmethod
    def is_pack_quantized(quant_config):
        return (
            hasattr(quant_config, "quantization_config")
            and hasattr(quant_config.quantization_config, "format")
            and quant_config.quantization_config.format == "pack-quantized"
        )

    def __init__(self, quantization_config, **kwargs):
        if self.is_pack_quantized(quantization_config):
            super().__init__(quantization_config, **kwargs)
        else:
            if not isinstance(quantization_config, QEffCompressedTensorsConfig):
                raise TypeError(
                    f"Only {QEffCompressedTensorsConfig} is supported for initialization got {type(quantization_config)}"
                )
            self.run_compressed = quantization_config.run_compressed
            self.quantization_config = quantization_config

            # -- Handle extra kwargs below --
            self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
            self.modules_to_not_convert = list(
                set(self.modules_to_not_convert if self.modules_to_not_convert else [])
                | set(self.quantization_config.ignore if self.quantization_config.ignore else [])
            )
            self.pre_quantized = kwargs.pop("pre_quantized", True)
            self.scale_dtype = getattr(self.quantization_config, "scale_dtype", torch.bfloat16)

            if not self.pre_quantized and self.requires_calibration:
                raise ValueError(
                    f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                    f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                    f"pass `pre_quantized=True` while knowing what you are doing."
                )

    def validate_environment(self, *args, **kwargs):
        if self.is_pack_quantized(self.quantization_config):
            return super().validate_environment(*args, **kwargs)
        return True

    def update_dtype(self, dtype):
        if self.is_pack_quantized(self.quantization_config):
            return super().update_dtype(dtype)

        if dtype is None:
            return dtype
        if dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(
                f"FP8 models require a supported floating-point dtype, got {dtype}. "
                f"Supported dtypes are {_SUPPORTED_SCALE_DTYPES}."
            )
        self.scale_dtype = dtype
        self.quantization_config.scale_dtype = dtype
        return dtype

    def update_torch_dtype(self, torch_dtype):
        """Keep compatibility with Transformers versions using the old hook name."""
        return self.update_dtype(torch_dtype)

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.is_pack_quantized(self.quantization_config):
            super()._process_model_before_weight_loading(model, **kwargs)
            return

        if self.quantization_config.targets != ["Linear"]:
            raise NotImplementedError(
                f"Only Linear layer with FP8 quantization are supported got targets = {self.quantization_config.targets}"
            )

        logger.warning(
            f"activations quantization scheme = {self.quantization_config.input_activations_quantization_scheme.__dict__}, will be ignored and the layers will be run with de-quantized weights"
        )

        scale_dtype = self.scale_dtype

        def replace_linear_with_fp8_dequant_layer(module):
            for name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear) and name not in self.modules_to_not_convert:
                    compressed_fp8_layer = FP8DeQuantLinear.for_compressed_tensors_fp8_layer(
                        child_module.in_features,
                        child_module.out_features,
                        self.quantization_config.weights_quantization_scheme,
                        self.quantization_config.input_activations_quantization_scheme,
                        child_module.bias is not None,
                        scale_dtype,
                    )
                    setattr(module, name, compressed_fp8_layer)
                else:
                    replace_linear_with_fp8_dequant_layer(child_module)

        replace_linear_with_fp8_dequant_layer(model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        if self.is_pack_quantized(self.quantization_config):
            super()._process_model_after_weight_loading(model, **kwargs)
            return
        _cast_fp8_scale_dtypes(model, self.scale_dtype)
        _squeeze_fp8_per_channel_scales(model)

    def update_missing_keys_after_loading(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        if self.is_pack_quantized(self.quantization_config):
            return super().update_missing_keys_after_loading(model, missing_keys=missing_keys, prefix=prefix)
        return missing_keys

    def update_unexpected_keys(self, model, unexpected_keys: List[str], prefix: str = None) -> List[str]:
        return unexpected_keys
