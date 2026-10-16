# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts
from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
from transformers.utils import is_compressed_tensors_available
from transformers.utils.quantization_config import CompressedTensorsConfig, QuantizationConfigMixin, QuantizationMethod

from QEfficient.transformers.quantizers.quantizer_utils import blockwise_dequantize, get_keys_to_not_convert
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


FP8_DTYPE = torch.float8_e4m3fn


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
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.empty(
                (out_features, in_features), dtype=FP8_DTYPE
            ),  # This is fixed for now and only e4m3fn quantization is prominent
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                ),
            )
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
    ):
        fp8_dequant_layer = cls(in_features, out_features, bias)
        fp8_dequant_layer.weights_quantization_scheme = weights_quant_scheme
        fp8_dequant_layer.input_activations_quantization_scheme = input_activations_quant_scheme

        if fp8_dequant_layer.weights_quantization_scheme.dynamic:
            raise NotImplementedError(
                f"Expected statically quantized weights but got weights quantization scheme dynamic = {fp8_dequant_layer.weights_quantization_scheme.dynamic}"
            )

        if fp8_dequant_layer.weights_quantization_scheme.strategy == "tensor":
            fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((1), dtype=torch.float32))
        elif fp8_dequant_layer.weights_quantization_scheme.strategy == "channel":
            fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((out_features, 1), dtype=torch.float32))
        else:
            raise NotImplementedError(
                f"Unknown weights quantization strategy {fp8_dequant_layer.weights_quantization_scheme.strategy}, ['channel' or 'tensor'] strategy supported."
            )

        if not fp8_dequant_layer.input_activations_quantization_scheme.dynamic:
            if fp8_dequant_layer.input_activations_quantization_scheme.strategy == "tensor":
                fp8_dequant_layer.register_buffer("input_scale", torch.zeros((1), dtype=torch.float32))
            elif fp8_dequant_layer.input_activations_quant_scheme.strategy == "token":
                fp8_dequant_layer.register_buffer("input_scale", torch.zeros((1, in_features), dtype=torch.float32))
            else:
                raise NotImplementedError(
                    f"Unknown input activations quantization strategy {fp8_dequant_layer.input_activations_quantization_scheme.strategy}, ['token' or 'tensor'] strategy supported."
                )

        return fp8_dequant_layer

    @classmethod
    def for_fp8_layer(cls, in_features, out_features, activation_quantization_strategy, bias):
        fp8_dequant_layer = cls(in_features, out_features, bias)

        # -- Always per tensor quantization assumed --
        fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((), dtype=torch.float32))

        if activation_quantization_strategy == "static":
            fp8_dequant_layer.register_buffer("input_scale", torch.zeros((), dtype=torch.float32))

        return fp8_dequant_layer

    def forward(self, x):
        # Only inference supported
        with torch.no_grad():
            dequantized_weights = self.weight.to(torch.float32) * self.weight_scale
            out = torch.matmul(x.float(), dequantized_weights.T)
            out = out + self.bias if self.bias is not None else out

        return out


class FP8BlockWiseDequantLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_block_size: List[int],
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_block_size = weight_block_size

        self.register_buffer(
            "weight",
            torch.empty(
                (out_features, in_features), dtype=FP8_DTYPE
            ),  # This is fixed for now and only e4m3fn quantization is prominent
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def for_fp8_layer_with_blocksize(cls, in_features, out_features, weight_block_size, fmt, bias):
        fp8_dequant_layer = cls(in_features, out_features, weight_block_size, bias)
        assert fmt == "e4m3", "e5m2 is not supposed yet!!"
        assert (in_features % weight_block_size[0]) == 0 and (out_features % weight_block_size[1]) == 0, (
            "weight shape is not divisible by block sizes in either rows or columns or both dimensions, \
            got in_features: {in_features}, out_features: {out_features}, weight_block_size: {weight_block_size}!!"
        )
        fp8_dequant_layer.register_buffer(
            "weight_scale_inv",
            torch.empty(
                (out_features // weight_block_size[0], in_features // weight_block_size[1]), dtype=torch.float32
            ),
        )
        return fp8_dequant_layer

    def __repr__(self):
        return f"FP8BlockWiseDequantLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"

    def forward(self, x):
        with torch.no_grad():
            dequantized_weights = blockwise_dequantize(self.weight, self.weight_scale_inv, self.weight_block_size)
            out = torch.matmul(x.float(), dequantized_weights.T)
            out = out + self.bias if self.bias is not None else out

        return out


class FP8BlockWiseDequantQwen3VLMoeTextExperts(torch.nn.Module):
    def __init__(self, num_experts, moe_intermediate_size, hidden_size, act_fn, weights_block_size):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = moe_intermediate_size
        self.hidden_size = hidden_size
        self.expert_dim = self.intermediate_size
        self.weights_block_size = weights_block_size
        r, c = weights_block_size
        self.register_buffer(
            "gate_up_proj", torch.empty((self.num_experts, self.hidden_size, 2 * self.expert_dim), dtype=FP8_DTYPE)
        )
        self.register_buffer(
            "down_proj", torch.empty((self.num_experts, self.expert_dim, self.hidden_size), dtype=FP8_DTYPE)
        )
        self.register_buffer(
            "gate_up_proj_scale_inv",
            torch.empty((self.num_experts, self.hidden_size // r, (2 * self.expert_dim) // c), dtype=torch.float32),
        )
        self.register_buffer(
            "down_proj_scale_inv",
            torch.empty((self.num_experts, self.expert_dim // r, self.hidden_size // c), dtype=torch.float32),
        )
        self.act_fn = act_fn

    @classmethod
    def for_fp8_layer_with_blocksize(cls, old_module, weight_block_size, fmt):
        assert fmt == "e4m3", "e5m2 is not supposed yet!!"
        fp8_experts = cls(
            num_experts=old_module.num_experts,
            moe_intermediate_size=old_module.intermediate_size,
            hidden_size=old_module.hidden_size,
            act_fn=old_module.act_fn,
            weights_block_size=weight_block_size,
        )
        return fp8_experts

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        hidden_states = hidden_states.repeat(self.num_experts, 1)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up_proj = blockwise_dequantize(self.gate_up_proj, self.gate_up_proj_inv_scale, self.weights_block_size)
        down_proj = blockwise_dequantize(self.down_proj, self.down_proj_inv_scale, self.weights_block_size)
        gate_up = torch.bmm(hidden_states, gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
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
    ):
        self.quant_method = quant_method
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers
        self.kv_cache_scheme = kv_cache_scheme
        self.run_compressed = run_compressed
        self.quantization_config = None
        self.sparsity_config = None
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


def _replace_with_fp8_dequant_linear_and_experts_if_qwen(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, has_been_replaced=False
):
    current_key_name = [] if current_key_name is None else current_key_name

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
                )
                has_been_replaced = True

        if isinstance(child_module, Qwen3VLMoeTextExperts) and name not in (modules_to_not_convert or []):
            # Replace the MoE experts
            current_key_name_str = ".".join(current_key_name)
            if not any(key in current_key_name_str for key in (modules_to_not_convert or [])):
                model._modules[name] = FP8BlockWiseDequantQwen3VLMoeTextExperts.for_fp8_layer_with_blocksize(
                    child_module,
                    quantization_config.weight_block_size,
                    quantization_config.fmt,
                )
                has_been_replaced = True

        if len(list(child_module.children())) > 0:
            _, has_been_replaced = _replace_with_fp8_dequant_linear_and_experts_if_qwen(
                child_module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )

        current_key_name.pop(-1)
    return model, has_been_replaced


class QEffFP8Quantizer(CompressedTensorsHfQuantizer):
    def __init__(self, quantization_config, **kwargs):
        # TODO: check if more checks are required
        if not isinstance(quantization_config, QEffFP8Config):
            raise TypeError(f"Only {QEffFP8Config} is supported for initialization got {type(quantization_config)}")

        self.quantization_config = quantization_config
        self.run_compressed = quantization_config.run_compressed
        # -- Handle extra kwargs below --
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.modules_to_not_convert = list(
            set(self.modules_to_not_convert if self.modules_to_not_convert else [])
            | set(self.quantization_config.ignored_layers if self.quantization_config.ignored_layers else [])
        )
        self.pre_quantized = kwargs.pop("pre_quantized", True)

        if not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                f"pass `pre_quantized=True` while knowing what you are doing."
            )

    def validate_environment(self, *args, **kwargs):
        return True

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype not in [None, torch.float32]:
            logger.warning(f"Requested dtype {torch_dtype} is not supported, overriding to float32")
        return torch.float32

    def _process_model_before_weight_loading(self, model, **kwargs):
        if not self.modules_to_not_convert or "lm_head" not in self.modules_to_not_convert:
            self.modules_to_not_convert.extend(get_keys_to_not_convert(model))

        logger.warning(
            f"activations quantization strategy = {self.quantization_config.activation_scheme}, will be ignored and the layers will be run with de-quantized weights"
        )

        if self.quantization_config.weight_block_size is not None:
            model, has_been_replaced = _replace_with_fp8_dequant_linear_and_experts_if_qwen(
                model, self.modules_to_not_convert, quantization_config=self.quantization_config
            )
            return

        # -- Defining local method as it uses lot of local variables --
        def replace_linear_with_fp8_dequant_layer(module):
            for name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear) and name not in self.modules_to_not_convert:
                    compressed_fp8_layer = FP8DeQuantLinear.for_fp8_layer(
                        child_module.in_features,
                        child_module.out_features,
                        self.quantization_config.activation_scheme,
                        child_module.bias is not None,
                    )
                    setattr(module, name, compressed_fp8_layer)
                else:
                    replace_linear_with_fp8_dequant_layer(child_module)

        replace_linear_with_fp8_dequant_layer(model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        pass

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
        # Validate configuration
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

    def update_torch_dtype(self, torch_dtype):
        if self.is_pack_quantized(self.quantization_config):
            return super().update_torch_dtype(torch_dtype)

        if torch_dtype not in [None, torch.float32]:
            logger.warning(f"Requested dtype {torch_dtype} is not supported, overriding to float32")
        return torch.float32

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

        # -- Defining local method as it uses lot of local variables --
        def replace_linear_with_fp8_dequant_layer(module):
            for name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear) and name not in self.modules_to_not_convert:
                    compressed_fp8_layer = FP8DeQuantLinear.for_compressed_tensors_fp8_layer(
                        child_module.in_features,
                        child_module.out_features,
                        self.quantization_config.weights_quantization_scheme,
                        self.quantization_config.input_activations_quantization_scheme,
                        child_module.bias is not None,
                    )
                    setattr(module, name, compressed_fp8_layer)
                else:
                    replace_linear_with_fp8_dequant_layer(child_module)

        replace_linear_with_fp8_dequant_layer(model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        if self.is_pack_quantized(self.quantization_config):
            super()._process_model_after_weight_loading(model, **kwargs)
            return
        pass

    def update_missing_keys_after_loading(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        if self.is_pack_quantized(self.quantization_config):
            return super().update_missing_keys_after_loading(model, missing_keys=missing_keys, prefix=prefix)
        return missing_keys

    def update_unexpected_keys(self, model, unexpected_keys: List[str], prefix: str = None) -> List[str]:
        return unexpected_keys
