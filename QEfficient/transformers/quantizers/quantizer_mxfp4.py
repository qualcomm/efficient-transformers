# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import re
from typing import Optional

import torch
import torch.nn as nn
from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
from transformers.utils.quantization_config import Mxfp4Config

from QEfficient.transformers.quantizers.quantizer_utils import convert_moe_packed_tensors, get_keys_to_not_convert
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


class QEffMxfp4GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_up_proj_blocks = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, 16, dtype=torch.uint8),
            requires_grad=False,
        )
        self.gate_up_proj_scales = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, dtype=torch.uint8),
            requires_grad=False,
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32), requires_grad=False
        )

        self.down_proj_blocks = nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32, 16), dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_scales = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, self.intermediate_size // 32, dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, dtype=torch.float32), requires_grad=False
        )
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_up_proj_precision_config = None
        self.down_proj_precision_config = None

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        gate_up_proj = convert_moe_packed_tensors(
            self.gate_up_proj_blocks, self.gate_up_proj_scales, dtype=torch.float32
        )
        down_proj = convert_moe_packed_tensors(self.down_proj_blocks, self.down_proj_scales, dtype=torch.float32)
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]
        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, gate_up_proj) + self.gate_up_proj_bias[..., None, :]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1) * glu), down_proj)
        next_states = next_states + self.down_proj_bias[..., None, :]
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)
        return next_states


def should_convert_module(current_key_name, patterns):
    current_key_name_str = ".".join(current_key_name)
    if not any(
        re.match(f"{key}\\.", current_key_name_str) or re.match(f"{key}", current_key_name_str) for key in patterns
    ):
        return True
    return False


class QEffMxfp4Config(Mxfp4Config):
    """
    Currently there is not need to change the implementation of Mxfp4Config
    This is placeholder for future when we would want to change this
    """

    pass


class QEffMxfp4HfQuantizer(Mxfp4HfQuantizer):
    def validate_environment(self, *args, **kwargs):
        return True

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype not in [None, torch.float32]:
            logger.warning(f"Requested dtype {torch_dtype} is not supported, overriding to float32")
        return torch.float32

    def update_dtype(self, dtype):
        return self.update_torch_dtype(dtype)

    def _process_model_before_weight_loading(
        self,
        model: torch.nn.Module,
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        self.modules_to_not_convert = get_keys_to_not_convert(model)
        self.modules_to_not_convert = (
            ["lm_head"] if self.modules_to_not_convert is None else self.modules_to_not_convert
        )
        self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)
        self.modules_to_not_convert = list(set(self.modules_to_not_convert))
        config = model.config

        # -- Defining local method as it uses lot of local variables --
        def _replace_with_mxfp4_linear(
            model,
            modules_to_not_convert=None,
            current_key_name=None,
            quantization_config=None,
            has_been_replaced=False,
        ):
            if current_key_name is None:
                current_key_name = []

            for name, module in model.named_children():
                current_key_name.append(name)
                if not should_convert_module(current_key_name, modules_to_not_convert):
                    current_key_name.pop(-1)
                    continue
                if module.__class__.__name__ == "GptOssExperts" and not quantization_config.dequantize:
                    model._modules[name] = QEffMxfp4GptOssExperts(config)
                    has_been_replaced = True
                if len(list(module.children())) > 0:
                    _, has_been_replaced = _replace_with_mxfp4_linear(
                        module,
                        modules_to_not_convert,
                        current_key_name,
                        quantization_config,
                        has_been_replaced=has_been_replaced,
                    )
                current_key_name.pop(-1)
            return model, has_been_replaced

        _replace_with_mxfp4_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        model.config.quantization_config = self.quantization_config
