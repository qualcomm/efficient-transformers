# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers.quantizers import HfQuantizer
from transformers.utils.quantization_config import GPTQConfig

from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_utils import (
    get_keys_to_not_convert,
    repack_zeros,
    replace_linear_layer_with_target_layer,
)
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


class QEffGPTQConfig(GPTQConfig):
    """
    Configuration class for QEffGPTQ, extending GPTQConfig.
    This class includes a post-initialization safety checker to ensure that the configuration arguments are correct.
    """

    def post_init(self):
        r"""
        Safety checker that arguments are correct.
        """
        if self.bits != 4:
            raise ValueError(f"Only 4-bit quantization is supported, got bits={self.bits}")
        if self.desc_act:
            raise ValueError("Only GPTQ model without descending activation size supported.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must be between 0 and 1.")


class QEffGPTQQuantizer(HfQuantizer):
    """
    Quantizer class for QEffGPTQ, extending HfQuantizer.
    This class handles the initialization, environment validation, dtype updating, and model processing for quantization.
    """

    target_cls = QuantLinearGPTQ

    def __init__(self, quantization_config: QEffGPTQConfig, **kwargs):
        """
        Initializes the QEffGPTQQuantizer with the given quantization configuration.

        Args:
            quantization_config (QEffGPTQConfig): The quantization configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        """
        Validates the environment for quantization.

        Args:
            device_map (dict): The device map for the model.
            **kwargs: Additional keyword arguments.

        Returns:
            :bool: True if the environment is valid, False otherwise.
        """
        return True

    def update_torch_dtype(self, torch_dtype):
        """
        Updates the torch data type for quantization.

        Args:
            torch_dtype (torch.dtype): The requested torch data type.

        Returns:
            :torch.dtype: The updated torch data type.
        """
        if torch_dtype not in [None, torch.float32]:
            logger.warning(f"Requested dtype {torch_dtype} is not supported, overriding to float32")
        return torch.float32

    def _process_model_before_weight_loading(self, model, **kwargs):
        """
        Processes the model before loading weights, ensuring it is suitable for quantization.

        Args:
            model (torch.nn.Module): The model to process.
            **kwargs: Additional keyword arguments.

        Returns:
            :torch.nn.Module: The processed model.
        """
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")
        if not self.pre_quantized:
            raise RuntimeError("Model is not quantized")

        self.modules_to_not_convert = get_keys_to_not_convert(model)

        model, has_been_replaced = replace_linear_layer_with_target_layer(
            model,
            target_cls=self.target_cls,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )
        if not has_been_replaced:
            logger.warning(
                "You are loading a GPTQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on GitHub if you think this is a bug."
            )
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        """
        Processes the model after loading weights, repacking quantization layers.

        Args:
            model (torch.nn.Module): The model to process.
            **kwargs: Additional keyword arguments.

        Returns:
            :torch.nn.Module: The processed model.
        """
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearGPTQ):
                izeros = repack_zeros(module.qzeros, module.bits)
                module.qzeros = izeros

    @property
    def is_trainable(self):
        """
        Indicates if the quantizer is trainable.

        Returns:
            :bool: False, indicating the quantizer is not trainable.
        """
        return False

    @property
    def is_serializable(self):
        """
        Indicates if the quantizer is serializable.

        Returns:
            :bool: True, indicating the quantizer is serializable.
        """
        return True
