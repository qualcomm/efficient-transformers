# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers.quantizers.quantizer_awq import AwqQuantizer
from transformers.utils.quantization_config import AwqConfig

try:
    # transformers>=5
    from transformers.utils.quantization_config import AwqBackend
except ImportError:  # transformers<5
    from transformers.utils.quantization_config import AwqBackendPackingMethod as AwqBackend

from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.quantizer_utils import (
    get_keys_to_not_convert,
    replace_linear_layer_with_target_layer,
    replace_quantization_scales,
)
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


class QEffAwqConfig(AwqConfig):
    def post_init(self):
        """
        Safety checker that arguments are correct
        """
        super().post_init()

        # Keep QEff limited to auto-awq style GEMM path while tolerating v5 enum renames.
        allowed_backends = {getattr(AwqBackend, "AUTOAWQ", None), getattr(AwqBackend, "AUTO", None)}
        if self.backend not in allowed_backends:
            raise ValueError(
                f"Only quantization backend AUTO/AUTOAWQ is supported - not recognized backend {self.backend}"
            )

        awq_format = getattr(self, "format", None)
        allowed_formats = {None, "gemm", getattr(type(awq_format), "GEMM", None)}
        if awq_format not in allowed_formats:
            raise ValueError(f"Only GEMM format is supported - not recognized format {awq_format}")

        do_fuse = getattr(self, "do_fuse", False)
        fuse_max_seq_len = getattr(self, "fuse_max_seq_len", None)
        if do_fuse or fuse_max_seq_len is not None:
            raise ValueError(
                f"fused modules are not supported, got do_fuse={do_fuse}, fuse_max_seq_len={fuse_max_seq_len}"
            )

        if self.bits != 4:
            raise ValueError(f"Only 4-bit AWQ quantization is supported, got bits={self.bits}")


class QEffAwqQuantizer(AwqQuantizer):
    target_cls = WQLinear_GEMM

    def __init__(self, quantization_config: QEffAwqConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        # No need to validate as we will always use pytorch CPU version.
        return True

    @property
    def is_trainable(self):
        return False

    def update_dtype(self, torch_dtype):
        if torch_dtype not in [None, torch.float32]:
            logger.warning(f"Requested dtype {torch_dtype} is not supported, overriding to float32")
        return torch.float32

    # transformers<5 compatibility
    def update_torch_dtype(self, torch_dtype):
        return self.update_dtype(torch_dtype)

    def _process_model_before_weight_loading(self, model, **kwargs):
        self.modules_to_not_convert = get_keys_to_not_convert(model)

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model, has_been_replaced = replace_linear_layer_with_target_layer(
            model,
            target_cls=self.target_cls,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )

        model = replace_quantization_scales(model, model.config.model_type)
        if not has_been_replaced:
            logger.warning(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )

    def _process_model_after_weight_loading(self, model, **kwargs):
        """
        Keep post-load processing independent from optional upstream extras (e.g. gptqmodel).
        """
        return model
