# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.utils import constants


def _weights_dtype(model: nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


class QEffChatterboxT3(QEFFBaseModel):
    """
    Chatterbox T3 (Llama backbone, speech-token head) with ONNX export and QAIC compilation.

    The graph takes ``inputs_embeds`` built on the host and keeps the KV cache on the device (retained state).
    Classifier-free guidance runs the conditional and unconditional rows as one batch of 2.
    """

    _pytorch_transforms = [CustomOpsTransform, KVCacheTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.__dict__

    def __init__(self, model: nn.Module, **kwargs) -> None:
        # The upstream Llama config says bfloat16, but the multilingual checkpoints load as float32: follow the weights.
        model.config.torch_dtype = _weights_dtype(model)
        super().__init__(model, **kwargs)
        self.hash_params["module_name"] = "t3"
        self.num_layers = model.config.num_hidden_layers

    def get_onnx_params(
        self,
        batch_size: int = constants.CHATTERBOX_T3_CFG_BATCH_SIZE,
        seq_len: int = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        ctx_len: int = 2 * constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
    ) -> Tuple[Dict, Dict, List[str]]:
        config = self.model.config
        dtype = config.torch_dtype
        kv_shape = (batch_size, config.num_key_value_heads, ctx_len, config.head_dim)
        example_inputs = {
            "inputs_embeds": torch.zeros((batch_size, seq_len, config.hidden_size), dtype=dtype),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(batch_size, 1),
            "past_key_values": [
                [torch.zeros(kv_shape, dtype=dtype), torch.zeros(kv_shape, dtype=dtype)] for _ in range(self.num_layers)
            ],
        }
        dynamic_axes = {
            "inputs_embeds": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        output_names = ["logits"]
        for i in range(self.num_layers):
            for kv in ("key", "value"):
                dynamic_axes[f"past_{kv}.{i}"] = {0: "batch_size", 2: "ctx_len"}
                output_names.append(f"past_{kv}.{i}_RetainedState")
        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(
        self,
        specializations: List[Dict],
        mxint8_kv_cache: bool = False,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        # The KV cache stays on the device in fp16 (or MXINT8). With ONNX subfunctions the graph names the retained
        # outputs *_InternalRetainedState; the base compile maps these public names onto them.
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        custom_io = {}
        for i in range(self.num_layers):
            for kv in ("key", "value"):
                custom_io[f"past_{kv}.{i}"] = kv_cache_dtype
                custom_io[f"past_{kv}.{i}_RetainedState"] = kv_cache_dtype
        compiler_options.setdefault("convert_to_fp16", True)  # the checkpoint is fp32; AI100 computes in fp16
        return self._compile(
            onnx_path=compiler_options.pop("onnx_path", None) or self.onnx_path,
            retained_state=True,
            specializations=specializations,
            custom_io=custom_io,
            mxint8_kv_cache=mxint8_kv_cache,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )
