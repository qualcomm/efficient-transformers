# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.tts.models.pytorch_transforms import ConvTranspose1dTransform, HiFTTransform, S3GenEncoderTransform
from QEfficient.tts.pipelines.pipeline_t3 import _weights_dtype
from QEfficient.utils import constants


class QEffChatterboxGraph(QEFFBaseModel):
    """
    Base for the stateless S3Gen graphs: one export, one compile with a specialization per length bucket.

    The upstream modules carry no Hugging Face config, so each graph gets a small one describing it; the hash also
    includes the graph name and the checkpoint id (``pretrained_model_name_or_path``) to keep exports apart.
    """

    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]
    graph_name: str = ""
    # HiFT modules stay in use on the host after export (the excitation source), so they keep their weights
    offload_pt_weights: bool = False

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.to_dict()

    def __init__(self, model: nn.Module, **kwargs) -> None:
        model.config = PretrainedConfig(
            architectures=[type(model).__name__], graph_name=self.graph_name, torch_dtype=_weights_dtype(model)
        )
        super().__init__(model, **kwargs)
        self.hash_params["module_name"] = self.graph_name

    def get_onnx_params(self) -> Tuple[Dict, Dict, List[str]]:
        raise NotImplementedError

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
            offload_pt_weights=self.offload_pt_weights,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> str:
        onnx_path = compiler_options.pop("onnx_path", None) or self.onnx_path
        return self._compile(onnx_path=onnx_path, specializations=specializations, **compiler_options)


class QEffS3GenFlowEncoder(QEffChatterboxGraph):
    """S3Gen flow encoder: prompt + speech tokens (padded to a bucket) -> ``mu`` (80 x 2 frames per token)."""

    _pytorch_transforms = [S3GenEncoderTransform]
    graph_name = "flow_encoder"
    offload_pt_weights = True

    def get_onnx_params(
        self, num_tokens: int = constants.CHATTERBOX_ONNX_EXPORT_TOKENS
    ) -> Tuple[Dict, Dict, List[str]]:
        example_inputs = {
            "token": torch.zeros((1, num_tokens), dtype=torch.int64),
            "token_len": torch.tensor([num_tokens], dtype=torch.int64),
        }
        dynamic_axes = {"token": {1: "n_tok"}, "mu": {2: "n_frames"}, "frame_mask": {2: "n_frames"}}
        return example_inputs, dynamic_axes, ["mu", "frame_mask"]


class QEffS3GenFlowEstimator(QEffChatterboxGraph):
    """S3Gen conditional flow-matching estimator: one Euler step for the CFG pair (batch 2)."""

    # Exports as-is: causal convs + masked attention; FP16ClipTransform clamps its -1e10 mask bias for fp16.
    _pytorch_transforms = []
    graph_name = "flow_estimator"
    offload_pt_weights = True

    def get_onnx_params(
        self, num_frames: int = constants.CHATTERBOX_MEL_FRAMES_PER_TOKEN * constants.CHATTERBOX_ONNX_EXPORT_TOKENS
    ) -> Tuple[Dict, Dict, List[str]]:
        dtype = self.model.config.torch_dtype
        channels = constants.CHATTERBOX_MEL_CHANNELS
        frames = (2, channels, num_frames)
        example_inputs = {
            "x": torch.zeros(frames, dtype=dtype),
            "mask": torch.ones((2, 1, num_frames), dtype=dtype),
            "mu": torch.zeros(frames, dtype=dtype),
            "t": torch.zeros(2, dtype=dtype),
            "spks": torch.zeros((2, channels), dtype=dtype),
            "cond": torch.zeros(frames, dtype=dtype),
        }
        dynamic_axes = {name: {2: "n_frames"} for name in ("x", "mask", "mu", "cond", "dxdt")}
        return example_inputs, dynamic_axes, ["dxdt"]


class QEffHiFTF0Predictor(QEffChatterboxGraph):
    """HiFT F0 predictor: mel (padded to a bucket) -> F0 per mel frame."""

    _pytorch_transforms = [HiFTTransform]
    graph_name = "hift_f0"

    def get_onnx_params(
        self, num_mel_frames: int = constants.CHATTERBOX_ONNX_EXPORT_MEL_FRAMES
    ) -> Tuple[Dict, Dict, List[str]]:
        dtype = self.model.config.torch_dtype
        # input named after ConvRNNF0Predictor.forward(x)
        example_inputs = {"x": torch.zeros((1, constants.CHATTERBOX_MEL_CHANNELS, num_mel_frames), dtype=dtype)}
        return example_inputs, {"x": {2: "n_mel"}, "f0": {1: "n_mel"}}, ["f0"]


class QEffHiFTDecoder(QEffChatterboxGraph):
    """HiFT decoder: mel + excitation source -> 24 kHz waveform (480 samples per mel frame)."""

    _pytorch_transforms = [HiFTTransform, ConvTranspose1dTransform]
    graph_name = "hift_decoder"

    def get_onnx_params(
        self, num_mel_frames: int = constants.CHATTERBOX_ONNX_EXPORT_MEL_FRAMES
    ) -> Tuple[Dict, Dict, List[str]]:
        dtype = self.model.config.torch_dtype
        num_samples = num_mel_frames * constants.CHATTERBOX_HIFT_SAMPLES_PER_MEL_FRAME
        example_inputs = {
            "mel": torch.zeros((1, constants.CHATTERBOX_MEL_CHANNELS, num_mel_frames), dtype=dtype),
            "source": torch.zeros((1, 1, num_samples), dtype=dtype),
        }
        dynamic_axes = {"mel": {2: "n_mel"}, "source": {2: "n_samples"}, "wav": {1: "n_samples"}}
        return example_inputs, dynamic_axes, ["wav"]
