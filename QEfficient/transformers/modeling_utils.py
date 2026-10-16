# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from collections import namedtuple
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import transformers.models.auto.modeling_auto as mapping
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenBlock,
    CodeGenForCausalLM,
    CodeGenModel,
)
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconForCausalLM,
    FalconModel,
)
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaDecoderLayer,
    GemmaForCausalLM,
    GemmaModel,
    GemmaRMSNorm,
)
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
    Gemma2Model,
    Gemma2RMSNorm,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4TextRouter,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    GPTBigCodeAttention,
    GPTBigCodeBlock,
    GPTBigCodeForCausalLM,
    GPTBigCodeModel,
)
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJForCausalLM, GPTJModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralModel,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from transformers.models.mllama.modeling_mllama import MllamaForCausalLM
from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptForCausalLM, MptModel
from transformers.models.phi.modeling_phi import PhiAttention, PhiForCausalLM, PhiModel
from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3ForCausalLM, Phi3Model, Phi3RMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2ForCausalLM, Qwen2Model, Qwen2RMSNorm
from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2DecoderLayer,
    Starcoder2ForCausalLM,
    Starcoder2Model,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperEncoder,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperPositionalEmbedding,
)

from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.customop import CustomRMSNormAIC
from QEfficient.proxy.pytorch_transform import QeffProxyModuleTransform
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.logging_utils import QEFFLogger

if TYPE_CHECKING:
    from QEfficient.base.modeling_qeff import QEFFBaseModel

# Placeholder for all non-transformer models
from .models.codegen.modeling_codegen import (
    QEffCodeGenAttention,
    QEffCodeGenBlock,
    QEffCodeGenForCausalLM,
    QEffCodeGenModel,
)
from .models.falcon.modeling_falcon import (
    QEffFalconAttention,
    QEffFalconForCausalLM,
    QEffFalconModel,
)
from .models.gemma.modeling_gemma import QEffGemmaAttention, QEffGemmaDecoderLayer, QEffGemmaForCausalLM, QEffGemmaModel
from .models.gemma2.modeling_gemma2 import (
    QEffGemma2Attention,
    QEffGemma2DecoderLayer,
    QEffGemma2ForCausalLM,
    QEffGemma2Model,
)
from .models.gemma4.modeling_gemma4 import (
    QEffGemma4CustomRMSNormAIC,
    QEffGemma4ForCausalLM,
    QEffGemma4TextAttention,
    QEffGemma4TextDecoderLayer,
    QEffGemma4TextModel,
    QEffGemma4TextRouter,
)
from .models.gpt2.modeling_gpt2 import QEffGPT2Attention, QEffGPT2Block, QEffGPT2LMHeadModel, QEffGPT2Model
from .models.gpt_bigcode.modeling_gpt_bigcode import (
    QEffGPTBigCodeAttention,
    QEffGPTBigCodeBlock,
    QEffGPTBigCodeForCausalLM,
    QEffGPTBigCodeModel,
)
from .models.gptj.modeling_gptj import QEffGPTJAttention, QEffGPTJForCausalLM, QEffGPTJModel
from .models.llama.modeling_llama import (
    QEffLlamaAttention,
    QEffLlamaDecoderLayer,
    QEffLlamaForCausalLM,
    QEffLlamaModel,
    QEffLlamaRotaryEmbedding,
)
from .models.mistral.modeling_mistral import (
    QEffMistralAttention,
    QEffMistralDecoderLayer,
    QEffMistralForCausalLM,
    QEffMistralModel,
)
from .models.mixtral_moe.modeling_mixtral import (
    QEffMixtralAttention,
    QeffMixtralDecoderLayer,
    QEffMixtralForCausalLM,
    QEffMixtralModel,
    QEffMixtralSparseMoeBlock,
)
from .models.mpt.modeling_mpt import QEffMptAttention, QEffMptBlock, QEffMptForCausalLM, QEFfMptModel
from .models.phi.modeling_phi import QEffPhiAttention, QEffPhiForCausalLM, QEffPhiModel
from .models.phi3.modeling_phi3 import QEffPhi3Attention, QEffPhi3ForCausalLM, QEffPhi3Model
from .models.qwen2.modeling_qwen2 import QEffQwen2Attention, QEffQwen2ForCausalLM, QEffQwen2Model
from .models.starcoder2.modeling_starcoder2 import (
    QEffStarcoder2Attention,
    QEFFStarcoder2DecoderLayer,
    QEffStarcoder2ForCausalLM,
    QEffStarcoder2Model,
)
from .models.whisper.modeling_whisper import (
    QEffWhisperAttention,
    QEffWhisperDecoder,
    QEffWhisperDecoderLayer,
    QEffWhisperEncoder,
    QEffWhisperForConditionalGeneration,
    QEffWhisperModel,
    QEffWhisperPositionalEmbedding,
)

logger = QEFFLogger.get_logger("INFRA")

# Define a named tuple for ModelArchitectures
# Required for the Automation tool
ModelArchitectures = namedtuple("ModelArchitectures", ["architectures"])

# Create an instance of the named tuple
qeff_supported_architectures = ModelArchitectures(
    [
        GPT2LMHeadModel.__name__,
        GPTJForCausalLM.__name__,
        MptForCausalLM.__name__,
        CodeGenForCausalLM.__name__,
        LlamaForCausalLM.__name__,
        GemmaForCausalLM.__name__,
        Gemma2ForCausalLM.__name__,
        Gemma4ForCausalLM.__name__,
        MistralForCausalLM.__name__,
        MixtralForCausalLM.__name__,
        Phi3ForCausalLM.__name__,
        PhiForCausalLM.__name__,
        FalconForCausalLM.__name__,
        Qwen2ForCausalLM.__name__,
        Starcoder2ForCausalLM.__name__,
        GPTBigCodeForCausalLM.__name__,
        MllamaForCausalLM.__name__,
        WhisperForConditionalGeneration.__name__,
    ]
)


# This is for supporting different seq_len for different layers for Sliding window attn, chunked attn etc.
DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH = {"gemma3", "gemma3_text", "gemma4_text", "llama4", "llama4_text"}

# This is for supporting different modelling classes specially written for prefill-only model
SPECIALIZED_DISAGG_SERVING_MODEL_ARCH = {"gpt_oss", "qwen3_moe", "glm4_moe", "kimi_k2", "kimi_k25", "gemma4"}

# This is for supporting dynamic prefill sequence length for prefill_only model
DYNAMIC_PREFILL_SEQ_LEN_SUPPORTED_MODEL_ARCH = {"gemma4"}

_PROXY_ONLY_ONNX_TRANSFORMS = (FP16ClipTransform, SplitTensorsTransform)


def _configure_proxy_for_model(instance: "QEFFBaseModel", enable_proxy: bool) -> None:
    """
    Configure per-instance transform lists based on proxy mode.

    Keep class-defined ONNX transforms by default.
    Proxy flow appends additional proxy-only transforms.
    """
    instance._pytorch_transforms = list(instance._pytorch_transforms)
    instance._onnx_transforms = list(instance._onnx_transforms)
    instance._enable_proxy = enable_proxy

    if enable_proxy:
        if QeffProxyModuleTransform not in instance._pytorch_transforms:
            instance._pytorch_transforms.append(QeffProxyModuleTransform)
        for transform in _PROXY_ONLY_ONNX_TRANSFORMS:
            if transform not in instance._onnx_transforms:
                instance._onnx_transforms.append(transform)
        logger.info("Proxy Model Enabled for QEfficient Model")


# Define a transformers layers to QEff layers dictionary
# While onboarding new models make sure to add the new layer maps to this dictionary.
TransformersToQEffModulesDict: Dict[Type[nn.Module], Type[nn.Module]] = {
    # GPT model layers
    GPT2Model: QEffGPT2Model,
    GPT2Block: QEffGPT2Block,
    GPT2Attention: QEffGPT2Attention,
    GPT2LMHeadModel: QEffGPT2LMHeadModel,
    # GPTJ model layers
    GPTJModel: QEffGPTJModel,
    GPTJAttention: QEffGPTJAttention,
    GPTJForCausalLM: QEffGPTJForCausalLM,
    # Llama model layers
    LlamaModel: QEffLlamaModel,
    LlamaAttention: QEffLlamaAttention,
    LlamaForCausalLM: QEffLlamaForCausalLM,
    LlamaDecoderLayer: QEffLlamaDecoderLayer,
    LlamaRMSNorm: CustomRMSNormAIC,
    LlamaRotaryEmbedding: QEffLlamaRotaryEmbedding,
    # Gemma model layers
    GemmaModel: QEffGemmaModel,
    GemmaAttention: QEffGemmaAttention,
    GemmaForCausalLM: QEffGemmaForCausalLM,
    GemmaDecoderLayer: QEffGemmaDecoderLayer,
    GemmaRMSNorm: CustomRMSNormAIC,
    # Gemma2 model layers
    Gemma2Model: QEffGemma2Model,
    Gemma2Attention: QEffGemma2Attention,
    Gemma2ForCausalLM: QEffGemma2ForCausalLM,
    Gemma2DecoderLayer: QEffGemma2DecoderLayer,
    Gemma2RMSNorm: CustomRMSNormAIC,
    # Gemma4 model layers
    Gemma4TextAttention: QEffGemma4TextAttention,
    Gemma4TextModel: QEffGemma4TextModel,
    Gemma4ForCausalLM: QEffGemma4ForCausalLM,
    Gemma4TextDecoderLayer: QEffGemma4TextDecoderLayer,
    Gemma4TextRouter: QEffGemma4TextRouter,
    Gemma4RMSNorm: QEffGemma4CustomRMSNormAIC,
    # MPT model layers
    MptAttention: QEffMptAttention,
    MptBlock: QEffMptBlock,
    MptModel: QEFfMptModel,
    MptForCausalLM: QEffMptForCausalLM,
    # CodeGen model layers
    CodeGenAttention: QEffCodeGenAttention,
    CodeGenModel: QEffCodeGenModel,
    CodeGenForCausalLM: QEffCodeGenForCausalLM,
    CodeGenBlock: QEffCodeGenBlock,
    # Mistral model layers
    MistralAttention: QEffMistralAttention,
    MistralDecoderLayer: QEffMistralDecoderLayer,
    MistralModel: QEffMistralModel,
    MistralForCausalLM: QEffMistralForCausalLM,
    MistralRMSNorm: CustomRMSNormAIC,
    # Mixtral model layers
    MixtralAttention: QEffMixtralAttention,
    MixtralDecoderLayer: QeffMixtralDecoderLayer,
    MixtralModel: QEffMixtralModel,
    MixtralForCausalLM: QEffMixtralForCausalLM,
    MixtralRMSNorm: CustomRMSNormAIC,
    MixtralSparseMoeBlock: QEffMixtralSparseMoeBlock,
    # Phi3 model layers
    Phi3Attention: QEffPhi3Attention,
    Phi3Model: QEffPhi3Model,
    Phi3ForCausalLM: QEffPhi3ForCausalLM,
    Phi3RMSNorm: CustomRMSNormAIC,
    # Phi model layers
    PhiAttention: QEffPhiAttention,
    PhiModel: QEffPhiModel,
    PhiForCausalLM: QEffPhiForCausalLM,
    # Falcon model layers
    FalconAttention: QEffFalconAttention,
    FalconForCausalLM: QEffFalconForCausalLM,
    FalconModel: QEffFalconModel,
    # Qwen2 model layers
    Qwen2Attention: QEffQwen2Attention,
    Qwen2ForCausalLM: QEffQwen2ForCausalLM,
    Qwen2Model: QEffQwen2Model,
    Qwen2RMSNorm: CustomRMSNormAIC,
    # Starcoder2 model layers
    Starcoder2Attention: QEffStarcoder2Attention,
    Starcoder2ForCausalLM: QEffStarcoder2ForCausalLM,
    Starcoder2Model: QEffStarcoder2Model,
    Starcoder2DecoderLayer: QEFFStarcoder2DecoderLayer,
    # Gpt_bigcode model layers
    GPTBigCodeForCausalLM: QEffGPTBigCodeForCausalLM,
    GPTBigCodeAttention: QEffGPTBigCodeAttention,
    GPTBigCodeBlock: QEffGPTBigCodeBlock,
    GPTBigCodeModel: QEffGPTBigCodeModel,
    # Whisper encoder and decoder layers
    WhisperAttention: QEffWhisperAttention,
    WhisperDecoderLayer: QEffWhisperDecoderLayer,
    WhisperEncoder: QEffWhisperEncoder,
    WhisperDecoder: QEffWhisperDecoder,
    WhisperPositionalEmbedding: QEffWhisperPositionalEmbedding,
    WhisperModel: QEffWhisperModel,
    WhisperForConditionalGeneration: QEffWhisperForConditionalGeneration,
}


def build_model_class_mapping(auto_model_class, qeff_class_name):
    """
    Build a mapping of model config class names to QEfficient model class names.
    """
    return {
        config_class.__name__: qeff_class_name for config_class, model_class in auto_model_class._model_mapping.items()
    }


EXTERNAL_MODEL_CLASS_MAPPING = {"Grok1Config": "QEFFAutoModelForCausalLM"}

MODEL_CLASS_MAPPING = {
    **build_model_class_mapping(mapping.AutoModelForCausalLM, "QEFFAutoModelForCausalLM"),
    **build_model_class_mapping(mapping.AutoModelForImageTextToText, "QEFFAutoModelForImageTextToText"),
}


def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32)
    )

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32)
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask, full_text_row_masked_out_mask


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = (
        attention_mask
        @ attention_mask.transpose(-1, -2)
        * torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32)
    )
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask


def _create_causal_mask(
    position_ids,
    target_length,
    sliding_window: Optional[int] = None,
):
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with sliding window
    """
    if sliding_window is not None:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, -1)
        # --- Rolling buffer ---
        pos_max = position_ids.max(1, keepdim=True).values
        kv_start = (pos_max // target_length) * target_length
        kv_indices_high = kv_indices + kv_start
        kv_indices_low = torch.where(kv_indices_high < target_length, kv_indices, kv_indices_high - target_length)
        kv_indices = torch.where(kv_indices_high > pos_max, kv_indices_low, kv_indices_high)
        kv_indices = kv_indices.unsqueeze(1)
        # ------
        causal_mask = kv_indices > query_indices
        attention_mask = causal_mask

        window_indices = query_indices - sliding_window + 1
        window_mask = kv_indices < window_indices
        attention_mask = attention_mask | window_mask
        attention_mask = attention_mask.unsqueeze(1)
    else:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, 1, -1)
        attention_mask = kv_indices > query_indices
        attention_mask = attention_mask.unsqueeze(1)

    return attention_mask
