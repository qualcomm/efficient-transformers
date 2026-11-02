# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Example pytorch_transforms.py showing common model onboarding patterns.

This file demonstrates three representative patterns:
1. Blueprint - Standard decoder-only model (example for onboarding)
2. Llama - Most common architecture pattern
3. Mixtral - Mixture of Experts (MoE) model

For more examples and patterns, see:
- Production transforms: QEfficient/base/pytorch_transforms.py
- All model implementations: QEfficient/transformers/models/
- Specific patterns:
  * Gemma (custom RMSNorm): QEfficient/transformers/models/gemma/
  * Multimodal (Llama4, Mllama): QEfficient/transformers/models/llama4/
  * External models (Grok): QEfficient/transformers/models/grok_1/
  * Vision-Language models: QEfficient/transformers/models/mllama/
"""

import warnings
from types import MethodType
from typing import Callable, Optional, Tuple, Union

from QEfficient.transformers.models.blueprint.modeling_blueprint import (
    QEffBlueprintAttention,
    QEffBlueprintDecoderLayer,
    QEffBlueprintForCausalLM,
    QEffBlueprintModel,
)
from torch import nn

# Example imports for three representative models
from transformers.models.blueprint.modeling_blueprint import (
    BlueprintAttention,
    BlueprintDecoderLayer,
    BlueprintForCausalLM,
    BlueprintModel,
    BlueprintRMSNorm,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)

from QEfficient.base.pytorch_transforms import ExternalModuleMapperTransform, ModuleMappingTransform
from QEfficient.customop import CustomRMSNormAIC
from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP, PooledModel, validate_user_pooling_function
from QEfficient.transformers.models.llama.modeling_llama import (
    QEffLlamaAttention,
    QEffLlamaDecoderLayer,
    QEffLlamaForCausalLM,
    QEffLlamaModel,
)
from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import (
    QEffMixtralAttention,
    QeffMixtralDecoderLayer,
    QEffMixtralForCausalLM,
    QEffMixtralModel,
    QEffMixtralSparseMoeBlock,
)
from QEfficient.transformers.post_processing import build_and_attach_mlp, model_type_registry
from QEfficient.transformers.sampler.sampler import sampler_forward
from QEfficient.transformers.spd.spd_transform_forward import tlm_forward

SPD_TARGET = "target"


class CustomOpsTransform(ModuleMappingTransform):
    """
    Maps RMSNorm classes to custom implementations optimized for Cloud AI 100.

    Most models use the standard CustomRMSNormAIC. For special cases (like Gemma),
    you can create custom RMSNorm in QEfficient.customop.
    """

    _module_mapping = {
        # Blueprint - Example model for onboarding
        BlueprintRMSNorm: CustomRMSNormAIC,
        # Llama - Most common pattern
        LlamaRMSNorm: CustomRMSNormAIC,
        # Mixtral - MoE model pattern
        MixtralRMSNorm: CustomRMSNormAIC,
        # TODO: Add your model's RMSNorm mapping here:
        # YourModelRMSNorm: CustomRMSNormAIC,
    }


class KVCacheTransform(ModuleMappingTransform):
    """
    Maps model classes to their QEfficient counterparts with KV cache support.

    This is the most critical transform for enabling efficient inference.
    All model classes (Attention, DecoderLayer, Model, ForCausalLM) must be mapped.
    """

    _module_mapping = {
        # Blueprint - Example model for onboarding
        BlueprintAttention: QEffBlueprintAttention,
        BlueprintDecoderLayer: QEffBlueprintDecoderLayer,
        BlueprintModel: QEffBlueprintModel,
        BlueprintForCausalLM: QEffBlueprintForCausalLM,
        # Llama - Most common pattern (standard decoder-only)
        LlamaAttention: QEffLlamaAttention,
        LlamaDecoderLayer: QEffLlamaDecoderLayer,
        LlamaModel: QEffLlamaModel,
        LlamaForCausalLM: QEffLlamaForCausalLM,
        # Mixtral - MoE model pattern (includes SparseMoeBlock)
        MixtralAttention: QEffMixtralAttention,
        MixtralSparseMoeBlock: QEffMixtralSparseMoeBlock,
        MixtralDecoderLayer: QeffMixtralDecoderLayer,
        MixtralModel: QEffMixtralModel,
        MixtralForCausalLM: QEffMixtralForCausalLM,
        # TODO: Add your model's class mappings here:
        # YourModelAttention: QEffYourModelAttention,
        # YourModelDecoderLayer: QEffYourModelDecoderLayer,
        # YourModelModel: QEffYourModelModel,
        # YourModelForCausalLM: QEffYourModelForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed


class SpDTransform:
    """
    Apply generic QEffForCausalLM forward pass to extract `num_speculative_tokens+1` hidden states before computing logits during decode phase and extract last predicted token during prefill.
    This is only needed if user is exporting Target Language Model (TLM) for Speculative Decoding to validate output logits
    against the speculated tokens from a smaller model.
    Other than the computed logits, there should be no difference between the SpD Transformed model and its corresponding cunterpart.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model.

    Returns:
        :model (nn.Module): PyTorch model.
        :transformed (bool): whether transformation was applied successfully.
    """

    # supported architectures
    _module_mapping = {
        QEffBlueprintForCausalLM,
        # TODO: Add your model's ForCausalLM class here if using Speculative Decoding:
        # QEffYourModelForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: Optional[dict] = None, **kwargs) -> Tuple[nn.Module, bool]:
        transformed = False
        pretrained_model_name_or_path_temp = kwargs.pop("pretrained_model_name_or_path", None)

        if qaic_config is None or (speculative_model_type := qaic_config.get("speculative_model_type")) is None:
            return model, transformed

        if speculative_model_type not in (supported_spd_model_types := [SPD_TARGET] + list(model_type_registry.keys())):
            raise ValueError(
                f"Speculative model type {speculative_model_type} is not supported. "
                f"Currently only support {supported_spd_model_types}"
            )

        if (model_class := model.__class__) in cls._module_mapping:
            model.forward = MethodType(tlm_forward, model)
            if speculative_model_type != SPD_TARGET:
                pretrained_model_name_or_path = qaic_config["pretrained_model_name_or_path"]
                model = build_and_attach_mlp(
                    model, pretrained_model_name_or_path, speculative_model_type=speculative_model_type, **kwargs
                )
            transformed = True
        else:
            raise NotImplementedError(
                f"Model class {model_class} does not yet support returning multiple logits to keep."
            )

        kwargs["pretrained_model_name_or_path"] = pretrained_model_name_or_path_temp
        return model, transformed


class SamplerTransform:
    """
    Add nodes at the output of any generic QEffForCausalLM model to enable the
    sampling of next tokens at the device (instead of the host) and return the
    next tokens and/or probability distributions.

    Note: To achieve this, the generic QEffForCausalLM model must provide the
    logits as output.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model.

    Returns:
        :model (nn.Module): PyTorch model.
        :transformed (bool): whether transformation was applied successfully.
    """

    # supported architectures
    _module_mapping = {
        # TODO: Add your model's ForCausalLM class here if using on-device sampling:
        # QEffYourModelForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: Optional[dict] = None, **kwargs) -> Tuple[nn.Module, bool]:
        transformed = False
        if qaic_config is None or not qaic_config.get("include_sampler", False):
            return model, transformed

        if (model_class := model.__class__) in cls._module_mapping:
            model.old_forward = model.forward
            model.forward = MethodType(sampler_forward, model)
            transformed = True
        else:
            raise NotImplementedError(f"Model class {model_class} does not support on device sampling.")

        return model, transformed


class VlmKVOffloadTransform(ModuleMappingTransform):
    """
    Vision-Language Model transform with KV offloading (two QPC setup).

    Used for multimodal models where vision and text processing are separated.
    See QEfficient/transformers/models/mllama/ for implementation examples.
    """

    _module_mapping = {
        # TODO: Add VLM models with KV offloading here:
        # YourVLMTextCrossAttention: QEffYourVLMTextCrossAttentionTwoQPC,
    }


class VlmNoKVOffloadTransform(ModuleMappingTransform):
    """
    Vision-Language Model transform without KV offloading (single QPC setup).

    Used for multimodal models in single QPC configuration.
    See QEfficient/transformers/models/mllama/ for implementation examples.
    """

    _module_mapping = {
        # TODO: Add VLM models without KV offloading here:
        # YourVLMTextCrossAttention: QEffYourVLMTextCrossAttentionSingleQPC,
    }


class KVCacheExternalModuleMapperTransform(ExternalModuleMapperTransform):
    _match_string_replace_method = {
        # TODO: Add external model mappings here (for models not in transformers library):
        # "YourExternalModelClass": {
        #     "forward": QEffYourExternalModel.forward,
        #     "__qeff_init__": QEffYourExternalModel.__qeff_init__,
        # },
    }

    _match_class_replace_method = {}


class PoolingTransform:
    """
    Apply a pooling transformation to the model. This transformation appends a pooling layer to the model, allowing for the reduction of spatial dimensions in the output.
    The pooling layer can be configured to use different pooling methods, such as max pooling or average pooling.
    """

    @classmethod
    def apply(cls, model: nn.Module, pooling: Union[str, Callable]) -> Tuple[nn.Module, bool]:
        transformed = False
        pooling_method = (
            POOLING_MAP[pooling]
            if isinstance(pooling, str) and pooling in POOLING_MAP
            else validate_user_pooling_function(pooling)
        )
        model = PooledModel(model, pooling_method)
        warnings.warn("Pooling is applied to the model.")
        return model, transformed
