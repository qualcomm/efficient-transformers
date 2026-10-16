# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.models.llava.modeling_llava import (
    LlavaForConditionalGeneration,
)

from QEfficient.utils._utils import IOInfo
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")
BS = 1
FBS = 4
NUM_CHANNEL = 3
SEQ_LEN = 592
CTX_LEN = 1024


class QEFFLlavaEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.model.vision_tower

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.model.vision_tower.vision_model.encoder.layers[0].__class__}

    def forward(self, pixel_values):
        # Image features
        image_outputs = self.model.model.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.model.config.vision_feature_layer]
        vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.model.config.vision_feature_select_strategy}")
        vision_embeds = self.model.model.multi_modal_projector(selected_image_feature)

        return vision_embeds


class QEFFLlavaDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.language_model = self.model.model.language_model
        self.lm_head = self.model.lm_head

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.language_model.layers[0].__class__}

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.model.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        vision_embeds_expanded = vision_embeds[indices0, indices1]
        vision_embeds_expanded = torch.where(mask.unsqueeze(-1), vision_embeds_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, vision_embeds_expanded)
        outputs = self.model.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            return_dict=True,
        )

        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits, vision_embeds, image_idx, outputs.past_key_values


class QEffLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEFFLlavaEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEFFLlavaDecoderWrapper(self)

    def forward(
        self,
        input_ids,
        position_ids,
        pixel_values,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # Image features
        image_outputs = self.model.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.config.vision_feature_layer]
        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        vision_embeds = self.model.multi_modal_projector(selected_image_feature)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        vision_embeds_expanded = vision_embeds[indices0, indices1]
        image_embeds = torch.where(mask.unsqueeze(-1), vision_embeds_expanded, inputs_embeds)
        # *where to skip image encoder for decode*
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_embeds)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
        )

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)
        return logits, pixel_values, image_idx, outputs.past_key_values

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        prefill_seq_len = kwargs.get("prefill_seq_len")
        if prefill_seq_len is None:
            prefill_seq_len = SEQ_LEN
        prefill_seq_len = int(prefill_seq_len)
        num_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 336)
        else:
            img_size = 336
        if img_size != 336 and kv_offload:
            raise NotImplementedError("Image Size other than 336 is not supported for Llava models yet.")
        vision_size = img_size // self.config.vision_config.patch_size
        vision_inputs = {
            "pixel_values": torch.zeros((BS, NUM_CHANNEL, img_size, img_size), dtype=self.config.torch_dtype),
        }
        lang_inputs = {
            "input_ids": torch.ones((BS, prefill_seq_len), dtype=torch.int64),
            "vision_embeds": torch.ones(
                (BS, vision_size, self.model.language_model.config.hidden_size), dtype=self.config.torch_dtype
            ),
            "attention_mask": torch.ones((BS, prefill_seq_len), dtype=torch.int64),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
        }
        lang_inputs["position_ids"] = lang_inputs.pop("attention_mask").cumsum(1)
        lang_inputs["past_key_values"] = []
        for i in range(num_layers):
            lang_inputs["past_key_values"].append(
                (
                    torch.zeros(
                        FBS if continuous_batching else BS,
                        num_key_value_heads,
                        CTX_LEN,
                        head_dim,
                        dtype=self.config.torch_dtype,
                    ),
                    torch.zeros(
                        FBS if continuous_batching else BS,
                        num_key_value_heads,
                        CTX_LEN,
                        head_dim,
                        dtype=self.config.torch_dtype,
                    ),
                )
            )
        lang_inputs["position_ids"] = torch.full(lang_inputs["position_ids"].shape, CTX_LEN - 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(BS).view(BS, 1)
        inputs = {}

        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            inputs = {**vision_inputs, **lang_inputs}
        return inputs

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        vision_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        max_num_images = compiler_options.pop("max_num_images", 1)
        # Preserve the legacy shape when callers do not request a separate vision batch.
        vision_batch_size = batch_size if vision_batch_size is None else vision_batch_size
        prefill_seq_len = prefill_seq_len if prefill_seq_len else SEQ_LEN
        ctx_len = ctx_len if ctx_len else CTX_LEN
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 336
            logger.warning("Setting img_size to be 336, as it was neither passed nor found in vision_config")
        if img_size != 336 and kv_offload:
            raise NotImplementedError("Image Size other than 336 is not supported for Llava models yet.")
        user_vision_size = compiler_options.pop("vision_size", None)
        if user_vision_size:
            if user_vision_size >= ctx_len:
                raise ValueError("vision_size must be less than ctx_len")
            vision_size = user_vision_size
        else:
            vision_size = (img_size // self.config.vision_config.patch_size) ** 2
        vision = [
            {
                "vision_batch_size": vision_batch_size,
                "max_num_images": max_num_images,
                "img_size": img_size,
            }
        ]

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            lang = []

            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                    "max_num_images": max_num_images,
                    "img_size": img_size,
                    "vision_size": vision_size,
                    "vision_batch_size": vision_batch_size,
                }
                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_prefill["batch_size"] = kv_cache_batch_size
                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size
                lang.append(lang_prefill)

            for i in range(0, len(comp_ctx_lengths_decode)):
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                    "max_num_images": max_num_images,
                    "img_size": img_size,
                    "vision_size": vision_size,
                    "vision_batch_size": vision_batch_size,
                }
                if continuous_batching:
                    lang_decode["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_decode["batch_size"] = kv_cache_batch_size
                lang.append(lang_decode)
        else:
            lang_prefill = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "max_num_images": max_num_images,
                "img_size": img_size,
                "vision_size": vision_size,
                "vision_batch_size": vision_batch_size,
            }
            if continuous_batching:
                lang_prefill["full_batch_size"] = kv_cache_batch_size
            else:
                lang_prefill["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                lang_prefill["full_batch_exec_size"] = full_batch_size

            lang_decode = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "max_num_images": max_num_images,
                "img_size": img_size,
                "vision_size": vision_size,
                "vision_batch_size": vision_batch_size,
            }
            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size
            else:
                lang_decode["batch_size"] = kv_cache_batch_size

            lang = [lang_prefill, lang_decode]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            lang[0].pop("vision_size")
            lang[1].pop("vision_size")
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        # Define dynamic axes
        num_layers = self.config.text_config.num_hidden_layers

        vision_dynamic_axes = {
            "pixel_values": {0: "vision_batch_size", 2: "img_size", 3: "img_size"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }
            lang_dynamic_axes[f"past_value.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.model.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return lang_output_names
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values", datatype=self.config.torch_dtype, shape=("batch_size", 3, "img_size", "img_size")
            ),
        ]
