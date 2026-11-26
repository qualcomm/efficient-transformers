# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import math
from typing import List

import numpy as np
import torch

from QEfficient.utils import (
    get_num_layers_from_config,
    get_padding_shape_from_config,
    get_sliding_window_layers,
    get_sliding_window_shapes,
    padding_check_and_fix,
)


def build_block_table(
    dp: int,
    cp: int,
    batch_size: int,
    num_logical_pages: int,
    generator: torch.Generator,
    physical_pages: int | None = None,
) -> torch.Tensor:
    """Build a DP-major logical-to-physical page table."""
    if batch_size % dp:
        raise ValueError("batch_size must be divisible by dp")
    batch_local = batch_size // dp
    num_page_groups = math.ceil(num_logical_pages / cp)
    required_pages = batch_local * num_page_groups
    physical_pages = required_pages if physical_pages is None else int(physical_pages)
    if physical_pages < required_pages:
        raise ValueError(
            f"physical page capacity ({physical_pages}) is smaller than the required logical entries "
            f"({required_pages})."
        )
    table = torch.empty((dp, batch_local, num_page_groups), dtype=torch.int32)
    for dp_idx in range(dp):
        table[dp_idx] = torch.randperm(
            physical_pages,
            dtype=torch.int32,
            generator=generator,
        )[:required_pages].view(batch_local, num_page_groups)
    return table


class InputHandler:
    def __init__(
        self,
        batch_size,
        tokenizer,
        config,
        prompt,
        prompt_len,
        ctx_len,
        full_batch_size,
        dtype=torch.float32,
        qaic_config=None,
    ):
        """
        Initialization

        ``Mandatory`` Args:
            :batch_size (int): Number of prompts to run in one batch.
            :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Pass model tokenizer.
            :config (AutoConfig): From pretrained model.
            :prompt (List[str]): String to used as input prompt for the model.
            :prompt_len (int): Prompt length for the model to compile.
            :ctx_len (int): Maximum context length to compile the model.
            :full_batch_size (int): Continuous batching batch size
        """
        # check and fix tokenizer viability
        padding_check_and_fix(tokenizer)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len
        self.full_batch_size = full_batch_size
        self.config = config
        self.dtype = dtype
        self.qaic_config = qaic_config
        self.n_layer = get_num_layers_from_config(config)
        self.padding_shape = get_padding_shape_from_config(
            config=config, batch_size=full_batch_size if full_batch_size else batch_size, seq_len=ctx_len
        )

        self.is_chunked_attention = get_sliding_window_layers(config)
        self.global_shape, self.sliding_shape = get_sliding_window_shapes(
            config=config, batch_size=full_batch_size if full_batch_size else batch_size, seq_len=ctx_len
        )

    def _get_layer_cache_shape(self, layer_idx):
        if not hasattr(self.config, "layer_types") or self.config.layer_types is None:
            if hasattr(self.config, "sliding_window") and hasattr(self.config, "sliding_window_pattern"):
                is_sliding = bool((layer_idx + 1) % self.config.sliding_window_pattern)
                if is_sliding:
                    return self.padding_shape[:2] + [self.config.sliding_window] + [self.padding_shape[-1]]
            return self.padding_shape

        head_dim = (
            getattr(self.config, "head_dim", None)
            or (
                self.config.hidden_size // self.config.num_attention_heads
                if getattr(self.config, "hidden_size", None) is not None
                and getattr(self.config, "num_attention_heads", None) is not None
                else None
            )
            or self.padding_shape[-1]
        )

        layer_type = self.config.layer_types[layer_idx]
        if layer_type == "sliding_attention":
            n_heads = self.config.num_key_value_heads
            d_head = head_dim
            ctx_len = min(self.config.sliding_window, self.ctx_len)
        else:
            use_alternative_attention = getattr(self.config, "attention_k_eq_v", False)
            n_heads = (
                self.config.num_global_key_value_heads
                if use_alternative_attention and getattr(self.config, "num_global_key_value_heads", None) is not None
                else self.config.num_key_value_heads
            )
            d_head = self.config.global_head_dim if getattr(self.config, "global_head_dim", None) else head_dim
            ctx_len = self.ctx_len

        batch = self.full_batch_size if self.full_batch_size else self.padding_shape[0]
        return [batch, n_heads, ctx_len, d_head]

    def _is_paged_attention(self) -> bool:
        if self.qaic_config is None:
            return False
        blocking_mode = self.qaic_config.get("blocking_mode")
        return blocking_mode is not None and "paged" in blocking_mode

    def _get_num_kv_blocks(self):
        if not self._is_paged_attention():
            return None
        return self.qaic_config.get("num_kv_blocks")

    def _get_kv_block_size(self):
        num_kv_blocks = self._get_num_kv_blocks()
        if num_kv_blocks is None:
            return None
        return -(-self.ctx_len // num_kv_blocks)

    def prepare_pytorch_inputs(self):
        """
        Function responsible for creating Prefill stage tensor inputs for PyTorch model.

        Return:
            :Dict: input_ids, position_ids, past_key_values
        """

        inputs = self.tokenizer(
            self.prompt,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        batch_size, input_len = input_ids.shape
        inputs.pop("attention_mask")
        inputs.pop("token_type_ids", None)
        usable_bs = self.full_batch_size if self.full_batch_size else 1
        position_ids = torch.arange(input_len).view(1, input_len).repeat(usable_bs, 1)
        inputs["input_ids"] = torch.concat(
            [
                input_ids,
                torch.ones((batch_size, self.prompt_len - input_len), dtype=torch.int64)
                * (self.tokenizer.pad_token_id),
            ],
            1,
        )
        inputs["position_ids"] = torch.concat(
            [
                position_ids,
                torch.ones((batch_size, self.prompt_len - input_len), dtype=torch.int64) * (-1),
            ],
            1,
        )

        if self.full_batch_size:
            inputs["input_ids"] = input_ids
            inputs["position_ids"] = position_ids
            inputs["batch_index"] = torch.arange(self.full_batch_size).view(-1, 1)

        past_key_values = []
        num_kv_blocks = self._get_num_kv_blocks()
        kv_block_size = self._get_kv_block_size()

        if self._is_paged_attention() and num_kv_blocks and kv_block_size:
            inputs["block_table"] = torch.arange(batch_size * num_kv_blocks, dtype=torch.int64).reshape(
                batch_size, num_kv_blocks
            )
            inputs["slot_id"] = torch.zeros(batch_size, dtype=torch.int64)
            pad_shape_paged = [batch_size * num_kv_blocks, self.padding_shape[1], kv_block_size, self.padding_shape[3]]

        for i in range(self.n_layer):
            if self._is_paged_attention() and num_kv_blocks and kv_block_size:
                pad_shape = pad_shape_paged
            else:
                pad_shape = self._get_layer_cache_shape(i)
            past_key = torch.zeros((pad_shape), dtype=self.dtype)
            past_value = torch.zeros((pad_shape), dtype=self.dtype)
            pkv = (past_key, past_value)
            past_key_values.append(pkv)
        inputs["past_key_values"] = tuple(past_key_values)

        return inputs

    def update_pytorch_inputs(self, inputs, pt_outputs):
        """
        Function responsible for updating Prefill stage inputs to create decode stage inputs for PyTorch model.

        ``Mandatory`` Args:
            :inputs (Dict): Pytorch inputs from previous iteration
            :pt_outputs (Dict): Pytorch outputs from previous iteration

        Return:
            :Dict: Updated input_ids, position_ids and past_key_values
        """
        updated_inputs = {}
        if self.full_batch_size:
            input_ids = pt_outputs.logits.detach().argmax(2)
            updated_inputs["input_ids"] = torch.full((self.full_batch_size, 1), self.tokenizer.pad_token_id)
            updated_inputs["input_ids"][inputs["batch_index"].view(-1)] = input_ids

            position_ids = inputs["position_ids"].max(1, keepdim=True).values + 1
            updated_inputs["position_ids"] = torch.full((self.full_batch_size, 1), 0)
            updated_inputs["position_ids"][inputs["batch_index"].view(-1)] = position_ids

            updated_inputs["batch_index"] = inputs["batch_index"]
        else:
            updated_inputs["input_ids"] = pt_outputs["logits"].argmax(-1).reshape(-1, 1)
            updated_inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1

        pkv = pt_outputs["past_key_values"]
        if isinstance(pkv, (list, tuple)):
            normalized_pkv = []
            for layer_cache in pkv:
                if isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
                    key, value = layer_cache[0], layer_cache[1]
                    normalized_pkv.append((key.detach(), value.detach()))
            updated_inputs["past_key_values"] = tuple(normalized_pkv)
        else:
            updated_inputs["past_key_values"] = pkv

        num_kv_blocks = self._get_num_kv_blocks()
        kv_block_size = self._get_kv_block_size()
        if self._is_paged_attention() and num_kv_blocks and kv_block_size and "slot_id" in inputs:
            updated_inputs["slot_id"] = ((updated_inputs["position_ids"]) % kv_block_size).view(
                updated_inputs["position_ids"].shape[0]
            )
            updated_inputs["block_table"] = torch.arange(
                updated_inputs["input_ids"].shape[0] * inputs["block_table"].shape[1], dtype=torch.int64
            ).reshape(updated_inputs["input_ids"].shape[0], inputs["block_table"].shape[1])

        return updated_inputs

    def prepare_ort_inputs(self):
        """
        Function responsible for creating Prefill stage numpy inputs for ONNX model to be run on ONNXRT.

        Return:
            :Dict: input_ids, position_ids, past_key_values
        """

        inputs = self.tokenizer(
            self.prompt,
            return_tensors="np",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        batch_size, input_len = input_ids.shape
        inputs.pop("attention_mask")
        inputs.pop("token_type_ids", None)
        position_ids = np.arange(input_len).reshape(1, -1)
        inputs["input_ids"] = np.concatenate(
            [input_ids, np.full((batch_size, self.prompt_len - input_len), self.tokenizer.pad_token_id)],
            axis=1,
        ).astype(np.int64)
        inputs["position_ids"] = np.concatenate(
            [position_ids, np.full((batch_size, self.prompt_len - input_len), -1)],
            axis=1,
        ).astype(np.int64)

        num_kv_blocks = self._get_num_kv_blocks()
        kv_block_size = self._get_kv_block_size()

        if self._is_paged_attention() and num_kv_blocks and kv_block_size:
            inputs["block_table"] = np.arange(batch_size * num_kv_blocks, dtype=np.int64).reshape(
                batch_size, num_kv_blocks
            )
            inputs["slot_id"] = np.zeros(batch_size, dtype=np.int64)
            pad_shape_paged = [batch_size * num_kv_blocks, self.padding_shape[1], kv_block_size, self.padding_shape[3]]

        for i in range(self.n_layer):
            if self._is_paged_attention() and num_kv_blocks and kv_block_size:
                pad_shape = pad_shape_paged
            else:
                pad_shape = self._get_layer_cache_shape(i)
            inputs["past_key." + str(i)] = np.zeros((pad_shape), dtype=np.float32)
            inputs["past_value." + str(i)] = np.zeros((pad_shape), dtype=np.float32)
        if self.full_batch_size:
            inputs["batch_index"] = np.arange(self.full_batch_size).reshape(-1, 1)

        return inputs

    def update_ort_inputs(self, inputs, ort_outputs):
        """
        Function responsible for updating Prefill stage inputs to create inputs for decode stage inputs for ONNX model to be run on ONNXRT.

        ``Mandatory`` Args:
            :inputs (Dict): NumPy inputs of Onnx model from previous iteration
            :ort_outputs (Dict): Numpy outputs of Onnx model from previous iteration

        Return:
            :Dict: Updated input_ids, position_ids and past_key_values
        """

        updated_inputs = {}
        updated_inputs["input_ids"] = ort_outputs["logits"][:, -1, :].argmax(-1).reshape(-1, 1)
        updated_inputs["position_ids"] = np.max(inputs["position_ids"], axis=1, keepdims=True) + 1
        for i in range(self.n_layer):
            updated_inputs["past_key." + str(i)] = ort_outputs["past_key_values"][i * 2]
            updated_inputs["past_value." + str(i)] = ort_outputs["past_key_values"][i * 2 + 1]
        if self.full_batch_size:
            updated_inputs["batch_index"] = inputs["batch_index"]

        num_kv_blocks = self._get_num_kv_blocks()
        kv_block_size = self._get_kv_block_size()
        if self._is_paged_attention() and num_kv_blocks and kv_block_size and "slot_id" in inputs:
            updated_inputs["slot_id"] = ((updated_inputs["position_ids"]) % kv_block_size).reshape(
                updated_inputs["position_ids"].shape[0]
            )
            updated_inputs["block_table"] = np.arange(
                updated_inputs["input_ids"].shape[0] * inputs["block_table"].shape[1], dtype=np.int64
            ).reshape(updated_inputs["input_ids"].shape[0], inputs["block_table"].shape[1])

        return updated_inputs

    def update_ort_outputs(self, ort_outputs):
        """
        Function responsible for updating ONNXRT session outputs.

        ``Mandatory`` Args:
            :ort_outputs (Dict): Numpy outputs of Onnx model from current iteration

        Return:
            updated_outputs (Dict): Updated past_key_values, logits
        """

        present_key_values = []
        for i in range(self.n_layer):
            if "past_key." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_key." + str(i) + "_RetainedState"])
            if "past_value." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_value." + str(i) + "_RetainedState"])

        outputs = {}
        outputs["past_key_values"] = present_key_values
        outputs["logits"] = ort_outputs["logits"]

        return outputs


class InputHandlerVLM:
    def __init__(
        self,
        batch_size,
        config,
        image,
        conversation,
        processor,
        prompt,
        prompt_len,
        ctx_len,
        max_gen_len,
        n_layer,
        dtype=torch.float32,
    ):
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.max_gen_len = max_gen_len
        self.config = config
        self.image = image
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.processor = processor
        self.conversation = conversation
        self.dtype = dtype

    def prepare_pytorch_inputs(self):
        """
        Function responsible for creating Prefill stage tensor inputs for PyTorch model.

        Return:
            :Dict: input_ids, position_ids, past_key_values
        """
        inputs = self.processor(images=self.image, text=self.prompt, return_tensors="pt")
        if hasattr(self.config, "text_config"):
            txt_cfg = self.config.text_config
        else:
            txt_cfg = self.config.llm_config

        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads)
        if hasattr(txt_cfg, "cross_attention_layers"):
            cross_attention_layers = txt_cfg.cross_attention_layers

            vis_cfg = self.config.vision_config
            num_patches = (vis_cfg.image_size // vis_cfg.patch_size) ** 2 + 1
            image_tokens_len = vis_cfg.max_num_tiles * num_patches

        inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1) - 1
        inputs["past_key_values"] = []
        for i in range(num_hidden_layers):
            # Specific to mllama as of now
            if hasattr(txt_cfg, "cross_attention_layers") and i in cross_attention_layers:
                idx = cross_attention_layers.index(i)
                assert idx == ((i - 3) // 5), f"{i}, {(i - 3) // 5}"
                inputs["past_key_values"].append(
                    (
                        torch.zeros((1, num_key_value_heads, image_tokens_len, head_dim), dtype=self.dtype),
                        torch.zeros((1, num_key_value_heads, image_tokens_len, head_dim), dtype=self.dtype),
                    )
                )
            else:
                inputs["past_key_values"].append(
                    (
                        torch.zeros((1, num_key_value_heads, self.ctx_len, head_dim), dtype=self.dtype),
                        torch.zeros((1, num_key_value_heads, self.ctx_len, head_dim), dtype=self.dtype),
                    )
                )

        return inputs

    def prepare_vlm_ort_inputs(self):
        if hasattr(self.config, "text_config"):
            txt_cfg = self.config.text_config
        else:
            txt_cfg = self.config.llm_config
        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads)
        if hasattr(txt_cfg, "cross_attention_layers"):
            cross_attention_layers = txt_cfg.cross_attention_layers
            vis_cfg = self.config.vision_config
            num_patches = (vis_cfg.image_size // vis_cfg.patch_size) ** 2 + 1
            image_tokens_len = vis_cfg.max_num_tiles * num_patches

        inputs = self.processor(images=self.image, text=self.prompt, return_tensors="np")
        if "attention_mask" in inputs.keys():
            inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1) - 1
        inputs["past_key_values"] = []
        inputs["image_idx"] = np.array([[0]])

        vision_inputs = {
            k: v
            for k, v in inputs.items()
            if k in {"pixel_values", "image_position_ids", "aspect_ratio_ids", "aspect_ratio_mask"}
        }

        for i in range(num_hidden_layers):
            if hasattr(txt_cfg, "cross_attention_layers") and i in cross_attention_layers:
                idx = cross_attention_layers.index(i)
                assert idx == ((i - 3) // 5), f"{i}, {(i - 3) // 5}"
                inputs["past_key." + str(i)] = np.zeros(
                    (self.batch_size, num_key_value_heads, image_tokens_len, head_dim), dtype=np.float32
                )
                inputs["past_value." + str(i)] = np.zeros(
                    (self.batch_size, num_key_value_heads, image_tokens_len, head_dim), dtype=np.float32
                )
            else:
                inputs["past_key." + str(i)] = np.zeros(
                    (self.batch_size, num_key_value_heads, self.ctx_len, head_dim), dtype=np.float32
                )
                inputs["past_value." + str(i)] = np.zeros(
                    (self.batch_size, num_key_value_heads, self.ctx_len, head_dim), dtype=np.float32
                )
        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        return vision_inputs, lang_inputs

    def update_vlm_ort_outputs(self, ort_outputs):
        """
        Function responsible for updating ONNXRT session outputs.

        ``Mandatory`` Args:
            :ort_outputs (Dict): Numpy outputs of Onnx model from current iteration

        Return:
            updated_outputs (Dict): Updated past_key_values, logits, pixel_values
        """
        present_key_values = []
        for i in range(self.n_layer[0]):
            if "past_key." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_key." + str(i) + "_RetainedState"])
            if "past_value." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_value." + str(i) + "_RetainedState"])

        outputs = {}
        outputs["past_key_values"] = present_key_values
        outputs["logits"] = ort_outputs["logits"]
        outputs["pixel_values_RetainedState"] = (
            ort_outputs["pixel_values_RetainedState"] if "pixel_values_RetainedState" in ort_outputs else None
        )
        outputs["image_features_RetainedState"] = (
            ort_outputs["image_features_RetainedState"] if "image_features_RetainedState" in ort_outputs else None
        )
        outputs["vision_embeds_RetainedState"] = (
            ort_outputs["vision_embeds_RetainedState"] if "vision_embeds_RetainedState" in ort_outputs else None
        )
        outputs["image_idx"] = ort_outputs["image_idx_output"]
        return outputs

    def update_vlm_ort_inputs(self, inputs, ort_outputs):
        """
        Function responsible for updating Prefill stage inputs to create inputs for decode stage inputs for ONNX model to be run on ONNXRT.

        ``Mandatory`` Args:
            :inputs (Dict): NumPy inputs of Onnx model from previous iteration
            :ort_outputs (Dict): Numpy outputs of Onnx model from previous iteration

        Return:
            :Dict: Updated input_ids, position_ids, pixel_values and past_key_values
        """
        updated_inputs = {}
        updated_inputs["input_ids"] = ort_outputs["logits"].argmax(-1)
        updated_inputs["position_ids"] = np.max(inputs["position_ids"], axis=1, keepdims=True) + 1
        for i in range(self.n_layer[0]):
            updated_inputs["past_key." + str(i)] = ort_outputs["past_key_values"][i * 2]
            updated_inputs["past_value." + str(i)] = ort_outputs["past_key_values"][i * 2 + 1]
        if "pixel_values_RetainedState" in ort_outputs.keys():
            updated_inputs["pixel_values"] = ort_outputs["pixel_values_RetainedState"]
        if "image_features_RetainedState" in ort_outputs.keys():
            updated_inputs["image_features"] = ort_outputs["image_features_RetainedState"]
        if "vision_embeds_RetainedState" in ort_outputs.keys():
            updated_inputs["vision_embeds"] = ort_outputs["vision_embeds_RetainedState"]
        if "mm_token_type_ids" in inputs.keys():
            updated_inputs["mm_token_type_ids"] = np.zeros_like(
                updated_inputs["input_ids"], dtype=inputs["mm_token_type_ids"].dtype
            )

        if "cross_attention_mask" in inputs.keys():
            bs, _, num_images, img_tiles = inputs["cross_attention_mask"].shape
            updated_inputs["cross_attention_mask"] = torch.ones(
                (bs, 1, num_images, img_tiles), dtype=torch.int64
            ).numpy()

        for k, v in inputs.items():
            if k not in updated_inputs.keys():
                updated_inputs[k] = v
        return updated_inputs


class InputHandlerInternVL(InputHandlerVLM):
    def __init__(
        self,
        batch_size,
        config,
        image,
        processor,
        prompt,
        prompt_len,
        ctx_len,
        max_gen_len,
        n_layer,
        dtype=torch.float32,
    ):
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.max_gen_len = max_gen_len
        self.config = config
        self.image = image
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.processor = processor
        self.dtype = dtype

    def prepare_pytorch_inputs(self):
        question = "<image>\n" + self.prompt
        pixel_values = self.processor.load_image(self.image, max_num=12)
        # Chat Template information for prompt preprocessing
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = self.processor(pixel_values, question, messages, roles)
        inputs = self.processor.tokenizer(prompt, return_tensors="pt")
        inputs["pixel_values"] = pixel_values.clone()

        if hasattr(self.config, "text_config"):
            txt_cfg = self.config.text_config
        else:
            txt_cfg = self.config.llm_config

        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads)

        inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1) - 1
        inputs["past_key_values"] = []
        for i in range(num_hidden_layers):
            inputs["past_key_values"].append(
                (
                    torch.zeros((1, num_key_value_heads, self.ctx_len, head_dim), dtype=self.dtype),
                    torch.zeros((1, num_key_value_heads, self.ctx_len, head_dim), dtype=self.dtype),
                )
            )

        return inputs

    def prepare_vlm_ort_inputs(self):
        if hasattr(self.config, "text_config"):
            txt_cfg = self.config.text_config
        else:
            txt_cfg = self.config.llm_config
        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads)

        question = "<image>\n" + self.prompt
        pixel_values = self.processor.load_image(self.image, max_num=12)
        # Chat Template information for prompt preprocessing
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = self.processor(pixel_values, question, messages, roles)
        inputs = self.processor.tokenizer(prompt, return_tensors="np")
        inputs["pixel_values"] = pixel_values.numpy()

        if "attention_mask" in inputs.keys():
            inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1) - 1
        inputs["past_key_values"] = []
        inputs["image_idx"] = np.array([[0]])

        vision_inputs = {
            k: v for k, v in inputs.items() if k in {"pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"}
        }

        for i in range(num_hidden_layers):
            inputs["past_key." + str(i)] = np.zeros(
                (self.batch_size, num_key_value_heads, self.ctx_len, head_dim), dtype=np.float32
            )
            inputs["past_value." + str(i)] = np.zeros(
                (self.batch_size, num_key_value_heads, self.ctx_len, head_dim), dtype=np.float32
            )
        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        return vision_inputs, lang_inputs
