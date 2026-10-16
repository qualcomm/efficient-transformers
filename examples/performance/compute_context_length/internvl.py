# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from io import BytesIO
from typing import List

import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Process the input messages to generate prompt for the model.
def get_prompt(messages) -> str:
    """Get the prompt for generation."""
    ## Chat template used for InternVL
    system_prompt = "<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    sep = "<|im_end|>\n"

    ret = system_prompt + sep
    for role, message in messages:
        if message:
            if type(message) is tuple:
                message, _, _ = message
            ret += role + message + sep
        else:
            ret += role
    return ret


# Processor class for InternVL models
class InternProcessor:
    """
    InternVL model only has an AutoTokenizer so this class performs the processing tasks similar to an AutoProcessor.
    The methods used here are borrowed from the original InternVL modelling files.
    "https://huggingface.co/OpenGVLab/InternVL2_5-1B/"
    """

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        patch_size = self.model.config.vision_config.patch_size
        self.template = model.config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.model.config.downsample_ratio**2))
        self.tokenizer = tokenizer

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def __call__(
        self,
        pixel_values,
        question,
        messages,
        roles,
        history=None,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ) -> str:
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        messages.append([roles[0], question])
        messages.append([roles[1], None])
        query = get_prompt(messages)
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            logger.info(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)
        return query


def run_intern_on_aic(
    model_name,
    prompt,
    image_url,
    messages,
    roles,
    kv_offload=False,
    prefill_seq_len=3840,
    num_devices=1,
    num_cores=16,
    ctx_len=512,
    ccl_enabled=False,
    comp_ctx_lengths_prefill=None,
    comp_ctx_lengths_decode=None,
):
    ## STEP 1 -- LOAD THE MODEL

    # The original Intern-VL model, despite being multimodal, is loaded using `AutoModelForCausalLM` in Huggingface.
    # To maintain compatibility, we load this model using `QEFFAutoModelForCausalLM`.

    model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name,
        kv_offload=kv_offload,
        trust_remote_code=True,
        qaic_config={
            "ccl_enabled": ccl_enabled,
        },
    )

    ## STEP 2 -- EXPORT & COMPILE THE MODEL

    model.compile(
        num_cores=num_cores,
        num_devices=num_devices,
        ctx_len=ctx_len,
        prefill_seq_len=prefill_seq_len,
        mxfp6_matmul=False,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )

    ## STEP 3 -- SETUP THE PROCESSOR

    # InternVL doesn't have an AutoProcessor yet, so we will use our own processor class "InternProcessor"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    internProcessor = InternProcessor(model.model, tokenizer)

    ## STEP 4 -- PREPROCESS THE INPUTS

    img = requests.get(image_url, stream=True)
    image = Image.open(BytesIO(img.content)).convert("RGB")

    # Images are resized to (1000, 747) for inference
    image = image.resize((1000, 747))

    # preprocess the resized image
    pixel_values = internProcessor.load_image(image, max_num=12)
    question = "<image>\n" + prompt
    query = internProcessor(pixel_values, question, messages, roles)
    inputs = tokenizer(
        query, return_tensors="pt", padding="max_length", max_length=prefill_seq_len, padding_side="right"
    )

    inputs["pixel_values"] = pixel_values

    ## STEP 5 -- RUN INFERENCE VIA GENERATE FUNCTION
    streamer = TextStreamer(tokenizer)
    model.generate(inputs=inputs, streamer=streamer, generation_len=128)


if __name__ == "__main__":
    model_name = "OpenGVLab/InternVL2_5-1B"

    # Chat Template information for prompt preprocessing
    messages: List[List[str]] = []
    roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")

    # Inputs for the model
    prompt = "Please describe the image in detail."
    image_url = "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg"

    ## Compilation parameters

    # `kv_offload` is used to compile the model in a Single QPC or 2 QPCs.
    # The Dual QPC approach splits the model to perform Image Encoding and Output generation in 2 different QPCs.
    # The outputs of the Vision Encoder are then passed to the Language model via host in this case.

    kv_offload = True

    # InternVL is an Early-Fusion model that uses placeholder tokens within the input_ids to interleave text_embeddings with
    # Image embeddings and generate final input_embeds for outout generation. Hence we need very large prefill_seq_len (3840 in this case) to
    # incorporate the memory for the merged embeddings.

    prefill_seq_len = 3840
    num_devices = 4
    num_cores = 16

    ctx_len = 8192
    ccl_enabled = True
    # Two optional lists, comp_ctx_lengths_prefill and comp_ctx_lengths_decode, define CCL values for prefilling and decoding. If both are None, the lists will be generated automatically based on the context length.
    comp_ctx_lengths_prefill = [4096]
    comp_ctx_lengths_decode = [6144, ctx_len]

    run_intern_on_aic(
        model_name=model_name,
        prompt=prompt,
        image_url=image_url,
        messages=messages,
        roles=roles,
        kv_offload=kv_offload,
        prefill_seq_len=prefill_seq_len,
        num_devices=num_devices,
        num_cores=num_cores,
        ctx_len=ctx_len,
        ccl_enabled=ccl_enabled,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )


"""
Expected Response:

The image is a promotional graphic for Microsoft Azure. It features a blue background with a hexagonal pattern on the left side. The hexagons are white and are arranged in a way that suggests a network or connectivity theme. 

On the right side of the image, the Microsoft Azure logo is prominently displayed. The logo consists of the Azure name in white, with the Microsoft logo above it, which includes four colored squares (blue, green, yellow, and red). Below the logo, the word "Azure" is written in large white letters.

Below the logo, there is text that reads:
- "By Dinesh Kumar Wick
"""
