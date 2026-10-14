# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download


def get_custom_n_layers(model_name):
    """
    Function to set number layers of the variuos types of models such as swiftkv models and others
    --------

    :model_name: str

    :return n_layer
    """
    if model_name in {"microsoft/Phi-3-mini-4k-instruct", "neuralmagic/Qwen2-0.5B-Instruct-FP8", "openai/gpt-oss-20b"}:
        return 2
    elif model_name in ModelConfig.SWIFTKV_MODELS:
        return -1
    return 1


def load_hf_causal_lm_model(
    model_name: str,
    num_hidden_layers: int = -1,
    config: Optional[AutoConfig] = None,
    torch_dtype: Optional[torch.dtype] = torch.float32,
):
    model_path = hf_download(
        repo_id=model_name,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    if config is None:
        model_kwargs = dict(attn_implementation="eager", low_cpu_mem_usage=False, torch_dtype=torch_dtype)
        if num_hidden_layers != -1:
            model_kwargs["num_hidden_layers"] = num_hidden_layers
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
            **model_kwargs,
        )
    else:
        model_hf = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
            trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        )

    try:
        model_hf = model_hf.to(torch_dtype)
        model_hf.config.torch_dtype = torch_dtype
    except ValueError:
        pass
    model_hf.eval()
    return model_hf


def load_qeff_causal_lm_model(
    model_name: str,
    num_hidden_layers: int = -1,
    continuous_batching: bool = False,
    qaic_config: Dict = None,
    config: Optional[AutoConfig] = None,
):
    kwargs = dict(continuous_batching=continuous_batching, qaic_config=qaic_config)
    if config is None:
        if num_hidden_layers != -1:
            kwargs["num_hidden_layers"] = num_hidden_layers
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model_hf = load_hf_causal_lm_model(model_name, num_hidden_layers, config)
        qeff_model = QEFFAutoModelForCausalLM(model_hf, **kwargs)
    return qeff_model


def set_num_layers_vlm(config: AutoConfig, n_layer: int = -1):
    if n_layer == -1:
        return config
    elif hasattr(config, "model_type") and "mllama" in config.model_type:
        config.text_config.num_hidden_layers = n_layer
        config.text_config.cross_attention_layers = [
            x for x in config.text_config.cross_attention_layers if x < n_layer
        ]
    elif getattr(config, "model_type", None) == "molmo_point":
        config.text_config.num_hidden_layers = n_layer
        config.text_config.rope_scaling = {"rope_type": "linear", "factor": 1.0}
        config.text_config.standardize_rope_params()
        config.vit_config.num_hidden_layers = n_layer
        config.adapter_config.vit_layers = [-1]
    elif hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
        if hasattr(config.vision_config, "vt_num_hidden_layers"):
            config.vision_config.vt_num_hidden_layers = n_layer
        if hasattr(config.vision_config, "depth"):
            config.vision_config.depth = n_layer
        if hasattr(config.vision_config, "deepstack_visual_indexes"):
            # Keep deepstack taps aligned with reduced vision depth for fast-test configs.
            deepstack_idxs = [idx for idx in config.vision_config.deepstack_visual_indexes if idx < n_layer]
            if not deepstack_idxs and n_layer > 0:
                deepstack_idxs = [n_layer - 1]
            config.vision_config.deepstack_visual_indexes = deepstack_idxs
    elif hasattr(config, "llm_config"):
        config.llm_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
        if hasattr(config.vision_config, "depth"):
            config.vision_config.depth = n_layer
    else:
        config.num_hidden_layers = n_layer
    return config


def load_hf_vlm_model(
    model_name: str,
    num_hidden_layers: int = -1,
    config: Optional[AutoConfig] = None,
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config = set_num_layers_vlm(config, num_hidden_layers)
        try:
            model_hf = AutoModelForImageTextToText.from_pretrained(
                config._name_or_path,
                low_cpu_mem_usage=False,
                config=config,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
            )
        except ValueError:
            model_hf = AutoModelForCausalLM.from_pretrained(
                config._name_or_path,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                config=config,
            )
    else:
        try:
            model_hf = AutoModelForImageTextToText.from_config(
                config,
                attn_implementation="eager",
                trust_remote_code=True,
            )
        except ValueError:
            model_hf = AutoModelForCausalLM.from_config(
                config,
                attn_implementation="eager",
                trust_remote_code=True,
            )
        torch_dtype = getattr(model_hf.config, "torch_dtype", None)
        if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
            model_hf = model_hf.to(torch.float32)

    model_hf.eval()
    return model_hf


def load_qeff_vlm_model(
    model_name: str,
    num_hidden_layers: int = -1,
    kv_offload: bool = False,
    continuous_batching: bool = False,
    config: Optional[AutoConfig] = None,
    qaic_config: Optional[dict] = None,
    torch_dtype: Optional[torch.dtype] = torch.float32,
):
    if config is None:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        config = set_num_layers_vlm(config, num_hidden_layers)
        try:
            qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
                model_name,
                low_cpu_mem_usage=False,
                config=config,
                kv_offload=kv_offload,
                continuous_batching=continuous_batching,
                qaic_config=qaic_config,
                torch_dtype=torch_dtype,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
            )
        except ValueError:
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=False,
                config=config,
                kv_offload=kv_offload,
                continuous_batching=continuous_batching,
                qaic_config=qaic_config,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                torch_dtype=torch_dtype,
            )
    else:
        model_hf = load_hf_vlm_model(
            model_name,
            config=config,
        )
        qeff_model = QEFFAutoModelForImageTextToText(
            model_hf,
            kv_offload=kv_offload,
            continuous_batching=continuous_batching,
            trust_remote_code=True,
            qaic_config=qaic_config,
            torch_dtype=torch_dtype,
        )

    return qeff_model


def load_vlm_model(config):
    try:
        model_hf = AutoModelForImageTextToText.from_pretrained(
            config._name_or_path,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            config=config,
            dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )
    except ValueError:
        model_hf = AutoModelForCausalLM.from_pretrained(
            config._name_or_path, low_cpu_mem_usage=False, trust_remote_code=True, config=config, dtype=torch.float32
        )
    model_hf.eval()
    return model_hf


def load_vlm_model_from_config(config):
    try:
        model_hf = AutoModelForImageTextToText.from_config(
            config,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    except ValueError:
        model_hf = AutoModelForCausalLM.from_config(
            config, attn_implementation="eager", trust_remote_code=True, ignore_mismatched_sizes=True
        )
    torch_dtype = getattr(model_hf.config, "torch_dtype", None)
    if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        model_hf = model_hf.to(torch.float32)
    model_hf.eval()
    return model_hf


def load_qeff_model_with_sampler(
    model_name: str,
    is_vlm: bool,
    continuous_batching: bool,
    num_hidden_layers: Optional[int] = -1,
    config: Optional[AutoConfig] = None,
    qaic_config: Optional[dict] = None,
):
    """
    Get a QEfficient model with the sampler transform.

    Args:
        model_name (str): The name of the model to test.
        is_vlm (bool): Whether the model is a vision-language model.
        continuous_batching (bool): Whether to use continuous batching.
        num_hidden_layers (Optional[int]): The number of hidden layers to use.
        config (Optional[AutoConfig]): The configuration to use.
        qaic_config (Optional[dict]): The QAIC configuration to use.
    """
    if is_vlm:
        qeff_model = load_qeff_vlm_model(
            model_name,
            continuous_batching=continuous_batching,
            num_hidden_layers=num_hidden_layers,
            kv_offload=True,
            config=config,
            qaic_config=qaic_config,
        )

    else:
        qeff_model = load_qeff_causal_lm_model(
            model_name,
            continuous_batching=continuous_batching,
            qaic_config=qaic_config,
            config=config,
            num_hidden_layers=num_hidden_layers,
        )

    return qeff_model


def get_text_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    elif hasattr(config, "llm_config"):
        return config.llm_config
    return config


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
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
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

    # Process the input messages to generate prompt for the model.
    def get_prompt(self, messages) -> str:
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
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            query = question[idx]
            if history is None and pixel_values is not None and "<image>" not in query:
                query = "<image>\n" + query
            template = messages.copy()
            template.append([roles[0], query])
            template.append([roles[1], None])
            query = self.get_prompt(template)

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)
        return queries


class ModelConfig:
    """
    Contains all the model types which are not default model like quantized models, external models, swiftkv models etc,.
    """

    QUANTIZED_MODELS = {
        "neuralmagic/Qwen2-0.5B-Instruct-FP8",
        "neuralmagic/Llama-3.2-3B-Instruct-FP8",
        "TheBloke/Llama-2-7B-GPTQ",
        "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
    }

    STANDARD_VLM_MODELS = {
        "llava-hf/llava-1.5-7b-hf",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "tiny-random/gemma-3",
        "tiny-random/mistral-3",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "tiny-random/qwen3-vl-moe",
        "tiny-random/qwen3.5",
        "tiny-random/qwen3.5-moe",
    }

    INTERNVL_MODELS = {
        "OpenGVLab/InternVL2_5-1B",
        "OpenGVLab/InternVL3_5-1B",
    }

    MOLMO_MODELS = {
        "allenai/Molmo-7B-D-0924",
    }
    # FIXME: Debug issue wrt Qwen 3.5, 3.6
    SKIPPED_MODELS = {
        "allenai/Molmo-7B-D-0924",
        "wtang06/mpt-125m-c4",
        "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
        "OpenGVLab/InternVL2_5-1B",
        "OpenGVLab/InternVL3_5-1B",
        "jinaai/jina-embeddings-v2-base-code",
        "hpcai-tech/grok-1",
    }

    DUAL_QPC_MODELS = {
        "moonshotai/Kimi-K2.5",
        "OpenGVLab/InternVL2_5-1B",
        "OpenGVLab/InternVL3_5-1B",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "tiny-random/qwen3-vl-moe",
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-Reranker-2B",
        "Qwen/Qwen3-VL-Reranker-8B",
        "tiny-random/qwen3.5",
        "tiny-random/qwen3.5-moe",
        "tiny-random/gemma-4-dense",
        "tiny-random/gemma-4-moe",
    }

    REPEAT_KV_TEST_MODELS = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "ibm-granite/granite-3.1-1b-a400m-base",
        "Qwen/Qwen2-0.5B",
        "bigcode/starcoder2-3b",
        "meta-llama/Llama-3.2-1B",
        "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
        "TheBloke/Llama-2-7B-GPTQ",
        "neuralmagic/Llama-3.2-3B-Instruct-FP8",
        "ibm-granite/granite-3.1-2b-instruct",
        "llava-hf/llava-1.5-7b-hf",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "tiny-random/mistral-3",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen3-VL-2B-Instruct",
        "tiny-random/qwen3-vl-moe",
        "allenai/Molmo-7B-D-0924",
        "OpenGVLab/InternVL2_5-1B",
        "tiny-random/qwen3.5",
    }

    EXTERNAL_MODELS = {
        "allenai/MolmoPoint-8B": {},
        "hpcai-tech/grok-1": {
            "pytorch_hf_tokens_custom_case": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "pytorch_hf_tokens_normal_case": [
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
                391,
            ],
        }
    }

    SWIFTKV_MODELS = {
        "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
    }

    FULL_MODEL_TESTS_TO_SKIP = {
        "hpcai-tech/grok-1",
    }
