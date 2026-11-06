# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generic image-text-to-text disaggregated example.

This script exports, compiles, and runs a vision-language model as three QPCs:
1. vision encoder QPC
2. language prefill QPC
3. language decode QPC

It can also exercise combined-QPC example flows such as continuous batching,
multi-specialization, and attention blocking.
"""

import argparse
import copy
import importlib.util
import inspect
import json
import os
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.vlm_generation import VisionLanguageGeneration

DEFAULT_PROMPT = "Describe this image."
DEFAULT_IMAGE_URL = "https://picsum.photos/id/237/536/354"
DEFAULT_QWEN_IMAGE_WIDTH = 536
DEFAULT_QWEN_IMAGE_HEIGHT = 354
DEFAULT_IMG_SIZE = 336
DEFAULT_CONTINUOUS_BATCHING_PROMPTS = [
    "Can you describe the image in detail?",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "What colors are predominant in the image?",
]
DEFAULT_MULTISPEC_HEIGHTS = [240, 354, 1024]
DEFAULT_MULTISPEC_WIDTHS = [360, 536, 1024]
DEFAULT_MULTISPEC_NUM_FRAMES = [3, 2, 1]
DEFAULT_MULTISPEC_MM_PROCESSOR_KWARGS = {
    "min_pixels": 4 * 32 * 32,
    "max_pixels": 16384 * 32 * 32,
}
QWEN_MODEL_TYPES = {
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}
KIMI_MODEL_TYPES = {"kimi_k25"}
MOLMO_MODEL_TYPES = {"molmo"}
INTERNVL_MODEL_TYPES = {"internvl_chat", "internvl"}

VISION_OUTPUT_NAMES = (
    "vision_embeds",
    "deepstack_features",
    "pixel_values",
    "cross_attention_states",
)
SEQUENCE_INPUT_NAMES = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "token_type_ids",
    "mm_token_type_ids",
    "cache_position",
}


class _GenericVisionLanguageGeneration(VisionLanguageGeneration):
    def _set_output_buffers(self, batch_size: int = 1, sequence_length: int = 1):
        if self.include_sampler:
            return super()._set_output_buffers(batch_size, sequence_length)

        logits_dtype = np.float32
        logits_binding_idx = self._session.binding_index_map.get("logits")
        if logits_binding_idx is not None:
            logits_dtype = self._session.aic_to_np_dtype_mapping[self._session.bindings[logits_binding_idx].type]
        logits_out_placeholder = np.zeros((batch_size, sequence_length, self._vocab_size), dtype=logits_dtype)
        self._session.set_buffers({"logits": logits_out_placeholder})


def _parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    if not value:
        return []
    value = value.strip()
    if value.startswith("["):
        parsed_value = json.loads(value)
        if not isinstance(parsed_value, list) or not all(isinstance(item, str) for item in parsed_value):
            raise argparse.ArgumentTypeError("Expected a JSON string list.")
        return parsed_value
    separator = "|" if "|" in value else ","
    return [item.strip() for item in value.split(separator) if item.strip()]


def _parse_int_or_list(value: str) -> int | list[int]:
    values = _parse_int_list(value)
    if len(values) == 1:
        return values[0]
    return values


def _parse_json_object(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed_value = json.loads(value)
    if not isinstance(parsed_value, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object.")
    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export, compile, and run a VLM as vision, prefill, and decode QPCs.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--cache-dir", default=os.getenv("HF_HUB_CACHE"), help="Hugging Face cache directory.")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token for gated models.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to pair with the image.")
    parser.add_argument(
        "--prompts",
        type=_parse_str_list,
        default=None,
        help="Prompts for continuous batching. Accepts JSON, comma-separated, or pipe-separated values.",
    )
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt for chat-template models.")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="HTTP(S) URL or local path for the image.")
    parser.add_argument(
        "--image-urls",
        type=_parse_str_list,
        default=None,
        help="Image URLs/paths for continuous batching. Accepts JSON, comma-separated, or pipe-separated values.",
    )
    parser.add_argument(
        "--processor-mode",
        choices=("auto", "qwen", "chat-template", "legacy", "molmo", "internvl", "kimi"),
        default="auto",
    )
    parser.add_argument("--message-image-key", choices=("auto", "image", "url", "image_url", "none"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--text-num-hidden-layers", type=int, default=2)
    parser.add_argument("--vision-num-hidden-layers", type=int, default=2)
    parser.add_argument("--vision-depth", type=int, default=None)
    parser.add_argument(
        "--deepstack-visual-indexes",
        type=_parse_int_list,
        default=None,
        help="Comma-separated deepstack indexes. Defaults to the last retained vision layer when available.",
    )
    parser.add_argument(
        "--num-experts", type=int, default=None, help="Optional MoE expert-count override for tiny tests."
    )
    parser.add_argument(
        "--num-experts-per-token", type=int, default=None, help="Optional MoE top-k expert override for tiny tests."
    )
    parser.add_argument("--torch-dtype", choices=("float16", "float32", "bfloat16", "auto"), default="float16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--compile-dir", type=Path, default=None)
    parser.add_argument("--vision-qpc-path", type=Path, default=None, help="Use a precompiled vision QPC.")
    parser.add_argument("--lang-qpc-path", type=Path, default=None, help="Use a precompiled combined language QPC.")
    parser.add_argument("--prefill-qpc-path", type=Path, default=None, help="Use a precompiled language prefill QPC.")
    parser.add_argument("--decode-qpc-path", type=Path, default=None, help="Use a precompiled language decode QPC.")
    parser.add_argument(
        "--compile-only", action="store_true", help="Stop after exporting and compiling the three QPCs."
    )
    parser.add_argument("--skip-compile", action="store_true", help="Run from the three provided QPC paths.")
    parser.add_argument(
        "--shape-arg-mode", choices=("auto", "img-size", "height-width", "image-height-width"), default="auto"
    )
    parser.add_argument(
        "--img-size", type=int, default=None, help="Square image size specialization for img_size models."
    )
    parser.add_argument(
        "--height", type=_parse_int_or_list, default=None, help="Image height specialization for Qwen-style models."
    )
    parser.add_argument(
        "--width", type=_parse_int_or_list, default=None, help="Image width specialization for Qwen-style models."
    )
    parser.add_argument(
        "--image-height", type=int, default=None, help="Image height specialization for Kimi-style models."
    )
    parser.add_argument(
        "--image-width", type=int, default=None, help="Image width specialization for Kimi-style models."
    )
    parser.add_argument(
        "--resize-image", action="store_true", help="Resize input image to the compile dimensions when possible."
    )
    parser.add_argument("--num-frames", type=_parse_int_or_list, default=None)
    parser.add_argument("--runtime-num-frames", type=int, default=None)
    parser.add_argument(
        "--multi-specialization",
        "--multi-specs",
        action="store_true",
        help="Compile multiple image/frame specializations and run with multi_specs=True.",
    )
    parser.add_argument("--max-num-tiles", type=int, default=None)
    parser.add_argument("--max-num-images", type=int, default=None)
    parser.add_argument("--num-patches", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--num-crops", type=int, default=None)
    parser.add_argument("--valid-size", type=int, default=None)
    parser.add_argument("--vision-size", type=int, default=None)
    parser.add_argument("--image-size-height", type=int, default=None)
    parser.add_argument("--image-size-width", type=int, default=None)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--vision-num-devices", type=int, default=1)
    parser.add_argument("--lang-num-devices", type=int, default=1)
    parser.add_argument(
        "--device-ids", type=_parse_int_list, default=None, help="Runtime device IDs for combined QPCs."
    )
    parser.add_argument("--continuous-batching", action="store_true", help="Run the model in continuous batching mode.")
    parser.add_argument(
        "--direct-generation",
        action="store_true",
        help="Compile/run combined vision+language QPCs with the model generate API.",
    )
    parser.add_argument("--skip-vision", action="store_true", help="Compile/run a text-only language QPC.")
    parser.add_argument("--full-batch-size", type=int, default=None, help="Full batch size for continuous batching.")
    parser.add_argument(
        "--split-model-io",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Split model I/O. Defaults to enabled for disaggregated mode and disabled for continuous batching.",
    )
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--mos", type=int, default=1)
    parser.add_argument("--aic-enable-depth-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-onnx-subfunctions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--node-precision-info", default=None)
    parser.add_argument("--mm-processor-kwargs", type=_parse_json_object, default=None)
    parser.add_argument("--enable-blocking", "--blocked", action="store_true")
    parser.add_argument("--blocking-mode", default="hqkv")
    parser.add_argument("--head-block-size", type=int, default=None)
    parser.add_argument("--num-kv-blocks", type=int, default=None)
    parser.add_argument("--num-q-blocks", type=int, default=None)
    parser.add_argument("--num-batch-blocks", type=int, default=None)
    parser.add_argument("--skip-kv", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--moe-prefill-packed-chunk-size", type=int, default=None)
    parser.add_argument("--offload-prefill-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--layerwise-window-size", type=int, default=1)
    parser.add_argument(
        "--qaic-config", type=_parse_json_object, default=None, help="JSON object passed to QEfficient."
    )
    parser.add_argument(
        "--compiler-options", type=_parse_json_object, default=None, help="Extra compile kwargs as JSON."
    )
    parser.add_argument(
        "--kimi-layer-subset",
        action="store_true",
        help="Use tests/utils/load_kimi_utils.py to load a small local Kimi-K2.5 subset.",
    )
    parser.add_argument("--kimi-expert-ids", type=_parse_int_list, default=None)
    parser.add_argument("--data-path-timeout-ms", type=int, default=60_000)
    parser.add_argument("--vision-device-ids", type=_parse_int_list, default=None)
    parser.add_argument("--prefill-device-ids", type=_parse_int_list, default=None)
    parser.add_argument("--decode-device-ids", type=_parse_int_list, default=None)
    parser.add_argument("--stop-on-eos", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    blocking_options = (
        args.head_block_size,
        args.num_kv_blocks,
        args.num_q_blocks,
        args.num_batch_blocks,
        args.skip_kv,
    )
    if any(value is not None for value in blocking_options):
        args.enable_blocking = True
    if args.multi_specialization:
        args.direct_generation = True
        if args.height is None and args.width is None:
            args.height = DEFAULT_MULTISPEC_HEIGHTS
            args.width = DEFAULT_MULTISPEC_WIDTHS
        if args.num_frames is None:
            args.num_frames = DEFAULT_MULTISPEC_NUM_FRAMES
        if args.mm_processor_kwargs is None:
            args.mm_processor_kwargs = DEFAULT_MULTISPEC_MM_PROCESSOR_KWARGS
        args.resize_image = True
    if args.skip_vision:
        args.direct_generation = True

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    if args.prefill_seq_len < 1:
        parser.error("--prefill-seq-len must be >= 1.")
    if args.generation_len < 1:
        parser.error("--generation-len must be >= 1.")
    if args.continuous_batching and args.direct_generation:
        parser.error("--continuous-batching cannot be combined with --direct-generation or --multi-specialization.")
    if args.multi_specialization and args.skip_vision:
        parser.error("--multi-specialization cannot be combined with --skip-vision.")
    if args.continuous_batching:
        if args.full_batch_size is None:
            args.full_batch_size = 4
        if args.full_batch_size < 1:
            parser.error("--full-batch-size must be >= 1.")
        if args.batch_size != 1:
            parser.error("--continuous-batching requires --batch-size 1.")
    if args.runtime_num_frames is not None and args.runtime_num_frames < 1:
        parser.error("--runtime-num-frames must be >= 1.")
    if args.multi_specialization:
        spec_lengths = [len(value) for value in (args.height, args.width, args.num_frames) if isinstance(value, list)]
        if spec_lengths and len(set(spec_lengths)) != 1:
            parser.error("--height, --width, and --num-frames lists must have the same length.")
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be provided together.")
    if (args.image_height is None) != (args.image_width is None):
        parser.error("--image-height and --image-width must be provided together.")
    if args.skip_compile and (args.continuous_batching or args.direct_generation):
        has_vision_qpc = args.skip_vision or args.vision_qpc_path
        has_lang_qpc = args.lang_qpc_path or args.prefill_qpc_path
        if not (has_vision_qpc and has_lang_qpc):
            parser.error("--skip-compile with combined modes requires --vision-qpc-path and --lang-qpc-path.")
    if (
        args.skip_compile
        and not (args.continuous_batching or args.direct_generation)
        and not (args.vision_qpc_path and args.prefill_qpc_path and args.decode_qpc_path)
    ):
        parser.error("--skip-compile requires --vision-qpc-path, --prefill-qpc-path, and --decode-qpc-path.")
    return args


def _torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def _model_type(config: Any, model_name: str) -> str:
    return str(getattr(config, "model_type", "") or model_name).lower()


def _is_qwen_model(model_type: str) -> bool:
    return model_type in QWEN_MODEL_TYPES or ("qwen" in model_type and "vl" in model_type)


def _is_kimi_model(model_type: str, model_name: str) -> bool:
    model_name = model_name.lower()
    return model_type in KIMI_MODEL_TYPES or "kimi" in model_type or "kimi-k2.5" in model_name


def _is_molmo_model(model_type: str, model_name: str) -> bool:
    model_name = model_name.lower()
    return model_type in MOLMO_MODEL_TYPES or "molmo" in model_type or "molmo" in model_name


def _is_internvl_model(model_type: str, model_name: str) -> bool:
    model_name = model_name.lower()
    return model_type in INTERNVL_MODEL_TYPES or "internvl" in model_type or "internvl" in model_name


def _apply_dtype(config: Any, dtype_name: str):
    if dtype_name == "auto":
        return
    torch_dtype = _torch_dtype(dtype_name)
    for cfg in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if cfg is None:
            continue
        if hasattr(cfg, "dtype"):
            cfg.dtype = dtype_name
        cfg.torch_dtype = torch_dtype


def _set_first_existing_attr(config_obj: Any, names: tuple[str, ...], value: int, label: str) -> list[str]:
    updated = []
    for name in names:
        if hasattr(config_obj, name):
            setattr(config_obj, name, value)
            updated.append(f"{label}.{name}")
    return updated


def _truncate_layer_types(config_obj: Any, num_layers: int, label: str) -> list[str]:
    if not hasattr(config_obj, "layer_types"):
        return []
    layer_types = list(config_obj.layer_types)
    config_obj.layer_types = layer_types[:num_layers]
    return [f"{label}.layer_types"]


def _apply_text_layer_limit(config: Any, num_layers: int | None) -> list[str]:
    if num_layers is None or num_layers < 0:
        return []
    text_config = getattr(config, "text_config", None) or getattr(config, "llm_config", None) or config
    updated = _set_first_existing_attr(
        text_config, ("num_hidden_layers", "n_layer", "num_layers"), num_layers, "text_config"
    )
    updated.extend(_truncate_layer_types(text_config, num_layers, "text_config"))
    if hasattr(text_config, "cross_attention_layers"):
        text_config.cross_attention_layers = [idx for idx in text_config.cross_attention_layers if idx < num_layers]
        updated.append("text_config.cross_attention_layers")
    if hasattr(text_config, "num_kv_shared_layers"):
        text_config.num_kv_shared_layers = min(int(text_config.num_kv_shared_layers), max(0, num_layers - 1))
        updated.append("text_config.num_kv_shared_layers")
    return updated


def _apply_vision_layer_limit(config: Any, num_layers: int | None, vision_depth: int | None) -> list[str]:
    requested_layers = vision_depth if vision_depth is not None else num_layers
    if requested_layers is None or requested_layers < 0:
        return []
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return []
    updated = _set_first_existing_attr(
        vision_config,
        ("num_hidden_layers", "depth", "vt_num_hidden_layers", "num_layers"),
        requested_layers,
        "vision_config",
    )
    return updated


def _apply_deepstack_indexes(config: Any, args: argparse.Namespace) -> list[str]:
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None or not hasattr(vision_config, "deepstack_visual_indexes"):
        return []

    if args.deepstack_visual_indexes is not None:
        vision_config.deepstack_visual_indexes = args.deepstack_visual_indexes
        return ["vision_config.deepstack_visual_indexes"]

    depth = getattr(vision_config, "depth", None) or getattr(vision_config, "num_hidden_layers", None)
    if depth is None:
        return []
    max_valid_idx = max(0, int(depth) - 1)
    existing_indexes = [int(idx) for idx in vision_config.deepstack_visual_indexes]
    clamped_indexes = [idx for idx in existing_indexes if 0 <= idx <= max_valid_idx]
    vision_config.deepstack_visual_indexes = clamped_indexes if clamped_indexes else [max_valid_idx]
    return ["vision_config.deepstack_visual_indexes"]


def _apply_expert_limit(config: Any, args: argparse.Namespace) -> list[str]:
    updated = []
    text_config = getattr(config, "text_config", None) or getattr(config, "llm_config", None) or config
    if args.num_experts is not None:
        updated.extend(
            _set_first_existing_attr(
                text_config,
                ("n_routed_experts", "num_experts", "num_local_experts", "num_routed_experts"),
                args.num_experts,
                "text_config",
            )
        )
    if args.num_experts_per_token is not None:
        updated.extend(
            _set_first_existing_attr(
                text_config,
                ("num_experts_per_tok", "num_experts_per_token", "router_top_k"),
                args.num_experts_per_token,
                "text_config",
            )
        )
    return updated


def _load_kimi_utils():
    utils_path = Path(__file__).resolve().parents[3] / "tests" / "utils" / "load_kimi_utils.py"
    spec = importlib.util.spec_from_file_location("load_kimi_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Kimi helpers from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_config(args: argparse.Namespace):
    config_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.cache_dir:
        config_kwargs["cache_dir"] = args.cache_dir
    if args.hf_token:
        config_kwargs["token"] = args.hf_token

    config = AutoConfig.from_pretrained(args.model_name, **config_kwargs)
    _apply_dtype(config, args.torch_dtype)
    updated = []
    updated.extend(_apply_text_layer_limit(config, args.text_num_hidden_layers))
    updated.extend(_apply_vision_layer_limit(config, args.vision_num_hidden_layers, args.vision_depth))
    updated.extend(_apply_deepstack_indexes(config, args))
    updated.extend(_apply_expert_limit(config, args))
    if updated:
        print("Applied small-layer config overrides: " + ", ".join(updated))
    return config


def _effective_qaic_config(args: argparse.Namespace, config: Any) -> dict[str, Any] | None:
    qaic_config = copy.deepcopy(args.qaic_config) or {}
    model_type = _model_type(config, args.model_name)
    if not qaic_config and _is_kimi_model(model_type, args.model_name):
        qaic_config.update({"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}})
    if args.enable_blocking:
        qaic_config.update({"enable_blocking": True, "blocking_mode": args.blocking_mode})
        optional_blocking_args = {
            "head_block_size": args.head_block_size,
            "num_kv_blocks": args.num_kv_blocks,
            "num_q_blocks": args.num_q_blocks,
            "num_batch_blocks": args.num_batch_blocks,
            "skip_kv": args.skip_kv,
        }
        qaic_config.update({name: value for name, value in optional_blocking_args.items() if value is not None})
    return qaic_config or None


def _load_model(args: argparse.Namespace, config: Any):
    dtype = _torch_dtype(args.torch_dtype)
    qaic_config = _effective_qaic_config(args, config)

    if args.kimi_layer_subset:
        kimi_utils = _load_kimi_utils()
        expert_ids = args.kimi_expert_ids if args.kimi_expert_ids is not None else kimi_utils.LOADED_EXPERT_IDS
        num_experts_per_token = args.num_experts_per_token or kimi_utils.NUM_EXPERTS_PER_TOKEN
        model, tokenizer, processor = kimi_utils.load_kimi_k25_layer_subset_model(
            model_path=Path(args.model_name),
            num_vision_layers=args.vision_num_hidden_layers,
            num_text_layers=args.text_num_hidden_layers,
            loaded_expert_ids=expert_ids,
            num_experts_per_tok=num_experts_per_token,
            dtype=torch.float32 if dtype == "auto" else dtype,
        )
        if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "patch_embed"):
            model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
        qeff_model = QEFFAutoModelForImageTextToText(
            model.eval().to("cpu"),
            kv_offload=True,
            config=model.config,
            torch_dtype=torch.float32 if dtype == "auto" else dtype,
            qaic_config=qaic_config,
            layerwise=args.layerwise,
            continuous_batching=args.continuous_batching,
        )
        return qeff_model, tokenizer, processor

    model_kwargs: dict[str, Any] = {
        "config": config,
        "attn_implementation": args.attn_implementation,
        "kv_offload": True,
        "trust_remote_code": args.trust_remote_code,
        "ignore_mismatched_sizes": True,
        "qaic_config": qaic_config,
        "layerwise": args.layerwise,
        "continuous_batching": args.continuous_batching,
    }
    if args.cache_dir:
        model_kwargs["cache_dir"] = args.cache_dir
    if args.hf_token:
        model_kwargs["token"] = args.hf_token
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    try:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)
    except ValueError:
        fallback_kwargs = model_kwargs.copy()
        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
            transformers.PreTrainedModel.all_tied_weights_keys = {}
        if _is_molmo_model(_model_type(config, args.model_name), args.model_name):
            molmo_cls = get_class_from_dynamic_module("modeling_molmo.MolmoForCausalLM", args.model_name)
            original_tie_weights = molmo_cls.tie_weights

            def _tie_weights_compat(self, missing_keys=None, recompute_mapping=True):
                return original_tie_weights(self)

            if "missing_keys" not in inspect.signature(original_tie_weights).parameters:
                molmo_cls.tie_weights = _tie_weights_compat
        model_type = _model_type(config, args.model_name)
        if _is_molmo_model(model_type, args.model_name):
            hf_kwargs = fallback_kwargs.copy()
            hf_kwargs.pop("kv_offload", None)
            hf_kwargs.pop("qaic_config", None)
            hf_kwargs.pop("layerwise", None)
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, **hf_kwargs)
            if dtype != "auto":
                for cfg in (
                    getattr(hf_model, "config", None),
                    getattr(getattr(hf_model, "model", None), "config", None),
                ):
                    if cfg is not None:
                        cfg.torch_dtype = dtype
            qeff_model = QEFFAutoModelForImageTextToText(
                hf_model,
                kv_offload=True,
                config=hf_model.config,
                torch_dtype=torch.float32 if dtype == "auto" else dtype,
                qaic_config=qaic_config,
                layerwise=args.layerwise,
                pretrained_model_name_or_path=args.model_name,
            )
        else:
            if _is_internvl_model(model_type, args.model_name):
                fallback_kwargs.pop("ignore_mismatched_sizes", None)
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, **fallback_kwargs)

    processor_kwargs = {"trust_remote_code": args.trust_remote_code}
    tokenizer_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.cache_dir:
        processor_kwargs["cache_dir"] = args.cache_dir
        tokenizer_kwargs["cache_dir"] = args.cache_dir
    if args.hf_token:
        processor_kwargs["token"] = args.hf_token
        tokenizer_kwargs["token"] = args.hf_token

    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)
    tokenizer = getattr(processor, "tokenizer", None) or transformers.AutoTokenizer.from_pretrained(
        args.model_name, **tokenizer_kwargs
    )
    return qeff_model, tokenizer, processor


def _is_url(path_or_url: str) -> bool:
    return urlparse(path_or_url).scheme in {"http", "https"}


def _load_image(path_or_url: str) -> Image.Image:
    if _is_url(path_or_url):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def _select_runtime_specialization_index(args: argparse.Namespace) -> int:
    if (
        args.runtime_num_frames is not None
        and isinstance(args.num_frames, list)
        and args.runtime_num_frames in args.num_frames
    ):
        return args.num_frames.index(args.runtime_num_frames)
    return 0


def _select_runtime_value(value: Any, args: argparse.Namespace) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        index = min(_select_runtime_specialization_index(args), len(value) - 1)
        return value[index]
    return value


def _runtime_num_frames(args: argparse.Namespace) -> int:
    if args.runtime_num_frames is not None:
        return args.runtime_num_frames
    num_frames = _select_runtime_value(args.num_frames, args)
    return int(num_frames) if num_frames is not None else 1


def _compile_image_size(
    args: argparse.Namespace, image: Image.Image
) -> tuple[int | list[int] | None, int | list[int] | None]:
    height = args.height if args.height is not None else args.image_height
    width = args.width if args.width is not None else args.image_width
    if height is None and width is None:
        return image.height, image.width
    return height, width


def _runtime_image_size(args: argparse.Namespace, image: Image.Image) -> tuple[int | None, int | None]:
    height, width = _compile_image_size(args, image)
    height = _select_runtime_value(height, args)
    width = _select_runtime_value(width, args)
    if height is None and width is None:
        return image.height, image.width
    return height, width


def _maybe_resize_image(image: Image.Image, args: argparse.Namespace) -> Image.Image:
    if not args.resize_image:
        return image
    height, width = _runtime_image_size(args, image)
    if height is None or width is None:
        img_size = args.img_size or DEFAULT_IMG_SIZE
        return image.resize((img_size, img_size))
    return image.resize((int(width), int(height)))


def _compile_shape_kwargs(config: Any, args: argparse.Namespace, image: Image.Image) -> dict[str, Any]:
    model_type = _model_type(config, args.model_name)
    mode = args.shape_arg_mode
    if mode == "auto":
        if _is_kimi_model(model_type, args.model_name):
            mode = "image-height-width"
        elif _is_qwen_model(model_type):
            mode = "height-width"
        else:
            mode = "img-size"

    shape_kwargs: dict[str, Any] = {}
    height, width = _compile_image_size(args, image)
    if mode == "height-width":
        shape_kwargs.update({"height": height or DEFAULT_QWEN_IMAGE_HEIGHT, "width": width or DEFAULT_QWEN_IMAGE_WIDTH})
    elif mode == "image-height-width":
        shape_kwargs.update({"image_height": height or image.height, "image_width": width or image.width})
    elif args.img_size is not None:
        shape_kwargs["img_size"] = args.img_size

    optional_kwargs = {
        "num_frames": args.num_frames,
        "max_num_tiles": args.max_num_tiles,
        "max_num_images": args.max_num_images,
        "num_patches": args.num_patches,
        "num_images": args.num_images,
        "num_crops": args.num_crops,
        "valid_size": args.valid_size,
        "vision_size": args.vision_size,
        "image_size_height": args.image_size_height,
        "image_size_width": args.image_size_width,
    }
    shape_kwargs.update({name: value for name, value in optional_kwargs.items() if value is not None})
    return shape_kwargs


def _common_compile_kwargs(config: Any, args: argparse.Namespace, image: Image.Image) -> dict[str, Any]:
    compile_kwargs = {
        "batch_size": args.batch_size,
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "mxfp6_matmul": args.mxfp6_matmul,
        "mxint8_kv_cache": args.mxint8_kv_cache,
        "split_model_io": args.split_model_io if args.split_model_io is not None else not args.continuous_batching,
        "mos": args.mos,
        "aic_enable_depth_first": args.aic_enable_depth_first,
        "use_onnx_subfunctions": (
            args.use_onnx_subfunctions
            if args.use_onnx_subfunctions is not None
            else not (args.continuous_batching or args.multi_specialization or args.enable_blocking)
        ),
        "layerwise": args.layerwise,
        "layerwise_window_size": args.layerwise_window_size,
    }
    compile_kwargs.update(_compile_shape_kwargs(config, args, image))
    qaic_config = _effective_qaic_config(args, config)
    if qaic_config is not None:
        compile_kwargs["qaic_config"] = qaic_config
    if args.compile_dir is not None:
        compile_kwargs["compile_dir"] = str(args.compile_dir)
    if args.node_precision_info is not None:
        compile_kwargs["node_precision_info"] = args.node_precision_info
    if args.mm_processor_kwargs is not None:
        compile_kwargs["mm_processor_kwargs"] = args.mm_processor_kwargs
    if args.compiler_options:
        compile_kwargs.update(args.compiler_options)
    if args.continuous_batching:
        compile_kwargs["full_batch_size"] = args.full_batch_size
    return compile_kwargs


def _compile_disagg_qpcs(qeff_model: Any, config: Any, args: argparse.Namespace, image: Image.Image):
    if args.skip_compile:
        return (
            {"vision_qpc_path": str(args.vision_qpc_path)},
            {"lang_prefill_qpc_path": str(args.prefill_qpc_path)},
            {"lang_decode_qpc_path": str(args.decode_qpc_path)},
        )

    common_kwargs = _common_compile_kwargs(config, args, image)
    print("Compiling vision QPC...")
    vision_qpc_path = qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        skip_vision=False,
        skip_lang=True,
        num_devices=args.vision_num_devices,
        **common_kwargs,
    )

    print("Compiling decode QPC...")
    decode_kwargs = common_kwargs.copy()
    decode_kwargs.update(
        {
            "prefill_only": False,
            "skip_vision": True,
            "skip_lang": False,
            "num_devices": args.lang_num_devices,
            "offload_pt_weights": False,
        }
    )
    decode_qpc_path = qeff_model.compile(prefill_seq_len=1, **decode_kwargs)

    print("Compiling prefill QPC...")
    prefill_kwargs = common_kwargs.copy()
    prefill_kwargs.update(
        {
            "prefill_only": True,
            "enable_chunking": True,
            "retain_full_kv": True,
            "skip_vision": True,
            "skip_lang": False,
            "num_devices": args.lang_num_devices,
            "offload_pt_weights": args.offload_prefill_weights,
        }
    )
    if args.moe_prefill_packed_chunk_size is not None:
        prefill_kwargs["moe_prefill_packed_chunk_size"] = args.moe_prefill_packed_chunk_size
    prefill_qpc_path = qeff_model.compile(prefill_seq_len=args.prefill_seq_len, **prefill_kwargs)
    return vision_qpc_path, prefill_qpc_path, decode_qpc_path


def _compile_continuous_batching_qpcs(qeff_model: Any, config: Any, args: argparse.Namespace, image: Image.Image):
    if args.skip_compile:
        return {
            "vision_qpc_path": str(args.vision_qpc_path),
            "lang_qpc_path": str(args.lang_qpc_path or args.prefill_qpc_path),
        }

    common_kwargs = _common_compile_kwargs(config, args, image)
    print("Compiling continuous-batching vision/language QPCs...")
    return qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        num_devices=args.lang_num_devices,
        **common_kwargs,
    )


def _compile_combined_qpcs(qeff_model: Any, config: Any, args: argparse.Namespace, image: Image.Image):
    if args.skip_compile:
        qpc_paths = {"lang_qpc_path": str(args.lang_qpc_path or args.prefill_qpc_path)}
        if not args.skip_vision:
            qpc_paths["vision_qpc_path"] = str(args.vision_qpc_path)
        _set_combined_qpc_paths(qeff_model, qpc_paths)
        return qpc_paths

    common_kwargs = _common_compile_kwargs(config, args, image)
    print("Compiling combined vision/language QPCs...")
    qpc_paths = qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        skip_vision=args.skip_vision,
        num_devices=args.lang_num_devices,
        **common_kwargs,
    )
    _set_combined_qpc_paths(qeff_model, qpc_paths)
    return qpc_paths


def _set_combined_qpc_paths(qeff_model: Any, qpc_paths: Any) -> None:
    if isinstance(qpc_paths, dict):
        vision_qpc_path = qpc_paths.get("vision_qpc_path")
        lang_qpc_path = qpc_paths.get("lang_qpc_path") or qpc_paths.get("lang_prefill_qpc_path")
    elif isinstance(qpc_paths, (list, tuple)):
        vision_qpc_path = qpc_paths[0] if len(qpc_paths) > 1 else None
        lang_qpc_path = qpc_paths[-1]
    else:
        vision_qpc_path = None
        lang_qpc_path = qpc_paths

    if vision_qpc_path is not None and hasattr(qeff_model, "vision_model"):
        qeff_model.vision_model.qpc_path = str(vision_qpc_path)
    if lang_qpc_path is not None and hasattr(qeff_model, "lang_model"):
        qeff_model.lang_model.qpc_path = str(lang_qpc_path)


def _resolve_qpc_path(qpc_paths: Any, preferred_keys: tuple[str, ...], tuple_index: int) -> str:
    if isinstance(qpc_paths, dict):
        for key in preferred_keys:
            if key in qpc_paths:
                return str(qpc_paths[key])
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_paths.keys())}")
    if isinstance(qpc_paths, (list, tuple)):
        return str(qpc_paths[tuple_index])
    return str(qpc_paths)


def _clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {name: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for name, value in inputs.items()}


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_numpy_inputs(inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: _numpy(value) for name, value in inputs.items() if value is not None}


def _session_input_names(session: QAICInferenceSession) -> set[str]:
    input_names = set(session.input_names)
    input_names.update(name.rsplit("/", 1)[-1] for name in session.input_names)
    return input_names


def _binding_index(session: QAICInferenceSession, name: str) -> int | None:
    binding_index = session.binding_index_map.get(name)
    if binding_index is not None:
        return binding_index
    for session_name in session.input_names:
        if session_name.rsplit("/", 1)[-1] == name:
            return session.binding_index_map.get(session_name)
    return None


def _cast_for_session(session: QAICInferenceSession, name: str, value: np.ndarray) -> np.ndarray:
    binding_index = _binding_index(session, name)
    if binding_index is None:
        return value
    expected_shape = tuple(int(dim) for dim in session.bindings[binding_index].dims)
    if value.shape != expected_shape and value.ndim + 1 == len(expected_shape) and expected_shape[0] == 1:
        value = np.expand_dims(value, axis=0)
    if value.shape != expected_shape and value.ndim == len(expected_shape):
        if all(actual <= expected for actual, expected in zip(value.shape, expected_shape)):
            padded_value = np.zeros(expected_shape, dtype=value.dtype)
            slices = tuple(slice(0, actual) for actual in value.shape)
            padded_value[slices] = value
            value = padded_value
    dtype = session.aic_to_np_dtype_mapping[session.bindings[binding_index].type]
    return value.astype(dtype, copy=False)


def _filter_session_inputs(session: QAICInferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = _session_input_names(session)
    return {name: _cast_for_session(session, name, value) for name, value in inputs.items() if name in input_names}


def _output_value(outputs: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    if name in outputs:
        return outputs[name]
    for output_name, value in outputs.items():
        if output_name.rsplit("/", 1)[-1] == name:
            return value
    return None


def _update_retained_states(target_inputs: dict[str, np.ndarray], source_outputs: dict[str, np.ndarray]):
    for output_name, value in source_outputs.items():
        output_basename = output_name.rsplit("/", 1)[-1]
        if output_basename.endswith("_RetainedState"):
            target_inputs[output_basename.removesuffix("_RetainedState")] = value


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 3:
        next_token_ids = logits[:, -1, :].argmax(axis=-1)
    elif logits.ndim == 2:
        next_token_ids = logits.argmax(axis=-1)
    else:
        next_token_ids = np.asarray([logits.argmax()])
    return next_token_ids.astype(np.int64).reshape(-1, 1)


def _pad_sequence_inputs(inputs: dict[str, np.ndarray], prefill_seq_len: int, pad_token_id: int) -> tuple[int, int]:
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len
    pad_width = padded_len - input_ids_length
    inputs["input_ids"] = np.pad(inputs["input_ids"], ((0, 0), (0, pad_width)), constant_values=pad_token_id)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = np.ones_like(inputs["input_ids"], dtype=np.int64)
    else:
        inputs["attention_mask"] = np.pad(inputs["attention_mask"], ((0, 0), (0, pad_width)), constant_values=0)
    return num_chunks, padded_len


def _add_kimi_grid_shapes(inputs: dict[str, np.ndarray], vision_inputs: dict[str, np.ndarray], vision_session):
    session_inputs = _session_input_names(vision_session)
    if "h_shape" not in session_inputs and "w_shape" not in session_inputs:
        return
    grid_thw = inputs.get("grid_thws")
    if grid_thw is None:
        grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        return
    flat_grid = np.asarray(grid_thw).reshape(-1, 3).astype(np.int64)
    h = int(flat_grid[0, 1])
    w = int(flat_grid[0, 2])
    vision_inputs.setdefault("h_shape", np.ones((h,), dtype=np.int64))
    vision_inputs.setdefault("w_shape", np.ones((w,), dtype=np.int64))


def _split_vision_inputs(inputs: dict[str, np.ndarray], vision_session) -> dict[str, np.ndarray]:
    vision_inputs = _filter_session_inputs(vision_session, inputs)
    _add_kimi_grid_shapes(inputs, vision_inputs, vision_session)
    return _filter_session_inputs(vision_session, vision_inputs)


def _prepare_language_inputs(
    inputs: dict[str, np.ndarray],
    vision_inputs: dict[str, np.ndarray],
    vision_outputs: dict[str, np.ndarray],
    prefill_session,
    padded_len: int,
) -> dict[str, np.ndarray]:
    vision_input_names = set(vision_inputs)
    lang_inputs = {name: value for name, value in inputs.items() if name not in vision_input_names}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        attention_mask = lang_inputs.pop("attention_mask")
        lang_inputs["position_ids"] = np.where(attention_mask > 0, np.arange(padded_len), -1).astype(np.int64)

    prefill_input_names = _session_input_names(prefill_session)
    if "image_idx" in prefill_input_names and "image_idx" not in lang_inputs:
        lang_inputs["image_idx"] = np.zeros((inputs["input_ids"].shape[0], 1), dtype=np.int64)

    for output_name in VISION_OUTPUT_NAMES:
        value = _output_value(vision_outputs, output_name)
        if value is not None:
            lang_inputs[output_name] = value
    return lang_inputs


def _slice_prefill_inputs(
    lang_inputs: dict[str, np.ndarray], start: int, end: int, padded_len: int
) -> dict[str, np.ndarray]:
    chunk_inputs = {}
    for name, value in lang_inputs.items():
        if name in SEQUENCE_INPUT_NAMES or (value.ndim >= 2 and value.shape[-1] == padded_len):
            chunk_inputs[name] = value[..., start:end]
        else:
            chunk_inputs[name] = value
    return chunk_inputs


def _prepare_inputs_for_generation(qeff_model: Any, inputs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model = getattr(qeff_model, "model", None)
    prepare_fn = getattr(model, "prepare_inputs_for_generation", None)
    if prepare_fn is None:
        return inputs
    try:
        return prepare_fn(inputs=inputs, prefill_seq_len=args.prefill_seq_len, batch_size=args.batch_size)
    except TypeError:
        return inputs


def _expand_to_min_length(values: list[str], min_length: int, label: str) -> list[str]:
    if not values:
        raise ValueError(f"At least one {label} is required.")
    if len(values) >= min_length:
        return values
    repeats = -(min_length // -len(values))
    return (values * repeats)[:min_length]


def _continuous_batch_items(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    prompts = args.prompts
    if prompts is None:
        prompts = [args.prompt] if args.prompt != DEFAULT_PROMPT else DEFAULT_CONTINUOUS_BATCHING_PROMPTS
    image_urls = args.image_urls if args.image_urls is not None else [args.image_url]

    prompts = _expand_to_min_length(prompts, args.full_batch_size, "prompt")
    if len(image_urls) == 1 and len(prompts) > 1:
        image_urls = image_urls * len(prompts)
    else:
        image_urls = _expand_to_min_length(image_urls, args.full_batch_size, "image URL/path")
    if len(image_urls) != len(prompts):
        raise ValueError(f"Number of image URLs/paths ({len(image_urls)}) must match prompts ({len(prompts)}).")
    return image_urls, prompts


def _continuous_batch_device_ids(args: argparse.Namespace) -> list[int] | None:
    return args.device_ids or args.prefill_device_ids or args.decode_device_ids or args.vision_device_ids


def _run_continuous_batching_generation(
    qeff_model: Any,
    tokenizer: Any,
    processor: Any,
    vision_qpc: str,
    lang_qpc: str,
    image: Image.Image,
    args: argparse.Namespace,
):
    height, width = _compile_image_size(args, image)
    if not isinstance(height, int):
        height = None
    if not isinstance(width, int):
        width = None

    generator = _GenericVisionLanguageGeneration(
        qeff_model=qeff_model,
        tokenizer=tokenizer,
        processor=processor,
        lang_qpc_path=lang_qpc,
        vision_qpc_path=vision_qpc,
        device_id=_continuous_batch_device_ids(args),
        ctx_len=args.ctx_len,
        full_batch_size=args.full_batch_size,
        image_height=height,
        image_width=width,
    )
    try:
        image_urls, prompts = _continuous_batch_items(args)
        print(f"Running continuous batching for {len(prompts)} prompt(s), full_batch_size={args.full_batch_size}...")
        return generator.generate(
            images=image_urls,
            prompts=prompts,
            generation_len=args.generation_len,
            stream=False,
        )
    finally:
        for session_name in ("_session", "_vision_session"):
            session = getattr(generator, session_name, None)
            if session is not None and getattr(session, "is_active", False):
                session.deactivate()


def _batch_items(args: argparse.Namespace, batch_size: int) -> tuple[list[str], list[str]]:
    prompts = args.prompts if args.prompts is not None else [args.prompt]
    image_urls = args.image_urls if args.image_urls is not None else [args.image_url]
    prompts = _expand_to_min_length(prompts, batch_size, "prompt")
    if len(image_urls) == 1 and len(prompts) > 1:
        image_urls = image_urls * len(prompts)
    else:
        image_urls = _expand_to_min_length(image_urls, batch_size, "image URL/path")
    if len(image_urls) != len(prompts):
        raise ValueError(f"Number of image URLs/paths ({len(image_urls)}) must match prompts ({len(prompts)}).")
    return image_urls, prompts


def _run_combined_generation(
    qeff_model: Any,
    tokenizer: Any,
    processor: Any,
    inputs: dict[str, Any] | None,
    vision_qpc: str | None,
    lang_qpc: str,
    image: Image.Image,
    args: argparse.Namespace,
):
    if args.skip_vision:
        print("Running text-only combined QPC...")
        return qeff_model.generate(
            inputs=inputs,
            device_ids=args.device_ids,
            generation_len=args.generation_len,
        )

    height, width = _runtime_image_size(args, image)
    generator = _GenericVisionLanguageGeneration(
        qeff_model=qeff_model,
        tokenizer=tokenizer,
        processor=processor,
        lang_qpc_path=lang_qpc,
        vision_qpc_path=vision_qpc,
        device_id=args.device_ids,
        ctx_len=args.ctx_len,
        image_height=height if isinstance(height, int) else None,
        image_width=width if isinstance(width, int) else None,
    )
    try:
        if args.multi_specialization:
            frames = _runtime_num_frames(args)
            print(f"Running multi-specialization generation with num_frames={frames}...")
            return generator.generate(
                images=[],
                prompts=[],
                inputs=inputs,
                generation_len=args.generation_len,
                multi_specs=True,
                num_frames=frames,
                stream=False,
            )

        image_urls, prompts = _batch_items(args, args.batch_size)
        print(f"Running combined generation for {len(prompts)} prompt(s)...")
        return generator.generate(
            images=image_urls,
            prompts=prompts,
            generation_len=args.generation_len,
            stream=False,
        )
    finally:
        for session_name in ("_session", "_vision_session"):
            session = getattr(generator, session_name, None)
            if session is not None and getattr(session, "is_active", False):
                session.deactivate()


def _run_disagg_generation(
    qeff_model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
    args: argparse.Namespace,
) -> np.ndarray:
    inputs = _prepare_inputs_for_generation(qeff_model, _clone_inputs(inputs), args)
    inputs = _as_numpy_inputs(inputs)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 1
    num_chunks, padded_len = _pad_sequence_inputs(inputs, args.prefill_seq_len, pad_token_id)

    vision_start = perf_counter()
    print("Running vision QPC...")
    vision_inputs = _split_vision_inputs(inputs, vision_session)
    vision_outputs = vision_session.run(vision_inputs)
    vision_time = perf_counter() - vision_start

    lang_inputs = _prepare_language_inputs(inputs, vision_inputs, vision_outputs, prefill_session, padded_len)
    prefill_session.set_buffers(vision_outputs)

    print("Running prefill QPC...")
    prefill_start = perf_counter()
    chunk_inputs = lang_inputs.copy()
    prefill_outputs = None
    for chunk_idx in range(num_chunks):
        start = chunk_idx * args.prefill_seq_len
        end = (chunk_idx + 1) * args.prefill_seq_len
        chunk_inputs.update(_slice_prefill_inputs(lang_inputs, start, end, padded_len))
        prefill_outputs = prefill_session.run(_filter_session_inputs(prefill_session, chunk_inputs))
        _update_retained_states(chunk_inputs, prefill_outputs)
        image_idx_output = _output_value(prefill_outputs, "image_idx_output")
        if image_idx_output is not None:
            chunk_inputs["image_idx"] = image_idx_output.astype(np.int64)

    if prefill_outputs is None:
        raise RuntimeError("QAIC prefill did not execute.")
    print(f"Prefill time, including vision: {perf_counter() - prefill_start + vision_time:.2f} secs")

    generated_ids = [_get_next_token_ids(prefill_outputs["logits"])]
    decode_inputs = chunk_inputs.copy()
    decode_inputs["input_ids"] = generated_ids[-1]
    decode_inputs["position_ids"] = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True).astype(np.int64) + 1
    _update_retained_states(decode_inputs, prefill_outputs)

    print("Running decode QPC...")
    decode_start = perf_counter()
    for _ in range(1, args.generation_len):
        decode_outputs = decode_session.run(_filter_session_inputs(decode_session, decode_inputs))
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        decode_inputs["input_ids"] = generated_ids[-1]
        decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
        image_idx_output = _output_value(decode_outputs, "image_idx_output")
        if image_idx_output is not None:
            decode_inputs["image_idx"] = image_idx_output.astype(np.int64)
        _update_retained_states(decode_inputs, decode_outputs)
        if (
            args.stop_on_eos
            and getattr(tokenizer, "eos_token_id", None) is not None
            and np.all(generated_ids[-1] == tokenizer.eos_token_id)
        ):
            break

    decode_time = perf_counter() - decode_start
    if len(generated_ids) > 1 and decode_time > 0:
        print(f"Decode tok/sec: {(len(generated_ids) - 1) / decode_time:.2f}")
    return np.concatenate(generated_ids, axis=1)


def _build_conversation(args: argparse.Namespace, image: Image.Image, model_type: str) -> list[dict[str, Any]]:
    image_key = args.message_image_key
    if image_key == "auto":
        if _is_qwen_model(model_type):
            image_key = "image"
        elif _is_kimi_model(model_type, args.model_name):
            image_key = "image_url"
        elif _is_url(args.image_url):
            image_key = "url"
        else:
            image_key = "image"

    content = []
    if args.skip_vision:
        content.append({"type": "text", "text": args.prompt})
    else:
        num_images = _runtime_num_frames(args) if args.multi_specialization else 1
        for _ in range(num_images):
            if image_key == "none":
                content.append({"type": "image"})
            else:
                content.append({"type": "image", image_key: args.image_url if image_key == "url" else image})
        content.append({"type": "text", "text": args.prompt})

    conversation = []
    if args.system_prompt:
        conversation.append({"role": "system", "content": [{"type": "text", "text": args.system_prompt}]})
    conversation.append({"role": "user", "content": content})
    return conversation


def _prepare_qwen_inputs(
    processor: AutoProcessor, conversations: list[list[dict[str, Any]]]
) -> dict[str, torch.Tensor]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as error:
        raise ImportError("Install qwen-vl-utils to prepare Qwen VL inputs for this example.") from error

    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in conversations
    ]
    image_inputs, video_inputs = process_vision_info(conversations)
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"))


def _prepare_molmo_inputs(
    processor: AutoProcessor, image: Image.Image, args: argparse.Namespace
) -> dict[str, torch.Tensor]:
    if args.batch_size != 1:
        raise ValueError("The generic Molmo path currently supports --batch-size 1.")
    inputs = processor.process(images=[image], text=args.prompt)
    inputs = {name: value.unsqueeze(0) if torch.is_tensor(value) else value for name, value in inputs.items()}
    inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, dtype=torch.int64)
    valid = inputs["image_input_idx"] > 0
    inputs["valid_idx"] = torch.nonzero(valid.reshape(1, -1))[:, 1].unsqueeze(0)
    inputs["pixel_values"] = inputs.pop("images")
    return inputs


def _prepare_internvl_inputs(
    qeff_model: Any, tokenizer: Any, image: Image.Image, args: argparse.Namespace
) -> dict[str, Any]:
    if args.batch_size != 1:
        raise ValueError("The generic InternVL path currently supports --batch-size 1.")
    from QEfficient.utils.test_utils import InternProcessor

    intern_processor = InternProcessor(qeff_model.model, tokenizer)
    pixel_values = intern_processor.load_image(image, max_num=args.num_patches or 12)
    num_patches_list = [pixel_values.shape[0]]
    question = "<image>\n" + args.prompt
    query = intern_processor(
        pixel_values, [question], [], ("<|im_start|>user\n", "<|im_start|>assistant\n"), num_patches_list
    )
    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding="max_length",
        max_length=args.prefill_seq_len,
        padding_side="right",
    )
    inputs["pixel_values"] = pixel_values
    return dict(inputs)


def _prepare_chat_template_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    conversation: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    conversations = [conversation for _ in range(args.batch_size)]
    try:
        inputs = processor.apply_chat_template(
            conversations if args.batch_size > 1 else conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return dict(inputs)
    except TypeError:
        input_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return dict(processor(images=image, text=input_text, return_tensors="pt"))


def _prepare_kimi_inputs(processor: AutoProcessor, conversation: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(processor(messages=conversation, add_generation_prompt=True, tokenize=False, return_tensors="pt"))


def _prepare_inputs(
    qeff_model: Any,
    processor: AutoProcessor,
    tokenizer: Any,
    config: Any,
    image: Image.Image,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_type = _model_type(config, args.model_name)
    mode = args.processor_mode
    if mode == "auto":
        if _is_qwen_model(model_type):
            mode = "qwen"
        elif _is_kimi_model(model_type, args.model_name):
            mode = "kimi"
        elif _is_molmo_model(model_type, args.model_name):
            mode = "molmo"
        elif _is_internvl_model(model_type, args.model_name):
            mode = "internvl"
        else:
            mode = "chat-template"

    conversation = _build_conversation(args, image, model_type)
    if mode == "qwen":
        inputs = _prepare_qwen_inputs(processor, [conversation for _ in range(args.batch_size)])
    elif mode == "kimi":
        if args.batch_size != 1:
            raise ValueError("The generic Kimi path currently supports --batch-size 1.")
        inputs = _prepare_kimi_inputs(processor, conversation)
    elif mode == "molmo":
        inputs = _prepare_molmo_inputs(processor, image, args)
    elif mode == "internvl":
        inputs = _prepare_internvl_inputs(qeff_model, tokenizer, image, args)
    elif mode == "legacy":
        input_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = dict(processor(images=image, text=input_text, return_tensors="pt"))
    else:
        inputs = _prepare_chat_template_inputs(processor, image, conversation, args)

    target_dtype = getattr(qeff_model.model.config, "torch_dtype", None)
    for name, value in list(inputs.items()):
        if (
            torch.is_tensor(value)
            and torch.is_floating_point(value)
            and target_dtype in {torch.float16, torch.float32, torch.bfloat16}
        ):
            inputs[name] = value.to(target_dtype)
    return inputs


def main():
    args = parse_args()
    config = _build_config(args)
    image = _maybe_resize_image(_load_image(args.image_url), args)
    qeff_model, tokenizer, processor = _load_model(args, config)

    if args.direct_generation:
        qpc_paths = _compile_combined_qpcs(qeff_model, config, args, image)
        vision_qpc = None if args.skip_vision else _resolve_qpc_path(qpc_paths, ("vision_qpc_path",), 0)
        lang_qpc = _resolve_qpc_path(qpc_paths, ("lang_qpc_path", "lang_prefill_qpc_path"), 1)
        if vision_qpc is not None:
            print(f"Vision QPC path: {vision_qpc}")
        print(f"Language QPC path: {lang_qpc}")

        if args.compile_only:
            return

        inputs = None
        if args.skip_vision or args.multi_specialization:
            inputs = _prepare_inputs(qeff_model, processor, tokenizer, config, image, args)
        output = _run_combined_generation(qeff_model, tokenizer, processor, inputs, vision_qpc, lang_qpc, image, args)
        print(output.generated_ids)
        if hasattr(output, "generated_texts"):
            print(output.generated_texts)
        else:
            print(tokenizer.batch_decode(output.generated_ids))
        print(output)
        return

    if args.continuous_batching:
        qpc_paths = _compile_continuous_batching_qpcs(qeff_model, config, args, image)
        vision_qpc = _resolve_qpc_path(qpc_paths, ("vision_qpc_path",), 0)
        lang_qpc = _resolve_qpc_path(qpc_paths, ("lang_qpc_path", "lang_prefill_qpc_path"), 1)
        print(f"Vision QPC path: {vision_qpc}")
        print(f"Language QPC path: {lang_qpc}")

        if args.compile_only:
            return

        output = _run_continuous_batching_generation(
            qeff_model, tokenizer, processor, vision_qpc, lang_qpc, image, args
        )
        print(output.generated_ids)
        print(output.generated_texts)
        print(output)
        return

    vision_qpc_paths, prefill_qpc_paths, decode_qpc_paths = _compile_disagg_qpcs(qeff_model, config, args, image)
    vision_qpc = _resolve_qpc_path(vision_qpc_paths, ("vision_qpc_path",), 0)
    prefill_qpc = _resolve_qpc_path(prefill_qpc_paths, ("lang_prefill_qpc_path", "lang_qpc_path"), 1)
    decode_qpc = _resolve_qpc_path(decode_qpc_paths, ("lang_decode_qpc_path", "lang_qpc_path"), 1)
    print(f"Vision QPC path: {vision_qpc}")
    print(f"Prefill QPC path: {prefill_qpc}")
    print(f"Decode QPC path: {decode_qpc}")

    if args.compile_only:
        return

    inputs = _prepare_inputs(qeff_model, processor, tokenizer, config, image, args)
    sessions = []
    try:
        session_kwargs = {"data_path_timeout_ms": args.data_path_timeout_ms}
        vision_session = QAICInferenceSession(vision_qpc, device_ids=args.vision_device_ids, **session_kwargs)
        sessions.append(vision_session)
        prefill_session = QAICInferenceSession(prefill_qpc, device_ids=args.prefill_device_ids, **session_kwargs)
        sessions.append(prefill_session)
        decode_session = QAICInferenceSession(decode_qpc, device_ids=args.decode_device_ids, **session_kwargs)
        sessions.append(decode_session)
        generated_ids = _run_disagg_generation(
            qeff_model, tokenizer, inputs, vision_session, prefill_session, decode_session, args
        )
    finally:
        for session in sessions:
            session.deactivate()

    print(generated_ids)
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
