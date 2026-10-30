# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free + Compute-Context-Length (CCL) compile/generate tests.
"""

from __future__ import annotations

import inspect
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import onnx
import pytest
import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES

from ._helpers import (
    BATCH_SIZE,
    WEIGHT_FREE_VLM_MODEL_PARAMS,
    skip_on_model_fetch_error,
)

# Mirrors tests/dynamo/test_ccl.py: CCL's specialization validation floors small
# comp_ctx_lengths values up to CCL_MIN_CTX_LEN (1024) then clamps back down to
# ctx_len; with a tiny CTX_LEN, both prefill and decode collapse to the same value
# and the collision-repair step lands on 0, which the compiler rejects. Use
# ctx_len/prefill_seq_len large enough to avoid that collision.
CCL_PREFILL_SEQ_LEN = 32
CCL_CTX_LEN = 128
CCL_LENGTHS = [1024, 2048]
GENERATION_LEN = 2
CCL_IMAGE_HEIGHT = 354
CCL_IMAGE_WIDTH = 536

_VLM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "image_text_model_configs.json"
with _VLM_CONFIG_PATH.open() as config_file:
    _VLM_CONFIGS = {entry["model_type"]: entry for entry in json.load(config_file)["image_text_models"]}


def _supports_ccl(model_type):
    config_model_type = "gemma4" if model_type == "gemma4_moe" else model_type
    return _VLM_CONFIGS[config_model_type].get("supports_ccl", True)


WEIGHT_FREE_CCL_VLM_MODEL_PARAMS = [
    model_param for model_param in WEIGHT_FREE_VLM_MODEL_PARAMS if _supports_ccl(model_param.values[0])
]


def _get_configured_image_size(model_config):
    path_parts = [part for part in urlsplit(model_config["img_url"]).path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[-1].isdigit() and path_parts[-2].isdigit():
        return int(path_parts[-1]), int(path_parts[-2])
    return CCL_IMAGE_HEIGHT, CCL_IMAGE_WIDTH


def _prepare_qwen_inputs(qeff_model, processor, image_url, prompt, model_config, image_height, image_width):
    image_response = requests.get(image_url[0], timeout=30)
    image_response.raise_for_status()
    image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
    image = image.resize((image_width, image_height), Image.Resampling.BICUBIC)
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt[0]}]}]
    prompt_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    qeff_inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    qeff_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in qeff_inputs.items()
    }
    prepared_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=qeff_inputs,
        prefill_seq_len=model_config["prompt_len"],
        batch_size=model_config["batch_size"],
    )
    return {
        key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        for key, value in prepared_inputs.items()
    }


def _get_explicit_ccl_lengths(lang_onnx_path, ccl_lengths):
    language_graph = onnx.load(str(lang_onnx_path), load_external_data=False)
    input_names = {value.name for value in language_graph.graph.input}
    if "comp_ctx_lengths" not in input_names and len(ccl_lengths) > 1:
        return [ccl_lengths[-1]], [ccl_lengths[-1]]
    return ccl_lengths, ccl_lengths


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_CCL_VLM_MODEL_PARAMS)
def test_weight_free_ccl_compile_and_generate(model_type, model_id, tmp_export_dir):
    """Export once, compile twice (normal and with explicit CCL), generate on each QPC."""
    config_model_type = "gemma4" if model_type == "gemma4_moe" else model_type
    model_config = _VLM_CONFIGS[config_model_type]
    image_height, image_width = _get_configured_image_size(model_config)
    prompt = [model_config["text_prompt"]]
    image_url = [model_config["img_url"]]
    generation_len = model_config.get("generation_len", GENERATION_LEN)
    ccl_lengths = model_config.get("comp_ctx_lengths_decode", CCL_LENGTHS)

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_id, config=config, weight_free=True, qaic_config={"ccl_enabled": True}
        )
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    export_paths = qeff_model.export(
        tmp_export_dir / "wf_ccl_export",
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    if not isinstance(export_paths, (list, tuple)) or len(export_paths) < 2:
        raise AssertionError(f"Expected separate vision and language ONNX paths, got {export_paths!r}")
    vision_onnx_path = Path(export_paths[0])
    lang_onnx_path = Path(export_paths[-1])
    assert vision_onnx_path.is_file(), f"Expected vision ONNX file at {vision_onnx_path}"
    assert lang_onnx_path.is_file(), f"Expected language ONNX file at {lang_onnx_path}"
    explicit_ccl_prefill, explicit_ccl_decode = _get_explicit_ccl_lengths(lang_onnx_path, ccl_lengths)

    compile_kwargs = {
        "vision_onnx_path": str(vision_onnx_path),
        "lang_onnx_path": str(lang_onnx_path),
        "img_size": model_config["img_size"],
        "prefill_seq_len": model_config.get("prompt_len", CCL_PREFILL_SEQ_LEN),
        "ctx_len": model_config.get("ctx_len", CCL_CTX_LEN),
        "num_cores": DEFAULT_AIC_NUM_CORES,
        "batch_size": BATCH_SIZE,
        "use_onnx_subfunctions": True,
    }
    get_specializations_signature = inspect.signature(qeff_model.model.get_specializations)
    img_size_parameter = get_specializations_signature.parameters.get("img_size")
    if img_size_parameter is not None and img_size_parameter.default is inspect.Parameter.empty:
        compile_kwargs["img_size"] = model_config["img_size"]
    if {"height", "width"}.issubset(get_specializations_signature.parameters):
        compile_kwargs.update(
            {
                "height": image_height,
                "width": image_width,
            }
        )

    # Compile normally (auto-generated CCL lists; no explicit comp_ctx_lengths).
    qeff_model.compile(
        compile_dir=str(tmp_export_dir / "wf_normal_compile"),
        **compile_kwargs,
    )
    if config_model_type == "qwen3_vl_moe":
        qeff_inputs = _prepare_qwen_inputs(
            qeff_model, processor, image_url, prompt, model_config, image_height, image_width
        )
        output = qeff_model.generate(inputs=qeff_inputs, generation_len=generation_len)
        assert output is not None
        assert output.generated_ids is not None
    else:
        output = qeff_model.generate(
            tokenizer=tokenizer,
            prompts=prompt,
            processor=processor,
            images=image_url,
            image_height=image_height,
            image_width=image_width,
            generation_len=generation_len,
        )
        assert output is not None
        assert output.generated_texts is not None

    # Compile with explicit CCL specializations, reusing the same export.
    qeff_model.compile(
        compile_dir=str(tmp_export_dir / "wf_ccl_compile"),
        comp_ctx_lengths_prefill=explicit_ccl_prefill,
        comp_ctx_lengths_decode=explicit_ccl_decode,
        **compile_kwargs,
    )
    if config_model_type == "qwen3_vl_moe":
        qeff_inputs = _prepare_qwen_inputs(
            qeff_model, processor, image_url, prompt, model_config, image_height, image_width
        )
        output = qeff_model.generate(inputs=qeff_inputs, generation_len=generation_len)
        assert output is not None
        assert output.generated_ids is not None
    else:
        output = qeff_model.generate(
            tokenizer=tokenizer,
            prompts=prompt,
            processor=processor,
            images=image_url,
            image_height=image_height,
            image_width=image_width,
            generation_len=generation_len,
        )
        assert output is not None
        assert output.generated_texts is not None
