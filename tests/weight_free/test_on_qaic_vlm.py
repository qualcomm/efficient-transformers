# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free VLM tests on QAIC.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
All tests run with weight_free=True (set on the QEff model, which forces
dynamo=True internally) and use_onnx_subfunctions=True.

Covers VLM export, multi-specialization compilation, and HF/QAIC parity.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
)

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils.device_utils import get_available_device_id, get_qaic_mdp_device_groups
from QEfficient.utils.test_utils import load_vlm_model_from_config

from ._helpers import (
    WEIGHT_FREE_BLOCKING_MODE_CASES,
    WEIGHT_FREE_VLM_MODEL_PARAMS,
    assert_blocked_kv_ops_for_mode,
    skip_on_model_fetch_error,
)

_VLM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "image_text_model_configs.json"
with _VLM_CONFIG_PATH.open() as config_file:
    _VLM_MODEL_CONFIGS = {entry["model_type"]: entry for entry in json.load(config_file)["image_text_models"]}

WEIGHT_FREE_VLM_QAIC_MODEL_PARAMS = [
    pytest.param("qwen2_5_vl", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen2-5-vl"), id="qwen2.5-vl"),
    pytest.param("qwen3_vl", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen3-vl"), id="qwen3-vl"),
    pytest.param("qwen3_vl_moe", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen3-vl-moe"), id="qwen3-vl-moe"),
    pytest.param("qwen3_5", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen3-5"), id="qwen3.5"),
    pytest.param("qwen3_5_moe", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen3-5-moe"), id="qwen3.5-moe"),
    pytest.param("gemma4", marks=pytest.mark.xdist_group(name="qaic-vlm-gemma4-dense"), id="gemma4"),
]
WEIGHT_FREE_VLM_QAIC_MODEL_IDS = {
    "qwen2_5_vl": "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
    "qwen3_vl": "tiny-random/qwen3-vl",
    "qwen3_vl_moe": "tiny-random/qwen3-vl-moe",
    "qwen3_5": "tiny-random/qwen3.5",
    "qwen3_5_moe": "tiny-random/qwen3.5-moe",
    "gemma4": "tiny-random/gemma-4-dense",
}
VLM_GENERATION_LEN = 2
VLM_PREFILL_SEQ_LEN = 64
VLM_CTX_LEN = 512
VLM_IMAGE_HEIGHT = 354
VLM_IMAGE_WIDTH = 536
VLM_NUM_CORES = 4
VLM_PROMPT = "Describe this image."
VLM_HEAD_BLOCKING_NUM_DEVICES = 4
VLM_DECODE_BLOCKING_GENERATION_LEN = 10
VLM_DECODE_BLOCKING_KEYS = ("h", "q", "kv", "qkv", "hqkv", "bhqkv", "kv_headpar")
VLM_DECODE_BLOCKING_MODEL_TYPES = ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "gemma4")


def _vlm_decode_blocking_marks(model_type: str, blocking_key: str):
    marks = [pytest.mark.xdist_group(name=f"qaic-vlm-decode-blocking-{model_type}")]
    if model_type == "gemma4" and blocking_key in {"h", "q"}:
        marks.append(
            pytest.mark.skip(
                reason=f"Gemma4 single-QPC {blocking_key}-blocking has compilation issues due to retained-state shape mismatch"
            )
        )
    return marks


VLM_DECODE_BLOCKING_QAIC_CASES = [
    pytest.param(
        model_type,
        blocking_key,
        marks=_vlm_decode_blocking_marks(model_type, blocking_key),
        id=f"{model_type.replace('_', '-')}-{blocking_key}",
    )
    for model_type in VLM_DECODE_BLOCKING_MODEL_TYPES
    for blocking_key in VLM_DECODE_BLOCKING_KEYS
]


def _vlm_blocking_runtime_options(model_type: str, blocking_key: str) -> tuple[dict, list[int] | None]:
    if model_type == "gemma4":
        device_ids = get_available_device_id()
        if device_ids is None:
            pytest.skip("No available QAIC device")
        return {"num_devices": 1}, device_ids[:1]

    if blocking_key == "h":
        device_groups = get_qaic_mdp_device_groups(devices_per_group=VLM_HEAD_BLOCKING_NUM_DEVICES)
        if not device_groups:
            pytest.skip(f"No available QAIC MDP device group with {VLM_HEAD_BLOCKING_NUM_DEVICES} devices")
        return {"num_devices": VLM_HEAD_BLOCKING_NUM_DEVICES, "user_tiled": True}, device_groups[0]

    device_ids = get_available_device_id()
    if device_ids is None:
        pytest.skip("No available QAIC device")
    return {}, device_ids


def _build_vlm_test_checkpoint(
    model_type: str,
    checkpoint_dir: Path,
    model_id: str | None = None,
    *,
    force_full_attention: bool = False,
):
    model_config = _VLM_MODEL_CONFIGS[model_type]
    if model_id is None:
        config = AutoConfig.for_model(
            model_type,
            trust_remote_code=True,
            **model_config.get("additional_params", {}),
        )
        config.name_or_path = model_config["model_name"]
    else:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.name_or_path = model_id
        if model_type == "qwen3_5" and hasattr(config.text_config, "layer_types"):
            config.text_config.layer_types[-1] = "full_attention"
        if force_full_attention and hasattr(config.text_config, "layer_types"):
            config.text_config.layer_types = ["full_attention"] * config.text_config.num_hidden_layers
        for nested_config in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
            if nested_config is not None:
                nested_config.dtype = "float32"
                nested_config.torch_dtype = torch.float32
    torch.manual_seed(42)
    model = load_vlm_model_from_config(config)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    return model, model_config


def _prepare_vlm_test_inputs(
    processor,
    model,
    qeff_model,
    model_config,
    image_size=(536, 354),
    skip_vision=False,
):
    if skip_vision:
        hf_inputs = processor(text=model_config["text_prompt"], return_tensors="pt")
        qeff_inputs = {
            key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in hf_inputs.items()
        }
        if qeff_model is not None:
            qeff_inputs = qeff_model.prepare_inputs_for_generation(
                inputs=qeff_inputs,
                prefill_seq_len=model_config["prompt_len"],
                batch_size=model_config["batch_size"],
            )
        return hf_inputs, qeff_inputs

    image_width, image_height = image_size
    image = Image.new("RGB", image_size, color=(32, 96, 160))
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": model_config["text_prompt"]},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    pixel_count = image_width * image_height
    hf_inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        min_pixels=pixel_count,
        max_pixels=pixel_count,
    )
    qeff_inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in hf_inputs.items()}
    if qeff_model is not None:
        config_model_type = getattr(qeff_model.model.config, "model_type", None)
        if config_model_type != "gemma4":
            qeff_inputs = qeff_model.prepare_inputs_for_generation(
                inputs=qeff_inputs,
                prefill_seq_len=model_config["prompt_len"],
                batch_size=model_config["batch_size"],
            )
    if "image_grid_thw" in qeff_inputs and qeff_inputs["image_grid_thw"].ndim == 2:
        grid = qeff_inputs["image_grid_thw"][0]
        time, grid_height, grid_width = (int(value.item()) for value in grid)
        qeff_inputs["image_grid_thw"] = torch.zeros((1, time, grid_height, grid_width), dtype=grid.dtype)
    if "pixel_values" in hf_inputs:
        hf_inputs["pixel_values"] = hf_inputs["pixel_values"].to(torch.float32)
    if "pixel_values" in qeff_inputs:
        qeff_inputs["pixel_values"] = qeff_inputs["pixel_values"].to(torch.float32)
    return hf_inputs, qeff_inputs


def _get_vlm_specialization_kwargs(model, model_config, heights, widths, num_frames):
    parameters = inspect.signature(model.get_specializations).parameters
    kwargs = {}
    if "img_size" in parameters:
        kwargs["img_size"] = model_config.get("img_size")
    if "height" in parameters and "width" in parameters:
        kwargs.update(
            {
                "height": heights,
                "width": widths,
                "num_frames": num_frames,
            }
        )
    return kwargs


def _get_vlm_compile_kwargs(
    model,
    model_config,
    image_size,
    heights=None,
    widths=None,
    num_frames=None,
    min_pixels=None,
    max_pixels=None,
):
    kwargs = {"img_size": model_config.get("img_size")}
    specialization_kwargs = _get_vlm_specialization_kwargs(
        model,
        model_config,
        heights if heights is not None else image_size[1],
        widths if widths is not None else image_size[0],
        num_frames if num_frames is not None else 1,
    )
    if "height" in specialization_kwargs:
        kwargs.update(
            {
                "height": specialization_kwargs["height"],
                "width": specialization_kwargs["width"],
                "num_frames": specialization_kwargs["num_frames"],
                "mm_processor_kwargs": {
                    "min_pixels": min_pixels if min_pixels is not None else image_size[0] * image_size[1],
                    "max_pixels": max_pixels if max_pixels is not None else image_size[0] * image_size[1],
                },
            }
        )
    return kwargs


def _assert_vlm_specialization_structure(specializations, expected_vision_specs):
    assert set(specializations) >= {"vision", "lang"}
    vision_specs = specializations["vision"]
    assert len(vision_specs) == expected_vision_specs
    assert vision_specs

    if all("grid_height" in spec and "grid_width" in spec for spec in vision_specs):
        assert len({(spec["grid_height"], spec["grid_width"]) for spec in vision_specs}) == expected_vision_specs
    else:
        assert all("max_patches" in spec for spec in vision_specs)
        assert all("vision_batch_size" in spec for spec in vision_specs)


def _assert_vlm_weight_free_graph(model_type: str, language_onnx: Path) -> None:
    graph = onnx.load(str(language_onnx), load_external_data=False)
    function_names = [function.name for function in graph.functions]
    expected_functions = {
        "qwen2_5_vl": ("QEffQwen2_5_VLDecoderLayer",),
        "qwen3_vl": ("QEffQwen3VLTextDecoderLayer",),
        "qwen3_vl_moe": ("QEffQwen3VLMoeTextDecoderLayer",),
        "qwen3_5": (
            "QEffQwen3_5LinearAttentionDecoderLayer",
            "QEffQwen3_5FullAttentionDecoderLayer",
        ),
        "qwen3_5_moe": (
            "QEffQwen3_5MoeLinearAttentionDecoderLayer",
            "QEffQwen3_5MoeFullAttentionDecoderLayer",
        ),
        "gemma4": ("QEffGemma4DecoderLayer"),
    }[model_type]
    missing_functions = [
        expected for expected in expected_functions if not any(expected in name for name in function_names)
    ]
    assert not missing_functions, f"Missing subfunctions {missing_functions} in {language_onnx}: {function_names}"

    retained_outputs = {output.name for output in graph.graph.output if output.name.endswith("_RetainedState")}
    assert retained_outputs, f"No retained-state outputs found in {language_onnx}"
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        assert any(name.startswith("conv_state.") for name in retained_outputs)
        assert any(name.startswith("recurrent_state.") for name in retained_outputs)
    else:
        assert any(name.startswith("past_key.") for name in retained_outputs)
        assert any(name.startswith("past_value.") for name in retained_outputs)


def _assert_input_trace_length(language_onnx: Path, expected_seq_len: int) -> None:
    graph = onnx.load(str(language_onnx), load_external_data=False)
    input_ids = next(value for value in graph.graph.input if value.name == "input_ids")
    seq_len = input_ids.type.tensor_type.shape.dim[1].dim_value
    assert seq_len == expected_seq_len, f"Expected input trace length {expected_seq_len}, got {seq_len}"


def _assert_weight_spec(onnx_path: Path) -> None:
    weight_spec = onnx_path.with_name("weight_spec.json")
    assert weight_spec.is_file(), f"Missing weight-free spec next to {onnx_path}"


def _update_retained_states(target_inputs, source_outputs, layer_types):
    for layer_idx, layer_type in enumerate(layer_types):
        state_names = (
            (f"past_key.{layer_idx}", f"past_value.{layer_idx}")
            if layer_type == "full_attention"
            else (f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}")
        )
        for state_name in state_names:
            retained_name = f"{state_name}_RetainedState"
            if retained_name in source_outputs:
                target_inputs[state_name] = np.array(source_outputs[retained_name], copy=True)


def _language_layer_types(qeff_model) -> list[str]:
    text_config = getattr(qeff_model.model.config, "text_config", qeff_model.model.config)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    return ["full_attention"] * int(getattr(text_config, "num_hidden_layers"))


def _ensure_position_ids(inputs: dict[str, np.ndarray]) -> None:
    if "position_ids" in inputs:
        return
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        inputs["position_ids"] = np.arange(inputs["input_ids"].shape[-1], dtype=np.int64).reshape(1, -1)
        return
    inputs["position_ids"] = np.where(attention_mask, np.arange(attention_mask.shape[-1]), -1).astype(np.int64)


def _pad_position_ids(position_ids: np.ndarray, pad_len: int) -> np.ndarray:
    pad_width = [(0, 0)] * position_ids.ndim
    pad_width[-1] = (0, pad_len)
    return np.pad(position_ids, pad_width, constant_values=-1)


def _reshape_vision_output_for_language_session(session, name: str, value: np.ndarray) -> np.ndarray:
    """Restore the ranked tensor shape expected by the language QPC input binding."""
    binding_index = session.binding_index_map.get(name)
    if binding_index is None:
        return value

    for allowed_shape in session.allowed_shapes:
        shape = allowed_shape[binding_index][1]
        if len(shape) > 1 and int(np.prod(shape)) == value.size:
            return value.reshape(shape)
    return value


def _set_vision_output_buffers(session, vision_inputs: dict[str, np.ndarray]) -> None:
    """Select output dimensions for the vision specialization matching the input image grid."""
    pixel_values_index = session.binding_index_map["pixel_values"]
    pixel_values_shape = list(vision_inputs["pixel_values"].shape)
    allowed_shapes = session.allowed_shapes or [[(None, list(binding.dims)) for binding in session.bindings]]
    matching_specializations = [
        index
        for index, allowed_shape in enumerate(allowed_shapes)
        if allowed_shape[pixel_values_index][1] == pixel_values_shape
    ]
    if not matching_specializations:
        allowed_pixel_shapes = [allowed_shape[pixel_values_index][1] for allowed_shape in allowed_shapes]
        raise ValueError(
            "No vision specialization matches pixel_values shape "
            f"{pixel_values_shape}; binding_index={pixel_values_index}, "
            f"allowed_shapes={allowed_pixel_shapes}"
        )
    specialization_index = matching_specializations[0]
    buffers = {}
    for output_name in session.output_names:
        binding_index = session.binding_index_map[output_name]
        output_shape = allowed_shapes[specialization_index][binding_index][1]
        if output_shape:
            buffers[output_name] = np.zeros(output_shape, dtype=np.float16)
    session.set_buffers(buffers)


def _run_qwen35_three_qpc(
    qeff_model,
    qeff_inputs,
    qpc_paths,
    layer_types,
    generation_len,
    *,
    decode_device_ids=None,
):
    vision_session = QAICInferenceSession(qpc_paths["vision_qpc_path"])
    prefill_session = None
    decode_session = None
    try:
        inputs = {
            name: np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for name, value in qeff_inputs.items()
        }
        _ensure_position_ids(inputs)
        vision_inputs = {name: inputs[name] for name in vision_session.input_names if name in inputs}
        for name in ("pixel_values", "image_embeds"):
            if name in vision_inputs:
                vision_inputs[name] = vision_inputs[name].astype(np.float16)
        _set_vision_output_buffers(vision_session, vision_inputs)
        vision_outputs = vision_session.run(vision_inputs)
        vision_session.deactivate()
        vision_session = None

        prefill_session = QAICInferenceSession(qpc_paths["lang_prefill_qpc_path"])
        if "vision_embeds" in vision_outputs:
            vision_outputs["vision_embeds"] = _reshape_vision_output_for_language_session(
                prefill_session, "vision_embeds", vision_outputs["vision_embeds"]
            )

        prefill_binding = prefill_session.binding_index_map["input_ids"]
        prefill_seq_len = prefill_session.bindings[prefill_binding].dims[1]
        input_ids = inputs["input_ids"]
        padded_len = int(np.ceil(input_ids.shape[1] / prefill_seq_len) * prefill_seq_len)
        if padded_len != input_ids.shape[1]:
            pad_len = padded_len - input_ids.shape[1]
            pad_token_id = int(input_ids[0, -1])
            inputs["input_ids"] = np.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
            inputs["position_ids"] = _pad_position_ids(inputs["position_ids"], pad_len)

        lang_inputs = {
            name: inputs[name] for name in prefill_session.input_names if name in inputs and name not in vision_inputs
        }
        lang_inputs.pop("attention_mask", None)
        lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        for name, value in vision_outputs.items():
            if name in prefill_session.input_names:
                lang_inputs[name] = value

        prefill_session.set_buffers(vision_outputs)
        chunk_inputs = dict(lang_inputs)
        for chunk_idx in range(padded_len // prefill_seq_len):
            start = chunk_idx * prefill_seq_len
            end = start + prefill_seq_len
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, start:end]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., start:end]
            prefill_outputs = prefill_session.run(chunk_inputs)
            _update_retained_states(chunk_inputs, prefill_outputs, layer_types)
            if "image_idx_output" in prefill_outputs:
                chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"]

        prefill_session.deactivate()
        prefill_session = None

        decode_session = QAICInferenceSession(qpc_paths["lang_decode_qpc_path"], device_ids=decode_device_ids)
        generated_tokens = [np.argmax(prefill_outputs["logits"], axis=-1)[:, -1]]
        decode_inputs = {
            "input_ids": generated_tokens[-1].reshape(1, 1),
            "position_ids": np.max(inputs["position_ids"], axis=-1, keepdims=True) + 1,
        }
        _update_retained_states(decode_inputs, prefill_outputs, layer_types)
        for name in ("image_idx_output", "vision_embeds_RetainedState"):
            if name in prefill_outputs:
                decode_inputs[
                    name.removesuffix("_RetainedState") if name.endswith("_RetainedState") else "image_idx"
                ] = prefill_outputs[name]

        for _ in range(generation_len - 1):
            decode_outputs = decode_session.run(decode_inputs)
            generated_tokens.append(np.argmax(decode_outputs["logits"], axis=-1)[:, -1])
            decode_inputs["input_ids"] = generated_tokens[-1].reshape(1, 1)
            decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
            _update_retained_states(decode_inputs, decode_outputs, layer_types)
            if "image_idx_output" in decode_outputs:
                decode_inputs["image_idx"] = decode_outputs["image_idx_output"]
            if "vision_embeds_RetainedState" in decode_outputs:
                decode_inputs["vision_embeds"] = decode_outputs["vision_embeds_RetainedState"]
        return np.stack(generated_tokens[:generation_len], axis=1)
    finally:
        for session in (vision_session, prefill_session, decode_session):
            if session is not None:
                session.deactivate()


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize("model_type", WEIGHT_FREE_VLM_QAIC_MODEL_PARAMS)
def test_weight_free_vlm_dummy_cpu_export_and_multispec_structure(model_type, tmp_export_dir):
    """Export every supported VLM from a synthesized checkpoint on CPU."""
    model_config = _VLM_MODEL_CONFIGS[model_type]
    checkpoint_dir = tmp_export_dir / f"{model_type}_cpu_checkpoint"
    tiny_model_types = {"qwen2_5_vl", "qwen3_vl"}
    dummy_model_id = WEIGHT_FREE_VLM_QAIC_MODEL_IDS.get(model_type) if model_type in tiny_model_types else None
    model, model_config = _build_vlm_test_checkpoint(model_type, checkpoint_dir, dummy_model_id)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        kv_offload=True,
        weight_free=True,
        dtype=torch.float32,
    )
    skip_vision = model_type == "qwen2_5_vl"
    # Skipped due to an ONNX issue in the vision module.

    export_paths = qeff_model.export(
        tmp_export_dir / f"{model_type}_cpu_export",
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        prefill_seq_len=model_config["prompt_len"],
        ctx_len=model_config["ctx_len"],
        skip_vision=skip_vision,
    )
    language_onnx = Path(export_paths[-1])
    assert language_onnx.is_file()
    _assert_weight_spec(language_onnx)
    _assert_vlm_weight_free_graph(model_type, language_onnx)
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        _assert_input_trace_length(language_onnx, model_config["prompt_len"])
    if skip_vision:
        return

    vision_onnx = Path(export_paths[0])
    assert vision_onnx.is_file()
    _assert_weight_spec(vision_onnx)

    specialization_kwargs = _get_vlm_specialization_kwargs(
        qeff_model.model,
        model_config,
        [224, VLM_IMAGE_HEIGHT],
        [224, VLM_IMAGE_WIDTH],
        [1, 1],
    )
    specializations, _ = qeff_model.model.get_specializations(
        batch_size=model_config["batch_size"],
        prefill_seq_len=model_config["prompt_len"],
        ctx_len=model_config["ctx_len"],
        kv_offload=True,
        **specialization_kwargs,
    )
    expected_vision_specs = 2 if "height" in specialization_kwargs else 1
    _assert_vlm_specialization_structure(specializations, expected_vision_specs)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
def test_weight_free_gemma4_decode_dummy_trace_length(tmp_export_dir):
    """Trace Gemma4 decode with a non-degenerate dummy KV cache."""
    model_config = _VLM_MODEL_CONFIGS["gemma4"]
    checkpoint_dir = tmp_export_dir / "gemma4_decode_cpu_checkpoint"
    _build_vlm_test_checkpoint("gemma4", checkpoint_dir)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        kv_offload=True,
        weight_free=True,
        dtype=torch.float32,
    )
    dummy_inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        prefill_seq_len=1,
        batch_size=constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
    )
    assert dummy_inputs["lang"]["input_ids"].shape[1] == constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
    assert dummy_inputs["lang"]["past_key_values"][0][0].shape[2] == constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN

    export_paths = qeff_model.export(
        tmp_export_dir / "gemma4_decode_cpu_export",
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        prefill_seq_len=1,
        ctx_len=model_config["ctx_len"],
        skip_vision=True,
    )
    language_onnx = Path(export_paths[-1])
    assert language_onnx.is_file()


def _prepare_vlm_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ]
    ]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    processor_kwargs = {"padding": True, "return_tensors": "pt"}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, **processor_kwargs))


def _run_vlm_hf_reference(model, inputs: dict) -> np.ndarray:
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=VLM_GENERATION_LEN,
            min_new_tokens=VLM_GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_VLM_MODEL_PARAMS)
def test_weight_free_vlm_combined_compile(model_type, model_id, tmp_export_dir):
    """Run a weight-free dual-QPC VLM and compare generated tokens with HF."""
    checkpoint_dir = tmp_export_dir / f"{model_type}_combined_checkpoint"
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        hf_model, model_config = _build_vlm_test_checkpoint(model_type, checkpoint_dir, model_id)
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            str(checkpoint_dir),
            kv_offload=True,
            weight_free=True,
            dtype=torch.float32,
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    image = Image.new("RGB", (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT), color=(127, 127, 127))
    image_size = (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT)
    compile_image_kwargs = _get_vlm_compile_kwargs(qeff_model.model, model_config, image_size)
    mm_processor_kwargs = compile_image_kwargs.get("mm_processor_kwargs", {})
    hf_inputs = _prepare_vlm_inputs(
        processor,
        image,
        min_pixels=mm_processor_kwargs.get("min_pixels"),
        max_pixels=mm_processor_kwargs.get("max_pixels"),
    )
    hf_tokens = _run_vlm_hf_reference(hf_model, hf_inputs)
    qeff_inputs = _prepare_vlm_inputs(
        processor,
        image,
        min_pixels=mm_processor_kwargs.get("min_pixels"),
        max_pixels=mm_processor_kwargs.get("max_pixels"),
    )
    if "height" in inspect.signature(qeff_model.model.get_specializations).parameters:
        qeff_inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=qeff_inputs,
            prefill_seq_len=VLM_PREFILL_SEQ_LEN,
            batch_size=1,
        )
    compiler_config = {
        "batch_size": 1,
        "prefill_seq_len": VLM_PREFILL_SEQ_LEN,
        "ctx_len": VLM_CTX_LEN,
        "num_cores": VLM_NUM_CORES,
        "num_devices": 1,
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
        "retain_full_kv": True,
    }
    compiler_config.update(compile_image_kwargs)
    qpc_paths = qeff_model.compile(compile_dir=str(tmp_export_dir / f"{model_type}_combined"), **compiler_config)

    assert qpc_paths.get("vision_qpc_path")
    assert qpc_paths.get("lang_qpc_path")

    qaic_output = qeff_model.generate(inputs=qeff_inputs, generation_len=VLM_GENERATION_LEN)
    assert qaic_output is not None, "Weight-free VLM QAIC generate returned None"
    assert qaic_output.generated_ids is not None, "Weight-free VLM QAIC generate returned no token IDs"

    qaic_tokens = qaic_output.generated_ids[:, :VLM_GENERATION_LEN]
    assert qaic_tokens.shape == hf_tokens.shape == (1, VLM_GENERATION_LEN)
    assert np.array_equal(qaic_tokens, hf_tokens), (
        f"Weight-free VLM QAIC/HF parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type", WEIGHT_FREE_VLM_QAIC_MODEL_PARAMS)
def test_weight_free_vlm_hf_qaic_parity(model_type, tmp_export_dir):
    """Validate one compact weight-free VLM export, compile, and HF/QAIC parity run."""
    model_config = _VLM_MODEL_CONFIGS[model_type]
    model_id = WEIGHT_FREE_VLM_QAIC_MODEL_IDS[model_type]
    checkpoint_dir = tmp_export_dir / f"{model_type}_checkpoint"

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        hf_model, model_config = _build_vlm_test_checkpoint(model_type, checkpoint_dir, model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    image_size = (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT)
    skip_vision = model_type == "qwen2_5_vl"
    compile_prefill_seq_len = model_config["prompt_len"]
    hf_inputs, _ = _prepare_vlm_test_inputs(
        processor,
        hf_model,
        None,
        model_config,
        image_size,
        skip_vision=skip_vision,
    )

    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            min_new_tokens=VLM_GENERATION_LEN,
            max_new_tokens=VLM_GENERATION_LEN,
            do_sample=False,
        )
    hf_tokens = hf_output[0, hf_inputs["input_ids"].shape[1] :].cpu().numpy()

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        kv_offload=True,
        weight_free=True,
        dtype=torch.float32,
    )
    hf_inputs, qeff_inputs = _prepare_vlm_test_inputs(
        processor,
        hf_model,
        qeff_model.model,
        model_config,
        image_size,
        skip_vision=skip_vision,
    )
    export_paths = qeff_model.export(
        tmp_export_dir / "weight_free_vlm_export",
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        prefill_seq_len=compile_prefill_seq_len,
        ctx_len=model_config["ctx_len"],
        skip_vision=skip_vision,
    )
    language_onnx = Path(export_paths[-1])
    assert language_onnx.is_file()
    _assert_weight_spec(language_onnx)
    _assert_vlm_weight_free_graph(model_type, language_onnx)
    if not skip_vision:
        vision_onnx = Path(export_paths[0])
        assert vision_onnx.is_file()
        _assert_weight_spec(vision_onnx)

    common_compile_kwargs = {
        "ctx_len": model_config["ctx_len"],
        "num_devices": 1,
        "num_cores": 16,
        "mxfp6_matmul": False,
        "use_onnx_subfunctions": True,
    }
    if skip_vision:
        common_compile_kwargs["skip_vision"] = True
    else:
        common_compile_kwargs.update(
            _get_vlm_compile_kwargs(
                qeff_model.model,
                model_config,
                image_size,
                heights=image_size[1],
                widths=image_size[0],
                num_frames=1,
            )
        )
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        vision_qpc = qeff_model.compile(
            **common_compile_kwargs,
            aic_enable_depth_first=False,
            split_model_io=True,
            prefill_seq_len=compile_prefill_seq_len,
            skip_lang=True,
            skip_vision=False,
        )
        prefill_qpc = qeff_model.compile(
            **common_compile_kwargs,
            retain_full_kv=True,
            split_model_io=True,
            aic_enable_depth_first=False,
            prefill_seq_len=compile_prefill_seq_len,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
        )
        decode_qpc = qeff_model.compile(
            **common_compile_kwargs,
            retain_full_kv=True,
            split_model_io=True,
            aic_enable_depth_first=True,
            prefill_seq_len=1,
            prefill_only=False,
            skip_vision=True,
        )
        layer_types = list(qeff_model.model.config.text_config.layer_types)
        qaic_tokens = _run_qwen35_three_qpc(
            qeff_model,
            qeff_inputs,
            {
                "vision_qpc_path": vision_qpc["vision_qpc_path"],
                "lang_prefill_qpc_path": prefill_qpc["lang_prefill_qpc_path"],
                "lang_decode_qpc_path": decode_qpc["lang_decode_qpc_path"],
            },
            layer_types,
            VLM_GENERATION_LEN,
        )[0]
    else:
        compile_kwargs = {**common_compile_kwargs, "prefill_seq_len": compile_prefill_seq_len}
        qeff_model.compile(**compile_kwargs)
        qaic_output = qeff_model.generate(inputs=qeff_inputs, generation_len=VLM_GENERATION_LEN)
        assert qaic_output.generated_ids is not None
        qaic_tokens = np.asarray(qaic_output.generated_ids)[0, :VLM_GENERATION_LEN]
    assert np.array_equal(hf_tokens[:VLM_GENERATION_LEN], qaic_tokens), (
        f"Weight-free HF/QAIC parity failed for {model_type}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )


@pytest.mark.weight_free
@pytest.mark.weight_free_blocking
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type,blocking_key", VLM_DECODE_BLOCKING_QAIC_CASES)
def test_weight_free_vlm_decode_blocked_hw_hf_parity_and_artifact_metrics(
    model_type,
    blocking_key,
    tmp_export_dir,
    manual_cleanup,
    record_property,
    request,
):
    """HF PT tokens == decode-blocked weight-free VLM QAIC tokens, with real vision inputs."""
    model_config = _VLM_MODEL_CONFIGS[model_type]
    model_id = WEIGHT_FREE_VLM_QAIC_MODEL_IDS[model_type]
    checkpoint_dir = tmp_export_dir / f"{model_type}_{blocking_key}_decode_blocking_checkpoint"
    image_size = (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT)
    qaic_config = dict(WEIGHT_FREE_BLOCKING_MODE_CASES[blocking_key])

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        hf_model, model_config = _build_vlm_test_checkpoint(
            model_type,
            checkpoint_dir,
            model_id,
            force_full_attention=model_type == "gemma4",
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    hf_inputs, _ = _prepare_vlm_test_inputs(
        processor,
        hf_model,
        None,
        model_config,
        image_size,
        skip_vision=False,
    )
    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            min_new_tokens=VLM_DECODE_BLOCKING_GENERATION_LEN,
            max_new_tokens=VLM_DECODE_BLOCKING_GENERATION_LEN,
            do_sample=False,
        )
    hf_tokens = hf_output[0, hf_inputs["input_ids"].shape[1] :].cpu().numpy()

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        kv_offload=True,
        weight_free=True,
        dtype=torch.float32,
        qaic_config=qaic_config,
    )

    def _cleanup_blocking_artifacts():
        cleanup_paths = [tmp_export_dir]
        for component_name in ("vision_model", "lang_model"):
            component = getattr(qeff_model, component_name, None)
            onnx_path = getattr(component, "onnx_path", None)
            if onnx_path is not None:
                cleanup_paths.append(onnx_path)
        manual_cleanup(cleanup_paths)

    request.addfinalizer(_cleanup_blocking_artifacts)

    _, qeff_inputs = _prepare_vlm_test_inputs(
        processor,
        hf_model,
        qeff_model.model,
        model_config,
        image_size,
        skip_vision=False,
    )

    common_compile_kwargs = {
        "batch_size": 1,
        "ctx_len": model_config["ctx_len"],
        "num_devices": 1,
        "num_cores": 16,
        "mxfp6_matmul": False,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
    }
    common_compile_kwargs.update(
        _get_vlm_compile_kwargs(
            qeff_model.model,
            model_config,
            image_size,
            heights=image_size[1],
            widths=image_size[0],
            num_frames=1,
        )
    )
    compile_options, device_ids = _vlm_blocking_runtime_options(model_type, blocking_key)
    decode_compile_kwargs = dict(common_compile_kwargs)
    decode_compile_kwargs.update(compile_options)
    decode_depth_first = not decode_compile_kwargs.get("user_tiled", False)

    if model_type == "gemma4":
        compile_start = time.perf_counter()
        qpc_paths = qeff_model.compile(
            compile_dir=str(tmp_export_dir / f"{model_type}_{blocking_key}_decode_blocking_compile"),
            **decode_compile_kwargs,
            aic_enable_depth_first=decode_depth_first,
            prefill_seq_len=model_config["prompt_len"],
            qaic_config=qaic_config,
        )
        compile_seconds = time.perf_counter() - compile_start
        language_onnx = Path(qeff_model.lang_model.onnx_path)
        qpc_size_bytes = sum(
            file.stat().st_size
            for qpc_path in qpc_paths.values()
            for file in Path(qpc_path).rglob("*")
            if file.is_file()
        )

        blocking_config = qeff_model.lang_model.hash_params.get("blocking_kwargs")
        assert blocking_config is not None
        assert blocking_config.mode == BlockingMode(qaic_config["blocking_mode"])
        assert qpc_paths.get("vision_qpc_path")
        assert qpc_paths.get("lang_qpc_path")
        assert language_onnx.is_file()
        _assert_weight_spec(language_onnx)
        _assert_vlm_weight_free_graph(model_type, language_onnx)
        assert_blocked_kv_ops_for_mode(language_onnx, qeff_model.lang_model, blocking_key)

        for name, value in {
            "qpc_compile_seconds": compile_seconds,
            "qpc_size_bytes": qpc_size_bytes,
        }.items():
            record_property(name, value)

        qaic_output = qeff_model.generate(
            inputs=qeff_inputs,
            generation_len=VLM_DECODE_BLOCKING_GENERATION_LEN,
            device_ids=device_ids,
        )
        assert qaic_output.generated_ids is not None
        qaic_tokens = np.asarray(qaic_output.generated_ids)[0, :VLM_DECODE_BLOCKING_GENERATION_LEN]
        hf_tokens = hf_tokens[:VLM_DECODE_BLOCKING_GENERATION_LEN]
        print("HF first 10 token IDs:", hf_tokens.tolist())
        print("QAIC first 10 token IDs:", qaic_tokens.tolist())
        assert np.array_equal(hf_tokens, qaic_tokens), (
            f"Weight-free VLM decode blocking HF/QAIC parity failed for {model_type}/{blocking_key}: "
            f"HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
        )
        return

    compile_start = time.perf_counter()
    vision_qpc = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_{blocking_key}_decode_blocking_vision"),
        **common_compile_kwargs,
        aic_enable_depth_first=False,
        split_model_io=True,
        prefill_seq_len=model_config["prompt_len"],
        skip_lang=True,
        skip_vision=False,
    )
    prefill_qpc = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_{blocking_key}_decode_blocking_prefill"),
        **common_compile_kwargs,
        retain_full_kv=True,
        split_model_io=True,
        aic_enable_depth_first=False,
        prefill_seq_len=model_config["prompt_len"],
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
    )
    decode_qpc = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_{blocking_key}_decode_blocking_decode"),
        **decode_compile_kwargs,
        retain_full_kv=True,
        split_model_io=True,
        aic_enable_depth_first=decode_depth_first,
        prefill_seq_len=1,
        prefill_only=False,
        skip_vision=True,
        qaic_config=qaic_config,
    )
    compile_seconds = time.perf_counter() - compile_start
    language_onnx = Path(qeff_model.lang_model.onnx_path)
    qpc_paths = {
        "vision_qpc_path": vision_qpc["vision_qpc_path"],
        "lang_prefill_qpc_path": prefill_qpc["lang_prefill_qpc_path"],
        "lang_decode_qpc_path": decode_qpc["lang_decode_qpc_path"],
    }
    qpc_size_bytes = sum(
        file.stat().st_size for qpc_path in qpc_paths.values() for file in Path(qpc_path).rglob("*") if file.is_file()
    )

    blocking_config = qeff_model.lang_model.hash_params.get("blocking_kwargs")
    assert blocking_config is not None
    assert blocking_config.mode == BlockingMode(qaic_config["blocking_mode"])
    assert qpc_paths.get("vision_qpc_path")
    assert qpc_paths.get("lang_prefill_qpc_path")
    assert qpc_paths.get("lang_decode_qpc_path")
    assert language_onnx.is_file()
    _assert_weight_spec(language_onnx)
    _assert_vlm_weight_free_graph(model_type, language_onnx)
    assert_blocked_kv_ops_for_mode(language_onnx, qeff_model.lang_model, blocking_key)

    for name, value in {
        "qpc_compile_seconds": compile_seconds,
        "qpc_size_bytes": qpc_size_bytes,
    }.items():
        record_property(name, value)

    qaic_tokens = _run_qwen35_three_qpc(
        qeff_model,
        qeff_inputs,
        qpc_paths,
        _language_layer_types(qeff_model),
        VLM_DECODE_BLOCKING_GENERATION_LEN,
        decode_device_ids=device_ids,
    )
    qaic_tokens = np.asarray(qaic_tokens)[0, :VLM_DECODE_BLOCKING_GENERATION_LEN]
    hf_tokens = hf_tokens[:VLM_DECODE_BLOCKING_GENERATION_LEN]
    print("HF first 10 token IDs:", hf_tokens.tolist())
    print("QAIC first 10 token IDs:", qaic_tokens.tolist())
    assert np.array_equal(hf_tokens, qaic_tokens), (
        f"Weight-free VLM decode blocking HF/QAIC parity failed for {model_type}/{blocking_key}: "
        f"HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type", WEIGHT_FREE_VLM_QAIC_MODEL_PARAMS)
def test_weight_free_vlm_multi_specialization_hf_qaic_parity(model_type, tmp_export_dir):
    """Compile two vision resolutions in one QPC and run the dummy VLM on QAIC."""
    if model_type == "qwen2_5_vl":
        pytest.skip("Qwen 2.5-VL vision QPC is deferred because QAIC reports a Conv3D bias-shape error")
    if model_type == "gemma4":
        pytest.skip("Gemma 4 does not support multi-specialization yet")
    model_config = _VLM_MODEL_CONFIGS[model_type]
    model_id = WEIGHT_FREE_VLM_QAIC_MODEL_IDS[model_type]
    checkpoint_dir = tmp_export_dir / f"{model_type}_multi_spec_checkpoint"
    image_size = (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT)
    specialization_heights = [224, VLM_IMAGE_HEIGHT]
    specialization_widths = [224, VLM_IMAGE_WIDTH]
    specialization_frames = [1, 1]

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        hf_model, model_config = _build_vlm_test_checkpoint(model_type, checkpoint_dir, model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        kv_offload=True,
        weight_free=True,
        dtype=torch.float32,
    )
    hf_inputs, qeff_inputs = _prepare_vlm_test_inputs(
        processor,
        hf_model,
        qeff_model.model,
        model_config,
        image_size,
    )

    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            min_new_tokens=VLM_GENERATION_LEN,
            max_new_tokens=VLM_GENERATION_LEN,
            do_sample=False,
        )
    hf_tokens = hf_output[0, hf_inputs["input_ids"].shape[1] :].cpu().numpy()

    export_paths = qeff_model.export(
        tmp_export_dir / f"{model_type}_multi_spec_export",
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        prefill_seq_len=model_config["prompt_len"],
        ctx_len=model_config["ctx_len"],
    )
    vision_onnx, language_onnx = (Path(export_paths[0]), Path(export_paths[-1]))
    _assert_weight_spec(vision_onnx)
    _assert_weight_spec(language_onnx)
    _assert_vlm_weight_free_graph(model_type, language_onnx)

    specialization_kwargs = _get_vlm_specialization_kwargs(
        qeff_model.model,
        model_config,
        [224, VLM_IMAGE_HEIGHT],
        [224, VLM_IMAGE_WIDTH],
        [1, 1],
    )
    specializations, _ = qeff_model.model.get_specializations(
        batch_size=model_config["batch_size"],
        prefill_seq_len=model_config["prompt_len"],
        ctx_len=model_config["ctx_len"],
        kv_offload=True,
        **specialization_kwargs,
    )
    expected_vision_specs = 2 if "height" in specialization_kwargs else 1
    _assert_vlm_specialization_structure(specializations, expected_vision_specs)

    compile_kwargs = {
        "prefill_seq_len": model_config.get("prompt_len"),
        "ctx_len": model_config.get("ctx_len"),
        "num_devices": 1,
        "num_cores": 16,
        "mxfp6_matmul": False,
        "use_onnx_subfunctions": True,
    }
    compile_kwargs.update(
        _get_vlm_compile_kwargs(
            qeff_model.model,
            model_config,
            image_size,
            heights=specialization_heights,
            widths=specialization_widths,
            num_frames=specialization_frames,
            min_pixels=224 * 224,
            max_pixels=VLM_IMAGE_WIDTH * VLM_IMAGE_HEIGHT,
        )
    )
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        vision_qpc = qeff_model.compile(
            **compile_kwargs,
            aic_enable_depth_first=False,
            split_model_io=True,
            skip_lang=True,
            skip_vision=False,
        )
        prefill_qpc = qeff_model.compile(
            **compile_kwargs,
            retain_full_kv=True,
            split_model_io=True,
            aic_enable_depth_first=False,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
        )
        decode_compile_kwargs = {**compile_kwargs, "prefill_seq_len": 1}
        decode_qpc = qeff_model.compile(
            **decode_compile_kwargs,
            retain_full_kv=True,
            split_model_io=True,
            aic_enable_depth_first=True,
            prefill_only=False,
            skip_vision=True,
        )
        qaic_tokens = _run_qwen35_three_qpc(
            qeff_model,
            qeff_inputs,
            {
                "vision_qpc_path": vision_qpc["vision_qpc_path"],
                "lang_prefill_qpc_path": prefill_qpc["lang_prefill_qpc_path"],
                "lang_decode_qpc_path": decode_qpc["lang_decode_qpc_path"],
            },
            list(qeff_model.model.config.text_config.layer_types),
            VLM_GENERATION_LEN,
        )[0]
    else:
        qeff_model.compile(**compile_kwargs)
        qaic_output = qeff_model.generate(
            inputs=qeff_inputs,
            tokenizer=getattr(processor, "tokenizer", None),
            generation_len=VLM_GENERATION_LEN,
            multi_specs=True,
            num_frames=1,
        )
        qaic_tokens = np.asarray(qaic_output.generated_ids)[0, :VLM_GENERATION_LEN]
    assert np.array_equal(hf_tokens[:VLM_GENERATION_LEN], qaic_tokens), (
        f"Weight-free multi-specialization HF/QAIC parity failed for {model_type}: "
        f"HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )
