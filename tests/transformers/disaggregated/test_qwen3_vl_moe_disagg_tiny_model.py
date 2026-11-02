# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from tests.two_phase import model_export_compile_lock, resolve_two_phase_cleanup

MODEL_NAME = "tiny-random/qwen3-vl-moe"
PREFILL_SEQ_LEN = 64
CTX_LEN = 512
BATCH_SIZE = 1
GENERATION_LEN = 4
# Vision, prefill, and decode sessions stay resident together; 3 * 4 cores fits on one 16-core card.
DISAGG_NUM_CORES = 4
IMAGE_SIZE = (536, 354)
TEXT_PROMPT = "Describe all the colors seen in the image."

VISION_INPUTS = {
    "pixel_values",
    "image_grid_thw",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_INPUTS = {"pixel_values", "image_masks"}
VISION_OUTPUTS = ("vision_embeds", "deepstack_features")


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_distinct_onnx_paths(onnx_paths: dict[str, Path]):
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def _load_hf_model_from_pretrained(config):
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
        )
    model.eval()
    return model


def _build_config(dtype: str = "float16"):
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = 1
        config.text_config.dtype = dtype
        config.text_config.torch_dtype = getattr(torch, dtype)
    return config


def _prepare_messages(image: Image.Image) -> list:
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": TEXT_PROMPT},
                ],
            }
        ]
        for _ in range(BATCH_SIZE)
    ]


def _prepare_processor_inputs(processor: AutoProcessor, messages: list) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info

    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"))


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _update_retained_states(target_inputs: dict, source_outputs: dict, num_hidden_layers: int):
    for layer_idx in range(num_hidden_layers):
        target_inputs[f"past_key.{layer_idx}"] = source_outputs[f"past_key.{layer_idx}_RetainedState"]
        target_inputs[f"past_value.{layer_idx}"] = source_outputs[f"past_value.{layer_idx}_RetainedState"]


def _run_hf_torch_fp32(model, processor: AutoProcessor, messages: list) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    inputs = _prepare_processor_inputs(processor, messages)
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=GENERATION_LEN, do_sample=False)

    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _run_disagg_qaic_generation(
    qeff_model: QEFFAutoModelForImageTextToText,
    processor: AutoProcessor,
    common_inputs: dict,
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=PREFILL_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    pad_token_id = processor.tokenizer.pad_token_id or 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"],
        (0, padded_len - input_ids_length),
        "constant",
        0,
    )
    inputs = {name: np.array(value) for name, value in inputs.items()}

    vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUTS}
    vision_inputs.update(
        {name: vision_inputs[name].astype("float16") for name in VISION_FP16_INPUTS if name in vision_inputs}
    )
    vision_outputs = vision_session.run(vision_inputs)
    vision_session.deactivate()

    lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    lang_inputs["image_idx"] = np.array([[0]])
    for output_name in VISION_OUTPUTS:
        if output_name in vision_outputs:
            lang_inputs[output_name] = vision_outputs[output_name]

    prefill_session.set_buffers(vision_outputs)
    chunk_inputs = lang_inputs.copy()
    outputs = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        outputs = prefill_session.run(chunk_inputs)
        _update_retained_states(chunk_inputs, outputs, qeff_model.model.config.text_config.num_hidden_layers)
        chunk_inputs["image_idx"] = outputs["image_idx_output"]

    prefill_session.deactivate()

    generated_ids = [_get_next_token_ids(outputs["logits"])]
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
    }
    _update_retained_states(decode_inputs, outputs, qeff_model.model.config.text_config.num_hidden_layers)

    decode_outputs = decode_session.run(decode_inputs)
    generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))

    position_ids = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
    loop_decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": position_ids,
    }
    _update_retained_states(loop_decode_inputs, decode_outputs, qeff_model.model.config.text_config.num_hidden_layers)

    for _ in range(GENERATION_LEN - 2):
        decode_outputs = decode_session.run(loop_decode_inputs)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        position_ids += 1
        _update_retained_states(
            loop_decode_inputs,
            decode_outputs,
            qeff_model.model.config.text_config.num_hidden_layers,
        )
        loop_decode_inputs.update(
            {
                "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
                "position_ids": position_ids,
            }
        )

    return np.stack(generated_ids, axis=1)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.disagg_dma
def test_qwen3_vl_moe_disagg_qaic_vs_hf_fp32(manual_cleanup):
    pytest.importorskip("qwen_vl_utils")
    torch.manual_seed(42)
    manual_cleanup, compile_only = resolve_two_phase_cleanup(manual_cleanup)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32")).to(dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))
    messages = _prepare_messages(image)
    common_inputs = _prepare_processor_inputs(processor, messages)
    hf_tokens = None if compile_only else _run_hf_torch_fp32(hf_model, processor, messages)

    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    qeff_model = QEFFAutoModelForImageTextToText(
        hf_model,
        kv_offload=True,
        config=hf_model.config,
        torch_dtype=torch.float32,
        layerwise=False,
    )

    sessions = []
    compiled_onnx_paths = {}
    try:
        with model_export_compile_lock(MODEL_NAME):
            vision_qpc_path = qeff_model.compile(
                batch_size=BATCH_SIZE,
                prefill_seq_len=PREFILL_SEQ_LEN,
                ctx_len=CTX_LEN,
                height=image.height,
                width=image.width,
                num_cores=DISAGG_NUM_CORES,
                num_devices=1,
                mos=1,
                aic_enable_depth_first=True,
                skip_vision=False,
                split_model_io=True,
                skip_lang=True,
                use_onnx_subfunctions=True,
                layerwise=False,
            )
            compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

            prefill_qpc_path = qeff_model.compile(
                batch_size=BATCH_SIZE,
                prefill_seq_len=PREFILL_SEQ_LEN,
                ctx_len=CTX_LEN,
                height=image.height,
                width=image.width,
                num_cores=DISAGG_NUM_CORES,
                num_devices=1,
                retain_full_kv=True,
                split_model_io=True,
                mos=1,
                aic_enable_depth_first=True,
                prefill_only=True,
                enable_chunking=True,
                skip_vision=True,
                use_onnx_subfunctions=True,
                layerwise=False,
                layerwise_window_size=1,
            )
            compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")

            decode_qpc_path = qeff_model.compile(
                batch_size=BATCH_SIZE,
                prefill_seq_len=1,
                ctx_len=CTX_LEN,
                height=image.height,
                width=image.width,
                num_cores=DISAGG_NUM_CORES,
                num_devices=1,
                split_model_io=True,
                mos=1,
                aic_enable_depth_first=True,
                prefill_only=False,
                skip_vision=True,
                use_onnx_subfunctions=True,
                layerwise=False,
                layerwise_window_size=1,
            )
            compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")
            _assert_distinct_onnx_paths(compiled_onnx_paths)
            print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        if compile_only:
            return

        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))
        prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"))
        decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))
        sessions.extend([vision_session, prefill_session, decode_session])

        qaic_tokens = _run_disagg_qaic_generation(
            qeff_model=qeff_model,
            processor=processor,
            common_inputs=common_inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert hf_tokens is not None
    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert np.issubdtype(hf_tokens.dtype, np.integer)
    assert (hf_tokens == qaic_tokens).all(), "Tokens don't match for HF Torch fp32 output and disagg QAIC output"
