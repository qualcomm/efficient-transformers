# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import gc
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
from PIL import Image

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from tests.two_phase import model_export_compile_lock, resolve_two_phase_cleanup
from tests.utils.load_kimi_utils import (
    KIMI_K25_MODEL_NAME,
    get_kimi_k25_test_config,
    load_kimi_k25_model_from_config,
    run_kimi_k25_hf_model_on_pytorch,
    set_deterministic,
)

PREFILL_SEQ_LEN = 256
CTX_LEN = 512
BATCH_SIZE = 1
GENERATION_LEN = 4
IMAGE_URL = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png"
TEXT_PROMPT = "Describe this image."
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "image_text_model_configs.json"


def _prepare_inputs(processor):
    image = Image.open(BytesIO(requests.get(IMAGE_URL, timeout=30).content)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]
    inputs = processor(
        messages=messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )
    return inputs, image.height, image.width


def _decode_tokens(tokenizer, token_ids: torch.Tensor) -> str:
    decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def _clone_inputs(inputs):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in inputs.items()}


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_distinct_onnx_paths(onnx_paths: dict[str, Path]):
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _session_input_names(session: QAICInferenceSession) -> set[str]:
    input_names = set(session.input_names)
    input_names.update(name.rsplit("/", 1)[-1] for name in session.input_names)
    return input_names


def _cast_for_session(session: QAICInferenceSession, name: str, value: np.ndarray) -> np.ndarray:
    binding_index = session.binding_index_map.get(name)
    if binding_index is None:
        return value
    dtype = session.aic_to_np_dtype_mapping[session.bindings[binding_index].type]
    return value.astype(dtype, copy=False)


def _filter_session_inputs(session: QAICInferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = _session_input_names(session)
    return {name: _cast_for_session(session, name, value) for name, value in inputs.items() if name in input_names}


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64).reshape(BATCH_SIZE, 1)


def _update_retained_states(target_inputs: dict[str, np.ndarray], source_outputs: dict[str, np.ndarray]):
    for output_name, value in source_outputs.items():
        output_basename = output_name.rsplit("/", 1)[-1]
        if not output_basename.endswith("_RetainedState"):
            continue
        target_inputs[output_basename.removesuffix("_RetainedState")] = value


def _load_kimi_random_model():
    set_deterministic(1234)
    with CONFIG_PATH.open() as config_file:
        multimodal_models = json.load(config_file)["image_text_models"]
    model_config_dict = {model_config["model_name"]: model_config for model_config in multimodal_models}
    config = get_kimi_k25_test_config(KIMI_K25_MODEL_NAME, model_config_dict, seed=1234)
    model, tokenizer, processor = load_kimi_k25_model_from_config(config, seed=1234)
    return model.eval().to("cpu"), tokenizer, processor


def _get_image_compile_dims(image_height: int, image_width: int) -> dict[str, int]:
    return {"image_height": image_height, "image_width": image_width}


def _compile_disagg_qpcs(qeff_model: QEFFAutoModelForImageTextToText, compile_dims: dict[str, int]):
    common_compile_kwargs = {
        "batch_size": BATCH_SIZE,
        "ctx_len": CTX_LEN,
        "num_cores": DEFAULT_AIC_NUM_CORES,
        "mxfp6_matmul": False,
        "split_model_io": True,
        "mos": 1,
        "aic_enable_depth_first": True,
        "use_onnx_subfunctions": True,
        "layerwise": False,
        **compile_dims,
    }

    compiled_onnx_paths = {}
    vision_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        skip_vision=False,
        skip_lang=True,
        num_devices=1,
        **common_compile_kwargs,
    )
    compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        prefill_only=True,
        skip_vision=True,
        skip_lang=False,
        num_devices=1,
        **common_compile_kwargs,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")

    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        prefill_only=False,
        skip_vision=True,
        skip_lang=False,
        num_devices=1,
        **common_compile_kwargs,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")
    _assert_distinct_onnx_paths(compiled_onnx_paths)

    return vision_qpc_path, prefill_qpc_path, decode_qpc_path, compiled_onnx_paths


def _run_disagg_qaic_generation(
    common_inputs: dict[str, torch.Tensor],
    vision_qpc_path,
    prefill_qpc_path,
    decode_qpc_path,
) -> np.ndarray:
    inputs = {name: _numpy(value) for name, value in _clone_inputs(common_inputs).items()}
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN

    inputs["input_ids"] = np.pad(
        inputs["input_ids"],
        ((0, 0), (0, padded_len - input_ids_length)),
        constant_values=1,
    )
    inputs["attention_mask"] = np.pad(
        inputs["attention_mask"],
        ((0, 0), (0, padded_len - input_ids_length)),
        constant_values=0,
    )

    grid_thws = inputs["grid_thws"].astype(np.int64)
    h = int(grid_thws[0, 1])
    w = int(grid_thws[0, 2])
    vision_inputs = {
        "pixel_values": inputs["pixel_values"],
        "h_shape": np.ones((h,), dtype=np.int64),
        "w_shape": np.ones((w,), dtype=np.int64),
    }
    vision_session = QAICInferenceSession(vision_qpc_path)
    try:
        vision_outputs = vision_session.run(_filter_session_inputs(vision_session, vision_inputs))
    finally:
        vision_session.deactivate()
        vision_session.program.unload()
        del vision_session
        gc.collect()

    vision_embeds = vision_outputs.get("vision_embeds")
    assert vision_embeds is not None, f"Vision QPC did not return vision_embeds. Outputs: {vision_outputs.keys()}"

    lang_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "position_ids": np.where(inputs["attention_mask"] > 0, np.arange(padded_len), -1).astype(np.int64),
        "vision_embeds": vision_embeds,
        "image_idx": np.zeros((BATCH_SIZE, 1), dtype=np.int64),
    }

    prefill_session = QAICInferenceSession(prefill_qpc_path)
    try:
        prefill_session.set_buffers(vision_outputs)
        chunk_inputs = lang_inputs.copy()
        prefill_outputs = None
        for chunk_idx in range(num_chunks):
            start = chunk_idx * PREFILL_SEQ_LEN
            end = (chunk_idx + 1) * PREFILL_SEQ_LEN
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, start:end]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][:, start:end]
            prefill_outputs = prefill_session.run(_filter_session_inputs(prefill_session, chunk_inputs))
            _update_retained_states(chunk_inputs, prefill_outputs)
            if "image_idx_output" in prefill_outputs:
                chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"].astype(np.int64)
    finally:
        prefill_session.deactivate()
        prefill_session.program.unload()
        del prefill_session
        gc.collect()

    assert prefill_outputs is not None, "QAIC prefill did not execute."

    generated_ids = [_get_next_token_ids(prefill_outputs["logits"])]
    decode_inputs = {
        "input_ids": generated_ids[-1],
        "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True).astype(np.int64) + 1,
        "vision_embeds": chunk_inputs.get("vision_embeds", vision_embeds),
        "image_idx": chunk_inputs.get("image_idx", np.zeros((BATCH_SIZE, 1), dtype=np.int64)),
    }
    _update_retained_states(decode_inputs, prefill_outputs)

    decode_session = QAICInferenceSession(decode_qpc_path)
    try:
        for _ in range(1, GENERATION_LEN):
            decode_outputs = decode_session.run(_filter_session_inputs(decode_session, decode_inputs))
            generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
            decode_inputs["input_ids"] = generated_ids[-1]
            decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
            if "image_idx_output" in decode_outputs:
                decode_inputs["image_idx"] = decode_outputs["image_idx_output"].astype(np.int64)
            _update_retained_states(decode_inputs, decode_outputs)
    finally:
        decode_session.deactivate()
        decode_session.program.unload()
        del decode_session
        gc.collect()

    return np.concatenate(generated_ids, axis=1)[0]


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.disagg_dma
def test_kimi_k25_disagg_qaic_vs_hf_fp32(manual_cleanup):
    manual_cleanup, compile_only = resolve_two_phase_cleanup(manual_cleanup)
    model, tokenizer, processor = _load_kimi_random_model()
    inputs, image_height, image_width = _prepare_inputs(processor)
    inputs = {name: (value.to("cpu") if torch.is_tensor(value) else value) for name, value in inputs.items()}
    hf_tokens = (
        None
        if compile_only
        else run_kimi_k25_hf_model_on_pytorch(
            copy.deepcopy(model), processor, _clone_inputs(inputs), max_gen_len=GENERATION_LEN
        )
    )

    qeff_model = QEFFAutoModelForImageTextToText(
        model,
        kv_offload=True,
        config=model.config,
        torch_dtype=torch.float32,
        layerwise=False,
    )

    compile_dims = _get_image_compile_dims(image_height, image_width)
    with model_export_compile_lock(KIMI_K25_MODEL_NAME):
        vision_qpc_path, prefill_qpc_path, decode_qpc_path, compiled_onnx_paths = _compile_disagg_qpcs(
            qeff_model,
            compile_dims,
        )
    print(f"Kimi-K2.5 disagg ONNX paths: {compiled_onnx_paths}")

    if compile_only:
        return

    try:
        qaic_tokens = _run_disagg_qaic_generation(
            common_inputs=inputs,
            vision_qpc_path=vision_qpc_path.get("vision_qpc_path"),
            prefill_qpc_path=prefill_qpc_path.get("lang_prefill_qpc_path"),
            decode_qpc_path=decode_qpc_path.get("lang_decode_qpc_path"),
        )
    finally:
        manual_cleanup(list(compiled_onnx_paths.values()))

    assert hf_tokens is not None
    print("HF:", _decode_tokens(tokenizer, hf_tokens), "\n", hf_tokens)
    print("Disagg QAIC:", _decode_tokens(tokenizer, torch.as_tensor(qaic_tokens)), "\n", qaic_tokens)

    assert torch.equal(hf_tokens, torch.as_tensor(qaic_tokens)), "HF and disagg QAIC tokens do not match"
