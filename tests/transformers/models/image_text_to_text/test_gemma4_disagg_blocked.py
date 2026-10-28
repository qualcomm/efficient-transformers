# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES

from .test_image_text_to_text_models import model_config_dict

MODEL_NAME = "tiny-random/gemma-4-moe"
BATCH_SIZE = 1
GENERATION_LEN = 4
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
GEMMA4_TEST_SLIDING_WINDOW = 16


def _assert_lang_only_compile(qeff_model, qpc_paths: dict, qpc_keys: tuple[str, ...]):
    assert any(qpc_paths.get(key) for key in qpc_keys), f"Compile did not return any of: {qpc_keys}"
    assert not qpc_paths.get("vision_qpc_path"), "Vision compile should be skipped"


def _resolve_lang_qpc_path(qpc_paths: dict, qpc_keys: tuple[str, ...]) -> Path:
    for key in qpc_keys:
        if qpc_paths.get(key):
            return Path(qpc_paths[key])
    raise KeyError(f"Could not find any of {qpc_keys} in compile output keys: {list(qpc_paths.keys())}")


def _prepare_text_inputs(processor: AutoProcessor) -> dict:
    cfg = model_config_dict[MODEL_NAME]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cfg["text_prompt"]},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return dict(processor(text=prompt, return_tensors="pt"))


def _run_hf_torch_fp32_from_inputs(model, inputs: dict) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_tensor(value) and torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=GENERATION_LEN, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _get_output(source_outputs: dict, output_name: str):
    if output_name in source_outputs:
        return source_outputs[output_name]
    for actual_name, value in source_outputs.items():
        if actual_name.rsplit("/", 1)[-1] == output_name:
            return value
    raise KeyError(output_name)


def _get_output_if_present(source_outputs: dict, output_name: str):
    try:
        return _get_output(source_outputs, output_name)
    except KeyError:
        return None


def _update_retained_states(target_inputs: dict, source_outputs: dict, num_hidden_layers: int):
    for layer_idx in range(num_hidden_layers):
        key_name = f"past_key.{layer_idx}_RetainedState"
        value_name = f"past_value.{layer_idx}_RetainedState"
        key_state = _get_output_if_present(source_outputs, key_name)
        value_state = _get_output_if_present(source_outputs, value_name)
        if key_state is not None:
            target_inputs[f"past_key.{layer_idx}"] = key_state
        if value_state is not None:
            target_inputs[f"past_value.{layer_idx}"] = value_state

    for input_name, output_name in (
        ("vision_embeds", "vision_embeds_RetainedState"),
        ("deepstack_features", "deepstack_features_RetainedState"),
        ("image_idx", "image_idx_output"),
    ):
        output_value = _get_output_if_present(source_outputs, output_name)
        if output_value is not None:
            target_inputs[input_name] = output_value


def _session_input_names(session) -> set[str]:
    input_names = set(session.input_names)
    input_names.update(name.rsplit("/", 1)[-1] for name in session.input_names)
    return input_names


def _filter_session_inputs(session, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    filtered_inputs = {}
    input_names = _session_input_names(session)
    for name, value in inputs.items():
        if name not in input_names:
            continue
        binding_index = session.binding_index_map.get(name)
        if binding_index is not None:
            dtype = session.aic_to_np_dtype_mapping[session.bindings[binding_index].type]
            value = value.astype(dtype, copy=False)
        filtered_inputs[name] = value
    return filtered_inputs


def _build_gemma4_disagg_test_config(model_name: str) -> AutoConfig:
    model_type = model_config_dict[model_name].get("model_type")
    custom_config = model_config_dict[model_name].get("additional_params", {})
    hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
    hf_config.name_or_path = model_name

    if hasattr(hf_config, "text_config"):
        hf_config.text_config.sliding_window = GEMMA4_TEST_SLIDING_WINDOW
        hf_config.text_config.num_kv_shared_layers = 0
        hf_config.text_config.num_hidden_layers = 2
        hf_config.text_config.layer_types = ["sliding_attention", "full_attention"]
    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.num_hidden_layers = 1
    return hf_config


def _load_hf_model_from_pretrained(config):
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
            ignore_mismatched_sizes=True,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
            ignore_mismatched_sizes=True,
        )
    model.eval()
    return model


def _load_gemma4_hf_model_for_disagg():
    config = _build_gemma4_disagg_test_config(MODEL_NAME)
    hf_model = _load_hf_model_from_pretrained(config).to(dtype=torch.float32)
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    return hf_model


def _build_qeff_model_from_hf(hf_model):
    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

    return QEFFAutoModelForImageTextToText(
        hf_model,
        kv_offload=True,
        config=hf_model.config,
        torch_dtype=torch.float32,
        layerwise=False,
    )


def _run_disagg_qaic_generation_lang_only(
    qeff_model,
    common_inputs: dict,
    prefill_qpc_path: Path,
    decode_qpc_path: Path,
    prefill_seq_len: int,
    pad_token_id: int,
) -> np.ndarray:
    from QEfficient.generation.cloud_infer import QAICInferenceSession

    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len

    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"],
            (0, padded_len - input_ids_length),
            "constant",
            0,
        )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.nn.functional.pad(
            inputs["mm_token_type_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            0,
        )
    if "position_ids" in inputs:
        inputs["position_ids"] = torch.nn.functional.pad(
            inputs["position_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            -1,
        )
    elif "attention_mask" in inputs:
        inputs["position_ids"] = torch.where(inputs["attention_mask"].to(torch.bool), torch.arange(padded_len), -1)
    else:
        inputs["position_ids"] = torch.arange(padded_len).unsqueeze(0).repeat(BATCH_SIZE, 1)

    np_inputs = {name: np.array(value) for name, value in inputs.items()}

    prefill_session = QAICInferenceSession(str(prefill_qpc_path))

    outputs = None
    chunk_inputs = dict(np_inputs)
    try:
        for chunk_idx in range(num_chunks):
            chunk_inputs["input_ids"] = np_inputs["input_ids"][
                :, chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
            ]
            chunk_inputs["position_ids"] = np_inputs["position_ids"][
                ..., chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
            ]
            if "mm_token_type_ids" in np_inputs:
                chunk_inputs["mm_token_type_ids"] = np_inputs["mm_token_type_ids"][
                    ..., chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
                ]
            outputs = prefill_session.run(_filter_session_inputs(prefill_session, chunk_inputs))
            _update_retained_states(chunk_inputs, outputs, qeff_model.model.config.text_config.num_hidden_layers)
    finally:
        prefill_session.deactivate()

    assert outputs is not None, "Prefill session did not return outputs"
    first_token = _get_next_token_ids(outputs["logits"])
    decode_inputs = {
        "input_ids": first_token.reshape(BATCH_SIZE, 1),
        "position_ids": np.max(np_inputs["position_ids"], axis=-1, keepdims=True) + 1,
    }
    _update_retained_states(decode_inputs, outputs, qeff_model.model.config.text_config.num_hidden_layers)

    generated_ids = [first_token]
    decode_session = QAICInferenceSession(str(decode_qpc_path))
    try:
        for _ in range(GENERATION_LEN - 1):
            decode_outputs = decode_session.run(_filter_session_inputs(decode_session, decode_inputs))
            next_token = _get_next_token_ids(decode_outputs["logits"])
            generated_ids.append(next_token)
            _update_retained_states(
                decode_inputs, decode_outputs, qeff_model.model.config.text_config.num_hidden_layers
            )
            decode_inputs["input_ids"] = next_token.reshape(BATCH_SIZE, 1)
            decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
    finally:
        decode_session.deactivate()

    return np.stack(generated_ids, axis=1)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
def test_gemma4_disagg_lang_prefill_qkv_decode_qkv_single_device_qaic_vs_hf_fp32(manual_cleanup):
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    common_inputs = _prepare_text_inputs(processor)

    hf_model = _load_gemma4_hf_model_for_disagg()
    hf_tokens = _run_hf_torch_fp32_from_inputs(hf_model, common_inputs)
    qeff_model = _build_qeff_model_from_hf(hf_model)

    compiled_onnx_paths = []
    prefill_seq_len = model_config_dict[MODEL_NAME]["prompt_len"]
    ctx_len = model_config_dict[MODEL_NAME]["ctx_len"]

    try:
        prefill_qpc_paths = qeff_model.compile(
            batch_size=1,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_cores=DEFAULT_AIC_NUM_CORES,
            num_devices=1,
            retain_full_kv=True,
            split_model_io=True,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
            use_onnx_subfunctions=True,
            qaic_config={
                "blocking_mode": "prefill_qkv",
                "num_kv_blocks": NUM_KV_BLOCKS,
                "num_q_blocks": NUM_Q_BLOCKS,
                "ctx_len": ctx_len,
            },
        )
        if getattr(qeff_model.lang_model, "onnx_path", None) is not None:
            compiled_onnx_paths.append(qeff_model.lang_model.onnx_path)
        _assert_lang_only_compile(qeff_model, prefill_qpc_paths, ("lang_prefill_qpc_path", "lang_qpc_path"))

        decode_qpc_paths = qeff_model.compile(
            batch_size=1,
            prefill_seq_len=1,
            ctx_len=ctx_len,
            num_cores=DEFAULT_AIC_NUM_CORES,
            num_devices=1,
            retain_full_kv=True,
            split_model_io=True,
            prefill_only=False,
            skip_vision=True,
            use_onnx_subfunctions=True,
            layerwise=False,
            qaic_config={
                "blocking_mode": "qkv",
                "num_kv_blocks": NUM_KV_BLOCKS,
                "num_q_blocks": NUM_Q_BLOCKS,
                "ctx_len": ctx_len,
            },
        )
        if getattr(qeff_model.lang_model, "onnx_path", None) is not None:
            compiled_onnx_paths.append(qeff_model.lang_model.onnx_path)
        _assert_lang_only_compile(qeff_model, decode_qpc_paths, ("lang_decode_qpc_path", "lang_qpc_path"))

        qaic_tokens = _run_disagg_qaic_generation_lang_only(
            qeff_model=qeff_model,
            common_inputs=common_inputs,
            prefill_qpc_path=_resolve_lang_qpc_path(prefill_qpc_paths, ("lang_prefill_qpc_path", "lang_qpc_path")),
            decode_qpc_path=_resolve_lang_qpc_path(decode_qpc_paths, ("lang_decode_qpc_path", "lang_qpc_path")),
            prefill_seq_len=prefill_seq_len,
            pad_token_id=processor.tokenizer.pad_token_id or 1,
        )
    finally:
        manual_cleanup(compiled_onnx_paths)

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert np.issubdtype(hf_tokens.dtype, np.integer)
    assert (qaic_tokens == hf_tokens).all(), "Tokens don't match for HF torch fp32 output and disagg QAIC output"
