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
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from tests.two_phase import model_export_compile_lock, resolve_two_phase_cleanup

MODEL_NAME = "tiny-random/qwen3-vl-moe"
PREFILL_SEQ_LEN = 64
CTX_LEN = 512
BATCH_SIZE = 1
GENERATION_LEN = 4
IMAGE_SIZE = (536, 354)
TEXT_PROMPT = "Describe all the colors seen in the image."
NUM_KV_BLOCKS = 2
HEAD_BLOCK_SIZE = 8
NUM_Q_BLOCKS = 2
PREFILL_MDP_NUM_LAYERS = 2
PREFILL_MDP_NUM_DEVICES = 2
PREFILL_MDP_NUM_PARTITIONS = 2
PREFILL_EXPERT_PARALLEL_CHUNK_SIZE = 32
PREFILL_ONLINE_QL_CHUNK = 32
PREFILL_ONLINE_N_REP_CHUNK = 2


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_distinct_onnx_paths(onnx_paths: dict[str, Path]):
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def _assert_lang_only_compile(qeff_model, qpc_paths: dict, qpc_key: str):
    assert qpc_paths.get(qpc_key), f"Compile did not return {qpc_key}"
    assert not qpc_paths.get("vision_qpc_path"), "Vision compile should be skipped"
    assert getattr(qeff_model.vision_model, "onnx_path", None) is None, "Vision export should be skipped"


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


def _build_config(dtype: str = "float16", num_hidden_layers: int = 1):
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = num_hidden_layers
        config.text_config.dtype = dtype
        config.text_config.torch_dtype = getattr(torch, dtype)
    return config


def _prepare_messages(batch_size: int = BATCH_SIZE) -> list:
    return [TEXT_PROMPT for _ in range(batch_size)]


def _prepare_processor_inputs(processor: AutoProcessor, messages: list) -> dict:
    return dict(processor(text=messages, padding=True, return_tensors="pt"))


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


def _update_retained_states(target_inputs: dict, source_outputs: dict, num_hidden_layers: int):
    for layer_idx in range(num_hidden_layers):
        target_inputs[f"past_key.{layer_idx}"] = _get_output(source_outputs, f"past_key.{layer_idx}_RetainedState")
        target_inputs[f"past_value.{layer_idx}"] = _get_output(source_outputs, f"past_value.{layer_idx}_RetainedState")
    for input_name, output_name in (
        ("vision_embeds", "vision_embeds_RetainedState"),
        ("deepstack_features", "deepstack_features_RetainedState"),
        ("image_idx", "image_idx_output"),
    ):
        try:
            target_inputs[input_name] = _get_output(source_outputs, output_name)
        except KeyError:
            pass


def _session_input_names(session: QAICInferenceSession) -> set[str]:
    input_names = set(session.input_names)
    input_names.update(name.rsplit("/", 1)[-1] for name in session.input_names)
    return input_names


def _filter_session_inputs(session: QAICInferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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


def _run_hf_torch_fp32(model, processor: AutoProcessor, messages: list) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    inputs = _prepare_processor_inputs(processor, messages)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=GENERATION_LEN, do_sample=False)

    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _run_disagg_qaic_generation(
    qeff_model: QEFFAutoModelForImageTextToText,
    processor: AutoProcessor,
    common_inputs: dict,
    prefill_qpc_path: Path,
    decode_qpc_path: Path,
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
    inputs = {name: np.array(value) for name, value in inputs.items()}

    dummy_lang_inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
    )["lang"]
    lang_inputs = {
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "vision_embeds": np.array(dummy_lang_inputs["vision_embeds"]),
        "deepstack_features": np.array(dummy_lang_inputs["deepstack_features"]),
        "image_idx": np.array(dummy_lang_inputs["image_idx"]),
    }

    chunk_inputs = lang_inputs.copy()
    outputs = None
    prefill_session = QAICInferenceSession(prefill_qpc_path)
    try:
        for chunk_idx in range(num_chunks):
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][
                :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
            ]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
            ]
            outputs = prefill_session.run(_filter_session_inputs(prefill_session, chunk_inputs))
            _update_retained_states(chunk_inputs, outputs, qeff_model.model.config.text_config.num_hidden_layers)
    finally:
        prefill_session.deactivate()

    first_token = _get_next_token_ids(outputs["logits"])
    decode_inputs = {
        "input_ids": first_token.reshape(BATCH_SIZE, 1),
        "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
        "vision_embeds": chunk_inputs["vision_embeds"],
        "deepstack_features": chunk_inputs["deepstack_features"],
        "image_idx": chunk_inputs["image_idx"],
    }
    _update_retained_states(decode_inputs, outputs, qeff_model.model.config.text_config.num_hidden_layers)

    generated_ids = [first_token]

    decode_session = QAICInferenceSession(decode_qpc_path)
    try:
        decode_outputs = decode_session.run(_filter_session_inputs(decode_session, decode_inputs))
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))

        position_ids = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
        loop_decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": position_ids,
            "vision_embeds": decode_inputs["vision_embeds"],
            "deepstack_features": decode_inputs["deepstack_features"],
            "image_idx": decode_inputs["image_idx"],
        }
        _update_retained_states(
            loop_decode_inputs, decode_outputs, qeff_model.model.config.text_config.num_hidden_layers
        )

        for _ in range(GENERATION_LEN - 2):
            decode_outputs = decode_session.run(_filter_session_inputs(decode_session, loop_decode_inputs))
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
    finally:
        decode_session.deactivate()

    return np.stack(generated_ids, axis=1)


def _build_blocking_qaic_config(blocking_mode: str) -> dict:
    qaic_config = {"blocking_mode": blocking_mode, "ctx_len": CTX_LEN}
    if blocking_mode in ("h", "hqkv"):
        qaic_config["head_block_size"] = HEAD_BLOCK_SIZE
    if blocking_mode in ("kv", "kv_headpar", "qkv", "hqkv"):
        qaic_config["num_kv_blocks"] = NUM_KV_BLOCKS
    if blocking_mode in ("q", "qkv", "hqkv"):
        qaic_config["num_q_blocks"] = NUM_Q_BLOCKS
    return qaic_config


def _build_decode_qaic_config(prefill_blocking_mode: str) -> dict:
    decode_blocking_mode = prefill_blocking_mode.replace("q", "")
    if not decode_blocking_mode:
        return {"ctx_len": CTX_LEN}
    return _build_blocking_qaic_config(decode_blocking_mode)


def _run_disagg_blocked(
    manual_cleanup,
    qaic_config: dict,
    prefill_qaic_config: dict | None = None,
) -> None:
    torch.manual_seed(42)
    manual_cleanup, compile_only = resolve_two_phase_cleanup(manual_cleanup)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32")).to(dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    messages = _prepare_messages()
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

    compiled_onnx_paths = {}
    try:
        qaic_config["ctx_len"] = CTX_LEN
        effective_prefill_qaic_config = copy.deepcopy(
            prefill_qaic_config if prefill_qaic_config is not None else qaic_config
        )
        with model_export_compile_lock(MODEL_NAME):
            prefill_qpc_path = qeff_model.compile(
                batch_size=BATCH_SIZE,
                prefill_seq_len=PREFILL_SEQ_LEN,
                ctx_len=CTX_LEN,
                height=IMAGE_SIZE[1],
                width=IMAGE_SIZE[0],
                num_cores=DEFAULT_AIC_NUM_CORES,
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
                qaic_config=effective_prefill_qaic_config,
            )
            compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")
            _assert_lang_only_compile(qeff_model, prefill_qpc_path, "lang_prefill_qpc_path")

            decode_qpc_path = qeff_model.compile(
                batch_size=BATCH_SIZE,
                prefill_seq_len=1,
                ctx_len=CTX_LEN,
                height=IMAGE_SIZE[1],
                width=IMAGE_SIZE[0],
                num_cores=DEFAULT_AIC_NUM_CORES,
                num_devices=1,
                split_model_io=True,
                mos=1,
                aic_enable_depth_first=True,
                prefill_only=False,
                skip_vision=True,
                use_onnx_subfunctions=True,
                layerwise=False,
                layerwise_window_size=1,
                qaic_config=copy.deepcopy(qaic_config),
            )
            compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")
            _assert_lang_only_compile(qeff_model, decode_qpc_path, "lang_decode_qpc_path")
            _assert_distinct_onnx_paths(compiled_onnx_paths)
            print(f"Disagg blocked lang-only ONNX paths: {compiled_onnx_paths}")

        if compile_only:
            return

        qaic_tokens = _run_disagg_qaic_generation(
            qeff_model=qeff_model,
            processor=processor,
            common_inputs=common_inputs,
            prefill_qpc_path=prefill_qpc_path.get("lang_prefill_qpc_path"),
            decode_qpc_path=decode_qpc_path.get("lang_decode_qpc_path"),
        )
    finally:
        manual_cleanup(list(compiled_onnx_paths.values()))

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert hf_tokens is not None
    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert np.issubdtype(hf_tokens.dtype, np.integer)
    assert (qaic_tokens == hf_tokens).all(), (
        "Tokens don't match for HF Torch fp32 output and disagg blocked QAIC output"
    )


def _build_prefill_mdp_qaic_config(blocking_mode: str) -> dict:
    qaic_config = {
        "blocking_mode": blocking_mode,
        "ctx_len": CTX_LEN,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": PREFILL_EXPERT_PARALLEL_CHUNK_SIZE,
        },
    }
    if blocking_mode == "prefill_online":
        qaic_config.update(
            {
                "num_kv_blocks": NUM_KV_BLOCKS,
                "num_q_blocks": -(-PREFILL_SEQ_LEN // PREFILL_ONLINE_QL_CHUNK),
                "n_rep_chunk": PREFILL_ONLINE_N_REP_CHUNK,
            }
        )
    else:
        qaic_config.update(
            {
                "head_block_size": HEAD_BLOCK_SIZE,
                "num_kv_blocks": NUM_KV_BLOCKS,
                "num_q_blocks": NUM_Q_BLOCKS,
            }
        )
    return qaic_config


def _load_qeff_model(num_hidden_layers: int = 1):
    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32", num_hidden_layers=num_hidden_layers)).to(
        dtype=torch.float32
    )
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32

    return QEFFAutoModelForImageTextToText(
        hf_model,
        kv_offload=True,
        config=hf_model.config,
        torch_dtype=torch.float32,
        layerwise=False,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.disagg_dma
@pytest.mark.parametrize("blocking_mode", ["h", "q", "kv", "qkv"])
def test_qwen3_vl_moe_disagg_blocked_qaic_vs_hf_fp32(blocking_mode, manual_cleanup):
    _run_disagg_blocked(
        manual_cleanup,
        _build_decode_qaic_config(blocking_mode),
        prefill_qaic_config=_build_blocking_qaic_config(blocking_mode),
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.disagg_dma
@pytest.mark.parametrize("blocking_mode", ["prefill_qkv", "prefill_online"])
def test_qwen3_vl_moe_disagg_prefill_mdp_intersection_compile_only(blocking_mode, manual_cleanup):
    torch.manual_seed(42)
    manual_cleanup, _ = resolve_two_phase_cleanup(manual_cleanup, compile_only=True)
    qeff_model = _load_qeff_model(num_hidden_layers=PREFILL_MDP_NUM_LAYERS)
    compiled_onnx_paths = {}

    try:
        with model_export_compile_lock(MODEL_NAME):
            qpc_paths = qeff_model.compile(
                batch_size=BATCH_SIZE,
                prefill_seq_len=PREFILL_SEQ_LEN,
                ctx_len=CTX_LEN,
                height=IMAGE_SIZE[1],
                width=IMAGE_SIZE[0],
                num_cores=DEFAULT_AIC_NUM_CORES,
                num_devices=PREFILL_MDP_NUM_DEVICES,
                mdp_num_partitions=PREFILL_MDP_NUM_PARTITIONS,
                mdp_strategy="intersection",
                retain_full_kv=True,
                split_model_io=True,
                mos=1,
                aic_enable_depth_first=True,
                prefill_only=True,
                enable_chunking=True,
                skip_vision=True,
                # FIXME: Re-enable subfunctions once MDP intersection handles compiler dump names
                # emitted for ONNX subfunction graphs.
                use_onnx_subfunctions=False,
                layerwise=False,
                layerwise_window_size=1,
                qaic_config=_build_prefill_mdp_qaic_config(blocking_mode),
            )
            compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")
            _assert_lang_only_compile(qeff_model, qpc_paths, "lang_prefill_qpc_path")
    finally:
        manual_cleanup(list(compiled_onnx_paths.values()))
