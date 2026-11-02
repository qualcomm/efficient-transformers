# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Token-level parity tests for the Qwen3.5-MoE disaggregated prefill/decode DMA path."""

import copy
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from tests.transformers.disaggregated._disagg_dma_config import disagg_dma_configs

MODEL_NAME = "tiny-random/qwen3.5-moe"
TINY_RANDOM_MODEL_NAMES = {"tiny-random/qwen3.5-moe"}
MDP_PARTITION_XFAIL_REASON = (
    "QAIC compilation rejects the generated two-partition MDP config because conv_state.2 is unassigned"
)


def _optional_int_env(name: str, default: int | None) -> int | None:
    """Read an optional int from the environment; empty / unset -> ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


# Optional depth truncation: set to an int to run a shallow model
NUM_HIDDEN_LAYERS = _optional_int_env("QEFF_QWEN35_NUM_HIDDEN_LAYERS", default=4)
VISION_DEPTH = _optional_int_env("QEFF_QWEN35_VISION_DEPTH", default=2)
PREFILL_SEQ_LEN = 64
CTX_LEN = 1024
BATCH_SIZE = 1
FULL_BATCH_SIZE = 4
GENERATION_LEN = 40
FULL_BATCH_GENERATION_LEN = 4
IMAGE_SIZE = (536, 354)
TEXT_PROMPT = "Describe all the colors seen in the image."
FULL_BATCH_TEXT_PROMPTS = (
    "Name the four seasons of the year.",
    "What is the capital of France?",
    "Write a short greeting.",
    "Name two primary colors.",
)

VISION_INPUT_KEYS = {
    "pixel_values",
    "image_grid_thw",  # required by the vision ONNX graph
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUTS = ("vision_embeds",)


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_distinct_onnx_paths(onnx_paths: dict[str, Path]):
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def _load_hf_model_from_pretrained(config, dtype: str = "float32", model_name: str = MODEL_NAME):
    from_pretrained_kwargs = {"config": config} if config is not None else {}
    torch_dtype = getattr(config, "torch_dtype", None) if config is not None else None
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is None:
        torch_dtype = getattr(torch, dtype)

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **from_pretrained_kwargs,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **from_pretrained_kwargs,
        )
    model.eval()
    return model


def _apply_reduced_layer_config(config, num_lang_layers: int | None, num_vision_layers: int | None):
    """prefer a ``num_lang_layers`` that already spans a
    full attention layer (e.g. a multiple of ``full_attention_interval``).
    """
    text_config = config.text_config if hasattr(config, "text_config") else config

    if num_lang_layers is not None:
        text_config.num_hidden_layers = num_lang_layers
        if getattr(text_config, "layer_types", None):
            layer_types = list(text_config.layer_types[:num_lang_layers])
            if "full_attention" not in layer_types:
                warnings.warn(
                    f"Truncating qwen3.5-moe to {num_lang_layers} layers yields a layer_types "
                    f"slice with no full_attention layer: {layer_types}. Forcing the last layer "
                    "to full_attention so the DMA KV-share handoff has a full-KV family to slice "
                    "into. This slice is NOT representative of the original model -- choose a "
                    f"num_lang_layers that spans a full_attention layer (a multiple of "
                    f"full_attention_interval={getattr(text_config, 'full_attention_interval', '?')}) "
                    "to keep the reduced model faithful.",
                    stacklevel=2,
                )
                layer_types[-1] = "full_attention"
            text_config.layer_types = layer_types

    if num_vision_layers is not None and hasattr(config, "vision_config"):
        config.vision_config.depth = num_vision_layers

    return config


def _build_config(dtype: str = "float16", full_model: bool = False, model_name: str = MODEL_NAME):
    """Load the real config; optionally truncate depth (see ``_apply_reduced_layer_config``)."""
    if model_name in TINY_RANDOM_MODEL_NAMES:
        return None

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)

    if not full_model:
        config = _apply_reduced_layer_config(
            config,
            num_lang_layers=NUM_HIDDEN_LAYERS,
            num_vision_layers=VISION_DEPTH,
        )

    text_config = config.text_config if hasattr(config, "text_config") else config
    text_config.dtype = dtype
    text_config.torch_dtype = getattr(torch, dtype)
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


def _prepare_text_messages(prompts: tuple[str, ...]) -> list:
    return [[{"role": "user", "content": [{"type": "text", "text": prompt}]}] for prompt in prompts]


def _prepare_processor_inputs(processor: AutoProcessor, messages: list) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info

    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"))


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _last_step_logits(logits: np.ndarray) -> np.ndarray:
    """The (B, vocab) logit slice that produced the next token for one step."""
    return np.asarray(logits, dtype=np.float32)[:, -1, :]


def _assert_hf_qaic_tokens_match(
    hf_tokens: np.ndarray,
    qaic_tokens: np.ndarray,
    processor: AutoProcessor,
    batch_size: int,
    generation_len: int = GENERATION_LEN,
) -> None:
    assert qaic_tokens.shape == (batch_size, generation_len)
    assert hf_tokens.shape == (batch_size, generation_len)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert np.issubdtype(hf_tokens.dtype, np.integer)

    matches = hf_tokens == qaic_tokens
    num_matched = int(matches.all(axis=0).cumprod().sum())
    qaic_text = processor.tokenizer.batch_decode(qaic_tokens, skip_special_tokens=True)
    hf_text = processor.tokenizer.batch_decode(hf_tokens, skip_special_tokens=True)
    print(f"Disagg QAIC DMA tokens : {qaic_tokens.tolist()}")
    print(f"Disagg QAIC DMA text   : {qaic_text}")
    print(f"HF Torch fp32 tokens   : {hf_tokens.tolist()}")
    print(f"HF Torch fp32 text     : {hf_text}")
    print(f"Matched leading tokens : {num_matched}/{generation_len}")

    if not matches.all():
        first_mismatch = int(np.argmin(matches.all(axis=0)))
        raise AssertionError(
            "Tokens don't match for HF Torch fp32 output and disagg QAIC DMA output; "
            f"first mismatch at token index {first_mismatch} "
            f"(matched {num_matched}/{generation_len} leading tokens): "
            f"HF={hf_tokens[:, first_mismatch].tolist()} vs QAIC={qaic_tokens[:, first_mismatch].tolist()}"
        )


def _run_hf_torch_fp32(
    model,
    processor: AutoProcessor,
    messages: list,
    collect_logits: bool = False,
    processor_inputs: dict | None = None,
    generation_len: int = GENERATION_LEN,
):
    model = model.to(dtype=torch.float32).eval()
    inputs = (
        _prepare_processor_inputs(processor, messages)
        if processor_inputs is None
        else {
            name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for name, value in processor_inputs.items()
        }
    )
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_len,
            min_new_tokens=generation_len,
            do_sample=False,
            temperature=None,
            top_p=None,
            output_logits=collect_logits,
            return_dict_in_generate=collect_logits,
        )

    if collect_logits:
        sequences = outputs.sequences
        step_logits = np.stack([step.float().cpu().numpy() for step in outputs.logits], axis=1)
        prompt_len = inputs["input_ids"].shape[-1]
        return sequences[:, prompt_len:].detach().cpu().numpy(), step_logits

    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _run_disagg_kv_share_qaic_generation(
    qeff_model: QEFFAutoModelForImageTextToText,
    processor: AutoProcessor,
    common_inputs: dict,
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
    collect_logits: bool = False,
):
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

    vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUT_KEYS}
    vision_inputs.update(
        {name: vision_inputs[name].astype("float16") for name in VISION_FP16_KEYS if name in vision_inputs}
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

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    vision_persist = {name: vision_outputs[name] for name in VISION_OUTPUTS if name in vision_outputs}
    prefill_session.set_persistent_inputs(vision_persist)
    decode_session.set_persistent_inputs(
        {name: value for name, value in vision_persist.items() if name in decode_session.binding_index_map}
    )

    # Hybrid: kv_cache_info carries mixed 4-D (full) and 3-D (linear) shapes.
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]

    chunk_inputs = dict(lang_inputs)
    exec_idx = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        last_chunk = chunk_idx == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    generated_ids = [_get_next_token_ids(prefill_out["logits"])]
    step_logits = [_last_step_logits(prefill_out["logits"])] if collect_logits else None

    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    num_pos_sections = lang_inputs["position_ids"].shape[0]
    phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
    mrope_pos = int(lang_inputs["position_ids"][1:].max()) + 1

    def _decode_position_ids(next_phys: int, next_mrope: int) -> np.ndarray:
        pos = np.empty((num_pos_sections, BATCH_SIZE, 1), dtype=np.int64)
        pos[0] = next_phys
        pos[1:] = next_mrope
        return pos

    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": _decode_position_ids(phys_pos, mrope_pos),
    }
    if decode_has_image_idx:
        decode_inputs["image_idx"] = prefill_out["image_idx_output"]

    for _ in range(GENERATION_LEN - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_outputs = decode_session.get_outputs(index=exec_idx)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        if collect_logits:
            step_logits.append(_last_step_logits(decode_outputs["logits"]))
        phys_pos += 1
        mrope_pos += 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": _decode_position_ids(phys_pos, mrope_pos),
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    tokens = np.stack(generated_ids, axis=1)
    if collect_logits:
        # (B, GEN, vocab) aligned with tokens: index 0 is the prefill/first-token logits.
        return tokens, np.stack(step_logits, axis=1)
    return tokens


def _run_disagg_cb_kv_share_qaic_generation(
    qeff_model: QEFFAutoModelForImageTextToText,
    common_inputs: list[dict],
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
    batch_size: int,
    generation_len: int,
) -> np.ndarray:
    assert "batch_index" in prefill_session.binding_index_map
    assert "batch_index" in decode_session.binding_index_map

    kv_caches = [np.zeros(shape, dtype=dtype) for shape, dtype in decode_session.kv_cache_info]
    assert kv_caches and kv_caches[0].shape[0] == batch_size
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    first_tokens = np.empty(batch_size, dtype=np.int64)
    phys_pos = np.empty((batch_size, 1), dtype=np.int64)
    mrope_pos = np.empty((batch_size, 1), dtype=np.int64)
    num_pos_sections = None

    assert len(common_inputs) == batch_size, "Expected one independently processed prompt per full-batch slot"
    for slot, common_input in enumerate(common_inputs):
        slot_inputs = {
            name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for name, value in common_input.items()
        }
        slot_inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=slot_inputs,
            prefill_seq_len=PREFILL_SEQ_LEN,
            batch_size=1,
        )

        pad_token_id = getattr(qeff_model.model.config, "pad_token_id", None) or 1
        input_ids_length = slot_inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
        padded_len = num_chunks * PREFILL_SEQ_LEN
        slot_inputs["input_ids"] = torch.nn.functional.pad(
            slot_inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            pad_token_id,
        )
        slot_inputs["attention_mask"] = torch.nn.functional.pad(
            slot_inputs["attention_mask"],
            (0, padded_len - input_ids_length),
            "constant",
            0,
        )
        slot_inputs = {name: np.asarray(value) for name, value in slot_inputs.items()}
        vision_inputs = {name: value for name, value in slot_inputs.items() if name in VISION_INPUT_KEYS}
        assert not vision_inputs, f"Text-only generation unexpectedly produced vision inputs: {vision_inputs.keys()}"

        lang_inputs = dict(slot_inputs)
        if "position_ids" in lang_inputs:
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(
                lang_inputs.pop("attention_mask"),
                np.arange(padded_len),
                -1,
            )
        lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        lang_inputs["batch_index"] = np.array([[slot]], dtype=np.int64)

        slot_kv_view = [kv_cache[slot : slot + 1] for kv_cache in kv_caches]
        chunk_inputs = dict(lang_inputs)
        exec_idx = None
        for chunk_idx in range(num_chunks):
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][
                :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
            ]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
            ]
            last_chunk = chunk_idx == num_chunks - 1
            exec_idx = prefill_session.np_run_pipeline(
                chunk_inputs,
                last_chunk=last_chunk,
                kv_cache_buffers=slot_kv_view if last_chunk else None,
            )
            prefill_session.complete_inf(exec_idx, is_prefill=True)
            chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]

        prefill_outputs = prefill_session.get_outputs(index=exec_idx)
        first_tokens[slot] = _get_next_token_ids(prefill_outputs["logits"])[0]
        position_ids = lang_inputs["position_ids"]
        num_pos_sections = position_ids.shape[0]
        phys_pos[slot, 0] = int(position_ids[0].max()) + 1
        mrope_pos[slot, 0] = int(position_ids[1:].max()) + 1 if num_pos_sections > 1 else phys_pos[slot, 0]

    def _decode_position_ids() -> np.ndarray:
        position_ids = np.empty((num_pos_sections, batch_size, 1), dtype=np.int64)
        position_ids[0] = phys_pos
        position_ids[1:] = mrope_pos
        return position_ids

    batch_index = np.arange(batch_size, dtype=np.int64).reshape(batch_size, 1)
    generated_ids = [first_tokens]
    for _ in range(generation_len - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(batch_size, 1),
            "position_ids": _decode_position_ids(),
            "batch_index": batch_index,
        }
        if "image_idx" in decode_session.binding_index_map:
            decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_outputs = decode_session.get_outputs(index=exec_idx)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        phys_pos += 1
        mrope_pos += 1

    return np.stack(generated_ids, axis=1)


def _wrap_qeff_model(hf_model, continuous_batching: bool = False) -> QEFFAutoModelForImageTextToText:
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    return QEFFAutoModelForImageTextToText(
        hf_model,
        attn_implementation="eager",
        kv_offload=True,
        config=hf_model.config,
        dtype=torch.float32,
        layerwise=False,
        continuous_batching=continuous_batching,
    )


def _compile_disagg_sessions(
    qeff_model,
    image,
    sessions: list,
    compiled_onnx_paths: dict,
    *,
    vision_num_devices: int = 1,
    prefill_num_devices: int = 2,
    decode_num_devices: int = 2,
    stages: int = 2,
    use_onnx_subfunctions: bool = True,
    prefill_qaic_config: dict | None = None,
):
    vision_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=vision_num_devices,
        mos=1,
        aic_enable_depth_first=True,
        skip_vision=False,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=use_onnx_subfunctions,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")
    decode_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=decode_num_devices,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        split_retained_state_io=True,
        mos=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=use_onnx_subfunctions,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")

    prefill_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=prefill_num_devices,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        mdp_num_partitions=stages,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=use_onnx_subfunctions,
        layerwise=False,
        qaic_config=prefill_qaic_config,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")
    _assert_distinct_onnx_paths(compiled_onnx_paths)
    print(f"Disagg ONNX paths: {compiled_onnx_paths}")

    vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))
    prefill_session = QAICInferenceSession(
        prefill_qpc_path.get("lang_prefill_qpc_path"), kv_dma_share=True, cluster_id="prefill"
    )
    decode_session = QAICInferenceSession(
        decode_qpc_path.get("lang_decode_qpc_path"), kv_dma_share=True, cluster_id="decode"
    )
    sessions.extend([vision_session, prefill_session, decode_session])
    return vision_session, prefill_session, decode_session


def _compile_gdn_full_batch_sessions(qeff_model, sessions: list, compiled_onnx_paths: dict):
    decode_qpc_path = qeff_model.compile(
        batch_size=FULL_BATCH_SIZE,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        height=IMAGE_SIZE[1],
        width=IMAGE_SIZE[0],
        num_cores=16,
        num_devices=2,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
        qaic_config={"gdn_full_state_update": True},
        full_batch_size=FULL_BATCH_SIZE,
        kv_cache_batch_size=FULL_BATCH_SIZE,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")

    prefill_qpc_path = qeff_model.compile(
        batch_size=1,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=IMAGE_SIZE[1],
        width=IMAGE_SIZE[0],
        num_cores=16,
        num_devices=1,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        full_batch_size=FULL_BATCH_SIZE,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")
    _assert_distinct_onnx_paths(compiled_onnx_paths)
    print(f"Disagg ONNX paths: {compiled_onnx_paths}")

    prefill_session = QAICInferenceSession(
        prefill_qpc_path.get("lang_prefill_qpc_path"),
        kv_dma_share=True,
        full_batch_size=FULL_BATCH_SIZE,
        cluster_id="prefill",
    )
    decode_session = QAICInferenceSession(
        decode_qpc_path.get("lang_decode_qpc_path"),
        kv_dma_share=True,
        full_batch_size=FULL_BATCH_SIZE,
        cluster_id="decode",
    )
    sessions.extend([prefill_session, decode_session])
    return prefill_session, decode_session


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.disagg_dma
def test_qwen3_5_gdn_full_batch_decode_qaic_vs_hf_fp32(manual_cleanup):
    pytest.importorskip("qwen_vl_utils")

    hf_model = _load_hf_model_from_pretrained(config=None)
    assert hf_model.config.text_config.layer_types == ["linear_attention"] * 3 + ["full_attention"]
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    messages = _prepare_text_messages(FULL_BATCH_TEXT_PROMPTS)
    common_inputs = [_prepare_processor_inputs(processor, [message]) for message in messages]
    hf_tokens = np.concatenate(
        [
            _run_hf_torch_fp32(
                hf_model,
                processor,
                [message],
                processor_inputs=inputs,
                generation_len=FULL_BATCH_GENERATION_LEN,
            )
            for message, inputs in zip(messages, common_inputs)
        ]
    )
    qeff_model = _wrap_qeff_model(hf_model, continuous_batching=True)

    sessions = []
    compiled_onnx_paths = {}
    try:
        prefill_session, decode_session = _compile_gdn_full_batch_sessions(
            qeff_model,
            sessions,
            compiled_onnx_paths,
        )
        qaic_tokens = _run_disagg_cb_kv_share_qaic_generation(
            qeff_model=qeff_model,
            common_inputs=common_inputs,
            prefill_session=prefill_session,
            decode_session=decode_session,
            batch_size=FULL_BATCH_SIZE,
            generation_len=FULL_BATCH_GENERATION_LEN,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [getattr(qeff_model.lang_model, "onnx_path", None)]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    _assert_hf_qaic_tokens_match(
        hf_tokens,
        qaic_tokens,
        processor,
        FULL_BATCH_SIZE,
        generation_len=FULL_BATCH_GENERATION_LEN,
    )


@pytest.mark.on_qaic
@pytest.mark.disagg_dma
@pytest.mark.xfail(reason=MDP_PARTITION_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("dma_config", disagg_dma_configs("qwen3_5_moe_tiny"))
def test_qwen3_5_disagg_kv_share_qaic_vs_hf_fp32(manual_cleanup, dma_config):
    pytest.importorskip("qwen_vl_utils")
    torch.manual_seed(42)

    model_id = dma_config["model_id"]
    use_onnx_subfunctions = dma_config.get("use_onnx_subfunctions", True)
    skip_hf_reference = dma_config.get("skip_hf_reference", False)
    blocking_mode = dma_config.get("blocking_mode")
    prefill_qaic_config = None
    if blocking_mode == "kv":
        prefill_qaic_config = {
            "enable_blocking": True,
            "blocking_mode": blocking_mode,
            "num_kv_blocks": 4,
            "skip_kv": True,
        }

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32", model_name=model_id), model_name=model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))
    messages = _prepare_messages(image)
    common_inputs = _prepare_processor_inputs(processor, messages)
    hf_tokens = (
        None if skip_hf_reference else _run_hf_torch_fp32(hf_model, processor, messages, processor_inputs=common_inputs)
    )

    qeff_model = _wrap_qeff_model(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_session, prefill_session, decode_session = _compile_disagg_sessions(
            qeff_model,
            image,
            sessions,
            compiled_onnx_paths,
            vision_num_devices=dma_config["vision_num_devices"],
            prefill_num_devices=dma_config["prefill_num_devices"],
            decode_num_devices=dma_config["decode_num_devices"],
            stages=dma_config["stages"],
            use_onnx_subfunctions=use_onnx_subfunctions,
            prefill_qaic_config=prefill_qaic_config,
        )

        qaic_tokens = _run_disagg_kv_share_qaic_generation(
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

    if skip_hf_reference:
        assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
        assert np.issubdtype(qaic_tokens.dtype, np.integer)
        return

    _assert_hf_qaic_tokens_match(hf_tokens, qaic_tokens, processor, BATCH_SIZE)
