# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Token-level parity tests for the Qwen3-VL-MoE disaggregated prefill/decode DMA path.

Run the regular HF/QAIC parity test with:
    pytest -m "on_qaic and disagg_dma" \
        tests/transformers/disaggregated/test_qwen3_vl_moe_disagg_kv_share_w_hf_fp32.py
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from tests.transformers.disaggregated._disagg_dma_config import disagg_dma_configs

MODEL_NAME = "tiny-random/qwen3-vl-moe"
TINY_RANDOM_MODEL_NAMES = {"tiny-random/qwen3-vl-moe"}
PREFILL_SEQ_LEN = 64
CTX_LEN = 4096
BATCH_SIZE = 1
GENERATION_LEN = 40
# IMAGE_SIZE = (1600, 1056)
IMAGE_SIZE = (536, 354)
TEXT_PROMPT = "Describe this image."

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


def _normalize_text_layer_types(config):
    text_config = config.text_config if hasattr(config, "text_config") else config
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    num_layers = getattr(text_config, "num_hidden_layers", len(layer_types))
    if layer_types:
        if len(layer_types) < num_layers:
            layer_types.extend(["full_attention"] * (num_layers - len(layer_types)))
        text_config.layer_types = layer_types[:num_layers]
    return config


def _build_config(dtype: str = "float32", full_model: bool = False, model_name: str = MODEL_NAME):
    if model_name in TINY_RANDOM_MODEL_NAMES:
        return None

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    if not full_model:
        config.vision_config.depth = 9
        config.text_config.num_hidden_layers = 2
        config.vision_config.deepstack_visual_indexes = [8]
    return _normalize_text_layer_types(config)


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


def _run_hf_torch_fp32(
    model, processor: AutoProcessor, messages: list, processor_inputs: dict | None = None
) -> np.ndarray:
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
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Shared QEff model / input preparation
# ---------------------------------------------------------------------------


def _build_qeff_model(hf_model) -> QEFFAutoModelForImageTextToText:
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


def _run_disagg_kv_share_qaic_generation(
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

    # ---- Vision (producer): run the vision QPC once; its outputs feed every lang step ----
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

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    vision_persist = {name: vision_outputs[name] for name in VISION_OUTPUTS if name in vision_outputs}
    prefill_session.set_persistent_inputs(vision_persist)
    decode_session.set_persistent_inputs(
        {name: value for name, value in vision_persist.items() if name in decode_session.binding_index_map}
    )
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

    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    num_pos_sections = lang_inputs["position_ids"].shape[0]
    phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
    mrope_pos = int(lang_inputs["position_ids"][1:].max()) + 1 if num_pos_sections > 1 else phys_pos

    def _decode_position_ids(next_phys: int, next_mrope: int) -> np.ndarray:
        pos = np.empty((num_pos_sections, BATCH_SIZE, 1), dtype=np.int64)
        pos[0] = next_phys
        if num_pos_sections > 1:
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
        phys_pos += 1
        mrope_pos += 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": _decode_position_ids(phys_pos, mrope_pos),
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    return np.stack(generated_ids, axis=1)


@pytest.mark.on_qaic
@pytest.mark.disagg_dma
@pytest.mark.parametrize("dma_config", disagg_dma_configs("qwen3_vl_moe_tiny"))
def test_qwen3_vl_moe_disagg_kv_share_qaic_vs_hf_fp32(manual_cleanup, dma_config):
    pytest.importorskip("qwen_vl_utils")
    torch.manual_seed(42)

    model_id = dma_config["model_id"]
    use_onnx_subfunctions = dma_config.get("use_onnx_subfunctions", True)
    skip_hf_reference = dma_config.get("skip_hf_reference", False)
    mdp_strategy = dma_config.get("mdp_strategy", "onnx")
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
    hf_tokens = None if skip_hf_reference else _run_hf_torch_fp32(hf_model, processor, messages, common_inputs)
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    qeff_model = _build_qeff_model(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_qpc_path = qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            height=image.height,
            width=image.width,
            num_cores=DEFAULT_AIC_NUM_CORES,
            num_devices=dma_config["vision_num_devices"],
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
            num_cores=DEFAULT_AIC_NUM_CORES,
            num_devices=dma_config["decode_num_devices"],
            retain_full_kv=True,
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
            num_cores=DEFAULT_AIC_NUM_CORES,
            num_devices=dma_config["prefill_num_devices"],
            retain_full_kv=True,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            split_retained_state_io=True,
            mos=1,
            aic_enable_depth_first=True,
            mdp_num_partitions=dma_config["stages"],
            mdp_strategy=mdp_strategy,
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

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    print(f"Disagg QAIC DMA tokens : {qaic_tokens.tolist()}")
    print("QAIC ouptut tokens     : ")
    print(processor.tokenizer.batch_decode(qaic_tokens))

    if skip_hf_reference:
        return

    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(hf_tokens.dtype, np.integer)

    matches = hf_tokens == qaic_tokens
    num_matched = int(matches.all(axis=0).cumprod().sum())  # leading run matched across all rows
    print(f"HF Torch fp32 tokens   : {hf_tokens.tolist()}")
    print("HF output token:")
    print(processor.tokenizer.batch_decode(hf_tokens))
    print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")

    if not matches.all():
        first_mismatch = int(np.argmin(matches.all(axis=0)))
        raise AssertionError(
            "Tokens don't match for HF Torch fp32 output and disagg QAIC DMA output; "
            f"first mismatch at token index {first_mismatch} "
            f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
            f"HF={hf_tokens[:, first_mismatch].tolist()} vs QAIC={qaic_tokens[:, first_mismatch].tolist()}"
        )
