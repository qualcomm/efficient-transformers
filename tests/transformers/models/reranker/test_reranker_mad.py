# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoProcessor

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    format_mm_content as _shared_format_mm_content,
)
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    format_mm_instruction as _shared_format_mm_instruction,
)
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    get_yes_no_token_ids as _shared_get_yes_no_token_ids,
)
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    resolve_model_source as _shared_resolve_model_source,
)
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    score_from_logits as _shared_score_from_logits,
)
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    tokenize_pair as _shared_tokenize_pair,
)
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    truncate_tokens_optimized as _shared_truncate_tokens_optimized,
)
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from QEfficient.utils.test_utils import load_vlm_model, set_num_layers_vlm

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")

PT_AI100_MAD_MAX = 5e-3
MAX_LENGTH = 8192
RERANKER_DOC_LIMIT = int(os.getenv("QEFF_RERANKER_DOC_LIMIT", "0"))

IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1280 * IMAGE_FACTOR * IMAGE_FACTOR

EXAMPLE_INPUTS = {
    "instruction": "Retrieve relevant content.",
    "query": {"text": "dog on beach"},
    "documents": [
        {"image": "https://picsum.photos/id/237/536/354"},
        {"text": "A dog running on the beach."},
    ],
}

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    reranker_models = config_data["image_text_reranker_models"]

test_reranker_models = [model_config["model_name"] for model_config in reranker_models]
reranker_model_config_dict = {model["model_name"]: model for model in reranker_models}


def _resolve_model_source(model_name_or_path: str) -> str:
    return _shared_resolve_model_source(model_name_or_path)


def _format_mm_content(text, image, video, prefix: str) -> List[Dict]:
    return _shared_format_mm_content(
        text=text,
        image=image,
        video=video,
        prefix=prefix,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        unsupported_video_error="Video input is not supported in this test.",
    )


def _format_mm_instruction(instruction: str, query: Dict, document: Dict) -> List[Dict]:
    return _shared_format_mm_instruction(
        instruction=instruction,
        query=query,
        document=document,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        unsupported_video_error="Video input is not supported in this test.",
    )


def _truncate_tokens_optimized(tokens: List[int], max_length: int, special_tokens: List[int]) -> List[int]:
    return _shared_truncate_tokens_optimized(tokens, max_length, special_tokens)


def _tokenize_pair(processor, pair: List[Dict]) -> Dict:
    return _shared_tokenize_pair(processor, pair, MAX_LENGTH)


def _get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    return _shared_get_yes_no_token_ids(tokenizer)


def _score_from_logits(logits, yes_token_id: int, no_token_id: int) -> np.ndarray:
    score = _shared_score_from_logits(logits, yes_token_id, no_token_id)
    return score.detach().cpu().numpy().astype(np.float64)


def _score_from_last_hidden(last_hidden_state: torch.Tensor, score_linear: torch.nn.Linear) -> np.ndarray:
    score = torch.sigmoid(score_linear(last_hidden_state[:, -1])).squeeze(-1)
    return score.detach().to(torch.float32).cpu().numpy().astype(np.float64)


def _make_score_linear(model_hf, yes_token_id: int, no_token_id: int) -> torch.nn.Linear:
    lm_head_weights = model_hf.lm_head.weight.data
    weight_yes = lm_head_weights[yes_token_id]
    weight_no = lm_head_weights[no_token_id]

    linear_layer = torch.nn.Linear(weight_yes.shape[0], 1, bias=False)
    with torch.no_grad():
        linear_layer.weight[0] = weight_yes - weight_no
    return linear_layer.eval()


def _mad_stats(reference: np.ndarray, candidate: np.ndarray) -> Tuple[float, float]:
    diff = np.abs(reference - candidate)
    return float(np.mean(diff)), float(np.max(diff))


def _prepare_qeff_inputs(qeff_model, tokenized_inputs: Dict, prefill_seq_len: int = None):
    runtime_prompt_len = int(tokenized_inputs["input_ids"].shape[1])
    effective_prefill_seq_len = runtime_prompt_len if prefill_seq_len is None else prefill_seq_len
    if effective_prefill_seq_len < runtime_prompt_len:
        raise ValueError(
            f"prefill_seq_len ({effective_prefill_seq_len}) must be >= runtime prompt length ({runtime_prompt_len})."
        )

    prepared_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=tokenized_inputs,
        prefill_seq_len=effective_prefill_seq_len,
        batch_size=1,
    )

    if "image_grid_thw" in prepared_inputs and prepared_inputs["image_grid_thw"].ndim == 2:
        thw = prepared_inputs["image_grid_thw"][0]
        t, h, w = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
        prepared_inputs["image_grid_thw"] = torch.zeros((1, t, h, w), dtype=thw.dtype)

    if "pixel_values" in prepared_inputs:
        prepared_inputs["pixel_values"] = prepared_inputs["pixel_values"].to(torch.float32)

    return prepared_inputs, runtime_prompt_len


def _zero_vision_outputs(vision_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: np.zeros_like(value) for name, value in vision_outputs.items()}


def _run_ai100_vision(vision_qpc_path: str, prepared_inputs) -> Dict[str, np.ndarray]:
    vision_session = QAICInferenceSession(vision_qpc_path)
    vision_inputs = {
        "pixel_values": prepared_inputs["pixel_values"].detach().cpu().numpy().astype(np.float16),
        "image_grid_thw": prepared_inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64),
    }
    vision_outputs = vision_session.run(vision_inputs)
    vision_session.deactivate()
    return vision_outputs


def _run_ai100_prefill(qpc_paths, prepared_inputs, vision_template):
    if not isinstance(qpc_paths, dict):
        raise ValueError("Expected qpc_paths to be a dict with vision/lang QPC keys.")

    vision_qpc_path = qpc_paths.get("vision_qpc_path")
    lang_qpc_path = qpc_paths.get("lang_qpc_path")
    if vision_qpc_path is None or lang_qpc_path is None:
        raise ValueError("Missing vision_qpc_path/lang_qpc_path in compiled QPC outputs.")

    prefill_len = prepared_inputs["position_ids"].shape[-1]
    input_ids = prepared_inputs["input_ids"]
    if input_ids.shape[1] < prefill_len:
        pad = torch.full(
            (input_ids.shape[0], prefill_len - input_ids.shape[1]),
            1,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids = input_ids[:, :prefill_len]
    position_ids = prepared_inputs["position_ids"][..., :prefill_len]

    if "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
        vision_outputs = _run_ai100_vision(vision_qpc_path, prepared_inputs)
    else:
        vision_outputs = _zero_vision_outputs(vision_template)

    lang_session = QAICInferenceSession(lang_qpc_path)
    lang_session.skip_buffers(
        [
            name
            for name in lang_session.input_names + lang_session.output_names
            if name.startswith("past_") or name.endswith("_RetainedState")
        ]
    )
    lang_session.set_buffers(vision_outputs)
    lang_inputs = {
        "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
        "position_ids": position_ids.detach().cpu().numpy().astype(np.int64),
        "image_idx": np.zeros((1, 1), dtype=np.int64),
    }
    outputs = lang_session.run(lang_inputs)
    lang_session.deactivate()
    return outputs["logits"]


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.regular
@pytest.mark.parametrize("model_name", test_reranker_models)
def test_qwen3_vl_reranker_mad_parity(model_name):
    torch.manual_seed(42)
    model_cfg = reranker_model_config_dict[model_name]
    model_source = _resolve_model_source(model_name)

    config = AutoConfig.from_pretrained(model_source, trust_remote_code=True, padding=True)
    config = set_num_layers_vlm(config, n_layer=model_cfg["num_layers"])
    if hasattr(config, "use_cache"):
        config.use_cache = True
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = True

    model_hf = load_vlm_model(config)
    model_hf.eval()

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_source,
        kv_offload=True,
        config=config,
    )
    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True, padding=True)

    yes_token_id, no_token_id = _get_yes_no_token_ids(processor.tokenizer)
    score_linear = _make_score_linear(model_hf, yes_token_id, no_token_id).to(next(model_hf.parameters()).device)
    score_linear = score_linear.to(dtype=next(model_hf.parameters()).dtype)

    doc_contexts = []
    max_prompt_len = 0
    max_grid_h = 22
    max_grid_w = 34

    hf_scores_list = []

    documents = EXAMPLE_INPUTS["documents"]
    if RERANKER_DOC_LIMIT > 0:
        documents = documents[:RERANKER_DOC_LIMIT]

    for document in documents:
        pair = _format_mm_instruction(
            instruction=EXAMPLE_INPUTS["instruction"],
            query=EXAMPLE_INPUTS["query"],
            document=document,
        )
        tokenized = _tokenize_pair(processor, pair)
        runtime_prompt_len = int(tokenized["input_ids"].shape[1])

        hf_inputs = {}
        for key, value in tokenized.items():
            hf_inputs[key] = value.to(next(model_hf.parameters()).device) if torch.is_tensor(value) else value
        with torch.no_grad():
            hf_last_hidden = model_hf.model(**hf_inputs).last_hidden_state
        hf_score = _score_from_last_hidden(hf_last_hidden, score_linear)[0]
        hf_scores_list.append(float(hf_score))

        if "image_grid_thw" in tokenized and tokenized["image_grid_thw"].numel() > 0:
            grid = tokenized["image_grid_thw"]
            max_grid_h = max(max_grid_h, int(grid[..., 1].max().item()))
            max_grid_w = max(max_grid_w, int(grid[..., 2].max().item()))

        doc_contexts.append(
            {
                "tokenized": tokenized,
            }
        )
        max_prompt_len = max(max_prompt_len, runtime_prompt_len)

    patch_size = int(qeff_model.model.config.vision_config.patch_size)
    compile_height = max_grid_h * patch_size
    compile_width = max_grid_w * patch_size

    qpc_paths = qeff_model.compile(
        img_size=max(compile_height, compile_width),
        height=compile_height,
        width=compile_width,
        prefill_seq_len=max_prompt_len,
        ctx_len=model_cfg["ctx_len"],
        num_devices=1,
        num_cores=DEFAULT_AIC_NUM_CORES,
        mxfp6_matmul=False,
    )

    ai100_scores_list = []

    prepared_contexts = []
    vision_template_ai100 = None
    for context in doc_contexts:
        prepared_inputs, _ = _prepare_qeff_inputs(
            qeff_model=qeff_model,
            tokenized_inputs=context["tokenized"],
            prefill_seq_len=max_prompt_len,
        )
        prepared_contexts.append(
            {
                "prepared_inputs": prepared_inputs,
            }
        )
        if vision_template_ai100 is None and "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
            vision_template_ai100 = _run_ai100_vision(qpc_paths["vision_qpc_path"], prepared_inputs)

    if vision_template_ai100 is None:
        raise ValueError("Expected at least one image document to initialize vision templates.")

    for context in prepared_contexts:
        prepared_inputs_runtime = context["prepared_inputs"]
        ai100_logits = _run_ai100_prefill(
            qpc_paths=qpc_paths,
            prepared_inputs=prepared_inputs_runtime,
            vision_template=vision_template_ai100,
        )
        ai100_score = _score_from_logits(ai100_logits, yes_token_id, no_token_id)[0]
        ai100_scores_list.append(float(ai100_score))

    hf_scores = np.array(hf_scores_list, dtype=np.float64)
    ai100_scores = np.array(ai100_scores_list, dtype=np.float64)

    print(f"[SCORES] PyTorch(original): {hf_scores.tolist()}")
    print(f"[SCORES] AI100: {ai100_scores.tolist()}")

    pt_ai100_mad_mean, pt_ai100_mad_max = _mad_stats(hf_scores, ai100_scores)
    print(f"[MAD] PyTorch(original) vs AI100: mean={pt_ai100_mad_mean:.6e}, max={pt_ai100_mad_max:.6e}")
    assert pt_ai100_mad_max <= PT_AI100_MAD_MAX, (
        f"PyTorch(original) vs AI100 MAD max {pt_ai100_mad_max:.6e} "
        f"exceeds threshold {PT_AI100_MAD_MAX:.6e}. "
        f"Check tokenizer ids, prompt formatting, runtime prompt length slicing, and compile dimensions."
    )
