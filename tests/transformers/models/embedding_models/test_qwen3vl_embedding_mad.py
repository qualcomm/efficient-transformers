# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import inspect
import json
from typing import Any, Dict, List

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.transformers.models.qwen3_vl._embedding_utils import (
    DEFAULT_MAD_MAX,
    EXAMPLE_DOCUMENTS,
    EXAMPLE_QUERIES,
    QEffQwen3VLEmbedder,
    configure_embedding_model_config,
    resolve_model_source,
)
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from QEfficient.utils.test_utils import load_vlm_model
from tests.utils.image_utils import load_test_image

CONFIG_PATH = "tests/configs/image_text_model_configs.json"

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    embedding_models = config_data["image_text_embedding_models"]

test_embedding_models = [model_config["model_name"] for model_config in embedding_models]
embedding_model_config_dict = {model["model_name"]: model for model in embedding_models}


def _load_embedding_test_images(model_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    loaded_inputs = []
    for entry in model_inputs:
        loaded_entry = dict(entry)
        if "image" in loaded_entry:
            loaded_entry["image"] = load_test_image(loaded_entry["image"])
        loaded_inputs.append(loaded_entry)
    return loaded_inputs


def _xfail_if_qwen_vl_utils_too_old():
    """Xfail when the installed ``qwen-vl-utils`` predates the ``image_patch_size`` kwarg.

    Qwen3-VL embedding tokenization calls ``process_vision_info(..., image_patch_size=...)``,
    which was added in ``qwen-vl-utils>=0.0.14``. ``pyproject.toml`` currently pins
    ``qwen-vl-utils==0.0.8``, whose ``process_vision_info`` lacks that kwarg, so the MAD parity
    harness cannot tokenize its multimodal inputs and raises ``TypeError`` before compile. Xfail
    (rather than hard-fail) until the pin is bumped; gating on the actual capability means the
    check self-clears once a compatible version is installed, so a real parity regression cannot
    hide behind a stale model-level xfail.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        pytest.xfail("qwen-vl-utils is not installed; required (>=0.0.14) for Qwen3-VL embedding tokenization.")
    if "image_patch_size" not in inspect.signature(process_vision_info).parameters:
        pytest.xfail(
            "qwen-vl-utils<0.0.14 lacks process_vision_info(image_patch_size=...) (pyproject pins ==0.0.8); "
            "bump the pin to >=0.0.14 to run Qwen3-VL embedding MAD parity."
        )


def _compute_cpu_embeddings(model_hf, embedder, model_inputs: List[Dict[str, Any]]) -> torch.Tensor:
    embedding_rows = []
    for entry in model_inputs:
        conversation = embedder.format_model_input(
            text=entry.get("text"),
            image=entry.get("image"),
            video=entry.get("video"),
            instruction=entry.get("instruction"),
        )
        tokenized = embedder._tokenize_conversation(conversation)
        hf_inputs = {}
        for key, value in tokenized.items():
            hf_inputs[key] = value.to(next(model_hf.parameters()).device) if torch.is_tensor(value) else value

        with torch.no_grad():
            last_hidden_state = model_hf.model(**hf_inputs).last_hidden_state

        last_idx = tokenized["input_ids"].shape[1] - 1
        row = last_hidden_state[:, last_idx : last_idx + 1, :].reshape(last_hidden_state.shape[0], -1)
        embedding_rows.append(row.detach().cpu().to(torch.float32))

    embeddings = torch.cat(embedding_rows, dim=0)
    return F.normalize(embeddings, p=2, dim=-1)


@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.nightly
@pytest.mark.parametrize("model_name", test_embedding_models)
def test_qwen3_vl_embedding_cpu_vs_ai100_mad_parity(model_name):
    _xfail_if_qwen_vl_utils_too_old()
    torch.manual_seed(42)
    model_cfg = embedding_model_config_dict[model_name]
    model_source = resolve_model_source(model_name)

    config = AutoConfig.from_pretrained(model_source, trust_remote_code=True, padding=True)
    # Keep parity runs lightweight by default (reduced text/vision depth from
    # test config). To validate full-layer quality, update the config entry.
    configure_embedding_model_config(
        config=config,
        num_hidden_layers=model_cfg["num_layers"],
        vision_depth=model_cfg["vision_depth"],
        deepstack_index=model_cfg["deepstack_index"],
        export_embedding=False,
    )

    model_hf = load_vlm_model(config)
    model_hf.eval()

    qeff_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True, padding=True)
    configure_embedding_model_config(
        config=qeff_config,
        num_hidden_layers=model_cfg["num_layers"],
        vision_depth=model_cfg["vision_depth"],
        deepstack_index=model_cfg["deepstack_index"],
        export_embedding=True,
    )

    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True, padding=True)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_source,
        kv_offload=True,
        trust_remote_code=True,
        config=qeff_config,
        qaic_config={"export_embedding": True},
    )

    embedder = QEffQwen3VLEmbedder(
        processor=processor,
        model=qeff_model,
    )

    model_inputs = _load_embedding_test_images(EXAMPLE_QUERIES + EXAMPLE_DOCUMENTS)
    compile_specs = embedder.get_compile_specs(
        inputs=model_inputs,
        ctx_len=model_cfg["ctx_len"],
        prefill_seq_len=model_cfg.get("compile_prefill_seq_len", None),
    )
    qpc_paths = qeff_model.compile(
        prefill_seq_len=compile_specs["prefill_seq_len"],
        ctx_len=compile_specs["ctx_len"],
        img_size=compile_specs["img_size"],
        height=compile_specs["height"],
        width=compile_specs["width"],
        num_devices=1,
        num_cores=DEFAULT_AIC_NUM_CORES,
        mxfp6_matmul=False,
    )

    cpu_embeddings = _compute_cpu_embeddings(model_hf=model_hf, embedder=embedder, model_inputs=model_inputs)
    ai100_embeddings = embedder.process(
        inputs=model_inputs,
        qpc_paths=qpc_paths,
        prefill_seq_len=compile_specs["prefill_seq_len"],
        normalize=True,
    )

    diff = torch.abs(cpu_embeddings - ai100_embeddings)
    mad_mean = float(diff.mean().item())
    mad_max = float(diff.max().item())
    threshold = float(model_cfg.get("mad_max_threshold", DEFAULT_MAD_MAX))

    print(f"[MAD] CPU vs AI100 mean={mad_mean:.6e}, max={mad_max:.6e}")
    assert mad_max <= threshold, (
        f"CPU vs AI100 MAD max {mad_max:.6e} exceeds threshold {threshold:.6e}. "
        f"Check prompt formatting, tokenization, prompt-length handling, and AI100 compile args."
    )
