# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from typing import Optional

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient.utils import load_hf_tokenizer
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from QEfficient.utils.test_utils import (
    InternProcessor,
    load_hf_vlm_model,
    load_qeff_model_with_sampler,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../configs/feature_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    sampler_models = config_data["sampler_config"]
test_models = [model["model_name"] for model in sampler_models]
model_config_dict = {model["model_name"]: model for model in sampler_models}


def check_guided_decoding_sampler(
    model_name: str,
    manual_cleanup: callable,
    num_hidden_layers: Optional[int] = -1,
    config: Optional[AutoConfig] = None,
):
    """
    Test QPCs compiled with and without guided decoding.
    """
    model_config = model_config_dict[model_name]
    is_vlm = model_config.get("is_vlm", False)
    prompts = model_config.get("prompts", [])
    prefill_seq_len = model_config.get("prefill_seq_len", 16)
    ctx_len = model_config.get("ctx_len", 32)
    full_batch_size = model_config.get("full_batch_size", 1)
    spec_length = model_config.get("spec_length", None)
    prompts = model_config.get("prompts", [])
    image_urls = model_config.get("image_urls", [])
    generation_len = model_config.get("generation_len", 20)

    model_w_sampler_w_guided_decoding = load_qeff_model_with_sampler(
        model_name,
        is_vlm,
        True,
        num_hidden_layers=num_hidden_layers,
        config=config,
        qaic_config=dict(
            {
                "include_sampler": True,
                "return_pdfs": False,
                "max_top_k_ids": 1024,
                "include_guided_decoding": True,
            }
        ),
    )
    model_w_sampler_wo_guided_decoding = load_qeff_model_with_sampler(
        model_name,
        is_vlm,
        True,
        num_hidden_layers=num_hidden_layers,
        config=config,
        qaic_config=dict(
            {
                "include_sampler": True,
                "return_pdfs": False,
                "max_top_k_ids": 1024,
            }
        ),
    )

    additional_params = {}
    if is_vlm:
        model_hf = load_hf_vlm_model(model_name, num_hidden_layers=num_hidden_layers, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)
        additional_params = {"processor": processor, "images": image_urls}
    else:
        spec_length = spec_length - 1

    model_w_sampler_w_guided_decoding.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=DEFAULT_AIC_NUM_CORES,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_w_sampler_wo_guided_decoding.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=DEFAULT_AIC_NUM_CORES,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Generate texts from prompts
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    np.random.seed(0)
    sampling_params = {
        "repetition_penalties": np.array(1.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "presence_penalties": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        # "frequency_penalties": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "temperatures": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "top_ks": np.array(1024, dtype=np.int32).repeat(full_batch_size).reshape(-1, 1),
        "top_ps": np.array(1.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "min_ps": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "random_numbers": np.zeros((full_batch_size, 1024), dtype=np.float32),
    }
    if is_vlm:
        vocab_size = model_w_sampler_w_guided_decoding.model.language_model.config.vocab_size
    else:
        vocab_size = model_w_sampler_w_guided_decoding.model.config.vocab_size
    model_w_sampler_w_guided_decoding_exec_info = model_w_sampler_w_guided_decoding.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        include_guided_decoding=True,
        sampling_params={
            **sampling_params,
            **{
                "token_bitmasks": np.tile(
                    np.random.choice([True, False], size=(vocab_size,)),
                    (full_batch_size, 1),
                )
            },
        },
        **additional_params,
    )
    model_w_sampler_wo_guided_decoding_exec_info = model_w_sampler_wo_guided_decoding.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        sampling_params=sampling_params,
        **additional_params,
    )
    assert (
        model_w_sampler_w_guided_decoding_exec_info.generated_ids
        != model_w_sampler_wo_guided_decoding_exec_info.generated_ids
    ).any(), "Sampler outputs with and without guided decoding should not match"

    manual_cleanup(model_w_sampler_w_guided_decoding.onnx_path)
    manual_cleanup(model_w_sampler_wo_guided_decoding.onnx_path)


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models)
def test_full_guided_decoding_sampler(model_name, manual_cleanup):
    """
    Test the full guided decoding with different models.
    """
    torch.manual_seed(42)
    check_guided_decoding_sampler(model_name, manual_cleanup=manual_cleanup)


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models)
def test_2layers_guided_decoding_sampler(model_name, manual_cleanup):
    """
    Test the guided decoding with 2 layers models.
    """
    torch.manual_seed(42)
    check_guided_decoding_sampler(model_name, num_hidden_layers=2, manual_cleanup=manual_cleanup)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models)
def test_dummy_guided_decoding_sampler(model_name, manual_cleanup):
    """
    Test the guided decoding with dummy models.
    """
    torch.manual_seed(42)
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    check_guided_decoding_sampler(model_name, config=hf_config, manual_cleanup=manual_cleanup)
