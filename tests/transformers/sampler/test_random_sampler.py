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


def check_random_sampler(
    model_name: str,
    manual_cleanup: callable,
    num_hidden_layers: Optional[int] = -1,
    config: Optional[AutoConfig] = None,
):
    """
    Test random sampling with QPCs compiled with and without On Device Sampling.
    """
    # Export and compile QEfficient models
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

    model_w_sampler = load_qeff_model_with_sampler(
        model_name,
        is_vlm,
        True,
        num_hidden_layers=num_hidden_layers,
        config=config,
        qaic_config=dict(
            {
                "include_sampler": True,
                "return_pdfs": False,
                "max_top_k_ids": 512,
            }
        ),
    )
    model_wo_sampler = load_qeff_model_with_sampler(
        model_name,
        is_vlm,
        True,
        num_hidden_layers=num_hidden_layers,
        config=config,
        qaic_config=dict(
            {
                "include_sampler": False,
                "return_pdfs": False,
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

    model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=DEFAULT_AIC_NUM_CORES,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler.compile(
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
    model_w_sampler_exec_info = model_w_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        sampling_params={
            "repetition_penalties": np.array(20.2, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "presence_penalties": np.array(10.5, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            # "frequency_penalties": np.array(0.5, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "temperatures": np.array(4.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "top_ks": np.array(512, dtype=np.int32).repeat(full_batch_size).reshape(-1, 1),
            "top_ps": np.array(0.89, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "min_ps": np.array(0.6, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "random_numbers": np.tile(np.random.uniform(low=0.0, high=1.0, size=512), (full_batch_size, 1)).astype(
                np.float32
            ),
        },
        **additional_params,
    )
    model_wo_sampler_exec_info = model_wo_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=False,
        return_pdfs=False,
        sampling_params=None,
        **additional_params,
    )

    # Compare generated texts
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        golden_texts = {
            "w_sampler": "Aiden and I am a freelance writer who loves to explore the world. With over",
            "wo_sampler": "John Smith and I am a software engineer. I have been working in the industry for the past ",
        }
        golden_ids = {
            "w_sampler": [
                [
                    319,
                    3615,
                    322,
                    306,
                    626,
                    263,
                    3005,
                    295,
                    749,
                    9227,
                    1058,
                    12355,
                    267,
                    304,
                    26987,
                    278,
                    3186,
                    29889,
                    2973,
                    975,
                ]
            ],
            "wo_sampler": [
                [
                    2259,
                    7075,
                    322,
                    306,
                    626,
                    263,
                    7047,
                    22055,
                    29889,
                    306,
                    505,
                    1063,
                    1985,
                    297,
                    278,
                    13661,
                    363,
                    278,
                    4940,
                    29871,
                ]
            ],
        }
    elif model_name == "OpenGVLab/InternVL2_5-1B":
        golden_texts = {
            "w_sampler": "The description of this vivid scene is as follows:\n\nIn a sepia-toned photograph, we see",
            "wo_sampler": "The image features a black puppy lying on a wooden surface. The puppy has a shiny, glossy coat",
        }
        golden_ids = {
            "w_sampler": [
                [
                    785,
                    4008,
                    315,
                    419,
                    42020,
                    6109,
                    374,
                    438,
                    11017,
                    1447,
                    641,
                    264,
                    21017,
                    685,
                    74635,
                    291,
                    10300,
                    11,
                    582,
                    1490,
                ]
            ],
            "wo_sampler": [
                [
                    785,
                    2168,
                    4419,
                    264,
                    3691,
                    41189,
                    20446,
                    389,
                    264,
                    22360,
                    7329,
                    13,
                    576,
                    41189,
                    702,
                    264,
                    41199,
                    11,
                    73056,
                    22875,
                ]
            ],
        }
    for i in range(full_batch_size):
        assert (
            tokenizer.decode(model_w_sampler_exec_info.generated_ids[i][:generation_len]) == golden_texts["w_sampler"]
        ), "Sampler generated texts does not match"
        assert (model_w_sampler_exec_info.generated_ids[i][:generation_len] == golden_ids["w_sampler"]).all(), (
            "Sampler generated ids do not match"
        )
        assert (
            tokenizer.decode(model_wo_sampler_exec_info.generated_ids[i][:generation_len]) == golden_texts["wo_sampler"]
        ), "Without sampler generated texts does not match"
        assert (model_wo_sampler_exec_info.generated_ids[i][:generation_len] == golden_ids["wo_sampler"]).all(), (
            "Without sampler generated ids do not match"
        )
    manual_cleanup(model_w_sampler.onnx_path)
    manual_cleanup(model_wo_sampler.onnx_path)


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models)
def test_full_random_sampler(model_name, manual_cleanup):
    """
    Test the full random sampler with different models.
    """
    torch.manual_seed(42)
    check_random_sampler(model_name, manual_cleanup=manual_cleanup)


# @pytest.mark.on_qaic
# @pytest.mark.feature
# @pytest.mark.parametrize("model_name",test_models)
# def test_2layers_random_sampler(model_name):
#     """
#     Test the random sampler with 2 layers models.
#     """
#     torch.manual_seed(42)
#     golden_texts = model_config_dict[model_name]["dummy_layers_output"]["golden_texts"]
#     golden_ids = model_config_dict[model_name]["dummy_layers_output"]["golden_ids"]
#     check_random_sampler(model_name, golden_texts=golden_texts, golden_ids=golden_ids, num_hidden_layers=2)

# @pytest.mark.on_qaic
# @pytest.mark.feature
# @pytest.mark.parametrize("model_name",test_models)
# def test_dummy_random_sampler(model_name):
#     """
#     Test the random sampler with dummy models.
#     """
#     torch.manual_seed(42)
#     hf_config = AutoConfig.from_pretrained(
#         model_name,
#         trust_remote_code=True,
#         **model_config_dict[model_name].get("additional_params", {}),
#     )
#     golden_texts = model_config_dict[model_name]["dummy_layers_output"]["golden_texts"]
#     golden_ids = model_config_dict[model_name]["dummy_layers_output"]["golden_ids"]
#     check_random_sampler(model_name, golden_texts=golden_texts, golden_ids=golden_ids, config=hf_config,)
