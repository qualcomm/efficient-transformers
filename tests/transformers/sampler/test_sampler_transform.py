# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from typing import Optional

import pytest
import torch
from transformers import AutoConfig

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES, Constants
from QEfficient.utils.test_utils import (
    load_qeff_model_with_sampler,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../configs/feature_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    sampler_models = config_data["sampler_config"]
test_models = [model["model_name"] for model in sampler_models]
model_config_dict = {model["model_name"]: model for model in sampler_models}


def check_sampler_transform(
    model_name: str,
    manual_cleanup: callable,
    num_hidden_layers: Optional[int] = -1,
    config: Optional[AutoConfig] = None,
):
    """
    Check the sampler transform for a given model.

    Args:
        model_name (str): The name of the model to test.
        num_hidden_layers (Optional[int]): The number of hidden layers to use.
        config (Optional[AutoConfig]): The configuration to use.
    """
    model_config = model_config_dict[model_name]
    is_vlm = model_config.get("is_vlm", False)
    prefill_seq_len = model_config.get("prefill_seq_len", 16)
    ctx_len = model_config.get("ctx_len", 32)
    full_batch_size = model_config.get("full_batch_size", 1)
    spec_length = model_config.get("spec_length", None)
    if not is_vlm:
        spec_length = spec_length - 1

    qaic_config = dict(
        {
            "include_sampler": True,
            "return_pdfs": False,
            "max_top_k_ids": 512,
        }
    )
    model_w_sampler = load_qeff_model_with_sampler(
        model_name, is_vlm, True, num_hidden_layers=num_hidden_layers, config=config, qaic_config=qaic_config
    )

    qaic_config = dict(
        {
            "include_sampler": True,
            "return_pdfs": False,
            "max_top_k_ids": 512,
            "include_guided_decoding": True,
        }
    )
    model_w_sampler_w_guided_decoding = load_qeff_model_with_sampler(
        model_name, is_vlm, True, num_hidden_layers=num_hidden_layers, config=config, qaic_config=qaic_config
    )

    qaic_config = dict(
        {
            "include_sampler": False,
            "return_pdfs": False,
        }
    )
    model_wo_sampler = load_qeff_model_with_sampler(
        model_name, is_vlm, True, num_hidden_layers=num_hidden_layers, config=config, qaic_config=qaic_config
    )

    model_w_sampler_qpc_path = model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=DEFAULT_AIC_NUM_CORES,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_w_sampler_w_guided_decoding_qpc_path = model_w_sampler_w_guided_decoding.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=DEFAULT_AIC_NUM_CORES,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler_qpc_path = model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=DEFAULT_AIC_NUM_CORES,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    if is_vlm:
        model_w_sampler_qpc_path = model_w_sampler_qpc_path[1]
        model_w_sampler_w_guided_decoding_qpc_path = model_w_sampler_w_guided_decoding_qpc_path[1]
        model_wo_sampler_qpc_path = model_wo_sampler_qpc_path[1]

    # Init qaic session
    model_w_sampler_session = QAICInferenceSession(model_w_sampler_qpc_path)
    model_w_sampler_w_guided_decoding_session = QAICInferenceSession(model_w_sampler_w_guided_decoding_qpc_path)
    model_wo_sampler_session = QAICInferenceSession(model_wo_sampler_qpc_path)

    # Skip inputs/outputs buffers
    model_w_sampler_session.skip_buffers(set([x for x in model_w_sampler_session.input_names if x.startswith("past_")]))
    model_w_sampler_session.skip_buffers(
        set([x for x in model_w_sampler_session.output_names if x.endswith("_RetainedState")])
    )
    model_w_sampler_w_guided_decoding_session.skip_buffers(
        set([x for x in model_w_sampler_w_guided_decoding_session.input_names if x.startswith("past_")])
    )
    model_w_sampler_w_guided_decoding_session.skip_buffers(
        set([x for x in model_w_sampler_w_guided_decoding_session.output_names if x.endswith("_RetainedState")])
    )
    model_wo_sampler_session.skip_buffers(
        set([x for x in model_wo_sampler_session.input_names if x.startswith("past_")])
    )
    model_wo_sampler_session.skip_buffers(
        set([x for x in model_wo_sampler_session.output_names if x.endswith("_RetainedState")])
    )

    # Validate sampler inputs
    sampler_inputs = Constants.SAMPLER_INPUTS
    for input_name in sampler_inputs:
        assert input_name in model_w_sampler_session.input_names, (
            f"Sampler input {input_name} not found in QPC compiled with On Device Sampler"
        )
        assert input_name in model_w_sampler_w_guided_decoding_session.input_names, (
            f"Sampler input {input_name} not found in QPC compiled with On Device Sampler and Guided Decoding"
        )
        assert input_name not in model_wo_sampler_session.input_names, (
            f"Sampler input {input_name} found in QPC compiled without On Device Sampler"
        )
    assert "token_bitmasks" in model_w_sampler_w_guided_decoding_session.input_names, (
        "Sampler input token_bitmasks not found in QPC compiled with On Device Sampler and Guided Decoding"
    )
    manual_cleanup(model_w_sampler.onnx_path)
    manual_cleanup(model_w_sampler_w_guided_decoding.onnx_path)
    manual_cleanup(model_wo_sampler.onnx_path)


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models)
def test_full_sampler_transform(model_name, manual_cleanup):
    """
    Test for full layer models if `SamplerTransform` adds nodes at the output of a `QEffForCausalLM model` to enable the
    sampling of next tokens at the device (instead of the host) and returns the
    next tokens and/or probability distributions.
    """
    # Export and compile QEfficient models
    torch.manual_seed(42)
    check_sampler_transform(model_name, manual_cleanup=manual_cleanup)


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models)
def test_2layers_sampler_transform(model_name, manual_cleanup):
    """
    Test for 2 layers model if `SamplerTransform` adds nodes at the output of a `QEffForCausalLM model` to enable the
    sampling of next tokens at the device (instead of the host) and returns the
    next tokens and/or probability distributions.
    """
    # Export and compile QEfficient models
    torch.manual_seed(42)
    check_sampler_transform(model_name, num_hidden_layers=2, manual_cleanup=manual_cleanup)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models)
def test_dummy_sampler_transform(model_name: str, manual_cleanup):
    """
    Test for dummy model if `SamplerTransform` adds nodes at the output of a `QEffForCausalLM model` to enable the
    sampling of next tokens at the device (instead of the host) and returns the
    next tokens and/or probability distributions.
    """
    # Export and compile QEfficient models
    torch.manual_seed(42)
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    check_sampler_transform(model_name, config=hf_config, manual_cleanup=manual_cleanup)
