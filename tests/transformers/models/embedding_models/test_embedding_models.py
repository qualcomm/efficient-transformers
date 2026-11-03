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
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP
from QEfficient.transformers.models.modeling_auto import QEFFAutoModel
from QEfficient.utils.constants import Constants
from QEfficient.utils.test_utils import ModelConfig
from tests import NUM_AIC_CORES

from ..check_model_results import dump_and_compare_results

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/embedding_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    embed_test_models = config_data["embedding_models"]


def load_embedding_model(model_name: str, n_layer: int = -1):
    """Load a pre-trained embedding model."""
    kwargs = {"attn_implementation": "eager", "trust_remote_code": True}
    if n_layer > 0:
        kwargs["num_hidden_layers"] = n_layer
    pt_model = AutoModel.from_pretrained(
        model_name,
        **kwargs,
    )
    pt_model.eval()
    return pt_model


def check_embed_pytorch_vs_ort_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    seq_len: int = Constants.CTX_LEN,
    n_layer: int = -1,
    pooling: Optional[str] = None,
    compare_results: Optional[bool] = False,
):
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("My name is", return_tensors="pt")

    pt_model = load_embedding_model(model_name, n_layer)
    # Original PyTorch model output
    pt_outputs = pt_model(**inputs)
    pooling_method = POOLING_MAP[pooling] if pooling else None
    pt_embeddings = (
        pooling_method(pt_outputs.last_hidden_state, inputs["attention_mask"])
        if pooling
        else pt_outputs.last_hidden_state
    )

    # QEff transformed PyTorch model
    qeff_model = QEFFAutoModel(pt_model, pretrained_model_name_or_path=model_name, pooling=pooling)

    # QEff transformed PyTorch model output
    qeff_pt_outputs = qeff_model.generate(inputs=inputs, runtime_ai100=False)
    qeff_pt_embeddings = qeff_pt_outputs if pooling else qeff_pt_outputs[0]

    mad = torch.mean(torch.abs(pt_embeddings - qeff_pt_embeddings))
    print("Mad for PyTorch and PyTorch transformed qeff_model is ", mad)
    assert mad <= 0, f"MAD is too high for onnx and Pytorch: {mad}"

    # ONNX session load
    onnx_model = qeff_model.export()
    ort_session = ort.InferenceSession(str(onnx_model))

    # Prepare the inputs for ONNX Runtime
    input_ids = np.array(inputs["input_ids"])
    attention_mask = np.array(inputs["attention_mask"])

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Run inference
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # Compare Transformed PyTorch and ONNX outputs
    mad = torch.mean(torch.abs(pt_embeddings - torch.tensor(onnx_outputs[0])))
    print("Mad for onnx and PyTorch is ", mad)
    assert mad <= 10**-5, f"MAD is too high for onnx and Pytorch: {mad}"

    qeff_model.compile(num_cores=NUM_AIC_CORES)
    ai100_output = qeff_model.generate(inputs=inputs)
    qeff_ai100_embeddings = (
        ai100_output["output"] if pooling else ai100_output["output"][:, : inputs["input_ids"].shape[1], :]
    )

    # Compare ONNX and AI 100 outputs
    mad = np.mean(np.abs(qeff_ai100_embeddings - onnx_outputs[0]))
    print("Mad for onnx and AI 100 output is ", mad)
    assert mad <= 10**-2, f"MAD is too high for onnx and Pytorch: {mad}"
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))

    manual_cleanup(qeff_model.onnx_path)  # Clean up the model files after the tests are done.
    if compare_results is False:
        return

    compile_params = {"pooling": pooling, "seq_len": seq_len}
    assert dump_and_compare_results(
        model_name,
        compile_params,
        "embedding_model_results.json",
        qeff_ai100_embeddings,
        pytorch_hf_tokens=pt_embeddings,
        pytorch_kv_tokens=qeff_pt_embeddings,
        ort_tokens=onnx_outputs[0],
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model", embed_test_models)
def test_full_embed_model_pytorch_vs_onnx_vs_ai100(model, manual_cleanup):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output.
    """
    if model["model_name"] in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model["model_name"], seq_len=32, compare_results=True, manual_cleanup=manual_cleanup
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model", embed_test_models)
def test_full_embed_model_pytorch_vs_onnx_vs_ai100_pooling(model, manual_cleanup):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output with pooling.
    """
    if model["model_name"] in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model["model_name"],
        seq_len=32,
        pooling=model["pooling"],
        compare_results=True,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model", embed_test_models[:1])
def test_full_embed_model_pytorch_vs_onnx_vs_ai100_multiple_seq_len(model, manual_cleanup):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output with multiple seq_len.
    """
    if model["model_name"] in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model["model_name"], seq_len=[32, 20], compare_results=True, manual_cleanup=manual_cleanup
    )


@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model", embed_test_models)
def test_embed_model_pytorch_vs_onnx_vs_ai100(model, manual_cleanup):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output.
    """
    if model["model_name"] in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model["model_name"], seq_len=32, n_layer=1, manual_cleanup=manual_cleanup
    )


@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model", embed_test_models)
def test_embed_model_pytorch_vs_onnx_vs_ai100_pooling(model, manual_cleanup):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output with pooling.
    """
    if model["model_name"] in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model["model_name"], seq_len=32, pooling=model["pooling"], n_layer=1, manual_cleanup=manual_cleanup
    )


@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model", embed_test_models[:1])
def test_embed_model_pytorch_vs_onnx_vs_ai100_multiple_seq_len(model, manual_cleanup):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output with multiple seq_len.
    """
    if model["model_name"] in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model["model_name"], seq_len=[32, 20], n_layer=1, manual_cleanup=manual_cleanup
    )
