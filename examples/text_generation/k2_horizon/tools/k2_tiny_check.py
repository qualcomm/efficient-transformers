# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Tiny random K2 Horizon models through the QEff stack, one per Hub size:
HF PyTorch -> QEff KV PyTorch -> ONNX export -> ONNX Runtime, token match on all three.

Each size's real config is reduced to 2 layers / 4 heads, so the features that differ
between sizes (norm groups, head_dim, rope type, vocabulary) are all exercised in seconds.
Run this after any wrapper change, before spending time on real weights.

    python3 k2_tiny_check.py                                   # all four dense sizes
    python3 k2_tiny_check.py IFM/K2-Horizon-0.9B               # one size
"""

import copy
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.transformers.modeling_utils import EXTERNAL_MODEL_CLASS_MAPPING
from QEfficient.utils.run_utils import ApiRunner

SIZES = ["IFM/K2-Horizon-0.9B", "IFM/K2-Horizon-3.7B", "IFM/K2-Horizon-7B", "IFM/K2-Horizon-32B"]


def check(name):
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    assert type(config).__name__ in EXTERNAL_MODEL_CLASS_MAPPING, f"{type(config).__name__} is not registered"
    head_dim = config.head_dim
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.hidden_size = 4 * head_dim
    config.intermediate_size = 512
    config.mlp_only_layers = [0, 1]
    config.max_position_embeddings = 256
    config.dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model_hf = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager").eval()
    runner = ApiRunner(1, tokenizer, config, ["My name is"], 8, 24)
    hf_tokens = np.asarray(runner.run_hf_model_on_pytorch_CB(copy.deepcopy(model_hf))[0]).reshape(-1)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=name)
    attn = qeff_model.model.model.layers[0].self_attn
    assert "k2_horizon" in attn.forward.__func__.__module__, "transforms did not apply"
    kv_tokens = np.asarray(runner.run_kv_model_on_pytorch(qeff_model.model)).reshape(-1)

    onnx_path = qeff_model.export()
    ort_tokens = np.asarray(runner.run_kv_model_on_ort(str(onnx_path))).reshape(-1)

    kv_ok, ort_ok = bool((hf_tokens == kv_tokens).all()), bool((hf_tokens == ort_tokens).all())
    print(
        f"RESULT {name}: groups={config.layernorm_num_groups} head_dim={head_dim} "
        f"rope={config.rope_parameters.get('rope_type')} vocab={config.vocab_size} HF==KV {kv_ok} HF==ORT {ort_ok}"
    )
    return kv_ok and ort_ok


if __name__ == "__main__":
    results = [check(name) for name in (sys.argv[1:] or SIZES)]
    sys.exit(0 if all(results) else 1)
