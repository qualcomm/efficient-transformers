# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os

import pytest
import torch
from transformers import AutoConfig

from QEfficient.utils.constants import Constants
from QEfficient.utils.test_utils import ModelConfig
from tests.two_phase import resolve_two_phase_cleanup

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
    check_kv_repeat_causal_lm_pytorch_vs_ai100,
    get_custom_n_layers,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    causal_lm_models = config_data["causal_lm_models"]
    per_pr_causal_text_models = config_data["per_pr_causal_text_models"]
test_models_causal = [model["model_name"] for model in causal_lm_models]
model_config_dict = {model["model_name"]: model for model in causal_lm_models}
test_models_per_pr_causal = per_pr_causal_text_models

# per_pr_causal_text_models already covers every architecture in causal_lm_models at
# dummy-layer scope except these: quantized checkpoints (AWQ/GPTQ/FP8, no tiny-random
# equivalent) and the custom-path grok-1/SwiftKV models. Dummy-layer parity tests below
# run only this subset to avoid re-validating architectures the per-PR suite already covers.
_dummy_layers_unique_models = (
    ModelConfig.QUANTIZED_MODELS | set(ModelConfig.EXTERNAL_MODELS) | ModelConfig.SWIFTKV_MODELS
)
test_models_causal_dummy_only = [m for m in test_models_causal if m in _dummy_layers_unique_models]

PER_PR_PROMPT_LEN = 1024
PER_PR_CTX_LEN = 2048
PER_PR_GENERATION_LEN = 8
PER_PR_CCL_PREFILL = None
PER_PR_CCL_DECODE = [2048]
# Blocked-KV walks the cache in fixed-size blocks, so a second, differently-sized prompt is
# what exercises block boundaries that a single prompt would never reach. These two prompts
# are tiled across the unchanged full_batch_size, so export/compile still happen exactly once.
PER_PR_BLOCKING_PROMPTS = [
    "My name is",
    "The capital city of France is called",
]


def _per_pr_id(model_config):
    return model_config["id"]


def _per_pr_dummy_config(model_config):
    config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True)
    if config_attr := model_config.get("config_attr"):
        config = getattr(config, config_attr, config)
    elif model_config.get("use_text_config"):
        config = getattr(config, "text_config", config)

    for attr, value in model_config.get("config_overrides", {}).items():
        setattr(config, attr, value)

    for attr in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
        if hasattr(config, attr):
            setattr(config, attr, max(getattr(config, attr), PER_PR_CTX_LEN))

    if num_hidden_layers := model_config.get("num_hidden_layers"):
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(config, attr):
                setattr(config, attr, num_hidden_layers)

    if layer_types := model_config.get("layer_types"):
        config.layer_types = layer_types
        config.num_hidden_layers = len(layer_types)
    return config


def _run_per_pr_causal_text_case(
    model_config,
    manual_cleanup,
    *,
    torch_dtype=torch.float16,
    compile_only=False,
    retain_full_kv=False,
    qaic_config=None,
    comp_ctx_lengths_prefill=None,
    comp_ctx_lengths_decode=None,
    kv_cache_batch_size=None,
    num_cores=16,
    compile_options=None,
    num_speculative_tokens=None,
    prompts=None,
):
    # Two-phase shared-QEFF_HOME run: the per-test cleanup must be suppressed in BOTH phases,
    # because variants of one model share a content-addressed export dir (the QPCs nest inside
    # it). If left active, a finishing variant's cleanup rmtree's the shared dir and destroys
    # sibling variants' warm QPCs / in-progress compiles. Phase A additionally forces
    # compile-only so it never touches a device. See tests/two_phase.py.
    manual_cleanup, compile_only = resolve_two_phase_cleanup(manual_cleanup, compile_only)

    # Per-test qaic_config takes precedence; fallback to per-model default.
    qaic_config = dict(qaic_config or model_config.get("qaic_config", {})) or None

    # CCL must be enabled before export so comp_ctx_lengths remains an ONNX input.
    if comp_ctx_lengths_prefill is not None or comp_ctx_lengths_decode is not None:
        qaic_config = dict(qaic_config or {})
        qaic_config["ccl_enabled"] = True

    if model_config.get("known_export_or_compile_issue"):
        pytest.xfail(model_config["known_export_or_compile_issue"])
    if model_config.get("known_runtime_parity_issue") and not compile_only:
        pytest.xfail(model_config["known_runtime_parity_issue"])

    config = _per_pr_dummy_config(model_config)
    aic_hw_version = "ai200" if torch_dtype == torch.bfloat16 else "ai100"
    num_cores = 4 if aic_hw_version == "ai200" else model_config.get("num_cores", num_cores)
    compile_options = {**model_config.get("compile_options", {}), **(compile_options or {})}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_config["model_name"],
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        prompt_len=PER_PR_PROMPT_LEN,
        ctx_len=PER_PR_CTX_LEN,
        torch_dtype=torch_dtype,
        generation_len=PER_PR_GENERATION_LEN,
        use_onnx_subfunctions=True,
        compile_only=compile_only,
        retain_full_kv=retain_full_kv,
        qaic_config=qaic_config,
        config=config,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
        kv_cache_batch_size=kv_cache_batch_size,
        num_cores=num_cores,
        aic_hw_version=aic_hw_version,
        compile_options=compile_options,
        num_speculative_tokens=num_speculative_tokens,
        tokenizer_name=model_config.get("tokenizer_id"),
        prompts=prompts,
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_full_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name, compare_results=True, manual_cleanup=manual_cleanup, num_devices=4
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=n_layer, manual_cleanup=manual_cleanup)


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True])
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_onnx_mdp_compile_only(model_name, use_onnx_subfunctions, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        compile_only=True,
        mdp_num_partitions=2,
        mdp_strategy="onnx",
        use_onnx_subfunctions=use_onnx_subfunctions,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal_dummy_only)
def test_dummy_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, manual_cleanup=manual_cleanup)
    else:
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, config=hf_config, manual_cleanup=manual_cleanup)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models_causal)
def test_check_kv_repeat_custom_causal_lm_pytorch_vs_ai100(model_name, manual_cleanup):
    """
    Test function to validate the PyTorch model and the Cloud AI 100 model with repeating original KV heads.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.REPEAT_KV_TEST_MODELS:
        if model_name in ModelConfig.QUANTIZED_MODELS:
            n_layer = get_custom_n_layers(model_name)
            check_kv_repeat_causal_lm_pytorch_vs_ai100(model_name, manual_cleanup=manual_cleanup, n_layer=n_layer)
        else:
            check_kv_repeat_causal_lm_pytorch_vs_ai100(model_name, manual_cleanup=manual_cleanup, config=hf_config)
    else:
        pytest.skip(f"Skipping {model_name} as it is not in REPEAT_KV_TEST_MODELS")


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_full_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal_dummy_only)
def test_dummy_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            n_layer=n_layer,
            continuous_batching=True,
            manual_cleanup=manual_cleanup,
        )
    else:
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            config=hf_config,
            continuous_batching=True,
            manual_cleanup=manual_cleanup,
        )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_fp16_subfunction_cb(model_config, manual_cleanup):
    if issue := model_config.get("known_fp16_subfunction_compile_issue"):
        pytest.xfail(issue)
    _run_per_pr_causal_text_case(model_config, manual_cleanup)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_fp16_subfunction_cb_prefix_caching(model_config, manual_cleanup):
    if issue := model_config.get("known_fp16_subfunction_compile_issue"):
        pytest.xfail(issue)
    _run_per_pr_causal_text_case(model_config, manual_cleanup, kv_cache_batch_size=8)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_fp16_subfunction_cb_ccl(model_config, manual_cleanup):
    if model_config.get("known_ccl_export_or_compile_issue"):
        pytest.xfail(model_config["known_ccl_export_or_compile_issue"])
    _run_per_pr_causal_text_case(
        model_config,
        manual_cleanup,
        comp_ctx_lengths_prefill=PER_PR_CCL_PREFILL,
        comp_ctx_lengths_decode=PER_PR_CCL_DECODE,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_config",
    [model for model in test_models_per_pr_causal if model["supports_blocking"]],
    ids=_per_pr_id,
)
def test_per_pr_causal_fp16_subfunction_cb_blocking(model_config, manual_cleanup):
    """Blocked-KV FP16 CB run, validated against two distinct prompts.

    The prompts are tiled across the same ``full_batch_size`` the other per-PR
    variants use, so the QPC shape is unchanged and export/compile still run once
    per model -- only the generate/reference legs see the extra prompt.
    """
    _run_per_pr_causal_text_case(
        model_config,
        manual_cleanup,
        qaic_config={"enable_blocking": True, "num_kv_blocks": 2},
        prompts=PER_PR_BLOCKING_PROMPTS,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_fp32_export_fp16_compile_subfunction_cb_ccl(model_config, manual_cleanup):
    if model_config.get("known_ccl_export_or_compile_issue"):
        pytest.xfail(model_config["known_ccl_export_or_compile_issue"])
    _run_per_pr_causal_text_case(
        model_config,
        manual_cleanup,
        torch_dtype=torch.float32,
        comp_ctx_lengths_prefill=PER_PR_CCL_PREFILL,
        comp_ctx_lengths_decode=PER_PR_CCL_DECODE,
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_bf16_subfunction_cb_ccl_compile_only(model_config, manual_cleanup):
    if model_config.get("known_ccl_export_or_compile_issue"):
        pytest.xfail(model_config["known_ccl_export_or_compile_issue"])
    if model_config.get("known_bf16_compile_issue"):
        pytest.xfail(model_config["known_bf16_compile_issue"])
    _run_per_pr_causal_text_case(
        model_config,
        manual_cleanup,
        torch_dtype=torch.bfloat16,
        compile_only=True,
        comp_ctx_lengths_prefill=PER_PR_CCL_PREFILL,
        comp_ctx_lengths_decode=PER_PR_CCL_DECODE,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_speculative_tlm_fp16_subfunction_cb(model_config, manual_cleanup):
    """Speculative-decoding (TLM) FP16 export/compile/generate in continuous-batching mode.

    Compiles each per-PR dummy model as a Target Language Model with
    ``Constants.NUM_SPECULATIVE_TOKENS`` speculative tokens, then validates the
    full HF/ORT/AI100 parity path (compile-only under the compile-warm phase).
    Models whose speculative export/compile is a known repo/compiler gap opt out
    via a ``known_speculative_export_or_compile_issue`` registry field, mirroring
    the CCL/BF16 escape hatches.
    """
    if issue := model_config.get("known_fp16_subfunction_compile_issue"):
        pytest.xfail(issue)
    if issue := model_config.get("known_speculative_export_or_compile_issue"):
        pytest.xfail(issue)
    _run_per_pr_causal_text_case(
        model_config,
        manual_cleanup,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
    )


# NOTE: The prompt_len=1 ("pl1") decode-only tests were removed. They fed the fixed
# 3-token prompt ("My name is") through a seq_len=1 PyTorch/ORT reference, so the reference
# input padding computed prompt_len - input_len = 1 - 3 = -2 and crashed before any real
# HF <-> KV <-> ORT parity could be checked. Decode-only behavior is already covered:
#   - runtime seq_len=1 decode with real prefill->decode KV handoff:
#     tests/unit_test/models/test_prefill_decode_kv_handoff.py (test_gpt2_decode_*),
#   - the prompt_len==1 specialization collapse (single "Decode" spec):
#     tests/unit_test/models/test_model_quickcheck.py::
#     test_compile_helper_prefill_only_when_prompt_len_1.
# On-device prompt_len=1 QPC compilation for gpt2 is intentionally not re-added here.
