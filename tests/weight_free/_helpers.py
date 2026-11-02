# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Shared helpers, model registry, and constants for tests/weight_free/.

Model IDs are the same tiny-random checkpoints used by tests/dynamo/ so
both suites exercise the same model families.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.exporter.weight_free import load_weight_free_ort_inputs
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

# ---------------------------------------------------------------------------
# Worker-level model cache
# ---------------------------------------------------------------------------
_HF_MODEL_CACHE: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

# ---------------------------------------------------------------------------
# Model registry — same tiny-random models as tests/dynamo/
# ---------------------------------------------------------------------------

WEIGHT_FREE_CAUSAL_LM_MODEL_IDS = {
    "codegen": "hf-tiny-v2/tiny-random-CodeGenForCausalLM",
    "falcon": "hf-tiny-v2/tiny-random-FalconForCausalLM",
    "gemma": "Xenova/tiny-random-GemmaForCausalLM",
    "gemma2": "hf-tiny-v2/tiny-random-Gemma2ForCausalLM",
    "glm4_moe": "tiny-random/glm-4-moe",
    "gpt2": "hf-tiny-v2/tiny-random-GPT2LMHeadModel",
    "gpt_bigcode": "hf-tiny-v2/tiny-random-GPTBigCodeForCausalLM",
    "gpt_oss": "tiny-random/gpt-oss-mxfp4",
    "gptj": "yujiepan/gptj-tiny-random",
    "granite": "hf-tiny-v2/tiny-random-GraniteForCausalLM",
    "granitemoe": "hf-tiny-v2/tiny-random-GraniteMoeForCausalLM",
    "llama": "hf-tiny-v2/tiny-random-LlamaForCausalLM",
    "mistral": "yujiepan/mistral-tiny-random",
    "mixtral": "optimum-intel-internal-testing/tiny-mixtral",
    "mpt": "hf-tiny-v2/tiny-random-MptForCausalLM",
    "olmo2": "hf-tiny-v2/tiny-random-Olmo2ForCausalLM",
    "phi": "hf-tiny-v2/tiny-random-PhiForCausalLM",
    "qwen2": "yujiepan/qwen2-tiny-random",
    "qwen3": "tiny-random/qwen3",
    "qwen3_moe": "tiny-random/qwen3-moe",
    "starcoder2": "hf-tiny-v2/tiny-random-Starcoder2ForCausalLM",
}

WEIGHT_FREE_QAIC_MODEL_PARAMS = [
    pytest.param(
        model_type,
        model_id,
        marks=pytest.mark.xdist_group(name=f"qaic-runtime-{model_type}"),
        id=model_type,
    )
    for model_type, model_id in sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items())
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_LEN = 8
CTX_LEN = 16
BATCH_SIZE = 1
MODEL_KWARGS = {"attn_implementation": "eager", "low_cpu_mem_usage": False, "torch_dtype": torch.float32}

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def load_hf_model(model_id: str) -> AutoModelForCausalLM:
    if model_id not in _HF_MODEL_CACHE:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **MODEL_KWARGS,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _HF_MODEL_CACHE[model_id] = (model, tokenizer)
    model, _ = _HF_MODEL_CACHE[model_id]
    return copy.deepcopy(model)


def load_tokenizer(model_id: str) -> AutoTokenizer:
    if model_id not in _HF_MODEL_CACHE:
        load_hf_model(model_id)
    _, tokenizer = _HF_MODEL_CACHE[model_id]
    return tokenizer


def exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file(), f"Expected ONNX file at {onnx_path}"
    return onnx_path


# ---------------------------------------------------------------------------
# Shared ONNX structure assertions (same as tests/dynamo/_helpers.py)
# ---------------------------------------------------------------------------


def assert_has_subfunctions(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Assert the ONNX contains at least one decoder-block subfunction."""
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return
    submodule_classes = get_submodules()
    if not submodule_classes:
        return
    decoder_names = {
        cls.__name__
        for cls in (submodule_classes if isinstance(submodule_classes, (set, list, tuple)) else [submodule_classes])
    }
    model = onnx.load(str(onnx_path), load_external_data=False)
    found = [fn.name for fn in model.functions if any(d in fn.name for d in decoder_names)]
    assert found, (
        f"Expected decoder-block subfunctions ({decoder_names}) in {onnx_path.name} but found none. "
        f"Functions present: {[fn.name for fn in model.functions]}"
    )


def assert_subfunction_names_match_decoder_class(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Verify RenameRepeatedSubgraphTransform renamed functions to decoder class names."""
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return
    submodule_classes = get_submodules()
    if not submodule_classes:
        return
    model = onnx.load(str(onnx_path), load_external_data=False)
    for fn in model.functions:
        assert not any(fn.name.startswith(pat) for pat in ("repeated_subgraph", "subgraph_", "invoke_subgraph_")), (
            f"Function '{fn.name}' still has raw dynamo name — RenameRepeatedSubgraphTransform did not rename it."
        )


def assert_retained_state_outputs(onnx_path: Path, expected_count: int) -> None:
    """Assert the ONNX graph has the expected number of _RetainedState outputs."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    retained = [o for o in model.graph.output if o.name.endswith("_RetainedState")]
    assert len(retained) == expected_count, (
        f"Expected {expected_count} _RetainedState outputs, got {len(retained)}: {[o.name for o in retained]}"
    )


# ---------------------------------------------------------------------------
# Weight-free-specific ONNX assertions
# ---------------------------------------------------------------------------

# ONNX elem_type constants
_ONNX_INT64 = 7


def assert_unique_graph_input_names(onnx_path: Path) -> None:
    """Assert no ONNX graph input name appears twice.

    Guards the regression where position_ids is mislabeled as past_key.0 due to
    a dict-order vs input_names-order mismatch in the weight-free export path,
    producing a duplicate graph input that causes a compiler error.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    names = [i.name for i in model.graph.input]
    duplicates = [n for n in set(names) if names.count(n) > 1]
    assert not duplicates, (
        f"Duplicate ONNX graph inputs found: {duplicates}. "
        "position_ids was likely mislabeled as a KV cache input during weight-free export."
    )


def assert_no_int64_kv_cache_inputs(onnx_path: Path) -> None:
    """Assert no past_key.X / past_value.X graph input has dtype int64.

    Guards the same regression: if position_ids (int64) is aliased to past_key.0,
    that KV cache slot will have the wrong dtype. A valid KV cache tensor is always
    a floating-point type (float16 or float32), never int64.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    for inp in model.graph.input:
        if inp.name.startswith(("past_key.", "past_value.")):
            dtype = inp.type.tensor_type.elem_type
            assert dtype != _ONNX_INT64, (
                f"Graph input '{inp.name}' has dtype int64 — this is position_ids mislabeled "
                "as a KV cache tensor due to a weight-free export input-naming mismatch."
            )


# ---------------------------------------------------------------------------
# Weight-free ORT generation loop
# ---------------------------------------------------------------------------


def run_weight_free_ort(api_runner, onnx_path: Path, weight_spec_path: Path) -> np.ndarray:
    """Run token generation on a weight-free ONNX using ORT, injecting real weights
    from the checkpoint at every step via load_weight_free_ort_inputs.

    This mirrors the loop in examples/text_generation/compare.py.local and is needed
    because weight-free ONNX has no embedded weights — they appear as extra ORT inputs
    that must be populated from the HF cache safetensors files each inference step.

    Returns generated token IDs with shape (1, gen_len) matching run_hf_model_on_pytorch.
    """
    session = onnxruntime.InferenceSession(str(onnx_path))

    # Prepare runtime inputs (input_ids, position_ids, initial KV cache zeros)
    inputs = api_runner.input_handler.prepare_ort_inputs()
    # Inject model weights from HF cache
    inputs = load_weight_free_ort_inputs(weight_spec_path, inputs)

    ort_outputs_raw = api_runner.run_ort_session(inputs, session)
    ort_outputs = api_runner.input_handler.update_ort_outputs(ort_outputs_raw)

    generated_ids = []
    for _ in range(1, api_runner.gen_len):
        generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        inputs = api_runner.input_handler.update_ort_inputs(inputs, ort_outputs)
        # Re-inject weights: update_ort_inputs only updates runtime tensors (input_ids,
        # position_ids, KV cache) and does not carry weights forward.
        inputs = load_weight_free_ort_inputs(weight_spec_path, inputs)
        ort_outputs_raw = api_runner.run_ort_session(inputs, session)
        ort_outputs = api_runner.input_handler.update_ort_outputs(ort_outputs_raw)

    generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
    return np.concatenate(generated_ids, axis=1)
