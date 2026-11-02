# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Shared helpers, model registry, and constants for tests/dynamo/."""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

# Worker-level caches. Model callers receive deep copies because transforms and
# weight offload can mutate model instances.
_HF_MODEL_CACHE: dict[tuple[str, torch.dtype], tuple[AutoModelForCausalLM, AutoTokenizer]] = {}
_HF_TOKEN_CACHE: dict[tuple[object, ...], object] = {}

DYNAMO_CAUSAL_LM_MODEL_IDS = {
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    # deepseek_v3 needs a newer compatible transformers environment.
    "falcon": "hf-internal-testing/tiny-random-FalconForCausalLM",
    "gemma": "Xenova/tiny-random-GemmaForCausalLM",
    "gemma2": "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "glm4_moe": "tiny-random/glm-4-moe",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "gpt_oss": "tiny-random/gpt-oss-bf16",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "granite": "hf-tiny-v2/tiny-random-GraniteForCausalLM",
    "granitemoe": "hf-tiny-v2/tiny-random-GraniteMoeForCausalLM",
    # grok_1 tiny config is not supported in legacy.
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # llama_swiftkv is not AutoModelForCausalLM-compatible.
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "olmo2": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    # phi3 is disabled until SplitToSequence is fixed.
    "qwen2": "yujiepan/qwen2-tiny-random",
    "qwen3": "tiny-random/qwen3",
    "qwen3_moe": "tiny-random/qwen3-moe",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
}

PROMPT_LEN = 8
CTX_LEN = 32
BATCH_SIZE = 1
FULL_BATCH_SIZE = 4
CB_PROMPTS = ["hello world"] * FULL_BATCH_SIZE
DYNAMO = True
DTYPE = torch.float32
MODEL_KWARGS = {"attn_implementation": "eager", "low_cpu_mem_usage": False}


def skip_on_hf_model_load_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: HF model/tokenizer unavailable or unsupported in this environment "
        f"({type(exc).__name__}: {exc})"
    )


def load_hf_model(model_id: str, torch_dtype: torch.dtype) -> AutoModelForCausalLM:
    key = (model_id, torch_dtype)
    if key not in _HF_MODEL_CACHE:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **MODEL_KWARGS,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _HF_MODEL_CACHE[key] = (model, tokenizer)
    model, _ = _HF_MODEL_CACHE[key]
    return copy.deepcopy(model)


def load_tokenizer(model_id: str, torch_dtype: torch.dtype) -> AutoTokenizer:
    key = (model_id, torch_dtype)
    if key not in _HF_MODEL_CACHE:
        load_hf_model(model_id, torch_dtype=torch_dtype)
    _, tokenizer = _HF_MODEL_CACHE[key]
    return tokenizer


def _hf_token_cache_key(tokenizer, model_hf, prompts, prompt_len, ctx_len, batch_size, full_batch_size):
    model_name = getattr(model_hf.config, "_name_or_path", "") or getattr(model_hf.config, "name_or_path", "")
    tokenizer_name = getattr(tokenizer, "name_or_path", "")
    dtype = str(getattr(model_hf, "dtype", next(model_hf.parameters()).dtype))
    fingerprint = hashlib.sha256()
    for name, tensor in list(model_hf.state_dict().items())[:4]:
        sample = tensor.detach().cpu().reshape(-1)[:16]
        if sample.is_floating_point():
            sample = sample.float()
        fingerprint.update(name.encode())
        fingerprint.update(str(tuple(tensor.shape)).encode())
        fingerprint.update(str(tensor.dtype).encode())
        fingerprint.update(sample.numpy().tobytes())
    return (
        model_name,
        tokenizer_name,
        dtype,
        fingerprint.hexdigest(),
        tuple(prompts),
        prompt_len,
        ctx_len,
        batch_size,
        full_batch_size,
    )


def get_hf_tokens(
    tokenizer,
    model_hf,
    prompts,
    *,
    prompt_len: int,
    ctx_len: int,
    batch_size: int = BATCH_SIZE,
    full_batch_size: int | None = None,
):
    key = _hf_token_cache_key(tokenizer, model_hf, prompts, prompt_len, ctx_len, batch_size, full_batch_size)
    if key in _HF_TOKEN_CACHE:
        return copy.deepcopy(_HF_TOKEN_CACHE[key])

    api_runner = ApiRunner(
        batch_size=batch_size,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompts,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        dtype=DTYPE,
    )
    if full_batch_size is None:
        hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
        assert hf_tokens is not None, "HF PT inference returned None"
        _HF_TOKEN_CACHE[key] = copy.deepcopy(hf_tokens)
        return copy.deepcopy(hf_tokens)

    hf_tokens = api_runner.run_hf_model_on_pytorch_CB(model_hf)
    assert hf_tokens is not None, "HF PT CB inference returned None"
    _HF_TOKEN_CACHE[key] = copy.deepcopy(hf_tokens)
    return copy.deepcopy(hf_tokens)


def assert_hf_hw_parity(
    model_id: str,
    hf_tokens,
    qaic_output,
    *,
    gen_len: int,
    full_batch_size: int | None = None,
    context: str = "",
) -> None:
    assert qaic_output is not None, "QAIC generate returned None"
    assert hasattr(qaic_output, "generated_ids"), "QAIC generate did not return generated_ids"
    assert qaic_output.generated_ids is not None, "QAIC generate returned generated_ids=None"

    label = f" {context}" if context else ""
    if full_batch_size is None:
        hf_tokens = np.asarray(hf_tokens).flatten()[:gen_len]
        qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
        if not np.array_equal(hf_tokens, qaic_tokens):
            assert False, (
                f"HF AIC HW{label} parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
            )
        return

    assert len(hf_tokens) == full_batch_size
    for batch_idx in range(full_batch_size):
        hf_batch_tokens = np.asarray(hf_tokens[batch_idx]).flatten()[:gen_len]
        qaic_batch_tokens = qaic_output.generated_ids[batch_idx].flatten()[:gen_len]
        if not np.array_equal(hf_batch_tokens, qaic_batch_tokens):
            assert False, (
                f"HF AIC HW{label} CB parity failed for {model_id} batch {batch_idx}: "
                f"HF={hf_batch_tokens.tolist()}, QAIC={qaic_batch_tokens.tolist()}"
            )


def exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file(), f"Expected ONNX file at {onnx_path}"
    return onnx_path


# Export cache shared by QAIC Dynamo tests. The key only includes export-time
# knobs; compile-time options still get their own QPCs per test.
_DYNAMO_ONNX_CACHE: dict[tuple[str, torch.dtype, bool, bool], tuple[str, QEFFAutoModelForCausalLM]] = {}


def get_dynamo_export(
    model_id: str,
    tmp_path_factory,
    *,
    torch_dtype: torch.dtype,
    continuous_batching: bool = False,
    ccl_enabled: bool = False,
) -> tuple[str, QEFFAutoModelForCausalLM]:
    """Export once per compatible Dynamo graph shape and cache the ONNX path."""
    key = (model_id, torch_dtype, continuous_batching, ccl_enabled)
    if key in _DYNAMO_ONNX_CACHE:
        onnx_path, qeff_model = _DYNAMO_ONNX_CACHE[key]
        return onnx_path, copy.deepcopy(qeff_model)

    kwargs: dict[str, object] = {}
    if continuous_batching:
        kwargs["continuous_batching"] = True
    if ccl_enabled:
        kwargs["qaic_config"] = {"ccl_enabled": True}

    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    except Exception as exc:
        skip_on_hf_model_load_error(exc, model_id)

    export_dir = tmp_path_factory.mktemp("dynamo_export", numbered=True)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            export_dir,
            dynamo=DYNAMO,
            use_onnx_subfunctions=True,
        )
    )
    _DYNAMO_ONNX_CACHE[key] = (str(onnx_path), copy.deepcopy(qeff_model))
    return str(onnx_path), qeff_model


def assert_has_subfunctions(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Assert the ONNX graph contains at least one decoder-block subfunction."""
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


_BLOCKED_KV_MARKER_MODES = {
    "kv": {"CtxGatherBlockedKV"},
    "qkv": {"CtxGatherBlockedKV"},
    "hkv": {"CtxGatherBlockedKV"},
    "hqkv": {"CtxGatherBlockedKV"},
    "bhqkv": {"CtxGatherBlockedKV"},
    "kv_headpar": {"CtxGatherBlockedKV"},
}
_CB_BLOCKED_KV_MARKER_MODES = {"kv", "qkv", "hkv", "hqkv", "bhqkv", "kv_headpar"}


def assert_blocked_kv_ops_for_mode(
    onnx_path: Path,
    qeff_model: QEFFAutoModelForCausalLM,
    blocking_key: str,
    *,
    continuous_batching: bool = False,
) -> None:
    """Assert stable blocked-KV custom op markers for KV-bearing blocking modes.

    Pure Q/H/HQ modes do not have a small reliable graph marker, so they are
    covered by dispatch tests plus export/compile/generation parity.
    """
    expected_ops = set(_BLOCKED_KV_MARKER_MODES.get(blocking_key, ()))
    if not expected_ops:
        return
    if continuous_batching and blocking_key in _CB_BLOCKED_KV_MARKER_MODES:
        expected_ops = {"CtxGatherBlockedKVCB"}

    model = onnx.load(str(onnx_path), load_external_data=False)
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    decoder_names = set()
    if callable(get_submodules):
        submodule_classes = get_submodules()
        decoder_names = {
            cls.__name__
            for cls in (submodule_classes if isinstance(submodule_classes, (set, list, tuple)) else [submodule_classes])
        }

    function_nodes = [
        node
        for fn in model.functions
        if not decoder_names or any(decoder_name in fn.name for decoder_name in decoder_names)
        for node in fn.node
    ]
    op_counts = Counter(node.op_type for node in list(model.graph.node) + function_nodes)

    missing_ops = sorted(op_name for op_name in expected_ops if not op_counts[op_name])
    found_ops = {op_name: op_counts[op_name] for op_name in sorted(expected_ops) if op_counts[op_name]}
    assert not missing_ops, (
        f"Expected blocked KV custom op marker(s) {sorted(expected_ops)} for mode {blocking_key!r} "
        f"in {onnx_path.name}, but missing {missing_ops}. "
        f"Found blocked KV ops: {found_ops}. Ops present: {dict(op_counts)}"
    )


def assert_subfunction_names_match_decoder_class(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Verify RenameRepeatedSubgraphTransform renamed functions to decoder class names."""
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return

    submodule_classes = get_submodules()
    if not submodule_classes:
        return

    expected_names = {
        cls.__name__
        for cls in (submodule_classes if isinstance(submodule_classes, (set, list, tuple)) else [submodule_classes])
    }

    model = onnx.load(str(onnx_path), load_external_data=False)
    for fn in model.functions:
        assert not any(fn.name.startswith(pat) for pat in ("repeated_subgraph", "subgraph_", "invoke_subgraph_")), (
            f"Function '{fn.name}' still has raw dynamo name: "
            f"RenameRepeatedSubgraphTransform did not rename it. "
            f"Expected a name derived from {expected_names}."
        )


def assert_retained_state_outputs(onnx_path: Path, expected_count: int) -> None:
    """Assert that the ONNX graph has the expected number of _RetainedState outputs."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    retained = [o for o in model.graph.output if o.name.endswith("_RetainedState")]
    assert len(retained) == expected_count, (
        f"Expected {expected_count} _RetainedState outputs, got {len(retained)}: {[o.name for o in retained]}"
    )
