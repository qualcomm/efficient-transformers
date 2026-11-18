# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""GPT-OSS disaggregated subfunction QAIC token parity tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import QEffGptOssDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.constants import DEFAULT_AIC_NUM_CORES
from QEfficient.utils.run_utils import ApiRunner

MODEL_ID = "tiny-random/gpt-oss-bf16"
BATCH_SIZE = 1
MODEL_KWARGS = {
    "attn_implementation": "eager",
    "low_cpu_mem_usage": False,
    "torch_dtype": torch.float32,
    "trust_remote_code": True,
}
PROMPT = ["once upon a time"]
PREFILL_SEQ_LEN = 64
PREFILL_CTX_LEN = 192
DECODE_CTX_LEN = PREFILL_CTX_LEN
DECODE_STEPS = 8
MOE_PREFILL_PACKED_CHUNK_SIZE = 32
EXPECTED_MOE_PREFILL_NUM_PACKED_CHUNKS = PREFILL_SEQ_LEN // MOE_PREFILL_PACKED_CHUNK_SIZE
NUM_CORES = DEFAULT_AIC_NUM_CORES


def _moe_qaic_config() -> dict:
    return {
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE,
        }
    }


@pytest.fixture
def tmp_export_dir(tmp_path):
    export_dir = tmp_path / "qeff_gpt_oss_disagg_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def assert_has_subfunctions(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
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
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    found = [fn.name for fn in onnx_model.functions if any(name in fn.name for name in decoder_names)]
    assert found, (
        f"Expected decoder-block subfunctions ({decoder_names}) in {onnx_path.name} but found none. "
        f"Functions present: {[fn.name for fn in onnx_model.functions]}"
    )


def assert_subfunction_names_match_decoder_class(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
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
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for fn in onnx_model.functions:
        assert not any(fn.name.startswith(pat) for pat in ("repeated_subgraph", "subgraph_", "invoke_subgraph_")), (
            f"Function {fn.name!r} still has raw subgraph name. Expected a name derived from {expected_names}."
        )


def _load_gpt_oss_hf_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        model_hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config, **MODEL_KWARGS)
    except Exception as exc:
        skip_on_model_fetch_error(exc, MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_hf.eval()
    return model_hf, tokenizer


def _load_gpt_oss_qeff_model(config):
    try:
        return QEFFAutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            config=config,
            continuous_batching=False,
            **MODEL_KWARGS,
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, MODEL_ID)


def _prompt_token_len(tokenizer) -> int:
    return tokenizer(PROMPT[0], return_tensors="np", padding=True)["input_ids"].shape[1]


def _api_runner(model_hf, tokenizer, *, prompt_len: int, ctx_len: int) -> ApiRunner:
    return ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=PROMPT,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=None,
    )


def _prefill_inputs(tokenizer) -> tuple[dict[str, np.ndarray], int]:
    prompt_len = _prompt_token_len(tokenizer)
    num_chunks = -(prompt_len // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    assert num_chunks >= 1
    assert padded_len % PREFILL_SEQ_LEN == 0

    inputs = tokenizer(PROMPT[0], return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    return inputs, num_chunks


def _filter_session_inputs(session, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = sorted(set(session.input_names) - set(inputs))
    assert not missing, f"Missing QAIC inputs: {missing}"
    return {name: inputs[name] for name in session.input_names}


def _assert_token_match(test_name: str, hf_tokens, qaic_tokens, gen_len: int) -> None:
    hf_tokens = np.asarray(hf_tokens).flatten()[:gen_len]
    qaic_tokens = np.asarray(qaic_tokens).flatten()[:gen_len]
    assert np.array_equal(hf_tokens, qaic_tokens), (
        f"{test_name} HF vs QAIC token parity failed: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )


def _decoder_function_op_counts(onnx_path: Path) -> Counter:
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    return Counter(
        node.op_type
        for function_proto in onnx_model.functions
        if function_proto.name.startswith(QEffGptOssDecoderLayer.__name__)
        for node in function_proto.node
    )


def _past_input_ctx_dim_params(onnx_path: Path) -> list[str]:
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    dim_params = []
    for graph_input in onnx_model.graph.input:
        if graph_input.name.startswith(("past_key.", "past_value.")):
            dim = graph_input.type.tensor_type.shape.dim[2]
            dim_params.append(dim.dim_param or str(dim.dim_value))
    return dim_params


def _run_chunked_prefill(qpc_path: str, tokenizer, config) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prefill_session = QAICInferenceSession(qpc_path)
    prefill_session.set_buffers({"logits": np.zeros((1, 1, config.vocab_size), dtype=np.float32)})
    inputs, num_chunks = _prefill_inputs(tokenizer)
    qaic_outputs = {}

    for chunk_idx in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["position_ids"] = inputs["position_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        assert chunk_inputs["input_ids"].shape[1] == PREFILL_SEQ_LEN
        assert chunk_inputs["position_ids"].shape[1] == PREFILL_SEQ_LEN
        qaic_outputs = prefill_session.run(chunk_inputs)

        for layer_idx in range(config.num_hidden_layers):
            inputs[f"past_key.{layer_idx}"] = qaic_outputs[f"past_key.{layer_idx}_RetainedState"]
            inputs[f"past_value.{layer_idx}"] = qaic_outputs[f"past_value.{layer_idx}_RetainedState"]

    return qaic_outputs, inputs


def _decode_inputs_from_prefill_outputs(
    prefill_outputs: dict[str, np.ndarray],
    prefill_inputs: dict[str, np.ndarray],
    config,
) -> dict[str, np.ndarray]:
    decode_inputs = {
        "input_ids": prefill_outputs["logits"].argmax(-1).reshape(1, 1),
        "position_ids": np.max(prefill_inputs["position_ids"], axis=1, keepdims=True) + 1,
    }
    for layer_idx in range(config.num_hidden_layers):
        decode_inputs[f"past_key.{layer_idx}"] = prefill_outputs[f"past_key.{layer_idx}_RetainedState"]
        decode_inputs[f"past_value.{layer_idx}"] = prefill_outputs[f"past_value.{layer_idx}_RetainedState"]
    return decode_inputs


def _run_decode_steps(
    qpc_path: str,
    decode_inputs: dict[str, np.ndarray],
    config,
    decode_steps: int,
) -> np.ndarray:
    decode_session = QAICInferenceSession(qpc_path)
    decode_session.set_buffers({"logits": np.zeros((1, 1, config.vocab_size), dtype=np.float32)})

    qaic_tokens = []
    for _ in range(decode_steps):
        decode_outputs = decode_session.run(_filter_session_inputs(decode_session, decode_inputs))
        next_token = int(decode_outputs["logits"].argmax(-1).reshape(-1)[0])
        qaic_tokens.append(next_token)

        decode_inputs = {
            "input_ids": np.asarray([[next_token]], dtype=np.int64),
            "position_ids": decode_inputs["position_ids"] + 1,
        }
        for layer_idx in range(config.num_hidden_layers):
            decode_inputs[f"past_key.{layer_idx}"] = decode_outputs[f"past_key.{layer_idx}_RetainedState"]
            decode_inputs[f"past_value.{layer_idx}"] = decode_outputs[f"past_value.{layer_idx}_RetainedState"]

    return np.asarray(qaic_tokens)


@pytest.mark.on_qaic
@pytest.mark.disagg_dma
def test_gpt_oss_prefill_chunked_expert_parallel_subfunction_hf_qaic_token_parity(tmp_export_dir, monkeypatch):
    monkeypatch.setenv("NUM_Q_BLOCKS", "1")
    model_hf, tokenizer = _load_gpt_oss_hf_and_tokenizer()
    prompt_len = _prompt_token_len(tokenizer)
    api_runner = _api_runner(model_hf, tokenizer, prompt_len=prompt_len, ctx_len=prompt_len + 1)
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

    qeff_model = _load_gpt_oss_qeff_model(model_hf.config)
    qpc_path = qeff_model.compile(
        compile_dir=str(tmp_export_dir / "gpt-oss-prefill-chunked-expert-parallel"),
        dynamo=False,
        use_onnx_subfunctions=True,
        prefill_only=True,
        enable_chunking=True,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=PREFILL_CTX_LEN,
        batch_size=BATCH_SIZE,
        num_cores=NUM_CORES,
        qaic_config=_moe_qaic_config(),
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        offload_pt_weights=False,
    )

    onnx_path = Path(qeff_model.onnx_path)
    assert_has_subfunctions(onnx_path, qeff_model)
    assert_subfunction_names_match_decoder_class(onnx_path, qeff_model)
    decoder_op_counts = _decoder_function_op_counts(onnx_path)
    assert decoder_op_counts["CtxGather3D"] > 0
    assert decoder_op_counts["CtxScatter3D"] > 0
    assert decoder_op_counts["CtxScatter3DInt"] > 0
    assert qeff_model.hash_params["moe_prefill_flavour"] == "expert_parallel"
    assert qeff_model.hash_params["moe_prefill_expert_parallel_chunk_size"] == MOE_PREFILL_PACKED_CHUNK_SIZE
    assert qeff_model.hash_params["moe_prefill_num_packed_chunks"] == EXPECTED_MOE_PREFILL_NUM_PACKED_CHUNKS

    qaic_outputs, _ = _run_chunked_prefill(qpc_path, tokenizer, model_hf.config)
    qaic_first_token = qaic_outputs["logits"].argmax(-1)
    _assert_token_match(
        "GPT-OSS prefill-only chunked expert-parallel",
        hf_tokens,
        qaic_first_token,
        gen_len=1,
    )


@pytest.mark.on_qaic
@pytest.mark.disagg_dma
def test_gpt_oss_decode_retain_full_kv_subfunction_hf_qaic_token_parity(tmp_export_dir):
    model_hf, tokenizer = _load_gpt_oss_hf_and_tokenizer()
    prompt_len = _prompt_token_len(tokenizer)
    api_runner = _api_runner(model_hf, tokenizer, prompt_len=prompt_len, ctx_len=prompt_len + DECODE_STEPS)
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

    qeff_model = _load_gpt_oss_qeff_model(model_hf.config)
    qpc_path = qeff_model.compile(
        compile_dir=str(tmp_export_dir / "gpt-oss-decode-retain-full-kv"),
        dynamo=False,
        use_onnx_subfunctions=True,
        prefill_only=False,
        prefill_seq_len=1,
        ctx_len=DECODE_CTX_LEN,
        batch_size=BATCH_SIZE,
        num_cores=NUM_CORES,
        retain_full_kv=True,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        offload_pt_weights=False,
    )

    onnx_path = Path(qeff_model.onnx_path)
    assert_has_subfunctions(onnx_path, qeff_model)
    assert_subfunction_names_match_decoder_class(onnx_path, qeff_model)
    assert set(_past_input_ctx_dim_params(onnx_path)) == {"ctx_len"}
    assert qeff_model.hash_params["retain_full_kv"] is True

    prefill_qeff_model = _load_gpt_oss_qeff_model(model_hf.config)
    prefill_qpc_path = prefill_qeff_model.compile(
        compile_dir=str(tmp_export_dir / "gpt-oss-decode-seed-prefill"),
        dynamo=False,
        use_onnx_subfunctions=True,
        prefill_only=True,
        enable_chunking=True,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=PREFILL_CTX_LEN,
        batch_size=BATCH_SIZE,
        num_cores=NUM_CORES,
        qaic_config=_moe_qaic_config(),
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        offload_pt_weights=False,
    )
    prefill_outputs, prefill_inputs = _run_chunked_prefill(prefill_qpc_path, tokenizer, model_hf.config)
    _assert_token_match(
        "GPT-OSS decode seed prefill",
        hf_tokens,
        prefill_outputs["logits"].argmax(-1),
        gen_len=1,
    )

    decode_inputs = _decode_inputs_from_prefill_outputs(prefill_outputs, prefill_inputs, model_hf.config)
    qaic_tokens = _run_decode_steps(
        qpc_path,
        decode_inputs,
        model_hf.config,
        decode_steps=DECODE_STEPS - 1,
    )
    _assert_token_match(
        "GPT-OSS decode-only retain-full-KV multi-token",
        np.asarray(hf_tokens).flatten()[1:DECODE_STEPS],
        qaic_tokens,
        gen_len=DECODE_STEPS - 1,
    )
