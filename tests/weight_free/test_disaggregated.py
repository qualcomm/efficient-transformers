# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Weight-free disaggregated prefill/decode compile tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import skip_on_model_fetch_error

DISAGG_MODEL_PARAMS = [
    pytest.param("glm4_moe", "tiny-random/glm-4-moe", id="glm4-moe"),
    pytest.param("qwen3_moe", "tiny-random/qwen3-moe", id="qwen3-moe"),
    pytest.param("gpt_oss", "tiny-random/gpt-oss-mxfp4", id="gpt-oss"),
]

CTX_LEN = 128
PREFILL_SEQ_LEN = 32
MOE_PREFILL_PACKED_CHUNK_SIZE = 16
MDP_NUM_PARTITIONS = 2


def _compile_dir(tmp_export_dir, name):
    """Create compile directories explicitly, matching the legacy test pattern."""
    compile_dir = tmp_export_dir / name
    compile_dir.mkdir(parents=True, exist_ok=True)
    return str(compile_dir)


PREFILL_NUM_DEVICES = 2
GPT_OSS_TINY_MODEL_ID = "tiny-random/gpt-oss"
GPT_OSS_PARITY_CTX_LEN = 256
GPT_OSS_PARITY_GENERATION_LEN = 40
GPT_OSS_PARITY_FULL_BATCH_SIZE = 2
GPT_OSS_PARITY_NUM_CORES = 16
GPT_OSS_PARITY_NUM_HIDDEN_LAYERS = 2
GPT_OSS_PARITY_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
]
GPT_OSS_PARITY_DMA_CONFIG = pytest.param(
    {
        "model_id": GPT_OSS_TINY_MODEL_ID,
        "prefill_num_devices": 2,
        "decode_num_devices": 1,
        "stages": 2,
        "use_onnx_subfunctions": True,
    },
    id="tiny_model_prefill2_decode1_stages2",
)
QWEN3_MOE_PARITY_CTX_LEN = 256
QWEN3_MOE_PARITY_GENERATION_LEN = 40
QWEN3_MOE_PARITY_FULL_BATCH_SIZE = 1
QWEN3_MOE_PARITY_NUM_CORES = 4
QWEN3_MOE_PARITY_PROMPT_IDS = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
QWEN3_MOE_PARITY_DMA_CONFIG = pytest.param(
    {
        "prefill_num_devices": 2,
        "decode_num_devices": 1,
        "stages": 2,
        "use_onnx_subfunctions": True,
    },
    id="local_qwen3_moe_prefill2_decode1_stages2",
)


def _load_weight_free_model(model_id: str, continuous_batching: bool = False, num_hidden_layers: int | None = 2):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers
    return QEFFAutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        weight_free=True,
        continuous_batching=continuous_batching,
    )


def _build_gpt_oss_parity_config(dtype: str = "float32"):
    config = AutoConfig.from_pretrained(GPT_OSS_TINY_MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = GPT_OSS_PARITY_NUM_HIDDEN_LAYERS
    config.layer_types = [
        "sliding_attention" if i % 2 == 0 else "full_attention" for i in range(GPT_OSS_PARITY_NUM_HIDDEN_LAYERS)
    ]
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    return config


def _load_gpt_oss_parity_hf_model(config) -> AutoModelForCausalLM:
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    # Keep activations numerically stable for strict token parity on device.
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model.eval()


def _run_gpt_oss_hf_torch_fp32(model, tokenizer, prompt: str) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=GPT_OSS_PARITY_GENERATION_LEN,
            min_new_tokens=GPT_OSS_PARITY_GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    prompt_len = input_ids.shape[-1]
    return outputs[0, prompt_len:].detach().cpu().numpy()


def _build_qwen3_moe_parity_config(dtype: str = "float32"):
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=QWEN3_MOE_PARITY_CTX_LEN,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        moe_intermediate_size=16,
        dtype=dtype,
    )
    config.torch_dtype = getattr(torch, dtype)
    return config


def _load_qwen3_moe_parity_hf_model(config) -> Qwen3MoeForCausalLM:
    torch.manual_seed(42)
    model = Qwen3MoeForCausalLM(config)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model.eval()


def _run_hf_torch_fp32_from_input_ids(model, input_ids: np.ndarray, generation_len: int) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    prompt_ids = torch.from_numpy(input_ids).to(dtype=torch.long)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=generation_len,
            min_new_tokens=generation_len,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    prompt_len = prompt_ids.shape[-1]
    return outputs[0, prompt_len:].detach().cpu().numpy()


def _prepare_gpt_oss_parity_prompt(tokenizer, prompt: str):
    enc = tokenizer(prompt, return_tensors="np", padding=True)
    prompt_len = enc["input_ids"].shape[1]
    num_chunks = -(prompt_len // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN

    enc = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    input_ids = enc["input_ids"]
    position_ids = np.where(enc["attention_mask"], np.arange(padded_len), -1)
    return input_ids, position_ids.astype(np.int64), num_chunks, prompt_len


def _prepare_parity_input_ids(input_ids: np.ndarray):
    prompt_len = input_ids.shape[1]
    num_chunks = -(prompt_len // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    padded_input_ids = np.pad(input_ids, ((0, 0), (0, padded_len - prompt_len)), constant_values=0)
    position_ids = np.full_like(padded_input_ids, -1, dtype=np.int64)
    position_ids[:, :prompt_len] = np.arange(prompt_len)
    return padded_input_ids.astype(np.int64), position_ids, num_chunks, prompt_len


def _next_token_ids_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 2:
        return logits.argmax(axis=-1).astype(np.int64)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _prefill_gpt_oss_parity_slot(prefill_session, input_ids, position_ids, num_chunks, slot: int, slot_kv_view):
    chunk_inputs = {"batch_index": np.array([[slot]], dtype=np.int64)}
    exec_idx = None
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = input_ids[:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = position_ids[:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        last_chunk = i == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=slot_kv_view if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    first_token = int(_next_token_ids_from_logits(prefill_out["logits"])[0])
    next_pos = int(np.max(position_ids)) + 1
    return first_token, next_pos


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", DISAGG_MODEL_PARAMS)
def test_weight_free_disaggregated_prefill_and_decode(model_type, model_id, tmp_export_dir):
    """Compile weight-free decode and chunked-prefill QPCs for disaggregated serving."""
    try:
        qeff_model = _load_weight_free_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    moe_config = {"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}}
    common = {
        "ctx_len": CTX_LEN,
        "num_cores": 4,
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
        "retain_full_kv": True,
    }

    decode_qpc = qeff_model.compile(
        compile_dir=_compile_dir(tmp_export_dir, f"{model_type}_decode"),
        prefill_seq_len=1,
        **common,
    )
    prefill_qpc = qeff_model.compile(
        compile_dir=_compile_dir(tmp_export_dir, f"{model_type}_prefill_mdp"),
        prefill_seq_len=PREFILL_SEQ_LEN,
        prefill_only=True,
        enable_chunking=True,
        num_devices=PREFILL_NUM_DEVICES,
        mdp_num_partitions=MDP_NUM_PARTITIONS,
        qaic_config=moe_config,
        **common,
    )

    assert decode_qpc
    assert prefill_qpc


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", DISAGG_MODEL_PARAMS)
def test_weight_free_disaggregated_continuous_batching(model_type, model_id, tmp_export_dir):
    """Compile CB decode and chunked-prefill QPCs for weight-free disaggregated serving."""
    try:
        qeff_model = _load_weight_free_model(model_id, continuous_batching=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    full_batch_size = 2
    common = {
        "ctx_len": CTX_LEN,
        "full_batch_size": full_batch_size,
        "num_cores": 4,
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        "split_retained_state_io": True,
        "retain_full_kv": True,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": True,
    }

    decode_qpc = qeff_model.compile(
        compile_dir=_compile_dir(tmp_export_dir, f"{model_type}_cb_decode"),
        prefill_seq_len=1,
        **common,
    )
    prefill_qpc = qeff_model.compile(
        compile_dir=_compile_dir(tmp_export_dir, f"{model_type}_cb_prefill"),
        prefill_seq_len=PREFILL_SEQ_LEN,
        prefill_only=True,
        enable_chunking=True,
        num_devices=PREFILL_NUM_DEVICES,
        mdp_num_partitions=MDP_NUM_PARTITIONS,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
        **common,
    )

    assert decode_qpc
    assert prefill_qpc


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.disagg_dma
@pytest.mark.parametrize("dma_config", [QWEN3_MOE_PARITY_DMA_CONFIG])
def test_weight_free_qwen3_moe_disagg_cb_kv_handoff_and_hf_parity(
    manual_cleanup,
    tmp_path,
    tmp_export_dir,
    dma_config,
):
    """Validate local Qwen3-MoE weight-free CB disagg KV handoff and HF fp32 token parity."""
    config = _build_qwen3_moe_parity_config(dtype="float32")
    hf_model = _load_qwen3_moe_parity_hf_model(config)
    checkpoint_dir = tmp_path / "qwen3_moe_fp32_checkpoint"
    hf_model.save_pretrained(checkpoint_dir)

    hf_tokens = _run_hf_torch_fp32_from_input_ids(
        hf_model,
        QWEN3_MOE_PARITY_PROMPT_IDS,
        QWEN3_MOE_PARITY_GENERATION_LEN,
    )
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        config=config,
        trust_remote_code=True,
        weight_free=True,
        continuous_batching=True,
    )

    sessions = []
    compiled_onnx_paths = {}
    try:
        decode_qpc_path = qeff_model.compile(
            compile_dir=_compile_dir(tmp_export_dir, "qwen3_moe_wf_cb_decode_parity"),
            prefill_seq_len=1,
            ctx_len=QWEN3_MOE_PARITY_CTX_LEN,
            full_batch_size=QWEN3_MOE_PARITY_FULL_BATCH_SIZE,
            num_cores=QWEN3_MOE_PARITY_NUM_CORES,
            num_devices=dma_config["decode_num_devices"],
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            offload_pt_weights=False,
            split_retained_state_io=True,
            retain_full_kv=True,
            use_onnx_subfunctions=dma_config.get("use_onnx_subfunctions", True),
        )
        compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.onnx_path, "decode")

        prefill_qpc_path = qeff_model.compile(
            compile_dir=_compile_dir(tmp_export_dir, "qwen3_moe_wf_cb_prefill_parity"),
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=QWEN3_MOE_PARITY_CTX_LEN,
            full_batch_size=QWEN3_MOE_PARITY_FULL_BATCH_SIZE,
            num_cores=QWEN3_MOE_PARITY_NUM_CORES,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
            num_devices=dma_config["prefill_num_devices"],
            mdp_num_partitions=dma_config["stages"],
            split_retained_state_io=True,
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
            use_onnx_subfunctions=dma_config.get("use_onnx_subfunctions", True),
        )
        compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.onnx_path, "prefill")
        print(f"Qwen3-MoE weight-free disagg CB ONNX paths: {compiled_onnx_paths}")

        prefill_session = QAICInferenceSession(
            prefill_qpc_path,
            kv_dma_share=True,
            full_batch_size=QWEN3_MOE_PARITY_FULL_BATCH_SIZE,
            cluster_id="prefill",
        )
        decode_session = QAICInferenceSession(
            decode_qpc_path,
            kv_dma_share=True,
            full_batch_size=QWEN3_MOE_PARITY_FULL_BATCH_SIZE,
            cluster_id="decode",
        )
        sessions.extend([prefill_session, decode_session])

        assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"

        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
        assert kv_caches[0].shape[0] == QWEN3_MOE_PARITY_FULL_BATCH_SIZE, (
            f"decode KV batch dim {kv_caches[0].shape[0]} != full_batch_size {QWEN3_MOE_PARITY_FULL_BATCH_SIZE}"
        )

        input_ids, position_ids, num_chunks, prompt_len = _prepare_parity_input_ids(QWEN3_MOE_PARITY_PROMPT_IDS)
        assert all(np.all(kv[0] == 0) for kv in kv_caches), "slot 0 KV row is not zero before prefill"
        first_token, next_pos = _prefill_gpt_oss_parity_slot(
            prefill_session,
            input_ids,
            position_ids,
            num_chunks,
            slot=0,
            slot_kv_view=[kv[:1] for kv in kv_caches],
        )
        written = [kv[0, :, :prompt_len, :] for kv in kv_caches]
        assert all(np.any(w != 0) for w in written), (
            "slot 0 KV row is still zero after prefill -- DMA handoff did not write it"
        )

        pre_decode_kv = [kv.copy() for kv in kv_caches]
        decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
        batch_index = np.array([[0]], dtype=np.int64)
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = {
            "input_ids": np.array([[first_token]], dtype=np.int64),
            "position_ids": np.array([[next_pos]], dtype=np.int64),
            "batch_index": batch_index,
        }
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_out = decode_session.get_outputs(index=exec_idx)
        decode_logits = decode_out["logits"].reshape(
            QWEN3_MOE_PARITY_FULL_BATCH_SIZE,
            -1,
            decode_out["logits"].shape[-1],
        )[:, -1, :]
        second_token = int(np.argmax(decode_logits, axis=-1)[0])

        for kv_before, kv_after in zip(pre_decode_kv, kv_caches):
            assert np.array_equal(kv_before[0, :, :prompt_len, :], kv_after[0, :, :prompt_len, :]), (
                "slot 0: decode step overwrote prefill-written KV prefix"
            )
            assert np.any(kv_after[0, :, next_pos, :] != 0), (
                f"slot 0: decode step did not write KV at the new position {next_pos}"
            )

        qaic_tokens = [first_token, second_token]
        pos = np.array([[next_pos + 1]], dtype=np.int64)
        last_token = np.array([[second_token]], dtype=np.int64)
        for _ in range(QWEN3_MOE_PARITY_GENERATION_LEN - 2):
            decode_session.set_data_for_kv_handoff(
                kv_caches + kv_caches,
                [("batch_index", 0), ("ctx_start", 0)],
                index=decode_session.decode_execObj_idx,
                buff_map=decode_kv_map,
            )
            decode_inputs = {
                "input_ids": last_token,
                "position_ids": pos,
                "batch_index": batch_index,
            }
            exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
            decode_session.complete_inf(exec_idx, is_prefill=False)
            out = decode_session.get_outputs(index=exec_idx)
            logits = out["logits"].reshape(QWEN3_MOE_PARITY_FULL_BATCH_SIZE, -1, out["logits"].shape[-1])[:, -1, :]
            next_token = int(np.argmax(logits, axis=-1)[0])
            qaic_tokens.append(next_token)
            last_token = np.array([[next_token]], dtype=np.int64)
            pos = pos + 1
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [getattr(qeff_model, "onnx_path", None)]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    qaic_tokens = np.array(qaic_tokens, dtype=np.int64)
    matches = hf_tokens == qaic_tokens
    num_matched = int(np.cumprod(matches).sum())
    print(f"HF Torch fp32 tokens        : {hf_tokens.tolist()}")
    print(f"Weight-free QAIC tokens     : {qaic_tokens.tolist()}")
    print(f"Matched leading tokens      : {num_matched}/{QWEN3_MOE_PARITY_GENERATION_LEN}")
    if not matches.all():
        first_mismatch = int(np.argmin(matches))
        raise AssertionError(
            f"Qwen3-MoE tokens don't match HF Torch fp32 output; first mismatch at token index "
            f"{first_mismatch} (matched {num_matched}/{QWEN3_MOE_PARITY_GENERATION_LEN} leading tokens): "
            f"HF={hf_tokens[first_mismatch]} vs QAIC={qaic_tokens[first_mismatch]}"
        )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.disagg_dma
@pytest.mark.parametrize("dma_config", [GPT_OSS_PARITY_DMA_CONFIG])
def test_weight_free_gpt_oss_disagg_cb_kv_handoff_and_hf_parity(
    manual_cleanup,
    tmp_path,
    tmp_export_dir,
    dma_config,
):
    """Validate weight-free CB disagg KV handoff and token parity against HF fp32."""
    torch.manual_seed(42)

    model_id = dma_config["model_id"]
    use_onnx_subfunctions = dma_config.get("use_onnx_subfunctions", True)

    try:
        config = _build_gpt_oss_parity_config(dtype="float32")
        hf_model = _load_gpt_oss_parity_hf_model(config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    checkpoint_dir = tmp_path / "gpt_oss_fp32_checkpoint"
    hf_model.save_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_tokens = [_run_gpt_oss_hf_torch_fp32(hf_model, tokenizer, prompt) for prompt in GPT_OSS_PARITY_PROMPTS]
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        config=config,
        trust_remote_code=True,
        weight_free=True,
        continuous_batching=True,
    )

    sessions = []
    compiled_onnx_paths = {}
    try:
        decode_qpc_path = qeff_model.compile(
            compile_dir=_compile_dir(tmp_export_dir, "gpt_oss_wf_cb_decode_parity"),
            prefill_seq_len=1,
            ctx_len=GPT_OSS_PARITY_CTX_LEN,
            full_batch_size=GPT_OSS_PARITY_FULL_BATCH_SIZE,
            num_cores=GPT_OSS_PARITY_NUM_CORES,
            num_devices=dma_config["decode_num_devices"],
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            offload_pt_weights=False,
            split_retained_state_io=True,
            retain_full_kv=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
        compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.onnx_path, "decode")

        prefill_qpc_path = qeff_model.compile(
            compile_dir=_compile_dir(tmp_export_dir, "gpt_oss_wf_cb_prefill_parity"),
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=GPT_OSS_PARITY_CTX_LEN,
            full_batch_size=GPT_OSS_PARITY_FULL_BATCH_SIZE,
            num_cores=GPT_OSS_PARITY_NUM_CORES,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
            num_devices=dma_config["prefill_num_devices"],
            mdp_num_partitions=dma_config["stages"],
            split_retained_state_io=True,
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
        compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.onnx_path, "prefill")
        print(f"Weight-free disagg CB ONNX paths: {compiled_onnx_paths}")

        prefill_session = QAICInferenceSession(
            prefill_qpc_path,
            kv_dma_share=True,
            full_batch_size=GPT_OSS_PARITY_FULL_BATCH_SIZE,
            cluster_id="prefill",
        )
        decode_session = QAICInferenceSession(
            decode_qpc_path,
            kv_dma_share=True,
            full_batch_size=GPT_OSS_PARITY_FULL_BATCH_SIZE,
            cluster_id="decode",
        )
        sessions.extend([prefill_session, decode_session])

        assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"

        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
        assert kv_caches[0].shape[0] == GPT_OSS_PARITY_FULL_BATCH_SIZE, (
            f"decode KV batch dim {kv_caches[0].shape[0]} != full_batch_size {GPT_OSS_PARITY_FULL_BATCH_SIZE}"
        )
        decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

        first_tokens = [None] * GPT_OSS_PARITY_FULL_BATCH_SIZE
        next_pos = [None] * GPT_OSS_PARITY_FULL_BATCH_SIZE
        prompt_len = [None] * GPT_OSS_PARITY_FULL_BATCH_SIZE
        for slot, prompt in enumerate(GPT_OSS_PARITY_PROMPTS):
            input_ids, position_ids, num_chunks, plen = _prepare_gpt_oss_parity_prompt(tokenizer, prompt)
            prompt_len[slot] = plen

            assert all(np.all(kv[slot] == 0) for kv in kv_caches), f"slot {slot} KV row is not zero before prefill"

            slot_kv_view = [kv[slot : slot + 1] for kv in kv_caches]
            ft, npos = _prefill_gpt_oss_parity_slot(
                prefill_session, input_ids, position_ids, num_chunks, slot, slot_kv_view
            )
            first_tokens[slot] = ft
            next_pos[slot] = npos

            written = [kv[slot, :, : prompt_len[slot], :] for kv in kv_caches]
            assert all(np.any(w != 0) for w in written), (
                f"slot {slot} KV row is still zero after prefill -- DMA handoff did not write it"
            )
            for other in range(GPT_OSS_PARITY_FULL_BATCH_SIZE):
                if other == slot or first_tokens[other] is None:
                    continue
                assert np.any(kv_caches[0][other] != 0), (
                    f"slot {other} KV row went to zero after prefilling slot {slot} -- cross-slot corruption"
                )

        pre_decode_kv = [kv.copy() for kv in kv_caches]
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        input_ids = np.array([[first_tokens[s]] for s in range(GPT_OSS_PARITY_FULL_BATCH_SIZE)], dtype=np.int64)
        position_ids = np.array([[next_pos[s]] for s in range(GPT_OSS_PARITY_FULL_BATCH_SIZE)], dtype=np.int64)
        batch_index = np.array([[s] for s in range(GPT_OSS_PARITY_FULL_BATCH_SIZE)], dtype=np.int64)
        decode_inputs = {"input_ids": input_ids, "position_ids": position_ids, "batch_index": batch_index}
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_out = decode_session.get_outputs(index=exec_idx)
        decode_logits = decode_out["logits"].reshape(
            GPT_OSS_PARITY_FULL_BATCH_SIZE,
            -1,
            decode_out["logits"].shape[-1],
        )[:, -1, :]
        second_tokens = np.argmax(decode_logits, axis=-1)

        for slot in range(GPT_OSS_PARITY_FULL_BATCH_SIZE):
            for kv_before, kv_after in zip(pre_decode_kv, kv_caches):
                prefix_before = kv_before[slot, :, : prompt_len[slot], :]
                prefix_after = kv_after[slot, :, : prompt_len[slot], :]
                assert np.array_equal(prefix_before, prefix_after), (
                    f"slot {slot}: decode step overwrote prefill-written KV prefix "
                    f"(positions 0..{prompt_len[slot]}) -- handoff/write-back is wiring the wrong offset"
                )
                new_pos_after = kv_after[slot, :, next_pos[slot], :]
                assert np.any(new_pos_after != 0), (
                    f"slot {slot}: decode step did not write KV at the new position {next_pos[slot]} "
                    "-- write-back side of the handoff is not wired"
                )

        gen_tokens = [[first_tokens[s], int(second_tokens[s])] for s in range(GPT_OSS_PARITY_FULL_BATCH_SIZE)]
        pos = position_ids + 1
        last_token = second_tokens.reshape(GPT_OSS_PARITY_FULL_BATCH_SIZE, 1)
        for _ in range(GPT_OSS_PARITY_GENERATION_LEN - 2):
            decode_session.set_data_for_kv_handoff(
                kv_caches + kv_caches,
                [("batch_index", 0), ("ctx_start", 0)],
                index=decode_session.decode_execObj_idx,
                buff_map=decode_kv_map,
            )
            decode_inputs = {
                "input_ids": last_token.astype(np.int64),
                "position_ids": pos.astype(np.int64),
                "batch_index": batch_index,
            }
            exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
            decode_session.complete_inf(exec_idx, is_prefill=False)
            out = decode_session.get_outputs(index=exec_idx)
            logits = out["logits"].reshape(GPT_OSS_PARITY_FULL_BATCH_SIZE, -1, out["logits"].shape[-1])[:, -1, :]
            next_tokens = np.argmax(logits, axis=-1)
            for slot in range(GPT_OSS_PARITY_FULL_BATCH_SIZE):
                gen_tokens[slot].append(int(next_tokens[slot]))
            last_token = next_tokens.reshape(GPT_OSS_PARITY_FULL_BATCH_SIZE, 1)
            pos = pos + 1
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [getattr(qeff_model, "onnx_path", None)]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    for slot in range(GPT_OSS_PARITY_FULL_BATCH_SIZE):
        qaic_tokens = np.array(gen_tokens[slot], dtype=np.int64)
        ref_tokens = hf_tokens[slot]
        matches = ref_tokens == qaic_tokens
        num_matched = int(np.cumprod(matches).sum())
        print(f"\nslot[{slot}] prompt: {GPT_OSS_PARITY_PROMPTS[slot]}")
        print(f"HF Torch fp32 tokens        : {ref_tokens.tolist()}")
        print(f"Weight-free QAIC tokens     : {qaic_tokens.tolist()}")
        print(f"Matched leading tokens      : {num_matched}/{GPT_OSS_PARITY_GENERATION_LEN}")
        if not matches.all():
            first_mismatch = int(np.argmin(matches))
            raise AssertionError(
                f"slot {slot}: tokens don't match HF Torch fp32 output; "
                f"first mismatch at token index {first_mismatch} "
                f"(matched {num_matched}/{GPT_OSS_PARITY_GENERATION_LEN} leading tokens): "
                f"HF={ref_tokens[first_mismatch]} vs QAIC={qaic_tokens[first_mismatch]}"
            )
