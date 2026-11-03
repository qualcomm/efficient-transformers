# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free causal-LM and embedding tests on QAIC.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
All tests run with weight_free=True (set on the QEff model, which forces
dynamo=True internally). Whisper runs with ONNX subfunctions disabled because
they are not supported for that model.

Covers causal-LM generation, HF/QAIC parity, and embedding parity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.exporter.weight_free import resolve_weight_spec_path
from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP
from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForSpeechSeq2Seq,
)
from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.constants import WAV2VEC2_MAX_SEQ_LEN
from QEfficient.utils.device_utils import get_available_device_id, get_qaic_mdp_device_groups

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_ASR_MODEL_IDS,
    WEIGHT_FREE_BLOCKING_MODE_CASES,
    WEIGHT_FREE_BLOCKING_QAIC_CASES,
    WEIGHT_FREE_EMBEDDING_MODEL_PARAMS,
    WEIGHT_FREE_QAIC_MODEL_PARAMS,
    assert_blocked_kv_ops_for_mode,
    assert_has_subfunctions,
    exported_onnx_path,
    load_causal_lm_config,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)

WEIGHT_FREE_ASR_QEFF_CLASSES = {
    "wav2vec2": QEFFAutoModelForCTC,
    "whisper": QEFFAutoModelForSpeechSeq2Seq,
}
WEIGHT_FREE_ASR_USE_SUBFUNCTIONS = {
    "wav2vec2": True,
    "whisper": False,
}
ASR_GENERATION_LEN = 8
ASR_WAV2VEC2_SEQ_LEN = WAV2VEC2_MAX_SEQ_LEN
WEIGHT_FREE_ASR_QAIC_MODEL_PARAMS = [
    pytest.param(
        model_type,
        model_id,
        marks=pytest.mark.xdist_group(name=f"qaic-runtime-asr-{model_type}"),
        id=model_type,
    )
    for model_type, model_id in sorted(WEIGHT_FREE_ASR_MODEL_IDS.items())
]

BLOCKING_PARITY_PROMPT = "Hello world"
HEAD_BLOCKING_NUM_DEVICES = 4


def _blocking_compile_options(blocking_key: str) -> dict:
    if blocking_key in {"h", "hq", "hkv"}:
        device_groups = get_qaic_mdp_device_groups(devices_per_group=HEAD_BLOCKING_NUM_DEVICES)
        if not device_groups:
            pytest.skip(f"No available QAIC MDP device group with {HEAD_BLOCKING_NUM_DEVICES} devices")
        return {"num_devices": HEAD_BLOCKING_NUM_DEVICES, "user_tiled": True}

    device_ids = get_available_device_id()
    if device_ids is None:
        pytest.skip("No available QAIC device")
    return {}


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model_type,model_id,pooling", WEIGHT_FREE_EMBEDDING_MODEL_PARAMS)
def test_weight_free_embedding_hw_hf_parity(model_type, model_id, pooling, tmp_export_dir):
    """HF embedding output matches weight-free embedding output on QAIC."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model_hf = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
        )
        model_hf.eval()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if model_type in {"jina", "nomic"}:
            raise
        skip_on_model_fetch_error(exc, model_id)

    inputs = tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        hf_hidden_states = model_hf(**inputs).last_hidden_state
        hf_output = POOLING_MAP[pooling](hf_hidden_states, inputs["attention_mask"]).cpu().numpy()

    qeff_model = QEFFAutoModel.from_pretrained(model_id, pooling=pooling, weight_free=True)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"{model_type}_export",
            dynamo=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"{model_type}_compile"),
        seq_len=32,
        batch_size=1,
        num_cores=16,
    )
    qaic_output = qeff_model.generate(inputs=inputs)

    assert qaic_output is not None, f"Weight-free QAIC generate returned None for {model_type}"
    qaic_embeddings = qaic_output["output"]
    if qaic_embeddings.ndim == 3:
        qaic_embeddings = qaic_embeddings[:, 0, :]
    mad = np.mean(np.abs(hf_output - qaic_embeddings))
    assert mad <= 1e-2, f"HF PT vs weight-free QAIC parity failed for {model_type}: MAD={mad}"


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_generate_fp16(model_type, model_id, tmp_export_dir):
    """End-to-end weight-free export -> compile -> generate on real QAIC hardware."""
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_gen_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    tokenizer = load_tokenizer(model_id)
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert output is not None
    assert output.generated_texts is not None


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_hw_hf_parity(model_type, model_id, tmp_export_dir):
    """HF PT tokens == weight-free QAIC FP16 tokens (exact equality)."""
    from QEfficient.utils.run_utils import ApiRunner

    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_hw_parity_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_hw_parity_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )

    assert qaic_output is not None, "QAIC generate returned None"
    if hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None:
        gen_len = CTX_LEN - PROMPT_LEN
        qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
        assert np.array_equal(hf_tokens, qaic_tokens), (
            f"Weight-free HW/HF parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
        )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_vs_legacy_qaic_parity(model_type, model_id, tmp_export_dir):
    """Weight-free-compiled and legacy dynamo-compiled QPCs produce identical tokens on QAIC."""
    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
        if model_type == "gpt_oss":
            model_hf = model_hf.to(torch.float32)
            model_hf.config.torch_dtype = torch.float32
            model_hf.config.dtype = torch.float32
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    # Legacy/dynamo leg — real weights, no weight-free export.
    qeff_legacy = QEFFAutoModelForCausalLM(model_hf)

    legacy_onnx_path = exported_onnx_path(
        qeff_legacy.export(
            tmp_export_dir / "legacy_export",
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_legacy.compile(
        onnx_path=str(legacy_onnx_path),
        compile_dir=str(tmp_export_dir / "legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    legacy_output = qeff_legacy.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert legacy_output is not None, "Legacy QAIC generate returned None"

    # Weight-free leg — meta-device model, matching layer count.
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_weight_free = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    weight_free_onnx_path = exported_onnx_path(
        qeff_weight_free.export(
            tmp_export_dir / "wf_vs_legacy_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_weight_free.compile(
        onnx_path=str(weight_free_onnx_path),
        compile_dir=str(tmp_export_dir / "wf_vs_legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    weight_free_output = qeff_weight_free.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert weight_free_output is not None, "Weight-free QAIC generate returned None"

    if (
        hasattr(legacy_output, "generated_ids")
        and legacy_output.generated_ids is not None
        and hasattr(weight_free_output, "generated_ids")
        and weight_free_output.generated_ids is not None
    ):
        legacy_tokens = legacy_output.generated_ids[0].flatten()
        weight_free_tokens = weight_free_output.generated_ids[0].flatten()
        assert np.array_equal(legacy_tokens, weight_free_tokens), (
            f"Weight-free vs legacy QAIC parity failed for {model_id}: "
            f"legacy={legacy_tokens.tolist()}, weight_free={weight_free_tokens.tolist()}"
        )


@pytest.mark.weight_free
@pytest.mark.weight_free_blocking
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id,blocking_key", WEIGHT_FREE_BLOCKING_QAIC_CASES)
def test_weight_free_blocked_hw_hf_parity_and_artifact_metrics(
    model_type,
    model_id,
    blocking_key,
    tmp_export_dir,
):
    """HF PT tokens match blocked weight-free QAIC tokens."""
    if model_type == "gpt_oss" and blocking_key == "hq":
        pytest.xfail("tokens not matching - to fix")

    from QEfficient.utils.run_utils import ApiRunner

    use_onnx_subfunctions = True
    qaic_config = dict(WEIGHT_FREE_BLOCKING_MODE_CASES[blocking_key])
    prompts = [BLOCKING_PARITY_PROMPT]

    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompts,
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

    config = load_causal_lm_config(model_id)
    config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        weight_free=True,
        qaic_config=qaic_config,
    )

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"wf_blocking_{model_type}_{blocking_key}_{use_onnx_subfunctions}_export",
            qaic_config=qaic_config,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=False,
        )
    )

    blocking_config = qeff_model.hash_params.get("blocking_kwargs")
    assert blocking_config is not None
    assert blocking_config.mode == BlockingMode(qaic_config["blocking_mode"])
    if use_onnx_subfunctions:
        assert_has_subfunctions(onnx_path, qeff_model)
    assert_blocked_kv_ops_for_mode(onnx_path, qeff_model, blocking_key)

    compile_options = _blocking_compile_options(blocking_key)
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"wf_blocking_{model_type}_{blocking_key}_{use_onnx_subfunctions}_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        qaic_config=qaic_config,
        use_onnx_subfunctions=use_onnx_subfunctions,
        **compile_options,
    )

    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
    )
    assert qaic_output is not None, "Blocked weight-free QAIC generate returned None"
    assert hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None

    gen_len = CTX_LEN - PROMPT_LEN
    qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
    qaic_text = getattr(qaic_output, "generated_texts", None)
    print("Blocked weight-free QAIC output:")
    print("Prompt:", repr(prompts))
    print("Completion:", repr(qaic_text))
    hf_tokens_for_compare = np.asarray(hf_tokens).flatten()[:gen_len]
    print("HF token IDs:", hf_tokens_for_compare.tolist())
    print("QAIC token IDs:", qaic_tokens.tolist())
    assert np.array_equal(hf_tokens_for_compare, qaic_tokens), (
        f"Blocked weight-free HW/HF parity failed for {model_id} mode={blocking_key}: "
        f"HF={hf_tokens_for_compare.tolist()}, QAIC={qaic_tokens.tolist()}"
    )


def _load_weight_free_asr(model_type, model_id):
    try:
        model_hf, processor = load_hf_model(model_id, model_type=model_type)
        qeff_model = WEIGHT_FREE_ASR_QEFF_CLASSES[model_type].from_pretrained(
            model_id,
            trust_remote_code=True,
            weight_free=True,
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)
    return model_hf, processor, qeff_model


def _export_weight_free_asr(qeff_model, model_type, export_dir):
    onnx_path = exported_onnx_path(
        qeff_model.export(
            export_dir,
            use_onnx_subfunctions=WEIGHT_FREE_ASR_USE_SUBFUNCTIONS[model_type],
            offload_pt_weights=False,
        )
    )
    assert resolve_weight_spec_path(onnx_path).is_file()
    return onnx_path


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_ASR_QAIC_MODEL_PARAMS)
def test_weight_free_asr_generate_fp16(model_type, model_id, tmp_export_dir):
    """Weight-free ASR export, QAIC compile, and execution smoke test."""
    _, processor, qeff_model = _load_weight_free_asr(model_type, model_id)
    onnx_path = _export_weight_free_asr(qeff_model, model_type, tmp_export_dir / "wf_asr_export")

    if model_type == "wav2vec2":
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_export_dir / "wf_asr_compile"),
            seq_len=ASR_WAV2VEC2_SEQ_LEN,
            batch_size=BATCH_SIZE,
            num_cores=16,
            use_onnx_subfunctions=True,
        )
        output = qeff_model.generate(processor, inputs=np.zeros(16000, dtype=np.float32))
        assert output is not None
        assert len(output) == BATCH_SIZE
    else:
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_export_dir / "wf_asr_compile"),
            ctx_len=ASR_GENERATION_LEN,
            batch_size=BATCH_SIZE,
            num_cores=16,
            use_onnx_subfunctions=False,
        )
        inputs = processor(np.zeros(16000, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
        output = qeff_model.generate(inputs=inputs, generation_len=ASR_GENERATION_LEN)
        assert output is not None
        assert output.generated_ids is not None


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_type,model_id", [pytest.param("wav2vec2", WEIGHT_FREE_ASR_MODEL_IDS["wav2vec2"])])
def test_weight_free_asr_hw_hf_parity(model_type, model_id, tmp_export_dir):
    """Weight-free QAIC ASR output matches the HF PyTorch reference output."""
    model_hf, processor, qeff_model = _load_weight_free_asr(model_type, model_id)
    audio = np.zeros(16000, dtype=np.float32)

    if model_type == "wav2vec2":
        hf_inputs = processor(
            audio,
            return_tensors="pt",
            max_length=ASR_WAV2VEC2_SEQ_LEN,
            truncation=True,
            padding="max_length",
        )
        with torch.no_grad():
            hf_logits = model_hf(input_values=hf_inputs.input_values).logits
        hf_output = processor.batch_decode(hf_logits.argmax(dim=-1))
    else:
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        decoder_input_ids = torch.full(
            (BATCH_SIZE, 1),
            model_hf.config.decoder_start_token_id,
            dtype=torch.int64,
        )
        hf_output = model_hf.generate(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=ASR_GENERATION_LEN,
            do_sample=False,
        )

    onnx_path = _export_weight_free_asr(qeff_model, model_type, tmp_export_dir / "wf_asr_parity_export")
    if model_type == "wav2vec2":
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_export_dir / "wf_asr_parity_compile"),
            seq_len=ASR_WAV2VEC2_SEQ_LEN,
            batch_size=BATCH_SIZE,
            num_cores=16,
            use_onnx_subfunctions=True,
        )
        qaic_output = qeff_model.generate(processor, inputs=audio)
        assert hf_output == qaic_output
    else:
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_export_dir / "wf_asr_parity_compile"),
            ctx_len=ASR_GENERATION_LEN,
            batch_size=BATCH_SIZE,
            num_cores=16,
            use_onnx_subfunctions=False,
        )
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        qaic_output = qeff_model.generate(inputs=inputs, generation_len=ASR_GENERATION_LEN)
        assert np.array_equal(hf_output.numpy(), qaic_output.generated_ids)
