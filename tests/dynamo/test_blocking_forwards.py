# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo blocking QAIC tests using explicit tiny model configs."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.generation.cloud_infer import QAICInferenceSession, is_retained_state_name
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.device_utils import get_available_device_id, get_qaic_mdp_device_groups
from QEfficient.utils.generate_inputs import InputHandler

from ._helpers import (
    DTYPE,
    DYNAMO,
    assert_blocked_kv_ops_for_mode,
    exported_onnx_path,
)

VOCAB_SIZE_FLOOR = 512
BATCH_SIZE = 1
FULL_BATCH_SIZE = 4
PROMPT_LEN_BLOCKING = 32
CTX_LEN_BLOCKING = 128
HEAD_BLOCK_SIZE = 2
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
NUM_BATCH_BLOCKS = 2
HEADPAR_SPLIT = 4
TOKENIZER_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = "hello world"
CB_PROMPTS = ["hello world", "quick brown fox", "machine learning", "open source"]


@dataclass(frozen=True)
class BlockingQaicCase:
    model_label: str
    config_factory: Callable[[int], object]
    qaic_config: dict
    num_devices: int = 1
    batch_size: int = BATCH_SIZE
    prompt_len: int = PROMPT_LEN_BLOCKING
    ctx_len: int = CTX_LEN_BLOCKING
    xfail_reason: str | None = None


@dataclass
class BlockingCompiledArtifact:
    qeff_model: QEFFAutoModelForCausalLM
    model_hf: torch.nn.Module
    tokenizer: AutoTokenizer
    compile_dir: Path
    qpc_path: Path
    onnx_path: Path


_BLOCKING_QPC_CACHE: dict[tuple, BlockingCompiledArtifact] = {}


def _make_tiny_llama_config(vocab_size: int):
    return LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
        vocab_size=vocab_size,
        max_position_embeddings=CTX_LEN_BLOCKING,
        pad_token_id=0,
    )


def _make_tiny_qwen3_moe_config(vocab_size: int):
    return AutoConfig.for_model(
        "qwen3_moe",
        max_position_embeddings=CTX_LEN_BLOCKING,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=32,
        vocab_size=vocab_size,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        head_dim=32,
        pad_token_id=0,
        dtype="float32",
    )


def _make_tiny_gpt_oss_config(vocab_size: int):
    pytest.importorskip("transformers")
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    return GptOssConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=128,
        head_dim=32,
        vocab_size=vocab_size,
        max_position_embeddings=CTX_LEN_BLOCKING,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=32,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default"},
    )


def _make_kimi_k25_language_config(vocab_size: int):
    from tests.utils.load_kimi_utils import KIMI_K25_MODEL_NAME, get_kimi_k25_test_config

    config_path = Path(__file__).parents[1] / "configs" / "image_text_model_configs.json"
    model_configs = json.loads(config_path.read_text())["image_text_models"]
    model_config_dict = {model["model_name"]: model for model in model_configs}
    config = get_kimi_k25_test_config(KIMI_K25_MODEL_NAME, model_config_dict)
    config.text_config.vocab_size = vocab_size
    config.text_config.max_position_embeddings = CTX_LEN_BLOCKING
    return config.text_config


def _make_tiny_qwen3_vl_moe_config(vocab_size: int):
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeConfig,
        Qwen3VLMoeTextConfig,
        Qwen3VLMoeVisionConfig,
    )

    text_config = Qwen3VLMoeTextConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_scaling={"rope_type": "default", "mrope_section": [11, 11, 10]},
        pad_token_id=0,
        dtype="float32",
    )
    vision_config = Qwen3VLMoeVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=4,
        temporal_patch_size=1,
        spatial_merge_size=1,
        out_hidden_size=16,
        num_position_embeddings=64,
        deepstack_visual_indexes=[],
        dtype="float32",
    )
    return Qwen3VLMoeConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _qaic_config(mode: str, **kwargs) -> dict:
    return {"blocking_mode": mode, **kwargs}


def _case_id(case: BlockingQaicCase) -> str:
    mode = case.qaic_config["blocking_mode"]
    suffix = "-mdp" if case.num_devices > 1 else ""
    return f"{case.model_label}-{mode}{suffix}"


GPT_OSS_RUNTIME_ERROR_XFAIL_CASE_IDS = {
    "gpt_oss-qkv": "GPT-OSS blocked QKV runtime shape mismatch with attention sinks",
    "gpt_oss-hq-mdp": "GPT-OSS blocked HQ runtime shape mismatch with attention sinks",
    "gpt_oss-hkv-mdp": "GPT-OSS blocked HKV runtime shape mismatch with attention sinks",
    "gpt_oss-hqkv-mdp": "GPT-OSS blocked HQKV runtime shape mismatch with attention sinks",
    "gpt_oss-bhqkv-mdp": "GPT-OSS blocked BHQKV runtime shape mismatch with attention sinks",
}


def _with_marks(case: BlockingQaicCase):
    marks = []
    if case.num_devices > 1:
        marks.append(pytest.mark.dynamo_multi_device)
    if xfail_reason := GPT_OSS_RUNTIME_ERROR_XFAIL_CASE_IDS.get(_case_id(case)):
        marks.append(pytest.mark.xfail(reason=xfail_reason))
    if case.xfail_reason:
        marks.append(pytest.mark.xfail(reason=case.xfail_reason))
    return pytest.param(case, marks=marks, id=_case_id(case))


@dataclass(frozen=True)
class BlockingModeSpec:
    mode: str
    kwargs: dict
    num_devices: int = 1
    batch_size: int = BATCH_SIZE


@dataclass(frozen=True)
class BlockingModelSpec:
    label: str
    config_factory: Callable[[int], object]


def _mode(mode: str, *, num_devices: int = 1, batch_size: int = BATCH_SIZE, **kwargs) -> BlockingModeSpec:
    return BlockingModeSpec(mode, kwargs, num_devices=num_devices, batch_size=batch_size)


STANDARD_BLOCKING_MODES = (
    _mode("kv", num_kv_blocks=NUM_KV_BLOCKS),
    _mode("h", head_block_size=HEAD_BLOCK_SIZE, num_devices=4),
    _mode("q", num_q_blocks=NUM_Q_BLOCKS),
    _mode("qkv", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS),
    _mode("hq", head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS, num_devices=4),
    _mode("hkv", head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS, num_devices=4),
    _mode(
        "hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_q_blocks=NUM_Q_BLOCKS,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_devices=4,
    ),
    _mode(
        "bhqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_q_blocks=NUM_Q_BLOCKS,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_batch_blocks=NUM_BATCH_BLOCKS,
        num_devices=4,
        batch_size=2,
    ),
    _mode("kv_headpar", num_kv_blocks=NUM_KV_BLOCKS, headpar_split=HEADPAR_SPLIT, num_devices=4),
)

MLA_BLOCKING_MODES = (
    _mode(
        "kv",
        num_kv_blocks=NUM_KV_BLOCKS,
        mla_absorption={"absorption": False, "online": False, "cache_compressed": True},
    ),
    _mode(
        "h",
        head_block_size=HEAD_BLOCK_SIZE,
        mla_absorption={"absorption": True, "online": False, "cache_compressed": True},
        num_devices=4,
    ),
)

QWEN3_VL_MOE_SPECIAL_MODES = (
    _mode("prefill_q", num_q_blocks=NUM_Q_BLOCKS),
    _mode("prefill_kv", num_kv_blocks=NUM_KV_BLOCKS, headpar_split=HEADPAR_SPLIT, num_devices=4),
    _mode(
        "prefill_qkv",
        num_q_blocks=NUM_Q_BLOCKS,
        num_kv_blocks=NUM_KV_BLOCKS,
        headpar_split=HEADPAR_SPLIT,
        num_devices=4,
    ),
    _mode("prefill_online", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS, n_rep_chunk=1),
)

STANDARD_MODEL_SPECS = (
    BlockingModelSpec("llama", _make_tiny_llama_config),
    BlockingModelSpec("qwen3_moe", _make_tiny_qwen3_moe_config),
    BlockingModelSpec("gpt_oss", _make_tiny_gpt_oss_config),
)

MLA_MODEL_SPECS = (BlockingModelSpec("kimi_k25_language", _make_kimi_k25_language_config),)

QWEN3_VL_MOE_MODEL_SPEC = BlockingModelSpec("qwen3_vl_moe_text", _make_tiny_qwen3_vl_moe_config)


def _expand_cases(
    model_specs: tuple[BlockingModelSpec, ...],
    mode_specs: tuple[BlockingModeSpec, ...],
    *,
    prompt_len: int = PROMPT_LEN_BLOCKING,
    ctx_len: int = CTX_LEN_BLOCKING,
) -> list:
    cases = []
    for model_spec in model_specs:
        for mode_spec in mode_specs:
            cases.append(
                _with_marks(
                    BlockingQaicCase(
                        model_spec.label,
                        model_spec.config_factory,
                        _qaic_config(mode_spec.mode, **mode_spec.kwargs),
                        num_devices=mode_spec.num_devices,
                        batch_size=mode_spec.batch_size,
                        prompt_len=prompt_len,
                        ctx_len=ctx_len,
                    )
                )
            )
    return cases


BLOCKING_QAIC_CASES = [*_expand_cases(STANDARD_MODEL_SPECS, STANDARD_BLOCKING_MODES)]

CB_BLOCKING_QAIC_CASES = [*_expand_cases(STANDARD_MODEL_SPECS, STANDARD_BLOCKING_MODES)]


def _load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _save_tiny_checkpoint(case: BlockingQaicCase, tmp_path: Path):
    tokenizer = _load_tokenizer()
    config = case.config_factory(max(len(tokenizer), VOCAB_SIZE_FLOOR))
    config.dtype = DTYPE
    config.torch_dtype = DTYPE

    model_hf = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation="eager", torch_dtype=DTYPE
    ).eval()
    checkpoint_dir = tmp_path / f"{_case_id(case)}_checkpoint"
    model_hf.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir, model_hf, tokenizer


def _build_qeff_from_checkpoint(checkpoint_dir: Path, *, continuous_batching: bool):
    return QEFFAutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        continuous_batching=continuous_batching,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    )


def _compile_and_check_blocking(qeff_model, case: BlockingQaicCase, tmp_export_dir: Path, *, continuous_batching: bool):
    compile_dir = tmp_export_dir / f"{_case_id(case)}_{'cb' if continuous_batching else 'decode'}"
    compile_kwargs = {}
    if continuous_batching:
        compile_kwargs["full_batch_size"] = FULL_BATCH_SIZE
    qpc_path = qeff_model.compile(
        compile_dir=str(compile_dir),
        prefill_seq_len=case.prompt_len,
        ctx_len=case.ctx_len,
        num_cores=16,
        num_devices=case.num_devices,
        batch_size=case.batch_size,
        qaic_config=copy.deepcopy(case.qaic_config),
        user_tiled=True,
        use_onnx_subfunctions=True,
        dynamo=DYNAMO,
        **compile_kwargs,
    )
    assert compile_dir.is_dir()
    assert Path(qpc_path).is_dir(), f"Expected QPC directory at {qpc_path}"
    assert qeff_model.qpc_path == Path(qpc_path)
    expected_mode = BlockingMode(case.qaic_config["blocking_mode"])
    attached_configs = [
        module.attn_blocking_config for module in qeff_model.model.modules() if hasattr(module, "attn_blocking_config")
    ]
    assert attached_configs, "Expected blocking config on at least one attention module"
    assert all(config.mode == expected_mode for config in attached_configs)
    onnx_path = exported_onnx_path(qeff_model.onnx_path)
    assert_blocked_kv_ops_for_mode(
        onnx_path,
        qeff_model,
        case.qaic_config["blocking_mode"],
        continuous_batching=continuous_batching,
    )
    return compile_dir, Path(qpc_path), onnx_path


def _compile_cache_key(case: BlockingQaicCase, *, continuous_batching: bool) -> tuple:
    return (
        _case_id(case),
        continuous_batching,
        case.prompt_len,
        case.ctx_len,
        case.batch_size,
        case.num_devices,
        json.dumps(case.qaic_config, sort_keys=True),
    )


def _get_or_compile_blocking_artifact(
    case: BlockingQaicCase,
    tmp_path_factory,
    *,
    continuous_batching: bool,
) -> BlockingCompiledArtifact:
    cache_key = _compile_cache_key(case, continuous_batching=continuous_batching)
    cached = _BLOCKING_QPC_CACHE.get(cache_key)
    if cached is not None and cached.compile_dir.is_dir() and cached.qpc_path.is_dir() and cached.onnx_path.is_file():
        return cached

    artifact_dir = tmp_path_factory.mktemp(
        f"dynamo_blocking_{_case_id(case)}_{'cb' if continuous_batching else 'decode'}",
        numbered=True,
    )
    checkpoint_dir, model_hf, tokenizer = _save_tiny_checkpoint(case, artifact_dir)
    qeff_model = _build_qeff_from_checkpoint(checkpoint_dir, continuous_batching=continuous_batching)
    export_dir = artifact_dir / "qeff_dynamo_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    compile_dir, qpc_path, onnx_path = _compile_and_check_blocking(
        qeff_model,
        case,
        export_dir,
        continuous_batching=continuous_batching,
    )

    artifact = BlockingCompiledArtifact(
        qeff_model=qeff_model,
        model_hf=model_hf,
        tokenizer=tokenizer,
        compile_dir=compile_dir,
        qpc_path=qpc_path,
        onnx_path=onnx_path,
    )
    _BLOCKING_QPC_CACHE[cache_key] = artifact
    return artifact


def _get_device_ids(case: BlockingQaicCase):
    if case.num_devices <= 1:
        return get_available_device_id()
    device_groups = get_qaic_mdp_device_groups(devices_per_group=case.num_devices)
    assert device_groups, f"No available QAIC MDP device group with {case.num_devices} devices"
    return device_groups[0]


def _make_prefill_raw_inputs(tokenizer, prompts: list[str], prompt_len: int) -> dict[str, np.ndarray]:
    inputs = tokenizer(prompts, return_tensors="np", padding="max_length", max_length=prompt_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(prompt_len), -1)
    inputs.pop("token_type_ids", None)
    return inputs


def _make_past_key_values(config, tokenizer, batch_size: int, ctx_len: int):
    input_handler = InputHandler(
        batch_size=batch_size,
        tokenizer=tokenizer,
        config=config,
        prompt=[],
        prompt_len=0,
        ctx_len=ctx_len,
        full_batch_size=None,
        dtype=DTYPE,
    )
    return tuple(
        (
            torch.zeros(input_handler._get_layer_cache_shape(layer_idx), dtype=DTYPE),
            torch.zeros(input_handler._get_layer_cache_shape(layer_idx), dtype=DTYPE),
        )
        for layer_idx in range(input_handler.n_layer)
    )


def _make_prefill_torch_inputs(
    case: BlockingQaicCase,
    tokenizer,
    prompts: list[str],
    config,
    *,
    full_batch_size: int | None = None,
    batch_index: int | None = None,
    past_key_values=None,
) -> dict[str, torch.Tensor]:
    raw_inputs = _make_prefill_raw_inputs(tokenizer, prompts, case.prompt_len)
    inputs = {key: torch.from_numpy(value) for key, value in raw_inputs.items()}
    if past_key_values is None:
        past_key_values = _make_past_key_values(config, tokenizer, full_batch_size or len(prompts), case.ctx_len)
    inputs["past_key_values"] = past_key_values
    inputs["use_cache"] = True
    if batch_index is not None:
        inputs["batch_index"] = torch.tensor([[batch_index]], dtype=torch.long)
    return inputs


def _make_unpadded_prefill_torch_inputs(
    tokenizer,
    prompt: str,
    *,
    past_key_values,
    batch_index: int,
) -> dict[str, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.pop("attention_mask", None)
    inputs.pop("token_type_ids", None)
    seq_len = inputs["input_ids"].shape[1]
    inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).view(1, seq_len)
    inputs["past_key_values"] = past_key_values
    inputs["use_cache"] = True
    inputs["batch_index"] = torch.tensor([[batch_index]], dtype=torch.long)
    return inputs


def _get_output_past_key_values(outputs):
    if hasattr(outputs, "past_key_values"):
        return outputs.past_key_values
    return outputs["past_key_values"]


@torch.no_grad()
def _get_qeff_generation_logits_and_tokens(
    case: BlockingQaicCase,
    qeff_model,
    tokenizer,
    prompts: list[str],
    *,
    full_batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gen_len = case.ctx_len - case.prompt_len
    config = qeff_model.model.config

    if full_batch_size is not None:
        past_key_values = _make_past_key_values(config, tokenizer, full_batch_size, case.ctx_len)
        prefill_logits = []
        next_tokens = []
        next_positions = []
        for batch_idx, prompt in enumerate(prompts):
            inputs = _make_unpadded_prefill_torch_inputs(
                tokenizer,
                prompt,
                past_key_values=past_key_values,
                batch_index=batch_idx,
            )
            outputs = qeff_model.model(**inputs)
            past_key_values = _get_output_past_key_values(outputs)
            prefill_logits.append(outputs.logits.detach().float().cpu().numpy())
            next_tokens.append(outputs.logits.argmax(-1).detach().cpu())
            next_positions.append(inputs["position_ids"].max(1, keepdim=True).values + 1)

        logits = [np.concatenate(prefill_logits, axis=0)]
        input_ids = torch.cat(next_tokens, dim=0).to(torch.long)
        position_ids = torch.cat(next_positions, dim=0).to(torch.long)
        batch_index = torch.arange(full_batch_size, dtype=torch.long).view(-1, 1)
        for _ in range(1, gen_len):
            outputs = qeff_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                batch_index=batch_index,
                use_cache=True,
            )
            past_key_values = _get_output_past_key_values(outputs)
            logits.append(outputs.logits.detach().float().cpu().numpy())
            input_ids = outputs.logits.argmax(-1).detach().to(torch.long)
            position_ids = position_ids + 1
        stacked_logits = np.concatenate(logits, axis=1)
        return stacked_logits, stacked_logits.argmax(-1)

    inputs = _make_prefill_torch_inputs(case, tokenizer, prompts, config)
    outputs = qeff_model.model(**inputs)
    past_key_values = _get_output_past_key_values(outputs)
    logits = [outputs.logits.detach().float().cpu().numpy()]
    input_ids = outputs.logits.argmax(-1).detach().to(torch.long)
    position_ids = inputs["position_ids"].max(1, keepdim=True).values + 1
    for _ in range(1, gen_len):
        outputs = qeff_model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = _get_output_past_key_values(outputs)
        logits.append(outputs.logits.detach().float().cpu().numpy())
        input_ids = outputs.logits.argmax(-1).detach().to(torch.long)
        position_ids = position_ids + 1
    stacked_logits = np.concatenate(logits, axis=1)
    return stacked_logits, stacked_logits.argmax(-1)


@torch.no_grad()
def _get_hf_generation_logits(model_hf, tokenizer, prompts: list[str], forced_tokens: np.ndarray) -> np.ndarray:
    logits = []
    for batch_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]
        prompt_logits = []
        for token_idx in range(forced_tokens.shape[1]):
            outputs = model_hf(input_ids=input_ids)
            prompt_logits.append(outputs.logits[:, -1:, :].detach().float().cpu().numpy())
            if token_idx + 1 < forced_tokens.shape[1]:
                next_token = torch.tensor([[forced_tokens[batch_idx, token_idx]]], dtype=torch.long)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        logits.append(np.concatenate(prompt_logits, axis=1))
    return np.concatenate(logits, axis=0)


def _get_qaic_generation_logits(
    case: BlockingQaicCase,
    qpc_path: Path,
    config,
    tokenizer,
    prompts: list[str],
    forced_tokens: np.ndarray,
    *,
    full_batch_size: int | None = None,
) -> np.ndarray:
    session = QAICInferenceSession(str(qpc_path), _get_device_ids(case))
    session.skip_buffers([name for name in session.input_names + session.output_names if is_retained_state_name(name)])
    try:
        gen_len = forced_tokens.shape[1]
        if full_batch_size is not None:
            assert len(prompts) == full_batch_size
            prefill_logits = []
            position_ids = []
            for batch_idx, prompt in enumerate(prompts):
                inputs = _make_prefill_raw_inputs(tokenizer, [prompt], case.prompt_len)
                inputs["batch_index"] = np.array(batch_idx, dtype=np.int64).reshape(1, 1)
                session.set_buffers({"logits": np.zeros((1, 1, config.vocab_size), dtype=np.float32)})
                outputs = session.run(inputs)
                prefill_logits.append(outputs["logits"])
                position_ids.append(inputs["position_ids"].max(1, keepdims=True) + 1)

            logits = [np.concatenate(prefill_logits, axis=0)]
            decode_position_ids = np.concatenate(position_ids, axis=0).astype(np.int64)
            batch_index = np.arange(full_batch_size, dtype=np.int64).reshape(-1, 1)
            for token_idx in range(1, gen_len):
                decode_inputs = {
                    "input_ids": forced_tokens[:, token_idx - 1].reshape(full_batch_size, 1).astype(np.int64),
                    "position_ids": decode_position_ids,
                    "batch_index": batch_index,
                }
                session.set_buffers({"logits": np.zeros((full_batch_size, 1, config.vocab_size), dtype=np.float32)})
                outputs = session.run(decode_inputs)
                logits.append(outputs["logits"])
                decode_position_ids = decode_position_ids + 1
            return np.concatenate(logits, axis=1)

        inputs = _make_prefill_raw_inputs(tokenizer, prompts, case.prompt_len)
        session.set_buffers({"logits": np.zeros((case.batch_size, 1, config.vocab_size), dtype=np.float32)})
        outputs = session.run(inputs)
        logits = [outputs["logits"]]
        position_ids = inputs["position_ids"].max(1, keepdims=True) + 1
        for token_idx in range(1, gen_len):
            decode_inputs = {
                "input_ids": forced_tokens[:, token_idx - 1].reshape(case.batch_size, 1).astype(np.int64),
                "position_ids": position_ids.astype(np.int64),
            }
            session.set_buffers({"logits": np.zeros((case.batch_size, 1, config.vocab_size), dtype=np.float32)})
            outputs = session.run(decode_inputs)
            logits.append(outputs["logits"])
            position_ids = position_ids + 1
        return np.concatenate(logits, axis=1)
    finally:
        del session


def _assert_logits_close(label: str, expected: np.ndarray, actual: np.ndarray, *, atol: float) -> None:
    assert expected.shape == actual.shape
    diff = np.abs(expected - actual)
    max_diff = float(diff.max())
    if max_diff >= atol:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        assert False, (
            f"{label} logits diverged: shape={expected.shape}, max_abs_diff={max_diff}, "
            f"mean_abs_diff={float(diff.mean())}, max_diff_index={max_idx}, atol={atol}"
        )


def _assert_generation_logits_parity(
    case: BlockingQaicCase,
    qeff_model,
    qpc_path: Path,
    model_hf,
    tokenizer,
    prompts,
    *,
    full_batch_size: int | None = None,
):
    qeff_logits, forced_tokens = _get_qeff_generation_logits_and_tokens(
        case,
        qeff_model,
        tokenizer,
        prompts,
        full_batch_size=full_batch_size,
    )
    hf_logits = _get_hf_generation_logits(model_hf, tokenizer, prompts, forced_tokens)
    _assert_logits_close("HF vs QEff PyTorch", hf_logits, qeff_logits, atol=1e-4)

    qaic_logits = _get_qaic_generation_logits(
        case,
        qpc_path,
        qeff_model.model.config,
        tokenizer,
        prompts,
        forced_tokens,
        full_batch_size=full_batch_size,
    )
    _assert_logits_close("QAIC vs QEff PyTorch", qeff_logits, qaic_logits, atol=5e-2)


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize("case", BLOCKING_QAIC_CASES)
def test_dynamo_tiny_blocking_compile_and_generate(case, tmp_path_factory):
    artifact = _get_or_compile_blocking_artifact(case, tmp_path_factory, continuous_batching=False)

    _assert_generation_logits_parity(
        case,
        artifact.qeff_model,
        artifact.qpc_path,
        artifact.model_hf,
        artifact.tokenizer,
        CB_PROMPTS[: case.batch_size],
    )


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize("case", CB_BLOCKING_QAIC_CASES)
def test_dynamo_tiny_cb_blocking_compile_and_generate(case, tmp_path_factory):
    artifact = _get_or_compile_blocking_artifact(case, tmp_path_factory, continuous_batching=True)

    _assert_generation_logits_parity(
        case,
        artifact.qeff_model,
        artifact.qpc_path,
        artifact.model_hf,
        artifact.tokenizer,
        CB_PROMPTS,
        full_batch_size=FULL_BATCH_SIZE,
    )
