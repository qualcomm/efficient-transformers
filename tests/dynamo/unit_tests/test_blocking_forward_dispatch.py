# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CPU-only Dynamo checks that attention blocking dispatches to the requested forward.

These tests intentionally avoid ONNX export/compile. They validate that an
explicit blocking config reaches the runtime attention path and selects the
expected implementation.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from QEfficient.blocking import attention_blocking
from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.blocking.blocking_configurator import build_transformer_blocking_config_for_transform
from QEfficient.transformers.models.pytorch_transforms import (
    BlockingAttentionTransform,
    CustomOpsTransform,
    KVCacheExternalModuleMapperTransform,
    KVCacheTransform,
)

VOCAB_SIZE = 512
SEQ_LEN = 8
CTX_LEN = 128
BATCH_SIZE = 1
HEAD_BLOCK_SIZE = 2
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
NUM_BATCH_BLOCKS = 2
HEADPAR_SPLIT = 4

MODEL_KWARGS = {"attn_implementation": "eager"}


@dataclass(frozen=True)
class DispatchCase:
    model_label: str
    make_model: Callable[[], nn.Module]
    qaic_config: dict
    expected_mode: BlockingMode
    seq_len: int = SEQ_LEN
    ctx_len: int = CTX_LEN
    batch_size: int = BATCH_SIZE
    prefill_only: bool = False
    skip_reason: str | None = None


def _make_tiny_llama():
    import transformers

    cfg = transformers.LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        pad_token_id=0,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def _make_tiny_glm4_moe():
    import transformers

    cfg = transformers.AutoConfig.for_model(
        "glm4_moe",
        max_position_embeddings=CTX_LEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=32,
        vocab_size=VOCAB_SIZE,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        n_group=1,
        topk_group=1,
        head_dim=32,
        pad_token_id=0,
    )
    return transformers.AutoModelForCausalLM.from_config(cfg, **MODEL_KWARGS).eval()


def _make_tiny_gpt_oss():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    cfg = GptOssConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=128,
        head_dim=32,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=32,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default"},
    )
    return GptOssForCausalLM(cfg).eval()


def _make_tiny_qwen3_vl_moe_text():
    from transformers import AutoConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextModel

    text_config = AutoConfig.for_model(
        "qwen3_vl_moe",
        text_config={
            "max_position_embeddings": CTX_LEN,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 32,
            "vocab_size": VOCAB_SIZE,
            "num_key_value_heads": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "head_dim": 32,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "rope_scaling": {"rope_type": "default", "mrope_section": [11, 11, 10]},
            "pad_token_id": 0,
        },
    ).text_config
    return Qwen3VLMoeTextModel._from_config(text_config, **MODEL_KWARGS).eval()


def _make_kimi_k25_language_model():
    from tests.utils.load_kimi_utils import get_kimi_k25_test_config, load_kimi_k25_model_from_config

    model_id = "moonshotai/Kimi-K2.5"
    config_path = Path(__file__).parents[2] / "configs" / "image_text_model_configs.json"
    model_configs = json.loads(config_path.read_text())["image_text_models"]
    model_config_dict = {model["model_name"]: model for model in model_configs}

    config = get_kimi_k25_test_config(model_id, model_config_dict)
    model_hf, _, _ = load_kimi_k25_model_from_config(config)
    return model_hf.language_model.eval()


def _qeff_attention_modules(model: nn.Module) -> list[nn.Module]:
    supported = {
        qeff_cls for qeff_cls in KVCacheTransform._module_mapping.values() if qeff_cls.__name__.endswith("Attention")
    }
    return [
        module for module in model.modules() if type(module) in supported or hasattr(module, "attn_blocking_config")
    ]


def _config_num_layers(config) -> int:
    return int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 1)))


def _config_num_heads(config) -> int:
    return int(getattr(config, "num_attention_heads", getattr(config, "n_head", 1)))


def _config_kv_heads(config) -> int:
    return int(getattr(config, "num_key_value_heads", _config_num_heads(config)))


def _config_hidden_size(config) -> int:
    return int(getattr(config, "hidden_size", getattr(config, "n_embd", 128)))


def _config_head_dim(config) -> int:
    return int(getattr(config, "head_dim", _config_hidden_size(config) // _config_num_heads(config)))


def _make_past_key_values(config, batch_size: int, ctx_len: int):
    n_layers = _config_num_layers(config)
    n_kv_heads = _config_kv_heads(config)
    head_dim = _config_head_dim(config)
    return tuple(
        (
            torch.zeros(batch_size, n_kv_heads, ctx_len, head_dim, dtype=torch.float32),
            torch.zeros(batch_size, n_kv_heads, ctx_len, head_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    )


def _apply_qeff_kv_transforms(model: nn.Module, model_label: str):
    model, _ = CustomOpsTransform.apply(model)
    model, external_kv_transformed = KVCacheExternalModuleMapperTransform.apply(model)
    model, kv_transformed = KVCacheTransform.apply(model)
    assert kv_transformed or external_kv_transformed, f"[{model_label}] KV cache transform did not transform the model"
    return model


def _clone_tensor_inputs(inputs):
    def clone_value(value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(clone_value(item) for item in value)
        if isinstance(value, list):
            return [clone_value(item) for item in value]
        return value

    return {key: clone_value(value) for key, value in inputs.items()}


def _make_inputs(model: nn.Module, case: DispatchCase, *, include_batch_index: bool = True):
    config = model.config
    input_ids = torch.randint(0, int(getattr(config, "vocab_size", VOCAB_SIZE)), (case.batch_size, case.seq_len))
    if case.model_label.startswith("qwen3_vl_moe"):
        position_ids = torch.arange(case.seq_len, dtype=torch.long).view(1, 1, -1).expand(4, case.batch_size, -1)
    else:
        position_ids = torch.arange(case.seq_len, dtype=torch.long).unsqueeze(0).expand(case.batch_size, -1)
    inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "use_cache": True,
    }
    if include_batch_index:
        inputs["batch_index"] = torch.arange(case.batch_size, dtype=torch.long)

    if "kimi" in case.model_label:
        inputs["compressed_kvs"] = model.get_dummy_pkv_cache(config, case.batch_size, case.ctx_len)
    else:
        inputs["past_key_values"] = _make_past_key_values(config, case.batch_size, case.ctx_len)
    return inputs


def _assert_valid_output(output, model: nn.Module, case: DispatchCase):
    if hasattr(output, "logits"):
        logits = output.logits
        assert logits.shape[0] == case.batch_size
        assert logits.shape[1] in {1, case.seq_len}
        assert logits.shape[-1] == model.config.vocab_size
        return

    hidden_states = getattr(output, "last_hidden_state", None)
    if hidden_states is None and isinstance(output, tuple):
        hidden_states = output[0]
    assert hidden_states is not None
    assert hidden_states.shape[:2] == (case.batch_size, case.seq_len)


def _apply_blocking(model: nn.Module, case: DispatchCase):
    model = _apply_qeff_kv_transforms(model, case.model_label)
    if (mla_absorption := case.qaic_config.get("mla_absorption")) is not None:
        model.mla_absorption = mla_absorption

    blocking_config = build_transformer_blocking_config_for_transform(
        model.config,
        ctx_len=case.ctx_len,
        seq_len=case.seq_len,
        bs=case.batch_size,
        num_devices=1,
        qaic_config=copy.deepcopy(case.qaic_config),
        aic_num_cores=4,
        prefill_only=case.prefill_only,
    )
    assert blocking_config is not None
    for optional_param in ("num_kv_blocks", "num_q_blocks", "head_block_size", "num_batch_blocks", "headpar_split"):
        if case.qaic_config.get(optional_param) is not None:
            setattr(blocking_config, optional_param, case.qaic_config[optional_param])

    expected_config_mode = BlockingMode.resolve(case.qaic_config["blocking_mode"])
    if case.expected_mode not in {BlockingMode.KV_MLA, BlockingMode.H_MLA}:
        expected_config_mode = case.expected_mode
    assert blocking_config.mode == expected_config_mode

    model, blocking_transformed = BlockingAttentionTransform.apply(model, blocking_config)
    assert blocking_transformed, f"[{case.model_label}] BlockingAttentionTransform did not transform the model"

    attn_modules = _qeff_attention_modules(model)
    assert attn_modules, f"[{case.model_label}] no QEff attention modules found"
    for module in attn_modules:
        assert module.attn_blocking_config is blocking_config
        assert module.attn_blocking_config.mode == expected_config_mode

    return model


def _case_id(case: DispatchCase) -> str:
    return f"{case.model_label}-{case.qaic_config['blocking_mode']}"


def _qaic_config(mode: str, **kwargs) -> dict:
    return {"blocking_mode": mode, **kwargs}


DISPATCH_CASES = [
    DispatchCase("llama", _make_tiny_llama, _qaic_config("kv", num_kv_blocks=NUM_KV_BLOCKS), BlockingMode.KV),
    DispatchCase("llama", _make_tiny_llama, _qaic_config("h", head_block_size=HEAD_BLOCK_SIZE), BlockingMode.H),
    DispatchCase("llama", _make_tiny_llama, _qaic_config("q", num_q_blocks=NUM_Q_BLOCKS), BlockingMode.Q),
    DispatchCase(
        "llama",
        _make_tiny_llama,
        _qaic_config("qkv", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS),
        BlockingMode.QKV,
    ),
    DispatchCase(
        "llama",
        _make_tiny_llama,
        _qaic_config("hq", head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS),
        BlockingMode.HQ,
        skip_reason="HQ uses shared HQKV forward that currently expects num_kv_blocks; pending implementation fix",
    ),
    DispatchCase(
        "llama",
        _make_tiny_llama,
        _qaic_config("hkv", head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS),
        BlockingMode.HKV,
    ),
    DispatchCase(
        "llama",
        _make_tiny_llama,
        _qaic_config("hqkv", head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS),
        BlockingMode.HQKV,
    ),
    DispatchCase(
        "llama",
        _make_tiny_llama,
        _qaic_config(
            "bhqkv",
            head_block_size=HEAD_BLOCK_SIZE,
            num_q_blocks=NUM_Q_BLOCKS,
            num_kv_blocks=NUM_KV_BLOCKS,
            num_batch_blocks=NUM_BATCH_BLOCKS,
        ),
        BlockingMode.BHQKV,
        batch_size=2,
    ),
    DispatchCase(
        "llama",
        _make_tiny_llama,
        _qaic_config("kv_headpar", num_kv_blocks=NUM_KV_BLOCKS, headpar_split=HEADPAR_SPLIT),
        BlockingMode.KV_HEADPAR,
    ),
    DispatchCase("gpt_oss", _make_tiny_gpt_oss, _qaic_config("kv", num_kv_blocks=NUM_KV_BLOCKS), BlockingMode.KV),
    DispatchCase("gpt_oss", _make_tiny_gpt_oss, _qaic_config("q", num_q_blocks=NUM_Q_BLOCKS), BlockingMode.Q),
    DispatchCase(
        "gpt_oss",
        _make_tiny_gpt_oss,
        _qaic_config("qkv", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS),
        BlockingMode.QKV,
        skip_reason="GPT-OSS QKV blocking sinks are full-sequence while QKV uses block-local softmax state",
    ),
    DispatchCase("glm4_moe", _make_tiny_glm4_moe, _qaic_config("kv", num_kv_blocks=NUM_KV_BLOCKS), BlockingMode.KV),
    DispatchCase("glm4_moe", _make_tiny_glm4_moe, _qaic_config("q", num_q_blocks=NUM_Q_BLOCKS), BlockingMode.Q),
    DispatchCase(
        "glm4_moe",
        _make_tiny_glm4_moe,
        _qaic_config("qkv", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS),
        BlockingMode.QKV,
    ),
    DispatchCase(
        "glm4_moe",
        _make_tiny_glm4_moe,
        _qaic_config("hqkv", head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS),
        BlockingMode.HQKV,
    ),
    DispatchCase(
        "qwen3_vl_moe_text",
        _make_tiny_qwen3_vl_moe_text,
        _qaic_config("prefill_q", num_q_blocks=NUM_Q_BLOCKS),
        BlockingMode.PREFILL_Q,
        prefill_only=True,
    ),
    DispatchCase(
        "qwen3_vl_moe_text",
        _make_tiny_qwen3_vl_moe_text,
        _qaic_config("prefill_kv", num_kv_blocks=NUM_KV_BLOCKS, headpar_split=HEADPAR_SPLIT),
        BlockingMode.PREFILL_KV,
        prefill_only=True,
    ),
    DispatchCase(
        "qwen3_vl_moe_text",
        _make_tiny_qwen3_vl_moe_text,
        _qaic_config(
            "prefill_qkv", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS, headpar_split=HEADPAR_SPLIT
        ),
        BlockingMode.PREFILL_QKV,
        prefill_only=True,
    ),
    DispatchCase(
        "qwen3_vl_moe_text",
        _make_tiny_qwen3_vl_moe_text,
        _qaic_config("prefill_online", num_q_blocks=NUM_Q_BLOCKS, num_kv_blocks=NUM_KV_BLOCKS, n_rep_chunk=1),
        BlockingMode.PREFILL_ONLINE,
        prefill_only=True,
    ),
    DispatchCase(
        "kimi_k25_language",
        _make_kimi_k25_language_model,
        _qaic_config(
            "kv",
            num_kv_blocks=NUM_KV_BLOCKS,
            mla_absorption={"absorption": False, "online": False, "cache_compressed": True},
        ),
        BlockingMode.KV_MLA,
    ),
    DispatchCase(
        "kimi_k25_language",
        _make_kimi_k25_language_model,
        _qaic_config(
            "h",
            head_block_size=HEAD_BLOCK_SIZE,
            mla_absorption={"absorption": True, "online": False, "cache_compressed": True},
        ),
        BlockingMode.H_MLA,
    ),
]


@pytest.mark.dynamo
@pytest.mark.parametrize("case", DISPATCH_CASES, ids=_case_id)
def test_dynamo_blocking_forward_dispatch(case: DispatchCase):
    if case.skip_reason:
        pytest.skip(case.skip_reason)

    torch.manual_seed(42)
    model = _apply_blocking(case.make_model(), case)
    real_strategy = attention_blocking._STRATEGIES[case.expected_mode]
    strategy_spy = Mock(wraps=real_strategy)

    with torch.no_grad(), patch.dict(attention_blocking._STRATEGIES, {case.expected_mode: strategy_spy}):
        output = model(**_make_inputs(model, case))

    assert strategy_spy.called, f"[{_case_id(case)}] expected {case.expected_mode.value} blocking forward to run"
    _assert_valid_output(output, model, case)


_LLAMA_CPU_PARITY_MODES = {
    BlockingMode.Q,
    BlockingMode.H,
    BlockingMode.KV,
    BlockingMode.QKV,
    BlockingMode.HQKV,
    BlockingMode.BHQKV,
    BlockingMode.KV_HEADPAR,
}
_LLAMA_CPU_PARITY_CASES = [
    case
    for case in DISPATCH_CASES
    if case.model_label == "llama" and case.expected_mode in _LLAMA_CPU_PARITY_MODES and case.skip_reason is None
]


@pytest.mark.dynamo
@pytest.mark.parametrize("case", _LLAMA_CPU_PARITY_CASES, ids=_case_id)
def test_dynamo_llama_blocking_cpu_logits_parity(case: DispatchCase):
    torch.manual_seed(7)
    base = case.make_model()
    unblocked = _apply_qeff_kv_transforms(copy.deepcopy(base), case.model_label).eval()
    blocked = _apply_blocking(copy.deepcopy(base), case).eval()
    inputs = _make_inputs(blocked, case, include_batch_index=False)

    with torch.no_grad():
        unblocked_output = unblocked(**_clone_tensor_inputs(inputs))
        blocked_output = blocked(**_clone_tensor_inputs(inputs))

    torch.testing.assert_close(
        blocked_output.logits.float(),
        unblocked_output.logits.float(),
        rtol=1e-3,
        atol=1e-3,
        msg=f"[{_case_id(case)}] blocked and unblocked QEff CPU logits diverged",
    )


@pytest.mark.dynamo
@pytest.mark.parametrize(
    ("attention_cfg", "expected_mode"),
    [
        (
            {"head_blocking_enabled": False, "head_block_size": 1, "num_q_blocks": 2, "num_kv_blocks": 1},
            BlockingMode.Q,
        ),
        (
            {"head_blocking_enabled": False, "head_block_size": 1, "num_q_blocks": 1, "num_kv_blocks": 2},
            BlockingMode.KV,
        ),
        (
            {"head_blocking_enabled": True, "head_block_size": 2, "num_q_blocks": 2, "num_kv_blocks": 2},
            BlockingMode.HQKV,
        ),
    ],
    ids=["hqkv_to_q", "hqkv_to_kv", "hqkv_stays_hqkv"],
)
def test_dynamo_blocking_auto_hqkv_reduction_is_explicit(attention_cfg, expected_mode):
    with patch("QEfficient.blocking.blocking_configurator.attention_configurator", return_value=attention_cfg):
        blocking_config = build_transformer_blocking_config_for_transform(
            _make_tiny_llama().config,
            ctx_len=CTX_LEN,
            seq_len=SEQ_LEN,
            bs=BATCH_SIZE,
            num_devices=1,
            qaic_config={"blocking_mode": "hqkv"},
            aic_num_cores=4,
        )

    assert blocking_config.mode == expected_mode
