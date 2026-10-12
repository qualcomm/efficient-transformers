# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import copy
from collections import Counter

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.transformers.models.pytorch_transforms import (
    ExternalOptimizedMoEMapperTransform,
    KVCacheTransform,
    OptimizedMoEExpertParallelWeightsTransform,
    OptimizedMoEExportConfigTransform,
    OptimizedMoEMapperTransform,
    OptimizedMoETransform,
    OptimizedMoEWeightsTransform,
    PrefillOnlyChunkedTransform,
    PrefillOnlyTransform,
)
from QEfficient.transformers.moe import MoEFlavour, QEffMoEBlockMixin, build_canonical_expert_weights

MODEL_KWARGS = {"attn_implementation": "eager"}

GLM4_MOE_CFG = dict(
    max_position_embeddings=1024,
    num_hidden_layers=2,
    num_attention_heads=4,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    vocab_size=127,
    num_key_value_heads=2,
    n_routed_experts=4,
    num_experts_per_tok=2,
    first_k_dense_replace=0,
    n_group=1,
    topk_group=1,
    head_dim=16,
)


MOE_BLOCK_SEQ_LEN = 8
MOE_BLOCK_HIDDEN_SIZE = 32
MOE_BLOCK_INTERMEDIATE_SIZE = 64
MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE = 16
MOE_BLOCK_NUM_EXPERTS = 4
MOE_BLOCK_TOP_K = 2

MOE_BLOCK_BASE_CFG = dict(
    max_position_embeddings=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    hidden_size=MOE_BLOCK_HIDDEN_SIZE,
    intermediate_size=MOE_BLOCK_INTERMEDIATE_SIZE,
    vocab_size=127,
    num_key_value_heads=2,
)


def _first_module_by_class_name(model: nn.Module, class_name: str) -> nn.Module:
    return next(module for module in model.modules() if module.__class__.__name__ == class_name)


def _first_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _match_expected_shape(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    if actual.shape == expected.shape:
        return actual
    assert actual.numel() == expected.numel()
    return actual.view_as(expected)


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def test_moe_block_mixin_allows_fx_attr_proxy_subclasses():
    class FxProxyTargetMoE(QEffMoEBlockMixin, nn.Module):
        supported_moe_flavours = (MoEFlavour.SIMPLE_LOOP,)

        def route(self, x: torch.Tensor):
            raise NotImplementedError

        def transform_weights(self):
            raise NotImplementedError

    proxy_cls = type(FxProxyTargetMoE.__name__, (FxProxyTargetMoE,), {"__module__": "abc"})

    assert proxy_cls.supported_moe_flavours == FxProxyTargetMoE.supported_moe_flavours
    with pytest.raises(TypeError, match="must explicitly define supported_moe_flavours"):
        type("RegularDerivedMoE", (FxProxyTargetMoE,), {})


def test_build_canonical_expert_weights_clone_breaks_fused_gate_up_aliasing():
    gate_up = torch.randn(MOE_BLOCK_NUM_EXPERTS, 2 * MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE, MOE_BLOCK_HIDDEN_SIZE)
    down = torch.randn(MOE_BLOCK_NUM_EXPERTS, MOE_BLOCK_HIDDEN_SIZE, MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE)

    weights = build_canonical_expert_weights(
        gate_up=gate_up,
        down=down,
        fused=True,
        fused_split_dim=1,
        transpose_gate_up=True,
        transpose_down=True,
        clone=True,
    )

    assert _storage_ptr(weights.gate) != _storage_ptr(weights.up)


def _make_tiny_causal_lm(model_type: str, **config_kwargs):
    config = AutoConfig.for_model(model_type, **{**MOE_BLOCK_BASE_CFG, **config_kwargs})
    return AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)


def _apply_prefill_compile_transforms(
    qeff,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    num_cores: int = 2,
    enable_chunking: bool = True,
    qaic_config=None,
):
    qeff.transform(
        ctx_len=ctx_len,
        seq_len=prefill_seq_len,
        bs=1,
        qaic_config=qaic_config,
        prefill_only=True,
        enable_chunking=enable_chunking,
        num_cores=num_cores,
        prefill_seq_len=prefill_seq_len,
    )


def _apply_decode_compile_transforms(qeff, *, seq_len: int, ctx_len: int, num_cores: int = 2, qaic_config=None):
    qeff.transform(
        ctx_len=ctx_len,
        seq_len=seq_len,
        bs=1,
        qaic_config=qaic_config,
        prefill_only=False,
        enable_chunking=False,
        num_cores=num_cores,
    )


def _make_tiny_qwen3_5_moe_text_model():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    text_config = AutoConfig.for_model(
        "qwen3_5_moe",
        text_config={
            **MOE_BLOCK_BASE_CFG,
            "moe_intermediate_size": MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE,
            "num_experts": MOE_BLOCK_NUM_EXPERTS,
            "num_experts_per_tok": MOE_BLOCK_TOP_K,
            "pad_token_id": 0,
            "shared_expert_intermediate_size": MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "partial_rotary_factor": 0.25,
            "layer_types": ["full_attention"],
        },
    ).text_config
    return Qwen3_5MoeTextModel._from_config(text_config, **MODEL_KWARGS)


def _make_tiny_qwen3_vl_moe_text_model():
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextModel

    text_config = AutoConfig.for_model(
        "qwen3_vl_moe",
        text_config={
            **MOE_BLOCK_BASE_CFG,
            "moe_intermediate_size": MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE,
            "num_local_experts": MOE_BLOCK_NUM_EXPERTS,
            "num_experts_per_tok": MOE_BLOCK_TOP_K,
            "pad_token_id": 0,
            "head_dim": 16,
        },
    ).text_config
    return Qwen3VLMoeTextModel._from_config(text_config, **MODEL_KWARGS)


def _make_tiny_llama4_text_model():
    from transformers.models.llama4.modeling_llama4 import Llama4TextModel

    text_config = AutoConfig.for_model(
        "llama4",
        text_config={
            **MOE_BLOCK_BASE_CFG,
            "intermediate_size_mlp": MOE_BLOCK_INTERMEDIATE_SIZE,
            "num_local_experts": MOE_BLOCK_NUM_EXPERTS,
            "num_experts_per_tok": MOE_BLOCK_TOP_K,
            "head_dim": 16,
            "pad_token_id": 0,
            "attn_scale": 0.1,
            "attn_temperature_tuning": True,
            "floor_scale": 8192,
            "attention_chunk_size": 8192,
            "interleave_moe_layer_step": 1,
            "moe_layers": [0],
            "no_rope_layers": [0],
            "layer_types": ["full_attention"],
            "use_qk_norm": False,
        },
    ).text_config
    return Llama4TextModel._from_config(text_config, **MODEL_KWARGS)


MOE_BLOCK_PARITY_CASES = (
    pytest.param(
        "qwen3_moe",
        lambda: _make_tiny_causal_lm(
            "qwen3_moe",
            moe_intermediate_size=MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE,
            num_experts=MOE_BLOCK_NUM_EXPERTS,
            num_experts_per_tok=MOE_BLOCK_TOP_K,
        ),
        "Qwen3MoeSparseMoeBlock",
        ("decode_bmm", "simple_loop", "expert_parallel"),
        {},
        id="qwen3_moe",
    ),
    pytest.param(
        "qwen3_vl_moe",
        _make_tiny_qwen3_vl_moe_text_model,
        "Qwen3VLMoeTextSparseMoeBlock",
        ("decode_bmm", "simple_loop", "expert_parallel"),
        {},
        id="qwen3_vl_moe",
    ),
    pytest.param(
        "qwen3_5_moe",
        _make_tiny_qwen3_5_moe_text_model,
        "Qwen3_5MoeSparseMoeBlock",
        ("decode_bmm", "simple_loop", "expert_parallel"),
        {},
        id="qwen3_5_moe",
    ),
    pytest.param(
        "glm4_moe",
        lambda: _make_tiny_causal_lm(
            "glm4_moe",
            moe_intermediate_size=MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE,
            n_routed_experts=MOE_BLOCK_NUM_EXPERTS,
            num_experts_per_tok=MOE_BLOCK_TOP_K,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            head_dim=16,
        ),
        "Glm4MoeMoE",
        ("decode_bmm", "simple_loop", "expert_parallel"),
        {},
        id="glm4_moe",
    ),
    pytest.param(
        "gpt_oss",
        lambda: _make_tiny_causal_lm(
            "gpt_oss",
            num_local_experts=MOE_BLOCK_NUM_EXPERTS,
            num_experts_per_tok=MOE_BLOCK_TOP_K,
        ),
        "GptOssMLP",
        ("decode_bmm", "simple_loop", "expert_parallel"),
        {"gpt_oss_prefill": True},
        id="gpt_oss",
    ),
    pytest.param(
        "mixtral",
        lambda: _make_tiny_causal_lm(
            "mixtral",
            num_local_experts=MOE_BLOCK_NUM_EXPERTS,
            num_experts_per_tok=MOE_BLOCK_TOP_K,
        ),
        "MixtralSparseMoeBlock",
        ("decode_bmm", "simple_loop"),
        {},
        id="mixtral",
    ),
    pytest.param(
        "granitemoe",
        lambda: _make_tiny_causal_lm(
            "granitemoe",
            num_local_experts=MOE_BLOCK_NUM_EXPERTS,
            num_experts_per_tok=MOE_BLOCK_TOP_K,
        ),
        "GraniteMoeMoE",
        ("simple_loop",),
        {},
        id="granitemoe",
    ),
    pytest.param(
        "llama4",
        _make_tiny_llama4_text_model,
        "Llama4TextMoe",
        ("simple_loop",),
        {},
        id="llama4",
    ),
    pytest.param(
        "deepseek_v3",
        lambda: _make_tiny_causal_lm(
            "deepseek_v3",
            moe_intermediate_size=MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE,
            n_routed_experts=MOE_BLOCK_NUM_EXPERTS,
            num_local_experts=MOE_BLOCK_NUM_EXPERTS,
            num_experts_per_tok=MOE_BLOCK_TOP_K,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            q_lora_rank=None,
            kv_lora_rank=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            qk_nope_head_dim=8,
        ),
        "DeepseekV3MoE",
        ("decode_bmm", "simple_loop", "expert_parallel"),
        {"external_mapper": True},
        id="deepseek_v3",
    ),
)


def _make_qeff_moe_block(original_block: nn.Module, flavour_name: str, options: dict) -> nn.Module:
    from QEfficient.transformers.moe import MoEFlavour

    qeff_block = copy.deepcopy(original_block).eval()
    if options.get("external_mapper"):
        _, transformed = ExternalOptimizedMoEMapperTransform.apply(qeff_block)
    else:
        _, transformed = OptimizedMoEMapperTransform.apply(qeff_block)
    assert transformed

    flavour = MoEFlavour(flavour_name)
    OptimizedMoEWeightsTransform.apply(qeff_block)
    qeff_block._moe_flavour = flavour
    if flavour is MoEFlavour.EXPERT_PARALLEL:
        qeff_block.num_parallelized_experts = 2
        qeff_block.num_pipeline_stages = 2
        qeff_block.expert_parallel_num_packed_chunks = 2
        OptimizedMoEExpertParallelWeightsTransform.apply(qeff_block)
    return qeff_block


@pytest.mark.parametrize(("model_family", "factory", "block_class_name", "flavours", "options"), MOE_BLOCK_PARITY_CASES)
def test_moe_block_flavour_forward_parity(model_family, factory, block_class_name, flavours, options):
    torch.manual_seed(17)
    model = factory().eval()
    original_block = _first_module_by_class_name(model, block_class_name)

    for flavour_name in flavours:
        seq_len = 1 if flavour_name == "decode_bmm" else MOE_BLOCK_SEQ_LEN
        hidden_states = torch.randn(1, seq_len, MOE_BLOCK_HIDDEN_SIZE)
        with torch.no_grad():
            expected = _first_tensor(original_block(hidden_states))

        qeff_block = _make_qeff_moe_block(original_block, flavour_name, options)
        if flavour_name == "expert_parallel":
            assert qeff_block.expert_parallel_num_packed_chunks == 2
        with torch.no_grad():
            actual = _first_tensor(qeff_block(hidden_states))
        actual = _match_expected_shape(actual, expected)

        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-4,
            rtol=1e-4,
            msg=f"{model_family} {flavour_name} block parity failed",
        )


@pytest.mark.parametrize(("model_family", "factory", "block_class_name", "flavours", "options"), MOE_BLOCK_PARITY_CASES)
def test_exact_mapped_moe_block_parity_cases_match_advertised_flavours(
    model_family, factory, block_class_name, flavours, options
):
    if options.get("external_mapper"):
        from QEfficient.transformers.models.deepseek_v3.modeling_deepseek import QEffDeepseekV3MoE
        from QEfficient.transformers.moe import MoEFlavour

        assert "supported_moe_flavours" in QEffDeepseekV3MoE.__dict__
        assert {flavour.value for flavour in QEffDeepseekV3MoE.supported_moe_flavours} == set(flavours)

        original_block = _first_module_by_class_name(factory().eval(), block_class_name)
        qeff_block = _make_qeff_moe_block(original_block, flavours[0], options)
        assert all(MoEFlavour(flavour_name) in qeff_block.get_supported_moe_flavours() for flavour_name in flavours)
        return

    from QEfficient.transformers.moe import MoEFlavour

    model = factory().eval()
    original_block = _first_module_by_class_name(model, block_class_name)
    qeff_block = _make_qeff_moe_block(original_block, flavours[0], options)
    advertised_flavours = {flavour.value for flavour in qeff_block.get_supported_moe_flavours()}

    assert "supported_moe_flavours" in type(qeff_block).__dict__, (
        f"{type(qeff_block).__name__} must explicitly advertise supported_moe_flavours"
    )
    assert advertised_flavours == set(flavours), f"{model_family} parity cases must cover every advertised flavour"
    assert all(MoEFlavour(flavour_name) in qeff_block.get_supported_moe_flavours() for flavour_name in flavours)


class _Grok1DummyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(MOE_BLOCK_HIDDEN_SIZE, MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE, bias=False)
        self.linear_v = nn.Linear(MOE_BLOCK_HIDDEN_SIZE, MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE, bias=False)
        self.linear_1 = nn.Linear(MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE, MOE_BLOCK_HIDDEN_SIZE, bias=False)
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_1(self.act_fn(self.linear(hidden_states)) * self.linear_v(hidden_states))


class MoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(MOE_BLOCK_HIDDEN_SIZE, MOE_BLOCK_NUM_EXPERTS, bias=False)
        self.top_k = MOE_BLOCK_TOP_K
        self.experts = nn.ModuleList(_Grok1DummyExpert() for _ in range(MOE_BLOCK_NUM_EXPERTS))

    def forward(self, hidden_states: torch.Tensor):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        topk_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        final_hidden_states = torch.zeros_like(hidden_states)

        for expert_idx, expert in enumerate(self.experts):
            token_indices, weight_indices = torch.where(selected_experts == expert_idx)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(hidden_states[token_indices])
            expert_output = expert_output * topk_weights[token_indices, weight_indices].to(
                hidden_states.dtype
            ).unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, expert_output)

        return final_hidden_states.view(orig_shape), router_logits


@pytest.mark.parametrize("flavour_name", ("decode_bmm", "simple_loop"))
def test_grok1_external_moe_block_flavour_forward_parity(flavour_name):
    torch.manual_seed(19)
    original_block = MoeBlock().eval()
    qeff_block = _make_qeff_moe_block(original_block, flavour_name, {"external_mapper": True})
    hidden_states = torch.randn(1, MOE_BLOCK_SEQ_LEN, MOE_BLOCK_HIDDEN_SIZE)

    with torch.no_grad():
        expected = _first_tensor(original_block(hidden_states))
        actual = _first_tensor(qeff_block(hidden_states))
    actual = _match_expected_shape(actual, expected)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_grok1_external_moe_block_does_not_advertise_expert_parallel():
    from QEfficient.transformers.models.grok_1.modeling_grok1 import QEffGrok1MoeBlock

    assert "supported_moe_flavours" in QEffGrok1MoeBlock.__dict__

    original_block = MoeBlock().eval()
    qeff_block = copy.deepcopy(original_block)

    _, transformed = ExternalOptimizedMoEMapperTransform.apply(qeff_block)

    assert transformed
    assert MoEFlavour.EXPERT_PARALLEL not in qeff_block.get_supported_moe_flavours()
    with pytest.raises(NotImplementedError, match="expert_parallel"):
        OptimizedMoEExportConfigTransform.apply(
            qeff_block,
            prefill_only=True,
            qaic_config={"moe_config": {"flavour": "expert_parallel"}},
        )


def test_glm4_moe_blocked_prefill_forward_parity():
    from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import QEffGlm4MoeMoE
    from QEfficient.transformers.moe import MoEFlavour

    config = AutoConfig.for_model("glm4_moe", **GLM4_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff_model = copy.deepcopy(model)
    KVCacheTransform.apply(qeff_model)
    OptimizedMoEMapperTransform.apply(qeff_model)
    OptimizedMoEWeightsTransform.apply(qeff_model)
    qeff_block = next(module for module in qeff_model.modules() if isinstance(module, QEffGlm4MoeMoE))

    chunked_model = copy.deepcopy(model)
    KVCacheTransform.apply(chunked_model)
    OptimizedMoEMapperTransform.apply(chunked_model)
    PrefillOnlyChunkedTransform.apply(chunked_model)
    OptimizedMoEWeightsTransform.apply(chunked_model)
    OptimizedMoEExportConfigTransform.apply(
        chunked_model,
        prefill_only=True,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
        prefill_seq_len=8,
    )
    OptimizedMoEExpertParallelWeightsTransform.apply(chunked_model)
    chunked_block = next(module for module in chunked_model.modules() if isinstance(module, QEffGlm4MoeMoE))

    assert chunked_block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
    assert chunked_block.num_parallelized_experts == 2
    assert chunked_block.num_pipeline_stages == 2
    assert chunked_block.expert_parallel_num_packed_chunks == 1

    x = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        orig = qeff_block(x)
        blocked = chunked_block(x)

    assert orig.shape == blocked.shape
    assert torch.allclose(orig, blocked, atol=1e-4, rtol=1e-4)


def test_glm4_moe_decode_export(tmp_path):
    config = AutoConfig.for_model("glm4_moe", **GLM4_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.export(tmp_path / "decode")
    assert qeff.onnx_path.is_file()


def test_glm4_moe_prefill_chunked_subfunction_export_contains_cumsum_custom_ops(tmp_path):
    import onnx

    config = AutoConfig.for_model("glm4_moe", **GLM4_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    _apply_prefill_compile_transforms(
        qeff,
        prefill_seq_len=512,
        ctx_len=512,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
    )
    onnx_path = qeff.export(
        tmp_path / "prefill-subfunction",
        prefill_only=True,
        prefill_seq_len=512,
        enable_chunking=True,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    decoder_functions = [func for func in onnx_model.functions if func.name.startswith("QEffGlm4MoeDecoderLayer")]
    assert len(decoder_functions) == config.num_hidden_layers

    for function_proto in decoder_functions:
        op_counts = Counter(node.op_type for node in function_proto.node)
        assert op_counts["Sin"] == 0
        assert op_counts["Cos"] == 0
        # prefill_seq_len=512 and expert_parallel_chunk_size=256 gives two packed chunks.
        # With n_routed_experts=4 and num_cores=2, each layer has two expert slots.
        assert op_counts["CtxGather3D"] == 12
        assert op_counts["CtxScatter3D"] == 4
        assert op_counts["CtxScatter3DInt"] == 2


def test_glm4_moe_kv_blocking_transform_and_prefill_export(tmp_path):
    import onnx

    from QEfficient.blocking.attention_blocking import BlockingMode
    from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import QEffGlm4MoeAttention

    config = AutoConfig.for_model("glm4_moe", **GLM4_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.transform(
        ctx_len=1024,
        seq_len=512,
        qaic_config={"blocking_mode": "kv", "num_kv_blocks": 2},
    )

    attn_modules = [module for module in qeff.model.modules() if isinstance(module, QEffGlm4MoeAttention)]
    assert attn_modules
    for attn_module in attn_modules:
        blocking_config = getattr(attn_module, "attn_blocking_config", None)
        assert blocking_config is not None
        assert blocking_config.mode == BlockingMode.KV
        assert blocking_config.num_kv_blocks == 2

    onnx_path = qeff.export(
        tmp_path / "prefill-kv-blocked",
        prefill_only=True,
        prefill_seq_len=512,
        enable_chunking=True,
        use_onnx_subfunctions=True,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
        offload_pt_weights=False,
    )
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    decoder_functions = [func for func in onnx_model.functions if func.name.startswith("QEffGlm4MoeDecoderLayer")]
    assert decoder_functions
    assert (
        Counter(node.op_type for node in onnx_model.graph.node)["QEffGlm4MoeDecoderLayer"] == config.num_hidden_layers
    )
    for function_proto in decoder_functions:
        op_counts = Counter(node.op_type for node in function_proto.node)
        assert op_counts["CtxGatherBlockedKV"] == 4


# ── Qwen3MOE ──────────────────────────────────────────────────────────────────


QWEN3_MOE_CFG = dict(
    max_position_embeddings=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    hidden_size=128,
    intermediate_size=512,
    vocab_size=127,
    num_key_value_heads=2,
)
GPTOSS_CFG = dict(
    max_position_embeddings=256,
    num_hidden_layers=2,
    num_attention_heads=2,
    hidden_size=32,
    intermediate_size=32,
    vocab_size=127,
    num_key_value_heads=2,
)


# ── Qwen3MOE ──────────────────────────────────────────────────────────────────


def test_qwen3moe_blocked_forward_parity():
    from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import QEffQwen3MoeSparseMoeBlock
    from QEfficient.transformers.moe import MoEFlavour

    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)

    blocks = [
        m
        for _, m in model.named_modules()
        if hasattr(m, "experts") and hasattr(m, "gate") and hasattr(m.gate, "num_experts")
    ]
    assert blocks

    qeff_model = copy.deepcopy(model)
    KVCacheTransform.apply(qeff_model)
    OptimizedMoEMapperTransform.apply(qeff_model)
    OptimizedMoEWeightsTransform.apply(qeff_model)
    qeff_block = next(module for module in qeff_model.modules() if isinstance(module, QEffQwen3MoeSparseMoeBlock))

    chunked_model = copy.deepcopy(model)
    KVCacheTransform.apply(chunked_model)
    OptimizedMoEMapperTransform.apply(chunked_model)
    PrefillOnlyChunkedTransform.apply(chunked_model)
    OptimizedMoEWeightsTransform.apply(chunked_model)
    OptimizedMoEExportConfigTransform.apply(
        chunked_model,
        prefill_only=True,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
        prefill_seq_len=8,
    )
    OptimizedMoEExpertParallelWeightsTransform.apply(chunked_model)
    chunked = next(module for module in chunked_model.modules() if isinstance(module, QEffQwen3MoeSparseMoeBlock))

    assert chunked._moe_flavour is MoEFlavour.EXPERT_PARALLEL
    assert chunked.num_parallelized_experts == 2
    assert chunked.num_pipeline_stages == chunked.num_experts // chunked.num_parallelized_experts
    assert chunked.expert_parallel_num_packed_chunks == 1

    x = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        orig, _ = qeff_block(x)
        blocked, _ = chunked.forward(x)

    assert orig.shape == blocked.shape
    assert (orig - blocked).abs().max().item() < 0.1, "Qwen3MOE parity failed"


def test_qwen3moe_decode_export(tmp_path):
    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.export(tmp_path / "decode")
    assert qeff.onnx_path.is_file()


def test_qwen3moe_prefill_chunked_export(tmp_path):
    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    _apply_prefill_compile_transforms(qeff, prefill_seq_len=32, ctx_len=32, num_cores=2)
    qeff.export(tmp_path / "prefill", prefill_only=True, enable_chunking=True, num_cores=2)
    assert qeff.onnx_path.is_file()


def test_qwen3moe_disagg_compile_uses_distinct_decode_and_prefill_onnx(tmp_path, monkeypatch):
    import subprocess

    compile_commands = []

    def fake_compile(command, *args, **kwargs):
        compile_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_compile)

    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)

    _apply_decode_compile_transforms(qeff, seq_len=1, ctx_len=128, num_cores=2)
    decode_onnx_path = qeff.export(
        tmp_path / "decode-export",
        prefill_only=False,
        offload_pt_weights=False,
        retain_full_kv=True,
    )
    qeff.compile(
        onnx_path=str(decode_onnx_path),
        compile_dir=tmp_path / "decode-compile",
        prefill_seq_len=1,
        ctx_len=128,
        num_cores=2,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        offload_pt_weights=False,
        retain_full_kv=True,
    )

    _apply_prefill_compile_transforms(
        qeff,
        prefill_seq_len=64,
        ctx_len=128,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 32}},
    )
    prefill_onnx_path = qeff.export(
        tmp_path / "prefill-export",
        prefill_only=True,
        prefill_seq_len=64,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 32}},
        enable_chunking=True,
        offload_pt_weights=False,
    )
    qeff.compile(
        onnx_path=str(prefill_onnx_path),
        compile_dir=tmp_path / "prefill-compile",
        prefill_seq_len=64,
        ctx_len=128,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 32}},
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        prefill_only=True,
        enable_chunking=True,
        offload_pt_weights=False,
    )

    compiled_onnx_args = [arg for command in compile_commands for arg in command if str(arg).startswith("-m=")]
    assert len(compiled_onnx_args) == 2
    assert decode_onnx_path != prefill_onnx_path
    assert decode_onnx_path.is_file()
    assert prefill_onnx_path.is_file()
    assert compiled_onnx_args[0] == f"-m={decode_onnx_path}"
    assert compiled_onnx_args[1] == f"-m={prefill_onnx_path}"


def test_qwen3moe_prefill_chunked_subfunction_export_contains_cumsum_custom_ops(tmp_path):
    import onnx
    from onnx import numpy_helper

    config = AutoConfig.for_model("qwen3_moe", **{**QWEN3_MOE_CFG, "max_position_embeddings": 1024})
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    _apply_prefill_compile_transforms(
        qeff,
        prefill_seq_len=512,
        ctx_len=512,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
    )
    onnx_path = qeff.export(
        tmp_path / "prefill-subfunction",
        prefill_only=True,
        enable_chunking=True,
        prefill_seq_len=512,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    function_names = {func.name for func in onnx_model.functions}
    used_op_types = {node.op_type for node in onnx_model.graph.node}
    slice_starts = []
    for function_proto in onnx_model.functions:
        constants = {}
        for node in function_proto.node:
            used_op_types.add(node.op_type)
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        constants[node.output[0]] = numpy_helper.to_array(attr.t).flatten().tolist()
        for node in function_proto.node:
            if node.op_type == "Slice" and len(node.input) > 1 and node.input[1] in constants:
                slice_starts.append(constants[node.input[1]])

    assert "CtxScatter3DInt" in function_names
    assert "CtxScatter3D" in function_names
    assert "CtxGather3D" in function_names
    assert "CtxScatter3DInt" in used_op_types
    assert "CtxScatter3D" in used_op_types
    assert "CtxGather3D" in used_op_types
    assert qeff.hash_params["moe_prefill_num_packed_chunks"] == 2
    assert qeff.hash_params["moe_prefill_expert_parallel_chunk_size"] == 256
    assert qeff.hash_params["moe_prefill_num_parallelized_experts"] == 2
    assert (
        qeff.hash_params["moe_prefill_num_pipeline_stages"]
        == config.num_experts // qeff.hash_params["moe_prefill_num_parallelized_experts"]
    )
    assert [256] not in slice_starts


# ── GPT-OSS ───────────────────────────────────────────────────────────────────


def test_gptoss_blocked_forward_parity():
    from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import QEffGptOssMLP
    from QEfficient.transformers.moe import MoEFlavour

    config = AutoConfig.for_model("gpt_oss", **GPTOSS_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)

    blocks_orig = [m for _, m in model.named_modules() if m.__class__.__name__ == "GptOssMLP"]
    assert blocks_orig

    x = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        orig, _ = blocks_orig[0].forward(x)

    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)

    chunked_model = copy.deepcopy(qeff.model)
    PrefillOnlyChunkedTransform.apply(chunked_model)
    OptimizedMoETransform.apply(
        chunked_model,
        prefill_only=True,
        num_cores=2,
        prefill_seq_len=8,
        qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 256}},
    )
    blocks_chunked = [m for _, m in chunked_model.named_modules() if isinstance(m, QEffGptOssMLP)]
    assert blocks_chunked
    assert blocks_chunked[0]._moe_flavour is MoEFlavour.EXPERT_PARALLEL
    assert blocks_chunked[0].num_parallelized_experts == 2
    assert (
        blocks_chunked[0].num_pipeline_stages
        == blocks_chunked[0].num_experts // blocks_chunked[0].num_parallelized_experts
    )
    assert blocks_chunked[0].expert_parallel_num_packed_chunks == 1

    with torch.no_grad():
        blocked, _ = blocks_chunked[0].forward(x)

    loop_model = copy.deepcopy(qeff.model)
    PrefillOnlyTransform.apply(loop_model)
    OptimizedMoETransform.apply(
        loop_model,
        prefill_only=True,
        qaic_config={"moe_config": {"flavour": "simple_loop"}},
    )
    blocks_loop = [m for _, m in loop_model.named_modules() if isinstance(m, QEffGptOssMLP)]
    assert blocks_loop
    assert blocks_loop[0]._moe_flavour is MoEFlavour.SIMPLE_LOOP
    with torch.no_grad():
        looped, _ = blocks_loop[0].forward(x)

    assert orig.shape == blocked.shape == looped.shape
    assert (orig - blocked).abs().max().item() < 0.1, "GPT-OSS HF vs expert-parallel parity failed"
    assert (orig - looped).abs().max().item() < 1e-3, "GPT-OSS HF vs simple-loop parity failed"


@pytest.mark.parametrize("flavour", ("decode_bmm", "simple_loop", "expert_parallel"))
def test_gemma4_text_moe_block_forward_parity(flavour):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextExperts, Gemma4TextRouter

    from QEfficient.transformers.models.gemma4.modeling_gemma4 import QEffGemma4TextMoeBlock
    from QEfficient.transformers.moe import MoEFlavour

    torch.manual_seed(29)
    config = Gemma4TextConfig(
        hidden_size=MOE_BLOCK_HIDDEN_SIZE,
        intermediate_size=MOE_BLOCK_INTERMEDIATE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=MOE_BLOCK_NUM_EXPERTS,
        top_k_experts=MOE_BLOCK_TOP_K,
        moe_intermediate_size=MOE_BLOCK_EXPERT_INTERMEDIATE_SIZE,
        enable_moe_block=True,
    )
    router = Gemma4TextRouter(config).eval()
    experts = Gemma4TextExperts(config).eval()
    with torch.no_grad():
        experts.gate_up_proj.normal_(mean=0.0, std=0.02)
        experts.down_proj.normal_(mean=0.0, std=0.02)
    pre_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    post_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    qeff_block = QEffGemma4TextMoeBlock(
        copy.deepcopy(router),
        copy.deepcopy(experts),
        copy.deepcopy(pre_norm),
        copy.deepcopy(post_norm),
    ).eval()

    hidden_states = torch.randn(1, MOE_BLOCK_SEQ_LEN, MOE_BLOCK_HIDDEN_SIZE)
    with torch.no_grad():
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, top_k_weights, top_k_index = router(flat)
        expected = pre_norm(flat)
        expected = experts(expected, top_k_index, top_k_weights).reshape(hidden_states.shape)
        expected = post_norm(expected)

    assert isinstance(qeff_block, QEffGemma4TextMoeBlock)
    assert hasattr(qeff_block.experts, "gate_up_proj")
    assert not hasattr(qeff_block.experts, "gate_proj")

    _, weights_ready = OptimizedMoEWeightsTransform.apply(qeff_block)

    assert weights_ready
    assert hasattr(qeff_block, "moe_weights")
    assert not hasattr(qeff_block.experts, "gate_up_proj")
    assert not hasattr(qeff_block.experts, "down_proj")

    if flavour == "expert_parallel":
        OptimizedMoEExportConfigTransform.apply(
            qeff_block,
            prefill_only=True,
            num_cores=2,
            qaic_config={"moe_config": {"flavour": flavour, "expert_parallel_chunk_size": 4}},
            prefill_seq_len=MOE_BLOCK_SEQ_LEN,
        )
        OptimizedMoEExpertParallelWeightsTransform.apply(qeff_block)
    else:
        qeff_block._moe_flavour = MoEFlavour(flavour)

    with torch.no_grad():
        actual = qeff_block(hidden_states)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_gptoss_decode_export(tmp_path):
    config = AutoConfig.for_model("gpt_oss", **GPTOSS_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.export(tmp_path / "decode")
    assert qeff.onnx_path.is_file()


def test_gptoss_prefill_chunked_export(tmp_path):
    config = AutoConfig.for_model("gpt_oss", **GPTOSS_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    _apply_prefill_compile_transforms(qeff, prefill_seq_len=32, ctx_len=32, num_cores=2)
    qeff.export(tmp_path / "prefill", prefill_only=True, enable_chunking=True, num_cores=2)
    assert qeff.onnx_path.is_file()


def test_gptoss_blocked_chunked_prefill_writes_current_chunk_once(monkeypatch):
    from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
    from QEfficient.transformers.cache_utils import QEffGPTOSSHybridCache
    from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import QEffPrefillOnlyChunkedGptOssAttention

    config = AutoConfig.for_model("gpt_oss", **{**GPTOSS_CFG, "num_hidden_layers": 2, "sliding_window": 4})
    attention = QEffPrefillOnlyChunkedGptOssAttention(config, layer_idx=1).eval()
    assert attention.sliding_window is None
    attention.attn_blocking_config = AttentionBlockingConfig(
        mode=BlockingMode.PREFILL_ONLINE,
        num_kv_blocks=2,
        num_q_blocks=2,
        ctx_len=8,
    )

    cache = QEffGPTOSSHybridCache(config=config)
    cache.update(
        torch.zeros(1, config.num_key_value_heads, 8, attention.head_dim),
        torch.zeros(1, config.num_key_value_heads, 8, attention.head_dim),
        layer_idx=1,
        cache_kwargs={"position_ids": torch.arange(8).reshape(1, -1)},
    )

    calls = {"write": 0}
    original_write_only = cache.write_only

    def fail_full_cache_update_chunked(*args, **kwargs):
        raise AssertionError("blocked chunked prefill must not pre-gather K/V before generic blocked attention")

    def tracking_write_only(key_states, value_states, layer_idx, cache_kwargs):
        calls["write"] += 1
        assert key_states.shape[2] == cache_kwargs["position_ids"].shape[1]
        assert value_states.shape[2] == cache_kwargs["position_ids"].shape[1]
        return original_write_only(key_states, value_states, layer_idx, cache_kwargs)

    monkeypatch.setattr(cache, "full_cache_update_chunked", fail_full_cache_update_chunked)
    monkeypatch.setattr(cache, "write_only", tracking_write_only)

    hidden_states = torch.randn(1, 4, config.hidden_size)
    position_ids = torch.arange(4, 8).reshape(1, -1)
    cos = torch.ones(1, 1, 4, attention.head_dim)
    sin = torch.zeros(1, 1, 4, attention.head_dim)

    with torch.no_grad():
        attn_output, _, returned_cache = attention(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=cache,
            cos_cached=cos,
            sin_cached=sin,
        )

    assert calls["write"] == 1
    assert attn_output.shape == hidden_states.shape
    assert returned_cache is cache


def test_gptoss_prefill_online_blocked_matches_eager_with_sinks():
    from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
    from QEfficient.transformers.cache_utils import QEffGPTOSSHybridCache
    from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
    from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import QEffPrefillOnlyChunkedGptOssAttention

    torch.manual_seed(31)
    config = AutoConfig.for_model(
        "gpt_oss",
        **{
            **GPTOSS_CFG,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "sliding_window": 4,
        },
    )
    eager_attention = QEffPrefillOnlyChunkedGptOssAttention(config, layer_idx=1).eval()
    blocked_attention = copy.deepcopy(eager_attention).eval()
    assert blocked_attention.sliding_window is None
    blocked_attention.attn_blocking_config = AttentionBlockingConfig(
        mode=BlockingMode.PREFILL_ONLINE,
        num_kv_blocks=2,
        num_q_blocks=2,
        ctx_len=8,
    )

    hidden_states = torch.randn(1, 8, config.hidden_size)
    position_ids = torch.arange(8).reshape(1, -1)
    attention_mask = _create_causal_mask(position_ids=position_ids, target_length=8)
    cos = torch.ones(1, 1, 8, blocked_attention.head_dim)
    sin = torch.zeros(1, 1, 8, blocked_attention.head_dim)

    eager_cache = QEffGPTOSSHybridCache(config=config)
    blocked_cache = QEffGPTOSSHybridCache(config=config)
    for cache in (eager_cache, blocked_cache):
        cache.update(
            torch.zeros(1, config.num_key_value_heads, 8, blocked_attention.head_dim),
            torch.zeros(1, config.num_key_value_heads, 8, blocked_attention.head_dim),
            layer_idx=1,
            cache_kwargs={"position_ids": torch.arange(8).reshape(1, -1)},
        )

    with torch.no_grad():
        eager_output, _, _ = eager_attention(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=eager_cache,
            cos_cached=cos,
            sin_cached=sin,
        )
        blocked_output, _, _ = blocked_attention(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=blocked_cache,
            cos_cached=cos,
            sin_cached=sin,
        )

    torch.testing.assert_close(blocked_output, eager_output, atol=1e-5, rtol=1e-5)


def test_gptoss_prefill_chunked_export_traces_packed_chunks(tmp_path):
    import onnx
    from onnx import numpy_helper

    config = AutoConfig.for_model("gpt_oss", **{**GPTOSS_CFG, "max_position_embeddings": 1024})
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
    _apply_prefill_compile_transforms(
        qeff,
        prefill_seq_len=512,
        ctx_len=512,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
    )
    onnx_path = qeff.export(
        tmp_path / "prefill-subfunction-512",
        prefill_only=True,
        enable_chunking=True,
        prefill_seq_len=512,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    op_types = []
    slice_starts = []
    for nodes in [onnx_model.graph.node] + [function.node for function in onnx_model.functions]:
        constants = {}
        for node in nodes:
            op_types.append(node.op_type)
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        constants[node.output[0]] = numpy_helper.to_array(attr.t).flatten().tolist()
        for node in nodes:
            if node.op_type == "Slice" and len(node.input) > 1 and node.input[1] in constants:
                slice_starts.append(constants[node.input[1]])

    assert qeff.hash_params["moe_prefill_num_packed_chunks"] == 2
    assert qeff.hash_params["moe_prefill_expert_parallel_chunk_size"] == 256
    assert qeff.hash_params["moe_prefill_num_parallelized_experts"] == 2
    assert (
        qeff.hash_params["moe_prefill_num_pipeline_stages"]
        == config.num_local_experts // qeff.hash_params["moe_prefill_num_parallelized_experts"]
    )
    assert [256] not in slice_starts
    assert op_types.count("CtxGather3D") >= 2 * op_types.count("CtxScatter3DInt")
