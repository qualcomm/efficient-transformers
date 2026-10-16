# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import platform

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# HybridCache was removed in transformers v5. DynamicCache is the v5
# equivalent and is used wherever a pre-built cache object is needed.
from transformers.cache_utils import DynamicCache

from QEfficient.customop.matmulnbits import QuantLinearORT
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.transformers.spd.turbo import ResBlock
from QEfficient.utils._utils import get_padding_shape_from_config
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


KVCacheTransformTestConfigs = [
    ("llama", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("llama", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("llama", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("llama", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("gpt2", 3, 12, 192, {"n_inner": 512}, 0.8),
    ("gpt2", 1, 12, 192, {"n_inner": 512}, 0.8),
    ("codegen", 1, 16, 1024, {"n_inner": 2048}, 0.8),
    ("codegen", 3, 16, 1024, {"n_inner": 2048}, 0.8),
    ("falcon", 1, 71, 4544, {"multi_query": True}, 1.5),
    ("falcon", 3, 71, 4544, {"multi_query": False}, 1.5),
    ("falcon", 1, 71, 4544, {"multi_query": False}, 1.5),
    ("falcon", 3, 71, 4544, {"multi_query": True}, 1.5),
    ("gptj", 3, 16, 4096, {"n_inner": 512}, 1),
    ("gptj", 1, 16, 4096, {"n_inner": 512}, 1.2),
    ("mistral", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("mistral", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("mistral", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("mistral", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("mixtral", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("mixtral", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("mixtral", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("mixtral", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("mpt", 1, 16, 2048, {}, 0.8),
    ("mpt", 3, 16, 2048, {}, 0.8),
    ("phi", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("phi", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("phi", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("phi", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("phi3", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("phi3", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("phi3", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("phi3", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("qwen2", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("qwen2", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("qwen2", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("qwen2", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("starcoder2", 3, 24, 192, {"num_key_value_heads": 2, "intermediate_size": 512}, 0.8),
    ("starcoder2", 1, 24, 192, {"num_key_value_heads": 2, "intermediate_size": 512}, 0.8),
    ("starcoder2", 3, 24, 192, {"num_key_value_heads": 24, "intermediate_size": 512}, 0.8),
    ("starcoder2", 1, 24, 192, {"num_key_value_heads": 24, "intermediate_size": 512}, 0.8),
    ("gemma", 3, 8, 2048, {"num_key_value_heads": 1, "intermediate_size": 512}, 0.8),
    ("gemma", 1, 8, 2048, {"num_key_value_heads": 1, "intermediate_size": 512}, 0.8),
]

SpDTransformTestConfigs = [
    ("llama", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("llama", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("llama", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("llama", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("qwen2", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
]

SpDTransformProjTestConfigs = [
    ("llama", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
]


def create_qaic_model_inputs(
    input_len: int, vocab_size: int, padding_shape: tuple, num_hidden_layers: int, is_tlm: bool = False
) -> dict:
    """create pytorch QEff model inputs

    ``Mandatory`` Args:
        :input_len (int): input length.
        :vocab_size (int): vocab size.
        :padding_shape (tuple): padding shape of KV$.
        :num_hidden_layers (int): number of hidden layers.
    ``Optional`` Args:
        :is_tlm (bool, optional): whether this is an SpD TLM model. Defaults to False.

    Returns:
        :dict: pytorch QEff model inputs
    """
    input_ids = torch.randint(0, vocab_size, size=(1, input_len))
    past_key_values = []
    for _ in range(num_hidden_layers):
        past_key = torch.zeros((padding_shape), dtype=torch.float32)
        past_value = torch.zeros((padding_shape), dtype=torch.float32)
        pkv = (past_key, past_value)
        past_key_values.append(pkv)
    inputs = dict(
        input_ids=input_ids,
        position_ids=torch.Tensor([range(input_ids.shape[1])]).long(),
        past_key_values=tuple(past_key_values),
        output_hidden_states=True,
    )
    if is_tlm:
        inputs["num_logits_to_keep"] = torch.zeros((input_len, 1))
    return inputs


def compare_original_vs_kv_model_pt_outputs(original_val, kv_val, tolerance=1e-6) -> bool:
    # Base case
    if original_val is None:
        assert kv_val is None
        return True
    elif isinstance(original_val, torch.Tensor):
        if original_val.shape != kv_val.shape:
            original_val = original_val[:, -1:, :]  # LM Head outputs
        mae = torch.mean(torch.abs(original_val - kv_val))
        if mae >= tolerance:
            logger.critical(f"MAE={mae} is greater than expected tolerance={tolerance}")
            return False
        return True
    # Call recursively if tuple/list
    elif isinstance(original_val, (tuple, list)):
        for sub_orig_val, sub_kv_val in zip(original_val, kv_val):
            if not compare_original_vs_kv_model_pt_outputs(sub_orig_val, sub_kv_val, tolerance):
                return False
        return True
    else:
        raise TypeError(f"got unexpected type inputs {type(original_val)}")


def run_kv_cache_transform_and_test(
    hf_model,
    qaic_model_inputs,
    logits_tolerance=0.8,
    kv_cache=None,
):
    hf_model.eval()
    # Run original model
    input_ids = qaic_model_inputs["input_ids"]
    input_len = input_ids.shape[1]
    with torch.inference_mode():
        if isinstance(kv_cache, type(None)):
            original_model_outputs = hf_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=kv_cache,
            )
            # In transformers v5 past_key_values is a DynamicCache whose
            # layers are accessed by iterating (not subscripting). Each layer
            # yields (key, value, ...) so we take only the first two elements.
            original_model_outputs["past_key_values"] = tuple(
                [
                    (
                        layer[0][:, :, :input_len, :],  # key cache
                        layer[1][:, :, :input_len, :],  # value cache
                    )
                    for layer in original_model_outputs["past_key_values"]
                ]
            )
        else:
            original_model_outputs = hf_model(input_ids=input_ids, output_hidden_states=True)
        hidden_size_projections = (
            hf_model.hidden_size_projections if hasattr(hf_model, "hidden_size_projections") else None
        )
        if hidden_size_projections:
            # compute projections
            last_hidden_size = original_model_outputs.hidden_states[-1]  # shape: [bsz, seq_len, d_model]
            proj_hidden_sizes = [last_hidden_size]
            for proj in hidden_size_projections:
                proj_i = proj(last_hidden_size)
                proj_hidden_sizes.append(proj_i)
            proj_hidden_sizes = torch.stack(proj_hidden_sizes, dim=2)
            logits = hf_model.lm_head(proj_hidden_sizes)
            original_model_outputs.logits = logits

    # Apply transforms
    qaic_config = None
    if "num_logits_to_keep" in qaic_model_inputs:
        qaic_config = dict(speculative_model_type="target")
    hf_model = QEFFAutoModelForCausalLM(hf_model, qaic_config=qaic_config).model
    if hidden_size_projections is not None:
        hf_model.projections = hidden_size_projections

    # Run KV model
    with torch.inference_mode():
        transformed_model_outputs = hf_model(**qaic_model_inputs)

    assert original_model_outputs.keys() == transformed_model_outputs.keys(), "Model output keys do not match!"

    # FIXME: Tolerance should not be so high for logits
    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["logits"], transformed_model_outputs["logits"], tolerance=logits_tolerance
    ), "Logits are not matching with tolerance=0.8"
    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["hidden_states"], transformed_model_outputs["hidden_states"], tolerance=1e-5
    )

    # Slice Past key values based on input_len
    pkv = transformed_model_outputs["past_key_values"][0]
    new_pkv = []
    for past_key_value in pkv:
        new_pkv.append(past_key_value[:, :, :input_len, :])
    transformed_model_outputs["past_key_values"] = (tuple(new_pkv),)

    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["past_key_values"], transformed_model_outputs["past_key_values"], tolerance=1e-10
    )


# FIXME: Temporarily skip because Qwen3.5 gated RMSNorm fails in this generic test without a gate input.
@pytest.mark.skip(reason="Qwen3.5 gated RMSNorm requires gate input; generic RMSNorm test needs update")
@pytest.mark.parametrize("input_size", [2, 5], ids=lambda x: "input_size=" + str(x))
@pytest.mark.parametrize("hidden_size", [64, 1024], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("module", CustomOpsTransform._module_mapping.keys(), ids=lambda x: "module=" + x.__name__)
def test_rms_norm_ops_transform(module: torch.nn.Module, hidden_size: int, input_size: int) -> None:
    """Test custom Ops transform individually

    Args:
        module (nn.Module): Pytorch module
        hidden_size (int): hidden_size for RMSNorm operation
        input_size (int): Random inputs shape for testing
    """
    import inspect

    first_param = list(inspect.signature(module.__init__).parameters.values())[1]
    if first_param.annotation not in (int, inspect.Parameter.empty) or first_param.name == "config":
        pytest.skip(f"{module.__name__} requires a full config object, not a plain int — not an RMSNorm-style module")
    model = module(hidden_size)
    rand_data = torch.rand(input_size, hidden_size)

    original_output = model(rand_data)

    model, transformed = CustomOpsTransform.apply(model)
    assert transformed

    transformed_model_output = model(rand_data)

    assert not isinstance(model, module)
    assert torch.all(original_output == transformed_model_output)


@pytest.mark.parametrize(
    "config_class, num_hidden_layers, num_attention_heads, hidden_size, kwargs, logits_tolerance",
    KVCacheTransformTestConfigs,
)
def test_kv_cache_transform(
    config_class, num_hidden_layers, num_attention_heads, hidden_size, kwargs, logits_tolerance
):
    config = AutoConfig.for_model(
        config_class,
        **kwargs,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        use_cache=True,
        cache_position=None,
        position_embeddings=None,
    )
    hf_model = AutoModelForCausalLM.from_config(config=config, attn_implementation="eager")

    kv_cache = None
    if hasattr(config, "cache_implementation") and config.cache_implementation == "hybrid":
        # HybridCache was removed in transformers v5. Use DynamicCache instead,
        # which is the standard cache type in v5 and handles all model types.
        kv_cache = DynamicCache()

    padding_shape = get_padding_shape_from_config(config=config, batch_size=1, seq_len=32)

    # Prepare KV model inputs
    qaic_model_inputs = create_qaic_model_inputs(
        input_len=8, vocab_size=config.vocab_size, padding_shape=padding_shape, num_hidden_layers=num_hidden_layers
    )

    run_kv_cache_transform_and_test(
        hf_model,
        qaic_model_inputs=qaic_model_inputs,
        logits_tolerance=logits_tolerance,
        kv_cache=kv_cache,
    )


@pytest.mark.parametrize(
    "config_class, num_hidden_layers, num_attention_heads, hidden_size, kwargs, logits_tolerance",
    SpDTransformTestConfigs,
)
def test_spd_transform(config_class, num_hidden_layers, num_attention_heads, hidden_size, kwargs, logits_tolerance):
    config = AutoConfig.for_model(
        config_class,
        **kwargs,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        use_cache=True,
        cache_position=None,
        position_embeddings=None,
    )
    hf_model = AutoModelForCausalLM.from_config(config=config, attn_implementation="eager")

    kv_cache = None
    if hasattr(config, "cache_implementation") and config.cache_implementation == "hybrid":
        # HybridCache was removed in transformers v5. Use DynamicCache instead,
        # which is the standard cache type in v5 and handles all model types.
        kv_cache = DynamicCache()

    padding_shape = get_padding_shape_from_config(config=config, batch_size=1, seq_len=32)

    # Prepare KV model inputs
    qaic_model_inputs = create_qaic_model_inputs(
        input_len=8,
        vocab_size=config.vocab_size,
        padding_shape=padding_shape,
        num_hidden_layers=num_hidden_layers,
        is_tlm=True,
    )

    run_kv_cache_transform_and_test(
        hf_model,
        qaic_model_inputs=qaic_model_inputs,
        logits_tolerance=logits_tolerance,
        kv_cache=kv_cache,
    )


@pytest.mark.parametrize(
    "config_class, num_hidden_layers, num_attention_heads, hidden_size, kwargs, logits_tolerance",
    SpDTransformProjTestConfigs,
)
def test_spd_proj_transform(
    config_class, num_hidden_layers, num_attention_heads, hidden_size, kwargs, logits_tolerance
):
    config = AutoConfig.for_model(
        config_class,
        **kwargs,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        use_cache=True,
        cache_position=None,
        position_embeddings=None,
    )
    hf_model = AutoModelForCausalLM.from_config(config=config, attn_implementation="eager")
    proj_num_layers = 1
    num_speculative_tokens = 3
    hidden_size_projections = torch.nn.ModuleList(
        [
            torch.nn.Sequential(
                *([ResBlock(hidden_size)] * proj_num_layers),
            )
            for _ in range(num_speculative_tokens)
        ],
    )
    hf_model.hidden_size_projections = hidden_size_projections

    kv_cache = None
    if hasattr(config, "cache_implementation") and config.cache_implementation == "hybrid":
        # HybridCache was removed in transformers v5. Use DynamicCache instead,
        # which is the standard cache type in v5 and handles all model types.
        kv_cache = DynamicCache()

    padding_shape = get_padding_shape_from_config(config=config, batch_size=1, seq_len=32)

    # Prepare KV model inputs
    qaic_model_inputs = create_qaic_model_inputs(
        input_len=8,
        vocab_size=config.vocab_size,
        padding_shape=padding_shape,
        num_hidden_layers=num_hidden_layers,
        is_tlm=True,
    )

    run_kv_cache_transform_and_test(
        hf_model,
        qaic_model_inputs=qaic_model_inputs,
        logits_tolerance=logits_tolerance,
        kv_cache=kv_cache,
    )


@pytest.mark.parametrize("in_features", [2048, 4096])
@pytest.mark.parametrize("out_features", [2048, 4096])
@pytest.mark.skipif(platform.machine() == "aarch64", reason="Test skipped on aarch64 platform")
def test_awq_to_matmulnbits_transform(in_features, out_features):
    wqlinear = WQLinear_GEMM(bits=4, group_size=128, in_features=in_features, out_features=out_features, bias=False)

    wqlinear.qweight = torch.randint(
        low=-(2**31), high=2**31 - 1, size=(in_features, out_features // 8), dtype=torch.int32
    )
    wqlinear.qzeros = torch.randint(
        low=-(2**31), high=2**31 - 1, size=(in_features // wqlinear.group_size, out_features // 8), dtype=torch.int32
    )
    wqlinear.scales = torch.rand(in_features // wqlinear.group_size, out_features, dtype=torch.float32)

    rand_data = torch.rand(4, in_features)
    old_out = wqlinear(rand_data)
    new_module, transformed = AwqToMatmulNbitsTransform.apply(wqlinear)
    assert transformed
    new_out = new_module(rand_data)
    assert isinstance(new_module, QuantLinearORT)
    assert compare_original_vs_kv_model_pt_outputs(old_out, new_out, tolerance=1e-8), (
        "Test failed because MAE is greater than tolerance"
    )


@pytest.mark.parametrize("in_features", [4096, 4096])
@pytest.mark.parametrize("out_features", [4096, 4096])
def test_gptq_to_matmulnbits_transform(in_features, out_features):
    quant_linear_gptq = QuantLinearGPTQ(
        bits=4, group_size=128, in_features=in_features, out_features=out_features, bias=False
    )
    quant_linear_gptq.qweight = torch.randint(
        low=-(2**31), high=2**31 - 1, size=(in_features // 8, out_features), dtype=torch.int32
    )
    quant_linear_gptq.qzeros = torch.randint(
        low=-(2**31),
        high=2**31 - 1,
        size=(in_features // quant_linear_gptq.group_size, out_features // 8),
        dtype=torch.int32,
    )
    quant_linear_gptq.scales = torch.rand(
        in_features // quant_linear_gptq.group_size, out_features, dtype=torch.float32
    )
    rand_data = torch.rand(4, in_features)
    old_out = quant_linear_gptq(rand_data)
    new_module, transformed = GPTQToMatmulNbitsTransform.apply(quant_linear_gptq)
    assert transformed
    new_out = new_module(rand_data)
    assert isinstance(new_module, QuantLinearORT)
    assert compare_original_vs_kv_model_pt_outputs(old_out, new_out, tolerance=1e-4), (
        "Test failed because MAE is greater than tolerance"
    )
