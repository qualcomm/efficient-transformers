# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import os
from time import perf_counter

import onnx
import pytest
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import constants, get_padding_shape_from_config
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.hash_utils import hash_dict_params
from QEfficient.utils.run_utils import ApiRunner

test_configs = [
    # name, max_position_embeddings, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, vocab_size, additional_params
    ("gpt2", 32, 2, 2, 32, 64, 127, {}),
    ("codegen", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    ("falcon", 32, 2, 2, 32, 64, 127, {}),
    ("gptj", 32, 2, 2, 32, 64, 127, {"rotary_dim": 16}),
    ("llama", 32, 2, 2, 32, 64, 127, {"num_key_value_heads": 1}),
    ("mistral", 32, 2, 2, 32, 64, 127, {"num_key_value_heads": 1}),
    ("mixtral", 32, 2, 2, 32, 64, 127, {"num_key_value_heads": 1}),
    # ("mpt", 32, 2, 2, 32, 64, 127, {}), # disabling for HF issues
    ("phi", 32, 2, 2, 32, 64, 127, {}),
    ("phi3", 32, 2, 2, 32, 64, 127, {"pad_token_id": 0}),
    ("qwen2", 32, 2, 2, 32, 64, 127, {"num_key_value_heads": 1}),
    ("starcoder2", 32, 2, 2, 32, 64, 127, {}),
    ("granite", 32, 2, 2, 32, 64, 127, {"num_key_value_heads": 1}),
    ("olmo2", 32, 2, 2, 32, 64, 127, {"num_key_value_heads": 1}),
    ("gpt_oss", 256, 3, 2, 32, 64, 127, {"num_key_value_heads": 1}),
]

test_prefill_only_specialized_models_configs = [
    ("gpt_oss", 32, 2, 2, 32, 64, 127, {"num_key_value_heads": 1}),
]


def get_auto_config_from_test_config(configs):
    auto_configs = [
        AutoConfig.for_model(
            model_name,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            **additional_params,
        )
        for (
            model_name,
            max_position_embeddings,
            num_hidden_layers,
            num_attention_heads,
            hidden_size,
            intermediate_size,
            vocab_size,
            additional_params,
        ) in configs
    ]
    return auto_configs


configs = get_auto_config_from_test_config(test_configs)
config_ids = [x.model_type for x in configs]

prefill_only_configs = get_auto_config_from_test_config(test_prefill_only_specialized_models_configs)
prefill_only_config_ids = [x.model_type for x in prefill_only_configs]

model_kwargs = {"attn_implementation": "eager"}


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
def test_causal_lm_unsupported(cb):
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model("opt"))
    with pytest.warns():
        QEFFAutoModelForCausalLM(model, cb)


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_init(config, cb):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, cb)
    with pytest.raises(TypeError):
        QEFFAutoModelForCausalLM(AutoModel.from_config(config, **model_kwargs), cb)
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_pretrained(config, cb, tmp_path):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    model.save_pretrained(tmp_path)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(tmp_path, cb)
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_export_and_hash(config, cb, tmp_path):
    model_0_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), cb)
    model_0_0.export(tmp_path)
    model_path = tmp_path.with_name(tmp_path.name + "-" + model_0_0.export_hash)
    assert model_path.is_dir()
    assert model_0_0.onnx_path.is_file()
    assert model_0_0.onnx_path.relative_to(model_path).parts == (model_0_0.model_name + ".onnx",)

    # Check if the KV-cache inputs and outputs are created
    onnx_model = onnx.load(model_0_0.onnx_path, load_external_data=False)
    retained_output_names = {
        x.name[: -len("_RetainedState")] for x in onnx_model.graph.output if x.name.endswith("_RetainedState")
    }
    retained_output_names.issubset({x.name for x in onnx_model.graph.input})

    # Check if there is no re-export
    start = perf_counter()
    model_0_0.export(tmp_path)
    end = perf_counter()
    export_time = end - start
    assert export_time < 2.0

    # Check if hashing is happening properly
    model_0_1 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), cb)
    model_0_1.export(tmp_path)
    hash_0_0 = model_0_0.export_hash
    hash_0_1 = model_0_1.export_hash

    assert hash_0_0 == hash_0_1

    cfg1 = copy.deepcopy(config)
    cfg1.num_hidden_layers -= 1
    model_1_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg1, **model_kwargs), cb)
    model_1_0.export(tmp_path)
    hash_1_0 = model_1_0.export_hash
    cfg2 = copy.deepcopy(config)
    cfg2.num_hidden_layers -= 1
    model_1_1 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg2, **model_kwargs), cb)
    model_1_1.export(tmp_path)
    hash_1_1 = model_1_1.export_hash
    assert hash_1_0 == hash_1_1

    assert hash_0_0 != hash_1_0

    if cb:
        model_0_no_cb = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), False)
        model_0_no_cb.export(tmp_path)
        hash_0_no_cb = model_0_no_cb.export_hash
        assert hash_0_0 != hash_0_no_cb


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("subfunc", [False, True], ids=["non-subfunc", "subfunc"])
@pytest.mark.parametrize("prefill_only", [False, True], ids=["pref+decode", "prefill-only"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_hash_creation(config, cb, subfunc, prefill_only, tmp_path):
    if config.model_type == "gpt_oss" and prefill_only:
        pytest.skip(
            "gpt_oss prefill_only mode has different logic to create hash as we have two different ONNX for prefill/decode for this model for disagg serving"
        )
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, cb)
    qeff_model.export(tmp_path, use_onnx_subfunctions=subfunc, prefill_only=prefill_only)
    hash_params = {}
    hash_params["config"] = qeff_model.model.config.to_diff_dict()
    hash_params["peft_config"] = None
    hash_params["applied_transform_names"] = qeff_model._transform_names()
    hash_params["qeff_auto_class"] = qeff_model.__class__.__name__
    hash_params["max_seq_len_cached"] = None
    hash_params["qaic_config"] = None
    hash_params["use_onnx_subfunctions"] = subfunc
    hash_params["onnx_transform_version"] = 1
    hash_params["dynamo"] = False
    if subfunc:
        hash_params["onnx_subfunction_version"] = 3

    # Create parameters separately for hash creation
    bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
    seq_len: int = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
    fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS
    kv_cache_shape = get_padding_shape_from_config(
        qeff_model.model.config, fbs if qeff_model.continuous_batching else bs, seq_len
    )
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
    }
    if len(kv_cache_shape) == 3:  # For GPTBigCode arch the pkv is 3d
        pkv_dynamic_axes = {
            0: "full_batch_size" if qeff_model.continuous_batching else "batch_size",
            1: "ctx_len",
        }
    else:  # pkv is 4d
        pkv_dynamic_axes = {
            0: "full_batch_size" if qeff_model.continuous_batching else "batch_size",
            2: "ctx_len",
        }
    pkv_dynamic_axes = (
        qeff_model.model.get_pkv_dynamic_axes()
        if hasattr(qeff_model.model, "get_pkv_dynamic_axes")
        else pkv_dynamic_axes
    )
    pkv_dynamic_axes = (
        [pkv_dynamic_axes] * qeff_model.model.config.num_hidden_layers
        if isinstance(pkv_dynamic_axes, dict)
        else pkv_dynamic_axes
    )
    output_names = []
    output_names.append("logits")
    onnx_out_name_suffix = "InternalRetainedState" if subfunc else "RetainedState"
    for i in range(qeff_model.num_layers):
        pkv_dynamic_axes[i][0] = "full_batch_size" if qeff_model.continuous_batching else "batch_size"
        for kv in ["key", "value"]:
            dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes[i]
            output_names.append(f"past_{kv}.{i}_{onnx_out_name_suffix}")

    if qeff_model.continuous_batching:
        dynamic_axes["batch_index"] = {0: "batch_size"}

    export_params = {}
    export_params["output_names"] = output_names
    export_params["dynamic_axes"] = dynamic_axes
    export_params["dynamic_shapes"] = None
    hash_params["export_params"] = export_params
    if subfunc:
        hash_params["export_modules_as_functions"] = qeff_model.model.get_submodules_for_export()

    manual_hash = hash_dict_params(hash_params)

    assert manual_hash == qeff_model.export_hash


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", prefill_only_configs, ids=prefill_only_config_ids)
def test_prefill_only_specialized_models(config, cb, tmp_path):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, cb)
    qeff_model.export(tmp_path, prefill_only=True, offload_pt_weights=False)
    assert qeff_model.hash_params.get("chunking") is True
    first_export_hash = qeff_model.export_hash

    qeff_model.export(tmp_path, prefill_only=False, offload_pt_weights=False)
    second_export_hash = qeff_model.export_hash

    assert first_export_hash != second_export_hash


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("QEfficient.utils.export_utils.QEFF_HOME", tmp_path)
    yield tmp_path


@pytest.mark.parametrize("prefill_only", [False, True], ids=["pref+decode", "prefill_only"])
@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_compile(config, cb, prefill_only, tmp_cache):
    if config.model_type == "gpt_oss":
        pytest.skip(
            "gpt_oss prefill_only mode has different logic to create hash as we have two different ONNX for prefill/decode for this model for disagg serving"
        )
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, cb)
    compile_params = {"prefill_seq_len": 8, "ctx_len": 16}
    if prefill_only:
        compile_params["prefill_only"] = True
    if cb:
        compile_params["full_batch_size"] = 32
        compile_params["batch_size"] = 8
    qeff_model.compile(**compile_params)
    model_path = tmp_cache / qeff_model.model_name / (qeff_model.model_name + "-" + qeff_model.export_hash)

    # Check if ONNX is exported properly
    assert model_path.is_dir()
    assert qeff_model.onnx_path.is_file()
    assert qeff_model.onnx_path.relative_to(model_path).parts == (qeff_model.model_name + ".onnx",)

    # Check if QPC is compiled properly
    assert qeff_model.qpc_path.is_dir()
    assert (qeff_model.qpc_path / "programqpc.bin").is_file()
    assert qeff_model.qpc_path.relative_to(tmp_cache).parts[1] == qeff_model.model_name + "-" + qeff_model.export_hash

    # Check if there is no re-compilation
    start = perf_counter()
    qeff_model.compile(**compile_params)
    end = perf_counter()
    compile_time = end - start
    assert compile_time < 2.0
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))


# FIXME: there should be a CB test here
@pytest.mark.parametrize("model_name", ["gpt2"], ids=lambda x: x)
def test_causal_lm_export_with_deprecated_api(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        num_hidden_layers=1,
    )
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    qeff_model = QEFFAutoModelForCausalLM(model, model_name=model_name, pretrained_model_name_or_path=model_name)
    new_api_onnx_model_path = qeff_model.export()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        num_hidden_layers=1,
    )
    qeff_model = QEFFAutoModelForCausalLM(model, model_name=model_name, pretrained_model_name_or_path=model_name)
    _, old_api_onnx_model_path = qualcomm_efficient_converter(
        model_name=model_name, model_kv=qeff_model, tokenizer=tokenizer
    )

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=model.config,
        prompt=Constants.INPUT_STR,
        prompt_len=Constants.PROMPT_LEN,
        ctx_len=Constants.CTX_LEN,
    )

    new_api_ort_tokens = api_runner.run_kv_model_on_ort(new_api_onnx_model_path)
    old_api_ort_tokens = api_runner.run_kv_model_on_ort(old_api_onnx_model_path)

    assert (new_api_ort_tokens == old_api_ort_tokens).all(), (
        "New API output does not match old API output for ONNX export function"
    )
