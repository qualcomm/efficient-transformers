# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import os
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.config_utils import get_first_config_value
from QEfficient.utils.constants import ATTENTION_HEAD_CONFIG_KEYS, KV_HEAD_CONFIG_KEYS, Constants
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils.test_utils import ModelConfig, load_hf_causal_lm_model
from tests.two_phase import model_export_compile_lock

from ..check_model_results import dump_and_compare_results
from ..golden_utils import config_fingerprint, golden_variant_key, resolve_hf_golden

# The QAIC LLM CI stage runs only the on-device leg and compares it against a committed
# golden HF reference. The CPU reference legs (torch-KV "qeff_hf" and ONNXRuntime) are
# retained for debugging / on-demand parity checks but are skipped unless this is set.
_RUN_CPU_REFERENCES = os.environ.get("QEFF_RUN_CPU_REFERENCES") == "1"


def get_custom_n_layers(model_name):
    """
    Function to set number layers of the variuos types of models such as swiftkv models and others
    --------

    :model_name: str

    :return n_layer
    """
    if model_name in {"microsoft/Phi-3-mini-4k-instruct", "neuralmagic/Qwen2-0.5B-Instruct-FP8", "openai/gpt-oss-20b"}:
        return 2
    elif model_name in ModelConfig.SWIFTKV_MODELS:
        return -1
    return 1


def check_kv_repeat_causal_lm_pytorch_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
    n_layer: int = -1,
    config: Optional[AutoConfig] = None,
):
    """
    Validate causal LM flow with repeated KV heads configuration.
    """
    if config is None:
        model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        )
    else:
        model_config = config

    num_attention_heads = get_first_config_value(model_config, ATTENTION_HEAD_CONFIG_KEYS, default=1, cast_int=True)
    num_key_value_heads = get_first_config_value(model_config, KV_HEAD_CONFIG_KEYS, default=None, cast_int=True)
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    if num_attention_heads < 1 or num_key_value_heads < 1:
        raise ValueError(
            f"Invalid heads in config for RepeatKV: "
            f"num_attention_heads={num_attention_heads}, num_key_value_heads={num_key_value_heads}"
        )
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"Invalid heads in config for RepeatKV: num_attention_heads ({num_attention_heads}) "
            f"is not divisible by num_key_value_heads ({num_key_value_heads})."
        )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        manual_cleanup=manual_cleanup,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        n_layer=n_layer,
        config=config,
        qaic_config={"replicate_kv_heads": True},
    )


def check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    num_devices: int = 1,
    continuous_batching: bool = False,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
    n_layer: int = -1,
    num_speculative_tokens: Optional[int] = None,
    prefill_only: Optional[bool] = None,
    config: Optional[AutoConfig] = None,
    pytorch_hf_tokens: Optional[list] = None,
    qaic_config: Optional[dict] = None,
    retain_full_kv: Optional[bool] = None,
    compare_results: bool = False,
    compile_only: bool = False,
    mdp_num_partitions: Optional[int] = None,
    mdp_strategy: Optional[str] = None,
    use_onnx_subfunctions: bool = False,
    torch_dtype: Optional[torch.dtype] = torch.float32,
    generation_len: Optional[int] = None,
    comp_ctx_lengths_prefill: Optional[list[int]] = None,
    comp_ctx_lengths_decode: Optional[list[int]] = None,
    kv_cache_batch_size: Optional[int] = None,
    num_cores: int = 16,
    aic_hw_version: Optional[str] = "ai100",
    compile_options: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    prompts: Optional[list[str]] = None,
    skip_onnxruntime: bool = False,
):
    torch.manual_seed(42)
    replace_transformers_quantizers()
    model_hf = load_hf_causal_lm_model(model_name, num_hidden_layers=n_layer, config=config, torch_dtype=torch_dtype)
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=tokenizer_name or model_name)
    config = model_hf.config
    # `prompts` lets a caller validate generation against several distinct prompts without
    # paying for an extra export/compile: the QPC is keyed on batch_size/full_batch_size, and
    # both stay fixed because the base prompts are tiled to exactly full_batch_size entries.
    # Only continuous batching can absorb extra prompts this way -- the non-CB QPC is compiled
    # with batch_size=1, so a multi-prompt override there would not match the compiled shape.
    base_prompts = list(prompts) if prompts else Constants.INPUT_STR
    if len(base_prompts) > len(Constants.INPUT_STR) and not continuous_batching:
        raise ValueError("Multiple prompts are only supported with continuous_batching=True.")
    batch_size = len(Constants.INPUT_STR)
    full_batch_size = kv_cache_batch_size or 4
    prompts = (
        [base_prompts[i % len(base_prompts)] for i in range(full_batch_size)] if continuous_batching else base_prompts
    )
    gen_len = generation_len or 24
    is_tlm = False if num_speculative_tokens is None else True
    pytorch_hf_tokens = None
    pytorch_kv_tokens = None
    ort_tokens = None
    reference_ctx_len = prompt_len + generation_len if generation_len is not None else ctx_len

    qeff_model = QEFFAutoModelForCausalLM(
        copy.deepcopy(model_hf),
        is_tlm=is_tlm,
        pretrained_model_name_or_path=model_name,
        continuous_batching=continuous_batching,
        qaic_config=qaic_config,
    )
    qeff_model.transform(
        ctx_len=ctx_len,
        seq_len=prompt_len,
        batch_size=full_batch_size if continuous_batching else batch_size,
        num_devices=num_devices,
        qaic_config=qaic_config,
    )
    # Build the reference runner from the *transformed* config: qaic_config transforms such as
    # KV-head replication / blocking mutate qeff_model.config.num_key_value_heads, and the
    # reference KV-cache buffers must be sized to match, or the PyTorch-KV leg scatters into a
    # wrongly-shaped cache. (model_hf.config still holds the pre-transform head count.)
    api_runner = ApiRunner(
        batch_size,
        tokenizer,
        qeff_model.config,
        prompts,
        prompt_len,
        reference_ctx_len,
        full_batch_size if continuous_batching else None,
        dtype=torch_dtype,
    )
    if _RUN_CPU_REFERENCES and not compile_only and continuous_batching is False:
        pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    if (
        not compile_only
        and model_name not in ModelConfig.SWIFTKV_MODELS
        and model_name not in ModelConfig.EXTERNAL_MODELS
    ):
        # The HF PyTorch reference is a pure function of the model + effective config, so it
        # is served from a committed golden and generated once on first use. It is unaffected
        # by qaic_config (blocking / CCL / speculative / KV replication), which is why a single
        # golden is reused across every QAIC variant of this model.
        config_fp = config_fingerprint(qeff_model.config)
        variant_key = golden_variant_key(
            continuous_batching=continuous_batching,
            torch_dtype=torch_dtype,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            generation_len=generation_len,
            full_batch_size=full_batch_size if continuous_batching else None,
            prompts=prompts,
            config_fp=config_fp,
        )

        def _compute_hf_tokens():
            if continuous_batching:
                return np.vstack(api_runner.run_hf_model_on_pytorch_CB(model_hf))
            return api_runner.run_hf_model_on_pytorch(model_hf)

        pytorch_hf_tokens = resolve_hf_golden(
            family="causal_lm",
            model_name=model_name,
            variant_key=variant_key,
            params={
                "config_fp": config_fp,
                "continuous_batching": continuous_batching,
                "dtype": str(torch_dtype),
                "prompt_len": prompt_len,
                "ctx_len": ctx_len,
                "generation_len": generation_len,
                "full_batch_size": full_batch_size if continuous_batching else None,
                "prompts": prompts,
            },
            compute_fn=_compute_hf_tokens,
        )

    with model_export_compile_lock(model_name):
        onnx_model_path = qeff_model.export(use_onnx_subfunctions=use_onnx_subfunctions)
        if _RUN_CPU_REFERENCES and not compile_only and continuous_batching is False and not skip_onnxruntime:
            ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=is_tlm)
            gen_len = ort_tokens.shape[-1]
        elif continuous_batching is False and pytorch_hf_tokens is not None:
            # Golden path (non-CB): ORT is not run, so take the generation length from the HF
            # reference (HF and ORT emit the same token count). CB keeps its own gen_len.
            gen_len = pytorch_hf_tokens.shape[-1]

        if pytorch_hf_tokens is not None and ort_tokens is not None:
            assert (pytorch_hf_tokens == ort_tokens).all(), (
                "Tokens don't match for HF PyTorch model output and ONNXRT output."
            )

        if pytorch_kv_tokens is not None and ort_tokens is not None:
            assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

        compiler_options = {}

        mdp_compile_kwargs = {}
        if mdp_num_partitions is not None:
            mdp_compile_kwargs["mdp_num_partitions"] = mdp_num_partitions
        if mdp_strategy is not None:
            mdp_compile_kwargs["mdp_strategy"] = mdp_strategy

        qpc_path = qeff_model.compile(
            prefill_seq_len=prompt_len,
            ctx_len=ctx_len,
            num_devices=num_devices,
            mxfp6_matmul=False,
            aic_enable_depth_first=False,
            num_speculative_tokens=num_speculative_tokens,
            retain_full_kv=retain_full_kv,
            prefill_only=prefill_only,
            batch_size=batch_size if continuous_batching else 1,
            full_batch_size=full_batch_size if continuous_batching else None,
            use_onnx_subfunctions=use_onnx_subfunctions,
            onnx_path=onnx_model_path,
            comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
            comp_ctx_lengths_decode=comp_ctx_lengths_decode,
            kv_cache_batch_size=kv_cache_batch_size,
            num_cores=num_cores,
            aic_hw_version=aic_hw_version,
            **compiler_options,
            **mdp_compile_kwargs,
            **(compile_options or {}),
        )
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))

    if compile_only:
        manual_cleanup(onnx_model_path)
        return

    # Generate
    generate_kwargs = {}
    if generation_len is not None:
        generate_kwargs["generation_len"] = generation_len
    exec_info = qeff_model.generate(tokenizer, prompts=prompts, **generate_kwargs)

    if continuous_batching:
        cloud_ai_100_tokens = exec_info.generated_ids
        if cloud_ai_100_tokens is not None and ort_tokens is not None:
            assert all(
                [
                    all(ort_token[:gen_len] == cloud_token[:gen_len])
                    for ort_token, cloud_token in zip(ort_tokens, cloud_ai_100_tokens)
                ]
            ), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
        if pytorch_hf_tokens is not None and cloud_ai_100_tokens is not None:
            assert all(
                [
                    all(pt_token[:gen_len] == cloud_token[:gen_len])
                    for pt_token, cloud_token in zip(pytorch_hf_tokens, cloud_ai_100_tokens)
                ]
            ), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
    else:
        cloud_ai_100_tokens = exec_info.generated_ids[0][:, :gen_len]
        if prefill_only:
            if ort_tokens is None:
                raise RuntimeError(
                    "prefill_only comparison requires the ORT reference leg; rerun with QEFF_RUN_CPU_REFERENCES=1."
                )
            assert (ort_tokens[0][0] == cloud_ai_100_tokens[0][0]).all(), (
                "prefill run output tokens don't match for ONNXRT output and Cloud AI 100 output."
            )
        else:
            # Prefer the live ORT reference when it was run; otherwise fall back to the
            # committed golden HF tokens. The golden is shape (gen_len,) and broadcasts
            # against the (1, gen_len) cloud output exactly as the HF<->ORT assert did.
            reference_tokens = ort_tokens if ort_tokens is not None else pytorch_hf_tokens
            reference_name = "ONNXRT" if ort_tokens is not None else "golden HF PyTorch"
            if reference_tokens is None:
                raise RuntimeError(
                    f"No reference tokens available for {model_name}: no golden HF output and "
                    "CPU reference legs are disabled. Rerun with QEFF_RUN_CPU_REFERENCES=1."
                )
            assert (reference_tokens == cloud_ai_100_tokens).all(), (
                f"Tokens don't match for {reference_name} output and Cloud AI 100 output."
            )

    manual_cleanup(onnx_model_path)  # Clean up the model files after the tests are done.
    if compare_results is False:
        return
    # Compare results for full model only.
    compile_params = {
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "num_devices": num_devices,
        "mxfp6_matmul": False,
        "aic_enable_depth_first": False,
        "num_speculative_tokens": num_speculative_tokens,
        "retain_full_kv": retain_full_kv,
        "prefill_only": prefill_only,
        "batch_size": batch_size if continuous_batching else 1,
        "full_batch_size": full_batch_size if continuous_batching else None,
        "compiler_options": compiler_options,
        "compile_only": compile_only,
        "num_cores": num_cores,
        "compile_options": compile_options,
        "mdp_num_partitions": mdp_num_partitions,
        "mdp_strategy": mdp_strategy,
        "use_onnx_subfunctions": use_onnx_subfunctions,
    }
    assert dump_and_compare_results(
        model_name,
        compile_params,
        "causal_lm_model_results.json",
        cloud_ai_100_tokens,
        exec_info,
        pytorch_hf_tokens,
        pytorch_kv_tokens,
        ort_tokens,
    )
