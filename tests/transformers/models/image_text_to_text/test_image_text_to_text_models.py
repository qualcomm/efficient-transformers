# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import json
import os
from typing import List, Optional

import numpy as np
import onnx
import pytest
import requests
import torch
from requests.adapters import HTTPAdapter
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
)
from transformers.processing_utils import ProcessorMixin
from urllib3.util.retry import Retry

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.molmo_point.modeling_molmo_point import _reset_molmo_point_patch_rope
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import (
    InternProcessor,
    ModelConfig,
    load_vlm_model,
    load_vlm_model_from_config,
    set_num_layers_vlm,
)
from tests.two_phase import model_export_compile_lock, resolve_two_phase_cleanup
from tests.utils.image_utils import load_test_image
from tests.utils.load_kimi_utils import (
    get_kimi_k25_test_config,
    is_kimi_k25,
    load_kimi_k25_model_from_config,
    run_kimi_k25_hf_model_on_pytorch,
)

from ..check_model_results import dump_and_compare_results
from ..golden_utils import config_to_dict_fingerprint, resolve_hf_golden, vlm_golden_variant_key

_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_models"]
test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}
test_mm_moe_models = [model["model_name"] for model in multimodal_models if "moe" in model.get("model_type", "")]
test_mm_blocking_models = [model["model_name"] for model in multimodal_models if model.get("supports_blocking")]

NEW_GENERATION_TOKENS = 10

MOLMO_POINT_VISION_STATES = (
    "vision_embeds",
    "vision_embeds_vit_features",
    "vision_embeds_vit_mask",
    "vision_embeds_subpatch_k",
)
MOLMO_POINT_POINT_STATES = (
    "vision_embeds_patch_k",
    "vision_embeds_patch_mask",
    "vision_embeds_image_pos_ids",
    "vision_embeds_last_patch_id",
)


def _load_vlm_processor(model_name: str):
    """Load a VLM processor, bridging old ProcessorMixin optional-kwarg handling.

    MolmoPoint's remote processor declares image/video formatting options as
    ``optional_attributes``.  Transformers 5.5.4 forwards those options to
    ``ProcessorMixin.__init__`` but that version only accepts modality
    attributes, so processor construction fails before the model is exercised.
    Keep the compatibility shim scoped to this load and preserve the options on
    the processor instance for the remote implementation.
    """
    try:
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    except TypeError as exc:
        if model_name != "allenai/MolmoPoint-8B" or "Unexpected keyword argument" not in str(exc):
            raise

    original_init = ProcessorMixin.__dict__["__init__"]

    def compatible_init(self, *args, **kwargs):
        optional = set(getattr(self, "optional_attributes", ()))
        deferred = {key: kwargs.pop(key) for key in list(kwargs) if key in optional}
        original_init(self, *args, **kwargs)
        for key, value in deferred.items():
            setattr(self, key, value)

    ProcessorMixin.__init__ = compatible_init
    try:
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    finally:
        ProcessorMixin.__init__ = original_init


def _patch_molmo_point_hf_generation(model):
    """Supply the cache positions expected by the Hub model on Transformers 5.5.4."""
    if getattr(getattr(model, "config", None), "model_type", None) != "molmo_point":
        return

    original_prepare = model.prepare_inputs_for_generation

    def compatible_prepare(input_ids, *args, **kwargs):
        if kwargs.get("cache_position") is None:
            source = input_ids if input_ids is not None else kwargs.get("inputs_embeds")
            past_key_values = kwargs.get("past_key_values")
            if source is None:
                raise ValueError("MolmoPoint generation requires input_ids or inputs_embeds")
            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values.get_seq_length()
            kwargs["cache_position"] = torch.arange(
                past_length, past_length + source.shape[1], device=source.device
            )
        return original_prepare(input_ids, *args, **kwargs)

    model.prepare_inputs_for_generation = compatible_prepare


def _assert_runtime_token_parity(reference_tokens, qpc_tokens, parity_issue=None):
    """Compare HF and QAIC tokens, xfail only a configured numerical parity mismatch."""
    if (reference_tokens == qpc_tokens).all():
        return
    if parity_issue:
        pytest.xfail(parity_issue)
    pytest.fail("Tokens don't match for pytorch HF output and QPC output")


def _molmo_point_dual_inputs(model, inputs, prompt_len, ctx_len):
    """Build the exact padded dual-QPC inputs used by the MolmoPoint runtime."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_ids_length = input_ids.shape[1]
    padded_len = -(input_ids_length // -prompt_len) * prompt_len
    input_ids = torch.nn.functional.pad(input_ids, (0, padded_len - input_ids_length), value=1)
    attention_mask = torch.nn.functional.pad(attention_mask, (0, padded_len - input_ids_length), value=0)
    position_ids = torch.where(
        attention_mask.to(torch.bool),
        torch.arange(padded_len, dtype=torch.int64).view(1, padded_len),
        -1,
    )
    token_type_ids = torch.nn.functional.pad(
        inputs["token_type_ids"], (0, padded_len - input_ids_length), value=0
    )

    pixel_values, image_token_pooling = model.model.merge_visual_inputs(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_token_pooling=inputs.get("image_token_pooling"),
        image_grids=inputs.get("image_grids"),
        image_num_crops=inputs.get("image_num_crops"),
        pixel_values_videos=inputs.get("pixel_values_videos"),
        video_token_pooling=inputs.get("video_token_pooling"),
        video_grids=inputs.get("video_grids"),
    )
    dummy_inputs = model.get_dummy_inputs(
        kv_offload=True,
        prefill_seq_len=padded_len,
        ctx_len=ctx_len,
        num_crops=pixel_values.shape[1],
        num_patches=pixel_values.shape[2],
        pixels_per_patch=pixel_values.shape[3],
        num_image_tokens=image_token_pooling.shape[1],
        pool_dim=image_token_pooling.shape[2],
    )
    lang_inputs = dummy_inputs["lang"]
    lang_inputs.update(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
        }
    )
    vision_inputs = {
        "pixel_values": pixel_values,
        "image_token_pooling": image_token_pooling,
    }
    return vision_inputs, lang_inputs


def _molmo_point_flatten_torch_outputs(model, outputs):
    output_names = model.get_output_names(kv_offload=True)["lang"]
    values = list(outputs[:-1])
    for key_cache, value_cache in outputs[-1]:
        values.extend((key_cache, value_cache))
    assert len(output_names) == len(values)
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in zip(output_names, values, strict=True)
    }


def _molmo_point_next_inputs(previous_inputs, outputs, num_hidden_layers):
    next_token = outputs["logits"].argmax(-1).reshape(1, 1)
    next_inputs = {
        "input_ids": next_token,
        "position_ids": np.max(previous_inputs["position_ids"], axis=1, keepdims=True) + 1,
        "token_type_ids": np.zeros_like(next_token, dtype=previous_inputs["token_type_ids"].dtype),
    }
    for state_name in MOLMO_POINT_VISION_STATES + MOLMO_POINT_POINT_STATES:
        next_inputs[state_name] = outputs[f"{state_name}_RetainedState"]
    next_inputs["past_key_values"] = [
        (
            outputs[f"past_key.{layer_idx}_RetainedState"],
            outputs[f"past_value.{layer_idx}_RetainedState"],
        )
        for layer_idx in range(num_hidden_layers)
    ]
    return next_inputs


@torch.no_grad()
def _run_molmo_point_qeff_pytorch(model, inputs, prompt_len, ctx_len, generation_len):
    """Run dual-QPC wrappers on CPU and retain every recurrent output for ORT comparison."""
    vision_inputs, lang_inputs = _molmo_point_dual_inputs(model, inputs, prompt_len, ctx_len)
    vision_model = model.get_qeff_vision_encoder()
    language_model = model.get_qeff_language_decoder()
    vision_values = vision_model(**vision_inputs)
    vision_outputs = {
        name: value.detach().cpu().numpy().copy()
        for name, value in zip(MOLMO_POINT_VISION_STATES, vision_values, strict=True)
    }
    lang_inputs.update(dict(zip(MOLMO_POINT_VISION_STATES, vision_values, strict=True)))

    generated_ids = []
    step_outputs = []
    for _ in range(generation_len):
        raw_outputs = language_model(**lang_inputs)
        step_outputs.append(_molmo_point_flatten_torch_outputs(model, raw_outputs))
        next_token = raw_outputs[0].argmax(-1).reshape(1, 1)
        generated_ids.append(next_token.detach().cpu().numpy())
        lang_inputs = {
            "input_ids": next_token,
            "position_ids": lang_inputs["position_ids"].max(1, keepdim=True).values + 1,
            "token_type_ids": torch.zeros_like(next_token, dtype=lang_inputs["token_type_ids"].dtype),
            "past_key_values": raw_outputs[-1],
        }
        lang_inputs.update(dict(zip(MOLMO_POINT_VISION_STATES, raw_outputs[1:5], strict=True)))
        lang_inputs.update(dict(zip(MOLMO_POINT_POINT_STATES, raw_outputs[5:9], strict=True)))
    return np.concatenate(generated_ids, axis=1), vision_outputs, step_outputs


def _assert_molmo_point_outputs_close(reference, actual, boundary, base_vocab_size):
    assert reference.keys() == actual.keys(), f"{boundary}: output names differ"
    for name in reference:
        reference_value = reference[name]
        actual_value = actual[name]
        assert reference_value.shape == actual_value.shape, f"{boundary}/{name}: shape mismatch"
        if name == "logits":
            reference_value = reference_value[..., :base_vocab_size]
            actual_value = actual_value[..., :base_vocab_size]
        if np.issubdtype(reference_value.dtype, np.floating):
            try:
                np.testing.assert_allclose(
                    actual_value,
                    reference_value,
                    rtol=1e-3,
                    atol=1e-4,
                    err_msg=f"{boundary}/{name}",
                )
            except AssertionError:
                delta = np.abs(actual_value.astype(np.float32) - reference_value.astype(np.float32))
                max_index = np.unravel_index(delta.argmax(), delta.shape)
                print(
                    f"[molmo-point-ort] {boundary}/{name}: max_index={max_index} "
                    f"reference={reference_value[max_index]} actual={actual_value[max_index]} "
                    f"reference_nonzero={np.count_nonzero(reference_value)} "
                    f"actual_nonzero={np.count_nonzero(actual_value)}"
                )
                raise
        else:
            np.testing.assert_array_equal(actual_value, reference_value, err_msg=f"{boundary}/{name}")


def _run_molmo_point_ort(
    api_runner,
    model,
    inputs,
    onnx_paths,
    prompt_len,
    ctx_len,
    generation_len,
    pytorch_vision_outputs,
    pytorch_step_outputs,
):
    """Run both ONNX graphs and compare all vision, point, and KV retained states."""
    vision_inputs, lang_inputs = _molmo_point_dual_inputs(model, inputs, prompt_len, ctx_len)
    _, vision_session = api_runner.setup_ort_session(onnx_paths[0])
    _, language_session = api_runner.setup_ort_session(onnx_paths[1])
    vision_inputs = {name: value.detach().cpu().numpy() for name, value in vision_inputs.items()}
    lang_inputs = {
        name: value.detach().cpu().numpy()
        for name, value in lang_inputs.items()
        if name != "past_key_values"
    }
    for layer_idx, (key_cache, value_cache) in enumerate(
        _molmo_point_dual_inputs(model, inputs, prompt_len, ctx_len)[1]["past_key_values"]
    ):
        lang_inputs[f"past_key.{layer_idx}"] = key_cache.detach().cpu().numpy()
        lang_inputs[f"past_value.{layer_idx}"] = value_cache.detach().cpu().numpy()

    base_vocab_size = model.config.text_config.vocab_size + model.config.text_config.additional_vocab_size
    vision_outputs = api_runner.run_ort_session(vision_inputs, vision_session)
    _assert_molmo_point_outputs_close(
        pytorch_vision_outputs, vision_outputs, "QEff-PyTorch/ORT vision", base_vocab_size
    )
    lang_inputs.update(vision_outputs)

    generated_ids = []
    step_outputs = []
    num_hidden_layers = model.config.text_config.num_hidden_layers
    for step in range(generation_len):
        outputs = api_runner.run_ort_session(lang_inputs, language_session)
        step_outputs.append({name: value.copy() for name, value in outputs.items()})
        _assert_molmo_point_outputs_close(
            pytorch_step_outputs[step], outputs, f"QEff-PyTorch/ORT language step {step}", base_vocab_size
        )
        next_inputs = _molmo_point_next_inputs(lang_inputs, outputs, num_hidden_layers)
        generated_ids.append(next_inputs["input_ids"])
        lang_inputs = {
            name: value
            for name, value in next_inputs.items()
            if name != "past_key_values"
        }
        for layer_idx, (key_cache, value_cache) in enumerate(next_inputs["past_key_values"]):
            lang_inputs[f"past_key.{layer_idx}"] = key_cache
            lang_inputs[f"past_value.{layer_idx}"] = value_cache
    return np.concatenate(generated_ids, axis=1), vision_outputs, step_outputs


def _cast_molmo_point_qpc_input(session, name, value):
    binding = session.bindings[session.binding_index_map[name]]
    dtype = session.aic_to_np_dtype_mapping[binding.type]
    return np.asarray(value).astype(dtype, copy=False)


def _report_molmo_point_qpc_delta(reference, actual, boundary, base_vocab_size):
    """Report the raw ORT/QPC delta without hiding an argmax mismatch."""
    assert reference.keys() <= actual.keys(), f"{boundary}: QPC outputs are missing {reference.keys() - actual.keys()}"
    for name, reference_value in reference.items():
        actual_value = actual[name]
        assert reference_value.shape == actual_value.shape, f"{boundary}/{name}: shape mismatch"
        if name == "logits":
            reference_value = reference_value[..., :base_vocab_size]
            actual_value = actual_value[..., :base_vocab_size]
        if not np.issubdtype(reference_value.dtype, np.floating):
            np.testing.assert_array_equal(actual_value, reference_value, err_msg=f"{boundary}/{name}")
            continue
        if name == "vision_embeds_patch_k_RetainedState":
            live_slots = reference["vision_embeds_patch_mask_RetainedState"] > 0
            reference_value = reference_value[live_slots]
            actual_value = actual_value[live_slots]
        masked = reference_value <= -60000
        assert np.all(actual_value[masked] <= -60000), f"{boundary}/{name}: masked values became live"
        live_actual = actual_value[~masked]
        live_reference = reference_value[~masked]
        if not np.isfinite(live_actual).all():
            invalid = np.argwhere(~np.isfinite(actual_value))[:10].tolist()
            pytest.fail(f"{boundary}/{name}: QPC produced non-finite live values at {invalid}")
        delta = np.abs(live_actual.astype(np.float32) - live_reference.astype(np.float32))
        actual_norm = np.linalg.norm(live_actual.astype(np.float64))
        reference_norm = np.linalg.norm(live_reference.astype(np.float64))
        if actual_norm == 0 or reference_norm == 0:
            cosine = 1.0 if np.array_equal(live_actual, live_reference) else 0.0
        else:
            cosine = float(
                np.dot(live_actual.astype(np.float64), live_reference.astype(np.float64))
                / (actual_norm * reference_norm)
            )
        print(
            f"[molmo-point-qpc] {boundary}/{name}: "
            f"max_abs={delta.max():.7g} mean_abs={delta.mean():.7g} cosine={cosine:.7f}"
        )
        assert cosine >= 0.995, f"{boundary}/{name}: cosine {cosine:.7f} is below 0.995"
        assert delta.mean() <= 0.1, f"{boundary}/{name}: mean absolute error exceeds 0.1"


def _run_molmo_point_qpc(
    model,
    inputs,
    qpc_paths,
    device_ids,
    prompt_len,
    ctx_len,
    generation_len,
    ort_vision_outputs,
    ort_step_outputs,
):
    """Run both MolmoPoint QPCs while exposing every retained state for comparison."""
    vision_inputs, lang_inputs = _molmo_point_dual_inputs(model, inputs, prompt_len, ctx_len)
    vision_session = QAICInferenceSession(qpc_paths[0], device_ids=device_ids)
    language_session = QAICInferenceSession(qpc_paths[1], device_ids=device_ids, activate=False)
    base_vocab_size = model.config.text_config.vocab_size + model.config.text_config.additional_vocab_size
    try:
        vision_inputs = {
            name: _cast_molmo_point_qpc_input(vision_session, name, value.detach().cpu().numpy())
            for name, value in vision_inputs.items()
        }
        vision_outputs = vision_session.run(vision_inputs)
        _report_molmo_point_qpc_delta(
            ort_vision_outputs, vision_outputs, "ORT/QPC vision", base_vocab_size
        )
        vision_session.deactivate()
        language_session.activate()

        qpc_inputs = {
            name: _cast_molmo_point_qpc_input(language_session, name, value.detach().cpu().numpy())
            for name, value in lang_inputs.items()
            if name in {"input_ids", "position_ids", "token_type_ids"}
        }
        qpc_inputs.update(
            {
                name: _cast_molmo_point_qpc_input(language_session, name, value)
                for name, value in vision_outputs.items()
            }
        )
        for name in language_session.input_names:
            if name in qpc_inputs:
                continue
            binding = language_session.bindings[language_session.binding_index_map[name]]
            qpc_inputs[name] = np.zeros(
                tuple(binding.dims), dtype=language_session.aic_to_np_dtype_mapping[binding.type]
            )

        generated_ids = []
        num_hidden_layers = model.config.text_config.num_hidden_layers
        for step in range(generation_len):
            outputs = language_session.run(qpc_inputs)
            _report_molmo_point_qpc_delta(
                ort_step_outputs[step], outputs, f"ORT/QPC language step {step}", base_vocab_size
            )
            next_inputs = _molmo_point_next_inputs(qpc_inputs, outputs, num_hidden_layers)
            generated_ids.append(next_inputs["input_ids"])
            qpc_inputs = {
                "input_ids": _cast_molmo_point_qpc_input(
                    language_session, "input_ids", next_inputs["input_ids"]
                ),
                "position_ids": _cast_molmo_point_qpc_input(
                    language_session, "position_ids", next_inputs["position_ids"]
                ),
                "token_type_ids": _cast_molmo_point_qpc_input(
                    language_session, "token_type_ids", next_inputs["token_type_ids"]
                ),
            }
            for state_name in MOLMO_POINT_VISION_STATES + MOLMO_POINT_POINT_STATES:
                qpc_inputs[state_name] = _cast_molmo_point_qpc_input(
                    language_session, state_name, next_inputs[state_name]
                )
            for layer_idx, (key_cache, value_cache) in enumerate(next_inputs["past_key_values"]):
                qpc_inputs[f"past_key.{layer_idx}"] = _cast_molmo_point_qpc_input(
                    language_session, f"past_key.{layer_idx}", key_cache
                )
                qpc_inputs[f"past_value.{layer_idx}"] = _cast_molmo_point_qpc_input(
                    language_session, f"past_value.{layer_idx}", value_cache
                )
        return np.concatenate(generated_ids, axis=1)
    finally:
        if vision_session.is_active:
            vision_session.deactivate()
        if language_session.is_active:
            language_session.deactivate()


def _resolve_vlm_hf_golden(
    model_name: str,
    config: AutoConfig,
    query: str,
    img_url: str,
    torch_dtype: torch.dtype,
    max_gen_len: int,
    compile_only: bool,
    compute_fn,
):
    """Resolve the HF PyTorch reference tokens for one VLM variant from the committed golden.

    The HF leg is a pure function of the model + effective config + fixed image/prompt pair
    (from ``image_text_model_configs.json``), independent of ``kv_offload``/``qaic_config``
    (those only steer the QEff/on-device leg), so it is generated once per variant and reused
    across every other knob. ``compile_only`` runs never reach the token comparison, so the
    (expensive) HF generate call is skipped for them entirely rather than golden-cached.
    """
    if compile_only:
        return None
    variant_key = vlm_golden_variant_key(
        torch_dtype=torch_dtype,
        prompt_text=query,
        image_url=img_url,
        generation_len=max_gen_len,
        config_fp=config_to_dict_fingerprint(config),
    )
    return resolve_hf_golden(
        family="image_text_to_text",
        model_name=model_name,
        variant_key=variant_key,
        params={
            "prompt": query,
            "image_url": img_url,
            "dtype": str(torch_dtype),
            "generation_len": max_gen_len,
        },
        compute_fn=compute_fn,
    )


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    num_hidden_layers: Optional[int] = -1,
    kv_offload: Optional[bool] = False,
    num_devices: Optional[int] = 1,
    config: Optional[AutoConfig] = None,
    qaic_config: Optional[dict] = None,
    test_kv_replicate: Optional[bool] = None,
    torch_dtype: Optional[torch.dtype] = torch.float32,
    compare_results: Optional[bool] = False,
    compile_only: bool = False,
    mdp_num_partitions: Optional[int] = None,
    mdp_strategy: Optional[str] = None,
    use_onnx_subfunctions: bool = False,
    comp_ctx_lengths_prefill: Optional[List[int]] = None,
    comp_ctx_lengths_decode: Optional[List[int]] = None,
    ccl_enabled: bool = False,
    known_runtime_parity_issue: Optional[str] = None,
):
    # Two-phase compile/execute split: suppress per-test cleanup in both phases (model variants
    # share a content-addressed export dir, so one variant's rmtree would destroy its siblings'
    # warm QPCs) and force compile-only in the warm phase. A no-op in normal runs.
    manual_cleanup, compile_only = resolve_two_phase_cleanup(manual_cleanup, compile_only)
    prompt_len = model_config_dict[model_name]["prompt_len"]
    ctx_len = model_config_dict[model_name]["ctx_len"]
    img_size = model_config_dict[model_name].get("img_size")
    img_url = model_config_dict[model_name]["img_url"]
    query = model_config_dict[model_name]["text_prompt"]
    batch_size = model_config_dict[model_name]["batch_size"]

    max_gen_len = model_config_dict[model_name].get("generation_len", NEW_GENERATION_TOKENS)
    pytorch_hf_tokens = None
    pytorch_kv_tokens = None
    ort_tokens = None
    n_layer = num_hidden_layers
    qaic_config = copy.deepcopy(qaic_config) if qaic_config is not None else None

    # CCL is opted into through the model's QAIC configuration. Merge it with any
    # caller-provided options so KV-head replication and blocking remain intact.
    if ccl_enabled or comp_ctx_lengths_prefill or comp_ctx_lengths_decode:
        qaic_config = qaic_config or {}
        qaic_config["ccl_enabled"] = True

    if is_kimi_k25(model_name):
        if config is None:
            # Build the reduced Kimi architecture directly with random weights. Loading a
            # checkpoint subset first would snapshot the complete ~595 GB model repository.
            config = get_kimi_k25_test_config(model_name, model_config_dict)
        model_hf, tokenizer, processor = load_kimi_k25_model_from_config(config)
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=model_hf.config,
            qaic_config=qaic_config,
            torch_dtype=torch_dtype,
        )
    elif config is None:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=model_name not in ModelConfig.MOLMO_MODELS
        )
        config = set_num_layers_vlm(config, n_layer=n_layer)
        if test_kv_replicate:
            qaic_config = qaic_config or {}
            qaic_config["replicate_kv_heads"] = True
        if hasattr(config, "model_type") and config.model_type in ["gemma3"]:
            config.text_config._sliding_window_pattern = 2
            config.text_config.layer_types = ["sliding_attention", "full_attention"]
        if hasattr(config, "model_type") and config.model_type in ["gemma4"]:
            config.text_config.num_kv_shared_layers = 0
            config.text_config.num_hidden_layers = 1
            config.vision_config.num_hidden_layers = 1
            config.text_config.layer_types = ["sliding_attention"]
            # Keep the sliding window below ctx_len (512); the hub value (1024) exceeds it, a
            # degenerate setup where the window never slides. See the CB test for the compile crash
            # this avoids at larger decode batches.
            config.text_config.sliding_window = 256
        if hasattr(config, "model_type") and config.model_type in [
            "qwen3_vl",
            "qwen3_vl_moe",
        ]:
            config.vision_config.depth = 9
            config.text_config.num_hidden_layers = 1
            config.vision_config.deepstack_visual_indexes = [8]
        if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
            config._attn_implementation = "eager"
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
                qaic_config=qaic_config,
                torch_dtype=torch_dtype,
                ignore_mismatched_sizes=True,
            )
        else:
            if model_name == "allenai/MolmoPoint-8B":
                config._attn_implementation = "eager"
            model_hf = load_vlm_model(config)
            if model_name == "allenai/MolmoPoint-8B":
                _reset_molmo_point_patch_rope(model_hf)
                qeff_model = QEFFAutoModelForImageTextToText(
                    copy.deepcopy(model_hf),
                    kv_offload=kv_offload,
                    config=model_hf.config,
                    qaic_config=qaic_config,
                    torch_dtype=torch.float32,
                    ignore_mismatched_sizes=True,
                )
            else:
                qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
                    model_name,
                    kv_offload=kv_offload,
                    config=config,
                    qaic_config=qaic_config,
                    torch_dtype=torch_dtype,
                    ignore_mismatched_sizes=True,
                )
    else:
        if test_kv_replicate:
            qaic_config = qaic_config or {}
            qaic_config["replicate_kv_heads"] = True
        model_hf = load_vlm_model_from_config(config)
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=model_hf.config,
            qaic_config=qaic_config,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True,
        )
    aic_hw_version = "ai200" if torch_dtype == torch.bfloat16 else "ai100"
    _patch_molmo_point_hf_generation(model_hf)
    compile_kwargs = {
        "num_devices": num_devices,
        "num_cores": 4 if aic_hw_version == "ai200" else 16,
        "aic_hw_version": aic_hw_version,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "mxfp6": False,
        "qaic_config": qaic_config,
        "use_onnx_subfunctions": use_onnx_subfunctions,
        "split-model-io": False if aic_hw_version == "ai200" else True,
    }

    # Left as None when CCL is auto-generated: compile() derives both lists from ctx_len.
    if comp_ctx_lengths_prefill is not None:
        compile_kwargs["comp_ctx_lengths_prefill"] = comp_ctx_lengths_prefill
    if comp_ctx_lengths_decode is not None:
        compile_kwargs["comp_ctx_lengths_decode"] = comp_ctx_lengths_decode

    mdp_compile_kwargs = {}
    if mdp_num_partitions is not None:
        mdp_compile_kwargs["mdp_num_partitions"] = mdp_num_partitions
    if mdp_strategy is not None:
        mdp_compile_kwargs["mdp_strategy"] = mdp_strategy
    if model_name == "tiny-random/gemma-4-dense" or model_name == "tiny-random/gemma-4-moe":
        compile_kwargs["node_precision_info"] = True
    if model_name in ModelConfig.INTERNVL_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)
        prompt = [query]
        img_url_list = [img_url]
        pixel_values = []
        num_patches_list = []
        questions = []
        for i in range(len(prompt)):
            image = load_test_image(img_url_list[i], size=(448, 448), session=_session)
            pixel_value = processor.load_image(image, max_num=12)
            num_patches_list.append(pixel_value.shape[0])
            pixel_values.append(pixel_value)
            question = "<image>\n" + prompt[i]
            questions.append(question)

        pixel_values = torch.cat(pixel_values, dim=0)
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = processor(pixel_values, questions, messages, roles, num_patches_list=num_patches_list)
        inputs = tokenizer(prompt, return_tensors="pt")
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["pixel_values"] = pixel_values.clone()
        generation_config = dict(max_new_tokens=max_gen_len, do_sample=False)
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())
        api_runner = ApiRunnerInternVL(
            batch_size,
            processor,
            config,
            image,
            query,
            prompt_len,
            ctx_len,
            max_gen_len,
            num_hidden_layers,
        )
        pytorch_hf_tokens = _resolve_vlm_hf_golden(
            model_name=model_name,
            config=config,
            query=query,
            img_url=img_url,
            torch_dtype=torch_dtype,
            max_gen_len=max_gen_len,
            compile_only=compile_only,
            compute_fn=lambda: api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs, generation_config),
        )
        compile_kwargs["num_patches"] = 1

    elif model_name in ModelConfig.MOLMO_MODELS:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        image = load_test_image(img_url, size=(536, 354), session=_session)
        inputs = processor.process(images=[image], text=query)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        generation_config = GenerationConfig(max_new_tokens=NEW_GENERATION_TOKENS, stop_strings="<|endoftext|>")
        api_runner = ApiRunnerMolmo(
            batch_size,
            processor,
            config,
            image,
            query,
            prompt_len,
            ctx_len,
            max_gen_len,
            (num_hidden_layers, num_hidden_layers),
        )
        pytorch_hf_tokens = _resolve_vlm_hf_golden(
            model_name=model_name,
            config=config,
            query=query,
            img_url=img_url,
            torch_dtype=torch_dtype,
            max_gen_len=max_gen_len,
            compile_only=compile_only,
            compute_fn=lambda: api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs, generation_config),
        )
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
        valid = inputs["image_input_idx"] > 0
        valid = valid.reshape(1, -1)
        inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
        inputs["pixel_values"] = inputs.pop("images")
        compile_kwargs["img_size"] = img_size

    elif is_kimi_k25(model_name):
        image = load_test_image(img_url, session=_session)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": query},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            messages=conversation,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )
        if not compile_only:
            pytorch_hf_tokens = run_kimi_k25_hf_model_on_pytorch(
                copy.deepcopy(model_hf), processor, inputs, max_gen_len
            )
        compile_kwargs.update(
            {
                "prefill_seq_len": 1,
                "image_height": image.height,
                "image_width": image.width,
            }
        )

    else:
        processor = _load_vlm_processor(model_name)
        image = load_test_image(img_url, session=_session)
        if model_name == "tiny-random/mistral-3":
            image = image.resize((1540, 1540))
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        api_runner = ApiRunnerVlm(
            batch_size,
            processor,
            config,
            image,
            conversation,
            prompt,
            prompt_len,
            ctx_len,
            max_gen_len,
            num_hidden_layers,
        )
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
        pytorch_hf_tokens = _resolve_vlm_hf_golden(
            model_name=model_name,
            config=config,
            query=query,
            img_url=img_url,
            torch_dtype=torch_dtype,
            max_gen_len=max_gen_len,
            compile_only=compile_only,
            compute_fn=lambda: api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs),
        )
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type in [
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        ]:
            inputs = qeff_model.model.prepare_inputs_for_generation(
                inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
            )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
        if model_name != "allenai/MolmoPoint-8B":
            compile_kwargs["img_size"] = img_size

    if model_name == "allenai/MolmoPoint-8B" and not compile_only:
        pytorch_kv_tokens, pytorch_vision_outputs, pytorch_step_outputs = _run_molmo_point_qeff_pytorch(
            qeff_model.model,
            inputs,
            prompt_len,
            ctx_len,
            max_gen_len,
        )
        _assert_runtime_token_parity(pytorch_hf_tokens, pytorch_kv_tokens)
        with model_export_compile_lock(model_name):
            onnx_model_path = qeff_model.export(
                use_onnx_subfunctions=use_onnx_subfunctions,
                offload_pt_weights=False,
            )
        for path in onnx_model_path:
            onnx.checker.check_model(path)
        ort_tokens, ort_vision_outputs, ort_step_outputs = _run_molmo_point_ort(
            api_runner,
            qeff_model.model,
            inputs,
            onnx_model_path,
            prompt_len,
            ctx_len,
            max_gen_len,
            pytorch_vision_outputs,
            pytorch_step_outputs,
        )
        _assert_runtime_token_parity(pytorch_kv_tokens, ort_tokens)
        compile_kwargs["vision_onnx_path"] = str(onnx_model_path[0])
        compile_kwargs["lang_onnx_path"] = str(onnx_model_path[1])

    if (
        mdp_compile_kwargs
        and model_name not in ModelConfig.INTERNVL_MODELS
        and model_name not in ModelConfig.MOLMO_MODELS
    ):
        compile_kwargs["skip_vision"] = True
        compile_kwargs.update(mdp_compile_kwargs)
    elif mdp_compile_kwargs:
        compile_kwargs.update(mdp_compile_kwargs)
    compile_kwargs["use_onnx_subfunctions"] = use_onnx_subfunctions
    with model_export_compile_lock(model_name):
        qeff_model.compile(**compile_kwargs)

    if compile_only:
        manual_cleanup(qeff_model.onnx_path)
        return

    if model_name == "allenai/MolmoPoint-8B":
        qpc_tokens = _run_molmo_point_qpc(
            qeff_model.model,
            inputs,
            [qeff_model.vision_model.qpc_path, qeff_model.lang_model.qpc_path],
            [0],
            prompt_len,
            ctx_len,
            max_gen_len,
            ort_vision_outputs,
            ort_step_outputs,
        )
        _assert_runtime_token_parity(ort_tokens, qpc_tokens)

    streamer = TextStreamer(processor.tokenizer)
    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(inputs=inputs, generation_len=max_gen_len, streamer=streamer)
    print(exec_info)
    cloud_ai_100_tokens = exec_info.generated_ids[:, :-1]
    parity_issue = known_runtime_parity_issue or model_config_dict[model_name].get("known_runtime_parity_issue")
    _assert_runtime_token_parity(pytorch_hf_tokens, cloud_ai_100_tokens, parity_issue)
    manual_cleanup(qeff_model.onnx_path)  # Clean up the model files after the tests are done.
    if compare_results is False:
        return

    dump_and_compare_results(
        model_name=model_name,
        compile_params=compile_kwargs,
        json_file_path="image_text_to_text_model_results.json",
        cloud_ai_100_tokens=cloud_ai_100_tokens.tolist(),
        pytorch_hf_tokens=pytorch_hf_tokens.tolist(),
        pytorch_kv_tokens=pytorch_kv_tokens.tolist() if pytorch_kv_tokens is not None else None,
        ort_tokens=ort_tokens.tolist() if ort_tokens is not None else None,
        exec_info=exec_info,
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # VLMs only need dual-QPC coverage; single-QPC isn't exercised.
def test_full_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        kv_offload=kv_offload,
        compare_results=True,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # VLMs only need dual-QPC coverage; single-QPC isn't exercised.
def test_few_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        num_hidden_layers=model_config_dict[model_name]["num_layers"],
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_moe_models)
@pytest.mark.parametrize("kv_offload", [True])  # VLMs only need dual-QPC coverage; single-QPC isn't exercised.
def test_few_image_text_to_text_onnx_mdp_compile_only(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        num_hidden_layers=model_config_dict[model_name]["num_layers"],
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
        compile_only=True,
        mdp_num_partitions=2,
        mdp_strategy="onnx",
        use_onnx_subfunctions=True,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # VLMs only need dual-QPC coverage; single-QPC isn't exercised.
def test_dummy_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    torch.manual_seed(7 if model_name == "allenai/MolmoPoint-8B" else 42)
    hf_config = None
    if is_kimi_k25(model_name):
        hf_config = get_kimi_k25_test_config(model_name, model_config_dict)
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            kv_offload=kv_offload,
            config=hf_config,
            manual_cleanup=manual_cleanup,
            known_runtime_parity_issue=model_config_dict[model_name].get("known_dummy_runtime_parity_issue"),
        )
    elif model_name in ModelConfig.STANDARD_VLM_MODELS:
        model_type = model_config_dict[model_name].get("model_type", None)
        custom_config = model_config_dict[model_name].get("additional_params", {})
        hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
        hf_config.name_or_path = model_name
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            kv_offload=kv_offload,
            config=hf_config,
            manual_cleanup=manual_cleanup,
            known_runtime_parity_issue=model_config_dict[model_name].get("known_dummy_runtime_parity_issue"),
        )
    else:
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            num_hidden_layers=model_config_dict[model_name]["num_layers"],
            kv_offload=kv_offload,
            manual_cleanup=manual_cleanup,
        )


def _run_dummy_dual_qpc_case(model_name, manual_cleanup, layer_types=None, **kwargs):
    """Run one dummy-layer dual-QPC case, resolving the config the same way for every variant.

    ``STANDARD_VLM_MODELS`` are built from a synthesized ``AutoConfig`` (random weights at
    the config's declared sizes); the rest carry a ``num_layers`` override applied to the
    checkpoint's own config. Both branches otherwise share the same call, so the per-variant
    knobs are passed through ``kwargs``.

    ``layer_types`` overrides the language-side attention pattern, and with it the truncation
    depth, for a variant that needs a specific mix of layer kinds; its length becomes the
    layer count so the pattern and the depth cannot drift apart.
    """
    if model_name in ModelConfig.STANDARD_VLM_MODELS:
        model_type = model_config_dict[model_name].get("model_type", None)
        custom_config = model_config_dict[model_name].get("additional_params", {})
        hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
    elif layer_types is not None:
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hf_config = set_num_layers_vlm(hf_config, n_layer=len(layer_types))
    else:
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            num_hidden_layers=model_config_dict[model_name]["num_layers"],
            kv_offload=True,
            manual_cleanup=manual_cleanup,
            **kwargs,
        )
        return

    if layer_types is not None:
        hf_config.text_config.num_hidden_layers = len(layer_types)
        hf_config.text_config.layer_types = layer_types
    hf_config.name_or_path = model_name
    kwargs.setdefault(
        "known_runtime_parity_issue", model_config_dict[model_name].get("known_dummy_runtime_parity_issue")
    )
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        kv_offload=True,
        config=hf_config,
        manual_cleanup=manual_cleanup,
        **kwargs,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
def test_dummy_image_text_to_text_ccl_dual_qpc(model_name, manual_cleanup):
    """Compute-context-length (CCL) parity for every VLM, dual QPC only.

    CCL only changes the language-side specializations, and the prefill/decode QPC
    split that consumes them exists solely on the dual-QPC path, so this runs
    ``kv_offload=True`` for all models rather than parametrizing over both.

    CCL values default to automatic generation from each model's ``ctx_len``. A model
    opts into explicit values with a ``comp_ctx_lengths_decode`` entry in
    ``image_text_model_configs.json``; a multi-value list is what pins the
    disagg prefill/decode specialization slicing, since a single-value list cannot
    distinguish a correct slice from one truncated to a single specialization.

    A hybrid linear-attention model additionally opts into a ``ccl_layer_types`` pattern
    when its default truncation depth keeps no ``full_attention`` layer; see the comment
    on that branch below for why CCL needs one.

    Qwen2.5-VL is xfailed rather than skipped so the dual-QPC CCL plumbing is still
    exercised on that model path. Its dummy-layer HF-vs-QAIC token parity is a
    pre-existing gap unrelated to CCL: the random-init 1-layer config yields near-flat
    decode logits (HF top1-top2 margin <0.11 on most positions), which fp16 rounding at
    the QPC flips into a different top-K member on those steps.
    """
    ccl_forced = {
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    }
    if model_name in ModelConfig.SKIPPED_MODELS and model_name not in ccl_forced:
        pytest.skip("Test skipped for this model due to some issues.")
    torch.manual_seed(7 if model_name == "allenai/MolmoPoint-8B" else 42)
    comp_ctx_lengths_decode = model_config_dict[model_name].get("comp_ctx_lengths_decode")

    # On hybrid linear-attention stacks only the full_attention layers consume
    # comp_ctx_lengths; the linear_attention (Gated-DeltaNet) path ignores it. Truncating
    # such a model to a depth that keeps no full_attention layer therefore leaves the input
    # dead, ONNX prunes it, and every CCL specialization becomes identical, which
    # qaic-compile rejects with "No input that uniquely identifies specialization". A model
    # whose default truncation lands short of its first full_attention layer pins a pattern
    # holding both layer kinds via ``ccl_layer_types``; both are required, since an
    # all-full_attention stack breaks the hybrid cache's linear-layer state indexing.
    _run_dummy_dual_qpc_case(
        model_name,
        manual_cleanup,
        layer_types=model_config_dict[model_name].get("ccl_layer_types"),
        ccl_enabled=True,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_blocking_models)
def test_dummy_image_text_to_text_blocking_dual_qpc(model_name, manual_cleanup):
    """Blocked-KV attention parity for VLMs, dual QPC only.

    Mirrors ``test_per_pr_causal_fp16_subfunction_cb_blocking`` on the causal-LM side.
    Blocking rewrites the language-side attention forward, so this runs ``kv_offload=True``.

    Parametrized over models flagged ``supports_blocking`` in
    ``image_text_model_configs.json`` rather than every VLM: ``BlockingAttentionTransform``
    attaches ``attn_blocking_config`` to every mapped ``*Attention`` module, but only some
    attention forwards read it. Running the families that ignore it would pass while
    exercising nothing.
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    torch.manual_seed(42)
    _run_dummy_dual_qpc_case(
        model_name,
        manual_cleanup,
        qaic_config={"blocking_mode": "kv", "num_kv_blocks": 2},
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
def test_dummy_image_text_to_text_bf16_compile_only(model_name, manual_cleanup):
    """BF16 export + compile for VLMs, dual QPC only.

    Mirrors ``test_per_pr_causal_bf16_subfunction_cb_ccl_compile_only``. Kept
    ``compile_only`` so the extra coverage costs an export/compile and no device run.

    A family whose BF16 compile is known-broken opts out with a
    ``known_bf16_compile_issue`` entry in ``image_text_model_configs.json``, so the gap
    stays visible as an xfail instead of disappearing into ``SKIPPED_MODELS`` (which would
    drop that model from every other VLM test too).
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if bf16_issue := model_config_dict[model_name].get("known_bf16_compile_issue"):
        pytest.xfail(bf16_issue)

    torch.manual_seed(42)
    _run_dummy_dual_qpc_case(
        model_name,
        manual_cleanup,
        torch_dtype=torch.bfloat16,
        compile_only=True,
    )


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.dummy_layers
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # VLMs only need dual-QPC coverage; single-QPC isn't exercised.
def test_custom_replicate_kv_pytorch_vs_ai100(
    model_name,
    kv_offload,
    manual_cleanup,
):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    torch.manual_seed(42)
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.REPEAT_KV_TEST_MODELS:
        hf_config = None
        if model_name in ModelConfig.STANDARD_VLM_MODELS:
            model_type = model_config_dict[model_name].get("model_type")
            custom_config = model_config_dict[model_name].get("additional_params", {})
            hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
            hf_config.name_or_path = model_name

        if hf_config is not None:
            check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
                model_name=model_name,
                kv_offload=kv_offload,
                config=hf_config,
                qaic_config={},
                test_kv_replicate=True,
                manual_cleanup=manual_cleanup,
                known_runtime_parity_issue=model_config_dict[model_name].get("known_dummy_runtime_parity_issue"),
            )
        else:
            check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
                model_name=model_name,
                num_hidden_layers=model_config_dict[model_name]["num_layers"],
                kv_offload=kv_offload,
                qaic_config={},
                test_kv_replicate=True,
                manual_cleanup=manual_cleanup,
            )
    else:
        pytest.skip(f"Skipping replicate KV test for {model_name} as it's not in REPEAT_KV_TEST_MODELS")
