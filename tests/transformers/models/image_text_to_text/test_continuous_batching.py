# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import json
import os
from typing import Optional

import pytest
import requests
import torch
from requests.adapters import HTTPAdapter
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)
from urllib3.util.retry import Retry

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
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
    run_kimi_k25_hf_model_on_pytorch_CB,
)

_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_models"]
test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}

NEW_GENERATION_TOKENS = 10


def _assert_runtime_token_parity(reference_tokens, qpc_tokens, message, parity_issue=None):
    """Compare HF and QAIC tokens, xfail only a configured numerical parity mismatch."""
    if (reference_tokens == qpc_tokens).all():
        return
    if parity_issue:
        pytest.xfail(parity_issue)
    pytest.fail(message)


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
    model_name: str,
    manual_cleanup: callable,
    num_hidden_layers: int = -1,
    kv_offload: bool = False,
    num_devices: int = 1,
    config: Optional[AutoConfig] = None,
    kv_cache_batch_size: Optional[int] = None,
    compile_only: bool = False,
    known_runtime_parity_issue: Optional[str] = None,
):
    # Two-phase compile/execute split: suppress per-test cleanup in both phases (model variants
    # share a content-addressed export dir, so one variant's rmtree would destroy its siblings'
    # warm QPCs) and force compile-only in the warm phase. A no-op in normal runs.
    manual_cleanup, compile_only = resolve_two_phase_cleanup(manual_cleanup, compile_only)
    parity_issue = known_runtime_parity_issue or model_config_dict[model_name].get("known_runtime_parity_issue")
    prompt_len = model_config_dict[model_name]["prompt_len"]
    ctx_len = model_config_dict[model_name]["ctx_len"]
    img_size = model_config_dict[model_name].get("img_size")
    image_urls = model_config_dict[model_name]["img_url_list"]
    queries = model_config_dict[model_name]["text_prompt_list"]
    n_layer = num_hidden_layers
    batch_size = model_config_dict[model_name]["batch_size"]
    full_batch_size = model_config_dict[model_name]["full_batch_size"]
    # Prefix caching sizes the KV cache from kv_cache_batch_size, and the decode
    # specialization records that value as its full_batch_size (modeling_auto.py:4149/4215).
    # The runtime reads the batch count back from the QPC, so the running batch has to match
    # kv_cache_batch_size or prefill runs out of prompts -- exactly like the causal-LM helper,
    # which sets `full_batch_size = kv_cache_batch_size or 4` (check_causal_models.py:175).
    # The image/prompt lists below are tiled to full_batch_size, so they follow along.
    if kv_cache_batch_size is not None:
        full_batch_size = kv_cache_batch_size
    max_gen_len = model_config_dict[model_name].get("generation_len", NEW_GENERATION_TOKENS)

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
            torch_dtype=torch.float32,
            continuous_batching=True,
        )
    elif config is None:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=model_name not in ModelConfig.MOLMO_MODELS
        )
        config = set_num_layers_vlm(config, n_layer=n_layer)
        if hasattr(config, "model_type") and config.model_type in ["gemma3"]:
            config.text_config._sliding_window_pattern = 2
            config.text_config.layer_types = ["sliding_attention", "full_attention"]
        if hasattr(config, "model_type") and config.model_type in ["gemma4"]:
            config.text_config.num_kv_shared_layers = 0
            config.text_config.layer_types = ["sliding_attention"]
            # Keep the sliding window below ctx_len (512). The hub value (1024) exceeds ctx_len --
            # a degenerate setup where the window never slides -- and that path crashes qaic-compile's
            # rolling-cache `where` selector once the decode batch reaches 4 (prefix caching uses fbs=4).
            config.text_config.sliding_window = 256
        if hasattr(config, "model_type") and config.model_type in ["qwen3_5"]:
            config.text_config.layer_types = [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
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
                continuous_batching=True,
            )
        else:
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
                continuous_batching=True,
            )
    else:
        model_hf = load_vlm_model_from_config(config)
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=model_hf.config,
            continuous_batching=True,
        )
    compile_kwargs = {
        "num_cores": 16,
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "batch_size": batch_size,
        "full_batch_size": full_batch_size,
        "mxfp6_matmul": False,
        "split-model-io": True,
    }
    # Sizes the KV cache independently of full_batch_size (prefix caching). Left out unless
    # requested so the default CB runs keep inferring it from full_batch_size.
    if kv_cache_batch_size is not None:
        compile_kwargs["kv_cache_batch_size"] = kv_cache_batch_size
    if model_name in ["qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "gemma4"]:
        compile_kwargs["use_onnx_subfunctions"] = True

    images = []
    generation_config = None
    if model_name in ModelConfig.INTERNVL_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)
        image_height = 448
        image_width = 448
        for img_url in image_urls:
            image = load_test_image(img_url, size=(image_height, image_width), session=_session)
            images.append(image)
        generation_config = dict(max_new_tokens=max_gen_len, do_sample=False)
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())
        api_runner = ApiRunnerInternVL(
            batch_size,
            processor,
            config,
            images[0],
            queries[0],
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        # For same prompt
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        if not compile_only:
            pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list)
        compile_kwargs["num_patches"] = 1
    elif model_name in ModelConfig.MOLMO_MODELS:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_height = 536
        image_width = 354
        for img_url in image_urls:
            image = load_test_image(img_url, size=(image_height, image_width), session=_session)
            images.append(image)
        api_runner = ApiRunnerMolmo(
            batch_size,
            processor,
            config,
            images[0],
            queries[0],
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        generation_config = GenerationConfig(max_new_tokens=NEW_GENERATION_TOKENS, stop_strings="<|endoftext|>")
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        if not compile_only:
            pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
                model_hf, image_list, prompt_list, generation_config
            )
        compile_kwargs["img_size"] = img_size
    elif is_kimi_k25(model_name):
        for img_url in image_urls:
            image = load_test_image(img_url, session=_session)
            images.append(image)

        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        if not compile_only:
            pytorch_hf_tokens = run_kimi_k25_hf_model_on_pytorch_CB(
                copy.deepcopy(model_hf), processor, image_list, prompt_list, max_gen_len
            )
        image_height = images[0].height
        image_width = images[0].width
        compile_kwargs.update(
            {
                "image_height": image_height,
                "image_width": image_width,
            }
        )
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        use_fast = model_name != "tiny-random/mistral-3"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=use_fast)
        image_height = None
        image_width = None
        for img_url in image_urls:
            image = load_test_image(img_url, session=_session)
            if model_name == "tiny-random/mistral-3":
                image_height = 1540
                image_width = 1540
                image = image.resize((image_height, image_width))
            images.append(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": queries[0]},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        api_runner = ApiRunnerVlm(
            batch_size,
            processor,
            config,
            images[0],
            conversation,
            prompt,
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        if not compile_only:
            pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list)
        compile_kwargs["img_size"] = img_size

    with model_export_compile_lock(model_name):
        qeff_model.compile(**compile_kwargs)

    if compile_only:
        manual_cleanup(qeff_model.onnx_path)
        return

    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=[images[0]] * full_batch_size,
        prompts=prompt_list,
        generation_len=max_gen_len,
        image_height=image_height,
        image_width=image_width,
    )
    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching with same prompt:")
    print(exec_info.generated_texts)
    for i in range(full_batch_size):
        _assert_runtime_token_parity(
            pytorch_hf_tokens[i],
            qpc_tokens[i],
            f"Tokens don't match for prompt {i} between HF and QPC output for same prompts",
            parity_issue,
        )
    # The distinct-prompt leg runs full_batch_size prompts. Prefix caching bumps
    # full_batch_size above the config's 2-entry prompt/image lists, so tile them to
    # full_batch_size (modulo) exactly as the causal helper does (check_causal_models.py:177).
    # A no-op when full_batch_size == len(queries) (the plain-CB case), so existing CB runs
    # are unchanged; without it the shorter lists under-fill the prompt queue and prefill
    # pops an empty deque (vlm_generation.py:935).
    diff_images = [images[i % len(images)] for i in range(full_batch_size)]
    diff_queries = [queries[i % len(queries)] for i in range(full_batch_size)]
    if model_name in ModelConfig.MOLMO_MODELS:
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
            model_hf, diff_images, diff_queries, generation_config=generation_config
        )
    else:
        if is_kimi_k25(model_name):
            # Kimi-K2.5 is compiled for the first image shape in this test; QPC generation resizes
            # each image to that shape, so resize HF reference images the same way while keeping content different.
            diff_images = [image.resize((image_width, image_height)) for image in diff_images]
            pytorch_hf_tokens = run_kimi_k25_hf_model_on_pytorch_CB(
                copy.deepcopy(model_hf), processor, diff_images, diff_queries, max_gen_len
            )
        else:
            pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, diff_images, diff_queries)

    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=diff_images,
        prompts=diff_queries,
        generation_len=max_gen_len,
        image_height=image_height,
        image_width=image_width,
    )
    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching with different prompt:")
    print(exec_info.generated_texts)
    for i in range(full_batch_size):
        _assert_runtime_token_parity(
            pytorch_hf_tokens[i],
            qpc_tokens[i],
            f"Tokens don't match for prompt {i} between HF and QPC output for different prompts",
            parity_issue,
        )
    manual_cleanup(qeff_model.onnx_path)  # Clean up the model files after the tests are done.


@pytest.mark.skip("Token Mismatch for full models")
@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_full_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name=model_name,
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_few_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name=model_name,
        num_hidden_layers=model_config_dict[model_name]["num_layers"],
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_dummy_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    hf_config = None
    if is_kimi_k25(model_name):
        hf_config = get_kimi_k25_test_config(model_name, model_config_dict)
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
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
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
            model_name,
            kv_offload=kv_offload,
            config=hf_config,
            manual_cleanup=manual_cleanup,
            known_runtime_parity_issue=model_config_dict[model_name].get("known_dummy_runtime_parity_issue"),
        )
    else:
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
            model_name,
            num_hidden_layers=model_config_dict[model_name]["num_layers"],
            kv_offload=kv_offload,
            manual_cleanup=manual_cleanup,
        )


# Larger than every config's full_batch_size (2) so the KV-cache buffer is sized apart from
# the plain-CB default and the kv_cache_batch_size compile arg is genuinely exercised. Capped
# at 4: Qwen-VL runs the shared base CB decode loop (run_continuous_batching_decode), which
# indexes the 4-D mrope decode_pos_ids `(4, batch, 1)` on axis 0 by decode_batch_id
# (text_generation_inference.py:964), so a running batch >4 IndexErrors. This is a pre-existing
# Qwen-VL CB ceiling -- reproducible with the plain CB test at full_batch_size=8 -- not a
# prefix-caching bug. The causal side runs 8; VLMs stay at 4 until that loop is generalized.
PREFIX_CACHING_KV_CACHE_BATCH_SIZE = 4


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_dummy_image_text_to_text_prefix_caching_cb(model_name, kv_offload, manual_cleanup):
    """Prefix-caching (``kv_cache_batch_size``) parity for VLMs.

    Mirrors ``test_per_pr_causal_fp16_subfunction_cb_prefix_caching`` on the causal-LM side.
    Lives with the CB tests because ``compile()`` rejects ``kv_cache_batch_size`` unless
    continuous batching is on.
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    hf_config = None
    if model_name in ModelConfig.STANDARD_VLM_MODELS:
        model_type = model_config_dict[model_name].get("model_type", None)
        custom_config = model_config_dict[model_name].get("additional_params", {})
        hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
        hf_config.name_or_path = model_name
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
            model_name,
            kv_offload=kv_offload,
            config=hf_config,
            manual_cleanup=manual_cleanup,
            kv_cache_batch_size=PREFIX_CACHING_KV_CACHE_BATCH_SIZE,
            known_runtime_parity_issue=model_config_dict[model_name].get("known_dummy_runtime_parity_issue"),
        )
    else:
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
            model_name,
            num_hidden_layers=model_config_dict[model_name]["num_layers"],
            kv_offload=kv_offload,
            manual_cleanup=manual_cleanup,
            kv_cache_batch_size=PREFIX_CACHING_KV_CACHE_BATCH_SIZE,
        )
