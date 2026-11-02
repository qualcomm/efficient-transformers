# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from gemma4_utils import (
    CHAT_TEMPLATE,
    build_compile_kwargs,
    build_messages,
    effective_lens,
    normalize_generated_ids,
    remove_fp16clip_transform_if_disabled,
    resolve_stop_token_ids,
    truncate_generated_ids_at_stop,
)
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "google/gemma-4-E2B-it"
SYSTEM_PROMPT = "You are a helpful assistant."
TEXT_PROMPT = "Tell me about Taj Mahal?"
IMAGE_PROMPT = "Can you Describe this image in detail?"
IMAGE_URL = "https://wallup.net/wp-content/uploads/2017/03/28/351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"
SKIP_VISION = False
BS = 1
PREFILL_SEQ_LEN = 128
CTX_LEN = 2048
GENERATION_LEN = 1920
NUM_LANG_HIDDEN_LAYER = 2
NUM_VISION_HIDDEN_LAYER = 2
# For CCL activation
# comp_ctx_lengths_prefill=[2048]
# comp_ctx_lengths_decode=[4096,65536]
# NODE_PRECISION_INFO:Optional argument
# If set to True, the NPI file will be generated automatically.
# If a file path is provided, that file will be used for compilation.
# If not specified or False, it will skip NPI file.
NODE_PRECISION_INFO = True

# Path to Node Precision Info YAML file.
# npi_file_path = "examples/image_text_to_text/models/gemma_vision/configs/gemma4_E4B_npi.yaml"
# npi_file_full_path = os.path.join(os.getcwd(), npi_file_path)

compiler_kwargs = {
    "NUM_CORES": 16,
    "NUM_DEVICES": 4,
    "MXFP6_MATMUL": True,
    "MXINT8_KV_CACHE": True,
    "AIC_ENABLE_DEPTH_FIRST": True,
    "MOS": 1,
    "USE_ONNX_SUBFUNCTIONS": True,
    "split_model_io": True,
    "BATCH_SIZE": BS,
    "node_precision_info": NODE_PRECISION_INFO,
    ## For CCL activation
    # "comp_ctx_lengths_prefill": comp_ctx_lengths_prefill,
    # "comp_ctx_lengths_decode": comp_ctx_lengths_decode,
}


def _apply_reduced_layer_config(config, num_lang_layers: int, num_vision_layers: int):
    config.text_config.num_hidden_layers = num_lang_layers
    config.vision_config.num_hidden_layers = num_vision_layers

    if hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]

    if hasattr(config.text_config, "num_kv_shared_layers"):
        # KV sharing to avoid invalid first_kv_shared_layer_idx=0 edge cases.
        config.text_config.num_kv_shared_layers = 0

    return config


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer
    chat_template = (
        getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE
    )
    config = AutoConfig.from_pretrained(MODEL_ID)

    # For Testing Purpose Only
    # config = _apply_reduced_layer_config(
    #     config,
    #     num_lang_layers=NUM_LANG_HIDDEN_LAYER,
    #     num_vision_layers=NUM_VISION_HIDDEN_LAYER,
    # )

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,
        ignore_mismatched_sizes=True,
        ## For CCL activation
        # qaic_config={
        #     "ccl_enabled": True,
        # },
    )
    remove_fp16clip_transform_if_disabled(qeff_model, True)

    if SKIP_VISION:
        messages = build_messages(SYSTEM_PROMPT, TEXT_PROMPT, use_image=False)
        text_inputs = processor.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = int(text_inputs["input_ids"].shape[1])
        effective_prefill_seq_len, effective_ctx_len = effective_lens(
            qeff_model,
            PREFILL_SEQ_LEN,
            CTX_LEN,
            prompt_len,
            GENERATION_LEN,
            SKIP_VISION,
        )

        compile_kwargs = build_compile_kwargs(
            effective_prefill_seq_len=effective_prefill_seq_len,
            effective_ctx_len=effective_ctx_len,
            skip_vision=SKIP_VISION,
            **compiler_kwargs,
        )
        # print("compile_kwargs:", compile_kwargs)
        qeff_model.compile(**compile_kwargs)

        output = qeff_model.generate(inputs=text_inputs, generation_len=GENERATION_LEN)
        qeff_ids = normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
        stop_token_ids = resolve_stop_token_ids(qeff_model, config, tokenizer)
        qeff_ids = truncate_generated_ids_at_stop(qeff_ids, stop_token_ids)
        print(tokenizer.batch_decode(qeff_ids, skip_special_tokens=True))
        print(output)
        return

    messages = build_messages(SYSTEM_PROMPT, IMAGE_PROMPT, use_image=True)
    messages[-1]["content"][0]["url"] = IMAGE_URL

    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_len = int(inputs["input_ids"].shape[1])
    effective_prefill_seq_len, effective_ctx_len = effective_lens(
        qeff_model,
        PREFILL_SEQ_LEN,
        CTX_LEN,
        prompt_len,
        GENERATION_LEN,
        SKIP_VISION,
    )

    compile_kwargs = build_compile_kwargs(
        effective_prefill_seq_len=effective_prefill_seq_len,
        effective_ctx_len=effective_ctx_len,
        skip_vision=SKIP_VISION,
        skip_model_io=True,
        **compiler_kwargs,
    )
    # print("compile_kwargs:", compile_kwargs)

    qeff_model.compile(**compile_kwargs)

    output = qeff_model.generate(
        inputs=inputs,
        generation_len=GENERATION_LEN,
    )
    qeff_ids = normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
    stop_token_ids = resolve_stop_token_ids(qeff_model, config, tokenizer)
    qeff_ids = truncate_generated_ids_at_stop(qeff_ids, stop_token_ids)
    print(tokenizer.batch_decode(qeff_ids, skip_special_tokens=True))
    print(output)


if __name__ == "__main__":
    main()
