# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

model_id = "Qwen/Qwen3.5-0.8B"
DECODE_NUM_DEVICES = int(os.environ.get("QEFF_DECODE_NUM_DEVICES", "1"))
WEIGHT_FREE = 1
config = AutoConfig.from_pretrained(model_id)

# For faster execution user can run with lesser layers, For Testing Purpose Only
# config.vision_config.depth = 4
# config.text_config.num_hidden_layers = 4
config.torch_dtype = "float32"
layer_types = list(getattr(config.text_config, "layer_types", []))
if len(layer_types) < config.text_config.num_hidden_layers:
    layer_types.extend(["full_attention"] * (config.text_config.num_hidden_layers - len(layer_types)))
config.text_config.layer_types = layer_types[: config.text_config.num_hidden_layers]


def _update_retained_states(target_inputs, source_outputs):
    for layer_idx, layer_type in enumerate(config.text_config.layer_types):
        if layer_type == "full_attention":
            state_names = (f"past_key.{layer_idx}", f"past_value.{layer_idx}")
        else:
            state_names = (f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}")

        # state_names = (f"past_key.{layer_idx}", f"past_value.{layer_idx}")
        for state_name in state_names:
            retained_state = source_outputs[f"{state_name}_RetainedState"]
            target_inputs[state_name] = np.array(retained_state, copy=True)


qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    weight_free=WEIGHT_FREE,
    config=config,
    # # For CCL activation
    # qaic_config={
    #     "ccl_enabled": True,
    # },
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Enable KV blocking for full-attention layers with 2 KV blocks
# To disable KV blocking, comment out the qaic_config line below
# Set skip_kv=True to skip future KV blocks during inference (optimization)
qaic_config = {"blocking_mode": "kv", "num_kv_blocks": 2, "skip_kv": True}

enable_blocking = False  # By default blocking is false
### use skip_vision=True, if want to run only text, or false ###
skip_vision = False

BS = 1
PREFILL_SEQ_LEN = 64
CTX_LEN = 4096

# Compute-Context-Length (CCL) lists for prefill and decode. When both are None and
# ccl_enabled=True, they are auto-generated from CTX_LEN.
# comp_ctx_lengths_prefill = [2048]
# comp_ctx_lengths_decode = [4096, 65536]

if skip_vision:
    ## Only Text ##

    prefill_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_model_io=True,
        aic_enable_depth_first=False,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        mos=1,
        use_onnx_subfunctions=True,
        # comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        # comp_ctx_lengths_decode=comp_ctx_lengths_decode,
        # qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
    )

    decode_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=DECODE_NUM_DEVICES,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_model_io=True,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        mos=1,
        use_onnx_subfunctions=True,
        # qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
    )

    if enable_blocking:
        print("\n" + "=" * 80)
        print("Verifying KV Blocking Applied During Compilation")
        print("=" * 80)

        if qaic_config and qaic_config.get("blocking_mode"):
            print("✓ qaic_config passed to compile():")
            print(f"    Blocking Mode: {qaic_config.get('blocking_mode')}")
            print(f"    Num KV Blocks: {qaic_config.get('num_kv_blocks')}")
            print(f"    Skip KV: {qaic_config.get('skip_kv', False)}")
            print("\n✓ BlockingAttentionTransform.apply() called during compile()")
            print("  - Sets attn_blocking_config on all supported attention modules")
            print("  - Blocked attention forward pass is used during ONNX export")
            print("  - Blocking operations are in the ONNX graph and QPC")
            print("\n  Status: ACTIVE")
            print("  Verification: Config-based verification")
            print("  Note: Blocking IS applied - torch model is freed after ONNX export")
        else:
            print("✗ No qaic_config provided - eager attention will be used")
            print("  Status: INACTIVE - Model compiled without blocking")

        print("=" * 80 + "\n")
else:
    ## Vision + Text ##

    vision_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=1,
        height=354,
        width=536,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=False,
        mos=1,
        split_model_io=True,
        skip_vision=False,
        skip_lang=True,
        use_onnx_subfunctions=True,
    )

    prefill_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=1,
        height=354,
        width=536,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_model_io=True,
        aic_enable_depth_first=False,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        mos=1,
        use_onnx_subfunctions=True,
        # qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
    )

    decode_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=DECODE_NUM_DEVICES,
        height=354,
        width=536,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_model_io=True,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        mos=1,
        use_onnx_subfunctions=True,
        # comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        # comp_ctx_lengths_decode=comp_ctx_lengths_decode,
        # qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
    )

    if enable_blocking:
        print("\n" + "=" * 80)
        print("Verifying KV Blocking Applied During Compilation")
        print("=" * 80)

        if qaic_config and qaic_config.get("blocking_mode"):
            print("✓ qaic_config passed to compile():")
            print(f"    Blocking Mode: {qaic_config.get('blocking_mode')}")
            print(f"    Num KV Blocks: {qaic_config.get('num_kv_blocks')}")
            print(f"    Skip KV: {qaic_config.get('skip_kv', False)}")
            print("\n✓ BlockingAttentionTransform.apply() called during compile()")
            print("  - Sets attn_blocking_config on all supported attention modules")
            print("  - Blocked attention forward pass is used during ONNX export")
            print("  - Blocking operations are in the ONNX graph and QPC")
            print("\n  Status: ACTIVE")
            print("  Verification: Config-based verification")
            print("  Note: Blocking IS applied - torch model is freed after ONNX export")
        else:
            print("✗ No qaic_config provided - eager attention will be used")
            print("  Status: INACTIVE - Model compiled without blocking")

        print("=" * 80 + "\n")

    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)

    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe all the colors seen in the image."},
            ],
        },
    ]

    messages = [messages_1] * BS

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs, prefill_seq_len=PREFILL_SEQ_LEN, batch_size=BS
    )

lang_prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"))
lang_decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))
prefill_input_binding = lang_prefill_session.binding_index_map["input_ids"]
compiled_prefill_seq_len = lang_prefill_session.bindings[prefill_input_binding].dims[1]
vision_session = None
if not skip_vision:
    vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))

if skip_vision:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe yourself as a large language model, including your purpose, capabilities, and limitations. Explain how you process and generate responses, interact with users, and handle uncertainty, while emphasizing accuracy, safety, and helpfulness in diverse conversations across various topics and domains.",
                },
            ],
        },
    ]
else:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe all the colors seen in the image."},
            ],
        },
    ]

messages = [messages] * BS
texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = qeff_model.model.prepare_inputs_for_generation(
    inputs=inputs, prefill_seq_len=compiled_prefill_seq_len, batch_size=BS
)

pad_token_id = tokenizer.pad_token_id
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -compiled_prefill_seq_len)
padded_len = num_chunks * compiled_prefill_seq_len

inputs["input_ids"] = torch.nn.functional.pad(
    inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
)
inputs["attention_mask"] = torch.nn.functional.pad(
    inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
)

for key, value in inputs.items():
    inputs[key] = np.array(value)

vision_inputs = {
    key: value
    for key, value in inputs.items()
    if key in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
}
for key in {"pixel_values", "image_masks"}:
    if key in vision_inputs:
        vision_inputs[key] = vision_inputs[key].astype("float16")

vision_outputs = {}
if vision_inputs:
    vision_outputs = vision_session.run(vision_inputs)

lang_inputs = {key: value for key, value in inputs.items() if key not in vision_inputs}
if "position_ids" not in lang_inputs:
    lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
else:
    lang_inputs.pop("attention_mask", None)
lang_inputs["image_idx"] = np.array([[0]])
if not skip_vision:
    lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]

generation_len = 100 if not skip_vision else 512
all_outputs = []
lang_prefill_session.set_buffers(vision_outputs)
chunk_inputs = lang_inputs.copy()
for chunk_idx in range(num_chunks):
    chunk_start = chunk_idx * compiled_prefill_seq_len
    chunk_end = (chunk_idx + 1) * compiled_prefill_seq_len
    chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, chunk_start:chunk_end]
    chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., chunk_start:chunk_end]
    outputs = lang_prefill_session.run(chunk_inputs)
    _update_retained_states(chunk_inputs, outputs)
    chunk_inputs["image_idx"] = outputs["image_idx_output"]

all_outputs.append(np.argmax(outputs["logits"]))
decode_inputs = {
    "input_ids": np.argmax(outputs["logits"]).reshape(BS, 1),
    "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
}
_update_retained_states(decode_inputs, outputs)
decode_inputs["image_idx"] = outputs["image_idx_output"]
if not skip_vision:
    decode_inputs["vision_embeds"] = outputs["vision_embeds_RetainedState"]
decode_out = lang_decode_session.run(decode_inputs)
all_outputs.append(np.argmax(decode_out["logits"]))
position_ids = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(BS, 1),
    "position_ids": position_ids,
}
_update_retained_states(loop_decode_inputs, decode_out)
loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]
if not skip_vision:
    loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]

for _ in range(generation_len - 2):
    decode_out = lang_decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"]))
    position_ids += 1
    _update_retained_states(loop_decode_inputs, decode_out)
    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(BS, 1),
            "position_ids": position_ids,
        }
    )
print(tokenizer.decode(np.asarray(all_outputs).reshape(-1).tolist()))
