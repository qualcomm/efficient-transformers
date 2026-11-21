# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

model_id = "Qwen/Qwen3.6-35B-A3B"
DECODE_NUM_DEVICES = int(os.environ.get("QEFF_DECODE_NUM_DEVICES", "1"))
config = AutoConfig.from_pretrained(model_id)

# For faster execution user can run with lesser layers, For Testing Purpose Only
config.vision_config.depth = 5
config.text_config.num_hidden_layers = 2
config.torch_dtype = "float16"
layer_types = list(getattr(config.text_config, "layer_types", []))
if len(layer_types) < config.text_config.num_hidden_layers:
    layer_types.extend(["full_attention"] * (config.text_config.num_hidden_layers - len(layer_types)))
config.text_config.layer_types = layer_types[: config.text_config.num_hidden_layers]


def _update_retained_states(target_inputs, source_outputs):
    for layer_idx, layer_type in enumerate(config.text_config.layer_types):
        if layer_type == "full_attention":
            target_inputs[f"past_key.{layer_idx}"] = source_outputs[f"past_key.{layer_idx}_RetainedState"]
            target_inputs[f"past_value.{layer_idx}"] = source_outputs[f"past_value.{layer_idx}_RetainedState"]
        else:
            target_inputs[f"conv_state.{layer_idx}"] = source_outputs[f"conv_state.{layer_idx}_RetainedState"]
            target_inputs[f"recurrent_state.{layer_idx}"] = source_outputs[f"recurrent_state.{layer_idx}_RetainedState"]


qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

PREFILL_SEQ_LEN = 512
CTX_LEN = 4096
BS = 1

# qaic_config here controls only gated-delta settings (no KV blocking).
# Use this for qwen3_5 family of models, ow it will affect hte accuracy.
qaic_config = {}
qaic_config["gdn_chunk_size"] = PREFILL_SEQ_LEN

# Update qaic_config here for Blocking settings.
# qaic_config.update({"blocking_mode": "kv", "num_kv_blocks": 2, "skip_kv": True})

generation_len = 256

skip_vision = False

if not skip_vision:
    vision_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=354,
        width=536,
        num_cores=16,
        num_devices=1,
        mos=1,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        skip_vision=skip_vision,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        # Uncomment for vision inputs with height > 2048 or width > 2048:
        # elf_va_limit=2048,
    )

prefill_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=1,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    retain_full_kv=True,
    split_model_io=True,  # This should be used for disagg serving via VLLM
    mos=1,
    user_tiled=True,
    aic_enable_depth_first=False,
    prefill_only=True,
    enable_chunking=True,
    skip_vision=True,
    use_onnx_subfunctions=True,
    qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
)


decode_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=DECODE_NUM_DEVICES,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    retain_full_kv=True,
    split_model_io=True,  # This should be used for disagg serving via VLLM
    mos=1,
    aic_enable_depth_first=True,
    prefill_only=False,
    skip_vision=True,
    use_onnx_subfunctions=True,
)

lang_prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"))
lang_decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))

if skip_vision:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]
else:
    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe all the colors seen in the image."},
            ],
        },
    ]
    vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))


messages = [messages] * BS

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)


inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=PREFILL_SEQ_LEN, batch_size=BS)

pad_token_id = 1
input_len = inputs["attention_mask"].sum(1, keepdims=True)
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

print(f"generation_len : {generation_len}")
generated_ids = np.full((BS, generation_len + 1), pad_token_id)


inputs["input_ids"] = torch.nn.functional.pad(
    inputs["input_ids"],
    (0, padded_len - input_ids_length),
    "constant",
    pad_token_id,
)
inputs["attention_mask"] = torch.nn.functional.pad(
    inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
)

for k, v in inputs.items():
    inputs[k] = np.array(v)


vision_inputs = {
    k: v
    for k, v in inputs.items()
    if k in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
}

vision_inputs_fp16 = {"pixel_values", "image_masks"}
vision_inputs.update({k: vision_inputs[k].astype("float16") for k in vision_inputs_fp16 if k in vision_inputs})

vision_start = perf_counter()
vision_outputs = {}
if vision_inputs:
    vision_outputs = vision_session.run(vision_inputs)
vision_end = perf_counter()

lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
if "position_ids" in inputs:
    lang_inputs["position_ids"] = inputs["position_ids"]
    lang_inputs.pop("attention_mask")
else:
    lang_inputs["position_ids"] = np.where(
        lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
    )  # Need to use -1 as position_ids for invalid tokens

lang_inputs["image_idx"] = np.array([[0]])

if not skip_vision:
    lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]

# RUN prefill
lang_start = perf_counter()
lang_prefill_session.set_buffers(vision_outputs)

all_outputs = []
chunk_inputs = lang_inputs.copy()
for i in range(num_chunks):
    chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    outputs = lang_prefill_session.run(chunk_inputs)
    _update_retained_states(chunk_inputs, outputs)
    chunk_inputs["image_idx"] = outputs["image_idx_output"]
prefill_time = perf_counter() - lang_start + vision_end - vision_start
print(f"Prefill time : {prefill_time:.2f} secs")

all_outputs.append(np.argmax(outputs["logits"]))
decode_inputs = {
    "input_ids": np.argmax(outputs["logits"]).reshape(1, 1),
    "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
}

_update_retained_states(decode_inputs, outputs)

decode_inputs["image_idx"] = outputs["image_idx_output"]

if not skip_vision:
    decode_inputs["vision_embeds"] = outputs["vision_embeds_RetainedState"]

st = perf_counter()
decode_out = lang_decode_session.run(decode_inputs)
print(f"time for first run of decode with KV as input = {perf_counter() - st} sec\n")

all_outputs.append(np.argmax(decode_out["logits"]))
pos_id = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
    "position_ids": pos_id,
}

_update_retained_states(loop_decode_inputs, decode_out)

loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]

if not skip_vision:
    loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]


st = perf_counter()
for i in range(generation_len - 2):
    decode_out = lang_decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"]))
    pos_id += 1
    _update_retained_states(loop_decode_inputs, decode_out)
    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
    )
ft = perf_counter()
print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
print(f"\noutput\n{tokenizer.decode(all_outputs)}")
