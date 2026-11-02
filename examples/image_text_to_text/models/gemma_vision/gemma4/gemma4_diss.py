# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from pathlib import Path
from time import perf_counter

# from qwen_vl_utils import process_vision_info
import numpy as np
import torch
import transformers
from gemma4_utils import (
    CHAT_TEMPLATE,
    build_messages,
    remove_fp16clip_transform_if_disabled,
)
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

model_id = "google/gemma-4-26B-A4B-it"
config = AutoConfig.from_pretrained(model_id)

# For faster execution user can run with lesser layers, For Testing Purpose Only
config.text_config.num_hidden_layers = 2
config.vision_config.num_hidden_layers = 2

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config, dtype="float32", trust_remote_code=True
)


tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

ENABLE_FP16_CLIP = True
remove_fp16clip_transform_if_disabled(qeff_model, ENABLE_FP16_CLIP)
PREFILL_SEQ_LEN = 256
CTX_LEN = 4096
BS = 1
gemma_vision_dir = Path(__file__).resolve().parent.parent
# vision_npi_file = str(gemma_vision_dir / "configs" / "fp32_nodes_gemma4_26B_A4B_it_vision_diss.yaml")


skip_vision = False
if not skip_vision:
    vision_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=1,
        mos=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        skip_vision=skip_vision,
        split_model_io=True,
        skip_lang=True,
    )
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
    mos=1,
    aic_enable_depth_first=True,
    node_precision_info=True,
    prefill_only=True,
    enable_chunking=True,
    qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 256}},
    use_onnx_subfunctions=True,
    skip_vision=True,
)

decode_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=16,
    num_devices=1,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    split_model_io=True,
    mos=1,
    aic_enable_depth_first=True,
    node_precision_info=True,
    prefill_only=False,
    use_onnx_subfunctions=True,
    skip_vision=True,
)


def _resolve_lang_qpc_path(qpc_obj, preferred_keys):
    if isinstance(qpc_obj, dict):
        for key in preferred_keys:
            if key in qpc_obj:
                return qpc_obj[key]
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_obj.keys())}")
    if isinstance(qpc_obj, (list, tuple)):
        # Backward-compat: some codepaths return (vision_qpc, lang_qpc)
        return qpc_obj[1]
    return qpc_obj


def _resolve_vision_qpc_path(qpc_obj, preferred_keys=("vision_qpc_path",)):
    if isinstance(qpc_obj, dict):
        for key in preferred_keys:
            if key in qpc_obj:
                return qpc_obj[key]
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_obj.keys())}")
    if isinstance(qpc_obj, (list, tuple)):
        # Backward-compat: some codepaths return (vision_qpc, lang_qpc)
        return qpc_obj[0]
    return qpc_obj


lang_prefill_qpc = _resolve_lang_qpc_path(prefill_qpc_path, ("lang_prefill_qpc_path", "lang_qpc_path"))
lang_decode_qpc = _resolve_lang_qpc_path(decode_qpc_path, ("lang_decode_qpc_path", "lang_qpc_path"))

lang_prefill_session = QAICInferenceSession(lang_prefill_qpc)
lang_decode_session = QAICInferenceSession(lang_decode_qpc)
MODEL_ID = "google/gemma-4-26B-A4B-it"
SYSTEM_PROMPT = "You are a helpful assistant."
TEXT_PROMPT = "Tell me about Taj Mahal?"
IMAGE_PROMPT = "Can you Describe this image in detail?"
IMAGE_URL = "https://wallup.net/wp-content/uploads/2017/03/28/351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"
chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE
if skip_vision:
    messages = build_messages(SYSTEM_PROMPT, TEXT_PROMPT, use_image=False)
    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
else:
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

    vision_qpc = _resolve_vision_qpc_path(vision_qpc_path)
    vision_session = QAICInferenceSession(vision_qpc)
pad_token_id = 1
input_len = inputs["attention_mask"].sum(1, keepdims=True)
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len
generation_len = 200
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
    if k
    in {
        "pixel_values",
        "image_position_ids",
        "image_masks",
        "image_input_idx",
        "valid_idx",
        "aspect_ratio_ids",
        "aspect_ratio_mask",
    }
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
    chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
        ..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN
    ]

    outputs = lang_prefill_session.run(chunk_inputs)
    for i in range(config.text_config.num_hidden_layers):
        chunk_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
        chunk_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]
    chunk_inputs["image_idx"] = outputs["image_idx_output"]
prefill_time = perf_counter() - lang_start + vision_end - vision_start
print(f"Prefill time : {prefill_time:.2f} secs")
all_outputs.append(np.argmax(outputs["logits"]))
decode_inputs = {
    "input_ids": np.argmax(outputs["logits"]).reshape(1, 1),
    "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
}
for i in range(config.text_config.num_hidden_layers):
    decode_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
    decode_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]
decode_inputs["image_idx"] = outputs["image_idx_output"]
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

for i in range(config.text_config.num_hidden_layers):
    loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
    loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]
loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]
loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]

st = perf_counter()
for i in range(generation_len - 2):
    decode_out = lang_decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"]))
    pos_id += 1
    for j in range(config.text_config.num_hidden_layers):
        loop_decode_inputs[f"past_key.{j}"] = decode_out[f"past_key.{j}_RetainedState"]
        loop_decode_inputs[f"past_value.{j}"] = decode_out[f"past_value.{j}_RetainedState"]
    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
    )
ft = perf_counter()
print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
print(f"\noutput\n{tokenizer.decode(all_outputs)}")
