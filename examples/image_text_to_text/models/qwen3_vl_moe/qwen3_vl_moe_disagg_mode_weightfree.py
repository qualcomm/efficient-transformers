# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Basic Qwen3-VL-MoE disaggregated inference without attention blocking."""

import argparse
from time import perf_counter

import numpy as np
import requests
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
VISION_INPUTS = {
    "pixel_values",
    "image_grid_thw",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_INPUTS = {"pixel_values", "image_masks"}
VISION_OUTPUTS = ("vision_embeds", "deepstack_features")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--image-url", default="https://picsum.photos/id/237/536/354")
    parser.add_argument("--prompt", default="Describe all the colors seen in the image.")
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--generation-len", type=int, default=100)
    parser.add_argument(
        "--num-cores",
        type=int,
        default=4,
        help="Cores per QPC; the default allows all three QPCs to fit on one 16-core device.",
    )
    parser.add_argument("--prefill-num-devices", type=int, default=1)
    parser.add_argument("--decode-num-devices", type=int, default=1)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Optionally truncate the text model to this many layers for a lightweight run.",
    )
    return parser.parse_args()


def update_retained_states(target_inputs, source_outputs, num_hidden_layers):
    for layer_idx in range(num_hidden_layers):
        target_inputs[f"past_key.{layer_idx}"] = source_outputs[f"past_key.{layer_idx}_RetainedState"]
        target_inputs[f"past_value.{layer_idx}"] = source_outputs[f"past_value.{layer_idx}_RetainedState"]


def get_next_token_ids(logits):
    return np.asarray(logits)[:, -1, :].argmax(axis=-1).astype(np.int64)


def main():
    args = parse_args()
    batch_size = 1

    image = Image.open(requests.get(args.image_url, stream=True, timeout=30).raw).convert("RGB")
    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        weight_free=True,
    )
    num_hidden_layers = qeff_model.model.config.text_config.num_hidden_layers
    processor = AutoProcessor.from_pretrained(args.model_id)

    vision_qpc = qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        height=image.height,
        width=image.width,
        num_cores=args.num_cores,
        num_devices=1,
        mos=1,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        split_model_io=True,
        skip_vision=False,
        skip_lang=True,
        use_onnx_subfunctions=True,
    )
    prefill_qpc = qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        height=image.height,
        width=image.width,
        num_cores=args.num_cores,
        num_devices=args.prefill_num_devices,
        mos=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        retain_full_kv=True,
        split_model_io=True,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )
    decode_qpc = qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        height=image.height,
        width=image.width,
        num_cores=args.num_cores,
        num_devices=args.decode_num_devices,
        mos=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        split_model_io=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )

    print(f"Vision QPC: {vision_qpc['vision_qpc_path']}")
    print(f"Language prefill QPC: {prefill_qpc['lang_prefill_qpc_path']}")
    print(f"Language decode QPC: {decode_qpc['lang_decode_qpc_path']}")

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
    ]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=args.prefill_seq_len,
        batch_size=batch_size,
    )

    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -args.prefill_seq_len)
    padded_len = num_chunks * args.prefill_seq_len
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 1
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"],
        (0, padded_len - input_ids_length),
        "constant",
        0,
    )
    inputs = {name: np.asarray(value) for name, value in inputs.items()}

    vision_session = QAICInferenceSession(vision_qpc["vision_qpc_path"])
    prefill_session = QAICInferenceSession(prefill_qpc["lang_prefill_qpc_path"])
    decode_session = QAICInferenceSession(decode_qpc["lang_decode_qpc_path"])
    try:
        vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUTS}
        vision_inputs.update(
            {name: vision_inputs[name].astype(np.float16) for name in VISION_FP16_INPUTS if name in vision_inputs}
        )

        prefill_start = perf_counter()
        vision_outputs = vision_session.run(vision_inputs)
        vision_session.deactivate()

        lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
        lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        for output_name in VISION_OUTPUTS:
            if output_name in vision_outputs:
                lang_inputs[output_name] = vision_outputs[output_name]

        prefill_session.set_buffers(vision_outputs)
        chunk_inputs = lang_inputs.copy()
        outputs = None
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * args.prefill_seq_len
            chunk_end = (chunk_idx + 1) * args.prefill_seq_len
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, chunk_start:chunk_end]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., chunk_start:chunk_end]
            outputs = prefill_session.run(chunk_inputs)
            update_retained_states(chunk_inputs, outputs, num_hidden_layers)
            chunk_inputs["image_idx"] = outputs["image_idx_output"]

        print(f"Vision + language prefill time: {perf_counter() - prefill_start:.3f}s")
        prefill_session.deactivate()

        generated_ids = [get_next_token_ids(outputs["logits"])]
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(batch_size, 1),
            "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
        }
        update_retained_states(decode_inputs, outputs, num_hidden_layers)

        decode_start = perf_counter()
        for _ in range(args.generation_len - 1):
            outputs = decode_session.run(decode_inputs)
            generated_ids.append(get_next_token_ids(outputs["logits"]))
            decode_inputs["input_ids"] = generated_ids[-1].reshape(batch_size, 1)
            decode_inputs["position_ids"] += 1
            update_retained_states(decode_inputs, outputs, num_hidden_layers)
        decode_time = perf_counter() - decode_start

        generated_ids = np.stack(generated_ids, axis=1)
        print(f"Decode throughput: {(args.generation_len - 1) / decode_time:.2f} tok/s")
        print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
    finally:
        vision_session.deactivate()
        prefill_session.deactivate()
        decode_session.deactivate()


if __name__ == "__main__":
    main()
