# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run MolmoPoint image-conditioned generation on Cloud AI target hardware."""

import argparse
import os
import tempfile
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

from QEfficient import QEFFAutoModelForImageTextToText


MODEL_ID = "allenai/MolmoPoint-8B"
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
PROMPT = "Describe this image."
IMAGE_SIZE = (378, 378)
PREFILL_SEQ_LEN = 640
DEFAULT_CTX_LEN = 1024
DEFAULT_DEVICE_IDS = "0,1,2,3"

VISION_NODE_PRECISION_INFO = """\
FP32NodeInstanceNames:
  - vision_embeds
"""


def parse_device_ids(value: str) -> list[int]:
    """Parse a comma-separated device list accepted by QEff ``generate``."""
    try:
        device_ids = [int(device.strip()) for device in value.split(",") if device.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("device IDs must be comma-separated integers") from exc
    if not device_ids:
        raise argparse.ArgumentTypeError("at least one device ID is required")
    return device_ids


def load_processor(model_id: str):
    """Load MolmoPoint's remote processor across ProcessorMixin API versions."""
    try:
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding=True)
    except TypeError as exc:
        if "Unexpected keyword argument" not in str(exc):
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
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding=True)
    finally:
        ProcessorMixin.__init__ = original_init


def load_image(image_url: str) -> Image.Image:
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB").resize(IMAGE_SIZE)


def prepare_inputs(
    qeff_model,
    processor,
    image: Image.Image,
    prompt_text: str,
    prefill_seq_len: int,
    batch_size: int,
):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        images=[image] * batch_size,
        text=[prompt] * batch_size,
        padding=True,
        return_tensors="pt",
    )
    if inputs["input_ids"].shape[1] > prefill_seq_len:
        raise ValueError(
            f"processed prompt has {inputs['input_ids'].shape[1]} tokens, "
            f"which exceeds the compiled prefill length {prefill_seq_len}"
        )

    pixel_values, image_token_pooling = qeff_model.model.model.merge_visual_inputs(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_token_pooling=inputs.get("image_token_pooling"),
        image_grids=inputs.get("image_grids"),
        image_num_crops=inputs.get("image_num_crops"),
        pixel_values_videos=inputs.get("pixel_values_videos"),
        video_token_pooling=inputs.get("video_token_pooling"),
        video_grids=inputs.get("video_grids"),
    )
    runtime_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": pixel_values,
        "image_token_pooling": image_token_pooling,
    }
    if "token_type_ids" in inputs:
        runtime_inputs["token_type_ids"] = inputs["token_type_ids"]
    return runtime_inputs


def main():
    parser = argparse.ArgumentParser(description="MolmoPoint VLM inference on Cloud AI target hardware")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--image-url", default=IMAGE_URL)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--prefill-seq-len", type=int, default=PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=parse_device_ids(os.environ.get("DEVICE_GROUP", DEFAULT_DEVICE_IDS)),
    )
    parser.add_argument("--aic-hw-version", default=os.environ.get("AIC_HW_VERSION", "ai100"))
    args = parser.parse_args()

    if args.ctx_len <= args.prefill_seq_len:
        parser.error(f"--ctx-len must be greater than {args.prefill_seq_len}")
    if args.generation_len <= 0:
        parser.error("--generation-len must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    processor = load_processor(args.model_id)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        kv_offload=True,
        trust_remote_code=True,
        dtype=torch.float32,
    )

    vision_npi = Path(tempfile.gettempdir()) / "qeff-molmo-point-vision-node-precision.yaml"
    vision_npi.write_text(VISION_NODE_PRECISION_INFO)
    qpc_paths = qeff_model.compile(
        num_devices=len(args.device_ids),
        num_cores=args.num_cores,
        aic_hw_version=args.aic_hw_version,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        vision_batch_size=args.batch_size,
        num_crops=2,
        num_patches=729,
        pixels_per_patch=588,
        num_image_tokens=392,
        pool_dim=4,
        use_onnx_subfunctions=False,
        offload_pt_weights=False,
        vision_node_precision_info=str(vision_npi),
    )

    image = load_image(args.image_url)
    inputs = prepare_inputs(
        qeff_model,
        processor,
        image,
        args.prompt,
        args.prefill_seq_len,
        args.batch_size,
    )
    output = qeff_model.generate(
        inputs=inputs,
        device_ids=args.device_ids,
        generation_len=args.generation_len,
    )
    generated_ids = output.generated_ids[:, : args.generation_len]

    print(f"QPCs: {qpc_paths}")
    print(f"Generated token IDs: {generated_ids.tolist()}")
    print("Generated text:", processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    print(f"Performance: {output.perf_metrics}")


if __name__ == "__main__":
    main()
