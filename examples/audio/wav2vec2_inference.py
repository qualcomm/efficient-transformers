# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import torch
from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForCTC


def main():
    parser = argparse.ArgumentParser(description="CTC speech recognition inference with Wav2Vec2")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="HuggingFace CTC model ID (e.g., Wav2Vec2)",
    )

    parser.add_argument("--num_cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=480000, help="Context length for generation")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices")
    parser.add_argument(
        "--weight-free",
        action="store_true",
        help="Build the model on meta tensors and load weights at compile time",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Build the model on meta tensors and load weights at compile time",
    )
    parser.add_argument(
        "--use-onnx-subfunction",
        action="store_true",
        help="Build the model on meta tensors and load weights at compile time",
    )
    args = parser.parse_args()

    print(f"Loading CTC model: {args.model_name}")

    ## STEP 1 -- load audio sample
    # Using a standard english dataset
    print("Loading audio sample from dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    data = [ds[i]["audio"]["array"] for i in range(args.batch_size)]

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    ## STEP 2 -- Load the model
    model = QEFFAutoModelForCTC.from_pretrained(
        args.model_name, torch_dtype=torch.float32, weight_free=args.weight_free
    )

    ## STEP 3 -- Compile the model
    model.compile(
        batch_size=args.batch_size,
        num_devices=args.num_devices,
        seq_len=args.seq_len,
        num_cores=args.num_cores,
        dynamo=True if args.weight_free else args.dynamo,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
    )

    ## STEP 4 -- Run the model and generate the output
    model_output = model.generate(processor, inputs=data)
    print(f"\nTranscription: {model_output}")


if __name__ == "__main__":
    main()
