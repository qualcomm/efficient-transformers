# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq


def main():
    parser = argparse.ArgumentParser(description="Speech-to-text inference with Whisper")
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/whisper-tiny",
        help="HuggingFace Whisper model ID",
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=25,
        help="Context length for generation",
    )
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument(
        "--weight-free",
        action="store_true",
        help="Build the model on meta tensors and load weights at compile time",
    )
    args = parser.parse_args()

    print(f"Loading Whisper model: {args.model_name}")

    ## STEP 1 -- load audio sample

    # Using a standard english dataset
    print("Loading audio sample from dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_rate = ds[0]["audio"]["sampling_rate"]
    data = ds[0]["audio"]["array"]

    # Reshape so shape corresponds to data with batch size 1
    data = data.reshape(-1)

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    ## STEP 2 -- init base model
    qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(args.model_name, weight_free=args.weight_free)

    ## STEP 3 -- export and compile model
    qeff_model.compile(num_cores=args.num_cores, dynamo=args.weight_free)

    ## STEP 4 -- generate output for loaded input and processor
    exec_info = qeff_model.generate(
        inputs=processor(data, sampling_rate=sample_rate, return_tensors="pt"), generation_len=args.ctx_len
    )

    ## STEP 5 -- use processor to decode output
    transcription = processor.batch_decode(exec_info.generated_ids)[0]
    print(f"\nTranscription: {transcription}")


if __name__ == "__main__":
    main()
