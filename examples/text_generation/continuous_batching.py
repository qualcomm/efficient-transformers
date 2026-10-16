# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Continuous batching inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument(
        "--prompts",
        type=str,
        default="Hello! How can I help?|Hi there! What’s up?|Hey! Need assistance?|Welcome! How can I support you today?",
        help="Pipe-separated prompts for batch processing",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=128, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=512, help="Context length")
    parser.add_argument("--num-hidden-layers", type=int, default=4, help="Num hidden layers")
    parser.add_argument("--full-batch-size", type=int, default=4, help="Full batch size for continuous batching")
    parser.add_argument("--dynamo", action="store_true", help="Export via dynamo")
    parser.add_argument(
        "--weight-free",
        action="store_true",
        help="Export without embedding weights; useful with --artifacts for a replayable compiler bundle",
    )
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Write compiler and runner artifacts without executing either tool",
    )
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    args = parser.parse_args()

    # Parse prompts
    prompt_list = args.prompts.split("|")
    print(f"Processing {len(prompt_list)} prompts with continuous batching")

    # Load tokenizer and model with continuous batching enabled
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        num_hidden_layers=args.num_hidden_layers,
        continuous_batching=True,
        weight_free=args.weight_free,
    )

    # Compile the model with full_batch_size for continuous batching
    compile_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        full_batch_size=args.full_batch_size,
        num_cores=args.num_cores,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        dynamo=args.dynamo,
        use_onnx_subfunctions=True,
        artifacts=args.artifacts,
    )
    if args.artifacts:
        print(f"Compiler artifacts written to: {compile_path}")
        if args.weight_free:
            print("Weight-free compiler bundle references checkpoint weights via AIC_EXTERNAL_DATA_ROOT.")
    else:
        print(f"Model compiled to: {compile_path}")

    # Generate text for all prompts
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=prompt_list,
        device_ids=args.device_group,
        generation_len=args.generation_len,
        artifacts=args.artifacts,
    )

    if args.artifacts:
        print(f"Runner inputs written to: {exec_info}")
        return

    # Display results
    print("\n" + "=" * 80)
    for i, (prompt, generated) in enumerate(zip(prompt_list, exec_info.generated_texts)):
        print(f"\nPrompt {i + 1}: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 80)


if __name__ == "__main__":
    main()
