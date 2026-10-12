# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

BLOCKING_MODE_REQUIRED_ARGS = {
    "kv": ("num_kv_blocks",),
    "kv_headpar": ("num_kv_blocks",),
    "q": ("num_q_blocks",),
    "h": ("head_block_size",),
    "qkv": ("num_kv_blocks", "num_q_blocks"),
    "hkv": ("head_block_size", "num_kv_blocks"),
    "hqkv": ("head_block_size", "num_kv_blocks", "num_q_blocks"),
    "bhqkv": ("head_block_size", "num_kv_blocks", "num_q_blocks", "num_batch_blocks"),
}


def build_blocking_config(args):
    blocking_mode = args.blocking_mode.lower()
    if blocking_mode not in BLOCKING_MODE_REQUIRED_ARGS:
        raise ValueError(f"Unsupported blocking mode: {args.blocking_mode}")

    qaic_config = {"blocking_mode": blocking_mode}
    for arg_name in BLOCKING_MODE_REQUIRED_ARGS[blocking_mode]:
        qaic_config[arg_name] = getattr(args, arg_name)
    return qaic_config


def assert_generated_tokens_match(reference_exec_info, blocked_exec_info, label):
    reference_ids = np.asarray(reference_exec_info.generated_ids)
    blocked_ids = np.asarray(blocked_exec_info.generated_ids)
    if np.array_equal(reference_ids, blocked_ids):
        print(f"Token comparison ({label} vs non-blocked): MATCH")
        print(f"Generated token IDs: {blocked_ids.tolist()}")
        return

    mismatch = np.argwhere(reference_ids != blocked_ids)
    first_mismatch = tuple(mismatch[0].tolist()) if mismatch.size else None
    raise AssertionError(
        f"Token comparison ({label} vs non-blocked): MISMATCH at {first_mismatch}; "
        f"non-blocked={reference_ids.tolist()}, blocked={blocked_ids.tolist()}"
    )


def main():
    parser = argparse.ArgumentParser(description="Basic text generation inference")
    parser.add_argument("--model-name", type=str, default="openai/gpt-oss-20b", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="Hello", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=1, help="Prefill sequence length")
    parser.add_argument(
        "--ctx-len", type=int, default=2048, help="Context length high enough to force blocking computation"
    )
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=[36, 37, 38, 39, 40, 41, 42, 43],
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    parser.add_argument(
        "--blocking-mode",
        type=str,
        default="q",
        help="Blocking mode, valid options: kv, kv_headpar, q, h, qkv, hkv, hqkv, bhqkv",
    )
    parser.add_argument("--num-q-blocks", type=int, default=2, help="Number of query blocks for q/qkv/hqkv/bhqkv modes")
    parser.add_argument(
        "--num-kv-blocks", type=int, default=2, help="Number of KV blocks for kv/kv_headpar/qkv/hkv/hqkv/bhqkv modes"
    )
    parser.add_argument("--head-block-size", type=int, default=16, help="Number of attention heads per head block")
    parser.add_argument("--num-batch-blocks", type=int, default=1, help="Number of batch blocks for bhqkv mode")
    parser.add_argument("--num-devices", type=int, default=4, help="Number of devices to compile for")
    parser.add_argument(
        "--compare-non-blocking",
        action="store_true",
        help="Compile and print results for non-blocked version of model as well",
    )
    parser.add_argument(
        "--subf",
        action="store_true",
        help="If flag is passed, onnx export is done with subfunction",
    )
    args = parser.parse_args()

    # Load tokenizer and model
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    qaic_config = build_blocking_config(args)
    print(
        "Blocking config: "
        f"qaic_config={qaic_config}, "
        f"attention_heads={config.num_attention_heads}, "
        f"kv_heads={config.num_key_value_heads}"
    )

    npi_file_path = None
    if args.model_name == "openai/gpt-oss-120b" and args.subf:
        npi_file_path = os.path.join("examples/disagg_serving/", "subfunction_120b_npi.yaml")
    elif args.model_name == "openai/gpt-oss-120b":
        npi_file_path = os.path.join("examples/disagg_serving/", "non_subfunction_120b_npi.yaml")

    common_compile_kwargs = {
        "prefill_seq_len": args.prefill_seq_len,
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "num_devices": args.num_devices,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "retain_full_kv": True,
        "use_onnx_subfunctions": args.subf,
    }
    if npi_file_path is not None:
        common_compile_kwargs["node_precision_info"] = npi_file_path

    if args.compare_non_blocking:
        if args.num_layers:
            model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, num_hidden_layers=args.num_layers)
        else:
            model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name)

        # Compile the model
        qpc_path = model.compile(**common_compile_kwargs)
        print(f"Model compiled to: {qpc_path}")

        # Generate text
        exec_info = model.generate(
            tokenizer=tokenizer,
            prompts=[args.prompt],
            generation_len=args.generation_len,
        )

        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {exec_info.generated_texts[0]}")

    if args.num_layers:
        model_blocked = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, num_hidden_layers=args.num_layers)
    else:
        model_blocked = QEFFAutoModelForCausalLM.from_pretrained(args.model_name)

    # model_blocked._offload_model_weights(True)

    # Compile the model
    qpc_path_blocked = model_blocked.compile(
        qaic_config=qaic_config,
        user_tiled=True,
        **common_compile_kwargs,
    )
    print(f"Model compiled to: {qpc_path_blocked}")

    # Generate text
    exec_info_blocked = model_blocked.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        generation_len=args.generation_len,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info_blocked.generated_texts[0]}")

    if args.compare_non_blocking:
        assert_generated_tokens_match(exec_info, exec_info_blocked, qaic_config["blocking_mode"])
        print("Performance non-blocked:")
        print(exec_info)

    print(f"Performance blocked ({qaic_config['blocking_mode']}):")
    print(exec_info_blocked)


if __name__ == "__main__":
    main()
