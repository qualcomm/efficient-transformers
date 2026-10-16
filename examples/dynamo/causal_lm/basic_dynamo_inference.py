# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo-based export and inference for Causal LM models on Cloud AI 100.

Requires PyTorch >= 2.13. Install dependencies before running:
    pip install -r examples/dynamo/causal_lm/requirements.txt
"""

import argparse
import time
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants


def _path_size_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _format_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0


def _build_qaic_config(args):
    if args.blocking_mode is None:
        return None

    qaic_config = {"blocking_mode": args.blocking_mode}
    optional_blocking_args = (
        ("num_kv_blocks", args.num_kv_blocks),
        ("num_q_blocks", args.num_q_blocks),
        ("head_block_size", args.head_block_size),
        ("headpar_split", args.headpar_split),
    )
    for key, value in optional_blocking_args:
        if value is not None:
            qaic_config[key] = value
    if args.disable_skip_kv:
        qaic_config["skip_kv"] = False
    return qaic_config


def main():
    parser = argparse.ArgumentParser(
        description="Dynamo-based export and inference for Causal LM models on Cloud AI 100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="openai/gpt-oss-20b", help="HuggingFace model ID")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Override number of hidden layers")
    parser.add_argument("--prompt", type=str, default="My name is", help="Input prompt for generation")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context (KV-cache) length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES, help="Number of AI 100 cores")
    parser.add_argument(
        "--aic-hw-version", type=str, default=constants.DEFAULT_AIC_HW_VERSION, help="AIC hardware version"
    )
    parser.add_argument(
        "--weight-free",
        action="store_true",
        help="Build the model on meta tensors and load weights at compile time",
    )
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated), e.g. [0,1]",
    )
    parser.add_argument(
        "--blocking-mode",
        choices=("kv", "qkv", "hqkv", "kv_headpar"),
        default=None,
        help="Enable CausalLM attention blocking through qaic_config",
    )
    parser.add_argument("--num-kv-blocks", type=int, default=None, help="Number of K/V cache blocks")
    parser.add_argument("--num-q-blocks", type=int, default=None, help="Number of query blocks")
    parser.add_argument("--head-block-size", type=int, default=None, help="Number of attention heads per head block")
    parser.add_argument("--headpar-split", type=int, default=None, help="Head-parallel split count for kv_headpar")
    parser.add_argument(
        "--disable-skip-kv",
        action="store_true",
        help="Set qaic_config['skip_kv']=False for blocking modes",
    )
    args = parser.parse_args()
    qaic_config = _build_qaic_config(args)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        weight_free=args.weight_free,
        qaic_config=qaic_config,
    )

    # Export (via torch.export / dynamo) + compile to QPC
    compile_start = time.perf_counter()
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        aic_hw_version=args.aic_hw_version,
        mxfp6_matmul=True,
        num_devices=(4 if args.device_group is None else len(args.device_group)),
        dynamo=True,
        use_onnx_subfunctions=True,
        qaic_config=qaic_config,
    )
    compile_time_seconds = time.perf_counter() - compile_start
    qpc_size_bytes = _path_size_bytes(qpc_path)
    print(f"Model compiled to: {qpc_path}")
    print(f"QPC compile time: {compile_time_seconds:.2f} seconds")
    print(f"QPC size: {_format_size(qpc_size_bytes)} ({qpc_size_bytes} bytes)")

    # Run inference on device
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_ids=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt   : {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
