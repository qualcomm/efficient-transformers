# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


def main():
    parser = argparse.ArgumentParser(description="Basic text generation inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Num hidden layers in the model")
    parser.add_argument("--enable-proxy", action="store_true", help="Enable proxy model export mode")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading HF assets")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length")
    parser.add_argument("--dynamo", action="store_true", help="Export via dynamo")
    parser.add_argument(
        "--weight-free",
        action="store_true",
        help="Export without embedding weights; useful with --artifacts for a replayable compiler bundle",
    )
    parser.add_argument("--use-onnx-subfunctions", action="store_true", help="Use subfunctions while exporting")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES, help="Number of cores")
    parser.add_argument(
        "--aic-hw-version", type=str, default=constants.DEFAULT_AIC_HW_VERSION, help="Version of aic hardware"
    )
    parser.add_argument("--compile-only", action="store_true", help="Compile the model and skip on-device generation")
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
    parser.add_argument(
        "--stats-level",
        dest="stats_level",
        type=int,
        default=None,
        help="Level of statistics to collect. Default: None",
    )
    parser.add_argument(
        "--profiling-type",
        dest="profiling_type",
        type=str,
        default=None,
        choices=["latency", "trace", "raw_device_stats", "stats"],
        help="Enable QAIC device profiling via the runtime Program/ExecObj profiling API and capture a "
        "report for the generate() call. Default: disabled",
    )
    parser.add_argument(
        "--profiling-output-dir",
        dest="profiling_output_dir",
        type=str,
        default=None,
        help="Directory to write the profiling report to. Default: <qpc dir>/profiling_output",
    )
    args = parser.parse_args()

    # Load tokenizer and model
    load_kwargs = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **load_kwargs)

    model_kwargs = dict(load_kwargs)
    if args.weight_free:
        model_kwargs["weight_free"] = True
    if args.enable_proxy:
        model_kwargs["enable_proxy"] = True
        if args.num_hidden_layers > 0:
            model_kwargs["num_hidden_layers"] = args.num_hidden_layers
    else:
        config = AutoConfig.from_pretrained(args.model_name, **load_kwargs)
        if args.num_hidden_layers > 0:
            config.num_hidden_layers = args.num_hidden_layers
        model_kwargs["config"] = config

    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.profiling_type is not None:
        if args.stats_level is None:
            logger.warning("Need to set --stats-level to enable profiling. Setting stats_level=100.")
            args.stats_level = 100

    compile_kwargs = {
        "prefill_seq_len": args.prefill_seq_len,
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "aic_hw_version": args.aic_hw_version,
        "num_devices": (1 if args.device_group is None else len(args.device_group)),
        "dynamo": args.dynamo,
        "use_onnx_subfunctions": args.use_onnx_subfunctions,
        "artifacts": args.artifacts,
    }

    if args.stats_level is not None:
        compile_kwargs["stats_level"] = args.stats_level

    # Compile the model
    qpc_path = model.compile(**compile_kwargs)
    if args.artifacts:
        print(f"Compiler artifacts written to: {qpc_path}")
        if args.weight_free:
            print("Weight-free compiler bundle references checkpoint weights via AIC_EXTERNAL_DATA_ROOT.")
    else:
        print(f"Model compiled to: {qpc_path}")
    if args.compile_only:
        return

    # Generate text
    generate_kwargs = {
        "tokenizer": tokenizer,
        "prompts": [args.prompt],
        "device_ids": args.device_group,
        "generation_len": args.generation_len,
        "artifacts": args.artifacts,
    }
    if args.profiling_type is not None:
        generate_kwargs["profiling_type"] = args.profiling_type
        if args.profiling_output_dir is not None:
            generate_kwargs["profiling_output_dir"] = args.profiling_output_dir
    exec_info = model.generate(**generate_kwargs)

    if args.artifacts:
        print(f"Runner inputs written to: {exec_info}")
        return

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")

    if args.profiling_type is not None:
        output_dir = (
            Path(args.profiling_output_dir)
            if args.profiling_output_dir
            else (Path(qpc_path) if Path(qpc_path).is_dir() else Path(qpc_path).parent) / "profiling_output"
        )
        report_files = sorted(p.name for p in output_dir.glob("*") if p.is_file())
        print(f"\nProfiling type: {args.profiling_type}")
        print(f"Profiling report directory: {output_dir}")
        print(f"Profiling report files: {report_files if report_files else '(none found)'}")


if __name__ == "__main__":
    main()
