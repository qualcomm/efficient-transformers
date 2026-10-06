# ----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Export, compile, and run a FP8 causal LM with Dynamo ONNX subfunctions.

Before running this example, configure the Hugging Face cache explicitly::

    export HF_HUB_CACHE=/path/to/huggingface/cache
    export HF_HUB_ENABLE_HF_TRANSFER=1

The ``retained`` FP8 mode keeps FP8 weights in ONNX and emits dequantization
nodes. The ``dequantized`` mode converts FP8 weights to regular Linear layers
before export.
One can use the following example command to run this script:
    ``python examples/text_generation/fp8_dynamo_inference.py --model-name Qwen/Qwen3-0.6B-FP8 --dtype fp16 --fp8-mode retained --action all --device-group [0] --output-dir ./qeff_output_fp8``
"""

import argparse
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _parse_device_group(value: str) -> list[int]:
    device_ids = value.strip().strip("[]")
    if not device_ids:
        raise argparse.ArgumentTypeError("device group must contain at least one device ID")
    try:
        return [int(device_id.strip()) for device_id in device_ids.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("device group must be comma-separated integers, such as [0,1]") from error


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def _default_artifact_dir(args: argparse.Namespace) -> Path:
    mode = "fp8_retained" if args.fp8_mode == "retained" else "fp8_dequantized"
    return Path(args.artifact_dir).expanduser() / f"{_safe_model_name(args.model_name)}_{args.dtype}_{mode}"


def _load_model(args: argparse.Namespace):
    compute_dtype = _DTYPE_MAP[args.dtype]
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if args.num_hidden_layers > 0 and hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = args.num_hidden_layers

    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
        dequantize_fp8_weights=args.fp8_mode == "dequantized",
    )
    return model


def _export_model(model, export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model.export(
        export_dir=str(export_dir),
        dynamo=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    print(f"ONNX model: {onnx_path}")
    return Path(onnx_path)


def _compile_model(model, args: argparse.Namespace, onnx_path: Path, compile_dir: Path) -> Path:
    compile_dir.mkdir(parents=True, exist_ok=True)
    qpc_path = model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(compile_dir),
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=len(args.device_group),
        aic_hw_version=args.aic_hw_version,
        dynamo=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    print(f"QPC package: {qpc_path}")
    return Path(qpc_path)


def _run_inference(model, tokenizer, args: argparse.Namespace, qpc_path: Optional[Path] = None) -> None:
    if qpc_path is not None:
        model.qpc_path = qpc_path

    execution = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )
    generated_text = execution.generated_texts[0]
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {generated_text}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dynamo export, compile, and inference for FP8 causal LMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model ID or local model path")
    parser.add_argument(
        "--action",
        choices=("export", "compile", "inference", "all"),
        default="all",
        help="Pipeline stage to run; compile exports first when --onnx-path is omitted",
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPE_MAP), default="fp16", help="Model compute dtype")
    parser.add_argument(
        "--fp8-mode",
        choices=("retained", "dequantized"),
        default="retained",
        help="Keep FP8 weights in ONNX or dequantize them before export",
    )
    parser.add_argument(
        "--artifact-dir",
        default="~/.cache/qeff_examples/fp8_dynamo",
        help="Root directory for mode- and dtype-specific artifacts",
    )
    parser.add_argument("--onnx-path", help="Existing Dynamo ONNX path for compile or inference workflows")
    parser.add_argument("--qpc-path", help="Existing QPC path for inference")
    parser.add_argument("--prompt", default="Explain why the sky is blue.", help="Prompt for inference")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Maximum KV-cache context length")
    parser.add_argument("--generation-len", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Optional model layer-count override")
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES, help="AI 100 core count")
    parser.add_argument(
        "--aic-hw-version",
        default=constants.DEFAULT_AIC_HW_VERSION,
        help="Target Cloud AI hardware version",
    )
    parser.add_argument(
        "--device-group",
        type=_parse_device_group,
        default=[0],
        help="Comma-separated device IDs, such as [0] or [0,1]",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when loading config and model",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    artifact_dir = _default_artifact_dir(args)
    export_dir = artifact_dir / "onnx"
    compile_dir = artifact_dir / "compile"

    if args.action == "inference" and not args.qpc_path:
        raise ValueError("--qpc-path is required when --action=inference")

    model = _load_model(args)
    tokenizer = None
    onnx_path = Path(args.onnx_path).expanduser() if args.onnx_path else None
    qpc_path = Path(args.qpc_path).expanduser() if args.qpc_path else None

    if args.action in ("export", "all"):
        onnx_path = _export_model(model, export_dir)

    if args.action == "compile" and onnx_path is None:
        onnx_path = _export_model(model, export_dir)

    if args.action in ("compile", "all"):
        qpc_path = _compile_model(model, args, onnx_path, compile_dir)

    if args.action in ("inference", "all"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        _run_inference(model, tokenizer, args, qpc_path)


if __name__ == "__main__":
    main()
