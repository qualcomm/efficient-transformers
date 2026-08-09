# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Export, compile, and run DeepSeek-V4-Flash in one-token decode mode."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_HF_CACHE = "/home/huggingface_hub"
DEFAULT_ARTIFACT_ROOT = "/home/ochougul/qeff_oc/qeff_artifacts/deepseek_v4_flash_full_decode"
PREFILL_PROMPT = "<replace with the prefill prompt>"


def parse_device_group(value: str) -> list[int]:
    return [int(device_id) for device_id in value.strip("[]").split(",") if device_id.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache", type=Path, default=Path(DEFAULT_HF_CACHE))
    parser.add_argument("--artifact-root", type=Path, default=Path(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--generation-len", type=int, default=250)
    parser.add_argument("--num-hidden-layers", type=int, default=43)
    parser.add_argument("--num-cores", type=int, default=12)
    parser.add_argument("--device-group", type=parse_device_group, default=[i for i in range(12)])
    parser.add_argument("--prefill-prompt", default=PREFILL_PROMPT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--automation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ctx_len < 2:
        raise ValueError("ctx_len must be at least 2.")
    if not 1 <= args.generation_len < args.ctx_len:
        raise ValueError("generation_len must be in [1, ctx_len).")
    if args.num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be at least 1.")
    if not args.device_group:
        raise ValueError("device_group must contain at least one QAIC device ID.")

    os.environ["HF_HUB_CACHE"] = str(args.hf_cache)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ.setdefault("QEFF_HOME", str(args.artifact_root))

    export_root = args.artifact_root / "onnx"
    compile_root = args.artifact_root / "compile"
    export_root.mkdir(parents=True, exist_ok=True)
    compile_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} with Transformers in float16")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.hf_cache,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(
        args.model_id,
        cache_dir=args.hf_cache,
        local_files_only=args.local_files_only,
    )
    if args.num_hidden_layers > config.num_hidden_layers:
        raise ValueError(f"num_hidden_layers cannot exceed the checkpoint's {config.num_hidden_layers} layers.")
    config.num_hidden_layers = args.num_hidden_layers
    config.layer_types = config.layer_types[: args.num_hidden_layers]
    config.mlp_layer_types = config.mlp_layer_types[: args.num_hidden_layers]

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.hf_cache,
        config=config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=args.local_files_only,
    ).eval()

    print("Applying QEfficient replacements to the loaded Transformers model")
    qeff_model = QEFFAutoModelForCausalLM(hf_model)
    qeff_model.model.to(dtype=torch.float16)

    print("Exporting the one-token decode graph through qeff_model.export()")
    onnx_path = Path(
        qeff_model.export(
            export_dir=str(export_root),
            prefill_only=False,
            use_onnx_subfunctions=False,
            offload_pt_weights=True,
        )
    )
    print(f"ONNX_PATH={onnx_path}")

    print("Compiling retained-state decode specialization: seq_len=1, ctx_len=%d" % args.ctx_len)
    qpc_path = Path(
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(compile_root),
            prefill_seq_len=1,
            ctx_len=args.ctx_len,
            batch_size=1,
            num_cores=args.num_cores,
            num_devices=len(args.device_group),
            prefill_only=False,
            use_onnx_subfunctions=False,
            mxint8_kv_cache=False,
            mxfp6_matmul=True,
        )
    )
    print(f"QPC_PATH={qpc_path}")
    print(f"SPECIALIZATIONS_PATH={qpc_path.parent / 'specializations.json'}")
    print(f"CUSTOM_IO_PATH={qpc_path.parent / 'custom_io.yaml'}")

    print("Running qeff_model.generate()")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[args.prefill_prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
        automation=args.automation,
        stream=False,
    )

    result_path = args.artifact_root / "generation_result.json"
    result = {
        "model_id": args.model_id,
        "prefill_prompt": args.prefill_prompt,
        "onnx_path": str(onnx_path),
        "qpc_path": str(qpc_path),
        "generated_texts": exec_info.generated_texts,
        "generated_ids": [ids.tolist() if hasattr(ids, "tolist") else ids for ids in exec_info.generated_ids],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"RESULT_PATH={result_path}")
    print(f"GENERATED_TEXT={exec_info.generated_texts}")


if __name__ == "__main__":
    main()
