#!/usr/bin/env python3
"""Export original safetensors and write a compiler replay bundle.

The bundle keeps the source checkpoint external.  ``weight_spec.json`` points
at the original safetensors directory and the adjacent directory created by
weight-free export is only a symlink.  Use separate bundles for AI100 and
AI200 because AI100 needs an explicit BF16-to-FP16 graph cast while AI200
consumes the BF16 inputs directly.

Examples
--------
AI100, decode::

    HF_HUB_CACHE=/home/huggingface_hub \
    python original_checkpoint_dtype_artifacts.py \
        --target-dtype float16 --hardware ai100 --stage decode \
        --output-dir /tmp/qeff-qwen-original-artifacts/ai100-decode

AI200, expert-parallel prefill::

    HF_HUB_CACHE=/home/huggingface_hub \
    python original_checkpoint_dtype_artifacts.py \
        --target-dtype bfloat16 --hardware ai200 --stage prefill-ep \
        --output-dir /tmp/qeff-qwen-original-artifacts/ai200-prefill-ep

The generated ``qaic-compile.sh`` is the compiler repro command.  Run it from
the artifact directory after checking that the original checkpoint path in
``weight_spec.json`` is visible to the compiler host.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="tiny-random/qwen3-moe")
    parser.add_argument("--target-dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--hardware", choices=("ai100", "ai200"), required=True)
    parser.add_argument("--stage", choices=("decode", "prefill-ep"), default="decode")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill-seq-len", type=int, default=4)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--num-cores", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

    target_dtype = torch.float16 if args.target_dtype == "float16" else torch.bfloat16
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        weight_free=True,
        use_original_checkpoint=True,
        torch_dtype=target_dtype,
    )

    common = {
        "compile_dir": str(args.output_dir / "compile"),
        "prefill_seq_len": args.prefill_seq_len,
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
        "artifacts": True,
        "aic_hw_version": args.hardware,
    }
    if args.stage == "prefill-ep":
        common.update(
            {
                "prefill_only": True,
                "enable_chunking": True,
                "qaic_config": {"moe_config": {"expert_parallel_chunk_size": args.prefill_seq_len}},
            }
        )
    else:
        common["prefill_only"] = False

    artifact_dir = Path(model.compile(**common))
    spec_path = Path(model.weight_spec_path)
    spec = json.loads(spec_path.read_text())
    print(f"ONNX: {model.onnx_path}")
    print(f"WEIGHT_SPEC: {spec_path}")
    print(f"COMPILER_ARTIFACTS: {artifact_dir}")
    print(f"ORIGINAL_CHECKPOINT: {spec['model_id']}")
    print(f"REPLAY: {artifact_dir / 'qaic-compile.sh'}")
    print("CHECKPOINT_COPIES_IN_OUTPUT:", list(args.output_dir.rglob("*.safetensors")))


if __name__ == "__main__":
    main()
