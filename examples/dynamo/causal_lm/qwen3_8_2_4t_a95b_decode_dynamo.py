# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Decode-oriented dynamo export example for Qwen3.8-2.4T-A95B.

The full target checkpoint is intended for weight-free export. For regular
dynamo export without weight-free, use ``--synthetic-tiny`` so no full
checkpoint weights are loaded.
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants

MODEL_ID = "Qwen/Qwen3.8-2.4T-A95B"
RANDOM_SEED = 42
SYNTHETIC_TINY_DTYPE = torch.bfloat16

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _set_model_dtype_config(qeff_model, dtype: torch.dtype) -> None:
    qeff_model.model.config.dtype = dtype
    qeff_model.model.config.torch_dtype = dtype


def _tiny_qwen3_5_moe_text_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=128,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
        },
        dtype=SYNTHETIC_TINY_DTYPE,
    )


def _default_synthetic_checkpoint_dir() -> Path:
    qeff_home = Path(os.environ.get("QEFF_HOME", Path.cwd() / ".qeff" / "qwen3_8_decode"))
    return qeff_home / "synthetic_tiny_checkpoint"


def _ensure_synthetic_tiny_checkpoint(config: Qwen3_5MoeTextConfig, checkpoint_dir: Path) -> Path:
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        checkpoint_config = Qwen3_5MoeTextConfig.from_pretrained(checkpoint_dir)
        if checkpoint_config.to_dict() != config.to_dict():
            shutil.rmtree(checkpoint_dir)

    if (checkpoint_dir / "model.safetensors").exists() or (checkpoint_dir / "model.safetensors.index.json").exists():
        return checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hf_model = Qwen3_5MoeForCausalLM(config).eval().to(SYNTHETIC_TINY_DTYPE)
    hf_model.save_pretrained(checkpoint_dir, safe_serialization=True)
    return checkpoint_dir


def _load_qeff_model(args):
    torch.manual_seed(RANDOM_SEED)
    if args.synthetic_tiny:
        config = _tiny_qwen3_5_moe_text_config()
        if args.weight_free:
            checkpoint_dir = _ensure_synthetic_tiny_checkpoint(
                config,
                args.synthetic_checkpoint_dir or _default_synthetic_checkpoint_dir(),
            )
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir),
                config=config,
                weight_free=True,
                dtype=SYNTHETIC_TINY_DTYPE,
                trust_remote_code=True,
            )
            _set_model_dtype_config(qeff_model, SYNTHETIC_TINY_DTYPE)
            qeff_model.model.eval()
            return qeff_model, None

        hf_model = Qwen3_5MoeForCausalLM(config).eval().to(SYNTHETIC_TINY_DTYPE)
        return QEFFAutoModelForCausalLM(hf_model), None

    if args.model_name == MODEL_ID and not args.weight_free and not args.allow_full_weight_load:
        raise ValueError(
            "Regular dynamo export for the full Qwen3.8-2.4T checkpoint would load full weights. "
            "Pass --weight-free for the target model, or pass --synthetic-tiny for regular dynamo export."
        )

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
        if hasattr(config, "layer_types"):
            config.layer_types = config.layer_types[: args.num_hidden_layers]

    load_kwargs = {
        "config": config,
        "weight_free": args.weight_free,
        "trust_remote_code": True,
    }
    if getattr(config, "dtype", None) is not None:
        load_kwargs["dtype"] = config.dtype

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        **load_kwargs,
    )
    if args.weight_free and getattr(config, "dtype", None) is not None:
        _set_model_dtype_config(qeff_model, config.dtype)
    qeff_model.model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    return qeff_model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Dynamo export/compile for Qwen3.8-2.4T-A95B decode graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default=MODEL_ID, help="Hugging Face model ID or local path")
    parser.add_argument("--export-dir", type=Path, default=None, help="Optional ONNX export directory")
    parser.add_argument("--compile-dir", type=Path, default=None, help="Optional QPC compile directory")
    parser.add_argument("--ctx-len", type=int, default=4096, help="Context length")
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES, help="Number of AI cores")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices for compile")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Debug-only layer limit for non-tiny models")
    parser.add_argument("--synthetic-tiny", action="store_true", help="Use a local 4-layer Qwen3.5-MoE text model")
    parser.add_argument("--weight-free", action="store_true", help="Build the model on meta tensors for export")
    parser.add_argument("--use-onnx-subfunctions", action="store_true", help="Export decoder layers as ONNX functions")
    parser.add_argument("--compile", action="store_true", help="Compile after export")
    parser.add_argument(
        "--blocking-mode",
        choices=("kv", "kv_headpar"),
        default=None,
        help="Decode attention blocking mode. Start with kv before trying kv_headpar.",
    )
    parser.add_argument("--num-kv-blocks", type=int, default=2, help="Number of KV blocks for decode blocking")
    parser.add_argument(
        "--headpar-split",
        type=int,
        default=None,
        help="Head-parallel split factor for --blocking-mode kv_headpar",
    )
    parser.add_argument(
        "--synthetic-checkpoint-dir",
        type=Path,
        default=None,
        help="Local checkpoint path used by --synthetic-tiny --weight-free",
    )
    parser.add_argument(
        "--allow-full-weight-load",
        action="store_true",
        help="Allow regular from_pretrained for the full target checkpoint",
    )
    args = parser.parse_args()

    qeff_model, tokenizer = _load_qeff_model(args)
    qaic_config = None
    if args.blocking_mode is not None:
        qaic_config = {
            "blocking_mode": args.blocking_mode,
            "num_kv_blocks": args.num_kv_blocks,
            "ctx_len": args.ctx_len,
        }
        if args.blocking_mode == "kv_headpar" and args.headpar_split is not None:
            qaic_config["headpar_split"] = args.headpar_split

    if args.compile:
        qpc_path = qeff_model.compile(
            compile_dir=str(args.compile_dir) if args.compile_dir else None,
            prefill_seq_len=1,
            ctx_len=args.ctx_len,
            num_cores=args.num_cores,
            num_devices=args.num_devices,
            dynamo=True,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
            qaic_config=qaic_config,
        )
        print(f"Compiled decode QPC: {qpc_path}")
        return

    onnx_path = qeff_model.export(
        export_dir=str(args.export_dir) if args.export_dir else None,
        prefill_seq_len=1,
        dynamo=True,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        offload_pt_weights=not args.synthetic_tiny,
    )
    print(f"Exported decode ONNX: {onnx_path}")
    if tokenizer is None:
        print("Synthetic tiny export completed without tokenizer/generation.")


if __name__ == "__main__":
    main()
