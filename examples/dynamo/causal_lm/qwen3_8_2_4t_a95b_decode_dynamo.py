# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Decode-oriented dynamo compile and generation example for Qwen3.8-2.4T-A95B.

The full target checkpoint is intended for weight-free export/compile. Use
``--synthetic-tiny`` to validate the same hybrid retained-state decode path with
a small local checkpoint.
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants

MODEL_ID = "Qwen/Qwen3.8-2.4T-A95B"
RANDOM_SEED = 42
DEFAULT_PROMPT = "Hello"
TINY_PAD_TOKEN = "[PAD]"
TINY_UNK_TOKEN = "[UNK]"
TINY_EOS_TOKEN = "[EOS]"
SYNTHETIC_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _parse_device_group(device_ids: str) -> list[int]:
    return [int(device_id) for device_id in device_ids.strip("[]").split(",")]


def _set_model_dtype_config(qeff_model, dtype: torch.dtype) -> None:
    qeff_model.model.config.dtype = dtype
    qeff_model.model.config.torch_dtype = dtype


def _resolve_synthetic_dtype(dtype_name: str, aic_hw_version: str) -> torch.dtype:
    if dtype_name != "auto":
        return SYNTHETIC_DTYPE_MAP[dtype_name]
    return torch.bfloat16 if aic_hw_version == "ai200" else torch.float16


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        TINY_PAD_TOKEN: 0,
        TINY_UNK_TOKEN: 1,
        TINY_EOS_TOKEN: 2,
        "Hello": 3,
    }
    vocab.update({f"token_{idx}": idx for idx in range(len(vocab), 128)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=TINY_UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=TINY_PAD_TOKEN,
        unk_token=TINY_UNK_TOKEN,
        eos_token=TINY_EOS_TOKEN,
    )


def _tiny_qwen3_5_moe_text_config(dtype: torch.dtype) -> Qwen3_5MoeTextConfig:
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
        dtype=dtype,
        pad_token_id=0,
        eos_token_id=2,
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
    hf_model = Qwen3_5MoeForCausalLM(config).eval().to(config.dtype)
    hf_model.save_pretrained(checkpoint_dir, safe_serialization=True)
    return checkpoint_dir


def _load_qeff_model(args):
    torch.manual_seed(RANDOM_SEED)
    if args.synthetic_tiny:
        synthetic_dtype = _resolve_synthetic_dtype(args.synthetic_dtype, args.aic_hw_version)
        config = _tiny_qwen3_5_moe_text_config(synthetic_dtype)
        tokenizer = _tiny_tokenizer()
        if args.weight_free:
            checkpoint_dir = _ensure_synthetic_tiny_checkpoint(
                config,
                args.synthetic_checkpoint_dir or _default_synthetic_checkpoint_dir(),
            )
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir),
                config=config,
                weight_free=True,
                dtype=synthetic_dtype,
                trust_remote_code=True,
            )
            _set_model_dtype_config(qeff_model, synthetic_dtype)
            qeff_model.model.eval()
            return qeff_model, tokenizer

        hf_model = Qwen3_5MoeForCausalLM(config).eval().to(synthetic_dtype)
        return QEFFAutoModelForCausalLM(hf_model), tokenizer

    if args.model_name == MODEL_ID and not args.weight_free:
        raise ValueError(
            "Regular dynamo compile for the full Qwen3.8-2.4T checkpoint would load full weights. "
            "Use the default weight-free path, or pass --synthetic-tiny for regular tiny-model validation."
        )

    from_pretrained_kwargs = {"trust_remote_code": True}
    if args.cache_dir is not None:
        from_pretrained_kwargs["cache_dir"] = str(args.cache_dir)

    config = AutoConfig.from_pretrained(args.model_name, **from_pretrained_kwargs)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
        if hasattr(config, "layer_types"):
            config.layer_types = config.layer_types[: args.num_hidden_layers]

    load_kwargs = {
        "config": config,
        "weight_free": args.weight_free,
        **from_pretrained_kwargs,
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **from_pretrained_kwargs)
    return qeff_model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Dynamo compile and decode generation for Qwen3.8-2.4T-A95B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default=MODEL_ID, help="Hugging Face model ID or local path")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Hugging Face cache directory for downloaded files")
    parser.add_argument("--compile-dir", type=Path, default=None, help="Optional QPC compile directory")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Input prompt for generation")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt batch size")
    parser.add_argument("--prefill-seq-len", type=int, default=1, help="Decode-only prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=4096, help="Context length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES, help="Number of AI cores")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices for compile")
    parser.add_argument(
        "--aic-hw-version", type=str, default=constants.DEFAULT_AIC_HW_VERSION, help="AIC hardware version"
    )
    parser.add_argument(
        "--device-group",
        type=_parse_device_group,
        default=None,
        help="Device IDs for generation, e.g. [0,1]",
    )
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Debug-only layer limit for non-tiny models")
    parser.add_argument("--synthetic-tiny", action="store_true", help="Use a local 4-layer Qwen3.5-MoE text model")
    parser.add_argument(
        "--synthetic-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Synthetic tiny model dtype. Auto uses float16 on AI100 and bfloat16 on AI200.",
    )
    parser.add_argument(
        "--weight-free",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the model on meta tensors and load weights at compile time",
    )
    parser.add_argument("--use-onnx-subfunctions", action="store_true", help="Export decoder layers as ONNX functions")
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

    num_devices = len(args.device_group) if args.device_group is not None else args.num_devices
    qpc_path = qeff_model.compile(
        batch_size=args.batch_size,
        compile_dir=str(args.compile_dir) if args.compile_dir else None,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=num_devices,
        aic_hw_version=args.aic_hw_version,
        dynamo=True,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        qaic_config=qaic_config,
    )
    print(f"Model compiled to: {qpc_path}")

    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt] * args.batch_size,
        device_id=args.device_group,
        generation_len=args.generation_len,
    )
    print(f"\nPrompt   : {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")
    print(exec_info)


if __name__ == "__main__":
    main()
