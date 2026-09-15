# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Simple decode-only dynamo example for Qwen3.8-2.4T-A95B."""

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

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

MODEL_ID = "Qwen/Qwen3.8-2.4T-A95B"
SYNTHETIC_CHECKPOINT_DIR = ".qeff/qwen3_8_decode/synthetic_tiny_checkpoint"
RANDOM_SEED = 42
DType = torch.dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hf-hub-cache", default=None)
    parser.add_argument("--qeff-home", default=None)
    parser.add_argument("--use-synthetic-tiny", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-checkpoint-dir", default=SYNTHETIC_CHECKPOINT_DIR)
    parser.add_argument("--torch-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--weight-free", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-onnx-subfunctions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-blocking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blocking-mode", choices=("kv", "kv_headpar"), default="kv")
    parser.add_argument("--num-kv-blocks", type=int, default=2)
    parser.add_argument("--headpar-split", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=1)
    parser.add_argument("--ctx-len", type=int, default=262144)
    parser.add_argument("--generation-len", type=int, default=100)
    parser.add_argument("--write-io", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--device-ids", type=int, nargs="*", default=None)
    parser.add_argument("--aic-hw-version", default="ai200")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def torch_dtype(dtype_name: str) -> DType:
    return getattr(torch, dtype_name)


def layer_types(num_hidden_layers: int) -> list[str]:
    return [
        "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
        for layer_idx in range(num_hidden_layers)
    ]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "Hello": 3}
    vocab.update({f"token_{idx}": idx for idx in range(len(vocab), 128)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )


def tiny_config(dtype: DType, num_hidden_layers: int) -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=layer_types(num_hidden_layers),
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


def ensure_synthetic_checkpoint(config: Qwen3_5MoeTextConfig, checkpoint_dir: Path, dtype: DType, seed: int) -> Path:
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        saved_config = Qwen3_5MoeTextConfig.from_pretrained(checkpoint_dir)
        if saved_config.to_dict() != config.to_dict():
            shutil.rmtree(checkpoint_dir)

    if (checkpoint_dir / "model.safetensors").exists():
        return checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = Qwen3_5MoeForCausalLM(config).eval().to(dtype)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    return checkpoint_dir


def load_model_and_tokenizer(args=None):
    args = args or parse_args()
    dtype = torch_dtype(args.torch_dtype)
    torch.manual_seed(args.random_seed)
    if args.use_synthetic_tiny:
        config = tiny_config(dtype, args.num_hidden_layers)
        tokenizer = tiny_tokenizer()
        if args.weight_free:
            checkpoint_dir = ensure_synthetic_checkpoint(
                config,
                Path(args.synthetic_checkpoint_dir),
                dtype,
                args.random_seed,
            )
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir),
                config=config,
                weight_free=True,
                dtype=dtype,
                trust_remote_code=True,
            )
        else:
            model = Qwen3_5MoeForCausalLM(config).eval().to(dtype)
            qeff_model = QEFFAutoModelForCausalLM(model, dtype=dtype)
        qeff_model.model.config.dtype = dtype
        qeff_model.model.config.torch_dtype = dtype
        return qeff_model, tokenizer

    hub_kwargs = {"cache_dir": args.hf_hub_cache} if args.hf_hub_cache else {}
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True, **hub_kwargs)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
        if hasattr(config, "layer_types"):
            config.layer_types = config.layer_types[: args.num_hidden_layers]
    config.dtype = dtype
    config.torch_dtype = dtype
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, **hub_kwargs)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        weight_free=args.weight_free,
        dtype=dtype,
        trust_remote_code=True,
        **hub_kwargs,
    )
    return qeff_model, tokenizer


def main():
    args = parse_args()
    if args.hf_hub_cache:
        os.environ["HF_HUB_CACHE"] = args.hf_hub_cache
    if args.qeff_home:
        os.environ["QEFF_HOME"] = args.qeff_home

    qeff_model, tokenizer = load_model_and_tokenizer(args)
    qeff_model.model.eval()

    qaic_config = None
    if args.enable_blocking:
        qaic_config = {
            "blocking_mode": args.blocking_mode,
            "num_kv_blocks": args.num_kv_blocks,
            "ctx_len": args.ctx_len,
        }
        if args.blocking_mode == "kv_headpar":
            qaic_config["headpar_split"] = args.headpar_split

    qpc_path = qeff_model.compile(
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        aic_hw_version=args.aic_hw_version,
        dynamo=True,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        qaic_config=qaic_config,
    )
    print(f"Final QPC path: {qpc_path}")

    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt] * args.batch_size,
        device_id=args.device_ids,
        generation_len=args.generation_len,
        write_io=args.write_io,
    )
    print(output.generated_ids)
    print(output.generated_texts)
    print(output)


if __name__ == "__main__":
    main()
