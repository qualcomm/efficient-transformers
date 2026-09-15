# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Simple decode-only dynamo example for Qwen3.8-2.4T-A95B."""

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
TORCH_DTYPE = getattr(torch, os.environ.get("TORCH_DTYPE", "float32"))

USE_SYNTHETIC_TINY = os.environ.get("USE_SYNTHETIC_TINY", "1").lower() in ("1", "true", "yes")
SYNTHETIC_CHECKPOINT_DIR = Path(".qeff/qwen3_8_decode/synthetic_tiny_checkpoint")
NUM_HIDDEN_LAYERS = int(os.environ.get("NUM_HIDDEN_LAYERS", "4"))

WEIGHT_FREE = True
USE_ONNX_SUBFUNCTIONS = True

ENABLE_BLOCKING = False
BLOCKING_MODE = "kv"  # "kv" or "kv_headpar"
NUM_KV_BLOCKS = 2
HEADPAR_SPLIT = 4

BATCH_SIZE = 1
PREFILL_SEQ_LEN = 1
CTX_LEN = 262144
GENERATION_LEN = 100
NUM_CORES = 4
NUM_DEVICES = 4
AIC_HW_VERSION = "ai200"
PROMPT = "Hello"
RANDOM_SEED = 42


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


def tiny_config() -> Qwen3_5MoeTextConfig:
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
        dtype=TORCH_DTYPE,
        pad_token_id=0,
        eos_token_id=2,
    )


def ensure_synthetic_checkpoint(config: Qwen3_5MoeTextConfig) -> Path:
    config_path = SYNTHETIC_CHECKPOINT_DIR / "config.json"
    if config_path.exists():
        saved_config = Qwen3_5MoeTextConfig.from_pretrained(SYNTHETIC_CHECKPOINT_DIR)
        if saved_config.to_dict() != config.to_dict():
            shutil.rmtree(SYNTHETIC_CHECKPOINT_DIR)

    if (SYNTHETIC_CHECKPOINT_DIR / "model.safetensors").exists():
        return SYNTHETIC_CHECKPOINT_DIR

    SYNTHETIC_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    model = Qwen3_5MoeForCausalLM(config).eval().to(TORCH_DTYPE)
    model.save_pretrained(SYNTHETIC_CHECKPOINT_DIR, safe_serialization=True)
    return SYNTHETIC_CHECKPOINT_DIR


def load_model_and_tokenizer():
    torch.manual_seed(RANDOM_SEED)
    if USE_SYNTHETIC_TINY:
        config = tiny_config()
        tokenizer = tiny_tokenizer()
        if WEIGHT_FREE:
            checkpoint_dir = ensure_synthetic_checkpoint(config)
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir),
                config=config,
                weight_free=True,
                dtype=TORCH_DTYPE,
                trust_remote_code=True,
            )
        else:
            model = Qwen3_5MoeForCausalLM(config).eval().to(TORCH_DTYPE)
            qeff_model = QEFFAutoModelForCausalLM(model, dtype=TORCH_DTYPE)
        qeff_model.model.config.dtype = TORCH_DTYPE
        qeff_model.model.config.torch_dtype = TORCH_DTYPE
        return qeff_model, tokenizer

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if NUM_HIDDEN_LAYERS > 0:
        config.num_hidden_layers = NUM_HIDDEN_LAYERS
        if hasattr(config, "layer_types"):
            config.layer_types = config.layer_types[:NUM_HIDDEN_LAYERS]
    config.dtype = TORCH_DTYPE
    config.torch_dtype = TORCH_DTYPE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        weight_free=WEIGHT_FREE,
        dtype=TORCH_DTYPE,
        trust_remote_code=True,
    )
    return qeff_model, tokenizer


def main():
    qeff_model, tokenizer = load_model_and_tokenizer()
    qeff_model.model.eval()

    qaic_config = None
    if ENABLE_BLOCKING:
        qaic_config = {
            "blocking_mode": BLOCKING_MODE,
            "num_kv_blocks": NUM_KV_BLOCKS,
            "ctx_len": CTX_LEN,
        }
        if BLOCKING_MODE == "kv_headpar":
            qaic_config["headpar_split"] = HEADPAR_SPLIT

    qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        num_devices=NUM_DEVICES,
        aic_hw_version=AIC_HW_VERSION,
        dynamo=True,
        use_onnx_subfunctions=USE_ONNX_SUBFUNCTIONS,
        qaic_config=qaic_config,
    )
    print(f"Final QPC path: {qpc_path}")

    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[PROMPT] * BATCH_SIZE,
        generation_len=GENERATION_LEN,
    )
    print(output.generated_ids)
    print(output.generated_texts)
    print(output)


if __name__ == "__main__":
    main()
