# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run Qwen3.8 disaggregated prefill and decode with separate Dynamo QPCs.

Qwen3.8 GDN prefill is statically specialized by Dynamo, so this example
compiles a fixed-length prefill QPC and a one-token decode QPC. It transfers
the hybrid retained state through host memory between the two workers. Set
``QEFF_HOME`` before invoking the script to choose the artifact root.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession, is_retained_state_name

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

MODEL_ID = "Qwen/Qwen3.8-2.4T-A95B"
RANDOM_SEED = 42
SYNTHETIC_CHECKPOINT_DIR = Path(".qeff/qwen3_8_disagg/synthetic_tiny_checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hf-hub-cache", default=None)
    parser.add_argument("--use-synthetic-tiny", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-checkpoint-dir", type=Path, default=SYNTHETIC_CHECKPOINT_DIR)
    parser.add_argument("--torch-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--weight-free", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefill-seq-len", type=int, default=64)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--device-ids", type=int, nargs="*", default=None)
    parser.add_argument("--aic-hw-version", default="ai200")
    parser.add_argument("--use-onnx-subfunctions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default="token_3 token_4")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def _torch_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def _layer_types(num_hidden_layers: int) -> list[str]:
    return [
        "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention" for layer_idx in range(num_hidden_layers)
    ]


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
    vocab.update({f"token_{idx}": idx for idx in range(3, 128)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )


def _tiny_config(dtype: torch.dtype, num_hidden_layers: int) -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=_layer_types(num_hidden_layers),
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


def _ensure_synthetic_checkpoint(args: argparse.Namespace, config: Qwen3_5MoeTextConfig, dtype: torch.dtype) -> Path:
    checkpoint_dir = args.synthetic_checkpoint_dir / (
        f"{args.torch_dtype}-layers{args.num_hidden_layers}-seed{args.random_seed}"
    )
    if (checkpoint_dir / "model.safetensors").is_file():
        return checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.random_seed)
    Qwen3_5MoeForCausalLM(config).eval().to(dtype).save_pretrained(checkpoint_dir, safe_serialization=True)
    return checkpoint_dir


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[QEFFAutoModelForCausalLM, PreTrainedTokenizerFast]:
    dtype = _torch_dtype(args.torch_dtype)
    torch.manual_seed(args.random_seed)
    if args.use_synthetic_tiny:
        config = _tiny_config(dtype, args.num_hidden_layers)
        if args.weight_free:
            checkpoint_dir = _ensure_synthetic_checkpoint(args, config, dtype)
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
        return qeff_model, _tiny_tokenizer()

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


def _compile_prefill(args: argparse.Namespace) -> str:
    model, _ = _load_model_and_tokenizer(args)
    model.model.eval()
    qpc_path = model.compile(
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        aic_hw_version=args.aic_hw_version,
        dynamo=True,
        prefill_only=True,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
    )
    del model
    gc.collect()
    return qpc_path


def _compile_decode(args: argparse.Namespace) -> str:
    model, _ = _load_model_and_tokenizer(args)
    model.model.eval()
    qpc_path = model.compile(
        batch_size=args.batch_size,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        aic_hw_version=args.aic_hw_version,
        dynamo=True,
        prefill_only=False,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
    )
    del model
    gc.collect()
    return qpc_path


def _retained_state_names(session: QAICInferenceSession) -> list[str]:
    return [name for name in session.input_names if is_retained_state_name(name)]


def _initial_retained_states(session: QAICInferenceSession) -> dict[str, np.ndarray]:
    state_inputs = {}
    for name in _retained_state_names(session):
        binding_index = session.binding_index_map[name]
        binding = session.bindings[binding_index]
        state_inputs[name] = np.zeros(
            tuple(binding.dims),
            dtype=session.aic_to_np_dtype_mapping[binding.type],
        )
    return state_inputs


def _retained_output(outputs: dict[str, np.ndarray], state_name: str) -> np.ndarray:
    output_name = f"{state_name}_RetainedState"
    if output_name not in outputs:
        raise KeyError(f"Missing retained-state output {output_name!r}; available outputs: {sorted(outputs)}")
    return outputs[output_name]


def _update_retained_states(
    inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray], state_names: list[str]
) -> None:
    for state_name in state_names:
        inputs[state_name] = _retained_output(outputs, state_name)


def _prepare_prefill_inputs(
    tokenizer: PreTrainedTokenizerFast, args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    encoded = tokenizer([args.prompt] * args.batch_size, return_tensors="np", padding=True)
    input_ids = encoded["input_ids"].astype(np.int64)
    prompt_len = input_ids.shape[1]
    chunk_count = -(-prompt_len // args.prefill_seq_len)
    padded_len = chunk_count * args.prefill_seq_len
    input_ids = np.pad(input_ids, ((0, 0), (0, padded_len - prompt_len)), constant_values=tokenizer.pad_token_id)
    positions = np.arange(padded_len, dtype=np.int64)
    positions[prompt_len:] = -1
    position_ids = np.broadcast_to(positions, (4, args.batch_size, padded_len)).copy()
    return {"input_ids": input_ids, "position_ids": position_ids}, np.full(
        (4, args.batch_size, 1), prompt_len, dtype=np.int64
    )


def _next_tokens(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=-1).reshape(logits.shape[0], 1).astype(np.int64)


def main() -> None:
    args = parse_args()
    if args.hf_hub_cache:
        os.environ["HF_HUB_CACHE"] = args.hf_hub_cache
    if args.prefill_seq_len <= 0 or args.ctx_len < args.prefill_seq_len:
        raise ValueError("ctx_len must be at least prefill_seq_len, and both lengths must be positive.")
    if args.generation_len <= 0:
        raise ValueError("generation_len must be positive.")

    tokenizer = (
        _tiny_tokenizer()
        if args.use_synthetic_tiny
        else AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            **({"cache_dir": args.hf_hub_cache} if args.hf_hub_cache else {}),
        )
    )
    prefill_qpc_path = _compile_prefill(args)
    decode_qpc_path = _compile_decode(args)
    print(f"Prefill QPC: {prefill_qpc_path}")
    print(f"Decode QPC : {decode_qpc_path}")

    prefill_session = QAICInferenceSession(prefill_qpc_path, device_ids=args.device_ids)
    decode_session = QAICInferenceSession(decode_qpc_path, device_ids=args.device_ids)
    prefill_state_names = _retained_state_names(prefill_session)
    decode_state_names = _retained_state_names(decode_session)
    if prefill_state_names != decode_state_names:
        raise ValueError(
            "Prefill and decode retained-state interfaces differ: "
            f"prefill={prefill_state_names}, decode={decode_state_names}"
        )

    prefill_inputs, next_position_ids = _prepare_prefill_inputs(tokenizer, args)
    prefill_inputs.update(_initial_retained_states(prefill_session))
    num_chunks = prefill_inputs["input_ids"].shape[1] // args.prefill_seq_len
    for chunk_idx in range(num_chunks):
        start = chunk_idx * args.prefill_seq_len
        end = start + args.prefill_seq_len
        chunk_inputs = prefill_inputs.copy()
        chunk_inputs["input_ids"] = prefill_inputs["input_ids"][:, start:end]
        chunk_inputs["position_ids"] = prefill_inputs["position_ids"][..., start:end]
        prefill_outputs = prefill_session.run(chunk_inputs)
        _update_retained_states(prefill_inputs, prefill_outputs, prefill_state_names)

    generated_ids = [_next_tokens(prefill_outputs["logits"])]
    decode_inputs = {
        "input_ids": generated_ids[-1],
        "position_ids": next_position_ids,
    }
    _update_retained_states(decode_inputs, prefill_outputs, decode_state_names)
    for _ in range(args.generation_len - 1):
        decode_outputs = decode_session.run(decode_inputs)
        token_ids = _next_tokens(decode_outputs["logits"])
        generated_ids.append(token_ids)
        _update_retained_states(decode_inputs, decode_outputs, decode_state_names)
        decode_inputs["input_ids"] = token_ids
        decode_inputs["position_ids"] += 1

    generated = np.concatenate(generated_ids, axis=1)
    print(f"Generated token IDs: {generated.tolist()}")
    for batch_idx, token_ids in enumerate(generated):
        print(f"Generated text [{batch_idx}]: {tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
