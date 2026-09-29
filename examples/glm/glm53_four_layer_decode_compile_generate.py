#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run a four-layer GLM-5.3 decode-only QEff export, compile, and generation experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "zai-org/GLM-5.3"
DEFAULT_HF_CACHE = "/home/huggingface_hub"
DEFAULT_QEFF_HOME = "/home/ochougul/efficient-transformers/artifacts/glm53_decode_only"

ATTENTION_PRESETS = {
    "dense": {
        "blocking_mode": "none",
        "mla_absorption": {"absorption": False, "online": False, "cache_compressed": True},
    },
    "dense_offline": {
        "blocking_mode": "none",
        "mla_absorption": {"absorption": True, "online": False, "cache_compressed": True},
    },
    "dense_online": {
        "blocking_mode": "none",
        "mla_absorption": {"absorption": True, "online": True, "cache_compressed": True},
    },
    "dense_parallel": {
        "blocking_mode": "par",
        "num_kv_blocks": 2,
        "par_num_split": 16,
        "mla_absorption": {"absorption": True, "online": False, "cache_compressed": True},
    },
    "dense_prefill_parallel": {
        "blocking_mode": "prefill_par",
        "num_kv_blocks": 2,
        "par_num_split": 16,
        "mla_absorption": {"absorption": True, "online": False, "cache_compressed": True},
    },
    "dense_prefill_parallel_online": {
        "blocking_mode": "prefill_par_online",
        "num_kv_blocks": 2,
        "par_num_split": 16,
        "mla_absorption": {"absorption": True, "online": True, "cache_compressed": True},
    },
    "dsa_cp1": {
        "indexer_dp": 1,
        "indexer_cp": 1,
        "indexer_kvp": 1,
        "attn_dp": 1,
        "attn_cp": 1,
        "attn_kvp": 1,
        "indexer_num_blocks": 1,
        "num_cores_per_device": 16,
    },
    "dsa_cp2": {
        "indexer_dp": 1,
        "indexer_cp": 2,
        "indexer_kvp": 1,
        "attn_dp": 1,
        "attn_cp": 2,
        "attn_kvp": 1,
        "indexer_num_blocks": 1,
        "num_cores_per_device": 16,
    },
    "dsa_ts16": {
        "indexer_dp": 1,
        "indexer_cp": 16,
        "indexer_kvp": 1,
        "attn_dp": 16,
        "attn_cp": 1,
        "attn_kvp": 1,
        "indexer_num_blocks": 16,
        "num_cores_per_device": 16,
    },
}

DEFAULT_RUNTIME_CONFIG = {"batch_size": 1, "ctx_len": 2048 + 128, "num_devices": 1}
PRESET_RUNTIME_DEFAULTS = {
    "dsa_cp2": {"num_devices": 2},
    "dsa_ts16": {"batch_size": 16, "ctx_len": 4096, "num_devices": 16},
}
DEFAULT_GENERATION_LEN = 32


def install_partial_fp8_dequant_patch(weight_block_size: tuple[int, int] = (128, 128)) -> None:
    """Allow reduced-layer CPU loads when an FP8 scale grid has a partial edge block.

    The real checkpoint is valid, but loading only the first few layers can expose
    a target tensor whose rows are not evenly divisible by the scale grid. This
    pads only for dequantization and slices back to the original weight shape.
    """

    from transformers.integrations import finegrained_fp8

    original_dequantize_one = finegrained_fp8.Fp8Dequantize._dequantize_one
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)

    def patched_dequantize_one(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)

        rows, cols = quantized_fp32.shape[-2:]
        try:
            scale_rows, scale_cols = scales.shape[-2:]
        except (AttributeError, TypeError, ValueError):
            return original_dequantize_one(self, quantized, scales, output_dtype=output_dtype)

        if rows % scale_rows == 0 and cols % scale_cols == 0:
            return original_dequantize_one(self, quantized, scales, output_dtype=output_dtype)

        block_m, block_n = weight_block_size
        padded_rows = scale_rows * block_m
        padded_cols = scale_cols * block_n
        if (
            padded_rows < rows
            or padded_cols < cols
            or rows <= (scale_rows - 1) * block_m
            or cols <= (scale_cols - 1) * block_n
        ):
            return original_dequantize_one(self, quantized, scales, output_dtype=output_dtype)

        if output_dtype is None:
            output_dtype = (
                scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
            )

        if scales.dtype == torch.uint8:
            scales_fp32 = (scales.to(torch.float32) - 127.0).exp2()
        else:
            scales_fp32 = scales.to(torch.float32)

        padded = torch.zeros(
            (*quantized_fp32.shape[:-2], padded_rows, padded_cols),
            dtype=torch.float32,
            device=quantized_fp32.device,
        )
        padded[..., :rows, :cols] = quantized_fp32
        dequantized = (
            padded.reshape(-1, scale_rows, block_m, scale_cols, block_n)
            * scales_fp32.reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
        ).reshape(padded.shape)
        return dequantized[..., :rows, :cols].to(output_dtype)

    finegrained_fp8.Fp8Dequantize._dequantize_one = patched_dequantize_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--qeff-home", default=DEFAULT_QEFF_HOME)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=None)
    parser.add_argument("--generation-len", type=int, default=None)
    parser.add_argument("--compile-dir", default=None)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--device-id", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--weight-free", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--attention-preset", choices=sorted(ATTENTION_PRESETS), default="dsa_cp1")
    parser.add_argument(
        "--attention-qaic-json",
        default=None,
        help="JSON object merged over the selected attention preset.",
    )
    parser.add_argument("--all-attention-configs", action="store_true")
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-devices", type=int, default=None)
    return parser.parse_args()


def resolve_runtime_dimensions(args: argparse.Namespace) -> tuple[int, int, int, int]:
    runtime_defaults = {**DEFAULT_RUNTIME_CONFIG, **PRESET_RUNTIME_DEFAULTS.get(args.attention_preset, {})}
    batch_size = args.batch_size if args.batch_size is not None else runtime_defaults["batch_size"]
    ctx_len = args.ctx_len if args.ctx_len is not None else runtime_defaults["ctx_len"]
    num_devices = args.num_devices if args.num_devices is not None else runtime_defaults["num_devices"]
    generation_len = args.generation_len if args.generation_len is not None else DEFAULT_GENERATION_LEN
    for name, value in {
        "batch_size": batch_size,
        "ctx_len": ctx_len,
        "num_devices": num_devices,
        "generation_len": generation_len,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if args.prompt_len + generation_len > ctx_len:
        raise ValueError("prompt_len + generation_len must not exceed ctx_len.")
    return batch_size, ctx_len, num_devices, generation_len


def validate_attention_runtime_dimensions(
    qaic_config: dict[str, Any],
    *,
    batch_size: int,
    context_length: int,
    num_devices: int,
    model_topk: int,
) -> None:
    if "indexer_dp" not in qaic_config and "attn_dp" not in qaic_config:
        return
    indexer_dp = int(qaic_config.get("indexer_dp", 1))
    indexer_cp = int(qaic_config.get("indexer_cp", 1))
    attn_dp = int(qaic_config.get("attn_dp", 1))
    attn_cp = int(qaic_config.get("attn_cp", 1))
    num_blocks = int(qaic_config.get("indexer_num_blocks", 1))
    cores = int(qaic_config.get("num_cores_per_device", 1))
    effective_topk = min(model_topk, context_length)
    for name, dp, cp in (("indexer", indexer_dp, indexer_cp), ("attention", attn_dp, attn_cp)):
        if dp * cp != num_devices:
            raise ValueError(f"{name}_dp * {name}_cp must equal num_devices ({num_devices}).")
        if batch_size % dp:
            raise ValueError(f"batch_size must be divisible by {name}_dp.")
        if context_length % cp:
            raise ValueError(f"context_length must be divisible by {name}_cp.")
    if effective_topk % cores:
        raise ValueError("model index_topk must be divisible by num_cores_per_device.")
    if (context_length // indexer_cp) % (num_blocks * cores):
        raise ValueError("indexer local context must be divisible by indexer_num_blocks * num_cores_per_device.")


def build_exact_length_prompt(tokenizer: Any, prompt_len: int, prompt: str | None) -> tuple[str, torch.Tensor]:
    if prompt is not None:
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids
        if input_ids.shape[1] != prompt_len:
            raise ValueError(f"Provided prompt tokenized to {input_ids.shape[1]} tokens, expected {prompt_len}.")
        return prompt, input_ids

    text = "GLM decode parity"
    for _ in range(512):
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        if input_ids.shape[1] == prompt_len:
            return text, input_ids
        if input_ids.shape[1] > prompt_len:
            candidate = tokenizer.decode(input_ids[0, :prompt_len], skip_special_tokens=False)
            candidate_ids = tokenizer(candidate, return_tensors="pt").input_ids
            if candidate_ids.shape[1] == prompt_len:
                return candidate, candidate_ids
        text += " test"
    raise RuntimeError(f"Could not synthesize a prompt with exactly {prompt_len} tokens.")


def tokens_from_qeff_output(exec_info: Any) -> np.ndarray:
    generated_ids = getattr(exec_info, "generated_ids", None)
    if generated_ids is None:
        raise RuntimeError("QEff generate did not return `generated_ids`.")
    generated_ids = np.asarray(generated_ids)
    if generated_ids.ndim == 3 and generated_ids.shape[1] == 1:
        generated_ids = generated_ids[:, 0, :]
    return generated_ids


def run_hf_decode_only(model: torch.nn.Module, input_ids: torch.Tensor, generation_len: int) -> torch.Tensor:
    """Run greedy HF generation using only single-token forward passes."""
    past_key_values = None
    outputs = None
    prompt_len = input_ids.shape[1]

    for position in range(prompt_len):
        cache_position = torch.tensor([position], dtype=torch.long)
        outputs = model(
            input_ids=input_ids[:, position : position + 1],
            attention_mask=torch.ones((input_ids.shape[0], position + 1), dtype=torch.long),
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    if outputs is None:
        raise ValueError("Decode-only HF generation requires at least one prompt token.")

    generated = []
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for step in range(generation_len):
        generated.append(next_token)
        if step == generation_len - 1:
            break
        position = prompt_len + step
        cache_position = torch.tensor([position], dtype=torch.long)
        outputs = model(
            input_ids=next_token,
            attention_mask=torch.ones((input_ids.shape[0], position + 1), dtype=torch.long),
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return torch.cat((input_ids, *generated), dim=1)


def compile_weight_free_decode_only(
    qeff_model,
    onnx_path: Path,
    compile_dir: Path,
    ctx_len: int,
    batch_size: int,
    num_devices: int,
    num_cores: int,
):
    """Compile the validated single-token GLM weight-free specialization."""
    custom_io = {}
    for layer_idx in range(qeff_model.model.config.num_hidden_layers):
        for cache_name in (f"compressed_kv.{layer_idx}", f"k_pe.{layer_idx}"):
            custom_io[cache_name] = "float16"
            custom_io[f"{cache_name}_RetainedState"] = "float16"
    for cache_idx, _ in enumerate(qeff_model.model.get_indexer_cache_layers(qeff_model.model.config)):
        cache_name = f"indexer_key.{cache_idx}"
        custom_io[cache_name] = "float16"
        custom_io[f"{cache_name}_RetainedState"] = "float16"

    return qeff_model._compile(
        onnx_path=str(onnx_path),
        compile_dir=str(compile_dir),
        specializations=[{"_graph_name": "Decode", "batch_size": batch_size, "seq_len": 1, "ctx_len": ctx_len}],
        custom_io=custom_io,
        prefill_only=False,
        retained_state=True,
        convert_to_fp16=True,
        mdp_ts_num_devices=num_devices,
        aic_num_cores=num_cores,
    )


def compare_tokens(hf_tokens: torch.Tensor, qeff_tokens: np.ndarray, prompt_len: int) -> dict[str, Any]:
    hf_np = hf_tokens.detach().cpu().numpy()
    qeff_np = np.asarray(qeff_tokens)
    if qeff_np.ndim == 1:
        qeff_np = qeff_np[None, :]

    comparisons = []
    if qeff_np.shape == hf_np.shape:
        comparisons.append(("full_sequence", hf_np, qeff_np))

    hf_new = hf_np[:, prompt_len:]
    if qeff_np.shape == hf_new.shape:
        comparisons.append(("generated_only", hf_new, qeff_np))
    if qeff_np.ndim == 2 and qeff_np.shape[0] == hf_new.shape[0] and qeff_np.shape[1] >= hf_new.shape[1]:
        comparisons.append(("generated_only_prefix", hf_new, qeff_np[:, : hf_new.shape[1]]))
    if qeff_np.ndim == 2 and qeff_np.shape[0] == hf_np.shape[0] and qeff_np.shape[1] >= hf_np.shape[1]:
        comparisons.append(("full_sequence_prefix", hf_np, qeff_np[:, : hf_np.shape[1]]))

    if not comparisons:
        return {
            "matched": False,
            "mode": "shape_mismatch",
            "hf_shape": list(hf_np.shape),
            "qeff_shape": list(qeff_np.shape),
        }

    mode, lhs, rhs = comparisons[0]
    mismatch = np.argwhere(lhs != rhs)
    result: dict[str, Any] = {
        "matched": mismatch.size == 0,
        "mode": mode,
        "hf_shape": list(hf_np.shape),
        "qeff_shape": list(qeff_np.shape),
    }
    if mismatch.size:
        batch_idx, token_idx = mismatch[0].tolist()
        result.update(
            {
                "first_mismatch": {
                    "batch": batch_idx,
                    "token_index": token_idx,
                    "hf_token": int(lhs[batch_idx, token_idx]),
                    "qeff_token": int(rhs[batch_idx, token_idx]),
                }
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.all_attention_configs:
        script_args = [arg for arg in sys.argv[1:] if arg != "--all-attention-configs"]
        for preset in ATTENTION_PRESETS:
            filtered = []
            skip_next = False
            for arg in script_args:
                if skip_next:
                    skip_next = False
                    continue
                if arg == "--attention-preset":
                    skip_next = True
                    continue
                if arg.startswith("--attention-preset="):
                    continue
                filtered.append(arg)
            subprocess.run([sys.executable, __file__, *filtered, "--attention-preset", preset], check=True)
        return
    if args.export_only and not args.weight_free:
        raise ValueError("--export-only is supported only with --weight-free.")
    args.batch_size, args.ctx_len, args.num_devices, generation_len = resolve_runtime_dimensions(args)
    qaic_config = copy.deepcopy(ATTENTION_PRESETS[args.attention_preset])
    if args.attention_qaic_json:
        qaic_config.update(json.loads(args.attention_qaic_json))
    os.environ.setdefault("HF_HUB_CACHE", args.hf_cache)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("QEFF_HOME", args.qeff_home)

    from QEfficient import QEFFAutoModelForCausalLM

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    config = AutoConfig.from_pretrained(args.model_id, cache_dir=args.hf_cache)
    config.num_hidden_layers = args.num_layers
    config.use_cache = True
    config.torch_dtype = torch.float32
    config.dtype = torch.float32
    validate_attention_runtime_dimensions(
        qaic_config,
        batch_size=args.batch_size,
        context_length=args.ctx_len,
        num_devices=args.num_devices,
        model_topk=int(config.index_topk),
    )
    native_hf_config = copy.deepcopy(config)
    is_dense_validation = args.attention_preset.startswith("dense")
    if is_dense_validation:
        config.layer_types = ["full_attention"] * args.num_layers
        if args.prompt_len + generation_len > native_hf_config.index_topk:
            raise ValueError(
                "Dense-vs-native-DSA validation requires prompt_len + generation_len <= index_topk "
                f"({native_hf_config.index_topk})."
            )
    if args.weight_free and config.num_hidden_layers != 4:
        raise ValueError("GLM-5.3 weight-free export is currently supported only with --num-layers 4.")

    print(
        json.dumps(
            {
                "event": "config",
                "model_id": args.model_id,
                "num_layers": args.num_layers,
                "dtype": "torch.float32",
                "prompt_len": args.prompt_len,
                "ctx_len": args.ctx_len,
                "generation_len": generation_len,
                "qeff_home": os.environ["QEFF_HOME"],
                "attention_preset": args.attention_preset,
                "qaic_config": qaic_config,
                "layer_types": list(config.layer_types[: args.num_layers]),
            }
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.hf_cache)
    prompt, input_ids = build_exact_length_prompt(tokenizer, args.prompt_len, args.prompt)
    input_ids = input_ids.expand(args.batch_size, -1).contiguous()
    attention_mask = torch.ones_like(input_ids)
    print(json.dumps({"event": "prompt", "prompt": prompt, "token_count": int(input_ids.shape[1])}), flush=True)

    hf_model = None
    onnx_path = None
    if args.weight_free:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            cache_dir=args.hf_cache,
            torch_dtype=torch.float32,
            weight_free=True,
            qaic_config=qaic_config,
        )
        qeff_model.model.eval()
        qeff_model.transform(
            ctx_len=args.ctx_len,
            seq_len=1,
            bs=args.batch_size,
            num_devices=args.num_devices,
            qaic_config=qaic_config,
            num_cores=args.num_cores,
        )
        print(
            json.dumps({"event": "qeff_initialized", "model_class": qeff_model.model.__class__.__name__}),
            flush=True,
        )
        export_dir = (
            Path(args.export_dir) if args.export_dir is not None else Path(args.qeff_home) / "weight_free_export"
        )
        onnx_path = qeff_model.export(
            export_dir=str(export_dir),
            prefill_only=False,
            offload_pt_weights=False,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
        )
        weight_spec_path = getattr(qeff_model, "weight_spec_path", None)
        weight_spec_path_str = str(weight_spec_path) if weight_spec_path is not None else None
        print(
            json.dumps(
                {
                    "event": "weight_free_export_done",
                    "onnx_path": str(onnx_path),
                    "onnx_exists": Path(onnx_path).is_file(),
                    "weight_spec_path": weight_spec_path_str,
                    "weight_spec_exists": bool(weight_spec_path_str and Path(weight_spec_path_str).is_file()),
                }
            ),
            flush=True,
        )
        if args.export_only:
            return
    else:
        install_partial_fp8_dequant_patch()
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=native_hf_config,
            cache_dir=args.hf_cache,
            torch_dtype=torch.float32,
            device_map="cpu",
        ).eval()

        qeff_source_model = copy.deepcopy(hf_model).eval()
        qeff_source_model.config.layer_types = list(config.layer_types)
        qeff_model = QEFFAutoModelForCausalLM(
            qeff_source_model,
            pretrained_model_name_or_path=args.model_id,
            qaic_config=qaic_config,
        )
        print(
            json.dumps({"event": "qeff_initialized", "model_class": qeff_model.model.__class__.__name__}),
            flush=True,
        )

    compile_dir = Path(args.compile_dir) if args.compile_dir is not None else Path(args.qeff_home) / "compile"
    if args.weight_free:
        qpc_path = compile_weight_free_decode_only(
            qeff_model,
            Path(onnx_path),
            compile_dir,
            args.ctx_len,
            args.batch_size,
            args.num_devices,
            args.num_cores,
        )
    else:
        qpc_path = qeff_model.compile(
            compile_dir=str(compile_dir),
            prefill_seq_len=1,
            ctx_len=args.ctx_len,
            batch_size=args.batch_size,
            num_devices=args.num_devices,
            num_cores=args.num_cores,
            prefill_only=False,
            offload_pt_weights=False,
        )
    print(
        json.dumps({"event": "compile_done", "qpc_path": str(qpc_path), "onnx_path": str(qeff_model.onnx_path)}),
        flush=True,
    )

    if args.skip_generate:
        print(json.dumps({"event": "skip_generate"}), flush=True)
        return

    exec_info = qeff_model.generate(
        tokenizer,
        prompts=[prompt] * args.batch_size,
        generation_len=generation_len,
        device_id=args.device_id,
    )
    qeff_generated = tokens_from_qeff_output(exec_info)
    print(json.dumps({"event": "qeff_generate_done", "shape": list(qeff_generated.shape)}), flush=True)

    if args.weight_free:
        install_partial_fp8_dequant_patch()
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=native_hf_config,
            cache_dir=args.hf_cache,
            torch_dtype=torch.float32,
            device_map="cpu",
        ).eval()
        hf_generated = run_hf_decode_only(hf_model, input_ids, generation_len)
        print(json.dumps({"event": "hf_decode_only_done", "shape": list(hf_generated.shape)}), flush=True)
    else:
        hf_generated = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_len,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(json.dumps({"event": "hf_generate_done", "shape": list(hf_generated.shape)}), flush=True)
    result = {"event": "token_compare", **compare_tokens(hf_generated, qeff_generated, args.prompt_len)}
    print(json.dumps(result), flush=True)
    if args.results_json:
        Path(args.results_json).write_text(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "attention_preset": args.attention_preset,
                    "qaic_config": qaic_config,
                    "onnx_path": str(qeff_model.onnx_path),
                    "qpc_path": str(qpc_path),
                    "token_comparison": result,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
