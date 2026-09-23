#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run a 4-layer GLM-5.3 decode-only QEff compile/generate experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "zai-org/GLM-5.3"
DEFAULT_HF_CACHE = "/home/huggingface_hub"
DEFAULT_QEFF_HOME = "/home/ochougul/efficient-transformers/artifacts/glm53_decode_only"


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
    parser.add_argument("--ctx-len", type=int, default=2048 + 128)
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
    return parser.parse_args()


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


def compile_weight_free_decode_only(qeff_model, onnx_path: Path, compile_dir: Path, ctx_len: int, num_cores: int):
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
        specializations=[{"_graph_name": "Decode", "batch_size": 1, "seq_len": 1, "ctx_len": ctx_len}],
        custom_io=custom_io,
        prefill_only=False,
        retained_state=True,
        convert_to_fp16=True,
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
    if args.export_only and not args.weight_free:
        raise ValueError("--export-only is supported only with --weight-free.")
    generation_len = args.generation_len if args.generation_len is not None else args.ctx_len - args.prompt_len
    if generation_len <= 0:
        raise ValueError("generation_len must be positive. Increase ctx_len or lower prompt_len.")

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
            }
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.hf_cache)
    prompt, input_ids = build_exact_length_prompt(tokenizer, args.prompt_len, args.prompt)
    attention_mask = torch.ones_like(input_ids)
    print(json.dumps({"event": "prompt", "prompt": prompt, "token_count": int(input_ids.shape[1])}), flush=True)

    qaic_config = {"mla_absorption": {"cache_compressed": True}}
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
            config=config,
            cache_dir=args.hf_cache,
            torch_dtype=torch.float32,
            device_map="cpu",
        ).eval()

        qeff_model = QEFFAutoModelForCausalLM(
            copy.deepcopy(hf_model).eval(),
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
            qeff_model, Path(onnx_path), compile_dir, args.ctx_len, args.num_cores
        )
    else:
        qpc_path = qeff_model.compile(
            compile_dir=str(compile_dir),
            prefill_seq_len=1,
            ctx_len=args.ctx_len,
            batch_size=1,
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
        prompts=[prompt],
        generation_len=generation_len,
        device_id=args.device_id,
    )
    qeff_generated = tokens_from_qeff_output(exec_info)
    print(json.dumps({"event": "qeff_generate_done", "shape": list(qeff_generated.shape)}), flush=True)

    if args.weight_free:
        install_partial_fp8_dequant_patch()
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
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
    print(
        json.dumps({"event": "token_compare", **compare_tokens(hf_generated, qeff_generated, args.prompt_len)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
