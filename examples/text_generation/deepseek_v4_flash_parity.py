# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compare one-layer DeepSeek-V4-Flash HF and QAIC logits for chunked prefill and decode."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_HF_CACHE = "/home/huggingface_hub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--hf-cache", type=Path, default=Path(DEFAULT_HF_CACHE))
    parser.add_argument("--prefill-qpc", type=Path, required=True)
    parser.add_argument("--decode-qpc", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--ctx-len", type=int, default=2048)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--prefill-chunk-len", type=int, default=512)
    parser.add_argument("--prompt", default="DeepSeek V4 sliding cache QAIC parity. ")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _download_metadata(model_id: str, revision: str, hf_cache: Path, local_files_only: bool) -> Path:
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    return Path(
        snapshot_download(
            model_id,
            revision=revision,
            cache_dir=hf_cache,
            local_files_only=local_files_only,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template.jinja",
                "*.model",
            ],
        )
    )


def _write_one_layer_checkpoint_view(
    model_id: str,
    revision: str,
    hf_cache: Path,
    snapshot: Path,
    output_dir: Path,
    local_files_only: bool,
) -> Path:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "deepseek_v4":
        raise ValueError(f"Expected a DeepSeek V4 checkpoint, got model_type={config.get('model_type')!r}.")
    config["num_hidden_layers"] = 1
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    selected_weights = {
        name: filename
        for name, filename in weight_map.items()
        if name in {"embed.weight", "norm.weight"} or name.startswith("layers.0.")
    }
    if not selected_weights:
        raise ValueError("The checkpoint index contains no weights for DeepSeek V4 layer 0.")

    for filename in set(selected_weights.values()):
        source_file = Path(
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=hf_cache,
                local_files_only=local_files_only,
            )
        )
        (output_dir / filename).symlink_to(source_file)

    sliced_index = {"metadata": {"total_size": 0}, "weight_map": selected_weights}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(sliced_index), encoding="utf-8")
    return output_dir


def _load_hf_reference(args: argparse.Namespace) -> tuple[torch.nn.Module, AutoTokenizer]:
    snapshot = _download_metadata(args.model_id, args.revision, args.hf_cache, args.local_files_only)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    with tempfile.TemporaryDirectory(prefix="deepseek_v4_flash_layer0_") as temp_dir:
        checkpoint_view = _write_one_layer_checkpoint_view(
            args.model_id,
            args.revision,
            args.hf_cache,
            snapshot,
            Path(temp_dir),
            args.local_files_only,
        )
        config = AutoConfig.from_pretrained(checkpoint_view, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_view,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    model.to(dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.hf_cache,
        local_files_only=args.local_files_only,
    )
    return model, tokenizer


def _runtime_logits_to_float32(values: np.ndarray, is_bfloat16: bool) -> np.ndarray:
    if is_bfloat16:
        bfloat16_values = torch.from_numpy(values.view(np.uint16).copy()).view(torch.bfloat16)
        return bfloat16_values.float().numpy()
    return values.astype(np.float32, copy=False)


def _mad(reference: torch.Tensor, actual: np.ndarray, actual_is_bfloat16: bool) -> dict[str, float]:
    reference_values = reference.detach().float().cpu().numpy()
    actual_values = _runtime_logits_to_float32(actual, actual_is_bfloat16)
    difference = reference_values - actual_values
    return {
        "mad": float(np.mean(np.abs(difference), dtype=np.float64)),
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
    }


def _unload_session(session) -> None:
    session.deactivate()
    if hasattr(session.program, "unload"):
        session.program.unload()


def run_parity(args: argparse.Namespace) -> dict:
    if args.ctx_len < 2 or args.prompt_len < 1 or args.prefill_chunk_len < 1:
        raise ValueError("ctx_len, prompt_len, and prefill_chunk_len must be positive; ctx_len must be >= 2.")
    if args.prompt_len + 1 > args.ctx_len:
        raise ValueError("ctx_len must leave room for one decode token after the prompt.")
    if args.prompt_len % args.prefill_chunk_len:
        raise ValueError("prompt_len must be divisible by prefill_chunk_len.")

    model, tokenizer = _load_hf_reference(args)
    if model.config.num_hidden_layers != 1:
        raise RuntimeError("The HF reference was not reduced to one layer.")

    seed_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"][0]
    repeated_ids = seed_ids.repeat((args.prompt_len + seed_ids.numel() - 1) // seed_ids.numel())
    input_ids = repeated_ids[: args.prompt_len].unsqueeze(0).contiguous()
    position_ids = torch.arange(args.prompt_len, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        hf_prefill = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )

    from QEfficient.generation.cloud_infer import QAICInferenceSession

    prefill_session = QAICInferenceSession(str(args.prefill_qpc), device_ids=[args.device_id])
    cache = np.zeros((1, 1, args.ctx_len, model.config.head_dim), dtype=np.float16)
    qpc_prefill_logits = None
    for start in range(0, args.prompt_len, args.prefill_chunk_len):
        end = start + args.prefill_chunk_len
        outputs = prefill_session.run(
            {
                "input_ids": input_ids[:, start:end].numpy().astype(np.int64, copy=False),
                "position_ids": position_ids[:, start:end].numpy().astype(np.int64, copy=False),
                "past_sliding_window_kv.0": cache,
            }
        )
        qpc_prefill_logits = outputs["logits"].copy()
        cache = outputs["past_sliding_window_kv.0_RetainedState"].copy()
    prefill_is_bfloat16 = prefill_session.binding_is_bfloat16("logits")
    _unload_session(prefill_session)

    prefill_mad = _mad(hf_prefill.logits[:, -1:, :], qpc_prefill_logits, prefill_is_bfloat16)
    next_token = hf_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    next_position = torch.full((1, 1), args.prompt_len, dtype=torch.long)
    with torch.inference_mode():
        hf_decode = model(
            input_ids=next_token,
            position_ids=next_position,
            attention_mask=torch.ones((1, args.prompt_len + 1), dtype=torch.long),
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )

    decode_session = QAICInferenceSession(str(args.decode_qpc), device_ids=[args.device_id])
    decode_outputs = decode_session.run(
        {
            "input_ids": next_token.numpy().astype(np.int64, copy=False),
            "position_ids": next_position.numpy().astype(np.int64, copy=False),
            "past_sliding_window_kv.0": cache,
        }
    )
    decode_is_bfloat16 = decode_session.binding_is_bfloat16("logits")
    _unload_session(decode_session)
    decode_mad = _mad(hf_decode.logits[:, -1:, :], decode_outputs["logits"], decode_is_bfloat16)

    return {
        "model_id": args.model_id,
        "revision": args.revision,
        "layers": 1,
        "layer_type": model.config.layer_types[0],
        "mlp_layer_type": model.config.mlp_layer_types[0],
        "dtype": args.dtype,
        "device_id": args.device_id,
        "ctx_len": args.ctx_len,
        "prompt_len": args.prompt_len,
        "prefill_chunk_len": args.prefill_chunk_len,
        "prefill_chunks": args.prompt_len // args.prefill_chunk_len,
        "prefill": prefill_mad,
        "decode": decode_mad,
        "decode_input_token": int(next_token.item()),
        "prefill_qpc": str(args.prefill_qpc),
        "decode_qpc": str(args.decode_qpc),
    }


def main() -> None:
    args = parse_args()
    report = run_parity(args)
    report_json = json.dumps(report, indent=2)
    print(report_json)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
