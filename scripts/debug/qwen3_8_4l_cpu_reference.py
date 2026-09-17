# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Save HF PyTorch CPU reference tokens and logits for Qwen3.8 decode parity."""

import argparse
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.8-2.4T-A95B"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hf-hub-cache", default=None)
    parser.add_argument("--torch-dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--generation-len", type=int, default=100)
    parser.add_argument("--output-dir", default=".newly_testing_4_layers/pytorch_cpu_reference_4l_fp16")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qwen3_8_cpu_reference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "pytorch_cpu_reference.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def dtype_from_name(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def truncate_config(config, num_hidden_layers: int):
    if num_hidden_layers <= 0:
        return config
    config.num_hidden_layers = num_hidden_layers
    if hasattr(config, "layer_types"):
        config.layer_types = config.layer_types[:num_hidden_layers]
    return config


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir)

    if args.hf_hub_cache:
        os.environ["HF_HUB_CACHE"] = args.hf_hub_cache

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.torch_dtype)
    hub_kwargs = {
        "cache_dir": args.hf_hub_cache,
        "local_files_only": args.local_files_only,
        "trust_remote_code": True,
    }

    logger.info("Loading config: model_id=%s cache=%s", args.model_id, args.hf_hub_cache)
    config = AutoConfig.from_pretrained(args.model_id, **hub_kwargs)
    config = truncate_config(config, args.num_hidden_layers)
    config.dtype = dtype
    config.torch_dtype = dtype

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **hub_kwargs)

    logger.info("Loading HF PyTorch model on CPU: dtype=%s layers=%s", args.torch_dtype, config.num_hidden_layers)
    start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        **hub_kwargs,
    ).eval()
    load_time = perf_counter() - start
    logger.info("Model loaded in %.2f sec", load_time)

    model_inputs = tokenizer(args.prompt, return_tensors="pt")
    model_inputs.pop("token_type_ids", None)
    prompt_len = int(model_inputs["input_ids"].shape[-1])
    logger.info("Prompt=%r prompt_len=%d generation_len=%d", args.prompt, prompt_len, args.generation_len)

    logger.info("Running greedy generate with output logits")
    start = perf_counter()
    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=args.generation_len,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
    elapsed = perf_counter() - start

    generated_tokens = generation.sequences[0, prompt_len:].detach().cpu().numpy()
    logits = torch.stack(generation.logits).squeeze(1).float().detach().cpu().numpy()

    np.save(output_dir / "tokens.npy", generated_tokens)
    np.save(output_dir / "logits.npy", logits)

    summary = {
        "model_id": args.model_id,
        "hf_hub_cache": args.hf_hub_cache,
        "dtype": args.torch_dtype,
        "num_hidden_layers": config.num_hidden_layers,
        "prompt": args.prompt,
        "prompt_len": prompt_len,
        "generation_len": args.generation_len,
        "elapsed_sec": elapsed,
        "tokens": generated_tokens.tolist(),
        "first_20_tokens": generated_tokens[:20].tolist(),
        "logits_shape": list(logits.shape),
        "tokens_path": str(output_dir / "tokens.npy"),
        "logits_path": str(output_dir / "logits.npy"),
    }
    with open(output_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Generated first 20 tokens: %s", summary["first_20_tokens"])
    logger.info("Saved tokens: %s", summary["tokens_path"])
    logger.info("Saved logits: %s shape=%s", summary["logits_path"], summary["logits_shape"])
    logger.info("Elapsed %.2f sec", elapsed)


if __name__ == "__main__":
    main()
