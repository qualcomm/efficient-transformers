# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Single-entry SPD single-prompt runner.

Given a TLM model_name (short name OR full HF repo path) and a prompt, this
script:
  1. Looks up the matching DFlash DLM repo on Hugging Face.
  2. Compiles TLM + DLM QPCs (only the side(s) not provided via
     --tlm_qpc / --dlm_qpc).
  3. Runs SPD single-prompt inference in-process via
     QEfficient.generation.dflash_generation.run_spd_inference_single.

Examples:
    # Compile + run with all defaults
    python basic_inference_text.py --model_name Qwen3-4B \\
        --prompt "Explain speculative decoding in two sentences."

    # Full HF path also accepted
    python basic_inference_text.py --model_name Qwen/Qwen3-4B \\
        --prompt "Hello"

    # Reuse pre-compiled QPCs
    python basic_inference_text.py --model_name Qwen3-4B \\
        --tlm_qpc /path/to/tlm/qpc --dlm_qpc /path/to/dlm/qpc \\
        --prompt "What is 17 * 23?"
"""

import argparse
import os
import sys

from rich.console import Console
from rich.markup import escape
from transformers import AutoConfig, AutoTokenizer
from utils import (
    MODEL_MAP,
    compile_dlm_qpc,
    compile_tlm_qpc,
    format_prompt,
    get_spd_prompt_chunk_size,
    load_spd_sessions,
    resolve_model_name,
    validate_spd_decode_specialization,
)

from QEfficient.generation.dflash_generation import run_spd_inference_single
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

console = Console()


def parse_device_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model_name",
        required=True,
        type=resolve_model_name,
        help="TLM name — either the short key (e.g. 'Qwen3-4B') or "
        "the full HF repo path (e.g. 'Qwen/Qwen3-4B'). "
        f"Supported: {', '.join(MODEL_MAP.keys())}",
    )
    p.add_argument("--prompt", required=True, help="Input prompt text.")
    p.add_argument(
        "--category",
        default="",
        help="Prompt category for formatting (math, coding, reasoning, …).",
    )
    p.add_argument(
        "--format_prompt",
        action="store_true",
        help="If set, wrap the prompt with the category-specific reasoning/coding template before sending to the model. "
        "Off by default — the prompt is used verbatim.",
    )
    p.add_argument(
        "--tlm_hf_path",
        default=None,
        help="Override TLM HF repo (required if mapping has None).",
    )

    # Optional pre-built QPCs (skip compilation)
    p.add_argument("--tlm_qpc", default=None, help="Pre-compiled TLM qpc dir (skip TLM compile).")
    p.add_argument("--dlm_qpc", default=None, help="Pre-compiled DLM qpc dir (skip DLM compile).")

    # Devices / cores
    p.add_argument(
        "--tlm_devices",
        type=parse_device_list,
        default=[40, 41, 42, 43],
        help="Comma-separated device IDs, e.g. '0,1,2,3' or '0'.",
    )
    p.add_argument(
        "--dlm_devices",
        type=parse_device_list,
        default=[40, 41, 42, 43],
        help="Comma-separated device IDs, e.g. '0,1,2,3' or '0'.",
    )
    p.add_argument("--tlm_cores", type=int, default=8)
    p.add_argument("--dlm_cores", type=int, default=8)

    # Compile / run knobs
    p.add_argument("--ctx_len", type=int, default=4096)
    p.add_argument("--prefill_seq_len", type=int, default=128)
    p.add_argument("--generation_len", type=int, default=256)
    p.add_argument("--iteration", type=int, default=300)
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Requested batch size. QEfficient currently supports batch_size=1 only; higher values fall back to 1.",
    )

    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def _resolve_batch_size(batch_size):
    if batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if batch_size > 1:
        logger.warning(
            "QEfficient currently does not support batch sizes greater than 1. "
            "Falling back to batch_size=1; use vLLM for higher batch sizes."
        )
        return 1
    return batch_size


def main():
    args = parse_args()
    batch_size = _resolve_batch_size(args.batch_size)

    tlm_repo_default, dlm_repo = MODEL_MAP[args.model_name]
    tlm_repo = args.tlm_hf_path or tlm_repo_default
    if tlm_repo is None:
        raise SystemExit(f"No default TLM HF path for '{args.model_name}'. Pass --tlm_hf_path.")

    if args.tlm_qpc:
        logger.info(f"[skip compile] using provided TLM qpc: {args.tlm_qpc}")
        tlm_qpc = args.tlm_qpc
    else:
        tlm_qpc = compile_tlm_qpc(
            tlm_repo,
            dlm_repo,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            num_cores=args.tlm_cores,
            num_devices=len(args.tlm_devices),
            hf_token=args.hf_token,
        )

    if args.dlm_qpc:
        logger.info(f"[skip compile] using provided DLM qpc: {args.dlm_qpc}")
        dlm_qpc = args.dlm_qpc
    else:
        dlm_qpc = compile_dlm_qpc(
            tlm_repo,
            dlm_repo,
            ctx_len=args.ctx_len,
            num_cores=args.dlm_cores,
            num_devices=len(args.dlm_devices),
            hf_token=args.hf_token,
        )
    logger.info(f"TLM qpc        : {tlm_qpc}")
    logger.info(f"DLM qpc        : {dlm_qpc}")

    prompt_text = format_prompt(args.prompt, args.category) if args.format_prompt else args.prompt
    tokenizer = AutoTokenizer.from_pretrained(tlm_repo, token=args.hf_token, trust_remote_code=True)
    config = AutoConfig.from_pretrained(dlm_repo, token=args.hf_token, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dlm_session, tlm_session = load_spd_sessions(tlm_qpc, dlm_qpc, args.tlm_devices, args.dlm_devices)
    prompt_chunk_size = get_spd_prompt_chunk_size(tlm_session)
    validate_spd_decode_specialization(tlm_session, config.block_size)

    dflash_config = getattr(config, "dflash_config", None) or config.to_dict().get("dflash_config", {})
    mask_token_id = dflash_config["mask_token_id"] if isinstance(dflash_config, dict) else dflash_config.mask_token_id
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    metrics = run_spd_inference_single(
        prompt_text=formatted_prompt,
        tokenizer=tokenizer,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        mask_token_id=mask_token_id,
        vocab_size=config.vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=args.ctx_len,
        block_size=config.block_size,
        max_iterations=args.iteration,
        hidden_size=config.hidden_size,
        generation_len=args.generation_len,
        batch_size=batch_size,
    )

    output_parts = ["Output: "]
    for tok_id, source in zip(metrics.generated_ids, metrics.generated_sources):
        text = escape(tokenizer.decode([tok_id], skip_special_tokens=True))
        if source == "dlm":
            output_parts.append(f"[blue]{text}[/blue]")
        else:
            output_parts.append(f"[white]{text}[/white]")
    console.print("".join(output_parts))

    ar = metrics.acceptance_rate()
    dlm_tps = metrics.dlm_tok_rate()
    tlm_tps = metrics.tlm_tok_rate()
    spd_tps = metrics.spd_tok_rate()

    w = 46
    print("\n" + "=" * w)
    print("  SPD Inference — Metrics")
    print("=" * w)
    print(f"  {'Acceptance Rate (tok/iter)':<30} {ar:>6.2f}")
    print(f"  {'DLM Throughput  (tok/s)':<30} {dlm_tps:>6.1f}")
    print(f"  {'TLM Throughput  (tok/s)':<30} {tlm_tps:>6.1f}")
    print(f"  {'SPD Decode Speed (tok/s)':<30} {spd_tps:>6.1f}")
    print(f"  {'Generated tokens':<30} {metrics.total_generated_tokens:>6}")
    print(f"  {'Iterations':<30} {metrics.num_total_iters:>6}")
    print(f"  {'Prefill time (s)':<30} {metrics.total_prefill_time:>6.3f}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
