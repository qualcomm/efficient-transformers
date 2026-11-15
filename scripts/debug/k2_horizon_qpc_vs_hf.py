# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Compare a compiled QPC with the original Hugging Face model, token by token, at the
shapes we ship (prefill length, context, devices, precision), including long context.

Two modes:

  prompt: one prompt, greedy on both sides, report the first token where they differ.
    python scripts/debug/k2_horizon_qpc_vs_hf.py --model-name IFM/K2-Horizon-7B --prefill-seq-len 128 --ctx-len 4096 --device-group '[0]'
    python scripts/debug/k2_horizon_qpc_vs_hf.py --model-name IFM/K2-Horizon-7B --prefill-seq-len 128 --ctx-len 4096 --device-group '[0,1,2,3]' \\
        --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions

  needle: a prompt of --needle-tokens tokens of prose with a code word planted near the
  start and a question at the end; pass = the QPC answers with the code word. Checks the
  rope and cache at long range. The HF leg is optional here (slow on CPU at 50K tokens).
    python scripts/debug/k2_horizon_qpc_vs_hf.py --model-name IFM/K2-Horizon-7B --needle-tokens 50000 --prefill-seq-len 1024 --ctx-len 65536 \\
        --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --skip-hf

Pass in prompt mode: the first --min-match tokens are identical (default 16; fp16 vs bf16
usually agree for 20+ tokens, then they drift, which is normal). For a quantized QPC
(MXFP6) pass --hf-topk K instead: the QPC's own tokens are fed back to HF and every one
must be inside HF's top-K at its position, which judges a run that forked on a near-tie
at the first token without calling it wrong. Junk text is a failure under both. Results go to --out-dir (default: scripts/debug/results/).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants

FILLER = (
    "The harbour town wakes before the sun. Fishing boats leave the quay in a line, their engines "
    "low, and the gulls follow them out past the breakwater. On the hill the bakery has been open "
    "for an hour; the smell of bread drifts down the steps to the market, where the first stalls "
    "are being set up under canvas. By eight the square is full. Farmers sell apples and cheese, "
    "a man repairs clocks at a folding table, and children chase each other between the carts. "
    "The church bell marks the hour. Nobody hurries, but everything gets done. "
)
NEEDLE = "Remember this: the secret code word is {word}. "
QUESTION = "\n\nWhat is the secret code word? Answer with the word only. The secret code word is"


def build_needle_prompt(tokenizer, total_tokens, word):
    """Prose of about total_tokens tokens with the code word planted at about 10% depth."""
    filler_ids = tokenizer(FILLER, add_special_tokens=False)["input_ids"]
    needle_ids = tokenizer(NEEDLE.format(word=word), add_special_tokens=False)["input_ids"]
    question_ids = tokenizer(QUESTION, add_special_tokens=False)["input_ids"]
    body_tokens = total_tokens - len(needle_ids) - len(question_ids) - 1
    repeats = body_tokens // len(filler_ids) + 1
    before = max(1, int(repeats * 0.1))
    text = FILLER * before + NEEDLE.format(word=word) + FILLER * (repeats - before)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:body_tokens]
    return tokenizer.decode(ids) + QUESTION


def trim(ids, eos_ids, pad_id):
    out = []
    for t in ids:
        if t in eos_ids or t == pad_id:
            break
        out.append(int(t))
    return out


def run_qpc(args, tokenizer, prompt):
    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=len(args.device_group),
        mxfp6_matmul=args.mxfp6,
        mxint8_kv_cache=args.mxint8_kv_cache,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        aic_enable_depth_first=True,
        mos=1,
    )
    print(f"QPC: {qpc_path}")
    exec_info = model.generate(
        tokenizer=tokenizer, prompts=[prompt], device_ids=args.device_group, generation_len=args.generation_len
    )
    print(exec_info.perf_metrics)
    return np.asarray(exec_info.generated_ids).reshape(-1).tolist()


def load_hf(args):
    dtype = getattr(torch, args.hf_dtype)
    return AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, dtype=dtype).eval()


def run_hf(model, tokenizer, prompt, generation_len):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=generation_len, do_sample=False)
    return out[0][inputs["input_ids"].shape[1] :].tolist()


def hf_ranks(model, tokenizer, prompt, qpc_ids):
    """Rank (0 = argmax) of each QPC token under HF's distribution, teacher-forced on the QPC's own prefix."""
    prompt_ids = tokenizer(prompt)["input_ids"]
    ids = torch.tensor([prompt_ids + qpc_ids[:-1]])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 :].float()
    chosen = logits.gather(1, torch.tensor(qpc_ids).unsqueeze(1))
    return (logits > chosen).sum(1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="IFM/K2-Horizon-7B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--needle-tokens", type=int, help="needle mode: prompt length in tokens")
    parser.add_argument("--needle-word", default="marigold")
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--generation-len", type=int, default=64)
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES)
    parser.add_argument(
        "--device-group",
        type=lambda ids: [int(x) for x in ids.strip("[]").split(",")],
        default=[0],
        help="e.g. '[0,1,2,3]'",
    )
    parser.add_argument("--mxfp6", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--hf-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--min-match", type=int, default=16, help="prompt mode: identical tokens needed to pass")
    parser.add_argument(
        "--hf-topk",
        type=int,
        help="prompt mode, quantized QPC: pass if every QPC token is in HF's top-K (teacher-forced)",
    )
    parser.add_argument("--skip-hf", action="store_true", help="needle mode: judge the QPC answer only")
    parser.add_argument("--skip-qpc", action="store_true", help="laptop dry run of the prompt and HF legs")
    parser.add_argument("--tag", default="", help="suffix for the results file")
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).with_name("results")), help="where the result JSON goes"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

    needle = args.needle_tokens is not None
    prompt = build_needle_prompt(tokenizer, args.needle_tokens, args.needle_word) if needle else args.prompt
    prompt_tokens = len(tokenizer(prompt)["input_ids"])
    if prompt_tokens + args.generation_len > args.ctx_len:
        raise SystemExit(
            f"prompt ({prompt_tokens} tokens) + generation ({args.generation_len}) exceeds ctx_len {args.ctx_len}"
        )
    print(f"prompt tokens: {prompt_tokens}, ctx_len: {args.ctx_len}, generation_len: {args.generation_len}")

    result = {
        "model": args.model_name,
        "mode": "needle" if needle else "prompt",
        "prompt_tokens": prompt_tokens,
        "prefill_seq_len": args.prefill_seq_len,
        "ctx_len": args.ctx_len,
        "devices": args.device_group,
        "mxfp6": args.mxfp6,
        "mxint8_kv_cache": args.mxint8_kv_cache,
    }

    qpc_ids = None if args.skip_qpc else trim(run_qpc(args, tokenizer, prompt), eos_ids, pad_id)
    hf_model = None if (needle and args.skip_hf) else load_hf(args)
    hf_ids = (
        None if hf_model is None else trim(run_hf(hf_model, tokenizer, prompt, args.generation_len), eos_ids, pad_id)
    )
    if qpc_ids is not None:
        result["qpc_text"] = tokenizer.decode(qpc_ids)
        print(f"QPC: {result['qpc_text']!r}")
    if hf_ids is not None:
        result["hf_text"] = tokenizer.decode(hf_ids)
        print(f"HF : {result['hf_text']!r}")

    if needle:
        word = args.needle_word.lower()
        checks = {}
        if qpc_ids is not None:
            checks["qpc"] = word in result["qpc_text"].lower()
        if hf_ids is not None:
            checks["hf"] = word in result["hf_text"].lower()
        result["found"] = checks
        ok = all(checks.values()) and bool(checks)
    else:
        if qpc_ids is None or hf_ids is None:
            raise SystemExit("prompt mode needs both legs")
        n = min(len(qpc_ids), len(hf_ids))
        first_diff = next((i for i in range(n) if qpc_ids[i] != hf_ids[i]), n)
        result.update({"first_diff": first_diff, "compared": n, "min_match": args.min_match})
        print(f"first differing token: {first_diff} of {n} compared (need {args.min_match})")
        if args.hf_topk is None:
            ok = first_diff >= args.min_match
        else:
            ranks = hf_ranks(hf_model, tokenizer, prompt, qpc_ids)
            result.update({"hf_topk": args.hf_topk, "ranks": ranks, "max_rank": max(ranks)})
            ok = max(ranks) < args.hf_topk
            print(
                f"teacher-forced ranks under HF: max {max(ranks)}, {sum(r < args.hf_topk for r in ranks)}/{len(ranks)} in top-{args.hf_topk}"
            )

    result["pass"] = ok
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or re.sub(
        r"[^A-Za-z0-9]+", "_", f"{result['mode']}_ctx{args.ctx_len}_{'mx' if args.mxfp6 else 'fp16'}"
    )
    out_path = out_dir / f"qpc_vs_hf_{args.model_name.split('/')[-1]}_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"RESULT {json.dumps({k: result[k] for k in ('model', 'mode', 'ctx_len', 'pass')})}")
    print(f"written: {out_path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
