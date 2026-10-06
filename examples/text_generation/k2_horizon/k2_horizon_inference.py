# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Compile and run a dense K2 Horizon model (IFM/K2-Horizon-*) on Cloud AI 100.
The model ships as remote code, so trust_remote_code is always passed.
"""

import argparse

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="K2 Horizon text generation on Cloud AI 100")
    parser.add_argument("--model-name", default="IFM/K2-Horizon-7B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--generation-len", type=int, default=128)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument(
        "--device-group",
        type=lambda ids: [int(x) for x in ids.strip("[]").split(",")],
        default=[0],
        help="Device ids, e.g. [0] or [0,1]",
    )
    parser.add_argument("--mxfp6", action="store_true", help="MXFP6 weights for the matmuls")
    parser.add_argument("--mxint8-kv-cache", action="store_true", help="MXINT8 KV cache")
    parser.add_argument("--use-onnx-subfunctions", action="store_true", help="Faster export and compile")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
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
    print(f"Model compiled to: {qpc_path}")
    if args.compile_only:
        return

    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_ids=args.device_group,
        generation_len=args.generation_len,
    )
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")
    print(exec_info.perf_metrics)


if __name__ == "__main__":
    main()
