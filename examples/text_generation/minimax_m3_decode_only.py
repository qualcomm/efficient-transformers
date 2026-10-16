# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
import tempfile
import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def _run_pytorch_parity_test(
    model_id: str,
    prompt: str,
    export_dir: str,
    ctx_len: int = 128,
    num_cores: int = 16,
    num_devices: int = 1,
    expert_parallel_chunk_size: int = 256,
    cores_per_expert: int = 2,
    tree_reduce: bool = True,
) -> None:
    """Compare HF PyTorch vs AIC on the last decode token of the prompt (prefill_seq_len=1)."""
    # Load real model architecture with 4 layers for a fast test (random weights).
    full_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    full_config.text_config.num_hidden_layers = 4

    torch.manual_seed(42)
    model_hf = AutoModelForImageTextToText.from_config(full_config).eval()
    model_dir = os.path.join(export_dir, "minimax-m3-parity")
    model_hf.save_pretrained(model_dir)

    # Tokenize the real prompt and take the last token as the single decode input.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    last_token_ids = inputs["input_ids"][:, -1:]

    with torch.no_grad():
        hf_logits = model_hf.language_model(input_ids=last_token_ids, use_cache=False).logits[:, -1:, :]
    expected_token = int(hf_logits.argmax(-1)[0, 0])

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.float32)
    qeff_model.compile(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        use_onnx_subfunctions=False,
        skip_vision=True,
        node_precision_info=True,
        offload_pt_weights=False,
        weight_free=True,
        qaic_config={
            "moe_config": {
                "flavour": "expert_parallel",
                "expert_parallel_chunk_size": expert_parallel_chunk_size,
                "cores_per_expert": cores_per_expert,
                "tree_reduce": tree_reduce,
            }
        },
    )

    aic_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs={"input_ids": last_token_ids}, prefill_seq_len=1, batch_size=1
    )
    output = qeff_model.generate(inputs=aic_inputs, generation_len=1)
    aic_token = int(output.generated_ids[0, 0])
    assert aic_token == expected_token, f"Parity check FAILED: expected {expected_token}, got {aic_token}"
    print(f"[PASS] PyTorch vs AIC parity check passed (token={aic_token})")


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M3 text-only decode (PL=1).")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--skip-generate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--expert-parallel-chunk-size",
        type=int,
        default=256,
        help="MoE expert-parallel chunk size (expert_parallel_chunk_size in moe_config).",
    )
    parser.add_argument(
        "--cores-per-expert",
        type=int,
        default=2,
        help="Number of NSP cores assigned to each expert during decode.",
    )
    parser.add_argument(
        "--no-tree-reduce",
        dest="tree_reduce",
        action="store_false",
        default=True,
        help="Disable tree-reduce for MoE expert-parallel dispatch.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run PyTorch vs ONNX parity check using a tiny random model.",
    )
    args = parser.parse_args()

    if args.test:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _run_pytorch_parity_test(
                model_id=args.model_id,
                prompt=args.prompt,
                export_dir=tmp_dir,
                ctx_len=args.ctx_len,
                num_cores=args.num_cores,
                num_devices=args.num_devices,
                expert_parallel_chunk_size=args.expert_parallel_chunk_size,
                cores_per_expert=args.cores_per_expert,
                tree_reduce=args.tree_reduce,
            )
        return

    factory_kwargs = dict(kv_offload=True, dtype=torch.float16)
    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers
    factory_kwargs["config"] = config

    t0 = time.perf_counter()
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_id, **factory_kwargs)
    print(f"[timing] model load:          {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    qpc_paths = qeff_model.compile(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_onnx_subfunctions=True,
        skip_vision=True,
        node_precision_info=True,
        offload_pt_weights=False,
        log_times=True,
        qaic_config={
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": 2,
            "moe_config": {
                "flavour": "expert_parallel",
                "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
                "cores_per_expert": args.cores_per_expert,
                "tree_reduce": args.tree_reduce,
            },
        },
    )
    print(f"[timing] compile total:       {time.perf_counter() - t0:.2f}s")
    print(f"QPC paths: {qpc_paths}")

    if args.skip_generate:
        return

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            }
        ]
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    t0 = time.perf_counter()
    output = qeff_model.generate(inputs=inputs, generation_len=args.generation_len)
    generate_time = time.perf_counter() - t0

    num_generated = output.generated_ids.shape[-1]
    toks_per_sec = num_generated / float(generate_time)
    print(f"[timing] generation:          {generate_time:.2f}s  ({num_generated} tokens, {toks_per_sec:.02f} tok/s)")

    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))


if __name__ == "__main__":
    main()
