# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import math
import os
import tempfile
import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "MiniMaxAI/MiniMax-M3"
# MODEL_ID = "/home/smedhe/MINIMAX/tiny-random-minimax-m3/model.safetensors"


def _expand_batch(inputs, batch_size: int):
    """Repeat single-prompt tokenizer tensors for the compiled execution batch."""
    expanded = {}
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            expanded[name] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            expanded[name] = value
    return expanded


def _execution_batch_size(batch_size: int, msa_indexer_dp: int, msa_attn_dp: int) -> int:
    return batch_size * math.lcm(msa_indexer_dp, msa_attn_dp)


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
    msa_indexer_dp: int = 1,
    msa_indexer_cp: int = 1,
    msa_attn_dp: int = 1,
    msa_attn_cp: int = 1,
    indexer_n_head: int = 1,
    batch_size: int = 1,
    skip_kv: bool = False,
    num_kv_blocks: int = 2,
) -> None:
    """Compare HF PyTorch vs AIC on the last decode token of the prompt (prefill_seq_len=1)."""
    full_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    full_config.text_config.num_hidden_layers = 4

    torch.manual_seed(42)
    model_hf = AutoModelForImageTextToText.from_config(full_config).eval()
    model_dir = os.path.join(export_dir, "minimax-m3-parity")
    model_hf.save_pretrained(model_dir)

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
    execution_batch_size = _execution_batch_size(batch_size, msa_indexer_dp, msa_attn_dp)
    last_token_ids = last_token_ids.repeat(execution_batch_size, 1)

    with torch.no_grad():
        hf_logits = model_hf.language_model(input_ids=last_token_ids, use_cache=False).logits[:, -1:, :]
    expected_token = int(hf_logits.argmax(-1)[0, 0])

    qaic_config = {
        "skip_kv": skip_kv,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": expert_parallel_chunk_size,
            "cores_per_expert": cores_per_expert,
            "tree_reduce": tree_reduce,
        },
    }
    if msa_indexer_dp > 1 or msa_attn_dp > 1 or msa_attn_cp > 1:
        qaic_config["blocking_mode"] = "kv_headpar"
        qaic_config["num_kv_blocks"] = num_kv_blocks
        if msa_indexer_dp > 1 or msa_indexer_cp > 1:
            qaic_config["msa_indexer_dp"] = msa_indexer_dp
            qaic_config["msa_indexer_cp"] = msa_indexer_cp
            qaic_config["indexer_n_head"] = indexer_n_head
            qaic_config["num_cores_per_device"] = num_cores
        if msa_attn_dp > 1 or msa_attn_cp > 1:
            qaic_config["msa_attn_cp"] = msa_attn_cp
            qaic_config["msa_attn_dp"] = msa_attn_dp

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(model_dir, weight_free=True, torch_dtype=torch.float32)
    qeff_model.compile(
        batch_size=execution_batch_size,
        prefill_seq_len=1,
        prefill_only=False,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        use_onnx_subfunctions=False,
        skip_vision=True,
        node_precision_info=True,
        offload_pt_weights=False,
        dynamo=True,
        qaic_config=qaic_config,
    )

    aic_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs={"input_ids": last_token_ids},
        prefill_seq_len=1,
        batch_size=execution_batch_size,
    )
    output = qeff_model.generate(inputs=aic_inputs, generation_len=1)
    aic_token = int(output.generated_ids[0, 0])
    assert aic_token == expected_token, f"Parity check FAILED: expected {expected_token}, got {aic_token}"
    print(f"[PASS] PyTorch vs AIC parity check passed (token={aic_token})")


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M3 text-only decode (PL=1) with DP and GP enabled.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Logical prompt batch size; QEfficient expands it by the DP LCM for execution.",
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--skip-generate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--skip-kv",
        action="store_true",
        help="Skip KV blocks that are entirely in the future (disabled by default).",
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=2,
        help="Number of KV cache blocks used by blocked attention.",
    )
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
        "--msa-indexer-dp",
        type=int,
        default=2,
        help="DP factor for the MSA sparse-attention indexer (_select_blocks_dp path).",
    )
    parser.add_argument(
        "--msa-indexer-cp",
        type=int,
        default=2,
        help="CP factor for the MSA sparse-attention indexer compact cache layout.",
    )
    parser.add_argument(
        "--msa-attn-dp",
        type=int,
        default=2,
        help="DP factor for GP attention (_baseline_attention_gp path). Must divide batch_size.",
    )
    parser.add_argument(
        "--msa-attn-cp",
        type=int,
        default=1,
        help="CP factor for GP attention cache layout.",
    )
    parser.add_argument(
        "--indexer-n-head",
        type=int,
        default=1,
        help="Number of KV heads used by the MSA indexer in the DP path.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run PyTorch vs ONNX parity check using a tiny random model.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_kv_blocks < 1:
        parser.error("--num-kv-blocks must be positive")
    execution_batch_size = _execution_batch_size(args.batch_size, args.msa_indexer_dp, args.msa_attn_dp)

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
                msa_indexer_dp=args.msa_indexer_dp,
                msa_indexer_cp=args.msa_indexer_cp,
                msa_attn_dp=args.msa_attn_dp,
                msa_attn_cp=args.msa_attn_cp,
                indexer_n_head=args.indexer_n_head,
                batch_size=args.batch_size,
                skip_kv=args.skip_kv,
                num_kv_blocks=args.num_kv_blocks,
            )
        return

    factory_kwargs = dict(kv_offload=True, dtype=torch.float16)
    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers
    factory_kwargs["config"] = config

    t0 = time.perf_counter()
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_id, weight_free=True, **factory_kwargs)
    print(f"[timing] model load:          {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    qpc_paths = qeff_model.compile(
        batch_size=execution_batch_size,
        prefill_seq_len=1,
        prefill_only=False,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_onnx_subfunctions=True,
        dynamo=True,
        skip_vision=True,
        node_precision_info=True,
        offload_pt_weights=False,
        log_times=True,
        qaic_config={
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": args.num_kv_blocks,
            "skip_kv": args.skip_kv,
            "msa_indexer_dp": args.msa_indexer_dp,
            "msa_indexer_cp": args.msa_indexer_cp,
            "msa_attn_dp": args.msa_attn_dp,
            "msa_attn_cp": args.msa_attn_cp,
            "indexer_n_head": args.indexer_n_head,
            "num_cores_per_device": args.num_cores,
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
    inputs = _expand_batch(inputs, execution_batch_size)
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
