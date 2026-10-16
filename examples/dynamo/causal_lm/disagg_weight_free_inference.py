# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compile weight-free disaggregated prefill and decode QPCs.

This example creates separate QPCs for the prefill and decode workers. It does
not run the serving loop; the resulting paths can be passed to the corresponding
prefill/decode runtime sessions or a disaggregated serving integration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile weight-free disaggregated prefill and decode QPCs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Hugging Face model ID")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.")
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("qeff_disagg_weight_free"))
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--full-batch-size", type=int, default=2)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-hidden-layers", type=int, default=-1)
    parser.add_argument("--expert-parallel-chunk-size", type=int, default=16)
    parser.add_argument("--continuous-batching", action="store_true")
    parser.add_argument("--split-retained-state-io", action="store_true")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers

    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        trust_remote_code=True,
        weight_free=True,
        continuous_batching=args.continuous_batching,
    )

    common = {
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "num_devices": args.num_devices,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "retain_full_kv": True,
        "use_onnx_subfunctions": True,
    }
    if args.continuous_batching:
        common["full_batch_size"] = args.full_batch_size
        common["split_retained_state_io"] = True
    elif args.split_retained_state_io:
        common["split_retained_state_io"] = True

    decode_qpc = model.compile(
        compile_dir=str(args.output_dir / "decode"),
        prefill_seq_len=1,
        **common,
    )
    prefill_qpc = model.compile(
        compile_dir=str(args.output_dir / "prefill"),
        prefill_seq_len=args.prefill_seq_len,
        prefill_only=True,
        enable_chunking=True,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": args.expert_parallel_chunk_size}},
        **common,
    )

    print(f"Decode QPC : {decode_qpc}")
    print(f"Prefill QPC: {prefill_qpc}")

    if args.continuous_batching:
        print("QPC compilation complete. Runtime CB KV-DMA handoff is not exercised by this example.")
        print("Use the generated QPCs with the CB disaggregated serving runtime.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    encoded = tokenizer(
        args.prompt,
        return_tensors="np",
        padding="max_length",
        max_length=args.prefill_seq_len,
    )
    attention_mask = encoded.pop("attention_mask")
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "position_ids": np.where(attention_mask, np.arange(args.prefill_seq_len), -1).astype(np.int64),
    }

    prefill_session = QAICInferenceSession(str(prefill_qpc))
    decode_session = QAICInferenceSession(str(decode_qpc))
    prefill_outputs = prefill_session.run(inputs)

    def last_token(logits):
        return int(np.argmax(logits.reshape(logits.shape[0], -1, logits.shape[-1])[:, -1, :], axis=-1)[0])

    token = last_token(prefill_outputs["logits"])
    generated = [token]
    decode_inputs = {
        "input_ids": np.array([[token]], dtype=np.int64),
        "position_ids": np.array([[args.prefill_seq_len]], dtype=np.int64),
    }
    for layer_idx in range(config.num_hidden_layers):
        for cache_name in ("past_key", "past_value"):
            output_name = f"{cache_name}.{layer_idx}_RetainedState"
            if output_name not in prefill_outputs:
                output_name = f"{cache_name}.{layer_idx}"
            decode_inputs[f"{cache_name}.{layer_idx}"] = prefill_outputs[output_name]

    for position in range(args.generation_len - 1):
        decode_outputs = decode_session.run(decode_inputs)
        token = last_token(decode_outputs["logits"])
        generated.append(token)
        decode_inputs["input_ids"] = np.array([[token]], dtype=np.int64)
        decode_inputs["position_ids"] = np.array([[args.prefill_seq_len + position + 1]], dtype=np.int64)

    print(f"Prompt   : {args.prompt}")
    print(f"Generated: {tokenizer.decode(generated, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
