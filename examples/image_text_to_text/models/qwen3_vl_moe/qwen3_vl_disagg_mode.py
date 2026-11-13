# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Qwen3-VL-MoE disaggregated inference example.

Select an attention blocking mode via --mode:

  batch_fold   (default) KV_BATCH blocking with continuous batching (BS=256).
               Separate prefill / decode QPCs. Prefill uses blocking_mode
               "prefill_<prefill_mode>" (--prefill-mode online|qkv).

  no_blocking  Standard KV attention, no blocking config (BS=1).
               Separate prefill / decode QPCs, depth-first scheduling.

  headpar      KV_BATCH_FOLD decode + head-parallel prefill (BS=1).
               Separate prefill / decode QPCs. Prefill uses blocking_mode
               "prefill_<prefill_mode>" (--prefill-mode online|qkv).
"""

import argparse
from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

# ── Defaults (from batch_fold / qwen3_vl_disagg_mode.py) ─────────────────────
MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"
PREFILL_SEQ_LEN = 1024
CTX_LEN = 10240
NUM_KV_BLOCKS = 4
NUM_Q_BLOCKS = 2
PREFILL_QL_CHUNK = 128
PREFILL_N_REP_CHUNK = 4
EXPERT_PARALLEL_CHUNK_SIZE = 256
NUM_CORES = 16
NUM_DEVICES_DECODE = 16
NUM_DEVICES_PREFILL = 1
GENERATION_LEN = 30
NUM_LAYERS = 2
VISION_DEPTH = 9


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-VL-MoE disaggregated inference example")
    p.add_argument(
        "--mode",
        choices=["batch_fold", "no_blocking", "headpar"],
        default="batch_fold",
        help="Attention blocking mode (default: batch_fold)",
    )
    p.add_argument(
        "--prefill-mode",
        choices=["online", "qkv"],
        default="online",
        help="Prefill blocking sub-mode for batch_fold and headpar (default: online)",
    )
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--prefill-seq-len", type=int, default=PREFILL_SEQ_LEN)
    p.add_argument("--ctx-len", type=int, default=CTX_LEN)
    p.add_argument(
        "--bs",
        type=int,
        default=None,
        help="Batch size (default: 256 for batch_fold, 1 for no_blocking / headpar)",
    )
    p.add_argument("--num-kv-blocks", type=int, default=NUM_KV_BLOCKS)
    p.add_argument("--num-q-blocks", type=int, default=NUM_Q_BLOCKS, help="Prefill Q blocks for headpar mode")
    p.add_argument(
        "--prefill-ql-chunk", type=int, default=PREFILL_QL_CHUNK, help="Prefill Q chunk size (batch_fold only)"
    )
    p.add_argument(
        "--prefill-n-rep-chunk", type=int, default=PREFILL_N_REP_CHUNK, help="Prefill n_rep_chunk (batch_fold only)"
    )
    p.add_argument(
        "--expert-parallel-chunk-size",
        type=int,
        default=EXPERT_PARALLEL_CHUNK_SIZE,
        help="MoE expert-parallel chunk size for prefill (batch_fold and headpar only)",
    )
    p.add_argument("--num-cores", type=int, default=NUM_CORES)
    p.add_argument("--num-devices-decode", type=int, default=NUM_DEVICES_DECODE)
    p.add_argument("--num-devices-prefill", type=int, default=NUM_DEVICES_PREFILL)
    p.add_argument("--generation-len", type=int, default=GENERATION_LEN)
    p.add_argument(
        "--num-layers",
        type=int,
        default=NUM_LAYERS,
        help="Number of text layers to export (default: 2)",
    )
    p.add_argument("--skip-vision", action="store_true", default=True, help="Skip vision encoding (default: True)")
    p.add_argument("--no-skip-vision", dest="skip_vision", action="store_false", help="Enable vision encoding")
    return p.parse_args()


# ── QAIC config builders ──────────────────────────────────────────────────────


def _decode_qaic_config(args) -> dict | None:
    moe_config = {"flavour": "expert_parallel", "tree_reduce": True, "cores_per_expert": 2}
    if args.mode == "batch_fold":
        return {
            "blocking_mode": "kv_batch_fold",
            "num_kv_blocks": args.num_kv_blocks,
            "ctx_len": args.ctx_len,
            "moe_config": moe_config,
        }
    if args.mode == "headpar":
        return {
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": args.num_kv_blocks,
            "ctx_len": args.ctx_len,
            "moe_config": moe_config,
        }
    return None  # no_blocking: no qaic_config


def _prefill_qaic_config(args, prefill_block_chunks: int) -> dict | None:
    if args.mode == "batch_fold" or args.mode == "headpar":
        return {
            "blocking_mode": f"prefill_{args.prefill_mode}",
            "num_kv_blocks": args.num_kv_blocks,
            "num_q_blocks": prefill_block_chunks,
            "n_rep_chunk": args.prefill_n_rep_chunk,
            "ctx_len": args.ctx_len,
            "moe_config": {"expert_parallel_chunk_size": args.expert_parallel_chunk_size},
        }
    return None  # no_blocking: no qaic_config


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    bs = args.bs if args.bs is not None else (256 if args.mode == "batch_fold" else 1)
    num_layers = args.num_layers

    # ── Config ───────────────────────────────────────────────────────────────
    config = AutoConfig.from_pretrained(args.model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16
    # For faster execution user can run with lesser layers, For Testing Purpose Only
    config.vision_config.depth = VISION_DEPTH
    config.text_config.num_hidden_layers = num_layers
    config.vision_config.deepstack_visual_indexes = [VISION_DEPTH - 1]

    # ── Model loading ─────────────────────────────────────────────────────────
    load_kwargs = dict(
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float16,
        layerwise=False,
    )
    if args.mode == "batch_fold":
        load_kwargs["continuous_batching"] = True

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_id, **load_kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ── Vision compile (optional) ─────────────────────────────────────────────
    if not args.skip_vision:
        vision_qpc_path = qeff_model.compile(
            batch_size=bs,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            height=354,
            width=536,
            num_cores=args.num_cores,
            num_devices=1,
            mos=1,
            mxfp6_matmul=True,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            layerwise=False,
        )

    # ── Decode compile ────────────────────────────────────────────────────────
    # batch_fold:   kv_batch blocking, large batched decode, expert parallel
    # no_blocking:  plain depth-first, no blocking config
    # headpar:      kv_batch_fold blocking, user-tiled
    prefill_block_chunks = -(args.prefill_seq_len // -args.prefill_ql_chunk)
    decode_qaic_config = _decode_qaic_config(args)
    print("decode", decode_qaic_config)
    decode_start_time = perf_counter()

    decode_compile_kwargs = dict(
        batch_size=bs,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        height=354,
        width=536,
        num_cores=args.num_cores,
        num_devices=args.num_devices_decode,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        split_model_io=True,
        mos=1,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    if args.mode == "batch_fold":
        decode_compile_kwargs.update(
            full_batch_size=bs,
            kv_cache_batch_size=bs,
            user_tiled=True,
            qaic_config=decode_qaic_config,
        )
    elif args.mode == "headpar":
        decode_compile_kwargs.update(
            retain_full_kv=True,
            user_tiled=True,
            qaic_config=decode_qaic_config,
        )
    else:  # no_blocking
        decode_compile_kwargs["aic_enable_depth_first"] = True

    decode_qpc_path = qeff_model.compile(**decode_compile_kwargs)
    print(f"Decode export + compile time is {(perf_counter() - decode_start_time):.3f}s")

    # ── Prefill compile ───────────────────────────────────────────────────────
    # batch_fold:   bs=1, kv_cache_batch_size=7, expert_parallel_chunk_size via moe_config
    # no_blocking:  bs=<bs>, depth-first, no blocking config
    # headpar:      bs=1, user-tiled, prefill blocking config
    prefill_qaic_config = _prefill_qaic_config(args, prefill_block_chunks)
    print("prefill", prefill_qaic_config)
    prefill_start_time = perf_counter()

    prefill_compile_kwargs = dict(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        height=354,
        width=536,
        num_cores=args.num_cores,
        num_devices=args.num_devices_prefill,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_model_io=True,
        mos=1,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=True,
    )
    if args.mode == "batch_fold":
        prefill_compile_kwargs.update(
            batch_size=1,
            full_batch_size=1,
            kv_cache_batch_size=1,
            user_tiled=True,
            qaic_config=prefill_qaic_config,
        )
    elif args.mode == "headpar":
        prefill_compile_kwargs.update(
            batch_size=1,
            user_tiled=True,
            qaic_config=prefill_qaic_config,
        )
    else:  # no_blocking
        prefill_compile_kwargs.update(
            batch_size=bs,
            aic_enable_depth_first=True,
        )

    prefill_qpc_path = qeff_model.compile(**prefill_compile_kwargs)
    print(f"Prefill export + compile time is {(perf_counter() - prefill_start_time):.3f}s")

    print(f"Prefill qpc path {prefill_qpc_path}")
    print(f"Decode qpc path {decode_qpc_path}")

    # ── Sessions ──────────────────────────────────────────────────────────────
    # For batch_fold the decode session is created after prefill deactivation;
    # for the other modes both sessions can coexist.
    lang_prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"))
    if args.mode != "batch_fold":
        lang_decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))

    # ── Messages / inputs ─────────────────────────────────────────────────────
    if args.skip_vision:
        messages = [{"role": "user", "content": [{"type": "text", "text": "Tell me about yourself."}]}]
    else:
        image_url = "https://picsum.photos/id/237/536/354"
        image = Image.open(requests.get(image_url, stream=True).raw)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe all the colors seen in the image."},
                ],
            }
        ]
        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))

    messages = [messages] * bs

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs, prefill_seq_len=args.prefill_seq_len, batch_size=bs
    )

    pad_token_id = 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -args.prefill_seq_len)
    padded_len = num_chunks * args.prefill_seq_len
    print(f"generation_len : {args.generation_len}")

    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
    )

    for k, v in inputs.items():
        inputs[k] = np.array(v)

    vision_inputs = {
        k: v
        for k, v in inputs.items()
        if k in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
    }
    vision_inputs_fp16 = {"pixel_values", "image_masks"}
    vision_inputs.update({k: vision_inputs[k].astype("float16") for k in vision_inputs_fp16 if k in vision_inputs})

    vision_start = perf_counter()
    vision_outputs = {}
    if vision_inputs:
        vision_outputs = vision_session.run(vision_inputs)
    vision_end = perf_counter()

    lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask")
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    lang_inputs["image_idx"] = np.array([[0]])

    if not args.skip_vision:
        lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]
        lang_inputs["deepstack_features"] = vision_outputs["deepstack_features"]

    # ── Prefill ───────────────────────────────────────────────────────────────
    # batch_fold slices to a single batch item (all requests are identical for
    # this example); no_blocking and headpar run the full batch.
    # batch_fold also requires batch_index in chunk_inputs.
    lang_start = perf_counter()
    lang_prefill_session.set_buffers(vision_outputs)
    all_outputs = []
    chunk_inputs = lang_inputs.copy()

    if args.mode == "batch_fold":
        chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)

    for i in range(num_chunks):
        if args.mode == "batch_fold":
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][
                0:1, i * args.prefill_seq_len : (i + 1) * args.prefill_seq_len
            ]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                :, 0:1, i * args.prefill_seq_len : (i + 1) * args.prefill_seq_len
            ]
        else:
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][
                :, i * args.prefill_seq_len : (i + 1) * args.prefill_seq_len
            ]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., i * args.prefill_seq_len : (i + 1) * args.prefill_seq_len
            ]
        outputs = lang_prefill_session.run(chunk_inputs)
        for j in range(num_layers):
            chunk_inputs[f"past_key.{j}"] = outputs[f"past_key.{j}_RetainedState"]
            chunk_inputs[f"past_value.{j}"] = outputs[f"past_value.{j}_RetainedState"]
        chunk_inputs["image_idx"] = outputs["image_idx_output"]

    prefill_time = perf_counter() - lang_start + vision_end - vision_start
    print(f"Prefill time : {prefill_time:.2f} secs")

    # ── Build initial decode inputs from prefill output ───────────────────────
    if args.mode == "batch_fold":
        # Deactivate prefill before loading decode on the same devices.
        lang_prefill_session.deactivate()
        lang_decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))

        next_token_id = np.argmax(outputs["logits"])
        all_outputs.append(next_token_id)
        next_pos = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1

        # Simulate vLLM physical-slot permutation.
        batch_index = np.random.default_rng(1234).permutation(bs).reshape(bs, 1).astype(np.int64)
        physical_slots = batch_index[:, 0]
        decode_inputs = {
            "input_ids": np.full((bs, 1), next_token_id, dtype=lang_inputs["input_ids"].dtype),
            "position_ids": next_pos,
            "batch_index": batch_index,
        }
        for layer_idx in range(num_layers):
            for cache_name in ("past_key", "past_value"):
                prefill_cache = outputs[f"{cache_name}.{layer_idx}_RetainedState"]
                logical_cache = np.tile(prefill_cache[0:1], (bs, 1, 1, 1))
                physical_cache = np.empty_like(logical_cache)
                physical_cache[physical_slots] = logical_cache
                decode_inputs[f"{cache_name}.{layer_idx}"] = physical_cache
    else:
        first_token = np.argmax(outputs["logits"])
        all_outputs.append(first_token)
        decode_inputs = {
            "input_ids": first_token.reshape(1, 1),
            "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
        }
        if args.mode == "headpar":
            decode_inputs["image_idx"] = outputs["image_idx_output"]
        for layer in range(num_layers):
            decode_inputs[f"past_key.{layer}"] = outputs[f"past_key.{layer}_RetainedState"]
            decode_inputs[f"past_value.{layer}"] = outputs[f"past_value.{layer}_RetainedState"]

    # ── First decode step ─────────────────────────────────────────────────────
    st = perf_counter()
    decode_out = lang_decode_session.run(decode_inputs)
    print(f"time for first run of decode with KV as input = {perf_counter() - st} sec\n")

    # ── Build loop inputs from first decode output ────────────────────────────
    pos_id = decode_inputs["position_ids"] + 1
    if args.mode == "batch_fold":
        all_outputs.append(np.argmax(decode_out["logits"][0]))
        loop_decode_inputs = {
            "input_ids": np.argmax(decode_out["logits"], axis=-1),
            "position_ids": pos_id,
            "batch_index": batch_index,
        }
    else:
        all_outputs.append(np.argmax(decode_out["logits"]))
        loop_decode_inputs = {
            "input_ids": np.argmax(decode_out["logits"], axis=-1).reshape(1, 1),
            "position_ids": pos_id,
        }
        if args.mode == "headpar":
            loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]

    for j in range(num_layers):
        loop_decode_inputs[f"past_key.{j}"] = decode_out[f"past_key.{j}_RetainedState"]
        loop_decode_inputs[f"past_value.{j}"] = decode_out[f"past_value.{j}_RetainedState"]

    # ── Decode loop ───────────────────────────────────────────────────────────
    st = perf_counter()
    for _ in range(args.generation_len - 2):
        decode_out = lang_decode_session.run(loop_decode_inputs)
        if args.mode == "batch_fold":
            all_outputs.append(np.argmax(decode_out["logits"][0]))
            next_ids = np.argmax(decode_out["logits"], axis=-1)
        else:
            all_outputs.append(np.argmax(decode_out["logits"]))
            next_ids = np.argmax(decode_out["logits"], axis=-1).reshape(1, 1)
        pos_id = pos_id + 1
        for j in range(num_layers):
            loop_decode_inputs[f"past_key.{j}"] = decode_out[f"past_key.{j}_RetainedState"]
            loop_decode_inputs[f"past_value.{j}"] = decode_out[f"past_value.{j}_RetainedState"]
        loop_decode_inputs["input_ids"] = next_ids
        loop_decode_inputs["position_ids"] = pos_id
        if args.mode == "batch_fold":
            loop_decode_inputs["batch_index"] = batch_index
        elif args.mode == "headpar":
            loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]

    ft = perf_counter()
    print(f"decode tok/sec={(args.generation_len - 2) / (ft - st)}")
    print(f"\noutput\n{tokenizer.decode(all_outputs)}")


if __name__ == "__main__":
    main()
