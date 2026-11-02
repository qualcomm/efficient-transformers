# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Continuous-batching disaggregated prefill/decode for Qwen3-VL-MoE — DMA KV handoff, with a
BATCH-FOLDED decode."""

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

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"
DEFAULT_PROMPTS = [
    "Tell me about yourself.",
    "What is the capital of France?",
    "Who is the President of USA?",
    "Explain quantum computing in simple terms?",
] * 64
DEFAULT_IMAGE_PROMPTS = [
    "Describe all the colors seen in the image",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "describe the image?",
]
DEFAULT_IMAGE_URLS = [
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/230/536/354",
    "https://picsum.photos/id/234/536/354",
    "https://picsum.photos/id/235/536/354",
]
DEFAULT_PREFILL_SEQ_LEN = 1024
DEFAULT_CTX_LEN = 10240
DEFAULT_GENERATION_LEN = 50
DEFAULT_FULL_BATCH_SIZE = 256

STAGES = 2  # 8
PREFILL_NUM_DEVICES = 2  # 16
DECODE_NUM_DEVICES = 1  # 16

NUM_KV_BLOCKS = 4


PREFILL_BLOCKING_MODE = "online"
PREFILL_QL_CHUNK = 128
PREFILL_N_REP_CHUNK = 4
MOE_PREFILL_PACKED_CHUNK_SIZE = 256

VISION_INPUT_KEYS = {
    "pixel_values",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
    "image_grid_thw",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUT_KEYS = ("vision_embeds", "deepstack_features")


def _build_config(model_id: str):
    """Load the model config, pinned to float16 (matches the single-request baseline)."""
    config = AutoConfig.from_pretrained(model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16

    # For faster execution user can run with fewer layers. For testing purposes only.
    config.vision_config.depth = 9
    config.text_config.num_hidden_layers = 2
    config.vision_config.deepstack_visual_indexes = [8]
    return config


def _decode_qaic_config(ctx_len: int, num_kv_blocks: int) -> dict:

    return {
        "blocking_mode": "kv_batch_fold",
        "num_kv_blocks": num_kv_blocks,
        "ctx_len": ctx_len,
        "moe_config": {"flavour": "expert_parallel", "tree_reduce": True, "cores_per_expert": 2},
    }


def _prefill_qaic_config(ctx_len: int, num_kv_blocks: int, prefill_seq_len: int) -> dict:

    cfg = _decode_qaic_config(ctx_len, num_kv_blocks)
    cfg["blocking_mode"] = f"prefill_{PREFILL_BLOCKING_MODE}"
    cfg["num_q_blocks"] = -(-prefill_seq_len // PREFILL_QL_CHUNK)  # ceil divide
    cfg["n_rep_chunk"] = PREFILL_N_REP_CHUNK
    cfg["moe_config"] = {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}
    return cfg


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompts=None,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    generation_len: int = DEFAULT_GENERATION_LEN,
    full_batch_size: int = DEFAULT_FULL_BATCH_SIZE,
    skip_vision: bool = False,
    image_urls=None,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
    num_kv_blocks: int = NUM_KV_BLOCKS,
):
    """Run CB (chunked-prefill + batch-folded decode) over ``prompts`` with the DMA KV handoff."""
    prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
    image_urls = list(image_urls) if image_urls else list(DEFAULT_IMAGE_URLS)

    if len(prompts) == 1:
        prompts = prompts * full_batch_size
    if len(image_urls) == 1:
        image_urls = image_urls * full_batch_size
    config = _build_config(model_id)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        continuous_batching=True,
        config=config,
        dtype=torch.float16,
        layerwise=False,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    vision_session = None
    if not skip_vision:
        vision_qpc_path = qeff_model.compile(
            batch_size=full_batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            height=354,
            width=536,
            num_cores=16,
            num_devices=2,
            mos=1,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
            layerwise=False,
        )
        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))

    decode_qpc_path = qeff_model.compile(
        batch_size=full_batch_size,
        full_batch_size=full_batch_size,
        kv_cache_batch_size=full_batch_size,
        prefill_seq_len=1,
        ctx_len=ctx_len,
        height=354,
        width=536,
        num_cores=16,
        num_devices=decode_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        user_tiled=True,
        split_retained_state_io=True,
        split_model_io=True,
        mos=1,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
        qaic_config=_decode_qaic_config(ctx_len, num_kv_blocks),
    )

    prefill_qpc_path = qeff_model.compile(
        batch_size=1,
        full_batch_size=1,
        kv_cache_batch_size=7,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        height=354,
        width=536,
        num_cores=16,
        num_devices=prefill_num_devices,
        mxfp6_matmul=True,
        retain_full_kv=True,
        mxint8_kv_cache=True,
        split_retained_state_io=True,
        mos=1,
        split_model_io=True,
        aic_enable_depth_first=True,
        mdp_num_partitions=stages,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=True,
        qaic_config=_prefill_qaic_config(ctx_len, num_kv_blocks, prefill_seq_len),
    )

    prefill_session = QAICInferenceSession(
        prefill_qpc_path.get("lang_prefill_qpc_path"),
        kv_dma_share=True,
        full_batch_size=full_batch_size,
        cluster_id="prefill",
    )
    decode_session = QAICInferenceSession(
        decode_qpc_path.get("lang_decode_qpc_path"),
        kv_dma_share=True,
        full_batch_size=full_batch_size,
        cluster_id="decode",
    )

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    assert kv_caches and kv_caches[0].shape[0] == full_batch_size, (
        f"decode KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != full_batch_size {full_batch_size}"
    )
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    def _prepare_prompt_batch(prompt: str, image_url: str):
        if skip_vision:
            content = [{"type": "text", "text": prompt}]
        else:
            image = Image.open(requests.get(image_url, stream=True).raw)
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        messages = [[{"role": "user", "content": content}]] * full_batch_size
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs, prefill_seq_len=prefill_seq_len, batch_size=full_batch_size
        )

        pad_token_id = 1
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -prefill_seq_len)
        padded_len = num_chunks * prefill_seq_len

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            pad_token_id,
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.nn.functional.pad(
                inputs["mm_token_type_ids"], (0, padded_len - input_ids_length), "constant", 0
            )

        for k, v in inputs.items():
            inputs[k] = np.array(v)

        vision_inputs = {k: v for k, v in inputs.items() if k in VISION_INPUT_KEYS}
        vision_inputs.update({k: vision_inputs[k].astype("float16") for k in VISION_FP16_KEYS if k in vision_inputs})

        vision_start = perf_counter()
        vision_outputs = {}
        if not skip_vision and vision_inputs:
            raw = vision_session.run(vision_inputs)
            vision_outputs = {k: raw[k] for k in VISION_OUTPUT_KEYS if k in raw}
        vision_end = perf_counter()

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
        if "mm_token_type_ids" in lang_inputs and "mm_token_type_ids" not in prefill_session.input_names:
            lang_inputs.pop("mm_token_type_ids")
        lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

        if not skip_vision:
            for k in VISION_OUTPUT_KEYS:
                if k in vision_outputs:
                    lang_inputs[k] = vision_outputs[k]

        return lang_inputs, vision_outputs, num_chunks, vision_end - vision_start

    prompt = prompts[0]
    prompt_image_url = image_urls[0 % len(image_urls)]
    lang_inputs, vision_outputs, num_chunks, vision_time = _prepare_prompt_batch(prompt, prompt_image_url)

    lang_start = perf_counter()
    all_outputs = []
    chunk_inputs = lang_inputs.copy()
    chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)
    slot_kv_view = [kv[0:1] for kv in kv_caches]
    exec_idx = None
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][0:1, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            :, 0:1, i * prefill_seq_len : (i + 1) * prefill_seq_len
        ]
        if "mm_token_type_ids" in lang_inputs:
            chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
                0:1, i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
        last_chunk = i == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=slot_kv_view if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        prefill_out = prefill_session.get_outputs(index=exec_idx)
        chunk_inputs["image_idx"] = prefill_out["image_idx_output"]
    prefill_time = perf_counter() - lang_start + vision_time
    print(f"Prefill time : {prefill_time:.2f} secs")

    next_token_id = int(np.argmax(prefill_out["logits"]))
    all_outputs.append(next_token_id)
    next_pos = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1

    batch_index = np.random.default_rng(1234).permutation(full_batch_size).reshape(full_batch_size, 1).astype(np.int64)
    physical_slots = batch_index[:, 0]
    for kv_cache in kv_caches:
        source_row = kv_cache[0:1].copy()
        reps = (full_batch_size,) + (1,) * (kv_cache.ndim - 1)
        kv_cache[physical_slots] = np.tile(source_row, reps)

    if not skip_vision and vision_outputs:
        persistent = {}
        for k in VISION_OUTPUT_KEYS:
            if k not in decode_session.binding_index_map:
                continue
            binding = decode_session.bindings[decode_session.binding_index_map[k]]
            dtype = decode_session.aic_to_np_dtype_mapping[binding.type]
            persistent[k] = np.zeros(tuple(binding.dims), dtype=dtype)
        decode_session.set_persistent_inputs(persistent)

    decode_inputs = {
        "input_ids": np.full((full_batch_size, 1), next_token_id, dtype=lang_inputs["input_ids"].dtype),
        "position_ids": next_pos,
        "batch_index": batch_index,
    }
    if decode_has_image_idx:
        decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

    def _run_decode_step(inputs):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        return decode_session.get_outputs(index=exec_idx)

    st = perf_counter()
    decode_out = _run_decode_step(decode_inputs)
    print(f"time for first run of decode with KV as input = {perf_counter() - st} sec\n")

    all_outputs.append(int(np.argmax(decode_out["logits"][0])))
    pos_id = decode_inputs["position_ids"] + 1
    loop_decode_inputs = {
        "input_ids": np.argmax(decode_out["logits"], axis=-1),
        "position_ids": pos_id,
        "batch_index": batch_index,
    }
    if decode_has_image_idx:
        loop_decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

    st = perf_counter()
    for _ in range(generation_len - 2):
        decode_out = _run_decode_step(loop_decode_inputs)
        all_outputs.append(int(np.argmax(decode_out["logits"][0])))
        pos_id += 1
        loop_decode_inputs.update(
            {
                "input_ids": np.argmax(decode_out["logits"], axis=-1),
                "position_ids": pos_id,
                "batch_index": batch_index,
            }
        )
        if decode_has_image_idx:
            loop_decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
    ft = perf_counter()
    print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
    print(f"\noutput\n{tokenizer.decode(all_outputs)}")

    return {"tokens": all_outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--prompt", action="append", dest="prompts", help="prompt (repeatable); defaults to a set of 4")
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument(
        "--full-batch-size",
        type=int,
        default=DEFAULT_FULL_BATCH_SIZE,
        help="CB decode width (N); also the folded decode batch, since batch_fold requires batch_size == N",
    )
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=NUM_KV_BLOCKS,
        help="decode KV-blocking chunk count (batch-folded 'kv' blocking)",
    )
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument("--skip-vision", dest="skip_vision", action="store_true", help="text-only lang path")
    vision.add_argument(
        "--with-vision",
        dest="skip_vision",
        action="store_false",
        help="image+text: compile and run the vision QPC, pairing each prompt with an --image-url (default)",
    )
    parser.set_defaults(skip_vision=False)
    parser.add_argument(
        "--image-url",
        action="append",
        dest="image_urls",
        help="image URL (repeatable); paired with prompts by index. Defaults to a set of 4",
    )
    args = parser.parse_args()

    # With vision the default text prompts are a poor fit; use the image prompts unless the
    # user supplied their own --prompt.
    prompts = args.prompts
    image_urls = args.image_urls
    if not args.skip_vision:
        if not prompts:
            prompts = list(DEFAULT_IMAGE_PROMPTS)
        if not image_urls:
            image_urls = list(DEFAULT_IMAGE_URLS)

    run(
        model_id=args.model_id,
        prompts=prompts,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        full_batch_size=args.full_batch_size,
        skip_vision=args.skip_vision,
        image_urls=image_urls,
        stages=args.stages,
        prefill_num_devices=args.prefill_num_devices,
        decode_num_devices=args.decode_num_devices,
        num_kv_blocks=args.num_kv_blocks,
    )
