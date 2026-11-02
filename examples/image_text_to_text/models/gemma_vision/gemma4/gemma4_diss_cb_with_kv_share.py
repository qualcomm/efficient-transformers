# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Continuous-batching disaggregated prefill/decode for Gemma4 — DMA KV handoff."""

import argparse
from collections import deque
from time import perf_counter

import numpy as np
import torch
import transformers
from gemma4_utils import (
    CHAT_TEMPLATE,
    build_messages,
    remove_fp16clip_transform_if_disabled,
)
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
# DEFAULT_MODEL_ID ='tiny-random/gemma4-moe'
SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROMPTS = [
    "Tell me about Taj Mahal?",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Name three primary colors.",
]
DEFAULT_IMAGE_PROMPTS = [
    "Can you Describe this image in detail?",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "What colors are predominant in the image?",
]
DEFAULT_IMAGE_URLS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
]
DEFAULT_PREFILL_SEQ_LEN = 256
DEFAULT_CTX_LEN = 4096
DEFAULT_GENERATION_LEN = 200
DEFAULT_FULL_BATCH_SIZE = 4

ENABLE_FP16_CLIP = True
STAGES = 4
PREFILL_NUM_DEVICES = 8
DECODE_NUM_DEVICES = 4
BS = 1  # prefill exec (input_ids) batch is always 1 under CB; the KV batch dim is full_batch_size

VISION_INPUT_KEYS = {
    "pixel_values",
    "image_position_ids",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}


def _build_config(model_id: str):
    """Load the model config (gemma4 defaults to float32 weights)."""
    config = AutoConfig.from_pretrained(model_id)

    # For faster execution user can run with fewer layers. For testing purposes only.
    # config.text_config.num_hidden_layers = 2
    # if getattr(config.text_config, "layer_types", None):
    #     config.text_config.layer_types = config.text_config.layer_types[: config.text_config.num_hidden_layers]
    # config.vision_config.num_hidden_layers = 2
    return config


def _resolve_lang_qpc_path(qpc_obj, preferred_keys):
    if isinstance(qpc_obj, dict):
        for key in preferred_keys:
            if key in qpc_obj:
                return qpc_obj[key]
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_obj.keys())}")
    if isinstance(qpc_obj, (list, tuple)):
        # Backward-compat: some codepaths return (vision_qpc, lang_qpc)
        return qpc_obj[1]
    return qpc_obj


def _resolve_vision_qpc_path(qpc_obj, preferred_keys=("vision_qpc_path",)):
    if isinstance(qpc_obj, dict):
        for key in preferred_keys:
            if key in qpc_obj:
                return qpc_obj[key]
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_obj.keys())}")
    if isinstance(qpc_obj, (list, tuple)):
        # Backward-compat: some codepaths return (vision_qpc, lang_qpc)
        return qpc_obj[0]
    return qpc_obj


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompts=None,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    generation_len: int = DEFAULT_GENERATION_LEN,
    full_batch_size: int = DEFAULT_FULL_BATCH_SIZE,
    skip_vision: bool = True,
    image_urls=None,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
):
    """Run CB (chunked-prefill + batched decode) over ``prompts`` with the DMA KV handoff.

    ``skip_vision=True`` (default) runs text-only prompts; ``skip_vision=False`` pairs each
    prompt with the image at the same index of ``image_urls`` (cycled if shorter) and runs it
    as an image+text turn through the vision QPC. Returns a dict with, per prompt, the
    ``first_tokens`` (prefill argmax) and the full decoded ``tokens`` list, for parity
    comparison against the single-request driver.
    """
    prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
    image_urls = list(image_urls) if image_urls else list(DEFAULT_IMAGE_URLS)
    config = _build_config(model_id)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        continuous_batching=True,
        config=config,
        dtype="float32",
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    remove_fp16clip_transform_if_disabled(qeff_model, ENABLE_FP16_CLIP)

    vision_session = None
    if not skip_vision:
        vision_qpc_path = qeff_model.compile(
            batch_size=BS,
            full_batch_size=full_batch_size,  # required by the CB compile guard; vision spec stays batch_size=1
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_cores=16,
            num_devices=2,
            mos=1,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            use_onnx_subfunctions=True,
            skip_lang=True,
        )
        vision_session = QAICInferenceSession(_resolve_vision_qpc_path(vision_qpc_path))

    # full_batch_size pins the KV/RetainedState batch axis to N on BOTH QPCs; enable_chunking
    # is mandatory for a CB prefill-only compile.
    prefill_qpc_path = qeff_model.compile(
        batch_size=BS,
        full_batch_size=full_batch_size,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=16,
        num_devices=prefill_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        node_precision_info=True,
        prefill_only=True,
        enable_chunking=True,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 256}},
        use_onnx_subfunctions=True,
        skip_vision=True,
        mdp_num_partitions=stages,
    )

    decode_qpc_path = qeff_model.compile(
        batch_size=BS,
        full_batch_size=full_batch_size,
        prefill_seq_len=1,
        ctx_len=ctx_len,
        num_cores=16,
        num_devices=decode_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        node_precision_info=True,
        prefill_only=False,
        use_onnx_subfunctions=True,
        skip_vision=True,
    )

    prefill_session = QAICInferenceSession(
        _resolve_lang_qpc_path(prefill_qpc_path, ("lang_prefill_qpc_path", "lang_qpc_path")),
        kv_dma_share=True,
        full_batch_size=full_batch_size,
        cluster_id="prefill",
    )
    decode_session = QAICInferenceSession(
        _resolve_lang_qpc_path(decode_qpc_path, ("lang_decode_qpc_path", "lang_qpc_path")),
        kv_dma_share=True,
        full_batch_size=full_batch_size,
        cluster_id="decode",
    )

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    chat_template = (
        getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE
    )

    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    assert kv_caches and kv_caches[0].shape[0] == full_batch_size, (
        f"decode KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != full_batch_size {full_batch_size}"
    )
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    def _prepare_prompt(prompt: str, image_url: str):
        if skip_vision:
            messages = build_messages(SYSTEM_PROMPT, prompt, use_image=False)
        else:
            messages = build_messages(SYSTEM_PROMPT, prompt, use_image=True)
            messages[-1]["content"][0]["url"] = image_url

        inputs = processor.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        pad_token_id = 1
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
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
        vision_embeds = None
        if not skip_vision and vision_inputs:
            vision_embeds = vision_session.run(vision_inputs)["vision_embeds"]

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(
                lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
            )  # -1 marks invalid positions

        # mm_token_type_ids is a per-chunk lang input; synthesize zeros if the processor
        # omitted it (e.g. text-only) so prefill slicing and the binding are always satisfied.
        if "mm_token_type_ids" not in lang_inputs:
            lang_inputs["mm_token_type_ids"] = np.zeros((BS, padded_len), dtype=np.int64)
        lang_inputs["image_idx"] = np.array([[0]])
        return lang_inputs, vision_embeds, num_chunks

    def _prefill_slot(lang_inputs, vision_embeds, num_chunks, slot: int):
        """Chunked prefill of one prompt into KV ``slot``."""
        chunk_inputs = dict(lang_inputs)
        chunk_inputs["batch_index"] = np.array([[slot]], dtype=np.int64)
        if not skip_vision and vision_embeds is not None:
            chunk_inputs["vision_embeds"] = vision_embeds
        slot_kv_view = [kv[slot : slot + 1] for kv in kv_caches]
        exec_idx = None
        for i in range(num_chunks):
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
                ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            last_chunk = i == num_chunks - 1
            exec_idx = prefill_session.np_run_pipeline(
                chunk_inputs,
                last_chunk=last_chunk,
                kv_cache_buffers=slot_kv_view if last_chunk else None,
            )
            prefill_session.complete_inf(exec_idx, is_prefill=True)
            chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]

        prefill_out = prefill_session.get_outputs(index=exec_idx)
        first_token = int(np.argmax(prefill_out["logits"]))
        next_pos = int(np.max(lang_inputs["position_ids"])) + 1
        return first_token, next_pos, prefill_out.get("image_idx_output", chunk_inputs["image_idx"])

    ongoing = [False] * full_batch_size
    last_token = [0] * full_batch_size
    pos = [0] * full_batch_size
    gen_count = [0] * full_batch_size
    slot_prompt_idx = [-1] * full_batch_size
    slot_tokens = [None] * full_batch_size
    results = [None] * len(prompts)
    decode_image_idx = np.array([[0]], dtype=np.int64)
    decode_vision_embeds = None

    def _seed_slot(slot, prompt_idx, first_token, next_pos):
        slot_prompt_idx[slot] = prompt_idx
        slot_tokens[slot] = [first_token]
        gen_count[slot] = 1
        last_token[slot] = first_token
        pos[slot] = next_pos
        ongoing[slot] = True

    prompt_queue = deque((idx, prompt, image_urls[idx % len(image_urls)]) for idx, prompt in enumerate(prompts))

    prefill_start = perf_counter()
    for slot in range(full_batch_size):
        if not prompt_queue:
            break
        prompt_idx, prompt, prompt_image_url = prompt_queue.popleft()
        lang_inputs, vision_embeds, num_chunks = _prepare_prompt(prompt, prompt_image_url)
        if vision_embeds is not None:
            decode_vision_embeds = vision_embeds
        ft, next_pos, decode_image_idx = _prefill_slot(lang_inputs, vision_embeds, num_chunks, slot)
        _seed_slot(slot, prompt_idx, ft, next_pos)
    print(f"Initial prefill time : {perf_counter() - prefill_start:.2f} secs")

    decode_persist = {"mm_token_type_ids": np.zeros((full_batch_size, 1), dtype=np.int64)}
    if not skip_vision and decode_vision_embeds is not None:
        decode_persist["vision_embeds"] = decode_vision_embeds
    decode_session.set_persistent_inputs(
        {k: v for k, v in decode_persist.items() if k in decode_session.binding_index_map}
    )

    def _build_decode_inputs():
        input_ids = np.full((full_batch_size, 1), -1, dtype=np.int64)
        position_ids = np.full((full_batch_size, 1), -1, dtype=np.int64)
        batch_index = np.full((full_batch_size, 1), -1, dtype=np.int64)
        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            input_ids[slot, 0] = last_token[slot]
            position_ids[slot, 0] = pos[slot]
            batch_index[slot, 0] = slot
        decode_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "batch_index": batch_index,
        }
        if decode_has_image_idx:
            # image_idx is a fixed (1,1) binding, NOT widened to full_batch_size. Keep it
            # stateful because the QPC returns image_idx_output across prefill/decode steps.
            decode_inputs["image_idx"] = decode_image_idx
        return decode_inputs

    st = perf_counter()
    decode_steps = 0
    while any(ongoing):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = _build_decode_inputs()
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        if decode_has_image_idx and "image_idx_output" in out:
            decode_image_idx = out["image_idx_output"]
        decode_steps += 1

        logits = out["logits"]
        logits = logits.reshape(full_batch_size, -1, logits.shape[-1])[:, -1, :]
        next_tokens = np.argmax(logits, axis=-1)

        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            tok = int(next_tokens[slot])
            if tok == tokenizer.eos_token_id or gen_count[slot] >= generation_len:
                # Slot finished: record its output, then refill from the queue or retire.
                results[slot_prompt_idx[slot]] = slot_tokens[slot]
                if prompt_queue:
                    prompt_idx, prompt, prompt_image_url = prompt_queue.popleft()
                    lang_inputs, vision_embeds, num_chunks = _prepare_prompt(prompt, prompt_image_url)
                    if vision_embeds is not None:
                        decode_vision_embeds = vision_embeds
                        decode_session.set_persistent_inputs(
                            {"vision_embeds": vision_embeds}
                            if "vision_embeds" in decode_session.binding_index_map
                            else {}
                        )
                    ft, next_pos, decode_image_idx = _prefill_slot(lang_inputs, vision_embeds, num_chunks, slot)
                    _seed_slot(slot, prompt_idx, ft, next_pos)
                else:
                    ongoing[slot] = False
            else:
                slot_tokens[slot].append(tok)
                gen_count[slot] += 1
                last_token[slot] = tok
                pos[slot] += 1
    ft = perf_counter()

    total_tokens = sum(len(t) for t in results if t)
    print(f"decode steps={decode_steps} tok/sec={total_tokens / (ft - st):.2f}")
    first_tokens = []
    for idx, prompt in enumerate(prompts):
        toks = results[idx] or []
        first_tokens.append(toks[0] if toks else None)
        print(f"\ninput [{idx}]\n{prompt}\noutput\n{tokenizer.decode(toks)}")

    return {"first_tokens": first_tokens, "tokens": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--prompt", action="append", dest="prompts", help="prompt (repeatable); defaults to a set of 4")
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument("--full-batch-size", type=int, default=DEFAULT_FULL_BATCH_SIZE, help="CB decode width (N)")
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
    )
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument("--skip-vision", dest="skip_vision", action="store_true", help="text-only lang path (default)")
    vision.add_argument(
        "--with-vision",
        dest="skip_vision",
        action="store_false",
        help="image+text: compile and run the vision QPC, pairing each prompt with an --image-url",
    )
    parser.set_defaults(skip_vision=True)
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
    )
