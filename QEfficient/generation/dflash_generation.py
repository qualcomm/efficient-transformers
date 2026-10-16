# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
DFlash speculative-decoding (SPD) generation: text-only and VLM (gemma4 / qwen3-vl).

Owns the on-device SPD core (TLM verify + DLM draft, block-wise accept/reject) and
model-specific adapters for text, Gemma4, and Qwen3-VL. Model loading, session setup,
image download, defaults, and output rendering belong to the example front ends under
``examples/performance/dflash``.
"""

import time
from collections.abc import Callable

import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.qwen3_vl.modeling_qwen3_vl import (
    QEffQwen3VLForConditionalGeneration,
)
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


# ===== GEMMA IMAGE/TEXT PROMPT (multimodal TLM) =====
# Mirrors examples/image_text_to_text/models/gemma_vision/gemma4/gemma4_example.py: the
# prompt is built with the gemma processor's chat template.
#
# The gemma4 vision build (SKIP_VISION=False, kv_offload=True) produces TWO QPCs:
#   * a vision-encoder QPC:  pixel_values, image_position_ids -> vision_embeds
#   * a language (DFlash TLM) QPC: input_ids, vision_embeds, position_ids, image_idx,
#     mm_token_type_ids -> logits, image_idx_output, target_hidden_states
# Multimodal callers run the vision QPC once to produce real vision_embeds, then feed those
# (plus mm_token_type_ids marking the image-placeholder span) through the SPD TLM so the
# image tokens attend to real image features. Text-only prompts bind zero vision_embeds.
# ===== METRICS =====


class SpecDecodingMetrics:
    def __init__(self, block_size: int = 10):
        self.block_size = block_size
        self.total_prefill_time = 0.0
        self.tlm_decode_time = 0.0
        self.dlm_decode_time = 0.0
        self.total_accepted_tokens = 0
        self.total_rejected_tokens = 0
        self.total_generated_tokens = 0
        self.num_total_iters = 0
        self.acceptance_history = []
        self.generated_ids: list = []
        self.generated_sources: list = []

    def acceptance_rate(self) -> float:
        """Return the average number of DLM draft tokens accepted per SPD iteration."""
        if self.num_total_iters == 0:
            return 0.0
        return self.total_accepted_tokens / self.num_total_iters

    def dlm_tok_rate(self) -> float:
        if self.dlm_decode_time <= 0:
            return 0.0
        return (self.block_size * self.num_total_iters) / self.dlm_decode_time

    def tlm_tok_rate(self) -> float:
        if self.tlm_decode_time <= 0:
            return 0.0
        ar = self.acceptance_rate()
        num_tok_tlm = self.total_generated_tokens / (1 + ar) if (1 + ar) > 0 else 0.0
        return num_tok_tlm / self.tlm_decode_time

    def spd_tok_rate(self) -> float:
        total_decode_s = self.tlm_decode_time + self.dlm_decode_time
        if total_decode_s <= 0:
            return 0.0
        return self.total_generated_tokens / total_decode_s


def _initialize_generated_ids(batch_size: int, ctx_len: int, padded_len: int, pad_token_id: int) -> np.ndarray:
    if ctx_len < padded_len:
        raise ValueError(f"ctx_len ({ctx_len}) must be greater than or equal to padded_len ({padded_len}).")
    return np.full((batch_size, ctx_len - padded_len), pad_token_id)


def _prepare_text_inputs(tokenizer, prompt_text: str, prompt_chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    prompt = [prompt_text]
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = -(inputs["input_ids"].shape[1] // -prompt_chunk_size) * prompt_chunk_size
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    position_ids = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs.pop("past_key_values", None)
    return inputs["input_ids"], position_ids


def _run_spd_core(
    *,
    input_ids: np.ndarray,
    flat_position_ids: np.ndarray,
    tlm_prefill: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    prepare_decode: Callable[[], None],
    tlm_decode: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    dlm_session: QAICInferenceSession,
    mask_token_id: int,
    vocab_size: int,
    prompt_chunk_size: int,
    block_size: int,
    hidden_size: int,
    generation_len: int,
    max_iterations: int,
    eos_token_ids: set[int],
    generated_ids: np.ndarray,
    metrics: SpecDecodingMetrics,
    prefill_start_time: float | None = None,
) -> SpecDecodingMetrics:
    """Run model-independent DFlash prefill, verification, and cache advancement.

    Model adapters normalize their TLM calls to token IDs and target hidden states.
    The core owns chunking, DLM prefill, speculative acceptance, EOS/length handling,
    and metrics; model-specific bindings and position-ID formats remain outside.
    """
    batch_size = input_ids.shape[0]
    padded_len = input_ids.shape[1]
    if flat_position_ids.shape != input_ids.shape:
        raise ValueError(
            f"flat_position_ids shape {flat_position_ids.shape} must match input_ids shape {input_ids.shape}."
        )
    if padded_len == 0 or padded_len % prompt_chunk_size != 0:
        raise ValueError(
            f"Padded prompt length ({padded_len}) must be a positive multiple of prompt_chunk_size "
            f"({prompt_chunk_size})."
        )

    dlm_session.set_buffers({"logits": np.zeros((batch_size, block_size, vocab_size), dtype=np.float32)})
    dlm_inputs = {}
    input_ids_block = np.full((batch_size, block_size), mask_token_id, dtype=np.int64)
    prefill_start = time.time() if prefill_start_time is None else prefill_start_time
    num_chunks = padded_len // prompt_chunk_size

    def run_dlm_prefill_block(hidden_states, position_ids, sub_start, sub_len):
        target_hidden = np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)
        position_ids_target = np.full((batch_size, block_size), -1, dtype=position_ids.dtype)
        target_hidden[:, :sub_len, :] = hidden_states[:, sub_start : sub_start + sub_len, :]
        position_ids_target[:, :sub_len] = position_ids[:, sub_start : sub_start + sub_len]
        dlm_inputs.update(
            {
                "input_ids": np.full((batch_size, block_size), mask_token_id, dtype=np.int64),
                "position_ids": position_ids_target + block_size,
                "position_ids_target": position_ids_target,
                "target_hidden": target_hidden,
            }
        )
        dlm_session.run(dlm_inputs)

    for chunk_index in range(num_chunks):
        chunk_start = chunk_index * prompt_chunk_size
        chunk_end = chunk_start + prompt_chunk_size
        tlm_logits, target_hidden = tlm_prefill(chunk_start, chunk_end)
        flat_pos_chunk = flat_position_ids[:, chunk_start:chunk_end]

        if chunk_index < num_chunks - 1:
            for sub_start in range(0, prompt_chunk_size, block_size):
                sub_len = min(block_size, prompt_chunk_size - sub_start)
                run_dlm_prefill_block(target_hidden, flat_pos_chunk, sub_start, sub_len)
            continue

        last_prefill_pos_in_chunk = int(flat_pos_chunk.argmax())
        new_tlm_token = tlm_logits[:, last_prefill_pos_in_chunk]
        last_sub_start = (last_prefill_pos_in_chunk // block_size) * block_size

        for sub_start in range(0, last_sub_start, block_size):
            run_dlm_prefill_block(target_hidden, flat_pos_chunk, sub_start, block_size)

        sub_len = min(block_size, prompt_chunk_size - last_sub_start)
        final_hidden = np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)
        final_position_ids = np.full((batch_size, block_size), -1, dtype=flat_pos_chunk.dtype)
        final_hidden[:, :sub_len, :] = target_hidden[:, last_sub_start : last_sub_start + sub_len, :]
        final_position_ids[:, :sub_len] = flat_pos_chunk[:, last_sub_start : last_sub_start + sub_len]

        input_ids_block[:, 0] = new_tlm_token
        spd_counter_idx = chunk_start + last_prefill_pos_in_chunk
        dlm_inputs.update(
            {
                "input_ids": input_ids_block,
                "position_ids": np.arange(spd_counter_idx + 1, spd_counter_idx + block_size + 1).reshape(1, -1),
                "position_ids_target": final_position_ids,
                "target_hidden": final_hidden,
            }
        )
        dlm_outputs = dlm_session.run(dlm_inputs)

    metrics.total_prefill_time += time.time() - prefill_start
    dlm_candidates = dlm_outputs["logits"].argmax(axis=-1)
    prepare_decode()

    generation_limit = min(generation_len, generated_ids.shape[1])
    gen_idx = 0
    iteration_count = 0
    continue_generation = True

    while gen_idx < generation_limit and iteration_count < max_iterations and continue_generation:
        iteration_count += 1
        iteration_generated_start = len(metrics.generated_ids)
        dlm_candidates[:, 0] = new_tlm_token

        tlm_decode_start = time.time()
        tlm_logits, target_hidden = tlm_decode(dlm_candidates, dlm_inputs["position_ids"])
        metrics.tlm_decode_time += time.time() - tlm_decode_start

        accepted_length = 0
        rejected_flag = False
        for spec_idx in range(min(block_size - 1, generation_limit - gen_idx)):
            tlm_token = tlm_logits[:, spec_idx]
            dlm_token = dlm_candidates[:, spec_idx + 1]
            if tlm_token == dlm_token:
                accepted_length += 1
                metrics.total_accepted_tokens += 1
                generated_ids[0, gen_idx] = dlm_token[0]
                gen_idx += 1
                metrics.generated_ids.append(int(dlm_token[0]))
                metrics.generated_sources.append("dlm")
            else:
                metrics.total_rejected_tokens += block_size - spec_idx - 1
                rejected_flag = True
                new_tlm_token = tlm_token
                generated_ids[0, gen_idx] = tlm_token[0]
                gen_idx += 1
                metrics.generated_ids.append(int(tlm_token[0]))
                metrics.generated_sources.append("tlm")
                break

        metrics.acceptance_history.append(accepted_length)
        if not rejected_flag and gen_idx < generation_limit:
            new_tlm_token = tlm_logits[:, block_size - 1]
            generated_ids[0, gen_idx] = new_tlm_token[0]
            gen_idx += 1
            metrics.generated_ids.append(int(new_tlm_token[0]))
            metrics.generated_sources.append("tlm")

        this_iter_gen_ids = metrics.generated_ids[iteration_generated_start:]
        metrics.total_generated_tokens += len(this_iter_gen_ids)
        if any(tok_id in eos_token_ids for tok_id in this_iter_gen_ids):
            continue_generation = False

        if not continue_generation or gen_idx >= generation_limit:
            break

        dlm_decode_start = time.time()
        dlm_inputs["position_ids_target"] = np.arange(spd_counter_idx + 1, spd_counter_idx + block_size + 1).reshape(
            1, -1
        )
        spd_counter_idx += accepted_length + 1
        dlm_inputs["position_ids_target"][:, accepted_length + 1 :] = -1
        dlm_inputs["position_ids"] = np.arange(spd_counter_idx + 1, spd_counter_idx + block_size + 1).reshape(1, -1)
        input_ids_block[:, 0] = new_tlm_token
        dlm_inputs["input_ids"] = input_ids_block
        dlm_inputs["target_hidden"] = target_hidden
        dlm_outputs = dlm_session.run(dlm_inputs)
        metrics.dlm_decode_time += time.time() - dlm_decode_start
        dlm_candidates = dlm_outputs["logits"].argmax(axis=-1)

    metrics.num_total_iters = iteration_count
    return metrics


# ===== TEXT-ONLY SPD INFERENCE =====


def run_spd_inference_single(
    prompt_text: str,
    tokenizer,
    dlm_session: QAICInferenceSession,
    tlm_session: QAICInferenceSession,
    mask_token_id: int,
    vocab_size: int,
    prompt_chunk_size: int,
    ctx_len: int = 4096,
    block_size: int = 16,
    max_iterations: int = 300,
    hidden_size: int = 4096,
    generation_len: int = 256,
    batch_size: int = 1,
) -> SpecDecodingMetrics:
    input_ids, position_ids = _prepare_text_inputs(tokenizer, prompt_text, prompt_chunk_size)
    generated_ids = _initialize_generated_ids(batch_size, ctx_len, input_ids.shape[1], tokenizer.pad_token_id)
    metrics = SpecDecodingMetrics(block_size=block_size)

    tlm_session.set_buffers({"logits": np.zeros((batch_size, prompt_chunk_size), dtype=np.int32)})
    tlm_session.set_buffers({"hidden_states": np.zeros((batch_size, prompt_chunk_size, hidden_size), dtype=np.float32)})

    def tlm_prefill(chunk_start, chunk_end):
        outputs = tlm_session.run(
            {
                "input_ids": input_ids[:, chunk_start:chunk_end],
                "position_ids": position_ids[:, chunk_start:chunk_end],
            }
        )
        return outputs["logits"], outputs["hidden_states"]

    def prepare_decode():
        tlm_session.set_buffers({"logits": np.zeros((batch_size, block_size), dtype=np.int32)})
        tlm_session.set_buffers({"hidden_states": np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)})

    def tlm_decode(draft_ids, decode_position_ids):
        outputs = tlm_session.run({"input_ids": draft_ids, "position_ids": decode_position_ids})
        return outputs["logits"], outputs["hidden_states"]

    eos_token_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    return _run_spd_core(
        input_ids=input_ids,
        flat_position_ids=position_ids,
        tlm_prefill=tlm_prefill,
        prepare_decode=prepare_decode,
        tlm_decode=tlm_decode,
        dlm_session=dlm_session,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        block_size=block_size,
        hidden_size=hidden_size,
        generation_len=generation_len,
        max_iterations=max_iterations,
        eos_token_ids=eos_token_ids,
        generated_ids=generated_ids,
        metrics=metrics,
    )


# ===== VISION SPD INFERENCE — GEMMA4 =====


def run_spd_inference_gemma4(
    prompt_text: str,
    tokenizer,
    dlm_session: QAICInferenceSession,
    tlm_session: QAICInferenceSession,
    mask_token_id: int,
    vocab_size: int,
    prompt_chunk_size: int,
    ctx_len: int = 4096,
    block_size: int = 16,
    max_iterations: int = 300,
    hidden_size: int = 4096,
    generation_len: int = 256,
    input_ids: np.ndarray | None = None,
    mm_token_type_ids: np.ndarray | None = None,
    vision_embeds: np.ndarray | None = None,
    batch_size: int = 1,
) -> SpecDecodingMetrics:
    mm_full = None
    if input_ids is None:
        padded_ids, position_ids = _prepare_text_inputs(tokenizer, prompt_text, prompt_chunk_size)
    else:
        raw_ids = np.asarray(input_ids, dtype=np.int64).reshape(batch_size, -1)
        unpadded_len = raw_ids.shape[1]
        padded_len = -(unpadded_len // -prompt_chunk_size) * prompt_chunk_size
        padded_ids = np.full((batch_size, padded_len), tokenizer.pad_token_id, dtype=np.int64)
        padded_ids[:, :unpadded_len] = raw_ids
        attention_mask = np.zeros((batch_size, padded_len), dtype=np.int64)
        attention_mask[:, :unpadded_len] = 1
        position_ids = np.where(attention_mask, np.arange(padded_len), -1)
        if mm_token_type_ids is not None:
            raw_mm = np.asarray(mm_token_type_ids, dtype=np.int64).reshape(batch_size, -1)
            mm_full = np.zeros((batch_size, padded_len), dtype=np.int64)
            copy_len = min(unpadded_len, raw_mm.shape[1])
            mm_full[:, :copy_len] = raw_mm[:, :copy_len]

    generated_ids = _initialize_generated_ids(batch_size, ctx_len, padded_ids.shape[1], tokenizer.pad_token_id)
    metrics = SpecDecodingMetrics(block_size=block_size)
    tlm_hidden_name = "target_hidden_states" if "target_hidden_states" in tlm_session.output_names else "hidden_states"

    tlm_inputs = set(tlm_session.input_names)
    tlm_needs_vision = "vision_embeds" in tlm_inputs
    tlm_needs_image_idx = "image_idx" in tlm_inputs
    tlm_needs_mm_ids = "mm_token_type_ids" in tlm_inputs
    logits_index = tlm_session.binding_index_map["logits"]
    logits_ranks = {len(shape[logits_index][1]) for shape in tlm_session.allowed_shapes}
    tlm_logits_is_full = max(logits_ranks) == 3 if logits_ranks else False
    logger.info(
        f"TLM contract: vision_embeds={tlm_needs_vision} image_idx={tlm_needs_image_idx} "
        f"mm_token_type_ids={tlm_needs_mm_ids} full_logits={tlm_logits_is_full}"
    )

    if tlm_needs_vision:
        vision_binding = tlm_session.bindings[tlm_session.binding_index_map["vision_embeds"]]
        vision_shape = tuple(int(dim) for dim in vision_binding.dims)
        vision_dtype = tlm_session.aic_to_np_dtype_mapping.get(vision_binding.type, np.dtype(np.float32))
        static_vision_embeds = (
            np.asarray(vision_embeds, dtype=vision_dtype)
            if vision_embeds is not None
            else np.zeros(vision_shape, dtype=vision_dtype)
        )
        tlm_session.set_buffers({"vision_embeds": static_vision_embeds})

    tlm_image_idx = np.array([[0]], dtype=np.int64)

    def tlm_run(ids, model_position_ids, mm=None):
        nonlocal tlm_image_idx
        feeds = {"input_ids": ids, "position_ids": model_position_ids}
        if tlm_needs_image_idx:
            feeds["image_idx"] = tlm_image_idx
        if tlm_needs_mm_ids:
            seq_len = ids.shape[1]
            mm_ids = np.zeros((ids.shape[0], seq_len), dtype=ids.dtype)
            if mm is not None:
                mm = np.asarray(mm, dtype=ids.dtype).reshape(ids.shape[0], -1)
                copy_len = min(seq_len, mm.shape[1])
                mm_ids[:, :copy_len] = mm[:, :copy_len]
            feeds["mm_token_type_ids"] = mm_ids
        outputs = tlm_session.run(feeds)
        if tlm_needs_image_idx and "image_idx_output" in outputs:
            tlm_image_idx = outputs["image_idx_output"]
        logits = outputs["logits"]
        token_ids = logits.argmax(axis=-1).astype(np.int64) if tlm_logits_is_full else logits
        return token_ids, outputs[tlm_hidden_name]

    if tlm_logits_is_full:
        tlm_session.set_buffers({"logits": np.zeros((batch_size, prompt_chunk_size, vocab_size), dtype=np.float32)})
    else:
        tlm_session.set_buffers({"logits": np.zeros((batch_size, prompt_chunk_size), dtype=np.int32)})
    tlm_session.set_buffers({tlm_hidden_name: np.zeros((batch_size, prompt_chunk_size, hidden_size), dtype=np.float32)})

    def tlm_prefill(chunk_start, chunk_end):
        mm_chunk = None if mm_full is None else mm_full[:, chunk_start:chunk_end]
        return tlm_run(
            padded_ids[:, chunk_start:chunk_end],
            position_ids[:, chunk_start:chunk_end],
            mm=mm_chunk,
        )

    def prepare_decode():
        if tlm_logits_is_full:
            tlm_session.set_buffers({"logits": np.zeros((batch_size, block_size, vocab_size), dtype=np.float32)})
        else:
            tlm_session.set_buffers({"logits": np.zeros((batch_size, block_size), dtype=np.int32)})
        tlm_session.set_buffers({tlm_hidden_name: np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)})
        if tlm_needs_vision:
            tlm_session.skip_buffers(["vision_embeds"])

    def tlm_decode(draft_ids, decode_position_ids):
        return tlm_run(draft_ids, decode_position_ids)

    eos_token_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    return _run_spd_core(
        input_ids=padded_ids,
        flat_position_ids=position_ids,
        tlm_prefill=tlm_prefill,
        prepare_decode=prepare_decode,
        tlm_decode=tlm_decode,
        dlm_session=dlm_session,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        block_size=block_size,
        hidden_size=hidden_size,
        generation_len=generation_len,
        max_iterations=max_iterations,
        eos_token_ids=eos_token_ids,
        generated_ids=generated_ids,
        metrics=metrics,
    )


def build_inputs_gemma4(processor, tokenizer, user_prompt, image=None, system_prompt=None):
    """Build processor inputs for a gemma text or image+text prompt.

    Returns (input_ids [1, L] int64, mm_token_type_ids [1, L] int64 or None,
    processor_inputs dict). With image=None this is the text-only
    path (no pixel_values / image tokens), matching gemma4_example.py at input image = false.

    Image downloads are intentionally left to callers such as the example front end.
    """
    chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].detach().cpu().numpy()
    mm_ids = None
    if "mm_token_type_ids" in inputs:
        mm_ids = inputs["mm_token_type_ids"].detach().cpu().numpy().astype(np.int64)
    return input_ids, mm_ids, inputs


def build_input_ids_gemma4(processor, tokenizer, user_prompt, system_prompt=None):
    """Back-compat shim: return only input_ids (text-only callers)."""
    input_ids, _, _ = build_inputs_gemma4(processor, tokenizer, user_prompt, system_prompt=system_prompt)
    return input_ids


def run_vision_encoder_gemma4(vision_session, processor_inputs):
    """Run the gemma4 vision-encoder QPC to produce vision_embeds.

    Feeds pixel_values (fp16) + image_position_ids exactly as kv_offload_generate does
    (see QEfficient/transformers/models/modeling_auto.py and embedding_handler.py).
    Returns the vision_embeds numpy array, or None if there are no pixel_values.
    """
    if "pixel_values" not in processor_inputs:
        return None

    vision_feeds = {}
    for key in ("pixel_values", "image_position_ids"):
        if key in processor_inputs:
            vision_feeds[key] = np.asarray(processor_inputs[key])
    # pixel_values feeds the vision encoder in fp16 (matches the compiled custom-io dtype).
    if "pixel_values" in vision_feeds:
        vision_feeds["pixel_values"] = vision_feeds["pixel_values"].astype("float16")

    vision_session.skip_buffers(
        {
            x
            for x in vision_session.input_names + vision_session.output_names
            if x.startswith("past_") or x.endswith("_RetainedState")
        }
    )
    vision_outputs = vision_session.run(vision_feeds)
    return vision_outputs["vision_embeds"]


# ===== VISION SPD INFERENCE — QWEN3-VL =====
# qwen3-vl's compiled DFlash TLM QPC always requires vision_embeds/deepstack_features/
# image_idx as inputs (unlike gemma4, which has separate text-only and multimodal TLM
# graphs). Its adapter owns the distinct M-RoPE position IDs and vision I/O contract,
# while the common prefill, acceptance, stopping, and metrics flow lives in _run_spd_core().
_rope_model_cache_qwen3_vl = {}


def _get_rope_model_qwen3_vl(config):
    """QEffQwen3VLForConditionalGeneration, built once per config on the meta device.
    get_rope_index (invoked via prepare_inputs_for_generation below) is a pure function
    of input_ids/config/grid_thw and never touches the weights, so a meta-device model
    is sufficient and avoids paying full-size construction cost on every call.
    """
    key = id(config)
    if key not in _rope_model_cache_qwen3_vl:
        with torch.device("meta"):
            _rope_model_cache_qwen3_vl[key] = QEffQwen3VLForConditionalGeneration._from_config(config)
    return _rope_model_cache_qwen3_vl[key]


def compute_position_ids_qwen3_vl(input_ids, attention_mask, image_grid_thw, config):
    """Delegates to QEffQwen3VLForConditionalGeneration.prepare_inputs_for_generation
    for M-RoPE construction. With image_grid_thw=None (text-only) this reduces to a
    plain arange position broadcast to all 4 rows with rope_deltas=0.
    Returns (position_ids [4, batch, seq_len] torch tensor, rope_deltas [batch, 1] torch tensor).
    """
    rope_model = _get_rope_model_qwen3_vl(config)
    batch_size, seq_len = input_ids.shape
    inputs = rope_model.prepare_inputs_for_generation(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
        },
        prefill_seq_len=seq_len,
        batch_size=batch_size,
    )
    return inputs["position_ids"], rope_model.model.rope_deltas


def build_decode_position_ids_qwen3_vl(flat_row: np.ndarray, rope_delta: int) -> np.ndarray:
    """flat_row: (batch, seq_len) plain causal positions for a decode block.
    Returns the (4, batch, seq_len) M-RoPE tensor to feed the lang QPC during decode:
    row 0 stays the plain causal position (used for KV-cache indexing), rows 1-3 all
    carry the same flat_row + rope_delta (post-image text has no spatial variation left,
    per qwen3-vl's M-RoPE convention).
    """
    mrope_row = flat_row + rope_delta
    return np.concatenate([flat_row[None], np.repeat(mrope_row[None], 3, axis=0)], axis=0)


def build_inputs_qwen3_vl(processor, user_prompt, image=None):
    """Build processor inputs for a qwen3-vl text or image+text prompt.

    Returns (processor_inputs dict). With image=None this is the text-only path (no
    pixel_values / image_grid_thw keys).
    """
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": user_prompt})
    messages = [{"role": "user", "content": content}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages) if image is not None else (None, None)
    proc_inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    proc_inputs.pop("token_type_ids", None)
    return proc_inputs


def run_vision_encoder_qwen3_vl(vision_session, pixel_values, image_grid_thw):
    """Run the qwen3-vl vision-encoder QPC to produce (vision_embeds, deepstack_features)."""
    vision_outputs = vision_session.run(
        {
            "pixel_values": pixel_values.astype(np.float16),
            "image_grid_thw": image_grid_thw,
        }
    )
    return vision_outputs["vision_embeds"], vision_outputs["deepstack_features"]


def run_spd_inference_qwen3_vl(
    prompt_text: str,
    tokenizer,
    processor,
    tlm_config,
    dlm_session: QAICInferenceSession,
    tlm_session: QAICInferenceSession,
    mask_token_id: int,
    vocab_size: int,
    prompt_chunk_size: int,
    ctx_len: int = 4096,
    block_size: int = 16,
    max_iterations: int = 300,
    hidden_size: int = 5120,
    generation_len: int = 256,
    image=None,
    vision_session: QAICInferenceSession | None = None,
    compiled_height: int = 354,
    compiled_width: int = 536,
    batch_size: int = 1,
) -> SpecDecodingMetrics:
    """Run Qwen3-VL DFlash inference for a text-only or single-image prompt."""
    metrics = SpecDecodingMetrics(block_size=block_size)
    image_processing_start = time.time()

    if image is not None:
        image = image.resize((compiled_width, compiled_height))
    processor_inputs = build_inputs_qwen3_vl(processor, prompt_text, image=image)
    input_ids_length = processor_inputs["input_ids"].shape[1]
    padded_len = -(input_ids_length // -prompt_chunk_size) * prompt_chunk_size

    image_grid_thw = processor_inputs.get("image_grid_thw")
    position_ids, rope_deltas = compute_position_ids_qwen3_vl(
        processor_inputs["input_ids"],
        processor_inputs["attention_mask"],
        image_grid_thw,
        tlm_config,
    )
    position_ids = torch.nn.functional.pad(position_ids, (0, padded_len - input_ids_length), value=-1)
    input_ids = torch.nn.functional.pad(
        processor_inputs["input_ids"],
        (0, padded_len - input_ids_length),
        value=tokenizer.pad_token_id,
    )
    input_ids_np = input_ids.numpy()
    position_ids_np = position_ids.numpy()
    flat_position_ids = position_ids_np[0]
    rope_delta = int(rope_deltas.reshape(-1)[0])
    generated_ids = _initialize_generated_ids(batch_size, ctx_len, padded_len, tokenizer.pad_token_id)
    metrics.image_processing_time = time.time() - image_processing_start

    prefill_start = time.time()
    vision_binding = tlm_session.bindings[tlm_session.binding_index_map["vision_embeds"]]
    vision_dtype = tlm_session.aic_to_np_dtype_mapping.get(vision_binding.type, np.dtype(np.float32))
    deepstack_binding = tlm_session.bindings[tlm_session.binding_index_map["deepstack_features"]]
    deepstack_dtype = tlm_session.aic_to_np_dtype_mapping.get(deepstack_binding.type, np.dtype(np.float32))

    if image is not None:
        if vision_session is None:
            raise ValueError("An image was given but no vision_session was provided.")
        vision_embeds, deepstack_features = run_vision_encoder_qwen3_vl(
            vision_session,
            processor_inputs["pixel_values"].numpy(),
            image_grid_thw.numpy(),
        )
        metrics.vision_prefill_time = time.time() - prefill_start
    else:
        vision_embeds = np.zeros(tuple(int(dim) for dim in vision_binding.dims), dtype=vision_dtype)
        deepstack_features = np.zeros(tuple(int(dim) for dim in deepstack_binding.dims), dtype=deepstack_dtype)
        metrics.vision_prefill_time = 0.0

    vision_outputs = {
        "vision_embeds": vision_embeds.astype(vision_dtype),
        "deepstack_features": deepstack_features.astype(deepstack_dtype),
    }
    tlm_session.set_buffers(vision_outputs)
    lang_extra = {"image_idx": np.array([[0]], dtype=np.int64)}
    tlm_session.set_buffers({"logits": np.zeros((batch_size, prompt_chunk_size), dtype=np.int32)})
    tlm_session.set_buffers({"hidden_states": np.zeros((batch_size, prompt_chunk_size, hidden_size), dtype=np.float32)})

    def tlm_prefill(chunk_start, chunk_end):
        outputs = tlm_session.run(
            {
                "input_ids": input_ids_np[:, chunk_start:chunk_end],
                "position_ids": position_ids_np[:, :, chunk_start:chunk_end],
                **lang_extra,
            }
        )
        lang_extra["image_idx"] = outputs["image_idx_output"]
        return outputs["logits"], outputs["hidden_states"]

    def prepare_decode():
        tlm_session.set_buffers({"logits": np.zeros((batch_size, block_size), dtype=np.int32)})
        tlm_session.set_buffers({"hidden_states": np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)})
        tlm_session.skip_buffers(vision_outputs.keys())

    def tlm_decode(draft_ids, decode_position_ids):
        outputs = tlm_session.run(
            {
                "input_ids": draft_ids,
                "position_ids": build_decode_position_ids_qwen3_vl(decode_position_ids, rope_delta),
                **lang_extra,
            }
        )
        return outputs["logits"], outputs["hidden_states"]

    eos_token_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    return _run_spd_core(
        input_ids=input_ids_np,
        flat_position_ids=flat_position_ids,
        tlm_prefill=tlm_prefill,
        prepare_decode=prepare_decode,
        tlm_decode=tlm_decode,
        dlm_session=dlm_session,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        block_size=block_size,
        hidden_size=hidden_size,
        generation_len=generation_len,
        max_iterations=max_iterations,
        eos_token_ids=eos_token_ids,
        generated_ids=generated_ids,
        metrics=metrics,
        prefill_start_time=prefill_start,
    )
