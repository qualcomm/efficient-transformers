# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
import re
import time
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
from typing import Callable, Dict, List, Optional

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.generation.cloud_infer import QAICInferenceSession

# from QEfficient.transformers.diffusion_gemma_utils import DiffusionGemmaRuntimeResult
from QEfficient.transformers.models.modeling_auto import QEffCausalLMForTextImageToTextModel

_DEVICE_ENTROPY_BOUND = 0.1


@dataclass
class DiffusionGemmaRuntimeResult:
    generated_ids: np.ndarray
    tokens_per_forward: np.ndarray
    decode_forward_passes: np.ndarray
    total_time: float


@dataclass
class DiffusionGemmaSingleQPCRuntimeResult(DiffusionGemmaRuntimeResult):
    ttft: float
    retained_kv_buffers: int
    total_steps: int
    executed_blocks: int
    total_canvas_time: float
    canvas_length: int


@dataclass
class DiffusionGemmaPrefillRuntimeResult:
    kv_caches: List[np.ndarray]
    encoder_outputs: Dict[str, np.ndarray]
    prompt_length: int
    next_position: int
    total_time: float


def _to_numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _session_feed(session: QAICInferenceSession, feed: Dict[str, np.ndarray]):
    input_names = set(session.input_names)
    return {name: value for name, value in feed.items() if name in input_names}


def _normalize_eos_token_ids(eos_token_id):
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    return [int(token_id) for token_id in eos_token_id]


def _infer_context_length(session: QAICInferenceSession) -> Optional[int]:
    context_lengths = []
    for binding in session.bindings:
        if binding.name.startswith("past_") and len(binding.dims) > 2:
            context_lengths.append(int(binding.dims[2]))
    return max(context_lengths) if context_lengths else None


def _binding_name(session: QAICInferenceSession, name: str) -> Optional[str]:
    if name in session.binding_index_map:
        return name
    return next((candidate for candidate in session.input_names if candidate.rsplit("/", 1)[-1] == name), None)


def _validate_prefill_stages(prefill_stages: int) -> int:
    if not isinstance(prefill_stages, int) or isinstance(prefill_stages, bool) or prefill_stages < 1:
        raise ValueError("`prefill_stages` must be an integer greater than or equal to 1.")
    return prefill_stages


class UnifiedQPC(QEffCausalLMForTextImageToTextModel):
    def __init__(self, model):
        QEFFBaseModel.__init__(self, model)
        self.model = model.get_qeff_unified_wrapper()
        self.model.qaic_config = None
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.continuous_batching = False

    @property
    def get_model_config(self):
        return self.model.model.config.__dict__

    def export(self, inputs, output_names, dynamic_axes, qaic_config, **kwargs):
        return self._export(inputs, output_names=output_names, dynamic_axes=dynamic_axes)


class DiffusionGemmaEncoderQPC(UnifiedQPC):
    def __init__(self, model):
        QEFFBaseModel.__init__(self, model)
        self.model = model.get_qeff_diffusion_encoder()
        self.model.qaic_config = None
        self.num_layers = self.model.text_config.num_hidden_layers
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.continuous_batching = False


class DiffusionGemmaDecoderQPC(UnifiedQPC):
    def __init__(self, model):
        QEFFBaseModel.__init__(self, model)
        self.model = model.get_qeff_diffusion_decoder()
        self.model.qaic_config = None
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.continuous_batching = False


class DiffusionGemmaSingleQPCGenerator:
    """Host driver for the shared-path unified DiffusionGemma QPC."""

    _MASK_VALUE = -1e4

    def __init__(self, *, model_config, session: QAICInferenceSession, seed: int = 1234):
        self.model_config = model_config
        self.session = session
        self.rng = np.random.RandomState(seed) if seed is not None and seed >= 0 else np.random.RandomState()

        input_dims = self._binding_dims("input_ids")
        if input_dims is None or int(input_dims[0]) != 1:
            raise ValueError("The shared DiffusionGemma QPC requires a batch-1 `input_ids` binding.")
        self.prefill_seq_len = int(input_dims[1])
        self.canvas_length = self.prefill_seq_len
        self.vocab_size = int(model_config.text_config.vocab_size)
        self.full_cache_length = self._cache_length("full_attention")
        self.sliding_cache_length = self._cache_length("sliding_attention")
        self.position_ids = None
        self.input_ids = None
        self.mm_token_type_ids = None
        self.vision_embeds = None
        self.image_idx = None
        self.pad_token_id = 0
        self._prompt_chunks = []
        self._prompt_pad_token_id = 0
        self._retained_last_position = -1
        self._full_slot_positions = np.full(self.full_cache_length, -1, dtype=np.int64)
        self._sliding_slot_positions = np.full(self.sliding_cache_length, -1, dtype=np.int64)
        self._active_canvas_position_ids = None

    def _binding_dims(self, name):
        for binding in self.session.bindings:
            if binding.name == name:
                return binding.dims
        return None

    def _cache_length(self, layer_type: str) -> int:
        mask_name = f"{layer_type}_mask"
        dims = self._binding_dims(mask_name)
        if dims is None:
            raise ValueError(f"The QPC is missing `{mask_name}`.")
        cache_length = int(dims[-1]) - self.prefill_seq_len
        if cache_length <= 0:
            raise ValueError(f"Invalid `{mask_name}` shape: {list(dims)}.")
        return cache_length

    @staticmethod
    def _project_slot_positions(slot_positions, cache_position_ids, *, sliding: bool):
        projected_slots = slot_positions.copy()
        write_positions = np.asarray(cache_position_ids, dtype=np.int64).reshape(-1)
        valid_positions = write_positions[write_positions >= 0]
        if valid_positions.size == 0:
            return projected_slots

        cache_length = len(projected_slots)
        rollover = sliding and int(valid_positions.max()) >= (cache_length - 1) * 2
        for logical_position in valid_positions:
            if sliding:
                physical_index = (
                    (int(logical_position) + 1) % cache_length if rollover else int(logical_position) % cache_length
                )
            elif logical_position < cache_length:
                physical_index = int(logical_position)
            else:
                continue
            projected_slots[physical_index] = logical_position
        return projected_slots

    def _projected_slot_positions(self, cache_position_ids, *, sliding: bool):
        slot_positions = self._sliding_slot_positions if sliding else self._full_slot_positions
        return self._project_slot_positions(slot_positions, cache_position_ids, sliding=sliding)

    def _record_cache_write(self, cache_position_ids):
        self._full_slot_positions = self._projected_slot_positions(cache_position_ids, sliding=False)
        self._sliding_slot_positions = self._projected_slot_positions(cache_position_ids, sliding=True)

    def _reset_cache_slot_tracking(self):
        self._full_slot_positions.fill(-1)
        self._sliding_slot_positions.fill(-1)
        self._retained_last_position = -1
        self._active_canvas_position_ids = None

    def _emit_debug(self, debug_callback, event):
        if debug_callback is None:
            return
        debug_callback(
            {
                **event,
                "retained_last_position": self._retained_last_position,
                "full_slot_positions": self._full_slot_positions.copy(),
                "sliding_slot_positions": self._sliding_slot_positions.copy(),
            }
        )

    def _next_encoder_position(self) -> int:
        valid_positions = self.position_ids[self.position_ids >= 0]
        return int(valid_positions.max()) + 1 if valid_positions.size else 0

    def _load_prompt_chunk(self, chunk_index: int, image_idx):
        chunk = self._prompt_chunks[chunk_index]
        block_length = chunk["input_ids"].shape[1]
        padding = self.prefill_seq_len - block_length
        self.input_ids = np.pad(chunk["input_ids"], ((0, 0), (0, padding)), constant_values=self._prompt_pad_token_id)
        self.position_ids = np.pad(chunk["position_ids"], ((0, 0), (0, padding)), constant_values=-1)
        self.mm_token_type_ids = np.pad(chunk["mm_token_type_ids"], ((0, 0), (0, padding)))
        input_session = getattr(self, "_input_session", self.session)
        vision_binding = next(
            (binding for binding in input_session.bindings if binding.name == "vision_embeds"),
            None,
        )
        if vision_binding is not None:
            vision_dtype = input_session.aic_to_np_dtype_mapping[vision_binding.type]
            self.vision_embeds = np.zeros(vision_binding.dims, dtype=vision_dtype)
            if chunk.get("vision_embeds") is not None:
                vision_embeds = _to_numpy(chunk["vision_embeds"], vision_dtype)
                self.vision_embeds[:, : vision_embeds.shape[1], :] = vision_embeds
        self.image_idx = _to_numpy(image_idx, np.int64)

    def _prepare_prompt(self, inputs, pad_token_id: int):
        input_ids = _to_numpy(inputs["input_ids"], np.int64)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("`input_ids` must have shape [1, sequence_length].")
        attention_mask = inputs.get("attention_mask")
        sequence_length = (
            int(_to_numpy(attention_mask, np.int64)[0].sum()) if attention_mask is not None else input_ids.shape[1]
        )
        if sequence_length <= 0:
            raise ValueError("The prompt must contain at least one token.")

        position_ids = inputs.get("position_ids")
        if position_ids is None:
            position_ids = np.arange(sequence_length, dtype=np.int64).reshape(1, -1)
        else:
            position_ids = _to_numpy(position_ids, np.int64)[:, :sequence_length]
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = np.zeros_like(input_ids[:, :sequence_length])
        else:
            mm_token_type_ids = _to_numpy(mm_token_type_ids, np.int64)[:, :sequence_length]

        self._prompt_chunks = []
        self._reset_cache_slot_tracking()
        for start in range(0, sequence_length, self.canvas_length):
            end = min(start + self.canvas_length, sequence_length)
            chunk = {
                "input_ids": input_ids[:, start:end],
                "position_ids": position_ids[:, start:end],
                "mm_token_type_ids": mm_token_type_ids[:, start:end],
            }
            if inputs.get("vision_embeds") is not None:
                chunk["vision_embeds"] = inputs["vision_embeds"]
            self._prompt_chunks.append(chunk)

        self._prompt_pad_token_id = int(pad_token_id)
        initial_image_idx = inputs.get("image_idx")
        if initial_image_idx is None:
            initial_image_idx = np.zeros((1, 1), dtype=np.int64)
        self._load_prompt_chunk(0, initial_image_idx)
        return sequence_length

    def _build_additive_mask(self, *, cache_length: int, sliding: bool, cache_position_ids, is_encode: bool):
        block_length = self.prefill_seq_len
        mask = np.full((1, 1, block_length, cache_length + block_length), self._MASK_VALUE, dtype=np.float32)
        current_positions = self.position_ids[0]
        cache_positions = cache_position_ids[0]
        valid_history_end = self._retained_last_position
        slot_positions = (
            self._projected_slot_positions(cache_position_ids, sliding=sliding)
            if is_encode
            else (self._sliding_slot_positions if sliding else self._full_slot_positions)
        )
        if is_encode:
            valid_history_end = max(valid_history_end, int(cache_positions.max(initial=-1)))

        for query_index, query_position in enumerate(current_positions):
            if query_position < 0:
                continue
            if is_encode:
                first_visible = max(0, query_position - cache_length + 1) if sliding else 0
                last_visible = query_position
            else:
                first_visible = max(0, valid_history_end - cache_length + 1) if sliding else 0
                last_visible = valid_history_end
            visible_slots = np.flatnonzero((slot_positions >= first_visible) & (slot_positions <= last_visible))
            mask[0, 0, query_index, visible_slots] = 0.0

        if is_encode:
            vision = (self.mm_token_type_ids[0] == 1) | (self.mm_token_type_ids[0] == 2)
            group_ids = np.full(block_length, -1, dtype=np.int64)
            group = -1
            previous = False
            for index, is_vision in enumerate(vision):
                if is_vision and not previous:
                    group += 1
                if is_vision:
                    group_ids[index] = group
                previous = bool(is_vision)
            for query_index, group_id in enumerate(group_ids):
                if group_id < 0 or current_positions[query_index] < 0:
                    continue
                for key_index, key_group_id in enumerate(group_ids):
                    if key_group_id != group_id or cache_positions[key_index] < 0:
                        continue
                    physical_indices = np.flatnonzero(slot_positions == cache_positions[key_index])
                    mask[0, 0, query_index, physical_indices] = 0.0
        else:
            mask[:, :, :, cache_length:] = 0.0
        return mask

    def _shared_feed(
        self,
        *,
        cache_position_ids,
        is_encode,
        sampling_uniforms,
        temperature,
        use_self_conditioning,
    ):
        cache_position_ids = np.asarray(cache_position_ids, dtype=np.int64)
        return {
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "cache_position_ids": cache_position_ids,
            "full_attention_mask": self._build_additive_mask(
                cache_length=self.full_cache_length,
                sliding=False,
                cache_position_ids=cache_position_ids,
                is_encode=is_encode,
            ),
            "sliding_attention_mask": self._build_additive_mask(
                cache_length=self.sliding_cache_length,
                sliding=True,
                cache_position_ids=cache_position_ids,
                is_encode=is_encode,
            ),
            "vision_embeds": self.vision_embeds,
            "image_idx": self.image_idx,
            "mm_token_type_ids": self.mm_token_type_ids,
            "sampling_uniforms": np.asarray(sampling_uniforms, dtype=np.float32),
            "temperature": np.asarray(temperature, dtype=np.float32),
            "is_encode": np.array([is_encode], dtype=np.int64),
            "use_self_conditioning": np.array([use_self_conditioning], dtype=np.int64),
        }

    def _initial_self_conditioning_state(self):
        probability_dims = self._binding_dims("self_conditioning_topk_probabilities")
        indices_dims = self._binding_dims("self_conditioning_topk_indices")
        if probability_dims is None or indices_dims is None:
            return {}
        return {
            "self_conditioning_topk_probabilities": np.zeros(probability_dims, dtype=np.float32),
            "self_conditioning_topk_indices": np.zeros(indices_dims, dtype=np.int64),
        }

    def _run_session(self, feed, *, is_prefill: bool):
        del is_prefill
        return self.session.run(_session_feed(self.session, feed))

    def prefill(self, debug_callback=None):
        retained_buffers = [
            name
            for name in self.session.input_names + self.session.output_names
            if name.startswith("past_") or name.startswith("self_conditioning_topk_")
        ]
        retained_kv_count = 0
        start = time.perf_counter()
        for chunk_index in range(len(self._prompt_chunks)):
            if chunk_index > 0:
                self._load_prompt_chunk(chunk_index, self.image_idx)
            feed = self._shared_feed(
                cache_position_ids=self.position_ids,
                is_encode=True,
                sampling_uniforms=np.zeros((1, self.canvas_length, 1), dtype=np.float32),
                temperature=np.ones((1, 1), dtype=np.float32),
                use_self_conditioning=False,
            )
            if chunk_index == 0:
                feed.update(self._initial_self_conditioning_state())
            outputs = self._run_session(feed, is_prefill=True)
            if "image_idx_output" in outputs:
                self.image_idx = _to_numpy(outputs["image_idx_output"], np.int64)
            self._record_cache_write(self.position_ids)
            self._retained_last_position = max(self._retained_last_position, int(self.position_ids.max()))
            self._emit_debug(
                debug_callback,
                {
                    "phase": "prefill",
                    "chunk_index": chunk_index,
                    "cache_position_ids": self.position_ids.copy(),
                },
            )
            if chunk_index == 0:
                retained_kv_count = len([name for name in outputs if name.startswith("past_")])
                self.session.skip_buffers(retained_buffers)
        return time.perf_counter() - start, retained_kv_count

    def _denoise_canvas(
        self,
        *,
        block_index: int,
        max_denoising_steps: int,
        sampler: str,
        entropy_bound: float,
        t_min: float,
        t_max: float,
        stability_threshold: int,
        confidence_threshold: float,
        step_callback,
        debug_callback,
    ):
        canvas_start = self._next_encoder_position()
        canvas = self.rng.randint(0, self.vocab_size, size=(1, self.canvas_length)).astype(np.int64)
        self.input_ids = canvas
        self.position_ids = np.arange(canvas_start, canvas_start + self.canvas_length, dtype=np.int64).reshape(1, -1)
        self.mm_token_type_ids = np.zeros_like(canvas)
        self._retained_last_position = canvas_start - 1
        self._active_canvas_position_ids = self.position_ids.copy()
        new_canvas = canvas.copy()
        accepted_mask = np.zeros((1, self.canvas_length), dtype=bool)
        no_cache_write = np.full((1, self.canvas_length), -1, dtype=np.int64)
        argmax_canvas_history = []

        start = time.perf_counter()
        # max_denoising_steps = 16
        for step in range(max_denoising_steps):
            current_step = max_denoising_steps - step
            temperature = t_min + (t_max - t_min) * current_step / max_denoising_steps
            self.input_ids = canvas
            feed = self._shared_feed(
                cache_position_ids=no_cache_write,
                is_encode=False,
                sampling_uniforms=self.rng.uniform(size=(1, self.canvas_length, 1)).astype(np.float32),
                temperature=np.full((1, 1), temperature, dtype=np.float32),
                use_self_conditioning=step > 0,
            )
            outputs = self._run_session(feed, is_prefill=False)
            denoiser_canvas = outputs["denoiser_canvas"].astype(np.int64)
            newly_accepted = outputs["newly_accepted_mask"].astype(bool)
            new_canvas = np.where(newly_accepted, denoiser_canvas, canvas)
            mean_entropy = float(outputs["mean_entropy"].mean())
            argmax_canvas = outputs["top1_indices"].astype(np.int64)
            stable = stability_threshold == 0 or (
                len(argmax_canvas_history) == stability_threshold
                and all(np.array_equal(previous_canvas, argmax_canvas) for previous_canvas in argmax_canvas_history)
            )
            confident = mean_entropy < confidence_threshold
            argmax_canvas_history.append(argmax_canvas)
            if len(argmax_canvas_history) > stability_threshold:
                argmax_canvas_history.pop(0)
            # accepted_mask = accepted_mask | newly_accepted #if sampler == "local" else newly_accepted
            accepted_mask = newly_accepted  # accepted_mask | newly_accepted #if sampler == "local" else newly_accepted
            canvas = np.where(
                ~accepted_mask,
                self.rng.randint(0, self.vocab_size, size=(1, self.canvas_length)).astype(np.int64),
                new_canvas,
            )

            accepted_count = int(accepted_mask.sum())
            self._emit_debug(
                debug_callback,
                {
                    "phase": "decode",
                    "block_index": block_index,
                    "step": step,
                    "temperature": temperature,
                    "entropy_mean": mean_entropy,
                    "accepted_count": accepted_count,
                    "stable": stable,
                    "confident": confident,
                },
            )
            if step_callback is not None:
                step_callback(
                    {
                        "block_index": block_index,
                        "step": step,
                        "temperature": temperature,
                        "accepted_count": accepted_count,
                        "canvas_length": self.canvas_length,
                        "tokens": new_canvas,
                        "stable": stable,
                        "confident": confident,
                    }
                )
            if accepted_count >= self.canvas_length or (stable and confident):
                break
        return new_canvas, step + 1, time.perf_counter() - start, int(accepted_mask.sum())

    def _commit_canvas(self, tokens: np.ndarray, debug_callback=None):
        commit_length = int(tokens.shape[1])
        if commit_length > self.prefill_seq_len:
            raise ValueError(f"Commit length {commit_length} exceeds compiled block length {self.prefill_seq_len}.")
        if self._active_canvas_position_ids is None:
            raise RuntimeError("No active canvas positions are available to commit.")
        commit_position_ids = self._active_canvas_position_ids[:, :commit_length]
        if np.any(commit_position_ids < 0):
            raise RuntimeError("Active canvas positions must be valid before commit.")
        self.input_ids = np.full((1, self.prefill_seq_len), self.pad_token_id, dtype=np.int64)
        self.input_ids[:, :commit_length] = tokens
        self.position_ids = np.full((1, self.prefill_seq_len), -1, dtype=np.int64)
        self.position_ids[:, :commit_length] = commit_position_ids
        self.mm_token_type_ids = np.zeros((1, self.prefill_seq_len), dtype=np.int64)
        outputs = self._run_session(
            self._shared_feed(
                cache_position_ids=self.position_ids,
                is_encode=True,
                sampling_uniforms=np.zeros((1, self.canvas_length, 1), dtype=np.float32),
                temperature=np.ones((1, 1), dtype=np.float32),
                use_self_conditioning=False,
            ),
            is_prefill=False,
        )
        if "image_idx_output" in outputs:
            self.image_idx = _to_numpy(outputs["image_idx_output"], np.int64)
        self._record_cache_write(self.position_ids)
        self._retained_last_position = int(commit_position_ids.max())
        self._emit_debug(
            debug_callback,
            {
                "phase": "commit",
                "cache_position_ids": self.position_ids.copy(),
            },
        )

    def generate(
        self,
        *,
        inputs,
        generation_len: int,
        max_denoising_steps: int = 48,
        sampler: str = "local",
        entropy_bound: float = 0.1,
        t_min: float = 0.4,
        t_max: float = 0.8,
        stability_threshold: int = 1,
        confidence_threshold: float = 0.005,
        ctx_len: Optional[int] = None,
        pad_token_id: int = 0,
        eos_token_id=None,
        stop_on_eos: bool = True,
        step_callback: Optional[Callable[[dict], None]] = None,
        debug_callback: Optional[Callable[[dict], None]] = None,
    ) -> DiffusionGemmaSingleQPCRuntimeResult:
        if generation_len <= 0:
            raise ValueError("`generation_len` must be positive.")
        if max_denoising_steps <= 0:
            raise ValueError("`max_denoising_steps` must be positive.")
        if sampler not in {"local", "hf"}:
            raise ValueError("`sampler` must be either 'local' or 'hf'.")
        if entropy_bound != _DEVICE_ENTROPY_BOUND:
            raise ValueError(f"`entropy_bound` is fixed at {_DEVICE_ENTROPY_BOUND} for the compiled QPC.")
        if not 0 <= t_min <= t_max:
            raise ValueError("Temperature bounds must satisfy 0 <= t_min <= t_max.")
        if isinstance(stability_threshold, bool) or not isinstance(stability_threshold, int) or stability_threshold < 0:
            raise ValueError("`stability_threshold` must be a non-negative integer.")
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or confidence_threshold < 0
        ):
            raise ValueError("`confidence_threshold` must be a non-negative number.")

        total_start = perf_counter()
        try:
            sequence_length = self._prepare_prompt(inputs, pad_token_id=pad_token_id)
            compiled_ctx_len = ctx_len or _infer_context_length(self.session)
            target_new_tokens = int(generation_len)
            if compiled_ctx_len is not None:
                target_new_tokens = min(target_new_tokens, max(0, int(compiled_ctx_len) - sequence_length))
            if target_new_tokens <= 0:
                raise ValueError("The compiled context length has no room for generated tokens.")

            ttft, retained_kv_buffers = self.prefill(debug_callback=debug_callback)
            eos_token_ids = _normalize_eos_token_ids(eos_token_id)
            generated = []
            total_steps = 0
            total_canvas_time = 0.0
            num_blocks = int(math.ceil(target_new_tokens / self.canvas_length))
            print(f"Total number of blocks is {num_blocks}")
            for block_index in range(num_blocks):
                emitted_tokens = sum(tokens.shape[1] for tokens in generated)
                remaining_tokens = target_new_tokens - emitted_tokens
                canvas_tokens, steps_run, canvas_time, _ = self._denoise_canvas(
                    block_index=block_index,
                    max_denoising_steps=max_denoising_steps,
                    sampler=sampler,
                    entropy_bound=entropy_bound,
                    t_min=t_min,
                    t_max=t_max,
                    stability_threshold=stability_threshold,
                    confidence_threshold=confidence_threshold,
                    step_callback=step_callback,
                    debug_callback=debug_callback,
                )
                total_steps += steps_run
                total_canvas_time += canvas_time
                canvas_tokens = canvas_tokens[:, :remaining_tokens]

                hit_eos = False
                if stop_on_eos and eos_token_ids:
                    eos_positions = np.where(np.isin(canvas_tokens[0], eos_token_ids))[0]
                    if eos_positions.size:
                        canvas_tokens = canvas_tokens[:, : int(eos_positions[0]) + 1]
                        hit_eos = True

                generated.append(canvas_tokens)
                if hit_eos:
                    break
                if block_index + 1 < num_blocks and canvas_tokens.shape[1] > 0:
                    self._commit_canvas(canvas_tokens, debug_callback=debug_callback)

            generated_ids = np.concatenate(generated, axis=1) if generated else np.zeros((1, 0), dtype=np.int64)
            valid_tokens = np.array([generated_ids.shape[1]], dtype=np.float32)
            decode_forward_passes = np.array([total_steps], dtype=np.int64)
            tokens_per_forward = valid_tokens / np.maximum(decode_forward_passes.astype(np.float32), 1.0)
            return DiffusionGemmaSingleQPCRuntimeResult(
                generated_ids=generated_ids,
                tokens_per_forward=tokens_per_forward,
                decode_forward_passes=decode_forward_passes,
                total_time=max(perf_counter() - total_start, 1e-6),
                ttft=ttft,
                retained_kv_buffers=retained_kv_buffers,
                total_steps=total_steps,
                executed_blocks=len(generated),
                total_canvas_time=total_canvas_time,
                canvas_length=self.canvas_length,
            )
        finally:
            self.session.deactivate()


class DiffusionGemmaSplitQPCGenerator(DiffusionGemmaSingleQPCGenerator):
    """Chunked prefill with DMA handoff into decoder-owned retained KV."""

    def __init__(
        self, *, model_config, encoder_session: QAICInferenceSession, decoder_session: QAICInferenceSession, seed=1234
    ):
        super().__init__(model_config=model_config, session=decoder_session, seed=seed)
        self.encoder_session = encoder_session
        self._input_session = encoder_session
        self._kv_caches = [np.zeros(shape, dtype=dtype) for shape, dtype in decoder_session.kv_cache_info]
        if not self._kv_caches:
            raise ValueError("Decoder QPC does not expose retained KV input bindings for DMA handoff.")
        self._decoder_kv_imported = False
        self._last_encoder_outputs = {}

    def _prepare_prompt(self, inputs, pad_token_id: int):
        self._input_session = self.encoder_session
        return super()._prepare_prompt(inputs, pad_token_id)

    def prefill(self, debug_callback=None):
        self._input_session = self.encoder_session
        self._decoder_kv_imported = False
        retained_kv_count = len(self._kv_caches)
        start = time.perf_counter()
        for chunk_index in range(len(self._prompt_chunks)):
            if chunk_index > 0:
                self._load_prompt_chunk(chunk_index, self.image_idx)
            feed = self._shared_feed(
                cache_position_ids=self.position_ids,
                is_encode=True,
                sampling_uniforms=np.zeros((1, self.canvas_length, 1), dtype=np.float32),
                temperature=np.ones((1, 1), dtype=np.float32),
                use_self_conditioning=False,
            )
            is_last_chunk = chunk_index == len(self._prompt_chunks) - 1
            exec_idx = self.encoder_session.np_run_pipeline(
                _session_feed(self.encoder_session, feed),
                last_chunk=is_last_chunk,
                kv_cache_buffers=self._kv_caches if is_last_chunk else None,
            )
            self.encoder_session.complete_inf(exec_idx, is_prefill=True)
            encoder_outputs = self.encoder_session.get_outputs(exec_idx)
            self._last_encoder_outputs = encoder_outputs
            if "image_idx_output" in encoder_outputs:
                self.image_idx = _to_numpy(encoder_outputs["image_idx_output"], np.int64)
            self._record_cache_write(self.position_ids)
            self._retained_last_position = max(self._retained_last_position, int(self.position_ids.max()))
            self._emit_debug(
                debug_callback,
                {
                    "phase": "prefill",
                    "chunk_index": chunk_index,
                    "cache_position_ids": self.position_ids.copy(),
                },
            )
        return time.perf_counter() - start, retained_kv_count

    def _run_session(self, feed, *, is_prefill: bool):
        if is_prefill:
            raise RuntimeError("Split decoder execution cannot use the prefill role.")
        importing_prefill_kv = not self._decoder_kv_imported
        if importing_prefill_kv:
            self.session.set_data_for_kv_handoff(
                self._kv_caches,
                [("batch_index", 0), ("ctx_start", 0)],
                index=self.session.decode_execObj_idx,
                buff_map=self.session.decode_buff_map,
            )
        exec_idx = self.session.np_run(_session_feed(self.session, feed), is_prefill=False)
        self.session.complete_inf(exec_idx, is_prefill=False)
        if importing_prefill_kv:
            self._decoder_kv_imported = True
        return self.session.get_outputs(exec_idx)

    def generate(self, **kwargs):
        try:
            return super().generate(**kwargs)
        finally:
            self.encoder_session.deactivate()


def diffusion_gemma_generate_single_qpc_chunked(
    *,
    qeff_model,
    inputs,
    generation_len: int,
    qpc_path,
    device_ids=None,
    **kwargs,
):
    generation_config = getattr(qeff_model.model, "generation_config", None)
    pad_token_id = kwargs.pop("pad_token_id", getattr(generation_config, "pad_token_id", None))
    eos_token_id = kwargs.pop("eos_token_id", getattr(generation_config, "eos_token_id", None))
    if pad_token_id is None:
        pad_token_id = 0

    session = QAICInferenceSession(str(qpc_path), device_ids=None)
    generator = DiffusionGemmaSingleQPCGenerator(
        model_config=qeff_model.model.config,
        session=session,
        seed=kwargs.pop("seed", 1234),
    )
    return generator.generate(
        inputs=inputs,
        generation_len=generation_len,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        **kwargs,
    )


def diffusion_gemma_generate_split_qpc_chunked(
    *,
    qeff_model,
    inputs,
    generation_len: int,
    encoder_qpc_path,
    decoder_qpc_path,
    device_ids=None,
    **kwargs,
):
    generation_config = getattr(qeff_model.model, "generation_config", None)
    pad_token_id = kwargs.pop("pad_token_id", getattr(generation_config, "pad_token_id", None))
    eos_token_id = kwargs.pop("eos_token_id", getattr(generation_config, "eos_token_id", None))
    prefill_stages = _validate_prefill_stages(kwargs.pop("prefill_stages", 1))
    if pad_token_id is None:
        pad_token_id = 0

    encoder_session = QAICInferenceSession(
        str(encoder_qpc_path),
        device_ids=device_ids,
        kv_dma_share=True,
        stages=prefill_stages,
        cluster_id="prefill",
        full_batch_size=1,
    )
    decoder_session = QAICInferenceSession(
        str(decoder_qpc_path),
        device_ids=device_ids,
        kv_dma_share=True,
        cluster_id="decode",
        full_batch_size=1,
    )
    generator = DiffusionGemmaSplitQPCGenerator(
        model_config=qeff_model.model.config,
        encoder_session=encoder_session,
        decoder_session=decoder_session,
        seed=kwargs.pop("seed", 1234),
    )
    return generator.generate(
        inputs=inputs,
        generation_len=generation_len,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        **kwargs,
    )


def diffusion_gemma_prefill_split_qpc_chunked(
    *,
    qeff_model,
    inputs,
    encoder_qpc_path,
    decoder_qpc_path,
    device_ids=None,
    seed: int = 1234,
    prefill_stages: int = 1,
) -> DiffusionGemmaPrefillRuntimeResult:
    """Run batch-1 split encoder prefill and return decoder-ready retained KV buffers."""
    prefill_stages = _validate_prefill_stages(prefill_stages)
    if inputs.get("vision_embeds") is not None:
        raise ValueError("The staged prefill helper supports text-only inputs only.")

    generation_config = getattr(qeff_model.model, "generation_config", None)
    pad_token_id = getattr(generation_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    encoder_session = QAICInferenceSession(
        str(encoder_qpc_path),
        device_ids=device_ids,
        kv_dma_share=True,
        stages=prefill_stages,
        cluster_id="prefill",
        full_batch_size=1,
    )
    decoder_session = QAICInferenceSession(
        str(decoder_qpc_path),
        device_ids=device_ids,
        kv_dma_share=True,
        cluster_id="decode",
        full_batch_size=1,
    )
    generator = DiffusionGemmaSplitQPCGenerator(
        model_config=qeff_model.model.config,
        encoder_session=encoder_session,
        decoder_session=decoder_session,
        seed=seed,
    )
    try:
        prompt_length = generator._prepare_prompt(inputs, pad_token_id)
        total_time, _ = generator.prefill()
        return DiffusionGemmaPrefillRuntimeResult(
            kv_caches=generator._kv_caches,
            encoder_outputs=generator._last_encoder_outputs,
            prompt_length=prompt_length,
            next_position=generator._next_encoder_position(),
            total_time=total_time,
        )
    finally:
        encoder_session.deactivate()
        decoder_session.deactivate()


def load_model_and_processor(model_id: str, canvas_length: int):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=False,
    )
    qeff_model.model.config.canvas_length = canvas_length
    return processor, qeff_model


def compile_unified_qpc(
    qeff_model,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    canvas_length: int,
    num_devices: int,
    num_cores: int,
):
    print(f"Compiling unified single-QPC ({num_devices} devices, {num_cores} cores)...")
    start = time.time()
    unified = UnifiedQPC(qeff_model)
    qaic_config_moe = {
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": 128,
            "tree_reduce": True,
        }
    }
    specializations, _ = unified.model.get_specializations(
        batch_size=1,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        canvas_length=canvas_length,
    )
    onnx_path = unified.get_onnx_path(
        specializations=specializations,
        mdp_ts_num_devices=num_devices,
        qaic_config=qaic_config_moe,
        aic_num_cores=num_cores,
    )
    # npi_file_path = '/home/jsaisaga//home/jsaisaga/.cache/qeff_models/DiffusionGemmaForBlockDiffusion/DiffusionGemmaUnifiedWrapper-9c317201f1a690a7/npi_fp32_unified_accum.yaml'
    npi_file_path = qeff_model.generate_npi_file(onnx_path)

    custom_io = {"vision_embeds": "float16"}
    for layer_index in range(qeff_model.config.text_config.num_hidden_layers):
        for kv_name in ("key", "value"):
            custom_io[f"past_{kv_name}.{layer_index}"] = "mxint8"
            custom_io[f"past_{kv_name}.{layer_index}_RetainedState"] = "mxint8"

    qpc_path = unified._compile(
        onnx_path=onnx_path,
        compile_dir=None,
        specializations=specializations,
        convert_to_fp16=True,
        mxfp6_matmul=True,
        mdp_ts_num_devices=num_devices,
        aic_num_cores=num_cores,
        custom_io=custom_io,
        retained_state=True,
        qaic_config=qaic_config_moe,
        user_tiled=True,
        node_precision_info=npi_file_path,
    )
    print(f"  unified QPC: {qpc_path} ({time.time() - start:.0f}s)")
    return qpc_path


def compile_split_qpcs(
    qeff_model,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    canvas_length: int,
    num_devices: int,
    num_cores: int,
    prefill_stages: int = 1,
):
    prefill_stages = _validate_prefill_stages(prefill_stages)
    if prefill_seq_len != canvas_length:
        raise ValueError("Split DiffusionGemma currently requires matching prefill and canvas lengths.")

    print(f"Compiling split DiffusionGemma QPCs ({num_devices} devices, {num_cores} cores)...")
    start = time.time()
    encoder = DiffusionGemmaEncoderQPC(qeff_model)
    decoder = DiffusionGemmaDecoderQPC(qeff_model)
    qaic_config_moe = {
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": 128,
            "tree_reduce": True,
        }
    }
    encoder_specializations, _ = encoder.model.get_specializations(
        batch_size=1,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
    )
    decoder_specializations, _ = decoder.model.get_specializations(
        batch_size=1,
        canvas_length=canvas_length,
        ctx_len=ctx_len,
    )
    custom_io = {}
    for layer_index in range(qeff_model.config.text_config.num_hidden_layers):
        for kv_name in ("key", "value"):
            custom_io[f"past_{kv_name}.{layer_index}"] = "mxint8"
            custom_io[f"past_{kv_name}.{layer_index}_RetainedState"] = "mxint8"
    common_compile_kwargs = {
        "compile_dir": None,
        "convert_to_fp16": True,
        "mxfp6_matmul": True,
        "mdp_ts_num_devices": num_devices,
        "aic_num_cores": num_cores,
        "retained_state": True,
        "retain_full_kv": True,
        "split_retained_state_io": True,
        "user_tiled": True,
        "custom_io": custom_io,
    }
    encoder_onnx_path = encoder.get_onnx_path(
        specializations=encoder_specializations,
        offload_pt_weights=False,
        retain_full_kv=True,
        mdp_ts_num_devices=num_devices,
        qaic_config=qaic_config_moe,
        aic_num_cores=num_cores,
    )
    print(f"  encoder ONNX: {encoder_onnx_path}")
    decoder_onnx_path = decoder.get_onnx_path(
        specializations=decoder_specializations,
        offload_pt_weights=False,
        retain_full_kv=True,
        mdp_ts_num_devices=num_devices,
        qaic_config=qaic_config_moe,
        aic_num_cores=num_cores,
    )
    print(f"  decoder ONNX: {decoder_onnx_path}")
    encoder_qpc_path = encoder._compile(
        onnx_path=encoder_onnx_path,
        specializations=encoder_specializations,
        qaic_config=qaic_config_moe,
        mdp_num_partitions=prefill_stages,
        **common_compile_kwargs,
    )
    print(f"  encoder QPC: {encoder_qpc_path}")
    decoder_qpc_path = decoder._compile(
        onnx_path=decoder_onnx_path,
        specializations=decoder_specializations,
        node_precision_info=qeff_model.generate_npi_file(decoder_onnx_path),
        qaic_config=qaic_config_moe,
        **common_compile_kwargs,
    )
    print(f"  decoder QPC: {decoder_qpc_path} ({time.time() - start:.0f}s)")
    return encoder_qpc_path, decoder_qpc_path


def _vision_embeds_cpu(model_id: str, text_model, vision_inputs):
    hf_vision = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    with torch.no_grad():
        encoder = hf_vision.model.encoder
        pixel_values = vision_inputs["pixel_values"]
        image_position_ids = vision_inputs["image_position_ids"]
        padding_positions = (image_position_ids == -1).all(dim=-1)
        hidden_states = encoder.vision_tower.patch_embedder(pixel_values, image_position_ids, padding_positions)
        attention_mask = (
            padding_positions.unsqueeze(1).unsqueeze(2).to(hidden_states.dtype) * torch.finfo(hidden_states.dtype).min
        )
        attention_mask = attention_mask.expand(-1, 1, hidden_states.shape[1], -1)
        position_embeddings = encoder.vision_tower.encoder.rotary_emb(hidden_states, image_position_ids)
        for layer in encoder.vision_tower.encoder.layers[: encoder.vision_tower.encoder.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=image_position_ids,
            )
        hidden_states, _ = encoder.vision_tower.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=image_position_ids,
            padding_positions=padding_positions,
            output_length=encoder.vision_tower.config.default_output_length,
        )
        if encoder.vision_tower.config.standardize:
            hidden_states = (hidden_states - encoder.vision_tower.std_bias) * encoder.vision_tower.std_scale
        vision_embeds = encoder.embed_vision(inputs_embeds=hidden_states).clamp(-60000.0, 60000.0)
        vision_embeds = vision_embeds[:, : text_model._get_mm_tokens_per_image(), :].float()
    del hf_vision
    return vision_embeds


def prepare_prompt_inputs(
    *,
    processor,
    qeff_model,
    model_id: str,
    prompt: str,
    text_only: bool,
    image_url: str,
):
    if text_only:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if not text_only:
        inputs["vision_embeds"] = _vision_embeds_cpu(model_id, qeff_model, inputs)
    return inputs


def clean_diffusion_text(text: str, truncate_first_sentence: bool = True):
    text = text.replace("\ufffd", " ").strip()
    text = re.sub(r"^\s*(thought\s*)+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).replace("。", ".")
    text = re.sub(r"\bfulling shot\b", "full shot", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(light|dark)\s+(blue|green|teal)ing\b", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(?:of|Of)\b.*$", ".", text)
    match = re.search(r"(.{12,}?[.!?])", text) if truncate_first_sentence else None
    if match:
        text = match.group(1)
    return text.strip(" \n\t\r\"'")


def build_step_callback(tokenizer, verbose_steps: bool):
    def callback(event):
        prefix = (
            f"  block {event['block_index'] + 1:2d} step {event['step'] + 1:2d} "
            f"t={event['temperature']:.2f} "
            f"acc={event['accepted_count']}/{event['canvas_length']}"
        )
        if verbose_steps:
            preview = tokenizer.decode(event["tokens"][0].tolist(), skip_special_tokens=True)
            prefix += f" :: {preview[:60]!r}"
        print(prefix)

    return callback
