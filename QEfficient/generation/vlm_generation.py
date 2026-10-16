# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
This module provides the VisionLanguageGeneration class that inherits from
QEffTextGenerationBase, enabling all advanced text generation features while
maintaining full API compatibility with the original VisionLanguageGeneration.

Key enhancements:
- Continuous batching support for vision models
- Advanced streaming capabilities
- On-device sampling support
- LoRA adapter support
- Better performance metrics
"""

from collections import deque
from time import perf_counter
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoImageProcessor, PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.cloud_infer import QAICInferenceSession, is_retained_state_name
from QEfficient.generation.embedding_handler import VisionHandler
from QEfficient.generation.text_generation_inference import (
    CloudAI100ExecInfo,
    PerfMetrics,
    QEffTextGenerationBase,
    TextGeneration,
    calculate_latency,
    get_compilation_dims,
)
from QEfficient.utils import LRUCache, constants
from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


class VisionLanguageGeneration(QEffTextGenerationBase):
    """
    Enhanced vision-language generation class inheriting from QEffTextGenerationBase.

    This class maintains full API compatibility with VisionLanguageGeneration while
    adding advanced features like continuous batching, streaming, and sampling.

    Example:
        >>> # Drop-in replacement for VisionLanguageGeneration
        >>> vlm = VisionLanguageGeneration(
        ...     tokenizer=tokenizer,
        ...     processor=processor,
        ...     lang_qpc_path="path/to/lang.qpc",
        ...     vision_qpc_path="path/to/vision.qpc",
        ...     device_ids=[0]
        ... )
        >>> result = vlm.generate(
        ...     images=["image1.jpg"],
        ...     prompts=["Describe this image"],
        ...     generation_len=512
        ... )

        >>> # Enhanced usage with new features
        >>> vlm_enhanced = VisionLanguageGeneration(
        ...     tokenizer=tokenizer,
        ...     processor=processor,
        ...     lang_qpc_path="path/to/lang.qpc",
        ...     vision_qpc_path="path/to/vision.qpc",
        ...     device_ids=[0],
        ...     full_batch_size=8,  # Enable continuous batching
        ...     include_sampler=True,  # Enable on-device sampling
        ...     sampling_params=sampling_config
        ... )
    """

    def __init__(
        self,
        qeff_model,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        processor: AutoImageProcessor,
        lang_qpc_path: str,
        vision_qpc_path: str,
        device_ids: Optional[List[int]] = None,
        ctx_len: Optional[int] = None,
        num_kv_blocks: Optional[int] = None,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        enable_debug_logs: bool = False,
        full_batch_size: Optional[int] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        is_tlm: bool = False,
        include_sampler: bool = False,
        return_pdfs: bool = False,
        include_guided_decoding: bool = False,
        sampling_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize vision-language generation with enhanced capabilities

        Args:
            qeff_model: QEff model instance
            tokenizer: Text tokenizer
            processor: Image processor
            lang_qpc_path: Path to language model QPC
            vision_qpc_path: Path to vision encoder QPC
            device_ids: Device IDs for execution (default: [0])
            ctx_len: Context length
            num_kv_blocks: Number of physical KV blocks for paged attention on the language
                QPC (optional). When None (default), paged attention is disabled and
                block_table/slot_id are never populated or sent to the language session.
            enable_debug_logs: Enable debug logging
            full_batch_size: Enable continuous batching (new feature)
            image_height: Desired image height for resizing
            image_width: Desired image width for resizing
            is_tlm: Target language model flag
            include_sampler: Enable on-device sampling (new feature)
            return_pdfs: Return probability distributions
            include_guided_decoding: Enable guided decoding in on-device sampling
            sampling_params: Sampling parameters for on-device sampling
        """
        # Validate required parameters
        if not lang_qpc_path:
            raise TypeError("lang_qpc_path is required")
        if not vision_qpc_path:
            raise TypeError("vision_qpc_path is required")

        # Initialize base class with language QPC
        # Pass activate=False to prevent premature activation before vision components are ready
        super().__init__(
            tokenizer=tokenizer,
            qpc_path=lang_qpc_path,
            full_batch_size=full_batch_size,
            num_kv_blocks=num_kv_blocks,
            ctx_len=ctx_len,
            comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
            comp_ctx_lengths_decode=comp_ctx_lengths_decode,
            device_ids=device_ids,
            enable_debug_logs=enable_debug_logs,
            is_tlm=is_tlm,
            include_sampler=include_sampler,
            return_pdfs=return_pdfs,
            include_guided_decoding=include_guided_decoding,
            sampling_params=sampling_params,
            activate=False,  # vision components need to be initialized first
        )

        # Physical KV block table for paged attention on the language QPC. None (default)
        # when num_kv_blocks isn't supplied, matching the non-VLM TextGeneration path
        # (_setup_model_execution_inputs) -- every paged-attention code path below is
        # already gated on self.block_table/self.num_kv_blocks being non-None.
        execution_batch_size = full_batch_size if full_batch_size else self.batch_size
        self.block_table = (
            np.arange(execution_batch_size * self.num_kv_blocks, dtype=np.int64).reshape(
                execution_batch_size, self.num_kv_blocks
            )
            if self.num_kv_blocks
            else None
        )

        # Vision-specific initialization
        model_type = getattr(getattr(qeff_model, "model", None).config, "model_type", "")
        self.is_qwen_vl = model_type in {
            "qwen2_5_vl",
            "qwen3_vl_moe",
            "qwen3_vl",
            "qwen3_5",
            "qwen3_5_moe",
        }
        self.qeff_model = qeff_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_height = image_height
        self.image_width = image_width
        self._vision_qpc_path = vision_qpc_path
        self.device_ids = device_ids  # Store device_ids for vision components
        self.enable_debug_logs = enable_debug_logs  # Store for vision components
        self._vision_outputs_cache = LRUCache(max_size=100)  # LRU cache for vision outputs
        self._vision_cache = {}  # Cache for vision outputs across batches
        self._init_vision_components()

        # Now that vision components are initialized, activate the text session
        self._session.activate()

        logger.info(
            f"VisionLanguageGeneration initialized: batch_size={self.batch_size}, "
            f"prefill_seq_len={self._prefill_seq_len}, ctx_len={ctx_len}, "
            f"continuous_batching={'enabled' if full_batch_size else 'disabled'}, "
            f"sampling={'enabled' if include_sampler else 'disabled'}"
        )

    def _init_vision_components(self):
        """Initialize vision-specific components"""
        # Vision session (separate from base class language session)
        self._vision_session = QAICInferenceSession(
            self._vision_qpc_path, self.device_ids, activate=False, enable_debug_logs=self.enable_debug_logs
        )

        # Vision handler with language session coordination
        vision_config = self._get_vision_config()
        self._vision_handler = VisionHandler(
            qeff_model=self.qeff_model,
            vision_session=self._vision_session,
            processor=self.processor,
            tokenizer=self.tokenizer,
            image_height=self.image_height,
            image_width=self.image_width,
            config=vision_config,
            lang_session=self._session,  # Pass language session for coordination
        )

        # Setup vision buffer skipping
        self._setup_vision_buffer_skipping()

    def _get_vision_config(self) -> Dict[str, Any]:
        """
        Derive vision config from session

        Returns:
            Dictionary with vision configuration
        """
        config = {}
        if self._vision_session:
            try:
                shapes = {}
                for output_name in self._vision_session.output_names:
                    if (
                        hasattr(self._vision_session, "bindings")
                        and output_name in self._vision_session.binding_index_map
                    ):
                        binding_idx = self._vision_session.binding_index_map[output_name]
                        if hasattr(self._vision_session.bindings[binding_idx], "dims"):
                            shapes[output_name] = tuple(self._vision_session.bindings[binding_idx].dims)

                if shapes:
                    config["vision_output_shapes"] = shapes
            except Exception as e:
                logger.warning(f"Could not derive vision config from session: {e}")

        return config

    def _setup_vision_buffer_skipping(self):
        """Skip KV cache and retained state buffers for vision session"""
        # Pre-compute skip buffers
        self._vision_skip_buffers = [
            x
            for x in self._vision_session.input_names + self._vision_session.output_names
            if is_retained_state_name(x) or x.endswith("_RetainedState")
        ]
        self._vision_session.skip_buffers(self._vision_skip_buffers)

        # Pre-compute language skip buffers
        self._lang_skip_buffers = [
            x
            for x in self._session.input_names + self._session.output_names
            if is_retained_state_name(x) or x.endswith("_RetainedState")
        ]

    def run_prefill_for_all_inputs(self, prompt_queue, generation_len):
        """
        Runs prefill for all inputs in the prompt queue and updates the decode input.

        Method iterates over the full batch size and for each decode batch ID, it pops the next prompt from the queue.  It then runs prefill for the next prompt and updates the decode input with the outputs.

        Args:
            prompt_queue (deque): The queue of prompts.
            generation_len (int): The generation length.

        """
        for decode_batch_id in range(self.full_batch_size):
            next_prompt = prompt_queue.popleft()

            # run prefill for num_chunks
            outputs, position_ids, generation_len = self.run_prefill(
                next_prompt, generation_len, decode_batch_id=np.array(decode_batch_id, dtype=np.int64).reshape(1, 1)
            )
            if self.is_qwen_vl:
                _ = self.update_decode_inputs_qwen_vl(outputs, position_ids, generation_len, decode_batch_id)
            else:
                _ = self.update_decode_input(outputs, position_ids, generation_len, decode_batch_id)

    def update_decode_inputs_qwen_vl(self, outputs, position_ids, generation_len, decode_batch_id=None):
        """
        Updates the decode input with the generated values.
        Args:
            outputs (dict): The outputs of the model.
            position_ids (array): The position IDs.
            generation_len (int): The generation length.
            decode_batch_id (int, optional): The decode batch ID. If None, all values are updated. Defaults to None.

        Returns:
            next_token_id (array): The next token ID.
        """
        next_token_id = self._fetch_next_token_id(outputs)
        if next_token_id.ndim == 2 and next_token_id.shape[1] > 1:
            next_token_id = next_token_id[:, -1:]

        # Store the generated values.
        decode_batch = decode_batch_id if decode_batch_id is not None else slice(None)
        self.decode_input_ids[decode_batch] = next_token_id
        self.decode_pos_ids[:, decode_batch] = position_ids.squeeze(1)
        self.generated_ids[decode_batch, 0] = next_token_id.squeeze()
        self.generation_len[decode_batch] = generation_len
        return next_token_id

    def _execute_chunked_prefill(
        self,
        lang_inputs: Dict[str, np.ndarray],
        num_chunks: int,
        decode_batch_id: Optional[np.ndarray] = None,
        prefill_logit_bs: int = 1,
        block_table: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Execute chunked prefill with language inputs

        Args:
            lang_inputs: Pre-processed language inputs with input_ids, position_ids, etc.
            num_chunks: Number of chunks to process
            decode_batch_id: Batch ID for continuous batching (optional)
            prefill_logit_bs: Batch size for prefill logits
            block_table: Physical KV block table for paged attention (optional)

        Returns:
            Final prefill outputs
        """
        # Skip buffers for dual-QPC coordination
        self._session.skip_buffers(self._lang_skip_buffers)

        # Run chunked prefill
        outputs = None
        chunk_image_idx = None

        if self.comp_ctx_lengths_prefill is not None:
            self.list_of_comp_ctx_lengths_prefill = [
                np.zeros(length, dtype=np.int64) for length in self.comp_ctx_lengths_prefill
            ]
            prefill_ccl_id = 0
            lang_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

        if self.include_sampler:
            for op in Constants.SAMPLER_OPS | ({"token_bitmasks"} if self.include_guided_decoding else set()):
                if decode_batch_id is not None:
                    lang_inputs[op] = self.sampling_params[op][decode_batch_id.flatten()]
                else:
                    lang_inputs[op] = self.sampling_params[op]

        prefill_output_seq_len_by_input_seq = {}
        default_prefill_output_seq_len = 1
        if not self.include_sampler:
            logits_binding_idx = self._session.binding_index_map.get("logits")
            input_ids_binding_idx = self._session.binding_index_map.get("input_ids")
            if logits_binding_idx is not None and input_ids_binding_idx is not None:
                if getattr(self._session, "allowed_shapes", None):
                    for shape_spec in self._session.allowed_shapes:
                        if len(shape_spec) <= max(input_ids_binding_idx, logits_binding_idx):
                            continue
                        input_dims = shape_spec[input_ids_binding_idx][1]
                        logits_dims = shape_spec[logits_binding_idx][1]
                        if len(input_dims) < 2 or len(logits_dims) < 2:
                            continue
                        prefill_output_seq_len_by_input_seq[int(input_dims[1])] = int(logits_dims[1])
                logits_binding = self._session.bindings[logits_binding_idx]
                if len(logits_binding.dims) > 1:
                    default_prefill_output_seq_len = int(logits_binding.dims[1])

        for i in range(num_chunks):
            input_ids_slice = lang_inputs["input_ids"][:, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len]
            position_ids_slice = lang_inputs["position_ids"][
                ..., i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]

            prefill_output_seq_len = prefill_output_seq_len_by_input_seq.get(
                int(input_ids_slice.shape[1]), default_prefill_output_seq_len
            )
            self._set_output_buffers(batch_size=prefill_logit_bs, sequence_length=prefill_output_seq_len)

            chunk_inputs = {
                "input_ids": input_ids_slice,
                "position_ids": position_ids_slice,
                "image_idx": chunk_image_idx if chunk_image_idx is not None else np.array([[0]], dtype=np.int64),
            }

            if block_table is not None:
                chunk_inputs["block_table"] = block_table
                chunk_start_position_id = i * self._prefill_seq_len
                chunk_inputs["slot_id"] = np.full(
                    (prefill_logit_bs,), chunk_start_position_id % self.kv_block_size, dtype=np.int64
                )

            if "mm_token_type_ids" in lang_inputs:
                chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
                    :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
                ]

            if decode_batch_id is not None:
                chunk_inputs["batch_index"] = decode_batch_id

            if "cross_attention_mask" in lang_inputs:
                chunk_inputs["cross_attention_mask"] = lang_inputs["cross_attention_mask"]

            if self.comp_ctx_lengths_prefill is not None:
                if (i + 1) * self._prefill_seq_len > self.comp_ctx_lengths_prefill[prefill_ccl_id]:
                    prefill_ccl_id = min(prefill_ccl_id + 1, len(self.comp_ctx_lengths_prefill) - 1)
                    lang_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

                chunk_inputs["comp_ctx_lengths"] = lang_inputs["comp_ctx_lengths"]

            if self.include_sampler:
                chunk_inputs["last_accepted_output_tokens"] = chunk_inputs["input_ids"]
                for op in Constants.SAMPLER_OPS | ({"token_bitmasks"} if self.include_guided_decoding else set()):
                    chunk_inputs[op] = lang_inputs[op]

            outputs = self._session.run(chunk_inputs)

            if "image_idx_output" in outputs:
                chunk_image_idx = outputs["image_idx_output"]

        # Prepare decode-time cross_attention_mask
        if "cross_attention_mask" in lang_inputs:
            bs, _, num_images, img_tiles = lang_inputs["cross_attention_mask"].shape
            self._decode_cross_attention_mask = np.ones((bs, 1, num_images, img_tiles), dtype=np.int64)
        else:
            self._decode_cross_attention_mask = None

        if "mm_token_type_ids" in lang_inputs:
            batch_size = (
                self.full_batch_size if self.full_batch_size is not None else lang_inputs["mm_token_type_ids"].shape[0]
            )
            self._decode_mm_token_type_ids = np.zeros((batch_size, 1), dtype=lang_inputs["mm_token_type_ids"].dtype)
        else:
            self._decode_mm_token_type_ids = None

        return outputs

    def run_prefill(self, prompt, generation_len, prefill_logit_bs=1, decode_batch_id=None):
        """
        Override base class prefill to handle vision processing

        Args:
            prompt: Can be string or tuple (image_path, text_prompt)
            generation_len: Generation length
            prefill_logit_bs: Prefill batch size
            decode_batch_id: Batch ID for continuous batching

        Returns:
            Same as base class: (outputs, position_ids, generation_len)
        """
        # Normalize prompt: TextGeneration passes a list even for batch_size=1
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], tuple) and len(prompt[0]) == 2:
            # Unwrap single (image_path, text_prompt) tuple
            if len(prompt) == 1:
                prompt = prompt[0]
            else:
                raise NotImplementedError(
                    "VisionLanguageGeneration.run_prefill currently supports a single (image, text) pair per call."
                )
        # Check if this is a vision-language prompt
        if isinstance(prompt, tuple) and len(prompt) == 2:
            image_path, text_prompt = prompt

            # Check cache for vision outputs
            cache_key = image_path if isinstance(image_path, str) else str(image_path)
            if cache_key in self._vision_cache:
                lang_inputs, vision_outputs, num_chunks = self._vision_cache[cache_key]
                logger.debug(f"Using cached vision outputs for {cache_key}")
            else:
                # Build language inputs with processor-aware vision/text integration
                lang_inputs, vision_outputs, num_chunks = self._vision_handler.get_processed_inputs(
                    image_url=image_path, query=text_prompt, prefill_seq_len=self._prefill_seq_len
                )
                # Cache for future use
                self._vision_cache[cache_key] = (lang_inputs, vision_outputs, num_chunks)
                logger.debug(f"Cached vision outputs for {cache_key}")

            # Set vision buffers in language session
            self._session.set_buffers(vision_outputs)
            logger.debug(f"Vision buffers set: {list(vision_outputs.keys())}")
            self._vision_processed = True
            self._vision_outputs = vision_outputs

            # Calculate generation_len consistent with ctx_len
            max_gen_len = self._ctx_len - np.where(lang_inputs["position_ids"] != -1, 1, 0).sum(1, keepdims=True).max()
            generation_len = self._fetch_generation_len(generation_len, max_gen_len)

            # Execute chunked prefill
            outputs = self._execute_chunked_prefill(
                lang_inputs, num_chunks, decode_batch_id, prefill_logit_bs, self.block_table
            )

            self._session.skip_buffers(vision_outputs)

            # Prepare position_ids for decode phase (next position after prefill)
            position_ids_decode = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1

            return outputs, position_ids_decode, generation_len
        else:
            # Fall back to base class for text-only
            return super().run_prefill(prompt, generation_len, prefill_logit_bs, decode_batch_id)

    def _prepare_vision_language_prompt(self, text_prompt, image_path):
        """
        Prepare text prompt with vision context

        This method handles the integration of vision and text inputs
        according to the specific model's requirements.
        """
        # For most vision-language models, we need to apply the chat template
        # that includes both image and text components
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image"},
                    ],
                },
            ]

            # Apply chat template
            processed_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            return processed_prompt

        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using original prompt.")
            return text_prompt

    def generate(
        self,
        images: List[str],
        prompts: List[str],
        inputs: torch.Tensor = None,
        num_frames: Optional[int] = None,
        multi_specs: Optional[bool] = None,
        generation_len: Optional[int] = None,
        stream: bool = True,
        **kwargs,
    ) -> CloudAI100ExecInfo:
        """
        Main generation method maintaining API compatibility with VisionLanguageGeneration

        Args:
            images: List of image URLs/paths
            prompts: List of text prompts
            generation_len: Max generation length
            stream: Enable streaming output
            **kwargs: Additional arguments passed to base class

        Returns:
            CloudAI100ExecInfo with results and metrics

        Raises:
            ValueError: If images and prompts lengths don't match
        """
        if num_frames or multi_specs:
            return self._generate_multi_frame_specialization(inputs, num_frames, generation_len)

        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")

        # Clear vision cache for fresh generation
        self._vision_cache.clear()

        logger.info(f"Generating for {len(images)} image-prompt pairs")

        # Convert to base class format: list of (image, prompt) tuples
        vision_prompts = [(img, prompt) for img, prompt in zip(images, prompts)]

        # Use base class generate method with vision prompts
        if self.full_batch_size is not None:
            # Continuous batching mode (new capability)
            return self._generate_continuous_batching(vision_prompts, generation_len, stream, **kwargs)
        else:
            # Regular batching mode
            return self._generate_regular_batching(vision_prompts, generation_len, stream, **kwargs)

    def run_prefill_multi_frame_specialization(
        self, inputs: Optional[torch.Tensor], num_frames: Optional[int] = 1, generation_len: int = None
    ):
        """
        Run prefill for multi-frame specialization. This is a special case where we have a fixed number of frames
        and we want to prefill the model with the input frames before generating the output.

        Args:
            inputs: Input tensor
            num_frames: Number of frames to process
            generation_len: Length of generation
            stream: Whether to stream the output

        Returns:
            Generation result
        """

        if not self._qpc_path:
            raise TypeError("Please run compile API for language model first!")

        self._session.deactivate()
        self._vision_session.activate()

        if not num_frames:
            logger.warning("num_frames not specified, defaulting to 1")
            num_frames = 1

        batch_size, ctx_len, fbs, _ = get_compilation_dims(self._qpc_path)

        pad_token_id = 1

        # Skip inputs/outputs
        self._session.skip_buffers(
            [
                x
                for x in self._session.input_names + self._session.output_names
                if x.startswith("past_") or x.endswith("_RetainedState")
            ]
        )

        prefill_seq_len = max(
            [x[self._session.binding_index_map["input_ids"]][1][1] for x in self._session.allowed_shapes]
            + [self._session.bindings[self._session.binding_index_map["input_ids"]].dims[1]]
        )

        input_len = inputs["attention_mask"].sum(1, keepdims=True)
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        if generation_len is None:
            generation_len = ctx_len - input_len.max()
        assert generation_len > 0, "generation length should be greater than zero"

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            pad_token_id,
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )

        for k, v in inputs.items():
            inputs[k] = np.array(v)

        vision_inputs = {
            k: v
            for k, v in inputs.items()
            if k
            in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
        }

        for k in constants.VISION_FP16_INPUTS:
            if k not in vision_inputs:
                continue
            if self._vision_session.binding_is_bfloat16(k):
                vision_inputs[k] = (
                    torch.from_numpy(vision_inputs[k]).to(torch.bfloat16).view(torch.int16).numpy().view(np.float16)
                )
            else:
                vision_inputs[k] = vision_inputs[k].astype(np.float16)

        vision_outputs = {}
        if vision_inputs:
            vision_size = vision_inputs["pixel_values"].shape[0] // num_frames

            pixel_values_shape = list(vision_inputs["pixel_values"][:vision_size].shape)

            idx = next(
                i for i, inner in enumerate(self._vision_session.allowed_shapes) if (2, pixel_values_shape) in inner
            )

            buffer_set = {
                "vision_embeds": np.zeros(
                    self._vision_session.allowed_shapes[idx][self._vision_session.binding_index_map["vision_embeds"]][
                        1
                    ],
                    dtype=np.float16,
                ),
                "image_grid_thw": np.zeros(
                    self._vision_session.allowed_shapes[idx][self._vision_session.binding_index_map["image_grid_thw"]][
                        1
                    ],
                    dtype=np.int64,
                ),
            }
            if "deepstack_features" in self._vision_session.binding_index_map:
                buffer_set["deepstack_features"] = np.zeros(
                    self._vision_session.allowed_shapes[idx][
                        self._vision_session.binding_index_map["deepstack_features"]
                    ][1],
                    dtype=np.float16,
                )

            self._vision_session.set_buffers(buffer_set)

            chunk_inputs = vision_inputs.copy()

            for i in range(num_frames):
                chunk_inputs["pixel_values"] = vision_inputs["pixel_values"][i * vision_size : (i + 1) * vision_size]
                chunk_outputs = self._vision_session.run(chunk_inputs)
                if i == 0:
                    vision_outputs = chunk_outputs
                else:
                    vision_outputs["vision_embeds"] = np.concatenate(
                        (vision_outputs["vision_embeds"], chunk_outputs["vision_embeds"]), axis=1
                    )

            vision_outputs["vision_embeds"] = np.pad(
                vision_outputs["vision_embeds"],
                pad_width=(
                    (0, 0),
                    (0, self._session.allowed_shapes[0][1][1][1] - vision_outputs["vision_embeds"].shape[-2]),
                    (0, 0),
                ),  # pad axis=1 only
                mode="constant",
                constant_values=0,
            )
            if "deepstack_features" in vision_outputs:
                vision_outputs["deepstack_features"] = np.pad(
                    vision_outputs["deepstack_features"],
                    pad_width=(
                        (0, 0),
                        (0, 0),
                        (0, self._session.allowed_shapes[0][1][1][1] - vision_outputs["deepstack_features"].shape[-2]),
                        (0, 0),
                    ),  # pad axis=1 only
                    mode="constant",
                    constant_values=0,
                )

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        lang_inputs.pop("attention_mask")

        if self._vision_qpc_path:
            self._vision_session.deactivate()

        self._session.activate()

        self._session.set_buffers(vision_outputs)
        logger.debug(f"Vision buffers set: {list(vision_outputs.keys())}")
        self._vision_processed = True
        self._vision_outputs = vision_outputs

        # Calculate generation_len consistent with ctx_len
        max_gen_len = self._ctx_len - np.where(lang_inputs["position_ids"] != -1, 1, 0).sum(1, keepdims=True).max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        # Execute chunked prefill
        outputs = self._execute_chunked_prefill(lang_inputs, num_chunks, block_table=self.block_table)

        self._session.skip_buffers(vision_outputs)

        # Prepare position_ids for decode phase (next position after prefill)
        position_ids_decode = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1

        return outputs, position_ids_decode, generation_len

    def _generate_multi_frame_specialization(
        self,
        inputs: Optional[torch.Tensor],
        num_frames: Optional[int] = 1,
        generation_len: int = None,
        stream: List[str] = None,
    ):
        exec_batch_size = self.batch_size
        max_gen_length = self._ctx_len if not generation_len else max(self._ctx_len, generation_len)
        self.initialize_decode_inputs(
            num_prompts=1, execution_batch_size=exec_batch_size, max_gen_length=max_gen_length
        )

        if self.is_qwen_vl:
            self.decode_pos_ids = np.zeros((4, exec_batch_size, 1), np.int64)

        # Prefill using VLM-aware run_prefill (batch is a list of (image, text))
        start = perf_counter()
        outputs, position_ids, generation_len_final = self.run_prefill_multi_frame_specialization(
            inputs, num_frames, generation_len
        )
        self.update_decode_input(outputs, position_ids, generation_len_final)

        # Prepare decode
        decode_inputs = self.prepare_decode_inputs()

        # Decode loop
        loop_start = perf_counter()
        num_token = self.run_decode(decode_inputs, generation_len_final, automation=False, streamer=None)
        end = perf_counter()

        # Decode generated texts
        generated_texts = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)

        # Latency metrics
        total_decode_tokens = num_token
        prefill_time, decode_perf, total_perf, total_time = calculate_latency(
            total_decode_tokens, loop_start, start, end
        )
        perf_metrics = PerfMetrics(prefill_time, decode_perf, total_perf, total_time)

        return CloudAI100ExecInfo(
            batch_size=self.batch_size,
            generated_texts=generated_texts,
            generated_ids=self.generated_ids,
            perf_metrics=perf_metrics,
        )

    def _generate_regular_batching(self, vision_prompts, generation_len, stream, **kwargs):
        """Handle regular batching for vision-language generation without creating a second language session"""
        batch_results = []
        for i in range(0, len(vision_prompts), self.batch_size):
            batch = vision_prompts[i : i + self.batch_size]

            if stream:
                print(
                    f"\nProcessing batch {i // self.batch_size + 1}/{(len(vision_prompts) - 1) // self.batch_size + 1}"
                )
                for j, (img, prompt) in enumerate(batch):
                    print(f"Image: {img}")
                    print(f"Prompt: {prompt}")
                    print("Completion:", flush=True, end="")

            # Setup decode storage arrays for this batch (use ctx_len or generation_len whichever is larger)
            exec_batch_size = self.batch_size
            max_gen_length = self._ctx_len if not generation_len else max(self._ctx_len, generation_len)
            self.initialize_decode_inputs(
                num_prompts=len(batch), execution_batch_size=exec_batch_size, max_gen_length=max_gen_length
            )

            # Prefill using VLM-aware run_prefill (batch is a list of (image, text))
            start = perf_counter()
            outputs, position_ids, generation_len_final = self.run_prefill(
                batch, generation_len, prefill_logit_bs=self.batch_size
            )
            self.update_decode_input(outputs, position_ids, generation_len_final)

            # Prepare decode
            decode_inputs = self.prepare_decode_inputs()

            # Decode loop
            loop_start = perf_counter()
            num_token = self.run_decode(decode_inputs, generation_len_final, automation=False, streamer=None)
            end = perf_counter()

            # Decode generated texts
            generated_texts = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)

            # Latency metrics
            total_decode_tokens = num_token
            prefill_time, decode_perf, total_perf, total_time = calculate_latency(
                total_decode_tokens, loop_start, start, end
            )
            perf_metrics = PerfMetrics(prefill_time, decode_perf, total_perf, total_time)

            # Package result for this batch
            batch_results.append(
                CloudAI100ExecInfo(
                    batch_size=self.batch_size,
                    generated_texts=generated_texts,
                    generated_ids=self.generated_ids,
                    perf_metrics=perf_metrics,
                )
            )

        # Aggregate results across batches
        return self._aggregate_batch_results(batch_results)

    def _generate_continuous_batching(self, vision_prompts, generation_len, stream, **kwargs):
        """Enable continuous batching for vision-language models (new capability)"""
        logger.info("Using continuous batching for vision-language generation")

        if stream:
            logger.warning("Streaming output not fully supported with continuous batching")

        # Reset vision processing state for new generation
        self._vision_processed = False
        self._vision_outputs = None
        self._vision_outputs_cache = {}

        # Initialize decode inputs
        num_prompts = len(vision_prompts)
        execution_batch_size = self.full_batch_size
        max_gen_length = self._ctx_len if not generation_len else max(self._ctx_len, generation_len)

        self.initialize_decode_inputs(num_prompts, execution_batch_size, max_gen_length)
        if self.is_qwen_vl:
            self.decode_pos_ids = np.zeros((4, execution_batch_size, 1), np.int64)
        # Create prompt queue
        prompt_queue = deque(vision_prompts)

        start = perf_counter()
        # Pre-process ALL vision inputs and cache them
        logger.info("Pre-processing all vision inputs...")
        for batch_id in range(min(self.full_batch_size, len(vision_prompts))):
            img, prompt = vision_prompts[batch_id]

            # Process vision for this slot
            lang_inputs, vision_outputs, num_chunks = self._vision_handler.get_processed_inputs(
                image_url=img, query=prompt, prefill_seq_len=self._prefill_seq_len
            )

            # Cache vision outputs for this batch slot
            self._vision_outputs_cache[batch_id] = {
                "vision_outputs": vision_outputs,
                "lang_inputs": lang_inputs,
                "num_chunks": num_chunks,
            }

            logger.debug(f"Cached vision outputs for batch_id {batch_id}")

        # Reset prompt queue for prefill
        prompt_queue = deque(vision_prompts)
        self.batch_index = None

        # Run prefill for all inputs using cached vision
        self.run_prefill_for_all_inputs_with_cached_vision(prompt_queue, generation_len)

        # Set vision buffers for decode (use first slot's vision for now)
        # For identical images, any slot's vision works
        cached_slot_0 = self._vision_outputs_cache.get(0)
        if cached_slot_0:
            self._session.set_buffers(cached_slot_0["vision_outputs"])
            logger.debug("Set vision buffers from slot 0 for decode phase")

        # Now set batch_index for decode phase
        self.batch_index = np.arange(self.full_batch_size).reshape(-1, 1)

        loop_start = perf_counter()
        decode_pause_time = self.run_continuous_batching_decode(prompt_queue, generation_len)
        end = perf_counter()

        generated_texts = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)

        total_decode_tokens = sum(
            np.sum(self.generated_ids[i] != self.tokenizer.pad_token_id) - 1 for i in range(len(vision_prompts))
        )
        prefill_time, decode_perf, total_perf, total_time = calculate_latency(
            total_decode_tokens, loop_start, start, end, decode_pause_time
        )
        prefill_time /= len(vision_prompts)  # Average prefill time for continuous batching

        perf_metrics = PerfMetrics(prefill_time, decode_perf, total_perf, total_time)

        return CloudAI100ExecInfo(
            batch_size=1, generated_texts=generated_texts, generated_ids=self.generated_ids, perf_metrics=perf_metrics
        )

    def run_prefill_for_all_inputs_with_cached_vision(self, prompt_queue, generation_len):
        """
        Runs prefill for all inputs using pre-cached vision outputs.

        This avoids the vision buffer overwriting issue by using cached vision
        outputs instead of processing vision during each prefill iteration.

        Args:
            prompt_queue (deque): The queue of prompts.
            generation_len (int): The generation length.
        """
        for decode_batch_id in range(self.full_batch_size):
            # Pop the promt as we are processing
            _ = prompt_queue.popleft()

            # Get cached vision outputs for this batch slot
            cached = self._vision_outputs_cache.get(decode_batch_id)
            if cached:
                vision_outputs = cached["vision_outputs"]
                lang_inputs = cached["lang_inputs"]
                num_chunks = cached["num_chunks"]

                # Set vision buffers for THIS prefill
                self._session.set_buffers(vision_outputs)
                logger.debug(f"Set vision buffers for batch_id {decode_batch_id} prefill")

                # Run prefill with cached inputs
                block_table = self.block_table[decode_batch_id].reshape(1, -1) if self.block_table is not None else None
                outputs = self._execute_chunked_prefill(
                    lang_inputs,
                    num_chunks,
                    decode_batch_id=np.array(decode_batch_id, dtype=np.int64).reshape(1, 1),
                    prefill_logit_bs=1,
                    block_table=block_table,
                )

                self._session.skip_buffers(vision_outputs.keys())

                # Calculate position_ids for decode
                position_ids_decode = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1

                # Calculate generation_len
                max_gen_len = (
                    self._ctx_len - np.where(lang_inputs["position_ids"] != -1, 1, 0).sum(1, keepdims=True).max()
                )
                generation_len_final = self._fetch_generation_len(generation_len, max_gen_len)

                # Update decode inputs
                if self.is_qwen_vl:
                    self.update_decode_inputs_qwen_vl(
                        outputs, position_ids_decode, generation_len_final, decode_batch_id
                    )
                else:
                    self.update_decode_input(outputs, position_ids_decode, generation_len_final, decode_batch_id)
            else:
                logger.error(f"No cached vision outputs for batch_id {decode_batch_id}")
                raise RuntimeError(f"Vision outputs not cached for batch_id {decode_batch_id}")

    def prepare_decode_inputs(self):
        """
        Override base class to handle vision-specific decode inputs
        """
        decode_inputs = super().prepare_decode_inputs()

        # Add image_idx for vision-language models in CB mode during decode only
        if self.batch_index is not None and hasattr(self, "_vision_outputs"):
            # image_idx should be a single slot selector; decoder expects shape (1,1)
            # Query binding dims if available to be robust
            try:
                if "image_idx" in getattr(self._session, "binding_index_map", {}):
                    idx = self._session.binding_index_map["image_idx"]
                    dims = tuple(self._session.bindings[idx].dims)
                    decode_inputs["image_idx"] = np.zeros(dims, dtype=np.int64)
                else:
                    decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
            except Exception:
                decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

        # Include cross_attention_mask during decode if present/required
        if hasattr(self, "_decode_cross_attention_mask") and self._decode_cross_attention_mask is not None:
            # Decoder specialization expects a single mask (batch dim = 1)
            decode_inputs["cross_attention_mask"] = self._decode_cross_attention_mask

        if hasattr(self, "_decode_mm_token_type_ids") and self._decode_mm_token_type_ids is not None:
            decode_inputs["mm_token_type_ids"] = self._decode_mm_token_type_ids

        return decode_inputs

    def _aggregate_batch_results(self, batch_results):
        """Aggregate results from multiple batches"""
        if not batch_results:
            raise ValueError("No batch results to aggregate")

        if len(batch_results) == 1:
            return batch_results[0]

        # Aggregate multiple batch results
        all_generated_texts = []
        all_generated_ids = []
        all_metrics = []

        for result in batch_results:
            if isinstance(result.generated_texts[0], list):
                # Flatten nested lists
                all_generated_texts.extend([text for batch in result.generated_texts for text in batch])
            else:
                all_generated_texts.extend(result.generated_texts)

            if isinstance(result.generated_ids, list):
                all_generated_ids.extend(result.generated_ids)
            else:
                all_generated_ids.append(result.generated_ids)

            all_metrics.append(result.perf_metrics)

        # Average metrics
        avg_metrics = PerfMetrics(
            prefill_time=np.mean([m.prefill_time for m in all_metrics]),
            decode_perf=np.mean([m.decode_perf for m in all_metrics]),
            total_perf=np.mean([m.total_perf for m in all_metrics]),
            total_time=np.mean([m.total_time for m in all_metrics]),
        )

        return CloudAI100ExecInfo(
            batch_size=batch_results[0].batch_size,
            generated_texts=all_generated_texts,
            generated_ids=all_generated_ids,
            perf_metrics=avg_metrics,
        )

    def generate_stream_tokens(
        self, images: List[str], prompts: List[str], generation_len: Optional[int] = None, **kwargs
    ):
        """
        Enable token-by-token streaming for vision models (new capability)

        Args:
            images: List of image URLs/paths
            prompts: List of text prompts
            generation_len: Max generation length
            **kwargs: Additional arguments

        Yields:
            List of decoded tokens for each batch position

        Raises:
            NotImplementedError: If continuous batching is enabled
        """
        if self.full_batch_size is not None:
            raise NotImplementedError("Token streaming not supported with continuous batching for VLM")

        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")

        logger.info(f"Starting token streaming for {len(images)} image-prompt pairs")

        vision_prompts = [(img, prompt) for img, prompt in zip(images, prompts)]

        text_gen = TextGeneration(
            tokenizer=self.tokenizer,
            qpc_path=self._qpc_path,
            ctx_len=self._ctx_len,
            device_ids=self.device_ids,
            enable_debug_logs=self.enable_debug_logs,
            is_tlm=self.is_tlm,
            include_sampler=self.include_sampler,
            return_pdfs=self.return_pdfs,
            include_guided_decoding=self.include_guided_decoding,
            sampling_params=self.sampling_params,
        )

        text_gen._qaic_model = self

        # Yield tokens as they're generated
        for tokens in text_gen.generate_stream_tokens(vision_prompts, generation_len, **kwargs):
            yield tokens

    def __repr__(self):
        """String representation of the class"""
        return (
            f"VisionLanguageGeneration("
            f"batch_size={self.batch_size}, "
            f"ctx_len={self._ctx_len}, "
            f"continuous_batching={'enabled' if self.full_batch_size else 'disabled'}, "
            f"sampling={'enabled' if self.include_sampler else 'disabled'})"
        )
