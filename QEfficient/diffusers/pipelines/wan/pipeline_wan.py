# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
QEfficient WAN Pipeline Implementation

This module provides an optimized implementation of the WAN pipeline
for high-performance text-to-video generation on Qualcomm AI hardware.
The pipeline supports WAN 2.2 architectures in:
1) unified mode (single transformer module with stage routing), and
2) non-unified mode (separate high/low transformer modules).

TODO: 1. Update umt5 to Qaic; present running on cpu
"""

import os
import time
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import WanPipeline
from tqdm import tqdm

from QEfficient.diffusers.first_block_cache.wan import (
    enable_wan_first_block_cache,
    run_wan_non_unified_first_block_cache_denoise,
)
from QEfficient.diffusers.models.transformers.transformer_wan import QEffWanUnifiedWrapper
from QEfficient.diffusers.pipelines.pipeline_module import QEffVAE, QEffWanTransformer, QEffWanUnifiedTransformer
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    ModulePerf,
    QEffPipelineOutput,
    calculate_latent_dimensions_with_frames,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_execute_params,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import QEFFLogger, log_pipeline_api

logger = QEFFLogger.get_logger("MODEL")


class QEffWanPipeline:
    """
    QEfficient-optimized WAN pipeline for high-performance text-to-video generation on Qualcomm AI hardware.

    This pipeline provides an optimized implementation of the WAN diffusion model
    specifically designed for deployment on Qualcomm AI Cloud (QAIC) devices. It extends the original
    HuggingFace WAN model with QEfficient-optimized components that can be exported to ONNX format
    and compiled into Qualcomm Program Container (QPC) files for efficient video generation.

    The pipeline supports the complete WAN workflow including:
    - UMT5 text encoding for rich semantic understanding
    - Unified transformer architecture: Combines multiple transformer stages into a single optimized model
    - VAE decoding for final video output
    - Performance monitoring and hardware optimization

    Attributes:
        text_encoder: UMT5 text encoder for semantic text understanding (TODO: QEfficient optimization)
        unified_wrapper (QEffWanUnifiedWrapper): Wrapper combining transformer stages (unified mode)
        transformer (QEffWanUnifiedTransformer): Optimized unified transformer for denoising (unified mode)
        transformer_high (QEffWanTransformer): High-noise transformer module (non-unified mode)
        transformer_low (QEffWanTransformer): Low-noise transformer module (non-unified mode)
        vae_decode: VAE decoder for latent-to-video conversion
        modules (Dict[str, Any]): Dictionary of pipeline modules for batch operations
        model (WanPipeline): Original HuggingFace WAN model reference
        tokenizer: Text tokenizer for preprocessing
        scheduler: Diffusion scheduler for timestep management

    Example:
        >>> from QEfficient.diffusers.pipelines.wan import QEffWanPipeline
        >>> pipeline = QEffWanPipeline.from_pretrained("path/to/wan/model")
        >>> videos = pipeline(
        ...     prompt="A cat playing in a garden",
        ...     height=480,
        ...     width=832,
        ...     num_frames=81,
        ...     num_inference_steps=4
        ... )
        >>> # Save generated video
        >>> videos.images[0].save("generated_video.mp4")
    """

    _hf_auto_class = WanPipeline

    def __init__(
        self,
        model,
        use_unified: bool = True,
        enable_first_block_cache: bool = False,
        first_block_cache_downsample_factor: int = 4,
        **kwargs,
    ):
        """
        Initialize the QEfficient WAN pipeline.

        This pipeline provides an optimized implementation of the WAN text-to-video model
        for deployment on Qualcomm AI hardware. It wraps the original HuggingFace WAN model
        components with QEfficient-optimized versions that can be exported to ONNX and compiled
        for QAIC devices.

        Args:
            model: Pre-loaded WanPipeline model with transformer and transformer_2 components
            use_unified (bool): If True, use a unified transformer module that internally
                selects high/low stage by `tsp`. If False, keep high/low transformers as
                separate compiled modules and dispatch explicitly at runtime.
            enable_first_block_cache (bool): Enable retained-state first-block-cache path.
                Supported only for non-unified mode.
            first_block_cache_downsample_factor (int): Downsample factor for the first-block
                residual cache key. Used only when first-block-cache is enabled.
            **kwargs: Additional keyword arguments including configuration parameters
        """
        # Store original model and configuration
        self.model = model
        self.use_unified = use_unified
        self.enable_first_block_cache = enable_first_block_cache
        self.first_block_cache_downsample_factor = first_block_cache_downsample_factor
        self.kwargs = kwargs
        self.custom_config = None

        if self.enable_first_block_cache and self.use_unified:
            raise ValueError("First-block-cache is currently supported only for non-unified WAN (`use_unified=False`).")

        # Text encoder (TODO: Replace with QEfficient UMT5 optimization)
        self.text_encoder = model.text_encoder

        # Build transformer modules based on selected architecture mode.
        if self.use_unified:
            # Unified mode: one wrapper containing both stages.
            self.unified_wrapper = QEffWanUnifiedWrapper(model.transformer, model.transformer_2)
            self.transformer = QEffWanUnifiedTransformer(self.unified_wrapper)
            self.modules = {"transformer": self.transformer}
            self._denoise_impl = self._run_denoise_loop_unified
        else:
            # Non-unified mode: independent high/low modules.
            self.unified_wrapper = None
            self.transformer_high = QEffWanTransformer(model.transformer, module_name="transformer_high")
            self.transformer_low = QEffWanTransformer(model.transformer_2, module_name="transformer_low")
            if self.enable_first_block_cache:
                enable_wan_first_block_cache(
                    self.transformer_high,
                    downsample_factor=self.first_block_cache_downsample_factor,
                )
                enable_wan_first_block_cache(
                    self.transformer_low,
                    downsample_factor=self.first_block_cache_downsample_factor,
                )
            self.modules = {
                "transformer_high": self.transformer_high,
                "transformer_low": self.transformer_low,
            }
            self._denoise_impl = (
                partial(run_wan_non_unified_first_block_cache_denoise, self)
                if self.enable_first_block_cache
                else self._run_denoise_loop_non_unified
            )
            # Keep a lightweight compatibility handle for existing scripts that access
            # `pipeline.transformer.model.transformer_high/low` to attach LoRA adapters.
            self.transformer = SimpleNamespace(
                model=SimpleNamespace(
                    transformer_high=self.transformer_high.model,
                    transformer_low=self.transformer_low.model,
                )
            )

        # VAE decoder for latent-to-video conversion
        self.vae_decoder = QEffVAE(model.vae, "decoder")
        # TODO: add text encoder on QAIC
        self.modules["vae_decoder"] = self.vae_decoder

        # Copy tokenizers and scheduler from the original model
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.scheduler = model.scheduler

        self.vae_decoder.model.forward = lambda latent_sample, return_dict: self.vae_decoder.model.decode(
            latent_sample, return_dict
        )

        self.vae_decoder.get_onnx_params = self.vae_decoder.get_video_onnx_params
        # Extract patch dimensions from transformer configuration
        _, self.patch_height, self.patch_width = model.transformer.config.patch_size

    @property
    def do_classifier_free_guidance(self):
        """
        Determine if classifier-free guidance should be used.

        Returns:
            bool: True if CFG should be applied based on current guidance scales
        """
        return self._guidance_scale > 1.0 and (self._guidance_scale_2 is None or self._guidance_scale_2 > 1.0)

    @classmethod
    @log_pipeline_api("from_pretrained", "Diffusion model loading completed.", "load_complete")
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        use_unified: bool = True,
        enable_first_block_cache: bool = False,
        first_block_cache_downsample_factor: int = 4,
        **kwargs,
    ):
        """
        Load a pretrained WAN model from HuggingFace Hub or local path and wrap it with QEfficient optimizations.

        This class method provides a convenient way to instantiate a QEffWanPipeline from a pretrained
        WAN model. It automatically loads the base WanPipeline model in float32 precision on CPU
        and wraps all components with QEfficient-optimized versions for QAIC deployment.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): Either a HuggingFace model identifier
                or a local path to a saved WAN model directory. Should contain transformer, transformer_2,
                text_encoder, and VAE components.
            use_unified (bool, optional): Selects WAN execution architecture.
                - True: unified high/low transformer module
                - False: separate high and low transformer modules
            enable_first_block_cache (bool, optional): Enables retained-state first-block-cache
                for non-unified mode.
            first_block_cache_downsample_factor (int, optional): Downsample factor for first-block
                cache key when cache is enabled.
            **kwargs: Additional keyword arguments passed to WanPipeline.from_pretrained().

        Returns:
            QEffWanPipeline: A fully initialized pipeline instance with QEfficient-optimized components
                ready for export, compilation, and inference on QAIC devices.

        Raises:
            ValueError: If the model path is invalid or model cannot be loaded
            OSError: If there are issues accessing the model files
            RuntimeError: If model initialization fails

        Example:
            >>> # Load from HuggingFace Hub
            >>> pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            >>>
            >>> # Load from local path
            >>> pipeline = QEffWanPipeline.from_pretrained("/local/path/to/wan")
            >>>
            >>> # Load with custom cache directory
            >>> pipeline = QEffWanPipeline.from_pretrained(
            ...     "wan-model-id",
            ...     cache_dir="/custom/cache/dir"
            ... )
        """
        # Load the base WAN model in float32 on CPU for optimization
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            **kwargs,
        )
        return cls(
            model=model,
            use_unified=use_unified,
            enable_first_block_cache=enable_first_block_cache,
            first_block_cache_downsample_factor=first_block_cache_downsample_factor,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )

    @log_pipeline_api("export", "ONNX export completed.", "export_complete")
    def export(
        self,
        export_dir: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """
        Export all pipeline modules to ONNX format for deployment preparation.

        This method systematically exports the unified transformer to ONNX format with
        video-specific configurations including temporal dimensions, dynamic axes, and
        optimization settings. The export process prepares the model for subsequent
        compilation to QPC format for efficient inference on QAIC hardware.

        Args:
            export_dir (str, optional): Target directory for saving ONNX model files. If None,
                uses the default export directory structure. The directory will be created
                if it doesn't exist.
            use_onnx_subfunctions (bool, default=False): Whether to enable ONNX subfunction
                optimization for supported modules. This can optimize the graph structure
                and improve compilation efficiency for complex models like the transformer.

        Returns:
            str: Absolute path to the export directory containing all ONNX model files.

        Raises:
            RuntimeError: If ONNX export fails for any module
            OSError: If there are issues creating the export directory or writing files
            ValueError: If module configurations are invalid

        Example:
            >>> pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            >>> export_path = pipeline.export(
            ...     export_dir="/path/to/export",
            ...     use_onnx_subfunctions=True
            ... )
        """

        # Export each module with video-specific parameters
        for module_name, module_obj in tqdm(self.modules.items(), desc="Exporting modules", unit="module"):
            # Get ONNX export configuration with video dimensions
            example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params()

            # Prepare export parameters
            export_params = {
                "inputs": example_inputs,
                "output_names": output_names,
                "dynamic_axes": dynamic_axes,
                "export_dir": export_dir,
            }

            # Enable ONNX subfunctions for supported modules if requested
            if use_onnx_subfunctions and module_name in ONNX_SUBFUNCTION_MODULE:
                export_params["use_onnx_subfunctions"] = True

            if module_obj.qpc_path is None:
                module_obj.export(**export_params)

    def get_default_config_path(self):
        """
        Get the default configuration file path for WAN pipeline.

        Returns:
            str: Path to the default WAN configuration JSON file.
        """
        config_name = "wan_config.json" if self.use_unified else "wan_non_unified_config.json"
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), f"configs/{config_name}")

    @log_pipeline_api("compile", "Compilation completed.", "compile_complete")
    def compile(
        self,
        compile_config: Optional[str] = None,
        parallel: bool = False,
        height: int = constants.WAN_ONNX_EXPORT_HEIGHT_45P,
        width: int = constants.WAN_ONNX_EXPORT_WIDTH_45P,
        num_frames: int = constants.WAN_ONNX_EXPORT_FRAMES,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """
        Compiles the ONNX graphs of the different model components for deployment on Qualcomm AI hardware.

        This method takes the ONNX paths of the transformer and compiles them into an optimized format
        for inference using JSON-based configuration.

        Args:
            compile_config (str, optional): Path to a JSON configuration file containing
                compilation settings, device mappings, and optimization parameters. If None,
                uses the default configuration.
            parallel (bool, default=False): Compilation mode selection:
                - True: Compile modules in parallel using ThreadPoolExecutor for faster processing
                - False: Compile modules sequentially for lower resource usage
            height (int, default=192): Target image height in pixels.
            width (int, default=320): Target image width in pixels.
            num_frames (int, deafult=81) : Target num of frames in pixel space
            use_onnx_subfunctions (bool, default=False): Whether to export models with ONNX
                subfunctions before compilation if not already exported.

        Raises:
            RuntimeError: If compilation fails for any module or if QAIC compiler is not available
            FileNotFoundError: If ONNX models haven't been exported or config file is missing
            ValueError: If configuration parameters are invalid
            OSError: If there are issues with file I/O during compilation

        Example:
            >>> pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            >>> # Sequential compilation with default config
            >>> pipeline.compile(height=480, width=832, num_frames=81)
            >>>
            >>> # Parallel compilation with custom config
            >>> pipeline.compile(
            ...     compile_config="/path/to/custom_config.json",
            ...     parallel=True,
            ...     height=480,
            ...     width=832,
            ...     num_frames=81
            ... )
        """
        # Load compilation configuration
        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)

        # Set device IDs, qpc path if precompiled qpc exist
        set_execute_params(self)

        # Ensure all modules are exported to ONNX before compilation
        onnx_paths = [module.onnx_path for module in self.modules.values()]
        if any(path is None for path in onnx_paths):
            self.export(use_onnx_subfunctions=use_onnx_subfunctions)

        # Configure pipeline dimensions and calculate compressed latent parameters
        cl, latent_height, latent_width, latent_frames = calculate_latent_dimensions_with_frames(
            height,
            width,
            num_frames,
            self.model.vae.config.scale_factor_spatial,
            self.model.vae.config.scale_factor_temporal,
            self.patch_height,
            self.patch_width,
        )
        if self.use_unified:
            # Unified mode: one transformer module with two model_type specializations.
            specialization_updates = {
                "transformer": [
                    {
                        "cl": cl,
                        "latent_height": latent_height,
                        "latent_width": latent_width,
                        "latent_frames": latent_frames,
                    },
                    {
                        "cl": cl,
                        "latent_height": latent_height,
                        "latent_width": latent_width,
                        "latent_frames": latent_frames,
                    },
                ],
                "vae_decoder": {
                    "latent_frames": latent_frames,
                    "latent_height": latent_height,
                    "latent_width": latent_width,
                },
            }
        else:
            # Non-unified mode: independent high/low modules.
            shared_transformer_spec = {
                "cl": cl,
                "latent_height": latent_height,
                "latent_width": latent_width,
                "latent_frames": latent_frames,
            }
            specialization_updates = {
                "transformer_high": shared_transformer_spec.copy(),
                "transformer_low": shared_transformer_spec.copy(),
                "vae_decoder": {
                    "latent_frames": latent_frames,
                    "latent_height": latent_height,
                    "latent_width": latent_width,
                },
            }

        # Use generic utility functions for compilation
        if parallel:
            compile_modules_parallel(self.modules, self.custom_config, specialization_updates)
        else:
            compile_modules_sequential(self.modules, self.custom_config, specialization_updates)

    def _get_transformer_dtype(self) -> torch.dtype:
        if self.use_unified:
            return self.transformer.model.transformer_high.dtype
        return self.transformer_high.model.dtype

    def _setup_transformer_session(self, module_obj, batch_size: int, cl: int) -> None:
        if module_obj.qpc_session is None:
            module_obj.qpc_session = QAICInferenceSession(str(module_obj.qpc_path), device_ids=module_obj.device_ids)
        output_buffer = {
            "output": np.random.rand(
                batch_size,
                cl,
                constants.WAN_DIT_OUT_CHANNELS,
            ).astype(np.int32),
        }
        module_obj.qpc_session.set_buffers(output_buffer)
        if getattr(module_obj, "_qeff_first_block_cache_enabled", False):
            module_obj.qpc_session.skip_buffers(
                [
                    tensor_name
                    for tensor_name in (module_obj.qpc_session.input_names + module_obj.qpc_session.output_names)
                    if tensor_name.startswith("prev_") or tensor_name.endswith("_RetainedState")
                ]
            )

    def _prepare_transformer_sessions(self, batch_size: int, cl: int) -> None:
        if self.use_unified:
            self._setup_transformer_session(self.transformer, batch_size, cl)
        else:
            self._setup_transformer_session(self.transformer_high, batch_size, cl)
            self._setup_transformer_session(self.transformer_low, batch_size, cl)

    @staticmethod
    def _reshape_noise_prediction(
        outputs: Dict[str, np.ndarray],
        batch_size: int,
        post_patch_num_frames: int,
        post_patch_height: int,
        post_patch_width: int,
        p_t: int,
        p_h: int,
        p_w: int,
    ) -> torch.Tensor:
        hidden_states = torch.tensor(outputs["output"])
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _run_denoise_loop_unified(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int,
        guidance_scale: float,
        guidance_scale_2: float,
        boundary_timestep: Optional[float],
        transformer_dtype: torch.dtype,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        mask: torch.Tensor,
        num_inference_steps: int,
        num_warmup_steps: int,
        callback_on_step_end: Optional[Callable],
        callback_on_step_end_tensor_inputs: List[str],
        cache_threshold_high: Optional[float] = None,
        cache_threshold_low: Optional[float] = None,
    ):
        transformer_perf = []
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer.model.transformer_high
                    current_guidance_scale = guidance_scale
                    model_type = torch.ones(1, dtype=torch.int64)
                else:
                    current_model = self.transformer.model.transformer_low
                    current_guidance_scale = guidance_scale_2
                    model_type = torch.ones(2, dtype=torch.int64)

                latent_model_input = latents.to(transformer_dtype)
                if self.model.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                _, _, latent_frames, latent_height, latent_width = latents.shape
                p_t, p_h, p_w = current_model.config.patch_size
                post_patch_num_frames = latent_frames // p_t
                post_patch_height = latent_height // p_h
                post_patch_width = latent_width // p_w

                rotary_emb = current_model.rope(latent_model_input)
                rotary_emb = torch.cat(rotary_emb, dim=0)
                timestep = timestep.flatten()

                temb, timestep_proj, encoder_hidden_states, _ = current_model.condition_embedder(
                    timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=None
                )

                if self.do_classifier_free_guidance:
                    _, _, encoder_hidden_states_neg, _ = current_model.condition_embedder(
                        timestep,
                        negative_prompt_embeds,
                        encoder_hidden_states_image=None,
                        timestep_seq_len=None,
                    )

                timestep_proj = timestep_proj.unflatten(1, (6, -1))
                inputs_aic = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy(),
                    "tsp": model_type.detach().numpy(),
                }

                if self.do_classifier_free_guidance:
                    inputs_aic2 = {
                        "hidden_states": latents.detach().numpy(),
                        "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                        "rotary_emb": rotary_emb.detach().numpy(),
                        "temb": temb.detach().numpy(),
                        "timestep_proj": timestep_proj.detach().numpy(),
                        "tsp": model_type.detach().numpy(),
                    }

                with current_model.cache_context("cond"):
                    start_transformer_step_time = time.perf_counter()
                    outputs = self.transformer.qpc_session.run(inputs_aic)
                    end_transformer_step_time = time.perf_counter()
                    transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                    noise_pred = self._reshape_noise_prediction(
                        outputs,
                        batch_size,
                        post_patch_num_frames,
                        post_patch_height,
                        post_patch_width,
                        p_t,
                        p_h,
                        p_w,
                    )

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        start_transformer_step_time = time.perf_counter()
                        outputs = self.transformer.qpc_session.run(inputs_aic2)
                        end_transformer_step_time = time.perf_counter()
                        transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                        noise_uncond = self._reshape_noise_prediction(
                            outputs,
                            batch_size,
                            post_patch_num_frames,
                            post_patch_height,
                            post_patch_width,
                            p_t,
                            p_h,
                            p_w,
                        )
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        return latents, transformer_perf

    def _run_denoise_loop_non_unified(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int,
        guidance_scale: float,
        guidance_scale_2: float,
        boundary_timestep: Optional[float],
        transformer_dtype: torch.dtype,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        mask: torch.Tensor,
        num_inference_steps: int,
        num_warmup_steps: int,
        callback_on_step_end: Optional[Callable],
        callback_on_step_end_tensor_inputs: List[str],
        cache_threshold_high: Optional[float] = None,
        cache_threshold_low: Optional[float] = None,
    ):
        transformer_perf = []
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    current_transformer_module = self.transformer_high
                    current_guidance_scale = guidance_scale
                else:
                    current_transformer_module = self.transformer_low
                    current_guidance_scale = guidance_scale_2
                current_model = current_transformer_module.model

                latent_model_input = latents.to(transformer_dtype)
                if self.model.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                _, _, latent_frames, latent_height, latent_width = latents.shape
                p_t, p_h, p_w = current_model.config.patch_size
                post_patch_num_frames = latent_frames // p_t
                post_patch_height = latent_height // p_h
                post_patch_width = latent_width // p_w

                rotary_emb = current_model.rope(latent_model_input)
                rotary_emb = torch.cat(rotary_emb, dim=0)
                timestep = timestep.flatten()

                temb, timestep_proj, encoder_hidden_states, _ = current_model.condition_embedder(
                    timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=None
                )

                if self.do_classifier_free_guidance:
                    _, _, encoder_hidden_states_neg, _ = current_model.condition_embedder(
                        timestep,
                        negative_prompt_embeds,
                        encoder_hidden_states_image=None,
                        timestep_seq_len=None,
                    )

                timestep_proj = timestep_proj.unflatten(1, (6, -1))
                inputs_aic = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy(),
                }

                if self.do_classifier_free_guidance:
                    inputs_aic2 = {
                        "hidden_states": latents.detach().numpy(),
                        "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                        "rotary_emb": rotary_emb.detach().numpy(),
                        "temb": temb.detach().numpy(),
                        "timestep_proj": timestep_proj.detach().numpy(),
                    }

                with current_model.cache_context("cond"):
                    start_transformer_step_time = time.perf_counter()
                    outputs = current_transformer_module.qpc_session.run(inputs_aic)
                    end_transformer_step_time = time.perf_counter()
                    transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                    noise_pred = self._reshape_noise_prediction(
                        outputs,
                        batch_size,
                        post_patch_num_frames,
                        post_patch_height,
                        post_patch_width,
                        p_t,
                        p_h,
                        p_w,
                    )

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        start_transformer_step_time = time.perf_counter()
                        outputs = current_transformer_module.qpc_session.run(inputs_aic2)
                        end_transformer_step_time = time.perf_counter()
                        transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                        noise_uncond = self._reshape_noise_prediction(
                            outputs,
                            batch_size,
                            post_patch_num_frames,
                            post_patch_height,
                            post_patch_width,
                            p_t,
                            p_h,
                            p_w,
                        )
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        return latents, transformer_perf

    @log_pipeline_api(
        "generate",
        "Generation completed.",
        "generation_complete",
        start_run=True,
        finish_run=True,
    )
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None]]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        custom_config_path: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        cache_threshold_high: Optional[float] = None,
        cache_threshold_low: Optional[float] = None,
        parallel_compile: bool = True,
    ):
        """
        Generate videos from text prompts using the QEfficient-optimized WAN pipeline on QAIC hardware.

        This is the main entry point for text-to-video generation. It orchestrates the complete WAN
        diffusion pipeline optimized for Qualcomm AI Cloud devices.

        Args:
            prompt (str or List[str]): Primary text prompt(s) describing the desired video content.
                Required unless `prompt_embeds` is provided.
            negative_prompt (str or List[str], optional): Negative prompt(s) describing what to avoid
                in the generated video. Used with classifier-free guidance.
            height (int, optional): Target video height in pixels. Must be divisible by VAE scale factor.
                Default: 480.
            width (int, optional): Target video width in pixels. Must be divisible by VAE scale factor.
                Default: 832.
            num_frames (int, optional): Number of video frames to generate. Must satisfy temporal
                divisibility requirements. Default: 81.
            num_inference_steps (int, optional): Number of denoising steps. More steps generally
                improve quality but increase generation time. Default: 50.
            guidance_scale (float, optional): Guidance scale for classifier-free guidance. Default: 3.0.
            guidance_scale_2 (float, optional): Guidance scale for low-noise stage in WAN 2.2.
                If None, uses guidance_scale value.
            num_videos_per_prompt (int, optional): Number of videos to generate per prompt. Default: 1.
            generator (torch.Generator or List[torch.Generator], optional): Random generator for
                reproducible generation.
            latents (torch.Tensor, optional): Pre-generated latent tensors. If None, random latents
                are generated based on video dimensions.
            prompt_embeds (torch.Tensor, optional): Pre-computed text embeddings from UMT5 encoder.
                Shape: [batch, seq_len, hidden_dim].
            negative_prompt_embeds (torch.Tensor, optional): Pre-computed negative text embeddings.
            output_type (str, optional): Output format. Options: "np" (default), "pil", or "latent".
            return_dict (bool, optional): Whether to return a dictionary or tuple. Default: True.
            attention_kwargs (Dict[str, Any], optional): Additional attention arguments for transformer.
            callback_on_step_end (Callable, optional): Callback function executed after each denoising step.
            callback_on_step_end_tensor_inputs (List[str], optional): Tensor names to pass to callback.
                Default: ["latents"].
            max_sequence_length (int, optional): Maximum token sequence length for text encoder. Default: 512.
            custom_config_path (str, optional): Path to custom JSON configuration file for compilation.
            use_onnx_subfunctions (bool, optional): Whether to export transformer blocks as ONNX subfunctions.
                Default: False.
            cache_threshold_high (float, optional): First-block-cache threshold for high-noise stage.
                Used only when `enable_first_block_cache=True`.
            cache_threshold_low (float, optional): First-block-cache threshold for low-noise stage.
                Used only when `enable_first_block_cache=True`.
            parallel_compile (bool, optional): Whether to compile modules in parallel. Default: True.

        Returns:
            QEffPipelineOutput: A dataclass containing:
                - images: Generated video(s) in the format specified by `output_type`
                - pipeline_module: Performance metrics for each pipeline component

        Raises:
            ValueError: If input validation fails or parameters are incompatible
            RuntimeError: If compilation fails or QAIC devices are unavailable
            FileNotFoundError: If custom config file is specified but not found

        Example:
            >>> from QEfficient.diffusers.pipelines.wan import QEffWanPipeline
            >>> pipeline = QEffWanPipeline.from_pretrained("path/to/wan/model")
            >>> result = pipeline(
            ...     prompt="A cat playing in a sunny garden",
            ...     height=480,
            ...     width=832,
            ...     num_frames=81,
            ...     num_inference_steps=4,
            ...     guidance_scale=3.0
            ... )
            >>> # Save generated video
            >>> result.images[0].save("cat_garden.mp4")
        """
        device = self.model._execution_device

        # Compile models with custom configuration if needed
        self.compile(
            compile_config=custom_config_path,
            parallel=parallel_compile,
            use_onnx_subfunctions=use_onnx_subfunctions,
            height=height,
            width=width,
            num_frames=num_frames,
        )

        # Step 1: Validate all inputs
        self.model.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        # Ensure num_frames satisfies temporal divisibility requirements
        if num_frames % self.model.vae.config.scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.model.vae.config.scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames // self.model.vae.config.scale_factor_temporal * self.model.vae.config.scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)

        if self.model.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        if not self.enable_first_block_cache and (cache_threshold_high is not None or cache_threshold_low is not None):
            logger.warning(
                "Ignoring cache thresholds because first-block-cache is disabled. "
                "Set `enable_first_block_cache=True` and `use_unified=False` to enable it."
            )

        # Initialize pipeline state
        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # Step 2: Determine batch size from inputs
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Step 3: Encode input prompts using UMT5 text encoder
        # TODO: Update UMT5 on QAIC
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Convert embeddings to transformer dtype for compatibility
        transformer_dtype = self._get_transformer_dtype()
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Step 4: Prepare timesteps for denoising process
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Step 5: Prepare initial latent variables for video generation
        num_channels_latents = self.model.transformer.config.in_channels

        latents = self.model.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Create mask for temporal processing (used in expand_timesteps mode)
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # Step 6: Configure dual-stage processing for WAN 2.2
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Calculate boundary timestep for stage switching in WAN 2.2
        if self.model.config.boundary_ratio is not None:
            boundary_timestep = self.model.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # Step 7: Initialize transformer sessions and buffers
        cl, _, _, _ = calculate_latent_dimensions_with_frames(
            height,
            width,
            num_frames,
            self.model.vae.config.scale_factor_spatial,
            self.model.vae.config.scale_factor_temporal,
            self.patch_height,
            self.patch_width,
        )
        self._prepare_transformer_sessions(batch_size, cl)
        latents, transformer_perf = self._denoise_impl(
            latents=latents,
            timesteps=timesteps,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            guidance_scale_2=self._guidance_scale_2,
            boundary_timestep=boundary_timestep,
            transformer_dtype=transformer_dtype,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            mask=mask,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            cache_threshold_high=cache_threshold_high,
            cache_threshold_low=cache_threshold_low,
        )

        self._current_timestep = None

        # Step 9: Decode latents to video
        vae_decoder_perf = 0.0
        if not output_type == "latent":
            # Prepare latents for VAE decoding
            latents = latents.to(self.vae_decoder.model.dtype)

            # Apply VAE normalization (denormalization)
            latents_mean = (
                torch.tensor(self.vae_decoder.model.config.latents_mean)
                .view(1, self.vae_decoder.model.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_decoder.model.config.latents_std).view(
                1, self.vae_decoder.model.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            # Initialize VAE decoder inference session
            if self.vae_decoder.qpc_session is None:
                self.vae_decoder.qpc_session = QAICInferenceSession(
                    str(self.vae_decoder.qpc_path), device_ids=self.vae_decoder.device_ids
                )

            # Allocate output buffer for VAE decoder
            output_buffer = {"sample": np.random.rand(batch_size, 3, num_frames, height, width).astype(np.int32)}
            self.vae_decoder.qpc_session.set_buffers(output_buffer)
            inputs = {"latent_sample": latents.numpy()}

            start_decode_time = time.perf_counter()
            video = self.vae_decoder.qpc_session.run(inputs)
            end_decode_time = time.perf_counter()
            vae_decoder_perf = end_decode_time - start_decode_time

            # Post-process video for output
            video_tensor = torch.from_numpy(video["sample"])
            video = self.model.video_processor.postprocess_video(video_tensor)
        else:
            video = latents

        # Step 10: Collect performance metrics
        perf_data = {
            "transformer": transformer_perf,
            "vae_decoder": vae_decoder_perf,
        }

        # Build performance metrics for output
        perf_metrics = [ModulePerf(module_name=name, perf=perf_data[name]) for name in perf_data.keys()]

        return QEffPipelineOutput(
            pipeline_module=perf_metrics,
            images=video,
        )
