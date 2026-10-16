# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

# TODO: Pipeline Architecture Improvements
# 1. Introduce QEffDiffusionPipeline base class to provide unified export, compile,
#    and inference APIs across all diffusion pipelines, promoting code reusability
#    and consistent interface design.
# 2. Implement persistent QPC session management strategy to retain/drop compiled model
#    sessions in memory across all pipeline modules.

import os
import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from tqdm import tqdm

from QEfficient.diffusers.first_block_cache.flux import enable_flux_first_block_cache
from QEfficient.diffusers.pipelines.pipeline_module import (
    QEffFluxTransformerModel,
    QEffTextEncoder,
    QEffVAE,
)
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    ModulePerf,
    QEffPipelineOutput,
    calculate_compressed_latent_dimension,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_execute_params,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import QEFFLogger, log_pipeline_api

logger = QEFFLogger.get_logger("MODEL")


class QEffFluxPipeline:
    """
    QEfficient-optimized Flux pipeline for high-performance text-to-image generation on Qualcomm AI hardware.

    This pipeline provides an optimized implementation of the Flux diffusion model specifically designed
    for deployment on Qualcomm AI Cloud (QAIC) devices. It wraps the original HuggingFace Flux model
    components with QEfficient-optimized versions that can be exported to ONNX format and compiled
    into Qualcomm Program Container (QPC) files for efficient inference.

    The pipeline supports the complete Flux workflow including:
    - Dual text encoding with CLIP and T5 encoders
    - Transformer-based denoising with adaptive layer normalization
    - VAE decoding for final image generation
    - Performance monitoring and optimization

    Attributes:
        text_encoder (QEffTextEncoder): Optimized CLIP text encoder for pooled embeddings
        text_encoder_2 (QEffTextEncoder): Optimized T5 text encoder for sequence embeddings
        transformer (QEffFluxTransformerModel): Optimized Flux transformer for denoising
        vae_decode (QEffVAE): Optimized VAE decoder for latent-to-image conversion
        modules (Dict[str, Any]): Dictionary of all pipeline modules for batch operations
        model (FluxPipeline): Original HuggingFace Flux model reference
        tokenizer: CLIP tokenizer for text preprocessing
        scheduler: Diffusion scheduler for timestep management

    Example:
        >>> from QEfficient.diffusers.pipelines.flux import QEffFluxPipeline
        >>> pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
        >>> images = pipeline(
        ...     prompt="A beautiful sunset over mountains",
        ...     height=512,
        ...     width=512,
        ...     num_inference_steps=28
        ... )
        >>> images.images[0].save("generated_image.png")
    """

    _hf_auto_class = FluxPipeline

    def __init__(
        self,
        model,
        enable_first_block_cache: bool = False,
        first_block_cache_downsample_factor: int = 4,
        *args,
        **kwargs,
    ):
        """
        Initialize the QEfficient Flux pipeline.

        This pipeline provides an optimized implementation of the Flux text-to-image model
        for deployment on Qualcomm AI hardware. It wraps the original HuggingFace Flux model
        components with QEfficient-optimized versions that can be exported to ONNX and compiled
        for QAIC devices.

        Args:
            model: Pre-loaded FluxPipeline model
            enable_first_block_cache (bool): Enable retained-state first-block-cache path.
            first_block_cache_downsample_factor (int): Downsample factor for the first-block
                residual cache key. Used only when first-block-cache is enabled.
            **kwargs: Additional arguments including height and width
        """

        # Wrap model components with QEfficient optimized versions
        self.model = model
        self.enable_first_block_cache = enable_first_block_cache
        self.first_block_cache_downsample_factor = first_block_cache_downsample_factor
        self.text_encoder = QEffTextEncoder(model.text_encoder)
        self.text_encoder_2 = QEffTextEncoder(model.text_encoder_2)
        self.transformer = QEffFluxTransformerModel(model.transformer)
        if self.enable_first_block_cache:
            enable_flux_first_block_cache(
                self.transformer,
                downsample_factor=self.first_block_cache_downsample_factor,
            )
        self.vae_decode = QEffVAE(model.vae, "decoder")

        # Store all modules in a dictionary for easy iteration during export/compile
        self.modules = {
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "transformer": self.transformer,
            "vae_decoder": self.vae_decode,
        }

        # Copy tokenizers and scheduler from the original model
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.text_encoder_2.tokenizer = model.tokenizer_2
        self.tokenizer_max_length = model.tokenizer_max_length
        self.scheduler = model.scheduler

        # Override VAE forward method to use decode directly
        self.vae_decode.model.forward = lambda latent_sample, return_dict: self.vae_decode.model.decode(
            latent_sample, return_dict
        )

        # Sync max position embeddings between text encoders
        self.text_encoder_2.model.config.max_position_embeddings = (
            self.text_encoder.model.config.max_position_embeddings
        )

    @classmethod
    @log_pipeline_api("from_pretrained", "Diffusion model loading completed.", "load_complete")
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        enable_first_block_cache: bool = False,
        first_block_cache_downsample_factor: int = 4,
        **kwargs,
    ):
        """
        Load a pretrained Flux model from HuggingFace Hub or local path and wrap it with QEfficient optimizations.

        This class method provides a convenient way to instantiate a QEffFluxPipeline from a pretrained
        Flux model. It automatically loads the base FluxPipeline model in float32 precision on CPU
        and wraps all components with QEfficient-optimized versions for QAIC deployment.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): Either a HuggingFace model identifier
                (e.g., "black-forest-labs/FLUX.1-schnell") or a local path to a saved model directory.
            enable_first_block_cache (bool, optional): Enables retained-state first-block-cache path.
            first_block_cache_downsample_factor (int, optional): Downsample factor for the first-block
                residual cache key when cache is enabled.
            **kwargs: Additional keyword arguments passed to FluxPipeline.from_pretrained().

        Returns:
            QEffFluxPipeline: A fully initialized pipeline instance with QEfficient-optimized components
                ready for export, compilation, and inference on QAIC devices.

        Raises:
            ValueError: If the model path is invalid or model cannot be loaded
            OSError: If there are issues accessing the model files
            RuntimeError: If model initialization fails

        Example:
            >>> # Load from HuggingFace Hub
            >>> pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
            >>>
            >>> # Load from local path
            >>> pipeline = QEffFluxPipeline.from_pretrained("/path/to/local/flux/model")
            >>>
            >>> # Load with custom cache directory
            >>> pipeline = QEffFluxPipeline.from_pretrained(
            ...     "black-forest-labs/FLUX.1-dev",
            ...     cache_dir="/custom/cache/dir"
            ... )
        """
        # Load the base Flux model in float32 on CPU
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            **kwargs,
        )

        return cls(
            model=model,
            enable_first_block_cache=enable_first_block_cache,
            first_block_cache_downsample_factor=first_block_cache_downsample_factor,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )

    @log_pipeline_api("export", "ONNX export completed.", "export_complete")
    def export(self, export_dir: Optional[str] = None, use_onnx_subfunctions: bool = False) -> str:
        """
        Export all pipeline modules to ONNX format for deployment preparation.

        This method systematically exports each pipeline component (CLIP text encoder, T5 text encoder,
        Flux transformer, and VAE decoder) to ONNX format. Each module is exported with its specific
        configuration including dynamic axes, input/output specifications, and optimization settings.

        The export process prepares the models for subsequent compilation to QPC format, enabling
        efficient inference on QAIC hardware. ONNX subfunctions can be used for certain modules
        to optimize memory usage and performance.

        Args:
            export_dir (str, optional): Target directory for saving ONNX model files. If None,
                uses the default export directory structure based on model name and configuration.
                The directory will be created if it doesn't exist.
            use_onnx_subfunctions (bool, default=False): Whether to enable ONNX subfunction
                optimization for supported modules. This can optimize thegraph and
                improve compilation efficiency for models like the transformer.

        Returns:
            str: Absolute path to the export directory containing all ONNX model files.
                Each module will have its own subdirectory with the exported ONNX file.

        Raises:
            RuntimeError: If ONNX export fails for any module
            OSError: If there are issues creating the export directory or writing files
            ValueError: If module configurations are invalid

        Note:
            - All models are exported in float32 precision for maximum compatibility
            - Dynamic axes are configured to support variable batch sizes and sequence lengths
            - The export process may take several minutes depending on model size
            - Exported ONNX files can be large (several GB for complete pipeline)

        Example:
            >>> pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
            >>> export_path = pipeline.export(
            ...     export_dir="/path/to/export",
            ...     use_onnx_subfunctions=True
            ... )
            >>> print(f"Models exported to: {export_path}")
        """
        for module_name, module_obj in tqdm(self.modules.items(), desc="Exporting modules", unit="module"):
            # Get ONNX export configuration for this module
            example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params()

            export_params = {
                "inputs": example_inputs,
                "output_names": output_names,
                "dynamic_axes": dynamic_axes,
                "export_dir": export_dir,
            }

            if use_onnx_subfunctions and module_name in ONNX_SUBFUNCTION_MODULE:
                export_params["use_onnx_subfunctions"] = True

            if module_obj.qpc_path is None:
                module_obj.export(**export_params)

    @staticmethod
    def get_default_config_path() -> str:
        """
        Get the absolute path to the default Flux pipeline configuration file.

        Returns:
            str: Absolute path to the flux_config.json file containing default pipeline
                configuration settings for compilation and device allocation.
        """
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/flux_config.json")

    @log_pipeline_api("compile", "Compilation completed.", "compile_complete")
    def compile(
        self,
        compile_config: Optional[str] = None,
        parallel: bool = False,
        height: int = 512,
        width: int = 512,
        use_onnx_subfunctions: bool = False,
    ) -> None:
        """
        Compile ONNX models into optimized QPC format for deployment on Qualcomm AI hardware.

        Args:
            compile_config (str, optional): Path to a JSON configuration file containing
                compilation settings, device mappings, and optimization parameters. If None,
                uses the default configuration from get_default_config_path().
            parallel (bool, default=False): Compilation mode selection:
                - True: Compile modules in parallel using ThreadPoolExecutor for faster processing
                - False: Compile modules sequentially for lower resource usage
            height (int, default=512): Target image height in pixels.
            width (int, default=512): Target image width in pixels.
            use_onnx_subfunctions (bool, default=False): Whether to export models with ONNX
                subfunctions before compilation.

        Raises:
            RuntimeError: If compilation fails for any module or if QAIC compiler is not available
            FileNotFoundError: If ONNX models haven't been exported or config file is missing
            ValueError: If configuration parameters are invalid
            OSError: If there are issues with file I/O during compilation

        Example:
            >>> pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
            >>> # Sequential compilation with default config
            >>> pipeline.compile(height=1024, width=1024)
            >>>
            >>> # Parallel compilation with custom config
            >>> pipeline.compile(
            ...     compile_config="/path/to/custom_config.json",
            ...     parallel=True,
            ...     height=512,
            ...     width=512
            ... )
        """
        # Load compilation configuration
        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)

        # Set device IDs, qpc path if precompiled qpc exist
        set_execute_params(self)

        # Ensure all modules are exported to ONNX before compilation
        if any(
            path is None
            for path in [
                self.text_encoder.onnx_path,
                self.text_encoder_2.onnx_path,
                self.transformer.onnx_path,
                self.vae_decode.onnx_path,
            ]
        ):
            self.export(use_onnx_subfunctions=use_onnx_subfunctions)

        # Calculate compressed latent dimension using utility function
        cl, latent_height, latent_width = calculate_compressed_latent_dimension(
            height, width, self.model.vae_scale_factor
        )

        # Prepare dynamic specialization updates based on image dimensions
        specialization_updates = {
            "transformer": {"cl": cl},
            "vae_decoder": {
                "latent_height": latent_height,
                "latent_width": latent_width,
            },
        }

        # Use generic utility functions for compilation
        if parallel:
            compile_modules_parallel(self.modules, self.custom_config, specialization_updates)
        else:
            compile_modules_sequential(self.modules, self.custom_config, specialization_updates)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device_ids: Optional[List[int]] = None,
    ):
        """
        Encode text prompts using the T5 text encoder for detailed semantic understanding.

        T5 provides rich sequence embeddings that capture fine-grained text details,
        complementing CLIP's global representation in Flux's dual encoder setup.

        Args:
            prompt (str or List[str]): Input prompt(s) to encode
            num_images_per_prompt (int): Number of images to generate per prompt
            max_sequence_length (int): Maximum token sequence length (default: 512)
            device_ids (List[int], optional): QAIC device IDs for inference

        Returns:
            tuple: (prompt_embeds, inference_time)
                - prompt_embeds (torch.Tensor): Encoded embeddings [batch*num_images, seq_len, 4096]
                - inference_time (float): T5 encoder inference time in seconds
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Tokenize prompts with padding and truncation
        text_inputs = self.text_encoder_2.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        # Check for truncation and warn user
        untruncated_ids = self.text_encoder_2.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.text_encoder_2.tokenizer.batch_decode(
                untruncated_ids[:, self.text_encoder_2.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                f"The following part of your input was truncated because `max_sequence_length` is set to "
                f"{self.text_encoder_2.tokenizer.model_max_length} tokens: {removed_text}"
            )

        # Initialize QAIC inference session if not already created
        if self.text_encoder_2.qpc_session is None:
            self.text_encoder_2.qpc_session = QAICInferenceSession(
                str(self.text_encoder_2.qpc_path), device_ids=device_ids
            )

        # Allocate output buffers for QAIC inference
        text_encoder_2_output = {
            "last_hidden_state": np.random.rand(
                batch_size, max_sequence_length, self.text_encoder_2.model.config.d_model
            ).astype(np.int32),
        }
        self.text_encoder_2.qpc_session.set_buffers(text_encoder_2_output)

        # Prepare input for QAIC inference
        aic_text_input = {"input_ids": text_input_ids.numpy().astype(np.int64)}

        # Run T5 encoder inference and measure time
        start_t5_time = time.perf_counter()
        prompt_embeds = torch.tensor(self.text_encoder_2.qpc_session.run(aic_text_input)["last_hidden_state"])
        end_t5_time = time.perf_counter()
        text_encoder_2_perf = end_t5_time - start_t5_time

        # Duplicate embeddings for multiple images per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, text_encoder_2_perf

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device_ids: Optional[List[int]] = None,
    ):
        """
        Encode text prompts using the CLIP text encoder for global semantic representation.

        CLIP provides pooled embeddings that capture high-level semantic meaning,
        working alongside T5's detailed sequence embeddings in Flux's dual encoder setup.

        Args:
            prompt (str or List[str]): Input prompt(s) to encode
            num_images_per_prompt (int): Number of images to generate per prompt
            device_ids (List[int], optional): QAIC device IDs for inference

        Returns:
            tuple: (pooled_prompt_embeds, inference_time)
                - pooled_prompt_embeds (torch.Tensor): Pooled embeddings [batch*num_images, 768]
                - inference_time (float): CLIP encoder inference time in seconds
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Tokenize prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids

        # Check for truncation and warn user
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                f"The following part of your input was truncated because CLIP can only handle sequences up to "
                f"{self.tokenizer_max_length} tokens: {removed_text}"
            )

        # Initialize QAIC inference session if not already created
        if self.text_encoder.qpc_session is None:
            self.text_encoder.qpc_session = QAICInferenceSession(str(self.text_encoder.qpc_path), device_ids=device_ids)

        # Allocate output buffers for QAIC inference
        text_encoder_output = {
            "last_hidden_state": np.random.rand(
                batch_size, self.tokenizer_max_length, self.text_encoder.model.config.hidden_size
            ).astype(np.float32),
            "pooler_output": np.random.rand(batch_size, self.text_encoder.model.config.hidden_size).astype(np.int32),
        }
        self.text_encoder.qpc_session.set_buffers(text_encoder_output)

        # Prepare input for QAIC inference
        aic_text_input = {"input_ids": text_input_ids.numpy().astype(np.int64)}

        # Run CLIP encoder inference and measure time
        start_text_encoder_time = time.perf_counter()
        aic_embeddings = self.text_encoder.qpc_session.run(aic_text_input)
        end_text_encoder_time = time.perf_counter()
        text_encoder_perf = end_text_encoder_time - start_text_encoder_time
        # Extract pooled output (used for conditioning in Flux)
        prompt_embeds = torch.tensor(aic_embeddings["pooler_output"])

        # Duplicate embeddings for multiple images per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, text_encoder_perf

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        """
        Encode text prompts using Flux's dual text encoder architecture.

        Flux employs both CLIP and T5 encoders for comprehensive text understanding:
        - CLIP provides pooled embeddings for global semantic conditioning
        - T5 provides detailed sequence embeddings for fine-grained text control

        Args:
            prompt (str or List[str]): Primary prompt(s) for both encoders
            prompt_2 (str or List[str], optional): Secondary prompt(s) for T5. If None, uses primary prompt
            num_images_per_prompt (int): Number of images to generate per prompt
            prompt_embeds (torch.FloatTensor, optional): Pre-computed T5 embeddings
            pooled_prompt_embeds (torch.FloatTensor, optional): Pre-computed CLIP pooled embeddings
            max_sequence_length (int): Maximum sequence length for T5 tokenization

        Returns:
            tuple: (prompt_embeds, pooled_prompt_embeds, text_ids, encoder_perf_times)
                - prompt_embeds (torch.Tensor): T5 sequence embeddings [batch*num_images, seq_len, 4096]
                - pooled_prompt_embeds (torch.Tensor): CLIP pooled embeddings [batch*num_images, 768]
                - text_ids (torch.Tensor): Position IDs for text tokens [seq_len, 3]
                - encoder_perf_times (List[float]): Performance times [CLIP_time, T5_time]
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_encoder_perf = 0.0
        text_encoder_2_perf = 0.0

        if prompt_embeds is None:
            # Use primary prompt for both encoders if secondary not provided
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # Encode with CLIP (returns pooled embeddings)
            pooled_prompt_embeds, text_encoder_perf = self._get_clip_prompt_embeds(
                prompt=prompt,
                device_ids=self.text_encoder.device_ids,
                num_images_per_prompt=num_images_per_prompt,
            )

            # Encode with T5 (returns sequence embeddings)
            prompt_embeds, text_encoder_2_perf = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device_ids=self.text_encoder_2.device_ids,
            )

        # Create text position IDs (required by Flux transformer)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3)

        return prompt_embeds, pooled_prompt_embeds, text_ids, [text_encoder_perf, text_encoder_2_perf]

    @log_pipeline_api(
        "generate",
        "Generation completed.",
        "generation_complete",
        start_run=True,
        finish_run=True,
    )
    def __call__(
        self,
        height: int = 512,
        width: int = 512,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        custom_config_path: Optional[str] = None,
        parallel_compile: bool = False,
        use_onnx_subfunctions: bool = False,
        cache_threshold: Optional[float] = None,
    ):
        """
        Generate images from text prompts using the QEfficient-optimized Flux pipeline on QAIC hardware.

        This is the main entry point for text-to-image generation. It orchestrates the complete Flux
        diffusion pipeline optimized for Qualcomm AI Cloud devices.

        Args:
            height (int, optional): Target image height in pixels. Must be divisible by 8. Default: 512.
            width (int, optional): Target image width in pixels. Must be divisible by 8. Default: 512.
            prompt (str or List[str]): Primary text prompt(s) describing the desired image(s).
                Required unless `prompt_embeds` is provided.
            prompt_2 (str or List[str], optional): Secondary prompt for T5 encoder. If None, uses `prompt`.
            negative_prompt (str or List[str], optional): Negative prompt(s) describing what to avoid.
                Only used when `true_cfg_scale > 1.0`.
            negative_prompt_2 (str or List[str], optional): Secondary negative prompt for T5. If None, uses `negative_prompt`.
            true_cfg_scale (float, optional): True classifier-free guidance scale. Values > 1.0 enable
                negative prompting. Default: 1.0 (disabled).
            num_inference_steps (int, optional): Number of denoising steps. Default: 28.
            timesteps (List[int], optional): Custom timestep schedule. If provided, overrides `num_inference_steps`.
            guidance_scale (float, optional): Guidance scale for classifier-free guidance. Default: 3.5.
            num_images_per_prompt (int, optional): Number of images to generate per prompt. Default: 1.
            generator (torch.Generator or List[torch.Generator], optional): Random generator for reproducibility.
            latents (torch.FloatTensor, optional): Pre-generated latent tensors. If None, random latents are generated.
            prompt_embeds (torch.FloatTensor, optional): Pre-computed T5 text embeddings. Shape: [batch, seq_len, 4096].
            pooled_prompt_embeds (torch.FloatTensor, optional): Pre-computed CLIP pooled embeddings. Shape: [batch, 768].
            negative_prompt_embeds (torch.FloatTensor, optional): Pre-computed negative T5 embeddings.
            negative_pooled_prompt_embeds (torch.FloatTensor, optional): Pre-computed negative CLIP embeddings.
            output_type (str, optional): Output format. Options: "pil" (default), "np", or "latent".
            callback_on_step_end (Callable, optional): Callback function executed after each denoising step.
            callback_on_step_end_tensor_inputs (List[str], optional): Tensor names to pass to callback. Default: ["latents"].
            max_sequence_length (int, optional): Maximum token sequence length for T5 encoder. Default: 512.
            custom_config_path (str, optional): Path to custom JSON configuration file for compilation settings.
            parallel_compile (bool, optional): Whether to compile modules in parallel. Default: False.
            use_onnx_subfunctions (bool, optional): Whether to export transformer blocks as ONNX subfunctions. Default: False.
            cache_threshold (float, optional): First-block-cache threshold.
                Used only when `enable_first_block_cache=True`.

        Returns:
            QEffPipelineOutput: A dataclass containing:
                - images: Generated image(s) in the format specified by `output_type`
                - pipeline_module: Performance metrics for each pipeline component (text encoders, transformer, VAE)

        Raises:
            ValueError: If input validation fails or parameters are incompatible.
            RuntimeError: If compilation fails or QAIC devices are unavailable.
            FileNotFoundError: If custom config file is specified but not found.

        Example:
            >>> from QEfficient.diffusers.pipelines.flux import QEffFluxPipeline
            >>> pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
            >>> result = pipeline(
            ...     prompt="A serene mountain landscape at sunset",
            ...     height=1024,
            ...     width=1024,
            ...     num_inference_steps=28,
            ...     guidance_scale=7.5
            ... )
            >>> result.images[0].save("mountain_sunset.png")
            >>> print(f"Transformer inference time: {sum(result.pipeline_module[2].perf):.2f}s")
        """
        device = self.model._execution_device

        if height is None or width is None:
            logger.warning("Height or width is None. Setting default values of 512 for both dimensions.")

        self.compile(
            compile_config=custom_config_path,
            parallel=parallel_compile,
            height=height,
            width=width,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

        # Validate all inputs
        self.model.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False
        if not self.enable_first_block_cache and cache_threshold is not None:
            logger.warning(
                "Ignoring cache_threshold because first-block-cache is disabled. "
                "Set `enable_first_block_cache=True` to enable it."
            )

        # Step 2: Determine batch size from inputs
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Step 3: Encode prompts with both text encoders
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        (prompt_embeds, pooled_prompt_embeds, text_ids, text_encoder_perf) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # Encode negative prompts if using true classifier-free guidance
        if do_true_cfg:
            (negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids, text_encoder_perf_2) = (
                self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )
            )

        # Step 4: Prepare timesteps for denoising
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Step 5: Prepare initial latents
        num_channels_latents = self.transformer.model.config.in_channels // 4
        latents, latent_image_ids = self.model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Step 6: Calculate compressed latent dimension for transformer buffer allocation
        cl, _, _ = calculate_compressed_latent_dimension(height, width, self.model.vae_scale_factor)

        # Deactivate text encoder sessions to free device resources before loading transformer
        if self.text_encoder.qpc_session is not None:
            self.text_encoder.qpc_session.deactivate()
        if self.text_encoder_2.qpc_session is not None:
            self.text_encoder_2.qpc_session.deactivate()

        # Initialize transformer inference session
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(
                str(self.transformer.qpc_path), device_ids=self.transformer.device_ids
            )

        # Allocate output buffer for transformer
        output_buffer = {
            "output": np.random.rand(batch_size, cl, self.transformer.model.config.in_channels).astype(np.float32),
        }
        self.transformer.qpc_session.set_buffers(output_buffer)
        if self.enable_first_block_cache:
            self.transformer.qpc_session.skip_buffers(
                [
                    tensor_name
                    for tensor_name in (
                        self.transformer.qpc_session.input_names + self.transformer.qpc_session.output_names
                    )
                    if tensor_name.startswith("prev_") or tensor_name.endswith("_RetainedState")
                ]
            )

        transformer_perf = []
        self.scheduler.set_begin_index(0)

        # Step 7: Denoising loop
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Prepare timestep embedding
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                temb = self.transformer.model.time_text_embed(timestep, pooled_prompt_embeds)

                # Compute AdaLN (Adaptive Layer Normalization) embeddings for dual transformer blocks
                adaln_emb = []
                for block_idx in range(len(self.transformer.model.transformer_blocks)):
                    block = self.transformer.model.transformer_blocks[block_idx]
                    # Process through norm1 and norm1_context
                    f1 = block.norm1.linear(block.norm1.silu(temb)).chunk(6, dim=1)
                    f2 = block.norm1_context.linear(block.norm1_context.silu(temb)).chunk(6, dim=1)
                    adaln_emb.append(torch.cat(list(f1) + list(f2)))
                adaln_dual_emb = torch.stack(adaln_emb)

                # Compute AdaLN embeddings for single transformer blocks
                adaln_emb = []
                for block_idx in range(len(self.transformer.model.single_transformer_blocks)):
                    block = self.transformer.model.single_transformer_blocks[block_idx]
                    f1 = block.norm.linear(block.norm.silu(temb)).chunk(3, dim=1)
                    adaln_emb.append(torch.cat(list(f1)))
                adaln_single_emb = torch.stack(adaln_emb)

                # Compute output AdaLN embedding
                temp = self.transformer.model.norm_out
                adaln_out = temp.linear(temp.silu(temb))

                # Normalize timestep to [0, 1] range
                timestep = timestep / 1000

                # Prepare all inputs for transformer inference
                inputs_aic = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": prompt_embeds.detach().numpy(),
                    "pooled_projections": pooled_prompt_embeds.detach().numpy(),
                    "timestep": timestep.detach().numpy(),
                    "img_ids": latent_image_ids.detach().numpy(),
                    "txt_ids": text_ids.detach().numpy(),
                    "adaln_emb": adaln_dual_emb.detach().numpy(),
                    "adaln_single_emb": adaln_single_emb.detach().numpy(),
                    "adaln_out": adaln_out.detach().numpy(),
                }
                if self.enable_first_block_cache:
                    stage_cache_threshold = 0.0 if cache_threshold is None else cache_threshold
                    inputs_aic["cache_threshold"] = np.array(stage_cache_threshold, dtype=np.float32)

                # Run transformer inference and measure time
                start_transformer_step_time = time.perf_counter()
                outputs = self.transformer.qpc_session.run(inputs_aic)
                end_transformer_step_time = time.perf_counter()
                print("Time taken", end_transformer_step_time - start_transformer_step_time)
                transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

                noise_pred = torch.from_numpy(outputs["output"])

                # Update latents using scheduler (x_t -> x_t-1)
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Handle dtype mismatch (workaround for MPS backend bug)
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # Execute callback if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Step 8: Decode latents to images (unless output_type is "latent")
        if output_type == "latent":
            image = latents
        else:
            # Unpack and denormalize latents
            latents = self.model._unpack_latents(latents, height, width, self.model.vae_scale_factor)
            latents = (latents / self.vae_decode.model.scaling_factor) + self.vae_decode.model.shift_factor

            # Deactivate transformer session to free device resources before loading VAE decoder
            if self.transformer.qpc_session is not None:
                self.transformer.qpc_session.deactivate()

            # Initialize VAE decoder inference session
            if self.vae_decode.qpc_session is None:
                self.vae_decode.qpc_session = QAICInferenceSession(
                    str(self.vae_decode.qpc_path), device_ids=self.vae_decode.device_ids
                )

            # Allocate output buffer for VAE decoder
            output_buffer = {"sample": np.random.rand(batch_size, 3, height, width).astype(np.int32)}
            self.vae_decode.qpc_session.set_buffers(output_buffer)

            # Run VAE decoder inference and measure time
            inputs = {"latent_sample": latents.numpy()}
            start_decode_time = time.perf_counter()
            image = self.vae_decode.qpc_session.run(inputs)
            end_decode_time = time.perf_counter()
            vae_decode_perf = end_decode_time - start_decode_time

            # Post-process image
            image_tensor = torch.from_numpy(image["sample"])
            image = self.model.image_processor.postprocess(image_tensor, output_type=output_type)

            # Build performance metrics
            perf_metrics = [
                ModulePerf(module_name="text_encoder", perf=text_encoder_perf[0]),
                ModulePerf(module_name="text_encoder_2", perf=text_encoder_perf[1]),
                ModulePerf(module_name="transformer", perf=transformer_perf),
                ModulePerf(module_name="vae_decoder", perf=vae_decode_perf),
            ]

            return QEffPipelineOutput(
                pipeline_module=perf_metrics,
                images=image,
            )
