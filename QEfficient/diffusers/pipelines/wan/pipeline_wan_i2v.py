# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
QEfficient WAN Image-to-Video Pipeline Implementation

This module provides an optimized implementation of the WAN image-to-video pipeline
for high-performance image-to-video generation on Qualcomm AI hardware.
The pipeline supports WAN 2.2 architectures with unified transformer for converting
static images into dynamic video sequences with temporal consistency.

TODO: 1. Update Umt5 to Qaic; present running on cpu
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from QEfficient.diffusers.models.transformers.transformer_wan import QEffWanUnifiedWrapper
from QEfficient.diffusers.pipelines.pipeline_module import QEffVAE, QEffWanUnifiedTransformer
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    ModulePerf,
    QEffPipelineOutput,
    calculate_latent_dimensions_with_frames,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_execute_params,
    update_npi_path,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import QEFFLogger, log_pipeline_api

logger = QEFFLogger.get_logger("MODEL")


class QEffWanImageToVideoPipeline:
    """
    QEfficient-optimized WAN image-to-video pipeline for high-performance video generation on Qualcomm AI hardware.

    This pipeline provides an optimized implementation of the WAN image-to-video diffusion model
    specifically designed for deployment on Qualcomm AI Cloud (QAIC) devices. It extends the original
    HuggingFace WAN image-to-video model with QEfficient-optimized components that can be exported to ONNX format
    and compiled into Qualcomm Program Container (QPC) files for efficient video generation from static images.

    The pipeline supports the complete WAN image-to-video workflow including:
    - Image conditioning and preprocessing for temporal consistency
    - UMT5 text encoding for rich semantic understanding
    - Unified transformer architecture: Combines multiple transformer stages into a single optimized model
    - VAE encoding/decoding for image-to-latent and latent-to-video conversion

    Attributes:
        text_encoder: UMT5 text encoder for semantic text understanding (TODO: QEfficient optimization)
        vae_encoder (QEffVAE): VAE encoder for converting input images to latent space
        unified_wrapper (QEffWanUnifiedWrapper): Wrapper combining transformer stages
        transformer (QEffWanUnifiedTransformer): Optimized unified transformer for denoising
        vae_decoder (QEffVAE): VAE decoder for latent-to-video conversion
        modules (Dict[str, Any]): Dictionary of pipeline modules for batch operations
        model (WanImageToVideoPipeline): Original HuggingFace WAN I2V model reference
        tokenizer: Text tokenizer for preprocessing
        scheduler: Diffusion scheduler for timestep management

    Example:
            >>> from QEfficient.diffusers.pipelines.wan import QEffWanImageToVideoPipeline
            >>> from PIL import Image
            >>>
            >>> # Load pipeline and input image
            >>> pipeline = QEffWanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            >>> image = Image.open("input_frame.jpg")
            >>>
            >>> # Generate video with motion
            >>> result = pipeline(
            ...     image=image,
            ...     prompt="A person walking through a sunny garden with flowing motion",
            ...     height=544,
            ...     width=720,
            ...     num_frames=81,
            ...     num_inference_steps=4,
            ...     guidance_scale=1.0
            ... )
            >>> # Save generated video
            >>> frames = result.images[0]
            >>> export_to_video(frames, "generated_video.mp4", fps=16)
    """

    _hf_auto_class = WanImageToVideoPipeline

    def __init__(self, model, **kwargs):
        """
        Initialize the QEfficient WAN image-to-video pipeline.

        This pipeline provides an optimized implementation of the WAN image-to-video model
        for deployment on Qualcomm AI hardware. It wraps the original HuggingFace WAN I2V model
        components with QEfficient-optimized versions that can be exported to ONNX and compiled
        for QAIC devices.

        Args:
            model (WanImageToVideoPipeline): Pre-loaded WanImageToVideoPipeline model with
                transformer, transformer_2, VAE, and text encoder components
            **kwargs: Additional keyword arguments including configuration parameters
        """
        # Wrap model components with QEfficient optimized versions
        self.model = model
        self.custom_config = None

        # Text encoder (TODO: Replace with QEfficient UMT5 optimization)
        self.text_encoder = model.text_encoder
        # Create unified transformer wrapper combining dual-stage models(high, low noise DiTs)
        self.unified_wrapper = QEffWanUnifiedWrapper(model.transformer, model.transformer_2)
        self.transformer = QEffWanUnifiedTransformer(self.unified_wrapper)
        # VAE encoder for image-to-latent conversion
        self.vae_encoder = QEffVAE(model.vae, "encoder")
        # VAE decoder for latent-to-video conversion
        self.vae_decoder = QEffVAE(model.vae, "decoder")

        # Store all modules in a dictionary for easy iteration during export/compile
        self.modules = {
            "vae_encoder": self.vae_encoder,
            "transformer": self.transformer,
            "vae_decoder": self.vae_decoder,
        }

        # Copy tokenizers and scheduler from the original model
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.scheduler = model.scheduler

        self.vae_encoder.get_onnx_params = self.vae_encoder.get_img_encoder_onnx_params
        self.vae_decoder.get_onnx_params = self.vae_decoder.get_video_onnx_params

        # Extract patch dimensions from transformer configuration
        _, self.patch_height, self.patch_width = self.transformer.model.config.patch_size

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
        **kwargs,
    ):
        """
        Load a pretrained WAN image-to-video model from HuggingFace Hub or local path and wrap it with QEfficient optimizations.

        This class method provides a convenient way to instantiate a QEffWanImageToVideoPipeline from a pretrained
        WAN I2V model. It automatically loads the base WanImageToVideoPipeline model in float32 precision on CPU
        and wraps all components with QEfficient-optimized versions for QAIC deployment.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): Either a HuggingFace model identifier
                or a local path to a saved WAN I2V model directory. Should contain transformer, transformer_2,
                text_encoder, and VAE components optimized for image-to-video generation.
            **kwargs: Additional keyword arguments passed to WanImageToVideoPipeline.from_pretrained().

        Returns:
            QEffWanImageToVideoPipeline: A fully initialized I2V pipeline instance with QEfficient-optimized components
                ready for export, compilation, and inference on QAIC devices.

        Raises:
            ValueError: If the model path is invalid or model cannot be loaded
            OSError: If there are issues accessing the model files
            RuntimeError: If model initialization fails

        Example:
            >>> # Load from HuggingFace Hub
            >>> pipeline = QEffWanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            >>>
            >>> # Load from local path
            >>> pipeline = QEffWanImageToVideoPipeline.from_pretrained("/local/path/to/wan/i2v")
            >>>
            >>> # Load with custom cache directory
            >>> pipeline = QEffWanImageToVideoPipeline.from_pretrained(
            ...     "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
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

        This method systematically exports the VAE encoder, unified transformer, and VAE decoder to ONNX format with
        image-to-video specific configurations including temporal dimensions, dynamic axes, and
        optimization settings.

        The export process prepares the models for subsequent compilation to QPC format, enabling
        efficient inference on QAIC hardware. ONNX subfunctions can be used for certain modules
        to optimize memory usage and performance.

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
            >>> pipeline = QEffWanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            >>> export_path = pipeline.export(
            ...     export_dir="/path/to/export",
            ...     use_onnx_subfunctions=True
            ... )
            >>> print(f"Models exported to: {export_path}")
        """

        # Export each module with corresponding parameters
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

            module_obj.export(**export_params)

    @staticmethod
    def get_default_config_path():
        """
        Get the default configuration file path for WAN pipeline.

        Returns:
            str: Path to the default WAN configuration JSON file.
        """
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/wan_i2v_config.json")

    @staticmethod
    def get_vae_encoder_npi_path():
        """
        Get the default VAE encoder NPI configuration file path for WAN I2V pipeline.

        Returns:
            str: Path to the default WAN I2V VAE encoder NPI file.
        """
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/npi_wan_i2v_vae_encoder.yaml")

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
            >>> pipeline = QEffWanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
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
        if any(
            path is None
            for path in [
                self.vae_encoder.onnx_path,
                self.transformer.onnx_path,
                self.vae_decoder.onnx_path,
            ]
        ):
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

        # # Update NPI path for vae encoder
        vae_npi_full_path = self.get_vae_encoder_npi_path()
        update_npi_path(self, vae_npi_full_path, module_name="vae_encoder")

        # Prepare dynamic specialization updates based on video dimensions
        specialization_updates = {
            "vae_encoder": {
                "num_frames": num_frames,
                "height": height,
                "width": width,
            },
            "transformer": [
                # high noise
                {
                    "cl": cl,  # Compressed latent dimension
                    "latent_height": latent_height,  # Latent space height
                    "latent_width": latent_width,  # Latent space width
                    "latent_frames": latent_frames,  # Latent frames
                },
                # low noise
                {
                    "cl": cl,  # Compressed latent dimension
                    "latent_height": latent_height,  # Latent space height
                    "latent_width": latent_width,  # Latent space width
                    "latent_frames": latent_frames,  # Latent frames
                },
            ],
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

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare latent variables for image-to-video generation with temporal conditioning.

        This method handles the complex process of preparing latent tensors for I2V generation,
        including image conditioning, temporal mask generation, and VAE encoding. It creates
        the initial noise latents and processes the input image(s) to create conditioning
        information that maintains temporal consistency throughout video generation.

        Args:
            image (PipelineImageInput): Input image(s) to condition the video generation.
                Can be PIL Image, numpy array, or torch tensor.
            batch_size (int): Number of videos to generate in parallel.
            num_channels_latents (int, default=16): Number of channels in the latent space.
            height (int, default=480): Target video height in pixels.
            width (int, default=832): Target video width in pixels.
            num_frames (int, default=81): Number of frames in the generated video.
            dtype (torch.dtype, optional): Data type for latent tensors. If None, uses float32.
            device (torch.device, optional): Device to place tensors on. If None, uses CPU.
            generator (torch.Generator or List[torch.Generator], optional): Random generator(s)
                for reproducible latent initialization.
            latents (torch.Tensor, optional): Pre-generated latent tensors. If None, random
                latents are created.
            last_image (torch.Tensor, optional): Optional last frame image for video completion
                tasks. Used to create temporal boundaries.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - latents: Initial noise latents for denoising process
                - condition: Conditioning tensor combining temporal masks and image latents
                  OR (if expand_timesteps=True):
                - latents: Initial noise latents
                - latent_condition: Image conditioning latents

        Raises:
            ValueError: If generator list length doesn't match batch size
            RuntimeError: If VAE encoding fails or tensor operations fail

        """
        num_latent_frames = (num_frames - 1) // self.model.vae.config.scale_factor_temporal + 1
        latent_height = height // self.model.vae.config.scale_factor_spatial
        latent_width = width // self.model.vae.config.scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if self.model.config.expand_timesteps:
            video_condition = image

        elif last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.model.vae.dtype)

        latents_mean = (
            torch.tensor(self.model.vae.config.latents_mean)
            .view(1, self.model.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.model.vae.config.latents_std).view(
            1, self.model.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        # Initialize VAE encoder inference session
        if self.vae_encoder.qpc_session is None:
            self.vae_encoder.qpc_session = QAICInferenceSession(
                str(self.vae_encoder.qpc_path), device_ids=self.vae_encoder.device_ids
            )

        # # Allocate output buffer for VAE encoder
        output_buffer = {
            "latents": np.random.rand(
                batch_size, constants.WAN_DIT_I2V_IMG_LATENT_CHANNELS, num_latent_frames, latent_height, latent_width
            ).astype(np.int32)
        }
        self.vae_encoder.qpc_session.set_buffers(output_buffer)

        aic_vae_encoder_input = {"image": video_condition.detach().numpy()}

        # Vae encoder QAIC inference
        start_vae_time = time.perf_counter()
        outputs = self.vae_encoder.qpc_session.run(aic_vae_encoder_input)
        end_vae_time = time.perf_counter()
        vae_encoder_perf = end_vae_time - start_vae_time

        qaic_op = torch.from_numpy(outputs["latents"])
        latent_condition_mean, logvar = torch.chunk(qaic_op, 2, dim=1)
        latent_condition = latent_condition_mean.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        if self.model.config.expand_timesteps:
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
            )
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask, vae_encoder_perf

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=self.model.vae.config.scale_factor_temporal
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, self.model.vae.config.scale_factor_temporal, latent_height, latent_width
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1), vae_encoder_perf

    @log_pipeline_api(
        "generate",
        "Generation completed.",
        "generation_complete",
        start_run=True,
        finish_run=True,
    )
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 544,
        width: int = 720,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None]]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        custom_config_path: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        parallel_compile: bool = True,
    ):
        """
        Generate videos from input images and text prompts using the QEfficient-optimized WAN I2V pipeline on QAIC hardware.

        This is the main entry point for image-to-video generation. It orchestrates the complete WAN I2V
        diffusion pipeline optimized for Qualcomm AI Cloud devices, converting static images into dynamic
        video sequences with temporal consistency and text-guided motion.

        Args:
            image (PipelineImageInput): Input image(s) to condition the video generation. Can be PIL Image,
                numpy array, or torch tensor. This serves as the first frame or conditioning frame for the video.
            prompt (str or List[str], optional): Primary text prompt(s) describing the desired motion and content
                for the video. Required unless `prompt_embeds` is provided.
            negative_prompt (str or List[str], optional): Negative prompt(s) describing what to avoid
                in the generated video. Used with classifier-free guidance.
            height (int, optional): Target video height in pixels. Must be divisible by VAE scale factor.
                Default: 480.
            width (int, optional): Target video width in pixels. Must be divisible by VAE scale factor.
                Default: 832.
            num_frames (int, optional): Number of video frames to generate. Must satisfy temporal
                divisibility requirements (num_frames - 1) % temporal_scale_factor == 0. Default: 81.
            num_inference_steps (int, optional): Number of denoising steps. More steps generally
                improve quality but increase generation time. Default: 50.
            guidance_scale (float, optional): Guidance scale for classifier-free guidance in high-noise stage.
                Default: 3.0.
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
            image_embeds (torch.Tensor, optional): Pre-computed image embeddings (currently unused).
            last_image (torch.Tensor, optional): Optional last frame image for video completion tasks.
                Used to create temporal boundaries in the generated video.
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
            parallel_compile (bool, optional): Whether to compile modules in parallel. Default: True.

        Returns:
            QEffPipelineOutput: A dataclass containing:
                - images: Generated video(s) in the format specified by `output_type`
                - pipeline_module: Performance metrics for each pipeline component (transformer, VAE decoder)

        Raises:
            ValueError: If input validation fails or parameters are incompatible
            RuntimeError: If compilation fails or QAIC devices are unavailable
            FileNotFoundError: If custom config file is specified but not found

        Example:
            >>> from QEfficient.diffusers.pipelines.wan import QEffWanImageToVideoPipeline
            >>> from PIL import Image
            >>>
            >>> # Load pipeline and input image
            >>> pipeline = QEffWanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            >>> image = Image.open("input_frame.jpg")
            >>>
            >>> # Generate video with motion
            >>> result = pipeline(
            ...     image=image,
            ...     prompt="A person walking through a sunny garden with flowing motion",
            ...     height=544,
            ...     width=720,
            ...     num_frames=81,
            ...     num_inference_steps=4,
            ...     guidance_scale=1.0
            ... )
            >>>
            >>> # Save generated video
            >>> frames = result.images[0]
            >>> export_to_video(frames, "generated_video.mp4", fps=16)
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
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
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
        transformer_dtype = self.transformer.model.transformer_high.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Step 4: Prepare timesteps for denoising process
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Step 5: Prepare initial latent variables for video generation
        num_channels_latents = self.model.vae.config.z_dim
        image = self.model.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )
        if self.model.config.expand_timesteps:
            # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
            latents, condition, first_frame_mask, vae_encoder_perf = latents_outputs
        else:
            latents, condition, vae_encoder_perf = latents_outputs

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.model.config.boundary_ratio is not None:
            boundary_timestep = self.model.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # Step 7: Initialize QAIC inference session for transformer
        if self.transformer.qpc_session is None:
            qpc_load_start = time.perf_counter()
            self.transformer.qpc_session = QAICInferenceSession(
                str(self.transformer.qpc_path), device_ids=self.transformer.device_ids
            )
            qpc_load_end = time.perf_counter()
            print(f" DIT QAICInferenceSession time {qpc_load_end - qpc_load_start:.2f} seconds")

        # Calculate compressed latent dimension for transformer buffer allocation
        cl, _, _, _ = calculate_latent_dimensions_with_frames(
            height,
            width,
            num_frames,
            self.model.vae.config.scale_factor_spatial,
            self.model.vae.config.scale_factor_temporal,
            self.patch_height,
            self.patch_width,
        )
        # Allocate output buffer for QAIC inference
        output_buffer = {
            "output": np.random.rand(
                batch_size,
                cl,  # Compressed latent dimension
                constants.WAN_DIT_OUT_CHANNELS,
            ).astype(np.int32),
        }
        self.transformer.qpc_session.set_buffers(output_buffer)
        transformer_perf = []

        # Step 8: Denoising loop with dual-stage processing
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                self._current_timestep = t

                # Determine which model to use based on boundary timestep
                if boundary_timestep is None or t >= boundary_timestep:
                    # High-noise stage
                    current_model = self.transformer.model.transformer_high
                    current_guidance_scale = guidance_scale
                    model_type = torch.ones(1, dtype=torch.int64)  # High-noise model indicator
                else:
                    # Low-noise stage
                    current_model = self.transformer.model.transformer_low
                    current_guidance_scale = guidance_scale_2
                    model_type = torch.ones(2, dtype=torch.int64)  # Low-noise model indicator

                # Prepare latent input with proper dtype
                latent_model_input = latents.to(transformer_dtype)

                # Handle timestep expansion for temporal consistency
                if self.model.config.expand_timesteps:
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                # Extract dimensions for patch processing
                batch_size, num_channels, latent_frames, latent_height, latent_width = latent_model_input.shape
                p_t, p_h, p_w = current_model.config.patch_size
                post_patch_num_frames = latent_frames // p_t
                post_patch_height = latent_height // p_h
                post_patch_width = latent_width // p_w

                # Generate rotary position embeddings
                rotary_emb = current_model.rope(latent_model_input)
                rotary_emb = torch.cat(rotary_emb, dim=0)
                ts_seq_len = None
                timestep = timestep.flatten()

                # Generate conditioning embeddings (time + text)
                temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
                    current_model.condition_embedder(
                        timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                    )
                )

                # Generate negative conditioning for classifier-free guidance
                if self.do_classifier_free_guidance:
                    temb, timestep_proj, encoder_hidden_states_neg, encoder_hidden_states_image = (
                        current_model.condition_embedder(
                            timestep,
                            negative_prompt_embeds,
                            encoder_hidden_states_image=None,
                            timestep_seq_len=ts_seq_len,
                        )
                    )

                # Reshape timestep projection for transformer input
                timestep_proj = timestep_proj.unflatten(1, (6, -1))

                # Prepare inputs for QAIC inference
                inputs_aic = {
                    "hidden_states": latent_model_input.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy(),
                    "tsp": model_type.detach().numpy(),  # Transformer stage pointer
                }

                # Prepare negative inputs for classifier-free guidance
                if self.do_classifier_free_guidance:
                    inputs_aic2 = {
                        "hidden_states": latent_model_input.detach().numpy(),
                        "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                        "rotary_emb": rotary_emb.detach().numpy(),
                        "temb": temb.detach().numpy(),
                        "timestep_proj": timestep_proj.detach().numpy(),
                    }

                # Run conditional prediction with caching context
                with current_model.cache_context("cond"):
                    # QAIC inference for conditional prediction
                    start_transformer_step_time = time.perf_counter()
                    outputs = self.transformer.qpc_session.run(inputs_aic)
                    end_transformer_step_time = time.perf_counter()
                    transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                    print(f"DIT {i} time {end_transformer_step_time - start_transformer_step_time:.2f} seconds")

                    # Process transformer output
                    hidden_states = torch.tensor(outputs["output"])

                    # Reshape output from patches back to video format
                    hidden_states = hidden_states.reshape(
                        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                    )

                    # Permute dimensions to reconstruct video tensor
                    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                # Run unconditional prediction for classifier-free guidance
                if self.do_classifier_free_guidance:  # Note: CFG will increase DIT num steps.
                    with current_model.cache_context("uncond"):
                        # QAIC inference for unconditional prediction
                        start_transformer_step_time = time.perf_counter()
                        outputs = self.transformer.qpc_session.run(inputs_aic2)
                        end_transformer_step_time = time.perf_counter()
                        transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

                        # Process unconditional output
                        hidden_states = torch.tensor(outputs["output"])

                        # Reshape unconditional output
                        hidden_states = hidden_states.reshape(
                            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                        )

                        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                        noise_uncond = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                        # Apply classifier-free guidance
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # Update latents using scheduler (x_t -> x_t-1)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Execute callback if provided # TODO: optimize to run DIT and vae in parallel
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, callback_kwargs, num_frames=num_frames)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        if self.model.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

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

        # # Allocate output buffer for VAE decoder
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

        # Step 10: Collect performance metrics
        perf_data = {
            "vae_encoder": vae_encoder_perf,
            "transformer": transformer_perf,
            "vae_decoder": vae_decoder_perf,
        }

        # Build performance metrics for output
        perf_metrics = [ModulePerf(module_name=name, perf=perf_data[name]) for name in perf_data.keys()]

        return QEffPipelineOutput(
            pipeline_module=perf_metrics,
            images=video,
        )
