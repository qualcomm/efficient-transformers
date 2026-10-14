# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import os
import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers import Flux2KleinPipeline
from tqdm import tqdm

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.diffusers.pipelines.pipeline_module import (
    Flux2VaeDecoderWrapper,
    Flux2VaeEncoderWrapper,
    QEffFlux2TransformerModel,
    QEffVAE,
)
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    ModulePerf,
    QEffPipelineOutput,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_execute_params,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import logger

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def to_numpy(x):
    """Convert a torch.Tensor (or pass-through numpy array) to numpy."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


# ---------------------------------------------------------------------------
# Helper: compute_empirical_mu
# ---------------------------------------------------------------------------


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)


# ---------------------------------------------------------------------------
# Helper: retrieve_timesteps
# ---------------------------------------------------------------------------


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Thin wrapper around scheduler.set_timesteps that supports custom schedules."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# ---------------------------------------------------------------------------
# Helper: retrieve_latents
# ---------------------------------------------------------------------------


def retrieve_latents(
    encoder_output,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    """
    Extract the latent tensor from a VAE encoder output.

    Handles three cases:
      1. HuggingFace ``AutoencoderKLOutput`` with a ``latent_dist`` attribute
         (the standard diffusers object returned by ``vae.encode()``).
      2. An object that directly exposes a ``latents`` attribute.
      3. A plain ``torch.Tensor`` — returned when the VAE encoder has already
         been run through QAIC (the ONNX wrapper bakes ``latent_dist.mode()``
         into the graph, so the session output is the latent tensor itself).
    """
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    elif isinstance(encoder_output, torch.Tensor):
        return encoder_output
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# ---------------------------------------------------------------------------
# QEffFlux2KleinPipeline
# ---------------------------------------------------------------------------


class QEffFlux2KleinPipeline:
    """
    QEfficient-optimised FLUX.2-klein-4B pipeline for text-to-image generation on
    Qualcomm AI hardware.
    """

    _hf_auto_class = Flux2KleinPipeline

    def __init__(
        self,
        model: Flux2KleinPipeline,
        transformer_qpc_path: Optional[str] = None,
        vae_qpc_path: Optional[str] = None,
        vae_encode_qpc_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialise the QEfficient FLUX.2-klein pipeline.

        Args:
            model (Flux2KleinPipeline): Pre-loaded HuggingFace Flux2KleinPipeline.
            transformer_qpc_path (str, optional): Path to the compiled transformer QPC directory.
            vae_qpc_path (str, optional): Path to the compiled VAE decoder QPC directory.
            vae_encode_qpc_path (str, optional): Path to the compiled VAE encoder QPC directory.
        """
        self.model = model

        # ------------------------------------------------------------------ #
        # Text encoder — wrapped with QEFFAutoModelForCausalLM               #
        # ------------------------------------------------------------------ #
        # Set is_flux2 flag on the text encoder config for FLUX2 Klein model
        model.text_encoder.config.is_flux2 = True

        self.text_encoder = QEFFAutoModelForCausalLM(
            model=model.text_encoder,
        )

        # ------------------------------------------------------------------ #
        # Transformer — QAIC session (lazy init on first call)               #
        # ------------------------------------------------------------------ #
        self.transformer = QEffFlux2TransformerModel(model.transformer)

        self._transformer_session: Optional[QAICInferenceSession] = None

        # ------------------------------------------------------------------ #
        # VAE decoder — QAIC session (lazy init on first call)               #
        # ------------------------------------------------------------------ #
        self.vae_qpc_path = vae_qpc_path
        self._vae_session: Optional[QAICInferenceSession] = None

        # ------------------------------------------------------------------ #
        # VAE encoder — QAIC session (lazy init on first call)               #
        # ------------------------------------------------------------------ #
        self.vae_encode_qpc_path = vae_encode_qpc_path
        self._vae_encode_session: Optional[QAICInferenceSession] = None

        # VAE decoder — QEffVAE wrapping a Flux2VaeDecoderWrapper.

        self.vae_decoder = QEffVAE(Flux2VaeDecoderWrapper(model.vae), "decoder")

        self.vae_encoder = QEffVAE(Flux2VaeEncoderWrapper(model.vae), "encoder")
        self.vae_encoder.get_onnx_params = self.vae_encoder.get_flux2_encoder_onnx_params

        self.modules = {
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "vae_encoder": self.vae_encoder,
            "vae_decoder": self.vae_decoder,
        }
        self.tokenizer = model.tokenizer
        self.tokenizer_max_length = model.tokenizer_max_length  # 512
        self.scheduler = model.scheduler
        self.image_processor = model.image_processor
        self.vae = model.vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.default_sample_size = 128

    def _get_diffusion_modules(self) -> dict[str, object]:
        return {module_name: module for module_name, module in self.modules.items() if module_name != "text_encoder"}

    def _compile_text_encoder_from_config(self) -> None:
        text_encoder_config = self.custom_config["modules"].get("text_encoder")
        if text_encoder_config is None or not hasattr(self, "text_encoder"):
            return

        execute_config = text_encoder_config.get("execute", {})
        self.text_encoder.device_ids = execute_config.get("device_ids")
        configured_qpc_path = execute_config.get("qpc_path")

        if configured_qpc_path:
            if not os.path.exists(configured_qpc_path):
                raise FileNotFoundError(
                    f"Given qpc path: {configured_qpc_path} does not exist. Please provide correct path or keep null"
                )
            self.text_encoder.qpc_path = configured_qpc_path
            return

        if self.text_encoder.qpc_path is not None:
            return

        specializations = text_encoder_config["specializations"].copy()
        compile_kwargs = text_encoder_config["compilation"].copy()
        onnx_path = compile_kwargs.pop("onnx_path", None)
        compile_dir = compile_kwargs.pop("compile_dir", None)
        mdp_ts_num_devices = compile_kwargs.pop("mdp_ts_num_devices", None)
        aic_num_cores = compile_kwargs.pop("aic_num_cores", None)
        compile_kwargs.pop("convert_to_fp16", None)

        prefill_seq_len = specializations.get("seq_len", 512)
        ctx_len = specializations.get("ctx_len", prefill_seq_len)
        batch_size = specializations.get("batch_size", 1)

        compiler_options = {
            "onnx_path": onnx_path,
            "compile_dir": compile_dir,
            "prefill_seq_len": prefill_seq_len,
            "ctx_len": ctx_len,
            "batch_size": batch_size,
            "prefill_only": True,
            **compile_kwargs,
        }
        if mdp_ts_num_devices is not None:
            compiler_options["num_devices"] = mdp_ts_num_devices
        if aic_num_cores is not None:
            compiler_options["num_cores"] = aic_num_cores

        self.text_encoder.compile(**compiler_options)

    def export(
        self,
        export_dir: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        compile_config: Optional[str] = None,
    ) -> str:
        """
        Export all pipeline modules to ONNX format for deployment preparation.


        Args:
            export_dir (str, optional): Target directory for saving ONNX model files. If None,
                uses the default export directory structure based on model name and configuration.
                The directory will be created if it doesn't exist.
            use_onnx_subfunctions (bool, default=False): Whether to enable ONNX subfunction
                optimization for supported modules. This can optimize thegraph and
                improve compilation efficiency for models like the transformer.
            compile_config (str, optional): Path to compilation config JSON. Used to read
                VAE encoder dimensions for ONNX export.

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
        # Load config to get VAE encoder dimensions for ONNX export
        from QEfficient.utils._utils import load_json

        if compile_config is None:
            compile_config = self.get_default_config_path()
        config_data = load_json(compile_config)
        vae_encoder_height = config_data["modules"]["vae_encoder"]["specializations"]["height"]
        vae_encoder_width = config_data["modules"]["vae_encoder"]["specializations"]["width"]

        for module_name, module_obj in tqdm(
            self._get_diffusion_modules().items(), desc="Exporting modules", unit="module"
        ):
            # Get ONNX export configuration for this module
            # For VAE encoder, use dimensions from config specializations to match compilation
            if module_name == "vae_encoder":
                bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
                example_inputs = {
                    "image": torch.randn(bs, 3, vae_encoder_height, vae_encoder_width),
                }
                output_names = ["latents"]
                dynamic_axes = {
                    "image": {0: "batch_size", 2: "height", 3: "width"},
                    "latents": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
                }
            else:
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
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/flux2_config.json")

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
        self._compile_text_encoder_from_config()

        # Ensure all modules are exported to ONNX before compilation
        if any(
            path is None
            for path in [self.vae_encoder.onnx_path, self.transformer.onnx_path, self.vae_decoder.onnx_path]
        ):
            self.export(use_onnx_subfunctions=use_onnx_subfunctions, compile_config=compile_config)

        specialization_updates = {}

        # Use generic utility functions for compilation
        diffusion_modules = self._get_diffusion_modules()
        if parallel:
            compile_modules_parallel(diffusion_modules, self.custom_config, specialization_updates)
        else:
            compile_modules_sequential(diffusion_modules, self.custom_config, specialization_updates)

    # ---------------------------------------------------------------------- #
    # Class method: from_pretrained                                           #
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        transformer_qpc_path: Optional[str] = None,
        vae_qpc_path: Optional[str] = None,
        vae_encode_qpc_path: Optional[str] = None,
        **kwargs,
    ) -> "QEffFlux2KleinPipeline":
        """
        Load a pretrained FLUX.2-klein model and wrap it with QEfficient optimisations.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path,
                e.g. ``"black-forest-labs/FLUX.2-klein-4B"``.
            transformer_qpc_path (str, optional): Path to the compiled transformer QPC directory.
            vae_qpc_path (str, optional): Path to the compiled VAE decoder QPC directory.
            vae_encode_qpc_path (str, optional): Path to the compiled VAE encoder QPC directory.
            **kwargs: Forwarded to ``Flux2KleinPipeline.from_pretrained()``.

        Returns:
            QEffFlux2KleinPipeline: Fully initialised pipeline ready for inference.
        """
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            # device_map="cpu",
            **kwargs,
        )
        return cls(
            model=model,
            transformer_qpc_path=transformer_qpc_path,
            vae_qpc_path=vae_qpc_path,
            vae_encode_qpc_path=vae_encode_qpc_path,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )

    # ---------------------------------------------------------------------- #
    # Session helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _get_transformer_session(self) -> QAICInferenceSession:
        """Lazily initialise and return the transformer QAIC session."""
        if self._transformer_session is None:
            if self.transformer_qpc_path is None:
                raise ValueError(
                    "transformer_qpc_path must be provided to run the transformer on QAIC. "
                    "Pass it to from_pretrained() or set pipeline.transformer_qpc_path."
                )
            logger.info(f"Initialising transformer QAIC session from: {self.transformer_qpc_path}")
            self._transformer_session = QAICInferenceSession(str(self.transformer_qpc_path))
        return self._transformer_session

    def _get_vae_session(self) -> QAICInferenceSession:
        """Lazily initialise and return the VAE decoder QAIC session."""
        if self._vae_session is None:
            if self.vae_qpc_path is None:
                raise ValueError(
                    "vae_qpc_path must be provided to run the VAE decoder on QAIC. "
                    "Pass it to from_pretrained() or set pipeline.vae_qpc_path."
                )
            logger.info(f"Initialising VAE decoder QAIC session from: {self.vae_qpc_path}")
            self._vae_session = QAICInferenceSession(str(self.vae_qpc_path))
        return self._vae_session

    def _get_vae_encode_session(self) -> QAICInferenceSession:
        """Lazily initialise and return the VAE encoder QAIC session."""
        if self._vae_encode_session is None:
            if self.vae_encode_qpc_path is None:
                raise ValueError(
                    "vae_encode_qpc_path must be provided to run the VAE encoder on QAIC. "
                    "Pass it to from_pretrained() or set pipeline.vae_encode_qpc_path."
                )
            logger.info(f"Initialising VAE encoder QAIC session from: {self.vae_encode_qpc_path}")
            self._vae_encode_session = QAICInferenceSession(str(self.vae_encode_qpc_path))
        return self._vae_encode_session

    def _run_transformer_step(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        adaln_double_img: torch.Tensor,
        adaln_double_txt: torch.Tensor,
        adaln_single: torch.Tensor,
        adaln_out: torch.Tensor,
    ):
        transformer_inputs = {
            "hidden_states": to_numpy(latent_model_input),
            "timestep": to_numpy(timestep / 1000),
            "encoder_hidden_states": to_numpy(prompt_embeds),
            "txt_ids": to_numpy(text_ids.to(torch.float32)),
            "img_ids": to_numpy(latent_image_ids.to(torch.float32)),
            "adaln_double_img": to_numpy(adaln_double_img),
            "adaln_double_txt": to_numpy(adaln_double_txt),
            "adaln_single": to_numpy(adaln_single),
            "adaln_out": to_numpy(adaln_out),
        }
        return self.transformer.qpc_session.run(transformer_inputs)

    # ---------------------------------------------------------------------- #
    # Text encoding                                                           #
    # ---------------------------------------------------------------------- #

    def _get_qwen3_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ):
        """
        Encode text prompts using the Qwen3 text encoder running on QAIC.

        The pipeline follows the same chat-template formatting as the reference
        Flux2KleinPipeline::

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, ..., enable_thinking=False)

        Args:
            prompt (str | List[str]): Input prompt(s).
            num_images_per_prompt (int): Images per prompt (for embedding repeat).
            max_sequence_length (int): Tokeniser max length (default: 512).

        Returns:
            Tuple[torch.Tensor, float]:
                - prompt_embeds: (batch * num_images, seq_len, hidden_size)
                - inference_time: seconds spent in QAIC inference
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])

        # ------------------------------------------------------------------ #
        # Initialise QAIC session                                     #
        # ------------------------------------------------------------------ #
        if self.text_encoder.qpc_session is None:
            self.text_encoder.qpc_session = QAICInferenceSession(
                str(self.text_encoder.qpc_path),
            )
            self.text_encoder.qpc_session.skip_buffers(
                [
                    name
                    for name in (self.text_encoder.qpc_session.input_names + self.text_encoder.qpc_session.output_names)
                    if name.startswith("past_key") or name.startswith("past_value") or name.endswith("_RetainedState")
                ]
            )
            hidden_size = self.text_encoder.model.config.hidden_size
            self.text_encoder.qpc_session.set_buffers(
                {"logits": np.zeros((1, 3, max_sequence_length, hidden_size), dtype=np.float32)}
            )

        # Process each prompt individually since QPC is compiled with batch_size=1
        all_prompt_embeds = []
        start = time.perf_counter()
        for input_id in all_input_ids:
            position_ids = torch.arange(input_id.shape[-1]).reshape(1, -1)
            position_ids = torch.where(input_id == 151643, -1, position_ids)

            aic_inputs = {
                "input_ids": input_id.numpy().astype(np.int64),
                "position_ids": position_ids.numpy().astype(np.int64),
            }

            outputs = self.text_encoder.qpc_session.run(aic_inputs)
            prompt_embeds_single = torch.from_numpy(outputs["logits"])  # (1, 3, seq_len, hidden_size)
            all_prompt_embeds.append(prompt_embeds_single)

        end = time.perf_counter()
        text_encoder_perf = end - start

        # Concatenate all prompt embeddings
        prompt_embeds = torch.cat(all_prompt_embeds, dim=0)  # (B, 3, seq_len, hidden_size)

        # Validate text encoder output
        if torch.isnan(prompt_embeds).any():
            raise RuntimeError("Text encoder output contains NaN!")
        if torch.isinf(prompt_embeds).any():
            raise RuntimeError("Text encoder output contains Inf!")
        if prompt_embeds.abs().max() < 1e-6:
            raise RuntimeError("Text encoder output is all zeros!")

        # Reshape: (B, 3, seq_len, hidden_size) -> (B, seq_len, 3*hidden_size)
        B, num_layers, seq_len, hidden_size = prompt_embeds.shape
        prompt_embeds = prompt_embeds.permute(0, 2, 1, 3).reshape(B, seq_len, num_layers * hidden_size)

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return prompt_embeds, text_encoder_perf

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        """
        Encode text prompts using the Qwen3 text encoder on QAIC.

        Args:
            prompt (str | List[str]): Input prompt(s).
            num_images_per_prompt (int): Number of images per prompt.
            prompt_embeds (torch.FloatTensor, optional): Pre-computed embeddings.
                If provided, skips the encoder inference.
            max_sequence_length (int): Tokeniser max length.

        Returns:
            Tuple:
                - prompt_embeds (torch.Tensor): (B*num_images, seq_len, embed_dim)
                - text_ids      (torch.Tensor): 4D position coords (B, seq_len, 4)
                - encoder_perf  (float): Inference time in seconds
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        encoder_perf = 0.0

        if prompt_embeds is None:
            prompt_embeds, encoder_perf = self._get_qwen3_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        return prompt_embeds, text_ids, encoder_perf

    # ---------------------------------------------------------------------- #
    # Static helpers (geometry / packing)                                     #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _prepare_text_ids(
        x: torch.Tensor,  # (B, L, D)
        t_coord: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        """
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor): Shape (B, C, H, W).

        Returns:
            torch.Tensor: Position IDs of shape (B, H*W, 4).
        """
        batch_size, _, height, width = latents.shape
        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    @staticmethod
    def _prepare_image_ids(
        image_latents: List[torch.Tensor],
        scale: int = 10,
    ) -> torch.Tensor:
        """
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

        Args:
            image_latents (List[torch.Tensor]): List of (1, C, H, W) tensors.
            scale (int): Time separation factor between latents.

        Returns:
            torch.Tensor: Shape (1, N_total, 4).
        """
        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]
        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape
            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)
        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)
        return image_latent_ids

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _unpack_latents_with_ids(
        x: torch.Tensor,
        x_ids: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """Scatter tokens back into spatial positions using position IDs."""
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)
            h = height if height is not None else torch.max(h_ids) + 1
            w = width if width is not None else torch.max(w_ids) + 1
            flat_ids = h_ids * w + w_ids
            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)
        return torch.stack(x_list, dim=0)

    # ---------------------------------------------------------------------- #
    # Latent preparation                                                      #
    # ---------------------------------------------------------------------- #

    def prepare_latents(
        self,
        batch_size: int,
        num_latents_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        from diffusers.utils.torch_utils import randn_tensor

        # VAE applies 8x compression; also account for packing (divisible by 2)
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective "
                f"batch size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents).to(device)
        latents = self._pack_latents(latents)  # (B, C, H, W) -> (B, H*W, C)
        return latents, latent_ids

    def prepare_image_latents(
        self,
        images: List[torch.Tensor],
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            image_latent, vae_encoder_perf = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(image_latent)

        image_latent_ids = self._prepare_image_ids(image_latents)

        packed_latents = []
        for latent in image_latents:
            packed = self._pack_latents(latent)  # (1, H*W, C)
            packed = packed.squeeze(0)  # (H*W, C)
            packed_latents.append(packed)

        image_latents = torch.cat(packed_latents, dim=0).unsqueeze(0)  # (1, N*H*W, C)
        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1).to(device)
        return image_latents, image_latent_ids, vae_encoder_perf

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """
        Encode a single image tensor through the VAE encoder running on QAIC.

        Args:
            image (torch.Tensor): Shape (1, C, H, W), dtype float32, values in [-1, 1].
            generator: Unused (kept for API compatibility; encode uses argmax).

        Returns:
            torch.Tensor: Patchified, BN-normalised latents of shape
                          (1, latent_channels * 4, H // (vae_scale_factor * 2), W // (vae_scale_factor * 2)).
        """
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        vae_encode_inputs = {"image": to_numpy(image.to(torch.float32))}
        t_start = time.perf_counter()

        vae_encode_output = self.vae_encoder.qpc_session.run(vae_encode_inputs)

        t_end = time.perf_counter()
        vae_encoder_perf = t_end - t_start
        logger.info(f"VAE encoder inference time: {t_end - t_start:.3f}s")

        image_latents = torch.from_numpy(vae_encode_output["latents"])

        image_latents = self._patchify_latents(image_latents)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            image_latents.device, image_latents.dtype
        )
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std
        return image_latents, vae_encoder_perf

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
    ):
        _callback_tensor_inputs = ["latents", "prompt_embeds"]

        if (
            height is not None
            and height % (self.vae_scale_factor * 2) != 0
            or width is not None
            and width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in _callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {_callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in _callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if guidance_scale is not None and guidance_scale > 1.0 and getattr(self.model.config, "is_distilled", False):
            logger.warning(f"Guidance scale {guidance_scale} is ignored for step-wise distilled models.")

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[Union[str, List[str]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple = (9, 18, 27),
    ):
        """
        Generate images from text prompts using the QEfficient FLUX.2-klein-4B pipeline.
        All three components (text encoder, transformer, VAE decoder) run on QAIC.

        Args:
            prompt (str | List[str]): Text prompt(s) describing the desired image.
            image (PIL.Image.Image | List[PIL.Image.Image], optional): Conditioning image(s).
            height (int): Target image height in pixels. Default: 1024.
            width (int): Target image width in pixels. Default: 1024.
            num_inference_steps (int): Denoising steps. Default: 50.
            sigmas (List[float], optional): Custom sigma schedule.
            guidance_scale (float): CFG scale. Default: 4.0.
            num_images_per_prompt (int): Images per prompt. Default: 1.
            generator (torch.Generator, optional): RNG for reproducibility.
            latents (torch.FloatTensor, optional): Pre-generated noise latents.
            prompt_embeds (torch.FloatTensor, optional): Pre-computed text embeddings.
            negative_prompt_embeds: Pre-computed negative text embeddings.
            output_type (str): "pil", "np", or "latent". Default: "pil".
            return_dict (bool): Return a named output object. Default: True.
            callback_on_step_end (Callable, optional): Step-end callback.
            callback_on_step_end_tensor_inputs (List[str]): Tensors for callback.
            max_sequence_length (int): Tokeniser max length. Default: 512.
            text_encoder_out_layers (tuple): Hidden-state layer indices for text encoder.

        Returns:
            QEffPipelineOutput or tuple: Generated images.
        """
        device = self.model._execution_device

        # ------------------------------------------------------------------ #
        # 1. Validate inputs                                                  #
        # ------------------------------------------------------------------ #
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # ------------------------------------------------------------------ #
        # 2. Compile / locate all QPCs before prompt encoding                #
        # ------------------------------------------------------------------ #
        self.compile(
            compile_config=None,
            parallel=False,
            height=height,
            width=width,
            use_onnx_subfunctions=False,
        )

        # ------------------------------------------------------------------ #
        # 3. Encode prompt via text encoder on QAIC                          #
        # ------------------------------------------------------------------ #
        prompt_embeds, text_ids, text_encoder_perf = self.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )
        logger.info(f"Text encoder inference time: {text_encoder_perf:.3f}s")

        do_classifier_free_guidance = guidance_scale > 1.0 and not getattr(self.model.config, "is_distilled", False)

        if do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids, _ = self.encode_prompt(
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )

        # ------------------------------------------------------------------ #
        # 3. Process conditioning images (optional)                           #
        # ------------------------------------------------------------------ #
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)
            condition_images = []
            target_height = height
            target_width = width
            for img in image:
                # Resize image to exactly match the target dimensions
                img = self.image_processor.preprocess(img, height=target_height, width=target_width, resize_mode="crop")
                condition_images.append(img)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(
                str(self.transformer.qpc_path), device_ids=self.transformer.device_ids
            )

        if self.vae_decoder.qpc_session is None:
            self.vae_decoder.qpc_session = QAICInferenceSession(
                str(self.vae_decoder.qpc_path), device_ids=self.vae_decoder.device_ids
            )

        if self.vae_encoder.qpc_session is None:
            self.vae_encoder.qpc_session = QAICInferenceSession(
                str(self.vae_encoder.qpc_path), device_ids=self.vae_encoder.device_ids
            )

        # ------------------------------------------------------------------ #
        # 3c. VAE-encode conditioning images                                 #
        # ------------------------------------------------------------------ #
        image_latents = None
        image_latent_ids = None
        vae_encoder_perf = 0.0
        if condition_images is not None:
            image_latents, image_latent_ids, vae_encoder_perf = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # ------------------------------------------------------------------ #
        # 4. Prepare latent variables                                         #
        # ------------------------------------------------------------------ #
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # ------------------------------------------------------------------ #
        # 5. Prepare timesteps                                                #
        # ------------------------------------------------------------------ #
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None

        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # ------------------------------------------------------------------ #
        # 7. Denoising loop — transformer runs on QAIC                        #
        # ------------------------------------------------------------------ #
        self.scheduler.set_begin_index(0)
        transformer_step_times = []

        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            if self._interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            latent_model_input = latents.to(torch.float32)

            latent_image_ids = latent_ids

            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(torch.float32)

                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            transformer_model = self.transformer.model  # Flux2Transformer2DModel
            temb_input = timestep.to(torch.float32)

            with torch.no_grad():
                # temb shape: [B, inner_dim]
                temb = transformer_model.time_guidance_embed(temb_input, None)

                # 3. Per-layer AdaLN modulation via SiLU + Linear
                double_mod_img = transformer_model.double_stream_modulation_img(temb)  # [B, inner_dim*6]
                double_mod_txt = transformer_model.double_stream_modulation_txt(temb)  # [B, inner_dim*6]
                single_mod = transformer_model.single_stream_modulation(temb)  # [B, inner_dim*3]

                # 4. norm_out
                adaln_out = transformer_model.norm_out.linear(transformer_model.norm_out.silu(temb))
            num_layers = transformer_model.config.num_layers
            num_single = transformer_model.config.num_single_layers
            adaln_double_img = (
                double_mod_img.squeeze(0).unsqueeze(0).expand(num_layers, -1).contiguous()
            )  # [num_layers, inner_dim*6]
            adaln_double_txt = (
                double_mod_txt.squeeze(0).unsqueeze(0).expand(num_layers, -1).contiguous()
            )  # [num_layers, inner_dim*6]
            adaln_single = (
                single_mod.squeeze(0).unsqueeze(0).expand(num_single, -1).contiguous()
            )  # [num_single, inner_dim*3]

            t_start = time.perf_counter()
            transformer_output = self._run_transformer_step(
                latent_model_input=latent_model_input,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                adaln_double_img=adaln_double_img,
                adaln_double_txt=adaln_double_txt,
                adaln_single=adaln_single,
                adaln_out=adaln_out,
            )
            t_end = time.perf_counter()
            transformer_step_times.append(t_end - t_start)
            logger.info(f"Transformer inference time (step {i}): {t_end - t_start:.3f}s")

            noise_pred = torch.tensor(transformer_output["sample"])
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_classifier_free_guidance:
                with self.transformer.cache_context("uncond"):
                    neg_output = self._run_transformer_step(
                        latent_model_input=latent_model_input,
                        timestep=timestep,
                        prompt_embeds=negative_prompt_embeds,
                        text_ids=negative_text_ids,
                        latent_image_ids=latent_image_ids,
                        adaln_double_img=adaln_double_img,
                        adaln_double_txt=adaln_double_txt,
                        adaln_single=adaln_single,
                        adaln_out=adaln_out,
                    )
                    neg_noise_pred = torch.tensor(neg_output["sample"])
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            # Scheduler step
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        self._current_timestep = None

        # ------------------------------------------------------------------ #
        # 8. Decode latents — VAE decoder runs on QAIC                        #
        # ------------------------------------------------------------------ #
        # Unpack latents back to spatial layout
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latents = self._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)

        # Reverse BN normalisation
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        if output_type == "latent":
            image = latents
        else:
            # Run VAE decoder on QAIC
            vae_in_np = {"latent_sample": to_numpy(latents)}
            vae_perf = 0.0

            t_start = time.perf_counter()
            vae_output = self.vae_decoder.qpc_session.run(vae_in_np)
            t_end = time.perf_counter()
            vae_perf = t_end - t_start
            logger.info(f"VAE decoder inference time: {t_end - t_start:.3f}s")

            decoded = torch.tensor(vae_output["sample"])
            image = self.image_processor.postprocess(decoded, output_type=output_type)

        # Release VAE encoder session if it was used
        if self._vae_encode_session is not None:
            self._vae_encode_session.deactivate()
            self._vae_encode_session = None

        if not return_dict:
            return (image,)
        pipeline_perf = [
            ModulePerf(module_name="text_encoder", perf=text_encoder_perf),
            ModulePerf(module_name="transformer", perf=transformer_step_times),
            ModulePerf(module_name="vae_decoder", perf=vae_perf),
            ModulePerf(module_name="vae_encoder", perf=vae_encoder_perf),
        ]
        return QEffPipelineOutput(pipeline_module=pipeline_perf, images=image)
