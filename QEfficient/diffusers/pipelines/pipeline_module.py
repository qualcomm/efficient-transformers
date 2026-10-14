# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.models.pytorch_transforms import (
    AttentionTransform,
    CLIPTextTransform,
    CustomOpsTransform,
    NormalizationTransform,
)
from QEfficient.transformers.models.pytorch_transforms import (
    T5ModelTransform,
)
from QEfficient.utils import constants


# ---------------------------------------------------------------------------
# Flux2 VAE wrapper modules
# ---------------------------------------------------------------------------


class Flux2VaeEncoderWrapper(torch.nn.Module):
    """
    Input:  image
    Output: latents
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.config = vae.config  # expose config for QEFFBaseModel

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoder_output = self.vae.encode(image)
        if hasattr(encoder_output, "latent_dist"):
            return encoder_output.latent_dist.mode()
        return encoder_output.latents


class Flux2VaeDecoderWrapper(torch.nn.Module):
    """
    Input:  latent_sample
    Output: sample
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.config = vae.config  # expose config for QEFFBaseModel

    def forward(self, latent_sample: torch.Tensor) -> torch.Tensor:
        # return_dict=False -> plain tuple; take first element (decoded image)
        decoded = self.vae.decode(latent_sample, return_dict=False)
        return decoded[0]


class QEffTextEncoder(QEFFBaseModel):
    """
    Wrapper for text encoder models with ONNX export and QAIC compilation capabilities.

    This class handles text encoder models (CLIP, T5) with specific transformations and
    optimizations for efficient inference on Qualcomm AI hardware. It applies custom
    PyTorch and ONNX transformations to prepare models for deployment.

    Attributes:
        model (nn.Module): The wrapped text encoder model (deep copy of original)
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CLIPTextTransform, CustomOpsTransform, T5ModelTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying text encoder model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the text encoder wrapper.

        Args:
            model (nn.Module): The text encoder model to wrap (CLIP or T5)
        """
        super().__init__(model)
        self.model = model

    def get_onnx_params(self) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the text encoder.

        Creates example inputs, dynamic axes specifications, and output names
        tailored to the specific text encoder type (CLIP vs T5).

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE

        # Create example input with max sequence length
        example_inputs = {
            "input_ids": torch.zeros((bs, self.model.config.max_position_embeddings), dtype=torch.int64),
        }

        # Define which dimensions can vary at runtime
        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}

        # T5 only outputs hidden states, CLIP outputs both hidden states and pooled output
        if self.model.__class__.__name__ == "T5EncoderModel":
            output_names = ["last_hidden_state"]
        else:
            output_names = ["last_hidden_state", "pooler_output"]
            example_inputs["output_hidden_states"] = False

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = {},
    ) -> str:
        """
        Export the text encoder model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            **export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options (e.g., num_cores, aic_num_of_activations)
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffUNet(QEFFBaseModel):
    """
    Wrapper for UNet models with ONNX export and QAIC compilation capabilities.

    This class handles UNet models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. UNet is commonly used in
    diffusion models for image generation tasks.

    Attributes:
        model (nn.Module): The wrapped UNet model
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying UNet model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the UNet wrapper.

        Args:
            model (nn.Module): The pipeline model containing the UNet
        """
        super().__init__(model.unet)
        self.model = model.unet

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = {},
    ) -> str:
        """
        Export the UNet model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            **export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffVAE(QEFFBaseModel):
    """
    Wrapper for Variational Autoencoder (VAE) models with ONNX export and QAIC compilation.

    This class handles VAE models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. VAE models are used in diffusion
    pipelines for encoding images to latent space and decoding latents back to images.

    Attributes:
        model (nn.Module): The wrapped VAE model (deep copy of original)
        type (str): VAE operation type ("encoder" or "decoder")
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform, AttentionTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying VAE model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module, type: str) -> None:
        """
        Initialize the VAE wrapper.

        Args:
            model (nn.Module): The pipeline model containing the VAE
            type (str): VAE operation type ("encoder" or "decoder")
        """
        super().__init__(model)
        self.model = model
        self.type = type
        # TODO: add vae type in hash file

    def get_onnx_params(self, latent_height: int = 32, latent_width: int = 32) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the VAE encoder or decoder.

        Args:
            latent_height (int): Height of latent representation (default: 32)
            latent_width (int): Width of latent representation (default: 32)

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE

        if self.type == "encoder":
            sample_size = getattr(self.model.config, "sample_size", None)
            height = sample_size if isinstance(sample_size, int) else 1024
            width = sample_size if isinstance(sample_size, int) else 1024
            example_inputs = {
                "image": torch.randn(bs, 3, height, width),
            }
            output_names = ["latents"]
            dynamic_axes = {
                "image": {0: "batch_size", 2: "height", 3: "width"},
            }
            return example_inputs, dynamic_axes, output_names

        # Read latent channel count from model config so this works for both
        # Flux1 (AutoencoderKL, latent_channels=16) and
        # Flux2 (AutoencoderKLFlux2, latent_channels=32) without hardcoding.
        latent_channels = getattr(self.model.config, "latent_channels", 16)

        # VAE decoder takes latent representation as input
        example_inputs = {
            "latent_sample": torch.randn(bs, latent_channels, latent_height, latent_width),
            "return_dict": False,
        }

        output_names = ["sample"]

        dynamic_axes = {
            "latent_sample": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        }

        return example_inputs, dynamic_axes, output_names

    def get_img_encoder_onnx_params(self) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the VAE Encoder.

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        num_frames = constants.WAN_ONNX_EXPORT_FRAMES
        height = constants.WAN_ONNX_EXPORT_HEIGHT_45P
        width = constants.WAN_ONNX_EXPORT_WIDTH_45P
        example_inputs = {
            "image": torch.randn(
                bs,
                3,  # channels
                num_frames,
                height,
                width,
            ),
        }
        output_names = ["latents"]
        # All dimensions except channels can be dynamic
        dynamic_axes = {
            "image": {
                0: "batch_size",
                # 1: "num_channels",
                2: "num_frames",
                3: "height",
                4: "width",
            },
        }

        return example_inputs, dynamic_axes, output_names

    def get_flux2_encoder_onnx_params(self) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the Flux2 VAE image encoder.
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        # sample_size is the native spatial resolution (e.g. 1024 for Flux2)
        # sample_size = getattr(self.model.config, "sample_size", 1024)
        # height = sample_size if isinstance(sample_size, int) else 1024
        # width  = sample_size if isinstance(sample_size, int) else 1024
        height = 480
        width = 832
        example_inputs = {
            "image": torch.randn(bs, 3, height, width),
        }
        output_names = ["latents"]
        dynamic_axes = {
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "latents": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        }
        return example_inputs, dynamic_axes, output_names

    def get_video_onnx_params(self) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the VAE decoder.

        Args:
            latent_height (int): Height of latent representation (default: 32)
            latent_width (int): Width of latent representation (default: 32)

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        latent_frames = constants.WAN_ONNX_EXPORT_LATENT_FRAMES
        latent_height = constants.WAN_ONNX_EXPORT_LATENT_HEIGHT_45P
        latent_width = constants.WAN_ONNX_EXPORT_LATENT_WIDTH_45P

        # VAE decoder takes latent representation as input
        example_inputs = {
            "latent_sample": torch.randn(bs, 16, latent_frames, latent_height, latent_width),
            "return_dict": False,
        }

        output_names = ["sample"]

        # All dimensions except channels can be dynamic
        dynamic_axes = {
            "latent_sample": {0: "batch_size", 2: "latent_frames", 3: "latent_height", 4: "latent_width"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = {},
    ) -> str:
        """
        Export the VAE model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """

        if hasattr(self.model.config, "_use_default_values"):
            self.model.config["_use_default_values"].sort()

        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=False,  # shared vae ref — must not deallocate between encoder/decoder exports
            **export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffFluxTransformerModel(QEFFBaseModel):
    """
    Wrapper for Flux Transformer2D models with ONNX export and QAIC compilation capabilities.

    This class handles Flux Transformer2D models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. Flux uses a transformer-based diffusion
    architecture instead of traditional UNet, with dual transformer blocks and adaptive layer
    normalization (AdaLN) for conditioning.

    Attributes:
        model (nn.Module): The wrapped Flux transformer model
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [AttentionTransform, NormalizationTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying Flux transformer model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the Flux transformer wrapper.

        Args:
            model (nn.Module): The Flux transformer model to wrap
        """
        super().__init__(model)

    def get_onnx_params(
        self,
        batch_size: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
        seq_length: int = constants.FLUX_ONNX_EXPORT_SEQ_LENGTH,
        cl: int = constants.FLUX_ONNX_EXPORT_COMPRESSED_LATENT_DIM,
    ) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the Flux transformer.

        Creates example inputs for all Flux-specific inputs including hidden states,
        text embeddings, timestep conditioning, and AdaLN embeddings.

        Args:
            batch_size (int): Batch size for example inputs (default: FLUX_ONNX_EXPORT_BATCH_SIZE)
            seq_length (int): Text sequence length (default: FLUX_ONNX_EXPORT_SEQ_LENGTH)
            cl (int): Compressed latent dimension (default: FLUX_ONNX_EXPORT_COMPRESSED_LATENT_DIM)

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        example_inputs = {
            # Latent representation of the image
            "hidden_states": torch.randn(batch_size, cl, self.model.config.in_channels, dtype=torch.float32),
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, self.model.config.joint_attention_dim, dtype=torch.float32
            ),
            "pooled_projections": torch.randn(batch_size, self.model.config.pooled_projection_dim, dtype=torch.float32),
            "timestep": torch.tensor([1.0], dtype=torch.float32),
            "img_ids": torch.randn(cl, 3, dtype=torch.float32),
            "txt_ids": torch.randn(seq_length, 3, dtype=torch.float32),
            # AdaLN embeddings for dual transformer blocks
            # Shape: [num_layers, FLUX_ADALN_DUAL_BLOCK_CHUNKS, FLUX_ADALN_HIDDEN_DIM]
            "adaln_emb": torch.randn(
                self.model.config["num_layers"],
                constants.FLUX_ADALN_DUAL_BLOCK_CHUNKS,
                constants.FLUX_ADALN_HIDDEN_DIM,
                dtype=torch.float32,
            ),
            # AdaLN embeddings for single transformer blocks
            # Shape: [num_single_layers, FLUX_ADALN_SINGLE_BLOCK_CHUNKS, FLUX_ADALN_HIDDEN_DIM]
            "adaln_single_emb": torch.randn(
                self.model.config["num_single_layers"],
                constants.FLUX_ADALN_SINGLE_BLOCK_CHUNKS,
                constants.FLUX_ADALN_HIDDEN_DIM,
                dtype=torch.float32,
            ),
            # Output AdaLN embedding
            # Shape: [batch_size, FLUX_ADALN_OUTPUT_DIM] for final projection
            "adaln_out": torch.randn(batch_size, constants.FLUX_ADALN_OUTPUT_DIM, dtype=torch.float32),
        }

        output_names = ["output"]

        # Define dynamic dimensions for runtime flexibility
        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "cl"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "steps"},
            "img_ids": {0: "cl"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """
        Export the Flux transformer model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            use_onnx_subfunctions (bool): Whether to export transformer blocks as ONNX functions
                                     for better modularity and potential optimization

        Returns:
            str: Path to the exported ONNX model
        """

        # Sort _use_default_values in config to ensure consistent hash generation during export
        self.model.config["_use_default_values"].sort()
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=False,  # As weights are needed with AdaLN changes
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options (e.g., num_cores, aic_num_of_activations)
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffFlux2VAE(QEFFBaseModel):
    """
    Wrapper for the Flux2 (AutoencoderKLFlux2) VAE decoder with ONNX export and
    QAIC compilation capabilities.

    Attributes:
        model (nn.Module): The wrapped AutoencoderKLFlux2 model
        type (str): Always ``"decoder"`` for this class
        _pytorch_transforms (List): Only CustomOpsTransform — no AttentionTransform
        _onnx_transforms (List): Standard FP16 clip + tensor split
    """

    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.__dict__

    def __init__(self, model: nn.Module, type: str = "decoder") -> None:
        """
        Args:
            model (nn.Module): The AutoencoderKLFlux2 instance.
            type (str): VAE operation type — always ``"decoder"`` here.
        """
        super().__init__(model)
        self.model = model
        self.type = type

    def get_onnx_params(self, latent_height: int = 32, latent_width: int = 32) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the Flux2 VAE decoder.
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        # self.model is Flux2VaeDecoderWrapper; .config is the VAE config
        latent_channels = self.model.config.latent_channels  # 32 for Flux2
        example_inputs = {
            "latent_sample": torch.randn(bs, latent_channels, latent_height, latent_width),
        }
        output_names = ["sample"]
        dynamic_axes = {
            "latent_sample": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        }
        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = {},
    ) -> str:
        """
        Export the Flux2 VAE decoder to ONNX format.
        """
        if hasattr(self.model.config, "_use_default_values"):
            self.model.config["_use_default_values"].sort()

        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=False,  # To do:
            **export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.
        """
        # self._compile(specializations=specializations, **compiler_options)
        self._compile(specializations=specializations, offload_pt_weights=False, **compiler_options)


class QEffFlux2TransformerModel(QEFFBaseModel):
    """
    Wrapper for Flux2 Transformer2D models with ONNX export and QAIC compilation capabilities.

    This class handles Flux2 Transformer2D models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. Flux2 uses a transformer-based diffusion
    architecture with dual-stream and single-stream transformer blocks. AdaLN modulation
    parameters are hoisted outside the model and passed as explicit ONNX inputs so that
    the SiLU+Linear projection ops (Flux2Modulation) are not part of the exported graph.

    Attributes:
        model (nn.Module): The wrapped Flux2 transformer model
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [AttentionTransform, NormalizationTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying Flux transformer model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the Flux transformer wrapper.

        Args:
            model (nn.Module): The Flux transformer model to wrap
        """
        super().__init__(model)

    def get_onnx_params(
        self,
        batch_size: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
        seq_length: int = constants.FLUX2_ONNX_EXPORT_SEQ_LENGTH,
        cl: int = constants.FLUX2_ONNX_EXPORT_COMPRESSED_LATENT_DIM,
    ) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the Flux2 transformer.

        Args:
            batch_size (int): Batch size for example inputs
            seq_length (int): Text sequence length
            cl (int): Compressed latent (image token) dimension

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        # Derive inner_dim from model config — no hardcoded constants needed
        inner_dim = self.model.config.num_attention_heads * self.model.config.attention_head_dim
        num_layers = self.model.config.num_layers
        num_single = self.model.config.num_single_layers

        # axes_dims_rope has 4 entries for Flux2; RoPE ids have 4 coords per token
        rope_dim = len(self.model.config.axes_dims_rope)

        example_inputs = {
            # Image latent tokens: [batch, cl, in_channels]
            "hidden_states": torch.randn(batch_size, cl, self.model.config.in_channels, dtype=torch.float32),
            # Text conditioning tokens: [batch, seq_length, joint_attention_dim]
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, self.model.config.joint_attention_dim, dtype=torch.float32
            ),
            "timestep": torch.tensor([1.0], dtype=torch.float32),
            "img_ids": torch.randn(cl, rope_dim, dtype=torch.float32),
            "txt_ids": torch.randn(seq_length, rope_dim, dtype=torch.float32),
            "adaln_double_img": torch.randn(num_layers, inner_dim * 6, dtype=torch.float32),
            # Dual-block txt-stream: same layout as img
            "adaln_double_txt": torch.randn(num_layers, inner_dim * 6, dtype=torch.float32),
            # Single-block: [mod_shift, mod_scale, mod_gate] per layer
            "adaln_single": torch.randn(num_single, inner_dim * 3, dtype=torch.float32),
            # Final AdaLayerNormContinuous: [scale, shift] — QEffAdaLayerNormContinuous
            "adaln_out": torch.randn(batch_size, inner_dim * 2, dtype=torch.float32),
        }

        output_names = ["sample"]

        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "cl"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "timestep": {0: "steps"},
            "img_ids": {0: "cl"},
            "sample": {0: "batch_size", 1: "cl"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """
        Export the Flux2 transformer model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            use_onnx_subfunctions (bool): Whether to export transformer blocks as ONNX functions
                                     for better modularity and potential optimization
        Returns:
            str: Path to the exported ONNX model
        """

        # Sort _use_default_values in config to ensure consistent hash generation during export
        if "_use_default_values" in self.model.config:
            self.model.config["_use_default_values"].sort()
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=False,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options (e.g., num_cores, aic_num_of_activations)
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffWanTransformer(QEFFBaseModel):
    """
    Wrapper for a single WAN Transformer3D model with ONNX export and QAIC compilation.

    This wrapper is used by the non-unified WAN pipeline mode where high-noise and low-noise
    transformers are exported/compiled/executed as separate modules.
    """

    _pytorch_transforms = [AttentionTransform, CustomOpsTransform, NormalizationTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, transformer, module_name: str = "transformer"):
        super().__init__(transformer, module_name=module_name)
        self.model = transformer
        # Ensure high/low non-unified transformers get distinct export hashes/paths
        # even when configs are identical.
        self.hash_params["module_name"] = module_name

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.__dict__

    def get_onnx_params(self):
        batch_size = constants.WAN_ONNX_EXPORT_BATCH_SIZE
        example_inputs = {
            "hidden_states": torch.randn(
                batch_size,
                self.model.config.in_channels,
                constants.WAN_ONNX_EXPORT_LATENT_FRAMES,
                constants.WAN_ONNX_EXPORT_LATENT_HEIGHT_45P,
                constants.WAN_ONNX_EXPORT_LATENT_WIDTH_45P,
                dtype=torch.float32,
            ),
            "encoder_hidden_states": torch.randn(
                batch_size, constants.WAN_ONNX_EXPORT_SEQ_LEN, constants.WAN_TEXT_EMBED_DIM, dtype=torch.float32
            ),
            "rotary_emb": torch.randn(
                2, constants.WAN_ONNX_EXPORT_CL_45P, 1, constants.WAN_ONNX_EXPORT_ROTARY_DIM, dtype=torch.float32
            ),
            "temb": torch.randn(batch_size, constants.WAN_TEXT_EMBED_DIM, dtype=torch.float32),
            "timestep_proj": torch.randn(
                batch_size,
                constants.WAN_PROJECTION_DIM,
                constants.WAN_TEXT_EMBED_DIM,
                dtype=torch.float32,
            ),
            "return_dict": False,
        }

        output_names = ["output"]
        dynamic_axes = {
            "hidden_states": {
                0: "batch_size",
                1: "num_channels",
                2: "latent_frames",
                3: "latent_height",
                4: "latent_width",
            },
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "rotary_emb": {1: "cl"},
            "temb": {0: "batch_size"},
            "timestep_proj": {0: "batch_size"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=False,  # condition_embedder holds learned weights needed on CPU at runtime (pipeline_wan.py denoise loop)
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(self, specializations, **compiler_options) -> None:
        self._compile(specializations=specializations, **compiler_options)


class QEffWanUnifiedTransformer(QEFFBaseModel):
    """
    Wrapper for WAN Unified Transformer with ONNX export and QAIC compilation capabilities.

    This class handles the unified WAN transformer model that combines high and low noise transformers
    into a single model for efficient deployment. Based on the timestep shape, the model dynamically
    selects between high and low noise transformers during inference.

    The wrapper applies specific transformations and optimizations for efficient inference on
    Qualcomm AI hardware, particularly for video diffusion models.

    Attributes:
        model (nn.Module): The QEffWanUnifiedWrapper model that combines high/low noise transformers
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [AttentionTransform, CustomOpsTransform, NormalizationTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, unified_transformer):
        """
        Initialize the Wan unified transformer.

        Args:
            model (nn.Module): Wan unified transformer model
        """
        super().__init__(unified_transformer)
        self.model = unified_transformer

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying Wan transformer model
        """
        return self.model.config.__dict__

    def get_onnx_params(self):
        """
        Generate ONNX export configuration for the Wan transformer.

        Creates example inputs for all Wan-specific inputs including hidden states,
        text embeddings, timestep conditioning,
        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        batch_size = constants.WAN_ONNX_EXPORT_BATCH_SIZE
        example_inputs = {
            # hidden_states = [ bs, in_channels, frames, latent_height, latent_width]
            "hidden_states": torch.randn(
                batch_size,
                self.model.config.in_channels,
                constants.WAN_ONNX_EXPORT_LATENT_FRAMES,
                constants.WAN_ONNX_EXPORT_LATENT_HEIGHT_45P,
                constants.WAN_ONNX_EXPORT_LATENT_WIDTH_45P,
                dtype=torch.float32,
            ),
            # encoder_hidden_states = [BS, seq len , text dim]
            "encoder_hidden_states": torch.randn(
                batch_size, constants.WAN_ONNX_EXPORT_SEQ_LEN, constants.WAN_TEXT_EMBED_DIM, dtype=torch.float32
            ),
            # Rotary position embeddings: [2, context_length, 1, rotary_dim]; 2 is from tuple of cos, sin freqs
            "rotary_emb": torch.randn(
                2, constants.WAN_ONNX_EXPORT_CL_45P, 1, constants.WAN_ONNX_EXPORT_ROTARY_DIM, dtype=torch.float32
            ),
            # Timestep embeddings: [batch_size=1, embedding_dim]
            "temb": torch.randn(batch_size, constants.WAN_TEXT_EMBED_DIM, dtype=torch.float32),
            # Projected timestep embeddings: [batch_size=1, projection_dim, embedding_dim]
            "timestep_proj": torch.randn(
                batch_size,
                constants.WAN_PROJECTION_DIM,
                constants.WAN_TEXT_EMBED_DIM,
                dtype=torch.float32,
            ),
            # Timestep parameter: Controls high/low noise transformer selection based on shape
            "tsp": torch.ones(1, dtype=torch.int64),
        }

        output_names = ["output"]

        dynamic_axes = {
            "hidden_states": {
                0: "batch_size",
                1: "num_channels",
                2: "latent_frames",
                3: "latent_height",
                4: "latent_width",
            },
            "timestep": {0: "steps"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "rotary_emb": {1: "cl"},
            "tsp": {0: "model_type"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """Export the Wan transformer model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            use_onnx_subfunctions (bool): Whether to export transformer blocks as ONNX functions
                                     for better modularity and potential optimization
        Returns:
            str: Path to the exported ONNX model
        """

        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=False,  # condition_embedder holds learned weights needed on CPU at runtime (pipeline_wan.py denoise loop)
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(self, specializations, **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options (e.g., num_cores, aic_num_of_activations)
        """
        self._compile(specializations=specializations, **compiler_options)
