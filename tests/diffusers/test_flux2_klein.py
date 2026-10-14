# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""QAIC integration tests for the QEfficient FLUX.2 Klein pipeline modules.

The tests use tiny random-weight FLUX.2 components so they exercise export,
compile, QAIC execution, and PyTorch-vs-QAIC MAD validation without downloading
full FLUX.2 Klein checkpoint weights.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel

from QEfficient.diffusers.pipelines.flux2.pipeline_flux2_klein import QEffFlux2KleinPipeline, to_numpy
from QEfficient.diffusers.pipelines.pipeline_module import (
    Flux2VaeDecoderWrapper,
    Flux2VaeEncoderWrapper,
    QEffFlux2TransformerModel,
    QEffVAE,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from tests.diffusers.diffusers_utils import MADValidator, release_pipeline_qpc_sessions

TEST_SEED = 42
TINY_HEIGHT = 16
TINY_WIDTH = 16
TINY_TEXT_SEQ_LEN = 6
TINY_PROMPT_DIM = 48
TINY_LATENT_CHANNELS = 8
TINY_TRANSFORMER_IN_CHANNELS = TINY_LATENT_CHANNELS * 4
TINY_CL = (TINY_HEIGHT // 2) * (TINY_WIDTH // 2)


def _build_tiny_flux2_klein_pipeline() -> QEffFlux2KleinPipeline:
    """Build a minimal QEffFlux2KleinPipeline object with tiny random-weight modules."""
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)

    vae = AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        latent_channels=TINY_LATENT_CHANNELS,
        norm_num_groups=8,
        sample_size=TINY_HEIGHT,
        patch_size=(2, 2),
        mid_block_add_attention=False,
    ).eval()

    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=TINY_TRANSFORMER_IN_CHANNELS,
        out_channels=TINY_TRANSFORMER_IN_CHANNELS,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=TINY_PROMPT_DIM,
        timestep_guidance_channels=16,
        mlp_ratio=2.0,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    ).eval()

    pipeline = QEffFlux2KleinPipeline.__new__(QEffFlux2KleinPipeline)
    pipeline.model = SimpleNamespace(_execution_device=torch.device("cpu"), config=SimpleNamespace(is_distilled=True))
    pipeline.transformer = QEffFlux2TransformerModel(transformer)
    transformer_get_onnx_params = pipeline.transformer.get_onnx_params
    pipeline.transformer.get_onnx_params = lambda: transformer_get_onnx_params(
        batch_size=1,
        seq_length=TINY_TEXT_SEQ_LEN,
        cl=TINY_CL,
    )
    pipeline.vae_decoder = QEffVAE(Flux2VaeDecoderWrapper(vae), "decoder")
    pipeline.vae_encoder = QEffVAE(Flux2VaeEncoderWrapper(vae), "encoder")
    pipeline.vae_encoder.get_onnx_params = pipeline.vae_encoder.get_flux2_encoder_onnx_params
    pipeline.modules = {
        "transformer": pipeline.transformer,
        "vae_encoder": pipeline.vae_encoder,
        "vae_decoder": pipeline.vae_decoder,
    }
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.vae = vae
    pipeline.vae_scale_factor = 1
    pipeline.default_sample_size = TINY_HEIGHT
    pipeline._transformer_session = None
    pipeline._vae_session = None
    pipeline._vae_encode_session = None
    return pipeline


def _write_tiny_flux2_config(tmp_path: Path) -> str:
    """Write a compile config aligned with the tiny random-weight FLUX.2 modules."""
    config = {
        "description": "Tiny FLUX.2 Klein QAIC test configuration",
        "modules": {
            "transformer": {
                "specializations": [{"batch_size": 1, "cl": TINY_CL, "seq_len": TINY_TEXT_SEQ_LEN}],
                "compilation": {
                    "onnx_path": None,
                    "compile_dir": None,
                    "mdp_ts_num_devices": 1,
                    "mxfp6_matmul": False,
                    "convert_to_fp16": True,
                    "aic_num_cores": 16,
                },
                "execute": {"device_ids": None, "qpc_path": None},
            },
            "vae_decoder": {
                "specializations": {
                    "batch_size": 1,
                    "channels": TINY_LATENT_CHANNELS,
                    "latent_height": TINY_HEIGHT,
                    "latent_width": TINY_WIDTH,
                },
                "compilation": {
                    "onnx_path": None,
                    "compile_dir": None,
                    "mdp_ts_num_devices": 1,
                    "mxfp6_matmul": False,
                    "convert_to_fp16": True,
                    "aic_num_cores": 16,
                },
                "execute": {"device_ids": None, "qpc_path": None},
            },
            "vae_encoder": {
                "specializations": {"batch_size": 1, "height": TINY_HEIGHT, "width": TINY_WIDTH},
                "compilation": {
                    "onnx_path": None,
                    "compile_dir": None,
                    "mdp_ts_num_devices": 1,
                    "mxfp6_matmul": False,
                    "convert_to_fp16": True,
                    "aic_num_cores": 16,
                },
                "execute": {"device_ids": None, "qpc_path": None},
            },
        },
    }
    config_path = tmp_path / "flux2_klein_tiny_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def _build_transformer_inputs(pipeline: QEffFlux2KleinPipeline):
    prompt_embeds = torch.randn(1, TINY_TEXT_SEQ_LEN, TINY_PROMPT_DIM, dtype=torch.float32)
    text_ids = pipeline._prepare_text_ids(prompt_embeds).to(torch.float32)
    latents, latent_ids = pipeline.prepare_latents(
        batch_size=1,
        num_latents_channels=TINY_LATENT_CHANNELS,
        height=TINY_HEIGHT,
        width=TINY_WIDTH,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(TEST_SEED),
    )
    timestep = torch.ones(1, dtype=torch.float32)

    transformer_model = pipeline.transformer.model
    with torch.no_grad():
        temb = transformer_model.time_guidance_embed(timestep, None)
        double_mod_img = transformer_model.double_stream_modulation_img(temb)
        double_mod_txt = transformer_model.double_stream_modulation_txt(temb)
        single_mod = transformer_model.single_stream_modulation(temb)
        adaln_out = transformer_model.norm_out.linear(transformer_model.norm_out.silu(temb))

    adaln_double_img = (
        double_mod_img.squeeze(0).unsqueeze(0).expand(transformer_model.config.num_layers, -1).contiguous()
    )
    adaln_double_txt = (
        double_mod_txt.squeeze(0).unsqueeze(0).expand(transformer_model.config.num_layers, -1).contiguous()
    )
    adaln_single = (
        single_mod.squeeze(0).unsqueeze(0).expand(transformer_model.config.num_single_layers, -1).contiguous()
    )

    torch_inputs = {
        "hidden_states": latents,
        "timestep": timestep / 1000,
        "encoder_hidden_states": prompt_embeds,
        "txt_ids": text_ids,
        "img_ids": latent_ids.to(torch.float32),
        "adaln_double_img": adaln_double_img,
        "adaln_double_txt": adaln_double_txt,
        "adaln_single": adaln_single,
        "adaln_out": adaln_out,
        "return_dict": False,
    }
    qaic_inputs = {k: to_numpy(v) for k, v in torch_inputs.items() if k != "return_dict"}
    return torch_inputs, qaic_inputs, latent_ids


@pytest.fixture(scope="function")
def compiled_tiny_flux2_pipeline(tmp_path):
    pipeline = _build_tiny_flux2_klein_pipeline()
    config_path = _write_tiny_flux2_config(tmp_path)
    try:
        pipeline.compile(compile_config=config_path, parallel=False, height=TINY_HEIGHT, width=TINY_WIDTH)
        yield pipeline
    finally:
        release_pipeline_qpc_sessions(pipeline, ["transformer", "vae_encoder", "vae_decoder"])


@pytest.mark.flux
@pytest.mark.flux2
@pytest.mark.diffusion_models
@pytest.mark.on_qaic
def test_flux2_klein_tiny_modules_export_compile_and_execute_on_qaic(compiled_tiny_flux2_pipeline):
    """Validate tiny FLUX.2 Klein transformer, VAE encoder, and VAE decoder on QAIC."""
    pipeline = compiled_tiny_flux2_pipeline
    mad_validator = MADValidator(
        tolerances={
            "transformer": 0.35,
            "vae_encoder": 0.05,
            "vae_decoder": 0.05,
        }
    )

    for module_name in ["transformer", "vae_encoder", "vae_decoder"]:
        module = getattr(pipeline, module_name)
        assert module.onnx_path is not None, f"{module_name} ONNX path was not set"
        assert module.qpc_path is not None, f"{module_name} QPC path was not set"
        assert Path(module.onnx_path).exists(), f"Missing {module_name} ONNX: {module.onnx_path}"
        assert Path(module.qpc_path).exists(), f"Missing {module_name} QPC: {module.qpc_path}"

    # Transformer QAIC vs transformed PyTorch reference.
    torch_inputs, qaic_inputs, _ = _build_transformer_inputs(pipeline)
    with torch.no_grad():
        transformer_ref = pipeline.transformer.model(**torch_inputs)[0]

    pipeline.transformer.qpc_session = QAICInferenceSession(
        str(pipeline.transformer.qpc_path), device_ids=pipeline.transformer.device_ids
    )
    transformer_out = pipeline.transformer.qpc_session.run(qaic_inputs)["sample"]
    assert transformer_out.shape == tuple(transformer_ref.shape)
    mad_validator.validate_module_mad(transformer_ref, transformer_out, "transformer")

    # VAE encoder QAIC vs PyTorch reference.
    image = torch.randn(1, 3, TINY_HEIGHT, TINY_WIDTH, dtype=torch.float32)
    with torch.no_grad():
        vae_encoder_ref = pipeline.vae.encode(image).latent_dist.mode()

    pipeline.vae_encoder.qpc_session = QAICInferenceSession(
        str(pipeline.vae_encoder.qpc_path), device_ids=pipeline.vae_encoder.device_ids
    )
    vae_encoder_out = pipeline.vae_encoder.qpc_session.run({"image": image.numpy()})["latents"]
    assert vae_encoder_out.shape == tuple(vae_encoder_ref.shape)
    mad_validator.validate_module_mad(vae_encoder_ref, vae_encoder_out, "vae_encoder")

    # VAE decoder QAIC vs PyTorch reference.
    latent_sample = torch.randn(1, TINY_LATENT_CHANNELS, TINY_HEIGHT, TINY_WIDTH, dtype=torch.float32)
    with torch.no_grad():
        vae_decoder_ref = pipeline.vae.decode(latent_sample, return_dict=False)[0]

    pipeline.vae_decoder.qpc_session = QAICInferenceSession(
        str(pipeline.vae_decoder.qpc_path), device_ids=pipeline.vae_decoder.device_ids
    )
    vae_decoder_out = pipeline.vae_decoder.qpc_session.run({"latent_sample": latent_sample.numpy()})["sample"]
    assert vae_decoder_out.shape == tuple(vae_decoder_ref.shape)
    mad_validator.validate_module_mad(vae_decoder_ref, vae_decoder_out, "vae_decoder")


@pytest.mark.flux
@pytest.mark.flux2
@pytest.mark.diffusion_models
@pytest.mark.on_qaic
def test_flux2_klein_tiny_one_step_latent_flow_on_qaic(compiled_tiny_flux2_pipeline):
    """Run one denoising-like transformer step and verify Flux2 latent unpacking flow."""
    pipeline = compiled_tiny_flux2_pipeline
    torch_inputs, qaic_inputs, latent_ids = _build_transformer_inputs(pipeline)

    pipeline.transformer.qpc_session = QAICInferenceSession(
        str(pipeline.transformer.qpc_path), device_ids=pipeline.transformer.device_ids
    )
    noise_pred = torch.from_numpy(pipeline.transformer.qpc_session.run(qaic_inputs)["sample"])
    latents = torch_inputs["hidden_states"] - 0.1 * noise_pred

    unpacked = pipeline._unpack_latents_with_ids(latents, latent_ids, TINY_HEIGHT // 2, TINY_WIDTH // 2)
    assert unpacked.shape == (1, TINY_TRANSFORMER_IN_CHANNELS, TINY_HEIGHT // 2, TINY_WIDTH // 2)
    restored_latents = pipeline._unpatchify_latents(unpacked)
    assert restored_latents.shape == (1, TINY_LATENT_CHANNELS, TINY_HEIGHT, TINY_WIDTH)
    assert torch.isfinite(restored_latents).all()
