# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""CPU-only unit tests for FLUX.2 Klein QEfficient integration."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from QEfficient.diffusers.pipelines.flux2 import pipeline_flux2_klein as flux2_klein_module
from QEfficient.diffusers.pipelines.flux2.pipeline_flux2_klein import (
    QEffFlux2KleinPipeline,
    compute_empirical_mu,
    retrieve_latents,
)


class _FakeLatentDist:
    def __init__(self, sample: torch.Tensor, mode: torch.Tensor):
        self._sample = sample
        self._mode = mode

    def sample(self, generator=None):
        del generator
        return self._sample

    def mode(self):
        return self._mode


class _FakeEncoderOutput:
    def __init__(self, latent_dist=None, latents=None):
        if latent_dist is not None:
            self.latent_dist = latent_dist
        if latents is not None:
            self.latents = latents


class _FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=4, block_out_channels=[1])
        self.encode_input = None
        self.decode_input = None

    def encode(self, image: torch.Tensor):
        self.encode_input = image
        latents = image.mean(dim=1, keepdim=True).repeat(1, self.config.latent_channels, 1, 1)
        return _FakeEncoderOutput(latent_dist=_FakeLatentDist(sample=latents + 1.0, mode=latents))

    def decode(self, latent_sample: torch.Tensor, return_dict: bool = False):
        self.decode_input = latent_sample
        decoded = latent_sample[:, :3]
        if return_dict:
            return SimpleNamespace(sample=decoded)
        return (decoded,)


def _make_minimal_flux2_pipeline() -> QEffFlux2KleinPipeline:
    pipeline = QEffFlux2KleinPipeline.__new__(QEffFlux2KleinPipeline)
    pipeline.vae_scale_factor = 8
    pipeline.model = SimpleNamespace(config=SimpleNamespace(is_distilled=False))
    return pipeline


class _FakeCompiledTextEncoder:
    def __init__(self):
        self.compile_calls = []
        self.device_ids = None
        self.qpc_path = None

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)
        self.qpc_path = "/compiled/text_encoder/qpc"


class _FakeTransformerSession:
    def __init__(self):
        self.inputs = None

    def run(self, inputs):
        self.inputs = inputs
        return {"sample": inputs["hidden_states"]}


def _make_text_encoder_compile_config(qpc_path=None):
    return {
        "modules": {
            "text_encoder": {
                "specializations": {
                    "batch_size": 2,
                    "seq_len": 384,
                    "ctx_len": 512,
                },
                "compilation": {
                    "onnx_path": "/tmp/text_encoder.onnx",
                    "compile_dir": "/tmp/text_encoder_compile",
                    "mdp_ts_num_devices": 2,
                    "mxfp6_matmul": True,
                    "convert_to_fp16": True,
                    "aic_num_cores": 12,
                },
                "execute": {
                    "device_ids": [0, 1],
                    "qpc_path": qpc_path,
                },
            }
        }
    }


def _make_tiny_flux2_transformer():
    try:
        from diffusers import Flux2Transformer2DModel

        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform, NormalizationTransform

        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=32,
            out_channels=32,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=48,
            timestep_guidance_channels=16,
            mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False,
        ).eval()
        model, _ = AttentionTransform.apply(model)
        model, _ = NormalizationTransform.apply(model)
        return model
    except Exception as exc:
        pytest.skip(f"Could not instantiate tiny Flux2Transformer2DModel: {exc}")


def _make_tiny_flux2_inputs(batch: int = 1, cl: int = 4, text_seq: int = 5):
    inner_dim = 16
    return {
        "hidden_states": torch.randn(batch, cl, 32),
        "encoder_hidden_states": torch.randn(batch, text_seq, 48),
        "timestep": torch.full((batch,), 0.5),
        "img_ids": torch.zeros(cl, 4),
        "txt_ids": torch.zeros(text_seq, 4),
        "adaln_double_img": torch.randn(1, inner_dim * 6),
        "adaln_double_txt": torch.randn(1, inner_dim * 6),
        "adaln_single": torch.randn(1, inner_dim * 3),
        "adaln_out": torch.randn(batch, inner_dim * 2),
        "return_dict": False,
    }


@pytest.mark.diffusers
class TestFlux2KleinHelpers:
    def test_compute_empirical_mu_small_sequence_formula(self):
        image_seq_len = 128
        num_steps = 20
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        expected = a * num_steps + (m_200 - 200.0 * a)
        assert mu == pytest.approx(expected)

    def test_compute_empirical_mu_large_sequence_formula(self):
        image_seq_len = 5000
        assert compute_empirical_mu(image_seq_len, 20) == pytest.approx(0.00016927 * image_seq_len + 0.45666666)

    def test_retrieve_latents_sample_mode_uses_latent_dist_sample(self):
        sample = torch.randn(1, 4, 2, 2)
        mode = torch.randn(1, 4, 2, 2)
        assert (
            retrieve_latents(_FakeEncoderOutput(latent_dist=_FakeLatentDist(sample, mode)), sample_mode="sample")
            is sample
        )

    def test_retrieve_latents_argmax_mode_uses_latent_dist_mode(self):
        sample = torch.randn(1, 4, 2, 2)
        mode = torch.randn(1, 4, 2, 2)
        assert (
            retrieve_latents(_FakeEncoderOutput(latent_dist=_FakeLatentDist(sample, mode)), sample_mode="argmax")
            is mode
        )

    def test_retrieve_latents_accepts_latents_attribute(self):
        latents = torch.randn(1, 4, 2, 2)
        assert retrieve_latents(_FakeEncoderOutput(latents=latents)) is latents

    def test_retrieve_latents_accepts_plain_tensor(self):
        latents = torch.randn(1, 4, 2, 2)
        assert retrieve_latents(latents) is latents

    def test_retrieve_latents_rejects_unknown_output(self):
        with pytest.raises(AttributeError):
            retrieve_latents(object())


@pytest.mark.diffusers
class TestFlux2TextEncoderCompilation:
    def test_init_wraps_text_encoder_without_compiling(self, monkeypatch):
        text_encoder_instances = []

        class FakeQEffWrapper:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def get_flux2_encoder_onnx_params(self):
                return {}, {}, []

        class FakeQEFFAutoModelForCausalLM:
            def __init__(self, model):
                self.model = model
                self.compile_calls = []
                text_encoder_instances.append(self)

            def compile(self, **kwargs):
                self.compile_calls.append(kwargs)

        monkeypatch.setattr(flux2_klein_module, "QEFFAutoModelForCausalLM", FakeQEFFAutoModelForCausalLM)
        monkeypatch.setattr(flux2_klein_module, "QEffFlux2TransformerModel", FakeQEffWrapper)
        monkeypatch.setattr(flux2_klein_module, "QEffVAE", FakeQEffWrapper)

        model = SimpleNamespace(
            text_encoder=SimpleNamespace(config=SimpleNamespace()),
            transformer=SimpleNamespace(),
            vae=_FakeVAE(),
            tokenizer=SimpleNamespace(),
            tokenizer_max_length=512,
            scheduler=SimpleNamespace(),
            image_processor=SimpleNamespace(),
        )

        pipeline = QEffFlux2KleinPipeline(model)

        assert model.text_encoder.config.is_flux2 is True
        assert pipeline.text_encoder is text_encoder_instances[0]
        assert pipeline.text_encoder.compile_calls == []
        assert pipeline.modules["text_encoder"] is pipeline.text_encoder

    def test_diffusion_modules_exclude_text_encoder(self):
        pipeline = QEffFlux2KleinPipeline.__new__(QEffFlux2KleinPipeline)
        pipeline.modules = {
            "text_encoder": object(),
            "transformer": object(),
            "vae_encoder": object(),
            "vae_decoder": object(),
        }

        assert set(pipeline._get_diffusion_modules()) == {"transformer", "vae_encoder", "vae_decoder"}

    def test_compile_text_encoder_from_config_uses_compile_config(self):
        pipeline = QEffFlux2KleinPipeline.__new__(QEffFlux2KleinPipeline)
        pipeline.text_encoder = _FakeCompiledTextEncoder()
        pipeline.custom_config = _make_text_encoder_compile_config()

        pipeline._compile_text_encoder_from_config()

        assert pipeline.text_encoder.device_ids == [0, 1]
        assert len(pipeline.text_encoder.compile_calls) == 1
        compile_call = pipeline.text_encoder.compile_calls[0]
        assert compile_call["onnx_path"] == "/tmp/text_encoder.onnx"
        assert compile_call["compile_dir"] == "/tmp/text_encoder_compile"
        assert compile_call["prefill_seq_len"] == 384
        assert compile_call["ctx_len"] == 512
        assert compile_call["batch_size"] == 2
        assert compile_call["prefill_only"] is True
        assert compile_call["num_devices"] == 2
        assert compile_call["num_cores"] == 12
        assert compile_call["mxfp6_matmul"] is True
        assert "convert_to_fp16" not in compile_call

    def test_compile_text_encoder_from_config_reuses_configured_qpc(self, tmp_path):
        qpc_path = tmp_path / "text_encoder_qpc"
        qpc_path.mkdir()
        pipeline = QEffFlux2KleinPipeline.__new__(QEffFlux2KleinPipeline)
        pipeline.text_encoder = _FakeCompiledTextEncoder()
        pipeline.custom_config = _make_text_encoder_compile_config(qpc_path=str(qpc_path))

        pipeline._compile_text_encoder_from_config()

        assert pipeline.text_encoder.qpc_path == str(qpc_path)
        assert pipeline.text_encoder.device_ids == [0, 1]
        assert pipeline.text_encoder.compile_calls == []


@pytest.mark.diffusers
class TestFlux2TransformerRuntimeInputs:
    def test_run_transformer_step_passes_complete_input_contract(self):
        session = _FakeTransformerSession()
        pipeline = QEffFlux2KleinPipeline.__new__(QEffFlux2KleinPipeline)
        pipeline.transformer = SimpleNamespace(qpc_session=session)

        latent_model_input = torch.randn(1, 4, 32)
        timestep = torch.tensor([500.0])
        prompt_embeds = torch.randn(1, 5, 48)
        text_ids = torch.zeros(1, 5, 4)
        latent_image_ids = torch.zeros(1, 4, 4)
        adaln_double_img = torch.randn(1, 96)
        adaln_double_txt = torch.randn(1, 96)
        adaln_single = torch.randn(1, 48)
        adaln_out = torch.randn(1, 32)

        output = pipeline._run_transformer_step(
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

        assert set(session.inputs) == {
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "txt_ids",
            "img_ids",
            "adaln_double_img",
            "adaln_double_txt",
            "adaln_single",
            "adaln_out",
        }
        assert torch.equal(torch.from_numpy(session.inputs["encoder_hidden_states"]), prompt_embeds)
        assert torch.equal(torch.from_numpy(output["sample"]), latent_model_input)


@pytest.mark.diffusers
class TestFlux2KleinPipelineGeometry:
    def test_prepare_text_ids_shape_and_coordinates(self):
        embeds = torch.randn(2, 5, 48)
        text_ids = QEffFlux2KleinPipeline._prepare_text_ids(embeds)
        assert text_ids.shape == (2, 5, 4)
        assert torch.equal(text_ids[0, :, 3], torch.arange(5, dtype=text_ids.dtype))
        assert torch.equal(text_ids[1, :, 3], torch.arange(5, dtype=text_ids.dtype))
        assert torch.equal(text_ids[:, :, :3], torch.zeros(2, 5, 3, dtype=text_ids.dtype))

    def test_prepare_latent_ids_shape_and_grid_coordinates(self):
        latents = torch.randn(2, 32, 3, 4)
        latent_ids = QEffFlux2KleinPipeline._prepare_latent_ids(latents)
        assert latent_ids.shape == (2, 12, 4)
        assert torch.equal(latent_ids[0, :, 1], torch.arange(3).repeat_interleave(4))
        assert torch.equal(latent_ids[0, :, 2], torch.arange(4).repeat(3))

    def test_prepare_image_ids_uses_time_offsets_and_concatenates_images(self):
        image_latents = [torch.randn(1, 32, 2, 3), torch.randn(1, 32, 1, 2)]
        image_ids = QEffFlux2KleinPipeline._prepare_image_ids(image_latents, scale=10)
        assert image_ids.shape == (1, 8, 4)
        assert torch.equal(image_ids[0, :6, 0], torch.full((6,), 10, dtype=image_ids.dtype))
        assert torch.equal(image_ids[0, 6:, 0], torch.full((2,), 20, dtype=image_ids.dtype))

    def test_prepare_image_ids_rejects_non_list(self):
        with pytest.raises(ValueError, match="Expected `image_latents` to be a list"):
            QEffFlux2KleinPipeline._prepare_image_ids(torch.randn(1, 32, 2, 2))

    def test_patchify_and_unpatchify_are_inverse(self):
        latents = torch.arange(1 * 8 * 4 * 6, dtype=torch.float32).reshape(1, 8, 4, 6)
        patchified = QEffFlux2KleinPipeline._patchify_latents(latents)
        restored = QEffFlux2KleinPipeline._unpatchify_latents(patchified)
        assert patchified.shape == (1, 32, 2, 3)
        assert torch.equal(restored, latents)

    def test_pack_and_unpack_latents_with_ids_are_inverse_for_grid_order(self):
        latents = torch.randn(1, 32, 3, 4)
        ids = QEffFlux2KleinPipeline._prepare_latent_ids(latents)
        packed = QEffFlux2KleinPipeline._pack_latents(latents)
        unpacked = QEffFlux2KleinPipeline._unpack_latents_with_ids(packed, ids, height=3, width=4)
        assert torch.allclose(unpacked, latents)

    def test_unpack_latents_with_ids_scatters_unsorted_tokens(self):
        latents = torch.randn(1, 32, 2, 2)
        ids = QEffFlux2KleinPipeline._prepare_latent_ids(latents)
        packed = QEffFlux2KleinPipeline._pack_latents(latents)
        permutation = torch.tensor([3, 1, 0, 2])
        unpacked = QEffFlux2KleinPipeline._unpack_latents_with_ids(
            packed[:, permutation], ids[:, permutation], height=2, width=2
        )
        assert torch.allclose(unpacked, latents)

    def test_prepare_latents_generates_expected_packed_shape_and_ids(self):
        pipeline = _make_minimal_flux2_pipeline()
        latents, latent_ids = pipeline.prepare_latents(
            batch_size=1,
            num_latents_channels=8,
            height=64,
            width=80,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(7),
        )
        assert latents.shape == (1, 20, 32)
        assert latent_ids.shape == (1, 20, 4)
        assert torch.isfinite(latents).all()

    def test_prepare_latents_uses_supplied_latents_and_casts_dtype(self):
        pipeline = _make_minimal_flux2_pipeline()
        supplied = torch.ones(1, 32, 4, 4, dtype=torch.float64)
        latents, latent_ids = pipeline.prepare_latents(
            batch_size=1,
            num_latents_channels=8,
            height=64,
            width=64,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu"),
            latents=supplied,
        )
        assert latents.dtype == torch.float32
        assert latents.shape == (1, 16, 32)
        assert latent_ids.shape == (1, 16, 4)
        assert torch.all(latents == 1)

    def test_prepare_latents_rejects_generator_list_with_wrong_length(self):
        pipeline = _make_minimal_flux2_pipeline()
        with pytest.raises(ValueError, match="list of generators of length"):
            pipeline.prepare_latents(
                batch_size=2,
                num_latents_channels=8,
                height=64,
                width=64,
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=[torch.Generator(device="cpu")],
            )


@pytest.mark.diffusers
class TestFlux2KleinInputValidation:
    def test_check_inputs_accepts_prompt_only(self):
        _make_minimal_flux2_pipeline().check_inputs(prompt="a cat", height=64, width=64, guidance_scale=1.0)

    def test_check_inputs_accepts_prompt_embeds_only(self):
        _make_minimal_flux2_pipeline().check_inputs(
            prompt=None, prompt_embeds=torch.randn(1, 5, 48), height=64, width=64
        )

    def test_check_inputs_rejects_prompt_and_prompt_embeds_together(self):
        with pytest.raises(ValueError, match="Cannot forward both `prompt`"):
            _make_minimal_flux2_pipeline().check_inputs(
                prompt="a cat", prompt_embeds=torch.randn(1, 5, 48), height=64, width=64
            )

    def test_check_inputs_rejects_missing_prompt_and_embeds(self):
        with pytest.raises(ValueError, match="Provide either `prompt` or `prompt_embeds`"):
            _make_minimal_flux2_pipeline().check_inputs(prompt=None, prompt_embeds=None, height=64, width=64)

    def test_check_inputs_rejects_bad_prompt_type(self):
        with pytest.raises(ValueError, match="`prompt` has to be of type"):
            _make_minimal_flux2_pipeline().check_inputs(prompt={"bad": "type"}, height=64, width=64)

    def test_check_inputs_rejects_unknown_callback_tensor_input(self):
        with pytest.raises(ValueError, match="callback_on_step_end_tensor_inputs"):
            _make_minimal_flux2_pipeline().check_inputs(
                prompt="a cat", height=64, width=64, callback_on_step_end_tensor_inputs=["not_supported"]
            )


@pytest.mark.diffusers
class TestFlux2VAEWrappers:
    def test_flux2_vae_encoder_wrapper_returns_latent_dist_mode(self):
        from QEfficient.diffusers.pipelines.pipeline_module import Flux2VaeEncoderWrapper

        vae = _FakeVAE()
        wrapper = Flux2VaeEncoderWrapper(vae)
        image = torch.randn(1, 3, 4, 4)
        latents = wrapper(image)
        assert latents.shape == (1, 4, 4, 4)
        assert vae.encode_input is image
        assert torch.allclose(latents, image.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1))

    def test_flux2_vae_encoder_wrapper_returns_latents_attribute_when_present(self):
        from QEfficient.diffusers.pipelines.pipeline_module import Flux2VaeEncoderWrapper

        class LatentsOnlyVAE(_FakeVAE):
            def encode(self, image: torch.Tensor):
                del image
                return _FakeEncoderOutput(latents=torch.ones(1, 4, 2, 2))

        assert torch.equal(Flux2VaeEncoderWrapper(LatentsOnlyVAE())(torch.randn(1, 3, 2, 2)), torch.ones(1, 4, 2, 2))

    def test_flux2_vae_decoder_wrapper_returns_first_decode_tensor(self):
        from QEfficient.diffusers.pipelines.pipeline_module import Flux2VaeDecoderWrapper

        vae = _FakeVAE()
        wrapper = Flux2VaeDecoderWrapper(vae)
        latent_sample = torch.randn(1, 4, 4, 4)
        decoded = wrapper(latent_sample)
        assert decoded.shape == (1, 3, 4, 4)
        assert vae.decode_input is latent_sample
        assert torch.equal(decoded, latent_sample[:, :3])


@pytest.mark.diffusers
class TestFlux2TransformMappingsAndWrappers:
    def test_flux2_transformer_module_is_importable(self):
        import QEfficient.diffusers.models.transformers.transformer_flux2 as transformer_flux2

        assert transformer_flux2 is not None

    def test_attention_transform_maps_flux2_classes(self):
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2Attention,
            Flux2ParallelSelfAttention,
            Flux2SingleTransformerBlock,
            Flux2Transformer2DModel,
            Flux2TransformerBlock,
        )

        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
        from QEfficient.diffusers.models.transformers.transformer_flux2 import (
            QEffFlux2Attention,
            QEffFlux2ParallelSelfAttention,
            QEffFlux2SingleTransformerBlock,
            QEffFlux2Transformer2DModel,
            QEffFlux2TransformerBlock,
        )

        assert AttentionTransform._module_mapping[Flux2Attention] is QEffFlux2Attention
        assert AttentionTransform._module_mapping[Flux2ParallelSelfAttention] is QEffFlux2ParallelSelfAttention
        assert AttentionTransform._module_mapping[Flux2TransformerBlock] is QEffFlux2TransformerBlock
        assert AttentionTransform._module_mapping[Flux2SingleTransformerBlock] is QEffFlux2SingleTransformerBlock
        assert AttentionTransform._module_mapping[Flux2Transformer2DModel] is QEffFlux2Transformer2DModel

    def test_qeff_flux2_transformer_model_has_expected_transforms(self):
        from QEfficient.diffusers.models.pytorch_transforms import (
            AttentionTransform,
            CustomOpsTransform,
            NormalizationTransform,
        )
        from QEfficient.diffusers.pipelines.pipeline_module import QEffFlux2TransformerModel

        assert AttentionTransform in QEffFlux2TransformerModel._pytorch_transforms
        assert NormalizationTransform in QEffFlux2TransformerModel._pytorch_transforms
        assert CustomOpsTransform in QEffFlux2TransformerModel._pytorch_transforms

    def test_qeff_flux2_transformer_get_onnx_params_contract(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffFlux2TransformerModel

        wrapper = QEffFlux2TransformerModel(_make_tiny_flux2_transformer())
        inputs, dynamic_axes, output_names = wrapper.get_onnx_params(batch_size=1, seq_length=5, cl=4)
        assert output_names == ["sample"]
        assert inputs["hidden_states"].shape == (1, 4, 32)
        assert inputs["encoder_hidden_states"].shape == (1, 5, 48)
        assert inputs["img_ids"].shape == (4, 4)
        assert inputs["txt_ids"].shape == (5, 4)
        assert inputs["adaln_double_img"].shape == (1, 96)
        assert inputs["adaln_double_txt"].shape == (1, 96)
        assert inputs["adaln_single"].shape == (1, 48)
        assert inputs["adaln_out"].shape == (1, 32)
        assert dynamic_axes["hidden_states"] == {0: "batch_size", 1: "cl"}
        assert dynamic_axes["encoder_hidden_states"] == {0: "batch_size", 1: "seq_len"}
        assert dynamic_axes["sample"] == {0: "batch_size", 1: "cl"}

    def test_qeff_vae_flux2_encoder_onnx_params_contract(self):
        from QEfficient.diffusers.pipelines.pipeline_module import Flux2VaeEncoderWrapper, QEffVAE

        wrapper = QEffVAE(Flux2VaeEncoderWrapper(_FakeVAE()), "encoder")
        inputs, dynamic_axes, output_names = wrapper.get_flux2_encoder_onnx_params()
        assert output_names == ["latents"]
        assert inputs["image"].shape == (1, 3, 480, 832)
        assert dynamic_axes["image"] == {0: "batch_size", 2: "height", 3: "width"}
        assert dynamic_axes["latents"] == {0: "batch_size", 2: "latent_height", 3: "latent_width"}


@pytest.mark.diffusers
@pytest.mark.accuracy
class TestFlux2TransformerBlocks:
    def test_qeff_flux2_transformer_2d_model_wraps_without_error(self):
        from QEfficient.diffusers.models.transformers.transformer_flux2 import QEffFlux2Transformer2DModel

        assert isinstance(_make_tiny_flux2_transformer(), QEffFlux2Transformer2DModel)

    def test_qeff_flux2_transformer_2d_model_forward_returns_expected_shape(self):
        model = _make_tiny_flux2_transformer()
        with torch.no_grad():
            out = model(**_make_tiny_flux2_inputs(batch=1, cl=4, text_seq=5))
        sample = out[0] if isinstance(out, (tuple, list)) else out.sample
        assert sample.shape == (1, 4, 32)
        assert torch.isfinite(sample).all()

    def test_qeff_flux2_transformer_2d_model_accepts_batched_ids(self):
        model = _make_tiny_flux2_transformer()
        inputs = _make_tiny_flux2_inputs(batch=1, cl=4, text_seq=5)
        inputs["img_ids"] = inputs["img_ids"].unsqueeze(0)
        inputs["txt_ids"] = inputs["txt_ids"].unsqueeze(0)
        with torch.no_grad():
            out = model(**inputs)
        sample = out[0] if isinstance(out, (tuple, list)) else out.sample
        assert sample.shape == (1, 4, 32)
        assert torch.isfinite(sample).all()

    def test_qeff_flux2_transformer_2d_model_is_deterministic(self):
        model = _make_tiny_flux2_transformer()
        inputs = _make_tiny_flux2_inputs(batch=1, cl=4, text_seq=5)
        with torch.no_grad():
            out1 = model(**inputs)
            out2 = model(**inputs)
        sample1 = out1[0] if isinstance(out1, (tuple, list)) else out1.sample
        sample2 = out2[0] if isinstance(out2, (tuple, list)) else out2.sample
        assert torch.allclose(sample1, sample2)

    def test_qeff_flux2_transformer_2d_model_get_submodules_for_export(self):
        from QEfficient.diffusers.models.transformers.transformer_flux2 import (
            QEffFlux2SingleTransformerBlock,
            QEffFlux2TransformerBlock,
        )

        submodules = _make_tiny_flux2_transformer().get_submodules_for_export()
        assert QEffFlux2TransformerBlock in submodules
        assert QEffFlux2SingleTransformerBlock in submodules

    def test_qeff_flux2_attention_processors_replace_originals(self):
        from QEfficient.diffusers.models.transformers.transformer_flux2 import (
            QEffFlux2Attention,
            QEffFlux2AttnProcessor,
            QEffFlux2ParallelSelfAttention,
            QEffFlux2ParallelSelfAttnProcessor,
        )

        model = _make_tiny_flux2_transformer()
        found_joint_attention = False
        found_parallel_attention = False
        for module in model.modules():
            if isinstance(module, QEffFlux2Attention):
                found_joint_attention = True
                assert isinstance(module.processor, QEffFlux2AttnProcessor)
            if isinstance(module, QEffFlux2ParallelSelfAttention):
                found_parallel_attention = True
                assert isinstance(module.processor, QEffFlux2ParallelSelfAttnProcessor)
        assert found_joint_attention
        assert found_parallel_attention

    def test_qeff_apply_rotary_emb_matches_real_valued_reference(self):
        from QEfficient.diffusers.models.transformers.transformer_flux2 import qeff_apply_rotary_emb

        x = torch.randn(1, 4, 2, 8)
        cos = torch.randn(4, 8)
        sin = torch.randn(4, 8)
        x_real, x_imag = x.reshape(1, 4, 2, 4, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        expected = x * cos[None, :, None, :] + x_rotated * sin[None, :, None, :]
        assert torch.allclose(qeff_apply_rotary_emb(x, (cos, sin)), expected)
