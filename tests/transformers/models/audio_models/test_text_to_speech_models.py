# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import json
import os

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download

from QEfficient import QEffChatterboxMultilingualTTS
from QEfficient.utils import constants

# Chatterbox is an optional dependency (installed with --no-deps, see examples/audio/README.md)
drop_invalid_tokens = pytest.importorskip("chatterbox.models.s3tokenizer").drop_invalid_tokens
from chatterbox.mtl_tts import MULTILINGUAL_T3_MODELS  # noqa: E402

from QEfficient.tts.pipelines.chatterbox.pipeline_chatterbox import CHATTERBOX_CHECKPOINT_FILES  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/audio_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    test_models = json.load(f)["text_to_speech_models"]

TEXT, LANGUAGE, T3_STEPS = "Hello from Cloud AI 100.", "en", 40
# Measured on AI100 (fp16) vs upstream fp32 over 8 languages: T3 top-1 >= 98.98 %, cosine >= 0.99994; with the same
# speech tokens, audio log-spectral distance 0.15-0.29 dB. Waveform SNR is not checked: the vocoder turns tiny input
# differences into phase shifts.
# A step whose two best tokens are nearly tied may swap its argmax under fp16: allow one such step.
MAX_T3_TOP1_MISMATCHES, MIN_T3_COSINE, MAX_LSD_DB = 1, 0.999, 0.5


def _cfg_logits(logits: torch.Tensor, cfg_weight: float = 0.5) -> torch.Tensor:
    cond, uncond = logits[0], logits[1]
    return cond + cfg_weight * (cond - uncond)


def _log_spectral_distance_db(audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
    """Mean over speech frames (within 60 dB of the loudest) of the RMS dB difference of the 80-band mel spectra."""
    import librosa

    n = min(len(audio), len(reference))

    def mel_db(wav):
        mel = librosa.feature.melspectrogram(y=wav[:n], sr=sample_rate, n_fft=1024, hop_length=256, n_mels=80)
        return 10 * np.log10(np.maximum(mel, 1e-12)), mel.sum(0)

    reference_db, energy = mel_db(reference)
    audio_db, _ = mel_db(audio)
    floor = reference_db.max() - 80.0
    reference_db, audio_db = np.maximum(reference_db, floor), np.maximum(audio_db, floor)
    energy_db = 10 * np.log10(np.maximum(energy, 1e-12))
    speech = energy_db > energy_db.max() - 60.0
    return float(np.mean(np.sqrt(np.mean((reference_db - audio_db)[:, speech] ** 2, axis=0))))


def _keep_t3_layers(model, n_layer: int):
    """Keep the first ``n_layer`` T3 decoder layers (the reduced-layer lane); S3Gen stays whole."""
    model.t3.tfmr.layers = model.t3.tfmr.layers[:n_layer]
    model.t3.cfg.num_hidden_layers = model.t3.tfmr.config.num_hidden_layers = n_layer
    return model


def _one_bucket_config(path: str, ctx_len: int = 512) -> str:
    """The default compile config with only the smallest bucket of each S3Gen graph and a short T3 context."""
    with open(QEffChatterboxMultilingualTTS.get_default_config_path()) as fp:
        config = json.load(fp)
    for module_name, module in config["modules"].items():
        if module_name == "t3":
            for spec in module["specializations"]:
                spec["ctx_len"] = ctx_len
        else:
            module["specializations"] = module["specializations"][:1]
    with open(path, "w") as fp:
        json.dump(config, fp)
    return path


def check_text_to_speech_pytorch_vs_ai100(
    model_name: str, manual_cleanup: callable, n_layer: int = -1, compile_config: str = None
):
    """
    Chatterbox Multilingual V3 on AI100 against upstream ChatterboxMultilingualTTS on CPU (fp32):
    1. T3, teacher-forced along the upstream greedy (CFG) path: top-1 agreement and logit cosine.
    2. S3Gen on the same speech tokens with the same seed: log-spectral distance of the waveform. A reduced-layer T3
       does not speak, so then S3Gen gets the first 60 speech tokens of the built-in voice prompt instead.
    3. Full model only: generate() is finite, not silent, and as long as upstream makes it (40 ms per speech token,
       the last one dropped).
    """
    upstream_class = QEffChatterboxMultilingualTTS._hf_auto_class
    if n_layer < 0:
        pipeline = QEffChatterboxMultilingualTTS.from_pretrained(model_name, t3_model="v3")
        upstream = upstream_class.from_local(pipeline.checkpoint_dir, "cpu", t3_model="v3")
    else:
        checkpoint_dir = snapshot_download(
            model_name, allow_patterns=[MULTILINGUAL_T3_MODELS["v3"], *CHATTERBOX_CHECKPOINT_FILES]
        )
        model = _keep_t3_layers(upstream_class.from_local(checkpoint_dir, "cpu", t3_model="v3"), n_layer)
        pipeline = QEffChatterboxMultilingualTTS(model, pretrained_model_name_or_path=model_name, t3_model="v3")
        pipeline.checkpoint_dir = checkpoint_dir
        upstream = _keep_t3_layers(upstream_class.from_local(checkpoint_dir, "cpu", t3_model="v3"), n_layer)
    pipeline.compile(compile_config=compile_config)
    t3 = upstream.t3

    # 1. T3: device logits vs upstream fp32, both fed the upstream greedy tokens
    prefill = pipeline._t3_prefill_embeds(TEXT, LANGUAGE, cfg_weight=0.5)
    with torch.inference_mode():
        outputs = t3.tfmr(inputs_embeds=prefill, use_cache=True)
        reference, device, tokens = [], [], []
        device_logits = pipeline._t3_prefill(prefill)
        for step in range(T3_STEPS):
            ref = _cfg_logits(t3.speech_head(outputs.last_hidden_state[:, -1]))
            dev = _cfg_logits(torch.from_numpy(np.asarray(device_logits, dtype=np.float32))[:, -1])
            reference.append(ref)
            device.append(dev)
            token = int(ref.argmax())
            tokens.append(token)
            if token == t3.hp.stop_speech_token:
                break
            embed = t3.speech_emb(torch.tensor([[token]])) + t3.speech_pos_emb.get_fixed_embedding(step + 1)
            embeds = torch.cat([embed, embed])
            outputs = t3.tfmr(inputs_embeds=embeds, past_key_values=outputs.past_key_values, use_cache=True)
            device_logits = pipeline._t3_forward(embeds.numpy(), np.full((2, 1), prefill.shape[1] + step))
    reference, device = torch.stack(reference), torch.stack(device)
    mismatches = int((reference.argmax(-1) != device.argmax(-1)).sum())
    cosine = torch.nn.functional.cosine_similarity(reference, device, dim=-1).min().item()
    assert mismatches <= MAX_T3_TOP1_MISMATCHES, f"T3 top-1 disagrees on {mismatches} of {len(reference)} steps"
    assert cosine >= MIN_T3_COSINE, f"T3 min logit cosine {cosine:.6f}"

    # 2. S3Gen: same speech tokens, same random draws (flow noise, then the HiFT source), upstream on CPU
    if n_layer < 0:
        speech_tokens = drop_invalid_tokens(torch.tensor(tokens))
    else:
        speech_tokens = upstream.conds.gen["prompt_token"][0, :60]
    torch.manual_seed(0)
    device_wav = pipeline._s3gen(speech_tokens, perf=[])[0].numpy()
    torch.manual_seed(0)
    with torch.inference_mode():
        reference_wav, _ = upstream.s3gen.inference(speech_tokens=speech_tokens, ref_dict=upstream.conds.gen)
    trim = len(upstream.s3gen.trim_fade)  # upstream fades in the first 40 ms; compare after it
    lsd = _log_spectral_distance_db(device_wav[trim:], reference_wav[0].numpy()[trim:], pipeline.sample_rate)
    assert lsd <= MAX_LSD_DB, f"S3Gen log-spectral distance {lsd:.3f} dB"

    # 3. End to end
    if n_layer < 0:
        output = pipeline.generate(TEXT, language_id=LANGUAGE, seed=0)
        generated = drop_invalid_tokens(torch.from_numpy(output.speech_tokens))
        samples_per_token = output.sample_rate // constants.CHATTERBOX_SPEECH_TOKENS_PER_SECOND
        assert len(generated) > 0
        assert len(output.audio) == max(1, len(generated) - 1) * samples_per_token
        assert np.isfinite(output.audio).all()
        assert np.sqrt(np.mean(output.audio**2)) > 1e-3
    manual_cleanup([module.onnx_path for module in pipeline.modules.values()])


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model_name", test_models)
def test_full_text_to_speech_ai100(model_name, manual_cleanup):
    """The full model with the default config (needs two SoCs' worth of cores)."""
    check_text_to_speech_pytorch_vs_ai100(model_name, manual_cleanup)


@pytest.mark.on_qaic
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model_name", test_models)
def test_few_text_to_speech_ai100(model_name, manual_cleanup, tmp_path):
    """T3 with 2 layers and one bucket per S3Gen graph: the export/compile/run path in a few minutes."""
    config = _one_bucket_config(str(tmp_path / "chatterbox_one_bucket.json"))
    check_text_to_speech_pytorch_vs_ai100(model_name, manual_cleanup, n_layer=2, compile_config=config)
