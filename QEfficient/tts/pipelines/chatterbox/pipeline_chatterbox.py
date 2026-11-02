# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers.generation.logits_process import MinPLogitsWarper, TopPLogitsWarper

try:
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.mtl_tts import MULTILINGUAL_T3_MODELS, SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS, punc_norm
except ImportError as error:
    raise ImportError(
        "QEffChatterboxMultilingualTTS needs Resemble AI's Chatterbox at git commit 5de7a54 (Multilingual V3 is not "
        "in any PyPI release). Its pins conflict with QEfficient's, so install it without dependencies: "
        "pip install --no-deps "
        "'chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox@5de7a54aa4e5e2baadb0182dde554908b48b85c2', "
        "then its runtime dependencies: s3tokenizer conformer einops omegaconf librosa resemble-perth pykakasi "
        "spacy-pkuseg 'setuptools<81' (see examples/audio/README.md)."
    ) from error

from QEfficient.diffusers.pipelines.pipeline_utils import ModulePerf, config_manager, set_execute_params
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.tts.models.chatterbox.modeling_chatterbox import QEffHiFTDecodeWrapper, QEffS3GenFlowEncoderWrapper
from QEfficient.tts.models.chatterbox.modeling_t3 import QEffChatterboxT3DecoderWrapper
from QEfficient.tts.pipelines.pipeline_module import (
    QEffHiFTDecoder,
    QEffHiFTF0Predictor,
    QEffS3GenFlowEncoder,
    QEffS3GenFlowEstimator,
)
from QEfficient.tts.pipelines.pipeline_t3 import QEffChatterboxT3
from QEfficient.tts.pipelines.pipeline_utils import QEffTTSOutput, check_bucket_config, select_bucket
from QEfficient.utils import constants
from QEfficient.utils.export_utils import export_from_compile
from QEfficient.utils.logging_utils import QEFFLogger, log_pipeline_api

logger = QEFFLogger.get_logger("MODEL")

# Files that ChatterboxMultilingualTTS.from_local reads, besides the T3 checkpoint
CHATTERBOX_CHECKPOINT_FILES = [
    "ve.pt",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]


def _load_local_cangjie(tokenizer, checkpoint_dir: str) -> None:
    """
    Fill the Chinese Cangjie table from ``<checkpoint_dir>/Cangjie5_TC.json`` if upstream could not load it.

    Upstream ``ChineseCangjieConverter`` fetches the table with ``hf_hub_download(cache_dir=<checkpoint dir>)``, which
    ignores the copy already in the checkpoint directory; offline (``HF_HUB_OFFLINE=1``, air-gapped servers, a
    ``--local-dir`` download) it only logs a warning and Chinese characters become ``[UNK]`` tokens.
    """
    converter = getattr(tokenizer, "cangjie_converter", None)
    path = os.path.join(checkpoint_dir, "Cangjie5_TC.json")
    if converter is None or converter.word2cj or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fp:
        for entry in json.load(fp):  # same parsing as ChineseCangjieConverter._load_cangjie_mapping
            word, code = entry.split("\t")[:2]
            converter.word2cj[word] = code
            converter.cj2word.setdefault(code, []).append(word)
    logger.info(f"Loaded the Cangjie table for Chinese from {path}.")


class QEffChatterboxMultilingualTTS:
    """
    Resemble AI Chatterbox Multilingual (text-to-speech) on Cloud AI 100.

    Five graphs run on the device: T3 (acoustic language model, KV cache retained on the device, classifier-free
    guidance as a batch of 2), and S3Gen's flow encoder, flow estimator (one call per Euler step), HiFT F0 predictor
    and HiFT decoder, each compiled with a specialization per length bucket. The host runs the upstream text
    processing, T3 conditioning and sampling, the Euler loop, the HiFT excitation source and the Perth watermark,
    drawing random numbers in the same order as upstream ``ChatterboxMultilingualTTS.generate``.

    Supported today: one request at a time (no continuous batching: T3's batch holds the two guidance rows of one
    request), graphs exported in float32 (the checkpoint's dtype) and compiled to fp16 (``convert_to_fp16``; AI100 has
    no bf16), optional ONNX subfunctions for T3. The default config gives T3 16 cores and the four S3Gen graphs 16
    cores together, and lets the runtime pick the devices. Voice conditioning from ``audio_prompt_path`` runs upstream
    code on the CPU.

    ``self.model`` (the wrapped upstream ``ChatterboxMultilingualTTS``) is transformed in place for export (QEff Llama
    layers, folded weight norm) and its exported weights may be offloaded, so ``self.model.generate()`` no longer works
    (``T3.inference`` raises). Load a separate upstream instance to compare against.

    Known limitation: when the speech is padded to a length bucket, the last ~5 mel frames (about 100 ms) of audio can
    differ slightly from upstream.

    Example:
        >>> from QEfficient import QEffChatterboxMultilingualTTS
        >>> pipeline = QEffChatterboxMultilingualTTS.from_pretrained("ResembleAI/chatterbox", t3_model="v3")
        >>> pipeline.compile()
        >>> output = pipeline.generate("Bonjour, comment allez-vous?", language_id="fr", seed=0)
    """

    _hf_auto_class = ChatterboxMultilingualTTS

    def __init__(self, model: ChatterboxMultilingualTTS, **kwargs):
        self.model = model
        self.sample_rate = model.sr
        hash_kwargs = {key: kwargs[key] for key in ("pretrained_model_name_or_path", "revision") if key in kwargs}
        self.t3 = QEffChatterboxT3(QEffChatterboxT3DecoderWrapper(model.t3), **hash_kwargs)
        self.t3.hash_params["t3_model"] = kwargs.get("t3_model")
        flow, hift = model.s3gen.flow, model.s3gen.mel2wav
        self.flow_encoder = QEffS3GenFlowEncoder(QEffS3GenFlowEncoderWrapper(flow), **hash_kwargs)
        self.flow_estimator = QEffS3GenFlowEstimator(flow.decoder.estimator, **hash_kwargs)
        self.hift_f0 = QEffHiFTF0Predictor(hift.f0_predictor, **hash_kwargs)
        self.hift_decoder = QEffHiFTDecoder(QEffHiFTDecodeWrapper(hift), **hash_kwargs)
        self.modules = {
            "t3": self.t3,
            "flow_encoder": self.flow_encoder,
            "flow_estimator": self.flow_estimator,
            "hift_f0": self.hift_f0,
            "hift_decoder": self.hift_decoder,
        }
        self.custom_config = None
        self.checkpoint_dir = None  # set by from_pretrained

    @classmethod
    @log_pipeline_api("from_pretrained", "Text-to-speech model loading completed.", "load_complete")
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        t3_model: str = "v3",
        revision: Optional[str] = None,
    ):
        """
        Load Chatterbox Multilingual on CPU in float32 from the Hugging Face Hub or a local checkpoint directory.

        Args:
            pretrained_model_name_or_path (str): Hub repo id or local directory with the checkpoint files.
            t3_model (str): Multilingual T3 checkpoint: ``"v3"`` (default, validated), ``"v2"`` (untested) or a
                ``.safetensors`` name.
            revision (str, optional): Hub revision to pin.
        """
        t3_file = MULTILINGUAL_T3_MODELS.get(t3_model, t3_model)
        if os.path.isdir(pretrained_model_name_or_path):
            checkpoint_dir = pretrained_model_name_or_path
        else:
            checkpoint_dir = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                revision=revision,
                allow_patterns=[t3_file, *CHATTERBOX_CHECKPOINT_FILES],
                token=os.getenv("HF_TOKEN"),
            )
        model = cls._hf_auto_class.from_local(checkpoint_dir, "cpu", t3_model=t3_model)
        _load_local_cangjie(model.tokenizer, str(checkpoint_dir))
        pipeline = cls(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=revision,
            t3_model=t3_model,
        )
        pipeline.checkpoint_dir = str(checkpoint_dir)
        return pipeline

    @staticmethod
    def get_default_config_path() -> str:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "chatterbox_config.json")

    @log_pipeline_api("export", "ONNX export completed.", "export_complete")
    def export(self, export_dir: Optional[str] = None, use_onnx_subfunctions: bool = False) -> None:
        """
        Export every graph to ONNX. ``use_onnx_subfunctions`` applies to T3 (30 repeated decoder layers).
        """
        for module_name, module_obj in self.modules.items():
            if module_obj.qpc_path is not None:
                continue
            example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params()
            module_obj.export(
                inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=export_dir,
                use_onnx_subfunctions=use_onnx_subfunctions and module_name == "t3",
            )

    @log_pipeline_api("compile", "Compilation completed.", "compile_complete")
    def compile(self, compile_config: Optional[str] = None, use_onnx_subfunctions: bool = False) -> None:
        """
        Compile every graph with the specializations, cores and devices of ``compile_config`` (JSON; default:
        ``configs/chatterbox_config.json``). Modules that already have a ``qpc_path`` in the config are not recompiled.
        """
        for module_obj in self.modules.values():  # sessions are bound to the QPCs of the previous compile
            if module_obj.qpc_session is not None:
                module_obj.qpc_session.deactivate()
                module_obj.qpc_session = None
        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)
        check_bucket_config(self.custom_config)
        self.custom_config["modules"]["t3"]["compilation"]["use_onnx_subfunctions"] = use_onnx_subfunctions
        set_execute_params(self)
        if any(module.onnx_path is None and module.qpc_path is None for module in self.modules.values()):
            with export_from_compile():
                self.export(use_onnx_subfunctions=use_onnx_subfunctions)
        # Compiled here rather than with QEfficient.diffusers' compile_modules_sequential: it raises on these list
        # specializations without per-bucket updates, and with empty updates renames the buckets <module>_0..n.
        for module_name, module_obj in self.modules.items():
            if module_obj.qpc_path is not None:
                continue
            module_config = self.custom_config["modules"][module_name]
            module_obj.compile(
                specializations=[dict(spec) for spec in module_config["specializations"]],
                **dict(module_config["compilation"]),
            )

    # ------------------------------------------------------------------ device execution
    def _session(self, module_name: str) -> QAICInferenceSession:
        module_obj = self.modules[module_name]
        if module_obj.qpc_session is None:
            session = QAICInferenceSession(str(module_obj.qpc_path), device_ids=module_obj.device_ids)
            if module_name == "t3":  # the KV cache stays on the device between calls
                session.skip_buffers([name for name in session.input_names + session.output_names if "past_" in name])
            module_obj.qpc_session = session
        return module_obj.qpc_session

    def _activate(self) -> None:
        """Open every graph's session, loading the five QPCs onto their devices."""
        for module_name in self.modules:
            self._session(module_name)

    def _run(self, module_name: str, output_shapes: Dict[str, tuple], **inputs) -> Dict[str, np.ndarray]:
        """Run one graph; output buffers sized for this call select the matching specialization (bucket)."""
        session = self._session(module_name)
        session.set_buffers({name: np.zeros(shape, dtype=np.float32) for name, shape in output_shapes.items()})
        try:
            return session.run({name: np.asarray(value) for name, value in inputs.items()})
        except Exception:
            self.modules[module_name].qpc_session = None  # a failed run unloads the program: reopen it next time
            raise

    def _buckets(self, module_name: str, symbol: str) -> List[int]:
        return sorted(spec[symbol] for spec in self.custom_config["modules"][module_name]["specializations"])

    def _max_speech_tokens(self, prefill_len: int, max_new_tokens: int) -> int:
        """Largest number of speech tokens that fits T3's context and the largest S3Gen buckets."""
        ctx_len = min(spec["ctx_len"] for spec in self.custom_config["modules"]["t3"]["specializations"])
        if prefill_len >= ctx_len:
            raise ValueError(
                f"The text needs {prefill_len} prompt positions; T3 is compiled for a context of {ctx_len}."
            )
        num_prompt_tokens = self.model.conds.gen["prompt_token"].shape[-1]
        num_prompt_mel = self.model.conds.gen["prompt_feat"].shape[1]
        frames_per_token = constants.CHATTERBOX_MEL_FRAMES_PER_TOKEN
        limits = {
            "max_new_tokens": max_new_tokens,
            "T3 context": ctx_len - prefill_len,
            "flow token bucket": max(self._buckets("flow_encoder", "n_tok")) - num_prompt_tokens,
            "HiFT mel bucket": (max(self._buckets("hift_f0", "n_mel")) + num_prompt_mel) // frames_per_token
            - num_prompt_tokens,
        }
        name, limit = min(limits.items(), key=lambda item: item[1])
        if limit <= 0:
            raise ValueError(f"No room for speech tokens: {limits}")
        if name != "max_new_tokens":
            logger.warning(f"Generating at most {limit} speech tokens (limited by the {name}); compile larger buckets.")
        return limit

    # ------------------------------------------------------------------ T3: text -> speech tokens
    @torch.inference_mode()
    def _t3_prefill_embeds(self, text: str, language_id: Optional[str], cfg_weight: float) -> torch.Tensor:
        """Host part of ``T3.inference`` up to the first forward pass (models/t3/t3.py:255-313)."""
        t3 = self.model.t3
        language_id = language_id.lower() if language_id else None
        text_tokens = self.model.tokenizer.text_to_tokens(punc_norm(text), language_id=language_id)
        if text_tokens.shape[1] > t3.hp.max_text_tokens:  # upstream fails in text_pos_emb with an IndexError
            raise ValueError(
                f"The text is {text_tokens.shape[1]} tokens long; T3 accepts at most {t3.hp.max_text_tokens}. Split it."
            )
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # conditional and unconditional (CFG) rows
        text_tokens = F.pad(text_tokens, (1, 0), value=t3.hp.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=t3.hp.stop_text_token).long()
        start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(
            t3_cond=self.model.conds.t3, text_tokens=text_tokens, speech_tokens=start, cfg_weight=cfg_weight
        )
        # T3.inference appends the start-of-speech embedding once more before the first forward pass
        bos = t3.speech_emb(torch.tensor([[t3.hp.start_speech_token]])) + t3.speech_pos_emb.get_fixed_embedding(0)
        return torch.cat([embeds, torch.cat([bos, bos])], dim=1)

    def _t3_forward(self, inputs_embeds: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        vocab_size = self.model.t3.hp.speech_tokens_dict_size
        output_shapes = {"logits": (constants.CHATTERBOX_T3_CFG_BATCH_SIZE, 1, vocab_size)}
        return self._run(
            "t3", output_shapes, inputs_embeds=np.asarray(inputs_embeds, dtype=np.float32), position_ids=position_ids
        )["logits"]

    def _t3_prefill(self, prefill: torch.Tensor) -> np.ndarray:
        """Chunked prefill; padded positions get position id -1, as in QEfficient text generation."""
        chunk = max(spec["seq_len"] for spec in self.custom_config["modules"]["t3"]["specializations"])
        batch_size, length, hidden = prefill.shape
        num_chunks = -(-length // chunk)
        embeds = np.zeros((batch_size, num_chunks * chunk, hidden), dtype=np.float32)
        embeds[:, :length] = prefill.numpy()
        positions = np.arange(num_chunks * chunk)
        positions = np.repeat(np.where(positions < length, positions, -1)[None], batch_size, axis=0)
        for i in range(num_chunks):
            window = slice(i * chunk, (i + 1) * chunk)
            logits = self._t3_forward(embeds[:, window], positions[:, window])
        return logits

    @torch.inference_mode()
    def _t3_sample(
        self,
        prefill: torch.Tensor,
        *,
        max_new_tokens: int,
        cfg_weight: float,
        temperature: float,
        repetition_penalty: float,
        min_p: float,
        top_p: float,
        perf: List[ModulePerf],
    ) -> torch.Tensor:
        """Device forward passes with the upstream CFG mix, logits processors and sampling (models/t3/t3.py:315-390).

        Same tokens as the upstream loop for the same seed and logits (checked token for token), with less host work:
        the repetition penalty is one ``torch.where`` over a running mask of seen tokens (same per-element arithmetic
        as the processor's gather/scatter); ``TopPLogitsWarper`` is skipped when ``top_p >= 1`` (it then only masks
        tokens whose probability is already exactly 0); the next input embedding is two fp32 table rows added in
        numpy into a reused buffer. ``torch.multinomial(torch.softmax(...))`` is unchanged, so the random stream is
        consumed exactly as upstream.
        """
        t3 = self.model.t3
        prefill_len = prefill.shape[1]
        start = time.perf_counter()
        logits = self._t3_prefill(prefill)
        perf.append(ModulePerf("t3_prefill", time.perf_counter() - start))
        penalty = float(repetition_penalty)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        top_p_warper = TopPLogitsWarper(top_p=top_p) if top_p < 1.0 else None
        speech_table = t3.speech_emb.weight.detach().numpy()
        position_table = t3.speech_pos_emb.emb.weight.detach().numpy()
        seen = torch.zeros(1, speech_table.shape[0], dtype=torch.bool)
        seen[0, t3.hp.start_speech_token] = True
        embeds = np.empty((constants.CHATTERBOX_T3_CFG_BATCH_SIZE, 1, speech_table.shape[1]), dtype=np.float32)
        positions = np.empty((constants.CHATTERBOX_T3_CFG_BATCH_SIZE, 1), dtype=np.int64)
        predicted, step_times = [], []
        for step in range(max_new_tokens):
            step_logits = torch.from_numpy(np.asarray(logits, dtype=np.float32))[:, -1, :]
            cond, uncond = step_logits[0:1], step_logits[1:2]
            scores = cond + torch.as_tensor(cfg_weight, dtype=cond.dtype) * (cond - uncond)
            scores = torch.where(seen, torch.where(scores < 0, scores * penalty, scores / penalty), scores)
            if temperature != 1.0:
                scores = scores / temperature
            scores = min_p_warper(None, scores)
            if top_p_warper is not None:
                scores = top_p_warper(None, scores)
            next_token = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)
            predicted.append(next_token)
            token = int(next_token)
            seen[0, token] = True
            if token == t3.hp.stop_speech_token:
                break
            np.add(speech_table[token], position_table[step + 1], out=embeds[0, 0])
            embeds[1:] = embeds[0]
            positions.fill(prefill_len + step)
            start = time.perf_counter()
            logits = self._t3_forward(embeds, positions)
            step_times.append(time.perf_counter() - start)
        perf.append(ModulePerf("t3_decode", step_times))
        return torch.cat(predicted, dim=1)

    # ------------------------------------------------------------------ S3Gen: speech tokens -> waveform
    @torch.inference_mode()
    def _flow_euler(self, x, mu, mask, spks, cond, perf: List[ModulePerf]) -> torch.Tensor:
        """``ConditionalCFM.solve_euler`` with the estimator on the device (models/s3gen/flow_matching.py:78-145)."""
        cfg_rate = self.model.s3gen.flow.decoder.inference_cfg_rate
        t_span = torch.linspace(0, 1, constants.CHATTERBOX_CFM_TIMESTEPS + 1)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)  # cosine scheduler
        output_shapes = {"dxdt": (2, constants.CHATTERBOX_MEL_CHANNELS, x.shape[-1])}
        step_times = []
        for t, r in zip(t_span[:-1], t_span[1:]):
            start = time.perf_counter()
            dxdt = self._run(
                "flow_estimator",
                output_shapes,
                x=torch.cat([x, x]).numpy(),
                mask=torch.cat([mask, mask]).numpy(),
                mu=torch.cat([mu, torch.zeros_like(mu)]).numpy(),
                t=torch.stack([t, t]).numpy(),
                spks=torch.cat([spks, torch.zeros_like(spks)]).numpy(),
                cond=torch.cat([cond, torch.zeros_like(cond)]).numpy(),
            )["dxdt"]
            step_times.append(time.perf_counter() - start)
            dxdt, cfg_dxdt = torch.from_numpy(dxdt).split(1)
            x = x + (r - t) * ((1.0 + cfg_rate) * dxdt - cfg_rate * cfg_dxdt)
        perf.append(ModulePerf("flow_estimator", step_times))
        return x

    @torch.inference_mode()
    def _s3gen(self, speech_tokens: torch.Tensor, perf: List[ModulePerf]) -> torch.Tensor:
        """``S3Token2Wav.inference`` with the four graphs on the device; returns the waveform (1, samples)."""
        ref, flow, hift = self.model.conds.gen, self.model.s3gen.flow, self.model.s3gen.mel2wav
        channels = constants.CHATTERBOX_MEL_CHANNELS
        # Upstream raises an IndexError in flow.input_embedding here; the device gather would not fail
        if (speech_tokens >= flow.vocab_size).any():
            raise ValueError(f"Speech token {int(speech_tokens.max())} is out of range for S3Gen ({flow.vocab_size}).")
        tokens = torch.cat([ref["prompt_token"][0].long(), speech_tokens.long()])[None]
        num_tokens, num_prompt_mel = tokens.shape[1], ref["prompt_feat"].shape[1]
        num_frames = constants.CHATTERBOX_MEL_FRAMES_PER_TOKEN * num_tokens
        num_mel = num_frames - num_prompt_mel
        token_bucket = select_bucket(self._buckets("flow_encoder", "n_tok"), num_tokens)
        mel_bucket = select_bucket(self._buckets("hift_f0", "n_mel"), num_mel)
        frame_bucket = constants.CHATTERBOX_MEL_FRAMES_PER_TOKEN * token_bucket

        start = time.perf_counter()
        encoded = self._run(
            "flow_encoder",
            {"mu": (1, channels, frame_bucket), "frame_mask": (1, 1, frame_bucket)},
            token=F.pad(tokens, (0, token_bucket - num_tokens)).numpy(),
            token_len=np.array([num_tokens], dtype=np.int64),
        )
        perf.append(ModulePerf("flow_encoder", time.perf_counter() - start))
        noise = torch.randn(1, channels, num_frames)  # same draw as torch.randn_like(mu) in CausalConditionalCFM
        speaker = flow.spk_embed_affine_layer(F.normalize(ref["embedding"], dim=1))
        prompt = torch.zeros(1, channels, frame_bucket)
        prompt[:, :, :num_prompt_mel] = ref["prompt_feat"].transpose(1, 2)
        mel = self._flow_euler(
            F.pad(noise, (0, frame_bucket - num_frames)),
            torch.from_numpy(encoded["mu"]),
            torch.from_numpy(encoded["frame_mask"]),
            speaker,
            prompt,
            perf,
        )[:, :, num_prompt_mel:num_frames]

        mel_padded = F.pad(mel, (0, mel_bucket - num_mel)).numpy()
        start = time.perf_counter()
        f0 = self._run("hift_f0", {"f0": (1, mel_bucket)}, x=mel_padded)["f0"][:, :num_mel]
        perf.append(ModulePerf("hift_f0", time.perf_counter() - start))
        # Excitation source on the host in float32 (HiFTGenerator.inference): the phase is a running sum over every
        # output sample, and the random phase and noise are drawn in upstream order.
        source = hift.f0_upsamp(torch.from_numpy(f0)[:, None]).transpose(1, 2)
        source = hift.m_source(source)[0].transpose(1, 2)
        num_samples = mel_bucket * constants.CHATTERBOX_HIFT_SAMPLES_PER_MEL_FRAME
        start = time.perf_counter()
        wav = self._run(
            "hift_decoder",
            {"wav": (1, num_samples)},
            mel=mel_padded,
            source=F.pad(source, (0, num_samples - source.shape[-1])).numpy(),
        )["wav"]
        perf.append(ModulePerf("hift_decoder", time.perf_counter() - start))
        # session outputs are read-only buffers; copy before the in-place trim fade in generate()
        return torch.from_numpy(wav[:, : num_mel * constants.CHATTERBOX_HIFT_SAMPLES_PER_MEL_FRAME].copy())

    @log_pipeline_api(
        "generate", "Speech generation completed.", "generation_complete", start_run=True, finish_run=True
    )
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        language_id: Optional[str],
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        max_new_tokens: int = 1000,
        seed: Optional[int] = None,
    ) -> QEffTTSOutput:
        """
        Synthesize ``text`` in ``language_id`` with the built-in voice or the voice of ``audio_prompt_path``.

        Arguments and defaults follow upstream ``ChatterboxMultilingualTTS.generate``. ``seed`` seeds torch's global
        generator first; with the same seed and torch release the random draws match upstream.
        """
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language_id '{language_id}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}.")
        if language_id and language_id.lower() == "zh" and not self.model.tokenizer.cangjie_converter.word2cj:
            raise ValueError("Chinese needs the Cangjie table (Cangjie5_TC.json in the checkpoint), which is missing.")
        if self.custom_config is None or any(module.qpc_path is None for module in self.modules.values()):
            self.compile()
        if seed is not None:
            torch.manual_seed(seed)
        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        elif self.model.conds is None:
            raise ValueError("No voice: pass audio_prompt_path, or use a checkpoint with a built-in voice (conds.pt).")
        if float(exaggeration) != float(self.model.conds.t3.emotion_adv[0, 0, 0].item()):
            current = self.model.conds.t3
            self.model.conds.t3 = T3Cond(
                speaker_emb=current.speaker_emb,
                cond_prompt_speech_tokens=current.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            )

        self._activate()  # before timing, so the module times below exclude QPC loading
        start_all = time.perf_counter()
        perf: List[ModulePerf] = []
        prefill = self._t3_prefill_embeds(text, language_id, cfg_weight)
        tokens = self._t3_sample(
            prefill,
            max_new_tokens=self._max_speech_tokens(prefill.shape[1], max_new_tokens),
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            perf=perf,
        )
        speech_tokens = drop_invalid_tokens(tokens[0])
        if speech_tokens.numel() == 0:  # end of speech sampled first: nothing to vocode (HiFT fails on 0 frames)
            logger.warning("T3 produced no speech tokens; returning empty audio.")
            return QEffTTSOutput(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=self.sample_rate,
                speech_tokens=tokens[0].numpy(),
                pipeline_module=perf,
            )
        wav = self._s3gen(speech_tokens, perf)
        trim_fade = self.model.s3gen.trim_fade[: wav.shape[1]]  # one token with an odd-length voice prompt: 480 samples
        wav[:, : len(trim_fade)] *= trim_fade
        # Upstream drops the audio of the last speech token, which is emitted just before EOS
        samples_per_token = self.sample_rate // constants.CHATTERBOX_SPEECH_TOKENS_PER_SECOND
        wav = wav[0, : max(1, int(speech_tokens.shape[-1]) - 1) * samples_per_token].numpy()
        if len(wav) < samples_per_token:  # 1 token + odd-length voice prompt: 480 samples, too short for Perth's STFT
            wav = np.pad(wav, (0, samples_per_token - len(wav)))
        start = time.perf_counter()
        audio = self.model.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
        perf.append(ModulePerf("watermark", time.perf_counter() - start))
        accounted = sum(sum(p.perf) if isinstance(p.perf, list) else p.perf for p in perf)
        perf.append(ModulePerf("host_other", time.perf_counter() - start_all - accounted))
        return QEffTTSOutput(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=self.sample_rate,
            speech_tokens=tokens[0].numpy(),
            pipeline_module=perf,
        )
