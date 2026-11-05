# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
DFlash speculative-decoding tests.

Includes a host-side regression test for generation-length enforcement and an
end-to-end QAIC test that builds a tiny from-scratch TLM/DLM pair, compiles both
to real QPCs, and drives them through the SPD decode loop.
"""

import os
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.dflash_generation import (
    SpecDecodingMetrics,
    _run_spd_core,
    run_spd_inference_gemma4,
    run_spd_inference_qwen3_vl,
    run_spd_inference_single,
)

VOCAB_SIZE = 64
HIDDEN_SIZE = 64
BLOCK_SIZE = 4  # DFlash decode block size; also the DLM's prefill_seq_len
# Different prefill and verification lengths exercise buffer resizing.
PROMPT_CHUNK_SIZE = 2 * BLOCK_SIZE
CTX_LEN = 32
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
MASK_TOKEN_ID = 2


@pytest.mark.parametrize("config", [{}, {"dflash_config": {}}], ids=["missing-block-size", "empty-dflash-config"])
def test_get_dflash_block_size_rejects_missing_value(config):
    from examples.performance.dflash import utils

    with pytest.raises(ValueError, match="block_size >= 2"):
        utils.get_dflash_block_size(config)


@pytest.mark.parametrize("block_size", [None, "16", 16.0, -1, 0, 1])
@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
def test_get_dflash_block_size_rejects_invalid_value(block_size, nested):
    from examples.performance.dflash import utils

    config = {"block_size": block_size}
    if nested:
        config = {"dflash_config": config}
    with pytest.raises(ValueError, match="block_size >= 2"):
        utils.get_dflash_block_size(config)


@pytest.mark.parametrize("ctx_len", [4096, 8192])
def test_qwen35_compile_accepts_context_larger_than_draft_window(monkeypatch, ctx_len):
    from examples.performance.dflash import utils

    class ReachedWeightLoading(Exception):
        pass

    monkeypatch.setattr(
        utils.AutoConfig,
        "from_pretrained",
        lambda repo, **kwargs: SimpleNamespace(model_type="qwen3_5", sliding_window=2048),
    )
    monkeypatch.setattr(utils, "read_dlm_meta", lambda *args: ({}, [1], 16))

    def stop_loading(*args, **kwargs):
        raise ReachedWeightLoading

    monkeypatch.setattr(utils.QEFFAutoModelForImageTextToText, "from_pretrained", stop_loading)
    with pytest.raises(ReachedWeightLoading):
        utils.compile_qwen3vl_vlm_qpcs(
            "Qwen/Qwen3.6-27B",
            "z-lab/Qwen3.6-27B-DFlash",
            ctx_len=ctx_len,
            prefill_seq_len=32,
            num_cores=8,
            num_devices=4,
        )


@pytest.mark.parametrize("ctx_len", [4096, 8192])
@pytest.mark.parametrize(
    "model_type,model_name",
    [
        ("qwen3_5", "Qwen/Qwen3.6-27B"),
        ("qwen3_5_moe", "Qwen/Qwen3.6-35B-A3B"),
    ],
)
def test_qwen35_vision_cli_accepts_long_context(monkeypatch, ctx_len, model_type, model_name):
    from examples.performance.dflash import utils

    class ReachedCompilation(Exception):
        pass

    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(
        sys,
        "argv",
        ["basic_inference_vision.py", "--model_name", model_name, "--image", "--ctx_len", str(ctx_len)],
    )
    monkeypatch.setattr(
        utils.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type=model_type, to_dict=lambda: {"block_size": 16}),
    )

    def stop_compile(*args, **kwargs):
        assert kwargs["ctx_len"] == ctx_len
        raise ReachedCompilation

    monkeypatch.setattr(utils, "compile_qwen3vl_vlm_qpcs", stop_compile)
    with pytest.raises(ReachedCompilation):
        runpy.run_path(os.path.join(os.path.dirname(utils.__file__), "basic_inference_vision.py"), run_name="__main__")


@pytest.mark.parametrize("sequence_lengths", [[], [16, 64]])
def test_dflash_decode_specialization_accepts_fixed_and_dynamic_shapes(sequence_lengths):
    from examples.performance.dflash.utils import validate_spd_decode_specialization

    session = SimpleNamespace(
        binding_index_map={"input_ids": 0},
        bindings=[SimpleNamespace(dims=[1, 64])],
        allowed_shapes=[[(8, [1, length])] for length in sequence_lengths],
    )
    validate_spd_decode_specialization(session, 64)
    with pytest.raises(ValueError, match="seq_len=32"):
        validate_spd_decode_specialization(session, 32)


@pytest.mark.parametrize("prompt_chunk_size", [8, 31, 127])
def test_qwen3_5_runtime_rejects_unsupported_prefill_qpc(prompt_chunk_size):
    with pytest.raises(ValueError, match="multiple of block_size"):
        run_spd_inference_qwen3_vl(
            prompt_text="test",
            tokenizer=None,
            processor=None,
            tlm_config=SimpleNamespace(model_type="qwen3_5"),
            dlm_session=None,
            tlm_session=SimpleNamespace(input_names=["cache_commit"]),
            mask_token_id=MASK_TOKEN_ID,
            vocab_size=VOCAB_SIZE,
            prompt_chunk_size=prompt_chunk_size,
        )


@pytest.mark.parametrize(
    "model_type,prompt_chunk_size,target_seq_len",
    [
        ("qwen3_5", 32, 32),
        ("qwen3_5", 64, 32),
        ("qwen3_5", 128, 32),
        ("qwen3_5", 256, 32),
        ("qwen3_5_moe", 32, 32),
        ("qwen3_5_moe", 128, 32),
    ],
)
def test_qwen3_5_runtime_concatenates_prefix_without_second_call(
    monkeypatch, model_type, prompt_chunk_size, target_seq_len
):
    import torch

    from QEfficient.generation import dflash_generation as generation

    calls = []
    buffers = []

    def run(feeds):
        calls.append({key: value.copy() for key, value in feeds.items()})
        return {
            "logits": np.zeros((1, feeds["num_logits_to_keep"].shape[0]), dtype=np.int32),
            "hidden_states": np.broadcast_to(np.arange(target_seq_len)[None, :, None], (1, target_seq_len, 32)),
            "image_idx_output": np.zeros((1, 1), dtype=np.int64),
        }

    session = SimpleNamespace(
        input_names=["cache_commit", "vision_embeds", "num_logits_to_keep"],
        binding_index_map={"vision_embeds": 0, "input_ids": 1, "num_logits_to_keep": 2},
        bindings=[SimpleNamespace(type=0, dims=(1, 1, 32)), SimpleNamespace(dims=(1, prompt_chunk_size))],
        allowed_shapes=[
            [(0, [1, 1, 32]), (8, [1, length]), (8, [keep, 1])]
            for length, keep in [(prompt_chunk_size, 1), (target_seq_len, 16)]
        ],
        aic_to_np_dtype_mapping={0: np.dtype(np.float32)},
        run=run,
        set_buffers=buffers.append,
        skip_buffers=lambda names: None,
    )
    monkeypatch.setattr(
        generation,
        "build_inputs_qwen3_vl",
        lambda *a, **k: {"input_ids": torch.ones((1, 5), dtype=torch.int64), "attention_mask": torch.ones((1, 5))},
    )
    monkeypatch.setattr(
        generation,
        "compute_position_ids_qwen3_vl",
        lambda *a, **k: (torch.arange(5).view(1, 1, 5).repeat(4, 1, 1), torch.zeros((1, 1), dtype=torch.int64)),
    )

    def core(**kwargs):
        kwargs["tlm_prefill"](0, prompt_chunk_size)
        assert calls[-1]["input_ids"].shape[1] == prompt_chunk_size
        assert calls[-1]["cache_commit"].item() == 1
        assert calls[-1]["num_logits_to_keep"].shape == (1, 1)
        kwargs["prepare_decode"]()
        ids = np.arange(16, dtype=np.int64).reshape(1, 16)
        positions = np.arange(5, 21, dtype=np.int64).reshape(1, 16)
        logits, hidden = kwargs["tlm_decode"](ids, positions)
        assert logits.shape == (1, 16)
        assert hidden.shape == (1, 16, 32)
        verify = calls[-1]
        assert verify["input_ids"].shape == (1, target_seq_len)
        np.testing.assert_array_equal(verify["position_ids"][0, :, 16:], positions)
        assert np.all(verify["position_ids"][..., :16] == -1)
        np.testing.assert_array_equal(hidden[0, :, 0], np.arange(16, 32))
        assert verify["cache_commit"].item() == 0
        assert verify["num_logits_to_keep"].shape == (16, 1)
        call_count = len(calls)
        kwargs["queue_accepted_prefix"](ids, positions, 2)
        assert len(calls) == call_count
        kwargs["tlm_decode"](ids + 1, positions + 2)
        combined = calls[-1]
        assert len(calls) == call_count + 1
        assert combined["cache_commit"].item() == 0
        np.testing.assert_array_equal(combined["input_ids"], np.concatenate([ids, ids + 1], axis=1))
        np.testing.assert_array_equal(combined["position_ids"][0, :, :2], positions[:, :2])
        assert np.all(combined["position_ids"][..., 2:16] == -1)
        np.testing.assert_array_equal(combined["position_ids"][0, :, 16:], positions + 2)
        return kwargs["metrics"]

    monkeypatch.setattr(generation, "_run_spd_core", core)
    generation.run_spd_inference_qwen3_vl(
        prompt_text="test",
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=None),
        processor=None,
        tlm_config=SimpleNamespace(model_type=model_type),
        dlm_session=None,
        tlm_session=session,
        mask_token_id=2,
        vocab_size=64,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=512,
        block_size=16,
        hidden_size=32,
    )
    assert [item["logits"].shape[1] for item in buffers if "logits" in item] == [1, 16]


@pytest.mark.parametrize("with_image", [False, True])
def test_qwen35_moe_runtime_mrope_matches_hf(with_image):
    import torch

    from QEfficient.generation.dflash_generation import compute_position_ids_qwen3_vl
    from tests.unit_test.transforms.test_dflash_transform import make_tiny_qwen3_5_vlm

    model = make_tiny_qwen3_5_vlm(moe=True)
    ids = torch.tensor([[10, 5, 3, 3, 3, 3, 6, 11]] if with_image else [[10, 11, 12]])
    mask = torch.ones_like(ids)
    types = (ids == model.config.image_token_id).long()
    grid = torch.tensor([[1, 4, 4]]) if with_image else None
    expected, expected_delta = model.model.get_rope_index(ids, types, image_grid_thw=grid, attention_mask=mask)
    actual, delta = compute_position_ids_qwen3_vl(ids, mask, grid, model.config, types)
    torch.testing.assert_close(actual[0], torch.arange(ids.shape[1]).view(1, -1))
    torch.testing.assert_close(actual[1:], expected)
    torch.testing.assert_close(delta, expected_delta)


@pytest.mark.parametrize("block_size", [8, 16])
@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_qwen35_compile_preserves_dflash_metadata(monkeypatch, block_size, model_type):
    import torch

    from examples.performance.dflash import utils

    calls = []
    text_model = SimpleNamespace(fc=torch.nn.Linear(8, 4, bias=False), hidden_norm=torch.nn.LayerNorm(4))
    qaic_config = {"target_layer_ids": [2, 4], "dflash_dlm_repo": "draft"}

    def compile_model(**kwargs):
        calls.append(kwargs)
        return {"lang_prefill_qpc_path": "target.qpc", "vision_qpc_path": "vision.qpc"}

    def load_model(*args, **kwargs):
        assert kwargs["qaic_config"] == qaic_config
        assert kwargs["dtype"] == "float32"
        return SimpleNamespace(
            lang_model=SimpleNamespace(model=SimpleNamespace(language_model=text_model)), compile=compile_model
        )

    monkeypatch.setattr(utils.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(model_type=model_type))
    monkeypatch.setattr(
        utils,
        "read_dlm_meta",
        lambda *a: ({"fc.weight": torch.ones(4, 8), "hidden_norm.weight": torch.ones(4)}, [1, 3], block_size),
    )
    monkeypatch.setattr(utils.QEFFAutoModelForImageTextToText, "from_pretrained", load_model)
    assert utils.compile_qwen3vl_vlm_qpcs(
        "target", "draft", ctx_len=8192, prefill_seq_len=128, num_cores=8, num_devices=4
    ) == ("target.qpc", "vision.qpc")
    assert calls[0]["prefill_only"]
    assert calls[0]["prefill_seq_len"] == 128 and calls[0]["dflash_block_size"] == block_size
    if model_type == "qwen3_5_moe":
        assert calls[0]["qaic_config"] == qaic_config
        assert calls[0]["enable_chunking"]
        assert calls[0]["mxint8_kv_cache"]


class _FakeTokenizer:
    """Minimal stand-in implementing only what run_spd_inference_single calls:
    __call__(prompts, return_tensors="np", padding=...) and pad/eos token ids.
    Deterministically maps each prompt string to a short fixed-length id sequence,
    keeping everything offline and within the tiny vocab."""

    pad_token_id = PAD_TOKEN_ID
    eos_token_id = EOS_TOKEN_ID

    def __call__(self, prompts, return_tensors="np", padding=True, max_length=None):
        assert return_tensors == "np"
        raw_ids = [[3 + (i % (VOCAB_SIZE - 3)) for i in range(1, len(p) + 1)] for p in prompts]
        length = max_length if max_length is not None else max(len(ids) for ids in raw_ids)
        input_ids = np.full((len(prompts), length), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(prompts), length), dtype=np.int64)
        for i, ids in enumerate(raw_ids):
            n = min(len(ids), length)
            input_ids[i, :n] = ids[:n]
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeTLMSession:
    def __init__(self):
        self.run_count = 0

    def set_buffers(self, _buffers):
        pass

    def run(self, inputs):
        self.run_count += 1
        seq_len = inputs["input_ids"].shape[1]
        hidden_states = np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32)
        if self.run_count == 1:
            logits = np.zeros((1, seq_len), dtype=np.int32)
            logits[:, 1] = 5
        else:
            logits = np.array([[6, 7, 8, 9]], dtype=np.int32)
        return {"logits": logits, "hidden_states": hidden_states}


class _FakeDLMSession:
    def set_buffers(self, _buffers):
        pass

    def run(self, _inputs):
        logits = np.zeros((1, BLOCK_SIZE, VOCAB_SIZE), dtype=np.float32)
        token_ids = np.array([5, 6, 7, 8])
        logits[0, np.arange(BLOCK_SIZE), token_ids] = 1
        return {"logits": logits}


class _FakeGemmaTLMSession:
    """Tracks static vision-buffer handling in the Gemma4 SPD path."""

    class _Binding:
        def __init__(self, name, index, dims, dtype="fp32"):
            self.name = name
            self.index = index
            self.dims = dims
            self.type = dtype

    def __init__(self):
        self.input_names = ["input_ids", "position_ids", "vision_embeds"]
        self.output_names = ["logits", "hidden_states"]
        self.bindings = [
            self._Binding("input_ids", 0, [1, BLOCK_SIZE]),
            self._Binding("position_ids", 1, [1, BLOCK_SIZE]),
            self._Binding("vision_embeds", 2, [1, 2, HIDDEN_SIZE]),
            self._Binding("logits", 3, [1, BLOCK_SIZE, VOCAB_SIZE]),
            self._Binding("hidden_states", 4, [1, BLOCK_SIZE, HIDDEN_SIZE]),
        ]
        self.binding_index_map = {binding.name: binding.index for binding in self.bindings}
        self.allowed_shapes = [
            [
                (8, [1, BLOCK_SIZE]),
                (8, [1, BLOCK_SIZE]),
                (4, [1, 2, HIDDEN_SIZE]),
                (4, [1, BLOCK_SIZE, VOCAB_SIZE]),
                (4, [1, BLOCK_SIZE, HIDDEN_SIZE]),
            ]
        ]
        self.aic_to_np_dtype_mapping = {"fp32": np.dtype(np.float32)}
        self.vision_buffer_sets = 0
        self.run_vision_feeds = 0
        self.skipped_buffers = []

    def set_buffers(self, buffers):
        if "vision_embeds" in buffers:
            self.vision_buffer_sets += 1

    def skip_buffers(self, buffers):
        self.skipped_buffers.extend(buffers)

    def run(self, inputs):
        self.run_vision_feeds += "vision_embeds" in inputs
        seq_len = inputs["input_ids"].shape[1]
        logits = np.zeros((1, seq_len, VOCAB_SIZE), dtype=np.float32)
        token_ids = np.resize(np.array([5, 6, 7, 8]), seq_len)
        logits[0, np.arange(seq_len), token_ids] = 1
        return {
            "logits": logits,
            "hidden_states": np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32),
        }


def test_dflash_acceptance_rate_uses_accepted_tokens():
    metrics = SpecDecodingMetrics(block_size=BLOCK_SIZE)
    metrics.total_accepted_tokens = 6
    metrics.total_generated_tokens = 9
    metrics.num_total_iters = 3

    assert metrics.acceptance_rate() == 2.0


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_dflash_queued_prefix_counts_seed_and_preserves_verification_features(accepted):
    commits = []
    hidden = np.ones((1, BLOCK_SIZE, HIDDEN_SIZE), dtype=np.float32)

    class Draft(_FakeDLMSession):
        def run(self, inputs):
            if commits:
                np.testing.assert_array_equal(inputs["target_hidden"], 1)
            return super().run(inputs)

    def verify(ids, positions):
        hidden.fill(1)
        logits = np.array([[6, 7, 8, 9]], dtype=np.int32)
        if accepted < BLOCK_SIZE - 1:
            logits[:, accepted] = 10
        return logits, hidden

    def queue_prefix(ids, positions, count):
        commits.append(count)

    metrics = _run_spd_core(
        input_ids=np.array([[3, 4, 0, 0]]),
        flat_position_ids=np.array([[0, 1, -1, -1]]),
        tlm_prefill=lambda start, end: (np.array([[0, 5, 0, 0]]), hidden),
        prepare_decode=lambda: None,
        tlm_decode=verify,
        queue_accepted_prefix=queue_prefix,
        dlm_session=Draft(),
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        generation_len=16,
        max_iterations=2,
        eos_token_ids=set(),
        generated_ids=np.zeros((1, 16), dtype=np.int64),
        metrics=SpecDecodingMetrics(block_size=BLOCK_SIZE),
    )
    assert commits == [accepted + 1, accepted + 1]
    assert metrics.tlm_decode_time > 0


def test_dflash_generation_len_caps_partial_block():
    metrics = run_spd_inference_single(
        prompt_text="hi",
        tokenizer=_FakeTokenizer(),
        dlm_session=_FakeDLMSession(),
        tlm_session=_FakeTLMSession(),
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=PROMPT_CHUNK_SIZE,
        ctx_len=CTX_LEN,
        block_size=BLOCK_SIZE,
        max_iterations=5,
        hidden_size=HIDDEN_SIZE,
        generation_len=1,
    )

    assert metrics.generated_ids == [6]
    assert metrics.generated_sources == ["dlm"]
    assert metrics.total_generated_tokens == 1
    assert metrics.total_accepted_tokens == 1
    assert metrics.num_total_iters == 1


def test_dflash_rejects_padded_prompt_longer_than_context():
    with pytest.raises(ValueError, match=r"ctx_len \(7\) must be greater than or equal to padded_len \(8\)"):
        run_spd_inference_single(
            prompt_text="hi",
            tokenizer=_FakeTokenizer(),
            dlm_session=_FakeDLMSession(),
            tlm_session=_FakeTLMSession(),
            mask_token_id=MASK_TOKEN_ID,
            vocab_size=VOCAB_SIZE,
            prompt_chunk_size=PROMPT_CHUNK_SIZE,
            ctx_len=PROMPT_CHUNK_SIZE - 1,
            block_size=BLOCK_SIZE,
            max_iterations=5,
            hidden_size=HIDDEN_SIZE,
            generation_len=1,
        )


def _make_tiny_qwen3_config():
    return Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )


def test_gemma4_binds_vision_embeddings_once_and_skips_them_for_decode():
    tlm_session = _FakeGemmaTLMSession()
    run_spd_inference_gemma4(
        prompt_text="ignored when input_ids are provided",
        tokenizer=_FakeTokenizer(),
        dlm_session=_FakeDLMSession(),
        tlm_session=tlm_session,
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=BLOCK_SIZE,
        ctx_len=CTX_LEN,
        block_size=BLOCK_SIZE,
        max_iterations=1,
        hidden_size=HIDDEN_SIZE,
        generation_len=1,
        input_ids=np.array([[3, 4]], dtype=np.int64),
        vision_embeds=np.zeros((1, 2, HIDDEN_SIZE), dtype=np.float32),
    )

    assert tlm_session.vision_buffer_sets == 1
    assert tlm_session.run_vision_feeds == 0
    assert tlm_session.skipped_buffers == ["vision_embeds"]


@pytest.mark.on_qaic
def test_dflash_spd_inference(manual_cleanup):
    tlm_model = QEFFAutoModelForCausalLM(
        Qwen3ForCausalLM(_make_tiny_qwen3_config()).eval(), qaic_config={"target_layer_ids": [1]}
    )
    dlm_model = QEFFAutoModelForCausalLM(
        Qwen3ForCausalLM(_make_tiny_qwen3_config()).eval(), qaic_config={"dflash_dlm": True}
    )

    tlm_qpc = tlm_model.compile(
        prefill_seq_len=PROMPT_CHUNK_SIZE,
        ctx_len=CTX_LEN,
        num_cores=2,
        num_devices=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        dflash_block_size=BLOCK_SIZE,
    )
    dlm_qpc = dlm_model.compile(
        prefill_seq_len=BLOCK_SIZE,
        ctx_len=CTX_LEN,
        num_cores=2,
        num_devices=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        prefill_only=True,
    )

    tlm_session = QAICInferenceSession(tlm_qpc)
    dlm_session = QAICInferenceSession(dlm_qpc)
    tlm_session.skip_buffers(
        {
            x
            for x in tlm_session.input_names + tlm_session.output_names
            if x.startswith("past_") or x.endswith("_RetainedState")
        }
    )
    dlm_session.skip_buffers(
        {
            x
            for x in dlm_session.input_names + dlm_session.output_names
            if x.startswith("past_") or x.endswith("_RetainedState")
        }
    )

    metrics = run_spd_inference_single(
        prompt_text="hi",
        tokenizer=_FakeTokenizer(),
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=PROMPT_CHUNK_SIZE,
        ctx_len=CTX_LEN,
        block_size=BLOCK_SIZE,
        max_iterations=5,
        hidden_size=HIDDEN_SIZE,
        generation_len=8,
    )

    assert metrics.num_total_iters > 0
    assert metrics.total_generated_tokens > 0
    assert len(metrics.generated_ids) == len(metrics.generated_sources)
    assert all(source in ("dlm", "tlm") for source in metrics.generated_sources)
    assert os.path.isfile(os.path.join(os.path.dirname(tlm_qpc), "qconfig.json"))
    assert os.path.isfile(os.path.join(os.path.dirname(dlm_qpc), "qconfig.json"))

    manual_cleanup([tlm_model.onnx_path, dlm_model.onnx_path])
