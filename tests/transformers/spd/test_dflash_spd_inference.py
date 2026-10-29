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

import numpy as np
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.dflash_generation import (
    SpecDecodingMetrics,
    run_spd_inference_gemma4,
    run_spd_inference_single,
)

VOCAB_SIZE = 64
HIDDEN_SIZE = 64
BLOCK_SIZE = 4  # DFlash decode block size; also the DLM's prefill_seq_len
# TLM prompt-chunk size for prefill. Must differ from BLOCK_SIZE: the TLM's decode
# specialization seq_len is always overridden to BLOCK_SIZE (see
# build_decode_specialization's dflash_tlm branch), so a prefill spec with the same
# seq_len would be indistinguishable from it and the AIC compiler would reject the
# network ("No input that uniquely identifies specialization"). A multiple of
# BLOCK_SIZE keeps the prefill loop's sub-block splitting exact (no remainder).
PROMPT_CHUNK_SIZE = 2 * BLOCK_SIZE
CTX_LEN = 32
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
MASK_TOKEN_ID = 2


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
        self.input_names = []

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


class _FakeCommitTLMSession(_FakeTLMSession):
    def __init__(self):
        super().__init__()
        self.input_names = ["cache_commit"]
        self.cache_commits = []
        self.position_history = []

    def run(self, inputs):
        self.cache_commits.append(int(inputs["cache_commit"][0]))
        self.position_history.append(inputs["position_ids"].copy())
        return super().run(inputs)


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


def test_dflash2_host_selector_rescores_top_k_chain():
    from examples.performance.dflash.utils import HostSelector

    predecessor = np.zeros((VOCAB_SIZE, 2), dtype=np.float32)
    successor = np.zeros_like(predecessor)
    predecessor[[0, 2], 0] = 1.0
    successor[2, 0] = 2.0
    selector = HostSelector(predecessor, successor, top_k=2)

    logits = np.zeros((1, BLOCK_SIZE, VOCAB_SIZE), dtype=np.float32)
    logits[:, :, 1] = 5.0
    logits[:, :, 2] = 4.5
    outputs = {
        "logits": logits,
        "selector_hidden": np.ones((1, BLOCK_SIZE, 2), dtype=np.float32),
    }

    selected = selector(outputs, np.array([0], dtype=np.int64))

    assert selected.tolist() == [[1, 2, 2, 2]]
    with pytest.raises(ValueError, match="selector_hidden rank"):
        selector(
            {
                "logits": logits,
                "selector_hidden": np.ones((1, BLOCK_SIZE, 3), dtype=np.float32),
            },
            np.array([0], dtype=np.int64),
        )


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


def test_dflash2_text_path_uses_selector_and_qwen35_commit_replay():
    tlm_session = _FakeCommitTLMSession()
    selector_calls = []

    def selector(outputs, anchor_ids):
        selector_calls.append((sorted(outputs), anchor_ids.copy()))
        return outputs["logits"].argmax(axis=-1)

    run_spd_inference_single(
        prompt_text="hi",
        tokenizer=_FakeTokenizer(),
        dlm_session=_FakeDLMSession(),
        tlm_session=tlm_session,
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=PROMPT_CHUNK_SIZE,
        ctx_len=CTX_LEN,
        block_size=BLOCK_SIZE,
        max_iterations=1,
        hidden_size=HIDDEN_SIZE,
        generation_len=1,
        draft_selector=selector,
    )

    assert len(selector_calls) == 1
    assert tlm_session.cache_commits == [1, 0, 1]
    assert np.all(tlm_session.position_history[-1][:, 2:] == -1)


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
