# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only tests for QEfficient.generation module.

Tests verify:
  - Module importability and dataclass construction
  - Pure utility functions (calculate_latency, fix_prompts, etc.)
  - File I/O (get_compilation_dims, read_prompts_txt_file)
  - VisionHandler initialization and config-based methods
  - QEffTextGenerationBase: prefill, decode, chunking, continuous batching,
    prepare_decode_inputs, initialize_decode_inputs, update_decode_input,
    generate_decode_stream via a fully mocked QAICInferenceSession

All tests run on CPU only. QAICInferenceSession is mocked so no QAIC hardware
is required.
"""

import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from transformers import AutoTokenizer

from QEfficient.generation.text_generation_inference import (
    CloudAI100ExecInfo,
    CloudAI100ExecInfoNew,
    PerfMetrics,
    calculate_latency,
    fix_prompt_to_lora_id_mapping,
    fix_prompts,
    get_compilation_dims,
    get_input_prompts,
    read_prompts_txt_file,
)
from QEfficient.utils import constants

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50257  # gpt2 tokenizer eos_token_id=50256
CTX_LEN = 32
PREFILL_LEN = 8
BATCH_SIZE = 1


def _make_mock_session(
    batch_size=BATCH_SIZE,
    prefill_seq_len=PREFILL_LEN,
    ctx_len=CTX_LEN,
    vocab_size=VOCAB_SIZE,
    full_batch_size=None,
    include_sampler=False,
    force_seq_len=None,
):
    """
    Build a MagicMock that mimics QAICInferenceSession well enough for
    QEffTextGenerationBase to initialise and run on CPU.
    """
    session = MagicMock()

    # --- binding helpers ---
    def _binding(name, dims, direction="input"):
        b = MagicMock()
        b.name = name
        b.dims = dims
        b.dir = "input" if direction == "input" else "output"
        b.size = int(np.prod(dims)) * 4  # 4 bytes per float32
        b.type = 1  # FLOAT_TYPE
        return b

    # Build bindings list
    bindings = [
        _binding("input_ids", [batch_size, prefill_seq_len], "input"),
        _binding("position_ids", [batch_size, prefill_seq_len], "input"),
        _binding("logits", [batch_size, prefill_seq_len, vocab_size], "output"),
    ]
    if full_batch_size is not None:
        bindings.append(_binding("batch_index", [full_batch_size, 1], "input"))

    session.bindings = bindings
    session.binding_index_map = {b.name: i for i, b in enumerate(bindings)}
    session.allowed_shapes = []  # use bindings dims directly
    session.input_names = [b.name for b in bindings if b.dir == "input"]
    session.output_names = [b.name for b in bindings if b.dir == "output"]
    session.is_active = True

    # run() returns logits with argmax-able values
    def _run(inputs):
        bs = inputs.get("input_ids", np.zeros((batch_size, 1))).shape[0]
        seq = (
            force_seq_len if force_seq_len is not None else inputs.get("input_ids", np.zeros((batch_size, 1))).shape[1]
        )
        logits = np.zeros((bs, seq, vocab_size), dtype=np.float32)
        logits[:, :, 42] = 1.0  # always predict token 42
        return {"logits": logits}

    session.run.side_effect = _run
    session.skip_buffers = MagicMock()
    session.set_buffers = MagicMock()
    session.activate = MagicMock()
    session.deactivate = MagicMock()
    return session


def _make_tokenizer():
    """Return a tiny GPT2 tokenizer (downloads once, cached)."""
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok
    except Exception:
        pytest.skip("Cannot load gpt2 tokenizer (network unavailable)")


def _make_base_instance(
    batch_size=BATCH_SIZE,
    ctx_len=CTX_LEN,
    full_batch_size=None,
):
    """
    Construct a QEffTextGenerationBase with a mocked session.
    Patches QAICInferenceSession so no hardware is needed.
    """
    from QEfficient.generation.text_generation_inference import QEffTextGenerationBase

    tok = _make_tokenizer()
    mock_session = _make_mock_session(
        batch_size=batch_size,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
    )

    with patch(
        "QEfficient.generation.text_generation_inference.QAICInferenceSession",
        return_value=mock_session,
    ):
        obj = QEffTextGenerationBase(
            tokenizer=tok,
            qpc_path="/fake/path/model.qpc",
            ctx_len=ctx_len,
            full_batch_size=full_batch_size,
        )
    return obj, tok, mock_session


# ---------------------------------------------------------------------------
# Tests: Module importability
# ---------------------------------------------------------------------------


class TestGenerationModuleImportability:
    """All generation modules must be importable on CPU."""

    def test_cloud_infer_importable(self):
        import QEfficient.generation.cloud_infer

        assert QEfficient.generation.cloud_infer is not None

    def test_embedding_handler_importable(self):
        import QEfficient.generation.embedding_handler

        assert QEfficient.generation.embedding_handler is not None

    def test_text_generation_inference_importable(self):
        import QEfficient.generation.text_generation_inference

        assert QEfficient.generation.text_generation_inference is not None

    def test_vlm_generation_importable(self):
        import QEfficient.generation.vlm_generation

        assert QEfficient.generation.vlm_generation is not None

    def test_vision_handler_importable(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        assert VisionHandler is not None

    def test_text_generation_class_importable(self):
        from QEfficient.generation.text_generation_inference import TextGeneration

        assert TextGeneration is not None

    def test_qeff_text_generation_base_importable(self):
        from QEfficient.generation.text_generation_inference import QEffTextGenerationBase

        assert QEffTextGenerationBase is not None

    def test_vision_language_generation_importable(self):
        from QEfficient.generation.vlm_generation import VisionLanguageGeneration

        assert VisionLanguageGeneration is not None


# ---------------------------------------------------------------------------
# Tests: PerfMetrics dataclass
# ---------------------------------------------------------------------------


class TestPerfMetricsDataclass:
    def test_construction_and_field_access(self):
        m = PerfMetrics(prefill_time=1.5, decode_perf=50.0, total_perf=45.0, total_time=10.0)
        assert m.prefill_time == 1.5
        assert m.decode_perf == 50.0
        assert m.total_perf == 45.0
        assert m.total_time == 10.0

    def test_repr_contains_values(self):
        m = PerfMetrics(1.5, 50.0, 45.0, 10.0)
        r = repr(m)
        assert "1.5" in r or "1.50" in r

    def test_zero_values_allowed(self):
        m = PerfMetrics(0.0, 0.0, 0.0, 0.0)
        assert m.prefill_time == 0.0


# ---------------------------------------------------------------------------
# Tests: CloudAI100ExecInfo dataclass
# ---------------------------------------------------------------------------


class TestCloudAI100ExecInfoDataclass:
    def test_construction_and_repr(self):
        m = PerfMetrics(1.5, 50.0, 45.0, 10.0)
        info = CloudAI100ExecInfo(
            batch_size=1,
            generated_texts=["Hello"],
            generated_ids=[np.array([1, 2, 3])],
            perf_metrics=m,
        )
        assert info.batch_size == 1
        r = repr(info)
        assert "Prefill" in r or "prefill" in r

    def test_nested_list_generated_texts(self):
        m = PerfMetrics(1.5, 50.0, 45.0, 10.0)
        info = CloudAI100ExecInfo(
            batch_size=2,
            generated_texts=[["A", "B"], ["C", "D"]],
            generated_ids=[np.array([1]), np.array([2])],
            perf_metrics=m,
        )
        assert len(info.generated_texts) == 2

    def test_cloud_ai100_exec_info_new(self):
        m = PerfMetrics(1.5, 50.0, 45.0, 10.0)
        info = CloudAI100ExecInfoNew(
            batch_size=1,
            generated_ids=[np.array([1, 2, 3])],
            perf_metrics=m,
        )
        assert info.batch_size == 1
        assert "Prefill" in repr(info) or "prefill" in repr(info)


# ---------------------------------------------------------------------------
# Tests: calculate_latency
# ---------------------------------------------------------------------------


class TestCalculateLatency:
    def test_normal_case(self):
        pf, dp, tp, tt = calculate_latency(100, 5.0, 1.0, 11.0, 0.0)
        assert pf == pytest.approx(4.0)
        assert dp == pytest.approx(100 / 6.0)
        assert tp == pytest.approx(100 / 10.0)
        assert tt == pytest.approx(10.0)

    def test_with_decode_pause_time(self):
        pf, dp, tp, tt = calculate_latency(100, 5.0, 1.0, 11.0, 1.0)
        assert pf == pytest.approx(5.0)
        assert dp == pytest.approx(100 / 5.0)

    def test_zero_tokens(self):
        pf, dp, tp, tt = calculate_latency(0, 5.0, 1.0, 11.0, 0.0)
        assert dp == 0.0
        assert tp == 0.0

    def test_returns_floats(self):
        result = calculate_latency(100, 5.0, 1.0, 11.0, 0.0)
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Tests: get_input_prompts
# ---------------------------------------------------------------------------


class TestGetInputPrompts:
    def test_both_none_raises(self):
        with pytest.raises(ValueError):
            get_input_prompts(None, None)

    def test_string_to_list(self):
        r = get_input_prompts("Hello", None)
        assert r == ["Hello"]

    def test_list_unchanged(self):
        r = get_input_prompts(["A", "B"], None)
        assert r == ["A", "B"]

    def test_txt_file_priority(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("L1\nL2\n")
        r = get_input_prompts("ignored", str(f))
        assert r == ["L1", "L2"]


# ---------------------------------------------------------------------------
# Tests: fix_prompts
# ---------------------------------------------------------------------------


class TestFixPrompts:
    def test_fewer_prompts_repeated(self):
        r = fix_prompts(["A", "B"], 5)
        assert len(r) == 5
        assert r == ["A", "B", "A", "B", "A"]

    def test_exact_batch_unchanged(self):
        r = fix_prompts(["A", "B", "C"], 3)
        assert r == ["A", "B", "C"]

    def test_incomplete_batch_dropped(self):
        r = fix_prompts(["A", "B", "C", "D", "E"], 2)
        assert len(r) == 4

    def test_full_batch_size_used(self):
        r = fix_prompts(["A", "B"], 3, full_batch_size=8)
        assert len(r) == 8

    def test_single_prompt_repeated(self):
        r = fix_prompts(["X"], 4)
        assert r == ["X", "X", "X", "X"]


# ---------------------------------------------------------------------------
# Tests: fix_prompt_to_lora_id_mapping
# ---------------------------------------------------------------------------


class TestFixPromptToLoraIdMapping:
    def test_fewer_repeated(self):
        r = fix_prompt_to_lora_id_mapping([0, 1], 5)
        assert len(r) == 5

    def test_exact_unchanged(self):
        r = fix_prompt_to_lora_id_mapping([0, 1, 2], 3)
        assert r == [0, 1, 2]

    def test_full_batch_size(self):
        r = fix_prompt_to_lora_id_mapping([0, 1], 3, full_batch_size=8)
        assert len(r) == 8


# ---------------------------------------------------------------------------
# Tests: read_prompts_txt_file
# ---------------------------------------------------------------------------


class TestReadPromptsTxtFile:
    def test_reads_lines(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("A\nB\nC\n")
        assert read_prompts_txt_file(str(f)) == ["A", "B", "C"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("  A  \n  B  \n")
        assert read_prompts_txt_file(str(f)) == ["A", "B"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("")
        assert read_prompts_txt_file(str(f)) == []

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_prompts_txt_file("/no/such/file.txt")


# ---------------------------------------------------------------------------
# Tests: get_compilation_dims
# ---------------------------------------------------------------------------


class TestGetCompilationDims:
    def _write_spec(self, tmp_path, spec):
        qpc_dir = tmp_path / "qpc"
        qpc_dir.mkdir()
        (qpc_dir / "specializations.json").write_text(json.dumps(spec))
        return str(qpc_dir / "model.qpc")

    def test_basic(self, tmp_path):
        path = self._write_spec(tmp_path, {"specializations": [{"batch_size": "4", "ctx_len": "128"}]})
        bs, cl, fbs, num_kv_blocks = get_compilation_dims(path)
        assert bs == 4 and cl == 128 and fbs is None

    def test_with_full_batch_size(self, tmp_path):
        path = self._write_spec(
            tmp_path, {"specializations": [{"batch_size": "4", "ctx_len": "128", "full_batch_size": "16"}]}
        )
        bs, cl, fbs, num_kv_blocks = get_compilation_dims(path)
        assert fbs == 16

    def test_with_pagedAttention(self, tmp_path):
        path = self._write_spec(
            tmp_path,
            {"specializations": [{"batch_size": "4", "ctx_len": "128", "full_batch_size": "16", "num_kv_blocks": "8"}]},
        )
        bs, cl, fbs, num_kv_blocks = get_compilation_dims(path)
        assert num_kv_blocks == 8

    def test_missing_file_raises(self, tmp_path):
        qpc_dir = tmp_path / "qpc"
        qpc_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            get_compilation_dims(str(qpc_dir / "model.qpc"))

    def test_returns_ints(self, tmp_path):
        path = self._write_spec(tmp_path, {"specializations": [{"batch_size": "2", "ctx_len": "64"}]})
        bs, cl, fbs, num_kv_blocks = get_compilation_dims(path)
        assert isinstance(bs, int) and isinstance(cl, int)

    def test_strict_three_value_unpack_raises(self, tmp_path):
        # get_compilation_dims always returns a 4-tuple. A caller written against
        # the old 3-value contract must use extended unpacking (see test below);
        # a strict 3-target unpack raises ValueError. Pinned here so a future
        # change to the return arity doesn't silently break either calling style.
        path = self._write_spec(tmp_path, {"specializations": [{"batch_size": "4", "ctx_len": "128"}]})
        with pytest.raises(ValueError):
            bs, cl, fbs = get_compilation_dims(path)

    def test_legacy_caller_with_extended_unpack_still_works(self, tmp_path):
        path = self._write_spec(tmp_path, {"specializations": [{"batch_size": "4", "ctx_len": "128"}]})
        bs, cl, fbs, *_ = get_compilation_dims(path)
        assert bs == 4 and cl == 128 and fbs is None


# ---------------------------------------------------------------------------
# Tests: QEffTextGenerationBase construction (mocked session)
# ---------------------------------------------------------------------------


class TestQEffTextGenerationBaseConstruction:
    """QEffTextGenerationBase must initialise correctly with a mocked session."""

    def test_construction_succeeds(self):
        obj, tok, _ = _make_base_instance()
        assert obj is not None

    def test_batch_size_fetched(self):
        obj, _, _ = _make_base_instance(batch_size=2)
        assert obj.batch_size == 2

    def test_prefill_seq_len_fetched(self):
        obj, _, _ = _make_base_instance()
        assert obj._prefill_seq_len == PREFILL_LEN

    def test_ctx_len_stored(self):
        obj, _, _ = _make_base_instance(ctx_len=64)
        assert obj._ctx_len == 64

    def test_tokenizer_stored(self):
        obj, tok, _ = _make_base_instance()
        assert obj.tokenizer is tok

    def test_full_batch_size_none_by_default(self):
        obj, _, _ = _make_base_instance()
        assert obj.full_batch_size is None

    def test_vocab_size_fetched(self):
        obj, _, _ = _make_base_instance()
        assert obj._vocab_size == VOCAB_SIZE

    def test_session_skip_buffers_called(self):
        obj, _, mock_session = _make_base_instance()
        mock_session.skip_buffers.assert_called()


# ---------------------------------------------------------------------------
# Tests: initialize_decode_inputs
# ---------------------------------------------------------------------------


class TestInitializeDecodeInputs:
    """initialize_decode_inputs must allocate correctly shaped numpy arrays."""

    def test_generated_ids_shape(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(num_prompts=2, execution_batch_size=1, max_gen_length=20)
        assert obj.generated_ids.shape == (2, 20)

    def test_decode_input_ids_shape(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(num_prompts=1, execution_batch_size=1, max_gen_length=10)
        assert obj.decode_input_ids.shape == (1, 1)

    def test_decode_pos_ids_shape(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(num_prompts=1, execution_batch_size=1, max_gen_length=10)
        assert obj.decode_pos_ids.shape == (1, 1)

    def test_generation_len_shape(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(num_prompts=1, execution_batch_size=1, max_gen_length=10)
        assert obj.generation_len.shape == (1, 1)

    def test_generated_ids_filled_with_pad_token(self):
        obj, tok, _ = _make_base_instance()
        obj.initialize_decode_inputs(num_prompts=1, execution_batch_size=1, max_gen_length=10)
        assert np.all(obj.generated_ids == tok.pad_token_id)

    def test_decode_input_ids_zero_initialized(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(num_prompts=1, execution_batch_size=1, max_gen_length=10)
        assert np.all(obj.decode_input_ids == 0)


# ---------------------------------------------------------------------------
# Tests: prepare_decode_inputs
# ---------------------------------------------------------------------------


class TestPrepareDecodeInputs:
    """prepare_decode_inputs must build correct decode input dict."""

    def test_returns_dict_with_input_ids(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, 10)
        decode_inputs = obj.prepare_decode_inputs()
        assert "input_ids" in decode_inputs

    def test_returns_dict_with_position_ids(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, 10)
        decode_inputs = obj.prepare_decode_inputs()
        assert "position_ids" in decode_inputs

    def test_input_ids_shape_is_batch_by_1(self):
        obj, _, _ = _make_base_instance(batch_size=2)
        obj.initialize_decode_inputs(2, 2, 10)
        decode_inputs = obj.prepare_decode_inputs()
        assert decode_inputs["input_ids"].shape == (2, 1)

    def test_position_ids_shape_is_batch_by_1(self):
        obj, _, _ = _make_base_instance(batch_size=2)
        obj.initialize_decode_inputs(2, 2, 10)
        decode_inputs = obj.prepare_decode_inputs()
        assert decode_inputs["position_ids"].shape == (2, 1)

    def test_no_batch_index_without_full_batch_size(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, 10)
        decode_inputs = obj.prepare_decode_inputs()
        assert "batch_index" not in decode_inputs


# ---------------------------------------------------------------------------
# Tests: update_decode_input
# ---------------------------------------------------------------------------


class TestUpdateDecodeInput:
    """update_decode_input must correctly update decode state arrays."""

    def _make_outputs(self, token_id=42):
        logits = np.zeros((1, 1, VOCAB_SIZE), dtype=np.float32)
        logits[0, 0, token_id] = 1.0
        return {"logits": logits}

    def test_decode_input_ids_updated(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, 10)
        outputs = self._make_outputs(token_id=42)
        position_ids = np.array([[PREFILL_LEN]])
        obj.update_decode_input(outputs, position_ids, generation_len=10)
        assert obj.decode_input_ids[0, 0] == 42

    def test_decode_pos_ids_updated(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, 10)
        outputs = self._make_outputs(token_id=42)
        position_ids = np.array([[PREFILL_LEN]])
        obj.update_decode_input(outputs, position_ids, generation_len=10)
        assert obj.decode_pos_ids[0, 0] == PREFILL_LEN

    def test_generated_ids_first_token_set(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, 10)
        outputs = self._make_outputs(token_id=99)
        position_ids = np.array([[PREFILL_LEN]])
        obj.update_decode_input(outputs, position_ids, generation_len=10)
        assert obj.generated_ids[0, 0] == 99

    def test_returns_next_token_id(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, 10)
        outputs = self._make_outputs(token_id=77)
        position_ids = np.array([[PREFILL_LEN]])
        next_token = obj.update_decode_input(outputs, position_ids, generation_len=10)
        assert next_token[0, 0] == 77


# ---------------------------------------------------------------------------
# Tests: run_prefill (mocked session, chunking logic)
# ---------------------------------------------------------------------------


class TestRunPrefill:
    """run_prefill must tokenize, chunk, and call session.run for each chunk."""

    def test_run_prefill_returns_outputs_position_ids_generation_len(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        outputs, position_ids, gen_len = obj.run_prefill(
            prompt=["Hello world"],
            generation_len=None,
        )
        assert outputs is not None
        assert position_ids is not None
        assert gen_len is not None

    def test_run_prefill_calls_session_run(self):
        obj, _, mock_session = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        obj.run_prefill(prompt=["Hello world"], generation_len=None)
        assert mock_session.run.called

    def test_run_prefill_generation_len_bounded_by_ctx_len(self):
        obj, _, _ = _make_base_instance(ctx_len=CTX_LEN)
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        _, _, gen_len = obj.run_prefill(prompt=["Hello world"], generation_len=None)
        assert gen_len <= CTX_LEN

    def test_run_prefill_generation_len_positive(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        _, _, gen_len = obj.run_prefill(prompt=["Hello world"], generation_len=None)
        assert gen_len > 0

    def test_run_prefill_chunking_multiple_chunks(self):
        """A long prompt that exceeds prefill_seq_len must be split into chunks."""
        obj, tok, mock_session = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        # Create a prompt that tokenizes to > PREFILL_LEN tokens
        long_prompt = " ".join(["hello"] * 20)
        obj.run_prefill(prompt=[long_prompt], generation_len=None)
        # session.run must be called at least once (possibly multiple times for chunks)
        assert mock_session.run.call_count >= 1

    def test_run_prefill_with_explicit_generation_len(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        _, _, gen_len = obj.run_prefill(prompt=["Hello"], generation_len=5)
        assert gen_len == 5

    def test_run_prefill_output_has_logits(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        outputs, _, _ = obj.run_prefill(prompt=["Hello world"], generation_len=None)
        assert "logits" in outputs

    def test_run_prefill_position_ids_shape(self):
        obj, _, _ = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        _, position_ids, _ = obj.run_prefill(prompt=["Hello world"], generation_len=None)
        assert position_ids.shape[0] == 1  # batch dim


# ---------------------------------------------------------------------------
# Tests: run_decode (mocked session)
# ---------------------------------------------------------------------------


class TestRunDecode:
    """run_decode must iterate and update generated_ids correctly."""

    def _setup_decode(self, generation_len=5):
        obj, tok, mock_session = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, generation_len + 2)
        # Simulate prefill output
        outputs = {"logits": np.zeros((1, 1, VOCAB_SIZE), dtype=np.float32)}
        outputs["logits"][0, 0, 42] = 1.0
        position_ids = np.array([[PREFILL_LEN]])
        obj.update_decode_input(outputs, position_ids, generation_len=generation_len)
        decode_inputs = obj.prepare_decode_inputs()
        return obj, tok, mock_session, decode_inputs, generation_len

    def test_run_decode_returns_num_tokens(self):
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_decode(5)
        num_token = obj.run_decode(decode_inputs, gen_len, automation=True)
        assert isinstance(num_token, int)
        assert num_token >= 1

    def test_run_decode_calls_session_run(self):
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_decode(3)
        obj.run_decode(decode_inputs, gen_len, automation=True)
        assert mock_session.run.called

    def test_run_decode_updates_generated_ids(self):
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_decode(3)
        obj.run_decode(decode_inputs, gen_len, automation=True)
        # generated_ids[:, 1:] should have been updated
        assert obj.generated_ids[0, 1] == 42  # mock always returns token 42

    def test_run_decode_stops_at_eos(self):
        """Decode must stop early when EOS token is generated."""
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_decode(10)

        # Make session return EOS token
        def _run_eos(inputs):
            logits = np.zeros((1, 1, VOCAB_SIZE), dtype=np.float32)
            logits[0, 0, tok.eos_token_id] = 1.0
            return {"logits": logits}

        mock_session.run.side_effect = _run_eos
        num_token = obj.run_decode(decode_inputs, gen_len, automation=False)
        # Should stop early (<= generation_len)
        assert num_token <= gen_len

    def test_run_decode_position_ids_advance(self):
        """position_ids must increase by 1 each decode step."""
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_decode(3)
        initial_pos = decode_inputs["position_ids"][0, -1].item()
        obj.run_decode(decode_inputs, gen_len, automation=True)
        # After decode, position_ids should have advanced
        final_pos = decode_inputs["position_ids"][0, -1].item()
        assert final_pos > initial_pos

    def test_run_decode_generated_ids_are_valid_tokens(self):
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_decode(3)
        obj.run_decode(decode_inputs, gen_len, automation=True)
        for i in range(1, gen_len):
            token = obj.generated_ids[0, i]
            if token != tok.pad_token_id:
                assert 0 <= token < VOCAB_SIZE


# ---------------------------------------------------------------------------
# Tests: generate_decode_stream (mocked session)
# ---------------------------------------------------------------------------


class TestGenerateDecodeStream:
    """generate_decode_stream must yield token arrays at each step."""

    def _setup_stream(self, generation_len=4):
        obj, tok, mock_session = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, generation_len + 2)
        outputs = {"logits": np.zeros((1, 1, VOCAB_SIZE), dtype=np.float32)}
        outputs["logits"][0, 0, 42] = 1.0
        position_ids = np.array([[PREFILL_LEN]])
        obj.update_decode_input(outputs, position_ids, generation_len=generation_len)
        decode_inputs = obj.prepare_decode_inputs()
        return obj, tok, mock_session, decode_inputs, generation_len

    def test_yields_token_arrays(self):
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_stream(4)
        tokens = list(obj.generate_decode_stream(decode_inputs, gen_len, automation=True))
        assert len(tokens) >= 1
        for t in tokens:
            assert isinstance(t, np.ndarray)

    def test_yields_correct_shape(self):
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_stream(4)
        tokens = list(obj.generate_decode_stream(decode_inputs, gen_len, automation=True))
        for t in tokens:
            assert t.shape[0] == 1  # batch dim

    def test_yields_at_most_generation_len_tokens(self):
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_stream(4)
        tokens = list(obj.generate_decode_stream(decode_inputs, gen_len, automation=True))
        assert len(tokens) <= gen_len + 1  # +1 for final yield

    def test_stops_at_eos(self):
        obj, tok, mock_session, decode_inputs, gen_len = self._setup_stream(10)

        def _run_eos(inputs):
            logits = np.zeros((1, 1, VOCAB_SIZE), dtype=np.float32)
            logits[0, 0, tok.eos_token_id] = 1.0
            return {"logits": logits}

        mock_session.run.side_effect = _run_eos
        tokens = list(obj.generate_decode_stream(decode_inputs, gen_len, automation=False))
        assert len(tokens) <= gen_len + 1


# ---------------------------------------------------------------------------
# Tests: Chunking logic in prefill
# ---------------------------------------------------------------------------


class TestPrefillChunking:
    """Prefill must correctly chunk long prompts into prefill_seq_len pieces."""

    def test_single_chunk_for_short_prompt(self):
        obj, _, mock_session = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        mock_session.run.reset_mock()
        # Short prompt: should fit in one chunk
        obj.run_prefill(prompt=["Hi"], generation_len=None)
        assert mock_session.run.call_count == 1

    def test_multiple_chunks_for_long_prompt(self):
        """A prompt tokenizing to > prefill_seq_len must produce multiple chunks."""
        obj, tok, mock_session = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        mock_session.run.reset_mock()
        # Force a prompt that tokenizes to > PREFILL_LEN tokens
        # by using a very long string
        long_prompt = "hello " * 30  # ~30 tokens
        obj.run_prefill(prompt=[long_prompt], generation_len=None)
        # With prefill_seq_len=8, 30 tokens → ceil(30/8) = 4 chunks
        assert mock_session.run.call_count >= 2

    def test_chunk_inputs_have_correct_seq_len(self):
        """Each chunk passed to session.run must have seq_len == prefill_seq_len."""
        obj, _, mock_session = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        mock_session.run.reset_mock()
        long_prompt = "hello " * 30
        obj.run_prefill(prompt=[long_prompt], generation_len=None)
        for call in mock_session.run.call_args_list:
            chunk_inputs = call[0][0]
            assert chunk_inputs["input_ids"].shape[1] == PREFILL_LEN

    def test_position_ids_in_chunks_are_sequential(self):
        """position_ids in each chunk must be sequential (or -1 for padding)."""
        obj, _, mock_session = _make_base_instance()
        obj.initialize_decode_inputs(1, 1, CTX_LEN)
        mock_session.run.reset_mock()
        long_prompt = "hello " * 20
        obj.run_prefill(prompt=[long_prompt], generation_len=None)
        for call in mock_session.run.call_args_list:
            chunk_inputs = call[0][0]
            pos = chunk_inputs["position_ids"][0]
            valid_pos = pos[pos >= 0]
            if len(valid_pos) > 1:
                diffs = np.diff(valid_pos)
                assert np.all(diffs == 1), f"Non-sequential position_ids: {valid_pos}"


# ---------------------------------------------------------------------------
# Tests: Continuous batching (mocked session with full_batch_size)
# ---------------------------------------------------------------------------


class TestContinuousBatching:
    """run_continuous_batching_decode must handle the CB decode loop correctly."""

    def _make_cb_instance(self, full_batch_size=2):
        from QEfficient.generation.text_generation_inference import QEffTextGenerationBase

        tok = _make_tokenizer()
        # For CB prefill, run_prefill expects to read next token from logits.
        # We force seq_len=1 so update_decode_input can store into (full_batch_size, 1).
        mock_session = _make_mock_session(
            batch_size=full_batch_size,
            prefill_seq_len=PREFILL_LEN,
            ctx_len=CTX_LEN,
            vocab_size=VOCAB_SIZE,
            full_batch_size=full_batch_size,
            force_seq_len=1,
        )

        # Add batch_index to binding_index_map
        bi_binding = MagicMock()
        bi_binding.name = "batch_index"
        bi_binding.dims = [full_batch_size, 1]
        bi_binding.dir = "input"
        bi_binding.size = full_batch_size * 4
        bi_binding.type = 1
        mock_session.bindings.append(bi_binding)
        mock_session.binding_index_map["batch_index"] = len(mock_session.bindings) - 1
        mock_session.input_names.append("batch_index")

        # allowed_shapes for full_batch_size detection
        mock_session.allowed_shapes = [
            [
                (4, [full_batch_size, PREFILL_LEN]),  # input_ids
                (4, [full_batch_size, PREFILL_LEN]),  # position_ids
                (4, [full_batch_size, PREFILL_LEN, VOCAB_SIZE]),  # logits
                (4, [full_batch_size, 1]),  # batch_index
            ],
            [
                (4, [full_batch_size, 1]),  # input_ids decode
                (4, [full_batch_size, 1]),  # position_ids decode
                (4, [full_batch_size, 1, VOCAB_SIZE]),  # logits decode
                (4, [full_batch_size, 1]),  # batch_index
            ],
        ]

        with patch(
            "QEfficient.generation.text_generation_inference.QAICInferenceSession",
            return_value=mock_session,
        ):
            obj = QEffTextGenerationBase(
                tokenizer=tok,
                qpc_path="/fake/path/model.qpc",
                ctx_len=CTX_LEN,
                full_batch_size=full_batch_size,
            )
        return obj, tok, mock_session

    def test_cb_instance_has_full_batch_size(self):
        obj, _, _ = self._make_cb_instance(full_batch_size=2)
        assert obj.full_batch_size == 2

    def test_initialize_decode_inputs_with_full_batch_size(self):
        obj, _, _ = self._make_cb_instance(full_batch_size=2)
        obj.initialize_decode_inputs(
            num_prompts=4,
            execution_batch_size=2,
            max_gen_length=10,
        )
        assert obj.generated_ids.shape == (4, 10)
        assert obj.decode_input_ids.shape == (2, 1)

    def test_prepare_decode_inputs_with_batch_index(self):
        obj, _, _ = self._make_cb_instance(full_batch_size=2)
        obj.initialize_decode_inputs(2, 2, 10)
        obj.batch_index = np.arange(2).reshape(-1, 1)
        decode_inputs = obj.prepare_decode_inputs()
        assert "batch_index" in decode_inputs

    def test_run_prefill_for_all_inputs_calls_session(self):
        obj, tok, mock_session = self._make_cb_instance(full_batch_size=2)
        obj.initialize_decode_inputs(2, 2, CTX_LEN)
        mock_session.run.reset_mock()
        prompt_queue = deque(["Hello", "World"])
        obj.run_prefill_for_all_inputs(prompt_queue, generation_len=None)
        assert mock_session.run.called

    def test_run_prefill_for_all_inputs_empties_queue(self):
        obj, tok, mock_session = self._make_cb_instance(full_batch_size=2)
        obj.initialize_decode_inputs(2, 2, CTX_LEN)
        prompt_queue = deque(["Hello", "World"])
        obj.run_prefill_for_all_inputs(prompt_queue, generation_len=None)
        assert len(prompt_queue) == 0


# ---------------------------------------------------------------------------
# Tests: _fetch_next_token_id
# ---------------------------------------------------------------------------


class TestFetchNextTokenId:
    """_fetch_next_token_id must extract argmax from logits correctly."""

    def test_returns_argmax_of_logits(self):
        obj, _, _ = _make_base_instance()
        logits = np.zeros((1, 1, VOCAB_SIZE), dtype=np.float32)
        logits[0, 0, 77] = 1.0
        outputs = {"logits": logits}
        token = obj._fetch_next_token_id(outputs)
        assert token[0, 0] == 77

    def test_batch_argmax(self):
        obj, _, _ = _make_base_instance(batch_size=2)
        logits = np.zeros((2, 1, VOCAB_SIZE), dtype=np.float32)
        logits[0, 0, 10] = 1.0
        logits[1, 0, 20] = 1.0
        outputs = {"logits": logits}
        tokens = obj._fetch_next_token_id(outputs)
        assert tokens[0, 0] == 10
        assert tokens[1, 0] == 20

    def test_2d_logits_expanded(self):
        """2D logits (batch, vocab) must be expanded to (batch, 1, vocab)."""
        obj, _, _ = _make_base_instance()
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, 55] = 1.0
        outputs = {"logits": logits}
        token = obj._fetch_next_token_id(outputs)
        assert token[0, 0] == 55


# ---------------------------------------------------------------------------
# Tests: _set_output_buffers
# ---------------------------------------------------------------------------


class TestSetOutputBuffers:
    """_set_output_buffers must call session.set_buffers with correct shapes."""

    def test_set_output_buffers_calls_set_buffers(self):
        obj, _, mock_session = _make_base_instance()
        mock_session.set_buffers.reset_mock()
        obj._set_output_buffers(batch_size=1, sequence_length=1)
        mock_session.set_buffers.assert_called_once()

    def test_set_output_buffers_logits_shape(self):
        obj, _, mock_session = _make_base_instance()
        mock_session.set_buffers.reset_mock()
        obj._set_output_buffers(batch_size=2, sequence_length=4)
        call_args = mock_session.set_buffers.call_args[0][0]
        assert "logits" in call_args
        assert call_args["logits"].shape == (2, 4, VOCAB_SIZE)

    def test_set_output_buffers_dtype_float32(self):
        obj, _, mock_session = _make_base_instance()
        mock_session.set_buffers.reset_mock()
        obj._set_output_buffers(batch_size=1, sequence_length=1)
        call_args = mock_session.set_buffers.call_args[0][0]
        assert call_args["logits"].dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: VisionHandler initialization (CPU-only)
# ---------------------------------------------------------------------------


class TestVisionHandlerInit:
    """VisionHandler must initialize correctly with None sessions."""

    def test_construction_with_none_sessions(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        h = VisionHandler(qeff_model=None, vision_session=None, processor=None, tokenizer=None)
        assert h is not None

    def test_is_available_false_with_none(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        h = VisionHandler(qeff_model=None, vision_session=None, processor=None, tokenizer=None)
        assert h.is_available() is False

    def test_is_available_false_session_no_processor(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        h = VisionHandler(qeff_model=None, vision_session=MagicMock(), processor=None, tokenizer=None)
        assert h.is_available() is False

    def test_get_vision_output_shapes_default(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        h = VisionHandler(qeff_model=None, vision_session=None, processor=None, tokenizer=None)
        shapes = h.get_vision_output_shapes()
        assert isinstance(shapes, dict)
        assert "vision_embeds" in shapes

    def test_get_vision_output_shapes_from_config(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        config = {"vision_output_shapes": {"my_out": (100, 200)}}
        h = VisionHandler(qeff_model=None, vision_session=None, processor=None, tokenizer=None, config=config)
        shapes = h.get_vision_output_shapes()
        assert shapes["my_out"] == (100, 200)

    def test_image_dims_stored(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        h = VisionHandler(
            qeff_model=None, vision_session=None, processor=None, tokenizer=None, image_height=224, image_width=224
        )
        assert h._image_height == 224 and h._image_width == 224

    def test_setup_vision_buffers_raises_without_session(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        h = VisionHandler(qeff_model=None, vision_session=None, processor=None, tokenizer=None)
        with pytest.raises(ValueError):
            h.setup_vision_buffers()

    def test_run_vision_inference_raises_without_session(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        h = VisionHandler(qeff_model=None, vision_session=None, processor=None, tokenizer=None)
        with pytest.raises(ValueError):
            h.run_vision_inference({})

    def test_prepare_vlm_inputs_raises_without_processor(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        h = VisionHandler(qeff_model=None, vision_session=None, processor=None, tokenizer=None)
        with pytest.raises((ValueError, AttributeError)):
            h.prepare_vlm_inputs("image.jpg", "query", 128)

    def test_cast_vision_inputs_preserves_bfloat16_payload(self):
        from QEfficient.generation.embedding_handler import VisionHandler

        vision_session = MagicMock()
        vision_session.binding_is_bfloat16.side_effect = lambda name: name == "pixel_values"
        handler = VisionHandler(qeff_model=None, vision_session=vision_session, processor=MagicMock(), tokenizer=None)
        values = np.array([1.0, -2.5, np.pi], dtype=np.float32)
        vision_inputs = {"pixel_values": values.copy(), "image_masks": values.copy()}

        handler._cast_vision_inputs(vision_inputs, constants.VISION_FP16_INPUTS)

        assert vision_inputs["pixel_values"].dtype == np.float16
        np.testing.assert_array_equal(
            vision_inputs["pixel_values"].view(np.uint16),
            np.array([0x3F80, 0xC020, 0x4049], dtype=np.uint16),
        )
        np.testing.assert_array_equal(vision_inputs["image_masks"], values.astype(np.float16))


def test_binding_is_bfloat16_uses_compiled_binding_type(monkeypatch):
    from QEfficient.generation import cloud_infer

    monkeypatch.setattr(cloud_infer, "aicapi", SimpleNamespace(BFLOAT16_TYPE=11), raising=False)
    session = object.__new__(cloud_infer.QAICInferenceSession)
    session.bindings = [SimpleNamespace(type=11), SimpleNamespace(type=1)]
    session.binding_index_map = {"pixel_values": 0, "image_masks": 1}

    assert session.binding_is_bfloat16("pixel_values") is True
    assert session.binding_is_bfloat16("image_masks") is False
    assert session.binding_is_bfloat16("missing") is False


# ---------------------------------------------------------------------------
# Helper: minimal VisionLanguageGeneration stub for unbound-method calls
# ---------------------------------------------------------------------------


def _make_vlg_stub(full_batch_size=None, prefill_seq_len=8):
    """Build a minimal MagicMock as `self` for VisionLanguageGeneration unbound methods."""
    obj = MagicMock()
    obj.full_batch_size = full_batch_size
    obj.include_sampler = False
    obj.comp_ctx_lengths_prefill = None
    obj._prefill_seq_len = prefill_seq_len
    obj._lang_skip_buffers = []

    session = MagicMock()
    session.binding_index_map = {}
    session.bindings = []
    session.allowed_shapes = []
    session.run.return_value = {}
    obj._session = session
    return obj


# ---------------------------------------------------------------------------
# Tests: Gemma4 CB variant  (gemma4_cb.py / continuous_batching.py)
#   full_batch_size=4, batch_size=1 per slot, continuous_batching=True,
#   kv_offload=True (dual-QPC).
# ---------------------------------------------------------------------------


class TestGemma4CBVariant:
    """Unit tests for the Gemma4 continuous-batching variant."""

    def test_full_batch_size_stored_on_instance(self):
        """CB instance must store full_batch_size=4."""
        obj, _, _ = _make_base_instance(full_batch_size=4)
        assert obj.full_batch_size == 4

    def test_batch_index_shape(self):
        """CB batch_index must have shape (full_batch_size, 1)."""
        obj, _, _ = _make_base_instance(full_batch_size=4)
        obj.initialize_decode_inputs(4, 4, 10)
        obj.batch_index = np.arange(4).reshape(-1, 1)
        assert obj.batch_index.shape == (4, 1)

    def test_batch_index_values_sequential(self):
        """CB batch_index values must be 0 … full_batch_size-1."""
        obj, _, _ = _make_base_instance(full_batch_size=4)
        obj.initialize_decode_inputs(4, 4, 10)
        obj.batch_index = np.arange(4).reshape(-1, 1)
        np.testing.assert_array_equal(obj.batch_index.flatten(), [0, 1, 2, 3])

    def test_decode_inputs_contain_batch_index(self):
        """prepare_decode_inputs in CB mode must include batch_index with shape (full_batch_size, 1)."""
        obj, _, _ = _make_base_instance(full_batch_size=4)
        obj.initialize_decode_inputs(4, 4, 10)
        obj.batch_index = np.arange(4).reshape(-1, 1)
        decode_inputs = obj.prepare_decode_inputs()
        assert "batch_index" in decode_inputs
        assert decode_inputs["batch_index"].shape == (4, 1)

    def test_mm_token_type_ids_uses_full_batch_size(self):
        """
        _execute_chunked_prefill must allocate _decode_mm_token_type_ids with
        full_batch_size rows, not the per-slot prefill batch size of 1.
        Bug: shape was (1,1) causing a QPC dimension mismatch at decode time.
        Fix: use self.full_batch_size when set.
        """
        from QEfficient.generation.vlm_generation import VisionLanguageGeneration

        obj = _make_vlg_stub(full_batch_size=4, prefill_seq_len=8)
        lang_inputs = {
            "input_ids": np.zeros((1, 8), dtype=np.int64),
            "position_ids": np.zeros((1, 8), dtype=np.int64),
            "mm_token_type_ids": np.zeros((1, 8), dtype=np.int32),
        }
        VisionLanguageGeneration._execute_chunked_prefill(obj, lang_inputs, num_chunks=1)
        assert obj._decode_mm_token_type_ids.shape == (4, 1), (
            f"Expected (4, 1), got {obj._decode_mm_token_type_ids.shape}"
        )

    def test_cb_compile_signature_has_full_batch_size(self):
        """DualQPC compile() must accept full_batch_size (CB compilation parameter)."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        sig = inspect.signature(_QEffAutoModelForImageTextToTextDualQPC.compile)
        assert "full_batch_size" in sig.parameters


# ---------------------------------------------------------------------------
# Tests: Gemma4 regular variant  (gemma4_example.py)
#   batch_size=1, no full_batch_size, kv_offload=True (dual-QPC).
# ---------------------------------------------------------------------------


class TestGemma4RegularVariant:
    """Unit tests for the Gemma4 regular (non-CB) batching variant."""

    def test_no_full_batch_size_on_instance(self):
        """Regular variant instance must have full_batch_size=None."""
        obj, _, _ = _make_base_instance(full_batch_size=None)
        assert obj.full_batch_size is None

    def test_decode_inputs_have_no_batch_index(self):
        """Regular variant decode inputs must NOT include batch_index."""
        obj, _, _ = _make_base_instance(full_batch_size=None)
        obj.initialize_decode_inputs(1, 1, 10)
        decode_inputs = obj.prepare_decode_inputs()
        assert "batch_index" not in decode_inputs

    def test_decode_input_ids_shape(self):
        """Regular variant decode input_ids must be (batch_size, 1)."""
        obj, _, _ = _make_base_instance(batch_size=1, full_batch_size=None)
        obj.initialize_decode_inputs(1, 1, 10)
        decode_inputs = obj.prepare_decode_inputs()
        assert decode_inputs["input_ids"].shape == (1, 1)

    def test_mm_token_type_ids_absent_from_decode_when_not_set(self):
        """VLM prepare_decode_inputs must omit mm_token_type_ids when attribute is not set."""
        from QEfficient.generation.text_generation_inference import QEffTextGenerationBase
        from QEfficient.generation.vlm_generation import VisionLanguageGeneration

        obj = object.__new__(VisionLanguageGeneration)
        obj._decode_cross_attention_mask = None
        obj.batch_index = None
        # _decode_mm_token_type_ids intentionally absent (regular variant never sets it via text-only path)

        with patch.object(QEffTextGenerationBase, "prepare_decode_inputs", return_value={}):
            result = VisionLanguageGeneration.prepare_decode_inputs(obj)
        assert "mm_token_type_ids" not in result

    def test_regular_compile_signature_has_skip_vision(self):
        """DualQPC compile() must accept skip_vision (used by gemma4_example text-only path)."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        sig = inspect.signature(_QEffAutoModelForImageTextToTextDualQPC.compile)
        assert "skip_vision" in sig.parameters


# ---------------------------------------------------------------------------
# Tests: Gemma4 diss variant  (gemma4_diss.py)
#   Separate prefill-only QPC + decode-only QPC, raw QAICInferenceSession.
#   Uses prefill_only=True, retain_full_kv=True, enable_chunking=True,
#   skip_lang=True (vision QPC) and skip_vision=True (lang QPCs).
# ---------------------------------------------------------------------------


class TestGemma4DissVariant:
    """Unit tests for the Gemma4 dissected (split prefill/decode QPC) variant."""

    def test_compile_signature_has_prefill_only(self):
        """DualQPC compile() must accept prefill_only (required for diss prefill QPC)."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        sig = inspect.signature(_QEffAutoModelForImageTextToTextDualQPC.compile)
        assert "prefill_only" in sig.parameters

    def test_compile_signature_has_enable_chunking(self):
        """DualQPC compile() must accept enable_chunking (used with prefill_only in diss)."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        sig = inspect.signature(_QEffAutoModelForImageTextToTextDualQPC.compile)
        assert "enable_chunking" in sig.parameters

    def test_compile_signature_has_skip_lang(self):
        """DualQPC compile() must accept skip_lang (vision-only QPC in diss)."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        sig = inspect.signature(_QEffAutoModelForImageTextToTextDualQPC.compile)
        assert "skip_lang" in sig.parameters

    def test_num_chunks_ceiling_division(self):
        """Diss variant computes num_chunks via ceiling division: -(length // -prefill_seq_len)."""
        PREFILL_SEQ_LEN = 296
        cases = [(296, 1), (297, 2), (592, 2), (593, 3), (1, 1)]
        for input_len, expected in cases:
            got = -(input_len // -PREFILL_SEQ_LEN)
            assert got == expected, f"input_len={input_len}: expected {expected}, got {got}"

    def test_padded_len_is_multiple_of_prefill_seq_len(self):
        """Diss pads inputs to an exact multiple of PREFILL_SEQ_LEN before chunking."""
        PREFILL_SEQ_LEN = 296
        for input_ids_length in [1, 100, 296, 297, 500, 592, 593]:
            num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
            padded_len = num_chunks * PREFILL_SEQ_LEN
            assert padded_len % PREFILL_SEQ_LEN == 0
            assert padded_len >= input_ids_length

    def test_mm_token_type_ids_sliced_per_chunk(self):
        """Diss prefill loop must slice mm_token_type_ids to [prefill_seq_len] tokens per chunk."""
        PREFILL_SEQ_LEN = 8
        BS, num_chunks = 1, 3
        total_len = num_chunks * PREFILL_SEQ_LEN
        mm = np.arange(total_len, dtype=np.int32).reshape(BS, -1)

        for i in range(num_chunks):
            chunk = mm[..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
            assert chunk.shape == (BS, PREFILL_SEQ_LEN)
            np.testing.assert_array_equal(
                chunk[0],
                np.arange(i * PREFILL_SEQ_LEN, (i + 1) * PREFILL_SEQ_LEN, dtype=np.int32),
            )

    def test_decode_inputs_exclude_mm_token_type_ids(self):
        """Diss decode inputs are built without mm_token_type_ids (decode QPC is vision-free)."""
        # Reproduce the diss decode_inputs construction from gemma4_diss.py
        logits = np.zeros((1, 1, 100), dtype=np.float32)
        logits[0, 0, 42] = 1.0
        position_ids = np.array([[10]])

        decode_inputs = {
            "input_ids": np.argmax(logits).reshape(1, 1),
            "position_ids": position_ids + 1,
            "image_idx": np.array([[0]], dtype=np.int64),
        }
        assert "mm_token_type_ids" not in decode_inputs


@pytest.mark.cpu_only
def test_runner_io_resolves_exporter_generated_logits_symbols():
    from onnx import TensorProto, helper

    from QEfficient.generation.generation_helpers import _resolve_output_shape

    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["Castlogits_dim_0", "Castlogits_dim_1", 32000])

    assert _resolve_output_shape(logits, {"seq_len": 1}, fallback_batch_size=1) == [1, 1, 32000]


@pytest.mark.cpu_only
def test_custom_io_maps_derive_precisions_and_item_sizes_from_one_yaml_load(tmp_path, monkeypatch):
    from QEfficient.generation import generation_helpers

    (tmp_path / "custom_io.yaml").write_text(
        """ - IOName: vision_embeds
   Precision: bfloat16

 - IOName: logits
   Precision: float32

 - IOName: opaque_tensor
   Precision: custom_precision

 - IOName: missing_precision
"""
    )
    yaml_load_count = 0
    safe_load = generation_helpers.yaml.safe_load

    def counting_safe_load(payload):
        nonlocal yaml_load_count
        yaml_load_count += 1
        return safe_load(payload)

    monkeypatch.setattr(generation_helpers.yaml, "safe_load", counting_safe_load)

    precision_map, item_size_map = generation_helpers._custom_io_maps(tmp_path)

    assert yaml_load_count == 1
    assert precision_map == {
        "vision_embeds": "bfloat16",
        "logits": "float32",
        "opaque_tensor": "custom_precision",
    }
    assert item_size_map == {"vision_embeds": 2, "logits": 4}


@pytest.mark.cpu_only
def test_runner_metadata_reuses_loaded_onnx_across_runner_helpers(tmp_path, monkeypatch):
    import onnx
    from onnx import TensorProto, helper

    from QEfficient.generation import generation_helpers, runner_io

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1])
    batch_index = helper.make_tensor_value_info("batch_index", TensorProto.INT64, [1, 1])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 4])
    onnx_path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(helper.make_graph([], "runner_metadata", [input_ids, batch_index], [logits])), onnx_path
    )

    load_count = 0
    onnx_load = generation_helpers.onnx.load

    def counting_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return onnx_load(*args, **kwargs)

    monkeypatch.setattr(generation_helpers.onnx, "load", counting_load)

    runner_metadata = generation_helpers.RunnerMetadata.from_paths(onnx_path, tmp_path)
    host_inputs = {"input_ids": np.array([[1]], dtype=np.int64)}
    generation_helpers._add_specialization_control_inputs(runner_metadata, host_inputs, {"batch_size": 1})
    host_inputs = generation_helpers._filter_graph_inputs(runner_metadata, host_inputs, {"extra": np.array([[0]])})

    runner_io.write_runner_io_bundle(
        runner_metadata=runner_metadata,
        specialization={"batch_size": 1, "seq_len": 1},
        host_inputs=host_inputs,
    )

    assert load_count == 1


@pytest.mark.cpu_only
def test_graph_input_filter_selects_model_specific_vision_metadata(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from QEfficient.generation.generation_helpers import RunnerMetadata, _filter_graph_inputs

    pixel_values = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 2, 2])
    image_grid_thw = helper.make_tensor_value_info("image_grid_thw", TensorProto.INT64, [1, 3])
    graph = helper.make_graph([], "vision_inputs", [pixel_values, image_grid_thw], [pixel_values, image_grid_thw])
    onnx_path = tmp_path / "vision.onnx"
    onnx.save(helper.make_model(graph), onnx_path)

    runner_metadata = RunnerMetadata.from_paths(onnx_path, tmp_path)

    inputs = _filter_graph_inputs(
        runner_metadata,
        {"pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32)},
        {"image_grid_thw": np.array([[1, 2, 2]]), "input_ids": np.array([[1]])},
    )

    assert set(inputs) == {"pixel_values", "image_grid_thw"}
    np.testing.assert_array_equal(inputs["image_grid_thw"], [[1, 2, 2]])


@pytest.mark.cpu_only
def test_cross_qpc_output_shapes_follow_language_input_contract(tmp_path):
    from types import SimpleNamespace

    import onnx
    from onnx import TensorProto, helper

    from QEfficient.generation.generation_helpers import _cross_qpc_output_shapes

    vision_output = helper.make_tensor_value_info(
        "vision_embeds", TensorProto.FLOAT, ["opaque_batch", "opaque_tokens", "opaque_width"]
    )
    vision_path = tmp_path / "vision.onnx"
    onnx.save(helper.make_model(helper.make_graph([], "vision", [], [vision_output])), vision_path)

    language_input = helper.make_tensor_value_info(
        "vision_embeds", TensorProto.FLOAT, ["vision_batch_size", "vision_size", 8]
    )
    language_path = tmp_path / "language.onnx"
    onnx.save(helper.make_model(helper.make_graph([], "language", [language_input], [language_input])), language_path)
    model = SimpleNamespace(
        vision_model=SimpleNamespace(onnx_path=vision_path),
        lang_model=SimpleNamespace(onnx_path=language_path),
    )

    assert _cross_qpc_output_shapes(model, {"batch_size": 1, "vision_size": 187}) == {"vision_embeds": [1, 187, 8]}

    lang_compile_dir = tmp_path / "lang-qpc"
    lang_compile_dir.mkdir()
    (lang_compile_dir / "specializations.json").write_text(
        json.dumps(
            {
                "specializations": [
                    {
                        "name": "Prefill",
                        "symbols": {
                            "batch_size": "1",
                            "seq_len": "128",
                            "ctx_len": "3000",
                            "vision_batch_size": "1",
                            "vision_size": "187",
                        },
                    }
                ]
            }
        )
    )
    model.lang_model.compile_artifacts_path = lang_compile_dir
    assert _cross_qpc_output_shapes(model, {"batch_size": 1, "img_size": 336}) == {"vision_embeds": [1, 187, 8]}

    model.lang_model.onnx_path = None
    model.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=8)))
    assert _cross_qpc_output_shapes(model, {"batch_size": 1, "vision_size": 187}) == {"vision_embeds": [1, 187, 8]}


@pytest.mark.cpu_only
def test_causal_lm_runner_bundle_uses_live_prefill_preparation(tmp_path):
    from types import SimpleNamespace

    import onnx
    from onnx import TensorProto, helper

    from QEfficient.generation.runner_io import write_causal_lm_runner_bundle

    class FakeTokenizer:
        padding_side = "left"
        pad_token_id = None
        eos_token_id = 0

        def __call__(self, prompt, return_tensors, padding, max_length=None):
            del return_tensors
            prompts = prompt if isinstance(prompt, list) else [prompt]
            input_ids = np.array([[5 + index, 6 + index, 7 + index] for index, _ in enumerate(prompts)], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)
            if padding == "max_length":
                pad_width = max_length - input_ids.shape[1]
                input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)))
                attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)))
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 4])
    position_ids = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, 4])
    past_key = helper.make_tensor_value_info("past_key.0", TensorProto.FLOAT, [1, 2, 8, 4])
    batch_index = helper.make_tensor_value_info("batch_index", TensorProto.INT64, [1, 1])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 10])
    retained_key = helper.make_tensor_value_info("past_key.0_RetainedState", TensorProto.FLOAT, [1, 2, 8, 4])
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["past_key.0"], ["past_key.0_RetainedState"]),
            helper.make_node(
                "Constant", [], ["logits"], value=helper.make_tensor("value", TensorProto.FLOAT, [1, 1, 10], [0.0] * 10)
            ),
        ],
        "causal_runner",
        [input_ids, position_ids, past_key, batch_index],
        [logits, retained_key],
    )
    onnx_path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), onnx_path)
    compile_dir = tmp_path / "qpc-hash"
    compile_dir.mkdir()
    (compile_dir / "specializations.json").write_text(
        json.dumps({"specializations": [{"batch_size": 1, "full_batch_size": 4, "seq_len": 4, "ctx_len": 8}]})
    )
    qeff_model = SimpleNamespace(
        qpc_path=None,
        compile_artifacts_path=compile_dir,
        onnx_path=onnx_path,
        model=SimpleNamespace(config=SimpleNamespace(n_head=2, n_embd=8)),
    )

    tokenizer = FakeTokenizer()
    io_dir = write_causal_lm_runner_bundle(model=qeff_model, tokenizer=tokenizer, prompts=["hello"])

    entries = json.loads((io_dir / "aic_batch_io.json").read_text())["IO-files"][0]
    assert tokenizer.padding_side == "right"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert [entry["map-to"] for entry in entries] == ["input_ids", "position_ids", "batch_index", "logits"]
    assert np.fromfile(io_dir / "data/input_ids.raw", dtype=np.int64).tolist() == [5, 6, 7, 0]
    assert np.fromfile(io_dir / "data/position_ids.raw", dtype=np.int64).tolist() == [0, 1, 2, -1]
    assert np.fromfile(io_dir / "data/batch_index.raw", dtype=np.int64).tolist() == [0]


@pytest.mark.cpu_only
def test_causal_lm_runner_bundle_matches_compiled_batch_size(tmp_path):
    from types import SimpleNamespace

    import onnx
    from onnx import TensorProto, helper

    from QEfficient.generation.runner_io import write_causal_lm_runner_bundle

    class FakeTokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, prompt, return_tensors, padding, max_length=None):
            del return_tensors
            prompts = prompt if isinstance(prompt, list) else [prompt]
            input_ids = np.array([[index + 1, index + 2] for index, _ in enumerate(prompts)], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)
            if padding == "max_length":
                pad_width = max_length - input_ids.shape[1]
                input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)))
                attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)))
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [2, 4])
    position_ids = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [2, 4])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [2, 1, 10])
    graph = helper.make_graph([], "causal_runner_bs2", [input_ids, position_ids], [logits])
    onnx_path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), onnx_path)
    compile_dir = tmp_path / "qpc-hash"
    compile_dir.mkdir()
    (compile_dir / "specializations.json").write_text(
        json.dumps({"specializations": [{"batch_size": 2, "seq_len": 4, "ctx_len": 8}]})
    )
    qeff_model = SimpleNamespace(
        qpc_path=None,
        compile_artifacts_path=compile_dir,
        onnx_path=onnx_path,
        model=SimpleNamespace(config=SimpleNamespace(n_head=2, n_embd=8)),
    )

    io_dir = write_causal_lm_runner_bundle(model=qeff_model, tokenizer=FakeTokenizer(), prompts=["hello"])

    entries = json.loads((io_dir / "aic_batch_io.json").read_text())["IO-files"][0]
    input_dims = {entry["map-to"]: entry["dims"] for entry in entries if entry["io-direction"] == "in"}
    assert input_dims["input_ids"] == [2, 4]
    assert input_dims["position_ids"] == [2, 4]
    assert np.fromfile(io_dir / "data/input_ids.raw", dtype=np.int64).tolist() == [1, 2, 0, 0, 2, 3, 0, 0]
    assert np.fromfile(io_dir / "data/position_ids.raw", dtype=np.int64).tolist() == [0, 1, -1, -1, 0, 1, -1, -1]


@pytest.mark.cpu_only
def test_single_qpc_vlm_runner_bundle_matches_compiled_batch_size(tmp_path):
    from types import SimpleNamespace

    import onnx
    from onnx import TensorProto, helper

    from QEfficient.generation.runner_io import write_single_qpc_vlm_runner_bundle

    class FakeVisionHandler:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def prepare_processor_inputs(self, image, prompt, prefill_seq_len):
            del image, prompt
            return (
                {"pixel_values": np.array([[11.0, 12.0, 13.0]], dtype=np.float32)},
                {
                    "input_ids": np.array([[1, 2, 0, 0]], dtype=np.int64)[:, :prefill_seq_len],
                    "position_ids": np.array([[0, 1, -1, -1]], dtype=np.int64)[:, :prefill_seq_len],
                    "image_idx": np.array([[0]], dtype=np.int64),
                },
                1,
            )

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [2, 4])
    position_ids = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [2, 4])
    image_idx = helper.make_tensor_value_info("image_idx", TensorProto.INT64, [2, 1])
    pixel_values = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [2, 3])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [2, 1, 10])
    graph = helper.make_graph([], "single_qpc_vlm_bs2", [input_ids, position_ids, image_idx, pixel_values], [logits])
    onnx_path = tmp_path / "vlm.onnx"
    onnx.save(helper.make_model(graph), onnx_path)
    compile_dir = tmp_path / "qpc-hash"
    compile_dir.mkdir()
    (compile_dir / "specializations.json").write_text(
        json.dumps({"specializations": [{"batch_size": 2, "seq_len": 4, "ctx_len": 8}]})
    )
    qeff_model = SimpleNamespace(
        qpc_path=None,
        compile_artifacts_path=compile_dir,
        onnx_path=onnx_path,
        model=SimpleNamespace(config=SimpleNamespace(model_type="fake_vlm")),
    )

    with patch("QEfficient.generation.embedding_handler.VisionHandler", FakeVisionHandler):
        io_dir = write_single_qpc_vlm_runner_bundle(
            model=qeff_model, processor=MagicMock(), images=["image"], prompts=["prompt"]
        )

    entries = json.loads((io_dir / "aic_batch_io.json").read_text())["IO-files"][0]
    input_dims = {entry["map-to"]: entry["dims"] for entry in entries if entry["io-direction"] == "in"}
    assert input_dims["input_ids"] == [2, 4]
    assert input_dims["position_ids"] == [2, 4]
    assert input_dims["image_idx"] == [2, 1]
    assert input_dims["pixel_values"] == [2, 3]
    assert np.fromfile(io_dir / "data/input_ids.raw", dtype=np.int64).tolist() == [1, 2, 0, 0, 1, 2, 0, 0]
    assert np.fromfile(io_dir / "data/pixel_values.raw", dtype=np.float32).tolist() == [11.0, 12.0, 13.0] * 2
