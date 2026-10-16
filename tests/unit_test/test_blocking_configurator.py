# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import pytest

from QEfficient.blocking.blocking_configurator import build_transformer_blocking_config_for_transform


def _minimax_config(*, index_block_size=8, num_kv_heads=2):
    text_config = SimpleNamespace(
        model_type="minimax_m3",
        layer_types=["minimax_m3_sparse"],
        index_block_size=index_block_size,
        num_attention_heads=8,
        num_key_value_heads=num_kv_heads,
        hidden_size=64,
    )
    return SimpleNamespace(model_type="minimax_m3_vl", text_config=text_config)


def _build_minimax_blocking_config(seq_len, **overrides):
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": 1,
        "num_cores_per_device": 4,
        "indexer_q_size": 8,
        "indexer_q_chunk": 32,
        "msa_q_chunk": 32,
        **overrides,
    }
    return build_transformer_blocking_config_for_transform(
        _minimax_config(),
        ctx_len=seq_len,
        seq_len=seq_len,
        qaic_config=qaic_config,
        aic_num_cores=4,
    )


@pytest.mark.cpu_only
def test_minimax_prefill_export_scaling_preserves_nested_loop_counts(caplog):
    with caplog.at_level("INFO", logger="QEfficient.blocking.blocking_configurator"):
        blocking_config = _build_minimax_blocking_config(128)

    assert blocking_config.prefill_export_seq_len == 64
    assert blocking_config.prefill_compile_seq_len == 128
    assert blocking_config.indexer_q_size == 4
    assert blocking_config.indexer_q_chunk == 16
    assert blocking_config.msa_q_chunk == 16
    assert blocking_config.indexer_q_proj_num_chunks == 1
    assert [32 // 8] * (128 // 32) == [16 // 4] * (64 // 16)
    assert 128 // 32 == 64 // 16
    assert "Exporting MiniMax M3 prefill ONNX with seq_len=64 (compiled prefill seq_len=128)." in caplog.messages


@pytest.mark.cpu_only
def test_minimax_parallel_indexer_uses_smallest_valid_export_length():
    blocking_config = _build_minimax_blocking_config(128, indexer_prefill_parallel=True)

    assert blocking_config.prefill_export_seq_len == 16
    assert blocking_config.indexer_q_size == 1
    assert blocking_config.indexer_q_chunk == 4
    assert blocking_config.msa_q_chunk == 4
    assert blocking_config.indexer_q_proj_num_chunks == 1
    assert 128 // 32 == 16 // 4
    assert 32 // 8 == 4 // 1


@pytest.mark.cpu_only
def test_minimax_prefill_scaling_preserves_index_projection_loop_count():
    blocking_config = build_transformer_blocking_config_for_transform(
        _minimax_config(index_block_size=128, num_kv_heads=8),
        ctx_len=4096,
        seq_len=4096,
        qaic_config={
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": 1,
            "num_cores_per_device": 8,
            "indexer_prefill_parallel": True,
            "indexer_q_size": 128,
            "indexer_q_chunk": 512,
            "msa_q_chunk": 64,
        },
        aic_num_cores=8,
    )

    assert blocking_config.prefill_export_seq_len == 64
    assert blocking_config.indexer_q_size == 2
    assert blocking_config.indexer_q_chunk == 8
    assert blocking_config.msa_q_chunk == 1
    assert blocking_config.indexer_q_proj_num_chunks == 8


@pytest.mark.cpu_only
def test_minimax_prefill_scaling_resolves_forward_defaults():
    blocking_config = _build_minimax_blocking_config(
        128,
        indexer_q_size=None,
        indexer_q_chunk=None,
        msa_q_chunk=None,
    )

    assert blocking_config.prefill_export_seq_len == 64
    assert blocking_config.indexer_q_size == 4
    assert blocking_config.indexer_q_chunk == 4
    assert blocking_config.msa_q_chunk == 64


@pytest.mark.cpu_only
def test_minimax_prefill_scaling_rejects_invalid_full_length_chunks():
    with pytest.raises(ValueError, match="divisible by indexer_q_chunk"):
        _build_minimax_blocking_config(130)


@pytest.mark.cpu_only
def test_minimax_prefill_scaling_rejects_non_positive_msa_chunk():
    with pytest.raises(ValueError, match="msa_q_chunk to be positive"):
        _build_minimax_blocking_config(128, msa_q_chunk=-1)


@pytest.mark.cpu_only
def test_non_minimax_blocking_values_are_not_scaled():
    model_config = SimpleNamespace(
        model_type="llama",
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=64,
    )
    blocking_config = build_transformer_blocking_config_for_transform(
        model_config,
        ctx_len=128,
        seq_len=128,
        qaic_config={
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": 1,
            "indexer_q_size": 8,
            "indexer_q_chunk": 32,
            "msa_q_chunk": 32,
        },
        aic_num_cores=4,
    )

    assert blocking_config.indexer_q_size == 8
    assert blocking_config.indexer_q_chunk == 32
    assert blocking_config.msa_q_chunk == 32
    assert blocking_config.prefill_export_seq_len is None
