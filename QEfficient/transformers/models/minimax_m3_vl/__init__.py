# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
    MiniMaxM3VLConfig,
    MiniMaxM3VLTextConfig,
    MiniMaxM3VLVisionConfig,
)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLAttention,
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLDenseMLP,
    MiniMaxM3VLForCausalLM,
    MiniMaxM3VLIndexer,
    MiniMaxM3VLModel,
    MiniMaxM3VLRMSNorm,
    MiniMaxM3VLSparseMoeBlock,
    MiniMaxM3VLTextModel,
    MiniMaxM3VLTopKRouter,
)

from QEfficient.transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    QEffMiniMaxM3SparseForConditionalGeneration,
    QEffMiniMaxM3VLAttention,
    QEffMiniMaxM3VLDecoderLayer,
    QEffMiniMaxM3VLDenseMLP,
    QEffMiniMaxM3VLForCausalLM,
    QEffMiniMaxM3VLIndexer,
    QEffMiniMaxM3VLRotaryEmbedding,
    QEffMiniMaxM3VLSparseMoeBlock,
    QEffMiniMaxM3VLTextModel,
    QEffMiniMaxM3VLTopKRouter,
)

__all__ = [
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3VLAttention",
    "MiniMaxM3VLConfig",
    "MiniMaxM3VLDecoderLayer",
    "MiniMaxM3VLDenseMLP",
    "MiniMaxM3VLForCausalLM",
    "MiniMaxM3VLIndexer",
    "MiniMaxM3VLModel",
    "MiniMaxM3VLRMSNorm",
    "MiniMaxM3VLSparseMoeBlock",
    "MiniMaxM3VLTextConfig",
    "MiniMaxM3VLTextModel",
    "MiniMaxM3VLTopKRouter",
    "MiniMaxM3VLVisionConfig",
    "QEffMiniMaxM3SparseForConditionalGeneration",
    "QEffMiniMaxM3VLAttention",
    "QEffMiniMaxM3VLDecoderLayer",
    "QEffMiniMaxM3VLDenseMLP",
    "QEffMiniMaxM3VLForCausalLM",
    "QEffMiniMaxM3VLIndexer",
    "QEffMiniMaxM3VLRotaryEmbedding",
    "QEffMiniMaxM3VLSparseMoeBlock",
    "QEffMiniMaxM3VLTextModel",
    "QEffMiniMaxM3VLTopKRouter",
]
