# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest
import torch

from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeVisionAttention

# Divides cleanly into both num_query_blocks (44) and num_key_value_blocks (4)
# on the class, so no block in either loop degenerates to an empty slice.
SEQ_LENGTH = 176
DIM = 16
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS


def _build_attention() -> QEffQwen3_5MoeVisionAttention:
    # QEffQwen3_5MoeVisionAttention.__init__ calls super().__init__() with no
    # args, but the parent HF class requires a config object -- it's never
    # actually constructed via __init__ in production (the class gets
    # attached to an already-built module instead). Bypass __init__ here,
    # same as the existing GDN head-blocking test does for its layer class.
    torch.manual_seed(0)
    attn = object.__new__(QEffQwen3_5MoeVisionAttention)
    torch.nn.Module.__init__(attn)
    attn.num_heads = NUM_HEADS
    attn.head_dim = HEAD_DIM
    attn.qkv = torch.nn.Linear(DIM, DIM * 3, bias=True)
    attn.proj = torch.nn.Linear(DIM, DIM)
    attn.eval()
    return attn


def _run_forward(attn, hidden_states, cu_seqlens, cos, sin, head_block_size):
    attn.head_block_size = head_block_size
    with torch.no_grad():
        return attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
        )


def test_vision_attention_default_head_block_size_is_four():
    attn = _build_attention()
    assert attn.head_block_size == 4


@pytest.mark.parametrize("head_block_size", [None, 1, 2, 4])
def test_vision_attention_head_blocking_matches_unblocked(head_block_size):
    """Head blocking only changes how many heads are processed together per
    step; it must not change the numerical result. Compare every blocked
    configuration against the unblocked (head_block_size=None) reference on
    a tiny randomly-initialized attention module."""
    torch.manual_seed(42)
    attn = _build_attention()

    hidden_states = torch.randn(SEQ_LENGTH, DIM)
    cu_seqlens = torch.tensor([0, SEQ_LENGTH], dtype=torch.int32)
    cos = torch.randn(SEQ_LENGTH, HEAD_DIM)
    sin = torch.randn(SEQ_LENGTH, HEAD_DIM)

    reference_output = _run_forward(attn, hidden_states, cu_seqlens, cos, sin, head_block_size=None)
    blocked_output = _run_forward(attn, hidden_states, cu_seqlens, cos, sin, head_block_size=head_block_size)

    torch.testing.assert_close(blocked_output, reference_output)


def test_vision_attention_head_blocking_output_shape_and_finite():
    torch.manual_seed(7)
    attn = _build_attention()

    hidden_states = torch.randn(SEQ_LENGTH, DIM)
    cu_seqlens = torch.tensor([0, SEQ_LENGTH], dtype=torch.int32)
    cos = torch.randn(SEQ_LENGTH, HEAD_DIM)
    sin = torch.randn(SEQ_LENGTH, HEAD_DIM)

    output = _run_forward(attn, hidden_states, cu_seqlens, cos, sin, head_block_size=2)

    assert output.shape == (SEQ_LENGTH, DIM)
    assert torch.isfinite(output).all()
