# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import torch

from QEfficient.blocking.glm_attention import _glm_tiled_sparse_mla_attention_prefill_benchmark_dataflow


def test_benchmark_dataflow_all_invalid_topk_contributes_zero():
    module = SimpleNamespace(
        scaling=1.0,
        per_head_v_up=torch.ones((1, 1, 1, 1), dtype=torch.float32),
    )
    layer_config = SimpleNamespace(
        attn_dp=1,
        attn_cp=1,
        num_cores_per_device=1,
        sparse_q_block_size=1,
        sparse_kv_num_blocks=1,
    )
    query = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    ckv_cache = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    k_pe_cache = torch.zeros_like(ckv_cache)
    topk_indices = torch.zeros((1, 1, 1), dtype=torch.int32)
    valid_topk = torch.zeros((1, 1, 1), dtype=torch.bool)

    output, _ = _glm_tiled_sparse_mla_attention_prefill_benchmark_dataflow(
        module=module,
        query_latent=query,
        query_rope=query,
        ckv_cache=ckv_cache,
        k_pe_cache=k_pe_cache,
        topk_indices=topk_indices,
        valid_topk=valid_topk,
        layer_config=layer_config,
    )

    torch.testing.assert_close(output, torch.zeros_like(output))
