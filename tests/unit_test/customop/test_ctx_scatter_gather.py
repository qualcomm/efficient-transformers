# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch

from QEfficient.customop.ctx_scatter_gather import CtxScatter3D, CtxScatterFunc3DGeneralized


def test_ctx_scatter3d_casts_updates_to_cache_dtype():
    function_proto = CtxScatter3D.to_function_proto()

    cast_like_nodes = [node for node in function_proto.node if node.op_type == "CastLike"]
    scatter_nodes = [node for node in function_proto.node if node.op_type == "ScatterND"]

    assert len(cast_like_nodes) == 1
    assert len(scatter_nodes) == 1
    assert list(cast_like_nodes[0].input) == ["updates", "data"]
    assert scatter_nodes[0].input[2] == cast_like_nodes[0].output[0]


def test_generalized_ctx_scatter3d_casts_updates_to_cache_dtype():
    data = torch.zeros((1, 3, 2), dtype=torch.bfloat16)
    position_ids = torch.tensor([[1, torch.iinfo(torch.int32).max]], dtype=torch.int32)
    updates = torch.tensor([[[1.25, -2.5], [7.0, 8.0]]], dtype=torch.float32)

    result = CtxScatterFunc3DGeneralized.apply(data, position_ids, updates)

    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result[0, 1], updates[0, 0].to(torch.bfloat16))
    torch.testing.assert_close(result[0, 0], torch.zeros(2, dtype=torch.bfloat16))
    torch.testing.assert_close(result[0, 2], torch.zeros(2, dtype=torch.bfloat16))
