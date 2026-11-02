# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import io

import onnx
import torch

from QEfficient.base.onnx_transforms import CustomOpTransform
from QEfficient.customop.dynamo_ops import DYNAMO_CUSTOM_OP_TABLE
from QEfficient.customop.glm_dsa import V4CtxGatherRows, v4_ctx_gather_rows
from QEfficient.customop.onnxscript_utils import get_dynamo_onnxscript_func


def test_v4_ctx_gather_rows_eager_matches_direct_cache_indexing():
    data = torch.arange(14, dtype=torch.float32).reshape(1, 1, 7, 2)
    indices = torch.tensor([[[[0, 4], [6, 2]]]], dtype=torch.int32)

    actual = v4_ctx_gather_rows(data, indices)
    expected = data[0, 0, indices[0, 0]].reshape(1, 1, 2, 2, 2)

    torch.testing.assert_close(actual, expected)


def test_v4_ctx_gather_rows_dynamo_export_preserves_vendor_op():
    class GatherWrapper(torch.nn.Module):
        def forward(self, data, ctx_indices):
            return v4_ctx_gather_rows(data, ctx_indices)

    data = torch.randn(1, 1, 8, 3)
    indices = torch.tensor([[[[0, 2], [5, 7]]]], dtype=torch.int32)
    output = io.BytesIO()
    torch.onnx.export(
        GatherWrapper().eval(),
        (data, indices),
        output,
        dynamo=True,
        custom_translation_table=DYNAMO_CUSTOM_OP_TABLE,
    )
    model = onnx.load_model_from_string(output.getvalue())
    CustomOpTransform.apply(model)

    gather_nodes = [node for node in model.graph.node if node.op_type == "V4CtxGatherRows"]
    assert len(gather_nodes) == 1
    assert gather_nodes[0].domain == "com.qti.aisw.onnx"
    assert any(function.name == "V4CtxGatherRows" for function in model.functions)

    function_proto = get_dynamo_onnxscript_func(V4CtxGatherRows).to_function_proto()
    assert function_proto.name == "V4CtxGatherRows"


def test_v4_ctx_gather_rows_legacy_export_has_gathered_output_shape():
    class GatherWrapper(torch.nn.Module):
        def forward(self, data, ctx_indices):
            return v4_ctx_gather_rows(data, ctx_indices)

    data = torch.randn(1, 1, 8, 3)
    indices = torch.tensor([[[[0, 2], [5, 7]]]], dtype=torch.int32)
    output = io.BytesIO()
    torch.onnx.export(
        GatherWrapper().eval(),
        (data, indices),
        output,
        dynamo=False,
        opset_version=17,
    )
    model = onnx.load_model_from_string(output.getvalue())

    assert len(model.graph.output) == 1
    output_shape = [dimension.dim_value for dimension in model.graph.output[0].type.tensor_type.shape.dim]
    assert output_shape == [1, 1, 2, 2, 3]
