# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.customop.ctx_scatter_gather import (
    CompressedAttnFunc,
    CtxChunkScatterBatchFunc,
    CtxGather1DFunc,
    CtxGatherDPCPFunc,
    CtxGatherDPFunc,
    CtxGatherFoldedRowsFunc,
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFunc3DGeneralized,
    CtxGatherFuncBlockedKV,
    CtxGatherFuncBlockedKVBatch,
    CtxGatherFuncPagedAttention,
    CtxScatterDPCPFunc,
    CtxScatterDPFunc,
    CtxScatterFoldedRowsFunc,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
    CtxScatterFuncPagedAttention,
    V4CtxScatter1DFunc,
)
from QEfficient.customop.ctx_scatter_gather_cb import (
    CtxGatherFuncBlockedKVCB,
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)

# Import dynamo_ops to register torch.ops.qefficient.* custom ops at package
# load time.  These ops must be registered before any model forward pass that
# uses select_interface, which evaluates torch.ops.qefficient.<op> eagerly.
from QEfficient.customop.dynamo_ops import DYNAMO_CUSTOM_OP_TABLE  # noqa: F401
from QEfficient.customop.rms_norm import CustomRMSNormAIC, GemmaCustomRMSNormAIC
from QEfficient.customop.utils import (
    ctx_chunk_scatter_batch,
    ctx_gather,
    ctx_gather_3d,
    ctx_gather_3d_generalized,
    ctx_gather_blocked_kv,
    ctx_gather_blocked_kv_batch,
    ctx_gather_blocked_kv_cb,
    ctx_gather_cb,
    ctx_gather_cb_3d,
    ctx_gather_dp,
    ctx_gather_dp_cp,
    ctx_gather_folded_rows,
    ctx_scatter,
    ctx_scatter_3d,
    ctx_scatter_3d_generalized,
    ctx_scatter_3d_int,
    ctx_scatter_cb,
    ctx_scatter_cb_3d,
    ctx_scatter_dp,
    ctx_scatter_dp_cp,
    ctx_scatter_folded_rows,
)

__all__ = [
    "CtxChunkScatterBatchFunc",
    "CtxGatherFuncBlockedKVBatch",
    "CustomRMSNormAIC",
    "GemmaCustomRMSNormAIC",
    # Func classes (for ONNX export symbolic registration and direct use)
    "CtxScatterFunc",
    "V4CtxScatter1DFunc",
    "CtxScatterFuncPagedAttention",
    "CtxScatterFunc3D",
    "CtxScatterFunc3DGeneralized",
    "CtxScatterFunc3DInt",
    "CtxGatherFunc",
    "CtxGather1DFunc",
    "CompressedAttnFunc",
    "CtxGatherFunc3D",
    "CtxGatherFunc3DGeneralized",
    "CtxGatherFuncBlockedKV",
    "CtxGatherFuncPagedAttention",
    "CtxGatherDPFunc",
    "CtxGatherFoldedRowsFunc",
    "CtxGatherDPCPFunc",
    "CtxScatterDPFunc",
    "CtxScatterFoldedRowsFunc",
    "CtxScatterDPCPFunc",
    "CtxScatterFuncCB",
    "CtxScatterFuncCB3D",
    "CtxGatherFuncCB",
    "CtxGatherFuncBlockedKVCB",
    "CtxGatherFuncCB3D",
    # Interface functions (dynamo-aware, prefer these at call sites)
    "ctx_scatter",
    "ctx_scatter_3d",
    "ctx_scatter_dp",
    "ctx_scatter_folded_rows",
    "ctx_scatter_dp_cp",
    "ctx_scatter_3d_generalized",
    "ctx_scatter_3d_int",
    "ctx_gather",
    "ctx_gather_3d",
    "ctx_gather_dp",
    "ctx_gather_folded_rows",
    "ctx_gather_dp_cp",
    "ctx_gather_3d_generalized",
    "ctx_chunk_scatter_batch",
    "ctx_gather_blocked_kv_batch",
    "ctx_gather_blocked_kv",
    "ctx_scatter_cb",
    "ctx_scatter_cb_3d",
    "ctx_gather_cb",
    "ctx_gather_blocked_kv_cb",
    "ctx_gather_cb_3d",
]
