# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.customop.ctx_scatter_gather import (
    CtxChunkScatterBatchFunc,
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFunc3DGeneralized,
    CtxGatherFuncBlockedKV,
    CtxGatherFuncBlockedKVBatch,
    CtxGatherFuncPagedAttention,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
    CtxScatterFuncPagedAttention,
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
    ctx_gather,
    ctx_gather_3d,
    ctx_gather_3d_generalized,
    ctx_gather_blocked_kv,
    ctx_gather_blocked_kv_cb,
    ctx_gather_cb,
    ctx_gather_cb_3d,
    ctx_scatter,
    ctx_scatter_3d,
    ctx_scatter_3d_generalized,
    ctx_scatter_3d_int,
    ctx_scatter_cb,
    ctx_scatter_cb_3d,
)

__all__ = [
    "CtxChunkScatterBatchFunc",
    "CtxGatherFuncBlockedKVBatch",
    "CustomRMSNormAIC",
    "GemmaCustomRMSNormAIC",
    # Func classes (for ONNX export symbolic registration and direct use)
    "CtxScatterFunc",
    "CtxScatterFuncPagedAttention",
    "CtxScatterFunc3D",
    "CtxScatterFunc3DGeneralized",
    "CtxScatterFunc3DInt",
    "CtxGatherFunc",
    "CtxGatherFunc3D",
    "CtxGatherFunc3DGeneralized",
    "CtxGatherFuncBlockedKV",
    "CtxGatherFuncPagedAttention",
    "CtxScatterFuncCB",
    "CtxScatterFuncCB3D",
    "CtxGatherFuncCB",
    "CtxGatherFuncBlockedKVCB",
    "CtxGatherFuncCB3D",
    # Interface functions (dynamo-aware, prefer these at call sites)
    "ctx_scatter",
    "ctx_scatter_3d",
    "ctx_scatter_3d_generalized",
    "ctx_scatter_3d_int",
    "ctx_gather",
    "ctx_gather_3d",
    "ctx_gather_3d_generalized",
    "ctx_gather_blocked_kv",
    "ctx_scatter_cb",
    "ctx_scatter_cb_3d",
    "ctx_gather_cb",
    "ctx_gather_blocked_kv_cb",
    "ctx_gather_cb_3d",
]
