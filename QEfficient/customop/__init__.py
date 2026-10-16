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
    CtxGatherFuncBlockedKVDP,
    CtxGatherFuncBlockRangeKVDP,
    CtxGatherFuncPagedAttention,
    CtxGatherFuncPagedKVDP,
    CtxGatherFuncPagedKVHeads,
    CtxPagedScatterFuncDP,
    CtxPagedScatterFuncPage,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
    CtxScatterFuncPagedAttention,
    M3CtxScatterFunc,
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
from QEfficient.customop.sequence_chunk import CompileLengthSequenceChunkFunc
from QEfficient.customop.utils import (
    compile_length_sequence_chunk,
    ctx_chunk_scatter_batch,
    ctx_gather,
    ctx_gather_3d,
    ctx_gather_3d_generalized,
    ctx_gather_block_range_kv_dp,
    ctx_gather_blocked_kv,
    ctx_gather_blocked_kv_batch,
    ctx_gather_blocked_kv_cb,
    ctx_gather_blocked_kv_dp,
    ctx_gather_cb,
    ctx_gather_cb_3d,
    ctx_gather_paged_attention,
    ctx_gather_paged_kv_dp,
    ctx_gather_paged_kv_heads,
    ctx_paged_scatter_dp,
    ctx_paged_scatter_page,
    ctx_scatter,
    ctx_scatter_3d,
    ctx_scatter_3d_generalized,
    ctx_scatter_3d_int,
    ctx_scatter_cb,
    ctx_scatter_cb_3d,
    ctx_scatter_paged_attention,
    m3_ctx_scatter,
)

__all__ = [
    "CtxChunkScatterBatchFunc",
    "CtxGatherFuncBlockedKVBatch",
    "CustomRMSNormAIC",
    "GemmaCustomRMSNormAIC",
    "CompileLengthSequenceChunkFunc",
    # Func classes (for ONNX export symbolic registration and direct use)
    "CtxScatterFunc",
    "CtxScatterFuncPagedAttention",
    "CtxScatterFunc3D",
    "CtxScatterFunc3DGeneralized",
    "CtxScatterFunc3DInt",
    "M3CtxScatterFunc",
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
    # DP-layout ops (com.qti.aisw.onnx namespace)
    "CtxPagedScatterFuncDP",
    "CtxPagedScatterFuncPage",
    "CtxGatherFuncBlockedKVDP",
    "CtxGatherFuncPagedKVDP",
    "CtxGatherFuncPagedKVHeads",
    "CtxGatherFuncBlockRangeKVDP",
    # Interface functions (dynamo-aware, prefer these at call sites)
    "compile_length_sequence_chunk",
    "ctx_scatter",
    "ctx_scatter_paged_attention",
    "ctx_scatter_3d",
    "ctx_scatter_3d_generalized",
    "ctx_scatter_3d_int",
    "ctx_chunk_scatter_batch",
    "m3_ctx_scatter",
    "ctx_gather",
    "ctx_gather_3d",
    "ctx_gather_3d_generalized",
    "ctx_gather_block_range_kv_dp",
    "ctx_gather_blocked_kv",
    "ctx_gather_blocked_kv_batch",
    "ctx_gather_blocked_kv_dp",
    "ctx_gather_paged_attention",
    "ctx_gather_paged_kv_dp",
    "ctx_gather_paged_kv_heads",
    "ctx_paged_scatter_dp",
    "ctx_paged_scatter_page",
    "ctx_scatter_cb",
    "ctx_scatter_cb_3d",
    "ctx_gather_cb",
    "ctx_gather_blocked_kv_cb",
    "ctx_gather_cb_3d",
]
