# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch


def select_interface(eager_impl, custom_op_impl):
    use_custom_op = torch.compiler.is_exporting() or torch._dynamo.is_compiling()
    return custom_op_impl if use_custom_op else eager_impl


def _dynamo_symbolic(domain_op: str, inputs: tuple[torch.Tensor, ...], dtype: torch.dtype, shape):
    """Emit a first-class ONNX custom node during Dynamo export.

    The ``torch.library`` path is retained for ``torch.compile``.  ONNX's
    symbolic API is more reliable for the QSA cache operators because it
    prevents Dynamo from decomposing their eager indexing implementation.
    """
    output = torch.onnx.ops.symbolic(domain_op, inputs, {}, dtype=dtype, shape=shape, version=1)
    # Weight-free Dynamo export traces model tensors on ``meta``.  ONNX
    # symbolic outputs default to CPU unless their device is made explicit,
    # which later causes mixed-device failures when the result is consumed by
    # ordinary ATen operators.
    return output.to(inputs[0].device)


# ---------------------------------------------------------------------------
# Interface functions for ctx_scatter_gather ops
# ---------------------------------------------------------------------------
# Each function uses select_interface to pick between FuncClass.apply (eager /
# ONNX export) and torch.ops.qefficient.<op> (dynamo compilation).
# Lazy imports inside each function break the circular dependency that would
# arise if ctx_scatter_gather.py (which imports select_interface from here)
# were imported at module load time.
# ---------------------------------------------------------------------------


def ctx_scatter(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxScatterFunc

    return select_interface(CtxScatterFunc.apply, torch.ops.qefficient.ctx_scatter)(data, position_ids, updates)


def ctx_scatter_3d(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxScatterFunc3D

    return select_interface(CtxScatterFunc3D.apply, torch.ops.qefficient.ctx_scatter_3d)(data, position_ids, updates)


def ctx_scatter_dp(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxScatterDPFunc

    return select_interface(CtxScatterDPFunc.apply, torch.ops.qefficient.ctx_scatter_dp)(data, position_ids, updates)


def ctx_scatter_folded_rows(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxScatterFoldedRowsFunc

    return select_interface(CtxScatterFoldedRowsFunc.apply, torch.ops.qefficient.ctx_scatter_folded_rows)(
        data, position_ids, updates
    )


def ctx_scatter_dp_cp(
    data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor, context_parallel: int
) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxScatterDPCPFunc

    return select_interface(CtxScatterDPCPFunc.apply, torch.ops.qefficient.ctx_scatter_dp_cp)(
        data, position_ids, updates, context_parallel
    )


def ctx_scatter_3d_generalized(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxScatterFunc3DGeneralized

    return select_interface(CtxScatterFunc3DGeneralized.apply, torch.ops.qefficient.ctx_scatter_3d_generalized)(
        data, position_ids, updates
    )


def ctx_scatter_3d_int(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxScatterFunc3DInt

    return select_interface(CtxScatterFunc3DInt.apply, torch.ops.qefficient.ctx_scatter_3d_int)(
        data, position_ids, updates
    )


def ctx_gather_3d(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxGatherFunc3D

    return select_interface(CtxGatherFunc3D.apply, torch.ops.qefficient.ctx_gather_3d)(data, ctx_indices)


def ctx_gather_dp(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxGatherDPFunc

    return select_interface(CtxGatherDPFunc.apply, torch.ops.qefficient.ctx_gather_dp)(data, ctx_indices)


def ctx_gather_folded_rows(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxGatherFoldedRowsFunc

    return select_interface(CtxGatherFoldedRowsFunc.apply, torch.ops.qefficient.ctx_gather_folded_rows)(
        data, ctx_indices
    )


def ctx_gather_dp_cp(data: torch.Tensor, ctx_indices: torch.Tensor, context_parallel: int) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxGatherDPCPFunc

    return select_interface(CtxGatherDPCPFunc.apply, torch.ops.qefficient.ctx_gather_dp_cp)(
        data, ctx_indices, context_parallel
    )


def ctx_gather_3d_generalized(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxGatherFunc3DGeneralized

    return select_interface(CtxGatherFunc3DGeneralized.apply, torch.ops.qefficient.ctx_gather_3d_generalized)(
        data, ctx_indices
    )


def ctx_gather(data: torch.Tensor, ctx_indices: torch.Tensor, comp_ctx_len: int) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxGatherFunc

    return select_interface(CtxGatherFunc.apply, torch.ops.qefficient.ctx_gather)(data, ctx_indices, comp_ctx_len)


def ctx_gather_blocked_kv(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxGatherFuncBlockedKV

    return select_interface(CtxGatherFuncBlockedKV.apply, torch.ops.qefficient.ctx_gather_blocked_kv)(data, ctx_indices)


def ctx_chunk_scatter_batch(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxChunkScatterBatchFunc

    return select_interface(CtxChunkScatterBatchFunc.apply, torch.ops.qefficient.ctx_chunk_scatter_batch)(
        data, position_ids, updates
    )


def ctx_gather_blocked_kv_batch(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather import CtxGatherFuncBlockedKVBatch

    return select_interface(CtxGatherFuncBlockedKVBatch.apply, torch.ops.qefficient.ctx_gather_blocked_kv_batch)(
        data, ctx_indices
    )


def ctx_paged_scatter(data, block_id, addr, updates):
    from QEfficient.customop.ctx_scatter_gather import CtxPagedScatterFunc

    if torch.compiler.is_exporting():
        return _dynamo_symbolic(
            "com.qualcomm.cloud::CtxPagedScatter",
            (data, block_id, addr, updates),
            data.dtype,
            data.shape,
        )
    return select_interface(CtxPagedScatterFunc.apply, torch.ops.qefficient.ctx_paged_scatter)(
        data, block_id, addr, updates
    )


def ctx_gather_rows(data, ctx_indices):
    from QEfficient.customop.ctx_scatter_gather import CtxGatherRowsFunc

    if torch.compiler.is_exporting():
        return _dynamo_symbolic(
            "com.qualcomm.cloud::CtxGatherRows",
            (data, ctx_indices),
            data.dtype,
            (*ctx_indices.shape, data.shape[-1]),
        )
    return select_interface(CtxGatherRowsFunc.apply, torch.ops.qefficient.ctx_gather_rows)(data, ctx_indices)


# ---------------------------------------------------------------------------
# Interface functions for ctx_scatter_gather_cb ops
# ---------------------------------------------------------------------------


def ctx_scatter_cb(
    data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor
) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather_cb import CtxScatterFuncCB

    return select_interface(CtxScatterFuncCB.apply, torch.ops.qefficient.ctx_scatter_cb)(
        data, batch_index, position_ids, updates
    )


def ctx_scatter_cb_3d(
    data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor
) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather_cb import CtxScatterFuncCB3D

    return select_interface(CtxScatterFuncCB3D.apply, torch.ops.qefficient.ctx_scatter_cb_3d)(
        data, batch_index, position_ids, updates
    )


def ctx_gather_cb(
    data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor, comp_ctx_len: int
) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather_cb import CtxGatherFuncCB

    return select_interface(CtxGatherFuncCB.apply, torch.ops.qefficient.ctx_gather_cb)(
        data, batch_index, ctx_indices, comp_ctx_len
    )


def ctx_gather_blocked_kv_cb(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather_cb import CtxGatherFuncBlockedKVCB

    return select_interface(CtxGatherFuncBlockedKVCB.apply, torch.ops.qefficient.ctx_gather_blocked_kv_cb)(
        data, batch_index, ctx_indices
    )


def ctx_gather_cb_3d(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    from QEfficient.customop.ctx_scatter_gather_cb import CtxGatherFuncCB3D

    return select_interface(CtxGatherFuncCB3D.apply, torch.ops.qefficient.ctx_gather_cb_3d)(
        data, batch_index, ctx_indices
    )


def custom_rms_norm_func(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    from QEfficient.customop.rms_norm import CustomRMSNormFunc

    return select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)(hidden_states, weight, epsilon)
