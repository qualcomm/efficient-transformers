# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch


def select_interface(eager_impl, custom_op_impl):
    use_custom_op = torch._dynamo.is_compiling()
    return custom_op_impl if use_custom_op else eager_impl


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
