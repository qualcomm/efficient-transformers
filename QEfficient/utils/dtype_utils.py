# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable, Optional, Set, Union

import torch


def resolve_torch_dtype(torch_dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    """
    Normalize a torch dtype input to a concrete torch.dtype.

    If torch_dtype is None or unrecognized, defaults to torch.float32.
    """
    if torch_dtype is None:
        return torch.float32
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        resolved = getattr(torch, torch_dtype, None)
        return resolved if isinstance(resolved, torch.dtype) else torch.float32
    return torch.float32


def _default_preserved_dtypes() -> Set[torch.dtype]:
    preserved = set()
    for name in ("float8_e4m3fn", "float8_e5m2"):
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            preserved.add(dtype)
    return preserved


def cast_non_quantized_tensors(
    model: torch.nn.Module,
    target_dtype: torch.dtype,
    *,
    preserve_dtypes: Optional[Iterable[torch.dtype]] = None,
) -> bool:
    """
    Cast floating-point parameters/buffers to target_dtype, preserving specific dtypes.

    Returns True if any tensor was cast. Preserved dtypes (e.g., FP8) are left untouched.
    """
    preserved = set(preserve_dtypes) if preserve_dtypes is not None else _default_preserved_dtypes()

    current_dtypes = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if not tensor.is_floating_point():
            continue
        if tensor.dtype in preserved:
            continue
        current_dtypes.add(tensor.dtype)

    if not current_dtypes or current_dtypes == {target_dtype}:
        return False

    changed = False
    for name, param in model.named_parameters(recurse=True):
        if not param.is_floating_point() or param.dtype in preserved:
            continue
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
            changed = True

    for name, buf in model.named_buffers(recurse=True):
        if not buf.is_floating_point() or buf.dtype in preserved:
            continue
        if buf.dtype != target_dtype:
            buf.data = buf.data.to(target_dtype)
            changed = True

    return changed
