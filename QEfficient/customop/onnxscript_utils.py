# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
from collections.abc import Callable

import onnxscript

from QEfficient.utils import constants

_DYNAMO_FUNC_ATTR = "_qeff_dynamo_onnxscript_func"
_MISSING = object()


def _onnx_opset(opset_version: int) -> onnxscript.values.Opset:
    return getattr(onnxscript, f"opset{opset_version}")


def _compile_with_default_opset(
    fn: Callable,
    custom_opset: onnxscript.values.Opset,
    default_opset_version: int,
):
    """Compile an ONNXScript function with the requested default-domain opset.

    Existing custom-op definitions refer to a module-level ``ops`` symbol. The
    ONNXScript compiler captures module globals while the decorator runs, so we
    temporarily point that symbol at the requested default opset and immediately
    restore it after compilation.
    """
    module = inspect.getmodule(fn)
    if module is None:
        raise RuntimeError(f"Unable to resolve module for ONNXScript function {fn.__name__}")

    previous_ops = module.__dict__.get("ops", _MISSING)
    module.__dict__["ops"] = _onnx_opset(default_opset_version)
    try:
        return onnxscript.script(custom_opset)(fn)
    finally:
        if previous_ops is _MISSING:
            module.__dict__.pop("ops", None)
        else:
            module.__dict__["ops"] = previous_ops


def qeff_custom_op(domain: str, version: int):
    # TODO: remove this decorator and the legacy compilation path once legacy (torch.onnx.export) is deprecated.
    """Compile one custom op body into legacy and dynamo ONNXScript variants."""
    custom_opset = onnxscript.values.Opset(domain, version)

    def decorator(fn: Callable):
        legacy_func = _compile_with_default_opset(fn, custom_opset, constants.ONNX_LEGACY_EXPORT_OPSET)
        dynamo_func = _compile_with_default_opset(fn, custom_opset, constants.ONNX_DYNAMO_EXPORT_OPSET)
        setattr(legacy_func, _DYNAMO_FUNC_ATTR, dynamo_func)
        return legacy_func

    return decorator


def get_dynamo_onnxscript_func(onnxscript_func):
    """Return the dynamo variant attached by ``qeff_custom_op`` when present."""
    return getattr(onnxscript_func, _DYNAMO_FUNC_ATTR, onnxscript_func)


def get_onnxscript_func(onnxscript_func, onnx_export_opset: int):
    """Return the ONNXScript variant matching the requested export opset."""
    if onnx_export_opset == constants.ONNX_DYNAMO_EXPORT_OPSET:
        return get_dynamo_onnxscript_func(onnxscript_func)
    return onnxscript_func
