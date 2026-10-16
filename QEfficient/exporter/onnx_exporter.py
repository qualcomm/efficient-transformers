# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Standalone ONNX export backend functions.

These functions contain the actual export logic and are called directly from
QEFFBaseModel._export(). Keeping them outside the base class makes the logic
testable in isolation and avoids polluting QEFFBaseModel with backend details.

_export_layerwise is intentionally kept in QEFFBaseModel because it contains
VLM-specific branching (is_vision checks) that requires a model-level hook to
refactor cleanly. It will be moved here once _build_weight_free_model() and
similar override hooks are in place.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch

from QEfficient.base.onnx_transforms import PruneFakeInitializersTransform
from QEfficient.utils import constants
from QEfficient.utils.torch_patches import (
    dynamo_invoke_subgraph_fallback_env,
    layerwise_safe_onnx_export_patches,
    preserve_subfunction_source_lines,
)


@dataclass
class ExportResult:
    onnx_path: Path
    onnx_transform_kwargs: Dict[str, Any] = field(default_factory=dict)
    excluded_onnx_transforms: Tuple[Type[Any], ...] = ()
    weight_spec_path: Optional[Path] = None


def export_via_legacy(
    qeff_model,
    onnx_path: Path,
    example_inputs: dict[str, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    export_kwargs: dict,
) -> ExportResult:
    """Export via TorchScript symbolic tracing (dynamo=False)."""
    from QEfficient.base.modeling_qeff import QEFFBaseModel

    with layerwise_safe_onnx_export_patches(enabled=bool(QEFFBaseModel._layerwise_active)):
        torch.onnx.export(
            qeff_model.model,
            (),
            str(onnx_path),
            kwargs=example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
            opset_version=constants.ONNX_LEGACY_EXPORT_OPSET,
            verbose=True,
            **export_kwargs,
        )
    return ExportResult(onnx_path=onnx_path)


def export_via_dynamo(
    qeff_model,
    onnx_path: Path,
    example_inputs: dict[str, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    dynamic_shapes: dict | None,
    export_kwargs: dict,
) -> ExportResult:
    """Export via torch.export (dynamo=True) with custom op translation."""
    from QEfficient.utils.export_utils import build_dynamo_export_kwargs, reorder_inputs_by_signature

    example_inputs, dynamic_shapes = reorder_inputs_by_signature(qeff_model.model, example_inputs, dynamic_shapes)
    export_kwargs = build_dynamo_export_kwargs(export_kwargs)

    with dynamo_invoke_subgraph_fallback_env(), preserve_subfunction_source_lines():
        onnx_program = torch.onnx.export(
            qeff_model.model,
            args=(),
            f=None,
            kwargs=example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            dynamic_shapes=dynamic_shapes,
            **export_kwargs,
        )
        if onnx_program is None:
            raise RuntimeError("torch.onnx.export returned None for dynamo export")
        PruneFakeInitializersTransform.apply(onnx_program)
        onnx_program.save(str(onnx_path))
    return ExportResult(onnx_path=onnx_path)


def export_via_weightfree(
    qeff_model,
    onnx_path: Path,
    example_inputs: dict[str, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    dynamic_shapes: dict | None,
    export_kwargs: dict,
    onnx_transform_kwargs: dict | None = None,
) -> ExportResult:
    """Export via weight-free dynamo path: meta-device model, no embedded weights.

    dynamic_shapes must be pre-computed by the caller (export_wrapper already does
    this for dynamo=True).  Accepting it directly keeps the signature parallel with
    export_via_dynamo and avoids a redundant convert_dynamic_axes_to_dynamic_shapes
    call.
    """
    from QEfficient.base.onnx_transforms import SplitTensorsTransform
    from QEfficient.exporter.weight_free.export import export_weight_free_onnx
    from QEfficient.utils.export_utils import build_dynamo_export_kwargs, reorder_inputs_by_signature

    example_inputs, dynamic_shapes = reorder_inputs_by_signature(qeff_model.model, example_inputs, dynamic_shapes)
    wf_export_kwargs = build_dynamo_export_kwargs(export_kwargs)

    _, updated_onnx_transform_kwargs = export_weight_free_onnx(
        qeff_model=qeff_model,
        onnx_path=onnx_path,
        example_inputs=example_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        export_kwargs=wf_export_kwargs,
        onnx_transform_kwargs=onnx_transform_kwargs or {},
    )
    weight_spec_path = onnx_path.with_name("weight_spec.json")
    return ExportResult(
        onnx_path=onnx_path,
        onnx_transform_kwargs=updated_onnx_transform_kwargs or {},
        excluded_onnx_transforms=(SplitTensorsTransform,),
        weight_spec_path=weight_spec_path,
    )
