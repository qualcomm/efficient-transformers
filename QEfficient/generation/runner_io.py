# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Utilities for writing replayable qaic-runner IO bundles.

This module prepares one host-side inference invocation from exported ONNX
metadata and compile artifacts. It serializes input tensors as raw files and
writes an ``aic_batch_io.json`` descriptor that external harnesses can replay
with ``qaic-runner``. The public helpers cover CausalLM, single-QPC VLM, and
dual-QPC VLM artifacts mode generation flows.
"""

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.generation_helpers import (
    RunnerMetadata,
    _add_cross_qpc_placeholders,
    _add_specialization_control_inputs,
    _apply_input_shapes,
    _cross_qpc_symbols,
    _cross_qpc_output_shapes,
    _execution_batch_size,
    _filter_graph_inputs,
    _first_execution_batch,
    _get_compile_dir,
    _prepare_vlm_execution_inputs,
    _proxy_logits_width,
    _resolve_output_shape,
    build_prefill_inputs,
    cross_qpc_tensors,
    load_prefill_specialization,
    prepare_tokenizer,
    select_prefill_specialization,
    select_vision_specialization,
    slice_prefill_inputs,
    validate_runner_input_shapes,
)
from QEfficient.utils import get_padding_shape_from_config
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")

_CUSTOM_IO_DTYPES = {
    "bfloat16": np.dtype(np.uint16),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "int8": np.dtype(np.int8),
    "mxint8": np.dtype(np.int8),
}


def _onnx_input_dtype(graph_input: onnx.ValueInfoProto) -> np.dtype:
    """Return a numpy storage dtype for an ONNX graph input."""
    element_type = graph_input.type.tensor_type.elem_type
    if element_type == onnx.TensorProto.BFLOAT16:
        return np.dtype(np.uint16)
    try:
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported ONNX input dtype for {graph_input.name!r}: {element_type}") from error


def _runner_input_dtype(runner_metadata: RunnerMetadata, name: str) -> tuple[str | None, np.dtype]:
    """Resolve runner input storage dtype, preferring custom-IO precision."""
    precision = runner_metadata.custom_io_precisions.get(name)
    if precision is not None:
        precision = precision.lower()
        dtype = _CUSTOM_IO_DTYPES.get(precision)
        if dtype is None:
            raise ValueError(f"Unsupported custom-IO precision {precision!r} for input {name!r}.")
        return precision, dtype

    graph_input = next((item for item in runner_metadata.graph.input if item.name == name), None)
    if graph_input is None:
        raise ValueError(f"Input {name!r} is not present in the exported ONNX graph.")
    if graph_input.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16:
        return "bfloat16", _onnx_input_dtype(graph_input)
    if graph_input.type.tensor_type.elem_type == onnx.TensorProto.INT8:
        return "int8", _onnx_input_dtype(graph_input)
    return None, _onnx_input_dtype(graph_input)


def _as_bfloat16_storage(value: np.ndarray, name: str) -> np.ndarray:
    """Encode numeric values as two-byte BF16 payloads without FP16 rounding."""
    if value.dtype == np.uint16:
        return np.ascontiguousarray(value)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"Input {name!r} cannot be encoded as bfloat16 from dtype {value.dtype}.")
    return (
        torch.as_tensor(value).to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16).copy()
    )


def _as_int8_storage(value: np.ndarray, name: str, precision: str) -> np.ndarray:
    """Convert integer-compatible inputs to INT8/MXINT8 without silent truncation."""
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise ValueError(f"Input {name!r} contains non-finite values incompatible with {precision}.")
    if value.dtype.kind == "f" and not np.all(value == np.trunc(value)):
        raise ValueError(f"Input {name!r} contains non-integral values incompatible with {precision}.")
    if value.dtype.kind not in "fiu":
        raise ValueError(f"Input {name!r} cannot be converted to {precision} from dtype {value.dtype}.")
    if value.size and (value.min() < np.iinfo(np.int8).min or value.max() > np.iinfo(np.int8).max):
        raise ValueError(f"Input {name!r} contains values outside the INT8 range required by {precision}.")
    return np.ascontiguousarray(value.astype(np.int8, copy=False))


def _normalize_runner_input(runner_metadata: RunnerMetadata, name: str, value) -> np.ndarray:
    """Convert one processor input to graph/custom-IO storage dtype for runner replay."""
    value = np.asarray(value)
    precision, expected_dtype = _runner_input_dtype(runner_metadata, name)
    if precision == "bfloat16":
        converted = _as_bfloat16_storage(value, name)
    elif precision in {"int8", "mxint8"}:
        converted = _as_int8_storage(value, name, precision)
    else:
        if value.dtype.kind not in "fiu":
            raise ValueError(f"Input {name!r} has incompatible processor dtype {value.dtype}.")
        try:
            converted = np.ascontiguousarray(value.astype(expected_dtype, casting="same_kind", copy=False))
        except TypeError as error:
            expected_name = precision or str(expected_dtype)
            raise ValueError(
                f"Input {name!r} with dtype {value.dtype} is incompatible with expected {expected_name}."
            ) from error

    expected_size = runner_metadata.custom_io_item_sizes.get(name, expected_dtype.itemsize)
    if converted.itemsize != expected_size:
        raise ValueError(
            f"Input {name!r} has element size {converted.itemsize}, expected {expected_size} from "
            f"{'custom-IO' if precision else 'ONNX'} metadata."
        )
    return converted

__all__ = [
    "load_prefill_specialization",
    "write_causal_lm_runner_bundle",
    "write_dual_qpc_vlm_runner_bundle",
    "write_runner_io_bundle",
    "write_single_qpc_vlm_runner_bundle",
]


def write_runner_io_bundle(
    *,
    specialization: Mapping[str, int],
    host_inputs: Mapping[str, np.ndarray],
    onnx_path: Optional[Union[str, Path]] = None,
    compile_dir: Optional[Union[str, Path]] = None,
    runner_metadata: Optional[RunnerMetadata] = None,
    input_shape_overrides: Optional[Mapping[str, Sequence[int]]] = None,
    output_shape_overrides: Optional[Mapping[str, Sequence[int]]] = None,
    fallback_logits_width: Optional[int] = None,
) -> Path:
    """Write raw inputs and ``aic_batch_io.json`` for one qaic-runner invocation."""
    if runner_metadata is None:
        if onnx_path is None or compile_dir is None:
            raise TypeError("`onnx_path` and `compile_dir` are required when `runner_metadata` is not provided.")
        runner_metadata = RunnerMetadata.from_paths(onnx_path, compile_dir)

    compile_dir = runner_metadata.compile_dir
    io_dir = compile_dir / "io"
    data_dir = io_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = onnx.ModelProto()
    onnx_model.CopyFrom(runner_metadata.model)
    missing_inputs = runner_metadata.required_host_input_names - host_inputs.keys()
    if missing_inputs:
        raise ValueError(f"Missing qaic-runner host inputs: {sorted(missing_inputs)}")
    unexpected_inputs = host_inputs.keys() - runner_metadata.input_names
    if unexpected_inputs:
        raise ValueError(f"Inputs are not present in the exported ONNX graph: {sorted(unexpected_inputs)}")

    io_entries = []
    numpy_inputs = {
        name: _normalize_runner_input(runner_metadata, name, value) for name, value in host_inputs.items()
    }
    validate_runner_input_shapes(runner_metadata, numpy_inputs, specialization)
    for name, value in numpy_inputs.items():
        value.tofile(data_dir / f"{name}.raw")
        io_entries.append(
            {
                "path": f"data/{name}.raw",
                "io-direction": "in",
                "elem-size": int(value.itemsize),
                "map-to": name,
                "dims": list(value.shape),
            }
        )

    shapes = {name: value.shape for name, value in numpy_inputs.items()}
    shapes.update(input_shape_overrides or {})
    _apply_input_shapes(onnx_model, shapes)
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=False, data_prop=True)
    except Exception as error:
        logger.warning(f"ONNX shape inference failed while creating runner inputs: {error}")

    symbols = {name: int(value) for name, value in specialization.items() if str(value).lstrip("-").isdigit()}
    symbols["seq_len"] = 1
    batch_source = numpy_inputs.get("input_ids")
    fallback_batch_size = int(batch_source.shape[0]) if batch_source is not None else None
    custom_item_sizes = runner_metadata.custom_io_item_sizes
    for output in onnx_model.graph.output:
        if output.name.endswith(("_RetainedState", "_InternalRetainedState")):
            continue
        dtype = onnx.helper.tensor_dtype_to_np_dtype(output.type.tensor_type.elem_type)
        output_shape = (output_shape_overrides or {}).get(output.name)
        io_entries.append(
            {
                "path": f"data/{output.name}.raw",
                "io-direction": "out",
                "elem-size": custom_item_sizes.get(output.name, int(np.dtype(dtype).itemsize)),
                "map-to": output.name,
                "dims": list(output_shape)
                if output_shape is not None
                else _resolve_output_shape(output, symbols, fallback_batch_size, fallback_logits_width),
            }
        )

    (io_dir / "aic_batch_io.json").write_text(json.dumps({"IO-files": [io_entries]}, indent=2))
    return io_dir


def write_causal_lm_runner_bundle(
    *,
    model,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompts: List[str],
    sampling_params: Optional[Mapping[str, np.ndarray]] = None,
) -> Path:
    """Prepare and write the first CausalLM prefill invocation."""
    if not prompts:
        raise ValueError("`prompts` must contain at least one prompt.")

    compile_dir = _get_compile_dir(model)
    runner_metadata = RunnerMetadata.from_paths(model.onnx_path, compile_dir)
    specialization = load_prefill_specialization(compile_dir)
    prefill_seq_len = int(specialization["seq_len"])
    batch_prompts = _first_execution_batch(prompts, _execution_batch_size(specialization), "prompts")
    prepare_tokenizer(tokenizer)
    prefill_inputs, _, _ = build_prefill_inputs(tokenizer, batch_prompts, prefill_seq_len)
    host_inputs = slice_prefill_inputs(prefill_inputs, 0, prefill_seq_len)

    full_batch_size = specialization.get("full_batch_size")
    _add_specialization_control_inputs(runner_metadata, host_inputs, specialization, sampling_params=sampling_params)

    shape_overrides = {}
    context_length = specialization.get("ctx_len")
    if context_length is not None:
        cache_batch_size = int(full_batch_size or specialization["batch_size"])
        cache_shape = get_padding_shape_from_config(model.model.config, cache_batch_size, int(context_length))
        shape_overrides = {
            graph_input.name: cache_shape
            for graph_input in runner_metadata.graph.input
            if graph_input.name.startswith(("past_key.", "past_value.")) and graph_input.name not in host_inputs
        }

    return write_runner_io_bundle(
        runner_metadata=runner_metadata,
        specialization=specialization,
        host_inputs=host_inputs,
        input_shape_overrides=shape_overrides,
        fallback_logits_width=_proxy_logits_width(model),
    )


def _prepare_vlm_runner_inputs(
    *,
    model,
    active_model,
    processor,
    images: Sequence[str],
    prompts: Sequence[str],
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> Tuple[Mapping[str, int], RunnerMetadata, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if processor is None or not images or not prompts:
        raise ValueError("`processor`, `images`, and `prompts` are required in artifacts mode.")
    if (image_height is None) != (image_width is None):
        raise ValueError("`image_height` and `image_width` must be provided together in artifacts mode.")
    if image_height is not None and (
        isinstance(image_height, bool)
        or isinstance(image_width, bool)
        or not isinstance(image_height, int)
        or not isinstance(image_width, int)
        or image_height <= 0
        or image_width <= 0
    ):
        raise ValueError("`image_height` and `image_width` must be positive integers in artifacts mode.")

    from QEfficient.generation.embedding_handler import VisionHandler

    compile_dir = _get_compile_dir(active_model)
    runner_metadata = RunnerMetadata.from_paths(active_model.onnx_path, compile_dir)
    try:
        language_specialization = select_prefill_specialization(compile_dir)
    except ValueError:
        # A standalone vision component has no language prefill record. Its
        # paired language graph defines the processor prompt chunk contract.
        lang_model = getattr(model, "lang_model", None)
        if lang_model is None:
            raise
        language_specialization = select_prefill_specialization(_get_compile_dir(lang_model))
    prefill_seq_len = max(int(language_specialization.get("seq_len", 1)), 1)
    batch_size = _execution_batch_size(language_specialization)
    handler = VisionHandler(
        qeff_model=model,
        vision_session=None,
        processor=processor,
        tokenizer=getattr(processor, "tokenizer", None),
        image_height=image_height,
        image_width=image_width,
    )
    vision_inputs, lang_inputs = _prepare_vlm_execution_inputs(handler, images, prompts, prefill_seq_len, batch_size)
    try:
        specialization = select_vision_specialization(compile_dir, runner_metadata, vision_inputs)
    except ValueError as error:
        # Fused and language-only graphs have no vision records. Preserve their
        # language prefill selection while making standalone vision strict.
        if "No vision specialization found" not in str(error):
            raise
        specialization = language_specialization
    return specialization, runner_metadata, vision_inputs, lang_inputs


def _write_vlm_runner_bundle(
    *,
    model,
    runner_metadata: RunnerMetadata,
    specialization: Mapping[str, int],
    host_inputs: Mapping[str, np.ndarray],
    output_shape_overrides: Optional[Mapping[str, Sequence[int]]] = None,
) -> Path:
    return write_runner_io_bundle(
        runner_metadata=runner_metadata,
        specialization=specialization,
        host_inputs=host_inputs,
        output_shape_overrides=output_shape_overrides,
        fallback_logits_width=_proxy_logits_width(model),
    )


def write_single_qpc_vlm_runner_bundle(
    *,
    model,
    processor,
    images: List[str],
    prompts: List[str],
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> Path:
    """Prepare and write the first fused vision-language prefill invocation."""
    specialization, runner_metadata, vision_inputs, host_inputs = _prepare_vlm_runner_inputs(
        model=model,
        active_model=model,
        processor=processor,
        images=images,
        prompts=prompts,
        image_height=image_height,
        image_width=image_width,
    )
    host_inputs.update(vision_inputs)
    _add_specialization_control_inputs(runner_metadata, host_inputs, specialization)
    host_inputs = _filter_graph_inputs(runner_metadata, host_inputs)
    return _write_vlm_runner_bundle(
        model=model, runner_metadata=runner_metadata, specialization=specialization, host_inputs=host_inputs
    )


def write_dual_qpc_vlm_runner_bundle(
    *,
    model,
    processor,
    images: List[str],
    prompts: List[str],
    skip_vision: bool,
    skip_lang: bool,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> Path:
    """Prepare one isolated vision or language invocation for a dual-QPC VLM."""
    if skip_vision == skip_lang:
        raise ValueError(
            "Artifacts mode dual-QPC generation requires exactly one of `skip_vision=True` or `skip_lang=True`; "
            "use the same component selection passed to compile()."
        )
    active_model = model.vision_model if skip_lang else model.lang_model
    if not getattr(active_model, "compile_artifacts_path", None) and not active_model.qpc_path:
        raise TypeError("Compile the active ImageTextToText component before generating runner inputs.")
    specialization, runner_metadata, vision_inputs, lang_inputs = _prepare_vlm_runner_inputs(
        model=model,
        active_model=active_model,
        processor=processor,
        images=images,
        prompts=prompts,
        image_height=image_height,
        image_width=image_width,
    )
    paired_model = model.lang_model if skip_lang else model.vision_model
    paired_metadata = RunnerMetadata.from_paths(paired_model.onnx_path, _get_compile_dir(paired_model))
    vision_metadata = runner_metadata if skip_lang else paired_metadata
    language_metadata = paired_metadata if skip_lang else runner_metadata
    cross_qpc = cross_qpc_tensors(vision_metadata, language_metadata)
    if skip_lang:
        # Some processors leave model-specific vision metadata in the language group.
        # The active ONNX graph is the source of truth for the replay invocation.
        host_inputs = _filter_graph_inputs(runner_metadata, vision_inputs, lang_inputs)
        output_shape_overrides = _cross_qpc_output_shapes(
            model, specialization, vision_metadata=vision_metadata, language_metadata=language_metadata
        )
    else:
        host_inputs = lang_inputs
        _add_specialization_control_inputs(runner_metadata, host_inputs, specialization)
        host_inputs = _filter_graph_inputs(runner_metadata, host_inputs)
        _add_cross_qpc_placeholders(
            runner_metadata, host_inputs, cross_qpc, _cross_qpc_symbols(model, specialization)
        )
        output_shape_overrides = None

    return _write_vlm_runner_bundle(
        model=active_model,
        runner_metadata=runner_metadata,
        specialization=specialization,
        host_inputs=host_inputs,
        output_shape_overrides=output_shape_overrides,
    )
