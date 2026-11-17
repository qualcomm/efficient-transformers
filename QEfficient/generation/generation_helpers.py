# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Shared helpers for qaic-runner IO bundle generation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import yaml
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.exporter.weight_free.weight_spec import load_weight_spec, resolve_weight_spec_path
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


def prepare_tokenizer(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
    """Configure tokenizer padding consistently for prefill generation."""
    if tokenizer is None:
        return
    if tokenizer.padding_side != "right":
        logger.warning("Please use padding_side='right' while initializing the tokenizer")
        tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def build_prefill_inputs(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt: Union[str, List[str]],
    prefill_seq_len: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int]:
    """Tokenize and pad host inputs to a whole number of prefill chunks."""
    unpadded_inputs = tokenizer(prompt, return_tensors="np", padding=True)
    position_ids = unpadded_inputs["attention_mask"].sum(1, keepdims=True)
    input_length = unpadded_inputs["input_ids"].shape[1]
    num_chunks = -(input_length // -prefill_seq_len)
    padded_length = num_chunks * prefill_seq_len

    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_length)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_length), -1)
    inputs.pop("token_type_ids", None)
    return inputs, position_ids, num_chunks


def slice_prefill_inputs(
    inputs: Dict[str, np.ndarray], chunk_index: int, prefill_seq_len: int
) -> Dict[str, np.ndarray]:
    """Return one prefill chunk while preserving non-sequence inputs."""
    chunk_inputs = inputs.copy()
    start = chunk_index * prefill_seq_len
    end = start + prefill_seq_len
    chunk_inputs["input_ids"] = inputs["input_ids"][:, start:end]
    chunk_inputs["position_ids"] = inputs["position_ids"][:, start:end]
    return chunk_inputs


_PRECISION_ITEM_SIZES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int8": 1,
    "mxint8": 1,
}


def _get_compile_dir(model) -> Path:
    qpc_path = getattr(model, "qpc_path", None)
    if qpc_path is not None:
        return Path(qpc_path).parent
    compile_dir = getattr(model, "compile_artifacts_path", None)
    if compile_dir is not None:
        return Path(compile_dir)
    raise TypeError("Compile the model or generate compile artifacts before writing runner inputs.")


def load_prefill_specialization(compile_dir: Union[str, Path]) -> Dict[str, int]:
    """Return the first compatible prefill specialization from a workspace.

    This preserves causal artifact generation's historic first-prefill behavior.
    Image-text callers that need a strict component selection use the selectors
    below instead.
    """
    return select_prefill_specialization(compile_dir)


def _specialization_records(compile_dir: Union[str, Path]) -> list[tuple[Optional[str], Dict[str, int]]]:
    """Load named and legacy specialization records without exporter coupling."""
    specializations_path = Path(compile_dir) / "specializations.json"
    if not specializations_path.is_file():
        raise FileNotFoundError(f"specializations.json not found at {specializations_path}.")
    specializations = json.loads(specializations_path.read_text())["specializations"]
    if not specializations:
        raise ValueError(f"No specializations found in {specializations_path}.")
    records = []
    for specialization in specializations:
        symbols = specialization.get("symbols", specialization)
        if not isinstance(symbols, Mapping):
            raise ValueError(f"Invalid specialization record in {specializations_path}.")
        records.append((specialization.get("name"), _specialization_symbols(symbols)))
    return records


def _named_role_candidates(
    records: Sequence[tuple[Optional[str], Dict[str, int]]], role: str
) -> list[tuple[Optional[str], Dict[str, int]]]:
    """Select records for a graph role, falling back to legacy shape rules."""
    # ``to_named_specializations`` makes duplicate compiler names unique with
    # a numeric suffix (for example, ``Vision_0`` and ``Vision_1``).
    named = [
        (name, symbols)
        for name, symbols in records
        if name and (name.lower() == role or name.lower().startswith(f"{role}_"))
    ]
    if named:
        return named
    if role == "vision":
        # Vision exports have no language sequence dimension.  This remains
        # format-based rather than architecture/name-based for legacy records.
        return [(name, symbols) for name, symbols in records if "seq_len" not in symbols]
    if role == "prefill":
        return [(name, symbols) for name, symbols in records if int(symbols.get("seq_len", 0)) > 1]
    if role == "decode":
        return [(name, symbols) for name, symbols in records if int(symbols.get("seq_len", 0)) == 1]
    raise ValueError(f"Unknown specialization role {role!r}.")


def _input_symbol_values(
    graph: onnx.GraphProto, inputs: Mapping[str, np.ndarray]
) -> Dict[str, int]:
    """Derive specialization symbol values from the active graph and tensors."""
    values = {}
    for graph_input in graph.input:
        if graph_input.name not in inputs:
            continue
        tensor = np.asarray(inputs[graph_input.name])
        dimensions = graph_input.type.tensor_type.shape.dim
        if tensor.ndim != len(dimensions):
            raise ValueError(
                f"Input {graph_input.name!r} has rank {tensor.ndim}, expected {len(dimensions)} for specialization selection."
            )
        for dimension, value in zip(dimensions, tensor.shape):
            if dimension.dim_param:
                previous = values.setdefault(dimension.dim_param, int(value))
                if previous != int(value):
                    raise ValueError(
                        f"Inputs disagree on specialization symbol {dimension.dim_param!r}: "
                        f"{previous} and {value}."
                    )
    return values


def _select_specialization(
    compile_dir: Union[str, Path],
    role: str,
    *,
    runner_metadata: Optional["RunnerMetadata"] = None,
    inputs: Optional[Mapping[str, np.ndarray]] = None,
    strict: bool = False,
) -> Dict[str, int]:
    """Select a specialization using graph symbols and observed input shapes."""
    candidates = _named_role_candidates(_specialization_records(compile_dir), role)
    if not candidates:
        raise ValueError(f"No {role} specialization found in {Path(compile_dir) / 'specializations.json'}.")

    observed_symbols = {}
    if runner_metadata is not None and inputs is not None:
        observed_symbols = _input_symbol_values(runner_metadata.graph, inputs)
        candidates = [
            (name, symbols)
            for name, symbols in candidates
            if all(symbols.get(symbol) == value for symbol, value in observed_symbols.items() if symbol in symbols)
        ]

    if not candidates:
        raise ValueError(
            f"No {role} specialization matches graph-derived symbols {dict(sorted(observed_symbols.items()))}."
        )
    if strict and len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {role} specialization matching "
            f"{dict(sorted(observed_symbols.items()))}, found {len(candidates)}."
        )
    return candidates[0][1]


def select_prefill_specialization(
    compile_dir: Union[str, Path],
    *,
    runner_metadata: Optional["RunnerMetadata"] = None,
    inputs: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, int]:
    """Select a language prefill record; preserve first-compatible behavior."""
    return _select_specialization(compile_dir, "prefill", runner_metadata=runner_metadata, inputs=inputs)


def select_decode_specialization(
    compile_dir: Union[str, Path],
    *,
    runner_metadata: Optional["RunnerMetadata"] = None,
    inputs: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, int]:
    """Select a language decode record, including explicitly named variants."""
    return _select_specialization(compile_dir, "decode", runner_metadata=runner_metadata, inputs=inputs)


def select_vision_specialization(
    compile_dir: Union[str, Path], runner_metadata: "RunnerMetadata", inputs: Mapping[str, np.ndarray]
) -> Dict[str, int]:
    """Select exactly one vision record from graph-derived processor shapes."""
    return _select_specialization(
        compile_dir, "vision", runner_metadata=runner_metadata, inputs=inputs, strict=True
    )


def _custom_io_maps(compile_dir: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    custom_io_path = compile_dir / "custom_io.yaml"
    if not custom_io_path.is_file():
        return {}, {}

    precision_map = {}
    item_size_map = {}
    for entry in yaml.safe_load(custom_io_path.read_text()) or []:
        io_name = entry.get("IOName")
        precision = entry.get("Precision")
        if not io_name or not precision:
            continue
        precision_map[io_name] = precision
        if precision in _PRECISION_ITEM_SIZES:
            item_size_map[io_name] = _PRECISION_ITEM_SIZES[precision]
    return precision_map, item_size_map


def _specialization_symbols(specialization: Mapping[str, int]) -> Dict[str, int]:
    return {name: int(value) for name, value in specialization.items() if str(value).lstrip("-").isdigit()}


def _execution_batch_size(specialization: Mapping[str, int]) -> int:
    return int(specialization.get("batch_size", 1))


def _first_execution_batch(values: Sequence, batch_size: int, name: str) -> List:
    if not values:
        raise ValueError(f"`{name}` must contain at least one value.")
    values = list(values)
    if len(values) < batch_size:
        logger.warning(f"Number of {name} is less than the compiled batch size; repeating to match it.")
        values = values * (batch_size // len(values) + 1)
    return values[:batch_size]


def _concat_input_batches(input_batches: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    merged_inputs = {}
    input_names = set().union(*(input_batch.keys() for input_batch in input_batches))
    for input_name in input_names:
        values = [np.asarray(input_batch[input_name]) for input_batch in input_batches if input_name in input_batch]
        if len(values) != len(input_batches):
            raise ValueError(f"Processor output {input_name!r} is missing from one or more batch entries.")
        try:
            merged_inputs[input_name] = np.concatenate(values, axis=0)
        except ValueError as error:
            raise ValueError(f"Processor output {input_name!r} cannot be batched for artifacts mode replay.") from error
    return merged_inputs


def _prepare_vlm_execution_inputs(
    handler, images: Sequence, prompts: Sequence[str], prefill_seq_len: int, batch_size: int
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    batch_images = _first_execution_batch(images, batch_size, "images")
    batch_prompts = _first_execution_batch(prompts, batch_size, "prompts")
    vision_batches = []
    lang_batches = []
    for image, prompt in zip(batch_images, batch_prompts):
        vision_inputs, lang_inputs, _ = handler.prepare_processor_inputs(image, prompt, prefill_seq_len)
        vision_batches.append(vision_inputs)
        lang_batches.append(_slice_vlm_prefill_inputs(lang_inputs, prefill_seq_len))
    return _concat_input_batches(vision_batches), _concat_input_batches(lang_batches)


def _component_prefill_symbols(component) -> Dict[str, int]:
    try:
        compile_dir = _get_compile_dir(component)
        specialization = load_prefill_specialization(compile_dir)
    except (FileNotFoundError, TypeError, ValueError):
        return {}
    return _specialization_symbols(specialization)


def _apply_input_shapes(model: onnx.ModelProto, shape_overrides: Mapping[str, Sequence[int]]) -> None:
    for graph_input in model.graph.input:
        shape = shape_overrides.get(graph_input.name)
        if shape is None or len(graph_input.type.tensor_type.shape.dim) != len(shape):
            continue
        for dimension, value in zip(graph_input.type.tensor_type.shape.dim, shape):
            dimension.ClearField("dim_param")
            dimension.dim_value = int(value)


def _resolve_output_shape(
    output: onnx.ValueInfoProto,
    symbols: Mapping[str, int],
    fallback_batch_size: Optional[int],
    fallback_logits_width: Optional[int] = None,
) -> List[int]:
    shape = []
    for axis, dimension in enumerate(output.type.tensor_type.shape.dim):
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            shape.append(int(dimension.dim_value))
        elif dimension.dim_param in symbols:
            shape.append(int(symbols[dimension.dim_param]))
        elif axis == 0 and fallback_batch_size is not None:
            shape.append(fallback_batch_size)
        elif output.name == "logits" and axis == 1 and "seq_len" in symbols:
            shape.append(int(symbols["seq_len"]))
        elif output.name == "logits" and axis == len(output.type.tensor_type.shape.dim) - 1 and fallback_logits_width:
            shape.append(fallback_logits_width)
        else:
            raise RuntimeError(f"Cannot resolve dimension {axis} ('{dimension.dim_param}') of output '{output.name}'.")
    return shape


def _weight_free_input_names(onnx_path: Path) -> set[str]:
    """Return the input names that refer to constants in a weight-free model."""
    weight_spec_path = resolve_weight_spec_path(onnx_path)
    if not weight_spec_path.is_file():
        return set()
    return {spec_input.name for spec_input in load_weight_spec(weight_spec_path).inputs}


def _required_host_input_names(graph: onnx.GraphProto, onnx_path: Path) -> set[str]:
    retained_inputs = set()
    for output in graph.output:
        for suffix in ("_InternalRetainedState", "_RetainedState"):
            if output.name.endswith(suffix):
                retained_inputs.add(output.name[: -len(suffix)])
    initializer_names = {initializer.name for initializer in graph.initializer}
    weight_free_inputs = _weight_free_input_names(onnx_path)
    return {
        graph_input.name
        for graph_input in graph.input
        if graph_input.name not in retained_inputs
        and graph_input.name not in initializer_names
        and graph_input.name not in weight_free_inputs
    }


@dataclass
class RunnerMetadata:
    """Cached ONNX and compile metadata for qaic-runner IO helpers."""

    onnx_path: Path
    compile_dir: Path
    model: onnx.ModelProto
    graph: onnx.GraphProto
    input_names: set[str]
    required_host_input_names: set[str]
    output_names: set[str]
    custom_io_precisions: Dict[str, str]
    custom_io_item_sizes: Dict[str, int]

    @classmethod
    def from_paths(cls, onnx_path: Union[str, Path], compile_dir: Union[str, Path]) -> "RunnerMetadata":
        onnx_path = Path(onnx_path)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Exported ONNX not found at {onnx_path}.")

        compile_dir = Path(compile_dir)
        model = onnx.load(str(onnx_path), load_external_data=False)
        graph = model.graph
        custom_io_precisions, custom_io_item_sizes = _custom_io_maps(compile_dir)
        return cls(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            model=model,
            graph=graph,
            input_names={graph_input.name for graph_input in graph.input},
            required_host_input_names=_required_host_input_names(graph, onnx_path),
            output_names={output.name for output in graph.output},
            custom_io_precisions=custom_io_precisions,
            custom_io_item_sizes=custom_io_item_sizes,
        )


def _add_specialization_control_inputs(
    runner_metadata: RunnerMetadata,
    host_inputs: Dict[str, np.ndarray],
    specialization: Mapping[str, int],
    sampling_params: Optional[Mapping[str, np.ndarray]] = None,
) -> None:
    input_names = runner_metadata.required_host_input_names
    batch_size = int(specialization.get("batch_size", 1))
    if "batch_index" in input_names and "batch_index" not in host_inputs:
        host_inputs["batch_index"] = np.zeros((batch_size, 1), dtype=np.int64)
    if "comp_ctx_lengths" in input_names and "comp_ctx_lengths" not in host_inputs:
        host_inputs["comp_ctx_lengths"] = np.zeros(int(specialization["comp_ctx_lengths"]), dtype=np.int64)
    if "num_logits_to_keep" in input_names and "num_logits_to_keep" not in host_inputs:
        host_inputs["num_logits_to_keep"] = np.zeros((batch_size, 1), dtype=np.int64)
    if "lora_ids" in input_names and "lora_ids" not in host_inputs:
        host_inputs["lora_ids"] = np.zeros((batch_size, 1), dtype=np.int64)
    if "last_accepted_output_tokens" in input_names and "input_ids" in host_inputs:
        host_inputs["last_accepted_output_tokens"] = host_inputs["input_ids"].copy()
    for name, value in (sampling_params or {}).items():
        if name in input_names:
            host_inputs[name] = np.asarray(value)


def _proxy_logits_width(model) -> Optional[int]:
    if not getattr(model, "_enable_proxy", False):
        return None

    config = model.model.config
    candidates = [config, getattr(config, "text_config", None), getattr(config, "language_config", None)]
    for candidate in filter(None, candidates):
        for attribute in ("hidden_size", "n_embd", "d_model"):
            if (hidden_size := getattr(candidate, attribute, None)) is not None:
                return int(hidden_size)
    raise AttributeError("Proxy model configuration does not expose its hidden size.")


def _slice_vlm_prefill_inputs(lang_inputs: Mapping[str, np.ndarray], prefill_seq_len: int) -> Dict[str, np.ndarray]:
    host_inputs = {}
    for name, value in lang_inputs.items():
        value = np.asarray(value)
        if name in {"input_ids", "position_ids", "mm_token_type_ids", "token_type_ids"}:
            host_inputs[name] = value[..., :prefill_seq_len]
        elif name == "cross_attention_mask":
            host_inputs[name] = value[:, :prefill_seq_len, ...]
        elif name in {"image_idx", "batch_index"}:
            host_inputs[name] = value
    return host_inputs


def _filter_graph_inputs(
    runner_metadata: RunnerMetadata, *input_groups: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    return {
        name: np.asarray(value)
        for inputs in input_groups
        for name, value in inputs.items()
        if name in runner_metadata.input_names
    }


def _resolve_tensor_shape(value_info: onnx.ValueInfoProto, symbols: Mapping[str, int]) -> List[int]:
    """Resolve concrete and declared symbolic dimensions for one ONNX tensor."""
    shape = []
    for axis, dimension in enumerate(value_info.type.tensor_type.shape.dim):
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            shape.append(int(dimension.dim_value))
        elif dimension.dim_param in symbols:
            shape.append(int(symbols[dimension.dim_param]))
        else:
            raise RuntimeError(
                f"Cannot resolve dimension {axis} ({dimension.dim_param!r}) of tensor {value_info.name!r}."
            )
    return shape


def validate_runner_input_shapes(
    runner_metadata: RunnerMetadata, host_inputs: Mapping[str, np.ndarray], specialization: Mapping[str, int]
) -> None:
    """Validate runner inputs against ONNX rank, dimensions, and specialization symbols."""
    symbols = _specialization_symbols(specialization)
    observed_symbols = {}
    graph_inputs = {graph_input.name: graph_input for graph_input in runner_metadata.graph.input}
    for name, value in host_inputs.items():
        graph_input = graph_inputs[name]
        tensor = np.asarray(value)
        dimensions = graph_input.type.tensor_type.shape.dim
        if tensor.ndim != len(dimensions):
            raise ValueError(f"Input {name!r} has rank {tensor.ndim}, expected {len(dimensions)}.")
        for axis, (dimension, actual) in enumerate(zip(dimensions, tensor.shape)):
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                if int(actual) != int(dimension.dim_value):
                    raise ValueError(
                        f"Input {name!r} dimension {axis} is {actual}, expected concrete ONNX value {dimension.dim_value}."
                    )
                continue
            symbol = dimension.dim_param
            if symbol in symbols and int(actual) != symbols[symbol]:
                raise ValueError(
                    f"Input {name!r} dimension {axis} is {actual}, expected specialization "
                    f"symbol {symbol}={symbols[symbol]}."
                )
            if symbol:
                previous = observed_symbols.setdefault(symbol, int(actual))
                if previous != int(actual):
                    raise ValueError(
                        f"Input {name!r} dimension {axis} is {actual}, inconsistent with "
                        f"symbol {symbol}={previous} in another runner input."
                    )


@dataclass(frozen=True)
class CrossQpcTensor:
    """A graph-derived tensor contract exchanged from vision QPC to language QPC."""

    name: str
    vision_output: onnx.ValueInfoProto
    language_input: onnx.ValueInfoProto


def _cross_qpc_storage_metadata(runner_metadata: RunnerMetadata, value_info: onnx.ValueInfoProto) -> tuple[str, int]:
    """Return the compiler storage contract for a cross-QPC ONNX tensor."""
    precision = runner_metadata.custom_io_precisions.get(value_info.name)
    if precision is not None:
        normalized = precision.lower()
        item_size = runner_metadata.custom_io_item_sizes.get(value_info.name)
        if item_size is None:
            raise ValueError(f"Unsupported custom-IO precision {precision!r} for cross-QPC tensor {value_info.name!r}.")
        return normalized, item_size
    element_type = value_info.type.tensor_type.elem_type
    if element_type == onnx.TensorProto.BFLOAT16:
        return "bfloat16", 2
    try:
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported ONNX dtype for cross-QPC tensor {value_info.name!r}.") from error
    return str(dtype), dtype.itemsize


def cross_qpc_tensors(vision_metadata: RunnerMetadata, language_metadata: RunnerMetadata) -> Dict[str, CrossQpcTensor]:
    """Return the normalized vision-output/language-input contract by ONNX name."""
    vision_outputs = {output.name: output for output in vision_metadata.graph.output}
    language_inputs = {graph_input.name: graph_input for graph_input in language_metadata.graph.input}
    contracts = {}
    for name, vision_output in vision_outputs.items():
        if name not in language_inputs:
            continue
        language_input = language_inputs[name]
        vision_storage = _cross_qpc_storage_metadata(vision_metadata, vision_output)
        language_storage = _cross_qpc_storage_metadata(language_metadata, language_input)
        if vision_storage != language_storage:
            raise ValueError(
                f"Cross-QPC tensor {name!r} has incompatible vision storage {vision_storage} "
                f"and language storage {language_storage}."
            )
        contracts[name] = CrossQpcTensor(name, vision_output, language_input)
    return contracts


def _cross_qpc_symbols(model, vision_specialization: Mapping[str, int]) -> Dict[str, int]:
    symbols = _specialization_symbols(vision_specialization)
    for name, value in _component_prefill_symbols(getattr(model, "lang_model", None)).items():
        symbols.setdefault(name, value)
    if "batch_size" in symbols:
        symbols.setdefault("vision_batch_size", symbols["batch_size"])
    return symbols


def _add_cross_qpc_placeholders(
    runner_metadata: RunnerMetadata,
    host_inputs: Dict[str, np.ndarray],
    contracts: Mapping[str, CrossQpcTensor],
    symbols: Mapping[str, int],
) -> None:
    custom_precisions = runner_metadata.custom_io_precisions
    precision_dtypes = {
        "bfloat16": np.uint16,
        "float16": np.float16,
        "float32": np.float32,
        "int8": np.int8,
        "mxint8": np.int8,
    }
    for name, contract in contracts.items():
        if name in host_inputs:
            continue
        graph_input = contract.language_input
        shape = _resolve_tensor_shape(graph_input, symbols)
        precision = custom_precisions.get(name)
        dtype = precision_dtypes.get(
            precision, onnx.helper.tensor_dtype_to_np_dtype(graph_input.type.tensor_type.elem_type)
        )
        host_inputs[name] = np.zeros(shape, dtype=dtype)
        logger.warning(
            f"Wrote a zero placeholder for {name!r}; replace it with the vision QPC output before replay."
        )


def _cross_qpc_output_shapes(
    model,
    specialization: Mapping[str, int],
    vision_metadata: RunnerMetadata,
    language_metadata: RunnerMetadata,
) -> Dict[str, List[int]]:
    """Resolve and validate vision output shapes from the language input contract."""
    symbols = _cross_qpc_symbols(model, specialization)
    output_shapes = {}
    for name, contract in cross_qpc_tensors(vision_metadata, language_metadata).items():
        vision_shape = _resolve_tensor_shape(contract.vision_output, symbols)
        language_shape = _resolve_tensor_shape(contract.language_input, symbols)
        if vision_shape != language_shape:
            raise ValueError(
                f"Cross-QPC tensor {name!r} has incompatible vision output shape {vision_shape} "
                f"and language input shape {language_shape}."
            )
        output_shapes[name] = vision_shape
    return output_shapes
