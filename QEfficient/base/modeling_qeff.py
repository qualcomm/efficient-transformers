# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import gc
import inspect
import os
import shlex
import shutil
import subprocess
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import onnx
import torch

from QEfficient.base.checkpoint_transforms import BaseCheckpointTransform
from QEfficient.base.onnx_transforms import (
    BaseOnnxTransform,
    CustomOpTransform,
    FP16ClipTransform,
    LocalizeFunctionReduceSumAxesTransform,
    OnnxTransformPipeline,
    RenameFunctionOutputsTransform,
    SplitTensorsTransform,
)
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.blocking.blocking_configurator import (
    build_gated_delta_config_for_transform,
    build_transformer_blocking_config_for_transform,
)
from QEfficient.compile.mdp_generator import (
    MdpStrategy,
    generate_disagg_mdp_config,
    generate_mdp_partition_config,
)
from QEfficient.compile.qnn_compiler import compile as qnn_compile
from QEfficient.exporter.weight_free.export import embed_weight_spec_as_metadata, link_prepared_checkpoint_dir
from QEfficient.exporter.weight_free.weight_spec import load_weight_spec, resolve_weight_spec_path
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.pytorch_transforms import (
    BlockingAttentionTransform,
    GatedDeltaConfigTransform,
    OptimizedMoETransform,
    ReplicateKVHeadTransform,
)
from QEfficient.utils import (
    align_kv_input_names_to_retained_outputs,
    apply_kv_cache_prefix,
    constants,
    create_json,
    create_model_params,
    dump_qconfig,
    get_attr_or_key,
    hash_dict_params,
    load_json,
    require_value,
    to_named_specializations,
)
from QEfficient.utils.checkpoint_utils import checkpoint_root, huggingface_hub_cache_dir
from QEfficient.utils.config_utils import calculate_num_replicate_kv_heads
from QEfficient.utils.export_utils import export_from_compile, export_wrapper
from QEfficient.utils.logging_utils import (
    QEFFLogger,
    log_api_arguments,
    log_from_pretrained_call,
    log_generate_call,
)
from QEfficient.utils.torch_patches import layerwise_safe_onnx_export_patches

logger = QEFFLogger.get_logger("INFRA")


_LEGACY_MOE_PREFILL_PACKED_CHUNK_SIZE_ERROR = (
    "moe_prefill_packed_chunk_size is no longer supported; use qaic_config['moe_config']['expert_parallel_chunk_size']"
)


def reject_legacy_moe_prefill_packed_chunk_size(kwargs: Optional[dict]) -> None:
    if kwargs and "moe_prefill_packed_chunk_size" in kwargs:
        raise TypeError(_LEGACY_MOE_PREFILL_PACKED_CHUNK_SIZE_ERROR)


def _copy_existing_compiler_input(command: List[str], flag: str, compile_dir: Path) -> None:
    """Copy a file-valued compiler input into compile_dir and update command in-place."""
    for index, argument in enumerate(command):
        option, separator, value = argument.partition("=")
        if option != flag or not separator:
            continue

        input_path = Path(value)
        if not input_path.is_file():
            continue

        artifact_path = compile_dir / input_path.name
        if input_path.resolve() != artifact_path.resolve():
            shutil.copy2(input_path, artifact_path)
        command[index] = f"{flag}={artifact_path}"


def _copy_onnx_external_data_files(source_onnx_path: Path, artifact_onnx_path: Path) -> None:
    """Copy ONNX-native external tensor data referenced by source_onnx_path.

    This handles ONNX ``TensorProto.EXTERNAL`` files that must remain beside the
    replay ONNX. It intentionally does not copy weight-free checkpoint files;
    those are resolved through ``AIC_EXTERNAL_DATA_ROOT``.
    """
    model = onnx.load(source_onnx_path, load_external_data=False)
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        location = next((entry.value for entry in tensor.external_data if entry.key == "location"), None)
        if not location:
            continue
        location_path = Path(location)
        if location_path.is_absolute():
            source_path = location_path
            artifact_path = artifact_onnx_path.parent / location_path.name
        else:
            source_path = source_onnx_path.parent / location_path
            artifact_path = artifact_onnx_path.parent / location_path
        if not source_path.is_file():
            raise FileNotFoundError(f"ONNX external data file not found at: {source_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != artifact_path.resolve():
            shutil.copy2(source_path, artifact_path)


def _weight_free_input_names(onnx_path: Path) -> set[str]:
    """Return the input names that refer to constants in a weight-free model."""
    weight_spec_path = resolve_weight_spec_path(onnx_path)
    return _weight_free_spec_input_names(weight_spec_path)


def _weight_free_spec_input_names(weight_spec_path: Path) -> set[str]:
    """Return the input names recorded in a weight-free sidecar."""
    if not weight_spec_path.is_file():
        return set()
    return {spec_input.name for spec_input in load_weight_spec(weight_spec_path).inputs}


def _copy_weight_free_spec(weight_spec_path: Path, artifact_dir: Path) -> None:
    """Copy only the weight-free sidecar metadata, not checkpoint weight files.

    The compiler consumes the same extdata metadata embedded in the ONNX and
    resolves its ``files`` entries from ``AIC_EXTERNAL_DATA_ROOT``. Copying the
    sidecar keeps the bundle inspectable without creating stale duplicate weight
    files that still point back to their original cache location.
    """
    if not weight_spec_path.is_file():
        raise FileNotFoundError(f"Weight spec file not found at: {weight_spec_path}")

    artifact_weight_spec_path = artifact_dir / weight_spec_path.name
    if weight_spec_path.resolve() != artifact_weight_spec_path.resolve():
        shutil.copy2(weight_spec_path, artifact_weight_spec_path)


def _root_containing_weight_free_files(model_id_path: Path, external_files) -> Optional[Path]:
    """Return the nearest ancestor that resolves all relative weight-spec files."""
    relative_paths = [
        Path(external_file.path) for external_file in external_files if not Path(external_file.path).is_absolute()
    ]
    if not relative_paths:
        return None

    for root in (model_id_path, *model_id_path.parents):
        if all((root / relative_path).exists() for relative_path in relative_paths):
            return root
    return None


def _weight_free_external_data_root(weight_spec_path: Optional[Union[str, Path]]) -> Optional[Path]:
    """Return the root for weight-free external data files, if a weight spec exists."""
    if weight_spec_path is None:
        return None

    weight_spec_path = Path(weight_spec_path)
    if not weight_spec_path.is_file():
        return None

    spec = load_weight_spec(weight_spec_path)
    if spec.external_data_root:
        return Path(spec.external_data_root).expanduser()

    model_id_path = Path(spec.model_id).expanduser()
    if model_id_path.exists():
        root = _root_containing_weight_free_files(model_id_path, spec.files)
        if root is not None:
            return root
        return checkpoint_root(spec.model_id, [str(model_id_path / external_file.path) for external_file in spec.files])

    return huggingface_hub_cache_dir()


def _compiler_env_with_external_data_root(weight_spec_path: Optional[Union[str, Path]]) -> Optional[Dict[str, str]]:
    """Return a compiler environment containing AIC_EXTERNAL_DATA_ROOT for weight-free models."""
    compiler_env = os.environ.copy()
    if compiler_env.get("AIC_EXTERNAL_DATA_ROOT"):
        return compiler_env

    external_data_root = _weight_free_external_data_root(weight_spec_path)
    if external_data_root is None:
        return None

    compiler_env["AIC_EXTERNAL_DATA_ROOT"] = str(external_data_root)
    return compiler_env


def _copy_model_compiler_input(
    command: List[str],
    compile_dir: Path,
    weight_spec_path: Optional[Union[str, Path]] = None,
) -> None:
    """Copy the ONNX compiler input and dependent files into compile_dir, updating command."""
    for index, argument in enumerate(command):
        option, separator, value = argument.partition("=")
        if option != "-m" or not separator:
            continue

        source_onnx_path = Path(value)
        if not source_onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found at: {source_onnx_path}")

        artifact_onnx_path = compile_dir / source_onnx_path.name
        if source_onnx_path.resolve() != artifact_onnx_path.resolve():
            shutil.copy2(source_onnx_path, artifact_onnx_path)
        _copy_onnx_external_data_files(source_onnx_path, artifact_onnx_path)

        resolved_weight_spec_path = (
            Path(weight_spec_path) if weight_spec_path is not None else source_onnx_path.with_name("weight_spec.json")
        )
        if _weight_free_spec_input_names(resolved_weight_spec_path):
            _copy_weight_free_spec(resolved_weight_spec_path, compile_dir)
        command[index] = f"-m={artifact_onnx_path}"
        return


def _rename_graph_value(graph: onnx.GraphProto, old_name: str, new_name: str) -> None:
    """Rename a graph value everywhere it can be referenced in an ONNX graph."""
    if old_name == new_name:
        return
    for node in graph.node:
        node.input[:] = [new_name if value == old_name else value for value in node.input]
        node.output[:] = [new_name if value == old_name else value for value in node.output]
    for initializer in graph.initializer:
        if initializer.name == old_name:
            initializer.name = new_name
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.name == old_name:
            value_info.name = new_name


def _restore_retained_state_output_names(model: onnx.ModelProto, output_names: List[str]) -> None:
    """Restore retained-state output names when ONNX subfunction transforms rewrite them."""
    for output_idx, expected_name in enumerate(output_names):
        if output_idx >= len(model.graph.output):
            break
        expected_name = expected_name.replace("_InternalRetainedState", "_RetainedState")
        current_name = model.graph.output[output_idx].name
        if "_RetainedState" not in expected_name:
            continue
        if current_name == expected_name:
            continue
        if current_name.isdigit() or "_InternalRetainedState" in current_name or "_RetainedState" in current_name:
            _rename_graph_value(model.graph, current_name, expected_name)


def _restore_output_names_exact(model: onnx.ModelProto, output_names: List[str]) -> None:
    """Force graph output names to match ``output_names`` by positional index."""
    for output_idx, expected_name in enumerate(output_names):
        if output_idx >= len(model.graph.output):
            break
        current_name = model.graph.output[output_idx].name
        _rename_graph_value(model.graph, current_name, expected_name)


def generate_mdp_compiler_dump(
    compile_dir: Path,
    mdp_ts_num_devices: int,
    mdp_num_partitions: int,
    compile_command: List[str],
    specializations: Optional[List[Dict[str, int]]] = None,
    specialization_module_name: Optional[str] = None,
    custom_io: Optional[Dict[str, str]] = None,
    compiler_env: Optional[Dict[str, str]] = None,
) -> str:
    """Generate the compiler MDP dump required by the intersection strategy."""
    # The compiler dump is generated before the final QPC compile hash directory exists.
    # Keep these intermediate files under their own input-keyed directory so changed
    # compile options, specializations, or custom IO cannot reuse a stale dump.
    dump_hash_params = {
        "compile_command": compile_command,
        "specializations": specializations,
        "specialization_module_name": specialization_module_name,
        "custom_io": custom_io,
        "mdp_ts_num_devices": mdp_ts_num_devices,
        "mdp_num_partitions": mdp_num_partitions,
    }
    mdp_dump_dir = compile_dir / f"mdp_{hash_dict_params(dump_hash_params)}"
    mdp_compiler_dump_path = str(
        mdp_dump_dir / f"tmp_mdp_compiler_dump_{mdp_ts_num_devices}d_{mdp_num_partitions}p.json"
    )
    dump_path = Path(mdp_compiler_dump_path)
    if dump_path.exists():
        return mdp_compiler_dump_path

    dump_path.parent.mkdir(parents=True, exist_ok=True)

    dump_command = list(compile_command)
    if specializations is not None:
        mdp_dump_dir.mkdir(parents=True, exist_ok=True)
        dump_specializations_json = mdp_dump_dir / "specializations.json"
        create_json(
            str(dump_specializations_json),
            {"specializations": to_named_specializations(specializations, module_name=specialization_module_name)},
        )
        dump_command.append(f"-network-specialization-config={dump_specializations_json}")
    if custom_io is not None:
        mdp_dump_dir.mkdir(parents=True, exist_ok=True)
        dump_custom_io_yaml = mdp_dump_dir / "custom_io.yaml"
        with open(dump_custom_io_yaml, "w") as fp:
            for io_name, dtype in custom_io.items():
                fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")
        dump_command.append(f"-custom-IO-list-file={dump_custom_io_yaml}")
    dump_command.append(f"-mdp-dump-partition-config={dump_path}")

    logger.info(f"Running compiler for MDP dump: {' '.join(dump_command)}")
    try:
        subprocess.run(dump_command, capture_output=True, check=True, env=compiler_env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "\n".join(
                [
                    "MDP compiler dump generation failed!",
                    f"Compiler command: {e.cmd}",
                    f"Compiler exitcode: {e.returncode}",
                    "Compiler stderr:",
                    e.stderr.decode(),
                ]
            )
        )
    if not dump_path.exists():
        raise FileNotFoundError(f"MDP compiler dump generation did not create the expected output file - {dump_path} ")
    return mdp_compiler_dump_path


class QEFFBaseModel(ABC):
    """
    Base class for all the model classes (i.e. LLMs, SD, quantized etc.).
    Provides certain utility methods to be used by child classes.

    Class variables:
    :_pytorch_transforms: Pytorch transformations to be applied after initialization.
    :_onnx_transforms: ONNX transformations to be applied after ONNX export.
    """

    _start = 0
    _end = 0
    _total_layers = None
    _layerwise_active = False
    _pytorch_transforms: List[PytorchTransform]
    _onnx_transforms = [BaseOnnxTransform]
    _checkpoint_transforms: List[Type[BaseCheckpointTransform]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from_pretrained = cls.__dict__.get("from_pretrained")
        if isinstance(from_pretrained, classmethod):
            function = from_pretrained.__func__
            if not getattr(function, "_qeff_api_dump", False):
                setattr(cls, "from_pretrained", classmethod(log_from_pretrained_call(function)))
        generate = cls.__dict__.get("generate")
        if generate is not None and callable(generate):
            if not getattr(generate, "_qeff_api_dump", False):
                setattr(cls, "generate", log_generate_call(generate))

    def _transform_names(self) -> List[str]:
        return [x.__name__ for x in self._pytorch_transforms + self._onnx_transforms]

    def maybe_apply_replicate_kv_transform(self, model_config, num_devices: int, qaic_config: Optional[dict]) -> int:
        if model_config is None or qaic_config is None or "EncoderWrapper" in self.model.__class__.__name__:
            return 1

        replicate_kv_heads = qaic_config.get("replicate_kv_heads", False)

        if not replicate_kv_heads:
            return 1

        text_config = getattr(model_config, "text_config", None)
        effective_config = text_config if text_config is not None else model_config

        num_replicate_kv_heads = calculate_num_replicate_kv_heads(
            num_devices=num_devices,
            text_model_config=effective_config,
        )
        if num_replicate_kv_heads is None or num_replicate_kv_heads <= 1:
            return 1

        self.model, replicate_kv_transformed = ReplicateKVHeadTransform.apply(
            self.model,
            num_replicate_kv_heads,
        )
        if not replicate_kv_transformed:
            return 1

        self.hash_params["config"] = (
            model_config.to_diff_dict() if hasattr(model_config, "to_diff_dict") else model_config
        )
        return num_replicate_kv_heads

    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        super().__init__()
        self.model = model
        if not QEFFLogger.has_active_run():
            QEFFLogger.start_run(self.model_name)
        self.config = model.config
        self.hash_params = create_model_params(self, **kwargs)
        if getattr(self, "_enable_proxy", False):
            language_config = self.config
            for config_name in ("text_config", "llm_config", "language_config"):
                if nested_config := getattr(self.config, config_name, None):
                    language_config = nested_config
                    break
            if (proxy_num_hidden_layers := getattr(language_config, "num_hidden_layers", None)) is not None:
                self.hash_params["proxy_num_hidden_layers"] = proxy_num_hidden_layers
        self.onnx_path: Optional[str] = None
        self.qpc_path: Optional[str] = None
        self.compile_artifacts_path: Optional[Path] = None
        self.qpc_session: Optional[QAICInferenceSession] = None
        self.weight_spec_path: Optional[str] = None
        self.model_architecture = (
            (arch := getattr(self.model.config, "architectures", None)) and len(arch) > 0 and arch[0]
        ) or None

        # Flag for checking if weights are offloaded
        self._is_weights_offloaded: bool = False
        self._weight_free: bool = kwargs.get("weight_free", False)
        # Flag for checking if model has been transformed yet
        self.is_transformed: bool = False

        self._normalize_torch_dtype()
        # Apply the transformations
        any_transformed = False
        for transform in self._pytorch_transforms:
            self.model, transformed = transform.apply(self.model)
            any_transformed = any_transformed or transformed

        if not any_transformed:
            warnings.warn(f"No transforms applied to model: {self.model_name}. It may be an unsupported model!")
        else:
            QEFFLogger.log_event(
                "milestone",
                "INFRA",
                f"Applied PyTorch transforms to model: {self.model_name}.",
                milestone=QEFFLogger.MILESTONE_LOAD_COMPLETE,
            )

    def _normalize_torch_dtype(self):
        """
        Normalizes torch_dtype across all nested configs to match the top-level config.

        This method ensures consistency by propagating the top-level torch_dtype
        to all nested configs (llm_config, vision_config, etc.) that may exist in
        multimodal models.
        """
        top_level_dtype = getattr(self.config, "torch_dtype", torch.float32)

        if top_level_dtype is None:
            top_level_dtype = torch.float32
        elif isinstance(top_level_dtype, str):
            top_level_dtype = getattr(torch, top_level_dtype, torch.float32)

        self.config.torch_dtype = top_level_dtype

        # Normalize llm_config if it exists
        if hasattr(self.config, "llm_config"):
            self.config.llm_config.torch_dtype = top_level_dtype
            if hasattr(self.config.llm_config, "use_bfloat16"):
                self.config.llm_config.use_bfloat16 = top_level_dtype == torch.bfloat16

        # Normalize vision_config if it exists
        if hasattr(self.config, "vision_config"):
            self.config.vision_config.torch_dtype = top_level_dtype
            if hasattr(self.config.vision_config, "use_bfloat16"):
                self.config.vision_config.use_bfloat16 = top_level_dtype == torch.bfloat16

        # Normalize text_config if it exists (for models like Qwen2.5-VL)
        if hasattr(self.config, "text_config"):
            self.config.text_config.torch_dtype = top_level_dtype

        logger.info(f"Normalized all config torch_dtype to: {top_level_dtype}")

    def _offload_model_weights(self, offload_pt_weights: bool) -> bool:
        """Clear PyTorch model weights to reduce memory usage after ONNX export."""
        if offload_pt_weights and not self._is_weights_offloaded:
            try:
                # Clear plain tensor attrs (not registered as params/buffers)
                param_data_ptrs = {p.data_ptr() for p in self.model.parameters()}
                buf_data_ptrs = {b.data_ptr() for b in self.model.buffers()}
                registered_ptrs = param_data_ptrs | buf_data_ptrs
                for module in self.model.modules():
                    for attr_name in list(vars(module).keys()):
                        attr = getattr(module, attr_name, None)
                        if isinstance(attr, torch.Tensor) and attr.data_ptr() not in registered_ptrs:
                            setattr(module, attr_name, torch.empty_like(attr, device="meta"))

                # Swap each parameter/buffer with a meta tensor of the same
                # shape, in place — so external Parameter refs also become meta.
                with torch.no_grad():
                    for p in self.model.parameters():
                        new_p = torch.nn.Parameter(
                            torch.empty(p.shape, dtype=p.dtype, device="meta"),
                            requires_grad=p.requires_grad,
                        )
                        torch.utils.swap_tensors(p, new_p)
                    for b in self.model.buffers():
                        new_b = torch.empty(b.shape, dtype=b.dtype, device="meta")
                        torch.utils.swap_tensors(b, new_b)

                gc.collect()

                self._is_weights_offloaded = True
                return True
            except Exception as e:
                logger.warning(f"Weight clearing failed, continuing: {e}")
                return False
        return False

    def _model_offloaded_check(self) -> None:
        """
        Check if the model is in meta state or weights are offloaded.

        Raises:
            RuntimeError: If model is in meta state or if weights are offloaded
        """
        if self._is_weights_offloaded or any(param.is_meta for param in self.model.parameters()):
            error_msg = (
                "Cannot re-export model: weights have been offloaded to save memory. "
                "To re-export, please create a new model instance using from_pretrained() method."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @property
    def model_name(self) -> str:
        """
        Get the model class name without QEff/QEFF prefix.

        This property extracts the underlying model's class name and removes
        any QEff or QEFF prefix that may have been added during wrapping.

        Returns:
            str: Model class name (e.g., "CLIPTextModel" instead of "QEffCLIPTextModel")
        """
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    @abstractmethod
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        This is an abstract property that must be implemented by all subclasses.
        Typically returns: self.model.config.__dict__

        Returns:
            Dict: The configuration dictionary of the underlying model
        """
        pass

    @abstractmethod
    def export(self, export_dir: Optional[str] = None) -> Path:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        .. deprecated::
            Use :meth:`compile` instead. Compilation performs the export with
            the complete model and compiler configuration.

        Args:
            :export_dir (str): Specify the export directory. The export_dir will be suffixed with a hash corresponding to current model.

        Returns:
            :Path: Path of the generated ``ONNX`` file.
        """

    @abstractmethod
    def compile(self, *args, **kwargs) -> Path:
        """
        Compile the exported onnx to run on AI100.
        If the model has not been exported yet, this method will handle the export process.

        Args:
            :onnx_path (str): Onnx file to compile
            :compile_dir (str): Directory path to compile the qpc. A suffix is added to the directory path to avoid reusing same qpc for different parameters.
            :num_devices (int): Number of devices to compile for. ``Defaults to 1``.
            :num_cores (int): Number of cores to utilize in each device ``Defaults to 16``.
            :mxfp6_matmul (bool): Use MXFP6 to compress weights for MatMul nodes to run faster on device. ``Defaults to False``.
            :mxint8_kv_cache (bool): Use MXINT8 to compress KV-cache on device to access and update KV-cache faster. ``Defaults to False``.
            :compiler_options: Pass any compiler option as input.

                Following flag can be passed in compiler_options to enable QNN Compilation path.
                    :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False. if not passed.``
                    :qnn_config (str): Path of QNN Config parameters file. ``Defaults to None. if not passed``

                for QAIC compilation path, any flag that is supported by ``qaic-compile`` can be passed. Params are converted to flags as below:

                    - aic_num_cores=16 -> -aic-num-cores=16
                    - convert_to_fp16=True -> -convert-to-fp16
                    - aic_hw_version=ai100 -> -aic-hw-version=ai100
                    - aic_hw_version=ai200 -> -aic-hw-version=ai200

        ``QEFFAutoModelForCausalLM`` Args:

            :full_batch_size (int): Full batch size to allocate cache lines.
            :batch_size (int): Batch size to compile for. ``Defaults to 1``.
            :prefill_seq_len (int): Prefill sequence length to compile for. Prompt will be chunked according to this length.
            :ctx_len (int): Context length to allocate space for KV-cache tensors.

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """

    @export_wrapper
    def _export(
        self,
        example_inputs: Dict[str, torch.Tensor],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        onnx_transform_kwargs: Optional[Dict[str, Any]] = None,
        export_dir: Optional[str] = None,
        offload_pt_weights: bool = True,
        dynamo: bool = False,
        dynamic_shapes: Optional[Dict[str, Dict[int, Any]]] = None,
        **export_kwargs,
    ) -> str:
        """
        Export the PyTorch model to ONNX and apply ONNX transforms


        This method:
        1. Exports PyTorch model to ONNX using torch.onnx.export
        2. Clears PyTorch weights after export
        3. Applies ONNX transforms with reduced memory footprint

        Args:
            :example_inputs (dict): Sample inputs to trace the model.
            :output_names (list): names to assign to the output nodes of the graph, in order.
            :dynamic_axes (dict): Same as dynamic_axes parameter to be passed to `torch.onnx.export`. Used when dynamo=False.
            :export_kwargs (dict): Additional arguments to be passed to `torch.onnx.export`.
            :onnx_transform_kwargs (dict): Additional arguments to be passed to `Transform.apply` for this class.
            :export_dir (str): Specify the export directory. The export_dir will be suffixed with a hash corresponding to current model.
            :offload_pt_weights (bool): If True, offload PyTorch model weights to meta device
            after successful export to reduce memory usage. Set to False if you need to
            keep weights for further operations. Defaults to True.
            :dynamo (bool): If True, export via torch.export (dynamo path) instead of the legacy torch.onnx.export TorchScript path. Defaults to False.
            :dynamic_shapes (dict): Dynamic shape constraints passed to torch.export when dynamo=True. Keys are input names; values are per-dimension constraint dicts. Ignored when dynamo=False.
            Note:
            Once weights are offloaded, the model cannot be re-exported. Create a new
            instance using from_pretrained() for re-export.

        """
        # TODO: Hack for retain_full_kv, handle this outside
        export_kwargs.pop("retain_full_kv", None)
        onnx_path = export_dir / f"{self.model_name}.onnx"

        # Resolve the weight spec path now — it is always co-located with the ONNX
        # (same directory, named weight_spec.json). Must happen before the early-return
        # so a cache hit restores the full export output state.
        _weight_spec_path = onnx_path.with_name("weight_spec.json")

        # Return early if ONNX already exists (restore complete export output state).
        if onnx_path.is_file():
            self.onnx_path = onnx_path
            self.weight_spec_path = str(_weight_spec_path) if _weight_spec_path.is_file() else None
            self._export_cache_hit = True
            return onnx_path

        export_dir.mkdir(parents=True, exist_ok=True)

        def _resolve_pkv_layers(pkv_obj):
            if isinstance(pkv_obj, (list, tuple)):
                return pkv_obj
            if hasattr(pkv_obj, "to_legacy_cache"):
                return pkv_obj.to_legacy_cache()
            if hasattr(pkv_obj, "layers"):
                layers = []
                for layer in pkv_obj.layers:
                    keys = getattr(layer, "keys", None)
                    values = getattr(layer, "values", None)
                    layers.append((keys, values))
                return tuple(layers)
            return None

        def _resolve_pkv_names(layer_idx, layer_state):
            if hasattr(self.model, "get_onnx_past_key_value_names"):
                names = self.model.get_onnx_past_key_value_names(layer_idx, layer_state)
                if names is not None:
                    return list(names)
            state_len = len(layer_state)
            if state_len == 2:
                return [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
            if state_len == 4:
                return [
                    f"past_key_self.{layer_idx}",
                    f"past_value_self.{layer_idx}",
                    f"past_key_cross.{layer_idx}",
                    f"past_value_cross.{layer_idx}",
                ]
            raise ValueError(
                f"Unknown shape of past_key_values! Expected length of past_key_values for each layer to be either 2 or 4 but got {state_len}"
            )

        # Create input_names from example_inputs
        input_names = []
        for param in inspect.signature(self.model.forward).parameters:
            if param in example_inputs:
                if param == "past_key_values":
                    pkv_layers = _resolve_pkv_layers(example_inputs["past_key_values"])
                    if pkv_layers is None:
                        input_names.append(param)
                        continue
                    for i in range(len(pkv_layers)):
                        input_names.extend(_resolve_pkv_names(i, pkv_layers[i]))
                elif param == "compressed_kvs":
                    for i in range(len(example_inputs["compressed_kvs"])):
                        input_names.extend(
                            [
                                f"compressed_kv.{i}",
                            ]
                        )
                        input_names.extend(
                            [
                                f"k_pe.{i}",
                            ]
                        )
                else:
                    input_names.append(param)

        # When retained-state outputs carry an injected KV-cache prefix
        # (past_key.0_<prefix>_RetainedState), rename the matching KV inputs (past_key.0 ->
        # past_key.0_<prefix>) so the compiler pairs and retains them, and carry the dynamic axes over
        # to the renamed inputs. No-op without a prefix.
        aligned_input_names = align_kv_input_names_to_retained_outputs(input_names, output_names)
        if aligned_input_names != input_names:
            rename_map = {old: new for old, new in zip(input_names, aligned_input_names) if old != new}
            dynamic_axes = {rename_map.get(k, k): v for k, v in dynamic_axes.items()}
            input_names = aligned_input_names

        try:
            if self._weight_free:
                from QEfficient.exporter.onnx_exporter import export_via_weightfree

                export_result = export_via_weightfree(
                    self,
                    onnx_path,
                    example_inputs,
                    input_names,
                    output_names,
                    dynamic_shapes,
                    export_kwargs,
                    onnx_transform_kwargs,
                )
            elif dynamo:
                from QEfficient.exporter.onnx_exporter import export_via_dynamo

                self._model_offloaded_check()
                export_result = export_via_dynamo(
                    self,
                    onnx_path,
                    example_inputs,
                    input_names,
                    output_names,
                    dynamic_shapes,
                    export_kwargs,
                )
                self._offload_model_weights(offload_pt_weights)
            else:
                from QEfficient.exporter.onnx_exporter import export_via_legacy

                self._model_offloaded_check()
                export_result = export_via_legacy(
                    self,
                    onnx_path,
                    example_inputs,
                    input_names,
                    output_names,
                    dynamic_axes,
                    export_kwargs,
                )
                self._offload_model_weights(offload_pt_weights)
            logger.info("PyTorch export successful")
            self.weight_spec_path = str(export_result.weight_spec_path) if export_result.weight_spec_path else None
            model = onnx.load(export_result.onnx_path, load_external_data=False)

            excluded_transforms = set(export_result.excluded_onnx_transforms)
            active_transforms = [
                transform for transform in self._onnx_transforms if transform not in excluded_transforms
            ]
            needs_external_tensor_data = any(
                transform in active_transforms for transform in (FP16ClipTransform, SplitTensorsTransform)
            )
            transform_kwargs = {
                "onnx_base_dir": str(export_dir) if needs_external_tensor_data else None,
                "model_name": self.model_name,
                "dynamic_axes": None if dynamo else dynamic_axes,
                "onnx_export_opset": constants.get_onnx_export_opset(dynamo),
            }
            if onnx_transform_kwargs is not None:
                transform_kwargs.update(onnx_transform_kwargs)
            transform_kwargs.update(export_result.onnx_transform_kwargs)

            onnx_transforms = OnnxTransformPipeline(transforms=active_transforms)
            model, transformed = onnx_transforms.apply(model, **transform_kwargs)

            # Keep this strictly layerwise-scoped so regular non-layerwise export
            # remains backward compatible.
            if QEFFBaseModel._layerwise_active:
                _restore_retained_state_output_names(model, output_names)

            transform_names = [transform.__name__ for transform in self._pytorch_transforms + active_transforms]
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="qeff_transforms", value=",".join(transform_names))
            )
            if self._weight_free and export_result.weight_spec_path:
                embed_weight_spec_as_metadata(model, export_result.weight_spec_path)
            logger.info("ONNX transforms applied")

            onnx_path_tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
            onnx.save(model, onnx_path_tmp)
            onnx_path_tmp.replace(onnx_path)
            del model
            gc.collect()
            logger.info("Transformed ONNX saved")

            if self._weight_free and export_result.weight_spec_path:
                link_prepared_checkpoint_dir(onnx_path, export_result.weight_spec_path)

        except Exception as e:
            logger.error(f"ONNX export or transforms failed: {e}")
            raise e

        self.onnx_path = onnx_path
        return onnx_path

    def get_onnx_path(
        self,
        prefill_only: Optional[bool] = False,
        enable_chunking: Optional[bool] = False,
        specializations: Optional[List[Dict[str, int]]] = None,
        offload_pt_weights: Optional[bool] = True,
        use_onnx_subfunctions: Optional[bool] = False,
        dynamo: Optional[bool] = False,
        retain_full_kv: Optional[bool] = False,
        qaic_config: Optional[dict] = None,
        kv_cache_prefix: Optional[str] = None,
        **compiler_options,
    ):
        kwargs = {
            "offload_pt_weights": offload_pt_weights,
            "use_onnx_subfunctions": use_onnx_subfunctions,
            "dynamo": dynamo,
            "retain_full_kv": retain_full_kv,
        }
        layerwise_cache_probe = compiler_options.pop("_layerwise_cache_probe", False)
        if layerwise_cache_probe:
            kwargs["_layerwise_cache_probe"] = True
        if kv_cache_prefix:
            kwargs["kv_cache_prefix"] = kv_cache_prefix
        num_devices = compiler_options.pop("num_devices", None)
        mdp_ts_num_devices = compiler_options.pop("mdp_ts_num_devices", 1)
        num_devices = int(mdp_ts_num_devices if num_devices is None else num_devices)
        if prefill_only:
            kwargs.update(
                {
                    "prefill_only": prefill_only,
                    "prefill_seq_len": specializations[0].get("seq_len"),
                    "enable_chunking": enable_chunking,
                    "num_cores": compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES),
                }
            )

        if qaic_config is not None:
            kwargs["qaic_config"] = qaic_config

        # Transform before export
        if specializations is not None:
            bs = require_value(get_attr_or_key(specializations[0], ("batch_size", "batch")), "batch size")
            seq_len = get_attr_or_key(specializations[0], ("cl", "seq_len", "sequence_length"))
            ctx_len = get_attr_or_key(specializations[0], ("ctx_len", "context_length"))
        else:
            bs = None
            seq_len = None
            ctx_len = None

        self.transform(
            ctx_len=ctx_len,
            seq_len=seq_len,
            bs=bs,
            num_devices=num_devices,
            qaic_config=qaic_config,
            prefill_only=prefill_only,
            enable_chunking=enable_chunking,
            num_cores=kwargs.get("num_cores", compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES)),
            prefill_seq_len=kwargs.get("prefill_seq_len"),
            **compiler_options,
        )

        with export_from_compile():
            self.export(**kwargs)
        return self.onnx_path

    @export_wrapper
    def _export_layerwise(
        self,
        example_inputs: Dict[str, torch.Tensor],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        onnx_transform_kwargs: Optional[Dict[str, any]] = None,
        export_dir: Optional[str] = None,
        offload_pt_weights: bool = True,
        prefill_only: Optional[bool] = False,
        enable_chunking: Optional[bool] = False,
        num_cores: Optional[int] = constants.DEFAULT_AIC_NUM_CORES,
        qaic_config: Optional[dict] = None,
        prefill_seq_len: Optional[int] = None,
        kv_cache_prefix: Optional[str] = None,
        **export_kwargs,
    ) -> str:
        cache_probe = export_kwargs.pop("_layerwise_cache_probe", False)
        idx = int(QEFFBaseModel._start)
        end_idx = int(getattr(QEFFBaseModel, "_end", idx + 1))
        if end_idx <= idx:
            raise ValueError(f"Invalid export window: start={idx}, end={end_idx}")

        # TODO: Hack for retain_full_kv, handle this outside
        export_kwargs.pop("retain_full_kv", None)
        onnx_path = export_dir / f"{self.model_name}.onnx"

        # Return early if ONNX already exists
        if onnx_path.is_file():
            self.onnx_path = onnx_path
            self._export_cache_hit = True
            return onnx_path

        # Layer-wise reuse: if the merged final ONNX from a prior run exists
        # under the export root (new layout) or final_data/ (legacy layout),
        # skip per-window export entirely. This preserves hash-stable reruns
        # without re-exporting layer shards.
        total_layers = int(getattr(QEFFBaseModel, "_total_layers", 0) or 0)
        cached_merged_paths = []
        if total_layers > 0:
            cached_merged_paths.append(export_dir / f"merged_0-{total_layers}.onnx")
            cached_merged_paths.append(export_dir / "final_data" / f"merged_0-{total_layers}.onnx")
        cached_merged_paths.extend(sorted(export_dir.glob("merged_0-*.onnx"), reverse=True))
        final_data_dir = export_dir / "final_data"
        if final_data_dir.is_dir():
            cached_merged_paths.extend(sorted(final_data_dir.glob("merged_0-*.onnx"), reverse=True))
        for cached_merged in cached_merged_paths:
            if cached_merged.is_file():
                self.onnx_path = cached_merged
                self._export_cache_hit = True
                return self.onnx_path
        if cache_probe:
            return None

        export_dir.mkdir(parents=True, exist_ok=True)

        # Setup temporary paths
        tmp_onnx_dir = export_dir / "onnx_layerwise_tmp"
        tmp_onnx_dir.mkdir(parents=True, exist_ok=True)

        def _resolve_pkv_layers(pkv_obj):
            if isinstance(pkv_obj, (list, tuple)):
                return pkv_obj
            if hasattr(pkv_obj, "to_legacy_cache"):
                return pkv_obj.to_legacy_cache()
            if hasattr(pkv_obj, "layers"):
                layers = []
                for layer in pkv_obj.layers:
                    keys = getattr(layer, "keys", None)
                    values = getattr(layer, "values", None)
                    layers.append((keys, values))
                return tuple(layers)
            return None

        def _resolve_pkv_names(layer_idx, layer_state):
            if hasattr(self.model, "get_onnx_past_key_value_names"):
                names = self.model.get_onnx_past_key_value_names(layer_idx, layer_state)
                if names is not None:
                    return list(names)
            state_len = len(layer_state)
            if state_len == 2:
                return [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
            if state_len == 4:
                return [
                    f"past_key_self.{layer_idx}",
                    f"past_value_self.{layer_idx}",
                    f"past_key_cross.{layer_idx}",
                    f"past_value_cross.{layer_idx}",
                ]
            raise ValueError(
                f"Unknown shape of past_key_values! Expected length of past_key_values for each layer to be either 2 or 4 but got {state_len}"
            )

        is_vision = hasattr(self.model, "language_model")
        output_name = []
        output_name.append("logits")
        if idx == 0:
            if is_vision:
                output_name.append("vision_embeds_RetainedState")
                if "deepstack_features_RetainedState" in output_names:
                    output_name.append("deepstack_features_RetainedState")
                output_name.append("image_idx_output")
        retained_state_suffix = (
            "_InternalRetainedState" if export_kwargs.get("use_onnx_subfunctions", False) else "_RetainedState"
        )
        for layer_idx in range(idx, end_idx):
            layer_states = _resolve_pkv_layers(example_inputs.get("past_key_values"))
            if layer_states is None:
                output_name.append(f"past_key.{layer_idx}{retained_state_suffix}")
                output_name.append(f"past_value.{layer_idx}{retained_state_suffix}")
            else:
                output_name.extend(
                    [
                        f"{name}{retained_state_suffix}"
                        for name in _resolve_pkv_names(layer_idx, layer_states[layer_idx])
                    ]
                )

        # Inject the optional vLLM KV-cache prefix into the freshly built per-window output names
        # (past_key.3_RetainedState -> past_key.3_<prefix>_RetainedState), using the same helper the
        # non-layerwise paths use. The matching input buffers are renamed to pair with these outputs
        # just below via align_kv_input_names_to_retained_outputs. No-op when kv_cache_prefix is falsy.
        output_name = apply_kv_cache_prefix(output_name, kv_cache_prefix)

        # For some decoder wrappers (e.g. VLM language wrappers), forward does not accept
        # `inputs_embeds`; keep `input_ids` in those cases.
        if idx >= 1:
            z = example_inputs.pop("input_ids")
            if is_vision:
                hidden_size = self.model.language_model.config.hidden_size
                embed_dtype = getattr(self.model.language_model.config, "torch_dtype", None)
            else:
                hidden_size = self.model.model.config.hidden_size
                embed_dtype = getattr(self.model.model.config, "torch_dtype", None)
            # Match the model's dtype so per-window export does not introduce a
            # float32/float16 mismatch when running through fp16 decoder layers.
            if embed_dtype is None:
                embed_dtype = next(self.model.parameters()).dtype
            inputs_embeds = torch.rand(z.shape[0], z.shape[1], hidden_size, device=z.device, dtype=embed_dtype)
            example_inputs["inputs_embeds"] = inputs_embeds
            dynamic_axes["inputs_embeds"] = dynamic_axes.pop("input_ids")

        window_size = end_idx - idx
        if "compressed_kvs" in example_inputs:
            example_inputs["compressed_kvs"] = [
                val for i, val in enumerate(example_inputs["compressed_kvs"]) if i < window_size
            ]

        if "past_key_values" in example_inputs:
            pkv_layers = _resolve_pkv_layers(example_inputs["past_key_values"])
            if pkv_layers is not None:
                if idx >= len(pkv_layers):
                    raise ValueError(
                        f"Invalid past_key_values index {idx} for length {len(pkv_layers)} in layerwise export"
                    )
                example_inputs["past_key_values"] = [pkv_layers[idx]]
        # Create input_names from example_inputs
        input_names = []
        for param in inspect.signature(self.model.forward).parameters:
            if param in example_inputs:
                if param == "past_key_values":
                    pkv_layers = _resolve_pkv_layers(example_inputs["past_key_values"])
                    if pkv_layers is None:
                        input_names.append(param)
                        continue
                    example_inputs["past_key_values"] = [val for i, val in enumerate(pkv_layers) if i < window_size]
                    for i in range(len(example_inputs["past_key_values"])):
                        for layer_offset in range(len(example_inputs["past_key_values"])):
                            layer_idx = idx + layer_offset
                            input_names.extend(
                                _resolve_pkv_names(layer_idx, example_inputs["past_key_values"][layer_offset])
                            )
                        break
                elif param == "compressed_kvs":
                    for layer_offset in range(len(example_inputs["compressed_kvs"])):
                        layer_idx = idx + layer_offset
                        input_names.extend([f"compressed_kv.{layer_idx}", f"k_pe.{layer_idx}"])
                else:
                    input_names.append(param)
        dynamic_axes = {k: v for k, v in dynamic_axes.items() if k in input_names}

        layerwise_dir = export_dir / "onnx_layerwise_tmp"
        start_time = time.time()

        # example_inputs["layer_indices_to_run"] = [i]
        current_layer_dir = layerwise_dir / f"layer_{idx}_{end_idx}"
        current_layer_dir.mkdir(parents=True, exist_ok=True)

        layer_onnx_path = str(current_layer_dir / f"{self.model_name}_layer_{idx}_{end_idx}.onnx")
        layer_onnx_path_tmp = str(current_layer_dir / f"{self.model_name}_layer_tmp_{idx}_{end_idx}.onnx")
        output_names = output_name
        # Align KV input names to match any prefix injected into the retained-state output names
        # (e.g. past_key.3 → past_key.3_vllmKvCache when output is past_key.3_vllmKvCache_RetainedState).
        aligned_input_names = align_kv_input_names_to_retained_outputs(input_names, output_names)
        if aligned_input_names != input_names:
            rename_map = {old: new for old, new in zip(input_names, aligned_input_names) if old != new}
            dynamic_axes = {rename_map.get(k, k): v for k, v in dynamic_axes.items()}
            input_names = aligned_input_names
        if not os.path.isfile(layer_onnx_path):
            with layerwise_safe_onnx_export_patches(enabled=bool(prefill_only)):
                torch.onnx.export(
                    self.model,
                    (),
                    layer_onnx_path_tmp,
                    kwargs=example_inputs,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=constants.ONNX_LEGACY_EXPORT_OPSET,
                    dynamo=False,
                    **export_kwargs,
                )
            total_end = time.time()
            print(f"\nTotal export time: {total_end - start_time:.2f} seconds")

        model = onnx.load(layer_onnx_path_tmp, load_external_data=False)
        transform_kwargs = {
            "onnx_base_dir": str(current_layer_dir),
            "model_name": self.model_name,
            "layer_idx": idx,
        }
        _onnx_transforms = [SplitTensorsTransform, CustomOpTransform, RenameFunctionOutputsTransform]
        if export_kwargs.get("use_onnx_subfunctions", False):
            _onnx_transforms.append(LocalizeFunctionReduceSumAxesTransform)
        onnx_transforms = OnnxTransformPipeline(transforms=_onnx_transforms)
        model, transformed = onnx_transforms.apply(model, **transform_kwargs)

        # Layer windows are stitched by name, so preserve the requested output
        # names after transforms normalize function/custom-op outputs.
        _restore_output_names_exact(model, output_names)

        onnx.save(model, layer_onnx_path_tmp)
        self.onnx_path = layer_onnx_path_tmp
        return layer_onnx_path_tmp

    def transform(
        self,
        ctx_len: Optional[int] = None,
        seq_len: Optional[int] = None,
        bs: Optional[int] = 1,
        num_devices: int = 1,
        qaic_config: Optional[dict] = None,
        **compiler_options,
    ):
        # Apply the transformations that are dependent on compilation parameters
        moe_batch_size = compiler_options.pop("moe_batch_size", bs)
        model_config = getattr(self.model, "config", None) or getattr(
            getattr(self.model, "model", None), "config", None
        )
        effective_num_replicate_kv_heads = self.maybe_apply_replicate_kv_transform(
            model_config,
            num_devices,
            qaic_config,
        )
        if model_config:
            blocking_config = build_transformer_blocking_config_for_transform(
                model_config,
                ctx_len=ctx_len,
                seq_len=seq_len,
                bs=bs,
                num_devices=num_devices,
                qaic_config=qaic_config,
                **compiler_options,
            )
        else:
            # without a model config, this is not a model that is possible to block
            blocking_config = None

        if blocking_config is not None:
            self.model, _ = BlockingAttentionTransform.apply(self.model, attn_blocking_config=blocking_config)
            self.hash_params["blocking_kwargs"] = blocking_config
        else:
            self.hash_params.pop("blocking_kwargs", None)
        gated_delta_config = build_gated_delta_config_for_transform(seq_len=seq_len, qaic_config=qaic_config)
        self.model, gated_delta_transformed = GatedDeltaConfigTransform.apply(
            self.model, gated_delta_config=gated_delta_config
        )
        if gated_delta_transformed and gated_delta_config is not None:
            self.hash_params["gated_delta_kwargs"] = gated_delta_config
        else:
            self.hash_params.pop("gated_delta_kwargs", None)
        if qaic_config is not None:
            self.hash_params["qaic_config"] = qaic_config
        else:
            self.hash_params.pop("qaic_config", None)
        self.hash_params["num_replicate_kv_heads"] = effective_num_replicate_kv_heads

        num_cores = compiler_options.get("num_cores", compiler_options.get("aic_num_cores"))
        if num_cores is None:
            num_cores = constants.DEFAULT_AIC_NUM_CORES
        prefill_seq_len = compiler_options.get("prefill_seq_len", seq_len)
        mdp_num_partitions = compiler_options.get("mdp_num_partitions", 1)
        if mdp_num_partitions is None:
            mdp_num_partitions = 1
        mdp_num_partitions = int(mdp_num_partitions)
        if mdp_num_partitions <= 0:
            raise ValueError("mdp_num_partitions must be greater than zero")
        moe_num_devices = int(num_devices)
        if mdp_num_partitions > 1:
            moe_num_devices = moe_num_devices // mdp_num_partitions
        reject_legacy_moe_prefill_packed_chunk_size(compiler_options)
        self.model, _ = OptimizedMoETransform.apply(
            self.model,
            prefill_only=bool(compiler_options.get("prefill_only", False)),
            batch_size=moe_batch_size,
            num_devices=moe_num_devices,
            num_cores=num_cores,
            qaic_config=qaic_config,
            prefill_seq_len=prefill_seq_len,
            hash_params=self.hash_params,
        )

    @dump_qconfig
    def _compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        mxint8_kv_cache: bool = False,
        specializations: Optional[List[Dict[str, int]]] = None,
        custom_io: Optional[Dict[str, str]] = None,
        mdp_ts_num_devices: int = 1,
        mdp_num_partitions: Optional[int] = 1,
        num_speculative_tokens: Optional[Union[int, List[int]]] = None,
        enable_qnn: Optional[bool] = False,
        qnn_config: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        dynamo: bool = False,
        prefill_only: Optional[str] = None,
        offload_pt_weights: Optional[bool] = True,
        enable_chunking: Optional[bool] = False,
        retain_full_kv: Optional[bool] = None,
        qaic_config: Optional[dict] = None,
        specialization_module_name: Optional[str] = None,
        kv_cache_prefix: Optional[str] = None,
        artifacts: bool = False,
        **compiler_options,
    ) -> str:
        """
        Interface for qaic-compile compiler

        Args:
            :onnx_path (str): Onnx file to compile
            :compile_dir (str): Directory path to compile the qpc. A suffix is added to the directory path to avoid reusing same qpc for different parameters.
            :mxint8_kv_cache (bool, optional): Whether to use ``mxint8`` compression for KV cache. ``Defaults to False``.
            :specializations (list): List of specializations to compile for
            :custom_io (dict): Custom IO to specify the input and outputs in different formats than default
            :mdp_ts_num_devices (int): Number of devices to partition to use Multi-Device Partitioning with tensor-slicing.
            :mdp_num_partitions (int): Number of pipeline-parallel partitions for disaggregated prefill serving.
                When > 1, the ONNX graph is read directly to generate a fully-populated MDP partition
                config (nodeList per partition) without requiring a compiler round-trip.
                Ignored when ``mdp_load_partition_config`` is already provided in compiler_options.
                Defaults to 1 (template / tensor-slice MDP, existing behaviour).
            :num_speculative_tokens (int | List[int], optional): Number of speculative tokens for TLM decode. A plain int K compiles one decode specialization (seq_len=K+1). A list [K0, K1, ...] compiles one specialization per value, enabling per-step dispatch to the cheapest kernel.
            :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False.``
            :qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file. ``Defaults to None.``
            :artifacts (bool): Export the model and write compiler inputs and a replay script without invoking the compiler. ``Defaults to False.``
            :compiler_options: Pass any compiler option as input.
                Any flag that is supported by `qaic-compile` can be passed. Params are converted to flags as below:

                - aic_num_cores=16 -> -aic-num-cores=16
                - convert_to_fp16=True -> -convert-to-fp16
                - aic_hw_version=ai100 -> -aic-hw-version=ai100
                - aic_hw_version=ai200 -> -aic-hw-version=ai200

                For QNN Compilation path, when enable_qnn is set to True, any parameter passed in compiler_options will be ignored.
        """
        layerwise_cache_probe = compiler_options.pop("_layerwise_cache_probe", False)

        for removed_option in ("compile_only", "compile-only"):
            if removed_option in compiler_options:
                logger.warning(f"'{removed_option}' is deprecated and is ignored; removing it from compiler options.")
                compiler_options.pop(removed_option, None)

        mdp_ts_json_path = compiler_options.pop("mdp_load_partition_config", None)
        mdp_strategy = MdpStrategy(compiler_options.pop("mdp_strategy", MdpStrategy.ONNX))
        mdp_compiler_dump_path = compiler_options.pop("mdp_compiler_dump_path", None)
        if mdp_compiler_dump_path is not None:
            logger.warning(
                "mdp_compiler_dump_path is deprecated and no longer used. "
                "QEfficient now generates the compiler dump automatically when mdp_strategy='intersection'; "
                "ignoring the provided value."
            )
            mdp_compiler_dump_path = None

        if onnx_path is None:
            # If weights were offloaded after export, compiling must use the existing
            # ONNX because re-exporting is no longer possible. Otherwise export for
            # the current compile mode, e.g. decode vs. disaggregated prefill.
            weights_offloaded = self._is_weights_offloaded or any(param.is_meta for param in self.model.parameters())
            if self.onnx_path is not None and weights_offloaded and not self._weight_free:
                onnx_path = self.onnx_path
            else:
                onnx_path = self.get_onnx_path(
                    prefill_only,
                    enable_chunking,
                    specializations,
                    offload_pt_weights,
                    use_onnx_subfunctions,
                    dynamo,
                    retain_full_kv,
                    mdp_ts_num_devices=mdp_ts_num_devices,
                    mdp_num_partitions=mdp_num_partitions,
                    qaic_config=qaic_config,
                    _layerwise_cache_probe=layerwise_cache_probe,
                    kv_cache_prefix=kv_cache_prefix,
                    **compiler_options,
                )
        if QEFFBaseModel._layerwise_active:
            if onnx_path is None:
                return None
            onnx_path = Path(onnx_path)
            return onnx_path
        onnx_path = Path(onnx_path)
        if artifacts:
            self.onnx_path = onnx_path
        existing_weight_spec_path = getattr(self, "weight_spec_path", None)
        weight_spec_path = (
            Path(existing_weight_spec_path) if existing_weight_spec_path else resolve_weight_spec_path(onnx_path)
        )
        weight_free_inputs = _weight_free_spec_input_names(weight_spec_path)
        compiler_env = _compiler_env_with_external_data_root(weight_spec_path) if weight_free_inputs else None

        compile_dir = Path(compile_dir or onnx_path.parent)
        qpc_path = compile_dir / "qpc"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

        if enable_qnn:
            if artifacts:
                raise NotImplementedError("`artifacts` is not supported by the QNN compilation path.")
            if compiler_options:
                logger.warning(
                    f"Extra arguments to QNN compilation are supported only via qnn_config file. Ignoring {compiler_options}"
                )

            self.qpc_path = qnn_compile(
                onnx_path=onnx_path,
                qpc_base_path=compile_dir,
                specializations=specializations,
                custom_io=custom_io,
                device_group=list(range(mdp_ts_num_devices)),
                num_cores=compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES),
                mxfp6=compiler_options.get("mxfp6_matmul", constants.DEFAULT_AIC_MXPF6_MATMUL),
                mxint8=mxint8_kv_cache,
                qnn_config=qnn_config,
            )

            return self.qpc_path

        command = (
            constants.COMPILER
            + [
                f"-aic-hw-version={compiler_options.pop('aic_hw_version', compiler_options.pop('aic-hw-version', constants.DEFAULT_AIC_HW_VERSION))}"
            ]
            + [f"-m={onnx_path}"]
        )

        for key, value in compiler_options.items():
            option = "-" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.append(f"{option}={value}")

        # Final custom-IO normalization against ONNX I/O names.
        # This only rewrites retained-state aliases:
        # *_InternalRetainedState <-> *_RetainedState.
        # Any other custom-IO key is preserved as-is for backward compatibility.
        if custom_io is not None and onnx_path is not None:
            try:
                model = onnx.load(onnx_path, load_external_data=False)
                io_names = {value.name for value in list(model.graph.input) + list(model.graph.output)}
                normalized_custom_io = {}
                for io_name, dtype in custom_io.items():
                    resolved_name = io_name
                    if io_name not in io_names:
                        if io_name.endswith("_InternalRetainedState"):
                            candidate = io_name[: -len("_InternalRetainedState")] + "_RetainedState"
                            if candidate in io_names:
                                resolved_name = candidate
                        elif io_name.endswith("_RetainedState"):
                            candidate = io_name[: -len("_RetainedState")] + "_InternalRetainedState"
                            if candidate in io_names:
                                resolved_name = candidate
                    normalized_custom_io[resolved_name] = dtype
                custom_io = normalized_custom_io
            except Exception:
                pass

        if use_onnx_subfunctions:
            logger.info("Using ONNX subfunctions for compilation.")
            command.append("-sub-functions")

        model_in_bfloat16 = hasattr(self, "config") and (self.config.torch_dtype == torch.bfloat16)
        io_name_prefix = ("past_", "pixel_values", "conv_", "recurrent_", "compressed_kv")
        pkv_in_bfloat16 = (custom_io is not None) and any(
            any(bfloat16_io_name in key for bfloat16_io_name in io_name_prefix) and "bfloat16" in value
            for key, value in custom_io.items()
        )
        custom_io_for_compiler = custom_io if not (model_in_bfloat16 and pkv_in_bfloat16) else None

        # MDP partition config selection (highest priority first):
        #   1. User-provided pre-built MDP JSON (mdp_load_partition_config).
        #   2. Disaggregated (pipeline-parallel) MDP — generated from ONNX topsort.
        #      Strategy ONNX (default): full superset from ONNX graph (~19 MB).
        #      Strategy INTERSECTION: intersect with compiler dump; compact (~1-2 MB),
        #        generating the temporary compiler dump automatically.
        #   3. Template (tensor-slice) MDP — single partition, nodeList absent.
        mdp_ts_json = None

        if mdp_ts_json_path:
            command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")
            mdp_ts_json = load_json(str(mdp_ts_json_path))
        elif mdp_num_partitions > 1:
            # Disaggregated (pipeline-parallel) MDP — delegate to focused helper.
            compile_dir.mkdir(parents=True, exist_ok=True)
            num_cores = compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES)
            num_layers = getattr(self, "num_layers", None)
            if getattr(self, "model", None) and getattr(self.model, "language_model", None) and not num_layers:
                num_layers = getattr(self.model.language_model.config, "num_hidden_layers", None)
            if num_layers is None:
                raise AttributeError(
                    "Model or Language Model does not expose 'num_layers' or 'num_hidden_layers' respectively. Cannot generate disagg MDP partition config."
                )
            if mdp_strategy is MdpStrategy.INTERSECTION and mdp_compiler_dump_path is None:
                mdp_compiler_dump_path = generate_mdp_compiler_dump(
                    compile_dir=compile_dir,
                    mdp_ts_num_devices=mdp_ts_num_devices,
                    mdp_num_partitions=mdp_num_partitions,
                    compile_command=command,
                    specializations=specializations,
                    specialization_module_name=specialization_module_name,
                    custom_io=custom_io_for_compiler,
                    compiler_env=compiler_env,
                )
            mdp_config_dir = (
                Path(mdp_compiler_dump_path).parent if mdp_strategy is MdpStrategy.INTERSECTION else compile_dir
            )
            mdp_ts_json_path, mdp_ts_json = generate_disagg_mdp_config(
                onnx_path=onnx_path,
                compile_dir=mdp_config_dir,
                mdp_ts_num_devices=mdp_ts_num_devices,
                mdp_num_partitions=mdp_num_partitions,
                mdp_strategy=mdp_strategy,
                mdp_compiler_dump_path=mdp_compiler_dump_path,
                num_cores=num_cores,
                num_layers=num_layers,
            )
            command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")
        elif mdp_ts_num_devices > 1 and not compiler_options.get("mdp_dump_partition_config", None):
            # Template (tensor-slice) MDP: single partition, empty nodeList; compiler fills it.
            # File write and command flag are deferred to after compile_dir is finalised (post-hash).
            mdp_ts_json = generate_mdp_partition_config(
                mdp_ts_num_devices, compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES)
            )

        compile_hash_params = {
            "command": command,
            "specializations": specializations,
            "custom_io": custom_io,
            "mdp_ts_num_devices": mdp_ts_num_devices,
            "mdp_num_partitions": mdp_num_partitions,
            "mdp_strategy": mdp_strategy.value,
            "mdp_ts_json": mdp_ts_json,
            "num_speculative_tokens": num_speculative_tokens,
            "prefill_only": prefill_only,
        }
        compile_hash = hash_dict_params(compile_hash_params)

        compile_dir = qpc_path.with_name(qpc_path.name + "-" + compile_hash)
        qpc_path = compile_dir / "qpc"
        if not artifacts:
            if (qpc_path / "programqpc.bin").is_file():
                self.qpc_path = qpc_path
                self.compile_artifacts_path = compile_dir

                # If export() already ran, its cache-hit message is the single
                # ONNX skip record for this run. Otherwise compilation returned
                # directly from a cached QPC and must report that export was
                # skipped as part of the QPC cache path.
                if not getattr(self, "_export_cache_hit", False):
                    QEFFLogger.log_event(
                        "milestone",
                        "INFRA",
                        "ONNX export skipped (cached QPC).",
                        milestone=QEFFLogger.MILESTONE_EXPORT_SKIPPED,
                    )
                log_api_arguments(
                    "compile",
                    self.__class__.__name__,
                    {"compile_hash_params": compile_hash_params},
                )
                QEFFLogger.log_event(
                    "milestone",
                    self.__class__.__name__,
                    "Compilation skipped (cached QPC).",
                    api="compile",
                    milestone=QEFFLogger.MILESTONE_COMPILE_SKIPPED,
                )
                logger.info(f"QPC path: {qpc_path}")
                return qpc_path

            log_api_arguments(
                "compile",
                self.__class__.__name__,
                {"compile_hash_params": compile_hash_params},
            )
            if qpc_path.is_dir():
                # Probably compilation failure last time, delete directory to start over.
                shutil.rmtree(qpc_path)
        compile_dir.mkdir(parents=True, exist_ok=True)

        # Write tensor-slice MDP partition config now that compile_dir exists.
        if mdp_ts_json is not None and mdp_ts_json_path is None:
            mdp_ts_json_path = compile_dir / f"mdp_ts_{mdp_ts_num_devices}.json"
            create_json(str(mdp_ts_json_path), mdp_ts_json)
            command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")

        # Write specializations.json file
        if specializations is not None:
            specializations_json = compile_dir / "specializations.json"
            specializations_data = {
                "specializations": to_named_specializations(specializations, module_name=specialization_module_name)
            }
            create_json(str(specializations_json), specializations_data)
            command.append(f"-network-specialization-config={specializations_json}")

        # Write custom_io.yaml file
        if custom_io is not None:
            custom_io_yaml = compile_dir / "custom_io.yaml"
            with open(custom_io_yaml, "w") as fp:
                for io_name, dtype in custom_io.items():
                    fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")
            if model_in_bfloat16 and pkv_in_bfloat16:
                logger.warning(
                    "Model and Past KV types are both bfloat16. Custom IO list file will be ignored during compile."
                )
            else:
                command.append(f"-custom-IO-list-file={custom_io_yaml}")

        command.append(f"-aic-binary-dir={qpc_path}")
        if artifacts:
            logger.info(f"Writing compiler replay command: {' '.join(command)}")
        else:
            logger.info(f"Running compiler: {' '.join(command)}")

        if artifacts:
            _copy_model_compiler_input(command, compile_dir, self.weight_spec_path)
            _copy_existing_compiler_input(command, "-node-precision-info", compile_dir)
            _copy_existing_compiler_input(command, "-mdp-load-partition-config", compile_dir)
            external_data_root = Path(compiler_env["AIC_EXTERNAL_DATA_ROOT"]) if compiler_env is not None else None
            path_flags = {
                "-aic-binary-dir",
                "-custom-IO-list-file",
                "-mdp-dump-partition-config",
                "-mdp-load-partition-config",
                "-m",
                "-network-specialization-config",
                "-node-precision-info",
                "-ols-config",
            }
            replay_command = [Path(command[0]).name]
            replay_base_dir = compile_dir.resolve()
            for argument in command[1:]:
                flag, separator, value = argument.partition("=")
                if separator and flag in path_flags:
                    argument = f"{flag}={os.path.relpath(Path(value).resolve(), replay_base_dir)}"
                replay_command.append(argument)
            compile_command = shlex.join(replay_command)
            script_lines = (
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'cd -- "$(dirname -- "$0")"',
                *(
                    (
                        'if [[ -z "${AIC_EXTERNAL_DATA_ROOT:-}" ]]; then',
                        f"  export AIC_EXTERNAL_DATA_ROOT={shlex.quote(str(external_data_root))}",
                        "else",
                        "  export AIC_EXTERNAL_DATA_ROOT",
                        "fi",
                    )
                    if external_data_root is not None
                    else ()
                ),
                compile_command,
                "",
            )
            script_path = compile_dir / "qaic-compile.sh"
            script_path.write_text("\n".join(script_lines))
            script_path.chmod(0o755)
            create_json(str(compile_dir / "hashed_compile_params.json"), compile_hash_params)
            self.qpc_path = None
            self.compile_artifacts_path = compile_dir
            logger.info(f"Compiler artifacts written to {compile_dir}")
            return compile_dir

        try:
            subprocess.run(command, capture_output=True, check=True, env=compiler_env)
        except subprocess.CalledProcessError as e:
            QEFFLogger.log_api_failure("compile", self.__class__.__name__, e)
            raise RuntimeError(
                "\n".join(
                    [
                        "Compilation failed!",
                        f"Compiler command: {e.cmd}",
                        f"Compiler exitcode: {e.returncode}",
                        "Compiler stderr:",
                        e.stderr.decode(),
                    ]
                )
            )
        # Dump JSON file with hashed parameters
        hashed_compile_params_path = compile_dir / "hashed_compile_params.json"
        create_json(hashed_compile_params_path, compile_hash_params)
        logger.info("Hashed parameters exported successfully.")

        self.qpc_path = qpc_path
        QEFFLogger.log_event(
            "milestone",
            self.__class__.__name__,
            "Compilation completed.",
            api="compile",
            milestone=QEFFLogger.MILESTONE_COMPILE_COMPLETE,
        )
        logger.info(f"QPC path: {qpc_path}")
        return qpc_path
