# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import math
import os
import warnings
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Union

import numpy as np
import onnx
import torch
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextStreamer,
)

import QEfficient
from QEfficient.base.modeling_qeff import (
    QEFFBaseModel,
    reject_legacy_moe_prefill_packed_chunk_size,
)
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.exporter.weight_free.checkpoint_transforms import (
    DtypeConversionCheckpointTransform,
    ExpertParallelPackingCheckpointTransform,
    GptOssMxfp4ExpertDequantSplitCheckpointTransform,
    MoEExpertStackingCheckpointTransform,
    MoEFusedExpertSplitCheckpointTransform,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession, is_retained_state_name
from QEfficient.generation.runner_io import (
    write_causal_lm_runner_bundle,
    write_dual_qpc_vlm_runner_bundle,
    write_single_qpc_vlm_runner_bundle,
)
from QEfficient.generation.text_generation_inference import (
    CloudAI100ExecInfoNew,
    PerfMetrics,
    calculate_latency,
    get_compilation_dims,
)
from QEfficient.generation.vlm_generation import VisionLanguageGeneration
from QEfficient.proxy.modeling_utils import (
    apply_proxy_layer_config,
    prepare_proxy_config,
)
from QEfficient.transformers.modeling_utils import (
    DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH,
    SPECIALIZED_DISAGG_SERVING_MODEL_ARCH,
    _configure_proxy_for_model,
)
from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import override_gptoss_prefill_chunking
from QEfficient.transformers.models.pytorch_transforms import (
    CustomOpsTransform,
    DFlashDLMTransform,
    DFlashTLMTransform,
    DFlashTransform,
    KVCacheExternalModuleMapperTransform,
    KVCacheTransform,
    PoolingTransform,
    PrefillOnlyChunkedTransform,
    PrefillOnlyExternalModuleMapperTransform,
    PrefillOnlyTransform,
    RevertPrefillKeepAttentionTransform,
    RevertPrefillOnlyExternalModuleMapperTransform,
    RevertPrefillOnlyTransform,
    SamplerTransform,
    SimpleDecodeMoeTransform,
    SpDTransform,
    TextClassificationTransform,
    VlmKVOffloadTransform,
    VlmNoKVOffloadTransform,
)
from QEfficient.transformers.moe.flavours import MoEFlavour
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import (
    AwqToMatmulNbitsTransform,
    FP8BlockWiseDequantLinearToLinearTransform,
    FP8BlockWiseDequantQwen3VLMoeTextExpertsToQwen3VLMoeTextExpertsTransform,
    FP8DeQuantLinearToLinearTransform,
    GPTQToMatmulNbitsTransform,
    Mxfp4GptOssExpertDequantizeTransform,
    PackQuantizedInt4ToMatMulNBitsTransform,
)
from QEfficient.utils import (
    apply_kv_cache_prefix,
    constants,
    get_padding_shape_from_config,
    validate_kv_cache_prefix,
)
from QEfficient.utils.check_ccl_specializations import process_ccl_specializations
from QEfficient.utils.export_utils import export_from_compile
from QEfficient.utils.logging_utils import QEFFLogger, log_from_pretrained_call, log_generate_call
from QEfficient.utils.runtime_requirements import validate_dynamo_export_requirements
from QEfficient.utils.sampler_utils import get_sampling_inputs_and_outputs

logger = QEFFLogger.get_logger("MODEL")
CUSTOM_IO_DTYPE_MAP = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float16",  # Since compiler doesn't support fp32
    "float32": "float16",  # Since compiler doesn't support fp32
}


def _should_convert_to_fp16(target_dtype: "torch.dtype", compiler_options: dict) -> bool:
    """Determine whether the compiler needs -convert-to-fp16.

    On ai100 hardware, bfloat16 is not natively supported so we must downcast
    to fp16 at compile time (same as fp32). On ai200, bfloat16 passes through
    natively and no conversion is needed.
    """
    if CUSTOM_IO_DTYPE_MAP[target_dtype] == "float16":
        return True
    if target_dtype == torch.bfloat16:
        aic_hw = compiler_options.get(
            "aic_hw_version", compiler_options.get("aic-hw-version", constants.DEFAULT_AIC_HW_VERSION)
        )
        return aic_hw != "ai200"
    return False


def _get_gdn_full_state_update_kwargs(
    model,
    qaic_config: dict | None,
    *,
    prefill_only: bool,
    batch_size: int = 1,
    full_batch_size: int = 1,
) -> dict:
    if not (qaic_config and qaic_config.get("gdn_full_state_update") is True):
        return {}

    get_kwargs = getattr(model, "get_gdn_full_state_update_kwargs", None)
    if get_kwargs is None:
        raise ValueError("qaic_config['gdn_full_state_update'] is not supported for this model")

    return get_kwargs(
        qaic_config=qaic_config,
        prefill_only=prefill_only,
        batch_size=batch_size,
        full_batch_size=full_batch_size,
    )


TORCH_TO_NUMPY_DTYPE_MAP = {
    torch.float16: np.float16,
    torch.bfloat16: np.float16,  # Since numpy doesn't support bfloat16
    torch.float32: np.float32,
}


def _disable_unsupported_weight_free(kwargs: dict, qeff_auto_class_name: str) -> None:
    """Remove unsupported weight-free mode from non-CausalLM wrappers."""

    if not kwargs.pop("weight_free", False):
        return

    logger.warning(
        "weight_free=True is only supported for QEFFAutoModelForCausalLM, QEffAutoModel; disabling it for %s.",
        qeff_auto_class_name,
    )


def _resolve_torch_dtype(kwargs: dict) -> None:
    """
    Resolve torch_dtype in kwargs before calling from_pretrained.

    Rules
    -----
    * If the caller already set torch_dtype to something other than
      bfloat16 (e.g. float16 or float32), leave it untouched.
    * If torch_dtype is bfloat16 **and** the target HW is ai100
      (the default), leave it as bfloat16 so export/compile still run in
      bfloat16, but warn that on-device generation is expected to fail
      because the ai100 runtime does not support bfloat16.
    * If torch_dtype is bfloat16 and the target HW is ai200,
      leave it as-is (ai200 supports bfloat16).
    * If torch_dtype is not set at all and the target HW is ai100 (the
      default), default to float32 so that models whose config.json
      declares bfloat16 are not silently loaded in bfloat16.

    Transformers v5 renamed the ``torch_dtype`` argument to ``dtype``. To keep
    backward compatibility for callers (and examples) that pass either name,
    a caller-supplied ``dtype`` is folded into ``torch_dtype`` here and the two
    are kept in sync on the way out so the loaded model and its config agree.
    """
    aic_hw_version = constants.DEFAULT_AIC_HW_VERSION
    # Normalize the transformers-v5 ``dtype`` alias onto ``torch_dtype`` so a
    # single code path governs the HW dtype policy below.
    if kwargs.get("torch_dtype", None) is None and kwargs.get("dtype", None) is not None:
        kwargs["torch_dtype"] = kwargs["dtype"]
    current_dtype = kwargs.get("torch_dtype", None)

    if aic_hw_version != "ai200":
        if current_dtype is None:
            kwargs["torch_dtype"] = torch.float32
        elif current_dtype == torch.bfloat16:
            logger.warning(
                "torch_dtype=bfloat16 is not supported on %s. Export and compilation will proceed in "
                "bfloat16, but on-device generation is expected to fail.",
                aic_hw_version,
            )

    # Keep the v5 alias in sync so HF from_pretrained and config see one dtype.
    if "dtype" in kwargs:
        kwargs["dtype"] = kwargs["torch_dtype"]


def _ignore_public_mdp_ts_num_devices(compiler_options: dict) -> None:
    if "mdp_ts_num_devices" not in compiler_options:
        return

    compiler_options.pop("mdp_ts_num_devices", None)
    logger.warning(
        "`mdp_ts_num_devices` passed to compile() is ignored. "
        "Tensor-slice devices per MDP partition are calculated internally as "
        "`num_devices // mdp_num_partitions`; "
        "pass `num_devices` and `mdp_num_partitions` instead."
    )


def _build_layerwise_vision_export_model(hf_auto_class, pretrained_model_name_or_path, kwargs):
    """Load a VLM with vision weights and only the first language window.

    This opt-in path keeps vision export usable while avoiding materializing
    every decoder layer up front. Language ONNX/QPC export still goes through
    the regular layerwise driver, which reloads each window independently.
    """
    from QEfficient.transformers.models import _layerwise

    config = kwargs.get("config", None)
    if config is None:
        from transformers import AutoConfig

        config_kwargs = {
            key: kwargs[key]
            for key in ("trust_remote_code", "revision", "token", "subfolder", "cache_dir")
            if key in kwargs
        }
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        kwargs["config"] = config
    model_type = _layerwise.assert_layerwise_supported(config)
    total_layers = _layerwise._resolve_text_total_layers(config)
    _layerwise._ensure_pretrained_window_attrs()
    _layerwise._install_window_patches_for(model_type)

    with _layerwise._layerwise_export_env():
        _layerwise._set_layer_windows(0, min(1, total_layers), total_layers)
        try:
            return hf_auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        finally:
            _layerwise._reset_layer_windows()


def _build_meta_model(hf_auto_class, pretrained_model_name_or_path, kwargs):
    """Construct an HF model on the meta device for layer-wise mode.

    Avoids materializing checkpoint weights at the outer ``from_pretrained``
    call site. The wrapper still has a fully-typed ``nn.Module`` (so config,
    architectures, and module structure are all real), but every parameter
    and buffer is a meta tensor — zero RAM. The layer-wise driver later
    rebuilds a real per-window model when ``compile()``/``export()`` runs.

    Uses accelerate's ``init_empty_weights()`` rather than a raw
    ``torch.device("meta")`` context: some models (e.g. Gemma's embed_scale)
    compute non-persistent buffers from a plain Python constant inside
    ``__init__`` via ``register_buffer(..., persistent=False)``. A bare
    ``torch.device("meta")`` context turns even those constant-valued buffers
    into empty meta tensors with no data, which later fails to serialize to
    ONNX. ``init_empty_weights()`` only intercepts parameter/persistent-buffer
    allocation, so such computed buffers still get their real values.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    config = kwargs.get("config", None)
    if config is None:
        config_kwargs = {
            k: kwargs[k] for k in ("trust_remote_code", "revision", "token", "subfolder", "cache_dir") if k in kwargs
        }
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
    torch_dtype = kwargs.get("torch_dtype", torch.float32)
    attn_implementation = kwargs.get("attn_implementation", "eager")
    # from_config's torch_dtype kwarg only governs newly-created parameters; internal
    # buffers computed during __init__ (e.g. rotary sin/cos caches) are derived from
    # config.dtype/torch_dtype directly, so those must be updated too or they end up
    # in the checkpoint's original dtype while parameters use torch_dtype instead.
    config.dtype = torch_dtype
    config.torch_dtype = torch_dtype
    with init_empty_weights():
        model = hf_auto_class.from_config(config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    return model


def _compile_io_name(name: str, *, use_onnx_subfunctions: bool) -> str:
    """Return the compiler-visible name for retained-state ONNX outputs."""
    if not use_onnx_subfunctions or not name.endswith("_RetainedState"):
        return name
    if any(token in name for token in ("key", "value", "conv_state", "recurrent_state", "compressed_kv", "k_pe")):
        return name[: -len("_RetainedState")] + "_InternalRetainedState"
    return name


def _state_input_name(output_name: str) -> str:
    """Map a retained-state output name to its matching state input name."""
    for suffix in ("_InternalRetainedState", "_RetainedState"):
        if output_name.endswith(suffix):
            state_name = output_name[: -len(suffix)]
            break
    else:
        state_name = output_name
    return state_name


def _add_retained_state_custom_io(
    custom_io: dict,
    output_name: str,
    *,
    dtype: str,
    use_onnx_subfunctions: bool,
) -> None:
    """Add the paired state-input and retained-state output custom-IO entries."""
    compiler_output_name = _compile_io_name(output_name, use_onnx_subfunctions=use_onnx_subfunctions)
    custom_io[_state_input_name(compiler_output_name)] = dtype
    custom_io[compiler_output_name] = dtype


def _filter_custom_io_for_onnx(custom_io: dict, onnx_path: str | Path | None) -> dict:
    """Keep custom-IO entries that exist in the ONNX graph.

    Layerwise stitched graphs may prefix I/O names (for example ``layer_0/``)
    and may expose public retained-state names even when subfunction export used
    internal names during per-window export. Matching by basename keeps compiler
    custom-IO compatible with both graph shapes.
    """
    if onnx_path is None:
        return custom_io
    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception:
        return custom_io

    io_names = {value.name for value in list(model.graph.input) + list(model.graph.output)}
    basename_to_name = {name.rsplit("/", 1)[-1]: name for name in io_names}

    def resolve_name(name: str) -> str | None:
        candidates = [name]
        if name.endswith("_InternalRetainedState"):
            candidates.append(name[: -len("_InternalRetainedState")] + "_RetainedState")
        elif name.endswith("_RetainedState"):
            candidates.append(name[: -len("_RetainedState")] + "_InternalRetainedState")
        for candidate in candidates:
            if candidate in io_names:
                return candidate
            if candidate in basename_to_name:
                return basename_to_name[candidate]
        return None

    filtered = {}
    for name, dtype in custom_io.items():
        resolved_name = resolve_name(name)
        if resolved_name is not None:
            filtered[resolved_name] = dtype
    return filtered


class QEFFTransformersBase(QEFFBaseModel):
    """
    Base class for QEfficient wrappers around HuggingFace transformer models.

    This class provides common functionality for loading, representing, and managing
    HuggingFace models within the QEfficient framework. It serves as a parent
    for specific model types like `AutoModel`, `AutoModelForCausalLM`, etc.
    """

    _hf_auto_class: type

    def __init__(self, model: nn.Module, **kwargs) -> None:
        _configure_proxy_for_model(self, kwargs.pop("enable_proxy", False))
        if self.__class__ is not QEFFAutoModel:
            _disable_unsupported_weight_free(kwargs, self.__class__.__name__)

        if (
            hasattr(model, "config")
            and hasattr(model.config, "quantization_config")
            and not isinstance(model.config.quantization_config, tuple(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.values()))
        ):
            raise AssertionError("Please use `from_pretrained` method to load quantized models")

        super().__init__(model, **kwargs)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__()

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Load a QEfficient transformer model from a pretrained HuggingFace model or local path.

        This is the recommended way to initialize any QEfficient transformer model.
        The interface is similar to ``transformers.AutoModel.from_pretrained``.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        QEFFTransformersBase
            An instance of the specific QEFFAutoModel subclass, initialized with the pretrained weights.
        """
        _disable_unsupported_weight_free(kwargs, cls.__name__)
        enable_proxy = kwargs.pop("enable_proxy", False)

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        _resolve_torch_dtype(kwargs)
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        kwargs.update({"enable_proxy": enable_proxy} if enable_proxy else {})

        return cls(model, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)


class MultimodalUtilityMixin:
    """
    Mixin for multimodal models providing utilities like input auto-correction.

    This mixin ensures that inputs to multimodal models conform to the expected
    names, shapes, and dtypes defined by the model's `get_inputs_info` method.
    """

    def __new__(cls, *args, **kwargs):
        if cls is MultimodalUtilityMixin:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)

    def auto_correct_inputs(self, inputs):
        """
        Validates and corrects model inputs to match expected specifications.

        Checks if the provided inputs dictionary contains all required keys and
        if the data types of the tensors match the model's specifications.
        It then filters the input dictionary to only include expected inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            A dictionary of input tensors, where keys are input names and values are `torch.Tensor` objects.

        Returns
        -------
        Dict[str, torch.Tensor]
            A filtered dictionary of input tensors that match the model's expected inputs.

        Raises
        ------
        RuntimeError
            If any expected input is missing or has a mismatched data type.
        """
        checked = True
        inputs_info = self.model.get_inputs_info()
        for valid_input_info in inputs_info:
            if valid_input_info.name not in inputs:
                checked = False
                break
            if inputs[valid_input_info.name].dtype != valid_input_info.datatype:
                checked = False
                break

        if not checked:
            err_str: str = (
                "Expected following input names and shapes to be passed\n"
                + "\n".join([val.__repr__() for val in inputs_info])
                + "\ngot"
                + f"{[(k, v.shape, v.dtype) for k, v in inputs.items()]}"
            )

            raise RuntimeError(err_str)

        return {k: v for k, v in inputs.items() if k in [iinfo.name for iinfo in inputs_info]}


class QEFFAutoModel(QEFFTransformersBase):
    """
    QEfficient class for general transformer models from the HuggingFace hub (e.g., BERT, Sentence Transformers).

    This class provides a unified interface for loading, exporting, compiling, and running
    various encoder-only transformer models on Cloud AI 100 hardware. It supports pooling
    for embedding extraction.

    Example
    -------
    .. code-block:: python

        from QEfficient import QEFFAutoModel
        from transformers import AutoTokenizer

        model = QEFFAutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", pooling="mean")
        model.compile(num_cores=16)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        inputs = tokenizer("My name is", return_tensors="pt")
        output = model.generate(inputs)
        print(output) # Output will be a dictionary containing extracted features.
    """

    _hf_auto_class = AutoModel
    _pytorch_transforms = [CustomOpsTransform, AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform]
    # FP16Clip inlines external weights; without Split the saved protobuf exceeds 2GB for large embedders.
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]
    _checkpoint_transforms = [DtypeConversionCheckpointTransform]

    def __init__(self, model: nn.Module, pooling=None, **kwargs):
        """
        Initializes a QEFFAutoModel instance.

        Parameters
        ----------
        model : nn.Module
            The underlying HuggingFace PyTorch model.
        pooling : str or Callable, optional
            The pooling method to use for feature extraction.
            Options include: "mean", "max", "cls", "avg", or a custom Callable.
            Default is None (no pooling applied).
        **kwargs :
            Additional keyword arguments passed to the base class constructor.
        """
        super().__init__(model, **kwargs)

        # Make Embedding specific transforms like appending pooling
        if pooling:
            self.model, _ = PoolingTransform.apply(self.model, pooling)

        # Encoder-only models (e.g. BERT) should not be forced into cache mode.
        if getattr(self.model.config, "is_decoder", False) or getattr(self.model.config, "is_encoder_decoder", False):
            self.model.base_model.config.use_cache = True
        else:
            object.__setattr__(self.model.base_model.config, "use_cache", None)

        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path, pooling=None, weight_free=False, *args, **kwargs):
        """
        Load a QEfficient transformer model from a pretrained HuggingFace model or local path.

        This is the recommended way to initialize a QEfficient transformer model. The interface is similar to
        ``transformers.AutoModel.from_pretrained``. Once initialized, you can use methods such as ``export``, ``compile``, and ``generate``.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        pooling : str or Callable, optional
            The pooling method to use. Options include:
            - "mean": Mean pooling
            - "max": Max pooling
            - "cls": CLS token pooling
            - "avg": Average pooling
            - Callable: A custom pooling function
            - None: No pooling applied. Default is None.
        weight_free : bool, optional
            If True, the model will be loaded in weight-free mode, which avoids materializing checkpoint weights. Default is False.
        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        QEFFAutoModel
            An instance initialized with the pretrained weights.
        """
        enable_proxy = kwargs.pop("enable_proxy", False)

        if weight_free:
            validate_dynamo_export_requirements("weight_free=True")

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        _resolve_torch_dtype(kwargs)
        if weight_free:
            # Weight-free mode: build the model on the meta device so no
            # checkpoint weights are ever materialized here. The real weights
            # are supplied later at export time via pretrained_model_name_or_path.
            model = _build_meta_model(cls._hf_auto_class, pretrained_model_name_or_path, kwargs)
        else:
            model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # This is support models that should be classified to in a different auto class but transformers load them via this class
        kv_offload = kwargs.pop("kv_offload", None)

        kwargs.update({"enable_proxy": enable_proxy} if enable_proxy else {})

        if model.__class__.__name__ in MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP:
            return MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP[model.__class__.__name__](
                model, kv_offload=kv_offload, **kwargs
            )

        return cls(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pooling=pooling,
            weight_free=weight_free,
            **kwargs,
        )

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def export(self, export_dir: str | None = None, dynamo: bool = False, **kwargs) -> str:
        """
        Export the model to ONNX format using ``torch.onnx.export``.

        This method prepares example inputs and dynamic axes based on the model configuration,
        then exports the model to an ONNX graph suitable for compilation and deployment on Cloud AI 100 hardware.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. If not provided,
            the default export directory is used.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file.
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN

        dynamo = kwargs.get("dynamo", False) or self._weight_free
        if dynamo:
            # torch.export requires example inputs to satisfy dynamic_shapes min=2; gpt_oss non-CB keeps bs=1.
            bs = max(2, bs)

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "attention_mask": torch.ones((bs, seq_len), dtype=torch.int64),
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}}

        output_names = ["output"]

        return self._export(
            example_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            dynamo=dynamo,
            use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
        )

    def compile(
        self,
        onnx_path: str | None = None,
        compile_dir: str | None = None,
        *,
        seq_len: int | list[int] = 32,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compile the exported ONNX model using the Cloud AI 100 Platform SDK compiler.

        This method generates a ``qpc`` package. If the model has not been exported yet,
        this method will handle the export process. Additional arguments for the `qaic-compile`
        compiler can be passed as keyword arguments.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        compile_dir : str, optional
            Directory to save the generated QPC package. If not provided, a default directory is used.
        seq_len : int or list of int, optional
            The length(s) of the prompt(s) to compile for. Can be a single integer or a list of integers
            to create multiple specializations. Default is 32.
        batch_size : int, optional
            Batch size. Default is 1.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers. These are passed directly
            to the underlying compilation command.

            **For QAIC Compiler:** Extra arguments for qaic-compile can be passed. Some common options include:

            - mos (int, optional): Effort level to reduce on-chip memory. Defaults to -1, meaning no effort. Defaults to -1.
            - aic_enable_depth_first (bool, optional): Enables DFS with default memory size. Defaults to False.
            - allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. Defaults to False.

            Params are converted to flags as below:

            - ``aic_num_cores=16`` -> ``-aic-num-cores=16``
            - ``convert_to_fp16=True`` -> ``-convert-to-fp16``

            **For QNN Compiler:** Following arguments can be passed as:

            - enable_qnn (bool): Enables QNN Compilation.
            - qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file.

        Returns
        -------
        str
            Path to the compiled QPC package.

        """

        if isinstance(seq_len, list) and len(seq_len) >= 15:
            warnings.warn("Recommended: `seq_len` should contain fewer than 15 items.")

        _seq_lens = seq_len if isinstance(seq_len, list) else [seq_len]
        specializations = [
            {
                "_graph_name": "Embedding" if len(_seq_lens) == 1 else f"Embedding_{i}",
                "batch_size": batch_size,
                "seq_len": sl,
            }
            for i, sl in enumerate(_seq_lens)
        ]

        target_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        return self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            specializations=specializations,
            convert_to_fp16=_should_convert_to_fp16(target_dtype, compiler_options),
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    def generate(
        self,
        inputs: torch.Tensor,
        device_ids: list[int] | None = None,
        runtime_ai100: bool = True,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor | np.ndarray:
        """
        Generate output by executing the compiled QPC on Cloud AI 100 hardware or using PyTorch runtime.

        This method runs sequential execution based on the compiled model's batch size and the number of prompts.
        If the number of prompts is not divisible by the batch size, the last batch will be dropped.

        Parameters
        ----------
        inputs : torch.Tensor or np.ndarray
            Input data for the model. For AI 100 runtime, this typically includes
            `input_ids` and `attention_mask`.
        device_ids : list of int, optional
            Device IDs for running the QPC. Defaults to `[0]` if not specified and `runtime_ai100` is True.
        runtime_ai100 : bool, optional
            Whether to use the AI 100 runtime for inference. If False, the PyTorch
            runtime will be used. Default is True.

        Returns
        -------
        torch.Tensor or np.ndarray
            Output from the AI 100 or PyTorch runtime. The type depends on the runtime and model.
        """
        # AI_100 runtime
        if runtime_ai100:
            if not isinstance(self.qpc_path, Path):
                raise TypeError("Please run compile API first!")

            return self.cloud_ai_100_feature_generate(inputs=inputs, device_ids=device_ids)
        # PyTorch runtime
        else:
            return self.pytorch_feature_generate(model=self.model, inputs=inputs)

    def cloud_ai_100_feature_generate(
        self,
        inputs: torch.Tensor,
        device_ids: list[int] | None = None,
        dtype: torch.dtype | None = torch.float32,
    ) -> np.ndarray:
        """
        Generate features for a batch of inputs using the Cloud AI 100 hardware runtime.

        This method runs inference on the compiled QPC using the Cloud AI 100 accelerator.
        It automatically pads input tensors to match the compiled sequence length and handles session setup.

        Parameters
        ----------
        inputs : torch.Tensor or np.ndarray
            Input tensors for feature extraction. Must be a dictionary-like object
            including `input_ids` and `attention_mask`.
        device_ids : List[int], optional
            List of device IDs to use for inference. Defaults to runtime automatic device selection.

        Returns
        -------
        np.ndarray
            Array containing the generated output features for each input in the batch.
        """

        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]

        # Dynamic switching to closest seq_Len based on input_ids_len
        input_ids_len = inputs["input_ids"].shape[1]

        for allowed_shape in self.qpc_session.allowed_shapes:
            seq_len_allowed = allowed_shape[1][1][1]

            if seq_len_allowed >= input_ids_len:
                self.seq_len = seq_len_allowed
                break

        # To handle single seq_len as we can't fetch allowed shapes for single seq_len
        self.seq_len = self.qpc_session.bindings[0].dims[1] if not hasattr(self, "seq_len") else self.seq_len

        input_ids = np.array(
            torch.nn.functional.pad(inputs["input_ids"], (0, self.seq_len - input_ids_len), "constant", 0)
        )
        attention_mask = np.array(
            torch.nn.functional.pad(
                inputs["attention_mask"], (0, self.seq_len - inputs["attention_mask"].size(1)), "constant", 0
            )
        )

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # TODO: Remove try and catch after compiler fix
        try:
            outputs = {
                "output": np.random.randn(*list(self.qpc_session.bindings[2].dims)).astype(
                    TORCH_TO_NUMPY_DTYPE_MAP[dtype]
                ),
            }
            self.qpc_session.set_buffers(outputs)
            outputs = self.qpc_session.run(inputs)
        except Exception:
            outputs = {
                "output": np.random.randn(self.batch_size, self.seq_len, self.qpc_session.bindings[2].dims[1]).astype(
                    TORCH_TO_NUMPY_DTYPE_MAP[dtype]
                ),
            }
            self.qpc_session.set_buffers(outputs)
            outputs = self.qpc_session.run(inputs)

        return outputs

    def pytorch_feature_generate(self, model, inputs: torch.Tensor | np.ndarray) -> list[torch.Tensor]:
        """
        Generate features from a batch of inputs using the PyTorch model.

        This method runs the model in PyTorch (CPU/GPU) mode for feature extraction.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model to use for inference.
        inputs : torch.Tensor or np.ndarray
            Input tensors for feature extraction. Expected to be a dictionary-like object.

        Returns
        -------
        List[torch.Tensor]
            List of output features generated by the model for each input.
        """
        outputs = model(**inputs)

        return outputs


class QEFFAutoModelForSequenceClassification(QEFFTransformersBase):
    """
    QEfficient class for sequence classification models from the HuggingFace hub (e.g., BERT, DebertaV2 for classification).

    This class provides a unified interface for loading, exporting, compiling, and running
    sequence classification models on Cloud AI 100 hardware.

    Example
    -------
    .. code-block:: python

        from QEfficient import QEFFAutoModelForSequenceClassification
        from transformers import AutoTokenizer

        model = QEFFAutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-Prompt-Guard-2-22M")
        model.compile(num_cores=16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Prompt-Guard-2-22M")
        inputs = tokenizer("Ignore your previous instructions.", return_tensors="pt")
        output = model.generate(inputs)
        predicted_class_id = output["logits"].argmax().item()
        print(model.model.config.id2label[predicted_class_id])
    """

    _hf_auto_class = AutoModelForSequenceClassification
    _pytorch_transforms = [CustomOpsTransform, TextClassificationTransform]
    _onnx_transforms = []

    def __init__(self, model: nn.Module, **kwargs):
        """
        Initializes a QEFFAutoModelForSequenceClassification instance.

        Parameters
        ----------
        model : nn.Module
            The underlying HuggingFace PyTorch sequence classification model.
        **kwargs :
            Additional keyword arguments passed to the base class constructor.
        """
        super().__init__(model, **kwargs)
        self.model.config.use_cache = True
        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a QEfficient sequence classification model from a pretrained HuggingFace model or local path.

        This is the recommended way to initialize a QEfficient sequence classification model.
        The interface is similar to ``transformers.AutoModelForSequenceClassification.from_pretrained``.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        QEFFAutoModelForSequenceClassification
            An instance initialized with the pretrained weights.
        """
        _disable_unsupported_weight_free(kwargs, cls.__name__)
        enable_proxy = kwargs.pop("enable_proxy", False)

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        _resolve_torch_dtype(kwargs)
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        kwargs.update({"enable_proxy": enable_proxy} if enable_proxy else {})
        return cls(model, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def export(self, export_dir: str | None = None, **kwargs) -> str:
        """
        Export the model to ONNX format using ``torch.onnx.export``.

        This method prepares example inputs and dynamic axes based on the model configuration,
        then exports the model to an ONNX graph suitable for compilation and deployment on Cloud AI 100 hardware.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. If not provided,
            the default export directory is used.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file.
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "attention_mask": torch.ones((bs, seq_len), dtype=torch.int64),
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}}

        output_names = ["logits"]

        return self._export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
        )

    def compile(
        self,
        onnx_path: str | None = None,
        compile_dir: str | None = None,
        *,
        seq_len: int | list[int] = 32,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,
        mxfp6_matmul: bool = False,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compile the exported ONNX model using the Cloud AI 100 Platform SDK compiler.

        This method generates a ``qpc`` package. If the model has not been exported yet,
        this method will handle the export process.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        compile_dir : str, optional
            Directory to save the generated QPC package. If not provided, a default directory is used.
        seq_len : int or list of int, optional
            The length(s) of the input sequence(s) to compile for. Can be a single integer or a list of integers
            to create multiple specializations. Default is 32.
        batch_size : int, optional
            Batch size. Default is 1.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers.

        Returns
        -------
        str
            Path to the compiled QPC package.
        """
        _ignore_public_mdp_ts_num_devices(compiler_options)
        if isinstance(seq_len, list) and len(seq_len) >= 15:
            warnings.warn("Recommended: `seq_len` should contain fewer than 15 items.")

        _seq_lens = seq_len if isinstance(seq_len, list) else [seq_len]
        specializations = [
            {
                "_graph_name": "SeqClassification" if len(_seq_lens) == 1 else f"SeqClassification_{i}",
                "batch_size": batch_size,
                "seq_len": sl,
            }
            for i, sl in enumerate(_seq_lens)
        ]
        target_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        return self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            specializations=specializations,
            convert_to_fp16=_should_convert_to_fp16(target_dtype, compiler_options),
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    def generate(
        self,
        inputs: torch.Tensor,
        device_ids: list[int] | None = None,
    ) -> dict:
        """
        Generate classification output using the Cloud AI 100 hardware runtime.

        Parameters
        ----------
        inputs : torch.Tensor or np.ndarray
            Input tensors for classification. Must be a dictionary-like object
            including `input_ids` and `attention_mask`.
        device_ids : List[int], optional
            List of device IDs to use for inference. Defaults to runtime automatic device selection.

        Returns
        -------
        dict
            Dictionary containing the classification logits.
        """
        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]

        # Dynamic switching to closest seq_len based on input_ids_len
        input_ids_len = inputs["input_ids"].shape[1]

        for allowed_shape in self.qpc_session.allowed_shapes:
            seq_len_allowed = allowed_shape[1][1][1]
            if seq_len_allowed >= input_ids_len:
                self.seq_len = seq_len_allowed
                break

        # To handle single seq_len as we can't fetch allowed shapes for single seq_len
        self.seq_len = self.qpc_session.bindings[0].dims[1] if not hasattr(self, "seq_len") else self.seq_len

        input_ids = np.array(
            torch.nn.functional.pad(inputs["input_ids"], (0, self.seq_len - input_ids_len), "constant", 0)
        )
        attention_mask = np.array(
            torch.nn.functional.pad(
                inputs["attention_mask"], (0, self.seq_len - inputs["attention_mask"].size(1)), "constant", 0
            )
        )

        inputs_np = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.qpc_session.run(inputs_np)

        return {"logits": torch.from_numpy(outputs["logits"])}


class QEffVisionEncoderForTextImageToTextModel(QEFFBaseModel):
    """
    QEfficient wrapper for the Vision Encoder component of a Text-to-Image-to-Text model.

    This class handles the export and compilation of the vision encoder part
    of multimodal models for optimal performance on Cloud AI 100 hardware.
    """

    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        PackQuantizedInt4ToMatMulNBitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        KVCacheExternalModuleMapperTransform,
    ]
    _onnx_transforms = []

    def __init__(self, model: nn.modules, **kwargs):
        """
        Initializes the vision encoder component for multimodal models.

        Parameters
        ----------
        model : nn.Module
            The full HuggingFace multimodal model from which the vision encoder is extracted.
        **kwargs :
            Additional keyword arguments passed to the base class constructor.
        """
        _configure_proxy_for_model(self, kwargs.pop("enable_proxy", False))
        _disable_unsupported_weight_free(kwargs, self.__class__.__name__)
        super().__init__(model, **kwargs)
        self.model = model.get_qeff_vision_encoder()
        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    def export(self, inputs, output_names, dynamic_axes, export_dir=None, offload_pt_weights=True, **kwargs):
        """
        Exports the vision encoder component to ONNX format.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Example inputs for the ONNX export.
        output_names : List[str]
            List of output names for the ONNX graph.
        dynamic_axes : Dict[str, Dict[int, str]]
            Dynamic axes configuration for the ONNX graph.
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. Default is None.
        offload_pt_weights : bool, optional
            If True, PyTorch weights will be offloaded after export. Default is True.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file for the vision encoder.
        """
        return self._export(
            inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=offload_pt_weights,
            use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
        )

    def compile(
        self,
        compile_dir,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the vision encoder component to a QPC package.

        Parameters
        ----------
        compile_dir : str
            Directory to save the generated QPC package.
        specializations : List[Dict[str, Union[int, str]]]
            List of dictionaries, each specifying a compilation specialization.
        convert_to_fp16 : bool
            If True, converts model to FP16 precision during compilation.
        mxfp6_matmul : bool
            If True, uses MXFP6 compression for MatMul weights.
        mdp_ts_num_devices : int
            Number of devices for multi-device (tensor slicing) compilation.
        aic_num_cores : int
            Number of cores to use for compilation.
        custom_io : Dict[str, str]
            Custom I/O configurations for the compiler.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options :
            Additional compiler options passed to the underlying compilation command.

        Returns
        -------
        str
            Path to the compiled QPC package for the vision encoder.
        """
        return self._compile(
            compile_dir=compile_dir,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    @property
    def get_model_config(self) -> dict:
        """
        Get the configuration dictionary of the underlying HuggingFace vision model.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        if hasattr(self.model.model, "vision_model"):
            return self.model.model.vision_model.config.__dict__
        return self.model.model.config.__dict__


class QEffCausalLMForTextImageToTextModel(QEFFBaseModel):
    """
    QEfficient wrapper for the Causal Language Model (decoder) component of a Text-to-Image-to-Text model.

    This class handles the export and compilation of the language decoder part
    of multimodal models for optimal performance on Cloud AI 100 hardware.
    """

    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        PackQuantizedInt4ToMatMulNBitsTransform,
        FP8BlockWiseDequantQwen3VLMoeTextExpertsToQwen3VLMoeTextExpertsTransform,
        FP8BlockWiseDequantLinearToLinearTransform,
        CustomOpsTransform,
        KVCacheTransform,
        VlmKVOffloadTransform,
        SimpleDecodeMoeTransform,
    ]
    _onnx_transforms = []

    def __init__(self, model, qaic_config: dict | None = None, **kwargs):
        """
        Initializes the language decoder component for multimodal models.

        Parameters
        ----------
        model : nn.Module
            The full HuggingFace multimodal model from which the language decoder is extracted.
        qaic_config : dict, optional
            A dictionary for QAIC-specific configurations. Supported keys include:
            - **num_kv_blocks** (int): Number of K/V blocks for BlockedKV attention implementation.
        **kwargs :
            Additional keyword arguments passed to the base class constructor.
        """
        _configure_proxy_for_model(self, kwargs.pop("enable_proxy", False))
        _disable_unsupported_weight_free(kwargs, self.__class__.__name__)
        super().__init__(model, **kwargs)
        self.model = model.get_qeff_language_decoder()
        self.qaic_config = qaic_config
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.continuous_batching = False
        if qaic_config and (mla_absorption := qaic_config.get("mla_absorption", None)):
            self.hash_params["mla_absorption"] = mla_absorption
            self.model.language_model.mla_absorption = mla_absorption

    def __update_prefill_transform(
        self,
        enable: bool | None = True,
        enable_chunking: bool | None = False,
        retain_full_kv: bool | None = False,
    ):
        if enable:
            if enable_chunking:
                self.model, tf = PrefillOnlyChunkedTransform.apply(self.model)
            else:
                self.model, tf = PrefillOnlyTransform.apply(self.model)

        else:
            if retain_full_kv:
                self.model, tf = RevertPrefillKeepAttentionTransform.apply(self.model)
            else:
                self.model, _tf = RevertPrefillOnlyTransform.apply(self.model)

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        offload_pt_weights=True,
        prefill_seq_len: int | None = None,
        prefill_only: bool = False,
        enable_chunking: bool = False,
        kv_cache_prefix: str | None = None,
        **kwargs,
    ):
        """
        Exports the language decoder component to ONNX format.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Example inputs for the ONNX export.
        output_names : List[str]
            List of output names for the ONNX graph.
        dynamic_axes : Dict[str, Dict[int, str]]
            Dynamic axes configuration for the ONNX graph.
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. Default is None.
        offload_pt_weights : bool, optional
            If True, PyTorch weights will be offloaded after export. Default is True.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file for the language decoder.
        """
        reject_legacy_moe_prefill_packed_chunk_size(kwargs)
        if prefill_only:
            assert prefill_seq_len > 1
            if not enable_chunking and self.continuous_batching:
                raise NotImplementedError(
                    "Looks like you are trying to run prefix-caching without chunking, this feature is not available yet!"
                )
            self.hash_params["prefill_only"] = True
            self.__update_prefill_transform(enable=True, enable_chunking=enable_chunking)
        else:
            self.hash_params["prefill_only"] = False
            self.__update_prefill_transform(False, retain_full_kv=kwargs.get("retain_full_kv", False))

        qaic_config = kwargs.pop("qaic_config", getattr(self.model, "qaic_config", None))

        if QEfficient.base.modeling_qeff.QEFFBaseModel._layerwise_active:
            return self._export_layerwise(
                inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=export_dir,
                offload_pt_weights=offload_pt_weights,
                use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
                _layerwise_cache_probe=kwargs.get("_layerwise_cache_probe", False),
                kv_cache_prefix=kv_cache_prefix,
                prefill_only=prefill_only,
                enable_chunking=enable_chunking,
                num_cores=kwargs.get("num_cores", constants.DEFAULT_AIC_NUM_CORES),
                qaic_config=qaic_config,
                prefill_seq_len=prefill_seq_len,
            )
        else:
            return self._export(
                inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=export_dir,
                offload_pt_weights=offload_pt_weights,
                use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
            )

    def compile(
        self,
        compile_dir,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the language decoder component to a QPC package.

        Parameters
        ----------
        compile_dir : str
            Directory to save the generated QPC package.
        specializations : List[Dict[str, Union[int, str]]]
            List of dictionaries, each specifying a compilation specialization.
        convert_to_fp16 : bool
            If True, converts model to FP16 precision during compilation.
        mxfp6_matmul : bool
            If True, uses MXFP6 compression for MatMul weights.
        mdp_ts_num_devices : int
            Number of devices for multi-device (tensor slicing) compilation.
        aic_num_cores : int
            Number of cores to use for compilation.
        custom_io : Dict[str, str]
            Custom I/O configurations for the compiler.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options :
            Additional compiler options passed to the underlying compilation command.

        Returns
        -------
        str
            Path to the compiled QPC package for the language decoder.
        """
        return self._compile(
            compile_dir=compile_dir,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    @property
    def get_model_config(self) -> dict:
        """
        Get the configuration dictionary of the underlying HuggingFace language model.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        if hasattr(self.model, "language_model"):
            return self.model.language_model.config.__dict__
        return self.model.config.__dict__


class _QEffAutoModelForImageTextToTextDualQPC:
    """
    Internal class handling multimodal image-text-to-text models using a dual QPC approach.

    In this approach, the vision encoder and language model decoder are compiled
    into separate QPC packages. The vision encoder's KV cache might be offloaded
    to CPU or managed differently from the language model's KV cache.
    """

    _hf_auto_class = AutoModelForImageTextToText

    def __init__(
        self,
        model: nn.Module,
        continuous_batching: bool = False,
        qaic_config: dict | None = None,
        **kwargs,
    ):
        """
        Initializes the dual QPC multimodal model wrapper.

        Parameters
        ----------
        model : nn.Module
            The full HuggingFace multimodal model.
        qaic_config : dict, optional
            A dictionary for QAIC-specific configurations.
        **kwargs :
            Additional keyword arguments.
        """
        _disable_unsupported_weight_free(kwargs, self.__class__.__name__)
        if kwargs.pop("full_batch_size", None):
            continuous_batching = True
            warnings.warn(
                "full_batch_size argument is deprecated. Use continuous_batching=True instead.", DeprecationWarning, 2
            )
        self.model = model
        self.config = model.config
        self._pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None)

        self.vision_model = QEffVisionEncoderForTextImageToTextModel(model, **kwargs)
        self.lang_model = QEffCausalLMForTextImageToTextModel(model, qaic_config=qaic_config, **kwargs)
        self.continuous_batching = continuous_batching
        self.ccl_enabled = False
        if qaic_config:
            self.ccl_enabled = qaic_config.get("ccl_enabled", False)

        self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = None, None
        self.input_shapes, self.output_names = None, None

        # DFlash TLM: attach fc/hidden_norm and enable target-layer hidden-state collect.
        self.dflash_tlm = bool(qaic_config and qaic_config.get("target_layer_ids", None))
        if self.dflash_tlm:
            qaic_config.setdefault("pretrained_model_name_or_path", kwargs.get("pretrained_model_name_or_path"))
            self.model, _ = DFlashTLMTransform.apply(self.model, qaic_config)

        # ---Sampling---
        # Note: SamplerTransform should be applied after all other transforms
        # are done. The role of the sampler is to just add nodes at the output of the
        # previous transform function.
        self.lang_model.model.qaic_config = qaic_config
        self.lang_model.model, _ = SamplerTransform.apply(self.lang_model.model, qaic_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, qaic_config: dict | None = None, **kwargs):
        """
        Load a QEfficient multimodal model for dual QPC from a pretrained HuggingFace model or local path.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.
            Note: `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        _QEffAutoModelForImageTextToTextDualQPC
            An instance initialized with the pretrained weights.
        """
        _disable_unsupported_weight_free(kwargs, cls.__name__)
        enable_proxy = kwargs.pop("enable_proxy", False)

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        # Respect an explicit low_cpu_mem_usage=True from the caller (used by the
        # layer-wise driver to keep RAM bounded by one window's weights via meta
        # init + sharded materialization). For everyone else, force False to
        # match prior behavior.
        explicit_low_cpu = kwargs.get("low_cpu_mem_usage", None) is True
        kwargs.update(
            {
                "attn_implementation": "eager",
                "low_cpu_mem_usage": bool(explicit_low_cpu),
            }
        )

        _resolve_torch_dtype(kwargs)
        if enable_proxy:
            prepare_proxy_config(pretrained_model_name_or_path, kwargs)
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        kwargs.update({"enable_proxy": enable_proxy} if enable_proxy else {})

        return cls(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            qaic_config=qaic_config,
            **kwargs,
        )

    @property
    def onnx_path(self):
        """
        Get the ONNX paths for the vision and language model components.

        Returns
        -------
        List[str]
            A list containing the ONNX paths of the vision model and the language model.
        """
        return [self.vision_model.onnx_path, self.lang_model.onnx_path]

    def __update_prefill_transform(
        self,
        enable: bool | None = True,
        enable_chunking: bool | None = False,
        retain_full_kv: bool | None = False,
    ):
        if enable:
            self.model, tf = PrefillOnlyExternalModuleMapperTransform.apply(self.model)
            if enable_chunking:
                self.model, tf = PrefillOnlyChunkedTransform.apply(self.model)
            else:
                self.model, tf = PrefillOnlyTransform.apply(self.model)

        else:
            self.model, tf = RevertPrefillOnlyExternalModuleMapperTransform.apply(self.model)
            if retain_full_kv:
                self.model, tf = RevertPrefillKeepAttentionTransform.apply(self.model)
            else:
                self.model, _tf = RevertPrefillOnlyTransform.apply(self.model)

    def export(
        self,
        export_dir: str | None = None,
        use_onnx_subfunctions: bool = False,
        skip_vision: bool | None = False,
        skip_lang: bool | None = False,
        prefill_seq_len: int | None = None,
        prefill_only: bool = False,
        enable_chunking: bool = False,
        num_cores: int = constants.DEFAULT_AIC_NUM_CORES,
        layerwise: bool = False,
        layerwise_window_size: int = 1,
        kv_cache_prefix: str | None = None,
        offload_pt_weights: bool | None = None,
        **kwargs,
    ) -> str:
        """
        Exports both the vision encoder and language decoder components to ONNX format.

        This method exports the vision component (optionally without offloading PyTorch weights)
        and the language component (with offloading PyTorch weights).

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graphs will be saved. Default is None.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        List[str]
            A list containing the paths to the generated ONNX graph files for both components.
        """
        layerwise_cache_probe = kwargs.pop("_layerwise_cache_probe", False)
        reject_legacy_moe_prefill_packed_chunk_size(kwargs)
        if layerwise:
            return self._run_layerwise_export(
                export_dir=export_dir,
                use_onnx_subfunctions=use_onnx_subfunctions,
                skip_vision=skip_vision,
                skip_lang=skip_lang,
                prefill_seq_len=prefill_seq_len,
                prefill_only=prefill_only,
                enable_chunking=enable_chunking,
                layerwise_window_size=layerwise_window_size,
                kv_cache_prefix=kv_cache_prefix,
                **kwargs,
            )
        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len: int = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        qaic_config = kwargs.get("qaic_config", getattr(self.lang_model.model, "qaic_config", None))
        gdn_export_kwargs = _get_gdn_full_state_update_kwargs(
            self.model,
            qaic_config,
            prefill_only=prefill_only,
        )
        # TODO: move this to a DA Serving utility class
        if self.model.config.model_type in SPECIALIZED_DISAGG_SERVING_MODEL_ARCH:
            if prefill_only:
                self.__update_prefill_transform(enable=True, enable_chunking=enable_chunking)
            else:
                self.__update_prefill_transform(False, retain_full_kv=kwargs.get("retain_full_kv", False))
        onnx_kwargs = {"prefill_seq_len": seq_len, "batch_size": bs}
        dynamic_axes_kwargs = {
            "kv_offload": True,
            "continuous_batching": self.continuous_batching,
            "comp_ctx_lengths": self.comp_ctx_lengths_decode,
        }
        if getattr(self.model.config, "model_type", None) in {"qwen3_vl_moe", "qwen3_5_moe", "qwen3_5"}:
            _blocking_cfg = self.lang_model.hash_params.get("blocking_kwargs", None)
            batch_fold = (
                not prefill_only and _blocking_cfg is not None and _blocking_cfg.mode == BlockingMode.KV_BATCH_FOLD
            )
            if batch_fold:
                # Batch-fold is used for the decode graph; export must trace
                # the same one-token decode shape used by compile/runtime.
                onnx_kwargs["prefill_seq_len"] = 1
                onnx_kwargs["batch_fold"] = batch_fold
                dynamic_axes_kwargs["batch_fold"] = batch_fold
        onnx_kwargs.update(gdn_export_kwargs)
        dynamic_axes_kwargs.update(gdn_export_kwargs)
        inputs = self.model.get_dummy_inputs(
            kv_offload=True,
            continuous_batching=self.continuous_batching,
            comp_ctx_lengths=self.comp_ctx_lengths_decode,
            **onnx_kwargs,
        )
        dynamic_axes = self.model.get_onnx_dynamic_axes(**dynamic_axes_kwargs)
        output_names = self.model.get_output_names(kv_offload=True)
        # Prefix only the language-side KV-cache retained buffers (vision buffers are untouched).
        output_names = apply_kv_cache_prefix(output_names, validate_kv_cache_prefix(kv_cache_prefix))
        if self.lang_model.qaic_config is not None and self.lang_model.qaic_config.get("include_sampler", False):
            logits_index = output_names["lang"].index("logits")
            output_names["lang"][logits_index] = "next_tokens"
            inputs["lang"], output_names["lang"], dynamic_axes["lang"] = get_sampling_inputs_and_outputs(
                example_inputs=inputs["lang"],
                output_names=output_names["lang"],
                dynamic_axes=dynamic_axes["lang"],
                continuous_batching=self.continuous_batching,
                vocab_size=self.lang_model.get_model_config["vocab_size"],
                qaic_config=self.lang_model.qaic_config,
            )

        layerwise_export = QEFFBaseModel._layerwise_active

        should_export = not skip_vision and (
            not layerwise_export
            or (
                layerwise_export
                and QEfficient.base.modeling_qeff.QEFFBaseModel._end
                == QEfficient.base.modeling_qeff.QEFFBaseModel._total_layers
            )
        )
        if should_export and not layerwise_cache_probe:
            self.vision_model.export(
                inputs["vision"],
                output_names["vision"],
                dynamic_axes["vision"],
                export_dir=export_dir,
                offload_pt_weights=False,
                use_onnx_subfunctions=use_onnx_subfunctions,
            )

        # TODO: remove the current pt weight offload capability once CustomLoader is in place
        if offload_pt_weights is None:
            if prefill_only and prefill_seq_len is not None and prefill_seq_len > 1:
                offload_pt_weights = False  # keep weights resident for the decode export
            else:
                offload_pt_weights = True

        if not skip_lang:
            self.lang_model.export(
                inputs["lang"],
                output_names["lang"],
                dynamic_axes["lang"],
                export_dir=export_dir,
                offload_pt_weights=offload_pt_weights,
                use_onnx_subfunctions=use_onnx_subfunctions,
                prefill_only=prefill_only,
                enable_chunking=enable_chunking,
                num_cores=num_cores,
                prefill_seq_len=prefill_seq_len,
                qaic_config=qaic_config,
                _layerwise_cache_probe=layerwise_cache_probe,
                kv_cache_prefix=kv_cache_prefix,
            )
        return self.onnx_path

    def transform(
        self,
        ctx_len: int | None = None,
        seq_len: int | None = None,
        bs: int | None = 1,
        num_devices: int = 1,
        qaic_config: dict | None = None,
        **compiler_options,
    ):
        self.vision_model.transform(
            ctx_len=ctx_len,
            seq_len=seq_len,
            bs=bs,
            num_devices=num_devices,
            qaic_config=qaic_config,
            **compiler_options,
        )

        self.lang_model.transform(
            ctx_len=ctx_len,
            seq_len=seq_len,
            bs=bs,
            num_devices=num_devices,
            qaic_config=qaic_config,
            **compiler_options,
        )

    def _layerwise_factory_kwargs(self):
        """Reproduce the from_pretrained kwargs needed to rebuild this wrapper per window."""
        # Mirror the dual-QPC from_pretrained surface; the layerwise driver passes
        # config explicitly per call, so we only carry torch_dtype + attn here.
        # low_cpu_mem_usage=True works with the shard-window patch to allocate
        # only the active window's weights instead of the full model.
        torch_dtype = getattr(self.config, "torch_dtype", None)
        return {
            "attn_implementation": "eager",
            "kv_offload": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }

    def _build_layerwise_factory(self):
        """Return a callable suitable for layerwise driver's qeff_factory hook."""
        from QEfficient import QEFFAutoModelForImageTextToText

        base_kwargs = self._layerwise_factory_kwargs()

        def _factory(model_id, config):
            return QEFFAutoModelForImageTextToText.from_pretrained(
                model_id,
                config=config,
                **base_kwargs,
            )

        return _factory

    def _build_layerwise_vision_wrapper(self):
        """Materialize vision weights while keeping language bounded to one window."""
        model_id = self._pretrained_model_name_or_path
        if model_id is None:
            raise RuntimeError("layerwise=True requires a model loaded via from_pretrained(...).")
        torch_dtype = getattr(self.config, "torch_dtype", None)
        kwargs = {
            "config": self.config,
            "attn_implementation": "eager",
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        _resolve_torch_dtype(kwargs)
        self.config.torch_dtype = kwargs["torch_dtype"]
        model = _build_layerwise_vision_export_model(self._hf_auto_class, model_id, kwargs)
        return self.__class__(
            model,
            continuous_batching=self.continuous_batching,
            pretrained_model_name_or_path=model_id,
        )

    def _run_layerwise_export(
        self,
        *,
        export_dir,
        use_onnx_subfunctions,
        skip_vision,
        skip_lang,
        prefill_seq_len,
        prefill_only,
        enable_chunking,
        layerwise_window_size,
        **kwargs,
    ):
        from QEfficient.transformers.models import _layerwise

        model_id = self._pretrained_model_name_or_path
        if model_id is None:
            raise RuntimeError(
                "layerwise=True requires the QEff model to be built via "
                "QEFFAutoModelForImageTextToText.from_pretrained(...). "
                "Direct __init__ does not preserve the model id needed for per-window reload."
            )
        compile_kwargs = dict(
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
            skip_vision=skip_vision,
            prefill_seq_len=prefill_seq_len,
            prefill_only=prefill_only,
            enable_chunking=enable_chunking,
            **kwargs,
        )
        return _layerwise.run_layerwise(
            model_id=model_id,
            config=self.config,
            qeff_factory=self._build_layerwise_factory(),
            compile_kwargs=compile_kwargs,
            probe_qeff_model=self,
            window_size=layerwise_window_size,
            final_compile=False,
        )

    def _run_layerwise_compile(
        self,
        *,
        layerwise_window_size,
        **compile_kwargs,
    ):
        from QEfficient.transformers.models import _layerwise

        model_id = self._pretrained_model_name_or_path
        if model_id is None:
            raise RuntimeError(
                "layerwise=True requires the QEff model to be built via "
                "QEFFAutoModelForImageTextToText.from_pretrained(...). "
                "Direct __init__ does not preserve the model id needed for per-window reload."
            )
        qpc_paths = _layerwise.run_layerwise(
            model_id=model_id,
            config=self.config,
            qeff_factory=self._build_layerwise_factory(),
            compile_kwargs=compile_kwargs,
            probe_qeff_model=self,
            window_size=layerwise_window_size,
            final_compile=True,
        )
        self.qpc_paths = qpc_paths
        if isinstance(qpc_paths, dict):
            self.vision_model.qpc_path = qpc_paths.get("vision_qpc_path") or self.vision_model.qpc_path
            self.lang_model.qpc_path = (
                qpc_paths.get("lang_decode_qpc_path")
                or qpc_paths.get("lang_prefill_qpc_path")
                or qpc_paths.get("lang_qpc_path")
                or self.lang_model.qpc_path
            )
        return qpc_paths

    def compile(
        self,
        img_size: int | None = None,
        vision_onnx_path: str | None = None,
        lang_onnx_path: str | None = None,
        compile_dir: str | None = None,
        *,
        prefill_seq_len: int | None = None,
        comp_ctx_lengths_prefill: list[int] | None = None,
        comp_ctx_lengths_decode: list[int] | None = None,
        ctx_len: int | None = None,
        batch_size: int = 1,
        vision_batch_size: int | None = None,
        full_batch_size: int | None = None,
        kv_cache_batch_size: int | None = None,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        skip_vision: bool | None = False,
        skip_lang: bool | None = False,
        use_onnx_subfunctions: bool = False,
        prefill_only=None,
        offload_pt_weights: bool | None = None,
        enable_chunking=False,
        dflash_block_size: int | None = None,
        qaic_config: dict | None = None,
        layerwise: bool = False,
        layerwise_window_size: int = 1,
        kv_cache_prefix: Optional[str] = None,
        artifacts: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles both the vision encoder and language decoder components into QPC packages.

        Parameters
        ----------
        img_size : int, optional
            The image size to compile the vision model for. Default is None.
        vision_onnx_path : str, optional
            Path to a pre-exported ONNX file for the vision encoder. If None, it will be exported.
        lang_onnx_path : str, optional
            Path to a pre-exported ONNX file for the language decoder. If None, it will be exported.
        compile_dir : str, optional
            Directory to save the generated QPC packages.
        prefill_seq_len : int, optional
            Length of the prefill prompt for the language model. Default is None.
        ctx_len : int, optional
            Maximum context length for the language model. Default is None.
        batch_size : int, optional
            Language prefill/decode batch size. Default is 1.
        vision_batch_size : int, optional
            Vision encoder batch size. Defaults to ``batch_size``.
        full_batch_size : int, optional
            Not supported for this model; must be None.
        kv_cache_batch_size : int, optional
            Not supported for this model; must be None.
        num_devices : int, optional
            Total number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights in the language model. Default is False.
        mxint8_kv_cache : bool, optional
            Use MXINT8 compression for KV cache. Default is False.
        num_speculative_tokens : int, optional
            Not supported for this model; must be None.
        skip_vision : bool, optional
            If True, skips compilation of the vision encoder. Default is False.
        skip_lang : bool, optional
            If True, skips compilation of the language decoder. Default is False.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers.
            Use ``mdp_num_partitions`` to select the number of pipeline-parallel
            MDP partitions.

        Returns
        -------
        Union[List[str], str, None]
            A list of paths to the compiled QPC packages, or a single path if only
            one component is compiled, or None if neither is compiled.

        Raises
        ------
        ValueError
            If `full_batch_size`, `kv_cache_batch_size`, or `num_speculative_tokens` are not None.
            If both `skip_lang` and `skip_vision` are True.
        """
        if artifacts:
            compiler_options["artifacts"] = True
        if skip_lang and skip_vision:
            raise ValueError("Expected at least one of 'skip_lang' or 'skip_vision' to be False")
        reject_legacy_moe_prefill_packed_chunk_size(compiler_options)
        _ignore_public_mdp_ts_num_devices(compiler_options)
        if qaic_config and qaic_config.get("gdn_full_state_update") is True:
            _get_gdn_full_state_update_kwargs(
                self.model,
                qaic_config,
                prefill_only=prefill_only,
                batch_size=batch_size,
                full_batch_size=full_batch_size,
            )

        if layerwise:
            if skip_lang and not skip_vision:
                vision_wrapper = self._build_layerwise_vision_wrapper()
                qpc_paths = vision_wrapper.compile(
                    img_size=img_size,
                    vision_onnx_path=vision_onnx_path,
                    lang_onnx_path=lang_onnx_path,
                    compile_dir=compile_dir,
                    prefill_seq_len=prefill_seq_len,
                    comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
                    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
                    ctx_len=ctx_len,
                    batch_size=batch_size,
                    vision_batch_size=vision_batch_size,
                    full_batch_size=full_batch_size,
                    kv_cache_batch_size=kv_cache_batch_size,
                    num_devices=num_devices,
                    num_cores=num_cores,
                    mxfp6_matmul=mxfp6_matmul,
                    mxint8_kv_cache=mxint8_kv_cache,
                    skip_vision=skip_vision,
                    skip_lang=skip_lang,
                    use_onnx_subfunctions=use_onnx_subfunctions,
                    prefill_only=prefill_only,
                    offload_pt_weights=offload_pt_weights,
                    enable_chunking=enable_chunking,
                    qaic_config=qaic_config,
                    kv_cache_prefix=kv_cache_prefix,
                    **compiler_options,
                )
                self.vision_model.onnx_path = vision_wrapper.vision_model.onnx_path
                self.vision_model.qpc_path = vision_wrapper.vision_model.qpc_path
                self.vision_model.compile_artifacts_path = vision_wrapper.vision_model.compile_artifacts_path
                self.qpc_paths = qpc_paths
                return qpc_paths
            return self._run_layerwise_compile(
                img_size=img_size,
                vision_onnx_path=vision_onnx_path,
                lang_onnx_path=lang_onnx_path,
                compile_dir=compile_dir,
                prefill_seq_len=prefill_seq_len,
                comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
                comp_ctx_lengths_decode=comp_ctx_lengths_decode,
                ctx_len=ctx_len,
                batch_size=batch_size,
                vision_batch_size=vision_batch_size,
                full_batch_size=full_batch_size,
                kv_cache_batch_size=kv_cache_batch_size,
                num_devices=num_devices,
                num_cores=num_cores,
                mxfp6_matmul=mxfp6_matmul,
                mxint8_kv_cache=mxint8_kv_cache,
                skip_vision=skip_vision,
                skip_lang=skip_lang,
                use_onnx_subfunctions=use_onnx_subfunctions,
                prefill_only=prefill_only,
                offload_pt_weights=offload_pt_weights,
                enable_chunking=enable_chunking,
                qaic_config=qaic_config,
                layerwise_window_size=layerwise_window_size,
                kv_cache_prefix=kv_cache_prefix,
                **compiler_options,
            )

        if self.continuous_batching and full_batch_size is None:
            raise TypeError("`full_batch_size` is required when `continuous_batching=True`.")

        if kv_cache_batch_size and not full_batch_size:
            raise ValueError(
                "KV caching requires continuous batching. Please set `full_batch_size` and "
                "enable `continuous_batching=True` in `from_pretrained`."
            )
        layerwise_cache_probe = compiler_options.pop("_layerwise_cache_probe", False)

        # Infer kv_cache_batch_size if not provided
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size

        kv_cache_prefix = validate_kv_cache_prefix(kv_cache_prefix)
        output_names = self.model.get_output_names(kv_offload=True)
        # Prefix only the language-side KV-cache retained buffers (vision buffers are untouched) so the
        # derived custom_io_lang keys match the prefixed names written into the exported graph.
        output_names = apply_kv_cache_prefix(output_names, kv_cache_prefix)

        # if ccl_enabled is True read Compute-Context-Length lists
        if self.ccl_enabled:
            if comp_ctx_lengths_prefill is None and comp_ctx_lengths_decode is None:
                logger.info("Auto-generating CCL-prefill and CCL-decode lists based on Context Length (CL).")
            self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode, ctx_len = process_ccl_specializations(
                comp_ctx_lengths_prefill, comp_ctx_lengths_decode, ctx_len, prefill_seq_len
            )
        # For supporting VLLM and Disaggregated with CCL
        elif comp_ctx_lengths_prefill is not None or comp_ctx_lengths_decode is not None:
            self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode, ctx_len = process_ccl_specializations(
                comp_ctx_lengths_prefill, comp_ctx_lengths_decode, ctx_len, prefill_seq_len
            )

        # Apply compile-dependent transforms like blocking transform
        moe_batch_size = (
            full_batch_size if self.continuous_batching and not prefill_only and prefill_seq_len == 1 else batch_size
        )
        self.transform(
            ctx_len=ctx_len,
            seq_len=prefill_seq_len,
            bs=batch_size,
            moe_batch_size=moe_batch_size,
            num_devices=num_devices,
            qaic_config=qaic_config,
            aic_num_cores=num_cores,
            mdp_num_partitions=compiler_options.get("mdp_num_partitions", 1),
            prefill_only=prefill_only,
            prefill_seq_len=prefill_seq_len,
        )

        specializations, compiler_options = self.model.get_specializations(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            comp_ctx_lengths_prefill=self.comp_ctx_lengths_prefill,
            comp_ctx_lengths_decode=self.comp_ctx_lengths_decode,
            img_size=img_size,
            kv_offload=True,
            continuous_batching=self.continuous_batching,
            kv_cache_batch_size=kv_cache_batch_size,
            full_batch_size=full_batch_size,
            vision_batch_size=vision_batch_size,
            **compiler_options,
        )

        # DFlash TLM: override the lang decode spec's seq_len to dflash_block_size.
        if getattr(self, "dflash_tlm", None) and dflash_block_size is not None:
            for spec in specializations["lang"]:
                if str(spec.get("seq_len")) == "1":
                    spec["seq_len"] = dflash_block_size

        custom_io_vision = {}
        target_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else CUSTOM_IO_DTYPE_MAP[target_dtype]
        molmo = hasattr(self.model.config, "model_type") and self.model.config.model_type == "molmo"
        if molmo:
            custom_io_vision["image_masks"] = CUSTOM_IO_DTYPE_MAP[target_dtype]
        custom_io_vision["pixel_values"] = CUSTOM_IO_DTYPE_MAP[target_dtype]

        for output_name in output_names["vision"]:
            if output_name.startswith("past_"):
                custom_io_vision[output_name] = kv_cache_dtype
            else:
                custom_io_vision[output_name] = CUSTOM_IO_DTYPE_MAP[target_dtype]

        if vision_onnx_path:
            self.vision_model.onnx_path = vision_onnx_path
        if lang_onnx_path:
            self.lang_model.onnx_path = lang_onnx_path

        needs_vision_export = not skip_vision and vision_onnx_path is None
        needs_lang_export = not skip_lang and lang_onnx_path is None

        if needs_vision_export or needs_lang_export:
            with export_from_compile():
                self.export(
                    use_onnx_subfunctions=use_onnx_subfunctions,
                    skip_vision=skip_vision,
                    skip_lang=skip_lang,
                    prefill_only=prefill_only,
                    enable_chunking=enable_chunking,
                    prefill_seq_len=prefill_seq_len,
                    num_cores=num_cores,
                    qaic_config=qaic_config,
                    _layerwise_cache_probe=layerwise_cache_probe,
                    kv_cache_prefix=kv_cache_prefix,
                    offload_pt_weights=offload_pt_weights,
                )
            if layerwise_cache_probe:
                return self.lang_model.onnx_path

        # generating NPI during compile
        if hasattr(self.model, "generate_npi_file") and "node_precision_info" in compiler_options:
            if isinstance(compiler_options["node_precision_info"], bool) and compiler_options["node_precision_info"]:
                if self.lang_model.onnx_path is None and not skip_lang:
                    raise ValueError("Language ONNX path is required to generate a language NPI file.")
                if self.lang_model.onnx_path:
                    compiler_options["node_precision_info"] = self.model.generate_npi_file(self.lang_model.onnx_path)
            elif (
                isinstance(compiler_options["node_precision_info"], bool)
                and not compiler_options["node_precision_info"]
            ):
                compiler_options.pop("node_precision_info", None)

        # TODO this should be removed once the continous batching is supported for all the models.
        compiler_options.pop("continuous_batching", None)
        compiler_options.pop("kv_cache_batch_size", None)
        compiler_options.pop("full_batch_size", None)
        self.qpc_paths = {}
        if not skip_vision:
            compiler_options_vision = compiler_options.copy()
            compiler_options_vision["node_precision_info"] = False
            vision_qpc_path = self.vision_model._compile(
                onnx_path=self.vision_model.onnx_path,
                compile_dir=compile_dir,
                specializations=specializations["vision"],
                specialization_module_name="Vision",
                convert_to_fp16=_should_convert_to_fp16(target_dtype, compiler_options),
                mxfp6_matmul=constants.VISION_MXFP6_MATMUL,
                mdp_ts_num_devices=num_devices,
                aic_num_cores=num_cores,
                custom_io=custom_io_vision,
                mxint8_kv_cache=mxint8_kv_cache,
                use_onnx_subfunctions=use_onnx_subfunctions,
                **compiler_options_vision,
            )
            self.qpc_paths["vision_qpc_path"] = vision_qpc_path

        # Custom NPI file options
        if hasattr(self.model, "get_npi_file") and "node_precision_info" not in compiler_options:
            compiler_options["node_precision_info"] = self.model.get_npi_file(self.model.name_or_path)

        if not skip_lang:
            custom_io_lang = {}
            for output_name in output_names["lang"]:
                if output_name.endswith("_RetainedState"):
                    dtype = (
                        CUSTOM_IO_DTYPE_MAP[target_dtype]
                        if ("vision_embeds" in output_name or "deepstack_features" in output_name)
                        else kv_cache_dtype
                    )
                    _add_retained_state_custom_io(
                        custom_io_lang,
                        output_name,
                        dtype=dtype,
                        use_onnx_subfunctions=use_onnx_subfunctions,
                    )

            custom_io_lang = _filter_custom_io_for_onnx(custom_io_lang, self.lang_model.onnx_path)

            # get_specializations lays the language specs out as
            # [prefill specs..., decode specs...], with one spec per CCL value when
            # CCL is enabled. The disagg prefill/decode QPCs must therefore keep the
            # whole prefill/decode slice (not just the first/last spec), otherwise all
            # but one CCL value is silently dropped.
            lang_specs = specializations["lang"]
            if prefill_only:
                if self.comp_ctx_lengths_prefill is not None:
                    specializations = lang_specs[: len(self.comp_ctx_lengths_prefill)]
                else:
                    specializations = lang_specs[:1]
                qpc_key = "lang_prefill_qpc_path"
            elif prefill_seq_len == 1:
                if self.comp_ctx_lengths_decode is not None:
                    specializations = lang_specs[-len(self.comp_ctx_lengths_decode) :]
                else:
                    specializations = lang_specs[-1:]
                qpc_key = "lang_decode_qpc_path"
            else:
                specializations = lang_specs
                qpc_key = "lang_qpc_path"

            lang_qpc_path = self.lang_model._compile(
                onnx_path=self.lang_model.onnx_path,
                compile_dir=compile_dir,
                retained_state=True,
                specializations=specializations,
                convert_to_fp16=_should_convert_to_fp16(target_dtype, compiler_options),
                mxfp6_matmul=mxfp6_matmul,
                mdp_ts_num_devices=num_devices,
                aic_num_cores=num_cores,
                custom_io=custom_io_lang,
                mxint8_kv_cache=mxint8_kv_cache,
                use_onnx_subfunctions=use_onnx_subfunctions,
                **compiler_options,
            )
            self.qpc_paths.update({qpc_key: lang_qpc_path})
        return self.qpc_paths

    @log_generate_call
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer = None,
        processor: AutoImageProcessor | None = None,
        images: list[str] | None = None,
        prompts: list[str] | None = None,
        streamer: TextStreamer | None = None,
        device_ids: list[int] | None = None,
        runtime_ai100: bool = True,
        generation_len: Optional[int] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        multi_specs: Optional[bool] = None,
        num_frames: Optional[int] = None,
        skip_vision: bool = False,
        skip_lang: bool = False,
        artifacts: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, np.ndarray, Path]:
        """
        Generates output by executing the compiled QPC(s) on Cloud AI 100 Hardware cards.

        This method coordinates inference between the vision encoder and language model decoder.

        Parameters
        ----------
        inputs : Dict[str, Union[torch.Tensor, np.ndarray]]
            Inputs to run the execution, typically includes `pixel_values`, `input_ids`,
            `attention_mask`, etc.
        tokenizer : PreTrainedTokenizer or PreTrainedTokenizerFast, optional
            Tokenizer for the model. Used when images and prompts are provided.
        processor : AutoImageProcessor, optional
            Processor for the model. Used when images and prompts are provided.
        images : List[str], optional
            List of image paths or PIL images to process.
        prompts : List[str], optional
            List of text prompts corresponding to the images.
        streamer : TextStreamer, optional
            A streamer object to display generated tokens in real-time. Default is None.
        device_ids : List[int], optional
            IDs of devices for running the QPC. E.g., `[0]` for a single device or
            `[0, 1, 2, 3]` for tensor slicing. Defaults to `[0]` if not specified.
        runtime_ai100 : bool, optional
            If True, uses the AI 100 runtime. PyTorch runtime is not supported for this model.
            Default is True.
        generation_len : int, optional
            The maximum number of tokens to generate. If None, it's inferred from `ctx_len`.

        artifacts : bool, optional
            Write first-prefill ``qaic-runner`` inputs without constructing a runtime session.
        Returns
        -------
        CloudAI100ExecInfoNew or np.ndarray
            Output from the AI 100 runtime, including generated IDs and performance metrics.

        Raises
        ------
        NotImplementedError
            If `runtime_ai100` is False.
        """
        if artifacts:
            return write_dual_qpc_vlm_runner_bundle(
                model=self,
                processor=processor,
                images=images,
                prompts=prompts,
                skip_vision=skip_vision,
                skip_lang=skip_lang,
            )
        if not runtime_ai100:
            raise NotImplementedError("PyTorch execution is not supported yet for this model!")

        # Use VisionLanguageGeneration for image-prompt pairs
        if (processor and images) or (tokenizer and prompts) or multi_specs or num_frames:
            # Create VisionLanguageGeneration instance
            batch_size_comp, ctx_len_comp, fbs, num_kv_blocks_comp = get_compilation_dims(self.lang_model.qpc_path)
            vlm_gen = VisionLanguageGeneration(
                qeff_model=self,
                lang_qpc_path=self.lang_model.qpc_path,
                vision_qpc_path=self.vision_model.qpc_path,
                tokenizer=tokenizer,
                processor=processor,
                device_ids=device_ids,  # if device_ids is not None else [0],
                ctx_len=ctx_len_comp,
                num_kv_blocks=num_kv_blocks_comp,
                full_batch_size=fbs,
                comp_ctx_lengths_prefill=self.comp_ctx_lengths_prefill,
                comp_ctx_lengths_decode=self.comp_ctx_lengths_decode,
                image_height=image_height,
                image_width=image_width,
                **kwargs,
            )

            # Call generate method
            return vlm_gen.generate(
                inputs=inputs,
                num_frames=num_frames,
                multi_specs=multi_specs,
                images=images,
                prompts=prompts,
                generation_len=generation_len,
                stream=streamer is not None,
            )

        # Fallback to kv_offload_generate for direct inputs (backward compatibility)
        return self.kv_offload_generate(
            inputs=inputs, device_ids=device_ids, streamer=streamer, generation_len=generation_len
        )

    def kv_offload_generate(
        self,
        inputs: list[str] | None = None,
        streamer: TextStreamer | None = None,
        device_ids: list[int] | None = None,
        generation_len: int | None = None,
    ):
        """
        Performs generation for multimodal models with KV offloading to CPU.

        This method orchestrates the inference by running the vision encoder (if compiled)
        and then iteratively running the language decoder, managing KV cache states.

        Parameters
        ----------
        inputs : Dict[str, Union[torch.Tensor, np.ndarray]]
            Input tensors for the multimodal model.
        streamer : TextStreamer, optional
            A streamer object to display generated tokens in real-time. Default is None.
        device_ids : List[int], optional
            IDs of devices for running the QPC. Defaults to `[0]` if not specified.
        generation_len : int, optional
            The maximum number of tokens to generate. If None, it's inferred from `ctx_len`.

        Returns
        -------
        CloudAI100ExecInfoNew
            Execution information including generated IDs and performance metrics.

        Raises
        ------
        TypeError
            If the language model QPC is not compiled.
        AssertionError
            If `generation_len` is not greater than zero.
        """
        if not self.lang_model.qpc_path:
            raise TypeError("Please run compile API for language model first!")

        lang_session = QAICInferenceSession(self.lang_model.qpc_path, device_ids, activate=False)

        if self.vision_model.qpc_path:
            vision_session = QAICInferenceSession(self.vision_model.qpc_path, device_ids)

        batch_size, ctx_len, _fbs, num_kv_blocks = get_compilation_dims(self.lang_model.qpc_path)

        pad_token_id = 1

        # Skip inputs/outputs
        lang_session.skip_buffers(
            [
                x
                for x in lang_session.input_names + lang_session.output_names
                if is_retained_state_name(x) or x.endswith("_RetainedState")
            ]
        )

        # Read prompt and ctx len from session
        batch_size = max(
            [x[lang_session.binding_index_map["input_ids"]][1][0] for x in lang_session.allowed_shapes]
            + [lang_session.bindings[lang_session.binding_index_map["input_ids"]].dims[0]]
        )

        prefill_seq_len = max(
            [x[lang_session.binding_index_map["input_ids"]][1][1] for x in lang_session.allowed_shapes]
            + [lang_session.bindings[lang_session.binding_index_map["input_ids"]].dims[1]]
        )
        input_len = inputs["attention_mask"].sum(1, keepdims=True)
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        if generation_len is None:
            generation_len = ctx_len - input_len.max()
        assert generation_len > 0, "generation length should be greater than zero"
        generated_ids = np.full((batch_size, generation_len + 1), pad_token_id)

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            pad_token_id,
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )

        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.nn.functional.pad(
                inputs["mm_token_type_ids"], (0, padded_len - input_ids_length), "constant", 0
            )

        if "cross_attention_mask" in inputs:
            inputs["cross_attention_mask"] = torch.nn.functional.pad(
                inputs["cross_attention_mask"], (0, 0, 0, 0, 0, padded_len - input_ids_length)
            )

        for k, v in inputs.items():
            inputs[k] = np.array(v)

        vision_inputs = {
            k: v
            for k, v in inputs.items()
            if k
            in {
                "pixel_values",
                "image_masks",
                "image_position_ids",
                "image_input_idx",
                "valid_idx",
                "aspect_ratio_ids",
                "aspect_ratio_mask",
            }
        }

        for k in constants.VISION_FP16_INPUTS:
            if k not in vision_inputs:
                continue
            if self.vision_model.qpc_path and vision_session.binding_is_bfloat16(k):
                vision_inputs[k] = (
                    torch.from_numpy(vision_inputs[k]).to(torch.bfloat16).view(torch.int16).numpy().view(np.float16)
                )
            else:
                vision_inputs[k] = vision_inputs[k].astype(np.float16)

        # Required for KIMI-K25
        grid_thws_val = inputs.pop("grid_thws", None)
        if grid_thws_val is not None:
            h_val = int(grid_thws_val[0, 1].item())
            w_val = int(grid_thws_val[0, 2].item())
            vision_inputs["h_shape"] = np.ones((h_val), dtype=np.int64)
            vision_inputs["w_shape"] = np.ones((w_val), dtype=np.int64)

        vision_start = perf_counter()

        vision_outputs = {}
        if vision_inputs:
            vision_outputs = vision_session.run(vision_inputs)
        vision_end = perf_counter()

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask")
        else:
            lang_inputs["position_ids"] = np.where(
                lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
            )  # Need to use -1 as position_ids for invalid tokens

        if "mm_token_type_ids" not in lang_inputs and "mm_token_type_ids" in lang_session.input_names:
            # Keep prefill/decode dynamic shapes aligned when callers omit multimodal token type ids.
            lang_inputs["mm_token_type_ids"] = np.zeros_like(
                lang_inputs["input_ids"], dtype=lang_inputs["input_ids"].dtype
            )
        not_mllama = hasattr(self.model.config, "model_type") and self.model.config.model_type != "mllama"
        if not_mllama:
            lang_inputs["image_idx"] = np.array([[0]])
        if self.vision_model.qpc_path:
            vision_session.deactivate()
        lang_session.activate()

        lang_session.set_buffers(vision_outputs)

        # If the vision pass produced no outputs (text-only prompt, or skip_vision=True
        # with an image-aware language QPC), the `vision_embeds` graph input is still part
        # of the lang QPC's signature. Leaving it unbound reads stale state and collapses
        # decoding; bind explicit zeros instead.
        if not vision_outputs and "vision_embeds" in lang_session.input_names:
            ve_binding = lang_session.bindings[lang_session.binding_index_map["vision_embeds"]]
            ve_shape = tuple(int(d) for d in ve_binding.dims)
            ve_dtype = lang_session.aic_to_np_dtype_mapping.get(ve_binding.type, np.dtype(np.float32))
            lang_inputs["vision_embeds"] = np.zeros(ve_shape, dtype=ve_dtype)

        if self.comp_ctx_lengths_prefill is not None:
            list_of_comp_ctx_lengths_prefill = [
                np.zeros(length, dtype=np.int64) for length in self.comp_ctx_lengths_prefill
            ]
            prefill_ccl_id = 0
            lang_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

        if num_kv_blocks:
            kv_block_size = -(-ctx_len // num_kv_blocks)
            lang_inputs["block_table"] = np.arange(batch_size * num_kv_blocks, dtype=np.int64).reshape(
                batch_size, num_kv_blocks
            )
            lang_inputs["slot_id"] = np.zeros(batch_size, dtype=np.int64)

        lang_start = perf_counter()
        # Run prefill
        chunk_inputs = lang_inputs.copy()
        logits_vocab_size = None
        logits_dtype = np.float32
        logits_binding_idx = lang_session.binding_index_map.get("logits")
        input_ids_binding_idx = lang_session.binding_index_map.get("input_ids")
        logits_seq_len_by_input_seq = {}
        default_logits_seq_len = 1
        if logits_binding_idx is not None and getattr(lang_session, "allowed_shapes", None):
            allowed_vocab_sizes = [
                int(shape_spec[logits_binding_idx][1][-1])
                for shape_spec in lang_session.allowed_shapes
                if len(shape_spec) > logits_binding_idx and len(shape_spec[logits_binding_idx][1]) >= 3
            ]
            if allowed_vocab_sizes:
                logits_vocab_size = max(allowed_vocab_sizes)
            if input_ids_binding_idx is not None:
                for shape_spec in lang_session.allowed_shapes:
                    if len(shape_spec) <= max(input_ids_binding_idx, logits_binding_idx):
                        continue
                    input_dims = shape_spec[input_ids_binding_idx][1]
                    logits_dims = shape_spec[logits_binding_idx][1]
                    if len(input_dims) < 2 or len(logits_dims) < 2:
                        continue
                    logits_seq_len_by_input_seq[int(input_dims[1])] = int(logits_dims[1])
        if logits_vocab_size is None and logits_binding_idx is not None:
            logits_dims = tuple(lang_session.bindings[logits_binding_idx].dims)
            if logits_dims:
                logits_vocab_size = int(logits_dims[-1])
            if len(logits_dims) > 1:
                default_logits_seq_len = int(logits_dims[1])
        if logits_binding_idx is not None:
            logits_dtype = lang_session.aic_to_np_dtype_mapping[lang_session.bindings[logits_binding_idx].type]
        for i in range(num_chunks):
            if (
                self.comp_ctx_lengths_prefill is not None
                and (i + 1) * prefill_seq_len > self.comp_ctx_lengths_prefill[prefill_ccl_id]
            ):
                prefill_ccl_id = min(prefill_ccl_id + 1, len(self.comp_ctx_lengths_prefill) - 1)
                chunk_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            if "mm_token_type_ids" in lang_inputs:
                chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
                    ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
                ]
            if "token_type_ids" in lang_inputs:
                chunk_inputs["token_type_ids"] = lang_inputs["token_type_ids"][
                    ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
                ]
            if "cross_attention_mask" in lang_inputs:
                chunk_inputs["cross_attention_mask"] = lang_inputs["cross_attention_mask"][
                    :, i * prefill_seq_len : (i + 1) * prefill_seq_len, :, :
                ]
            if logits_vocab_size is not None:
                input_seq_len = int(chunk_inputs["input_ids"].shape[1])
                logits_seq_len = logits_seq_len_by_input_seq.get(input_seq_len, default_logits_seq_len)
                logits_shape = (
                    chunk_inputs["input_ids"].shape[0],
                    logits_seq_len,
                    logits_vocab_size,
                )
                lang_session.set_buffers({"logits": np.zeros(logits_shape, dtype=logits_dtype)})
            outputs = lang_session.run(chunk_inputs)
            chunk_inputs["image_idx"] = outputs["image_idx_output"]

        prefill_time = perf_counter() - lang_start + vision_end - vision_start
        # Skip inputs/outputs again
        lang_session.skip_buffers(
            [
                x
                for x in lang_session.input_names + lang_session.output_names
                if is_retained_state_name(x) or x.endswith("_RetainedState")
            ]
        )
        if not_mllama:
            lang_session.skip_buffers(vision_outputs.keys())
        # Get first token
        lang_inputs["input_ids"] = outputs["logits"].argmax(2)
        if lang_inputs["input_ids"].ndim == 2 and lang_inputs["input_ids"].shape[1] > 1:
            lang_inputs["input_ids"] = lang_inputs["input_ids"][:, -1:]
        lang_inputs["position_ids"] = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1

        if "mm_token_type_ids" in lang_inputs:
            lang_inputs["mm_token_type_ids"] = np.zeros_like(
                lang_inputs["input_ids"], dtype=lang_inputs["mm_token_type_ids"].dtype
            )
        if num_kv_blocks:
            lang_inputs["slot_id"] = lang_inputs["position_ids"].reshape(-1, batch_size).max(axis=0) % kv_block_size
        if "cross_attention_mask" in lang_inputs:
            bs, _, num_images, img_tiles = lang_inputs["cross_attention_mask"].shape
            lang_inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64).numpy()
        generated_ids[:, 0] = lang_inputs["input_ids"].squeeze(1)

        if streamer:
            streamer.put(lang_inputs["input_ids"][0])

        # Decode loop
        if self.comp_ctx_lengths_decode is not None:
            max_ccl_id = len(self.comp_ctx_lengths_decode) - 1
            list_of_comp_ctx_lengths_decode = [
                np.zeros(length, dtype=np.int64) for length in self.comp_ctx_lengths_decode
            ]
            max_position_id = np.max(lang_inputs["position_ids"])
            ccl_id_initial = 0
            ccl_id = ccl_id_initial
            for i in range(ccl_id_initial, len(self.comp_ctx_lengths_decode)):
                if max_position_id < self.comp_ctx_lengths_decode[i]:
                    ccl_id = i
                    break
            lang_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

        if logits_vocab_size is not None:
            decode_logits_shape = (lang_inputs["input_ids"].shape[0], 1, logits_vocab_size)
            lang_session.set_buffers({"logits": np.zeros(decode_logits_shape, dtype=logits_dtype)})

        decode_start = perf_counter()
        for num_token in range(1, generation_len):
            if self.comp_ctx_lengths_decode is not None:
                if max_position_id >= self.comp_ctx_lengths_decode[ccl_id] - 1:
                    ccl_id = min(ccl_id + 1, max_ccl_id)
                    lang_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

            outputs = lang_session.run(lang_inputs)

            # Prepare inputs for next iteration
            lang_inputs["input_ids"] = outputs["logits"].argmax(2)
            lang_inputs["position_ids"] += 1
            if "mm_token_type_ids" in lang_inputs:
                lang_inputs["mm_token_type_ids"] = np.zeros_like(
                    lang_inputs["input_ids"], dtype=lang_inputs["mm_token_type_ids"].dtype
                )
            if num_kv_blocks:
                lang_inputs["slot_id"] += 1
                lang_inputs["slot_id"] %= kv_block_size
            generated_ids[:, num_token] = lang_inputs["input_ids"].squeeze(1)
            if streamer:
                streamer.put(lang_inputs["input_ids"][0])

        decode_end = perf_counter()
        if streamer:
            streamer.end()

        decode_perf = (num_token - 1) / (decode_end - decode_start)
        total_time = decode_end - decode_start + prefill_time
        total_perf = num_token / total_time

        return CloudAI100ExecInfoNew(
            batch_size=batch_size,
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(
                prefill_time=prefill_time, decode_perf=decode_perf, total_perf=total_perf, total_time=total_time
            ),
        )


class _QEFFAutoModelForImageTextToTextSingleQPC(QEFFTransformersBase, MultimodalUtilityMixin):
    """
    Internal class handling multimodal image-text-to-text models using a single QPC approach.

    In this approach, the entire multimodal model (vision encoder + language model decoder)
    is compiled into a single QPC package.
    """

    _hf_auto_class = AutoModelForImageTextToText
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        KVCacheExternalModuleMapperTransform,
        VlmNoKVOffloadTransform,
        SimpleDecodeMoeTransform,
    ]
    _onnx_transforms = []

    def __init__(
        self,
        model: nn.Module,
        qaic_config: dict | None = None,
        **kwargs,
    ):
        """
        Initializes the single QPC multimodal model wrapper.

        Parameters
        ----------
        model : nn.Module
            The full HuggingFace multimodal model.
        qaic_config : dict, optional
            A dictionary for QAIC-specific configurations. Supported keys include:
            - **num_kv_blocks** (int): Number of K/V blocks for BlockedKV attention implementation.
        **kwargs :
            Additional keyword arguments. `full_batch_size` is not supported here.

        Raises
        ------
        NotImplementedError
            If `full_batch_size` is provided or `include_sampler` is True.
        """
        if kwargs.pop("full_batch_size", None):
            warnings.warn(
                "full_batch_size argument is deprecated. Use continuous_batching=True instead.", DeprecationWarning, 2
            )
            raise NotImplementedError("Continuous batching is not supported for image-text-to-text models yet.")
        if qaic_config is not None and qaic_config.pop("include_sampler", False):
            raise NotImplementedError("On-device sampling is not supported for single QPC multimodal models yet.")

        super().__init__(model, **kwargs)

        self.model.qaic_config = qaic_config

        # to handle internvl models
        if hasattr(self.model.config, "llm_config") and hasattr(self.model.config, "vision_config"):
            self.model.config.llm_config.use_cache = True
            self.model.config.llm_config._attn_implementation = "eager"
            self.model.config.vision_config.use_flash_attn = "false"
        else:
            if hasattr(self.model.config, "text_config"):
                self.model.config.text_config.use_cache = True
            else:
                self.model.config.use_cache = True
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.ccl_enabled = False
        if qaic_config:
            self.ccl_enabled = qaic_config.get("ccl_enabled", False)
        self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = None, None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        qaic_config: dict | None = None,
        *args,
        **kwargs,
    ):
        """
        Load a QEfficient multimodal model for single QPC from a pretrained HuggingFace model or local path.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.
            Note: `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.
            Also, `_attn_implementation` and `use_flash_attn` are configured for VLM models.

        Returns
        -------
        _QEFFAutoModelForImageTextToTextSingleQPC
            An instance initialized with the pretrained weights.
        """
        _disable_unsupported_weight_free(kwargs, cls.__name__)
        enable_proxy = kwargs.pop("enable_proxy", False)

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        if enable_proxy:
            explicit_num_hidden_layers = kwargs.pop("num_hidden_layers", None)
            if explicit_num_hidden_layers is None:
                apply_proxy_layer_config(config)
            else:
                apply_proxy_layer_config(config, num_hidden_layers=explicit_num_hidden_layers)
        config._attn_implementation = "eager"
        config.vision_config.use_flash_attn = "false"
        _resolve_torch_dtype(kwargs)
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, config, *args, **kwargs)

        kwargs.update({"enable_proxy": enable_proxy} if enable_proxy else {})

        return cls(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            qaic_config=qaic_config,
            **kwargs,
        )

    def __update_prefill_transform(
        self,
        enable: bool | None = True,
        enable_chunking: bool | None = False,
        retain_full_kv: bool | None = False,
    ):
        if enable:
            if enable_chunking:
                self.model, tf = PrefillOnlyChunkedTransform.apply(self.model)
            else:
                self.model, tf = PrefillOnlyTransform.apply(self.model)

        else:
            if retain_full_kv:
                self.model, tf = RevertPrefillKeepAttentionTransform.apply(self.model)
            else:
                self.model, _tf = RevertPrefillOnlyTransform.apply(self.model)

    def export(
        self,
        export_dir: str | None = None,
        use_onnx_subfunctions: bool = False,
        prefill_seq_len: int | None = None,
        prefill_only: bool = False,
        enable_chunking: bool = False,
        kv_cache_prefix: str | None = None,
        **kwargs,
    ) -> str:
        """
        Exports the entire multimodal model to ONNX format.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. Default is None.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        str
            Path to the generated ONNX graph file.
        """
        reject_legacy_moe_prefill_packed_chunk_size(kwargs)
        if prefill_only:
            assert prefill_seq_len > 1
            if not enable_chunking and self.continuous_batching:
                raise NotImplementedError(
                    "Looks like you are trying to run prefix-caching without chunking, this feature is not available yet!"
                )
            self.hash_params["prefill_only"] = True
            self.__update_prefill_transform(enable=True, enable_chunking=enable_chunking)
        else:
            self.hash_params["prefill_only"] = False
            self.__update_prefill_transform(False, retain_full_kv=kwargs.get("retain_full_kv", False))

        inputs = self.model.get_dummy_inputs(comp_ctx_lengths=self.comp_ctx_lengths_decode)
        dynamic_axes = self.model.get_onnx_dynamic_axes(comp_ctx_lengths=self.comp_ctx_lengths_decode)
        output_names = self.model.get_output_names()
        # Prefix only the LLM KV-cache retained buffers (vision/multimodal buffers untouched).
        output_names = apply_kv_cache_prefix(output_names, validate_kv_cache_prefix(kv_cache_prefix))
        return self._export(
            inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(
        self,
        onnx_path: str | None = None,
        img_size: int | None = None,
        compile_dir: str | None = None,
        *,
        prefill_seq_len: int | None = None,
        ctx_len: int | None = None,
        comp_ctx_lengths_prefill: list[int] | None = None,
        comp_ctx_lengths_decode: list[int] | None = None,
        batch_size: int = 1,
        full_batch_size: int | None = None,
        kv_cache_batch_size: int | None = None,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        num_speculative_tokens: int | None = None,
        use_onnx_subfunctions: bool = False,
        qaic_config: Optional[dict] = None,
        kv_cache_prefix: Optional[str] = None,
        artifacts: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the exported ONNX model (single QPC) using the Cloud AI 100 Platform SDK compiler.

        This method generates a single ``qpc`` package for the entire multimodal model.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        img_size : int, optional
            The image size to compile the vision part of the model for. Default is None.
        compile_dir : str, optional
            Directory to save the generated QPC package.
        prefill_seq_len : int, optional
            Length of the prefill prompt. Default is None.
        ctx_len : int, optional
            Maximum context length the compiled model can remember. Default is None.
        batch_size : int, optional
            Batch size. Default is 1.
        full_batch_size : int, optional
            Not supported for this model; must be None.
        kv_cache_batch_size : int, optional
            Not supported for this model; must be None.
        num_devices : int, optional
            Total number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        mxint8_kv_cache : bool, optional
            Use MXINT8 compression for KV cache. Default is False.
        num_speculative_tokens : int, optional
            Not supported for this model; must be None.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers.
            Use ``mdp_num_partitions`` to select the number of pipeline-parallel
            MDP partitions.

        Returns
        -------
        str
            Path to the compiled QPC package.

        Raises
        ------
        ValueError
            If `full_batch_size`, `kv_cache_batch_size`, or `num_speculative_tokens` are not None.
        """
        _ignore_public_mdp_ts_num_devices(compiler_options)
        if any(param is not None for param in [full_batch_size, kv_cache_batch_size, num_speculative_tokens]):
            raise ValueError(
                f"Expected 'full_batch_size', 'kv_cache_batch_size', 'num_speculative_tokens' to be None but got: "
                f"full_batch_size={full_batch_size}, kv_cache_batch_size={kv_cache_batch_size}, num_speculative_tokens={num_speculative_tokens}, "
            )

        # Infer kv_cache_batch_size if not provided
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size
        kv_cache_prefix = validate_kv_cache_prefix(kv_cache_prefix)
        output_names = self.model.get_output_names()
        # Prefix only the LLM KV-cache retained buffers so the derived custom_io (and the names baked
        # into the exported graph) stay consistent; vision/multimodal buffers are untouched.
        output_names = apply_kv_cache_prefix(output_names, kv_cache_prefix)

        # if ccl_enabled is True read Compute-Context-Length lists
        if artifacts:
            compiler_options["artifacts"] = True
        if self.ccl_enabled:
            if comp_ctx_lengths_prefill is None and comp_ctx_lengths_decode is None:
                logger.info("Auto-generating CCL-prefill and CCL-decode lists based on Context Length (CL).")
            self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode, ctx_len = process_ccl_specializations(
                comp_ctx_lengths_prefill, comp_ctx_lengths_decode, ctx_len, prefill_seq_len
            )
        # For supporting VLLM and Disaggregated with CCL
        elif comp_ctx_lengths_prefill is not None or comp_ctx_lengths_decode is not None:
            self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode, ctx_len = process_ccl_specializations(
                comp_ctx_lengths_prefill, comp_ctx_lengths_decode, ctx_len, prefill_seq_len
            )

        # Get specializations from modelling file
        # TODO: expose this via the auto class as well
        specializations, compiler_options = self.model.get_specializations(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            comp_ctx_lengths_prefill=self.comp_ctx_lengths_prefill,
            comp_ctx_lengths_decode=self.comp_ctx_lengths_decode,
            kv_cache_batch_size=kv_cache_batch_size,
            img_size=img_size,
            **compiler_options,
        )

        if hasattr(self.model, "get_npi_file") and "node_precision_info" not in compiler_options:
            compiler_options["node_precision_info"] = self.model.get_npi_file(self.model.name_or_path)

        custom_io = {}
        target_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else CUSTOM_IO_DTYPE_MAP[target_dtype]
        for input_name in output_names:
            if input_name.endswith("_RetainedState"):
                dtype = CUSTOM_IO_DTYPE_MAP[target_dtype] if "pixel_values" in input_name else kv_cache_dtype
                _add_retained_state_custom_io(
                    custom_io,
                    input_name,
                    dtype=dtype,
                    use_onnx_subfunctions=use_onnx_subfunctions,
                )

        # TODO this hould be removed once the continous batching is supported for all the models.
        compiler_options.pop("continuous_batching", None)
        compiler_options.pop("kv_cache_batch_size", None)
        compiler_options.pop("full_batch_size", None)
        compile_path = self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=_should_convert_to_fp16(target_dtype, compiler_options),
            mxfp6_matmul=mxfp6_matmul,
            custom_io=custom_io,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            mxint8_kv_cache=mxint8_kv_cache,
            use_onnx_subfunctions=use_onnx_subfunctions,
            qaic_config=qaic_config,
            kv_cache_prefix=kv_cache_prefix,
            **compiler_options,
        )
        return compile_path

    def get_onnx_dynamic_axes(self):
        """
        Retrieves the dynamic axes configuration for ONNX export for this model.

        Returns
        -------
        Dict[str, Dict[int, str]]
            A dictionary specifying the dynamic axes for inputs.
        """
        return self.model.get_onnx_dynamic_axes()

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        streamer: Optional[TextStreamer] = None,
        device_ids: List[int] = None,
        runtime_ai100: bool = True,
        generation_len: int | None = None,
        processor: Optional[AutoImageProcessor] = None,
        images: List[str] = None,
        prompts: List[str] = None,
        artifacts: bool = False,
    ) -> Union[torch.Tensor, np.ndarray, Path]:
        """
        Generates output by executing the compiled single QPC on Cloud AI 100 Hardware cards.

        Parameters
        ----------
        inputs : Dict[str, Union[torch.Tensor, np.ndarray]]
            Inputs to run the execution, typically includes `pixel_values`, `input_ids`,
            `attention_mask`, etc.
        streamer : TextStreamer, optional
            A streamer object to display generated tokens in real-time. Default is None.
        device_ids : List[int], optional
            IDs of devices for running the QPC. E.g., `[0]` for a single device or
            `[0, 1, 2, 3]` for tensor slicing. Defaults to `[0]` if not specified.
        runtime_ai100 : bool, optional
            If True, uses the AI 100 runtime. PyTorch runtime is not supported for this model.
            Default is True.
        generation_len : int, optional
            The maximum number of tokens to generate. If None, it's inferred from `ctx_len`.

        artifacts : bool, optional
            Write first-prefill ``qaic-runner`` inputs without constructing a runtime session.
        Returns
        -------
        CloudAI100ExecInfoNew or np.ndarray
            Output from the AI 100 runtime, including generated IDs and performance metrics.

        Raises
        ------
        NotImplementedError
            If `runtime_ai100` is False.
        """
        if artifacts:
            return write_single_qpc_vlm_runner_bundle(model=self, processor=processor, images=images, prompts=prompts)
        if not runtime_ai100:
            raise NotImplementedError("PyTorch execution is not supported yet for this model!")

        return self.cloud_ai_100_generate(
            inputs=inputs, device_ids=device_ids, generation_len=generation_len, streamer=streamer
        )

    def cloud_ai_100_generate(
        self,
        inputs: torch.Tensor,
        device_ids: list[int],
        enable_debug_logs: bool = False,
        generation_len: int | None = None,
        streamer: TextStreamer | None = None,
    ) -> np.ndarray:
        """
        Performs generation for multimodal models using a single QPC on Cloud AI 100 hardware.

        Parameters
        ----------
        inputs : Dict[str, Union[torch.Tensor, np.ndarray]]
            Input tensors for the multimodal model.
        device_ids : List[int]
            IDs of devices for running the QPC.
        enable_debug_logs : bool, optional
            If True, enables debug logging for the QAIC inference session. Default is False.
        generation_len : int, optional
            The maximum number of tokens to generate. If None, it's inferred from `ctx_len`.
        streamer : TextStreamer, optional
            A streamer object to display generated tokens in real-time. Default is None.

        Returns
        -------
        CloudAI100ExecInfoNew
            Execution information including generated IDs and performance metrics.

        Raises
        ------
        AssertionError
            If `generation_len` is not greater than zero.
        """
        inputs = self.auto_correct_inputs(inputs)
        qpc_session = QAICInferenceSession(
            self.qpc_path, device_ids, enable_debug_logs=enable_debug_logs, activate=False
        )
        batch_size, ctx_len, _fbs, _ = get_compilation_dims(self.qpc_path)
        pad_token_id = 1
        # Skip inputs/outputs
        qpc_session.skip_buffers(
            [
                x
                for x in qpc_session.input_names + qpc_session.output_names
                if is_retained_state_name(x) or x.endswith("_RetainedState")
            ]
        )

        # Read prompt and ctx len from session
        batch_size = max(
            [x[qpc_session.binding_index_map["input_ids"]][1][0] for x in qpc_session.allowed_shapes]
            + [qpc_session.bindings[qpc_session.binding_index_map["input_ids"]].dims[0]]
        )

        prefill_seq_len = max(
            [x[qpc_session.binding_index_map["input_ids"]][1][1] for x in qpc_session.allowed_shapes]
            + [qpc_session.bindings[qpc_session.binding_index_map["input_ids"]].dims[1]]
        )

        input_len = inputs["attention_mask"].sum(1, keepdims=True)
        input_ids_length = inputs["input_ids"].shape[1]

        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float

        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len
        if generation_len is None:
            generation_len = ctx_len - input_len.max()

        assert generation_len > 0, "generation length should be greater than zero"
        generated_ids = np.full((batch_size, generation_len + 1), pad_token_id)

        # Prepare inputs for prefill
        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            pad_token_id,
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        if "cross_attention_mask" in inputs:
            inputs["cross_attention_mask"] = torch.nn.functional.pad(
                inputs["cross_attention_mask"], (0, 0, 0, 0, 0, padded_len - input_ids_length)
            )
        for k, v in inputs.items():
            inputs[k] = np.array(v)

        if "pixel_values_RetainedState" in qpc_session.output_names:
            if qpc_session.binding_is_bfloat16("pixel_values"):
                inputs["pixel_values"] = (
                    torch.from_numpy(inputs["pixel_values"])
                    .to(torch.bfloat16)
                    .view(torch.int16)
                    .numpy()
                    .view(np.float16)
                )
            else:
                inputs["pixel_values"] = inputs["pixel_values"].astype(np.float16)

        inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
        inputs["image_idx"] = np.array([[0]])

        if self.comp_ctx_lengths_prefill is not None:
            list_of_comp_ctx_lengths_prefill = [
                np.zeros(length, dtype=np.int64) for length in self.comp_ctx_lengths_prefill
            ]
            prefill_ccl_id = 0
            inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

        qpc_session.activate()
        chunk_inputs = inputs.copy()
        prefill_start = perf_counter()

        # Run prefill
        for i in range(num_chunks):
            if (
                self.comp_ctx_lengths_prefill is not None
                and (i + 1) * prefill_seq_len > self.comp_ctx_lengths_prefill[prefill_ccl_id]
            ):
                prefill_ccl_id = min(prefill_ccl_id + 1, len(self.comp_ctx_lengths_prefill) - 1)
                chunk_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

            chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            outputs = qpc_session.run(chunk_inputs)

            chunk_inputs["image_idx"] = outputs["image_idx_output"]

        prefill_time = perf_counter() - prefill_start
        # Get first token
        inputs["input_ids"] = outputs["logits"].argmax(2)
        inputs["position_ids"] = input_len.numpy()

        if "cross_attention_mask" in inputs:
            bs, _, num_images, img_tiles = inputs["cross_attention_mask"].shape
            inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64).numpy()

        generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
        if streamer:
            streamer.put(inputs["input_ids"][0])

        if "pixel_values_RetainedState" in qpc_session.output_names:
            qpc_session.skip_buffers(["pixel_values"])
            inputs.pop("pixel_values")

        # Decode loop
        if self.comp_ctx_lengths_decode is not None:
            list_of_comp_ctx_lengths_decode = [
                np.zeros(length, dtype=np.int64) for length in self.comp_ctx_lengths_decode
            ]
            max_ccl_id = len(self.comp_ctx_lengths_decode) - 1
            max_position_id = np.max(inputs["position_ids"])
            ccl_id_initial = 0
            ccl_id = ccl_id_initial
            for i in range(ccl_id_initial, len(self.comp_ctx_lengths_decode)):
                if max_position_id < self.comp_ctx_lengths_decode[i]:
                    ccl_id = i
                    break
            inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

        decode_start = perf_counter()
        for num_token in range(1, generation_len):
            if self.comp_ctx_lengths_decode is not None:
                if max_position_id >= self.comp_ctx_lengths_decode[ccl_id] - 1:
                    ccl_id = min(ccl_id + 1, max_ccl_id)
                    inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

            outputs = qpc_session.run(inputs)

            # Prepare inputs for next iteration
            inputs["input_ids"] = outputs["logits"].argmax(2)
            inputs["position_ids"] += 1
            generated_ids[:, num_token] = inputs["input_ids"].squeeze(1)
            if streamer:
                streamer.put(inputs["input_ids"][0])

        decode_end = perf_counter()
        if streamer:
            streamer.end()

        decode_perf = (num_token - 1) / (decode_end - decode_start)
        total_time = decode_end - prefill_start
        total_perf = num_token / total_time

        return CloudAI100ExecInfoNew(
            batch_size=batch_size,
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(
                prefill_time=prefill_time, decode_perf=decode_perf, total_perf=total_perf, total_time=total_time
            ),
        )

    @property
    def get_model_config(self) -> dict:
        """
        Get the configuration dictionary of the underlying HuggingFace model.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        return self.model.config.__dict__


class QEFFAutoModelForImageTextToText:
    """
    QEfficient class for multimodal (image-text-to-text) models from the HuggingFace hub.

    This class supports both single and dual QPC (Quantized Package Compilation) approaches for efficient deployment on Cloud AI 100 hardware.
    It is recommended to use the ``from_pretrained`` method for initialization.

    Example
    -------
    .. code-block:: python

        import requests
        from PIL import Image
        from transformers import AutoProcessor, TextStreamer
        from QEfficient import QEFFAutoModelForImageTextToText

        HF_TOKEN = "" # Your HuggingFace token if needed
        model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        query = "Describe this image."
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

        # STEP 1: Load processor and model
        processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
        model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name, token=HF_TOKEN, attn_implementation="eager", kv_offload=False # kv_offload=False for single QPC
        )

        # STEP 2: Export & Compile
        model.compile(
            prefill_seq_len=32,
            ctx_len=512,
            img_size=560,
            num_cores=16,
            num_devices=1,
            mxfp6_matmul=False,
        )

        # STEP 3: Prepare inputs
        image = Image.open(requests.get(image_url, stream=True).raw)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }
        ]
        input_text = [processor.apply_chat_template(messages, add_generation_prompt=True)]
        inputs = processor(
            text=input_text,
            images=image,
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length", # Consider padding strategy if max_length is crucial
            max_length=32,
        )

        # STEP 4: Run inference
        streamer = TextStreamer(processor.tokenizer)
        model.generate(inputs=inputs, streamer=streamer, generation_len=512)
    """

    _hf_auto_class = AutoModelForImageTextToText

    def __new__(
        self,
        model: nn.Module,
        kv_offload: bool | None = True,
        continuous_batching: bool = False,
        qaic_config: dict | None = None,
        **kwargs,
    ):
        """
        Instantiate the appropriate internal class for single or dual QPC mode.

        Parameters
        ----------
        model : nn.Module
            The loaded HuggingFace multimodal model.
        kv_offload : bool, optional
            If True, uses the dual QPC approach (vision encoder KV offloaded).
            If False, uses the single QPC approach (entire model in one QPC).
            Default is True.
        **kwargs :
            Additional keyword arguments passed to the constructor of the selected internal class.

        Returns
        -------
        Union[_QEffAutoModelForImageTextToTextDualQPC, _QEFFAutoModelForImageTextToTextSingleQPC]
            The wrapped model instance, configured for either dual or single QPC.
        """
        _disable_unsupported_weight_free(kwargs, self.__name__)
        if kv_offload:
            return _QEffAutoModelForImageTextToTextDualQPC(
                model, continuous_batching, qaic_config=qaic_config, **kwargs
            )
        else:
            return _QEFFAutoModelForImageTextToTextSingleQPC(model, qaic_config=qaic_config, **kwargs)

    @classmethod
    @log_from_pretrained_call
    @with_replaced_quantizers
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        kv_offload: bool | None = None,
        continuous_batching: bool = False,
        qaic_config: dict | None = None,
        layerwise: bool = False,
        **kwargs,
    ):
        """
        Load a QEfficient image-text-to-text model from a pretrained HuggingFace model or local path.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        kv_offload : bool, optional
            If True, uses the dual QPC approach (vision encoder KV offloaded).
            If False, uses the single QPC approach (entire model in one QPC).
            If None, the default behavior of the internal classes is used (typically dual QPC).
        qaic_config : dict, optional
            A dictionary for QAIC-specific configurations.
        **kwargs :
            Additional arguments passed to HuggingFace's ``from_pretrained``.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically set to "eager" and False respectively to ensure compatibility.
            `continuous_batching` is not supported for image-text-to-text models.

        Returns
        -------
        QEFFAutoModelForImageTextToText
            An instance initialized with the pretrained weights, wrapped for QEfficient.

        Raises
        ------
        NotImplementedError
            If `continuous_batching` is provided as True.
        """
        _disable_unsupported_weight_free(kwargs, cls.__name__)
        enable_proxy = kwargs.pop("enable_proxy", False)

        # TODO: add a check to see if kv_offload is allowed for given model by loading the config and checking architecture or type of config here.
        if continuous_batching and not kv_offload:
            raise NotImplementedError("Continuous batching is not supported for kv_offload = False")

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        # Respect explicit low_cpu_mem_usage=True (used by the layer-wise driver
        # to materialize only the active window's weights via meta init +
        # sharded materialization). Default behavior remains False.
        explicit_low_cpu = kwargs.get("low_cpu_mem_usage", None) is True
        kwargs.update(
            {
                "attn_implementation": "eager",
                "low_cpu_mem_usage": bool(explicit_low_cpu),
            }
        )

        _resolve_torch_dtype(kwargs)
        if enable_proxy:
            prepare_proxy_config(pretrained_model_name_or_path, kwargs)
        if layerwise:
            # Layer-wise mode: build the outer model on the meta device so the
            # caller's ``from_pretrained`` does not pull the full checkpoint
            # into RAM. compile()/export() rebuilds a real per-window model
            # internally via the layer-wise driver, so the outer instance is
            # only used as a config holder.
            model = _build_meta_model(cls._hf_auto_class, pretrained_model_name_or_path, kwargs)
        else:
            model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        kwargs.update({"enable_proxy": enable_proxy} if enable_proxy else {})

        instance = cls(
            model,
            kv_offload=kv_offload,
            continuous_batching=continuous_batching,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            qaic_config=qaic_config,
            **kwargs,
        )
        # Mark the wrapper so its compile() can default ``layerwise=True`` if
        # the user forgot to pass it (the meta model cannot be exported any
        # other way) and so the driver knows weights still need to be loaded.
        if layerwise:
            instance._layerwise_outer_meta = True
        return instance


MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP = {
    "InternVLChatModel": QEFFAutoModelForImageTextToText,
    "MolmoForCausalLM": QEFFAutoModelForImageTextToText,
}


class QEFFAutoModelForCausalLM(QEFFBaseModel):
    """
    QEfficient class for Causal Language Models from the HuggingFace hub (e.g., GPT-2, Llama).

    This class provides a unified interface for loading, exporting, compiling, and generating
    text with causal language models on Cloud AI 100 hardware. It supports features like
    continuous batching, speculative decoding (TLM), and on-device sampling.

    Example
    -------
    .. code-block:: python

        from QEfficient import QEFFAutoModelForCausalLM
        from transformers import AutoTokenizer

        model = QEFFAutoModelForCausalLM.from_pretrained("gpt2")
        model.compile(num_cores=16)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model.generate(prompts=["Hi there!!"], tokenizer=tokenizer)
    """

    _hf_auto_class = AutoModelForCausalLM
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        FP8DeQuantLinearToLinearTransform,
        PackQuantizedInt4ToMatMulNBitsTransform,
        Mxfp4GptOssExpertDequantizeTransform,
        CustomOpsTransform,
        KVCacheTransform,
        KVCacheExternalModuleMapperTransform,
        SimpleDecodeMoeTransform,
    ]

    _onnx_transforms = []

    _checkpoint_transforms = [
        GptOssMxfp4ExpertDequantSplitCheckpointTransform,
        MoEExpertStackingCheckpointTransform,
        MoEFusedExpertSplitCheckpointTransform,
        ExpertParallelPackingCheckpointTransform,
        DtypeConversionCheckpointTransform,
    ]

    def prefill(
        self,
        enable: bool | None = True,
        enable_chunking: bool | None = False,
        retain_full_kv: bool | None = False,
    ):
        if enable:
            self.model, tf = PrefillOnlyExternalModuleMapperTransform.apply(self.model)
            if enable_chunking:
                self.model, tf = PrefillOnlyChunkedTransform.apply(self.model)
            else:
                self.model, tf = PrefillOnlyTransform.apply(self.model)

        else:
            self.model, tf = RevertPrefillOnlyExternalModuleMapperTransform.apply(self.model)
            if retain_full_kv:
                self.model, tf = RevertPrefillKeepAttentionTransform.apply(self.model)
            else:
                self.model, _tf = RevertPrefillOnlyTransform.apply(self.model)

    def __update_prefill_transform(
        self,
        enable: bool | None = True,
        enable_chunking: bool | None = False,
        retain_full_kv: bool | None = False,
    ):
        if enable:
            self.model, tf = PrefillOnlyExternalModuleMapperTransform.apply(self.model)
            if enable_chunking:
                self.model, tf = PrefillOnlyChunkedTransform.apply(self.model)
            else:
                self.model, tf = PrefillOnlyTransform.apply(self.model)

        else:
            self.model, tf = RevertPrefillOnlyExternalModuleMapperTransform.apply(self.model)
            if retain_full_kv:
                self.model, tf = RevertPrefillKeepAttentionTransform.apply(self.model)
            else:
                self.model, _tf = RevertPrefillOnlyTransform.apply(self.model)

    def __init__(
        self,
        model: nn.Module,
        continuous_batching: bool = False,
        qaic_config: dict | None = None,
        max_seq_len_cached: int | None = None,
        **kwargs,
    ):
        """
        Initializes a QEFFAutoModelForCausalLM instance.

        Parameters
        ----------
        model : nn.Module
            The underlying HuggingFace PyTorch Causal Language Model.
        continuous_batching : bool, optional
            If True, enables continuous batching mode for future compilation and execution.
            This setting must be consistent across `from_pretrained` and `compile` calls. Default is False.
        qaic_config : dict, optional
            A dictionary for QAIC-specific configurations. Supported keys include:
            - **speculative_model_type** (str): Specifies the type of Speculative Decoding model (e.g., "target").
            - **include_sampler** (bool): If True, enables on-device sampling of next tokens.
            - **return_pdfs** (bool): If True, returns probability distributions along with sampled tokens.
              For Speculative Decoding Target Language Models, this is always True.
            - **max_top_k_ids** (int): Maximum number of top K tokens (<= vocab size) to consider during sampling.
            - **include_guided_decoding** (bool): If True, enables guided token-level filtering
              during decoding. Only works when include_sampler=True.
            - **num_kv_blocks** (int): Number of K/V blocks for BlockedKV attention implementation.
        **kwargs :
            Additional keyword arguments passed to the base class constructor.

        Raises
        ------
        TypeError
            If the provided `model` is not a CausalLM or LMHeadModel type.
        """
        model_class_name = model.__class__.__name__
        if not (model_class_name.endswith(("ForCausalLM", "LMHeadModel"))):
            raise TypeError(f"Required pytorch module for CausalLM or LMHeadModel, got {model_class_name}")
        _configure_proxy_for_model(self, kwargs.pop("enable_proxy", False))

        # TODO: remove from version 1.20
        if kwargs.pop("full_batch_size", None):
            continuous_batching = True
            warnings.warn(
                "full_batch_size argument is deprecated. Use continuous_batching=True instead.", DeprecationWarning, 2
            )
        if hasattr(model.config, "quantization_config") and not isinstance(
            model.config.quantization_config, tuple(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.values())
        ):
            logger.warning(
                "Please use `from_pretrained` method to load quantized models, might give unexpected results"
            )
        # Set use_cache=True to get KV values as output during ONNX export
        model.config.use_cache = True

        model.config.max_seq_len_cached = max_seq_len_cached
        super().__init__(model, qaic_config=qaic_config, **kwargs)
        self.num_layers = model.config.num_hidden_layers
        self.continuous_batching = continuous_batching
        self.model.qaic_config = qaic_config
        self.model.pretrained_path = kwargs.pop("pretrained_model_name_or_path", None)

        # DFlash changes the model structure and output contract, so apply it before
        # the generic speculative-decoding and sampler transforms wrap model.forward.
        self.dflash_dlm = bool(qaic_config and qaic_config.get("dflash_dlm", False))
        self.dflash_tlm = bool(qaic_config and qaic_config.get("target_layer_ids", None))
        if self.dflash_dlm:
            self.model, _ = DFlashTransform.apply(self.model, qaic_config)
            self.model, _ = DFlashDLMTransform.apply(self.model, qaic_config)
        if self.dflash_tlm:
            self.model, _ = DFlashTLMTransform.apply(self.model, qaic_config)

        self.model, transformed = SpDTransform.apply(self.model, qaic_config=qaic_config, **kwargs)
        self.is_tlm = transformed

        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.ccl_enabled = False
        if qaic_config:
            self.ccl_enabled = qaic_config.get("ccl_enabled", False)
            if mla_absorption := qaic_config.get("mla_absorption", None):
                self.hash_params["mla_absorption"] = mla_absorption
                self.model.mla_absorption = mla_absorption
        self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = None, None
        self.hash_params["max_seq_len_cached"] = max_seq_len_cached

        # ---Sampling---
        # Note: SamplerTransform should be applied after all other transforms
        # are done. The role of the sampler is to just add nodes at the output of the
        # previous transform function.
        self.model, transformed = SamplerTransform.apply(self.model, qaic_config=qaic_config, **kwargs)
        # TODO : Update in qaic_config isn't updated in the hash due to SpDTransforms. Need to move
        # SpDTransforms to PytorchTransforms.
        if self.is_tlm:
            self.model.qaic_config["return_pdfs"] = True

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__()

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        continuous_batching: bool = False,
        qaic_config: dict | None = None,
        max_seq_len_cached: int | None = None,
        layerwise: bool = False,
        weight_free: bool = False,
        *args,
        **kwargs,
    ):
        """
        Load a QEfficient Causal Language Model from a pretrained HuggingFace model or local path.

        This is the recommended way to initialize a QEfficient Causal Language Model.
        The interface is similar to ``transformers.AutoModelForCausalLM.from_pretrained``.
        Once initialized, you can use methods such as ``export``, ``compile``, and ``generate``.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        continuous_batching : bool, optional
            Whether this model will be used for continuous batching in the future.
            If not set to True here, the model cannot be exported/compiled for
            continuous batching later. Default is False.
        qaic_config : dict, optional
            QAIC config dictionary. Supported keys include:

            - **speculative_model_type** (str): Specify Speculative Decoding Target Language Models.
            - **include_sampler** (bool): Enable/Disable sampling of next tokens.
            - **return_pdfs** (bool): Return probability distributions along with sampled next tokens.
              For Speculative Decoding Target Language Model, ``return_pdfs=True`` always.
              Otherwise, ``return_pdfs=True`` for Speculative Decoding Draft Language Model
              and ``return_pdfs=False`` for regular model.
            - **max_top_k_ids** (int): Maximum number of top K tokens (<= vocab size) to consider during sampling.
              The values provided in ``top_ks`` tensor must be less than this maximum limit.
            - **include_guided_decoding** (bool): If True, enables guided token-level filtering
              during decoding. Only works when include_sampler=True.
        weight_free : bool, optional
            If True, builds the model on the meta device instead of loading real
            checkpoint weights — no weights are materialized into RAM. This is
            the single place to enable weight-free export; ``export()``/``compile()``
            automatically route through the weight-free path afterward, with no
            further flag needed. The real checkpoint weights are supplied at
            export time via ``pretrained_model_name_or_path``. Mutually exclusive
            with ``layerwise=True``. Default is False.

            The prepared/converted checkpoint produced during weight-free export
            (dtype conversion, MoE expert restacking, etc.) is saved under the
            ``QEFF_CHECKPOINT_HOME`` environment variable if set, otherwise next
            to the source checkpoint under the Hugging Face cache.

        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        QEFFAutoModelForCausalLM
            An instance initialized with the pretrained weights.
        """
        if layerwise and weight_free:
            raise ValueError(
                "`layerwise=True` and `weight_free=True` are mutually exclusive; weight_free replaces layerwise mode."
            )
        if weight_free:
            validate_dynamo_export_requirements("weight_free=True")

        enable_proxy = kwargs.pop("enable_proxy", False)
        if kwargs.pop("full_batch_size", None):
            continuous_batching = True
            warnings.warn(
                "full_batch_size argument is deprecated. Use continuous_batching=True instead.", DeprecationWarning, 2
            )

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        kv_offload = kwargs.pop("kv_offload", None)

        # Respect explicit low_cpu_mem_usage=True (used by the layer-wise driver
        # to materialize only the active window's weights via meta init +
        # sharded materialization). Default behavior remains False.
        explicit_low_cpu = kwargs.get("low_cpu_mem_usage", None) is True
        kwargs.update(
            {
                "attn_implementation": "eager",
                "low_cpu_mem_usage": bool(explicit_low_cpu),
            }
        )

        _resolve_torch_dtype(kwargs)
        if enable_proxy:
            prepare_proxy_config(pretrained_model_name_or_path, kwargs)
        if layerwise:
            warnings.warn(
                "layerwise export is deprecated and will be removed in a future release. "
                "Use weight-free export (weight_free=True) instead, which provides "
                "the same memory benefit without the complexity of per-window re-export.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Layer-wise mode: build the outer model on the meta device. The
            # caller still gets a typed wrapper, but no checkpoint weights are
            # pulled into RAM. compile()/export() rebuilds a real per-window
            # model internally via the layer-wise driver.
            model = _build_meta_model(cls._hf_auto_class, pretrained_model_name_or_path, kwargs)
        elif weight_free:
            # Weight-free mode: build the model on the meta device so no
            # checkpoint weights are ever materialized here. The real weights
            # are supplied later at export time via pretrained_model_name_or_path.
            model = _build_meta_model(cls._hf_auto_class, pretrained_model_name_or_path, kwargs)
        else:
            model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if qaic_config is not None:
            qaic_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path

        # This is support models that should be classified to in a different auto class but transformers load them via this class
        kwargs.update({"enable_proxy": enable_proxy} if enable_proxy else {})
        if model.__class__.__name__ in MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP:
            return MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP[model.__class__.__name__](
                model,
                kv_offload=kv_offload,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                qaic_config=qaic_config,
                continuous_batching=continuous_batching,
                **kwargs,
            )
        instance = cls(
            model,
            continuous_batching=continuous_batching,
            qaic_config=qaic_config,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            max_seq_len_cached=max_seq_len_cached,
            weight_free=weight_free,
            **kwargs,
        )
        if layerwise:
            instance._layerwise_outer_meta = True
        return instance

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def handle_gpt_oss_env_variable_legacy_burden(self, prefill_seq_len: int | None = None) -> int:
        num_q_blocks = (
            self.hash_params["blocking_kwargs"].num_q_blocks if self.hash_params.get("blocking_kwargs", None) else None
        )
        if num_q_blocks is None:
            if (
                prefill_seq_len is None
                or prefill_seq_len % constants.GPT_OSS_PREFILL_Q_BLOCK_SIZE != 0
                or prefill_seq_len < constants.GPT_OSS_PREFILL_Q_BLOCK_SIZE
            ):
                raise ValueError(
                    f"When prefill_only=True, 'prefill_seq_len' must be explicitly set and divisible by block_size={constants.GPT_OSS_PREFILL_Q_BLOCK_SIZE}. "
                    f"Or set `NUM_Q_BLOCKS` ENV variable"
                    f"Received: prefill_seq_len={prefill_seq_len}"
                )

            num_q_blocks = prefill_seq_len // constants.GPT_OSS_PREFILL_Q_BLOCK_SIZE
            logger.warning(
                f"Setting NUM_Q_BLOCKS={num_q_blocks} used in attention Q-blocking for prefill_only model, please pass `NUM_Q_BLOCKS` in qaic_config to override"
            )
        num_q_blocks = int(num_q_blocks)

        num_ffn_blocks = os.environ.get("NUM_FFN_BLOCKS", None)
        num_ffn_blocks = int(num_ffn_blocks) if num_ffn_blocks else num_ffn_blocks
        min_seq_len = max(num_q_blocks, num_ffn_blocks) if num_ffn_blocks else num_q_blocks
        if (num_ffn_blocks and min_seq_len % num_ffn_blocks != 0) or min_seq_len % num_q_blocks != 0:
            raise ValueError(
                f"Got NUM_FFN_BLOCKS={num_ffn_blocks} and NUM_Q_BLOCKS={num_q_blocks}, tried to set seq_len={min_seq_len} for export but,"
                "seq_len is not divisible by either num_ffn_blocks or num_q_blocks, try chaning the values."
            )

        self.hash_params["NUM_Q_BLOCKS"] = num_q_blocks
        self.hash_params["NUM_FFN_BLOCKS"] = num_ffn_blocks
        self.hash_params["ENABLE_OPT_SWA"] = os.environ.get("ENABLE_OPT_SWA", "0")
        return max(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, min_seq_len)

    def _run_layerwise(self, *, final_compile: bool, layerwise_window_size: int, **forward_kwargs):
        """Drive the layer-wise export/compile loop for CausalLM models."""
        from QEfficient.transformers.models import _layerwise

        model_id = getattr(self.model, "pretrained_path", None)
        if model_id is None:
            raise RuntimeError(
                "layerwise=True requires the QEff model to be built via "
                "QEFFAutoModelForCausalLM.from_pretrained(...). "
                "Direct __init__ does not preserve the model id needed for per-window reload."
            )
        config = getattr(self.model, "config", None)
        torch_dtype = getattr(config, "torch_dtype", None)

        def _factory(model_id, config):
            return QEFFAutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                torch_dtype=torch_dtype,
                continuous_batching=self.continuous_batching,
                low_cpu_mem_usage=True,
            )

        return _layerwise.run_layerwise(
            model_id=model_id,
            config=config,
            qeff_factory=_factory,
            compile_kwargs=forward_kwargs,
            probe_qeff_model=self,
            window_size=layerwise_window_size,
            final_compile=final_compile,
        )

    def export(
        self,
        export_dir: str | None = None,
        prefill_only: bool | None = False,
        prefill_seq_len: int | None = None,
        num_cores: int = constants.DEFAULT_AIC_NUM_CORES,
        layerwise: bool = False,
        layerwise_window_size: int = 1,
        kv_cache_prefix: str | None = None,
        dynamo: bool = False,
        **kwargs,
    ) -> str:
        """
        Export the model to ONNX format using ``torch.onnx.export``.

        This method prepares example inputs and dynamic axes based on the model configuration,
        then exports the model to an ONNX graph suitable for compilation and deployment
        on Cloud AI 100 hardware. It handles KV cache inputs/outputs and sampler-related inputs.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved.
            If not provided, the default export directory is used.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        dynamo: bool, optional
            whether to enable dynamo during export.
        Returns
        -------
        str
            Path to the generated ONNX graph file.
        """
        if kwargs.pop("decode_only", False):
            raise NotImplementedError(
                "decode_only=True is not supported by QEFFAutoModelForCausalLM.export(). "
                "Use the default non-prefill export path for standard CausalLM decode graphs."
            )
        reject_legacy_moe_prefill_packed_chunk_size(kwargs)
        enable_chunking = override_gptoss_prefill_chunking(
            self.model.config, prefill_only, kwargs.get("enable_chunking", False)
        )
        kwargs["enable_chunking"] = enable_chunking
        qaic_config = kwargs.pop("qaic_config", getattr(self.model, "qaic_config", None))
        # Weight-free export always uses the dynamo (torch.export) path.
        # Must be set here — @export_wrapper reads dynamo from kwargs before _export() body runs.
        dynamo = dynamo or self._weight_free
        if (
            kwargs.get("retain_full_kv", False)
            and self.model.config.model_type not in SPECIALIZED_DISAGG_SERVING_MODEL_ARCH
        ):
            logger.warning(
                "retain_full_kv=True is only supported for specialized disaggregated serving models "
                f"{sorted(SPECIALIZED_DISAGG_SERVING_MODEL_ARCH)}; ignoring it for model_type "
                f"'{self.model.config.model_type}'."
            )

        ## TODO: Remove this deprecated code ###
        if layerwise:
            return self._run_layerwise(
                final_compile=False,
                layerwise_window_size=layerwise_window_size,
                export_dir=export_dir,
                prefill_only=prefill_only,
                prefill_seq_len=prefill_seq_len,
                num_cores=num_cores,
                qaic_config=qaic_config,
                kv_cache_prefix=kv_cache_prefix,
                **kwargs,
            )
        ########################################

        ####### HANDLE DA PREFILL And REVERT PREFILL Transform ################
        # TODO: move this code inside self.transform in modeling_qeff.py
        if self.model.config.model_type in SPECIALIZED_DISAGG_SERVING_MODEL_ARCH:
            if prefill_only:
                if not enable_chunking and self.continuous_batching:
                    raise NotImplementedError(
                        "Looks like you are trying to run prefix-caching without chunking, this feature is not available yet!"
                    )
                self.__update_prefill_transform(enable=True, enable_chunking=enable_chunking)
                self.hash_params.pop("retain_full_kv", None)
                if "DeepseekV3ForCausalLM" not in (getattr(self.model.config, "architectures", None) or []):
                    self.hash_params["prefill_only"] = True
                    if enable_chunking:
                        self.hash_params["chunking"] = True
            else:
                self.__update_prefill_transform(False, retain_full_kv=kwargs.get("retain_full_kv", False))
                self.hash_params.pop("prefill_only", None)
                self.hash_params.pop("NUM_Q_BLOCKS", None)
                self.hash_params.pop("NUM_FFN_BLOCKS", None)
                self.hash_params.pop("ENABLE_OPT_SWA", None)
                self.hash_params.pop("chunking", None)
                self.hash_params.pop("chunking_seq_len", None)
                if kwargs.get("retain_full_kv", False):
                    self.hash_params["retain_full_kv"] = True
        #######################################################################

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len: int = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        supports_paged_attention = False
        # increase seq_len if using a larger number of blocks and set PagedAttention params if required
        if self.hash_params.get("blocking_kwargs", None):
            blocking_kwargs = self.hash_params["blocking_kwargs"]
            block_counts = [
                blocking_kwargs.num_q_blocks,
                blocking_kwargs.num_kv_blocks,
                blocking_kwargs.num_batch_blocks,
            ]
            max_blocks = max([b for b in block_counts if b is not None], default=1)
            block_size = -(-seq_len // max_blocks)
            seq_len = block_size * max_blocks
            num_kv_blocks = self.hash_params["blocking_kwargs"].num_kv_blocks
            supports_paged_attention = self.hash_params["blocking_kwargs"].paged_attention

            seq_len = kv_block_size = (
                -(-blocking_kwargs.ctx_len // num_kv_blocks) if supports_paged_attention else seq_len
            )

        # TODO: Remove this hack ##################
        if dynamo:
            # torch.export requires example inputs to satisfy dynamic_shapes min=2; gpt_oss non-CB keeps bs=1.
            bs = max(2, bs)
        ###########################################

        #### HANDLE KV CACHE SHAPE ################
        kv_cache_shape = get_padding_shape_from_config(
            self.model.config,
            fbs if self.continuous_batching else bs,
            seq_len * 2 if getattr(self, "dflash_dlm", None) else seq_len,
        )
        if dynamo:
            kv_cache_shape = list(kv_cache_shape)
            kv_cache_shape[1 if len(kv_cache_shape) == 3 else 2] = max(
                2, kv_cache_shape[1 if len(kv_cache_shape) == 3 else 2]
            )
        ############################################

        ############################################
        # Handle seq_len for export to succeed based on expert-parallel and blocking for loop requirements
        ############################################
        # TODO: We should change the ctx_len i.e. the past_key_values shape based on num_kv_blocks which is not yet handled.
        if self.model.config.model_type in SPECIALIZED_DISAGG_SERVING_MODEL_ARCH:
            if prefill_only and "DeepseekV3ForCausalLM" not in (
                getattr(self.model.config, "architectures", None) or []
            ):
                if enable_chunking:
                    if self.model.config.model_type in {"qwen3_moe", "gpt_oss", "glm4_moe"}:
                        seq_len = max(prefill_seq_len or 0, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)

                elif self.model.config.model_type == "gpt_oss":
                    seq_len = self.handle_gpt_oss_env_variable_legacy_burden(prefill_seq_len)
                sliding_window = getattr(self.model.config, "sliding_window", None)
                kv_cache_shape[2] = (
                    seq_len + (sliding_window if sliding_window is not None else 0) if enable_chunking else seq_len
                )
            else:
                if kwargs.get("retain_full_kv", False):
                    sliding_window = getattr(self.model.config, "sliding_window", None)
                    kv_cache_shape[2] = seq_len + (sliding_window if sliding_window is not None else 0)

        expert_parallel_is_triggered = self.hash_params.get("moe_prefill_flavour") == MoEFlavour.EXPERT_PARALLEL
        blocking_enabled = self.hash_params.get("blocking_kwargs", None) is not None
        for_loop_number_forced_by_expert_parallel = self.hash_params.get("moe_prefill_num_packed_chunks", None)
        if blocking_enabled:
            max_blocks = -1
            for key, num_blocks in self.hash_params.get("blocking_kwargs").__dict__.items():
                if isinstance(num_blocks, int) and key in ["num_kv_blocks", "num_q_blocks"]:
                    max_blocks = max(max_blocks, num_blocks)
            block_size = -(-seq_len // max_blocks)
            for_loop_number_forced_by_blocking = block_size * max_blocks

        if expert_parallel_is_triggered:
            seq_len = (for_loop_number_forced_by_expert_parallel * seq_len) // math.gcd(
                for_loop_number_forced_by_expert_parallel, seq_len
            )
        if blocking_enabled:
            seq_len = (for_loop_number_forced_by_blocking * seq_len) // math.gcd(
                for_loop_number_forced_by_blocking, seq_len
            )
        ##################################

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "past_key_values": [[] for _ in range(self.num_layers)],
        }
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }

        if getattr(self, "dflash_dlm", None):
            example_inputs["target_hidden"] = torch.ones((bs, seq_len, self.hidden_size), dtype=torch.float)
            example_inputs["position_ids"] = (
                torch.arange(seq_len, 2 * seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1)
            )
            example_inputs["position_ids_target"] = (
                torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1)
            )
            dynamic_axes["target_hidden"] = {0: "batch_size", 1: "seq_len"}
            dynamic_axes["position_ids_target"] = {0: "batch_size", 1: "seq_len"}

        if self.ccl_enabled:
            example_inputs["comp_ctx_lengths"] = torch.randint(0, 127, (seq_len,), dtype=torch.int64)
            dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if len(kv_cache_shape) == 3:  # For GPTBigCode arch the pkv is 3d
            pkv_dynamic_axes = {
                0: "full_batch_size" if self.continuous_batching else "batch_size",
                1: "ctx_len",
            }
        else:  # pkv is 4d
            pkv_dynamic_axes = {
                0: "full_batch_size" if self.continuous_batching else "batch_size",
                2: "ctx_len",
            }
        output_names = []
        if self.model.qaic_config is not None and self.model.qaic_config.get("include_sampler", False):
            if self.model.qaic_config.get("return_pdfs", False):
                output_names.append("probs")
            output_names.append("next_tokens")
        else:
            output_names.append("logits")

        if supports_paged_attention:
            batch, num_kv_heads, CL, dh = kv_cache_shape
            total_num_kv_blocks = batch * num_kv_blocks
            kv_cache_shape = [total_num_kv_blocks, num_kv_heads, kv_block_size, dh]
            example_inputs["block_table"] = torch.arange((bs * num_kv_blocks), dtype=torch.int64).view(
                bs, num_kv_blocks
            )
            example_inputs["slot_id"] = torch.zeros(bs, dtype=torch.int64)
            dynamic_axes["block_table"] = {0: "batch_size"}
            dynamic_axes["slot_id"] = {0: "batch_size"}
            # Assuming 4d pkv, might have to recheck for GPTBigCode with 3d pkv
            pkv_dynamic_axes = {
                0: "total_num_kv_blocks",
                2: "kv_block_size",
            }

        # TODO Update the get_padding_shape_from_config method to handle the case when the model config has attention_chunk_size or sliding_window and it should return a list of shapes for each layer
        if hasattr(self.model, "get_onnx_retained_state_specs"):
            retained_state_specs = self.model.get_onnx_retained_state_specs(
                batch_size=fbs if self.continuous_batching else bs,
                seq_len=seq_len,
                kv_cache_shape=kv_cache_shape,
                continuous_batching=self.continuous_batching,
                retain_full_kv=kwargs.get("retain_full_kv", False)
                or (prefill_only and kwargs.get("enable_chunking", False)),
            )
            example_inputs["past_key_values"] = retained_state_specs["past_key_values"]
            dynamic_axes.update(retained_state_specs["dynamic_axes"])
            output_names.extend(retained_state_specs["output_names"])
        elif (
            hasattr(self.model.config, "model_type")
            and self.model.config.model_type in DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH
            and hasattr(self.model, "get_dummy_pkv_cache")
        ):
            pkv_cache = self.model.get_dummy_pkv_cache(
                self.model.config, fbs if self.continuous_batching else bs, seq_len
            )
            for i in range(self.num_layers):
                for kv in ["key", "value"]:
                    example_inputs["past_key_values"][i].append(
                        torch.zeros(pkv_cache[0][0].shape, dtype=self.model.config.torch_dtype)
                    )
                    dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes
                    output_names.append(f"past_{kv}.{i}_RetainedState")

        else:
            # HACK: create common function for this including above if condition code
            pkv_dynamic_axes = (
                self.model.get_pkv_dynamic_axes(
                    retain_full_kv=kwargs.get("retain_full_kv", False)
                    or (prefill_only and kwargs.get("enable_chunking", False)),
                    continuous_batching=self.continuous_batching,
                )
                if hasattr(self.model, "get_pkv_dynamic_axes")
                else pkv_dynamic_axes
            )
            pkv_dynamic_axes = (
                [pkv_dynamic_axes] * self.model.config.num_hidden_layers
                if isinstance(pkv_dynamic_axes, dict)
                else pkv_dynamic_axes
            )

            for i in range(self.num_layers):
                for kv in ["key", "value"]:
                    example_inputs["past_key_values"][i].append(
                        torch.zeros(kv_cache_shape, dtype=self.model.config.torch_dtype)
                    )
                    dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes[i]
                    output_names.append(f"past_{kv}.{i}_RetainedState")

        if "DeepseekV3ForCausalLM" in (getattr(self.model.config, "architectures", None) or []):
            if self.model.qaic_config is not None and self.model.qaic_config.get("mla_absorption", None) is not None:
                mla_absorption = self.model.qaic_config["mla_absorption"]
                cache_compressed = mla_absorption.get("cache_compressed", False)
            else:
                cache_compressed = False
            pkv_cache = self.model.get_dummy_pkv_cache(
                self.model.config, fbs if self.continuous_batching else bs, seq_len
            )
            if cache_compressed:
                example_inputs = {k: v for k, v in example_inputs.items() if "past" not in k}
                dynamic_axes = {k: v for k, v in dynamic_axes.items() if "past" not in k}
                output_names = [v for v in output_names if "past" not in v]
                example_inputs["compressed_kvs"] = [[] for _ in range(self.num_layers)]
                for i in range(self.num_layers):
                    example_inputs["compressed_kvs"][i].append(
                        torch.zeros(pkv_cache[0][0].shape, dtype=self.model.config.torch_dtype)
                    )
                    example_inputs["compressed_kvs"][i].append(
                        torch.zeros(pkv_cache[0][1].shape, dtype=self.model.config.torch_dtype)
                    )
                    dynamic_axes[f"compressed_kv.{i}"] = {
                        0: "full_batch_size" if self.continuous_batching else "batch_size",
                        2: "ctx_len",
                    }
                    dynamic_axes[f"k_pe.{i}"] = {
                        0: "full_batch_size" if self.continuous_batching else "batch_size",
                        2: "ctx_len",
                    }
                    output_names.append(f"compressed_kv.{i}_RetainedState")
                    output_names.append(f"k_pe.{i}_RetainedState")
            else:
                example_inputs["past_key_values"] = [[] for _ in range(self.num_layers)]
                for i in range(self.num_layers):
                    example_inputs["past_key_values"][i].append(
                        torch.zeros(pkv_cache[0][0].shape, dtype=self.model.config.torch_dtype)
                    )
                    example_inputs["past_key_values"][i].append(
                        torch.zeros(pkv_cache[0][1].shape, dtype=self.model.config.torch_dtype)
                    )

        if self.continuous_batching:
            example_inputs["batch_index"] = torch.arange(bs).view(bs, 1)
            dynamic_axes["batch_index"] = {0: "batch_size"}

        if self.is_tlm:
            nlk = constants.ONNX_EXPORT_EXAMPLE_NLK  # Number of Logits to Keep
            example_inputs["num_logits_to_keep"] = torch.arange(nlk).view(nlk, 1)
            dynamic_axes["num_logits_to_keep"] = {0: "num_logits_to_keep"}

        if self.model.qaic_config is not None and self.model.qaic_config.get("include_sampler", False):
            example_inputs, output_names, dynamic_axes = get_sampling_inputs_and_outputs(
                example_inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                continuous_batching=self.continuous_batching,
                vocab_size=self.model.config.vocab_size,
                qaic_config=self.model.qaic_config,
            )

        # transformers>=5.3 Gemma3 models require Cache I/O internally; keep tensor/list
        # inputs for tracing and bridge to cache objects inside a temporary wrapper.
        if (
            hasattr(self.model.config, "model_type")
            and str(self.model.config.model_type).startswith("gemma3")
            and not getattr(self.model, "_qeff_export_gemma3_cache_patch", False)
        ):
            import functools
            import inspect

            from transformers.cache_utils import Cache

            model_forward = self.model.forward
            model_forward_sig = inspect.signature(model_forward)

            @functools.wraps(model_forward)
            def _qeff_patched_forward(*args, **kwargs):
                def _legacyify_cache(obj):
                    if hasattr(obj, "to_legacy_cache"):
                        return obj.to_legacy_cache()
                    if isinstance(obj, Cache):
                        if hasattr(obj, "to_legacy_cache"):
                            return obj.to_legacy_cache()
                        if hasattr(obj, "layers"):
                            legacy_cache = ()
                            for layer in obj.layers:
                                keys = getattr(layer, "keys", None)
                                values = getattr(layer, "values", None)
                                legacy_cache += ((keys, values),)
                            return legacy_cache
                    if isinstance(obj, (tuple, list)):
                        return type(obj)(_legacyify_cache(x) for x in obj)
                    return obj

                bound_args = model_forward_sig.bind_partial(*args, **kwargs)
                outputs = model_forward(*bound_args.args, **bound_args.kwargs)
                if torch.onnx.is_in_onnx_export():
                    if hasattr(outputs, "logits") and hasattr(outputs, "past_key_values"):
                        return outputs.logits, _legacyify_cache(outputs.past_key_values)
                    return _legacyify_cache(outputs)
                return outputs

            self.model.forward = _qeff_patched_forward
            self.model._qeff_export_gemma3_cache_patch = True

        # Optionally inject a user-provided infix token into the LLM KV-cache retained-state names
        # (e.g. past_key.0_RetainedState -> past_key.0_<prefix>_RetainedState) so downstream consumers
        # (vLLM disaggregated serving) can regex-select only the LLM KV buffers for transfer.
        kv_cache_prefix = validate_kv_cache_prefix(kv_cache_prefix)
        if kv_cache_prefix:
            output_names = apply_kv_cache_prefix(output_names, kv_cache_prefix)
            self.hash_params["kv_cache_prefix"] = kv_cache_prefix

        if getattr(self, "dflash_tlm", None):
            output_names.append("hidden_states")

        if QEFFBaseModel._layerwise_active:
            return self._export_layerwise(
                example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=export_dir,
                use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
                offload_pt_weights=kwargs.get("offload_pt_weights", True),
                prefill_only=prefill_only,
                enable_chunking=enable_chunking,
                num_cores=num_cores,
                qaic_config=qaic_config,
                prefill_seq_len=prefill_seq_len,
                kv_cache_prefix=kv_cache_prefix,
                _layerwise_cache_probe=kwargs.get("_layerwise_cache_probe", False),
            )
        else:
            return self._export(
                example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=export_dir,
                use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
                dynamo=dynamo,
                offload_pt_weights=kwargs.get("offload_pt_weights", True),
            )

    def build_prefill_specialization(
        self,
        prefill_seq_len: int = 32,
        ctx_len: int = 128,
        comp_ctx_lengths: int | None = None,
        batch_size: int = 1,
        kv_cache_batch_size: int | None = None,
        full_batch_size: int | None = None,
        **kwargs,
    ):
        """
        Builds a dictionary representing a compilation specialization for the prefill phase.

        Parameters
        ----------
        prefill_seq_len : int, optional
            Length of the prefill prompt. Default is 32.
        ctx_len : int, optional
            Maximum context length the compiled model can remember. Default is 128.
        batch_size : int, optional
            Batch size for the prefill. Default is 1.
        kv_cache_batch_size : int, optional
            Batch size for KV cache. If not provided, it defaults based on `full_batch_size` or `batch_size`.
        full_batch_size : int, optional
            Continuous batching batch size. Used if `continuous_batching` is enabled. Default is None.

        Returns
        -------
        Dict[str, Union[int, str]]
            A dictionary defining the prefill specialization.
        """
        if not self.continuous_batching:
            exec_batch_size = batch_size
        elif getattr(self, "dflash_dlm", None):
            # DFlash DLM: route decode_bsz rows via batch_index; use full_batch_size.
            exec_batch_size = full_batch_size or batch_size
        elif prefill_seq_len == 1:
            exec_batch_size = full_batch_size
        else:
            exec_batch_size = 1

        if hasattr(self.model, "get_specializations"):
            spec = self.model.get_specializations(
                batch_size=exec_batch_size,
                prefill_seq_len=prefill_seq_len,
                ctx_len=ctx_len,
                **kwargs,
            )[0]
        else:
            spec = {
                "batch_size": exec_batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
            }
        if comp_ctx_lengths is not None:
            spec["comp_ctx_lengths"] = comp_ctx_lengths
        spec["num_logits_to_keep"] = 1 if self.is_tlm else None
        if self.continuous_batching:
            spec["full_batch_size"] = kv_cache_batch_size
        else:
            spec["batch_size"] = kv_cache_batch_size
        # TODO: remove this; not required
        if full_batch_size:
            spec["full_batch_exec_size"] = exec_batch_size
        if self.model.qaic_config is not None and "paged" in self.model.qaic_config.get("blocking_mode", ""):
            num_kv_blocks = self.model.qaic_config["num_kv_blocks"]
            spec["num_kv_blocks"] = num_kv_blocks
            spec["total_num_kv_blocks"] = kv_cache_batch_size * num_kv_blocks
            spec["kv_block_size"] = -(-ctx_len // num_kv_blocks)
        result = {k: v for k, v in spec.items() if v is not None}
        result["_graph_name"] = "Decode" if prefill_seq_len == 1 and kwargs.get("prefill_only") is False else "Prefill"
        return result

    def build_decode_specialization(
        self,
        prefill_seq_len: int = 32,
        ctx_len: int = 128,
        comp_ctx_lengths: int | None = None,
        batch_size: int = 1,
        kv_cache_batch_size: int | None = None,
        full_batch_size: int | None = None,
        num_speculative_tokens: int | None = None,
        dflash_block_size: int | None = None,
        **kwargs,
    ):
        """
        Builds a dictionary representing a compilation specialization for the decode phase.

        Parameters
        ----------
        prefill_seq_len : int, optional
            Length of the prefill prompt. Used to avoid duplicate specializations. Default is 32.
        ctx_len : int, optional
            Maximum context length the compiled model can remember. Default is 128.
        batch_size : int, optional
            Batch size for the decode phase. Default is 1.
        kv_cache_batch_size : int, optional
            Batch size for KV cache. If not provided, it defaults based on `full_batch_size` or `batch_size`.
        full_batch_size : int, optional
            Continuous batching batch size. Used if `continuous_batching` is enabled. Default is None.
        num_speculative_tokens : int, optional
            Number of speculative tokens for Speculative Decoding Target Language Model. Default is None.

        Returns
        -------
        Optional[Dict[str, Union[int, str]]]
            A dictionary defining the decode specialization, or None if it would be a duplicate
            of the prefill specialization (e.g., if prefill_seq_len is 1 and not continuous batching).
        """
        decode_seq_len = (num_speculative_tokens + 1) if self.is_tlm else 1
        if decode_seq_len == prefill_seq_len and not self.continuous_batching:
            return None
        if hasattr(self.model, "get_specializations"):
            spec = self.model.get_specializations(
                batch_size=full_batch_size if self.continuous_batching else batch_size,
                prefill_seq_len=(num_speculative_tokens + 1) if self.is_tlm else 1,
                ctx_len=ctx_len,
            )[1]
        else:
            spec = {
                "batch_size": full_batch_size if self.continuous_batching else batch_size,
                "seq_len": (num_speculative_tokens + 1) if self.is_tlm else 1,
                "ctx_len": ctx_len,
            }
        if comp_ctx_lengths is not None:
            spec["comp_ctx_lengths"] = comp_ctx_lengths

        spec["num_logits_to_keep"] = (num_speculative_tokens + 1) if self.is_tlm else None

        if getattr(self, "dflash_tlm", None) or getattr(self, "dflash_dlm", None):
            spec["seq_len"] = dflash_block_size

        if self.continuous_batching:
            spec["full_batch_size"] = kv_cache_batch_size
        else:
            spec["batch_size"] = kv_cache_batch_size
        if self.model.qaic_config is not None and "paged" in self.model.qaic_config.get("blocking_mode", ""):
            num_kv_blocks = self.model.qaic_config["num_kv_blocks"]
            spec["num_kv_blocks"] = num_kv_blocks
            spec["total_num_kv_blocks"] = kv_cache_batch_size * num_kv_blocks
            spec["kv_block_size"] = -(-ctx_len // num_kv_blocks)
        result = {k: v for k, v in spec.items() if v is not None}
        result["_graph_name"] = "Decode"
        return result

    def compile(
        self,
        onnx_path: str | None = None,
        compile_dir: str | None = None,
        *,
        prefill_seq_len: int = 32,
        ctx_len: int = 128,
        comp_ctx_lengths_prefill: list[int] | None = None,
        comp_ctx_lengths_decode: list[int] | None = None,
        batch_size: int = 1,
        full_batch_size: int | None = None,
        kv_cache_batch_size: int | None = None,
        dflash_block_size: int | None = None,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        num_speculative_tokens: int | list[int] | None = None,
        prefill_only: bool | None = None,
        use_onnx_subfunctions: bool = False,
        offload_pt_weights: bool | None = True,
        enable_chunking: bool | None = False,
        retain_full_kv: bool | None = None,
        layerwise: bool = False,
        layerwise_window_size: int = 1,
        kv_cache_prefix: Optional[str] = None,
        artifacts: bool = False,
        **compiler_options,
    ) -> str:
        """

        Compile the exported ONNX model using the Cloud AI 100 Platform SDK compiler.

        This method generates a ``qpc`` package. If the model has not been exported yet,
        this method will handle the export process. Additional arguments for the `qaic-compile`
        compiler can be passed as keyword arguments.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        compile_dir : str, optional
            Directory to save the generated QPC package. If not provided, a default directory is used.
        prefill_seq_len : int, optional
            Length of the prefill prompt. Default is 32.
        ctx_len : int, optional
            Maximum context length the compiled model can remember. Default is 128.
        batch_size : int, optional
            Batch size. Default is 1.
        full_batch_size : int, optional
            Continuous batching batch size. Required if `continuous_batching=True` was
            set during `from_pretrained`.
        kv_cache_batch_size : int, optional
            Batch size for KV cache. If not provided, it defaults to `full_batch_size` (if
            continuous batching) or `batch_size`.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        mxint8_kv_cache : bool, optional
            Use MXINT8 compression for KV cache. Default is False.
        num_speculative_tokens : int or list[int], optional
            Proposal length(s) for Speculative Decoding Target Language Model.
            A plain int K is treated as ``[K]`` (backward compatible).
            Each value K generates a decode specialization with seq_len=K+1 and
            num_logits_to_keep=K+1. Include 0 to compile a cheap single-token fallback
            (e.g. ``[0, 3]`` for a fallback + full K=3 decode). Required if the model is
            configured as a Target Language Model (``is_tlm=True``).
        prefill_only : bool, optional
            If True, compiles only for the prefill stage. If False, compiles only for
            the decode stage. If None, compiles for both stages. Default is None.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers.

            **For QAIC Compiler:** Extra arguments for qaic-compile can be passed. Some common options include:

            - mos (int, optional): Effort level to reduce on-chip memory. Defaults to -1, meaning no effort. Defaults to -1.
            - aic_enable_depth_first (bool, optional): Enables DFS with default memory size. Defaults to False.
            - allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. Defaults to False.

            Params are converted to flags as below:

            - ``aic_num_cores=16`` -> ``-aic-num-cores=16``
            - ``convert_to_fp16=True`` -> ``-convert-to-fp16``

            **For QNN Compiler:** Following arguments can be passed as:

            - enable_qnn (bool): Enables QNN Compilation.
            - qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file.

        Returns
        -------
        str
            Path to the compiled QPC package.

        Raises
        ------
        TypeError
            If `prefill_only` is not a boolean.
            If `full_batch_size` is None when `continuous_batching` is True.
            If `num_speculative_tokens` is None or empty when the model is a TLM.
        ValueError
            If KV caching is requested without continuous batching (`full_batch_size`).
            If `include_sampler` is True and `num_speculative_tokens` contains a value > 0.
            If `num_speculative_tokens` is not an integer greater than 1.
            If `prefill_seq_len` is less than `num_speculative_tokens + 1` for TLM models.

        """
        if artifacts:
            compiler_options["artifacts"] = True
        reject_legacy_moe_prefill_packed_chunk_size(compiler_options)
        _ignore_public_mdp_ts_num_devices(compiler_options)
        enable_chunking = override_gptoss_prefill_chunking(self.model.config, prefill_only, enable_chunking)
        if layerwise:
            warnings.warn(
                "layerwise export is deprecated and will be removed in a future release. "
                "Use weight-free export (weight_free=True) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self._run_layerwise(
                final_compile=True,
                layerwise_window_size=layerwise_window_size,
                onnx_path=onnx_path,
                compile_dir=compile_dir,
                prefill_seq_len=prefill_seq_len,
                ctx_len=ctx_len,
                comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
                comp_ctx_lengths_decode=comp_ctx_lengths_decode,
                batch_size=batch_size,
                full_batch_size=full_batch_size,
                kv_cache_batch_size=kv_cache_batch_size,
                num_devices=num_devices,
                num_cores=num_cores,
                mxfp6_matmul=mxfp6_matmul,
                mxint8_kv_cache=mxint8_kv_cache,
                num_speculative_tokens=num_speculative_tokens,
                prefill_only=prefill_only,
                use_onnx_subfunctions=use_onnx_subfunctions,
                offload_pt_weights=offload_pt_weights,
                enable_chunking=enable_chunking,
                retain_full_kv=retain_full_kv,
                kv_cache_prefix=kv_cache_prefix,
                **compiler_options,
            )
        if self.model.qaic_config is not None and self.model.qaic_config.get("mla_absorption", None) is not None:
            mla_absorption = self.model.qaic_config["mla_absorption"]
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False
        if (
            self.model.qaic_config is not None
            and self.model.qaic_config.get("mla_absorption", None) is not None
            and not cache_compressed
        ):
            logger.warning("mla_absorption will be ignored as cache_compressed is set to False")
        if (kv_cache_batch_size or full_batch_size) and not self.continuous_batching:
            logger.warning(
                "`kv_cache_batch_size` or `full_batch_size` is being passed"
                "This will be ignored as `continuous_batching` is set to `False` in `from_pretrained`"
            )

        if prefill_only is None or not prefill_only:
            if self.continuous_batching and full_batch_size is None:
                raise TypeError("`full_batch_size` is required when `continuous_batching=True`.")
        else:
            if self.continuous_batching and kv_cache_batch_size is None and full_batch_size is None:
                raise ValueError(
                    "Please pass valid integer for kv_cache_batch_size or full_batch_size, both have same meaning, as continuous_batching is enabled for prefill-only model"
                )

        # Infer kv_cache_batch_size if not provided
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size

        # if ccl_enabled is True read Compute-Context-Length lists
        if self.ccl_enabled:
            if comp_ctx_lengths_prefill is None and comp_ctx_lengths_decode is None:
                logger.info("Auto-generating CCL-prefill and CCL-decode lists based on Context Length (CL).")
            self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode, ctx_len = process_ccl_specializations(
                comp_ctx_lengths_prefill, comp_ctx_lengths_decode, ctx_len, prefill_seq_len, enable_chunking
            )
        # For supporting VLLM and Disaggregated with CCL
        elif comp_ctx_lengths_prefill is not None or comp_ctx_lengths_decode is not None:
            self.ccl_enabled = True
            if isinstance(comp_ctx_lengths_prefill, str):
                import ast

                try:
                    # Safely evaluate the string to a Python list for disaggregated input
                    self.comp_ctx_lengths_prefill = ast.literal_eval(comp_ctx_lengths_prefill)
                    self.comp_ctx_lengths_decode = ast.literal_eval(comp_ctx_lengths_decode)

                except (ValueError, SyntaxError):
                    raise ValueError("Invalid format for comp_ctx_lengths. Expected a list-like string.")
            else:
                self.comp_ctx_lengths_prefill = comp_ctx_lengths_prefill
                self.comp_ctx_lengths_decode = comp_ctx_lengths_decode

            self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode, ctx_len = process_ccl_specializations(
                self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode, ctx_len, prefill_seq_len, enable_chunking
            )
        # --- Validation ---
        if prefill_only is not None and not isinstance(prefill_only, bool):
            raise TypeError("`prefill_only` must be a boolean.")

        _decode_ks = (
            sorted(set(num_speculative_tokens))
            if isinstance(num_speculative_tokens, (list, tuple))
            else ([num_speculative_tokens] if num_speculative_tokens is not None else None)
        )

        if self.is_tlm:
            _max_k = _decode_ks[-1] if _decode_ks else None
            validated_k = self.check_and_get_num_speculative_tokens(_max_k, prefill_seq_len)
            if validated_k is not None and validated_k != _max_k:
                # speculative_config in model.config overrides num_speculative_tokens.
                # Warn if the user passed a list — the extra values are discarded.
                if _decode_ks is not None and len(_decode_ks) > 1:
                    discarded = [k for k in _decode_ks if k != validated_k]
                    logger.warning(
                        f"speculative_config in model.config fixes num_speculative_tokens={validated_k}. "
                        f"Ignoring user-supplied values {discarded}. "
                        f"Pass num_speculative_tokens={validated_k} (or [{validated_k}]) to suppress this warning."
                    )
                _decode_ks = [validated_k]

        if (
            self.model.qaic_config is not None
            and self.model.qaic_config.get("include_sampler", False)
            and _decode_ks is not None
            and max(_decode_ks) > 0
        ):
            raise ValueError("Currently, sampler does not support `num_speculative_tokens` > 0.")

        if retain_full_kv and self.model.config.model_type not in SPECIALIZED_DISAGG_SERVING_MODEL_ARCH:
            logger.warning(
                "retain_full_kv=True is only supported for specialized disaggregated serving models "
                f"{sorted(SPECIALIZED_DISAGG_SERVING_MODEL_ARCH)}; ignoring it for model_type "
                f"'{self.model.config.model_type}'."
            )
            retain_full_kv = False

        # --- Specializations ---
        specializations = []
        if prefill_only is None or prefill_only or prefill_seq_len == 1:
            # TODO: we are handling decode-only case inside prefill call which is utterly mis-leading
            if self.comp_ctx_lengths_prefill is not None or self.comp_ctx_lengths_decode is not None:
                ccl_lengths = self.comp_ctx_lengths_decode if prefill_seq_len == 1 else self.comp_ctx_lengths_prefill
                # Adding elements from self.comp_ctx_lengths_prefill to prefill_specialization
                for i in range(len(ccl_lengths)):
                    specializations.append(
                        self.build_prefill_specialization(
                            prefill_seq_len=prefill_seq_len,
                            ctx_len=ctx_len,
                            comp_ctx_lengths=ccl_lengths[i],
                            batch_size=batch_size,
                            kv_cache_batch_size=kv_cache_batch_size,
                            full_batch_size=full_batch_size,
                            prefill_only=prefill_only,
                        )
                    )

            else:
                specializations.append(
                    self.build_prefill_specialization(
                        prefill_seq_len=prefill_seq_len,
                        ctx_len=ctx_len,
                        batch_size=batch_size,
                        kv_cache_batch_size=kv_cache_batch_size,
                        full_batch_size=full_batch_size,
                        prefill_only=prefill_only,
                        enable_chunking=enable_chunking,
                    )
                )

        if (prefill_only is None or not prefill_only) and prefill_seq_len != 1:
            if _decode_ks is not None and self.is_tlm:
                # TLM multi-spec path: one decode specialization per K in num_speculative_tokens.
                # CCL (comp_ctx_lengths) + multi-spec TLM is not yet supported: the per-K call
                # to build_decode_specialization would need to iterate over CCL values, producing
                # len(decode_ks) × len(comp_ctx_lengths_decode) decode specializations whose
                # naming and ordering is untested.  Reject early so users get a clear error
                # instead of a silently wrong QPC.
                if self.comp_ctx_lengths_decode is not None:
                    raise NotImplementedError(
                        "TLM multi-spec (num_speculative_tokens as a list) combined with "
                        "comp_ctx_lengths_decode is not yet supported. Pass a plain int for "
                        "num_speculative_tokens when using CCL."
                    )
                for k in _decode_ks:
                    spec = self.build_decode_specialization(
                        num_speculative_tokens=k,
                        prefill_seq_len=prefill_seq_len,
                        ctx_len=ctx_len,
                        batch_size=batch_size,
                        kv_cache_batch_size=kv_cache_batch_size,
                        full_batch_size=full_batch_size,
                        dflash_block_size=dflash_block_size,
                    )
                    if spec is not None:
                        specializations.append(spec)

            elif self.comp_ctx_lengths_decode is not None:
                # CCL loop (non-TLM)
                for i in range(len(self.comp_ctx_lengths_decode)):
                    decode_spec = self.build_decode_specialization(
                        prefill_seq_len=prefill_seq_len,
                        ctx_len=ctx_len,
                        comp_ctx_lengths=self.comp_ctx_lengths_decode[i],
                        batch_size=batch_size,
                        kv_cache_batch_size=kv_cache_batch_size,
                        full_batch_size=full_batch_size,
                        num_speculative_tokens=None,
                        dflash_block_size=dflash_block_size,
                    )
                    if decode_spec:
                        specializations.append(decode_spec)

            else:
                decode_spec = self.build_decode_specialization(
                    prefill_seq_len=prefill_seq_len,
                    ctx_len=ctx_len,
                    batch_size=batch_size,
                    kv_cache_batch_size=kv_cache_batch_size,
                    full_batch_size=full_batch_size,
                    num_speculative_tokens=num_speculative_tokens,
                    dflash_block_size=dflash_block_size,
                    prefill_only=prefill_only,
                )
                if decode_spec:
                    specializations.append(decode_spec)

        if kw_spec := compiler_options.pop("specializations", None):
            specializations = kw_spec

        target_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else CUSTOM_IO_DTYPE_MAP[target_dtype]
        # --- Compilation ---
        # When a KV-cache prefix is requested, the exported buffers are named
        # past_key.{i}_<prefix> (input) and past_key.{i}_<prefix>_RetainedState (output); the custom_io
        # keys must match those names so the compiler pairs and retains them correctly.
        kv_cache_prefix = validate_kv_cache_prefix(kv_cache_prefix)
        custom_io = {}
        if not cache_compressed:
            kv_infix = f"_{kv_cache_prefix}" if kv_cache_prefix else ""
            retained_state_names = (
                self.model.get_retained_state_names()
                if hasattr(self.model, "get_retained_state_names")
                else [f"past_{kv}.{i}" for i in range(self.num_layers) for kv in ["key", "value"]]
            )
            for state_name in retained_state_names:
                output_name = _compile_io_name(
                    f"{state_name}{kv_infix}_RetainedState",
                    use_onnx_subfunctions=use_onnx_subfunctions,
                )
                _add_retained_state_custom_io(
                    custom_io,
                    output_name,
                    dtype=kv_cache_dtype,
                    use_onnx_subfunctions=False,
                )
        else:
            kv_infix = f"_{kv_cache_prefix}" if kv_cache_prefix else ""
            for i in range(self.num_layers):
                for prefix in ("compressed_kv", "k_pe"):
                    output_name = _compile_io_name(
                        f"{prefix}.{i}{kv_infix}_RetainedState",
                        use_onnx_subfunctions=use_onnx_subfunctions,
                    )
                    _add_retained_state_custom_io(
                        custom_io,
                        output_name,
                        dtype=kv_cache_dtype,
                        use_onnx_subfunctions=False,
                    )

        custom_io = _filter_custom_io_for_onnx(custom_io, onnx_path)

        qpc_path = self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=_should_convert_to_fp16(target_dtype, compiler_options),
            mxfp6_matmul=mxfp6_matmul,
            custom_io=custom_io,
            mdp_ts_num_devices=num_devices,
            num_speculative_tokens=num_speculative_tokens,
            aic_num_cores=num_cores,
            mxint8_kv_cache=mxint8_kv_cache,
            use_onnx_subfunctions=use_onnx_subfunctions,
            prefill_only=prefill_only,
            offload_pt_weights=offload_pt_weights,
            enable_chunking=enable_chunking,
            retain_full_kv=retain_full_kv,
            kv_cache_prefix=kv_cache_prefix,
            **compiler_options,
        )

        return qpc_path

    # FIXME: Update this method to match with transformers AutoModelForCausalLM.generate
    def generate(
        self,
        tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
        prompts: list[str],
        device_ids: list[int] | None = None,
        runtime_ai100: bool = True,
        artifacts: bool = False,
        **kwargs,
    ) -> Union[CloudAI100ExecInfoNew, Path]:
        """
        Generate output by executing the compiled QPC on Cloud AI 100 hardware.

        This method runs sequential execution based on the compiled model's batch size and the number of prompts.
        If the number of prompts is not divisible by the batch size, the last batch will be dropped.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizer or PreTrainedTokenizerFast
            Tokenizer for the model.
        prompts : list of str
            List of prompts to generate output for.
        device_ids : list of int, optional
            Device IDs for running the QPC. Defaults to `[0]` if not specified.
        runtime_ai100 : bool, optional
            Whether to use AI 100 runtime. Default is True.
        **kwargs :
            Additional keyword arguments. Currently supports:
            - `generation_len (int, optional)`: The maximum number of tokens to generate.

        artifacts : bool, optional
            Write first-prefill ``qaic-runner`` inputs without constructing a runtime session.
        Returns
        -------
        CloudAI100ExecInfoNew
            Output from the AI 100 runtime, containing generated IDs and performance metrics.

        Raises
        ------
        TypeError
            If the QPC path is not set (i.e., `compile` was not run).
        NotImplementedError
            If `runtime_ai100` is False.
        """
        if artifacts:
            return write_causal_lm_runner_bundle(
                model=self,
                tokenizer=tokenizer,
                prompts=prompts,
                sampling_params=kwargs.get("sampling_params"),
            )
        if runtime_ai100:
            if not isinstance(self.qpc_path, Path):
                raise TypeError("Please run compile API first!")
            generation_len = kwargs.pop("generation_len", None)
            return QEfficient.cloud_ai_100_exec_kv(
                tokenizer=tokenizer,
                qpc_path=self.qpc_path,
                prompt=prompts,
                comp_ctx_lengths_prefill=self.comp_ctx_lengths_prefill,
                comp_ctx_lengths_decode=self.comp_ctx_lengths_decode,
                device_ids=device_ids,
                generation_len=generation_len,
                automation=kwargs.pop("automation", False),
                iteration=kwargs.pop("iteration", 1),
                is_tlm=self.is_tlm,
                **kwargs,
            )
        else:
            raise NotImplementedError("Only AI_100 runtime is supported right now via generate API")

    def check_and_get_num_speculative_tokens(self, num_speculative_tokens: int | None, prefill_seq_len: int):
        """
        Validates and retrieves the number of speculative tokens for TLM models.

        Parameters
        ----------
        num_speculative_tokens : int, optional
            The number of speculative tokens provided by the user.
        prefill_seq_len : int
            The prefill sequence length.

        Returns
        -------
        int
            The determined number of speculative tokens.

        Raises
        ------
        TypeError
            If `num_speculative_tokens` is None when `is_tlm` is True.
        ValueError
            If `num_speculative_tokens` is not an integer greater than 1.
            If `prefill_seq_len` is less than `num_speculative_tokens + 1`.
        """
        if not self.is_tlm:
            return None
        if hasattr(self.model.config, "speculative_config"):
            num_speculative_tokens_ = self.model.config.speculative_config["num_speculative_tokens"]
            if num_speculative_tokens is not None:
                logger.warning(
                    f"arg `num_speculative_tokens` is a fixed value of {num_speculative_tokens_} for this model."
                    f" Passed value of {num_speculative_tokens} will be ignored."
                )
            num_speculative_tokens = num_speculative_tokens_
        elif num_speculative_tokens is None:
            raise TypeError("missing required argument `num_speculative_tokens` as `is_tlm` instance variable is True.")

        if not isinstance(num_speculative_tokens, int) and num_speculative_tokens:
            raise ValueError(
                f"`num_speculative_tokens` arg should be an integer greater than 1, got {num_speculative_tokens}"
            )
        num_logits_to_keep = num_speculative_tokens + 1
        if prefill_seq_len < num_logits_to_keep:
            raise ValueError(
                f"sequence length ({prefill_seq_len}) must be at least `num_speculative_tokens+1` ({num_logits_to_keep})"
            )
        return num_speculative_tokens


class QEFFAutoModelForSpeechSeq2Seq(QEFFTransformersBase, MultimodalUtilityMixin):
    """
    QEfficient class for sequence-to-sequence speech-to-text models (e.g., Whisper, Encoder-Decoder speech models).

    This class enables efficient export, compilation, and inference of speech models on Cloud AI 100 hardware.
    It is recommended to use the ``from_pretrained`` method for initialization.

    Example
    -------
    .. code-block:: python

        from datasets import load_dataset
        from transformers import AutoProcessor
        from QEfficient import QEFFAutoModelForSpeechSeq2Seq

        base_model_name = "openai/whisper-tiny"
        ## STEP 1 -- load audio sample, using a standard english dataset, can load specific files if longer audio needs to be tested; also load initial processor
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        data = ds[0]["audio"]["array"]
        # reshape to so shape corresponds to data with batch size 1
        data = data.reshape(-1)
        sample_rate = ds[0]["audio"]["sampling_rate"]
        processor = AutoProcessor.from_pretrained(base_model_name)

        ## STEP 2 -- init base model
        qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(base_model_name)

        ## STEP 3 -- export and compile model
        qeff_model.compile()

        ## STEP 4 -- generate output for loaded input and processor
        exec_info = qeff_model.generate(inputs=processor(data, sampling_rate=sample_rate, return_tensors="pt"), generation_len=25)

        ## STEP 5 (optional) -- use processor to decode output
        print(processor.batch_decode(exec_info.generated_ids)[0])
    """

    _hf_auto_class = AutoModelForSpeechSeq2Seq
    _pytorch_transforms = [CustomOpsTransform, AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, KVCacheTransform]
    _onnx_transforms = []

    def __init__(self, model: nn.Module, **kwargs):
        """
        Initialize a QEFFAutoModelForSpeechSeq2Seq instance.

        Parameters
        ----------
        model : nn.Module
            A PyTorch model with a sequence-to-sequence speech-to-text head (e.g., Whisper).
        **kwargs :
            Additional keyword arguments passed to the base class constructor.

        Raises
        ------
        TypeError
            If the model is not a supported speech-to-text model (i.e., not a `ForConditionalGeneration` model).
        """
        model_class_name = model.__class__.__name__

        if not (model_class_name.endswith("ForConditionalGeneration")):
            raise TypeError(f"Required pytorch module with ForConditionalGeneration, got {model_class_name}")

        model.config.use_cache = True
        super().__init__(model, **kwargs)
        self.num_layers = model.config.num_hidden_layers
        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    @property
    def get_model_config(self) -> dict:
        """
        Get the configuration dictionary of the underlying HuggingFace model.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        return self.model.config.__dict__

    def export(self, export_dir: str | None = None, **kwargs) -> str:
        """
        Export the model to ONNX format using ``torch.onnx.export``.

        This method prepares example inputs and dynamic axes based on the model configuration,
        then exports the model to an ONNX graph suitable for compilation and deployment on Cloud AI 100 hardware.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved.
            If not provided, the default export directory is used.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file.
        """
        inputs = self.model.get_dummy_inputs()
        dynamic_axes = self.model.get_onnx_dynamic_axes()
        output_names = self.model.get_output_names()
        return self._export(
            inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
        )

    def compile(
        self,
        onnx_path: str | None = None,
        compile_dir: str | None = None,
        *,
        prefill_seq_len: int | None = 1,
        encoder_ctx_len: int | None = None,
        ctx_len: int = 150,
        full_batch_size: int | None = None,
        kv_cache_batch_size: int | None = None,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        num_speculative_tokens: int | None = None,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compile the exported ONNX model using the Cloud AI 100 Platform SDK compiler.

        This method generates a ``qpc`` package. If the model has not been exported yet,
        this method will handle the export process. Additional arguments for the `qaic-compile`
        compiler can be passed as keyword arguments.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        compile_dir : str, optional
            Directory to save the generated QPC package.
        prefill_seq_len : int, optional
            Prefill sequence length. This parameter is typically not critically used for
            SpeechSeq2Seq models' decoder compilation as the first decoder input is `seq_len=1`.
            Default is 1.
        encoder_ctx_len : int, optional
            Maximum context length for the encoder part of the model. If None, it's inferred
            from the model configuration or defaults (e.g., 1500 for Whisper).
        ctx_len : int, optional
            Maximum decoder context length. This defines the maximum output sequence length
            the compiled model can handle. Default is 150.
        batch_size : int, optional
            Batch size. Default is 1.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        mxint8_kv_cache : bool, optional
            Use MXINT8 compression for KV cache. Default is False.
        full_batch_size : int, optional
            Not yet supported for this model.
        kv_cache_batch_size : int, optional
            Not yet supported for this model.
        num_speculative_tokens : int, optional
            Not yet supported for this model.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC.

            **For QAIC Compiler:** Extra arguments for qaic-compile can be passed. Some common options include:

            - mos (int, optional): Effort level to reduce on-chip memory. Defaults to -1, meaning no effort. Defaults to -1.
            - aic_enable_depth_first (bool, optional): Enables DFS with default memory size. Defaults to False.
            - allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. Defaults to False.

            Params are converted to flags as below:

            - ``aic_num_cores=16`` -> ``-aic-num-cores=16``
            - ``convert_to_fp16=True`` -> ``-convert-to-fp16``

        Returns
        -------
        str
            Path to the compiled QPC package.

        """
        _ignore_public_mdp_ts_num_devices(compiler_options)
        specializations, compiler_options = self.model.get_specializations(
            batch_size,
            encoder_ctx_len,
            ctx_len,
            **compiler_options,
        )

        if full_batch_size:
            logger.warning("Continuous batching is not yet enabled for AutoModelForSpeechSeq2Seq")

        if kv_cache_batch_size:
            logger.warning("Prefix caching is not yet enabled for AutoModelForSpeechSeq2Seq")

        if mxint8_kv_cache:
            logger.warning("mxint8 cache is not yet enabled for AutoModelForSpeechSeq2Seq")

        if num_speculative_tokens:
            logger.warning("Speculative decoding is not yet enabled for AutoModelForSpeechSeq2Seq")

        output_names = self.model.get_output_names()

        target_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        kv_cache_dtype = CUSTOM_IO_DTYPE_MAP[target_dtype]
        custom_io = {}

        custom_io["input_features"] = kv_cache_dtype

        for output_name in output_names:
            if output_name.endswith("_RetainedState"):
                _add_retained_state_custom_io(
                    custom_io,
                    output_name,
                    dtype=kv_cache_dtype,
                    use_onnx_subfunctions=use_onnx_subfunctions,
                )

        return self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=_should_convert_to_fp16(target_dtype, compiler_options),
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            custom_io=custom_io,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    def generate(
        self,
        inputs: torch.Tensor,
        generation_len: int,
        streamer: TextStreamer | None = None,
        device_ids: list[int] | None = None,
    ) -> torch.Tensor | np.ndarray:
        """
        Generate output until ``<|endoftext|>`` token or `generation_len` is reached,
        by executing the compiled QPC on Cloud AI 100 hardware.

        This method performs sequential execution based on the compiled model's batch size
        and the provided audio tensors. It manages the iterative decoding process and KV cache.

        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Model inputs for inference, typically a dictionary containing:
            - `input_features` (np.ndarray): Preprocessed audio features.
            - `decoder_input_ids` (np.ndarray): Initial decoder input IDs (e.g., start token).
            - `decoder_position_ids` (np.ndarray): Initial decoder position IDs.
            These should be prepared to match the compiled model's expectations.
        generation_len : int
            Maximum number of tokens to generate. The generation stops if this limit is reached
            or the model generates an end-of-sequence token.
        streamer : TextStreamer, optional
            Streamer to receive generated tokens in real-time. Default is None.
        device_ids : List[int], optional
            Device IDs for running the QPC. Defaults to `[0]` if not specified.

        Returns
        -------
        CloudAI100ExecInfoNew
            Output from the AI 100 runtime, including generated IDs and performance metrics.

        Raises
        ------
        TypeError
            If the QPC path is not set (i.e., `compile` was not run).
        """
        if not isinstance(self.qpc_path, Path):
            raise TypeError("Please run compile API first!")

        inputs = self.auto_correct_inputs(inputs)
        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]

        inputs["input_features"] = inputs["input_features"].numpy().astype(np.float16)

        # add start token id and initial position ids to inputs
        seq_len = 1
        inputs["input_ids"] = (
            torch.ones((self.batch_size, seq_len), dtype=torch.int64) * self.model.config.decoder_start_token_id
        ).numpy()
        inputs["position_ids"] = (
            torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(self.batch_size, 1).numpy()
        )

        self.qpc_session.skip_buffers(
            [x for x in self.qpc_session.input_names + self.qpc_session.output_names if is_retained_state_name(x)]
        )

        outputs = {
            "logits": np.random.randn(self.batch_size, 1, self.model.config.vocab_size).astype(np.float32),
        }
        self.qpc_session.set_buffers(outputs)

        # encoder run
        start = perf_counter()
        outputs = self.qpc_session.run(inputs)

        # array to hold generated tokens
        generated_ids = np.full((self.batch_size, generation_len + 1), self.model.config.eos_token_id)
        generated_ids[:, 0] = [self.model.config.decoder_start_token_id]
        logits = outputs["logits"]
        next_token = logits.argmax(-1)
        generated_ids[:, 1] = next_token.squeeze(1)

        if streamer:
            streamer.put(next_token)

        inputs["input_features"] = np.zeros((self.batch_size, self.model.config.num_mel_bins, 1)).astype(np.float16)

        loop_start = perf_counter()
        for num_tokens in range(generation_len):
            outputs = self.qpc_session.run(inputs)

            logits = outputs["logits"]
            next_token = logits.argmax(-1)
            generated_ids[:, num_tokens + 1] = next_token.squeeze(1)

            if next_token[0][0] == self.model.config.eos_token_id:
                break

            inputs["input_ids"] = next_token
            inputs["position_ids"] += 1

            if streamer:
                streamer.put(next_token)
        end = perf_counter()

        prefill_time, decode_perf, total_perf, total_time = calculate_latency(num_tokens, loop_start, start, end)

        return CloudAI100ExecInfoNew(
            batch_size=self.batch_size,
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(prefill_time, decode_perf, total_perf, total_time),
        )


class QEFFAutoModelForCTC(QEFFTransformersBase):
    """
    The QEFFAutoModelForCTC class is designed for transformer models with a Connectionist Temporal Classification (CTC) speech-to-text head,
    including Wav2Vec2 and other encoder-only speech models optimized for alignment-free transcription.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.

    Example
    -------
    .. code-block:: python

        import torchaudio
        from QEfficient import QEFFAutoModelForCTC
        from transformers import AutoProcessor

        # Initialize the model using from_pretrained similar to transformers.AutoModelForCTC.
        model=QEFFAutoModelForCTC.from_pretrained(model_name)

        # Now you can directly compile the model for Cloud AI 100
        model.compile(num_cores=16)  # Considering you have a Cloud AI 100 SKU

        #prepare input
        processor = AutoProcessor.from_pretrained(model_name)
        input_audio, sample_rate = [...] # audio data loaded in via some external audio package, such as librosa or soundfile

        # Resample the input_audio if necessary
        if input_audio.shape[0] > 1:
            input_audio = input_audio.mean(dim=0)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            input_audio = resampler(input_audio)

        # You can now execute the model
        out = model.generate(processor,inputs=input_audio)
    """

    _hf_auto_class = AutoModelForCTC
    _pytorch_transforms = [CustomOpsTransform, AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform]
    _onnx_transforms = []

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.model.base_model.config.use_cache = True

        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path, pooling=None, *args, **kwargs):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCTC.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pre-trained model.

        .. code-block:: python

        import torchaudio
        from QEfficient import QEFFAutoModelForCTC
        from transformers import AutoProcessor

        # Initialize the model using from_pretrained similar to transformers.AutoModelForCTC.
        model=QEFFAutoModelForCTC.from_pretrained(model_name)

        # Now you can directly compile the model for Cloud AI 100
        model.compile(num_cores=16)  # Considering you have a Cloud AI 100 SKU

        #prepare input
        processor = AutoProcessor.from_pretrained(model_name)
        input_audio, sample_rate = [...] # audio data loaded in via some external audio package, such as librosa or soundfile

        # Resample the input_audio if necessary
        if input_audio.shape[0] > 1:
            input_audio = input_audio.mean(dim=0)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            input_audio = resampler(input_audio)

        # You can now execute the model
        out = model.generate(processor,inputs=input_audio)
        """
        _disable_unsupported_weight_free(kwargs, cls.__name__)
        enable_proxy = kwargs.pop("enable_proxy", False)
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        _resolve_torch_dtype(kwargs)
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # This is support models that should be classified to in a different auto class but transformers load them via this class
        kv_offload = kwargs.pop("kv_offload", None)

        kwargs.update({"enable_proxy": enable_proxy} if enable_proxy else {})

        if model.__class__.__name__ in MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP:
            return MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP[model.__class__.__name__](
                model, kv_offload=kv_offload, **kwargs
            )

        return cls(model, pretrained_model_name_or_path=pretrained_model_name_or_path, pooling=pooling, **kwargs)

    @property
    def get_model_config(self) -> dict:
        return self.model.config.__dict__

    def export(self, export_dir: str | None = None, **kwargs) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.
           :use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.WAV2VEC2_MAX_SEQ_LEN

        example_inputs = {
            "input_values": torch.zeros((bs, seq_len), dtype=self.model.config.torch_dtype),
        }

        dynamic_axes = {"input_values": {0: "batch_size", 1: "seq_len"}}

        output_names = ["logits"]

        return self._export(
            example_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=kwargs.get("use_onnx_subfunctions", False),
        )

    def compile(
        self,
        onnx_path: str | None = None,
        compile_dir: str | None = None,
        *,
        seq_len: int | list[int] = 480000,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        This method compiles the exported ``ONNX`` model using the Cloud AI 100 Platform SDK compiler binary found at ``/opt/qti-aic/exec/qaic-compile`` and generates a ``qpc`` package.
        If the model has not been exported yet, this method will handle the export process.
        You can pass any other arguments that the `qaic-compile` takes as extra kwargs.

        ``Optional`` Args:
            :onnx_path (str, optional): Path to pre-exported onnx model.
            :compile_dir (str, optional): Path for saving the qpc generated.
            :seq_len (Union[int, List[int]]): The length of the prompt should be less that ``seq_len``. ``Defaults to 32``.
            :batch_size (int, optional): Batch size. ``Defaults to 1``.
            :num_devices (int): Number of devices the model needs to be compiled for. Defaults to 1.
            :num_cores (int): Number of cores used to compile the model.
            :mxfp6_matmul (bool, optional): Whether to use ``mxfp6`` compression for weights. ``Defaults to False``.
            :use_onnx_subfunctions: bool, optional: whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
            :compiler_options (dict, optional): Additional compiler options.

                For QAIC Compiler: Extra arguments for qaic-compile can be passed.
                    :aic_enable_depth_first (bool, optional): Enables DFS with default memory size. ``Defaults to False``.
                    :allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. ``Defaults to False.``

                    Params are converted to flags as below:

                    - aic_hw_version=ai100 -> -aic-hw-version=ai100
                    - aic_hw_version=ai200 -> -aic-hw-version=ai200

                For QNN Compiler: Following arguments can be passed.
                    :enable_qnn (bool): Enables QNN Compilation.
                    :qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file.

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """

        _ignore_public_mdp_ts_num_devices(compiler_options)
        _seq_lens = seq_len if isinstance(seq_len, list) else [seq_len]
        specializations = [
            {"_graph_name": "CTC" if len(_seq_lens) == 1 else f"CTC_{i}", "batch_size": batch_size, "seq_len": sl}
            for i, sl in enumerate(_seq_lens)
        ]

        target_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        return self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            specializations=specializations,
            convert_to_fp16=_should_convert_to_fp16(target_dtype, compiler_options),
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    def generate(
        self,
        processor,
        inputs: torch.Tensor,
        device_ids: list[int] | None = None,
        runtime_ai100: bool = True,
    ) -> torch.Tensor | np.ndarray:
        """
        This method generates output by executing PyTorch runtime or the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        ``Mandatory`` Args:
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
            :processor (AutoProcessor): The Processor to use for encoding the waveform.
        ``optional`` Args:
            :device_ids (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
            :runtime_ai100 (bool, optional): ``AI_100`` and ``PyTorch`` runtime is supported as of now. Defaults to ``True`` for ``AI_100`` runtime.
        Returns:
            :dict: Output from the ``AI_100`` or ``PyTorch`` runtime.
        """
        # AI_100 runtime
        if runtime_ai100:
            if not isinstance(self.qpc_path, Path):
                raise TypeError("Please run compile API first!")

            return self.cloud_ai_100_feature_generate(processor, inputs=inputs, device_ids=device_ids)
        # PyTorch runtime
        else:
            return self.pytorch_feature_generate(processor, model=self.model, inputs=inputs)

    def cloud_ai_100_feature_generate(
        self,
        processor,
        inputs: torch.Tensor,
        device_ids: list[int] | None = None,
    ) -> np.ndarray:
        """
        Generates features with list of prompts using AI 100 runtime.

        ``Mandatory`` Args:
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
            :processor (AutoProcessor): The Processor to use for encoding the waveform.
        ``Optional`` Args:
            device_ids (List[int], optional): A list of device IDs to use for the session. Defaults to runtime automatic device selection.

        """

        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]
            self.seq_len = self.qpc_session.bindings[0].dims[1]

        # To handle single seq_len as we can't fetch allowed shapes for single seq_len
        self.seq_len = self.qpc_session.bindings[0].dims[1] if not hasattr(self, "seq_len") else self.seq_len
        inputs = processor(inputs, return_tensors="pt", max_length=self.seq_len, truncation=True, padding="max_length")
        input_ids_len = inputs["input_values"].shape[-1]
        input_values = np.array(
            torch.nn.functional.pad(inputs["input_values"], (0, self.seq_len - input_ids_len), "constant", 0)
        )
        target_dtype = getattr(self.model.config, "torch_dtype", torch.float32)
        input_values = input_values.astype(TORCH_TO_NUMPY_DTYPE_MAP[target_dtype])
        inputs = {"input_values": input_values}
        outputs = self.qpc_session.run(inputs)

        logits = outputs["logits"]
        predicted_ids = np.argmax(logits, axis=-1)
        transcriptions = processor.batch_decode(torch.tensor(predicted_ids))
        return transcriptions

    def pytorch_feature_generate(self, processor, model, inputs: torch.Tensor | np.ndarray) -> list[torch.Tensor]:
        """
        Generates features from a list of text prompts using a PyTorch model.

        ``Mandatory`` Args:
            :model: The transformed PyTorch model used for generating features.
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
            :processor (AutoProcessor): The Processor to use for encoding the waveform.

        """
        input_values = processor(
            inputs[0], return_tensors="pt", max_length=self.seq_len, truncation=True, padding="max_length"
        ).input_values
        outputs = model(input_values[0])

        logits = outputs.logits
        logits = logits.detach().numpy()
        predicted_ids = np.argmax(logits, axis=-1)
        transcriptions = processor.batch_decode(predicted_ids)
        return transcriptions
