# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import onnx
import torch
from accelerate import init_empty_weights

from QEfficient.exporter.weight_free.checkpoint_key_resolver import promote_initializers_and_build_spec
from QEfficient.exporter.weight_free.weight_spec import load_weight_spec, resolve_weight_spec_path, save_weight_spec
from QEfficient.utils import load_json
from QEfficient.utils.checkpoint_utils import resolve_checkpoint_dir
from QEfficient.utils.logging_utils import logger
from QEfficient.utils.torch_patches import dynamo_invoke_subgraph_fallback_env


def _to_meta(value: Any) -> Any:
    """Recursively move tensor-like example inputs to equivalent meta tensors."""
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value, device="meta")
    if isinstance(value, tuple):
        return tuple(_to_meta(item) for item in value)
    if isinstance(value, list):
        return [_to_meta(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_meta(item) for key, item in value.items()}
    return value


def _run_quantizer_for_wf(qeff_model, target_dtype: torch.dtype):
    """Finish preparing a meta-device QEfficient wrapper for weight-free tracing, in place."""
    model_ref = qeff_model.hash_params.get("pretrained_model_name_or_path")
    if not model_ref:
        raise ValueError(
            "Weight-free export requires checkpoint metadata. "
            "Pass `pretrained_model_name_or_path=...` when constructing the QEff model manually."
        )

    quant_config = getattr(qeff_model.model.config, "quantization_config", None)

    if quant_config is not None:
        # For quantized models the meta model must use the same quantized layer types as the
        # checkpoint so that ONNX initializer names match the checkpoint's storage keys.
        # qeff_model.model was built via from_config (no real quantized checkpoint load), so
        # Mxfp4GptOssExpertDequantizeTransform already ran as a no-op during __init__ and will
        # not undo the replacement applied here.
        from QEfficient.transformers.quantizers.auto import (
            QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING,
            QEFF_AUTO_QUANTIZER_MAPPING,
        )

        # quantization_config may be a plain dict (AutoConfig.from_pretrained) or a proper
        # config object (QEFFAutoModelForCausalLM.from_pretrained).  Normalise to an object.
        if isinstance(quant_config, dict):
            quant_type = quant_config.get("quant_method") or quant_config.get("quant_type")
            if quant_type == "fp8" and getattr(qeff_model.model.config, "model_type", None) == "glm_moe_dsa":
                qeff_model.model = qeff_model.model.to(dtype=target_dtype)
                return qeff_model
            config_cls = QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.get(quant_type)
            if config_cls is None:
                raise NotImplementedError(
                    f"Weight-free export is not implemented for quantization type '{quant_type}'. Supported: mxfp4"
                )
            init_kwargs = {k: v for k, v in quant_config.items() if k != "quant_method"}
            quant_config = config_cls(**init_kwargs)
        else:
            quant_method = getattr(quant_config, "quant_method", None) or getattr(quant_config, "quant_type", None)
            quant_type = quant_method.value if hasattr(quant_method, "value") else quant_method

        if quant_type == "fp8" and getattr(qeff_model.model.config, "model_type", None) == "glm_moe_dsa":
            qeff_model.model = qeff_model.model.to(dtype=target_dtype)
            return qeff_model

        quantizer_cls = QEFF_AUTO_QUANTIZER_MAPPING.get(quant_type) if quant_type else None
        if quantizer_cls is None:
            raise NotImplementedError(
                f"Weight-free export is not implemented for quantization type '{quant_type}'. Supported: mxfp4"
            )
        quantizer = quantizer_cls(quant_config)
        # Run inside init_empty_weights so newly created quantized layer buffers stay on
        # the meta device and are treated as weight-spec entries, not embedded constants.
        with init_empty_weights():
            quantizer._process_model_before_weight_loading(qeff_model.model)
    else:
        qeff_model.model = qeff_model.model.to(dtype=target_dtype)

    return qeff_model


def _resolve_weight_free_target_dtype(config) -> torch.dtype:
    """Return the dtype used by the meta model and prepared checkpoint."""
    target_dtype = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None) or torch.float32
    if isinstance(target_dtype, str):
        target_dtype = target_dtype.removeprefix("torch.")
        target_dtype = getattr(torch, target_dtype, torch.float32)
    return target_dtype


def _resolve_expert_parallel_checkpoint_layout(qeff_model) -> dict[str, Any] | None:
    """Return the EP weight layout that must also be materialized on disk."""
    config = qeff_model.model.config
    architectures = getattr(config, "architectures", None) or []
    is_glm_moe_dsa = (
        "GlmMoeDsaForCausalLM" in architectures
        or getattr(config, "model_type", None) == "glm_moe_dsa"
    )
    if not is_glm_moe_dsa:
        return None
    hash_params = getattr(qeff_model, "hash_params", {}) or {}
    flavour = hash_params.get("moe_prefill_flavour")
    flavour = getattr(flavour, "value", flavour)
    if flavour != "expert_parallel":
        return None

    layout_fields = {
        "num_pipeline_stages": "moe_prefill_num_pipeline_stages",
        "num_parallelized_experts": "moe_prefill_num_parallelized_experts",
    }
    layout = {name: hash_params.get(param_name) for name, param_name in layout_fields.items()}
    if any(value is None for value in layout.values()):
        raise ValueError(
            "Weight-free EP export requires num_pipeline_stages and num_parallelized_experts "
            "in the model's resolved export parameters."
        )
    layout = {name: int(value) for name, value in layout.items()}
    if any(value <= 0 for value in layout.values()):
        raise ValueError(f"Invalid weight-free EP checkpoint layout: {layout}")
    return {"flavour": flavour, "layout_version": 1, **layout}


def _prune_unused_fake_initializers(onnx_program) -> None:
    """Remove FakeTensor initializers not referenced by any graph node.

    During weight-free dynamo export, meta-device parameters that are not
    actually consumed in the forward graph can end up as orphan initializers.
    Serialising them fails (no data to write), so we drop them here before save.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    initializers = onnx_program.model.graph.initializers
    used_names = {name for node in onnx_program.model.graph for name in node.inputs}
    used_names.update(output.name for output in onnx_program.model.graph.outputs)
    for name in list(initializers):
        const_value = getattr(initializers[name], "const_value", None)
        raw_value = getattr(const_value, "raw", None)
        if isinstance(raw_value, FakeTensor) and name not in used_names:
            del initializers[name]


def _prepare_checkpoint_for_weight_free_export(
    qeff_model,
    model_ref: str,
    target_dtype: torch.dtype,
) -> str:
    """Prepare a checkpoint directory for weight-free ONNX export.

    Parameters
    ----------
    qeff_model
        QEfficient model wrapper that provides checkpoint transform classes.
    model_ref : str
        Local model path or Hugging Face Hub model id.
    target_dtype : torch.dtype
        Floating-point dtype expected by the exported ONNX graph.

    Returns
    -------
    str
        Path to the prepared checkpoint directory.
    """
    from QEfficient.base.checkpoint_transforms import CheckpointTransformPipeline
    from QEfficient.utils.cache import QEFF_CHECKPOINT_HOME

    source_dir = resolve_checkpoint_dir(model_ref)
    dtype_suffix = str(target_dtype).replace("torch.", "")
    model_config = qeff_model.model.config
    model_type = getattr(model_config, "model_type", "unknown")
    num_hidden_layers = getattr(model_config, "num_hidden_layers", "unknown")
    prepared_name = source_dir.name + f"-qeff-prepared-{model_type}-layers{num_hidden_layers}-{dtype_suffix}"
    expert_parallel_layout = _resolve_expert_parallel_checkpoint_layout(qeff_model)
    if expert_parallel_layout is not None:
        layout_signature = json.dumps(expert_parallel_layout, sort_keys=True, separators=(",", ":"))
        layout_hash = hashlib.sha256(layout_signature.encode("utf-8")).hexdigest()[:12]
        prepared_name += f"-moe-ep-v{expert_parallel_layout['layout_version']}-{layout_hash}"
    if QEFF_CHECKPOINT_HOME:
        prepared_out = QEFF_CHECKPOINT_HOME.expanduser() / prepared_name
    else:
        prepared_out = source_dir.parent / prepared_name
    prep_pipeline = CheckpointTransformPipeline(transforms=qeff_model._checkpoint_transforms)
    return str(
        prep_pipeline.apply(
            src=source_dir,
            out=prepared_out,
            target_dtype=target_dtype,
            model_config=model_config,
            expert_parallel_layout=expert_parallel_layout,
            checkpoint_transform_context=(
                {"moe_weight_layout": expert_parallel_layout} if expert_parallel_layout is not None else None
            ),
        )
    )


def export_weight_free_onnx(
    qeff_model,
    onnx_path: Path,
    example_inputs: dict[str, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    dynamic_shapes: dict[str, Any],
    export_kwargs: dict[str, Any],
    onnx_transform_kwargs: dict[str, Any],
):
    """Export a QEfficient model to ONNX with checkpoint weights externalized.

    Parameters
    ----------
    qeff_model
        Loaded QEfficient model wrapper used as the source of config and export options.
    onnx_path : Path
        Destination path where the dynamo export is saved.
    example_inputs : Dict[str, torch.Tensor]
        Example inputs used to trace the model.
    input_names : List[str]
        ONNX graph input names.
    output_names : List[str]
        ONNX graph output names.
    dynamic_shapes : Dict[str, Any]
        Dynamo export dynamic shape specification.
    export_kwargs : Dict[str, Any]
        Additional keyword arguments forwarded to ``torch.onnx.export``.
    onnx_transform_kwargs : Dict[str, Any]
        ONNX transform options carried through the export flow.

    Returns
    -------
    tuple
        Meta QEfficient model, updated ONNX transform kwargs, and cleanup callback.
    """
    target_dtype = _resolve_weight_free_target_dtype(qeff_model.model.config)
    meta_qeff_model = _run_quantizer_for_wf(qeff_model, target_dtype)

    # export_wrapper (the @export_wrapper decorator on _export) already ran
    # _setup_onnx_subfunctions on this same object before calling into this
    # function, and already has temporarily_enable_nested_compile_regions active
    # around the whole call chain — meta_qeff_model is qeff_model is self, mutated
    # in place, not a clone. Re-running subfunction setup here would only be
    # necessary if _run_quantizer_for_wf's quantizer rewrite (which runs after
    # export_wrapper's setup) changed the repeated decoder-block class itself;
    # quantizer rewrites replace layer types inside a decoder block (e.g. MoE
    # experts), not the decoder block class, so the classes export_wrapper already
    # resolved remain correct.

    meta_example_inputs = _to_meta(example_inputs)
    model_ref = meta_qeff_model.hash_params["pretrained_model_name_or_path"]

    meta_qeff_model.model.requires_grad_(False)
    torch_onnx_export_start = time.perf_counter()
    with dynamo_invoke_subgraph_fallback_env():
        onnx_program = torch.onnx.export(
            meta_qeff_model.model,
            args=(),
            f=None,
            kwargs=meta_example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            dynamic_shapes=dynamic_shapes,
            **export_kwargs,
        )
    if onnx_program is None:
        raise RuntimeError("torch.onnx.export returned None for weight-free dynamo export")
    torch_onnx_export_duration_seconds = time.perf_counter() - torch_onnx_export_start
    logger.info(
        "Weight-free torch.onnx.export completed in %.2fs",
        torch_onnx_export_duration_seconds,
    )

    prep_start = time.perf_counter()
    prepared_model_ref = _prepare_checkpoint_for_weight_free_export(meta_qeff_model, model_ref, target_dtype)
    prep_duration_seconds = time.perf_counter() - prep_start
    logger.info(
        "Weight-free checkpoint preparation completed in %.2fs: %s",
        prep_duration_seconds,
        prepared_model_ref,
    )
    qeff_model._weight_free_export_metrics = {
        "torch_onnx_export_seconds": torch_onnx_export_duration_seconds,
        "checkpoint_preparation_seconds": prep_duration_seconds,
    }

    spec = promote_initializers_and_build_spec(
        onnx_program=onnx_program,
        model_ref=prepared_model_ref,
        model_name=qeff_model.model_name,
        qeff_model=meta_qeff_model,
    )
    _prune_unused_fake_initializers(onnx_program)
    onnx_program.save(str(onnx_path))
    save_weight_spec(resolve_weight_spec_path(onnx_path), spec)

    return meta_qeff_model, onnx_transform_kwargs


def embed_weight_spec_as_metadata(model, weight_spec_path) -> None:
    """Embed weight_spec.json into the ONNX model as com.qti.aisw.extdata metadata.

    The QAIC compiler reads this key to locate and load external checkpoint weights
    at compile time.
    """

    weight_spec_json = json.dumps(load_json(Path(weight_spec_path)), separators=(",", ":"), sort_keys=True)
    model.metadata_props.append(onnx.StringStringEntryProto(key="com.qti.aisw.extdata", value=weight_spec_json))


def link_prepared_checkpoint_dir(onnx_path: Path, weight_spec_path: Path) -> None:
    """Place the checkpoint directory referenced by the weight spec next to the ONNX export.

    Weight specs store checkpoint files relative to their checkpoint root.
    Keeping this link beside the ONNX lets the compiler resolve those paths when
    it consumes the embedded weight spec.
    """
    spec = load_weight_spec(Path(weight_spec_path))
    prepared_out = Path(spec.model_id)
    symlink = Path(onnx_path).parent / prepared_out.name
    if prepared_out.exists() and not symlink.exists():
        try:
            symlink.symlink_to(prepared_out)
        except OSError as exc:
            logger.warning(
                "Could not create symlink %s -> %s: %s. "
                "Checkpoint files may not resolve at compile time if paths are relative.",
                symlink,
                prepared_out,
                exc,
            )
