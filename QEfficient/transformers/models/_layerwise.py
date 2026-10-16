# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Layer-wise export/compile orchestration for selected MoE architectures.

Provisional API. Scheduled for removal once first-class multi-window export lands.
Today only ``qwen3_vl_moe``, ``qwen3_5_moe`` and ``qwen3_moe`` carry the
windowing hooks (``_start``/``_end`` class attributes) that this driver relies
on; calling :func:`run_layerwise` for any other architecture will raise.
"""

from __future__ import annotations

import functools
import random
import shutil
import time
import warnings
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
import transformers

import QEfficient
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")

# Architectures whose modeling files declare _start/_end class attributes the
# layer-wise driver pokes. Keep this list narrow on purpose - adding a new
# architecture must come with the corresponding modeling-file hooks.
_LAYERWISE_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "qwen3_vl_moe",
        "qwen3_vl_moe_text",
        "qwen3_5_moe",
        "qwen3_moe",
    }
)

# Hard cap on the RoPE base table rows we serialize per window. Chosen as a
# constant (not a function of ctx_len) on purpose: changing ctx_len at compile
# time should re-use the cached ONNX and only re-run the QPC compile. Any
# inference-time position id is bounded by ctx_len, and ctx_len in practice
# stays well under 32K for the supported MoE families today, so dropping
# the unreachable rows past 32K is lossless.
_LAYERWISE_ROPE_MAX_POSITIONS = 76800

# Process-local layer-wise window state. We deliberately avoid setting class
# attributes on transformers.modeling_utils.PreTrainedModel - those would leak
# to every HF model in the process and survive past the layer-wise run. The
# patched HF hooks (shard filter, model-init nuller) close over this dict so
# they can be installed once and behave as no-ops whenever ``active`` is False.
_LAYERWISE_STATE: Dict[str, int] = {
    "active": 0,
    "start": 0,
    "end": 0,
    "total_layers": 0,
    "text_start": 0,
    "text_end": 0,
    "text_total_layers": 0,
    "force_full_init": 0,
}

_DEPRECATION_WARNED = False


def _maybe_warn_deprecation() -> None:
    global _DEPRECATION_WARNED
    if _DEPRECATION_WARNED:
        return
    warnings.warn(
        "layerwise=True is a provisional API and will be deprecated once "
        "first-class multi-window export lands. Use only for the supported "
        f"model types: {sorted(_LAYERWISE_SUPPORTED_MODEL_TYPES)}.",
        DeprecationWarning,
        stacklevel=3,
    )
    _DEPRECATION_WARNED = True


def _resolve_model_type(config) -> str:
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "model_type", None):
        return text_config.model_type
    return getattr(config, "model_type", "") or ""


def assert_layerwise_supported(config) -> str:
    """Raise a clear error if the model architecture has no layerwise hooks."""
    top_type = getattr(config, "model_type", "") or ""
    text_type = _resolve_model_type(config)
    if top_type in _LAYERWISE_SUPPORTED_MODEL_TYPES or text_type in _LAYERWISE_SUPPORTED_MODEL_TYPES:
        return text_type or top_type
    raise NotImplementedError(
        "layerwise=True is only supported for model types: "
        f"{sorted(_LAYERWISE_SUPPORTED_MODEL_TYPES)}. Got model_type="
        f"'{top_type}' (text_config.model_type='{text_type}'). "
        "Run with layerwise=False (the default) for this model."
    )


def is_layerwise_active() -> bool:
    """True only while the layer-wise export driver is running.

    The driver flips this on inside :func:`_layerwise_export_env`. Outside that
    scope (the default, non-layerwise path) it is always False, which lets the
    modeling forwards short-circuit every window branch and behave exactly as
    they did before layer-wise support was added.
    """
    return bool(_LAYERWISE_STATE["active"])


def resolve_layer_window(model_cls, total_layers: int) -> Tuple[int, int]:
    """Return the ``[start, end)`` decoder-layer window to run this forward.

    When layer-wise export is inactive this always returns ``(0, total_layers)``
    regardless of any ``_start``/``_end`` class attributes, so the default path
    is independent of (possibly stale) window state left on the modeling class.
    When the driver is active it honors the window it poked onto ``model_cls``.
    """
    if not is_layerwise_active():
        return 0, total_layers
    start = int(getattr(model_cls, "_start", 0) or 0)
    end = getattr(model_cls, "_end", 0) or 0
    end = int(end) if end else total_layers
    return start, end


def is_last_layer_window(model_cls, total_layers: int) -> bool:
    """True if this forward owns the final decoder window (applies final norm / lm_head).

    Always True on the default path; on the layer-wise path it is True only for
    the window whose ``_end`` reaches the total layer count.
    """
    if not is_layerwise_active():
        return True
    _, end = resolve_layer_window(model_cls, total_layers)
    return end >= total_layers


# ---------------------------------------------------------------------------
# Internal helpers (lifted from the legacy example script)
# ---------------------------------------------------------------------------


def _ensure_pretrained_window_attrs() -> None:
    """No-op kept for compatibility with prior call sites.

    Layer-wise window state lives in the module-local ``_LAYERWISE_STATE``
    dict (see top of file); we no longer pollute ``PreTrainedModel`` with
    class attributes.
    """
    return


def _build_layer_windows(total_layers: int, window_size: int) -> List[Tuple[int, int]]:
    if total_layers <= 0:
        raise ValueError(f"Invalid total_layers={total_layers}; expected > 0.")
    if window_size is None:
        raise ValueError(
            "layerwise=True requires `layerwise_window_size` to be a positive integer. "
            "Got None. Pass an explicit value (for example, layerwise_window_size=1)."
        )
    if isinstance(window_size, bool) or not isinstance(window_size, Integral):
        raise TypeError(
            "layerwise=True requires `layerwise_window_size` to be a positive integer. "
            f"Got {type(window_size).__name__}: {window_size!r}."
        )
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError(
            f"layerwise=True requires `layerwise_window_size` to be a positive integer. Got {window_size}."
        )
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < total_layers:
        end = min(total_layers, start + window_size)
        windows.append((start, end))
        start = end
    return windows


@functools.lru_cache(maxsize=32)
def _checkpoint_has_shard_index(model_id: str) -> bool:
    """Return True when checkpoint provides an index file for shard filtering."""
    hub_kwargs = {"_raise_exceptions_for_missing_entries": False}
    for index_filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            index_path = transformers.utils.hub.cached_file(model_id, index_filename, **hub_kwargs)
        except Exception:
            index_path = None
        if index_path is not None:
            return True
    return False


def _get_text_layers_container(model):
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _null_outside_window_layers(model, *, apply_text: bool = True) -> None:
    if not apply_text:
        return
    text_start = int(_LAYERWISE_STATE["text_start"] or _LAYERWISE_STATE["start"])
    text_end = int(_LAYERWISE_STATE["text_end"] or _LAYERWISE_STATE["end"])
    text_layers = _get_text_layers_container(model)
    if text_layers is not None and text_end > text_start:
        for idx, _ in enumerate(text_layers):
            if idx < text_start or idx >= text_end:
                text_layers[idx] = None


def _find_language_model(model):
    """Locate the inner language_model that owns sin_cached / cos_cached / embed_tokens."""
    candidates = []
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        candidates.append(model.model.language_model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    if hasattr(model, "model"):
        candidates.append(model.model)
    candidates.append(model)
    for cand in candidates:
        if any(hasattr(cand, attr) for attr in ("sin_cached", "cos_cached", "embed_tokens")):
            return cand
    return None


def _slim_for_window_export(qeff_model, *, ctx_len: Optional[int]) -> None:
    """Shrink top-level params that are unused (or oversized) for this window.

    Without this, every per-window ONNX shard re-bakes the full RoPE base table
    (``sin_cached``/``cos_cached`` of shape ``[max_position_embeddings, head_dim]``,
    typically tens of MB in fp32) plus the full vocab embedding, blowing each
    layer-window shard up by 1-2 orders of magnitude over the actual layer
    weight footprint. Each top-level param is touched in-place; the next
    window rebuilds the model from scratch via the factory so there is no
    leakage across windows.
    """
    import torch

    inner = qeff_model.model if hasattr(qeff_model, "model") else qeff_model
    lm = _find_language_model(inner)
    if lm is None:
        return

    text_start = int(_LAYERWISE_STATE["text_start"] or _LAYERWISE_STATE["start"])
    text_end = int(_LAYERWISE_STATE["text_end"] or _LAYERWISE_STATE["end"])
    text_total = int(_LAYERWISE_STATE["text_total_layers"] or _LAYERWISE_STATE["total_layers"] or 0)

    # 1) Truncate sin_cached / cos_cached to a fixed cap (32K rows) instead
    #    of ctx_len. A constant cap keeps the export hash invariant when the
    #    user changes ctx_len, so re-compiling at a different context length
    #    only re-runs the QPC compile - the cached layer-wise ONNX is reused.
    #    Inference-time position ids are bounded by ctx_len which is well
    #    under the cap for the supported MoE families today.
    del ctx_len  # signature retained for forward compat, value intentionally unused
    rope_cap = _LAYERWISE_ROPE_MAX_POSITIONS
    for attr in ("sin_cached", "cos_cached"):
        param = getattr(lm, attr, None)
        if param is None or not hasattr(param, "shape") or param.dim() < 2:
            continue
        cur_rows = int(param.shape[0])
        if cur_rows <= rope_cap:
            continue
        with torch.no_grad():
            truncated = param.detach()[:rope_cap].clone().contiguous()
        new_param = torch.nn.Parameter(truncated, requires_grad=False)
        setattr(lm, attr, new_param)

    # 2) Drop the vocab embedding for windows that don't run input-id lookup.
    #    The first window (text_start == 0) is the only one that calls
    #    ``get_input_embeddings()(input_ids)``; later windows take
    #    ``inputs_embeds`` directly so the embedding matrix is unreached but
    #    still serialized. Replace its weight with a tiny placeholder of the
    #    same dtype/device so module attributes stay valid.
    if text_start > 0 and hasattr(lm, "embed_tokens"):
        embed = getattr(lm, "embed_tokens", None)
        weight = getattr(embed, "weight", None)
        if weight is not None and weight.dim() == 2 and weight.shape[0] > 1:
            with torch.no_grad():
                tiny = torch.zeros((1, weight.shape[1]), dtype=weight.dtype, device=weight.device)
            embed.weight = torch.nn.Parameter(tiny, requires_grad=False)

    # 3) Drop the lm_head for windows that aren't the last one. Only the final
    #    window applies ``self.model.lm_head(hidden_states)``.
    outer = qeff_model.model if hasattr(qeff_model, "model") else qeff_model
    lm_head = getattr(outer, "lm_head", None)
    if (
        lm_head is not None
        and text_total > 0
        and text_end < text_total
        and hasattr(lm_head, "weight")
        and lm_head.weight is not None
        and lm_head.weight.dim() == 2
        and lm_head.weight.shape[0] > 1
    ):
        with torch.no_grad():
            tiny = torch.zeros(
                (1, lm_head.weight.shape[1]),
                dtype=lm_head.weight.dtype,
                device=lm_head.weight.device,
            )
        lm_head.weight = torch.nn.Parameter(tiny, requires_grad=False)
        if getattr(lm_head, "bias", None) is not None:
            lm_head.bias = torch.nn.Parameter(
                torch.zeros((1,), dtype=lm_head.bias.dtype, device=lm_head.bias.device),
                requires_grad=False,
            )


def _install_window_patch(model_cls) -> None:
    if getattr(model_cls, "_window_patch_installed", False):
        return
    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Only nullify decoder layers when the layer-wise driver is actively
        # exporting; idle calls to from_pretrained must behave normally.
        if _LAYERWISE_STATE["active"] and not _LAYERWISE_STATE["force_full_init"]:
            _null_outside_window_layers(self, apply_text=True)

    model_cls.__init__ = patched_init
    model_cls._window_patch_installed = True


def _install_shard_window_patch() -> None:
    if getattr(transformers.modeling_utils, "_window_shard_patch_installed", False):
        return

    original_get_checkpoint_shard_files = transformers.modeling_utils.get_checkpoint_shard_files

    @functools.wraps(original_get_checkpoint_shard_files)
    def patched_get_checkpoint_shard_files(*args, **kwargs):
        shard_files, metadata = original_get_checkpoint_shard_files(*args, **kwargs)
        # Honor the module-local state instead of polluting PreTrainedModel.
        # When layerwise is not active this reduces to a no-op for any HF
        # caller that happens to load checkpoint shards in this process.
        if not _LAYERWISE_STATE["active"]:
            return shard_files, metadata
        weight_map = metadata.get("weight_map")
        if not weight_map:
            return shard_files, metadata

        start = int(_LAYERWISE_STATE["start"])
        end = int(_LAYERWISE_STATE["end"])
        text_start = int(_LAYERWISE_STATE["text_start"] or start)
        text_end = int(_LAYERWISE_STATE["text_end"] or end)
        if text_end <= text_start:
            return shard_files, metadata

        selected_text_prefixes = tuple(
            [f"model.layers.{i}." for i in range(text_start, text_end)]
            + [f"model.language_model.layers.{i}." for i in range(text_start, text_end)]
        )
        filtered_weight_map: Dict[str, str] = {}
        for checkpoint_key, shard_name in weight_map.items():
            if checkpoint_key.startswith("model.layers.") or checkpoint_key.startswith("model.language_model.layers."):
                if checkpoint_key.startswith(selected_text_prefixes):
                    filtered_weight_map[checkpoint_key] = shard_name
                continue
            filtered_weight_map[checkpoint_key] = shard_name

        if not filtered_weight_map:
            return shard_files, metadata

        shard_name_to_path = {path.split("/")[-1]: path for path in shard_files}
        filtered_shard_names = sorted(set(filtered_weight_map.values()))
        filtered_shard_files = [shard_name_to_path[name] for name in filtered_shard_names if name in shard_name_to_path]
        if not filtered_shard_files:
            return shard_files, metadata

        metadata["weight_map"] = filtered_weight_map
        metadata["all_checkpoint_keys"] = list(filtered_weight_map.keys())
        return filtered_shard_files, metadata

    transformers.modeling_utils.get_checkpoint_shard_files = patched_get_checkpoint_shard_files
    transformers.modeling_utils._window_shard_patch_installed = True


def _set_layer_windows(text_start: int, text_end: int, text_total_layers: int) -> None:
    # Update the module-local state used by patched HF hooks. We deliberately
    # do NOT set class attributes on transformers.modeling_utils.PreTrainedModel
    # here - that would leak to every HF model in the process.
    _LAYERWISE_STATE["start"] = text_start
    _LAYERWISE_STATE["end"] = text_end
    _LAYERWISE_STATE["total_layers"] = text_total_layers
    _LAYERWISE_STATE["text_start"] = text_start
    _LAYERWISE_STATE["text_end"] = text_end
    _LAYERWISE_STATE["text_total_layers"] = text_total_layers
    _LAYERWISE_STATE["force_full_init"] = 0

    # The QEff modeling classes themselves expose _start/_end/_total_layers as
    # part of their windowing contract (they are read inside their forward
    # implementations). Those are ours to set.
    qeff_vl_mod = getattr(QEfficient.transformers.models, "qwen3_vl_moe", None)
    if qeff_vl_mod is not None:
        cls = getattr(qeff_vl_mod.modeling_qwen3_vl_moe, "QEffQwen3VLMoeTextModel", None)
        if cls is not None:
            cls._start = text_start
            cls._end = text_end
            cls._total_layers = text_total_layers

    qeff_35_mod = getattr(QEfficient.transformers.models, "qwen3_5_moe", None)
    if qeff_35_mod is not None:
        for class_name in ("QEffQwen3_5MoeTextModel", "QEffQwen3_5MoeModel"):
            cls = getattr(qeff_35_mod.modeling_qwen3_5_moe, class_name, None)
            if cls is not None:
                cls._start = text_start
                cls._end = text_end
                cls._total_layers = text_total_layers

    qeff_3_mod = getattr(QEfficient.transformers.models, "qwen3_moe", None)
    if qeff_3_mod is not None:
        cls = getattr(qeff_3_mod.modeling_qwen3_moe, "QEffQwen3MoeModel", None)
        if cls is not None:
            cls._start = text_start
            cls._end = text_end
            cls._total_layers = text_total_layers

    QEfficient.base.modeling_qeff.QEFFBaseModel._start = text_start
    QEfficient.base.modeling_qeff.QEFFBaseModel._end = text_end
    QEfficient.base.modeling_qeff.QEFFBaseModel._total_layers = text_total_layers


def _reset_layer_windows() -> None:
    _set_layer_windows(0, 0, 0)


def _resolve_export_root(onnx_path: Path) -> Path:
    parts = list(onnx_path.parts)
    if "onnx_layerwise_tmp" in parts:
        return Path(*parts[: parts.index("onnx_layerwise_tmp")])
    if "final_data" in parts:
        return Path(*parts[: parts.index("final_data")])
    return onnx_path.parent


def _is_cached_merged(onnx_path: Path) -> bool:
    """Return True for layerwise-stitched ONNX files produced by the merge pipeline."""
    return onnx_path.name.startswith("merged_")


def _cached_merged_onnx(export_root: Path, total_layers: Optional[int] = None) -> Optional[Path]:
    """Return the complete cached merged ONNX, if it exists."""
    search_dirs = [export_root]
    cached_dir = export_root / "final_data"
    if cached_dir.is_dir():
        search_dirs.append(cached_dir)

    if total_layers is not None:
        for search_dir in search_dirs:
            expected = search_dir / f"merged_0-{total_layers}.onnx"
            if expected.is_file():
                return expected

    cached_merged = [path for search_dir in search_dirs for path in search_dir.glob("merged_0-*.onnx")]
    cached_merged = sorted(cached_merged)
    return cached_merged[-1] if cached_merged else None


def _cached_root_onnx(export_root: Path, model_name: Optional[str]) -> Optional[Path]:
    """Return a canonical non-merged ONNX in the hash root, if present."""
    if model_name is None:
        return None
    candidate = export_root / f"{model_name}.onnx"
    return candidate if candidate.is_file() else None


def _resolve_layerwise_model_name(qeff_model) -> Optional[str]:
    """Resolve the decoder model name used for canonical ONNX cache lookup."""
    lang_model = getattr(qeff_model, "lang_model", None)
    if lang_model is not None and hasattr(lang_model, "model_name"):
        return lang_model.model_name
    return getattr(qeff_model, "model_name", None)


def _relocate_merged_onnx_to_root(export_root: Path, merged_onnx: Path) -> Path:
    """Move a merged ONNX and its external tensor blobs into the hash root.

    The move keeps the layerwise cache layout aligned with regular export and
    avoids copying large external data files, which can temporarily double disk
    usage for large models.
    """
    if merged_onnx.parent == export_root:
        return merged_onnx

    model = onnx.load(str(merged_onnx), load_external_data=False)
    external_files = set()
    for tensor in model.graph.initializer:
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    external_files.add(entry.value)

    src_dir = merged_onnx.parent
    for rel_name in external_files:
        src = src_dir / rel_name
        dst = export_root / rel_name
        if src.exists() and src != dst and not dst.exists():
            shutil.move(str(src), str(dst))

    dst_onnx = export_root / merged_onnx.name
    if merged_onnx != dst_onnx:
        shutil.move(str(merged_onnx), str(dst_onnx))
    return dst_onnx


def _cleanup_layerwise_intermediates(export_root: Path, *, keep_onnx: Optional[Path] = None) -> None:
    """Remove transient layerwise export directories after final ONNX relocation."""
    for dirname in ("final_data", "onnx_layerwise_tmp"):
        path = export_root / dirname
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for pattern in ("pref_*.onnx",):
        for path in export_root.glob(pattern):
            if keep_onnx is not None and path == keep_onnx:
                continue
            path.unlink(missing_ok=True)


def _stitch_layerwise_if_available(export_root: Path, total_layers: Optional[int] = None) -> str:
    # If a prior run already produced the complete merged ONNX, just return it.
    cached_merged = _cached_merged_onnx(export_root, total_layers)
    if cached_merged is not None:
        return str(cached_merged)
    pipeline_fn = getattr(QEfficient.utils, "layerwise_pipeline", None)
    if callable(pipeline_fn):
        return pipeline_fn(str(export_root), final_data_dir=".")
    return str(export_root / "onnx_layerwise_tmp")


def _cached_layerwise_onnx_path(qeff_model, compile_kwargs: Dict[str, Any]) -> Optional[Path]:
    """Return a cached merged ONNX path without exporting or loading weights."""
    probe_kwargs = dict(compile_kwargs)
    probe_kwargs["_layerwise_cache_probe"] = True
    cached = qeff_model.compile(**probe_kwargs)
    if isinstance(cached, dict):
        cached = next(
            (
                cached.get(key)
                for key in ("lang_decode_qpc_path", "lang_prefill_qpc_path", "lang_qpc_path")
                if cached.get(key) is not None
            ),
            None,
        )
    return Path(cached) if cached is not None else None


def _install_window_patches_for(model_type: str) -> None:
    """Install the HF __init__/shard patches needed for the given model_type.

    The shard-file patch makes ``from_pretrained`` skip checkpoint shards that
    only contain weights for layers outside the active window, so loading an
    N-layer model in a window of size 1 reads ~1/N of the disk. The init
    patch nulls the unused decoder layers right after the model is built so
    the full layer list is never instantiated in memory.
    """
    _install_shard_window_patch()
    candidates = []
    qwen3_vl_moe_mod = getattr(getattr(transformers.models, "qwen3_vl_moe", None), "modeling_qwen3_vl_moe", None)
    if qwen3_vl_moe_mod is not None and "qwen3_vl_moe" in model_type:
        candidates.extend(
            cls
            for name in ("Qwen3VLMoeForConditionalGeneration", "Qwen3VLMoeForCausalLM")
            if (cls := getattr(qwen3_vl_moe_mod, name, None)) is not None
        )
    qwen3_5_moe_mod = getattr(getattr(transformers.models, "qwen3_5_moe", None), "modeling_qwen3_5_moe", None)
    if qwen3_5_moe_mod is not None and model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        candidates.extend(
            cls
            for name in ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM")
            if (cls := getattr(qwen3_5_moe_mod, name, None)) is not None
        )
    qwen3_moe_mod = getattr(getattr(transformers.models, "qwen3_moe", None), "modeling_qwen3_moe", None)
    if qwen3_moe_mod is not None and model_type in {"qwen3_moe"}:
        candidates.extend(
            cls for name in ("Qwen3MoeForCausalLM",) if (cls := getattr(qwen3_moe_mod, name, None)) is not None
        )
    for cls in candidates:
        _install_window_patch(cls)


@contextmanager
def _layerwise_export_env():
    """Toggle the layerwise-active flag on QEFFBaseModel for the inner block.

    No environment variables are touched - the flag is a pure in-process class
    attribute, which keeps the API self-contained and lets concurrent Python
    interpreters (e.g. test workers) operate independently.
    """
    base = QEfficient.base.modeling_qeff.QEFFBaseModel
    prev_active = getattr(base, "_layerwise_active", False)
    prev_state_active = _LAYERWISE_STATE["active"]
    base._layerwise_active = True
    _LAYERWISE_STATE["active"] = 1
    try:
        yield
    finally:
        base._layerwise_active = prev_active
        _LAYERWISE_STATE["active"] = prev_state_active


def _resolve_text_total_layers(config) -> int:
    text_config = getattr(config, "text_config", config)
    total = getattr(text_config, "num_hidden_layers", None)
    if total is None:
        raise ValueError("Could not resolve `num_hidden_layers` from config.text_config / config.")
    return int(total)


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_layerwise(
    *,
    model_id: str,
    config,
    qeff_factory,
    compile_kwargs: Dict[str, Any],
    probe_qeff_model=None,
    window_size: int = 1,
    final_compile: bool = True,
) -> Any:
    """Drive the per-window export loop and (optionally) the final stitched compile.

    Parameters
    ----------
    model_id : str
        HF id / path passed to ``qeff_factory(model_id, config)`` to (re)build a
        QEff model fresh per window.
    config : transformers.PretrainedConfig
        Already-mutated config (the caller is responsible for any vision tweaks
        like ``deepstack_visual_indexes``).
    qeff_factory : Callable[[str, PretrainedConfig], QEffModel]
        Factory invoked once per window to materialize a fresh QEff model that
        only loads the active window's weights.
    compile_kwargs : dict
        Forwarded verbatim to ``qeff_model.compile(...)`` per window. The driver
        injects ``skip_lang`` per-window and ``lang_onnx_path`` for the final
        stitched compile.
    probe_qeff_model : QEffModel, optional
        Existing wrapper used for cache probing before any per-window weights are
        loaded. In normal layerwise use this is the outer meta wrapper.
    window_size : int
        Number of text-decoder layers per window. ``1`` matches the legacy
        example.
    final_compile : bool
        When True (compile path), do the final QPC compile on the merged ONNX
        and return ``qpc_paths``. When False (export path), return the merged
        ONNX path only.

    Returns
    -------
    Either the qpc_paths dict (final_compile=True) or the merged ONNX path
    (final_compile=False).
    """
    _maybe_warn_deprecation()
    model_type = assert_layerwise_supported(config)

    text_total_layers = _resolve_text_total_layers(config)
    text_cfg = getattr(config, "text_config", config)
    text_cfg.num_hidden_layers = text_total_layers

    _ensure_pretrained_window_attrs()
    _install_window_patches_for(model_type)

    windows = _build_layer_windows(text_total_layers, window_size)
    logger.info(
        "Layerwise export start: model_type=%s total_layers=%d window_size=%d windows=%d "
        "torch_threads=%d torch_interop_threads=%d",
        model_type,
        text_total_layers,
        window_size,
        len(windows),
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )
    if torch.get_num_threads() <= 1:
        logger.warning(
            "Layerwise export is running with torch.get_num_threads()=%d; ONNX export may appear single-core. "
            "For faster export, increase PyTorch/OMP threads (for example, TORCH_NUM_THREADS or OMP_NUM_THREADS).",
            torch.get_num_threads(),
        )
    first_onnx_path: Optional[Path] = None
    last_qeff_model = None
    needs_deterministic_init = model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} and not _checkpoint_has_shard_index(
        model_id
    )
    if needs_deterministic_init:
        init_rng_snapshot = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }

        def _build_window_model():
            # Tiny/test checkpoints can miss some tensors and rely on random
            # initialization. Reusing one RNG snapshot per window preserves
            # layerwise-vs-default parity for those models.
            prev_python_state = random.getstate()
            prev_numpy_state = np.random.get_state()
            prev_torch_state = torch.random.get_rng_state()
            try:
                random.setstate(init_rng_snapshot["python"])
                np.random.set_state(init_rng_snapshot["numpy"])
                torch.random.set_rng_state(init_rng_snapshot["torch"])
                return qeff_factory(model_id, config)
            finally:
                random.setstate(prev_python_state)
                np.random.set_state(prev_numpy_state)
                torch.random.set_rng_state(prev_torch_state)

    else:

        def _build_window_model():
            return qeff_factory(model_id, config)

    try:
        with _layerwise_export_env():
            _set_layer_windows(0, min(window_size, text_total_layers), text_total_layers)
            if needs_deterministic_init:
                _LAYERWISE_STATE["force_full_init"] = 1
            cached_probe = probe_qeff_model or _build_window_model()
            cached_onnx_path = _cached_layerwise_onnx_path(cached_probe, compile_kwargs)
            if cached_onnx_path is not None:
                logger.info("Layerwise cache hit: reusing merged ONNX at %s", cached_onnx_path)
                first_onnx_path = cached_onnx_path
                last_qeff_model = cached_probe

            for text_start, text_end in windows:
                if cached_onnx_path is not None:
                    break
                window_t0 = time.perf_counter()
                logger.info("Layerwise window export start: [%d, %d)", text_start, text_end)
                _set_layer_windows(text_start, text_end, text_total_layers)
                if needs_deterministic_init:
                    _LAYERWISE_STATE["force_full_init"] = 1

                qeff_model = _build_window_model()
                last_qeff_model = qeff_model
                if hasattr(qeff_model, "model"):
                    _null_outside_window_layers(qeff_model.model, apply_text=True)
                _slim_for_window_export(qeff_model, ctx_len=compile_kwargs.get("ctx_len"))

                window_kwargs = dict(compile_kwargs)
                # skip_lang is a VLM-only kwarg; only inject when present in caller's kwargs.
                if "skip_lang" in window_kwargs:
                    window_kwargs["skip_lang"] = False
                onnx_path = qeff_model.compile(**window_kwargs)
                if first_onnx_path is None:
                    if isinstance(onnx_path, dict):
                        lang_key = next(
                            (
                                k
                                for k in (
                                    "lang_decode_qpc_path",
                                    "lang_prefill_qpc_path",
                                    "lang_qpc_path",
                                )
                                if k in onnx_path
                            ),
                            None,
                        )
                        if lang_key is None:
                            raise RuntimeError(f"Layer-wise window produced no lang_*_qpc_path: keys={list(onnx_path)}")
                        lang_path = onnx_path[lang_key]
                    else:
                        lang_path = onnx_path
                    first_onnx_path = Path(str(lang_path))
                logger.info(
                    "Layerwise window export done: [%d, %d) in %.2fs",
                    text_start,
                    text_end,
                    time.perf_counter() - window_t0,
                )
    finally:
        _reset_layer_windows()

    if first_onnx_path is None:
        raise RuntimeError("Layer-wise export produced no ONNX shards.")

    export_root = _resolve_export_root(first_onnx_path)
    model_name = _resolve_layerwise_model_name(last_qeff_model)
    root_cached_onnx = _cached_root_onnx(export_root, model_name)
    if root_cached_onnx is not None:
        canonical_onnx = root_cached_onnx
    elif cached_onnx_path is not None:
        cached_path = Path(cached_onnx_path)
        canonical_onnx = (
            _relocate_merged_onnx_to_root(export_root, cached_path) if _is_cached_merged(cached_path) else cached_path
        )
    else:
        final_artifact = _stitch_layerwise_if_available(export_root, text_total_layers)
        logger.info("Layerwise merged ONNX: %s", final_artifact)
        canonical_onnx = Path(final_artifact)
    if canonical_onnx.parent == export_root:
        _cleanup_layerwise_intermediates(export_root, keep_onnx=canonical_onnx)
    logger.info("Layerwise canonical ONNX: %s", canonical_onnx)

    if not final_compile:
        return str(canonical_onnx)

    final_kwargs = dict(compile_kwargs)
    if hasattr(last_qeff_model, "lang_model"):
        final_kwargs["lang_onnx_path"] = str(canonical_onnx)
    else:
        final_kwargs["onnx_path"] = str(canonical_onnx)
    final_kwargs.setdefault("skip_lang", False)
    return last_qeff_model.compile(**final_kwargs)
