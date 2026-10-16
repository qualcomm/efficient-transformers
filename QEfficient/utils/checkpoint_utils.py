# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import shutil
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from QEfficient.utils._utils import hf_download
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


def huggingface_hub_cache_dir() -> Path:
    """Return the Hugging Face Hub cache root used by ``snapshot_download``."""
    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        return Path(hf_hub_cache).expanduser()
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home).expanduser() / "hub"
    if xdg_cache_home := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg_cache_home).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _all_checkpoint_files_under(root: Path, checkpoint_files: Sequence[str]) -> bool:
    """Return True when every checkpoint file is contained under ``root``."""
    root = root.expanduser()
    for checkpoint_file in checkpoint_files:
        try:
            Path(checkpoint_file).expanduser().relative_to(root)
        except ValueError:
            return False
    return True


def load_checkpoint_weights(checkpoint_path: str, keys: set[str]) -> dict[str, torch.Tensor]:
    """Read selected tensors from safetensors; pickle-backed PyTorch checkpoints are unsupported for security."""
    path = Path(checkpoint_path)
    if not path.is_dir():
        path = Path(hf_download(repo_id=checkpoint_path, allow_patterns=["*.safetensors"]))
    found: dict[str, torch.Tensor] = {}
    safetensors_files = sorted(path.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(
            f"No safetensors checkpoint found in {path}. Pickle-backed PyTorch checkpoints are not supported."
        )
    for safetensors_file in safetensors_files:
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as checkpoint:
            available_keys = set(checkpoint.keys())
            found.update({key: checkpoint.get_tensor(key) for key in keys & available_keys})
    return found


def load_checkpoint(model, checkpoint: str, strict=False, post_process_func=None):
    """load weights ending with `.safetensors` extension
    Args:
        model: model to load wights into
        checkpoint (str): checkpoint path
        strict (bool, optional): strictness of loading weights. Defaults to False.
        post_process_func (optional): Optional post-processing of loaded state dict. Defaults to None.
    Returns:
        model: model with applied weights
    """
    state_dict: dict = load_file(checkpoint)
    if post_process_func is not None:
        state_dict = post_process_func(state_dict)
    model.load_state_dict(state_dict, strict=strict)
    return model


# ---------------------------------------------------------------------------
# System-state helpers — used to derive worker counts at runtime
# ---------------------------------------------------------------------------


def available_ram_gb() -> float:
    """Available (free + reclaimable) RAM on the current machine in GB."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except ModuleNotFoundError:
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
        return 1.0


def cpu_count() -> int:
    """Logical CPU count with a safe fallback."""
    return os.cpu_count() or 8


# ---------------------------------------------------------------------------
# Safetensors checkpoint helpers
# ---------------------------------------------------------------------------


def safetensors_dtype_to_torch(dtype: str) -> torch.dtype | None:
    """Map a safetensors dtype string to the matching torch dtype."""
    return {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
    }.get(dtype)


def requires_dtype_conversion(src: Path, weight_map: dict[str, str], target_dtype: torch.dtype) -> bool:
    """Return True when any floating-point checkpoint tensor differs from ``target_dtype``."""
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(str(src / shard_name), framework="pt") as handle:
            keys = handle.keys()
            for key in keys:
                dtype = safetensors_dtype_to_torch(handle.get_slice(key).get_dtype())
                if dtype is not None and dtype != target_dtype:
                    return True
    return False


def read_weight_map(src: Path) -> dict[str, str]:
    """Return {tensor_key: shard_filename} from model.safetensors.index.json,
    or from model.safetensors directly for single-file checkpoints."""
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    shard_files = sorted(src.glob("*.safetensors"))
    if len(shard_files) == 1:
        with safe_open(str(shard_files[0]), framework="pt") as f:
            return {k: shard_files[0].name for k in f.keys()}
    if len(shard_files) == 0:
        raise FileNotFoundError(
            f"No safetensors files found in {src}. Weight-free export requires a safetensors checkpoint."
        )
    raise FileNotFoundError(
        f"{len(shard_files)} safetensors shards found in {src} but no "
        "model.safetensors.index.json. The checkpoint is malformed — "
        "re-download or regenerate the index file."
    )


def checkpoint_files_complete(src: Path) -> bool:
    """Return True when all indexed checkpoint shards are present and readable."""
    try:
        weight_map = read_weight_map(src)
    except Exception:
        return False

    for shard_name in set(weight_map.values()):
        shard_path = src / shard_name
        if not shard_path.is_file() or not os.access(shard_path, os.R_OK):
            return False
        try:
            with safe_open(str(shard_path), framework="pt"):
                pass
        except Exception:
            return False
    return True


@cache
def resolve_checkpoint_dir(model_id_or_path: str) -> Path:
    """Resolve a local or remote model reference to a checkpoint directory.

    Parameters
    ----------
    model_id_or_path : str
        Local model path or Hugging Face Hub model id.

    Returns
    -------
    Path
        Directory containing the downloaded or local checkpoint files.
    """
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate

    snapshot_dir = hf_download(
        repo_id=model_id_or_path,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.txt", "*.pdf", "*.msgpack", "*.h5", "*.pth"],
    )
    snapshot_path = Path(snapshot_dir)
    has_weights = (
        bool(list(snapshot_path.glob("*.safetensors"))) or (snapshot_path / "model.safetensors.index.json").exists()
    )

    if not has_weights:
        snapshot_dir = hf_download(
            repo_id=model_id_or_path,
            allow_patterns=["*.bin", "*.json"],
            ignore_patterns=[
                "*.onnx",
                "*.ot",
                "*.md",
                "*.txt",
                "*.pdf",
                "*.msgpack",
                "*.h5",
                "*.pth",
                "flax_model*",
                "tf_model*",
            ],
        )

    return Path(snapshot_dir)


def resolve_checkpoint_files(model_id_or_path: str) -> list[str]:
    """Return safetensors checkpoint files for a model reference.

    Parameters
    ----------
    model_id_or_path : str
        Local model path or Hugging Face Hub model id.

    Returns
    -------
    List[str]
        Sorted safetensors shard paths.
    """
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    checkpoint_files = sorted(str(path) for path in checkpoint_dir.glob("*.safetensors"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No safetensors checkpoint files found for {model_id_or_path}")
    return checkpoint_files


def checkpoint_root(model_id_or_path: str, checkpoint_files: Sequence[str]) -> Path | None:
    """Return the root directory used for relative checkpoint file paths.

    Parameters
    ----------
    model_id_or_path : str
        Local model path or Hugging Face Hub model id.
    checkpoint_files : Sequence[str]
        Checkpoint shard paths resolved for the model.

    Returns
    -------
    Optional[Path]
        Parent root for relative paths, or ``None`` when no checkpoint files exist.
    """
    if not checkpoint_files:
        return None

    hf_cache_root = huggingface_hub_cache_dir()
    if _all_checkpoint_files_under(hf_cache_root, checkpoint_files):
        return hf_cache_root

    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate.parent

    first_checkpoint = Path(checkpoint_files[0])
    for parent in first_checkpoint.parents:
        if parent.name.startswith("models--"):
            return parent.parent
    return first_checkpoint.parent


def load_checkpoint_index(checkpoint_files: list[str]) -> dict[str, str]:
    """Build a tensor-to-shard map by scanning safetensors checkpoint files.

    Parameters
    ----------
    checkpoint_files : List[str]
        Safetensors shard paths.

    Returns
    -------
    Dict[str, str]
        Mapping from checkpoint tensor key to the shard path containing it.
    """
    tensor_to_file = {}
    for checkpoint_file in checkpoint_files:
        with safe_open(checkpoint_file, framework="pt") as handle:
            keys = handle.keys()
            for key in keys:
                tensor_to_file[key] = checkpoint_file
    return tensor_to_file


def atomic_save(tensors: dict[str, torch.Tensor], dst: Path) -> None:
    """Write safetensors through a temporary file before replacing ``dst``."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
    tmp.replace(dst)
    try:
        dst.chmod(dst.stat().st_mode | 0o660)
    except OSError:
        pass


def write_index(out: Path, weight_map: dict[str, str]) -> None:
    """Write ``model.safetensors.index.json`` for a prepared checkpoint."""
    files = set(weight_map.values())
    total_size = sum((out / f).stat().st_size for f in files if (out / f).exists())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


# Sidecar files copied alongside a prepared checkpoint so it remains loadable
# via from_pretrained() without re-fetching the source checkpoint.
CHECKPOINT_AUX_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
]


def copy_checkpoint_aux_files(src: Path, out: Path) -> None:
    """Copy tokenizer and config sidecar files required by ``from_pretrained``."""
    for name in CHECKPOINT_AUX_FILES:
        src_file = src / name
        if src_file.exists() and not (out / name).exists():
            shutil.copy2(str(src_file), str(out / name))


def convert_bin_to_safetensors(src: Path, out: Path) -> None:
    """Load a .bin checkpoint via transformers and re-save as safetensors under ``out``.

    Uses save_pretrained(safe_serialization=True) which correctly handles tied
    weights and multi-shard layouts, writing model.safetensors (single file) or
    model-NNNNN-of-MMMMM.safetensors + model.safetensors.index.json (multi-shard).
    """
    import gc

    from transformers import AutoConfig, AutoModelForCausalLM

    if checkpoint_files_complete(out):
        return

    if out.exists():
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()

    out.mkdir(parents=True, exist_ok=True)
    copy_checkpoint_aux_files(src, out)
    logger.info(f"No safetensors files found in {src}. Auto-converting .bin → safetensors in {out}.")
    config = AutoConfig.from_pretrained(str(src), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.save_pretrained(str(out), safe_serialization=True)
    del model
    gc.collect()
    if not checkpoint_files_complete(out):
        raise FileNotFoundError(f"Failed to create a complete safetensors checkpoint in {out}")
    logger.info(f"Conversion complete — safetensors files written to {out}")
