# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Golden-output store for the causal-LM parity tests.

The QAIC LLM CI stage used to run four legs per model on every build -- HuggingFace
PyTorch (``hf``), the transformed KV PyTorch model (``qeff_hf``), ONNXRuntime (``ORT``)
and the on-device QAIC run. The three CPU legs dominate wall time yet the ``hf``
reference is a pure function of the model, so we persist it once as a committed
"golden" and, in the steady state, run only the QAIC leg and compare its tokens
against the golden.

The golden token stream depends only on the model, its *effective* config
(layer/head/dim counts), dtype, continuous-batching flag, prompts and the
prompt/ctx/generation lengths. It is independent of ``qaic_config`` (blocking, CCL,
speculative decoding, KV-head replication), so one golden variant is reused across
every QAIC variant of the same model.

Layout (a single committed file under ``tests/golden_outputs``)::

    golden_outputs/goldens.json
        {
          "<family>": {
            "<model_name>": {
              "<variant_key>": {"pytorch_hf_tokens": [...], "config_fp": "...", ...}
            }
          }
        }

All families and models share one file; ``<family>`` (e.g. ``causal_lm``) is the
top-level key, so the mechanism still scales to new model families without adding
files. Set ``QEFF_REGENERATE_GOLDEN=1`` to recompute and overwrite goldens.
"""

import hashlib
import json
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux is the supported test environment.
    fcntl = None

import numpy as np

# ``tests/transformers/models/golden_utils.py`` -> ``tests`` is parents[2].
GOLDEN_ROOT = Path(__file__).resolve().parents[2] / "golden_outputs"

# Single committed store holding every family / model / variant.
GOLDEN_FILE = GOLDEN_ROOT / "goldens.json"

# Fixed, ordered set of numeric config attributes that determine the HF token stream.
# Kept small and explicit so the fingerprint is stable across HF versions; any alias
# naming (e.g. ``n_head`` vs ``num_attention_heads``) is folded in via the *_ALIASES map.
_FINGERPRINT_ATTRS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "intermediate_size",
)
_FINGERPRINT_ALIASES = {
    "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "hidden_size": ("hidden_size", "n_embd", "d_model"),
    "num_attention_heads": ("num_attention_heads", "n_head", "num_heads"),
    "num_key_value_heads": ("num_key_value_heads", "num_kv_heads"),
    "vocab_size": ("vocab_size",),
    "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
}


def _dtype_str(torch_dtype) -> str:
    return str(torch_dtype).replace("torch.", "") if torch_dtype is not None else "none"


def config_fingerprint(config) -> str:
    """Short deterministic digest of the config attributes that steer HF generation.

    Two runs with the same fingerprint (same model_type, layer/head/dim/vocab counts)
    produce the same greedy token stream, so their goldens are interchangeable. This is
    what lets dummy / few / full layer scopes key distinct goldens without threading the
    ``n_layer`` argument through the store.
    """
    values = {}
    for attr in _FINGERPRINT_ATTRS:
        value = None
        for alias in _FINGERPRINT_ALIASES.get(attr, (attr,)):
            candidate = getattr(config, alias, None)
            if candidate is not None:
                value = candidate
                break
        values[attr] = value
    canonical = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


# Volatile keys HF stamps on every config that don't affect the generated token stream
# (library version, checkpoint path, dtype metadata -- dtype is threaded through the VLM
# variant key explicitly instead).
_VOLATILE_CONFIG_KEYS = frozenset({"transformers_version", "_name_or_path", "_commit_hash", "torch_dtype", "dtype"})


def config_to_dict_fingerprint(config) -> str:
    """Content-hash of a full HF config, for architectures too varied for a fixed attribute list.

    ``config_fingerprint`` above works for causal-LM configs because their token-relevant
    attributes are a small, stable set. VLM configs nest a ``text_config``/``vision_config``/
    ``llm_config`` whose architecture-defining fields differ per model family (e.g. Qwen3-VL's
    ``deepstack_visual_indexes`` vs Gemma3's ``layer_types``), so a fixed whitelist would need
    updating for every onboarded model. Hashing the full serialized dict picks up any test-time
    override -- layer counts, sliding-window pattern, vision depth -- without maintaining one.
    """

    def _strip(node):
        if isinstance(node, dict):
            return {k: _strip(v) for k, v in node.items() if k not in _VOLATILE_CONFIG_KEYS}
        if isinstance(node, list):
            return [_strip(v) for v in node]
        return node

    canonical = json.dumps(_strip(config.to_dict()), sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


def vlm_golden_variant_key(
    *,
    torch_dtype,
    prompt_text: str,
    image_url: str,
    image_fingerprint: str,
    generation_len: int,
    config_fp: str,
) -> str:
    """Deterministic variant key for one VLM golden.

    Mirrors ``golden_variant_key``: the HF token stream for a VLM is a pure function of the
    model + effective config plus the processed image/prompt pair, dtype, and the generation
    length. The URL alone is insufficient because unavailable external images are replaced by a
    checked-in fallback image. It does not depend on ``kv_offload`` or ``qaic_config`` (those
    only steer the QEff / on-device leg), so one golden is reused across every variant of those
    knobs for the same model.
    """
    payload = {
        "dtype": _dtype_str(torch_dtype),
        "prompt": prompt_text,
        "image_url": image_url,
        "image_fingerprint": image_fingerprint,
        "generation_len": generation_len,
        "config_fp": config_fp,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{_dtype_str(torch_dtype)}_gl{generation_len}_{config_fp}_{digest}"


def golden_variant_key(
    *,
    continuous_batching: bool,
    torch_dtype,
    prompt_len: int,
    ctx_len: int,
    generation_len: Optional[int],
    full_batch_size: Optional[int],
    prompts,
    config_fp: str,
) -> str:
    """Deterministic, semi-readable key for one golden variant.

    The readable prefix aids manual inspection of the JSON; the trailing digest folds in
    the full parameter set (including prompts and config fingerprint) so distinct inputs
    never collide.
    """
    payload = {
        "cb": bool(continuous_batching),
        "dtype": _dtype_str(torch_dtype),
        "prompt_len": prompt_len,
        "ctx_len": ctx_len,
        "generation_len": generation_len,
        "full_batch_size": full_batch_size,
        "prompts": list(prompts),
        "config_fp": config_fp,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    scope = "cb" if continuous_batching else "nocb"
    return f"{scope}_{_dtype_str(torch_dtype)}_pl{prompt_len}_cl{ctx_len}_gl{generation_len}_{config_fp}_{digest}"


def _read_variant(family: str, model_name: str, variant_key: str) -> Optional[dict]:
    """Return the stored record for a variant or ``None`` if absent/unreadable.

    Unreadable is treated as absent: an outside-lock read can race a concurrent write,
    in which case the caller falls through to the locked generate path.
    """
    if not GOLDEN_FILE.is_file():
        return None
    try:
        with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return doc.get(family, {}).get(model_name, {}).get(variant_key)


def _write_variant(family: str, model_name: str, variant_key: str, record: dict) -> None:
    """Merge ``record`` into the single golden store with an atomic replace.

    The store is a ``family -> model -> variant -> record`` nesting; existing entries are
    preserved so concurrent models accumulate into one file. Callers must hold the write
    lock, which serializes the read-modify-write against other workers.
    """
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc = {}
    if GOLDEN_FILE.is_file():
        try:
            with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            doc = {}
    doc.setdefault(family, {}).setdefault(model_name, {})[variant_key] = record

    fd, tmp_path = tempfile.mkstemp(dir=str(GOLDEN_FILE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            # Keep the committed file newline-terminated so regeneration doesn't churn the diff.
            f.write("\n")
        os.replace(tmp_path, GOLDEN_FILE)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class _golden_write_lock:
    """Serialize golden generation across concurrent xdist workers.

    A single process-wide lock guards the one shared store, so a read-modify-write in one
    worker never races another. The lock file lives in a ``.locks`` dir under the golden
    tree (gitignored), mirroring the ``QEFF_HOME/.locks`` idiom used for export/compile.
    Anchoring it there -- rather than the system temp dir, which can fall back to the CWD
    when ``/tmp`` is full -- keeps stray lock files out of the committed repo. A no-op when
    ``fcntl`` is unavailable.
    """

    def __init__(self):
        self._lockfile = None
        if fcntl is None:
            return
        lock_dir = GOLDEN_ROOT / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = lock_dir / "goldens.lock"

    def __enter__(self):
        if fcntl is None:
            return self
        self._lockfile = open(self._lock_path, "a+", encoding="utf-8")
        fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc):
        if self._lockfile is not None:
            fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_UN)
            self._lockfile.close()
            self._lockfile = None


def resolve_hf_golden(
    family: str,
    model_name: str,
    variant_key: str,
    params: dict,
    compute_fn: Callable[[], np.ndarray],
) -> np.ndarray:
    """Return the golden HF tokens for a variant, computing them live if missing.

    Reads the committed golden first; only when the variant is absent (or
    ``QEFF_REGENERATE_GOLDEN=1``) does ``compute_fn`` run the HF model.

    The store is only ever written under ``QEFF_REGENERATE_GOLDEN=1``. A plain run that
    misses still computes the reference so the parity assert stays meaningful, but it must
    not mutate a committed file as a side effect -- that would leave the tree dirty in CI
    and let an unreviewed token stream become the baseline. Regeneration is deliberately a
    separate, explicit step. The expensive HF run happens outside the lock; the store is
    re-checked under the lock before writing so parallel workers compute each variant at
    most once without serializing unrelated models behind one another.
    """
    regenerate = os.environ.get("QEFF_REGENERATE_GOLDEN") == "1"

    if not regenerate:
        record = _read_variant(family, model_name, variant_key)
        if record is not None:
            return np.array(record["pytorch_hf_tokens"])
        warnings.warn(
            f"No committed golden for {family}/{model_name}/{variant_key}; running the HF "
            "reference live. Commit a golden (QEFF_REGENERATE_GOLDEN=1) to skip the CPU leg.",
            stacklevel=2,
        )
        return np.asarray(compute_fn())

    tokens = np.asarray(compute_fn())
    record = {
        **params,
        "variant_key": variant_key,
        "pytorch_hf_tokens": tokens.tolist(),
        "gen_len": int(tokens.shape[-1]),
        "timestamp": datetime.now().isoformat(),
    }
    with _golden_write_lock():
        _write_variant(family, model_name, variant_key, record)
    return tokens
