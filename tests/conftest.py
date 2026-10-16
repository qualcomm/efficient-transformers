# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import functools
import json
import logging as py_logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import pytest
from transformers import logging as hf_logging

_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")
_two_phase_session = os.environ.get("QEFF_PER_PR_SHARED_HOME") or os.environ.get("QEFF_PER_PR_COMPILE_WARM_ONLY")
if _xdist_worker and os.environ.get("QEFF_HOME") and not _two_phase_session:
    os.environ["QEFF_HOME"] = str(Path(os.environ["QEFF_HOME"]) / _xdist_worker)
    Path(os.environ["QEFF_HOME"]).mkdir(parents=True, exist_ok=True)

from QEfficient.utils.cache import QEFF_HOME  # noqa: E402
from QEfficient.utils.device_utils import (  # noqa: E402
    get_qaic_mdp_device_groups,
    parse_qaic_device_groups,
)
from QEfficient.utils.logging_utils import QEFFLogger  # noqa: E402
from tests.two_phase import is_compile_warm_phase, is_two_phase_session  # noqa: E402


class _NoOpLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


logger = _NoOpLogger()


@pytest.fixture(autouse=True)
def finalize_qeff_run_after_test():
    """Close the model timing run at the end of each pytest test."""
    yield
    QEFFLogger.finish_run()


_QUICKCHECK_FILE = "tests/unit_test/models/test_model_quickcheck.py"
_QUICKCHECK_SUMMARY = {}
_QUICKCHECK_META = {
    "test_causal_lm_cpu_runtime_parity_with_api_runner": (
        "Causal LM",
        "Full parity: HF PyTorch vs QEff PyTorch vs ORT tokens",
    ),
    "test_vlm_text_side_runtime_parity_and_full_export": (
        "VLM",
        "Text-side full parity + full VLM export smoke",
    ),
    "test_vlm_export_smoke_additional_models": (
        "VLM",
        "Export smoke with text-side fallback when needed",
    ),
    "test_text_embedding_cpu_parity_and_export": (
        "Text Embedding",
        "Tensor parity: HF vs QEff PyTorch vs ORT",
    ),
    "test_audio_embedding_ctc_cpu_parity_and_export": (
        "Audio CTC",
        "Logits parity: HF vs ORT + export",
    ),
    "test_seq_classification_cpu_parity_and_export": (
        "Sequence Classification",
        "Logits parity: HF vs QEff PyTorch vs ORT",
    ),
    "test_whisper_export_smoke": (
        "Whisper",
        "Export smoke + retained-state outputs check",
    ),
    "test_causal_subfunction_export_smoke": (
        "Causal LM",
        "Subfunction export check (with/without QEffGPT2Block)",
    ),
    "test_qwen_quickcheck_subfunction_registration": (
        "Qwen",
        "Tiny synthetic decoder subfunction registration",
    ),
    "test_qwen_quickcheck_subfunction_setup_toggle": (
        "Qwen",
        "Subfunction/non-subfunction setup without ONNX export",
    ),
    "test_qwen_moe_quickcheck_layerwise_mode": (
        "Qwen MoE",
        "Layerwise/non-layerwise decoder registration",
    ),
    "test_qwen_quickcheck_hf_qeff_ort_default_parity": (
        "Qwen",
        "Full logits parity: HF PyTorch vs QEff PyTorch vs ORT (default export)",
    ),
    "test_qwen_moe_quickcheck_hf_qeff_ort_prefill_only_parity": (
        "Qwen MoE",
        "Full logits parity: HF PyTorch vs QEff PyTorch vs ORT (prefill-only transform)",
    ),
    "test_qwen_moe_quickcheck_layerwise_hf_qeff_ort_parity": (
        "Qwen MoE",
        "Full logits parity: HF PyTorch vs QEff PyTorch vs ORT (layerwise)",
    ),
    "test_repeat_kv_quickcheck_hf_qeff_ort_parity": (
        "Causal LM",
        "RepeatKV parity: HF PyTorch vs QEff PyTorch vs ORT logits",
    ),
    "test_causal_subfunction_export_smoke_all_models": (
        "Causal LM",
        "Full parity: HF PyTorch vs QEff PyTorch vs ORT tokens (subfunctions)",
    ),
    "test_prefix_caching_continuous_batching_export_and_ort_smoke": (
        "Prefix Caching",
        "Continuous-batching export structural checks",
    ),
    "test_awq_export_smoke": (
        "AWQ",
        "Export smoke + MatMulNBits presence check",
    ),
}

# Reduce noisy PyTorch C++ warning logs (e.g., torchvision op registration warnings)
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


def _is_nightly_pipeline_session(session):
    """Check if this is a nightly_pipeline test session"""
    # Check invocation args
    if hasattr(session.config, "invocation_params"):
        args_str = " ".join(session.config.invocation_params.args)
        if "nightly_pipeline" in args_str:
            return True

    # Check if any collected items are from nightly_pipeline
    if hasattr(session, "items") and session.items:
        for item in session.items:
            if "nightly_pipeline" in item.nodeid:
                return True

    return False


def qeff_models_clean_up(qeff_dir=QEFF_HOME):
    """
    Clean up QEFF models and cache.

    Args:
        qeff_dir: Can be a string (file/dir path), PosixPath, or list of strings/PosixPath objects
                 If a file path is provided, its parent directory will be deleted
    """
    if isinstance(qeff_dir, (str, Path)):
        paths = [qeff_dir]
    else:
        paths = qeff_dir

    for path in paths:
        try:
            path_str = str(path)
            if os.path.isfile(path_str):
                dir_to_delete = os.path.dirname(path_str)
                if os.path.exists(dir_to_delete):
                    shutil.rmtree(dir_to_delete)
                    print(f"\n.............Cleaned up {dir_to_delete}")
            elif os.path.isdir(path_str):
                if os.path.exists(path_str):
                    shutil.rmtree(path_str)
                    print(f"\n.............Cleaned up {path_str}")
        except Exception as e:
            print(f"\n.............Error cleaning up {path}: {e}")


@pytest.fixture
def manual_cleanup():
    """Fixture to manually trigger cleanup"""
    return qeff_models_clean_up


# Number of QAic cards on the CI machine. Workers are sharded round-robin
# across these cards via QAIC_VISIBLE_DEVICES. Override with QEFF_NUM_QAIC_CARDS
# if a host has a different count.
_QAIC_CARDS_DEFAULT = 4


def _xdist_worker_index():
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker or not worker.startswith("gw"):
        return None
    try:
        return int(worker[2:])
    except ValueError:
        return None


def _is_two_phase_shared_home_session():
    """True when the run uses one shared QEFF_HOME across a compile/execute split.

    In this mode the per-worker QEFF_HOME remap and the session-level cache wipe
    are both skipped: every worker must share one QEFF_HOME so Phase B hits the
    QPCs Phase A warmed, and the session-start/finish rmtree of QEFF_HOME would
    otherwise destroy that warm cache (Phase A on finish, Phase B on start). The
    caller owns the shared QEFF_HOME lifecycle (starts clean, cleans up when the
    whole two-phase run is done). See tests/two_phase.py for the phase flags.
    """
    return is_two_phase_session()


@pytest.fixture(scope="session", autouse=True)
def _qaic_device_for_xdist_worker():
    """Pin each pytest-xdist worker to its own slice of the QAic cards.

    Serial runs (no xdist) and runs that already export QAIC_VISIBLE_DEVICES
    are left untouched. Under ``pytest -n 4`` on a 4-card host, gw0..gw3 each
    own one card -- so .compile()/.generate() across workers run in parallel,
    while same-worker calls remain sequential on that card.

    QEFF_QAIC_CARDS_PER_WORKER widens that slice for tests that load more than one
    QPC at a time. A dual-QPC (kv_offload=True) VLM builds a vision *and* a language
    session, and each is compiled at the default num_cores=16 -- a whole card -- so
    such a test needs two visible cards or the second qaicrt.Program() fails with
    "Failed to create program with Qpc Buffer". Width 1 (the default) is correct for
    single-QPC LLM tests; the multimodal execute phase sets 2, giving gw0 cards 0,1
    and gw1 cards 2,3. Note QAIC_VISIBLE_DEVICES renumbers, so a worker masked to
    "2,3" sees device ids 0,1 -- absolute ids must never be passed as device_ids
    alongside a mask.

    QEFF_QAIC_DEVICE_GROUPS takes precedence when set by the CI stage. It assigns
    an explicit semicolon-delimited group to each worker during pytest_configure,
    before this fixture runs. Both paths fail instead of wrapping worker indexes,
    so two workers can never silently receive the same device slice.

    QEFF_QAIC_CARD_OFFSET allows two stages to run simultaneously on non-
    overlapping card slices. E.g. stage A sets QEFF_NUM_QAIC_CARDS=2 + offset=0
    -> cards 0,1; stage B sets QEFF_NUM_QAIC_CARDS=2 + offset=2 -> cards 2,3.

    The compile-warm phase is left unmasked: it never reaches a device, and masking
    would break the mdp_num_partitions>1 compiles, which require that many *visible*
    devices even though no program is ever created.
    """
    if "QAIC_VISIBLE_DEVICES" in os.environ or is_compile_warm_phase():
        return
    idx = _xdist_worker_index()
    if idx is None:
        return
    cards = max(1, int(os.environ.get("QEFF_NUM_QAIC_CARDS", _QAIC_CARDS_DEFAULT)))
    offset = int(os.environ.get("QEFF_QAIC_CARD_OFFSET", 0))
    per_worker = max(1, int(os.environ.get("QEFF_QAIC_CARDS_PER_WORKER", 1)))
    slots = cards // per_worker
    if idx >= slots:
        raise pytest.UsageError(
            f"QAIC worker gw{idx} has no exclusive {per_worker}-device slice; "
            f"only {slots} slices are available from {cards} devices"
        )
    base = offset + idx * per_worker
    os.environ["QAIC_VISIBLE_DEVICES"] = ",".join(str(base + i) for i in range(per_worker))


@pytest.fixture(scope="session", autouse=True)
def _qeff_home_per_xdist_worker():
    """Give each xdist worker its own QEFF_HOME and TMPDIR subdirs.

    This keeps compile-cache writes from racing and prevents concurrent
    qaic-compile processes from sharing the stage-level scratch directory.
    Serial runs are untouched.

    Setting os.environ alone is not enough because QEfficient.utils.cache and
    QEfficient.utils.export_utils bind QEFF_HOME to a module-level constant at
    import time.  We patch those constants directly so every runtime call to
    _prepare_export_directory() resolves to the per-worker path.

    Falls back to QEfficient.utils.cache's own resolved default when the caller
    hasn't exported QEFF_HOME: without this, unset QEFF_HOME made this fixture a
    no-op, so every worker silently shared one content-addressed QPC cache dir
    with no lock -- concurrent workers compiling the same model tore each
    other's QPC files, surfacing as a "Failed to create program with Qpc Buffer"
    error at generate() time with no clue it was a race. Isolation must not
    depend on the caller remembering to set QEFF_HOME first.

    Exception: in the two-phase compile-warm mode, every worker must share one
    QEFF_HOME so the execute phase hits the QPC cache warmed by the compile
    phase; the per-worker QEFF_HOME remap is skipped (see
    _is_two_phase_shared_home_session). The per-worker TMPDIR remap is NOT
    skipped: QEFF_HOME is content-addressed so concurrent workers write disjoint
    QPC dirs, but qaic-compile's scratch is TMPDIR-relative, so a shared TMPDIR
    lets concurrent compiles tear each other's intermediate files and emit a
    valid-looking but numerically wrong QPC. That warm QPC is then cached and
    reused by the execute phase, surfacing as a silent on-device parity failure.
    """
    idx = _xdist_worker_index()
    if idx is None:
        return

    import QEfficient.utils.cache as _cache_mod
    import QEfficient.utils.export_utils as _export_mod

    base = os.environ.get("QEFF_HOME") or str(_cache_mod.QEFF_HOME)

    # Two-phase shared-home run: keep QEFF_HOME shared (Phase B must hit Phase A's
    # warm QPCs) but still give each worker a private TMPDIR to isolate the
    # qaic-compile scratch. The private TMPDIR nests under the shared base so the
    # caller's single rm -rf of QEFF_HOME still reclaims it.
    if _is_two_phase_shared_home_session():
        worker_tmp = Path(base) / ".worker_tmp" / f"worker_{idx}"
        worker_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(worker_tmp)
        return

    worker_home = Path(base) / f"worker_{idx}"
    worker_home.mkdir(parents=True, exist_ok=True)
    worker_tmp = worker_home / ".tmp"
    worker_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["QEFF_HOME"] = str(worker_home)
    os.environ["TMPDIR"] = str(worker_tmp)
    _cache_mod.QEFF_HOME = worker_home
    _export_mod.QEFF_HOME = worker_home


def pytest_sessionstart(session):
    logger.info("PYTEST Session Starting ...")
    # Skip cleanup for nightly_pipeline tests
    if _is_nightly_pipeline_session(session):
        logger.info("Skipping cleanup for nightly_pipeline tests")
        return
    # Suppress transformers warnings about unused weights when loading models with fewer layers
    hf_logging.set_verbosity_error()

    # Suppress noisy ONNX torchvision-missing warnings from torch exporter internals.
    py_logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(py_logging.ERROR)
    py_logging.getLogger("torch.onnx").setLevel(py_logging.ERROR)

    if _is_two_phase_shared_home_session():
        logger.info("Skipping session-start cleanup: two-phase shared QEFF_HOME run")
        return

    # Only the controller wipes the shared QEFF_HOME base, and it does so before
    # any worker is spawned. Workers must not run this rmtree: it targets the base
    # dir (the parent of every per-worker QEFF_HOME/TMPDIR subdir), so a late-booting
    # worker's sessionstart would recursively delete a sibling worker's in-flight
    # compile scratch, tearing qaic-compile temp files ('fd != -1') and QPC buffers
    # ('Failed to create program with Qpc Buffer') mid-run. sessionfinish already
    # guards this the same way.
    if getattr(session.config, "workerinput", None) is None:
        qeff_models_clean_up()


def pytest_configure(config):
    """Register custom markers for test categorization."""
    global logger
    # When using multiple workers, pytest creates one controller process and
    # one process for each worker. Suppress logging in the controller and keep
    # real QEfficient log files only for worker processes.
    config_option = getattr(config, "option", None)
    if getattr(config_option, "numprocesses", None) and getattr(config, "workerinput", None) is None:
        logger = _NoOpLogger()
    else:
        logger = QEFFLogger.get_logger("INFRA")

    explicit_device_groups = os.environ.get("QEFF_QAIC_DEVICE_GROUPS")
    if _xdist_worker and (explicit_device_groups or os.environ.get("QEFF_ISOLATE_QAIC_WORKERS") == "1"):
        try:
            devices_per_worker = int(os.environ.get("QEFF_QAIC_DEVICES_PER_WORKER", 4))
            if devices_per_worker < 1:
                raise ValueError(f"QEFF_QAIC_DEVICES_PER_WORKER must be positive, got {devices_per_worker}")
            device_groups = (
                parse_qaic_device_groups(explicit_device_groups)
                if explicit_device_groups
                else get_qaic_mdp_device_groups(devices_per_group=devices_per_worker)
            )
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
        invalid_groups = [device_ids for device_ids in device_groups if len(device_ids) != devices_per_worker]
        if invalid_groups:
            raise pytest.UsageError(
                f"Every QAIC worker must have exactly {devices_per_worker} devices; invalid groups: {invalid_groups}"
            )
        worker_index = int(_xdist_worker.removeprefix("gw"))
        if worker_index >= len(device_groups):
            raise pytest.UsageError(
                f"QAIC worker {_xdist_worker} has no isolated {devices_per_worker}-device group; "
                f"found {len(device_groups)} groups"
            )
        os.environ["QAIC_VISIBLE_DEVICES"] = ",".join(map(str, device_groups[worker_index]))
        logger.info("Assigned %s to QAIC devices %s", _xdist_worker, os.environ["QAIC_VISIBLE_DEVICES"])

    config.addinivalue_line("markers", "llm_model: mark test as a pure LLM model inference test")
    config.addinivalue_line(
        "markers", "feature: mark test as a feature-specific test (SPD, sampler, prefix caching, LoRA, etc.)"
    )
    config.addinivalue_line(
        "markers",
        "embedding_audio_model: mark test as a text-embedding / audio (CTC, speech-seq2seq) model test",
    )

    # Validate QEFF_MODEL_TIER here rather than in the collection hook. pytest_configure
    # runs in the controller AND in every pytest-xdist worker before collection, and a
    # pytest.UsageError raised here is reported cleanly; the same error raised inside
    # pytest_collection_modifyitems crashes an xdist worker with an INTERNALERROR. When a
    # real tier is selected, build the identity index now so a conflicting model_type in
    # tests/configs fails fast at configure time instead of mid-collection. With the
    # default tier (all) the index is never built, so default runs are untouched.
    tier = (os.environ.get("QEFF_MODEL_TIER") or _TIER_ALL).strip().lower()
    if tier not in _VALID_TIERS:
        raise pytest.UsageError(f"Invalid QEFF_MODEL_TIER={tier!r}. Expected one of: {', '.join(_VALID_TIERS)}.")
    if tier != _TIER_ALL:
        _model_identity_index()


# ---------------------------------------------------------------------------
# Model tiering (QEFF_MODEL_TIER)
# ---------------------------------------------------------------------------
# Narrows the *model matrix* a run covers, so a per-PR job can test only the
# actively-developed families instead of every onboarded architecture.
#
#   all       (default) every model -- unchanged historical behaviour
#   priority  only models matched by tests/configs/model_tiers.json
#   legacy    only models NOT matched there
#
# Filtering happens once here, at collection, rather than in the ~96 parametrize
# sites across the suite. Tests that carry no model identity (unit tests, CLI,
# feature/infra tests) are never deselected: tiering may only narrow the model
# matrix, never drop infrastructure coverage.

_TIER_ALL = "all"
_TIER_PRIORITY = "priority"
_TIER_LEGACY = "legacy"
_VALID_TIERS = (_TIER_ALL, _TIER_PRIORITY, _TIER_LEGACY)

_TIER_REGISTRY_PATH = Path(__file__).parent / "configs" / "model_tiers.json"
_CONFIG_DIR = Path(__file__).parent / "configs"

# Parameter names that carry a model identity, in resolution order. A dict param
# (``model_config``) is authoritative because it holds ``model_type`` directly;
# the string params are resolved through the config index built below.
_MODEL_PARAM_KEYS = ("model_config", "model_name", "model_id", "model", "model_type")

# Config entry fields that can identify a model. ``model_name`` is the primary
# identity; the rest are aliases resolved in a second pass, because some suites
# parametrize on ``entry["id"]`` (e.g. "CB qwen") rather than a Hugging Face card,
# and those entries carry no ``model_type`` of their own.
_ENTRY_PRIMARY_NAME_FIELD = "model_name"
_ENTRY_ALIAS_FIELDS = ("id",)
_ENTRY_REFERENT_FIELDS = ("model_name", "target_model_name", "draft_model_name")


def _merge_card_model_types(model_name, previous, current):
    """Resolve a card shared by a VLM config and its nested text config.

    Some tiny-random VLM cards are also used to instantiate their standalone language
    backbone. In that case the two valid types differ only by the ``_text`` suffix;
    retain the top-level VLM type for string-parametrized tests. Dict-parametrized text
    tests carry their own model_type and do not use this index value.
    """
    if previous == current or current is None:
        return previous
    if previous is None:
        return current

    previous_base = previous.removesuffix("_text")
    current_base = current.removesuffix("_text")
    if previous_base == current_base:
        return previous_base

    raise pytest.UsageError(
        f"Conflicting model_type for {model_name!r} in tests/configs: "
        f"{previous!r} vs {current!r}. A card must map to one model family."
    )


@functools.lru_cache(maxsize=1)
def _priority_tier_spec():
    """Load the priority tier declaration from ``tests/configs/model_tiers.json``."""
    with open(_TIER_REGISTRY_PATH, "r") as handle:
        registry = json.load(handle)
    priority = registry.get(_TIER_PRIORITY, {})
    return (
        frozenset(priority.get("model_types", ())),
        frozenset(priority.get("markers", ())),
        frozenset(priority.get("model_names", ())),
    )


@functools.lru_cache(maxsize=1)
def _model_marker_universe():
    """Every pytest marker that identifies a model family rather than a capability.

    A test carrying one of these is a model test even when it has no model parameter
    (the diffusers pipelines), so it can be tiered. Capability markers such as
    ``on_qaic`` or ``diffusion_models`` are deliberately absent: they describe how a
    test runs, not which model it covers.
    """
    with open(_TIER_REGISTRY_PATH, "r") as handle:
        registry = json.load(handle)
    return frozenset(registry.get("model_markers", ()))


@functools.lru_cache(maxsize=1)
def _model_identity_index():
    """Map every model identifier found in ``tests/configs/*.json`` to its ``model_type``.

    Built in two passes. The first indexes ``model_name`` -> ``model_type`` and records
    bare-string entries (audio / sequence-classification cards that carry no metadata)
    as untyped. The second resolves alias keys (an entry's ``id``) by chaining through
    the model the entry actually refers to, so a suite parametrizing on ``"CB qwen"``
    still resolves to ``qwen2`` via that entry's ``target_model_name``.

    A key mapping to ``None`` is known but untyped -- it can only be tiered by an
    explicit ``model_names`` listing. A VLM card may also represent its nested language
    config when the types differ only by ``_text``; the top-level type is retained.
    Other conflicting ``model_type`` values raise ``pytest.UsageError``.
    """
    entries = []
    index = {}
    for config_path in sorted(_CONFIG_DIR.glob("*.json")):
        if config_path == _TIER_REGISTRY_PATH:
            continue
        try:
            with open(config_path, "r") as handle:
                config_data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(config_data, dict):
            continue
        for entry_list in config_data.values():
            if not isinstance(entry_list, list):
                continue
            for entry in entry_list:
                if isinstance(entry, str):
                    # Bare-string entry (audio / sequence-classification configs): a
                    # Hugging Face card with no model_type. Record it so it can be tiered
                    # by an explicit model_names listing; never clobber a typed entry.
                    index.setdefault(entry, None)
                    continue
                if not isinstance(entry, dict):
                    continue
                entries.append(entry)
                name = entry.get(_ENTRY_PRIMARY_NAME_FIELD)
                if not isinstance(name, str):
                    continue
                model_type = entry.get("model_type")
                index[name] = _merge_card_model_types(name, index.get(name), model_type)

    for entry in entries:
        for alias_field in _ENTRY_ALIAS_FIELDS:
            alias = entry.get(alias_field)
            if not isinstance(alias, str) or index.get(alias) is not None:
                continue
            model_type = entry.get("model_type")
            if model_type is None:
                # Untyped alias (SPD entries): inherit from the model it points at.
                for referent_field in _ENTRY_REFERENT_FIELDS:
                    referent = entry.get(referent_field)
                    if isinstance(referent, str) and index.get(referent) is not None:
                        model_type = index[referent]
                        break
            index[alias] = model_type

    return index


def _is_priority_model(model_type, model_name, marker_names):
    """True when a resolved model identity belongs to the priority tier."""
    priority_types, priority_markers, priority_names = _priority_tier_spec()
    if marker_names & priority_markers:
        return True
    if model_type is not None and model_type in priority_types:
        return True
    return model_name is not None and model_name in priority_names


def _resolve_item_model(item):
    """Extract ``(model_type, model_name)`` for a collected test item.

    Returns ``None`` when the item carries no model identity at all -- such items are
    tier-agnostic and must never be deselected. A test with no model parameter but
    with a model-family marker (the diffusers pipelines) resolves to ``(None, None)``:
    tierable, with the marker alone deciding the tier.
    """
    params = getattr(getattr(item, "callspec", None), "params", None) or {}
    model_type = None
    model_name = None

    for key in _MODEL_PARAM_KEYS:
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, dict):
            # Registry-style dict param: authoritative, carries model_type directly.
            model_type = model_type or value.get("model_type")
            model_name = model_name or value.get("model_name")
        elif isinstance(value, str):
            if key == "model_type":
                model_type = model_type or value
            else:
                model_name = model_name or value

    if model_type is None and model_name is None:
        # No model parameter. Only tierable if a model-family marker identifies it.
        if {marker.name for marker in item.iter_markers()} & _model_marker_universe():
            return None, None
        return None

    if model_type is None:
        # A bare string param: resolve through the config index. An unknown name
        # leaves model_type None, which _is_priority_model treats as non-priority.
        model_type = _model_identity_index().get(model_name)

    return model_type, model_name


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Deselect model tests outside the tier requested by ``QEFF_MODEL_TIER``.

    Runs ``trylast`` so it sees the stage's true selection: the builtin ``mark`` plugin
    applies ``-m`` / ``-k`` in its own ``pytest_collection_modifyitems`` at default order,
    which runs *before* this one. Tiering must narrow whatever that leaves, not the full
    pre-``-m`` collection.
    """
    tier = (os.environ.get("QEFF_MODEL_TIER") or _TIER_ALL).strip().lower()
    # Tier already validated in pytest_configure; re-normalise defensively.
    if tier not in _VALID_TIERS or tier == _TIER_ALL:
        return

    want_priority = tier == _TIER_PRIORITY
    kept, deselected = [], []
    for item in items:
        resolved = _resolve_item_model(item)
        if resolved is None:
            # No model identity -- unit/CLI/feature/infra test. Always kept.
            kept.append(item)
            continue
        model_type, model_name = resolved
        marker_names = {marker.name for marker in item.iter_markers()}
        if _is_priority_model(model_type, model_name, marker_names) == want_priority:
            kept.append(item)
        else:
            deselected.append(item)

    if not kept and deselected:
        # Tiering emptied a stage that DID collect model tests (e.g. Reranker under
        # legacy: every model is priority). Deselecting them all would leave pytest with
        # nothing collected and exit 5, failing the Jenkins stage for a legitimately
        # empty tier. Convert to skips so the stage passes clean and reports what was
        # skipped. A stage that was already empty (typo'd -m, no model tests) is left to
        # exit 5 as before, because deselected is empty here.
        skip_marker = pytest.mark.skip(reason=f"QEFF_MODEL_TIER={tier}: no {tier}-tier model tests in this stage")
        for item in deselected:
            item.add_marker(skip_marker)
        logger.info("QEFF_MODEL_TIER=%s: stage has no %s-tier model tests; skipping %d", tier, tier, len(deselected))
        return

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
    logger.info("QEFF_MODEL_TIER=%s: selected %d model tests, deselected %d", tier, len(kept), len(deselected))


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
    # Skip cleanup for nightly_pipeline tests
    if _is_nightly_pipeline_session(session):
        logger.info("Skipping cleanup for nightly_pipeline tests")
        return
    if _is_two_phase_shared_home_session():
        logger.info("Skipping session-finish cleanup: two-phase shared QEFF_HOME run")
        return
    if inside_worker is None:
        qeff_models_clean_up()
        logger.info("...PYTEST Session Ended.")


def pytest_runtest_logreport(report):
    if _QUICKCHECK_FILE not in report.nodeid:
        return

    if report.when == "call":
        _QUICKCHECK_SUMMARY[report.nodeid] = report.outcome
        return

    if report.when == "setup" and report.outcome == "skipped":
        _QUICKCHECK_SUMMARY.setdefault(report.nodeid, report.outcome)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    group_stats = defaultdict(lambda: defaultdict(int))
    seen_status = set()
    seen_total = set()

    def _group_from_report(report):
        keywords = getattr(report, "keywords", {}) or {}
        if "llm_model" in keywords:
            return "llm_model"
        if "embedding_audio_model" in keywords:
            return "embedding_audio_model"
        if "feature" in keywords:
            return "feature"
        return "unmarked"

    for status in ("passed", "failed", "skipped", "xfailed", "xpassed", "error"):
        for report in terminalreporter.stats.get(status, []):
            nodeid = getattr(report, "nodeid", None)
            when = getattr(report, "when", "call")
            if not nodeid or when != "call":
                continue
            group = _group_from_report(report)

            status_key = (group, nodeid, status)
            if status_key in seen_status:
                continue
            seen_status.add(status_key)
            group_stats[group][status] += 1

            total_key = (group, nodeid)
            if total_key not in seen_total:
                seen_total.add(total_key)
                group_stats[group]["total"] += 1

    headers = ["group", "total", "passed", "failed", "skipped", "xfailed", "xpassed", "error"]
    rows = []
    order = ["llm_model", "embedding_audio_model", "feature", "unmarked"]
    for group in order:
        if group not in group_stats:
            continue
        rows.append([group] + [str(group_stats[group][name]) for name in headers[1:]])

    if rows:
        widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

        def fmt(row):
            return " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

        terminalreporter.write_sep("-", "QEff Test Summary")
        terminalreporter.write_line(fmt(headers))
        terminalreporter.write_line("-+-".join("-" * w for w in widths))
        for row in rows:
            terminalreporter.write_line(fmt(row))

        xfailed_reports = [r for r in terminalreporter.stats.get("xfailed", []) if getattr(r, "when", "call") == "call"]
        failed_reports = [r for r in terminalreporter.stats.get("failed", []) if getattr(r, "when", "call") == "call"]

        if xfailed_reports:
            terminalreporter.write_sep("-", "Known Limitations (xfailed)")
            for report in xfailed_reports:
                reason = getattr(getattr(report, "longrepr", None), "reprcrash", None)
                reason_text = reason.message if reason and hasattr(reason, "message") else "expected failure"
                terminalreporter.write_line(f"- {report.nodeid}: {reason_text}")

        if failed_reports:
            terminalreporter.write_sep("-", "Failures")
            for report in failed_reports:
                terminalreporter.write_line(f"- {report.nodeid}")

    if _QUICKCHECK_SUMMARY:
        terminalreporter.section("Quickcheck Coverage Summary", sep="=")
        header = f"{'Status':7}  {'Test Case':58}  {'Category':24}  Validation"
        terminalreporter.write_line(header)
        terminalreporter.write_line("-" * len(header))

        for nodeid in sorted(_QUICKCHECK_SUMMARY):
            test_case = nodeid.split("::", 1)[1]
            base_name = test_case.split("[", 1)[0]
            category, validation = _QUICKCHECK_META.get(base_name, ("Other", "N/A"))
            status = _QUICKCHECK_SUMMARY[nodeid].upper()
            terminalreporter.write_line(f"{status:7}  {test_case:58}  {category:24}  {validation}")
