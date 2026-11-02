# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Tests for the QEFF_MODEL_TIER collection filter declared in ``tests/conftest.py``.

The tier hook narrows the model matrix a CI run covers (``all`` / ``priority`` /
``legacy``). These tests pin the invariants that keep it safe:

  * priority and legacy partition the matrix exactly -- no model is lost or double-run
  * tests carrying no model identity are never deselected (infra coverage is preserved)
  * every ``model_type`` declared in the registry actually exists in ``tests/configs``
"""

import ast
import json
import os
from pathlib import Path

import pytest

from tests.conftest import (
    _MODEL_PARAM_KEYS,
    _is_priority_model,
    _merge_card_model_types,
    _model_identity_index,
    _model_marker_universe,
    _priority_tier_spec,
    _resolve_item_model,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
REGISTRY_PATH = CONFIG_DIR / "model_tiers.json"
TESTS_ROOT = Path(__file__).resolve().parents[2]


class _FakeMarker:
    def __init__(self, name):
        self.name = name


class _FakeCallspec:
    def __init__(self, params):
        self.params = params


class _FakeItem:
    """Minimal stand-in for a pytest item -- only what the tier hook reads."""

    def __init__(self, params=None, markers=()):
        if params is not None:
            self.callspec = _FakeCallspec(params)
        self._markers = [_FakeMarker(name) for name in markers]

    def iter_markers(self):
        return iter(self._markers)


def _all_config_entries():
    """Every dict entry across ``tests/configs/*.json``, excluding the registry itself."""
    for config_path in sorted(CONFIG_DIR.glob("*.json")):
        if config_path == REGISTRY_PATH:
            continue
        with open(config_path, "r") as handle:
            config_data = json.load(handle)
        for key, entries in config_data.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    yield config_path.name, key, entry


# --------------------------------------------------------------------------- #
# Registry hygiene
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_registry_declares_only_priority_tier():
    """Legacy must stay defined by exclusion, so new models can never fall out of CI."""
    with open(REGISTRY_PATH, "r") as handle:
        registry = json.load(handle)
    tier_keys = {key for key in registry if not key.startswith("_") and key != "model_markers"}
    assert tier_keys == {"priority"}, (
        f"Only the priority tier may be declared (legacy is everything else); found {sorted(tier_keys)}"
    )


@pytest.mark.cpu_only
def test_priority_markers_are_declared_model_markers():
    """A priority marker outside the universe would be kept in BOTH tiers.

    Marker-only tests (the diffusers pipelines) are tierable solely because the marker
    is listed in ``model_markers``. If a priority marker were missing from that list,
    its tests would resolve as tier-agnostic and run in priority *and* legacy.
    """
    _, priority_markers, _ = _priority_tier_spec()
    undeclared = sorted(priority_markers - _model_marker_universe())
    assert not undeclared, (
        f"priority.markers entries missing from model_markers: {undeclared}. "
        "Add them to model_markers or those tests will run in both tiers."
    )


@pytest.mark.cpu_only
def test_model_markers_exclude_capability_markers():
    """Capability markers describe how a test runs, not which model it covers.

    Listing one would make every test in that stage tierable and silently halve
    infrastructure coverage.
    """
    capability_markers = {
        "on_qaic",
        "qnn",
        "cli",
        "finetune",
        "multimodal",
        "diffusion_models",
        "regular",
        "nightly",
        "vllm",
        "dummy_layers",
        "few_layers",
        "full_layers",
        "llm_model",
        "feature",
        "embedding_audio_model",
    }
    overlap = sorted(_model_marker_universe() & capability_markers)
    assert not overlap, f"model_markers must name model families, not capabilities: {overlap}"


@pytest.mark.cpu_only
def test_registry_model_types_exist_in_configs():
    """A typo'd model_type would silently match nothing -- fail loudly instead."""
    priority_types, _, _ = _priority_tier_spec()
    known_types = {entry.get("model_type") for _, _, entry in _all_config_entries()}
    known_types.discard(None)
    unknown = sorted(priority_types - known_types)
    assert not unknown, (
        f"model_tiers.json lists model_types absent from tests/configs: {unknown}. "
        "Either the family is not onboarded yet or the name is misspelled."
    )


@pytest.mark.cpu_only
def test_registry_model_names_exist_in_configs():
    _, _, priority_names = _priority_tier_spec()
    known_names = set(_model_identity_index())
    unknown = sorted(priority_names - known_names)
    assert not unknown, f"model_tiers.json lists model_names absent from tests/configs: {unknown}"


@pytest.mark.cpu_only
def test_audio_and_embedding_cards_resolve_to_priority():
    """The current audio / text-embedding cards are pinned to priority by model_names.

    These carry no ``model_type`` (bare strings or ``model_type: null``), so they can
    only reach priority through an explicit ``model_names`` listing. If one is dropped
    from the registry -- or the index stops recording bare strings -- it silently falls
    to legacy and stops running in a priority per-PR job. Pin the promotion here.
    """
    index = _model_identity_index()
    promoted = [
        "openai/whisper-tiny",
        "facebook/wav2vec2-base-960h",
        "jinaai/jina-embeddings-v2-base-code",
        "sentence-transformers/nli-bert-base-cls-pooling",
    ]
    for card in promoted:
        assert card in index, f"{card!r} not indexed from tests/configs -- bare-string entry dropped?"
        assert index[card] is None, f"{card!r} unexpectedly carries a model_type; update this guard"
        assert _is_priority_model(index[card], card, set()) is True, (
            f"{card!r} no longer resolves to priority -- check model_tiers.json model_names"
        )


# --------------------------------------------------------------------------- #
# Drift guards
# --------------------------------------------------------------------------- #
# The tier hook resolves a test's model identity from a fixed set of parametrize
# argnames and treats an unresolvable identity as legacy. Two kinds of silent drift
# would erode that: a new untyped card that no one pins, and a new identity-shaped
# argname the hook does not read. Both fail safe (extra legacy coverage, never a
# dropped model), but silently -- these guards surface them at review time instead.

# Cards known to carry no model_type: bare-string configs (audio / sequence
# classification) and dict entries with an explicit ``model_type: null`` (grok-1,
# SwiftKV). Each can only be tiered priority by an explicit model_names listing;
# left unpinned it is legacy. This is the allowlist of *accepted* untyped cards --
# a new one forces a conscious decision (pin to priority, or accept it as legacy).
_KNOWN_UNTYPED = frozenset(
    {
        "CB llama",  # SPD alias -> JackFram/llama-160m, itself untyped
        "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",  # model_type: null in causal configs
        "hpcai-tech/grok-1",  # model_type: null in causal configs
        "meta-llama/Llama-Prompt-Guard-2-22M",  # bare string, sequence_model_configs
        "openai/whisper-tiny",  # bare string, audio_model_configs (pinned priority)
        "facebook/wav2vec2-base-960h",  # bare string, audio_model_configs (pinned priority)
        "jinaai/jina-embeddings-v2-base-code",  # model_type null, embedding (pinned priority)
        "sentence-transformers/nli-bert-base-cls-pooling",  # model_type null, embedding (pinned priority)
    }
)

# Parametrize argnames that look like a model identity but that the tier hook
# deliberately does NOT read (they are not in _MODEL_PARAM_KEYS), with the reason
# each is safe to leave tier-agnostic. A new model-identity argname outside both
# this set and _MODEL_PARAM_KEYS means real model coverage the hook cannot tier --
# it would run in both priority and legacy. Add it to _MODEL_PARAM_KEYS (so it is
# tiered) or waive it here (documenting why it is not a tierable identity).
_WAIVED_IDENTITY_ARGNAMES = frozenset(
    {
        "base_model_name",  # LoRA base; adapters are the unit under test, not the family
        "make_model",  # a model-factory callable, not a card/type
        "model_cfg",  # reranker unit-test config dict, tests config reduction not a family
        "model_class",  # nightly artifact filename, not a live model run
    }
)


@pytest.mark.cpu_only
def test_untyped_identifiers_are_an_explicit_allowlist():
    """A new untyped card must be a conscious choice, not silent legacy drift."""
    index = _model_identity_index()
    untyped = {name for name, model_type in index.items() if model_type is None}
    unexpected = sorted(untyped - _KNOWN_UNTYPED)
    stale = sorted(_KNOWN_UNTYPED - untyped)
    assert not unexpected, (
        f"New untyped model identifiers in tests/configs: {unexpected}. "
        "Give the config entry a model_type, or -- if it is intentionally untyped -- "
        "add it to _KNOWN_UNTYPED and decide whether it belongs in model_tiers.json "
        "model_names (priority) or should stay legacy."
    )
    assert not stale, (
        f"_KNOWN_UNTYPED lists identifiers no longer present or now typed: {stale}. Remove them from the allowlist."
    )


@pytest.mark.cpu_only
def test_no_new_identity_shaped_parametrize_argnames():
    """Guard against a model-identity argname the tier hook cannot see.

    Sweeps every ``@pytest.mark.parametrize`` under ``tests/`` for argnames that look
    like a model identity. Any such argname must either be read by the hook
    (``_MODEL_PARAM_KEYS``) or be explicitly waived, so new model coverage can never
    slip in on an argname the hook silently ignores and runs in both tiers.
    """
    readable = set(_MODEL_PARAM_KEYS)
    found = {}  # argname -> first "path:line" seen
    for py_path in sorted(TESTS_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(py_path.read_text(), str(py_path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "parametrize" or not node.args:
                continue
            first = node.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            for argname in (part.strip() for part in first.value.split(",")):
                if not argname or "model" not in argname.lower():
                    continue
                found.setdefault(argname, f"{py_path.relative_to(TESTS_ROOT)}:{first.lineno}")

    unaccounted = {
        argname: where
        for argname, where in found.items()
        if argname not in readable and argname not in _WAIVED_IDENTITY_ARGNAMES
    }
    assert not unaccounted, (
        "New model-identity-shaped parametrize argname(s) the tier hook does not read: "
        f"{unaccounted}. If one carries a real model identity, add it to _MODEL_PARAM_KEYS "
        "in tests/conftest.py so it is tiered; otherwise waive it in "
        "_WAIVED_IDENTITY_ARGNAMES with the reason it is not a tierable identity."
    )


# --------------------------------------------------------------------------- #
# Identity index
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_identity_index_resolves_model_type_for_known_card():
    index = _model_identity_index()
    assert index["tiny-random/qwen3.5"] == "qwen3_5"
    assert index["hf-internal-testing/tiny-random-GPT2LMHeadModel"] == "gpt2"


@pytest.mark.cpu_only
def test_identity_index_merges_vlm_and_nested_text_types():
    assert _merge_card_model_types("tiny", "qwen3_5_text", "qwen3_5") == "qwen3_5"
    assert _merge_card_model_types("tiny", "gemma3", "gemma3_text") == "gemma3"
    with pytest.raises(pytest.UsageError, match="Conflicting model_type"):
        _merge_card_model_types("tiny", "qwen3_5", "gemma3")


@pytest.mark.cpu_only
def test_identity_index_keys_on_entry_id_for_spd_style_params():
    """SPD suites parametrize on entry["id"] (e.g. "CB qwen"), not on an HF card.

    Those entries carry no ``model_type``, so the index must chain the alias through
    the model the entry refers to -- otherwise a Qwen SPD case lands in legacy.
    """
    index = _model_identity_index()
    assert index["CB qwen"] == "qwen2"
    # "CB llama" points at JackFram/llama-160m, which has no typed entry anywhere in
    # tests/configs, so it stays unresolved -- known, but untyped, hence legacy.
    assert "CB llama" in index
    assert index["CB llama"] is None


@pytest.mark.cpu_only
def test_spd_alias_tiers_follow_the_referenced_model():
    index = _model_identity_index()
    assert _is_priority_model(index["CB qwen"], "CB qwen", set()) is True
    assert _is_priority_model(index["CB llama"], "CB llama", set()) is False


@pytest.mark.cpu_only
def test_conflicting_model_type_for_one_card_is_rejected(tmp_path, monkeypatch):
    """One card mapping to two model_types would tier inconsistently by file order.

    The index build must fail loudly instead. Point the loader at a throwaway config
    dir carrying the conflict and clear the cache around the call so the real index is
    untouched.
    """
    import tests.conftest as tier

    (tmp_path / "a_config.json").write_text(
        json.dumps({"models": [{"model_name": "acme/collide", "model_type": "llama"}]})
    )
    (tmp_path / "b_config.json").write_text(
        json.dumps({"models": [{"model_name": "acme/collide", "model_type": "qwen2"}]})
    )

    monkeypatch.setattr(tier, "_CONFIG_DIR", tmp_path)
    tier._model_identity_index.cache_clear()
    try:
        with pytest.raises(pytest.UsageError, match="Conflicting model_type for 'acme/collide'"):
            tier._model_identity_index()
    finally:
        tier._model_identity_index.cache_clear()


@pytest.mark.cpu_only
def test_repeated_model_type_for_one_card_is_accepted(tmp_path, monkeypatch):
    """A card legitimately appears in several lists; the same model_type is not a conflict."""
    import tests.conftest as tier

    (tmp_path / "a_config.json").write_text(json.dumps({"models": [{"model_name": "acme/dup", "model_type": "llama"}]}))
    (tmp_path / "b_config.json").write_text(json.dumps({"more": [{"model_name": "acme/dup", "model_type": "llama"}]}))

    monkeypatch.setattr(tier, "_CONFIG_DIR", tmp_path)
    tier._model_identity_index.cache_clear()
    try:
        index = tier._model_identity_index()
        assert index["acme/dup"] == "llama"
    finally:
        tier._model_identity_index.cache_clear()


@pytest.mark.cpu_only
def test_xdist_worker_gets_isolated_qeff_home_and_tmpdir(tmp_path, monkeypatch):
    import QEfficient.utils.cache as cache_module
    import QEfficient.utils.export_utils as export_module
    import tests.conftest as test_config

    base_home = tmp_path / "qeff_home"
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw2")
    monkeypatch.setenv("QEFF_HOME", str(base_home))
    monkeypatch.delenv("QEFF_PER_PR_SHARED_HOME", raising=False)
    monkeypatch.delenv("QEFF_PER_PR_COMPILE_WARM_ONLY", raising=False)
    monkeypatch.setattr(cache_module, "QEFF_HOME", base_home)
    monkeypatch.setattr(export_module, "QEFF_HOME", base_home)

    test_config._qeff_home_per_xdist_worker.__wrapped__()

    worker_home = base_home / "worker_2"
    worker_tmp = worker_home / ".tmp"
    assert Path(os.environ["QEFF_HOME"]) == worker_home
    assert Path(os.environ["TMPDIR"]) == worker_tmp
    assert worker_home.is_dir()
    assert worker_tmp.is_dir()
    assert cache_module.QEFF_HOME == worker_home
    assert export_module.QEFF_HOME == worker_home


@pytest.mark.cpu_only
def test_xdist_worker_gets_exclusive_qaic_device_slice(monkeypatch):
    import tests.conftest as test_config

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    monkeypatch.setenv("QEFF_NUM_QAIC_CARDS", "4")
    monkeypatch.setenv("QEFF_QAIC_CARDS_PER_WORKER", "2")
    monkeypatch.delenv("QAIC_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("QEFF_PER_PR_COMPILE_WARM_ONLY", raising=False)

    test_config._qaic_device_for_xdist_worker.__wrapped__()

    assert os.environ["QAIC_VISIBLE_DEVICES"] == "2,3"


@pytest.mark.cpu_only
def test_xdist_worker_rejects_overlapping_qaic_device_slice(monkeypatch):
    import tests.conftest as test_config

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw2")
    monkeypatch.setenv("QEFF_NUM_QAIC_CARDS", "4")
    monkeypatch.setenv("QEFF_QAIC_CARDS_PER_WORKER", "2")
    monkeypatch.delenv("QAIC_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("QEFF_PER_PR_COMPILE_WARM_ONLY", raising=False)

    with pytest.raises(pytest.UsageError, match="no exclusive 2-device slice"):
        test_config._qaic_device_for_xdist_worker.__wrapped__()


@pytest.mark.cpu_only
def test_explicit_qaic_device_group_is_assigned_to_worker(monkeypatch):
    import tests.conftest as test_config

    class Config:
        @staticmethod
        def addinivalue_line(*args):
            return None

    monkeypatch.setattr(test_config, "_xdist_worker", "gw1")
    monkeypatch.setenv("QEFF_QAIC_DEVICE_GROUPS", "0,1,2,3;4,5,6,7")
    monkeypatch.setenv("QEFF_QAIC_DEVICES_PER_WORKER", "4")
    monkeypatch.setenv("QEFF_MODEL_TIER", "all")
    monkeypatch.delenv("QAIC_VISIBLE_DEVICES", raising=False)

    test_config.pytest_configure(Config())

    assert os.environ["QAIC_VISIBLE_DEVICES"] == "4,5,6,7"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_cores,free,total,eligible",
    [(14, 14, 14, True), (14, 14, 16, True), (14, 13, 14, False), (16, 14, 14, False), (16, 16, 16, True)],
)
def test_empty_explicit_qaic_device_groups_fall_back_to_discovery(monkeypatch, num_cores, free, total, eligible):
    from types import SimpleNamespace

    import QEfficient.utils.device_utils as device_utils
    import tests.conftest as test_config

    class Config:
        @staticmethod
        def addinivalue_line(*args):
            return None

    output = "".join(
        f"QID {device_id}\n    Status:Ready\n    Device Capabilities: HybridBoot+ MDP+\n"
        f"    Nsp Free:{free}\n    Nsp Total:{total}\n    Board serial:board-{device_id // 4}\n"
        for device_id in range(20)
    )
    monkeypatch.setattr(
        device_utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )
    monkeypatch.setattr(test_config, "NUM_AIC_CORES", num_cores)
    monkeypatch.setattr(test_config, "_xdist_worker", "gw4")
    monkeypatch.setenv("QEFF_QAIC_DEVICE_GROUPS", "")
    monkeypatch.setenv("QEFF_QAIC_DEVICES_PER_WORKER", "4")
    monkeypatch.setenv("QEFF_ISOLATE_QAIC_WORKERS", "1")
    monkeypatch.setenv("QEFF_MODEL_TIER", "all")
    monkeypatch.delenv("QAIC_VISIBLE_DEVICES", raising=False)

    if eligible:
        test_config.pytest_configure(Config())
        assert os.environ["QAIC_VISIBLE_DEVICES"] == "16,17,18,19"
    else:
        with pytest.raises(pytest.UsageError, match="no isolated 4-device group"):
            test_config.pytest_configure(Config())
        assert "QAIC_VISIBLE_DEVICES" not in os.environ


# --------------------------------------------------------------------------- #
# Tier classification
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "model_type",
    ["qwen3_5_text", "qwen3_moe", "llama4_text", "gpt_oss", "gemma4"],
)
def test_named_priority_families_are_priority(model_type):
    assert _is_priority_model(model_type, None, set()) is True


@pytest.mark.cpu_only
@pytest.mark.parametrize("model_type", ["gpt2", "falcon", "mpt", "codegen", "mixtral"])
def test_unlisted_families_are_legacy(model_type):
    assert _is_priority_model(model_type, None, set()) is False


@pytest.mark.cpu_only
def test_priority_marker_promotes_diffusion_pipelines():
    """Diffusers models are pipelines behind markers, not config entries."""
    assert _is_priority_model(None, None, {"wan"}) is True
    assert _is_priority_model(None, None, {"diffusion_models"}) is False


@pytest.mark.cpu_only
def test_unresolvable_model_is_treated_as_legacy():
    """An unknown card must not be promoted to priority by accident."""
    assert _is_priority_model(None, "some/never-onboarded-model", set()) is False


# --------------------------------------------------------------------------- #
# Item resolution
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_item_without_model_param_is_tier_agnostic():
    """Unit/CLI/feature tests must never be deselected by tiering."""
    assert _resolve_item_model(_FakeItem()) is None
    assert _resolve_item_model(_FakeItem(params={"kv_offload": True})) is None


@pytest.mark.cpu_only
def test_capability_marker_alone_stays_tier_agnostic():
    """diffusion_models/on_qaic describe a stage, not a model -- must not be tierable."""
    assert _resolve_item_model(_FakeItem(markers=("diffusion_models", "on_qaic"))) is None


@pytest.mark.cpu_only
def test_model_family_marker_makes_a_paramless_test_tierable():
    """Diffusers pipelines have no model param; the marker alone must decide the tier.

    Regression guard: while these resolved as tier-agnostic they were kept in BOTH
    the priority and legacy runs.
    """
    wan_item = _FakeItem(markers=("wan", "diffusion_models", "on_qaic"))
    assert _resolve_item_model(wan_item) == (None, None)
    assert _is_priority_model(None, None, {"wan", "diffusion_models"}) is True

    flux_item = _FakeItem(markers=("flux", "diffusion_models", "on_qaic"))
    assert _resolve_item_model(flux_item) == (None, None)
    assert _is_priority_model(None, None, {"flux", "diffusion_models"}) is False


@pytest.mark.cpu_only
def test_item_with_dict_param_resolves_directly():
    item = _FakeItem(params={"model_config": {"model_name": "tiny-random/qwen3.5", "model_type": "qwen3_5_text"}})
    assert _resolve_item_model(item) == ("qwen3_5_text", "tiny-random/qwen3.5")


@pytest.mark.cpu_only
def test_item_with_string_param_resolves_via_index():
    item = _FakeItem(params={"model_name": "tiny-random/gpt-oss-bf16"})
    model_type, model_name = _resolve_item_model(item)
    assert (model_type, model_name) == ("gpt_oss", "tiny-random/gpt-oss-bf16")


@pytest.mark.cpu_only
def test_item_with_explicit_model_type_param_is_honoured():
    """tests/dynamo parametrizes ("model_type", "model_id") together."""
    item = _FakeItem(params={"model_type": "gpt_oss", "model_id": "tiny-random/gpt-oss-bf16"})
    model_type, _ = _resolve_item_model(item)
    assert model_type == "gpt_oss"


# --------------------------------------------------------------------------- #
# Partition invariant
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_priority_and_legacy_partition_the_per_pr_causal_matrix():
    """Every per-PR model lands in exactly one tier, and both tiers are non-empty."""
    with open(CONFIG_DIR / "causal_model_configs.json", "r") as handle:
        models = json.load(handle)["per_pr_causal_text_models"]

    priority, legacy = [], []
    for entry in models:
        bucket = priority if _is_priority_model(entry["model_type"], entry["model_name"], set()) else legacy
        bucket.append(entry["id"])

    assert len(priority) + len(legacy) == len(models)
    assert not set(priority) & set(legacy)
    assert priority, "priority tier selected no per-PR causal model"
    assert legacy, "legacy tier selected no per-PR causal model"
    # The families the tier split was introduced for must be in the priority bucket.
    assert {"qwen3_5_dense_text", "llama4_text", "gpt_oss_moe_text"} <= set(priority)
