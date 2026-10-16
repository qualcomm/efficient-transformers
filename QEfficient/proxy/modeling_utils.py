# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Configuration helpers for representative proxy language stacks."""

from collections import Counter

from transformers import AutoConfig

from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")
_CONFIG_LOAD_KWARGS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "subfolder",
    "token",
    "trust_remote_code",
)


def _get_language_config(config):
    for config_name in ("text_config", "llm_config", "language_config"):
        if language_config := getattr(config, config_name, None):
            return language_config
    return config


def _get_layer_signature(config, layer_idx: int):
    signature = []

    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        signature.append(("layer_type", layer_types[layer_idx % len(layer_types)]))
    elif getattr(config, "sliding_window", None) is not None and hasattr(config, "sliding_window_pattern"):
        pattern = config.sliding_window_pattern
        attention_type = "sliding_attention" if (layer_idx + 1) % pattern else "full_attention"
        signature.append(("layer_type", attention_type))

    num_experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", 0)
    sparse_step = getattr(config, "decoder_sparse_step", None)
    if num_experts and sparse_step:
        mlp_only_layers = set(getattr(config, "mlp_only_layers", None) or [])
        is_sparse = layer_idx not in mlp_only_layers and (layer_idx + 1) % sparse_step == 0
        signature.append(("feed_forward", "moe" if is_sparse else "dense"))
    elif num_experts and (first_dense_layer := getattr(config, "first_k_dense_replace", 0)):
        signature.append(("feed_forward", "dense" if layer_idx < first_dense_layer else "moe"))

    return tuple(signature) or (("decoder", "default"),)


def _get_signature_horizon(config, configured_num_layers: int) -> int:
    """Return how many layer positions are needed to observe the configured cadence."""
    horizon = configured_num_layers
    if layer_types := getattr(config, "layer_types", None):
        horizon = max(horizon, len(layer_types))
    if getattr(config, "sliding_window", None) is not None:
        horizon = max(horizon, getattr(config, "sliding_window_pattern", 1))
    horizon = max(horizon, getattr(config, "decoder_sparse_step", 1) or 1)
    horizon = max(horizon, (getattr(config, "first_k_dense_replace", 0) or 0) + 1)
    return horizon


def apply_proxy_layer_config(
    config, minimum_calls_per_layer_type: int = 2, num_hidden_layers: int | None = None
) -> int:
    """Reduce the language stack while retaining repeated instances of every layer type."""
    language_config = _get_language_config(config)
    configured_num_layers = (
        num_hidden_layers if num_hidden_layers is not None else get_num_layers_from_config(language_config)
    )
    if configured_num_layers is None:
        raise ValueError("Proxy mode requires the language config to define `num_hidden_layers`.")
    if configured_num_layers < 1:
        raise ValueError("Proxy mode requires at least one language layer.")
    if num_hidden_layers is not None:
        language_config.num_hidden_layers = num_hidden_layers
        logger.info("Proxy language model uses caller-provided layer count: %d", num_hidden_layers)
        return num_hidden_layers

    signature_horizon = _get_signature_horizon(language_config, configured_num_layers)
    original_signatures = [_get_layer_signature(language_config, layer_idx) for layer_idx in range(signature_horizon)]
    required_signatures = set(original_signatures)
    signature_counts = Counter()
    proxy_num_layers = signature_horizon
    for layer_idx in range(signature_horizon * minimum_calls_per_layer_type):
        signature_counts[original_signatures[layer_idx % signature_horizon]] += 1
        if all(signature_counts[signature] >= minimum_calls_per_layer_type for signature in required_signatures):
            proxy_num_layers = layer_idx + 1
            break

    layer_types = getattr(language_config, "layer_types", None)
    if layer_types:
        language_config.layer_types = [
            layer_types[layer_idx % len(layer_types)] for layer_idx in range(proxy_num_layers)
        ]

    mlp_only_layers = getattr(language_config, "mlp_only_layers", None)
    if mlp_only_layers is not None:
        mlp_only_layer_set = set(mlp_only_layers)
        language_config.mlp_only_layers = [
            layer_idx for layer_idx in range(proxy_num_layers) if layer_idx % signature_horizon in mlp_only_layer_set
        ]

    language_config.num_hidden_layers = proxy_num_layers
    logger.info(
        "Proxy language model uses %d of %d layers with at least %d calls for each of %d layer types",
        proxy_num_layers,
        signature_horizon,
        minimum_calls_per_layer_type,
        len(required_signatures),
    )
    return proxy_num_layers


def prepare_proxy_config(pretrained_model_name_or_path, kwargs: dict):
    """Load and mutate the model config before checkpoint weights are materialized."""
    explicit_num_hidden_layers = kwargs.pop("num_hidden_layers", None)
    config = kwargs.get("config")
    if config is None:
        config_kwargs = {key: kwargs[key] for key in _CONFIG_LOAD_KWARGS if key in kwargs}
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)

    if explicit_num_hidden_layers is None:
        apply_proxy_layer_config(config)
    else:
        apply_proxy_layer_config(config, num_hidden_layers=explicit_num_hidden_layers)
    kwargs["config"] = config
    return config
