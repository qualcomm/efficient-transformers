# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Configuration helpers for representative proxy stacks."""

from collections import Counter

from transformers import AutoConfig

from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")
PROXY_COMPONENT_LANGUAGE = "language"
PROXY_COMPONENT_VISION = "vision"
SUPPORTED_PROXY_COMPONENTS = (PROXY_COMPONENT_LANGUAGE, PROXY_COMPONENT_VISION)
_VISION_LAYER_COUNT_ATTRIBUTES = ("depth", "num_hidden_layers", "vt_num_hidden_layers")
_LAYER_INDEXED_SEQUENCE_ATTRIBUTES = {
    "cross_attention_layers",
    "cross_attn_layers",
    "deepstack_visual_indexes",
    "fullatt_block_indexes",
    "intermediate_layers_indices",
    "out_indices",
}
_LAYER_INDEXED_SCALAR_ATTRIBUTES = {"vision_feature_layer"}
_LAYER_COUNT_SCALAR_ATTRIBUTES = {"num_global_layers"}
_CONFIG_LOAD_KWARGS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "subfolder",
    "token",
    "trust_remote_code",
)


def normalize_proxy_component(proxy_component: str | None) -> str:
    proxy_component = PROXY_COMPONENT_LANGUAGE if proxy_component is None else str(proxy_component).lower()
    if proxy_component not in SUPPORTED_PROXY_COMPONENTS:
        raise ValueError(
            f"Unsupported proxy_component={proxy_component!r}. Expected one of {', '.join(SUPPORTED_PROXY_COMPONENTS)}."
        )
    return proxy_component


def _get_language_config(config):
    for config_name in ("text_config", "llm_config", "language_config"):
        if language_config := getattr(config, config_name, None):
            return language_config
    return config


def _get_vision_config(config):
    return getattr(config, "vision_config", None) or config


def _get_vision_num_layers(vision_config):
    for attribute in _VISION_LAYER_COUNT_ATTRIBUTES:
        if (num_layers := getattr(vision_config, attribute, None)) is not None:
            return int(num_layers)
    return None


def _is_int_sequence(value):
    return isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value)


def _restore_sequence_type(original_value, values):
    return tuple(values) if isinstance(original_value, tuple) else values


def _map_layer_index(layer_idx: int, original_num_layers: int, proxy_num_layers: int) -> int:
    if layer_idx < 0:
        return layer_idx
    if proxy_num_layers == 1 or original_num_layers <= 1:
        return 0
    return min(proxy_num_layers - 1, round(layer_idx * (proxy_num_layers - 1) / (original_num_layers - 1)))


def _normalize_index_sequence(attribute: str, value, original_num_layers: int, proxy_num_layers: int):
    if attribute in {"cross_attention_layers", "cross_attn_layers", "fullatt_block_indexes"}:
        mapped_values = [proxy_num_layers - 1] if value and proxy_num_layers > 0 else []
    else:
        mapped_values = []
        for layer_idx in value:
            mapped_idx = _map_layer_index(int(layer_idx), original_num_layers, proxy_num_layers)
            if mapped_idx < proxy_num_layers and mapped_idx not in mapped_values:
                mapped_values.append(mapped_idx)
        if value and not mapped_values and proxy_num_layers > 0:
            mapped_values = [proxy_num_layers - 1]
    return _restore_sequence_type(value, mapped_values)


def _normalize_layer_indexed_metadata(config, original_num_layers: int, proxy_num_layers: int) -> None:
    if config is None or original_num_layers < 1 or proxy_num_layers < 1:
        return

    for attribute, value in vars(config).items():
        attribute_lower = attribute.lower()
        is_known_sequence = attribute in _LAYER_INDEXED_SEQUENCE_ATTRIBUTES
        is_probable_layer_sequence = _is_int_sequence(value) and (
            "layer" in attribute_lower or "index" in attribute_lower or "indices" in attribute_lower
        )
        if _is_int_sequence(value) and (is_known_sequence or is_probable_layer_sequence):
            setattr(
                config,
                attribute,
                _normalize_index_sequence(attribute, value, original_num_layers, proxy_num_layers),
            )
        elif attribute in _LAYER_INDEXED_SCALAR_ATTRIBUTES and isinstance(value, int) and value >= 0:
            setattr(config, attribute, _map_layer_index(value, original_num_layers, proxy_num_layers))
        elif attribute in _LAYER_COUNT_SCALAR_ATTRIBUTES and isinstance(value, int):
            setattr(config, attribute, min(value, proxy_num_layers))


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
    try:
        original_num_layers = get_num_layers_from_config(language_config)
    except ValueError:
        original_num_layers = None
    configured_num_layers = num_hidden_layers if num_hidden_layers is not None else original_num_layers
    if configured_num_layers is None:
        raise ValueError("Proxy mode requires the language config to define `num_hidden_layers`.")
    if configured_num_layers < 1:
        raise ValueError("Proxy mode requires at least one language layer.")
    original_num_layers = original_num_layers or configured_num_layers
    if num_hidden_layers is not None:
        language_config.num_hidden_layers = num_hidden_layers
        _normalize_layer_indexed_metadata(language_config, original_num_layers, num_hidden_layers)
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
    _normalize_layer_indexed_metadata(language_config, configured_num_layers, proxy_num_layers)
    logger.info(
        "Proxy language model uses %d of %d layers with at least %d calls for each of %d layer types",
        proxy_num_layers,
        signature_horizon,
        minimum_calls_per_layer_type,
        len(required_signatures),
    )
    return proxy_num_layers


def apply_vision_proxy_layer_config(
    config, minimum_representative_layers: int = 2, num_hidden_layers: int | None = None
) -> int:
    """Reduce the vision stack while keeping representative layer-indexed metadata valid."""
    vision_config = _get_vision_config(config)
    original_num_layers = _get_vision_num_layers(vision_config)
    configured_num_layers = num_hidden_layers if num_hidden_layers is not None else original_num_layers
    if configured_num_layers is None:
        raise ValueError(
            f"Vision proxy mode requires `vision_config` to define one of {', '.join(_VISION_LAYER_COUNT_ATTRIBUTES)}."
        )
    configured_num_layers = int(configured_num_layers)
    original_num_layers = int(original_num_layers or configured_num_layers)
    if configured_num_layers < 1:
        raise ValueError("Vision proxy mode requires at least one vision layer.")

    if num_hidden_layers is not None:
        proxy_num_layers = configured_num_layers
        logger.info("Proxy vision model uses caller-provided layer count: %d", proxy_num_layers)
    else:
        proxy_num_layers = min(configured_num_layers, minimum_representative_layers)
        logger.info("Proxy vision model uses %d of %d layers", proxy_num_layers, configured_num_layers)

    for attribute in _VISION_LAYER_COUNT_ATTRIBUTES:
        if hasattr(vision_config, attribute):
            setattr(vision_config, attribute, proxy_num_layers)

    _normalize_layer_indexed_metadata(vision_config, original_num_layers, proxy_num_layers)
    _normalize_layer_indexed_metadata(config, original_num_layers, proxy_num_layers)
    return proxy_num_layers


def prepare_proxy_config(pretrained_model_name_or_path, kwargs: dict, proxy_component: str = PROXY_COMPONENT_LANGUAGE):
    """Load and mutate the model config before checkpoint weights are materialized."""
    proxy_component = normalize_proxy_component(kwargs.pop("proxy_component", proxy_component))
    explicit_num_hidden_layers = kwargs.pop("num_hidden_layers", None)
    config = kwargs.get("config")
    if config is None:
        config_kwargs = {key: kwargs[key] for key in _CONFIG_LOAD_KWARGS if key in kwargs}
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)

    if proxy_component == PROXY_COMPONENT_LANGUAGE:
        if explicit_num_hidden_layers is None:
            apply_proxy_layer_config(config)
        else:
            apply_proxy_layer_config(config, num_hidden_layers=explicit_num_hidden_layers)
    else:
        if explicit_num_hidden_layers is None:
            apply_vision_proxy_layer_config(config)
        else:
            apply_vision_proxy_layer_config(config, num_hidden_layers=explicit_num_hidden_layers)
    kwargs["config"] = config
    return config
