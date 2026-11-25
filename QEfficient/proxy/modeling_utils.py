# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Configuration helpers for representative proxy stacks."""

from collections import Counter
from copy import copy

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
_LAYER_INDEXED_ORDERED_SEQUENCE_ATTRIBUTES = {"intermediate_layers_indices", "out_indices"}
_LAYER_INDEXED_SCALAR_ATTRIBUTES = {"vision_feature_layer"}
_LAYER_COUNT_SCALAR_ATTRIBUTES = {"num_global_layers"}
_LAYER_TYPE_SEQUENCE_ATTRIBUTES = {"fullatt_block_indexes"}
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


def _require_vision_proxy_config(config) -> None:
    if getattr(config, "vision_config", None) is None:
        raise ValueError(
            "Vision proxy mode requires an image-text config with `vision_config`; "
            f"got {config.__class__.__name__}. Use proxy_component='language' for language-only models."
        )


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


def _normalize_index_sequence(value, original_num_layers: int, proxy_num_layers: int, deduplicate: bool = True):
    mapped_values = []
    for layer_idx in value:
        mapped_idx = _map_layer_index(int(layer_idx), original_num_layers, proxy_num_layers)
        if mapped_idx < proxy_num_layers and (not deduplicate or mapped_idx not in mapped_values):
            mapped_values.append(mapped_idx)
    if value and not mapped_values and proxy_num_layers > 0:
        mapped_values = [proxy_num_layers - 1]
    return _restore_sequence_type(value, mapped_values)


def _map_hidden_state_selector(selector_idx: int, original_num_layers: int, proxy_num_layers: int) -> int:
    original_num_hidden_states = original_num_layers + 1
    proxy_num_hidden_states = proxy_num_layers + 1
    if selector_idx < 0:
        positive_idx = max(0, original_num_hidden_states + selector_idx)
        mapped_idx = _map_layer_index(positive_idx, original_num_hidden_states, proxy_num_hidden_states)
        return mapped_idx - proxy_num_hidden_states
    return _map_layer_index(selector_idx, original_num_hidden_states, proxy_num_hidden_states)


def _normalize_hidden_state_selector(value, original_num_layers: int, proxy_num_layers: int):
    if isinstance(value, int):
        return _map_hidden_state_selector(value, original_num_layers, proxy_num_layers)
    if _is_int_sequence(value):
        return _restore_sequence_type(
            value,
            [
                _map_hidden_state_selector(int(selector_idx), original_num_layers, proxy_num_layers)
                for selector_idx in value
            ],
        )
    return value


def _normalize_layer_type_sequence(value, original_num_layers: int, proxy_num_layers: int):
    original_values = {int(layer_idx) for layer_idx in value if 0 <= int(layer_idx) < original_num_layers}
    if not original_values:
        return _restore_sequence_type(value, [])
    if len(original_values) >= original_num_layers:
        return _restore_sequence_type(value, list(range(proxy_num_layers)))
    if proxy_num_layers == 1:
        return _restore_sequence_type(value, [0])
    mapped_values = _normalize_index_sequence(value, original_num_layers, proxy_num_layers)
    if len(mapped_values) >= proxy_num_layers:
        return _restore_sequence_type(value, [proxy_num_layers - 1])
    return mapped_values


def _normalize_layer_indexed_metadata(
    config, original_num_layers: int, proxy_num_layers: int, skipped_attributes: set[str] | None = None
) -> None:
    if config is None or original_num_layers < 1 or proxy_num_layers < 1:
        return

    skipped_attributes = skipped_attributes or set()
    for attribute, value in vars(config).items():
        if attribute in skipped_attributes:
            continue
        attribute_lower = attribute.lower()
        is_known_sequence = attribute in _LAYER_INDEXED_SEQUENCE_ATTRIBUTES
        is_probable_layer_sequence = _is_int_sequence(value) and (
            "layer" in attribute_lower or "index" in attribute_lower or "indices" in attribute_lower
        )
        if attribute in _LAYER_INDEXED_SCALAR_ATTRIBUTES:
            setattr(config, attribute, _normalize_hidden_state_selector(value, original_num_layers, proxy_num_layers))
        elif _is_int_sequence(value) and attribute in _LAYER_TYPE_SEQUENCE_ATTRIBUTES:
            setattr(config, attribute, _normalize_layer_type_sequence(value, original_num_layers, proxy_num_layers))
        elif _is_int_sequence(value) and (is_known_sequence or is_probable_layer_sequence):
            setattr(
                config,
                attribute,
                _normalize_index_sequence(
                    value,
                    original_num_layers,
                    proxy_num_layers,
                    deduplicate=attribute not in _LAYER_INDEXED_ORDERED_SEQUENCE_ATTRIBUTES,
                ),
            )
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
    elif _is_int_sequence(fullatt_block_indexes := getattr(config, "fullatt_block_indexes", None)):
        fullatt_block_indexes = {int(fullatt_idx) for fullatt_idx in fullatt_block_indexes}
        attention_type = "full_attention" if layer_idx in fullatt_block_indexes else "window_attention"
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


def _rebuild_mlp_only_layers(config, pattern_horizon: int, proxy_num_layers: int) -> None:
    mlp_only_layers = getattr(config, "mlp_only_layers", None)
    if mlp_only_layers is None:
        return

    mlp_only_layer_set = set(mlp_only_layers)
    config.mlp_only_layers = [
        layer_idx for layer_idx in range(proxy_num_layers) if layer_idx % pattern_horizon in mlp_only_layer_set
    ]


def _get_reduced_vision_config(vision_config, original_num_layers: int, proxy_num_layers: int):
    reduced_config = copy(vision_config)
    for attribute in _VISION_LAYER_COUNT_ATTRIBUTES:
        if hasattr(reduced_config, attribute):
            setattr(reduced_config, attribute, proxy_num_layers)
    _normalize_layer_indexed_metadata(reduced_config, original_num_layers, proxy_num_layers)
    return reduced_config


def _get_representative_vision_layer_count(
    vision_config, original_num_layers: int, minimum_representative_layers: int
) -> int:
    if original_num_layers <= minimum_representative_layers:
        return original_num_layers

    required_signatures = {_get_layer_signature(vision_config, layer_idx) for layer_idx in range(original_num_layers)}
    for proxy_num_layers in range(minimum_representative_layers, original_num_layers + 1):
        proxy_config = _get_reduced_vision_config(vision_config, original_num_layers, proxy_num_layers)
        signature_counts = Counter(
            _get_layer_signature(proxy_config, layer_idx) for layer_idx in range(proxy_num_layers)
        )
        if all(signature_counts[signature] >= minimum_representative_layers for signature in required_signatures):
            return proxy_num_layers
    return original_num_layers


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
        _rebuild_mlp_only_layers(language_config, original_num_layers, num_hidden_layers)
        language_config.num_hidden_layers = num_hidden_layers
        _normalize_layer_indexed_metadata(language_config, original_num_layers, num_hidden_layers, {"mlp_only_layers"})
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

    _rebuild_mlp_only_layers(language_config, signature_horizon, proxy_num_layers)

    language_config.num_hidden_layers = proxy_num_layers
    _normalize_layer_indexed_metadata(language_config, configured_num_layers, proxy_num_layers, {"mlp_only_layers"})
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
        proxy_num_layers = _get_representative_vision_layer_count(
            vision_config, configured_num_layers, minimum_representative_layers
        )
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
        _require_vision_proxy_config(config)
        if explicit_num_hidden_layers is None:
            apply_vision_proxy_layer_config(config)
        else:
            apply_vision_proxy_layer_config(config, num_hidden_layers=explicit_num_hidden_layers)
    kwargs["config"] = config
    return config
