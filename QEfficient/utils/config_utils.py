# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Iterable, Optional

from QEfficient.utils.constants import ATTENTION_HEAD_CONFIG_KEYS, HIDDEN_SIZE_CONFIG_KEYS, KV_HEAD_CONFIG_KEYS


def get_first_config_value(config, names: Iterable[str], default=None, cast_int: bool = False):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value) if cast_int else value
    return default


def resolve_attention_heads(config) -> Optional[int]:
    return get_first_config_value(config, ATTENTION_HEAD_CONFIG_KEYS, cast_int=True)


def resolve_kv_heads(config) -> Optional[int]:
    value = get_first_config_value(config, KV_HEAD_CONFIG_KEYS, cast_int=True)
    if value is None:
        value = resolve_attention_heads(config)
    return value


def resolve_hidden_size(config) -> Optional[int]:
    return get_first_config_value(config, HIDDEN_SIZE_CONFIG_KEYS, cast_int=True)


def set_kv_head_aliases(config, value: int):
    setattr(config, "num_key_value_heads", value)
    for key in KV_HEAD_CONFIG_KEYS:
        if hasattr(config, key):
            setattr(config, key, value)


def calculate_num_replicate_kv_heads(num_devices: int, text_model_config) -> int:
    """
    Choose a KV-repeat value from model config and device count.

    Primary criteria:
    1. MQA models do not request replication; the compiler handles them internally.
    2. GQA models reach one replicated KV head per device.
    3. num_attention_heads is divisible by the replicated KV-head count.

    Returns 1 if no valid repeat exists (replication not applicable or not achievable).
    """
    num_attention_heads = resolve_attention_heads(text_model_config)
    num_kv_heads = resolve_kv_heads(text_model_config)

    if num_attention_heads is None or num_kv_heads is None or num_attention_heads < 1 or num_kv_heads < 1:
        return None

    # MQA already has a single shared KV head.  Its expansion is a compiler
    # concern, so keep the PyTorch model and its checkpoint tensors untouched.
    if num_kv_heads == 1:
        return 1

    num_devices = max(1, int(num_devices))
    max_repeat = max(1, num_attention_heads // num_kv_heads)
    if num_devices <= num_kv_heads or num_devices % num_kv_heads != 0:
        # There is no GQA replication to perform when the existing KV layout
        # already covers the device group, or when the device count cannot be
        # reached by an integral repeat of the original KV-head count.
        return None
    for repeat in range(2, max_repeat + 1):
        new_kv_heads = num_kv_heads * repeat
        # The device-count divisibility check alone also accepts a smaller
        # KV-head count (for example, 2 heads across 4 devices).  For GQA we
        # need the repeated layout to provide one effective KV head per device:
        # together, ``num_devices % new_kv_heads == 0`` and
        # ``num_devices <= new_kv_heads`` require ``new_kv_heads`` to equal
        # ``num_devices``.  The attention-head check preserves a valid GQA
        # grouping after the projection weights are expanded.
        if (
            (num_devices % new_kv_heads == 0)
            and (num_devices <= new_kv_heads)
            and (num_attention_heads % new_kv_heads == 0)
        ):
            return repeat

    return 1
