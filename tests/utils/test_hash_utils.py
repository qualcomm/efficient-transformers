# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import random

import pytest
import torch

from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
from QEfficient.utils.constants import HASH_HEXDIGEST_STR_LEN
from QEfficient.utils.hash_utils import hash_dict_params, json_serializable, to_hashable


def get_random_string(length: int) -> str:
    return "".join([chr(random.randint(0x20, 0x7E)) for _ in range(length)])


def test_to_hashable_dict():
    dct = {get_random_string(i): i for i in range(5)}
    dct = dict(sorted(dct.items()))
    hash1 = to_hashable(dct)

    dct = dict(reversed(dct.items()))
    hash2 = to_hashable(dct)

    assert hash1 == hash2


def test_to_hashable_set():
    assert to_hashable(set(range(4))) == to_hashable(set(range(4 - 1, -1, -1)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_to_hashable_float_nan(value):
    with pytest.raises(ValueError):
        to_hashable(value)


def test_json_serializable():
    # Test with a set
    assert json_serializable({1, 2, 3}) == ["1", "2", "3"]
    # Test with an unsupported type
    with pytest.raises(TypeError):
        json_serializable({1, 2, 3, {4, 5}})


def test_json_serializable_supports_dataclass_and_torch_dtype():
    blocking_config = AttentionBlockingConfig()

    assert isinstance(json_serializable(blocking_config), dict)
    assert to_hashable({"blocking_kwargs": blocking_config})
    assert json_serializable(torch.bfloat16) == "torch.bfloat16"


def test_to_hashable():
    # Test with a simple dictionary
    obj = {"key": "value"}
    expected = json.dumps(
        obj,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
        default=json_serializable,
        sort_keys=True,
    ).encode()
    assert to_hashable(obj) == expected

    # Test with a dictionary containing a set
    obj_with_set = {"key": {1, 2, 3}}
    expected_with_set = json.dumps(
        obj_with_set,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
        default=json_serializable,
        sort_keys=True,
    ).encode()
    assert to_hashable(obj_with_set) == expected_with_set


def test_hash_dict_params():
    # Test with a simple dictionary
    dict_items = {"key": "value"}
    hash_result = hash_dict_params(dict_items)
    assert len(hash_result) == HASH_HEXDIGEST_STR_LEN
    assert isinstance(hash_result, str)

    # Test with a dictionary containing a set
    dict_items_with_set = {"key": {1, 2, 3}}
    hash_result_with_set = hash_dict_params(dict_items_with_set)
    assert len(hash_result_with_set) == HASH_HEXDIGEST_STR_LEN
    assert isinstance(hash_result_with_set, str)

    # Test with a custom hash string size
    custom_hash_size = 10
    hash_result_custom_size = hash_dict_params(dict_items, custom_hash_size)
    assert len(hash_result_custom_size) == custom_hash_size
    assert isinstance(hash_result_custom_size, str)
