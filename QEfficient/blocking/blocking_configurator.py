# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Utility helpers to suggest attention/FFN blocking configs for diffusers transformers and transformers

This module adapts the standalone configurator script into a clean, importable API
that can be fed model config + pipeline compile config to derive blocking settings.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, List, Optional

from QEfficient.blocking.attention_blocking import BLOCKING_MODE_REQUIRED_PARAMS, AttentionBlockingConfig, BlockingMode
from QEfficient.utils import get_attr_or_key, require_value
from QEfficient.utils.constants import (
    DEFAULT_AIC_NUM_CORES,
    DEFAULT_NUM_HEADS,
    FP16_BYTES,
    KV_LORA_RANK,
    ROPE_DIM,
    VTCM_SIZE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _infer_head_dim(model_config: Any, num_heads: int) -> int:
    head_dim = get_attr_or_key(model_config, ("attention_head_dim", "head_dim", "head_dim_per_head"))
    if head_dim is not None:
        return int(head_dim)
    hidden_size = get_attr_or_key(model_config, ("hidden_size", "d_model", "model_dim", "attention_dim"))
    if hidden_size is None:
        raise ValueError("Missing head_dim or hidden_size to compute attention blocking configuration.")
    return int(hidden_size) // int(num_heads)


def _infer_data_bytes(compile_config: Dict[str, Any]) -> int:
    explicit = compile_config.get("data_bytes")
    if explicit is not None:
        return int(explicit)
    if compile_config.get("convert_to_fp16", False):
        return 2
    return 4


def _normalize_attention_mode(raw_mode: str) -> str:
    mode = raw_mode.lower()
    if "h" in mode and "q" in mode and "kv" in mode:
        return "hqkv"
    if "h" in mode and "q" in mode:
        return "hq"
    if "h" in mode and "kv" in mode:
        return "hkv"
    if "h" in mode:
        return "h"
    if "q" in mode and "kv" in mode:
        return "qkv"
    if "kv" in mode:
        return "kv"
    if "q" in mode:
        return "q"
    return ""


def _resolve_effective_blocking_mode(attention_cfg: Dict[str, Any], requested_mode: str) -> str:
    mode = _normalize_attention_mode(requested_mode)
    if mode == "":
        return ""
    num_q_blocks = attention_cfg.get("num_q_blocks") or 1
    num_kv_blocks = attention_cfg.get("num_kv_blocks") or 1
    head_block_size = (attention_cfg.get("head_block_size") or 1) if attention_cfg.get("head_blocking_enabled") else 1

    if head_block_size > 1 and num_q_blocks > 1 and num_kv_blocks > 1:
        return "hqkv"
    if head_block_size > 1 and num_q_blocks > 1:
        return "hq"
    if head_block_size > 1 and num_kv_blocks > 1:
        return "hkv"
    if head_block_size > 1:
        return "h"
    if num_q_blocks > 1 and num_kv_blocks > 1:
        return "qkv"
    if num_q_blocks > 1:
        return "q"
    if num_kv_blocks > 1:
        return "kv"
    return ""


def _get_valid_num_blocks(config: Dict, requested_key: str) -> int:
    if config.get(requested_key) < 1:
        raise ValueError(f"Invalid value {requested_key} passed in qaic_config: {config.get(requested_key)}")
    return config.get(requested_key)


def block_candidates_generator(max_length: int) -> List[int]:
    block_list = []
    i = 1
    step = 1
    while i <= max_length:
        block_list.append(i)
        if i % (4 * step) == 0:
            step *= 2
        i += step
    return block_list


def matmul1_bytes(q_len: int, kv_block_size: int, num_heads: int = DEFAULT_NUM_HEADS) -> int:
    """Bytes for [1,num_heads,q,kv] x [1,1,kv,512] -> [1,num_heads,q,512] in fp16."""
    elems_a = num_heads * q_len * kv_block_size
    elems_b = kv_block_size * KV_LORA_RANK
    elems_out = num_heads * q_len * KV_LORA_RANK
    return FP16_BYTES * (elems_a + elems_b + elems_out)


def matmul2_bytes(q_len: int, kv_block_size: int, num_heads: int = DEFAULT_NUM_HEADS) -> int:
    """Bytes for [1,num_heads,q,576] x [1,1,576,kv] -> [1,num_heads,q,kv] in fp16."""
    elems_a = num_heads * q_len * (KV_LORA_RANK + ROPE_DIM)
    elems_b = 576 * kv_block_size
    elems_out = num_heads * q_len * kv_block_size
    return FP16_BYTES * (elems_a + elems_b + elems_out)


def max_kv_block_size(
    q_len: int,
    budget_bytes: int = VTCM_SIZE_THRESHOLD,
    num_heads: int = DEFAULT_NUM_HEADS,
) -> int:
    """Return the largest integer kv_block_size that satisfies both matmul budgets.

    Returns 0 if no positive kv_block_size can satisfy the constraints.
    """
    if q_len < 0:
        raise ValueError("q_len must be non-negative")
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    # Enforce strict inequality in bytes:
    # FP16_BYTES * elems < budget_bytes  =>  elems <= floor((budget_bytes - 1)/FP16_BYTES)
    max_elems = (budget_bytes - 1) // FP16_BYTES

    # Matmul1 elements:
    #   A_elems = num_heads*q_len*kv
    #   B_elems = kv*512
    #   C_elems = num_heads*q_len*512
    # Enforce A_elems + B_elems + C_elems <= max_elems
    c1_elems = num_heads * q_len * KV_LORA_RANK
    rem1 = max_elems - c1_elems
    den1 = num_heads * q_len + KV_LORA_RANK  # kv coefficient from A_elems + B_elems
    k1 = rem1 // den1 if rem1 >= 0 else -1

    # Matmul2 elements:
    #   A_elems = num_heads*q_len*576
    #   B_elems = 576*kv
    #   C_elems = num_heads*q_len*kv
    # Enforce A_elems + B_elems + C_elems <= max_elems
    a2_elems = num_heads * q_len * 576
    rem2 = max_elems - a2_elems
    den2 = num_heads * q_len + 576  # kv coefficient from B_elems + C_elems
    k2 = rem2 // den2 if rem2 >= 0 else -1

    kv = min(k1, k2)
    return max(0, kv)


def get_num_kv_blocks_for_mla(q_len, num_heads, ctx_len):
    """Compute the maximum kv_block_size under an fp16 memory budget.

    Constraints (bytes) per matmul:
    1) [1, num_heads, q_len, 576] x [1, 1, 576, kv] -> [1, num_heads, q_len, kv]
    2) [1, num_heads, q_len, kv] x [1, 1, kv, 512] -> [1, num_heads, q_len, 512]

    For each matmul, sum(input_a + input_b + output) must be < budget.
    The returned kv_block_size satisfies both constraints.
    """
    budget_bytes = VTCM_SIZE_THRESHOLD
    kv = max_kv_block_size(q_len, budget_bytes, num_heads)
    b1 = matmul1_bytes(q_len, kv, num_heads)
    b2 = matmul2_bytes(q_len, kv, num_heads)

    assert b1 < budget_bytes, "matmul1 is not under the budget"
    assert b2 < budget_bytes, "matmul2 is not under the budget"

    kv_block_size = ctx_len
    kv_block_size_list = block_candidates_generator(ctx_len)
    for i in range(len(kv_block_size_list) - 1):
        if kv_block_size_list[i] < kv < kv_block_size_list[i + 1]:
            kv_block_size = kv_block_size_list[i]
    return ctx_len // kv_block_size


def attention_configurator(
    bs: int,
    seq_len: int,
    ctx_len: int,
    num_heads: int,
    head_dim: int,
    num_socs: int,
    num_nsps: int,
    data_bytes: int,
    blocking_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Suggest attention blocking configuration based on model and device constraints.
    """
    mode = (blocking_mode or "hqkv").lower()

    num_kv_blocks_list = block_candidates_generator(ctx_len) if "kv" in mode else [1]
    num_q_blocks_list = block_candidates_generator(ctx_len) if "q" in mode else [1]

    head_block_size = num_socs if "h" in mode else num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)
    num_heads_per_iter = math.ceil(head_block_size / num_socs)

    best_config = {
        "head_block_size": head_block_size,
        "num_head_blocks": num_head_blocks,
        "head_blocking_enabled": num_head_blocks > 1,
        "num_q_blocks": None,
        "num_kv_blocks": None,
        "q_kv_ratio": None,
        "vtcm_footprint": None,
    }

    def update_best_config(num_q_blocks: int, num_kv_blocks: int, q_kv_ratio: float, footprint: float) -> None:
        best_config["num_q_blocks"] = num_q_blocks
        best_config["num_kv_blocks"] = num_kv_blocks
        best_config["q_kv_ratio"] = q_kv_ratio
        best_config["vtcm_footprint"] = footprint

    kv_metrics = []
    for num_kv_blocks in num_kv_blocks_list:
        kv_cl_per_nsp = math.ceil(ctx_len / num_kv_blocks)
        kv_size_per_nsp = num_heads_per_iter * bs * kv_cl_per_nsp * head_dim * data_bytes
        kv_metrics.append((num_kv_blocks, kv_cl_per_nsp, kv_size_per_nsp))

    for num_q_blocks in num_q_blocks_list:
        q_sl_per_nsp = math.ceil(seq_len / num_nsps / num_q_blocks)
        q_size_per_nsp = num_heads_per_iter * bs * q_sl_per_nsp * head_dim * data_bytes

        for num_kv_blocks, kv_cl_per_nsp, kv_size_per_nsp in kv_metrics:
            qk_size_per_nsp = num_heads_per_iter * bs * q_sl_per_nsp * kv_cl_per_nsp * data_bytes
            qkv_size_per_nsp = q_size_per_nsp
            # For KV Blocking for loop, q input and qkv output should be persistent in VTCM
            vtcm_footprint = q_size_per_nsp + kv_size_per_nsp + qk_size_per_nsp + qkv_size_per_nsp
            q_kv_ratio = max(q_size_per_nsp / kv_size_per_nsp, kv_size_per_nsp / q_size_per_nsp)
            num_total_blocks = num_q_blocks * num_kv_blocks

            if vtcm_footprint < VTCM_SIZE_THRESHOLD:
                if best_config["num_q_blocks"] is None:
                    update_best_config(num_q_blocks, num_kv_blocks, q_kv_ratio, vtcm_footprint)
                elif best_config["num_q_blocks"] * best_config["num_kv_blocks"] > num_total_blocks:
                    update_best_config(num_q_blocks, num_kv_blocks, q_kv_ratio, vtcm_footprint)
                elif (
                    best_config["num_q_blocks"] * best_config["num_kv_blocks"] == num_total_blocks
                    and best_config["q_kv_ratio"] >= q_kv_ratio
                ):
                    update_best_config(num_q_blocks, num_kv_blocks, q_kv_ratio, vtcm_footprint)
                break

    return best_config


def build_transformer_blocking_config(
    model_config: Any,
    pipeline_config: Optional[Any] = None,
    module_name: str = "transformer",
    blocking_mode: Optional[str] = None,
    ctx_len: Optional[int] = None,
    seq_len: Optional[int] = None,
    bs: Optional[int] = 1,
    compile_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build blocking configuration based on model config + pipeline compile config.
    """
    if ctx_len is None:
        ctx_len = seq_len

    if seq_len is None and ctx_len is None:
        return AttentionBlockingConfig(mode="")

    # we only block on text configs in the case of VLMs
    if hasattr(model_config, "text_config"):
        model_config = model_config.text_config
    num_heads = require_value(
        get_attr_or_key(model_config, ("num_attention_heads", "num_heads", "attention_heads", "n_heads")),
        "num attention heads",
    )
    head_dim = _infer_head_dim(model_config, int(num_heads))

    num_socs = int(compile_config.get("mdp_ts_num_devices", 1))
    num_nsps = int(compile_config.get("aic_num_cores", 1))
    data_bytes = _infer_data_bytes(compile_config)

    attention_cfg = attention_configurator(
        int(bs),
        int(seq_len),
        int(ctx_len),
        int(num_heads),
        int(head_dim),
        int(num_socs),
        int(num_nsps),
        int(data_bytes),
        blocking_mode=blocking_mode,
    )

    if "DeepseekV3ForCausalLM" in (getattr(model_config, "architectures", None) or []):
        if "kv" in blocking_mode:
            attention_cfg["num_kv_blocks"] = get_num_kv_blocks_for_mla(seq_len, num_heads, ctx_len)

    effective_mode = _resolve_effective_blocking_mode(attention_cfg, blocking_mode or "hqkv")
    paged_attention = "paged" in str(blocking_mode or "").lower()

    # PagedAttention reads the cache block-wise, so it is only valid alongside KV blocking.
    if paged_attention and "kv" not in effective_mode:
        raise ValueError(f"Paged attention is only valid alongside KV blocking, got mode {effective_mode}")

    requested = blocking_mode or "hqkv"
    if effective_mode != requested:
        warnings.warn(
            f"Requested blocking_mode '{requested}'/'AUTO' reduced to '{effective_mode}' based on calculated attention config: {attention_cfg}",
        )

    return AttentionBlockingConfig(
        mode=BlockingMode(effective_mode),
        num_kv_blocks=attention_cfg["num_kv_blocks"],
        num_q_blocks=attention_cfg["num_q_blocks"],
        head_block_size=attention_cfg["head_block_size"],
        paged_attention=paged_attention,
    )


def build_transformer_blocking_config_for_transform(
    model_config: Any,
    ctx_len: Optional[int] = None,
    seq_len: Optional[int] = None,
    bs: Optional[int] = 1,
    num_devices: Optional[int] = 1,
    qaic_config: Optional[dict] = None,
    **compile_options,
) -> Dict[str, Any]:
    if qaic_config and "enable_blocking" in qaic_config:
        logger.warning(
            "'enable_blocking' in qaic_config is deprecated and ignored. "
            "Use 'blocking_mode' to enable attention blocking; omit 'blocking_mode' to disable it."
        )

    requested_blocking_mode = None if not qaic_config else qaic_config.get("blocking_mode")
    if requested_blocking_mode is None:
        return None

    # "paged" is an orthogonal flag on top of a base blocking mode (e.g. "kv_paged"), not its own
    # BlockingMode member, so it must be stripped before resolving the base mode via the enum.
    base_blocking_mode = str(requested_blocking_mode).lower().replace("_paged", "").replace("paged", "")
    blocking_mode = BlockingMode.resolve(base_blocking_mode)

    required_keys = BLOCKING_MODE_REQUIRED_PARAMS.get(blocking_mode, [])
    provided_keys = [key for key in required_keys if qaic_config.get(key)]
    missing_keys = [key for key in required_keys if not qaic_config.get(key)]

    if provided_keys and missing_keys:
        raise ValueError(
            f"blocking_mode '{requested_blocking_mode}' requires {required_keys} but only {provided_keys} were provided; "
            f"either supply all of {missing_keys} or none to use automatic calculation"
        )

    # if we haven't been passed manual number of blocks, do automatic calculation
    if missing_keys:
        # Pass the raw requested mode (not the paged-stripped enum) through so the "paged"
        # substring is still visible to build_transformer_blocking_config's own paged detection.
        blocking_config = build_transformer_blocking_config(
            model_config,
            blocking_mode=str(requested_blocking_mode),
            ctx_len=ctx_len,
            seq_len=seq_len,
            bs=bs,
            compile_config={"mdp_ts_num_devices": num_devices, **compile_options},
        )
    # if we have been passed all number of blocks via qaic_config, set each in blocking config
    else:
        blocking_config = AttentionBlockingConfig()
        for key in required_keys:
            setattr(blocking_config, key, _get_valid_num_blocks(qaic_config, key))
        blocking_config.mode = BlockingMode(blocking_mode)
        # PagedAttention reads the cache block-wise, so it is only valid alongside KV blocking.
        blocking_config.paged_attention = (
            "paged" in str(requested_blocking_mode or "").lower() and "num_kv_blocks" in required_keys
        )

    # For headpar modes: set headpar_split, defaulting to aic_num_cores when omitted
    _HEADPAR_MODES = {BlockingMode.KV_HEADPAR, BlockingMode.PREFILL_KV, BlockingMode.PREFILL_QKV}
    if blocking_mode in _HEADPAR_MODES:
        headpar_split = qaic_config.get("headpar_split")
        if headpar_split is None:
            headpar_split = compile_options.get("aic_num_cores", DEFAULT_AIC_NUM_CORES)
        blocking_config.headpar_split = headpar_split

    if blocking_mode == BlockingMode.KV_BATCH_FOLD:
        blocking_config.batch_fold = True

    # optional blocking parameters to set if given in qaic_config
    for param in ("skip_kv", "n_rep_chunk", "ctx_len", "kv_block_unroll", "num_cores_per_device"):
        if qaic_config.get(param) is not None:
            setattr(blocking_config, param, qaic_config.get(param))

    # KV_BATCH_FOLD also uses this value to block Qwen3.5 linear-attention
    # recurrence across the head dimension. It is a required field for BHQKV,
    # but an optional tuning knob for the folded decode path.
    if qaic_config.get("gdn_num_head_blocks") is not None:
        blocking_config.gdn_num_head_blocks = _get_valid_num_blocks(qaic_config, "gdn_num_head_blocks")

    if qaic_config.get("ctx_len") is None:
        blocking_config.ctx_len = ctx_len

    if qaic_config.get("num_cores_per_device") is None:
        blocking_config.num_cores_per_device = compile_options.get("aic_num_cores")

    return blocking_config


def build_gated_delta_config_for_transform(
    seq_len: Optional[int] = None, qaic_config: Optional[dict] = None
) -> Optional[Dict[str, Any]]:
    del seq_len
    if not qaic_config:
        return None

    gated_delta_config = {}
    chunk_size = qaic_config.get("gdn_chunk_size")

    if chunk_size is not None:
        chunk_size = int(chunk_size)
        if chunk_size > 0:
            gated_delta_config["chunk_size"] = chunk_size

    if qaic_config.get("gdn_full_state_update") is True:
        gated_delta_config["full_state_update"] = True

    return gated_delta_config or None
