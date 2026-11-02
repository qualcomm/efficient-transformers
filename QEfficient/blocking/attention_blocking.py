# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch
from transformers.cache_utils import Cache

from QEfficient.blocking.blocked_attention_forwards import (
    blocked_bhqkv_attention_forward,
    blocked_h_attention_forward,
    blocked_h_mla_attention_forward,
    blocked_hqkv_attention_forward,
    blocked_kv_attention_forward,
    blocked_kv_attention_forward_decode_headpar_batch,
    blocked_kv_attention_forward_headpar_offline,
    blocked_kv_attention_forward_prefill_headpar_offline,
    blocked_kv_mla_attention_forward,
    blocked_q_attention_forward,
    blocked_q_attention_forward_prefill,
    blocked_qkv_attention_forward,
    blocked_qkv_attention_forward_prefill_headpar_offline,
    blocked_qkv_attention_forward_prefill_online,
)


class BlockingMode(str, Enum):
    NONE = ""  # No blocking
    AUTO = "auto"  # We choose the best blocking mode based on the input configuration
    # decode
    KV = "kv"
    KV_HEADPAR = "kv_headpar"
    KV_BATCH_FOLD = "kv_batch_fold"
    Q = "q"
    H = "h"
    QKV = "qkv"
    HQ = "hq"
    HKV = "hkv"
    HQKV = "hqkv"
    BHQKV = "bhqkv"
    # MLA
    KV_MLA = "kv_mla"
    H_MLA = "h_mla"
    # prefill
    PREFILL_Q = "prefill_q"
    PREFILL_KV = "prefill_kv"
    PREFILL_QKV = "prefill_qkv"
    PREFILL_ONLINE = "prefill_online"

    @classmethod
    def resolve(cls, mode: Optional[str | "BlockingMode"]) -> "BlockingMode":
        if mode is None:
            return cls.NONE
        resolved_mode = cls(mode)
        if resolved_mode == cls.AUTO:
            return cls.HQKV
        return resolved_mode

    @property
    def is_prefill(self) -> bool:
        return self.value.startswith("prefill_")

    @classmethod
    def get_final_mode(
        cls,
        blocking_config: "AttentionBlockingConfig",
        *,
        prefill_only: bool = False,
        is_mla: bool = False,
        mla_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "BlockingMode":
        requested_mode = cls.resolve(blocking_config.mode)
        if is_mla:
            _mla_map = {cls.KV: cls.KV_MLA, cls.H: cls.H_MLA}
            mode = _mla_map[requested_mode]
        else:
            mode = requested_mode
        cls._validate_config(mode, blocking_config, prefill_only=prefill_only, mla_kwargs=mla_kwargs)
        return mode

    @classmethod
    def _validate_config(
        cls,
        mode: "BlockingMode",
        blocking_config: "AttentionBlockingConfig",
        prefill_only: bool = False,
        mla_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if mode.is_prefill and not prefill_only:
            raise ValueError(
                f"BlockingMode.{mode.name} is a prefill-only mode; set prefill_only=True when calling the blocked attention interface"
            )
        missing = [f for f in BLOCKING_MODE_REQUIRED_PARAMS.get(mode, []) if getattr(blocking_config, f, None) is None]
        if missing:
            raise ValueError(f"BlockingMode.{mode.name} requires {missing} to be set in AttentionBlockingConfig")

        _REQUIRED_MLA_KWARGS: Dict["BlockingMode", list] = {
            cls.KV_MLA: ["per_head_k_up_normal", "per_head_v_up", "mla_absorption"],
            cls.H_MLA: [
                "q_a_proj_out",
                "fusedqk",
                "q_nope",
                "q_pe",
                "kva",
                "k_pe",
                "per_head_q_up",
                "per_head_k_up",
                "per_head_v_up",
                "per_head_k_up_normal",
                "mla_absorption",
            ],
        }
        if mode in _REQUIRED_MLA_KWARGS:
            mla = mla_kwargs or {}
            missing_mla = [k for k in _REQUIRED_MLA_KWARGS[mode] if mla.get(k) is None]
            if missing_mla:
                raise ValueError(f"BlockingMode.{mode.name} requires {missing_mla} to be set in mla_kwargs")


@dataclass
class AttentionBlockingConfig:
    mode: BlockingMode = BlockingMode.NONE
    num_kv_blocks: Optional[int] = None
    num_q_blocks: Optional[int] = None
    head_block_size: Optional[int] = None
    skip_kv: Optional[bool] = True
    num_batch_blocks: Optional[int] = None
    gdn_num_head_blocks: Optional[int] = None
    headpar_split: Optional[int] = None
    batch_fold: Optional[bool] = False
    n_rep_chunk: Optional[int] = None
    ctx_len: Optional[int] = None
    kv_block_unroll: Optional[int] = 1
    num_cores_per_device: Optional[int] = None
    paged_attention: Optional[bool] = False


def get_gdn_num_head_blocks(blocking_config: Optional[AttentionBlockingConfig], batch_fold: bool) -> int:
    """Resolve the GDN head-block count for folded decode."""
    if not batch_fold or blocking_config is None:
        return 1
    return max(1, int(getattr(blocking_config, "gdn_num_head_blocks", 1) or 1))


def blocked_gdn_decode_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    gdn_num_head_blocks: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the one-token GDN update with optional head blocking.

    Folded decode treats the token dimension as static one and folds batch and
    head dimensions for each block, keeping recurrent-state contractions local.
    """
    dtype = query.dtype
    batch_size, sequence_length, num_heads, key_head_dim = query.shape
    value_head_dim = value.shape[-1]

    if sequence_length != 1:
        raise ValueError("blocked_gdn_decode_forward requires sequence_length == 1")

    q = query.float()
    k = key.float()
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    v = value.float()
    b = beta.float()
    decay = g.float().exp()
    state = recurrent_state.float()

    q = q * (1.0 / (key_head_dim**0.5))
    num_blocks = max(1, min(num_heads, int(gdn_num_head_blocks)))
    heads_per_block = -(-num_heads // num_blocks)

    output_blocks = []
    state_blocks = []
    for block_idx in range(num_blocks):
        head_start = block_idx * heads_per_block
        if head_start >= num_heads:
            break
        head_end = min(head_start + heads_per_block, num_heads)
        block_heads = head_end - head_start
        folded_batch_heads = batch_size * block_heads

        q_block = q[:, :, head_start:head_end].reshape(1, folded_batch_heads, sequence_length, key_head_dim)
        k_block = k[:, :, head_start:head_end].reshape(1, folded_batch_heads, sequence_length, key_head_dim)
        v_block = v[:, :, head_start:head_end].reshape(1, folded_batch_heads, sequence_length, value_head_dim)
        beta_block = b[:, :, head_start:head_end].reshape(1, folded_batch_heads, sequence_length, 1)
        decay_block = decay[:, :, head_start:head_end].reshape(1, folded_batch_heads, 1, 1)
        state_block = state[:, head_start:head_end].reshape(1, folded_batch_heads, key_head_dim, value_head_dim)

        decayed_state = state_block * decay_block
        kv_memory = torch.matmul(k_block, decayed_state)
        delta = (v_block - kv_memory) * beta_block
        updated_state = decayed_state + k_block[:, :, 0].unsqueeze(-1) * delta[:, :, 0].unsqueeze(-2)
        block_output = torch.matmul(q_block, updated_state)

        output_blocks.append(block_output.reshape(batch_size, block_heads, sequence_length, value_head_dim))
        state_blocks.append(updated_state.reshape(batch_size, block_heads, key_head_dim, value_head_dim))

    output = torch.cat(output_blocks, dim=1).reshape(batch_size, sequence_length, num_heads, value_head_dim)
    updated_state = torch.cat(state_blocks, dim=1)
    return output.to(dtype), updated_state.to(recurrent_state.dtype)


def recurrent_gdn_decode_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the original unblocked GDN decode update."""
    dtype = query.dtype
    _, _, _, key_head_dim = query.shape

    query = query.float()
    key = key.float()
    query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    value = value.float()
    state = recurrent_state.float()

    query = (query * (1.0 / (key_head_dim**0.5))).transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    beta = beta.transpose(1, 2).float().unsqueeze(-1)
    decay = g.transpose(1, 2).float().exp().unsqueeze(-1).unsqueeze(-1)

    state_decayed = state * decay[:, :, 0]
    kv_memory = (state_decayed * key[:, :, 0].unsqueeze(-1)).sum(dim=-2)
    delta = (value[:, :, 0] - kv_memory) * beta[:, :, 0]
    state_new = state_decayed + key[:, :, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    output = (state_new * query[:, :, 0].unsqueeze(-1)).sum(dim=-2)
    output = output.unsqueeze(2).transpose(1, 2).to(dtype)
    return output, state_new.to(recurrent_state.dtype)


# Required AttentionBlockingConfig fields per blocking mode.
BLOCKING_MODE_REQUIRED_PARAMS: Dict[BlockingMode, list] = {
    # decode
    BlockingMode.KV: ["num_kv_blocks"],
    BlockingMode.KV_BATCH_FOLD: ["num_kv_blocks"],
    BlockingMode.KV_HEADPAR: ["num_kv_blocks"],
    BlockingMode.Q: ["num_q_blocks"],
    BlockingMode.H: ["head_block_size"],
    BlockingMode.QKV: ["num_kv_blocks", "num_q_blocks"],
    BlockingMode.HQ: ["head_block_size", "num_q_blocks"],
    BlockingMode.HKV: ["head_block_size", "num_kv_blocks"],
    BlockingMode.HQKV: ["head_block_size", "num_kv_blocks", "num_q_blocks"],
    BlockingMode.BHQKV: ["head_block_size", "num_kv_blocks", "num_q_blocks", "num_batch_blocks"],
    # MLA
    BlockingMode.KV_MLA: ["num_kv_blocks"],
    BlockingMode.H_MLA: ["head_block_size"],
    # prefill
    BlockingMode.PREFILL_Q: ["num_q_blocks"],
    BlockingMode.PREFILL_KV: ["num_kv_blocks"],
    BlockingMode.PREFILL_QKV: ["num_kv_blocks", "num_q_blocks"],
    BlockingMode.PREFILL_ONLINE: ["num_kv_blocks", "num_q_blocks"],
}


def supports_blocked_kv(past_key_value: Optional[Cache]) -> bool:
    return past_key_value is not None and hasattr(past_key_value, "read_only_blocked_kv")


def supports_paged_attention_blocked_kv(past_key_value: Optional[Cache]) -> bool:
    return past_key_value is not None and hasattr(past_key_value, "read_only_paged_attention")


_STRATEGIES: Dict[BlockingMode, Callable] = {
    # decode
    BlockingMode.KV: blocked_kv_attention_forward,
    BlockingMode.KV_HEADPAR: blocked_kv_attention_forward_headpar_offline,
    BlockingMode.KV_BATCH_FOLD: blocked_kv_attention_forward_decode_headpar_batch,
    BlockingMode.Q: blocked_q_attention_forward,
    BlockingMode.H: blocked_h_attention_forward,
    BlockingMode.QKV: blocked_qkv_attention_forward,
    BlockingMode.HQ: blocked_hqkv_attention_forward,
    BlockingMode.HKV: blocked_hqkv_attention_forward,
    BlockingMode.HQKV: blocked_hqkv_attention_forward,
    BlockingMode.BHQKV: blocked_bhqkv_attention_forward,
    # MLA
    BlockingMode.KV_MLA: blocked_kv_mla_attention_forward,
    BlockingMode.H_MLA: blocked_h_mla_attention_forward,
    # prefill
    BlockingMode.PREFILL_Q: blocked_q_attention_forward_prefill,
    BlockingMode.PREFILL_KV: blocked_kv_attention_forward_prefill_headpar_offline,
    BlockingMode.PREFILL_QKV: blocked_qkv_attention_forward_prefill_headpar_offline,
    BlockingMode.PREFILL_ONLINE: blocked_qkv_attention_forward_prefill_online,
}


def _get_sliding_window_len(past_key_value: Cache, layer_idx: Optional[int] = None) -> int:
    if hasattr(past_key_value, "sliding_window_len"):
        return past_key_value.sliding_window_len
    return past_key_value.get_sliding_window_len(layer_idx)


# helper function needed both in generic blocked approach and in other modeling files for non-blocked approach
def past_key_value_update(
    module,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_value: Cache,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    sliding_window: Optional[int] = None,
    sliding_window_len: Optional[int] = None,
):
    if past_key_value is not None:
        cache_kwargs = {
            "batch_index": batch_index,
            "position_ids": position_ids,
        }
        if sliding_window is not None:
            cache_kwargs.update(
                {
                    "is_sliding": sliding_window is not None,
                    "sliding_window": sliding_window_len
                    if sliding_window_len is not None
                    else _get_sliding_window_len(past_key_value, module.layer_idx),
                }
            )
        if comp_ctx_lengths is not None:
            attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
            cache_kwargs["CCL"] = attention_mask.shape[-1]
        key, value = past_key_value.update(key, value, module.layer_idx, cache_kwargs)
    return key, value, attention_mask, cache_kwargs


def generic_blocked_attention_interface(
    module,
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    layer_idx: Optional[int] = None,
    past_key_value: Optional[Cache] = None,
    blocking_config: Optional[AttentionBlockingConfig] = None,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    block_table: Optional[torch.LongTensor] = None,
    slot_id: Optional[torch.LongTensor] = None,
    past_seen_tokens: Optional[int] = None,
    non_blocked_forward: Optional[Callable] = None,
    score_mod: Optional[Callable] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    mla_kwargs: Optional[Dict[str, Any]] = None,
    is_mla: bool = False,
    prefill_only: bool = False,
    **kwargs,
):
    blocking_mode = BlockingMode.resolve(blocking_config.mode)
    prefill_only = prefill_only or blocking_mode.is_prefill
    strategy = _STRATEGIES[
        BlockingMode.get_final_mode(blocking_config, prefill_only=prefill_only, is_mla=is_mla, mla_kwargs=mla_kwargs)
    ]

    # Keep a zero-valued, layer-local dependency in the traced graph for
    # batch-folded decode. The folded-cache index/mask expressions are built
    # by the layer-specific attention forward below. Without this dependency,
    # ONNX export can treat those expressions as identical across decoder
    # layers, lift them out of the subfunction body, and expose them as extra
    # subfunction inputs/outputs. Besides producing unnecessary subfunction
    # variants, that prevents the AIC compiler from applying its range
    # optimizations. The dependency is intentionally numerical no-op: it
    # preserves position_ids while anchoring the graph to this layer's cache.
    if blocking_mode == BlockingMode.KV_BATCH_FOLD and past_key_value is not None and position_ids is not None:
        layers = getattr(past_key_value, "layers", None)
        # Dynamic caches can start with no layer entries during initial tracing.
        # In that case, there is no cache tensor to use as a graph dependency;
        # leave position_ids unchanged until the requested layer is available.
        layer = layers[layer_idx] if layers is not None and layer_idx is not None and layer_idx < len(layers) else None
        layer_keys = getattr(layer, "keys", None)
        if layer_keys is not None:
            cache_shape = torch._shape_as_tensor(layer_keys)
            layer_local_zero = (cache_shape[0] - cache_shape[0]).to(dtype=position_ids.dtype)
            position_ids = position_ids + layer_local_zero

    cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}

    use_paged_kv_blocked = (
        blocking_config is not None
        and blocking_config.paged_attention
        and supports_paged_attention_blocked_kv(past_key_value)
    )

    if not is_mla:
        cache_kwargs["past_seen_tokens"] = past_seen_tokens
        if use_paged_kv_blocked and sliding_window is None:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids,
                "block_table": block_table,
                "slot_id": slot_id,
            }
            past_key_value.write_only_paged_attention(key, value, module.layer_idx, cache_kwargs)
        elif use_paged_kv_blocked and sliding_window is not None:
            raise NotImplementedError(
                "Sliding window attention is not supported with blocked KV caching. Please set `sliding_window` to None or use a different caching strategy."
            )
        elif prefill_only:
            if sliding_window is not None:
                cache_kwargs.update(
                    {
                        "is_sliding": sliding_window is not None,
                        "sliding_window": _get_sliding_window_len(past_key_value, module.layer_idx),
                    }
                )
            past_key_value.write_only(key, value, module.layer_idx, cache_kwargs)
        elif past_key_value is not None:
            use_kv_blocked = "kv" in blocking_config.mode and supports_blocked_kv(past_key_value)
            if blocking_mode == BlockingMode.KV_BATCH_FOLD:
                past_key_value.write_only_batch(key, value, module.layer_idx, cache_kwargs)
            elif use_kv_blocked and sliding_window is None:
                past_key_value.write_only(key, value, module.layer_idx, cache_kwargs)
            elif use_kv_blocked and sliding_window is not None:
                raise NotImplementedError(
                    "Sliding window attention is not supported with blocked KV caching. Please set `sliding_window` to None or use a different caching strategy."
                )
            else:
                key, value, attention_mask, cache_kwargs = past_key_value_update(
                    module=module,
                    key=key,
                    value=value,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    comp_ctx_lengths=comp_ctx_lengths,
                    batch_index=batch_index,
                    position_ids=position_ids,
                    sliding_window=sliding_window,
                )

    attn_output, attn_weights = strategy(
        # common
        module=module,
        query=query,
        attention_mask=attention_mask,
        scaling=scaling,
        cache_kwargs=cache_kwargs,
        layer_idx=layer_idx,
        score_mod=score_mod,
        position_bias=position_bias,
        sinks=sinks,
        # standard (decode / prefill) inputs
        key=key,
        value=value,
        past_key_value=past_key_value,
        sliding_window=sliding_window,
        # blocking dimensions
        num_kv_blocks=blocking_config.num_kv_blocks,
        num_q_blocks=blocking_config.num_q_blocks,
        head_block_size=blocking_config.head_block_size,
        num_batch_blocks=blocking_config.num_batch_blocks,
        configured_split=blocking_config.headpar_split,
        ctx_len=blocking_config.ctx_len,
        kv_block_unroll=blocking_config.kv_block_unroll,
        skip_kv=blocking_config.skip_kv or False,
        paged_attention=blocking_config.paged_attention,
        # prefill-specific
        n_rep_chunk=blocking_config.n_rep_chunk,
        num_cores_per_device=blocking_config.num_cores_per_device,
        # MLA-specific
        **(mla_kwargs or {}),
        **kwargs,
    )

    return attn_output, attn_weights
