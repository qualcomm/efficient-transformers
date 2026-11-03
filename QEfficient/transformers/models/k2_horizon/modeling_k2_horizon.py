# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEff wrappers for the K2 Horizon family (IFM/K2-Horizon-*), a remote-code model
whose dense sizes are Llama-style GQA decoders with a grouped RMSNorm. The
upstream classes live in the Hub repo (modeling_k2_horizon.py), so these
wrappers are wired by class name through KVCacheExternalModuleMapperTransform
instead of by class object.

Only the dense sizes (0.9B, 3.7B, 7B, 32B: K2HorizonAttention + K2HorizonMLP in
every layer) are covered. Configs with experts, partial rotary, a sliding window,
QK norm or an attention gate are refused at transform time; no published dense
size sets any of them.
"""

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from QEfficient.blocking.attention_blocking import (
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.customop.utils import select_interface
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def qeff_apply_rotary_pos_emb(q, k, cos, sin):
    """Applies the rotary embedding from the pre-gathered cos/sin tables and casts back to the input dtype."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        masked_fill = torch.full_like(attn_weights, constants.MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype)
        attn_weights = torch.where(attention_mask, masked_fill, attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffK2HorizonRMSNorm(nn.Module):
    """
    Grouped RMSNorm (K2HorizonRMSNorm) on the compiler custom op. The norm runs
    over each of the n_groups slices of the last dim, then the full weight is applied.
    """

    def __qeff_init__(self):
        # unit scale for the grouped op, built once instead of on every forward (same as gemma4)
        if self.n_groups > 1 and not hasattr(self, "_qeff_unit_weight"):
            self.register_buffer(
                "_qeff_unit_weight", torch.ones_like(self.weight[: self.hidden_size // self.n_groups]), persistent=False
            )

    def forward(self, hidden_states):
        rms_interface = select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)
        if self.n_groups == 1:
            return rms_interface(hidden_states, self.weight, self.variance_epsilon)
        group_size = self.hidden_size // self.n_groups
        grouped = hidden_states.reshape(*hidden_states.shape[:-1], self.n_groups, group_size)
        normed = rms_interface(grouped, self._qeff_unit_weight, self.variance_epsilon)
        return self.weight * normed.reshape(hidden_states.shape)


class QEffK2HorizonAttention(nn.Module):
    """
    Dense K2HorizonAttention with the QEff KV cache. Rotary tables come from the
    model (cos_cached/sin_cached) instead of being recomputed per call. With an
    attn_blocking_config (qaic_config["blocking_mode"]) attention runs through
    generic_blocked_attention_interface, otherwise through the eager path.
    """

    def __qeff_init__(self):
        if self.rope_head_dim != self.head_dim:
            raise NotImplementedError("K2 Horizon partial rotary (rope_head_dim != head_dim) is not supported yet")
        if self.sliding_window is not None:
            raise NotImplementedError("K2 Horizon sliding-window attention is not supported yet")
        if self.config.query_key_norm:
            raise NotImplementedError("K2 Horizon query_key_norm is not supported; no dense size sets it")
        if self.gate_func is not None:
            raise NotImplementedError("K2 Horizon attention_gate_func is not supported; no dense size sets it")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        cos_cached: torch.Tensor | None = None,
        sin_cached: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        for key in ("output_attentions", "return_dict", "labels", "position_embeddings"):
            kwargs.pop(key, None)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        blocking_config = getattr(self, "attn_blocking_config", None)
        if blocking_config is not None and blocking_config.mode != BlockingMode.NONE:
            past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            attn_output, attn_weights = generic_blocked_attention_interface(
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                layer_idx=self.layer_idx,
                past_key_value=past_key_values,
                blocking_config=blocking_config,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
                past_seen_tokens=past_seen_tokens,
            )
        else:
            key, value, attention_mask, _ = past_key_value_update(
                module=self,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
            )
            attn_output, attn_weights = eager_attention_forward(
                self, query_states, key, value, attention_mask, scaling=self.scaling, **kwargs
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffK2HorizonDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class QEffK2HorizonModel(nn.Module):
    def __qeff_init__(self):
        if getattr(self.config, "mova_num_experts", 0) or getattr(self.config, "num_experts", 0):
            raise NotImplementedError(
                "only the dense K2 Horizon sizes (0.9B, 3.7B, 7B, 32B) are supported; this config has experts"
            )
        max_seq_len = min(self.config.max_position_embeddings, constants.K2_HORIZON_MAX_POSITION_EMBEDDINGS)
        if max_seq_len < self.config.max_position_embeddings:
            logger.info(
                f"K2 Horizon rotary tables are capped at {max_seq_len} positions "
                f"(config has {self.config.max_position_embeddings}); ctx_len must stay within the cap"
            )
        rotary = self.rotary_emb
        positions = torch.arange(max_seq_len, dtype=torch.int64).type_as(rotary.inv_freq)
        freqs = torch.outer(positions, rotary.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = self.embed_tokens.weight.dtype
        self.cos_cached = nn.Parameter((emb.cos() * rotary.attention_scaling).to(dtype))
        self.sin_cached = nn.Parameter((emb.sin() * rotary.attention_scaling).to(dtype))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)
        sin = self.sin_cached[position_ids].unsqueeze(1).to(device=inputs_embeds.device)
        cos = self.cos_cached[position_ids].unsqueeze(1).to(device=inputs_embeds.device)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class QEffK2HorizonForCausalLM(nn.Module):
    def get_submodules_for_export(self) -> set[type[nn.Module]]:
        return {type(self.model.layers[0])}

    def get_specializations(self, batch_size: int, prefill_seq_len: int, ctx_len: int, **kwargs):
        # the baked rotary table (see QEffK2HorizonModel.__qeff_init__) is the real limit
        cap = self.model.cos_cached.shape[0]
        if prefill_seq_len > cap or ctx_len > cap:
            raise ValueError(
                f"K2 Horizon rotary tables hold {cap} positions; "
                f"prefill_seq_len={prefill_seq_len} and ctx_len={ctx_len} must not exceed it"
            )
        return [
            {"batch_size": batch_size, "seq_len": prefill_seq_len, "ctx_len": ctx_len},
            {"batch_size": batch_size, "seq_len": 1, "ctx_len": ctx_len},
        ]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        for key in ("output_router_logits", "logits_to_keep", "labels"):
            kwargs.pop(key, None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        # Cast to INT32 to avoid issue while running in ONNXRT
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
