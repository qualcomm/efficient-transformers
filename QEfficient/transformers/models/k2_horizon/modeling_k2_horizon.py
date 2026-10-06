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

Only the dense layers (K2HorizonAttention + K2HorizonMLP) are covered; MoVA/MoE
configs are refused at transform time. The upstream attention gate and QK norm
are config options that no published dense size enables; they are supported and
covered by the dummy-layer test config.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from QEfficient.blocking.attention_blocking import past_key_value_update
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.customop.utils import select_interface
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.models.llama.modeling_llama import eager_attention_forward, qeff_apply_rotary_pos_emb
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


class QEffK2HorizonRMSNorm(nn.Module):
    """
    Grouped RMSNorm (K2HorizonRMSNorm) on the compiler custom op. The norm runs
    over each of the n_groups slices of the last dim, then the full weight is applied.
    """

    def forward(self, hidden_states):
        rms_interface = select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)
        if self.n_groups == 1:
            return rms_interface(hidden_states, self.weight, self.variance_epsilon)
        group_size = self.hidden_size // self.n_groups
        grouped = hidden_states.reshape(*hidden_states.shape[:-1], self.n_groups, group_size)
        unit_weight = torch.ones(group_size, dtype=self.weight.dtype, device=self.weight.device)
        normed = rms_interface(grouped, unit_weight, self.variance_epsilon)
        return self.weight * normed.reshape(hidden_states.shape)


class QEffK2HorizonAttention(nn.Module):
    """
    Dense K2HorizonAttention with the QEff KV cache. Rotary tables come from the
    model (cos_cached/sin_cached) instead of being recomputed per call.
    """

    def __qeff_init__(self):
        if self.rope_head_dim != self.head_dim:
            raise NotImplementedError("K2 Horizon partial rotary (rope_head_dim != head_dim) is not supported yet")
        if self.sliding_window is not None:
            raise NotImplementedError("K2 Horizon sliding-window attention is not supported yet")

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
        if self.config.query_key_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

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

        if self.gate_func is not None:
            gate = self.gate_proj(hidden_states).view(*input_shape, -1, self.head_dim)
            if self.gate_func == "silu":
                gate = F.silu(gate)
            else:
                # softplus has no bf16 kernel on the compiler side, keep it in float32
                gate = F.softplus(gate.float(), beta=math.log(2)).to(gate.dtype)
            attn_output = attn_output * gate

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
        # sparse MoE layers return (hidden_states, router_logits)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return residual + hidden_states


class QEffK2HorizonModel(nn.Module):
    def __qeff_init__(self):
        if getattr(self.config, "mova_num_experts", 0) or getattr(self.config, "num_experts", 0):
            raise NotImplementedError(
                "K2 Horizon MoVA/MoE sizes are not supported yet; only the dense models "
                "(0.9B, 3.7B, 7B, 32B) are covered"
            )
        max_seq_len = min(self.config.max_position_embeddings, constants.K2_HORIZON_MAX_POSITION_EMBEDDINGS)
        if max_seq_len < self.config.max_position_embeddings:
            logger.warning(
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
