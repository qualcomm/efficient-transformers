# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""QEfficient methods for the remote-code MolmoPoint architecture."""

import math
from dataclasses import dataclass
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.transformers.cache_utils import InvalidIndexProvider, QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

VISION_OUTPUTS = [
    "vision_embeds",
    "vision_embeds_vit_features",
    "vision_embeds_vit_mask",
    "vision_embeds_subpatch_k",
]
POINT_STATES = [
    "vision_embeds_patch_k",
    "vision_embeds_patch_mask",
    "vision_embeds_image_pos_ids",
    "vision_embeds_last_patch_id",
]
POINT_ROTARY_MAX_POSITIONS = 4096
VISION_EMBEDS_BRIDGE_SCALE = 4096.0


@dataclass
class QEffMolmoPointTextOutput(BaseModelOutputWithPast):
    """Text-model output retaining the pre-final-norm state used by the point head."""

    pre_ln_hidden_state: Optional[torch.FloatTensor] = None


def _reset_molmo_point_patch_rope(model: nn.Module) -> None:
    """Rebuild the remote model's nonpersistent point-RoPE frequencies."""

    patch_rotary = model.model.point_predictor.patch_rotary
    if patch_rotary is None:
        return
    theta = getattr(model.config, "token_prediction_rotary_theta", None)
    if theta is None:
        text_config = getattr(model.config, "text_config", getattr(model.config, "llm", None))
        theta = text_config.rope_theta
    dim = model.config.patch_embed_dim
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=patch_rotary.inv_freq.device, dtype=torch.float32)
            / dim
        )
    )
    patch_rotary.register_buffer("inv_freq", inv_freq, persistent=False)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    head_dim = hidden_states.shape[-1]
    first_half = hidden_states[..., : head_dim // 2]
    second_half = hidden_states[..., head_dim // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_rotary_pos_emb(query, key, cos, sin):
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query, key


def _apply_point_rotary(module, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Apply the point head's one-dimensional RoPE without data-dependent reshapes."""
    return module(hidden_states, position_ids)


def _mask_base_logits(logits: torch.Tensor, token_ids: tuple[int, ...], value: float) -> torch.Tensor:
    vocab_indices = torch.arange(logits.shape[-1], device=logits.device).view(1, 1, logits.shape[-1])
    selected = vocab_indices == token_ids[0]
    for token_id in token_ids[1:]:
        selected = selected | (vocab_indices == token_id)
    return torch.where(selected, torch.full_like(logits, value), logits)


def _is_decode(input_ids: torch.Tensor) -> torch.Tensor:
    return torch._shape_as_tensor(input_ids).to(device=input_ids.device)[1] == 1


def _gather_image_position_ids(
    image_position_ids: torch.Tensor, last_patch_ids: torch.Tensor
) -> torch.Tensor:
    """Gather point-RoPE positions without exporting out-of-range sentinel indices."""
    safe_patch_ids = last_patch_ids.to(torch.int64).clamp(
        min=0, max=image_position_ids.shape[1] - 1
    )
    gathered_positions = image_position_ids.gather(1, safe_patch_ids)
    return torch.where(last_patch_ids >= 0, gathered_positions, torch.zeros_like(gathered_positions))


class QEffMolmoPointRotaryEmbedding(nn.Module):
    """Static RoPE lookup used by the Molmo2-derived text decoder."""

    def __qeff_init__(self):
        positions = torch.arange(
            self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=torch.int64,
        ).to(self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", embeddings.cos(), persistent=False)
        self.register_buffer("sin_cached", embeddings.sin(), persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        cos = self.cos_cached[position_ids] * self.attention_scaling
        sin = self.sin_cached[position_ids] * self.attention_scaling
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


class QEffMolmoPointPatchRope(nn.Module):
    """Static lookup for point-head RoPE; AI100 cannot safely lower runtime trigonometry."""

    def __qeff_init__(self):
        positions = torch.arange(
            -1,
            POINT_ROTARY_MAX_POSITIONS,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        frequencies = torch.outer(positions, self.inv_freq.float())
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", embeddings.cos(), persistent=False)
        self.register_buffer("sin_cached", embeddings.sin(), persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        cache_index = (position_ids + 1).clamp(min=0, max=POINT_ROTARY_MAX_POSITIONS)
        cos = self.cos_cached[cache_index].to(hidden_states.device)
        sin = self.sin_cached[cache_index].to(hidden_states.device)
        hidden_states_float = hidden_states.float()
        output = (hidden_states_float * cos) + (_rotate_half(hidden_states_float) * sin)
        return output.to(original_dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
):
    batch_size, num_key_value_heads, sequence_length, head_dim = key.shape
    if module.num_key_value_groups != 1:
        key = key[:, :, None, :, :].expand(
            batch_size,
            num_key_value_heads,
            module.num_key_value_groups,
            sequence_length,
            head_dim,
        )
        value = value[:, :, None, :, :].expand(
            batch_size,
            num_key_value_heads,
            module.num_key_value_groups,
            sequence_length,
            head_dim,
        )
        key = key.reshape(batch_size, module.num_heads, sequence_length, head_dim)
        value = value.reshape(batch_size, module.num_heads, sequence_length, head_dim)

    attention_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        masked = torch.full_like(attention_weights, MIN_MASKED_ATTENTION_VALUE)
        attention_weights = torch.where(attention_mask, masked, attention_weights)
    attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if attention_mask is not None:
        attention_weights = torch.where(attention_mask, torch.zeros_like(attention_weights), attention_weights)
    attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous()
    return attention_output, attention_weights


class QEffMolmoPointAttention(nn.Module):
    """Molmo2 fused-QKV attention with QEff retained-cache updates."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size, query_length, _ = hidden_states.shape
        query_dim, key_dim, value_dim = self.fused_dims
        query_states, key_states, value_states = self.att_proj(hidden_states).split(
            (query_dim, key_dim, value_dim), dim=-1
        )

        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type != "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(batch_size, query_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type == "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(
            query_states,
            key_states,
            cos.unsqueeze(1),
            sin.unsqueeze(1),
        )

        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        if blocking_config is not None and blocking_config.mode != BlockingMode.NONE:
            attention_output, attention_weights = generic_blocked_attention_interface(
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
                block_table=block_table,
                slot_id=slot_id,
                past_seen_tokens=past_seen_tokens,
            )
        else:
            key_states = key_states.float()
            value_states = value_states.float()
            key_states, value_states, attention_mask, _ = past_key_value_update(
                module=self,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
            )
            attention_output, attention_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
            )

        attention_output = attention_output.reshape(
            batch_size, query_length, self.num_heads * self.head_dim
        ).contiguous()
        return self.attn_out(attention_output), attention_weights


class QEffMolmoPointDecoderLayer(nn.Module):
    """Pre-norm and post-norm decoder-layer methods used by MolmoPoint."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attention_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            block_table=block_table,
            slot_id=slot_id,
        )
        hidden_states = residual + self.dropout(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.ff_norm(hidden_states))
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states, attention_weights

    def forward_post_norm(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, attention_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            block_table=block_table,
            slot_id=slot_id,
        )
        hidden_states = residual + self.dropout(self.attn_norm(hidden_states))
        residual = hidden_states
        hidden_states = residual + self.dropout(self.ff_norm(self.mlp(hidden_states)))
        return hidden_states, attention_weights


class QEffMolmoPointTextModel(nn.Module):
    """MolmoPoint text decoder with static causal masking and legacy-cache boundaries."""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        if inputs_embeds is None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            inputs_embeds = self.wte(input_ids)

        return_legacy_cache = use_cache and not isinstance(past_key_values, Cache)
        if return_legacy_cache:
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
        if past_key_values is None:
            past_key_values = QEffDynamicCache()

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).view(
                1, inputs_embeds.shape[1]
            )
        target_length = past_key_values.get_seq_length()
        if target_length == 0:
            target_length = inputs_embeds.shape[1]
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_length)
        if token_type_ids is not None and inputs_embeds.shape[1] != 1:
            key_token_types = F.pad(
                token_type_ids.to(torch.bool),
                (0, max(0, target_length - token_type_ids.shape[1])),
            )[:, :target_length]
            image_block = token_type_ids.to(torch.bool).unsqueeze(-1) & key_token_types.unsqueeze(1)
            causal_mask = causal_mask & ~image_block.unsqueeze(1)

        hidden_states = inputs_embeds
        if self.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": self.rotary_embs["default"](hidden_states, position_ids),
                "scaling": self.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        for layer_idx, decoder_block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in self.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings
            hidden_states = decoder_block(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                block_table=block_table,
                slot_id=slot_id,
                position_embeddings=position_embeddings_i,
            )[0]

        pre_ln_hidden_state = hidden_states
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()
        return QEffMolmoPointTextOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            pre_ln_hidden_state=pre_ln_hidden_state,
        )


class QEffMolmoPointVisionAttention(nn.Module):
    """ONNX-friendly Molmo2 ViT attention matching the Hub eager path."""

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        inputs_k = inputs_q if inputs_kv is None else inputs_kv
        inputs_v = inputs_q if inputs_kv is None else inputs_kv
        batch_size, query_length, _ = inputs_q.shape
        key_length = inputs_k.shape[1]
        query = self.wq(inputs_q).reshape(batch_size, query_length, self.num_heads, self.head_dim)
        key = self.wk(inputs_k).reshape(batch_size, key_length, self.num_key_value_heads, self.head_dim)
        value = self.wv(inputs_v).reshape(batch_size, key_length, self.num_key_value_heads, self.head_dim)
        if self.num_key_value_groups != 1:
            key = key[:, :, :, None, :].expand(
                batch_size,
                key_length,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            ).reshape(batch_size, key_length, self.num_heads, self.head_dim)
            value = value[:, :, :, None, :].expand(
                batch_size,
                key_length,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            ).reshape(batch_size, key_length, self.num_heads, self.head_dim)

        original_dtype = query.dtype
        if self.float32_attention:
            query = query.float()
            key = key.float()
            value = value.float()
        scores = torch.matmul(
            query.permute(0, 2, 1, 3) / math.sqrt(self.head_dim),
            key.permute(0, 2, 3, 1),
        )
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        attention_output = torch.matmul(probabilities, value.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        attention_output = attention_output.to(original_dtype).reshape(
            batch_size, query_length, self.num_heads * self.head_dim
        )
        if self.wo is not None:
            attention_output = self.wo(attention_output)
        return self.residual_dropout(attention_output)


def _vision_forward(model, pixel_values: torch.Tensor, image_token_pooling: torch.Tensor):
    batch_size, num_crops, num_patches, pixels_per_patch = pixel_values.shape
    images = pixel_values.to(device=model.model.device, dtype=model.model.dtype).reshape(
        batch_size * num_crops, num_patches, pixels_per_patch
    )
    all_features = model.model.vit(images)
    selected_features = [all_features[layer] for layer in model.model.vit_layers]
    vision_features = torch.cat(selected_features, dim=-1)
    vision_dim = vision_features.shape[-1]
    image_token_count = image_token_pooling.shape[1]
    pool_dim = image_token_pooling.shape[2]
    batch_indices = torch.arange(batch_size, device=image_token_pooling.device).view(batch_size, 1, 1)
    batch_indices = batch_indices.expand(batch_size, image_token_count, pool_dim)
    vision_features = vision_features.reshape(batch_size, num_crops * num_patches, vision_dim)
    vision_features = vision_features[batch_indices, image_token_pooling.clamp(min=0)]
    vision_mask = image_token_pooling >= 0
    vision_features = vision_features * vision_mask.unsqueeze(-1).to(vision_features.dtype)

    if model.config.adapter_config.positional_embeddings:
        vision_features = model.model.connector.positional_embeddings(vision_features)

    flat_features = vision_features.reshape(batch_size * image_token_count, pool_dim, vision_dim)
    flat_mask = vision_mask.reshape(batch_size * image_token_count, pool_dim)
    denominator = torch.einsum("bp->b", flat_mask.float()).clamp(min=1.0)
    query = torch.einsum("bpd->bd", flat_features).unsqueeze(1) / denominator[:, None, None]
    attention_mask = flat_mask[:, None, None, :] if model.config.adapter_config.pooling_attention_mask else None
    pooled_features = model.model.connector.image_pooling_2d(query, flat_features, attn_mask=attention_mask)
    vision_embeds = model.model.connector.image_projector(pooled_features)
    vision_embeds = vision_embeds.reshape(batch_size, image_token_count, model.config.text_config.hidden_size)
    vision_embeds = vision_embeds.clamp(-57000.0, 57000.0) / VISION_EMBEDS_BRIDGE_SCALE
    subpatch_keys = model.model.point_predictor.subpatch_k(vision_features)
    subpatch_keys = subpatch_keys.clamp(-57000.0, 57000.0)
    return vision_embeds, vision_features, vision_mask.float(), subpatch_keys


def _point_bounds(model, vision_embeds: torch.Tensor, vision_features: torch.Tensor):
    base_vocab_size = model.config.text_config.vocab_size + model.config.text_config.additional_vocab_size
    num_patches = vision_embeds.shape[1]
    num_patch_classes = num_patches + int(model.config.no_more_points_class)
    num_subpatches = vision_features.shape[2]
    patch_start = base_vocab_size
    patch_end_without_no_more = patch_start + num_patches
    subpatch_start = patch_start + num_patch_classes
    location_start = subpatch_start + num_subpatches
    return patch_start, patch_end_without_no_more, subpatch_start, location_start


def _language_forward(
    model,
    input_ids,
    vision_embeds,
    vision_embeds_vit_features,
    vision_embeds_vit_mask,
    vision_embeds_subpatch_k,
    vision_embeds_patch_k,
    vision_embeds_patch_mask,
    vision_embeds_image_pos_ids,
    vision_embeds_last_patch_id,
    position_ids,
    past_key_values,
    token_type_ids=None,
    comp_ctx_lengths=None,
    batch_index=None,
    block_table=None,
    slot_id=None,
):
    vision_embeds = vision_embeds.float()
    vision_embeds_vit_features = vision_embeds_vit_features.float()
    vision_embeds_vit_mask = vision_embeds_vit_mask.float()
    vision_embeds_subpatch_k = vision_embeds_subpatch_k.float()
    vision_embeds_patch_k = vision_embeds_patch_k.float()
    vision_embeds_patch_mask = vision_embeds_patch_mask.float()
    vision_embeds_image_pos_ids = vision_embeds_image_pos_ids.float()
    vision_embeds_last_patch_id = vision_embeds_last_patch_id.float()
    batch_size, sequence_length = input_ids.shape
    point_dim = model.config.patch_embed_dim
    num_image_tokens = vision_embeds.shape[1]
    pool_dim = vision_embeds_vit_features.shape[2]
    decoder_vision_embeds = vision_embeds * VISION_EMBEDS_BRIDGE_SCALE
    patch_start, patch_end, subpatch_start, location_start = _point_bounds(
        model, vision_embeds, vision_embeds_vit_features
    )

    is_patch = (input_ids >= patch_start) & (input_ids < patch_end)
    no_more_token_id = patch_end if model.config.no_more_points_class else -1
    is_no_more = (
        input_ids == no_more_token_id
        if model.config.no_more_points_class
        else torch.zeros_like(input_ids).bool()
    )
    is_subpatch = (input_ids >= subpatch_start) & (input_ids < location_start)
    is_location = (input_ids >= location_start) & (input_ids < location_start + 9)
    patch_ids = torch.where(is_patch, input_ids - patch_start, 0)
    subpatch_ids = torch.where(is_subpatch, input_ids - subpatch_start, 0)
    embedding_ids = torch.where(is_patch | is_no_more, model.config.patch_token_id, input_ids)
    embedding_ids = torch.where(is_subpatch, model.config.subpatch_token_id, embedding_ids)
    embedding_ids = torch.where(is_location, model.config.location_token_id, embedding_ids)
    embedding_ids = embedding_ids * (embedding_ids != -1).to(embedding_ids.dtype)
    text_embeds = model.model.transformer.wte(embedding_ids)

    image_mask = (embedding_ids == model.config.image_patch_id) | (
        embedding_ids == model.config.image_non_indexable_patch_id
    )
    image_slots = image_mask.to(torch.int64).cumsum(1) - 1
    batch_indices = torch.arange(batch_size, device=input_ids.device).view(batch_size, 1)
    gathered_vision = decoder_vision_embeds[
        batch_indices, image_slots.clamp(min=0, max=num_image_tokens - 1)
    ]
    merged_embeds = text_embeds + torch.where(image_mask.unsqueeze(-1), gathered_vision, torch.zeros_like(text_embeds))

    patch_embeddings = decoder_vision_embeds[
        batch_indices, patch_ids.clamp(min=0, max=num_image_tokens - 1)
    ]
    text_embeds = torch.where(is_patch.unsqueeze(-1), text_embeds + patch_embeddings, text_embeds)
    last_patch = vision_embeds_last_patch_id.to(torch.int64).clamp(min=0, max=num_image_tokens - 1)
    last_patch = last_patch.expand(batch_size, sequence_length)
    chosen_patch = torch.where(is_subpatch, last_patch, torch.zeros_like(last_patch))
    chosen_features = vision_embeds_vit_features[
        batch_indices,
        chosen_patch,
        subpatch_ids.clamp(min=0, max=pool_dim - 1),
    ]
    subpatch_embeddings = model.model.build_vit_embedding(chosen_features)
    text_embeds = torch.where(is_subpatch.unsqueeze(-1), subpatch_embeddings.to(text_embeds.dtype), text_embeds)
    inputs_embeds = torch.where(_is_decode(input_ids), text_embeds, merged_embeds)
    inputs_embeds = model.model.transformer.emb_drop(inputs_embeds)

    outputs = model.model.transformer(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        past_key_values=past_key_values,
        token_type_ids=token_type_ids,
        comp_ctx_lengths=comp_ctx_lengths,
        batch_index=batch_index,
        block_table=block_table,
        slot_id=slot_id,
        use_cache=True,
    )

    pre_ln_states = outputs.pre_ln_hidden_state
    normalized_states = model.model.point_predictor.x_norm(pre_ln_states)
    slot_ids = image_mask.to(torch.int64).cumsum(1) - 1
    slots = torch.arange(num_image_tokens, device=input_ids.device).view(1, 1, num_image_tokens)
    slot_assignment = (slot_ids.unsqueeze(-1) == slots) & image_mask.unsqueeze(-1)
    patch_keys_for_tokens = model.model.point_predictor.patch_k(normalized_states)
    patch_keys_for_tokens = patch_keys_for_tokens.clamp(-57000.0, 57000.0)
    patch_keys_for_tokens = torch.where(
        image_mask.unsqueeze(-1),
        patch_keys_for_tokens,
        torch.zeros_like(patch_keys_for_tokens),
    )
    new_patch_keys = torch.matmul(slot_assignment.transpose(1, 2).to(normalized_states.dtype), patch_keys_for_tokens)
    indexable_mask = embedding_ids == model.config.image_patch_id
    new_patch_mask = torch.matmul(
        slot_assignment.transpose(1, 2).to(normalized_states.dtype),
        indexable_mask.unsqueeze(-1).to(normalized_states.dtype),
    ).squeeze(-1)
    image_positions = indexable_mask.to(torch.int64).cumsum(1) - 1
    new_image_pos_ids = torch.matmul(
        slot_assignment.transpose(1, 2).to(normalized_states.dtype),
        image_positions.unsqueeze(-1).to(normalized_states.dtype),
    ).squeeze(-1)
    if model.model.point_predictor.patch_rotary is not None:
        flat_patch_keys = new_patch_keys.reshape(batch_size * num_image_tokens, point_dim)
        flat_positions = new_image_pos_ids.reshape(batch_size * num_image_tokens).to(torch.int64)
        new_patch_keys = _apply_point_rotary(
            model.model.point_predictor.patch_rotary, flat_patch_keys, flat_positions
        ).reshape(batch_size, num_image_tokens, point_dim)
    if model.config.no_more_points_class:
        no_point_key = model.model.point_predictor.add_no_point_class_embed.vector.view(1, 1, point_dim)
        no_point_key = no_point_key.expand(batch_size, 1, point_dim)
        new_patch_keys = torch.cat((new_patch_keys, no_point_key), dim=1)
        new_patch_mask = torch.cat((new_patch_mask, torch.ones_like(new_patch_mask[:, :1])), dim=1)

    is_decode = _is_decode(input_ids)
    patch_keys = torch.where(is_decode, vision_embeds_patch_k, new_patch_keys)
    patch_mask = torch.where(is_decode, vision_embeds_patch_mask, new_patch_mask)
    image_pos_ids = torch.where(is_decode, vision_embeds_image_pos_ids, new_image_pos_ids)

    logit_index = position_ids.argmax(1, keepdim=True).to(torch.int32)
    output_batch_indices = torch.arange(batch_size, device=input_ids.device, dtype=torch.int32).view(batch_size, 1)
    hidden_states = outputs.last_hidden_state[output_batch_indices, logit_index].float()
    point_states = normalized_states[output_batch_indices, logit_index]
    language_weights = torch.cat((model.lm_head.output_embeddings, model.lm_head.new_output_embeddings), dim=0)
    logits = F.linear(hidden_states, language_weights).float()

    logits = logits + vision_embeds[:, :1, :1] * 0
    logits = logits + vision_embeds_patch_k[:, :1, :1] * 0
    logits = logits + vision_embeds_patch_mask[:, :1, None] * 0
    logits = logits + vision_embeds_image_pos_ids[:, :1, None] * 0

    patch_query = model.model.point_predictor.patch_q(point_states)
    rotate_by = _gather_image_position_ids(image_pos_ids, vision_embeds_last_patch_id)
    if model.model.point_predictor.patch_rotary is not None:
        patch_query = _apply_point_rotary(
            model.model.point_predictor.patch_rotary,
            patch_query.reshape(batch_size, point_dim),
            rotate_by.reshape(batch_size).to(torch.int64),
        ).reshape(batch_size, 1, point_dim)
    patch_logits = torch.matmul(patch_query, patch_keys.transpose(1, 2))
    if model.config.norm_logits:
        patch_logits = patch_logits / math.sqrt(patch_logits.shape[-1])
    patch_logits = torch.where(patch_mask[:, None, :] > 0, patch_logits, torch.full_like(patch_logits, -100000.0))

    selected_patch = patch_ids[:, -1].clamp(min=0, max=num_image_tokens - 1)
    chosen_subpatch_keys = vision_embeds_subpatch_k[batch_indices[:, 0], selected_patch]
    subpatch_query = model.model.point_predictor.subpatch_q(point_states)
    subpatch_logits = torch.matmul(subpatch_query, chosen_subpatch_keys.transpose(1, 2))
    if model.config.norm_logits:
        subpatch_logits = subpatch_logits / math.sqrt(point_dim)
    chosen_subpatch_mask = vision_embeds_vit_mask[batch_indices[:, 0], selected_patch]
    subpatch_logits = torch.where(
        chosen_subpatch_mask[:, None, :] > 0,
        subpatch_logits,
        torch.full_like(subpatch_logits, -100000.0),
    )
    subpatch_logits = torch.where(is_patch[:, -1:, None], subpatch_logits, torch.full_like(subpatch_logits, -100000.0))
    location_logits = model.model.point_predictor.subpatch_loc_k(point_states)
    location_logits = torch.where(
        is_subpatch[:, -1:, None], location_logits, torch.full_like(location_logits, -100000.0)
    )

    patch_token_logits = logits[:, :, model.config.patch_token_id : model.config.patch_token_id + 1]
    selected_patch_class = patch_logits.argmax(-1, keepdim=True)
    patch_classes = torch.arange(patch_logits.shape[-1], device=logits.device).view(1, 1, patch_logits.shape[-1])
    emitted_patch_logits = torch.where(
        patch_classes == selected_patch_class,
        patch_token_logits.expand(batch_size, 1, patch_logits.shape[-1]),
        torch.full_like(patch_logits, -100000.0),
    )
    emitted_patch_logits = torch.where(
        is_decode,
        emitted_patch_logits,
        torch.full_like(emitted_patch_logits, -100000.0),
    )
    logits = _mask_base_logits(
        logits,
        (model.config.patch_token_id, model.config.subpatch_token_id, model.config.location_token_id),
        -100000.0,
    )
    logits = torch.cat((logits, emitted_patch_logits, subpatch_logits, location_logits), dim=-1)

    current_patch_id = torch.where(
        is_patch[:, -1:], patch_ids[:, -1:], vision_embeds_last_patch_id.to(torch.int64)
    )
    prefill_patch_id = torch.minimum(current_patch_id, torch.full_like(current_patch_id, -1))
    current_patch_id = torch.where(is_decode, current_patch_id, prefill_patch_id)
    last_patch_id = current_patch_id.to(vision_embeds_last_patch_id.dtype)
    return (
        logits,
        vision_embeds,
        vision_embeds_vit_features,
        vision_embeds_vit_mask,
        vision_embeds_subpatch_k,
        patch_keys,
        patch_mask,
        image_pos_ids,
        last_patch_id,
        outputs.past_key_values,
    )


class QEffMolmoPointVisionBackbone(nn.Module):
    """Vision-only QPC wrapper for MolmoPoint."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.model.vit.transformer.resblocks[0].__class__}

    def forward(self, pixel_values, image_token_pooling):
        return _vision_forward(self.model, pixel_values, image_token_pooling)


class QEffMolmoPointLanguageDecoder(nn.Module):
    """Language/pointing QPC wrapper with explicit non-KV retained state."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.language_model = model.model.transformer

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.model.transformer.blocks[0].__class__}

    def forward(
        self,
        input_ids,
        vision_embeds,
        vision_embeds_vit_features,
        vision_embeds_vit_mask,
        vision_embeds_subpatch_k,
        vision_embeds_patch_k,
        vision_embeds_patch_mask,
        vision_embeds_image_pos_ids,
        vision_embeds_last_patch_id,
        position_ids,
        past_key_values,
        token_type_ids=None,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
    ):
        return _language_forward(
            self.model,
            input_ids,
            vision_embeds,
            vision_embeds_vit_features,
            vision_embeds_vit_mask,
            vision_embeds_subpatch_k,
            vision_embeds_patch_k,
            vision_embeds_patch_mask,
            vision_embeds_image_pos_ids,
            vision_embeds_last_patch_id,
            position_ids,
            past_key_values,
            token_type_ids,
            comp_ctx_lengths,
            batch_index,
            block_table,
            slot_id,
        )


class QEffMolmoPointForConditionalGeneration(nn.Module):
    """QEff export interface for MolmoPoint single- and dual-QPC execution."""

    def __qeff_init__(self):
        InvalidIndexProvider.enable_subfunc()
        _reset_molmo_point_patch_rope(self)

    def get_qeff_vision_encoder(self):
        return QEffMolmoPointVisionBackbone(self)

    def get_qeff_language_decoder(self):
        return QEffMolmoPointLanguageDecoder(self)

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {
            self.model.vit.transformer.resblocks[0].__class__,
            self.model.transformer.blocks[0].__class__,
        }

    def forward(
        self,
        pixel_values,
        image_token_pooling,
        input_ids,
        vision_embeds,
        vision_embeds_vit_features,
        vision_embeds_vit_mask,
        vision_embeds_subpatch_k,
        vision_embeds_patch_k,
        vision_embeds_patch_mask,
        vision_embeds_image_pos_ids,
        vision_embeds_last_patch_id,
        position_ids,
        past_key_values,
        token_type_ids=None,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        new_vision_states = _vision_forward(self, pixel_values, image_token_pooling)
        is_decode = _is_decode(input_ids)
        vision_embeds = torch.where(is_decode, vision_embeds, new_vision_states[0])
        vision_embeds_vit_features = torch.where(is_decode, vision_embeds_vit_features, new_vision_states[1])
        vision_embeds_vit_mask = torch.where(is_decode, vision_embeds_vit_mask, new_vision_states[2])
        vision_embeds_subpatch_k = torch.where(is_decode, vision_embeds_subpatch_k, new_vision_states[3])
        return _language_forward(
            self,
            input_ids,
            vision_embeds,
            vision_embeds_vit_features,
            vision_embeds_vit_mask,
            vision_embeds_subpatch_k,
            vision_embeds_patch_k,
            vision_embeds_patch_mask,
            vision_embeds_image_pos_ids,
            vision_embeds_last_patch_id,
            position_ids,
            past_key_values,
            token_type_ids,
            comp_ctx_lengths,
        )

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        vision_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        compiler_options.pop("img_size", None)
        prefill_seq_len = prefill_seq_len or constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        ctx_len = ctx_len or constants.ONNX_EXPORT_CTX_LEN
        vision_batch_size = batch_size if vision_batch_size is None else vision_batch_size
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size
        vision_values = {
            "vision_batch_size": vision_batch_size,
            "num_crops": int(compiler_options.pop("num_crops", 2)),
            "num_patches": int(compiler_options.pop("num_patches", self.config.vit_config.image_num_pos)),
            "pixels_per_patch": int(
                compiler_options.pop("pixels_per_patch", self.config.vit_config.image_patch_size**2 * 3)
            ),
            "num_image_tokens": int(compiler_options.pop("num_image_tokens", 392)),
            "pool_dim": int(compiler_options.pop("pool_dim", 4)),
        }
        num_patch_classes = vision_values["num_image_tokens"] + int(self.config.no_more_points_class)
        vision = [vision_values.copy()]

        def language_spec(sequence_length, comp_ctx_length=None):
            spec = {
                "batch_size": full_batch_size if continuous_batching and sequence_length == "1" else batch_size,
                "seq_len": sequence_length,
                "ctx_len": ctx_len,
                "vision_batch_size": vision_batch_size,
                "num_image_tokens": vision_values["num_image_tokens"],
                "num_patch_classes": num_patch_classes,
                "pool_dim": vision_values["pool_dim"],
            }
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            if comp_ctx_length is not None:
                spec["comp_ctx_lengths"] = comp_ctx_length
            if not kv_offload:
                spec.update(vision_values)
            return spec

        if comp_ctx_lengths_prefill is not None and comp_ctx_lengths_decode is not None:
            lang = [language_spec(prefill_seq_len, value) for value in comp_ctx_lengths_prefill]
            lang.extend(language_spec("1", value) for value in comp_ctx_lengths_decode)
        else:
            lang = [language_spec(prefill_seq_len), language_spec("1")]
        return ({"vision": vision, "lang": lang}, compiler_options) if kv_offload else (lang, compiler_options)

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        vision_axes = {
            "pixel_values": {0: "vision_batch_size", 1: "num_crops", 2: "num_patches", 3: "pixels_per_patch"},
            "image_token_pooling": {0: "vision_batch_size", 1: "num_image_tokens", 2: "pool_dim"},
            "vision_embeds": {0: "vision_batch_size", 1: "num_image_tokens"},
            "vision_embeds_vit_features": {0: "vision_batch_size", 1: "num_image_tokens", 2: "pool_dim"},
            "vision_embeds_vit_mask": {0: "vision_batch_size", 1: "num_image_tokens", 2: "pool_dim"},
            "vision_embeds_subpatch_k": {0: "vision_batch_size", 1: "num_image_tokens", 2: "pool_dim"},
        }
        lang_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "token_type_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "num_image_tokens"},
            "vision_embeds_vit_features": {0: "vision_batch_size", 1: "num_image_tokens", 2: "pool_dim"},
            "vision_embeds_vit_mask": {0: "vision_batch_size", 1: "num_image_tokens", 2: "pool_dim"},
            "vision_embeds_subpatch_k": {0: "vision_batch_size", 1: "num_image_tokens", 2: "pool_dim"},
            "vision_embeds_patch_k": {0: "batch_size", 1: "num_patch_classes"},
            "vision_embeds_patch_mask": {0: "batch_size", 1: "num_patch_classes"},
            "vision_embeds_image_pos_ids": {0: "batch_size", 1: "num_image_tokens"},
            "vision_embeds_last_patch_id": {0: "batch_size"},
        }
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            cache_batch_axis = "full_batch_size" if continuous_batching else "batch_size"
            lang_axes[f"past_key.{layer_idx}"] = {0: cache_batch_axis, 2: "ctx_len"}
            lang_axes[f"past_value.{layer_idx}"] = {0: cache_batch_axis, 2: "ctx_len"}
        if comp_ctx_lengths is not None:
            lang_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}
        if continuous_batching:
            lang_axes["batch_index"] = {0: "batch_size"}
        return {"vision": vision_axes, "lang": lang_axes} if kv_offload else {**vision_axes, **lang_axes}

    def get_output_names(self, kv_offload: bool = False):
        state_names = [f"{name}_RetainedState" for name in VISION_OUTPUTS + POINT_STATES]
        language_outputs = ["logits", *state_names]
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            language_outputs.extend(
                [f"past_key.{layer_idx}_RetainedState", f"past_value.{layer_idx}_RetainedState"]
            )
        if kv_offload:
            return {"vision": VISION_OUTPUTS.copy(), "lang": language_outputs}
        return language_outputs

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        batch_size = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        sequence_length = int(kwargs.get("prefill_seq_len") or constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        context_length = int(kwargs.get("ctx_len") or constants.ONNX_EXPORT_CTX_LEN)
        num_crops = int(kwargs.get("num_crops", 1))
        num_patches = int(kwargs.get("num_patches", self.config.vit_config.image_num_pos))
        pixels_per_patch = int(kwargs.get("pixels_per_patch", self.config.vit_config.image_patch_size**2 * 3))
        num_image_tokens = int(kwargs.get("num_image_tokens", min(16, sequence_length)))
        pool_dim = int(kwargs.get("pool_dim", 4))
        vision_dim = self.config.vit_config.hidden_size * len(self.config.adapter_config.vit_layers)
        point_dim = self.config.patch_embed_dim
        text_dim = self.config.text_config.hidden_size
        model_dtype = getattr(self.config, "torch_dtype", None) or getattr(self.config, "dtype", torch.float32)

        pixel_values = torch.zeros(
            (batch_size, num_crops, num_patches, pixels_per_patch), dtype=model_dtype
        )
        image_token_pooling = torch.arange(pool_dim, dtype=torch.int64).view(1, 1, pool_dim)
        image_token_pooling = image_token_pooling.expand(batch_size, num_image_tokens, pool_dim).clone()
        vision_inputs = {"pixel_values": pixel_values, "image_token_pooling": image_token_pooling}
        input_ids = torch.zeros((batch_size, sequence_length), dtype=torch.int64)
        input_ids[:, :1] = self.config.image_patch_id
        position_ids = torch.arange(sequence_length, dtype=torch.int64).view(1, sequence_length)
        position_ids = position_ids.expand(batch_size, sequence_length).clone()
        point_classes = num_image_tokens + int(self.config.no_more_points_class)
        lang_inputs = {
            "input_ids": input_ids,
            "vision_embeds": torch.zeros((batch_size, num_image_tokens, text_dim), dtype=model_dtype),
            "vision_embeds_vit_features": torch.zeros(
                (batch_size, num_image_tokens, pool_dim, vision_dim), dtype=model_dtype
            ),
            "vision_embeds_vit_mask": torch.ones(
                (batch_size, num_image_tokens, pool_dim), dtype=model_dtype
            ),
            "vision_embeds_subpatch_k": torch.zeros(
                (batch_size, num_image_tokens, pool_dim, point_dim), dtype=model_dtype
            ),
            "vision_embeds_patch_k": torch.zeros((batch_size, point_classes, point_dim), dtype=model_dtype),
            "vision_embeds_patch_mask": torch.ones((batch_size, point_classes), dtype=model_dtype),
            "vision_embeds_image_pos_ids": torch.zeros((batch_size, num_image_tokens), dtype=model_dtype),
            "vision_embeds_last_patch_id": torch.full((batch_size, 1), -1.0, dtype=model_dtype),
            "position_ids": position_ids,
            "token_type_ids": torch.zeros((batch_size, sequence_length), dtype=torch.bool),
        }
        cache_batch_size = constants.ONNX_EXPORT_EXAMPLE_FBS if continuous_batching else batch_size
        cache_shape = get_padding_shape_from_config(self.config.text_config, cache_batch_size, context_length)
        lang_inputs["past_key_values"] = [
            (
                torch.zeros(cache_shape, dtype=model_dtype),
                torch.zeros(cache_shape, dtype=model_dtype),
            )
            for _ in range(self.config.text_config.num_hidden_layers)
        ]
        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.zeros((len(comp_ctx_lengths),), dtype=torch.int64)
        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(batch_size, dtype=torch.int64).view(batch_size, 1)
        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        return {**vision_inputs, **lang_inputs}

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=getattr(self.config, "torch_dtype", torch.float32),
                shape=("vision_batch_size", "num_crops", "num_patches", "pixels_per_patch"),
            ),
            IOInfo(
                name="image_token_pooling",
                datatype=torch.int64,
                shape=("vision_batch_size", "num_image_tokens", "pool_dim"),
            ),
        ]
