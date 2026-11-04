# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""PyTorch Mllama model."""

import warnings
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.mllama.modeling_mllama import (
    MllamaConfig,
    MllamaCrossAttentionDecoderLayer,
    MllamaForCausalLM,
    MllamaForConditionalGeneration,
    MllamaModel,
    MllamaRotaryEmbedding,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextModel,
    MllamaTextSelfAttention,
    MllamaVisionModel,
    logger,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_utils import (
    _create_causal_mask,
    _prepare_aspect_ratio_attention_mask,
    _prepare_cross_attention_mask,
)
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

_MLLAMA_DEPRECATION_MSG = (
    "Support for Mllama (Llama 3.2 Vision) in QEfficient is deprecated and will be removed in a future release. "
    "Please migrate to Llama-4 (meta-llama/Llama-4-Scout-17B-16E-Instruct) which provides equivalent "
    "vision-language capabilities with continued support."
)

MAX_NUM_IMG = 1
NUM_CHANNEL = 3


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def eager_self_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=module.config.torch_dtype), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffMllamaRotaryEmbedding(MllamaRotaryEmbedding):
    """
    Copied from MllamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: MllamaConfig, device=None):
        super().__init__(config=config)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=config.torch_dtype,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def qeff_apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # Cast back to original dtype
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class QEffMllamaTextCrossAttentionSingleQPC(MllamaTextCrossAttention):
    """
    Copied from MllamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
    The only differences are:
        - add new args cache idx for the kv retention
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        # elif past_key_value is not None:
        # Fetch old cache
        key_states_old = past_key_values.layers[self.layer_idx].keys
        value_states_old = past_key_values.layers[self.layer_idx].values

        # if cross_attention_states is not None:
        # Compute new KV states
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Out-of-place Scatter new into old
        # out-of-place is important so the original tensor is not affected,
        # otherwise leads to same operations in both graphs
        indices = (torch.arange(bsz),)
        key_states_new = torch.index_put(key_states_old, indices, key_states)
        value_states_new = torch.index_put(value_states_old, indices, value_states)

        # Select old or new image KV states based on q_len
        key_states = torch.where(torch.tensor(q_len == 1), key_states_old, key_states_new)
        value_states = torch.where(torch.tensor(q_len == 1), value_states_old, value_states_new)

        # Update the image cache
        past_key_values.layers[self.layer_idx].keys = key_states
        past_key_values.layers[self.layer_idx].values = value_states

        key_states = self.k_norm(key_states)

        attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class QEffMllamaTextSelfAttention(MllamaTextSelfAttention):
    """
    Copied from MllamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
    The only differences are:
        - add new args cache idx for the kv retention
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        position_embeddings: torch.Tensor = None,
        use_cache: bool = False,
        cache_position=None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            # kv_seq_len = past_key_value.get_seq_length(self.layer_idx, cache_position)

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        if past_key_values is not None:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids,
            }
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_self_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class QEffMllamaSelfAttentionDecoderLayer(MllamaSelfAttentionDecoderLayer):
    """
    Copied from MllamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
    The only differences are:
        - add new args cache idx for the kv retention
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        sin_cached=None,
        cos_cached=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QEffMllamaTextCrossAttentionTwoQPC(MllamaTextCrossAttention):
    """
    Copied from MllamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
    The only differences are:
        - add new args cache idx for the kv retention
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if past_key_values is not None:
                cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
                if comp_ctx_lengths is not None:
                    attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                    cache_kwargs["CCL"] = attention_mask.shape[-1]
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )
        elif past_key_values is not None:
            key_states, value_states = (
                past_key_values.layers[self.layer_idx].keys,
                past_key_values.layers[self.layer_idx].values,
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        key_states = self.k_norm(key_states)

        attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class QEffMllamaCrossAttentionDecoderLayer(MllamaCrossAttentionDecoderLayer):
    """
    Copied from MllamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
    The only differences are:
        - add new args cache idx for the kv retention
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            cache_position=cache_position,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        return hidden_states


class QEffMllamaVisionModel(MllamaVisionModel):
    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values.to(self.dtype).to(self.device))
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (
            0,
            0,
            0,
            num_padding_patches,
        )  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state = output.last_hidden_state

        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim
        )
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state = global_output.last_hidden_state

        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = [output.last_hidden_state for _ in self.intermediate_layers_indices]
        intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )
        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
        )


class QEffMllamaTextModel(MllamaTextModel):
    """
    Copied from MllamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
    The only differences are:
        - add new args cache idx for the kv retention
    """

    def __qeff_init__(self):
        self.rotary_emb = QEffMllamaRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.FloatTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        # embed positions
        hidden_states = inputs_embeds
        sin = self.sin_cached[position_ids].unsqueeze(1)
        cos = self.cos_cached[position_ids].unsqueeze(1)

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have cross attention states
            # or cached cross attention states.
            is_cross_attention_layer = idx in self.cross_attention_layers
            is_cross_attention_cache_empty = past_key_values is None or (
                past_key_values is not None and past_key_values.get_seq_length(idx) == 0
            )

            # original modeling to check the is_cross_attention_layers
            # if is_cross_attention_layer and cross_attention_states is None and is_cross_attention_cache_empty:
            #     continue

            # TODO: vbaddi: since past_key_values are retained from previous states, the condition for is_cross_attention_cache_empty is False
            # so explicitly making it true in order to skip the cross attention for language model
            # comment once there is vision and cross attention support
            if is_cross_attention_layer and cross_attention_states is None and is_cross_attention_cache_empty:
                continue

            hidden_states = decoder_layer(
                hidden_states,
                cross_attention_states=cross_attention_states,
                cross_attention_mask=cross_attention_mask,
                attention_mask=causal_mask,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                use_cache=use_cache,
                cache_position=cache_position,
                sin_cached=sin,
                cos_cached=cos,
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffMllamaForCausalLM(MllamaForCausalLM):
    """
    Copied from MllamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
    The only differences are:
        - add new args cache idx for the kv retention
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(_MLLAMA_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            batch_index=batch_index,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QEffMllamaVisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.cross_attention_layers = self.model.config.get_text_config().cross_attention_layers

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.vision_model.transformer.layers[0].__class__}

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor]]:
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
        )
        cross_attention_states = vision_outputs[0]
        cross_attention_states = self.model.multi_modal_projector(cross_attention_states).reshape(
            -1, cross_attention_states.shape[-2], self.model.hidden_size
        )

        bsz = pixel_values.shape[0]
        outputs = []
        for i in self.cross_attention_layers:
            cross_attn = self.model.language_model.layers[i].cross_attn
            key_states = cross_attn.k_proj(cross_attention_states)
            value_states = cross_attn.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, cross_attn.num_key_value_heads, cross_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, cross_attn.num_key_value_heads, cross_attn.head_dim).transpose(
                1, 2
            )
            outputs.append((key_states, value_states))
        return outputs


class QEffMllamaModel(MllamaModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_idx: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
            )
            cross_attention_states = vision_outputs.last_hidden_state
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QEffMllamaForConditionalGeneration(MllamaForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        warnings.warn(_MLLAMA_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def get_qeff_vision_encoder(self):
        return QEffMllamaVisionEncoder(self)

    def get_qeff_language_decoder(self):
        return self

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QEffMllamaSelfAttentionDecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_idx: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            aspect_ratio_mask=aspect_ratio_mask,
            aspect_ratio_ids=aspect_ratio_ids,
            cross_attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()
        return logits, image_idx, outputs.past_key_values, pixel_values

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        if continuous_batching:
            raise NotImplementedError("Continuous batching is not supported for Mllama.")

        BS = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = kwargs.get("prefill_seq_len")
        if seq_len is None:
            seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        SEQ_LEN = int(seq_len)
        CTX_LEN = constants.ONNX_EXPORT_CTX_LEN

        txt_cfg = self.config.get_text_config()
        num_hidden_layers = txt_cfg.num_hidden_layers
        cross_attention_layers = txt_cfg.cross_attention_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = txt_cfg.hidden_size // txt_cfg.num_attention_heads

        vis_cfg = self.config.vision_config
        num_patches = (vis_cfg.image_size // vis_cfg.patch_size) ** 2 + 1
        image_tokens_len = vis_cfg.max_num_tiles * num_patches

        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 448)
            max_num_img_tiles = getattr(vis_cfg, "max_num_tiles", 4)
        else:
            img_size = 448
            max_num_img_tiles = 4

        # vision inputs
        vision_inputs = {
            "pixel_values": torch.zeros(
                (BS, MAX_NUM_IMG, max_num_img_tiles, NUM_CHANNEL, img_size, img_size),
                dtype=self.config.torch_dtype,
            ),
            "aspect_ratio_ids": torch.ones((BS, MAX_NUM_IMG), dtype=torch.int64),
            "aspect_ratio_mask": torch.ones((BS, MAX_NUM_IMG, max_num_img_tiles), dtype=torch.int64),
        }

        # lang_inputs
        lang_inputs = {
            "input_ids": torch.zeros((BS, SEQ_LEN), dtype=torch.int64),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "cross_attention_mask": torch.zeros((BS, SEQ_LEN, MAX_NUM_IMG, max_num_img_tiles), dtype=torch.int64),
            "attention_mask": torch.ones((BS, SEQ_LEN), dtype=torch.int64),
        }

        lang_inputs["position_ids"] = torch.where(
            lang_inputs.pop("attention_mask") == 1,
            torch.arange(lang_inputs["input_ids"].shape[1]).view(1, -1),
            -1,
        )

        lang_inputs["past_key_values"] = QEffDynamicCache()
        lang_inputs["past_key_values"].append_new_layers(num_hidden_layers - 1)

        for i in range(num_hidden_layers):
            if i in cross_attention_layers:
                idx = cross_attention_layers.index(i)
                assert idx == ((i - 3) // 5), f"{i}, {(i - 3) // 5}"
                lang_inputs["past_key_values"].layers[i].keys = torch.zeros(
                    1,
                    num_key_value_heads,
                    image_tokens_len,
                    head_dim,
                    dtype=self.config.torch_dtype,
                )
                lang_inputs["past_key_values"].layers[i].values = torch.zeros(
                    1,
                    num_key_value_heads,
                    image_tokens_len,
                    head_dim,
                    dtype=self.config.torch_dtype,
                )
            else:
                lang_inputs["past_key_values"].layers[i].keys = torch.zeros(
                    1, num_key_value_heads, CTX_LEN, head_dim, dtype=self.config.torch_dtype
                )
                lang_inputs["past_key_values"].layers[i].values = torch.zeros(
                    1, num_key_value_heads, CTX_LEN, head_dim, dtype=self.config.torch_dtype
                )

        lang_inputs["past_key_values"] = lang_inputs["past_key_values"].to_legacy_cache()
        lang_inputs["position_ids"] = torch.full(lang_inputs["position_ids"].shape, CTX_LEN - 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        inputs = {}

        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        vision_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        vis_cfg = self.config.vision_config
        max_num_images = compiler_options.pop("max_num_images", 1)
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else 128
        if img_size is None and hasattr(vis_cfg, "image_size"):
            img_size = getattr(vis_cfg, "image_size")
        elif img_size is None:
            img_size = 448
            logger.warning("Setting `img_size=448` as it was neither passed nor found in vision_config")

        vision = [{"batch_size": batch_size, "max_num_images": max_num_images, "img_size": img_size}]

        if comp_ctx_lengths_prefill is not None:
            lang = []

            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang.append(
                    {
                        "batch_size": batch_size,
                        "seq_len": prefill_seq_len,
                        "ctx_len": ctx_len,
                        "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                        "max_num_images": max_num_images,
                        "img_size": img_size,
                    }
                )

            # Remaining elements use comp_ctx_lengths[1:] in a loop
            for i in range(0, len(comp_ctx_lengths_decode)):
                lang.append(
                    {
                        "batch_size": batch_size,
                        "seq_len": "1",
                        "ctx_len": ctx_len,
                        "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                        "max_num_images": max_num_images,
                        "img_size": img_size,
                    }
                )

        else:
            lang = [
                {
                    "batch_size": batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "max_num_images": max_num_images,
                    "img_size": img_size,
                },
                {
                    "batch_size": batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "max_num_images": max_num_images,
                    "img_size": img_size,
                },
            ]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        if continuous_batching:
            raise NotImplementedError("Continuous batching is not supported for Mllama.")

        txt_cfg = self.config.get_text_config()
        num_hidden_layers = txt_cfg.num_hidden_layers
        cross_attention_layers = txt_cfg.cross_attention_layers

        vision_dynamic_axes = {
            "pixel_values": {0: "batch_size", 1: "max_num_images", 4: "img_size", 5: "img_size"},
            "aspect_ratio_ids": {0: "batch_size", 1: "max_num_images"},
            "aspect_ratio_mask": {0: "batch_size", 1: "max_num_images"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "cross_attention_mask": {0: "batch_size", 1: "seq_len", 2: "max_num_images"},
        }

        for i in range(num_hidden_layers):
            if i in cross_attention_layers:
                lang_dynamic_axes[f"past_key.{i}"] = {0: "batch_size"}
                lang_dynamic_axes[f"past_value.{i}"] = {0: "batch_size"}
            else:
                lang_dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
                lang_dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        txt_cfg = self.config.get_text_config()
        num_hidden_layers = txt_cfg.num_hidden_layers

        vision_output_names = []
        for i in self.config.text_config.cross_attention_layers:
            vision_output_names.append(f"past_key.{i}")
            vision_output_names.append(f"past_value.{i}")
        lang_output_names = [
            "logits",
            *[f"past_{kv}.{i}_RetainedState" for i in range(num_hidden_layers) for kv in ["key", "value"]],
        ]
        if not kv_offload:
            lang_output_names.append("pixel_values_RetainedState")

        output_names = {}
        lang_output_names.insert(1, "image_idx_output")
        if kv_offload:
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            return lang_output_names
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(
                name="pixel_values",
                datatype=self.config.torch_dtype,
                shape=("batch_size", "max_num_images", 4, 3, "img_size", "img_size"),
            ),
            IOInfo(name="aspect_ratio_ids", datatype=torch.int64, shape=("batch_size", "max_num_images")),
            IOInfo(name="aspect_ratio_mask", datatype=torch.int64, shape=("batch_size", "max_num_images", 4)),
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="cross_attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len", "max_num_images", 4)
            ),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
        ]
