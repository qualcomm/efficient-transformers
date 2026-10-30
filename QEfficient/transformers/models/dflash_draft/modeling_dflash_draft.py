# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""PyTorch DFlash draft model (architecture-agnostic diffusion LLM draft for SpD)."""

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RotaryEmbedding,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def _create_mask(
    position_ids: torch.Tensor,
    target_length: int,
    # valid_kv_length: int,
    sliding_window: int | None = None,
    start_index: int | None = 0,
    is_causal: bool = False,
):
    """
    Args:
        position_ids: [1, target_length]
        target_length: 3 * block_size
        valid_kv_length: number of valid KV cache positions
        sliding_window: optional local attention window
        start_index: offset into KV cache (default 0)
        is_causal: restrict each query to its own position and earlier keys

    Returns:
        attention_mask: [1, 1, num_queries, target_length]
    """

    num_queries = position_ids.shape[1]  # = block_size (B)
    bsz = position_ids.shape[0]

    kv_positions = torch.arange(start_index, start_index + target_length, device=position_ids.device)
    row_max = position_ids.max(dim=-1, keepdim=True).values  # [bsz, 1]
    valid_kv_mask = kv_positions.view(1, target_length) > (start_index + row_max)  # [bsz, target_length]

    # ---- Step 2: Expand to [bsz, num_queries, target_length] ----
    attention_mask = valid_kv_mask.unsqueeze(1).expand(bsz, num_queries, target_length)

    # Full DFlash layers see the whole noise block. Local layers use a per-query
    # window, with causality selected from the draft config as in upstream DFlash.
    query_positions = (start_index + position_ids).unsqueeze(-1)
    key_positions = kv_positions.view(1, 1, target_length)
    if is_causal:
        attention_mask = attention_mask | (key_positions > query_positions)
    if sliding_window is not None:
        attention_mask = attention_mask | (key_positions <= query_positions - sliding_window)
        if not is_causal:
            attention_mask = attention_mask | (key_positions >= query_positions + sliding_window)

    # ---- Step 4: Add head dimension ----
    # Final shape: [bsz, 1, B, 3B] — broadcasts over heads in eager attention.
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask


#  Can be replaced with llama/modeling_llama.py::QEffLlamaRotaryEmbedding but keeping it following transformers ideology
class QEffQwen3RotaryEmbedding(Qwen3RotaryEmbedding):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__(config=config)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=config.torch_dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_len = q.size(-2)

    q_embed = (q * cos[..., :q_len, :]) + (rotate_half(q) * sin[..., :q_len, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def qeff_apply_rope_two_streams(q_noise, k_ctx, k_noise, cos, sin, pos_ctx, pos_noise, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    cos_t = cos[pos_ctx].unsqueeze(unsqueeze_dim)
    sin_t = sin[pos_ctx].unsqueeze(unsqueeze_dim)

    rotate_half_k_ctx = rotate_half(k_ctx)

    k_t_embed = k_ctx * cos_t + rotate_half_k_ctx * sin_t

    # ---- NOISE ----
    cos_n = cos[pos_noise].unsqueeze(unsqueeze_dim)
    sin_n = sin[pos_noise].unsqueeze(unsqueeze_dim)

    rotate_half_q_noise = rotate_half(q_noise)
    q_n_embed = q_noise * cos_n + rotate_half_q_noise * sin_n

    rotate_half_k_noise = rotate_half(k_noise)

    k_n_embed = k_noise * cos_n + rotate_half_k_noise * sin_n

    return q_n_embed, k_t_embed, k_n_embed


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
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffDFlashAttention(Qwen3Attention):
    """
    Copied from Qwen3Attention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    """

    def __qeff_init__(self):
        self.dflash_dlm = True
        layer_types = getattr(self.config, "layer_types", None)
        layer_type = layer_types[self.layer_idx] if layer_types else "full_attention"
        is_causal = getattr(self.config, "is_causal", None)
        self.is_causal = layer_type == "sliding_attention" if is_causal is None else bool(is_causal)
        self.sliding_window = self.config.sliding_window if layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids_target: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        kwargs.pop("output_attentions", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("labels", None)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)

        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k_ctx = self.k_norm(k_ctx.view(bsz, ctx_len, -1, self.head_dim)).transpose(1, 2)
        k_noise = self.k_norm(k_noise.view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)

        v_ctx = (v_ctx.view(bsz, ctx_len, -1, self.head_dim)).transpose(1, 2)
        v_noise = (v_noise.view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)

        # Assuming position_id [77,78,79,80, 75,76,-1,-1] first 4 pos id of noise next four position_id for target
        query_states, k_ctx, k_noise = qeff_apply_rope_two_streams(
            query_states, k_ctx, k_noise, cos_cached, sin_cached, position_ids_target, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids_target}
            # first write for target positon_id
            past_key_value.write_only(k_ctx, v_ctx, self.layer_idx, cache_kwargs)

            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_value.update(k_noise, v_noise, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class QEffDFlashDecoderLayer(Qwen3DecoderLayer):
    """
    Copied from Qwen3ForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    - update the hidden_states, and fix for onnx model
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor = None,
        position_ids_target: torch.LongTensor | None = None,
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
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_ids_target=position_ids_target,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QEffDFlashModel(Qwen3Model):
    """
    Copied from Qwen3Model: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    - update causal attention mask
    """

    def __qeff_init__(self):
        self.rotary_emb = QEffQwen3RotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    def forward(
        self,
        target_hidden: torch.Tensor = None,
        noise_embeds: torch.FloatTensor = None,
        position_ids_target: torch.LongTensor | None = None,
        input_ids: torch.LongTensor | None = None,
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
    ) -> tuple | BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (noise_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if input_ids is not None and noise_embeds is None:
            noise_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + noise_embeds.shape[1])
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        attention_masks = {}
        for decoder_layer in self.layers:
            attn = decoder_layer.self_attn
            mask_key = (attn.is_causal, attn.sliding_window)
            if mask_key not in attention_masks:
                attention_masks[mask_key] = _create_mask(
                    position_ids=position_ids,
                    target_length=target_length,
                    sliding_window=attn.sliding_window,
                    is_causal=attn.is_causal,
                )
        hidden_states = noise_embeds

        # Pass the whole rotary table down: `qeff_apply_rope_two_streams` gathers it once per stream
        # (`position_ids_target` for the context keys, `position_ids` for the noise keys), so
        # pre-gathering by `position_ids` here would drop the target stream. `attention_scaling` is
        # already folded into the cached buffers by `__qeff_init__`.
        sin = self.sin_cached.to(device=hidden_states.device, dtype=hidden_states.dtype)
        cos = self.cos_cached.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                target_hidden=target_hidden,
                position_ids_target=position_ids_target,
                attention_mask=attention_masks[
                    (decoder_layer.self_attn.is_causal, decoder_layer.self_attn.sliding_window)
                ],
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                sin_cached=sin,
                cos_cached=cos,
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class QEffDFlashForCausalLM(Qwen3ForCausalLM):
    """
    Copied from Qwen3ForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    - update the hidden_states, and fix for onnx model
    """

    def forward(
        self,
        target_hidden: torch.Tensor = None,
        noise_embeds: torch.FloatTensor = None,
        position_ids_target: torch.LongTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        # block_size: Optional[torch.Tensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            target_hidden=target_hidden,
            noise_embeds=noise_embeds,
            position_ids_target=position_ids_target,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            # block_size=block_size,
            output_hidden_states=output_hidden_states,
        )

        # Cast to INT32 to avoid issue while running in ONNXRT
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
