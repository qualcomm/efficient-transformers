# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperEncoder,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperPositionalEmbedding,
    logger,
)

from QEfficient.transformers.cache_utils import QEffEncoderDecoderCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE, ONNX_EXPORT_EXAMPLE_SEQ_LEN


class QEffWhisperPositionalEmbedding(WhisperPositionalEmbedding):
    def forward(self, input_ids, past_key_values_length=0):
        return self.weight[past_key_values_length, :]


class QEffWhisperAttention(WhisperAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper
    Copied from WhisperAttention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    The only differences are:
    - attention weights computation updated to handle overflow in fp16 computation
    - add new args cache idx for the kv retention
    - manually takes is_cross_attention instead of using key_value_states to determine
    - added torch.where based on new argument input_features to determine if we compute cross attentions or use old values
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids_layer: torch.Tensor = None,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        is_cross_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        q_input_shape = (bsz, tgt_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(*q_input_shape)
        query_states = query_states.transpose(1, 2).contiguous()

        if self.is_decoder:
            if is_cross_attention and past_key_value:
                # cross_attentions
                key_states_old = past_key_value[self.layer_idx][0]
                value_states_old = past_key_value[self.layer_idx][1]
                key_states = self.k_proj(key_value_states).view(bsz, -1, self.num_heads, self.head_dim)
                value_states = self.v_proj(key_value_states).view(bsz, -1, self.num_heads, self.head_dim)
                key_states = key_states.transpose(1, 2).contiguous()
                value_states = value_states.transpose(1, 2).contiguous()
                indices = (torch.arange(bsz),)
                key_states_new = torch.index_put(key_states_old, indices, key_states)
                value_states_new = torch.index_put(value_states_old, indices, value_states)

                # Select old or new image KV states based on q_len
                key_states = torch.where(input_features.shape[2] == torch.tensor(1), key_states_old, key_states_new)
                value_states = torch.where(
                    input_features.shape[2] == torch.tensor(1), value_states_old, value_states_new
                )

                past_key_value.layers[self.layer_idx].keys = key_states
                past_key_value.layers[self.layer_idx].values = value_states
            else:
                # self attention decoder
                key_states = self.k_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
                value_states = self.v_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
                key_states = key_states.transpose(1, 2).contiguous()
                value_states = value_states.transpose(1, 2).contiguous()
                if past_key_value is not None:
                    cache_kwargs = {"position_ids": position_ids_layer}
                    if comp_ctx_lengths is not None:
                        attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                        cache_kwargs["CCL"] = attention_mask.shape[-1]
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
        else:
            # self_attention Encoder
            key_states = self.k_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()

        src_len = key_states.size(2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if tuple(attn_weights.size()) != (bsz, self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None and attention_mask.size(-1) == 0:
            attention_mask = None

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                attention_mask = None
            else:
                # updated to use torch.where, to prevent overflow in fp16 computation
                attn_weights = torch.where(
                    attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
                )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout)
        attn_output = torch.matmul(attn_weights, value_states)

        if tuple(attn_output.size()) != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return [attn_output, attn_weights]


class QEffWhisperDecoderLayer(WhisperDecoderLayer):
    """
    Copied from WhisperAttention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    The only differences are:
    - self attention and cross attention caches are explicitly picked before entering attention blocks
    - use passed argument is_encoder_decoder instead of encoder_hidden_states
    - added input_features argument to pass forward to attention
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids_layer: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        is_encoder_decoder: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size *(decoder_attention_heads,)*.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        self_attn_past_key_value = past_key_value.self_attention_cache if past_key_value is not None else None
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            position_ids_layer=position_ids_layer,
            cache_position=cache_position,
            input_features=input_features,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if is_encoder_decoder:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value.cross_attention_cache if past_key_value is not None else None
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                position_ids_layer=position_ids_layer,
                input_features=input_features,
                is_cross_attention=True,  # explicitly pass this argument, instead of figuring it out form key_value_states
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class QEffWhisperEncoder(WhisperEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].
    Copied from WhisperEncoder: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    The only differences are:
    - can run with input features of length 1, which is needed for exporting with torch.where

    Args:
        config: WhisperConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def forward(
        self,
        input_features,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`WhisperFeatureExtractor`] should be used for extracting the mel features,
                padding and conversion into a tensor of type `torch.FloatTensor`. See
                [`~WhisperFeatureExtractor.__call__`]
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            cross_key_values:
                Torch.tensor to give shape of cross_attention_values
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            hidden_states = encoder_layer(
                hidden_states,
                None,
                output_attentions=output_attentions,
            )

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states] + [encoder_states, all_attentions] if v is not None)

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            cross_attentions=None,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class QEffWhisperDecoder(WhisperDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]
    Copied form WhisperDecoder: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    The only differences are:
    - Added position_ids as argument to attention, and removed attention_mask
    - Added is_encoder_decoder input to determine whether Decoder is part of Encoder-Decoder or standalone
    - Added input_features as input to pass forward to attention

    Args:
        config: WhisperConfig
    """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        position_ids=None,
        past_key_values=None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        is_encoder_decoder: Optional[bool] = False,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        position = position_ids
        input_shape = input_ids.size()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        if use_cache or past_key_values is not None:
            if not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                past_key_values = QEffEncoderDecoderCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            cache_position = position_ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            position_ids,
            past_key_values.self_attention_cache,
            output_attentions,
        )

        # embed positions
        positions = self.embed_positions(input_ids, past_key_values_length=position)
        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_ids_layer=position_ids,
                cache_position=cache_position,
                input_features=input_features,
                is_encoder_decoder=is_encoder_decoder,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            else:
                causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_length)

        return causal_mask


class QEffWhisperModel(WhisperModel):
    """
    Transformer encoder-decoder model
    Copied from WhisperModel: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    The only differences are:
    - Added input_features as input to pass forward to attention
    """

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_position_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )[0]

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=decoder_position_ids,
            input_features=input_features,
            is_encoder_decoder=True,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )


class QEffWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    """
    Encoder-decoder model with LM head for automatic speech recognition

    The only differences are:
    - Added get_dummy_inputs, get_onnx_dynamic_axes, get_output_names for AutoModel export
    - changed forward inputs decoder_input_ids and decoder_position_ids to input_ids and position_ids
    """

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.encoder.layers[0].__class__, QEffWhisperDecoderLayer}

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_dummy_inputs(
        self,
        **kwargs,
    ):
        bs = 1
        seq_len = int(kwargs.get("prefill_seq_len", ONNX_EXPORT_EXAMPLE_SEQ_LEN))
        encoder_seq_len = self.config.max_source_positions
        encoder_feature_count = self.config.num_mel_bins
        num_key_value_heads = self.config.decoder_attention_heads
        head_dim = self.config.d_model // num_key_value_heads
        num_layers = self.config.num_hidden_layers
        # feature_len is kept static for the dynamo/weight-free export: torch.export can't certify it as dynamic, since
        # QEffWhisperEncoder's positional-embedding addition only holds at exactly 1 or exactly the
        # encoder's fixed size. Trace with the real encoder feature length so that static value is the
        # one baked into the graph; this also means torch.where(input_features.shape[2] == 1, ...) in
        # QEffWhisperAttention.forward always recomputes cross-attention rather than reusing the cache
        # (see get_specializations for the compile-time counterpart).
        feature_len = 2 * encoder_seq_len if kwargs.get("dynamo", False) else 1

        inputs = {
            "input_features": torch.zeros((bs, encoder_feature_count, feature_len), dtype=torch.float32),
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "past_key_values": [[] for _ in range(num_layers)],
        }

        kv_cache_shape = (bs, num_key_value_heads, seq_len, head_dim)
        kv_cross_cache_shape = (bs, num_key_value_heads, encoder_seq_len, head_dim)

        for i in range(num_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    inputs["past_key_values"][i].append(
                        torch.zeros(
                            kv_cache_shape if self_cross == "self" else kv_cross_cache_shape, dtype=torch.float32
                        )
                    )

        return inputs

    def get_specializations(
        self, batch_size: int, encoder_ctx_len, ctx_len, weight_free: bool = False, **compiler_options
    ):
        if encoder_ctx_len is None and hasattr(self.config, "max_source_positions"):
            encoder_ctx_len = self.config.max_source_positions
        elif encoder_ctx_len is None:
            encoder_ctx_len = 1500
            logger.warning("Setting `encoder_ctx_len=1500` as it was neither passed nor found in config")
        feature_len = encoder_ctx_len * 2

        encoder_specializations = {
            "_graph_name": "Encoder",
            "batch_size": batch_size,
            "seq_len": 1,
            "encoder_ctx_len": encoder_ctx_len,
            "decoder_ctx_len": ctx_len,
            "feature_len": feature_len,
        }

        if weight_free:
            # The weight-free/dynamo export bakes feature_len as a static axis (see
            # get_onnx_dynamic_axes), so there is no dynamic ONNX axis left for the
            # compiler to specialize into a second, feature_len=1 "Decode" graph; the single exported
            # graph only accepts input_features of this fixed feature_len, on every call.
            return [encoder_specializations], compiler_options

        decoder_specializations = {
            "_graph_name": "Decode",
            "batch_size": batch_size,
            "seq_len": 1,
            "encoder_ctx_len": encoder_ctx_len,
            "decoder_ctx_len": ctx_len,
            "feature_len": 1,  # important dummy feature so that torch.where knows whether to run cross attention or not
        }

        specializations = [encoder_specializations, decoder_specializations]

        return specializations, compiler_options

    def get_onnx_dynamic_axes(
        self,
        dynamo: bool = False,
    ):
        num_layers = self.config.num_hidden_layers

        dynamic_axes = {
            "input_features": {0: "batch_size", 2: "feature_len"},
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        pkv_self_dynamic_axes = {
            0: "batch_size",
            2: "decoder_ctx_len",
        }
        pkv_cross_dynamic_axes = {
            0: "batch_size",
            2: "encoder_ctx_len",
        }
        for i in range(num_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    dynamic_axes[f"past_{kv}_{self_cross}.{i}"] = (
                        pkv_self_dynamic_axes if self_cross == "self" else pkv_cross_dynamic_axes
                    )

        if dynamo:
            # The encoder positional embedding only supports the traced feature lengths of 1
            # (decode/cache reuse) and 2 * max_source_positions (encoder execution). A torch.export
            # Dim cannot express that discrete constraint, so keep the raw audio feature length static
            # in the Dynamo graph while preserving the dynamic axis for the legacy exporter.
            dynamic_axes["input_features"].pop(2)

        return dynamic_axes

    def get_output_names(
        self,
    ):
        output_names = ["logits"]
        for i in range(self.config.num_hidden_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    output_names.append(f"past_{kv}_{self_cross}.{i}_RetainedState")
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_features", datatype=torch.float32, shape=("batch_size", "num_mel_bins", "feature_len")),
        ]
