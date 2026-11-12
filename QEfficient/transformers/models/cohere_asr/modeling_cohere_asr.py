# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEff CohereAsr wrapper.

CohereAsr is an encoder-decoder ASR model: a `ParakeetEncoder`
(fast-conformer, bidirectional, no KV cache) feeding a causal, KV-cached text
decoder that cross-attends to the fixed encoder output.

The only differences from the upstream `CohereAsrDecoder`/`CohereAsrCrossAttention`
are:
- `create_causal_mask`/`create_bidirectional_mask` (from `transformers.masking_utils`)
  crash during ONNX export tracing (`IndexError: tuple index out of range` in
  `sdpa_mask`, since both dispatch through it). Replaced with QEfficient's
  ONNX-safe `_create_causal_mask` and `_prepare_4d_attention_mask`.
- self-attention cache update passes `position_ids` via `cache_kwargs`, as
  required by `QEffDynamicLayer.update()`.
- cross-attention KV reuse is rewritten branch-free with `torch.where`/
  `torch.index_put`, gated on `input_features.shape[2] == 1` to select the
  encoder and decoder export specializations,
  instead of the Python-bool `past_key_values.is_updated` dict lookup that
  cannot be traced into a single ONNX graph shared by both specializations.
- masking uses QEfficient's `torch.where`-based fp16-safe convention instead
  of HF's additive `attn_weights + attention_mask`.
- `feature_lengths` is a per-batch int64 QPC input derived from the processor's
  attention mask. The processor excludes its final convolution-boundary feature
  frame, so the forward pass uses a clamped `feature_lengths + 1` encoder mask,
  then derives the decoder cross-attention mask with Parakeet's repeated
  convolution output-length calculation.
"""

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.cohere_asr.modeling_cohere_asr import (
    CohereAsrCrossAttention,
    CohereAsrDecoder,
    CohereAsrDecoderLayer,
    CohereAsrForConditionalGeneration,
    CohereAsrSelfAttention,
    repeat_kv,
)
from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderAttention, ParakeetEncoderSubsamplingConv2D

from QEfficient.transformers.cache_utils import QEffEncoderDecoderCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE, ONNX_EXPORT_EXAMPLE_SEQ_LEN


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Support both bool masks (causal, from _create_causal_mask) and float additive
        # masks (cross-attention, from _prepare_4d_attention_mask which uses 0.0 = attend,
        # large-negative = mask). Convert float masks via < -1.0 threshold to bool.
        bool_mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask < -1.0
        attn_weights = torch.where(
            bool_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffParakeetEncoderAttention(ParakeetEncoderAttention):
    """Prevents fully masked encoder query rows from propagating NaNs through convolution."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        attn_output, attn_weights = super().forward(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        if attention_mask is not None:
            valid_query_rows = attention_mask.any(dim=-1).transpose(1, 2)
            attn_output = torch.where(valid_query_rows, attn_output, torch.zeros_like(attn_output))
        return attn_output, attn_weights


class QEffParakeetEncoderSubsamplingConv2D(ParakeetEncoderSubsamplingConv2D):
    """Keeps the final subsampling projection bias out of masked encoder rows."""

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden_states = super().forward(input_features, attention_mask=attention_mask)
        if attention_mask is None:
            return hidden_states

        output_lengths = attention_mask.sum(-1)
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                output_lengths = self._get_output_length(output_lengths, layer)
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        output_mask = positions.unsqueeze(0) < output_lengths.unsqueeze(1)
        return torch.where(output_mask.unsqueeze(-1), hidden_states, torch.zeros_like(hidden_states))


class QEffCohereAsrSelfAttention(CohereAsrSelfAttention):
    """
    Copied from CohereAsrSelfAttention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only differences are:
    - pass `position_ids` via `cache_kwargs` to `past_key_values.update`, as required by `QEffDynamicLayer`
    - use QEfficient's `eager_attention_forward` (torch.where-based fp16-safe masking)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            self_attention_cache = past_key_values.self_attention_cache
            cache_kwargs = {"position_ids": position_ids}
            key_states, value_states = self_attention_cache.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffCohereAsrCrossAttention(CohereAsrCrossAttention):
    """
    Copied from CohereAsrCrossAttention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only differences are:
    - the encoder K/V cache-or-recompute decision is rewritten branch-free with
      torch.where/torch.index_put (gated on `input_features.shape[2] == 1`) instead
      of the Python-bool `past_key_values.is_updated` dict lookup, so both the
      "Encoder" and "Decode" compile specializations trace to the same ONNX graph
    - use QEfficient's `eager_attention_forward` (torch.where-based fp16-safe masking)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        input_features: torch.Tensor | None = None,
        **kwargs,
    ):
        bsz, tgt_len = hidden_states.shape[:-1]

        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        # Keep src_len symbolic at trace time so the view remains compatible
        # with Cohere's encoder cross-attention cache specialization.
        # A fixed src_len here would produce a Reshape with a concrete output shape that
        # conflicts with the cross-cache shape (encoder_ctx_len) during qaic-compile.
        kv_input_shape = (bsz, -1, self.config.num_key_value_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)

        if past_key_values is not None:
            # past_key_values here is already the cross_attention_cache (QEffDynamicCache),
            # split off at the decoder-layer level for Cohere's combined
            # encoder-decoder cache.
            # __getitem__ returns (keys_tensor, values_tensor) with traceable tensor identity.
            key_states_old = past_key_values[self.layer_idx][0]
            value_states_old = past_key_values[self.layer_idx][1]

            key_states_computed = self.k_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)
            value_states_computed = self.v_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)

            indices = (torch.arange(bsz),)
            key_cache_updated = torch.index_put(key_states_old, indices, key_states_computed)
            value_cache_updated = torch.index_put(value_states_old, indices, value_states_computed)
            key_states_cached, value_states_cached = key_states_old, value_states_old
            key_states_updated, value_states_updated = key_cache_updated, value_cache_updated

            # Select cache (Decode) or freshly written (Encode) based on the dummy
            # input_features shape used by compiler specializations.
            is_decode = input_features.shape[2] == torch.tensor(1)
            key_states = torch.where(is_decode, key_states_cached, key_states_updated)
            value_states = torch.where(is_decode, value_states_cached, value_states_updated)

            past_key_values.layers[self.layer_idx].keys = torch.where(is_decode, key_states_old, key_cache_updated)
            past_key_values.layers[self.layer_idx].values = torch.where(
                is_decode, value_states_old, value_cache_updated
            )
        else:
            key_states = self.k_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)
            value_states = self.v_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffCohereAsrDecoderLayer(CohereAsrDecoderLayer):
    """
    Copied from CohereAsrDecoderLayer: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only difference is threading `input_features` through to the cross-attention block.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        input_features: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            # Split the cross-attention sub-cache out of Cohere's combined
            # encoder-decoder cache before passing it to encoder_attn. This
            # lets the cross-attention block index its retained-state tensors
            # directly.
            cross_attn_past_key_value = past_key_values.cross_attention_cache if past_key_values is not None else None
            hidden_states, _ = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=cross_attn_past_key_value,
                input_features=input_features,
            )
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffCohereAsrDecoder(CohereAsrDecoder):
    """
    Copied from CohereAsrDecoder: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only differences are:
    - `create_causal_mask` -> QEfficient's ONNX-traceable `_create_causal_mask`
    - `create_bidirectional_mask` -> `_prepare_4d_attention_mask`, both of which use
      only standard tensor ops, unlike `create_causal_mask`/`create_bidirectional_mask`
      which dispatch through `sdpa_mask` and crash with
      `IndexError: tuple index out of range` while tracing to ONNX
    - threads `input_features` through to the decoder layers for cross-attention KV reuse
    """

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        encoder_hidden_states = self.proj(encoder_hidden_states)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (use_cache or past_key_values is not None) and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffEncoderDecoderCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        pos_emb = self.pos_emb(position_ids.squeeze(0))
        inputs_embeds = self.embedding_layernorm(inputs_embeds + pos_emb)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)
        encoder_attention_mask = (
            _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=inputs_embeds.shape[1])
            if encoder_attention_mask is not None
            else None
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                input_features=input_features,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        next_cache = past_key_values if (use_cache or past_key_values is not None) else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class QEffCohereAsrForConditionalGeneration(CohereAsrForConditionalGeneration):
    """
    Copied from CohereAsrForConditionalGeneration: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only differences are:
    - added get_dummy_inputs, get_onnx_dynamic_axes, get_output_names, get_specializations,
      get_inputs_info, get_submodules_for_export for AutoModel export
    - changed forward inputs decoder_input_ids/decoder_position_ids to input_ids/position_ids
    - transposes `input_features` from the QPC's mel-bins-major
      `(batch, num_mel_bins, feature_len)` contract to Parakeet's native
      `(batch, seq_len, num_mel_bins)` layout
    - adds `feature_lengths` (int64 per batch item) as a new model input derived from
      the processor attention mask. It rebuilds Parakeet's frame mask before encoding and
      its output-length mask before decoder cross-attention.
    """

    def get_submodules_for_export(self) -> type[nn.Module]:
        return {self.model.encoder.layers[0].__class__, QEffCohereAsrDecoderLayer}

    def _get_encoder_output_lengths(self, feature_lengths: torch.LongTensor) -> torch.LongTensor:
        output_lengths = feature_lengths
        for layer in self.model.encoder.subsampling.layers:
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                padding = layer.padding[0]
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                output_lengths = (output_lengths + 2 * padding - kernel_size) // stride + 1
        return output_lengths

    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
        past_key_values: EncoderDecoderCache | tuple[torch.FloatTensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        feature_lengths: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | Seq2SeqLMOutput:
        # Parakeet's native input_features is (batch, seq_len, num_mel_bins); the runtime
        # boundary (QEFFAutoModelForSpeechSeq2Seq) presents mel-bins-major
        # (batch, num_mel_bins, feature_len), matching Whisper's Conv1d convention.
        encoder_input_features = input_features.transpose(1, 2) if input_features is not None else None
        encoder_output_feature_lengths = feature_lengths + 1 if feature_lengths is not None else None

        if feature_lengths is not None:
            encoder_input_feature_lengths = torch.minimum(
                encoder_output_feature_lengths,
                torch.full_like(feature_lengths, input_features.shape[2]),
            )
            positions = torch.arange(input_features.shape[2], device=input_features.device)
            attention_mask = positions.unsqueeze(0) < encoder_input_feature_lengths.unsqueeze(1)

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(encoder_input_features, attention_mask=attention_mask)

        encoder_hidden_states = encoder_outputs.last_hidden_state
        if encoder_output_feature_lengths is not None:
            positions = torch.arange(encoder_hidden_states.shape[1], device=encoder_hidden_states.device)
            encoder_output_lengths = self._get_encoder_output_lengths(encoder_output_feature_lengths)
            valid_encoder_output_mask = positions.unsqueeze(0) < encoder_output_lengths.unsqueeze(1)
            encoder_hidden_states = torch.where(
                valid_encoder_output_mask.unsqueeze(-1), encoder_hidden_states, torch.zeros_like(encoder_hidden_states)
            )

        # Pad encoder output to the cross-KV cache length so QAIC ScatterND shape checks
        # pass in every compile specialization.
        #
        # CohereAsr's Fast Conformer emits ceil(feature_len / subsampling_factor) frames, so
        # the Decode spec (feature_len=1 → 1 frame) produces a cross-KV tensor shorter than
        # the cache, causing qaic-compile to reject Reshape_3's shape in the ScatterND write path.
        # encoder_ctx_len is read from the cross-KV cache shape when available so it matches
        # the specialization used at compile time (438 for 35s, 312 for 25s, 625 for 50s).
        # Fallback to config only when past_key_values is absent (non-QEff inference path).
        if (
            past_key_values is not None
            and isinstance(past_key_values, (list, tuple))
            and len(past_key_values) > 0
            and isinstance(past_key_values[0], (list, tuple))
            and len(past_key_values[0]) >= 3
        ):
            encoder_ctx_len = past_key_values[0][2].shape[2]
        else:
            encoder_ctx_len = (
                self.config.encoder_config.max_position_embeddings // self.config.encoder_config.subsampling_factor
            )
        pad_size = encoder_ctx_len - encoder_hidden_states.shape[1]
        if pad_size < 0:
            raise ValueError(
                f"Encoder output length {encoder_hidden_states.shape[1]} exceeds the configured "
                f"encoder context length {encoder_ctx_len}. Increase `encoder_ctx_len` when compiling."
            )
        padding = encoder_hidden_states.new_zeros(
            encoder_hidden_states.shape[0], pad_size, encoder_hidden_states.shape[2]
        )
        encoder_hidden_states = torch.cat([encoder_hidden_states, padding], dim=1)

        if encoder_output_feature_lengths is not None:
            positions = torch.arange(encoder_ctx_len, device=encoder_hidden_states.device)
            encoder_output_lengths = self._get_encoder_output_lengths(encoder_output_feature_lengths)
            enc_mask = (positions.unsqueeze(0) < encoder_output_lengths.unsqueeze(1)).long()
        else:
            enc_mask = getattr(encoder_outputs, "attention_mask", None)

        decoder_outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=enc_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            input_features=input_features,
        )
        logits = self.proj_out(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def get_dummy_inputs(self, **kwargs):
        bs = int(kwargs.get("batch_size", 1))
        seq_len = int(kwargs.get("prefill_seq_len", ONNX_EXPORT_EXAMPLE_SEQ_LEN))
        # encoder_ctx_len is the encoder OUTPUT sequence length (after subsampling).
        # encoder_config.max_position_embeddings is the INPUT length (max audio frames).
        # Divide by subsampling_factor to get the output length used for cross-KV cache sizing.
        subsampling_factor = self.config.encoder_config.subsampling_factor
        encoder_ctx_len = kwargs.get("encoder_ctx_len") or (
            self.config.encoder_config.max_position_embeddings // subsampling_factor
        )
        encoder_feature_count = self.config.encoder_config.num_mel_bins
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_layers = self.config.num_hidden_layers
        model_dtype = self.config.torch_dtype

        full_feature_len = encoder_ctx_len * subsampling_factor
        inputs = {
            # Use the full encoder input length (encoder_ctx_len * subsampling_factor) so
            # that the encoder output has exactly encoder_ctx_len frames at trace time.
            # This is necessary for the ScatterND / Reshape_3 in cross-attention to get
            # a matching shape from the cache (past_key_cross.{i}).  feature_len=1 would
            # produce a 1-frame encoder output, but the zero-pad in forward() corrects
            # this to encoder_ctx_len before the cross-attention, so both specializations
            # receive an encoder_ctx_len-frame encoder output.  Note: we still trace with
            # the full length so the Where condition (input_features.shape[2] == 1)
            # evaluates to False at trace time, keeping both branches in the ONNX.
            "input_features": torch.zeros((bs, encoder_feature_count, full_feature_len), dtype=model_dtype),
            "feature_lengths": torch.full((bs,), full_feature_len, dtype=torch.int64),
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "past_key_values": [[] for _ in range(num_layers)],
        }
        kv_cache_shape = (bs, num_key_value_heads, seq_len, head_dim)
        kv_cross_cache_shape = (bs, num_key_value_heads, encoder_ctx_len, head_dim)

        for i in range(num_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    inputs["past_key_values"][i].append(
                        torch.zeros(kv_cache_shape if self_cross == "self" else kv_cross_cache_shape, dtype=model_dtype)
                    )

        return inputs

    def get_export_hash_params(self, **kwargs):
        batch_size = int(kwargs.get("batch_size", 1))
        encoder_ctx_len = kwargs.get("encoder_ctx_len") or (
            self.config.encoder_config.max_position_embeddings // self.config.encoder_config.subsampling_factor
        )
        return {
            "batch_size": batch_size,
            "encoder_ctx_len": int(encoder_ctx_len),
        }

    def prepare_qpc_generation_inputs(self, inputs, input_features_shape):
        inputs = dict(inputs)
        if "feature_lengths" not in inputs:
            lengths = inputs.get("length")
            if lengths is not None:
                inputs["feature_lengths"] = lengths.to(dtype=torch.int64)
            else:
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    raise RuntimeError("Cohere ASR requires `length` or an attention mask to derive `feature_lengths`")
                inputs["feature_lengths"] = attention_mask.sum(dim=-1, dtype=torch.int64)

        input_features = inputs.get("input_features")
        if not isinstance(input_features, torch.Tensor) or input_features.ndim != 3:
            raise RuntimeError("Cohere ASR `input_features` must be a rank-3 torch.Tensor")

        if input_features.shape[1] != input_features_shape[1] and input_features.shape[2] == input_features_shape[1]:
            input_features = input_features.transpose(1, 2)
        if input_features.shape[:2] != input_features_shape[:2]:
            raise RuntimeError(
                f"input_features must have batch/mel dimensions {input_features_shape[:2]}, "
                f"got {tuple(input_features.shape[:2])}"
            )
        feature_padding = input_features_shape[2] - input_features.shape[2]
        if feature_padding < 0:
            raise RuntimeError(
                f"input_features length {input_features.shape[2]} exceeds compiled length {input_features_shape[2]}"
            )
        if feature_padding:
            input_features = nn.functional.pad(input_features, (0, feature_padding))
        inputs["input_features"] = input_features
        return inputs

    def get_specializations(self, batch_size: int, encoder_ctx_len, ctx_len, **compiler_options):
        subsampling_factor = self.config.encoder_config.subsampling_factor
        if encoder_ctx_len is None:
            # encoder_ctx_len = encoder OUTPUT length (after subsampling).
            # encoder_config.max_position_embeddings is the INPUT length; divide by subsampling_factor.
            encoder_ctx_len = self.config.encoder_config.max_position_embeddings // subsampling_factor
        # feature_len is the encoder INPUT length (before subsampling = encoder_ctx_len * subsampling_factor).
        feature_len = encoder_ctx_len * subsampling_factor

        encoder_specializations = {
            "_graph_name": "Encoder",
            "batch_size": batch_size,
            "seq_len": 1,
            "encoder_ctx_len": encoder_ctx_len,
            "decoder_ctx_len": ctx_len,
            "feature_len": feature_len,
        }

        decoder_specializations = {
            "_graph_name": "Decode",
            "batch_size": batch_size,
            "seq_len": 1,
            "encoder_ctx_len": encoder_ctx_len,
            "decoder_ctx_len": ctx_len,
            "feature_len": 1,  # dummy feature so torch.where knows whether to run cross attention or not
        }

        specializations = [encoder_specializations, decoder_specializations]

        return specializations, compiler_options

    def get_onnx_dynamic_axes(self):
        num_layers = self.config.num_hidden_layers

        dynamic_axes = {
            "input_features": {0: "batch_size", 2: "feature_len"},
            "feature_lengths": {0: "batch_size"},
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

        return dynamic_axes

    def get_output_names(self):
        output_names = ["logits"]
        for i in range(self.config.num_hidden_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    output_names.append(f"past_{kv}_{self_cross}.{i}_RetainedState")
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_features", datatype=torch.float32, shape=("batch_size", "num_mel_bins", "feature_len")),
            IOInfo(name="feature_lengths", datatype=torch.int64, shape=("batch_size",)),
        ]
