# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_asr.modeling_qwen3_asr import (
    Qwen3ASRAudioAttention,
    Qwen3ASRAudioEncoderLayer,
    Qwen3ASRCausalLMOutputWithPast,
    Qwen3ASREncoder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRModel,
    Qwen3ASRModelOutputWithPast,
    eager_attention_forward,
)

from QEfficient.transformers.models.qwen3.modeling_qwen3 import QEffQwen3DecoderLayer
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffQwen3ASRAudioAttention(Qwen3ASRAudioAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()
        hidden_shape = (seq_length, self.num_heads, -1)

        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose(0, 1).unsqueeze(0)
        key_states = self.k_proj(hidden_states).reshape(hidden_shape).transpose(0, 1).unsqueeze(0)
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose(0, 1).unsqueeze(0)

        chunk_len = self.config.n_window * 2
        chunk_after_cnn = (((chunk_len - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1
        window_length = (self.config.n_window_infer // chunk_len) * chunk_after_cnn
        attention_mask = self._window_attention_mask(cu_seqlens, seq_length, query_states.dtype, window_length)
        attn_output, _ = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.out_proj(attn_output)

    @staticmethod
    def _window_attention_mask(
        cu_seqlens: torch.Tensor, seq_length: int, dtype: torch.dtype, window_length: int
    ) -> torch.Tensor:
        device = cu_seqlens.device
        positions = torch.arange(seq_length, dtype=torch.int64, device=device)
        window_ids = torch.div(positions, window_length, rounding_mode="floor")
        same_window = window_ids.unsqueeze(0) == window_ids.unsqueeze(1)
        zero = torch.zeros((), dtype=dtype, device=device)
        masked = torch.full((), MIN_MASKED_ATTENTION_VALUE, dtype=dtype, device=device)
        return torch.where(same_window, zero, masked).unsqueeze(0).unsqueeze(0)


class QEffQwen3ASRAudioEncoderLayer(Qwen3ASRAudioEncoderLayer):
    pass


class QEffQwen3ASREncoder(Qwen3ASREncoder):
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        if input_features_mask is None:
            input_features_mask = torch.ones(
                input_features.shape[0], input_features.shape[2], dtype=torch.long, device=input_features.device
            )

        # The ASR deployment path supplies an aligned, all-valid feature tensor.
        # Keeping every post-CNN position avoids the dynamic NonZero/IndexSelect
        # sequence packing emitted by the upstream encoder and rejected by QAIC.
        input_features = input_features.to(self.conv2d1.weight.dtype)
        input_features = torch.where(input_features_mask[:, None, :1] < 0, -input_features, input_features)
        batch_size, num_mel_bins, padded_feature_length = input_features.shape
        chunk_len = self.n_window * 2
        if padded_feature_length % chunk_len != 0:
            raise ValueError(
                f"Qwen3ASREncoder expects `padded_feature_length` to be a multiple of "
                f"`n_window * 2` ({chunk_len}), but got {padded_feature_length}."
            )

        num_chunks = padded_feature_length // chunk_len
        chunked = (
            input_features.view(batch_size, num_mel_bins, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
        )
        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))
        total_chunks, conv_channels, freq_bins, time_steps = conv_out.size()
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous().view(total_chunks, time_steps, conv_channels * freq_bins)
        )
        conv_out += self.positional_embedding.positional_embedding[:time_steps].to(conv_out.dtype)
        hidden_states = conv_out.reshape(batch_size * num_chunks * time_steps, -1)

        # n_window_infer is eight 100-frame chunks for the released checkpoint.
        # Build fixed cumulative boundaries for the aligned input instead of
        # deriving them through mask-dependent Python packing.
        chunks_per_window = max(1, self.n_window_infer // chunk_len)
        window_lengths = []
        for batch_idx in range(batch_size):
            remaining = num_chunks
            while remaining > 0:
                window_lengths.append(min(remaining, chunks_per_window) * time_steps)
                remaining -= chunks_per_window
        cu_seqlens = torch.zeros(len(window_lengths) + 1, dtype=torch.int32, device=input_features.device)
        if window_lengths:
            cu_seqlens[1:] = torch.tensor(window_lengths, dtype=torch.int32, device=input_features.device).cumsum(0)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens, **kwargs)[0]

        hidden_states = self.ln_post(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class QEffQwen3ASRModel(Qwen3ASRModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3ASRModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        if input_features is not None and input_ids is not None:
            input_features = input_features.to(next(self.audio_tower.parameters()).dtype)
            audio_embeds = self.get_audio_features(input_features, input_features_mask, return_dict=True).pooler_output
            inputs_embeds = self._merge_audio_features(input_ids, inputs_embeds, audio_embeds)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        return Qwen3ASRModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
        )

    def _merge_audio_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        audio_embeds: torch.FloatTensor,
    ) -> torch.FloatTensor:
        selected = input_ids == self.config.audio_token_id
        flat_indices = selected.reshape(-1).to(torch.int64).cumsum(0).reshape_as(selected) - 1
        flat_indices = torch.where(flat_indices >= 0, flat_indices, torch.zeros_like(flat_indices))
        audio_features_expanded = audio_embeds.to(inputs_embeds.device)[flat_indices]
        audio_input_embeds = torch.where(selected.unsqueeze(-1), audio_features_expanded, inputs_embeds)
        return torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, audio_input_embeds)


class QEffQwen3ASRForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    def __qeff_init__(self):
        text_config = self.config.text_config
        audio_config = self.config.audio_config
        self.config.num_hidden_layers = text_config.num_hidden_layers
        self.config.num_attention_heads = text_config.num_attention_heads
        self.config.num_key_value_heads = text_config.num_key_value_heads
        self.config.hidden_size = text_config.hidden_size
        self.config.head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        self.config.vocab_size = text_config.vocab_size
        self.config.num_mel_bins = audio_config.num_mel_bins
        self.config.max_source_positions = getattr(audio_config, "n_window", 50) * 2
        self.config.decoder_start_token_id = self._scalar_config_value(
            getattr(self.config, "decoder_start_token_id", None), getattr(self.config, "pad_token_id", None)
        )
        self.config.eos_token_id = self._scalar_config_value(
            getattr(self.config, "eos_token_id", None), getattr(self.config, "pad_token_id", None)
        )
        self.config.use_cache = True

    @staticmethod
    def _scalar_config_value(value, fallback):
        if isinstance(value, (list, tuple)):
            return value[0]
        if value is None:
            return fallback
        return value

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3ASRAudioEncoderLayer, QEffQwen3DecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, Qwen3ASRCausalLMOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None and input_ids is not None:
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64, device=input_ids.device).view(1, -1)

        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        return Qwen3ASRCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def get_dummy_inputs(self, **kwargs):
        batch_size = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = int(kwargs.get("prefill_seq_len", constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN))
        feature_len = self._audio_feature_len(kwargs.get("encoder_ctx_len"))
        audio_token_count = self._audio_token_count(feature_len)
        text_config = self.config.text_config

        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.int64)
        input_ids[:, : min(audio_token_count, seq_len)] = self.config.audio_token_id
        inputs = {
            "input_features": torch.zeros(
                (batch_size, self.config.audio_config.num_mel_bins, feature_len), dtype=torch.float32
            ),
            "input_features_mask": torch.ones((batch_size, feature_len), dtype=torch.int64),
            "input_ids": input_ids,
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(batch_size, 1),
            "past_key_values": [[] for _ in range(text_config.num_hidden_layers)],
        }

        kv_cache_shape = get_padding_shape_from_config(text_config, batch_size, seq_len)
        for layer_idx in range(text_config.num_hidden_layers):
            for _ in ["key", "value"]:
                inputs["past_key_values"][layer_idx].append(torch.zeros(kv_cache_shape, dtype=torch.float32))
        return inputs

    def get_specializations(self, batch_size: int, encoder_ctx_len, ctx_len, **compiler_options):
        feature_len = self._audio_feature_len(encoder_ctx_len)
        prefill_seq_len = max(ctx_len, self._audio_token_count(feature_len))
        specializations = [
            {
                "_graph_name": "Prefill",
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "feature_len": feature_len,
            },
            {
                "_graph_name": "Decode",
                "batch_size": batch_size,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "feature_len": feature_len,
            },
        ]
        return specializations, compiler_options

    def get_onnx_dynamic_axes(self):
        dynamic_axes = {
            "input_features": {0: "batch_size", 2: "feature_len"},
            "input_features_mask": {0: "batch_size", 1: "feature_len"},
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                dynamic_axes[f"past_{kv}.{layer_idx}"] = {0: "batch_size", 2: "ctx_len"}
        return dynamic_axes

    def get_output_names(self):
        output_names = ["logits"]
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                output_names.append(f"past_{kv}.{layer_idx}_RetainedState")
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_features", datatype=torch.float32, shape=("batch_size", "num_mel_bins", "feature_len")),
            IOInfo(name="input_features_mask", datatype=torch.int64, shape=("batch_size", "feature_len")),
        ]

    def _audio_feature_len(self, encoder_ctx_len=None) -> int:
        chunk_len = self.config.audio_config.n_window * 2
        if encoder_ctx_len is None:
            return chunk_len
        return int(-(-int(encoder_ctx_len) // chunk_len) * chunk_len)

    def _audio_token_count(self, feature_len: int) -> int:
        chunk_len = self.config.audio_config.n_window * 2
        num_chunks = feature_len // chunk_len
        return num_chunks * self.config.audio_config.max_position_embeddings
