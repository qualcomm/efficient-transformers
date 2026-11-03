# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEff Wav2Vec2 wrapper — rebased for Transformers v5.5.

The only change vs the upstream model: replace `create_bidirectional_mask`
(which calls `sdpa_mask`/`eager_mask` and breaks ONNX tracing because
`inputs_embeds.shape[1]` becomes a 0-dim symbolic tensor during export)
with `_prepare_4d_attention_mask`, which uses standard tensor ops and is
fully ONNX-traceable.
"""

from typing import Optional

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
)


class QEffWav2Vec2Encoder(Wav2Vec2Encoder):
    """
    Replaces `create_bidirectional_mask` with `_prepare_4d_attention_mask` so
    that ONNX export succeeds.  `create_bidirectional_mask` internally calls
    `sdpa_mask`/`eager_mask`, which reads `inputs_embeds.shape[1]` as a
    0-dim symbolic tensor during tracing and crashes with
    `IndexError: tuple index out of range` in `sdpa_mask`.
    `_prepare_4d_attention_mask` uses only standard tensor ops and is safe.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            # _prepare_4d_attention_mask is ONNX-traceable; create_bidirectional_mask is not.
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.device)
        hidden_states = self.layer_norm(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if output_attentions and isinstance(layer_outputs, tuple):
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QEffWav2Vec2EncoderStableLayerNorm(Wav2Vec2EncoderStableLayerNorm):
    """
    Same fix as QEffWav2Vec2Encoder but for the stable-layer-norm variant
    (used when `config.do_stable_layer_norm=True`).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if output_attentions and isinstance(layer_outputs, tuple):
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
