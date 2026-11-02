# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEff wrappers for BERT-family encoder models — rebased for Transformers v5.5.

In TF v5.5, BertModel / RobertaModel / XLMRobertaModel gained a
`_create_attention_masks` helper that calls `create_bidirectional_mask`.
`create_bidirectional_mask` internally calls `sdpa_mask` / `eager_mask`,
which reads `inputs_embeds.shape[1]` as a 0-dim symbolic tensor during
ONNX tracing and crashes with `IndexError: tuple index out of range`.

Fix: override `_create_attention_masks` to use `_prepare_4d_attention_mask`
(standard tensor ops, fully ONNX-traceable) for the encoder (non-decoder) path.
"""

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel


class _QEffBertFamilyMixin:
    """
    Mixin that replaces `_create_attention_masks` with an ONNX-traceable version.

    `create_bidirectional_mask` (used in TF v5.5) calls `sdpa_mask`/`eager_mask`
    which reads `inputs_embeds.shape[1]` as a symbolic 0-dim tensor during tracing,
    causing `IndexError: tuple index out of range` in `sdpa_mask`.
    `_prepare_4d_attention_mask` uses only standard tensor ops and is safe.
    """

    def _create_attention_masks(
        self,
        attention_mask,
        encoder_attention_mask,
        embedding_output,
        encoder_hidden_states,
        past_key_values,
    ):
        if self.config.is_decoder:
            # Decoder path: delegate to the upstream implementation unchanged.
            return super()._create_attention_masks(
                attention_mask,
                encoder_attention_mask,
                embedding_output,
                encoder_hidden_states,
                past_key_values,
            )

        # Encoder path: use _prepare_4d_attention_mask instead of create_bidirectional_mask.
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, embedding_output.dtype)
        else:
            attention_mask = None

        if encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, embedding_output.dtype, tgt_len=embedding_output.shape[1]
            )

        return attention_mask, encoder_attention_mask


class QEffBertModel(_QEffBertFamilyMixin, BertModel):
    pass


class QEffRobertaModel(_QEffBertFamilyMixin, RobertaModel):
    pass


class QEffXLMRobertaModel(_QEffBertFamilyMixin, XLMRobertaModel):
    pass
