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

Separately, `RobertaEmbeddings` / `XLMRobertaEmbeddings` / `NomicBertEmbeddings`
`.forward` build `token_type_ids` from a non-persistent `token_type_ids` buffer
(created on a concrete device) gathered against `position_ids`. Under
FakeTensor/meta-device tracing (e.g. dynamo + use_onnx_subfunctions,
weight-free export) `position_ids` can be a fake tensor on the `meta` device
while the buffer stays on a concrete device, so `torch.gather` fails with
"found at least two devices". The `QEff*Embeddings` classes below override
`forward` to move the buffer to `position_ids.device` before the gather;
otherwise identical to the upstream implementation.
"""

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.nomic_bert.modeling_nomic_bert import NomicBertEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings, XLMRobertaModel


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


class _QEffRobertaFamilyEmbeddingsMixin:
    """
    Shared fixed `forward` for `RobertaEmbeddings` / `XLMRobertaEmbeddings` (identical
    upstream implementations): moves the buffered `token_type_ids` to `position_ids.device`
    before the `torch.gather` call. See module docstring for the FakeTensor/meta-device
    rationale.
    """

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, self.padding_idx)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = buffered_token_type_ids.to(position_ids.device)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QEffRobertaEmbeddings(_QEffRobertaFamilyEmbeddingsMixin, RobertaEmbeddings):
    pass


class QEffXLMRobertaEmbeddings(_QEffRobertaFamilyEmbeddingsMixin, XLMRobertaEmbeddings):
    pass


class QEffNomicBertEmbeddings(NomicBertEmbeddings):
    """
    Fixes the same buffered-`token_type_ids` device mismatch as `QEffRobertaEmbeddings`,
    for `NomicBertEmbeddings` (native in Transformers v5.5, generated from
    `modular_nomic_bert.py`). Otherwise identical to the upstream implementation.
    """

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        embeddings = inputs_embeds
        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)

        input_shape = embeddings.shape[:-1]
        device = embeddings.device

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = buffered_token_type_ids.to(position_ids.device)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(*input_shape)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
