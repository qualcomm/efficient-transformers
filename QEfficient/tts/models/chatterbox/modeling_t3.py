# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from torch import nn

from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer


class QEffChatterboxT3DecoderWrapper(nn.Module):
    """T3 backbone for export: Llama layers driven by ``inputs_embeds`` plus the speech-token head.

    Text, speaker and learned position embeddings are summed on the host, so the graph takes embeddings, not ids.
    Returns float32 speech logits for the last valid position of every row and the retained KV cache.
    """

    def __init__(self, t3: nn.Module):
        super().__init__()
        self.model = t3.tfmr
        self.speech_head = t3.speech_head
        self.config = t3.cfg

    def get_submodules_for_export(self) -> set:
        return {QEffLlamaDecoderLayer}

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, past_key_values):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        # Cast to INT32 to avoid issue while running in ONNXRT
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        return self.speech_head(hidden_states).float(), outputs.past_key_values
