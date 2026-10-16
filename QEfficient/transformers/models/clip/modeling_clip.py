# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEfficient CLIP model overrides for ONNX-tracing compatibility with transformers >= 5.5.

In transformers 5.5+, CLIPTextModel.forward calls create_causal_mask(), which
internally calls sdpa_mask(). During ONNX tracing, inputs_embeds.shape[1] is a tensor
(not an int), and the backward-compat branch in sdpa_mask does q_length[0].to(device)
on a 0-dim tensor, raising "IndexError: tuple index out of range".

The fix: override CLIPTextModel.forward to skip create_causal_mask entirely and
pass attention_mask=None directly to the encoder (CLIP uses causal attention via
is_causal=True, so no explicit mask tensor is needed for export).
"""

from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModel


class QEffCLIPTextModel(CLIPTextModel):
    """
    CLIP text transformer with create_causal_mask bypassed for ONNX-tracing compatibility.

    Overrides forward() to skip the create_causal_mask() call that breaks during
    torch.onnx.export tracing when running with transformers >= 5.5.
    """

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # Skip create_causal_mask() — it breaks during ONNX tracing in transformers >= 5.5
        # because shape[1] is a tensor during tracing and sdpa_mask's backward-compat branch
        # does q_length[0].to(device) on a 0-dim tensor.
        # CLIP uses causal self-attention via is_causal=True, so no explicit mask is needed.
        kwargs.pop("is_causal", None)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            is_causal=True,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )
