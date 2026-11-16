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

The fix: override CLIPTextModel.forward to skip create_causal_mask's tracing-
incompatible construction while preserving the caller-provided padding mask.
When no padding mask is provided, CLIP still relies on causal attention via
is_causal=True.
"""

from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModel


def _create_trace_safe_causal_mask(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None

    if attention_mask.ndim == 4:
        return attention_mask

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2D or 4D")

    attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
    attn_implementation = getattr(config, "_attn_implementation", None)
    if isinstance(attn_implementation, str) and attn_implementation.startswith("flash_attention"):
        return attention_mask

    query_length = inputs_embeds.shape[1]
    key_value_length = attention_mask.shape[-1]
    query_positions = torch.arange(query_length, device=inputs_embeds.device)
    key_positions = torch.arange(key_value_length, device=inputs_embeds.device)
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & attention_mask[:, None, None, :]

    if attn_implementation == "eager":
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        return torch.where(
            combined_mask, torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype), min_dtype
        )

    return combined_mask


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
        # does q_length[0].to(device) on a 0-dim tensor. Preserve the caller-provided padding
        # mask and combine it with causality without using the tracing-incompatible helper.
        attention_mask = _create_trace_safe_causal_mask(self.config, hidden_states, attention_mask)
        kwargs.pop("is_causal", None)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
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
