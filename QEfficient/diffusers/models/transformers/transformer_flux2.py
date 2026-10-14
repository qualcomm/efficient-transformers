# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
    Flux2SingleTransformerBlock,
    Flux2Transformer2DModel,
    Flux2TransformerBlock,
    Flux2Transformer2DModelOutput,
    _get_qkv_projections,
)

from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention, get_attention_blocking_config
from QEfficient.utils.logging_utils import logger


def qeff_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Apply rotary embeddings using real-valued cos/sin decomposition.

    Replaces diffusers' apply_rotary_emb which internally calls
    torch.view_as_complex / torch.view_as_real — ops not supported by
    ONNX opset 17.

    Args:
        x          : Query or key tensor, shape [B, S, H, D]
        freqs_cis  : Tuple of (cos, sin) each shape [S, D]

    Returns:
        Rotary-embedded tensor, same shape as x.
    """
    cos, sin = freqs_cis  # [S, D]
    cos = cos[None, :, None, :]  # [1, S, 1, D]
    sin = sin[None, :, None, :]  # [1, S, 1, D]
    cos = cos.to(x.device)
    sin = sin.to(x.device)
    B, S, H, D = x.shape
    x_real, x_imag = x.reshape(B, S, H, D // 2, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class QEffFlux2AttnProcessor(Flux2AttnProcessor):
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "QEffFlux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # RoPE — ONNX-safe real-valued implementation
        if image_rotary_emb is not None:
            query = qeff_apply_rotary_emb(query, image_rotary_emb)
            key = qeff_apply_rotary_emb(key, image_rotary_emb)

        # Blocked attention
        blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config()
        hidden_states = compute_blocked_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            blocking_mode=blocking_mode,
            head_block_size=head_block_size,
            num_kv_blocks=num_kv_blocks,
            num_q_blocks=num_q_blocks,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class QEffFlux2Attention(Flux2Attention):
    def __qeff_init__(self):
        self.processor = QEffFlux2AttnProcessor()


class QEffFlux2ParallelSelfAttnProcessor(Flux2ParallelSelfAttnProcessor):
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "QEffFlux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Fused QKV + MLP-in projection
        proj = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            proj,
            [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
            dim=-1,
        )

        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = qeff_apply_rotary_emb(query, image_rotary_emb)
            key = qeff_apply_rotary_emb(key, image_rotary_emb)

        blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config()
        hidden_states = compute_blocked_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            blocking_mode=blocking_mode,
            head_block_size=head_block_size,
            num_kv_blocks=num_kv_blocks,
            num_q_blocks=num_q_blocks,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)
        return hidden_states


class QEffFlux2ParallelSelfAttention(Flux2ParallelSelfAttention):
    def __qeff_init__(self):
        self.processor = QEffFlux2ParallelSelfAttnProcessor()


class QEffFlux2TransformerBlock(Flux2TransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(temb_mod_img, 6, dim=-1)

        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = torch.chunk(
            temb_mod_txt, 6, dim=-1
        )

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        attn_output, context_attn_output = attention_outputs

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QEffFlux2SingleTransformerBlock(Flux2SingleTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        text_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        # Concatenate text + image tokens only if encoder_hidden_states provided

        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output

        hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class QEffFlux2Transformer2DModel(Flux2Transformer2DModel):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of classes used as repeated layers for ONNX
        subfunction extraction.
        """
        return {QEffFlux2TransformerBlock, QEffFlux2SingleTransformerBlock}

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        adaln_double_img: torch.Tensor = None,
        adaln_double_txt: torch.Tensor = None,
        adaln_single: torch.Tensor = None,
        adaln_out: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Flux2Transformer2DModelOutput]:
        """
        QEff forward for Flux2Transformer2DModel.
        """
        num_txt_tokens = encoder_hidden_states.shape[1]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)

        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # ------------------------------------------------------------------
        # 3. Double-stream transformer blocks
        # ------------------------------------------------------------------
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=adaln_double_img[index_block],
                temb_mod_txt=adaln_double_txt[index_block],
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # ------------------------------------------------------------------
        # 4. Concatenate text + image for single-stream blocks
        # ------------------------------------------------------------------
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # ------------------------------------------------------------------
        # 5. Single-stream transformer blocks
        # ------------------------------------------------------------------
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=adaln_single[index_block],
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                text_seq_len=num_txt_tokens,
            )

        # Remove text tokens — keep only image tokens
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # ------------------------------------------------------------------
        # 6. Output norm + projection
        # ------------------------------------------------------------------
        hidden_states = self.norm_out(hidden_states, adaln_out)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Flux2Transformer2DModelOutput(sample=output)
