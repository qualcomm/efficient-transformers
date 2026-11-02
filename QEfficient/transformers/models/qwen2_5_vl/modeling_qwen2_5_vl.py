# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from qwen_vl_utils import smart_resize
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLConfig,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    repeat_kv,
    rotate_half,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.transformers.cache_utils import QEffDynamicCache

# from transformers import Qw
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


def qeff_prepare_mrope_cos_sin(cos, sin, position_ids):
    cos = cos[position_ids]
    sin = sin[position_ids]
    cos = torch.cat([cos[0, ..., 0:32], cos[1, ..., 32:80], cos[2, ..., 80:128]], dim=-1).unsqueeze(1)
    sin = torch.cat([sin[0, ..., 0:32], sin[1, ..., 32:80], sin[2, ..., 80:128]], dim=-1).unsqueeze(1)
    return cos, sin


def qeff_apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def qeff_apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class QEffQwen2_5_VLAttentionMask(nn.Module):
    """Builds the windowed attention mask used by the vision blocks.

    The mask is computed once per cu_seqlens outside of the VisionBlock loop
    and reused across every block that shares that cu_seqlens value. The output
    always has shape (1, seq_len, seq_len) and is passed to VisionAttention.
    """

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        dtype = hidden_states.dtype
        min_val = torch.finfo(dtype).min

        # Create index grids
        rows = torch.arange(seq_len).view(1, -1)
        cols = torch.arange(seq_len).view(-1, 1)

        # Prepare start and end indices
        start = cu_seqlens[:-1].view(-1, 1, 1)
        end = cu_seqlens[1:].view(-1, 1, 1)

        # Create block masks using broadcasting
        row_mask = (rows >= start) & (rows < end)
        col_mask = (cols >= start) & (cols < end)
        block_mask = row_mask & col_mask

        # Combine all blocks into one mask
        final_mask = torch.ones((seq_len, seq_len), dtype=dtype)
        final_mask[block_mask.any(dim=0)] = 0
        final_mask = torch.where(final_mask == 1.0, min_val, final_mask)

        return final_mask.unsqueeze(0)


class QEffQwen2_5_VLVisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = qeff_apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class QEffQwen2_5_VLVisionBlock(Qwen2_5_VLVisionBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class QEffQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __qeff_init__(self):
        self.attention_mask_builder = QEffQwen2_5_VLAttentionMask()

    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        bs, t, h, w = grid_thw.shape

        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)

        x_expanded = pos_ids.unsqueeze(0)
        x_expanded = x_expanded.expand(bs, -1, -1)
        pos_ids = x_expanded.reshape(-1, pos_ids.size(1))

        max_grid_size = max(grid_thw.shape)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        bs, grid_t, grid_h, grid_w = grid_thw.shape

        llm_grid_h, llm_grid_w = (
            grid_h // self.spatial_merge_size,
            grid_w // self.spatial_merge_size,
        )
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)

        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )

        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)

        x_expanded = seqlens.unsqueeze(0)
        x_expanded = x_expanded.expand(bs, -1)
        seqlens = x_expanded.reshape(-1)

        index_padded = index_padded.reshape(-1)

        mask = (index_padded == -100).to(torch.int32)

        if torch.jit.is_tracing():
            order = torch.argsort(mask)
        else:
            order = torch.argsort(mask, stable=True)

        index_new = index_padded[order]
        index_new = index_new[: index.reshape(-1).size(0)]

        step = grid_t * llm_grid_h * llm_grid_w
        batch_indices = torch.arange(bs)
        batch_indices = batch_indices.view(-1, 1)
        offsets = batch_indices * step
        window_index_tmp = index_new.unsqueeze(0) + offsets
        window_index = window_index_tmp.reshape(-1)

        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]

        cu_window_seqlens = torch.cat([torch.tensor([0], dtype=cu_seqlens_tmp.dtype), cu_seqlens_tmp])

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        cu_window_seqlens = cu_window_seqlens.to(
            device=hidden_states.device, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )

        # cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)

        hidden_states = hidden_states[window_index, :, :]

        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos().unsqueeze(-2).float(), emb.sin().unsqueeze(-2).float())

        bs, t, h, w = grid_thw.shape

        t = torch.arange(t, t + 1).squeeze().expand(bs)
        h = torch.arange(h, h + 1).squeeze().expand(bs)
        w = torch.arange(w, w + 1).squeeze().expand(bs)

        cu_seqlens = (h * w).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )

        cu_seqlens = torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype), cu_seqlens])

        full_attention_mask = self.attention_mask_builder(hidden_states, cu_seqlens)
        window_attention_mask = self.attention_mask_builder(hidden_states, cu_window_seqlens)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = full_attention_mask
            else:
                attention_mask_now = window_attention_mask

            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask_now,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class QEffQwen2_5_VLRotaryEmbedding(Qwen2_5_VLRotaryEmbedding):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__(config=config)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_kwargs: Optional[Dict[str, Any]] = None,
    layer_idx: int = None,
    past_key_value: Optional[Cache] = None,
    **kwargs,
):
    """
    default attention
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)

    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=module.config.torch_dtype), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffQwen2_5_VLAttention(Qwen2_5_VLAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        num_kv_blocks: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        use_blocking = blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)
        if use_blocking:
            attn_output, attn_weights = generic_blocked_attention_interface(
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                layer_idx=self.layer_idx,
                past_key_value=past_key_values,
                blocking_config=blocking_config,
                comp_ctx_length=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids[0],
                block_table=block_table,
                slot_id=slot_id,
                past_seen_tokens=past_seen_tokens,
            )
        else:
            key_states, value_states, attention_mask, _ = past_key_value_update(
                module=self,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids[0],
            )
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values


class QEffQwen2_5_VLDecoderLayer(Qwen2_5_VLDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        sin_cached=None,
        cos_cached=None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QEffQwen2_5_VLTextModel(Qwen2_5_VLTextModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffQwen2_5_VLRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(
            position_ids=position_ids[0], target_length=target_length, sliding_window=self.config.sliding_window
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        cos, sin = qeff_prepare_mrope_cos_sin(self.cos_cached, self.sin_cached, position_ids[1:])

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                block_table=block_table,
                slot_id=slot_id,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        return (hidden_states, past_key_values)


class QEffQwen2_5_VLModel(Qwen2_5_VLModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class QEffQwen_2_5_vl_EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.model.vision_model = self.model.visual
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.visual.blocks[0].__class__}

    def forward(self, pixel_values, image_grid_thw):
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        bs = image_grid_thw.shape[0]
        split_size = torch.floor_divide(torch.tensor(image_embeds.size(0)), bs)
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(1))

        return image_embeds


class QEffQwen_2_5_vl_DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QEffQwen2_5_VLDecoderLayer}

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)
        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids[0].shape[0]).view(-1, 1), logit_index]
        logits = self.model.lm_head(hidden_states)
        logits = logits.float()
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)

        return logits, vision_embeds, image_idx, outputs.past_key_values


class QEffQwen_2_5_vl_ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffQwen_2_5_vl_EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffQwen_2_5_vl_DecoderWrapper(self)

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        prefill_seq_len = kwargs.get("prefill_seq_len", constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        if prefill_seq_len is None:
            prefill_seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        prefill_seq_len = int(prefill_seq_len)
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, prefill_seq_len)

        vision_size = 3577
        inputs_shapes["vision_embeds"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            vision_size,
            self.model.config.text_config.hidden_size,
        )
        inputs_shapes["image_grid_thw"] = (1, 1, 98, 146)
        inputs_shapes["position_ids"] = (
            3,
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            prefill_seq_len,
        )
        inputs_shapes["pixel_values"] = (14308, 1176)
        inputs_shapes["image_idx"] = (1, 1)
        inputs_shapes["image_sizes"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 2)
        # Define inputs
        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=self.config.torch_dtype)
        vision_inputs["image_grid_thw"] = torch.zeros((inputs_shapes["image_grid_thw"]), dtype=torch.int64)

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        # Add data for KV
        kv_cache_shape = get_padding_shape_from_config(
            config=self.model.config.text_config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=prefill_seq_len,
        )

        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        qaic_config = getattr(self.model, "qaic_config", None)
        blocking_mode = qaic_config.get("blocking_mode") if qaic_config is not None else None
        if blocking_mode is not None and "paged" in blocking_mode:
            num_kv_blocks = qaic_config["num_kv_blocks"]
            batch, num_kv_heads, CL, dh = kv_cache_shape
            prefill_seq_len = kv_block_size = -(-CL // num_kv_blocks)
            total_num_kv_blocks = batch * num_kv_blocks
            kv_cache_shape = [total_num_kv_blocks, num_kv_heads, kv_block_size, dh]
            lang_inputs["block_table"] = torch.arange((bs * num_kv_blocks), dtype=torch.int64).view(bs, num_kv_blocks)
            lang_inputs["slot_id"] = torch.zeros(bs, dtype=torch.int64)
            lang_inputs["input_ids"] = torch.zeros((bs, prefill_seq_len), dtype=torch.int64)

        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=self.config.torch_dtype)
        lang_inputs["position_ids"] = (
            (torch.arange(prefill_seq_len, dtype=torch.int64).view(1, prefill_seq_len).repeat(bs, 1))
            .unsqueeze(0)
            .repeat(4, 1, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)

        lang_inputs["past_key_values"] = [[] for _ in range(self.model.config.text_config.num_hidden_layers)]
        for i in range(self.model.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=self.config.torch_dtype))

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        height: int | List[int] = None,
        width: int | List[int] = None,
        img_size=None,
        num_frames: int | List[int] = 1,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        vision_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill", None)
        comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode", None)
        # Preserve the legacy shape when callers do not request a separate vision batch.
        vision_batch_size = batch_size if vision_batch_size is None else vision_batch_size

        if height is None or width is None:
            height = constants.QWEN2_5_VL_HEIGHT
            width = constants.QWEN2_5_VL_WIDTH
            logger.warning(
                f"Setting height and width to be {height} and {width} respectively, as it was neither passed nor found in vision_config"
            )
        height = [height] if isinstance(height, int) else height
        width = [width] if isinstance(width, int) else width
        num_frames = [num_frames] * len(height) if isinstance(num_frames, int) else num_frames

        prefill_seq_len = prefill_seq_len if prefill_seq_len else 128
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        channel = 3
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = self.config.vision_config.temporal_patch_size

        IMAGE_FACTOR = constants.IMAGE_FACTOR_QWEN_2_5
        IMAGE_MIN_TOKEN_NUM = constants.IMAGE_MIN_TOKEN_NUM
        IMAGE_MAX_TOKEN_NUM = constants.IMAGE_MAX_TOKEN_NUM
        min_pixels = IMAGE_MIN_TOKEN_NUM * IMAGE_FACTOR**2
        max_pixels = IMAGE_MAX_TOKEN_NUM * IMAGE_FACTOR**2
        mm_processor_kwargs = compiler_options.pop("mm_processor_kwargs", None)
        if mm_processor_kwargs:
            min_pixels = mm_processor_kwargs.get("min_pixels", min_pixels)
            max_pixels = mm_processor_kwargs.get("max_pixels", max_pixels)

        vision = []
        max_vision_size = 0
        user_vision_size = compiler_options.pop("vision_size", None)
        if user_vision_size:
            assert user_vision_size < ctx_len, "vision_size must be less than ctx_len"
            max_vision_size = user_vision_size

        for h, w, f in zip(height, width, num_frames):
            resized_height, resized_width = smart_resize(
                height=h, width=w, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
            )
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            grid_height = grid_h * grid_w
            grid_width = patch_size * patch_size * temporal_patch_size * channel
            vision_size = grid_height // 4
            grid_height = grid_height * vision_batch_size
            if not user_vision_size:
                max_vision_size = max(max_vision_size, vision_size * f)
                assert max_vision_size < ctx_len, (
                    f"Computed vision_size of {vision_size * f} tokens "
                    f"(vision_size={vision_size}, num_frames={f}) for image resolution "
                    f"(width={w}, height={h}) must be less than ctx_len. Please adjust the image "
                    "resolution."
                )
            else:
                if vision_size * f > user_vision_size:
                    logger.warning_once(
                        f"Computed vision_size of {vision_size * f} tokens "
                        f"(vision_size={vision_size}, num_frames={f}) for image resolution "
                        f"(width={w}, height={h}) exceeds the provided "
                        f"vision_size={user_vision_size}. "
                        f"Vision embedding needs to be chunked during prefill."
                    )

            vision.append(
                {
                    "vision_batch_size": vision_batch_size,
                    "vision_size": vision_size,
                    "grid_height": grid_height,
                    "grid_width": grid_width,
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                }
            )
        if comp_ctx_lengths_prefill is not None:
            lang = []

            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "vision_size": max_vision_size,
                    "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                    "vision_batch_size": vision_batch_size,
                }

                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_prefill["batch_size"] = kv_cache_batch_size
                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size

                lang.append(lang_prefill)

            for i in range(0, len(comp_ctx_lengths_decode)):
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "vision_size": max_vision_size,
                    "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                    "vision_batch_size": vision_batch_size,
                }

                if continuous_batching:
                    lang_decode["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_decode["batch_size"] = kv_cache_batch_size

                lang.append(lang_decode)
        else:
            lang_prefill = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "vision_size": max_vision_size,
                "vision_batch_size": vision_batch_size,
            }

            if continuous_batching:
                lang_prefill["full_batch_size"] = kv_cache_batch_size
            else:
                lang_prefill["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                lang_prefill["full_batch_exec_size"] = full_batch_size

            qaic_config = getattr(self.model, "qaic_config", None)
            blocking_mode = qaic_config.get("blocking_mode") if qaic_config is not None else None
            if blocking_mode is not None and "paged" in blocking_mode:
                num_kv_blocks = self.model.qaic_config["num_kv_blocks"]
                lang_prefill["num_kv_blocks"] = num_kv_blocks
                lang_prefill["total_num_kv_blocks"] = kv_cache_batch_size * num_kv_blocks
                lang_prefill["kv_block_size"] = -(-ctx_len // num_kv_blocks)

            lang_decode = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "vision_size": max_vision_size,
                "vision_batch_size": vision_batch_size,
            }

            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size
            else:
                lang_decode["batch_size"] = kv_cache_batch_size

            qaic_config = getattr(self.model, "qaic_config", None)
            blocking_mode = qaic_config.get("blocking_mode") if qaic_config is not None else None
            if blocking_mode is not None and "paged" in blocking_mode:
                num_kv_blocks = self.model.qaic_config["num_kv_blocks"]
                lang_decode["num_kv_blocks"] = num_kv_blocks
                lang_decode["total_num_kv_blocks"] = kv_cache_batch_size * num_kv_blocks
                lang_decode["kv_block_size"] = -(-ctx_len // num_kv_blocks)

            lang = [lang_prefill, lang_decode]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            lang[0].pop("vision_size")
            lang[1].pop("vision_size")
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        # Define dynamic axes
        num_layers = self.config.text_config.num_hidden_layers

        vision_dynamic_axes = {
            "pixel_values": {0: "grid_height", 1: "grid_width"},
            "image_grid_thw": {0: "vision_batch_size", 2: "grid_h", 3: "grid_w"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {1: "batch_size", 2: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }

        pkv_dynamic_axes = {
            0: "full_batch_size" if continuous_batching else "batch_size",
            2: "ctx_len",
        }

        qaic_config = getattr(self.model, "qaic_config", None)
        blocking_mode = qaic_config.get("blocking_mode") if qaic_config is not None else None
        if blocking_mode is not None and "paged" in blocking_mode:
            lang_dynamic_axes["block_table"] = {0: "batch_size"}
            lang_dynamic_axes["slot_id"] = {0: "batch_size"}
            pkv_dynamic_axes = {
                0: "total_num_kv_blocks",
                2: "kv_block_size",
            }

        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = pkv_dynamic_axes
            lang_dynamic_axes[f"past_value.{i}"] = pkv_dynamic_axes

        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}

        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            lang_dynamic_axes.pop("vision_embeds")
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.model.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return lang_output_names
        return output_names

    def prepare_inputs_for_generation(self, inputs, prefill_seq_len=128, batch_size=1):
        input_ids_length = inputs["input_ids"].shape[1]

        inputs["position_ids"] = torch.arange(input_ids_length).view(1, 1, input_ids_length).expand(-1, batch_size, -1)

        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            # transformers>=5.5 get_rope_index expects modality token types (text=0, image=1, video=2).
            mm_token_type_ids = torch.zeros_like(inputs["input_ids"], dtype=torch.int32)
            mm_token_type_ids = mm_token_type_ids.masked_fill(inputs["input_ids"] == self.config.image_token_id, 1)
            mm_token_type_ids = mm_token_type_ids.masked_fill(inputs["input_ids"] == self.config.video_token_id, 2)

        pos_ids, rope_deltas = self.model.get_rope_index(
            input_ids=inputs["input_ids"],
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=None if "image_grid_thw" not in inputs else inputs["image_grid_thw"],
            video_grid_thw=None if "video_grid_thw" not in inputs else inputs["video_grid_thw"],
            second_per_grid_ts=None if "second_per_grid_ts" not in inputs else inputs["second_per_grid_ts"],
            attention_mask=inputs["attention_mask"],
        )
        self.model.rope_deltas = rope_deltas

        inputs["position_ids"] = torch.cat((inputs["position_ids"], pos_ids), dim=0)

        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        inputs["position_ids"] = F.pad(
            inputs["position_ids"], pad=(0, padded_len - input_ids_length), mode="constant", value=-1
        )

        inputs.pop("image_grid_thw", None)

        return inputs

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=self.config.torch_dtype,
                shape=("batch_size", 3, "image_size", "image_size"),
            ),
        ]
