# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import math
from typing import Any

import torch
import torch.nn.functional as F
from qwen_vl_utils import smart_resize
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextAttention,
    Qwen3VLTextConfig,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionAttention,
    Qwen3VLVisionModel,
    apply_rotary_pos_emb_vision,
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
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.spd.dflash import compute_dflash_target_hidden_states
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


def _should_export_embedding_output(module) -> bool:
    for holder in (module, getattr(module, "model", None)):
        if holder is None:
            continue
        qaic_config = getattr(holder, "qaic_config", None)
        if isinstance(qaic_config, dict) and qaic_config.get("export_embedding", False):
            return True
        config = getattr(holder, "config", None)
        if config is not None and getattr(config, "export_embedding", False):
            return True
    return False


def qeff_apply_interleaved_mrope(freqs, mrope_section):
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity.
    args:
        x: (3, bs, seq_len, head_dim // 2)
        mrope_section: (3,)
    returns:
        x_t: (bs, seq_len, head_dim // 2)
    """
    freqs_t = freqs[0]  # just overwrite the first dimension T
    half_shape = freqs.shape[-1] // 2
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
        offset += half_shape
        length += half_shape
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def qeff_prepare_mrope_cos_sin(cos, sin, position_ids, mrope_section):
    cos = cos[position_ids]
    sin = sin[position_ids]
    cos = qeff_apply_interleaved_mrope(cos, mrope_section).unsqueeze(1)
    sin = qeff_apply_interleaved_mrope(sin, mrope_section).unsqueeze(1)
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


class QEffQwen3VLTextRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__(config=config)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=config.torch_dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class QEffQwen3VLVisionModel(Qwen3VLVisionModel):
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = max(grid_thw.shape)
        freq_table = self.rotary_pos_emb(torch.arange(max_hw, device=grid_thw.device))  # (max_hw, dim // 2)
        device = freq_table.device
        _bs, num_frames, height, width = grid_thw.shape
        grid_thw = (torch.tensor(grid_thw.shape, dtype=torch.int64)).unsqueeze(0)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)  # block row indices
        block_cols = torch.arange(merged_w, device=device)  # block col indices
        intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
        intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

        # Compute full-resolution positions
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        pos_ids = coords
        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        bs, t, h, w = grid_thw.shape
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        max_t = torch.tensor(self.num_grid_per_side - 1, device=h_idxs.device)

        h_idxs_ceil = torch.minimum(h_idxs_floor + 1, max_t)  # working
        w_idxs_ceil = torch.minimum(w_idxs_floor + 1, max_t)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        idx_tensor = torch.stack(indices, dim=0).to(dtype=torch.long, device=self.pos_embed.weight.device)  # [4, h*w]

        weight_tensor = torch.stack(weights, dim=0).to(
            dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        pos_embed = patch_pos_embeds[0]
        pos_embed = pos_embed.repeat(t, 1)

        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        x_expanded = patch_pos_embeds.unsqueeze(0)
        x_expanded = x_expanded.expand(bs, -1, -1)
        patch_pos_embeds = x_expanded.reshape(-1, patch_pos_embeds.size(1))
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        bs, t, h, w = grid_thw.shape

        t = torch.arange(t, t + 1).squeeze().expand(bs)
        h = torch.arange(h, h + 1).squeeze().expand(bs)
        w = torch.arange(w, w + 1).squeeze().expand(bs)

        cu_seqlens = (h * w).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype), cu_seqlens])

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


class QEffQwen3VLVisionAttention(Qwen3VLVisionAttention):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
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
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.full(
            [1, seq_length, seq_length], MIN_MASKED_ATTENTION_VALUE, device=q.device, dtype=q.dtype
        )

        # Create index grids
        seq_len = attention_mask.shape[-1]
        rows = torch.arange(seq_len).view(1, -1)
        cols = torch.arange(seq_len).view(-1, 1)

        # Prepare start and end indices
        start = cu_seqlens[:-1].view(-1, 1, 1)
        end = cu_seqlens[1:].view(-1, 1, 1)

        # Create block masks using broadcasting
        row_mask = (rows >= start) & (rows < end)
        col_mask = (cols >= start) & (cols < end)
        block_mask = row_mask & col_mask  # shape: (num_blocks, seq_len, seq_len)

        # Combine all blocks into one mask
        final_mask = torch.ones((seq_len, seq_len), dtype=self.config.dtype)
        final_mask[block_mask.any(dim=0)] = 0

        final_mask = torch.where(final_mask == 1.0, MIN_MASKED_ATTENTION_VALUE, final_mask)

        attention_mask[0] = final_mask

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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_kwargs: dict[str, Any] | None = None,
    layer_idx: int | None = None,
    past_key_value: Cache | None = None,
    **kwargs,
):
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


class QEffQwen3VLTextAttention(Qwen3VLTextAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        block_table: torch.LongTensor | None = None,
        slot_id: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)
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


class QEffQwen3VLTextDecoderLayer(Qwen3VLTextDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        block_table: torch.LongTensor | None = None,
        slot_id: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        batch_index: torch.LongTensor | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
        sin_cached=None,
        cos_cached=None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
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
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
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
            past_key_values=past_key_values,
            batch_index=batch_index,
            comp_ctx_lengths=comp_ctx_lengths,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
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


class QEffQwen3VLTextModel(Qwen3VLTextModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffQwen3VLTextRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        block_table: torch.LongTensor | None = None,
        slot_id: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if self.config.use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if getattr(self, "target_layer_ids", None):
            # DFlash target model: position_ids[0] is the flat position (causal mask / logit
            # gather); position_ids[1:] is the hard-coded `3`-way temporal/height/width mrope
            # triple consumed by the rotary embedding.
            if position_ids is None:
                flat_position_ids = cache_position.view(1, 1, -1).expand(1, inputs_embeds.shape[0], -1)
                mrope_position_ids = flat_position_ids.expand(3, inputs_embeds.shape[0], -1)
                position_ids = torch.cat((flat_position_ids, mrope_position_ids), dim=0)
            elif position_ids.dim() == 2:
                flat_position_ids = position_ids[None, ...]
                mrope_position_ids = flat_position_ids.expand(3, position_ids.shape[0], -1)
                position_ids = torch.cat((flat_position_ids, mrope_position_ids), dim=0)
        else:
            # the hard coded `3` is for temporal, height and width.
            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            elif position_ids.dim() == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(
            position_ids=position_ids[0], target_length=target_length, sliding_window=None
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids[1:])
        cos, sin = qeff_prepare_mrope_cos_sin(
            self.cos_cached, self.sin_cached, position_ids[1:], self.config.rope_scaling["mrope_section"]
        )
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        self.target_layer_ids = getattr(self, "target_layer_ids", None)
        target_hidden_list = []

        layer_idx = 0
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.target_layer_ids and layer_idx in self.target_layer_ids:
                target_hidden_list.append(hidden_states)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                block_table=block_table,
                slot_id=slot_id,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if deepstack_visual_embeds is not None and layer_idx in range(deepstack_visual_embeds.shape[0]):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
            layer_idx += 1

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        target_hidden = None
        if self.target_layer_ids:
            target_hidden = compute_dflash_target_hidden_states(target_hidden_list, self.fc, self.hidden_norm)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=target_hidden if target_hidden is not None else all_hidden_states,
            attentions=all_self_attns,
        )

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_pos_masks = visual_pos_masks.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        mixed_embeds = hidden_states + visual_embeds

        local_this = torch.where(visual_pos_masks, mixed_embeds, hidden_states)

        return local_this


class QEffQwen3VLEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.model.vision_model = self.model.visual
        self.config = model.config

    def get_submodules_for_export(self) -> type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.visual.blocks[0].__class__}

    def forward(self, pixel_values, image_grid_thw):
        image_embeds, deepstack_feature_lists = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        bs = image_grid_thw.shape[0]
        split_size = torch.floor_divide(torch.tensor(image_embeds.size(0)), bs)
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(1))
        deepstack_features = torch.stack(
            [feature.reshape(bs, split_size, feature.size(1)) for feature in deepstack_feature_lists],
            dim=0,  # new axis for "features"
        )
        return image_embeds, deepstack_features


class QEffQwen3VLDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model
        self.config = model.config

    def get_submodules_for_export(self) -> type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QEffQwen3VLTextDecoderLayer}

    def forward(
        self,
        input_ids,
        vision_embeds,
        deepstack_features,
        position_ids,
        image_idx,
        past_key_values,
        block_table: torch.LongTensor | None = None,
        slot_id: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        comp_ctx_lengths: list[int] | None = None,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        _B, _N, C = inputs_embeds.shape
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]

        num_features, bs, split_size, C = deepstack_features.shape
        x = deepstack_features.reshape(num_features, bs * split_size, C)
        deepstack_features_expanded = x[:, indices1, :]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)

        image_mask = selected.clone()

        visual_pos_masks = None
        deepstack_visual_embeds = None

        if image_mask is not None:
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_features_expanded

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)

        if getattr(self.language_model, "target_layer_ids", None):
            target_hidden = outputs.hidden_states
            logits = self.model.lm_head(outputs.last_hidden_state).float()
            predicted_token_ids = logits.argmax(dim=-1).to(torch.int32)
            return (
                predicted_token_ids,
                vision_embeds,
                deepstack_features,
                image_idx,
                target_hidden,
                outputs.past_key_values,
            )

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids[0].shape[0]).view(-1, 1), logit_index]
        logits = self.model.lm_head(hidden_states)
        if _should_export_embedding_output(self):
            return logits, vision_embeds, deepstack_features, image_idx, hidden_states, outputs.past_key_values
        return logits, vision_embeds, deepstack_features, image_idx, outputs.past_key_values


class QEffQwen3VLModel(Qwen3VLModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        block_table: torch.LongTensor | None = None,
        slot_id: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
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

        output = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class QEffQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffQwen3VLEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffQwen3VLDecoderWrapper(self)

    def forward(
        self,
        input_ids,
        position_ids,
        past_key_values,
        block_table: torch.LongTensor | None = None,
        slot_id: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_idx: torch.LongTensor | None = None,
        comp_ctx_lengths: list[int] | None = None,
        batch_index: torch.LongTensor | None = None,
        image_grid_thw=None,
    ):
        image_embeds, _deepstack_feature_lists = self.model.visual(pixel_values, grid_thw=image_grid_thw)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        _B, _N, C = inputs_embeds.shape
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = image_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        # TODO: deepstack_features are not processed for single QPC setup yet. Will do if required.
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)
        outputs = self.language_model(
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
        logits = self.lm_head(hidden_states)
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        if _should_export_embedding_output(self):
            return logits, image_embeds, image_idx, hidden_states, outputs.past_key_values
        return logits, image_embeds, image_idx, outputs.past_key_values

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: list[int] | None = None,
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
        # vision_size = 1024
        vision_size = 187
        inputs_shapes["vision_embeds"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            vision_size,
            self.model.config.vision_config.out_hidden_size,
        )
        inputs_shapes["image_grid_thw"] = (1, 1, 22, 34)
        inputs_shapes["position_ids"] = (
            3,
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            prefill_seq_len,
        )
        inputs_shapes["pixel_values"] = (748, 1536)
        inputs_shapes["image_idx"] = (1, 1)
        inputs_shapes["image_sizes"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 2)
        inputs_shapes["deepstack_features"] = (
            len(self.config.vision_config.deepstack_visual_indexes),
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            vision_size,
            self.model.config.vision_config.out_hidden_size,
        )

        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros(
            (inputs_shapes["pixel_values"]), dtype=self.model.config.torch_dtype
        )
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
        lang_inputs["vision_embeds"] = torch.zeros(
            (inputs_shapes["vision_embeds"]), dtype=self.model.config.torch_dtype
        )
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

        lang_inputs["position_ids"] = (
            (torch.arange(prefill_seq_len, dtype=torch.int64).view(1, prefill_seq_len).repeat(bs, 1))
            .unsqueeze(0)
            .repeat(4, 1, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)
        lang_inputs["deepstack_features"] = torch.zeros(
            (inputs_shapes["deepstack_features"]), dtype=self.model.config.torch_dtype
        )

        lang_inputs["past_key_values"] = [[] for _ in range(self.model.config.text_config.num_hidden_layers)]
        for i in range(self.model.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(
                    torch.zeros(kv_cache_shape, dtype=self.model.config.torch_dtype)
                )

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
        height: int | list[int] | None = None,
        width: int | list[int] | None = None,
        img_size=None,
        time: int = 1,
        num_frames: int | list[int] = 1,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: int | None = None,
        full_batch_size: int | None = None,
        vision_batch_size: int | None = None,
        **compiler_options,
    ):
        comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill", None)
        comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode", None)
        # Preserve the legacy shape when callers do not request a separate vision batch.
        vision_batch_size = batch_size if vision_batch_size is None else vision_batch_size
        if height is None or width is None:
            height = constants.QWEN3_VL_HEIGHT
            width = constants.QWEN3_VL_WIDTH
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

        IMAGE_FACTOR = constants.IMAGE_FACTOR_QWEN_3
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
            vision_size = vision_size * time
            grid_height = grid_height * time * vision_batch_size
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
                    "time": time,
                    "num_feature_layers": len(self.config.vision_config.deepstack_visual_indexes),
                }
            )

        if comp_ctx_lengths_prefill is not None or comp_ctx_lengths_decode is not None:
            lang = []

            for i in range(len(comp_ctx_lengths_prefill or [])):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "vision_size": max_vision_size,
                    "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                    "vision_batch_size": vision_batch_size,
                    "num_feature_layers": len(self.config.vision_config.deepstack_visual_indexes),
                }

                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_prefill["batch_size"] = kv_cache_batch_size
                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size

                lang.append(lang_prefill)

            for i in range(len(comp_ctx_lengths_decode or [])):
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": 1,
                    "ctx_len": ctx_len,
                    "vision_size": max_vision_size,
                    "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                    "vision_batch_size": vision_batch_size,
                    "num_feature_layers": len(self.config.vision_config.deepstack_visual_indexes),
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
                "num_feature_layers": len(self.config.vision_config.deepstack_visual_indexes),
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
                "num_feature_layers": len(self.config.vision_config.deepstack_visual_indexes),
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
        self, comp_ctx_lengths: list[int] | None = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        # Define dynamic axes
        num_layers = self.config.text_config.num_hidden_layers
        vision_dynamic_axes = {
            "pixel_values": {0: "grid_height", 1: "grid_width"},
            "image_grid_thw": {0: "vision_batch_size", 1: "time", 2: "grid_h", 3: "grid_w"},
            "deepstack_features": {0: "num_feature_layers", 1: "vision_batch_size", 2: "vision_size"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {1: "batch_size", 2: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
            "deepstack_features": {0: "num_feature_layers", 1: "vision_batch_size", 2: "vision_size"},
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
        vision_output_names.append("deepstack_features")
        lang_output_names = ["logits"]
        for i in range(self.model.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            lang_output_names.insert(2, "deepstack_features_RetainedState")
            if _should_export_embedding_output(self):
                lang_output_names.insert(4, "embedding_output")
            elif getattr(self.model.language_model, "target_layer_ids", None):
                lang_output_names.insert(4, "hidden_states")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            if _should_export_embedding_output(self):
                lang_output_names.insert(3, "embedding_output")
            elif getattr(self.model.language_model, "target_layer_ids", None):
                lang_output_names.insert(3, "hidden_states")
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
            image_grid_thw=inputs.get("image_grid_thw", None),
            video_grid_thw=inputs.get("video_grid_thw", None),
            second_per_grid_ts=inputs.get("second_per_grid_ts", None),
            attention_mask=inputs["attention_mask"],
        )
        self.model.rope_deltas = rope_deltas
        inputs["position_ids"] = torch.cat((inputs["position_ids"], pos_ids), dim=0)

        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        inputs["position_ids"] = F.pad(
            inputs["position_ids"], pad=(0, padded_len - input_ids_length), mode="constant", value=-1
        )

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
