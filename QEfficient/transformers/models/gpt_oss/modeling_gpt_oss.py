# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os
from functools import partial
from typing import Callable, Optional, Type, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssConfig,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssForCausalLM,
    GptOssMLP,
    GptOssModel,
    GptOssRotaryEmbedding,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.transformers.cache_utils import QEffGPTOSSHybridCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    MoEWeights,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    gptoss_clamp,
    gptoss_clamped_glu_mlp,
    resolve_routing,
    unpack_moe_weights_from_expert_parallel,
)
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("MODEL")


def override_gptoss_prefill_chunking(
    config, prefill_only: Optional[bool], enable_chunking: Optional[bool]
) -> Optional[bool]:
    if prefill_only is True and enable_chunking is False and getattr(config, "model_type", None) == "gpt_oss":
        logger.warning(
            "For gpt_oss, chunking is always enabled for prefill-only mode; overriding enable_chunking=True."
        )
        return True
    return enable_chunking


class QEffGptOssExperts(GptOssExperts):
    def __qeff_init__(self):
        self.weights_transformed = False
        self.expert_dim = getattr(self, "intermediate_size", self.gate_up_proj.shape[-1] // 2)

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        self.moe_weights = build_canonical_expert_weights(
            gate_up=self.gate_up_proj,
            down=self.down_proj,
            gate_up_bias=self.gate_up_proj_bias,
            down_bias=self.down_proj_bias,
            fused=True,
            fused_split_dim=2,
            interleaved=True,
            transpose_gate_up=False,
            transpose_down=False,
            clone=True,
        )
        delete_module_attrs(self, "gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias")
        self.weights_transformed = True
        return self.moe_weights


class QEffGptOssMLP(QEffMoEBlockMixin, GptOssMLP):
    _moe_return_router_logits = True
    supported_moe_flavours = (
        MoEFlavour.SIMPLE_LOOP,
        MoEFlavour.DECODE_BMM,
        MoEFlavour.EXPERT_PARALLEL,
    )

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        weights = self.experts.transform_weights()
        self.moe_weights = weights
        self.weights_transformed = True
        return self.moe_weights

    @property
    def moe_profile(self) -> MoEProfile:
        return MoEProfile(
            expert_mlp=partial(gptoss_clamped_glu_mlp, limit=self.experts.limit, alpha=self.experts.alpha),
            has_bias=True,
        )

    def route(self, x: torch.Tensor):
        router_logits = F.linear(x, self.router.weight, self.router.bias)
        top_w, top_i = torch.topk(router_logits, self.router.top_k, dim=-1)
        top_w = F.softmax(top_w, dim=1, dtype=top_w.dtype)
        return (top_i, top_w), router_logits

    def execute_moe_flavour(self, x: torch.Tensor, routing) -> torch.Tensor:
        """Override for token-blocked simple-loop when num_ffn_blocks is configured.

        When ``expert_num_ffn_blocks`` is set (via ``moe_config['num_ffn_blocks']``),
        iterates outer-expert inner-token-block so each expert's weights are loaded once
        and reused across all token blocks before moving to the next expert. This
        preserves the CopyToVTCM+compute pipelining that the compiler can exploit
        (weights stay hot across token blocks). When not set, dispatch falls through
        to the standard flavour path.
        """
        num_ffn_blocks = getattr(self, "expert_num_ffn_blocks", None)
        if num_ffn_blocks is None:
            env_val = os.environ.get("NUM_FFN_BLOCKS", None)
            if env_val is not None:
                logger.warning(
                    "NUM_FFN_BLOCKS environment variable is deprecated. "
                    "Pass num_ffn_blocks via qaic_config['moe_config']['num_ffn_blocks'] instead."
                )
                num_ffn_blocks = int(env_val)

        # TODO: NUM_Q_BLOCKS is not currently supported in the MoE FFN path.
        # It only applies to opt_eager_attention_forward_blocked (attention Q-tiling).
        if os.environ.get("NUM_Q_BLOCKS", None) is not None:
            logger.warning(
                "NUM_Q_BLOCKS is not supported in the MoE FFN path. "
                "It only applies to the prefill attention path (opt_eager_attention_forward_blocked)."
            )
        if num_ffn_blocks is None:
            return super().execute_moe_flavour(x, routing)

        if not getattr(self, "weights_transformed", False):
            raise RuntimeError(
                f"{type(self).__name__} weights are not transformed; run OptimizedMoEWeightsTransform before forward"
            )

        T, H = x.shape
        dense, _ = resolve_routing(routing, self.moe_weights.num_experts)
        block_size = T // num_ffn_blocks

        weights = self.moe_weights
        if weights.gate.ndim == 4:
            weights = unpack_moe_weights_from_expert_parallel(weights)

        limit = self.experts.limit
        alpha = self.experts.alpha
        _fp16_min = torch.finfo(torch.float16).min

        i_block_size = getattr(self, "expert_intermediate_block_size", None)

        expert_out = x.new_zeros((T, H))
        for e in range(weights.num_experts):
            routing_weight = dense[:, e].unsqueeze(-1)
            W_g = weights.gate[e]
            W_u = weights.up[e]
            W_d = weights.down[e]
            b_g = weights.gate_bias[e] if weights.gate_bias is not None else None
            b_u = weights.up_bias[e] if weights.up_bias is not None else None
            b_d = weights.down_bias[e] if weights.down_bias is not None else None
            outs = []
            for i in range(num_ffn_blocks):
                start = i * block_size
                end = T if i == num_ffn_blocks - 1 else start + block_size
                xb = x[start:end]
                if i_block_size is None:
                    gate = xb @ W_g
                    if b_g is not None:
                        gate = gate + b_g
                    up = xb @ W_u
                    if b_u is not None:
                        up = up + b_u
                    gate = gptoss_clamp(gate, min_value=_fp16_min, max_value=limit)
                    up = gptoss_clamp(up, min_value=-limit, max_value=limit)
                    glu = gate * torch.sigmoid(gate * alpha)
                    intermediate = (up + 1) * glu
                    down = intermediate @ W_d
                else:
                    inter_size = W_d.shape[0]
                    down = xb.new_zeros((xb.shape[0], H))
                    for i_start in range(0, inter_size, i_block_size):
                        i_end = min(i_start + i_block_size, inter_size)
                        gate_s = xb @ W_g[:, i_start:i_end]
                        if b_g is not None:
                            gate_s = gate_s + b_g[i_start:i_end]
                        up_s = xb @ W_u[:, i_start:i_end]
                        if b_u is not None:
                            up_s = up_s + b_u[i_start:i_end]
                        gate_s = gptoss_clamp(gate_s, min_value=_fp16_min, max_value=limit)
                        up_s = gptoss_clamp(up_s, min_value=-limit, max_value=limit)
                        glu_s = gate_s * torch.sigmoid(gate_s * alpha)
                        intermediate_s = (up_s + 1) * glu_s
                        down = down + intermediate_s @ W_d[i_start:i_end, :]
                if b_d is not None:
                    down = down + b_d
                outs.append(down)
            expert_out = expert_out + torch.cat(outs, dim=0) * routing_weight
        return expert_out


#  Can be replaced with llama/modeling_llama.py::QEffLlamaRotaryEmbedding but keeping it following transformers ideology
class QEffGptOssRotaryEmbedding(GptOssRotaryEmbedding):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: GptOssConfig, device=None):
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask,
            torch.full_like(attn_weights, MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype),
            attn_weights,
        )

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def opt_eager_attention_forward_blocked(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    BS, NH, CL, DH = query.shape
    target_blocks = int(os.environ.get("NUM_Q_BLOCKS", 1))
    block_positions = []
    for j in range(target_blocks):
        block_positions.append(j * (CL // target_blocks))
    block_count = 0
    outs = []
    for block_idx in range(target_blocks):
        block_count += 1
        qi = block_positions[block_idx]
        # Calculate block size (last block should be handled with remainder)

        if block_idx == target_blocks - 1:
            real_q_len = CL - qi
        else:
            real_q_len = block_positions[block_idx + 1] - qi

        if block_idx == 0:
            kv_start_idx = 0
        else:
            kv_start_idx = qi - 128

        q_block = query[:, :, qi : qi + real_q_len, :]
        if kwargs.get("sliding_window"):
            k_block = key_states[:, :, kv_start_idx : qi + real_q_len, :]
            v_block = value_states[:, :, kv_start_idx : qi + real_q_len, :]
            attn_mask_block = attention_mask[:, :, qi : qi + real_q_len, kv_start_idx : qi + real_q_len]
        else:
            k_block = key_states
            v_block = value_states
            attn_mask_block = attention_mask[:, :, qi : qi + real_q_len, :]

        scores = torch.matmul(q_block, k_block.transpose(2, 3)) * scaling

        curr_attn_weights = torch.where(
            attn_mask_block,
            torch.full_like(scores, MIN_MASKED_ATTENTION_VALUE, dtype=scores.dtype),
            scores,
        )
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(
            curr_attn_weights.shape[0], -1, curr_attn_weights.shape[-2], -1
        )
        combined_logits = torch.cat([curr_attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        curr_attn_weights = nn.functional.softmax(combined_logits, dim=-1, dtype=torch.float32)
        curr_attn_weights = curr_attn_weights[..., :-1]
        out_block = torch.matmul(curr_attn_weights, v_block)
        outs.append(out_block)
    output = torch.cat(outs, dim=2)

    output = output.view(BS, NH, CL, DH).transpose(1, 2).contiguous()
    return output, output


def _prepare_gpt_oss_sliding_chunked_attention_mask(
    sliding_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    sliding_window: int,
) -> torch.Tensor:
    ctx_len = position_ids.shape[1] + sliding_window
    ctx_indices = torch.arange(ctx_len, device=position_ids.device)
    first_pos_idx = position_ids[0][0]
    add_idx = torch.where(first_pos_idx >= sliding_window, first_pos_idx - sliding_window, 0)
    ctx_indices += add_idx
    return sliding_mask[:, :, :, ctx_indices]


class QEffPrefillOnlyChunkedGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        hidden_shape = (*input_shape, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        blocking_mode = BlockingMode.resolve(blocking_config.mode) if blocking_config is not None else BlockingMode.NONE
        use_blocking = (
            blocking_config is not None and blocking_mode != BlockingMode.NONE and (self.sliding_window is None)
        )
        use_decode_blocking = use_blocking and not blocking_mode.is_prefill

        if past_key_values is not None and not use_decode_blocking:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin_cached,
                "cos": cos_cached,
                "batch_index": batch_index,
                "position_ids": position_ids,
                "config": self.config,
                "is_sliding": self.sliding_window is not None,
                "sliding_window": self.sliding_window,
            }
            if self.sliding_window is not None:
                key_states, value_states = past_key_values.sliding_window_update_chunked(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                if comp_ctx_lengths is not None:
                    attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                    cache_kwargs["CCL"] = attention_mask.shape[-1]
                key_states, value_states = past_key_values.full_cache_update_chunked(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

        if use_blocking:
            attention_interface = generic_blocked_attention_interface
        else:
            attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            sinks=self.sinks,
            s_aux=self.sinks,  # diff with Llama
            layer_idx=self.layer_idx,
            blocking_config=blocking_config,
            position_ids=position_ids,
            past_key_value=past_key_values,
            batch_index=batch_index,
            prefill_only=True,
            # KV cache already written above via sliding_window_update_chunked /
            # full_cache_update_chunked; skip the duplicate write inside the interface.
            skip_kv_write=True,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_values


class QEffPrefillOnlyGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        hidden_shape = (*input_shape, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin_cached,
                "cos": cos_cached,
                "batch_index": batch_index,
                "position_ids": position_ids,
                "config": self.config,
                "is_sliding": self.sliding_window is not None,
                "sliding_window": past_key_values.get_sliding_window_len(),
            }
            if self.sliding_window is not None:
                sliding_window_len = past_key_values.get_sliding_window_len()
                short_read_idx = torch.arange(
                    past_key_values.layers[self.layer_idx].keys.shape[2], device=position_ids.device
                )
                read_idx = short_read_idx + torch.where(
                    position_ids.max() > sliding_window_len - 1, position_ids.max() - sliding_window_len + 1, 0
                )
                # This is a trick to export with seq_len<sliding_window_len
                read_idx = torch.where(read_idx > position_ids.max(), 0, read_idx)
                k_cache = key_states[:, :, read_idx, :]
                v_cache = value_states[:, :, read_idx, :]
            else:
                k_cache, v_cache = key_states, value_states

            _, _ = past_key_values.write_only(k_cache, v_cache, self.layer_idx, cache_kwargs)

        if os.environ.get("ENABLE_OPT_SWA", "0") == "1":
            attention_interface: Callable = opt_eager_attention_forward_blocked
        else:
            attention_interface: Callable = eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_values


class QEffGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        use_blocking = (
            blocking_config is not None
            and (blocking_config.mode != BlockingMode.NONE)
            and (self.sliding_window is None)
        )
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
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
                block_table=block_table,
                slot_id=slot_id,
                past_seen_tokens=past_seen_tokens,
                sliding_window=self.sliding_window,
                sinks=self.sinks,
            )
        else:
            key_states, value_states, attention_mask, _ = past_key_value_update(
                module=self,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                ccl_length=(
                    attention_mask.shape[-1]
                    if comp_ctx_lengths is not None and not torch._dynamo.is_compiling()
                    else None
                ),
                batch_index=batch_index,
                position_ids=position_ids,
                sliding_window=self.sliding_window,
                sliding_window_len=(
                    past_key_values.get_sliding_window_len(self.layer_idx)
                    if self.sliding_window is not None and past_key_values is not None
                    else None
                ),
            )
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,  # diff with Llama
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_values


class QEffGptOssDecoderLayer(GptOssDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        sin_cached=None,
        cos_cached=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
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
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
        hidden_states = hidden_states.reshape(residual.shape)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class QEffPrefillOnlyGptOssModel(GptOssModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffGptOssRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffGPTOSSHybridCache.from_legacy_cache(self.config, past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values.get_max_cache_len())
        sliding_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=past_key_values.get_max_cache_len(),
            sliding_window=self.config.sliding_window,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        sin = self.sin_cached[position_ids].unsqueeze(1).to(device=hidden_states.device)
        cos = self.cos_cached[position_ids].unsqueeze(1).to(device=hidden_states.device)
        layer_sliding_mask = sliding_mask
        if isinstance(self.layers[0].self_attn, QEffPrefillOnlyChunkedGptOssAttention):
            layer_sliding_mask = _prepare_gpt_oss_sliding_chunked_attention_mask(
                sliding_mask=sliding_mask,
                position_ids=position_ids,
                sliding_window=self.config.sliding_window,
            )

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            self_attn = decoder_layer.self_attn
            if self_attn.sliding_window is None:
                layer_attention_mask = causal_mask
            else:
                layer_attention_mask = layer_sliding_mask

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                block_table=block_table,
                slot_id=slot_id,
                past_key_value=past_key_values,
                batch_index=batch_index,
                use_cache=use_cache,
                output_attentions=output_attentions,
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

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffGptOssModel(GptOssModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffGptOssRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffGPTOSSHybridCache.from_legacy_cache(self.config, past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        is_dynamo_export = torch._dynamo.is_compiling()
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values.get_max_cache_len())
        if comp_ctx_lengths is not None and not is_dynamo_export:
            causal_mask = causal_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
        sliding_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=past_key_values.get_sliding_window_len(),
            sliding_window=past_key_values.get_sliding_window_len(),
        )
        if comp_ctx_lengths is not None and not is_dynamo_export:
            sliding_mask = sliding_mask[:, :, :, : comp_ctx_lengths.shape[-1]]

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        sin = self.sin_cached[position_ids].unsqueeze(1).to(device=hidden_states.device)
        cos = self.cos_cached[position_ids].unsqueeze(1).to(device=hidden_states.device)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            self_attn = decoder_layer.self_attn
            if self_attn.sliding_window is None:
                layer_attention_mask = causal_mask
            else:
                layer_attention_mask = sliding_mask

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                block_table=block_table,
                slot_id=slot_id,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                output_attentions=output_attentions,
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

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffGptOssForCausalLM(GptOssForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.

        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QEffGptOssDecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GptOssForCausalLM

        >>> model = GptOssForCausalLM.from_pretrained("mistralai/GptOss-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/GptOss-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return MoeCausalLMOutputWithPast(
            loss=None,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def get_pkv_dynamic_axes(self, retain_full_kv: Optional[bool] = False, continuous_batching: Optional[bool] = False):
        pkv_dynamic_axes = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention" and not retain_full_kv:
                pkv_dynamic_axes.append(
                    {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
                )
            else:
                pkv_dynamic_axes.append({0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"})
        return pkv_dynamic_axes

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        **kwargs,
    ):
        batch_size = batch_size if batch_size else 1
        if kwargs.get("prefill_only") and not kwargs.get("enable_chunking") and ctx_len != prefill_seq_len:
            ctx_len = prefill_seq_len
            logger.warning(
                f"overriding ctx_len={prefill_seq_len}, currently we don't support ctx_len different than prefill_seq_len for prefill_only model"
            )

        specializations = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": 128,
            },
            {
                "batch_size": batch_size,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "sliding_window": 128,
            },
        ]
        return specializations
