# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import math
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
    gptoss_clamped_glu_mlp,
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


class _QEffGptOssLegacyBlockedMixin:
    def forward(self, hidden: torch.Tensor):
        if os.environ.get("NUM_FFN_BLOCKS", None) is not None:
            return self.blocked_ffn_forward(hidden)
        return super().forward(hidden)

    def blocked_ffn_forward(self, hidden: torch.Tensor):
        if not getattr(self, "weights_transformed", False):
            raise RuntimeError(f"{type(self).__name__} weights are not transformed; run OptimizedMoEWeightsTransform")
        B, S, H = hidden.shape
        T = B * S
        hidden = hidden.view(T, H)
        weights = self.moe_weights

        # Router computation
        router_logits = F.linear(hidden, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, top_i = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        masked_logits = torch.zeros_like(router_logits)
        masked_logits.scatter_(1, top_i, top_w)

        # Routing weights for each expert [T, E]
        routing_weights = masked_logits

        # ────────────────── allocate the output tensor ─────
        expert_out = hidden.new_zeros((T, H))  # accumulation buffer
        target_blocks = int(os.environ.get("NUM_FFN_BLOCKS", 1))
        block_positions = []
        for j in range(target_blocks):
            block_positions.append(j * (T // target_blocks))
        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)  # [T, 1]

            W_g, W_u = weights.gate[e], weights.up[e]  # [H, I], [H, I]
            b_g, b_u = weights.gate_bias[e], weights.up_bias[e]  # [I], [I]
            W_d = weights.down[e]  # [I, H]
            b_d = weights.down_bias[e]  # [H]

            block_count = 0
            outs = []
            for block_idx in range(target_blocks):
                block_count += 1
                qi = block_positions[block_idx]

                # Calculate block size (last block should be handled with remainder)
                if block_idx == target_blocks - 1:
                    real_q_len = T - qi
                else:
                    real_q_len = block_positions[block_idx + 1] - qi

                tgb = hidden[qi : qi + real_q_len, :]
                # Gate and Up projections
                # Gate and Up projections
                gate = (tgb @ W_g) + b_g  # [T, I]
                up = (tgb @ W_u) + b_u  # [T, I]

                # Apply GptOss activation with clamping
                gate = gate.clamp(min=torch.finfo(torch.float16).min, max=self.experts.limit)
                up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

                # GLU activation
                glu = gate * torch.sigmoid(gate * self.experts.alpha)
                intermediate = (up + 1) * glu  # [T, I]

                # Down projection
                down_out_block = (intermediate @ W_d) + b_d  # [T, H]

                outs.append(down_out_block)

            down_out = torch.cat(outs, dim=0)

            # Apply routing weights and accumulate
            expert_out += down_out * routing_weight

        # original shape [B, S, H]
        return expert_out.view(B, S, H), router_logits

    def blocked_ffn_forward_block_weights(self, hidden: torch.Tensor):
        if not getattr(self, "weights_transformed", False):
            raise RuntimeError(f"{type(self).__name__} weights are not transformed; run OptimizedMoEWeightsTransform")
        B, S, H = hidden.shape
        T = B * S
        hidden = hidden.view(T, H)
        weights = self.moe_weights

        # Router computation
        router_logits = F.linear(hidden, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, top_i = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        masked_logits = torch.zeros_like(router_logits)
        masked_logits.scatter_(1, top_i, top_w)

        # Routing weights for each expert [T, E]
        routing_weights = masked_logits

        # ────────────────── allocate the output tensor ─────
        expert_out = hidden.new_zeros((T, H))  # accumulation buffer
        target_blocks = int(os.environ.get("NUM_BLOCKS", 1))
        block_positions = []
        for j in range(target_blocks):
            block_positions.append(j * (T // target_blocks))
        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)  # [T, 1]

            W_g, W_u = weights.gate[e], weights.up[e]  # [H, I], [H, I]
            b_g, b_u = weights.gate_bias[e], weights.up_bias[e]  # [I], [I]
            W_d = weights.down[e]  # [I, H]
            b_d = weights.down_bias[e]  # [H]

            block_count = 0
            outs = []
            for block_idx in range(target_blocks):
                block_count += 1
                qi = block_positions[block_idx]

                # Calculate block size (last block should be handled with remainder)
                if block_idx == target_blocks - 1:
                    real_q_len = T - qi
                else:
                    real_q_len = block_positions[block_idx + 1] - qi

                tgb = hidden[qi : qi + real_q_len, :]
                # Gate and Up projections

                wg_col_shape = W_g.shape[1]
                wg_num_blocks = math.ceil(wg_col_shape / 128)
                last_block_size = wg_col_shape % 128 if wg_col_shape % 128 != 0 else 128

                intermediates = []
                for i in range(wg_num_blocks):
                    if i == wg_num_blocks - 1:
                        cur_gate = (tgb @ W_g[:, -last_block_size:]) + b_g[-last_block_size:]
                        cur_up = (tgb @ W_u[:, -last_block_size:]) + b_u[-last_block_size:]
                    else:
                        cur_gate = (tgb @ W_g[:, i * 128 : (i + 1) * 128]) + b_g[i * 128 : (i + 1) * 128]
                        cur_up = (tgb @ W_u[:, i * 128 : (i + 1) * 128]) + b_u[i * 128 : (i + 1) * 128]

                    cur_gate = cur_gate.clamp(min=torch.finfo(torch.float16).min, max=self.experts.limit)
                    cur_up = cur_up.clamp(min=-self.experts.limit, max=self.experts.limit)
                    cur_glu = cur_gate * torch.sigmoid(cur_gate * self.experts.alpha)
                    cur_intermediate = (cur_up + 1) * cur_glu
                    intermediates.append(cur_intermediate)

                intermediate = torch.cat(intermediates, dim=-1)

                downs = []
                for i in range(wg_num_blocks):
                    if i == wg_num_blocks - 1:
                        downs.append((intermediate @ W_d[:, -last_block_size:]) + b_d[-last_block_size:])
                    else:
                        downs.append((intermediate @ W_d[:, i * 128 : (i + 1) * 128]) + b_d[i * 128 : (i + 1) * 128])

                down_out_block = torch.cat(downs, dim=1)
                outs.append(down_out_block)

            down_out = torch.cat(outs, dim=0)

            # Apply routing weights and accumulate
            masked_down = torch.where(routing_weight > 0, down_out * routing_weight, torch.zeros_like(expert_out))
            expert_out += masked_down

        # original shape [B, S, H]
        return expert_out.view(B, S, H), router_logits


class QEffGptOssMLP(_QEffGptOssLegacyBlockedMixin, QEffMoEBlockMixin, GptOssMLP):
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

    # ------------------- Gather based, weights as activation approach ---------------
    def forward_weights_as_activation(self, hidden_states):
        if not getattr(self, "weights_transformed", False):
            raise RuntimeError(f"{type(self).__name__} weights are not transformed; run OptimizedMoEWeightsTransform")
        bs, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(bs * seq_len, self.experts.hidden_size)
        weights = self.moe_weights

        # Router computation
        router_logits = F.linear(hidden_states, self.router.weight, self.router.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.router.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)

        # GATHER - collect weights for selected experts
        gate_proj = weights.gate[router_indices.flatten()]
        up_proj = weights.up[router_indices.flatten()]
        gate_proj_bias = weights.gate_bias[router_indices.flatten()]
        up_proj_bias = weights.up_bias[router_indices.flatten()]
        down_proj = weights.down[router_indices.flatten()]
        down_proj_bias = weights.down_bias[router_indices.flatten()]

        # Apply Chosen Experts (without routing weights first)
        # expert_in = hidden_states.repeat_interleave(self.router.top_k, dim=0)
        # expert_in = expert_in.view(-1, 1, self.experts.hidden_size)
        # Reshape for bmm: (bs*seq_len*top_k, 1, hidden_size)
        expert_in = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.router.top_k, -1)
            .contiguous()
            .view(-1, 1, self.experts.hidden_size)
        )

        gate = torch.bmm(expert_in, gate_proj) + gate_proj_bias.unsqueeze(1)
        up = torch.bmm(expert_in, up_proj) + up_proj_bias.unsqueeze(1)

        # Apply activation with clamping
        gate = gate.clamp(min=None, max=self.experts.limit)
        up = up.clamp(min=-self.experts.limit, max=self.experts.limit)
        glu = gate * torch.sigmoid(gate * self.experts.alpha)
        gated_output = (up + 1) * glu

        experts_out = torch.bmm(gated_output, down_proj) + down_proj_bias.unsqueeze(1)
        experts_out = experts_out.view(bs * seq_len, self.router.top_k, self.experts.hidden_size)

        # Apply routing weights AFTER expert computation (This is before on Llama4)
        experts_out = experts_out * router_top_value.unsqueeze(-1)
        experts_out = experts_out.sum(dim=1)

        return experts_out, router_logits

    def forward(self, hidden_states):
        if os.environ.get("NUM_FFN_BLOCKS", None) is not None:
            return self.blocked_ffn_forward(hidden_states)
        return QEffMoEBlockMixin.forward(self, hidden_states)

    def optimized_moe_forward(self, hidden_states: torch.Tensor):
        if not getattr(self, "weights_transformed", False):
            raise RuntimeError(f"{type(self).__name__} weights are not transformed; run OptimizedMoEWeightsTransform")
        B, S, H = hidden_states.shape
        T = B * S
        hidden_states = hidden_states.view(T, H)
        weights = self.moe_weights

        # Router computation
        router_logits = F.linear(hidden_states, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, selected_experts = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        # Creating experts mask and routing weights masked
        awesome_experts_mask_1 = (
            torch.nn.functional.one_hot(selected_experts[:, 0], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_2 = (
            torch.nn.functional.one_hot(selected_experts[:, 1], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_3 = (
            torch.nn.functional.one_hot(selected_experts[:, 2], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_4 = (
            torch.nn.functional.one_hot(selected_experts[:, 3], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )

        gateupout1 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout2 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout3 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout4 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs

        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            W_g, W_u = weights.gate[e], weights.up[e]  # [H, I], [H, I]
            b_g, b_u = weights.gate_bias[e], weights.up_bias[e]  # [I], [I]

            # Gate and Up projections
            gate = (hidden_states @ W_g) + b_g  # [T, I]
            up = (hidden_states @ W_u) + b_u  # [T, I]

            # Apply GptOss activation with clamping
            gate = gate.clamp(min=None, max=self.experts.limit)
            up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

            # GLU activation
            glu = gate * torch.sigmoid(gate * self.experts.alpha)
            intermediate = (up + 1) * glu  # [T, I]

            gateupout1 += torch.where(awesome_experts_mask_1[e], intermediate, torch.zeros_like(gateupout1))
            gateupout2 += torch.where(awesome_experts_mask_2[e], intermediate, torch.zeros_like(gateupout2))
            gateupout3 += torch.where(awesome_experts_mask_3[e], intermediate, torch.zeros_like(gateupout3))
            gateupout4 += torch.where(awesome_experts_mask_4[e], intermediate, torch.zeros_like(gateupout4))

        concat_down = torch.zeros((self.router.top_k, T, H))
        concat_mask = torch.cat(
            (
                awesome_experts_mask_1.unsqueeze(0),
                awesome_experts_mask_2.unsqueeze(0),
                awesome_experts_mask_3.unsqueeze(0),
                awesome_experts_mask_4.unsqueeze(0),
            ),
            dim=0,
        )

        concat_gateout = torch.cat(
            (gateupout1.unsqueeze(0), gateupout2.unsqueeze(0), gateupout3.unsqueeze(0), gateupout4.unsqueeze(0)), dim=0
        )

        for e in range(self.experts.num_experts):
            W_d = weights.down[e]  # [I, H]
            b_d = weights.down_bias[e]  # [H]

            # Down projection
            down_out = (concat_gateout @ W_d) + b_d  # [T, H]

            concat_down += torch.where(concat_mask[:, e, :], down_out, torch.zeros_like(concat_down))

        downout1, downout2, downout3, downout4 = concat_down[0], concat_down[1], concat_down[2], concat_down[3]
        hidden_states = (
            downout1 * top_w[:, 0].unsqueeze(-1)
            + downout2 * top_w[:, 1].unsqueeze(-1)
            + downout3 * top_w[:, 2].unsqueeze(-1)
            + downout4 * top_w[:, 3].unsqueeze(-1)
        ).reshape(B, S, H)

        # original shape [B, S, H]
        return hidden_states, router_logits


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

        if past_key_values is not None:
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

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        use_blocking = blocking_config is not None and blocking_config.mode.is_prefill and (self.sliding_window is None)

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
            s_aux=self.sinks,  # diff with Llama
            layer_idx=self.layer_idx,
            blocking_config=blocking_config,
            position_ids=position_ids,
            past_key_value=past_key_values,
            batch_index=batch_index,
            prefill_only=True,
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

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values.get_max_cache_len())
        sliding_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=past_key_values.get_sliding_window_len(),
            sliding_window=past_key_values.get_sliding_window_len(),
        )

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
