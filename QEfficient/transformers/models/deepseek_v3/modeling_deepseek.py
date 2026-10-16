# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    generic_blocked_attention_interface,
)
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.customop.utils import select_interface
from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffDynamicCompressedKVRopeCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    MoEWeights,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    silu_glu_mlp,
    stack_expert_linears,
)
from QEfficient.utils.constants import MAX_POSITION_EMBEDDINGS, MIN_MASKED_ATTENTION_VALUE


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class QEffDeepseekV3CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def forward(self, hidden_states):
        """
        Forward pass of the RMSNorm module.

        Args:
            hidden_states (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)(
            hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, dtype=torch.get_default_dtype()):
        super().__init__()

        if dtype is None:
            dtype = torch.get_default_dtype()
        if not isinstance(dtype, torch.dtype):
            raise TypeError(
                f"DeepseekV3RotaryEmbedding: dtype must be a torch.dtype, got {type(dtype).__name__} with value {dtype}"
            )

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=dtype,
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dtype,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device, dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def orig_apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QEffDeepseekV3Attention(nn.Module):
    """Adapted DeepseekV3Attention with QEff logic, adding batch_index and proper position_ids handling."""

    def __qeff_init__(
        self,
    ):
        q_up, q_rope = self.q_b_proj.weight.T.view(
            -1, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        ).split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_up = q_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0)
        self.q_up = torch.nn.Parameter(q_up.detach().clone())

        q_rope = q_rope.reshape(-1, self.num_heads * self.qk_rope_head_dim).unsqueeze(0)
        self.q_rope = torch.nn.Parameter(q_rope.detach().clone())

        k_up, v_up = self.kv_b_proj.weight.T.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_up = k_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0)
        v_up = v_up.reshape(-1, self.num_heads * self.v_head_dim).unsqueeze(0)

        self.k_up = torch.nn.Parameter(k_up.detach())
        self.v_up = torch.nn.Parameter(v_up.detach())
        per_head_q_up = self.q_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
        per_head_k_up = (
            self.k_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1).transpose(1, 2)
        )
        per_head_v_up = self.v_up.squeeze(0).view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)
        self.per_head_v_up = torch.nn.Parameter(per_head_v_up.unsqueeze(0).detach().clone())
        self.per_head_q_up = torch.nn.Parameter(per_head_q_up.unsqueeze(0).detach().clone())
        self.per_head_k_up = torch.nn.Parameter(per_head_k_up.unsqueeze(0).detach().clone())
        per_head_k_up_normal = self.per_head_k_up.transpose(2, 3)
        self.per_head_k_up_normal = torch.nn.Parameter(per_head_k_up_normal.detach().clone())

        fusedqk = torch.bmm(per_head_q_up, per_head_k_up).reshape(
            -1, self.num_heads, self.q_lora_rank, self.kv_lora_rank
        )
        self.fusedqk = torch.nn.Parameter(fusedqk.detach().clone())

    def fused_forward_h_blocking(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pe = torch.matmul(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_nope = torch.bmm(q_a_proj_out, self.q_up)
        q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)

        kva = self.kv_a_layernorm(kva)
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if compressed_kvs is not None:
            kva = compressed_kvs.update_ckv(kva, self.layer_idx, cache_kwargs)

        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos_cached, sin_cached)

        if compressed_kvs is not None:
            k_pe = compressed_kvs.update_k_pe(k_pe, self.layer_idx, cache_kwargs)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())

        attn_output, attn_weights = generic_blocked_attention_interface(
            module=self,
            attention_mask=attention_mask,
            scaling=self.softmax_scale,
            blocking_config=blocking_config,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
            is_mla=True,
            mla_kwargs=dict(
                q_a_proj_out=q_a_proj_out,
                fusedqk=self.fusedqk,
                q_nope=q_nope,
                q_pe=q_pe,
                kva=kva,
                k_pe=k_pe,
                per_head_q_up=self.per_head_q_up,
                per_head_k_up=self.per_head_k_up,
                per_head_v_up=self.per_head_v_up,
                per_head_k_up_normal=self.per_head_k_up_normal,
                mla_absorption=mla_absorption,
            ),
            **kwargs,
        )

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, compressed_kvs

    def fused_forward_kv_blocking(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # -- KV compression (write to cache) ----------------------------------
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pe = torch.matmul(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        kva = self.kv_a_layernorm(kva)
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}

        ## Write Only
        if compressed_kvs is not None:
            compressed_kvs.write_only_ckv(kva, self.layer_idx, cache_kwargs)

        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos_cached, sin_cached)

        if compressed_kvs is not None:
            compressed_kvs.write_only_k_pe(k_pe, self.layer_idx, cache_kwargs)

        if mla_absorption is not None:
            absorption = mla_absorption.get("absorption", False)
            online = mla_absorption.get("online", False)
        else:
            absorption = False

        if absorption:
            if online:
                qup_kupT = torch.matmul(self.per_head_q_up, self.per_head_k_up)
                dq_qup_kupT = torch.matmul(q_a_proj_out, qup_kupT)
            else:
                dq_qup_kupT = torch.matmul(q_a_proj_out, self.fusedqk)
            query = torch.cat((dq_qup_kupT, q_pe), dim=-1)  # [B, num_heads, q_len, d_abs]
        else:
            q_nope = torch.bmm(q_a_proj_out, self.q_up)
            q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
            query = torch.cat((q_nope, q_pe), dim=-1)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())

        attn_output, attn_weights = generic_blocked_attention_interface(
            module=self,
            query=query,
            attention_mask=attention_mask,
            batch_index=batch_index,
            scaling=self.softmax_scale,
            layer_idx=self.layer_idx,
            blocking_config=blocking_config,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
            is_mla=True,
            mla_kwargs=dict(
                per_head_k_up_normal=self.per_head_k_up_normal,
                per_head_v_up=self.per_head_v_up,
                compressed_kvs=compressed_kvs,
                mla_absorption=mla_absorption,
            ),
            **kwargs,
        )

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, compressed_kvs

    def fused_forward_orig(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # ---- KV compression ----
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        # ---- Q projections ----
        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))

        q_pe = torch.bmm(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        kva = self.kv_a_layernorm(kva)

        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if compressed_kvs is not None:
            kva = compressed_kvs.update_ckv(kva, self.layer_idx, cache_kwargs)

        # ---- MLA absorption flags ----
        if mla_absorption is not None:
            absorption = mla_absorption.get("absorption", False)
            online = mla_absorption.get("online", False)
        else:
            absorption = False

        # ---- Rotary ----
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos_cached, sin_cached)

        if compressed_kvs is not None:
            k_pe = compressed_kvs.update_k_pe(k_pe, self.layer_idx, cache_kwargs)

        k_heads, q_heads = kva.shape[1], q_pe.shape[1]

        if k_heads > 1:
            num_heads_to_repeat = math.ceil(q_heads / k_heads)

            kva_expanded = (
                kva.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(bsz, num_heads_to_repeat * k_heads, -1, self.config.kv_lora_rank)
            )
            kva_expanded = kva_expanded[:, :q_heads, :, :]

            k_pe_expanded = (
                k_pe.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(bsz, num_heads_to_repeat * k_heads, -1, self.config.qk_rope_head_dim)
            )
            k_pe_expanded = k_pe_expanded[:, :q_heads, :, :]
        else:
            kva_expanded = kva
            k_pe_expanded = k_pe

        v_up_per_head = self.v_up.squeeze(0).view(self.kv_lora_rank, self.num_heads, self.v_head_dim).permute(1, 0, 2)
        value_states = torch.matmul(kva_expanded, v_up_per_head)

        if absorption:
            if online:
                out = torch.matmul(self.per_head_q_up, self.per_head_k_up)
                q_nope_compressed = torch.matmul(q_a_proj_out.unsqueeze(1), out)
            else:
                q_nope_compressed = torch.matmul(
                    q_a_proj_out.unsqueeze(1),
                    self.fusedqk,
                )
            query_states = torch.cat((q_nope_compressed, q_pe), dim=-1)
            key_states = torch.cat((kva_expanded, k_pe_expanded), dim=-1)
        else:
            q_nope = torch.bmm(q_a_proj_out, self.q_up)
            q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
            query_states = torch.cat((q_nope, q_pe), dim=-1)

            k_up_per_head = (
                self.k_up.squeeze(0).view(self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim).permute(1, 0, 2)
            )
            k_nope = torch.matmul(kva_expanded, k_up_per_head)
            if k_heads == 1:
                k_pe_expanded = (
                    k_pe_expanded.unsqueeze(1)
                    .expand(-1, self.num_heads, -1, -1, -1)
                    .reshape(bsz, self.num_heads, -1, self.qk_rope_head_dim)
                )
            key_states = torch.cat((k_nope, k_pe_expanded), dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        mask_value = torch.full_like(attn_weights, MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = torch.where(attention_mask, mask_value, attn_weights)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
        ## Do v_proj here
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, compressed_kvs

    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        if getattr(blocking_config, "mode", None) == "h":
            return self.fused_forward_h_blocking(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                block_table,
                slot_id,
                past_key_value,
                compressed_kvs,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                mla_absorption,
                cos_cached,
                sin_cached,
                **kwargs,
            )
        elif getattr(blocking_config, "mode", None) == "kv":
            return self.fused_forward_kv_blocking(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                block_table,
                slot_id,
                past_key_value,
                compressed_kvs,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                mla_absorption,
                cos_cached,
                sin_cached,
                **kwargs,
            )
        else:
            return self.fused_forward_orig(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                block_table,
                slot_id,
                past_key_value,
                compressed_kvs,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                mla_absorption,
                cos_cached,
                sin_cached,
                **kwargs,
            )

    def forward_full_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        q_nope = q[:, :, :, : self.qk_nope_head_dim]
        q_pe = q[:, :, :, self.qk_nope_head_dim :]

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        kva = compressed_kv[:, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, self.kv_lora_rank :]

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(kva))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope = kv[:, :, :, : self.qk_nope_head_dim]
        value_states = kv[:, :, :, self.qk_nope_head_dim :]

        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos_cached, sin_cached)

        query_states = torch.cat((q_nope, q_pe), -1)
        k_pe_new = k_pe.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat((k_nope, k_pe_new), -1)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin_cached,
                "cos": cos_cached,
                "batch_index": batch_index,
                "position_ids": position_ids,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        mask_value = torch.full_like(attn_weights, MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = torch.where(attention_mask, mask_value, attn_weights)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def forward_full_kv_h_blocking(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        q_nope = q[:, :, :, : self.qk_nope_head_dim]
        q_pe = q[:, :, :, self.qk_nope_head_dim :]

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        kv = (
            self.kv_b_proj(self.kv_a_layernorm(kva))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope = kv[:, :, :, : self.qk_nope_head_dim]
        value_states = kv[:, :, :, self.qk_nope_head_dim :]

        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos_cached, sin_cached)

        query_states = torch.cat((q_nope, q_pe), -1)
        k_pe_new = k_pe.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat((k_nope, k_pe_new), -1)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())

        attn_output, attn_weights = generic_blocked_attention_interface(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            scaling=self.softmax_scale,
            layer_idx=self.layer_idx,
            past_key_value=past_key_value,
            blocking_config=blocking_config,
            batch_index=batch_index,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        if getattr(blocking_config, "mode", None) == "h":
            return self.forward_full_kv_h_blocking(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                block_table,
                slot_id,
                past_key_value,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                cos_cached,
                sin_cached,
                **kwargs,
            )
        else:
            return self.forward_full_kv(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                block_table,
                slot_id,
                past_key_value,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                cos_cached,
                sin_cached,
                **kwargs,
            )


def _deepseek_expert_weight(proj: nn.Module) -> torch.Tensor:
    if hasattr(proj, "compressor"):
        return proj.compressor.decompress_module(proj)
    return proj.weight


def _deepseek_has_quantized_expert_projection(proj: nn.Module) -> bool:
    return all(hasattr(proj, name) for name in ("qweight", "qzeros", "scales", "bits", "group_size"))


def _deepseek_first_unfused_expert(experts: nn.Module) -> Optional[nn.Module]:
    if hasattr(experts, "gate_up_proj"):
        return None
    if not hasattr(experts, "__getitem__"):
        return None
    try:
        return experts[0]
    except (IndexError, KeyError, TypeError):
        return None


def _deepseek_act_fn(experts: nn.Module) -> Callable:
    if hasattr(experts, "act_fn"):
        return experts.act_fn
    return experts[0].act_fn


def _deepseek_route_tokens(module: nn.Module, hidden_states: torch.Tensor):
    router_output = module.gate(hidden_states)
    if isinstance(router_output, tuple):
        if len(router_output) == 2:
            topk_indices, topk_weights = router_output
            return topk_indices, topk_weights, None
        if len(router_output) == 3:
            first, topk_weights, third = router_output
            if torch.is_floating_point(third):
                topk_indices, router_logits = first, third
            else:
                router_logits, topk_indices = first, third
            return topk_indices, topk_weights, router_logits
        raise ValueError(f"Expected DeepSeek router to return 2 or 3 tensors, got {len(router_output)}")
    topk_indices, topk_weights = module.route_tokens_to_experts(router_output)
    return topk_indices, topk_weights, router_output


class QEffDeepseekV3MoE(QEffMoEBlockMixin, nn.Module):
    supported_moe_flavours = (MoEFlavour.SIMPLE_LOOP, MoEFlavour.DECODE_BMM, MoEFlavour.EXPERT_PARALLEL)
    quantized_supported_moe_flavours = (MoEFlavour.DECODE_BMM, MoEFlavour.EXPERT_PARALLEL)
    supports_moe_decode_bmm = True

    def __qeff_init__(
        self,
    ):
        QEffMoEBlockMixin.__qeff_init__(self)
        if hasattr(self, "all_gate_qweight"):
            self._qeff_quantized_experts = True
            return
        self.act_fn = _deepseek_act_fn(self.experts)
        first_expert = _deepseek_first_unfused_expert(self.experts)
        self._qeff_quantized_experts = (
            first_expert is not None
            and hasattr(first_expert, "gate_proj")
            and _deepseek_has_quantized_expert_projection(first_expert.gate_proj)
        )

    def _stack_quantized_projection_params(self, projection_name: str) -> None:
        first_projection = getattr(self.experts[0], projection_name)
        name_prefix = projection_name[: -len("_proj")]
        in_features = first_projection.in_features
        out_features = first_projection.out_features

        setattr(self, f"in_features_{name_prefix}", in_features)
        setattr(self, f"out_features_{name_prefix}", out_features)
        setattr(
            self,
            f"all_{name_prefix}_qweight",
            torch.nn.Parameter(
                torch.stack([getattr(expert, projection_name).qweight for expert in self.experts], dim=0).reshape(
                    -1, out_features, in_features // 2
                ),
                requires_grad=False,
            ),
        )
        setattr(
            self,
            f"all_{name_prefix}_scales",
            torch.nn.Parameter(
                torch.stack([getattr(expert, projection_name).scales for expert in self.experts], dim=0).reshape(
                    -1, out_features, in_features // self.group_size
                ),
                requires_grad=False,
            ),
        )
        setattr(
            self,
            f"all_{name_prefix}_qzeros",
            torch.nn.Parameter(
                torch.stack([getattr(expert, projection_name).qzeros for expert in self.experts], dim=0).reshape(
                    -1, out_features, in_features // (self.group_size * 2)
                ),
                requires_grad=False,
            ),
        )

    def _transform_quantized_expert_weights(self) -> None:
        if hasattr(self, "all_gate_qweight"):
            return

        first_expert = _deepseek_first_unfused_expert(self.experts)
        if first_expert is None:
            raise RuntimeError("Quantized DeepSeek experts must use an unfused expert container.")

        projections = tuple(
            getattr(expert, projection_name)
            for expert in self.experts
            for projection_name in ("gate_proj", "up_proj", "down_proj")
        )
        if any(projection.bits != 4 for projection in projections):
            raise NotImplementedError("Quantized DeepSeek experts require 4-bit projections.")
        if any(projection.act_order for projection in projections):
            raise NotImplementedError("Quantized DeepSeek experts with act_order=True are not supported.")

        self.group_size = first_expert.gate_proj.group_size
        if any(projection.group_size != self.group_size for projection in projections):
            raise ValueError("Quantized DeepSeek experts must use one shared group size.")

        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            QEffDeepseekV3MoE._stack_quantized_projection_params(self, projection_name)
        for expert in self.experts:
            delete_module_attrs(expert, "gate_proj", "up_proj", "down_proj")

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            if getattr(self, "_qeff_quantized_experts", False):
                return None
            return self.moe_weights
        if getattr(self, "_qeff_quantized_experts", False):
            QEffDeepseekV3MoE._transform_quantized_expert_weights(self)
            self.weights_transformed = True
            return None
        if hasattr(self.experts, "gate_up_proj"):
            self.moe_weights = build_canonical_expert_weights(
                gate_up=self.experts.gate_up_proj,
                down=self.experts.down_proj,
                fused=True,
                fused_split_dim=1,
                transpose_gate_up=True,
                transpose_down=True,
                clone=True,
            )
            delete_module_attrs(self.experts, "gate_up_proj", "down_proj")
        else:
            self.moe_weights = MoEWeights(
                gate=stack_expert_linears(self.experts, lambda expert: _deepseek_expert_weight(expert.gate_proj)),
                up=stack_expert_linears(self.experts, lambda expert: _deepseek_expert_weight(expert.up_proj)),
                down=stack_expert_linears(self.experts, lambda expert: _deepseek_expert_weight(expert.down_proj)),
            )
            for expert in self.experts:
                delete_module_attrs(expert, "gate_proj", "up_proj", "down_proj")
        self.weights_transformed = True
        return self.moe_weights

    def moe_profile(self) -> MoEProfile:
        return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=self.act_fn))

    def get_supported_moe_flavours(self) -> Tuple[MoEFlavour, ...]:
        if getattr(self, "_qeff_quantized_experts", False):
            return QEffDeepseekV3MoE.quantized_supported_moe_flavours
        return QEffMoEBlockMixin.get_supported_moe_flavours(self)

    def routing_input(self, x: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def route(self, x: torch.Tensor):
        topk_indices, topk_weights, router_logits = _deepseek_route_tokens(self, x)
        return (topk_indices, topk_weights.to(x.dtype)), router_logits

    def apply_shared_experts(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return out + self.shared_experts(residual)


class QEffDeepseekV3DecoderLayer(nn.Module):
    """Adapted DeepseekV3DecoderLayer with batch_index and proper position_ids handling."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        orig_hidden_states = self.input_layernorm(hidden_states)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False
        if cache_compressed:
            hidden_states, self_attn_weights, present_compressed_kvs = self.self_attn.fused_forward(
                hidden_states=orig_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                block_table=block_table,
                slot_id=slot_id,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                compressed_kvs=compressed_kvs,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                mla_absorption=mla_absorption,
                sin_cached=sin_cached,
                cos_cached=cos_cached,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=orig_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                block_table=block_table,
                slot_id=slot_id,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                sin_cached=sin_cached,
                cos_cached=cos_cached,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            if cache_compressed:
                outputs += (present_compressed_kvs,)
            else:
                outputs += (present_key_value,)

        return outputs


class QEffDeepseekV3Model(nn.Module):
    """Adapted DeepseekV3Model with batch_index and QEff rotary embedding."""

    def __qeff_init__(self):
        scaling_factor = self.config.rope_scaling["factor"]
        kwargs = {
            key: self.config.rope_scaling[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in self.config.rope_scaling
        }
        self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
            self.config.torch_dtype,
            self.config.qk_rope_head_dim,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            scaling_factor=scaling_factor,
            base=self.config.rope_theta,
            **kwargs,
        )
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached.to(torch.float16))
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached.to(torch.float16))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        compressed_kvs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Cache) and past_key_values is not None:
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        if cache_compressed:
            compressed_kvs = QEffDynamicCompressedKVRopeCache.from_legacy_cache(compressed_kvs)
            target_len = compressed_kvs.layers[0].ckv.shape[-2]
        else:
            target_len = past_key_values[0][0].shape[2]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_len)
        hidden_states = inputs_embeds
        position_embeddings = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        sin = self.sin_cached[position_ids].unsqueeze(1)
        cos = self.cos_cached[position_ids].unsqueeze(1)
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                block_table=block_table,
                slot_id=slot_id,
                compressed_kvs=compressed_kvs,
                past_key_value=past_key_values,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                mla_absorption=mla_absorption,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QEffDeepseekV3ForCausalLM(nn.Module):
    """Adapted DeepseekV3ForCausalLM with batch_index and QEff optimizations."""

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.layers[0].__class__}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        slot_id: Optional[torch.LongTensor] = None,
        compressed_kvs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mla_absorption = getattr(self, "mla_absorption", None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            block_table=block_table,
            slot_id=slot_id,
            compressed_kvs=compressed_kvs,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            mla_absorption=mla_absorption,
            **kwargs,
        )

        hidden_states = outputs[0]
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        mla_absorption = getattr(self, "mla_absorption", None)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        dummy_cache = [[] for _ in range(config.num_hidden_layers)]
        if cache_compressed:
            for layer in self.model.layers:
                if layer is not None:
                    num_heads = layer.self_attn.kv_a_proj_with_mqa.weight.shape[0] // (
                        self.model.config.kv_lora_rank + config.qk_rope_head_dim
                    )
            cache_shape_1 = (batch_size, num_heads, seq_len, config.kv_lora_rank)
            cache_shape_2 = (batch_size, num_heads, seq_len, config.qk_rope_head_dim)
        else:
            cache_shape_1 = (
                batch_size,
                config.num_attention_heads,
                seq_len,
                config.qk_nope_head_dim + config.qk_rope_head_dim,
            )
            cache_shape_2 = (batch_size, config.num_attention_heads, seq_len, config.v_head_dim)

        for i in range(config.num_hidden_layers):
            dummy_cache[i].append(torch.zeros(cache_shape_1, dtype=config.torch_dtype))
            dummy_cache[i].append(torch.zeros(cache_shape_2, dtype=config.torch_dtype))

        return dummy_cache
