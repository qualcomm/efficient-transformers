# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaConfig,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaIndexer,
    GlmMoeDsaModel,
    GlmMoeDsaMoE,
    GlmMoeDsaRMSNorm,
    GlmMoeDsaRotaryEmbedding,
    GlmMoeDsaTopkRouter,
    apply_rotary_pos_emb_interleave,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from QEfficient.customop import ctx_gather_3d, ctx_scatter_3d
from QEfficient.transformers.cache_utils import QEffDynamicCompressedKVRopeCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    MoEWeights,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    silu_glu_mlp,
)
from QEfficient.utils.constants import MAX_POSITION_EMBEDDINGS, MIN_MASKED_ATTENTION_VALUE


def _trim_live_context_for_pytorch(
    position_ids: torch.Tensor | None, attention_mask: torch.Tensor | None, *states: torch.Tensor
) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
    if position_ids is None or torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
        return attention_mask, states

    live_context = int(position_ids.max().item()) + 1
    trimmed_states = tuple(state[..., :live_context, :] for state in states)
    if attention_mask is not None:
        attention_mask = attention_mask[..., :live_context]
    return attention_mask, trimmed_states


class QEffDynamicGlmMoeDsaIndexerLayer:
    def __init__(self, indexer_key: torch.Tensor):
        self.indexer_key = indexer_key

    def update_indexer(self, indexer_key: torch.Tensor, cache_kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        position_ids = cache_kwargs["position_ids"].to(torch.int32)
        self.indexer_key = ctx_scatter_3d(self.indexer_key, position_ids, indexer_key)

        ctx_len = self.indexer_key.shape[1]
        ctx_indices = torch.arange(ctx_len, dtype=position_ids.dtype, device=position_ids.device)[None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.to(position_ids.dtype)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
        indexer_key = ctx_gather_3d(self.indexer_key, ctx_indices)
        return torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(indexer_key), indexer_key)


class QEffDynamicGlmMoeDsaIndexerCache:
    def __init__(self, full_layer_indices: tuple[int, ...]):
        self.full_layer_indices = tuple(full_layer_indices)
        self.layer_to_cache_idx = {layer_idx: cache_idx for cache_idx, layer_idx in enumerate(self.full_layer_indices)}
        self.layers: list[QEffDynamicGlmMoeDsaIndexerLayer] = []

    def add_new(self, indexer_key: torch.Tensor) -> None:
        self.layers.append(QEffDynamicGlmMoeDsaIndexerLayer(indexer_key))

    @classmethod
    def from_legacy_cache(
        cls,
        indexer_key_cache: list[torch.Tensor] | None,
        full_layer_indices: tuple[int, ...],
    ) -> "QEffDynamicGlmMoeDsaIndexerCache":
        cache = cls(full_layer_indices)
        if indexer_key_cache is not None:
            for indexer_key in indexer_key_cache:
                cache.add_new(indexer_key)
        return cache

    def update_indexer(
        self,
        indexer_key: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cache_idx = self.layer_to_cache_idx[layer_idx]
        return self.layers[cache_idx].update_indexer(indexer_key, cache_kwargs)

    def to_legacy_cache(self) -> tuple[torch.Tensor, ...]:
        return tuple(layer.indexer_key for layer in self.layers)


class QEffGlmMoeDsaRotaryEmbedding(GlmMoeDsaRotaryEmbedding):
    def __init__(self, config: GlmMoeDsaConfig, device=None):
        super().__init__(config=config)
        self._set_cos_sin_cache(MAX_POSITION_EMBEDDINGS, self.inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        return (
            self.cos_cached[position_ids].to(dtype=x.dtype),
            self.sin_cached[position_ids].to(dtype=x.dtype),
        )


class QEffGlmMoeDsaRMSNorm(GlmMoeDsaRMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # GLM DSA RMSNorm needs FP32 accumulation; lowering this to BF16 causes measurable accuracy deviation.
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QEffGlmMoeDsaIndexer(GlmMoeDsaIndexer):
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        indexer_key_cache: QEffDynamicGlmMoeDsaIndexerCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)

        if indexer_key_cache is not None:
            cache_kwargs = {"position_ids": position_ids}
            k = indexer_key_cache.update_indexer(k, self.layer_idx, cache_kwargs)
            attention_mask, (k,) = _trim_live_context_for_pytorch(position_ids, attention_mask, k)

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        index_scores = torch.where(
            attention_mask,
            torch.full_like(index_scores, float("-inf"), dtype=index_scores.dtype),
            index_scores,
        )
        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)


def _expand_glm_moe_dsa_kv(module: nn.Module, kv_nope: torch.Tensor, k_rot: torch.Tensor):
    batch_size, _, seq_length, _ = kv_nope.shape
    key_shape = (batch_size, seq_length, -1, module.qk_nope_head_dim + module.v_head_dim)
    kv_nope = module.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)
    k_nope, value_states = torch.split(kv_nope, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)
    k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)
    key_states = torch.cat((k_nope, k_rot), dim=-1)
    return key_states, value_states


class QEffGlmMoeDsaAttention(GlmMoeDsaAttention):
    def __qeff_init__(self):
        q_up, q_rope = self.q_b_proj.weight.T.view(-1, self.num_heads, self.qk_head_dim).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        self.q_up = nn.Parameter(q_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0).detach())
        self.q_rope = nn.Parameter(q_rope.reshape(-1, self.num_heads * self.qk_rope_head_dim).unsqueeze(0).detach())

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        compressed_kvs: QEffDynamicCompressedKVRopeCache | None = None,
        indexer_key_cache: QEffDynamicGlmMoeDsaIndexerCache | None = None,
        position_ids: torch.LongTensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        batch_index: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        del kwargs
        batch_size, seq_length = hidden_states.shape[:-1]

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pass = torch.bmm(q_resid, self.q_up)
        q_pass = q_pass.view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        q_rot = torch.bmm(q_resid, self.q_rope)
        q_rot = q_rot.view(batch_size, seq_length, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if compressed_kvs is not None:
            kv_pass = compressed_kvs.update_ckv(kv_pass, self.layer_idx, cache_kwargs)
            k_rot = compressed_kvs.update_k_pe(k_rot, self.layer_idx, cache_kwargs)
            attention_mask, (kv_pass, k_rot) = _trim_live_context_for_pytorch(
                position_ids, attention_mask, kv_pass, k_rot
            )

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states, value_states = _expand_glm_moe_dsa_kv(self, kv_pass, k_rot)

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask[:, 0, :, :],
                position_ids,
                indexer_key_cache=indexer_key_cache,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        index_mask = topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
        index_mask = index_mask.scatter(-1, topk_indices.long(), torch.zeros_like(topk_indices, dtype=torch.bool))
        index_mask = index_mask.unsqueeze(1)
        attention_mask = attention_mask | index_mask
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        mask_value = torch.full_like(attn_weights, MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype)
        attn_weights = torch.where(attention_mask, mask_value, attn_weights)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices


class QEffGlmMoeDsaDecoderLayer(GlmMoeDsaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        compressed_kvs: QEffDynamicCompressedKVRopeCache | None = None,
        indexer_key_cache: QEffDynamicGlmMoeDsaIndexerCache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        batch_index: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compressed_kvs=compressed_kvs,
            indexer_key_cache=indexer_key_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            batch_index=batch_index,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk_indices


class QEffGlmMoeDsaModel(GlmMoeDsaModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffGlmMoeDsaRotaryEmbedding(config=self.config)
        self.sin_cached = nn.Parameter(self.rotary_emb.sin_cached.detach().clone(), requires_grad=False)
        self.cos_cached = nn.Parameter(self.rotary_emb.cos_cached.detach().clone(), requires_grad=False)
        full_layers = []
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if getattr(decoder_layer.self_attn, "indexer", None) is not None:
                full_layers.append(layer_idx)
        self._qeff_indexer_cache_layers = tuple(full_layers)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        compressed_kvs: list[torch.FloatTensor] | None = None,
        indexer_key_cache: list[torch.FloatTensor] | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if compressed_kvs is None and isinstance(past_key_values, (tuple, list)) and len(past_key_values) == 2:
            compressed_kvs, indexer_key_cache = past_key_values

        compressed_cache = None
        indexer_cache = None
        if compressed_kvs is not None:
            compressed_cache = QEffDynamicCompressedKVRopeCache.from_legacy_cache(compressed_kvs)
        if indexer_key_cache is not None:
            indexer_cache = QEffDynamicGlmMoeDsaIndexerCache.from_legacy_cache(
                indexer_key_cache,
                self._qeff_indexer_cache_layers,
            )

        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        target_len = (
            compressed_cache.layers[0].ckv.shape[-2] if compressed_cache is not None else inputs_embeds.shape[1]
        )
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_len)
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].to(torch.bool)
            causal_mask = causal_mask | ~padding_mask

        hidden_states = inputs_embeds
        position_embeddings = (
            self.cos_cached[position_ids].to(dtype=hidden_states.dtype, device=hidden_states.device),
            self.sin_cached[position_ids].to(dtype=hidden_states.dtype, device=hidden_states.device),
        )
        all_hidden_states = () if output_hidden_states else None
        topk_indices = None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                compressed_kvs=compressed_cache,
                indexer_key_cache=indexer_cache,
                batch_index=batch_index,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            compressed_legacy = compressed_cache.to_legacy_cache() if compressed_cache is not None else None
            indexer_legacy = indexer_cache.to_legacy_cache() if indexer_cache is not None else None
            next_cache = (compressed_legacy, indexer_legacy)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
        )


class QEffGlmMoeDsaTopkRouter(GlmMoeDsaTopkRouter):
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.to(device=scores.device)
        group_scores_top2 = scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group).topk(
            2, dim=-1
        )[0]
        group_scores = group_scores_top2.sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask = group_mask.scatter(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class QEffGlmMoeDsaMoE(QEffMoEBlockMixin, GlmMoeDsaMoE):
    supported_moe_flavours = (MoEFlavour.SIMPLE_LOOP, MoEFlavour.DECODE_BMM, MoEFlavour.EXPERT_PARALLEL)

    def __qeff_init__(self):
        QEffMoEBlockMixin.__qeff_init__(self)
        self.act_fn = self.experts.act_fn
        self.num_experts = self.experts.num_experts

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        self.moe_weights = build_canonical_expert_weights(
            gate_up=self.experts.gate_up_proj,
            down=self.experts.down_proj,
            fused=True,
            fused_split_dim=1,
            transpose_gate_up=True,
            transpose_down=True,
            clone=True,
        )
        delete_module_attrs(self, "experts")
        self.weights_transformed = True
        return self.moe_weights

    @property
    def moe_profile(self) -> MoEProfile:
        return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=self.act_fn))

    def route(self, x: torch.Tensor):
        return self.gate(x), None

    def execute_moe_flavour(self, x: torch.Tensor, routing) -> torch.Tensor:
        return super().execute_moe_flavour(x, routing).to(x.dtype)

    def apply_shared_experts(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return out + self.shared_experts(residual.view(out.shape[0], -1)).view_as(out)


class QEffGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    def get_submodules_for_export(self) -> type[nn.Module]:
        return {QEffGlmMoeDsaDecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        compressed_kvs: list[torch.FloatTensor] | None = None,
        indexer_key_cache: list[torch.FloatTensor] | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if position_ids is None:
            seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            position_ids = torch.arange(
                seq_len, device=input_ids.device if input_ids is not None else inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compressed_kvs=compressed_kvs,
            indexer_key_cache=indexer_key_cache,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        export_with_cache_outputs = (
            getattr(self, "_qeff_export_with_cache_outputs", False)
            or torch.onnx.is_in_onnx_export()
            or torch.jit.is_tracing()
        )
        if export_with_cache_outputs:
            pass
        elif isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logit_index = position_ids.to(torch.int64).argmax(1, keepdim=True)
            logit_index = logit_index.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            hidden_states = torch.gather(hidden_states, 1, logit_index)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(hidden_states).to(hidden_states.dtype)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if export_with_cache_outputs and outputs.past_key_values is not None:
            compressed_legacy, indexer_legacy = outputs.past_key_values
            return (logits, compressed_legacy, indexer_legacy)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        num_kv_heads = 1
        cache_shape_1 = (batch_size, num_kv_heads, seq_len, config.kv_lora_rank)
        cache_shape_2 = (batch_size, num_kv_heads, seq_len, config.qk_rope_head_dim)
        return tuple(
            (
                torch.zeros(cache_shape_1, dtype=config.torch_dtype),
                torch.zeros(cache_shape_2, dtype=config.torch_dtype),
            )
            for _ in range(config.num_hidden_layers)
        )

    def get_dummy_indexer_cache(self, config, batch_size, seq_len):
        return tuple(
            torch.zeros((batch_size, seq_len, config.index_head_dim), dtype=config.torch_dtype)
            for layer_idx in self.get_indexer_cache_layers(config)
        )

    @staticmethod
    def get_indexer_cache_layers(config) -> tuple[int, ...]:
        return tuple(
            layer_idx
            for layer_idx, indexer_type in enumerate(config.indexer_types[: config.num_hidden_layers])
            if indexer_type != "shared"
        )
