# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""DSpark draft backbone and checkpoint-compatible Markov/confidence heads."""

from types import MethodType
from typing import ClassVar

import torch
from torch import nn
from transformers import AutoConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextMLP,
    Gemma4TextScaledWordEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
)

from QEfficient.customop import CustomRMSNormAIC
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.models.dflash_draft.modeling_dflash_draft import (
    QEffDFlashAttention,
    QEffDFlashDecoderLayer,
    QEffQwen3RotaryEmbedding,
    eager_attention_forward,
    qeff_apply_rope_two_streams,
)
from QEfficient.transformers.models.gemma4.modeling_gemma4 import (
    QEffGemma4CustomRMSNormAIC,
    QEffGemma4TextRotaryEmbedding,
    qeff_apply_rotary_pos_emb,
)
from QEfficient.utils.checkpoint_utils import load_checkpoint_weights
from QEfficient.utils.constants import _DFLASH_TARGET_ABSMAX


class QEffDSparkRMSNorm(Qwen3RMSNorm):
    """Keep DSpark RMSNorm stable when the compiler lowers arithmetic to FP16."""

    def forward(self, hidden_states):
        scale = hidden_states.abs().amax(dim=-1, keepdim=True) + self.variance_epsilon**0.5
        hidden_states = hidden_states / scale
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        epsilon = self.variance_epsilon / scale / scale
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return self.weight * hidden_states


def _dspark_config_value(config, name, default=None):
    value = getattr(config, name, None)
    if value is not None:
        return value
    for config_name in ("dspark_config", "dflash_config"):
        nested_config = getattr(config, config_name, None) or {}
        if name in nested_config:
            return nested_config[name]
    return default


class VanillaMarkov(nn.Module):
    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def apply_step_logits(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return logits + self.markov_w2(self.get_prev_embeddings(token_ids))


class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def dspark_target_causal_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    comp_ctx_lengths=None,
    batch_index=None,
    inputs_embeds=None,
    use_cache=None,
    output_hidden_states=None,
    cache_position=None,
    cache_commit=None,
    num_logits_to_keep=None,
    logits_to_keep=0,
    **kwargs,
):
    return self._dspark_original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        comp_ctx_lengths=comp_ctx_lengths,
        batch_index=batch_index,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        cache_commit=cache_commit,
        num_logits_to_keep=num_logits_to_keep,
        logits_to_keep=logits_to_keep,
        **kwargs,
    )


class DSparkTLMTransform:
    @classmethod
    def apply(cls, model, qaic_config=None, **kwargs):
        if not (qaic_config and qaic_config.get("dspark_tlm")):
            return model, False
        if qaic_config.get("include_sampler") or qaic_config.get("speculative_model_type"):
            raise NotImplementedError(
                "DSpark target verification supports greedy predictions without generic sampling."
            )
        config = getattr(model.config, "text_config", model.config)
        if not ("qwen3" in config.model_type or "gemma4" in config.model_type):
            raise NotImplementedError("DSpark target verification supports Qwen3, Qwen3.5, and Gemma4 targets.")
        target_layer_ids = list(qaic_config.get("target_layer_ids") or [])
        if not target_layer_ids or target_layer_ids != sorted(set(target_layer_ids)):
            raise ValueError("DSpark target_layer_ids must be nonempty, sorted, unique hidden-state tuple indices.")
        if target_layer_ids[0] < 1 or target_layer_ids[-1] > config.num_hidden_layers:
            raise ValueError("DSpark target_layer_ids must select decoder outputs within the target model.")
        draft_block_size = int(qaic_config.get("dspark_block_size", 7))
        verify_width = draft_block_size + 1
        if draft_block_size < 1 or verify_width & (verify_width - 1):
            raise ValueError("DSpark verification width (block_size + 1) must be a power of two.")
        inner = getattr(model.model, "language_model", model.model)
        reference_weight = inner.embed_tokens.weight
        module_device = reference_weight.device
        module_dtype = reference_weight.dtype
        feature_size = len(target_layer_ids) * config.hidden_size
        inner.fc = nn.Linear(feature_size, config.hidden_size, bias=False).to(device=module_device, dtype=module_dtype)
        inner.hidden_norm = CustomRMSNormAIC(config.hidden_size, eps=config.rms_norm_eps).to(
            device=module_device, dtype=module_dtype
        )
        draft_repo = qaic_config.get("dspark_dlm_repo")
        if draft_repo:
            weights = load_checkpoint_weights(draft_repo, {"fc.weight", "hidden_norm.weight"})
            expected_fc_shape = (config.hidden_size, feature_size)
            if tuple(weights.get("fc.weight", torch.empty(0)).shape) != expected_fc_shape:
                raise ValueError(f"DSpark fc.weight must have shape {expected_fc_shape}.")
            if tuple(weights.get("hidden_norm.weight", torch.empty(0)).shape) != (config.hidden_size,):
                raise ValueError(f"DSpark hidden_norm.weight must have shape {(config.hidden_size,)}.")
            with torch.no_grad():
                inner.fc.weight.copy_(weights["fc.weight"].to(device=module_device, dtype=module_dtype))
                inner.hidden_norm.weight.copy_(
                    weights["hidden_norm.weight"].to(device=module_device, dtype=module_dtype)
                )
                bound = (inner.fc.in_features**0.5) * inner.fc.weight.float().norm(dim=1).max().item()
                inner.fc.weight.div_(max(bound / _DFLASH_TARGET_ABSMAX, 1.0))
        inner.target_layer_ids = target_layer_ids
        inner.dflash_chunk_size = verify_width
        inner.dspark_fused_replay = config.model_type == "qwen3_5_text"
        if (
            config.model_type == "qwen3_5_text"
            and not hasattr(model.config, "text_config")
            and not hasattr(model, "_dspark_original_forward")
        ):
            model._dspark_original_forward = model.forward
            model.forward = MethodType(dspark_target_causal_forward, model)
        return model, True


class QEffDSparkAttention(QEffDFlashAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm.__class__ = QEffDSparkRMSNorm
        self.k_norm.__class__ = QEffDSparkRMSNorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids_target: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ):
        batch_size, query_length = hidden_states.shape[:2]
        context_length = target_hidden.shape[1]
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(batch_size, query_length, -1, self.head_dim)
        ).transpose(1, 2)
        context_keys = self.k_norm(
            self.k_proj(target_hidden).view(batch_size, context_length, -1, self.head_dim)
        ).transpose(1, 2)
        noise_keys = self.k_norm(
            self.k_proj(hidden_states).view(batch_size, query_length, -1, self.head_dim)
        ).transpose(1, 2)
        context_values = self.v_proj(target_hidden).view(batch_size, context_length, -1, self.head_dim).transpose(1, 2)
        noise_values = self.v_proj(hidden_states).view(batch_size, query_length, -1, self.head_dim).transpose(1, 2)
        query_states, context_keys, noise_keys = qeff_apply_rope_two_streams(
            query_states,
            context_keys,
            noise_keys,
            cos_cached,
            sin_cached,
            position_ids_target.clamp(min=0),
            position_ids.clamp(min=0),
        )
        if past_key_value is None:
            key_states = torch.cat((context_keys, noise_keys), dim=2)
            value_states = torch.cat((context_values, noise_values), dim=2)
        else:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids_target,
            }
            past_key_value.write_only(context_keys, context_values, self.layer_idx, cache_kwargs)
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[..., : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_value.update(noise_keys, noise_values, self.layer_idx, cache_kwargs)
        attention_output, attention_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, self.scaling
        )
        attention_output = attention_output.reshape(batch_size, query_length, -1)
        return self.o_proj(attention_output), attention_weights


class QEffDSparkDecoderLayer(QEffDFlashDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm.__class__ = QEffDSparkRMSNorm
        self.post_attention_layernorm.__class__ = QEffDSparkRMSNorm
        self.self_attn = QEffDSparkAttention(config, layer_idx)


class DSparkDraftModel(Qwen3PreTrainedModel):
    """Load the flat DSpark checkpoint without requiring SpecForge or remote code."""

    config_class = Qwen3Config
    base_model_prefix = ""
    _no_split_modules: ClassVar[list[str]] = ["QEffDSparkDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.target_layer_ids = _dspark_config_value(config, "target_layer_ids")
        if not self.target_layer_ids:
            raise ValueError("DSpark requires nonempty target_layer_ids.")
        if _dspark_config_value(config, "projector_type", "dspark") != "dspark":
            raise ValueError("DSpark requires projector_type='dspark'.")
        if _dspark_config_value(config, "attention_mode", "gqa") != "gqa":
            raise NotImplementedError("DSpark currently supports GQA attention only.")
        target_hidden_size = _dspark_config_value(config, "target_hidden_size", config.hidden_size)
        self.raw_target_feature_size = len(self.target_layer_ids) * target_hidden_size
        self.target_feature_size = config.hidden_size
        self.block_size = int(config.block_size)
        if self.block_size <= 0:
            raise ValueError("DSpark block_size must be positive.")
        self.mask_token_id = _dspark_config_value(config, "mask_token_id")
        self.layers = nn.ModuleList(
            QEffDSparkDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = QEffDSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = QEffQwen3RotaryEmbedding(config)
        self.register_buffer(
            "cos_cached",
            self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling,
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling,
            persistent=False,
        )
        markov_rank = int(_dspark_config_value(config, "markov_rank", 0))
        if markov_rank < 0:
            raise ValueError("DSpark markov_rank must be nonnegative.")
        if markov_rank and _dspark_config_value(config, "markov_head_type", "vanilla") != "vanilla":
            raise NotImplementedError("DSpark currently supports the vanilla Markov head only.")
        self.markov_head = VanillaMarkov(config.vocab_size, markov_rank) if markov_rank else None
        self.confidence_head_with_markov = bool(_dspark_config_value(config, "confidence_head_with_markov", False))
        self.confidence_head = None
        if _dspark_config_value(config, "enable_confidence_head", False):
            if self.confidence_head_with_markov and self.markov_head is None:
                raise ValueError("A Markov-conditioned confidence head requires markov_rank > 0.")
            input_dim = config.hidden_size + (markov_rank if self.confidence_head_with_markov else 0)
            self.confidence_head = AcceptRatePredictor(input_dim)
        self.post_init()

    def _refresh_rotary_cache(self):
        self.rotary_emb._set_cos_sin_cache(
            seq_len=self.rotary_emb.original_max_seq_len,
            device=self.rotary_emb.inv_freq.device,
            dtype=self.norm.weight.dtype,
        )
        self.cos_cached = self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling
        self.sin_cached = self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling

    def predict_confidence_step(self, hidden_states, prev_token_ids=None):
        if self.confidence_head is None:
            return None
        features = hidden_states
        if self.confidence_head_with_markov:
            if prev_token_ids is None:
                raise ValueError("prev_token_ids is required for Markov-conditioned confidence.")
            previous_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(hidden_states.dtype)
            features = torch.cat((features, previous_embeddings), dim=-1)
        return self.confidence_head(features).float()

    def forward(
        self,
        target_hidden: torch.Tensor,
        noise_embeds: torch.Tensor,
        position_ids_target: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        comp_ctx_lengths=None,
        batch_index=None,
        use_cache=None,
        **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_legacy_cache = use_cache and not isinstance(past_key_values, Cache)
        if use_cache:
            if past_key_values is None:
                raise ValueError("DSpark retained-cache inference requires a preallocated past_key_values cache.")
            if return_legacy_cache:
                past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
            if not isinstance(past_key_values, QEffDynamicCache):
                raise TypeError("DSpark requires QEffDynamicCache or a legacy tuple cache.")
            target_length = past_key_values.get_seq_length()
            key_positions = torch.arange(target_length, device=position_ids.device).unsqueeze(0)
        else:
            past_key_values = None
            key_positions = torch.cat((position_ids_target, position_ids), dim=-1)
        last_position = position_ids.max(dim=-1, keepdim=True).values
        mask = ((key_positions < 0) | (key_positions > last_position)).unsqueeze(1).unsqueeze(1)
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool or attention_mask.ndim != 4:
                raise ValueError("DSpark attention_mask must be a 4-D boolean mask, with True denoting masked keys.")
            mask = mask | attention_mask
        if target_hidden.shape[-1] != self.target_feature_size:
            raise ValueError(
                f"DSpark target_hidden must contain projected features of width {self.target_feature_size}."
            )
        target_hidden = target_hidden.to(noise_embeds.dtype)
        hidden_states = noise_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                target_hidden=target_hidden,
                position_ids_target=position_ids_target,
                position_ids=position_ids,
                attention_mask=mask,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                sin_cached=self.sin_cached.to(hidden_states.dtype),
                cos_cached=self.cos_cached.to(hidden_states.dtype),
            )
        hidden_states = self.norm(hidden_states)
        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def _gemma_norm(dim, eps, *, with_scale=True):
    norm = Gemma4RMSNorm(dim, eps=eps, with_scale=with_scale)
    norm.__class__ = QEffGemma4CustomRMSNormAIC
    if not with_scale:
        norm.register_buffer("_qeff_unit_weight", torch.ones(dim))
    return norm


class QEffDSparkGemmaAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.global_head_dim
        self.use_alternative_attention = bool(config.attention_k_eq_v)
        self.num_key_value_heads = (
            config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = (
            None
            if self.use_alternative_attention
            else nn.Linear(
                config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = _gemma_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _gemma_norm(self.head_dim, config.rms_norm_eps)
        self.v_norm = _gemma_norm(self.head_dim, config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states,
        target_hidden,
        position_ids_target,
        position_ids,
        attention_mask=None,
        past_key_value=None,
        comp_ctx_lengths=None,
        batch_index=None,
        cos_cached=None,
        sin_cached=None,
        **kwargs,
    ):
        batch_size, query_length = hidden_states.shape[:2]
        context_length = target_hidden.shape[1]
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(batch_size, query_length, self.num_attention_heads, self.head_dim)
        )
        context_keys = self.k_proj(target_hidden)
        noise_keys = self.k_proj(hidden_states)
        if self.use_alternative_attention:
            context_values = context_keys
            noise_values = noise_keys
        else:
            context_values = self.v_proj(target_hidden)
            noise_values = self.v_proj(hidden_states)
        context_keys = self.k_norm(
            context_keys.view(batch_size, context_length, self.num_key_value_heads, self.head_dim)
        )
        noise_keys = self.k_norm(noise_keys.view(batch_size, query_length, self.num_key_value_heads, self.head_dim))
        context_values = self.v_norm(
            context_values.view(batch_size, context_length, self.num_key_value_heads, self.head_dim)
        )
        noise_values = self.v_norm(noise_values.view(batch_size, query_length, self.num_key_value_heads, self.head_dim))
        query_cos = cos_cached[position_ids.clamp(min=0)].unsqueeze(2)
        query_sin = sin_cached[position_ids.clamp(min=0)].unsqueeze(2)
        context_cos = cos_cached[position_ids_target.clamp(min=0)].unsqueeze(2)
        context_sin = sin_cached[position_ids_target.clamp(min=0)].unsqueeze(2)
        query_states = qeff_apply_rotary_pos_emb(query_states, query_cos, query_sin).transpose(1, 2)
        context_keys = qeff_apply_rotary_pos_emb(context_keys, context_cos, context_sin).transpose(1, 2)
        noise_keys = qeff_apply_rotary_pos_emb(noise_keys, query_cos, query_sin).transpose(1, 2)
        context_values = context_values.transpose(1, 2)
        noise_values = noise_values.transpose(1, 2)
        if past_key_value is None:
            key_states = torch.cat((context_keys, noise_keys), dim=2)
            value_states = torch.cat((context_values, noise_values), dim=2)
        else:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids_target,
            }
            past_key_value.write_only(context_keys, context_values, self.layer_idx, cache_kwargs)
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[..., : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_value.update(
                noise_keys,
                noise_values,
                self.layer_idx,
                cache_kwargs,
            )
        attention_output, attention_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
        )
        attention_output = attention_output.reshape(batch_size, query_length, -1)
        return self.o_proj(attention_output), attention_weights


class QEffDSparkGemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        if config.enable_moe_block or config.hidden_size_per_layer_input:
            raise NotImplementedError("Gemma4 DSpark supports dense drafts without per-layer input gates.")
        self.self_attn = QEffDSparkGemmaAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, **kwargs)[0]
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states * self.layer_scalar


class DSparkGemmaDraftModel(Gemma4PreTrainedModel):
    config_class = None
    base_model_prefix = ""
    _no_split_modules: ClassVar[list[str]] = ["QEffDSparkGemmaDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.target_layer_ids = _dspark_config_value(config, "target_layer_ids")
        if not self.target_layer_ids:
            raise ValueError("DSpark requires nonempty target_layer_ids.")
        self.raw_target_feature_size = len(self.target_layer_ids) * config.hidden_size
        self.target_feature_size = config.hidden_size
        self.block_size = int(config.block_size)
        self.mask_token_id = _dspark_config_value(config, "mask_token_id")
        self.layers = nn.ModuleList(
            QEffDSparkGemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = _gemma_norm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = QEffGemma4TextRotaryEmbedding(config)
        markov_rank = int(_dspark_config_value(config, "markov_rank", 0))
        if markov_rank and _dspark_config_value(config, "markov_head_type", "vanilla") != "vanilla":
            raise NotImplementedError("DSpark currently supports the vanilla Markov head only.")
        self.markov_head = VanillaMarkov(config.vocab_size, markov_rank) if markov_rank else None
        self.confidence_head_with_markov = bool(_dspark_config_value(config, "confidence_head_with_markov", False))
        self.confidence_head = None
        if _dspark_config_value(config, "enable_confidence_head", False):
            input_dim = config.hidden_size + (markov_rank if self.confidence_head_with_markov else 0)
            self.confidence_head = AcceptRatePredictor(input_dim)
        self.post_init()

    def predict_confidence_step(self, hidden_states, prev_token_ids=None):
        if self.confidence_head is None:
            return None
        features = hidden_states
        if self.confidence_head_with_markov:
            if prev_token_ids is None:
                raise ValueError("prev_token_ids is required for Markov-conditioned confidence.")
            features = torch.cat(
                (
                    features,
                    self.markov_head.get_prev_embeddings(prev_token_ids).to(hidden_states.dtype),
                ),
                dim=-1,
            )
        return self.confidence_head(features).float()

    def forward(
        self,
        target_hidden,
        noise_embeds,
        position_ids_target,
        position_ids,
        attention_mask=None,
        past_key_values=None,
        comp_ctx_lengths=None,
        batch_index=None,
        use_cache=None,
        **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_legacy_cache = use_cache and not isinstance(past_key_values, Cache)
        if use_cache:
            if past_key_values is None:
                raise ValueError("DSpark retained-cache inference requires a preallocated past_key_values cache.")
            if return_legacy_cache:
                past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
            target_length = past_key_values.get_seq_length()
            key_positions = torch.arange(target_length, device=position_ids.device).unsqueeze(0)
        else:
            past_key_values = None
            key_positions = torch.cat((position_ids_target, position_ids), dim=-1)
        last_position = position_ids.max(dim=-1, keepdim=True).values
        mask = ((key_positions < 0) | (key_positions > last_position)).unsqueeze(1).unsqueeze(1)
        if attention_mask is not None:
            mask = mask | attention_mask
        target_hidden = target_hidden.to(noise_embeds.dtype)
        hidden_states = noise_embeds
        cos_cached = self.rotary_emb.full_attention_cos_cached
        sin_cached = self.rotary_emb.full_attention_sin_cached
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                target_hidden=target_hidden,
                position_ids_target=position_ids_target,
                position_ids=position_ids,
                attention_mask=mask,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                cos_cached=cos_cached.to(hidden_states.dtype),
                sin_cached=sin_cached.to(hidden_states.dtype),
            )
        hidden_states = self.norm(hidden_states)
        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class QEffDSparkForCausalLM(PreTrainedModel):
    """DSpark with target embeddings/head and sequential greedy Markov proposals."""

    def __init__(self, config, draft_model=None, embed_tokens=None, lm_head=None):
        config._attn_implementation_internal = "eager"
        super().__init__(config)
        is_gemma = "gemma4" in config.model_type
        draft_class = DSparkGemmaDraftModel if is_gemma else DSparkDraftModel
        self.model = draft_model if draft_model is not None else draft_class(config)
        if embed_tokens is None:
            if is_gemma:
                embed_tokens = Gemma4TextScaledWordEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                    config.pad_token_id,
                    embed_scale=config.hidden_size**0.5,
                )
            else:
                embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.embed_tokens = embed_tokens
        self.lm_head = lm_head if lm_head is not None else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.return_token_ids = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, tlm_repo=None, **kwargs):
        if not tlm_repo:
            raise ValueError("DSpark checkpoints require qaic_config['dspark_tlm_repo'] for target embeddings/lm_head.")
        config = kwargs.pop("config", None) or AutoConfig.from_pretrained(pretrained_model_name_or_path)
        is_gemma = "gemma4" in config.model_type
        draft_class = DSparkGemmaDraftModel if is_gemma else DSparkDraftModel
        draft_model = draft_class.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            config=config,
            **kwargs,
        )
        weights = load_checkpoint_weights(
            tlm_repo,
            {
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            },
        )
        embedding_weight = weights.get("model.embed_tokens.weight")
        if embedding_weight is None:
            embedding_weight = weights.get("model.language_model.embed_tokens.weight")
        head_weight = weights.get("lm_head.weight", embedding_weight)
        expected_shape = (draft_model.config.vocab_size, draft_model.config.hidden_size)
        if embedding_weight is None or head_weight is None:
            raise ValueError(f"Target checkpoint {tlm_repo!r} is missing embeddings or lm_head weights.")
        if tuple(embedding_weight.shape) != expected_shape or tuple(head_weight.shape) != expected_shape:
            raise ValueError(f"DSpark target embedding/lm_head weights must have shape {expected_shape}.")
        device = draft_model.norm.weight.device
        dtype = draft_model.norm.weight.dtype
        if hasattr(draft_model, "_refresh_rotary_cache"):
            draft_model._refresh_rotary_cache()
        if is_gemma:
            embed_tokens = Gemma4TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                embed_scale=config.hidden_size**0.5,
            )
            embed_tokens.weight = nn.Parameter(
                embedding_weight.to(device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            embed_tokens = nn.Embedding.from_pretrained(embedding_weight.to(device=device, dtype=dtype))
        head_bias = weights.get("lm_head.bias")
        lm_head = nn.Linear(
            *reversed(expected_shape),
            bias=head_bias is not None,
            device=device,
            dtype=dtype,
        )
        lm_head.weight = nn.Parameter(head_weight.to(device=device, dtype=dtype), requires_grad=False)
        if head_bias is not None:
            lm_head.bias = nn.Parameter(head_bias.to(device=device, dtype=dtype), requires_grad=False)
        return cls(draft_model.config, draft_model, embed_tokens, lm_head).eval()

    def forward(
        self,
        input_ids=None,
        target_hidden=None,
        position_ids_target=None,
        position_ids=None,
        prev_token_ids=None,
        noise_embeds=None,
        past_key_values=None,
        attention_mask=None,
        comp_ctx_lengths=None,
        batch_index=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) == (noise_embeds is None):
            raise ValueError("Specify exactly one of input_ids or noise_embeds.")
        if noise_embeds is None:
            noise_embeds = self.model.embed_tokens(input_ids)
        if noise_embeds.shape[1] != self.model.block_size:
            raise ValueError(f"DSpark greedy drafting requires seq_len == block_size ({self.model.block_size}).")
        if self.model.markov_head is not None and prev_token_ids is None:
            raise ValueError("prev_token_ids [batch_size, 1] is required for DSpark Markov drafting.")
        if prev_token_ids is not None and prev_token_ids.shape != (
            noise_embeds.shape[0],
            1,
        ):
            raise ValueError("DSpark prev_token_ids must have shape [batch_size, 1].")
        outputs = self.model(
            target_hidden=target_hidden,
            noise_embeds=noise_embeds,
            position_ids_target=position_ids_target,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        softcap = getattr(self.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        if self.model.markov_head is not None:
            previous_tokens = prev_token_ids[:, 0]
            if self.return_token_ids:
                token_ids = []
                for step_index in range(self.model.block_size):
                    corrected_logits = self.model.markov_head.apply_step_logits(logits[:, step_index], previous_tokens)
                    previous_tokens = corrected_logits.argmax(dim=-1)
                    token_ids.append(previous_tokens)
                logits = torch.stack(token_ids, dim=1)
            else:
                logits = logits.float()
                step_logits = []
                for step_index in range(self.model.block_size):
                    corrected_logits = self.model.markov_head.apply_step_logits(logits[:, step_index], previous_tokens)
                    step_logits.append(corrected_logits)
                    previous_tokens = corrected_logits.argmax(dim=-1)
                logits = torch.stack(step_logits, dim=1)
        elif self.return_token_ids:
            logits = logits.argmax(dim=-1)
        else:
            logits = logits.float()
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)
