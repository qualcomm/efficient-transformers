# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

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

from QEfficient.blocking.attention_blocking import generic_blocked_attention_interface
from QEfficient.blocking.glm_attention import blocked_glm_dsa_topk, glm_attention_strategy
from QEfficient.customop import ctx_gather_3d, ctx_scatter_3d
from QEfficient.transformers.cache_utils import QEffDynamicCompressedKVRopeCache, glm_dsa_scatter_cache
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
from QEfficient.utils.constants import MAX_POSITION_EMBEDDINGS


@dataclass(frozen=True)
class GlmAttentionLayerConfig:
    attention_type: str
    indexer_type: str
    blocking_mode: str
    absorption: bool
    online: bool
    cache_compressed: bool
    num_kv_blocks: int
    par_num_split: int
    dsa_topk: int
    indexer_dp: int
    indexer_cp: int
    indexer_kvp: int
    attn_dp: int
    attn_cp: int
    attn_kvp: int
    indexer_num_blocks: int
    num_cores_per_device: int
    indexer_local_context: int
    indexer_block_width: int
    indexer_tokens_per_core: int
    indexer_block_topk: int
    attention_tokens_per_core: int

    def to_hash_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_glm_attention_layer_configs(
    config: GlmMoeDsaConfig,
    qaic_config: dict[str, Any] | None,
    *,
    batch_size: int,
    context_length: int,
    num_devices: int,
    num_cores: int,
    seq_len: int = 1,
    prefill_only: bool = False,
) -> tuple[GlmAttentionLayerConfig, ...]:
    """Resolve and validate the production attention plan for every GLM layer."""
    qaic_config = qaic_config or {}
    layer_types = list(getattr(config, "layer_types", []) or [])
    if not layer_types:
        layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
    if len(layer_types) < config.num_hidden_layers:
        raise ValueError("config.layer_types must contain one entry per GLM attention layer.")
    indexer_types = list(getattr(config, "indexer_types", []) or ["full"] * config.num_hidden_layers)
    if len(indexer_types) < config.num_hidden_layers:
        raise ValueError("config.indexer_types must contain one entry per GLM attention layer.")

    blocking_mode = str(qaic_config.get("blocking_mode", "none") or "none")
    allowed_dense_modes = {"none", "par", "prefill_par", "prefill_par_online"}
    if blocking_mode not in allowed_dense_modes:
        raise ValueError(
            f"GLM dense MLA does not support blocking_mode={blocking_mode!r}; "
            f"expected one of {sorted(allowed_dense_modes)}."
        )
    absorption_config = qaic_config.get("mla_absorption") or {}
    absorption = bool(absorption_config.get("absorption", False))
    online = bool(absorption_config.get("online", False))
    cache_compressed = bool(absorption_config.get("cache_compressed", True))
    if online and not absorption:
        raise ValueError("GLM online MLA requires mla_absorption['absorption']=True.")
    if blocking_mode == "prefill_par_online" and not online:
        raise ValueError("blocking_mode='prefill_par_online' requires online MLA absorption.")
    if blocking_mode == "par" and seq_len != 1:
        raise ValueError("GLM blocking_mode='par' is a decode-only topology and requires seq_len=1.")
    if blocking_mode in {"prefill_par", "prefill_par_online"}:
        if not prefill_only or seq_len <= 1:
            raise ValueError(
                f"GLM blocking_mode={blocking_mode!r} requires prefill_only=True and a multi-token seq_len."
            )
        if not cache_compressed:
            raise ValueError(f"GLM blocking_mode={blocking_mode!r} requires compressed MLA cache.")

    explicit_dsa_tuning = any(
        key in qaic_config
        for key in (
            "indexer_dp",
            "indexer_cp",
            "indexer_kvp",
            "attn_dp",
            "attn_cp",
            "attn_kvp",
            "indexer_num_blocks",
            "num_cores_per_device",
        )
    )
    configured_topk = int(getattr(config, "index_topk", 2048))
    dsa_topk = min(configured_topk, context_length)
    indexer_dp = int(qaic_config.get("indexer_dp", 1))
    indexer_cp = int(qaic_config.get("indexer_cp", num_devices if explicit_dsa_tuning else 1))
    indexer_kvp = int(qaic_config.get("indexer_kvp", 1))
    attn_dp = int(qaic_config.get("attn_dp", num_devices if explicit_dsa_tuning else 1))
    attn_cp = int(qaic_config.get("attn_cp", 1))
    attn_kvp = int(qaic_config.get("attn_kvp", 1))
    indexer_num_blocks = int(qaic_config.get("indexer_num_blocks", num_cores if explicit_dsa_tuning else 1))
    num_cores_per_device = int(qaic_config.get("num_cores_per_device", num_cores if explicit_dsa_tuning else 1))
    num_kv_blocks = int(qaic_config.get("num_kv_blocks", 1))
    par_num_split = int(qaic_config.get("par_num_split", num_cores_per_device))

    positive_values = {
        "batch_size": batch_size,
        "context_length": context_length,
        "num_devices": num_devices,
        "num_kv_blocks": num_kv_blocks,
        "par_num_split": par_num_split,
        "dsa_topk": dsa_topk,
        "indexer_dp": indexer_dp,
        "indexer_cp": indexer_cp,
        "attn_dp": attn_dp,
        "attn_cp": attn_cp,
        "indexer_num_blocks": indexer_num_blocks,
        "num_cores_per_device": num_cores_per_device,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"GLM attention values must be positive: {invalid}.")
    if indexer_kvp != 1 or attn_kvp != 1:
        raise ValueError("GLM DSA supports only indexer_kvp=1 and attn_kvp=1.")
    for name, dp, cp in (("indexer", indexer_dp, indexer_cp), ("attention", attn_dp, attn_cp)):
        if explicit_dsa_tuning and dp * cp != num_devices:
            raise ValueError(f"GLM DSA {name}_dp * {name}_cp must equal num_devices ({num_devices}).")
        if batch_size % dp:
            raise ValueError(f"GLM DSA batch_size must be divisible by {name}_dp.")
        if context_length % cp:
            raise ValueError(f"GLM DSA context_length must be divisible by {name}_cp.")
    if dsa_topk > context_length:
        raise ValueError("GLM dsa_topk cannot exceed context_length.")
    if explicit_dsa_tuning and dsa_topk % num_cores_per_device:
        raise ValueError("GLM dsa_topk must be divisible by num_cores_per_device.")
    if num_kv_blocks > context_length:
        raise ValueError("GLM num_kv_blocks cannot exceed context_length.")
    if blocking_mode != "none" and par_num_split > max(1, context_length // num_kv_blocks):
        raise ValueError("GLM par_num_split cannot exceed the dense MLA KV block width.")
    indexer_local_context = context_length // indexer_cp
    if explicit_dsa_tuning and indexer_local_context % (indexer_num_blocks * num_cores_per_device):
        raise ValueError("GLM indexer local context must be divisible by indexer_num_blocks * num_cores_per_device.")
    indexer_block_width = indexer_local_context // indexer_num_blocks
    indexer_tokens_per_core = indexer_block_width // num_cores_per_device
    indexer_block_topk = min(dsa_topk, indexer_block_width)
    attention_tokens_per_core = dsa_topk // num_cores_per_device

    resolved = []
    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_types[layer_idx]
        is_dsa = layer_type == "deepseek_sparse_attention"
        if not is_dsa and layer_type not in {"full_attention", "dense_attention"}:
            raise ValueError(f"Unsupported GLM layer_types[{layer_idx}]={layer_type!r}.")
        indexer_type = indexer_types[layer_idx]
        if is_dsa and indexer_type not in {"full", "shared"}:
            raise ValueError(f"Unsupported GLM indexer_types[{layer_idx}]={indexer_type!r}.")
        resolved.append(
            GlmAttentionLayerConfig(
                attention_type="dsa" if is_dsa else "dense_mla",
                indexer_type=indexer_type,
                blocking_mode="dsa" if is_dsa else blocking_mode,
                absorption=True if is_dsa else absorption,
                online=False if is_dsa else online,
                cache_compressed=True if is_dsa else cache_compressed,
                num_kv_blocks=num_kv_blocks,
                par_num_split=par_num_split,
                dsa_topk=dsa_topk,
                indexer_dp=indexer_dp,
                indexer_cp=indexer_cp,
                indexer_kvp=indexer_kvp,
                attn_dp=attn_dp,
                attn_cp=attn_cp,
                attn_kvp=attn_kvp,
                indexer_num_blocks=indexer_num_blocks,
                num_cores_per_device=num_cores_per_device,
                indexer_local_context=indexer_local_context,
                indexer_block_width=indexer_block_width,
                indexer_tokens_per_core=indexer_tokens_per_core,
                indexer_block_topk=indexer_block_topk,
                attention_tokens_per_core=attention_tokens_per_core,
            )
        )
    return tuple(resolved)


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
    def __init__(self, indexer_key: torch.Tensor, layout_config: GlmAttentionLayerConfig | None = None):
        self.indexer_key = indexer_key
        self.layout_config = layout_config

    def update_indexer(self, indexer_key: torch.Tensor, cache_kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        position_ids = cache_kwargs["position_ids"].to(torch.int32)
        if self.layout_config is not None and (self.layout_config.indexer_dp > 1 or self.layout_config.indexer_cp > 1):
            self.indexer_key = glm_dsa_scatter_cache(
                self.indexer_key,
                position_ids,
                indexer_key,
                dp=self.layout_config.indexer_dp,
                cp=self.layout_config.indexer_cp,
            )
            return self.indexer_key
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
    def __init__(self, full_layer_indices: tuple[int, ...], layer_configs=None):
        self.full_layer_indices = tuple(full_layer_indices)
        self.layer_to_cache_idx = {layer_idx: cache_idx for cache_idx, layer_idx in enumerate(self.full_layer_indices)}
        self.layers: list[QEffDynamicGlmMoeDsaIndexerLayer] = []
        self.layer_configs = layer_configs

    def add_new(self, indexer_key: torch.Tensor, layer_idx: int) -> None:
        layout_config = self.layer_configs[layer_idx] if self.layer_configs is not None else None
        self.layers.append(QEffDynamicGlmMoeDsaIndexerLayer(indexer_key, layout_config))

    @classmethod
    def from_legacy_cache(
        cls,
        indexer_key_cache: list[torch.Tensor] | None,
        full_layer_indices: tuple[int, ...],
        layer_configs=None,
    ) -> "QEffDynamicGlmMoeDsaIndexerCache":
        cache = cls(full_layer_indices, layer_configs)
        if indexer_key_cache is not None:
            for layer_idx, indexer_key in zip(full_layer_indices, indexer_key_cache):
                cache.add_new(indexer_key, layer_idx)
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
            cache_layer = indexer_key_cache.layers[indexer_key_cache.layer_to_cache_idx[self.layer_idx]]
            layer_config = cache_layer.layout_config
            if layer_config is not None and (layer_config.indexer_dp > 1 or layer_config.indexer_cp > 1):
                weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
                weights = weights * (self.n_heads**-0.5)
                return blocked_glm_dsa_topk(
                    q,
                    weights,
                    k,
                    attention_mask,
                    position_ids,
                    scale=self.softmax_scale,
                    dp=layer_config.indexer_dp,
                    cp=layer_config.indexer_cp,
                    num_blocks=layer_config.indexer_num_blocks,
                    num_cores_per_device=layer_config.num_cores_per_device,
                    tokens_per_core=layer_config.indexer_tokens_per_core,
                    block_topk=layer_config.indexer_block_topk,
                    final_topk=layer_config.dsa_topk,
                )
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
    key_shape = (batch_size, seq_length, module.num_heads, module.qk_nope_head_dim + module.v_head_dim)
    kv_nope = kv_nope[:, 0]
    kv_nope = module.kv_b_proj(kv_nope).reshape(key_shape).transpose(1, 2)
    k_nope, value_states = torch.split(kv_nope, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)
    k_rot = k_rot[:, 0].unsqueeze(1)
    k_rot = k_rot.expand(batch_size, module.num_heads, seq_length, module.qk_rope_head_dim)
    key_states = torch.cat((k_nope, k_rot), dim=-1)
    return key_states, value_states


def split_glm_moe_dsa_q_b_proj(
    weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build QEff query projection weights from upstream fused query weight."""
    q_up, q_rope = weight.T.view(-1, num_heads, qk_nope_head_dim + qk_rope_head_dim).split(
        [qk_nope_head_dim, qk_rope_head_dim],
        dim=-1,
    )
    return (
        q_up.reshape(-1, num_heads * qk_nope_head_dim).unsqueeze(0).contiguous(),
        q_rope.reshape(-1, num_heads * qk_rope_head_dim).unsqueeze(0).contiguous(),
    )


def derive_glm_mla_parameters(
    q_b_weight: torch.Tensor,
    kv_b_weight: torch.Tensor,
    *,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
) -> dict[str, torch.Tensor]:
    """Materialize all checkpoint-independent projections used by GLM MLA graphs."""
    q_up, q_rope = split_glm_moe_dsa_q_b_proj(q_b_weight, num_heads, qk_nope_head_dim, qk_rope_head_dim)
    k_up, v_up = kv_b_weight.T.view(-1, num_heads, qk_nope_head_dim + v_head_dim).split(
        [qk_nope_head_dim, v_head_dim], dim=-1
    )
    k_up = k_up.reshape(-1, num_heads * qk_nope_head_dim).unsqueeze(0).contiguous()
    v_up = v_up.reshape(-1, num_heads * v_head_dim).unsqueeze(0).contiguous()
    per_head_q_up = q_up.squeeze(0).view(q_lora_rank, num_heads, qk_nope_head_dim).transpose(0, 1)
    per_head_k_up = k_up.squeeze(0).view(kv_lora_rank, num_heads, qk_nope_head_dim).transpose(0, 1).transpose(1, 2)
    per_head_v_up = v_up.squeeze(0).view(kv_lora_rank, num_heads, v_head_dim).transpose(0, 1)
    return {
        "q_up": q_up,
        "q_rope": q_rope,
        "k_up": k_up,
        "v_up": v_up,
        "per_head_q_up": per_head_q_up.unsqueeze(0).contiguous(),
        "per_head_k_up": per_head_k_up.unsqueeze(0).contiguous(),
        "per_head_k_up_normal": per_head_k_up.transpose(1, 2).unsqueeze(0).contiguous(),
        "per_head_v_up": per_head_v_up.unsqueeze(0).contiguous(),
        "fusedqk": torch.bmm(per_head_q_up, per_head_k_up)
        .reshape(-1, num_heads, q_lora_rank, kv_lora_rank)
        .contiguous(),
    }


class QEffGlmMoeDsaAttention(GlmMoeDsaAttention):
    def __qeff_init__(self):
        derived = derive_glm_mla_parameters(
            self.q_b_proj.weight,
            self.kv_b_proj.weight,
            num_heads=self.num_heads,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
        )
        for name, tensor in derived.items():
            setattr(self, name, nn.Parameter(tensor.detach().clone()))
        layer_types = list(getattr(self.config, "layer_types", []) or [])
        layer_type = layer_types[self.layer_idx] if self.layer_idx < len(layer_types) else "deepseek_sparse_attention"
        indexer_types = list(getattr(self.config, "indexer_types", []) or [])
        indexer_type = indexer_types[self.layer_idx] if self.layer_idx < len(indexer_types) else "full"
        self.glm_attention_config = GlmAttentionLayerConfig(
            attention_type="dsa" if layer_type == "deepseek_sparse_attention" else "dense_mla",
            indexer_type=indexer_type,
            blocking_mode="dsa" if layer_type == "deepseek_sparse_attention" else "none",
            absorption=layer_type == "deepseek_sparse_attention",
            online=False,
            cache_compressed=True,
            num_kv_blocks=1,
            par_num_split=1,
            dsa_topk=int(getattr(self.config, "index_topk", 2048)),
            indexer_dp=1,
            indexer_cp=1,
            indexer_kvp=1,
            attn_dp=1,
            attn_cp=1,
            attn_kvp=1,
            indexer_num_blocks=1,
            num_cores_per_device=1,
            indexer_local_context=int(getattr(self.config, "max_position_embeddings", 1)),
            indexer_block_width=int(getattr(self.config, "max_position_embeddings", 1)),
            indexer_tokens_per_core=int(getattr(self.config, "max_position_embeddings", 1)),
            indexer_block_topk=min(
                int(getattr(self.config, "index_topk", 2048)),
                int(getattr(self.config, "max_position_embeddings", 1)),
            ),
            attention_tokens_per_core=int(getattr(self.config, "index_topk", 2048)),
        )
        self.qeff_attention_strategy = glm_attention_strategy

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
        return generic_blocked_attention_interface(
            module=self,
            attention_mask=attention_mask,
            scaling=self.scaling,
            layer_idx=self.layer_idx,
            blocking_config=getattr(self, "attn_blocking_config", None),
            batch_index=batch_index,
            position_ids=position_ids,
            auxiliary_state={
                "hidden_states": hidden_states,
                "position_embeddings": position_embeddings,
                "compressed_kvs": compressed_kvs,
                "indexer_key_cache": indexer_key_cache,
                "prev_topk_indices": prev_topk_indices,
            },
            **kwargs,
        )


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
            layer_configs = tuple(layer.self_attn.glm_attention_config for layer in self.layers)
            compressed_cache = QEffDynamicCompressedKVRopeCache.from_legacy_cache(compressed_kvs, layer_configs)
        if indexer_key_cache is not None:
            layer_configs = tuple(layer.self_attn.glm_attention_config for layer in self.layers)
            indexer_cache = QEffDynamicGlmMoeDsaIndexerCache.from_legacy_cache(
                indexer_key_cache,
                self._qeff_indexer_cache_layers,
                layer_configs,
            )

        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        target_len = inputs_embeds.shape[1]
        if compressed_cache is not None:
            target_len = max(
                layer.ckv.shape[-2]
                * (
                    layer.layout_config.attn_cp
                    if layer.layout_config is not None and layer.layout_config.attention_type == "dsa"
                    else 1
                )
                for layer in compressed_cache.layers
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
        caches = []
        for layer_idx in range(config.num_hidden_layers):
            layer_config = self.model.layers[layer_idx].self_attn.glm_attention_config
            if layer_config.attention_type == "dsa" and (layer_config.attn_dp > 1 or layer_config.attn_cp > 1):
                prefix = (batch_size // layer_config.attn_dp, layer_config.attn_dp * layer_config.attn_cp)
                local_context = seq_len // layer_config.attn_cp
            else:
                prefix = (batch_size, 1)
                local_context = seq_len
            caches.append(
                (
                    torch.zeros((*prefix, local_context, config.kv_lora_rank), dtype=config.torch_dtype),
                    torch.zeros((*prefix, local_context, config.qk_rope_head_dim), dtype=config.torch_dtype),
                )
            )
        return tuple(caches)

    def get_dummy_indexer_cache(self, config, batch_size, seq_len):
        caches = []
        for layer_idx in self.get_indexer_cache_layers(config):
            layer_config = self.model.layers[layer_idx].self_attn.glm_attention_config
            if layer_config.indexer_dp > 1 or layer_config.indexer_cp > 1:
                shape = (
                    batch_size // layer_config.indexer_dp,
                    layer_config.indexer_dp * layer_config.indexer_cp,
                    seq_len // layer_config.indexer_cp,
                    config.index_head_dim,
                )
            else:
                shape = (batch_size, seq_len, config.index_head_dim)
            caches.append(torch.zeros(shape, dtype=config.torch_dtype))
        return tuple(caches)

    def get_export_example_dimensions(
        self,
        *,
        batch_size: int,
        seq_len: int,
        full_batch_size: int,
        continuous_batching: bool,
    ) -> dict[str, int | bool]:
        """Keep trace sequence length independent from compile-sized retained states."""
        compile_batch_size = int(getattr(self, "_qeff_compile_batch_size", batch_size))
        compile_seq_len = int(getattr(self, "_qeff_compile_seq_len", seq_len))
        compile_context_length = int(getattr(self, "_qeff_compile_context_length", seq_len))
        return {
            "batch_size": max(batch_size, compile_batch_size),
            "full_batch_size": max(full_batch_size, compile_batch_size) if continuous_batching else full_batch_size,
            "seq_len": compile_seq_len,
            "cache_context_length": compile_context_length,
            "dynamic_seq_len": False,
        }

    def get_glm_cache_dynamic_axes(self, continuous_batching: bool = False):
        batch_symbol = "full_batch_size" if continuous_batching else "batch_size"
        compressed_axes = []
        for layer_idx, layer in enumerate(self.model.layers[: self.config.num_hidden_layers]):
            layer_config = layer.self_attn.glm_attention_config
            if layer_config.attention_type == "dsa":
                axes = {}
                if layer_config.attn_dp == 1:
                    axes[0] = batch_symbol
                if layer_config.attn_cp == 1:
                    axes[2] = "ctx_len"
                compressed_axes.append(axes)
            else:
                compressed_axes.append({0: batch_symbol, 2: "ctx_len"})
        indexer_axes = {}
        for layer_idx in self.get_indexer_cache_layers(self.config):
            layer_config = self.model.layers[layer_idx].self_attn.glm_attention_config
            if layer_config.indexer_dp > 1 or layer_config.indexer_cp > 1:
                axes = {}
                if layer_config.indexer_dp == 1:
                    axes[0] = batch_symbol
                if layer_config.indexer_cp == 1:
                    axes[2] = "ctx_len"
                indexer_axes[layer_idx] = axes
            else:
                indexer_axes[layer_idx] = {0: batch_symbol, 1: "ctx_len"}
        return compressed_axes, indexer_axes

    @staticmethod
    def get_indexer_cache_layers(config) -> tuple[int, ...]:
        layer_types = list(getattr(config, "layer_types", []) or [])
        if not layer_types:
            layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
        indexer_types = list(getattr(config, "indexer_types", []) or ["full"] * config.num_hidden_layers)
        return tuple(
            layer_idx
            for layer_idx, (layer_type, indexer_type) in enumerate(
                zip(
                    layer_types[: config.num_hidden_layers],
                    indexer_types[: config.num_hidden_layers],
                )
            )
            if layer_type == "deepseek_sparse_attention" and indexer_type != "shared"
        )
