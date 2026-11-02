# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    Gemma3TextModel,
    repeat_kv,
    rotate_half,
)
from transformers.utils import logging

from QEfficient.customop.rms_norm import CustomRMSNorm
from QEfficient.transformers.cache_utils import QEffSlidingWindowCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

logger = logging.get_logger(__name__)


class GemmaRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        div_first = hidden_states * torch.rsqrt(torch.tensor(hidden_states.shape[-1], dtype=hidden_states.dtype))
        variance = div_first.pow(2).sum(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)


class QEffGemma3CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def forward(self, hidden_states):
        out = GemmaRMSNormFunc.apply(
            hidden_states,
            (self.weight).to(hidden_states.dtype) + 1.0,
            self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps,
        )
        return out.to(hidden_states.dtype)


class QEffGemma3RotaryEmbedding(nn.Module):
    """
    Copied from Gemma2RotaryEmbedding: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, dim, config, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

        if hasattr(config, "rope_scaling") and "factor" in config.rope_scaling:
            factor = config.rope_scaling["factor"]
            inv_freq /= factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=config.torch_dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding, already indexed by position_ids and shaped
            `[batch_size, seq_len, head_dim]`. In Transformers v5+, position_ids are consumed inside
            `Gemma3RotaryEmbedding.forward` via `inv_freq @ position_ids`, so no gather step is needed here.
        sin (`torch.Tensor`): The sine part of the rotary embedding, already indexed by position_ids and shaped
            `[batch_size, seq_len, head_dim]`. Same contract as `cos`.
        position_ids (`torch.Tensor`):
            Retained for API compatibility but not used in this function. Position indexing is performed upstream
            inside `Gemma3RotaryEmbedding.forward` before `cos`/`sin` are passed here.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze `cos` and `sin` so that they broadcast correctly against `q`
            and `k`. For example, if q and k have shape `[batch_size, heads, seq_len, head_dim]`, set
            `unsqueeze_dim=1` so that `cos`/`sin` of shape `[batch_size, 1, seq_len, head_dim]` broadcast
            across the heads dimension. If q and k have shape `[batch_size, seq_len, heads, head_dim]`, set
            `unsqueeze_dim=2`.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # Cast back to original dtype
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=module.config.torch_dtype), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _is_local(layer_idx: int, pattern: int = 6) -> bool:
    return (layer_idx % pattern) != (pattern - 1)


class QEffGemma3Attention(Gemma3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3TextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # Define the general __qeff_init__() for any changes in the init calls
        # Set the init in the module mapping pytorch transforms
        self.__qeff_init__()

    def __qeff_init__(self):
        # In Transformers v4.57, each Gemma3Attention owned its own `rotary_emb` and `rotary_emb_local`
        # instances, and __qeff_init__ replaced them with QEffGemma3RotaryEmbedding
        #
        # In Transformers v5.5+, `rotary_emb` was lifted out of each attention layer and placed as a
        # single shared instance on Gemma3TextModel (see Gemma3TextModel.__init__). It now accepts a
        # `layer_type` argument to handle both "sliding_attention" and "full_attention" in one call.
        # Gemma3Attention no longer owns a `rotary_emb` at all, so there is nothing to replace here.
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )

        cos, sin = position_embeddings
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "batch_index": batch_index,
                "position_ids": position_ids,
                "is_sliding": self.is_sliding,
                "sliding_window_pattern": self.config._sliding_window_pattern,
                "sliding_window": getattr(past_key_values, "sliding_window_len", self.config.sliding_window),
            }
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = torch.where(
                attention_mask.bool(),
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=self.config.torch_dtype),
                attn_weights,
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGemma3DecoderLayer(Gemma3DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # `attention_mask` is built once at the model level in ``QEffGemma3TextModel.forward``
        # and passed in keyed by layer_type (sliding_attention / full_attention).
        # Building it here would emit a ``torch.arange`` inside the decoder-layer subfunction
        # during ONNX subfunction export, which the AIC compiler rejects because the ``Range``
        # ``limit`` input becomes a function parameter rather than a constant.

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class QEffGemma3TextModel(Gemma3TextModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Gemma3TextDecoderLayer`]

    Args:
        config: Gemma3TextConfig
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Convert legacy tuple cache to QEffSlidingWindowCache (QEff ONNX export path).
        # Standard Cache subclasses (e.g. DynamicCache from model.generate()) are left as-is.
        if (
            use_cache
            and past_key_values is not None
            and not isinstance(past_key_values, (Cache, QEffSlidingWindowCache))
        ):
            past_key_values = QEffSlidingWindowCache.from_legacy_cache(
                config=self.config, past_key_values=past_key_values
            )
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if last_cache_position is None:
            last_cache_position = 0
            if attention_mask is not None:
                # In case a 4d mask is passed directly without using `generate`, we have to rely on cache_position
                # It will break dynamo tracing but there are no way around it (and it should never happen in practice)
                last_cache_position = (
                    attention_mask.shape[-1] if attention_mask.dim() == 2 else cache_position[-1].item()
                )

        hidden_states = inputs_embeds

        # Compute position embeddings per layer_type using the single rotary_emb
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Build per-layer-type causal masks using _create_causal_mask for the model.generate() path
        # (DynamicCache from model.generate(), or None on first prefill).
        # For QEffSlidingWindowCache the masks are built here at the model level (keyed by layer_type)
        # so that no torch.arange is emitted inside a decoder-layer subfunction during ONNX export.
        if not isinstance(past_key_values, QEffSlidingWindowCache):
            sliding_window = self.config.sliding_window
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            full_ctx_len = past_seen + inputs_embeds.shape[1]
            _causal_mask_mapping = {
                "full_attention": _create_causal_mask(
                    position_ids=position_ids,
                    target_length=full_ctx_len,
                ),
                "sliding_attention": _create_causal_mask(
                    position_ids=position_ids,
                    target_length=min(sliding_window, full_ctx_len),
                    sliding_window=min(sliding_window, full_ctx_len),
                ),
            }
        else:
            # QEffSlidingWindowCache path (ONNX export / on-device runtime).
            # Replicate the per-layer-type mask logic that previously lived inside
            # QEffGemma3DecoderLayer.forward, but build it once here so that the
            # torch.arange inside _create_causal_mask stays outside any subfunction.
            sliding_window_len = past_key_values.sliding_window_len
            full_ctx_len = past_key_values.key_cache[self.config._sliding_window_pattern - 1].shape[-2]
            _causal_mask_mapping = {
                "sliding_attention": _create_causal_mask(
                    position_ids=position_ids,
                    target_length=sliding_window_len,
                    sliding_window=sliding_window_len,
                ),
                "full_attention": _create_causal_mask(
                    position_ids=position_ids,
                    target_length=full_ctx_len,
                ),
            }

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_type = self.config.layer_types[i]
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=_causal_mask_mapping[layer_type] if _causal_mask_mapping is not None else None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                last_cache_position=last_cache_position,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_cache:
            # DynamicCache (model.generate() path) is returned as-is.
            # QEffSlidingWindowCache (ONNX export path) is converted back to legacy tuple format.
            if isinstance(past_key_values, Cache):
                next_cache = past_key_values
            else:
                next_cache = past_key_values.to_legacy_cache()
        else:
            next_cache = None

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class QEffGemma3ForCausalLMModel(Gemma3ForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the class used as the repeated layer across the text decoder for subfunction extraction.
        """
        return {QEffGemma3DecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

        >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.torch_dtype == torch.float16:
            logger.warning("Accuracy might drop with float16 as torch_dtype")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **loss_kwargs,
        )
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len, dtype=None):
        dtype = dtype or getattr(config, "torch_dtype", torch.float32)
        n_heads = config.num_key_value_heads
        d_head = config.head_dim
        layer_switch = (
            config._sliding_window_pattern if hasattr(config, "_sliding_window_pattern") else 2
        )  # 2 is for BC
        is_sliding = torch.tensor(
            [bool((i + 1) % layer_switch) for i in range(config.num_hidden_layers)], dtype=torch.bool
        )
        global_cache_shape = [batch_size, n_heads, seq_len, d_head]
        if hasattr(config, "sliding_window"):
            sliding_cache_shape = [batch_size, n_heads, min(config.sliding_window, seq_len), d_head]
        past_key_values = []
        cache_shape = global_cache_shape
        for i in range(config.num_hidden_layers):
            if hasattr(config, "sliding_window"):
                cache_shape = global_cache_shape if not is_sliding[i] else sliding_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=dtype)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=dtype)
            pkv = (new_layer_key_cache, new_layer_value_cache)
            past_key_values.append(pkv)
        return past_key_values


class QEffGemma3EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.model.vision_model = self.model.vision_tower

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.vision_tower.vision_model.encoder.layers[0].__class__}

    def forward(self, pixel_values):
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        return image_features


class QEffGemma3DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.language_model
        self.config = self.model.config
        self.lm_head = self.model.lm_head

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QEffGemma3DecoderLayer}

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape
        vision_embeds = vision_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        selected = input_ids == self.model.config.image_token_index
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.shape[0], device=selected.device).view(-1, 1)
        image_features_expanded = vision_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        present = outputs.past_key_values
        if isinstance(present, Cache):
            if hasattr(present, "to_legacy_cache"):
                present = present.to_legacy_cache()
            elif hasattr(present, "layers"):
                legacy_cache = ()
                for layer in present.layers:
                    legacy_cache += ((getattr(layer, "keys", None), getattr(layer, "values", None)),)
                present = legacy_cache
        return logits, vision_embeds, image_idx, present


class QEffGemma3ForConditionalGeneration(Gemma3ForConditionalGeneration):
    def __qeff_init__(self):
        # Module mapping swaps class post-init, so set aliases here.
        self.language_model = self.model.language_model

    def get_qeff_vision_encoder(self):
        return QEffGemma3EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffGemma3DecoderWrapper(self)

    def forward(
        self,
        input_ids,
        position_ids,
        pixel_values,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        image_features = self.get_image_features(pixel_values=pixel_values)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        inputs_embeds = self.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape
        image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        selected = input_ids == self.config.image_token_index
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.shape[0], device=selected.device).view(-1, 1)
        image_features_expanded = image_features.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)
        if past_key_values is not None and not isinstance(past_key_values, (Cache, QEffSlidingWindowCache)):
            past_key_values = QEffSlidingWindowCache.from_legacy_cache(
                config=self.language_model.config, past_key_values=past_key_values
            )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            use_cache=True,
        )
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        present = outputs.past_key_values
        if isinstance(present, Cache):
            if hasattr(present, "to_legacy_cache"):
                present = present.to_legacy_cache()
            elif hasattr(present, "layers"):
                legacy_cache = ()
                for layer in present.layers:
                    legacy_cache += ((getattr(layer, "keys", None), getattr(layer, "values", None)),)
                present = legacy_cache
        return logits, pixel_values, image_idx, present

    def get_npi_file(self, model_name: str) -> str:
        if constants.NPI_MAPPING[model_name] is not None:
            return constants.NPI_MAPPING[model_name]
        else:
            raise ValueError(
                f"For Model {self.pretrained_model_name_or_path} default NPI file is not supported/added for this particular model. Please use one of the following: google/gemma-3-4b-it, google/gemma-3-27b-it"
            )

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        vision_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        # Preserve the legacy shape when callers do not request a separate vision batch.
        vision_batch_size = batch_size if vision_batch_size is None else vision_batch_size
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 896  # FIXME based on gemma3 Image size
            logger.warning("Setting img_size to be 336, as it was neither passed nor found in vision_config")
        user_vision_size = compiler_options.pop("vision_size", None)
        if user_vision_size:
            if user_vision_size >= ctx_len:
                raise ValueError("vision_size must be less than ctx_len")
            vision_size = user_vision_size
        else:
            _mm_tokens_per_image = getattr(self.config, "mm_tokens_per_image", None)
            if _mm_tokens_per_image:
                logger.warning_once("mm_tokens_per_image is deprecated and will be removed in the next release.")
                vision_size = _mm_tokens_per_image
            else:
                vision_size = 256

        vision = [
            {
                "vision_batch_size": vision_batch_size,
                "img_size": img_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
            }
        ]
        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            lang = []

            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                    "sliding_window": self.language_model.config.sliding_window,
                    "img_size": img_size,
                    "vision_size": vision_size,
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
                    "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                    "sliding_window": self.language_model.config.sliding_window,
                    "img_size": img_size,
                    "vision_size": vision_size,
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
                "sliding_window": self.language_model.config.sliding_window,
                "img_size": img_size,
                "vision_size": vision_size,
                "vision_batch_size": vision_batch_size,
            }
            if continuous_batching:
                lang_prefill["full_batch_size"] = kv_cache_batch_size
            else:
                lang_prefill["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                lang_prefill["full_batch_exec_size"] = full_batch_size

            lang_decode = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "sliding_window": self.language_model.config.sliding_window,
                "img_size": img_size,
                "vision_size": vision_size,
                "vision_batch_size": vision_batch_size,
            }
            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size
            else:
                lang_decode["batch_size"] = kv_cache_batch_size
            lang = [lang_prefill, lang_decode]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        # Define dynamic axes
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}
        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["vision_embeds"] = {0: "vision_batch_size", 1: "vision_size"}
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        vision_dynamic_axes["pixel_values"] = {0: "vision_batch_size", 2: "img_size", 3: "img_size"}

        pkv_dynamic_axes = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
        pkv_dynamic_sliding_axes = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
        layer_switch = (
            self.language_model.config._sliding_window_pattern
            if hasattr(self.language_model.config, "_sliding_window_pattern")
            else 2
        )
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                apply_dynamic_axes = (
                    pkv_dynamic_sliding_axes
                    if ((i + 1) % layer_switch and hasattr(self.language_model.config, "_sliding_window_pattern"))
                    else pkv_dynamic_axes
                )
                lang_dynamic_axes[f"past_{kv}.{i}"] = apply_dynamic_axes

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.language_model.config.num_hidden_layers):
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

    def get_dummy_pkv_cache(self, config, batch_size, seq_len, dtype=None):
        dtype = dtype or getattr(config, "torch_dtype", torch.float32)
        n_heads = config.num_key_value_heads
        d_head = config.head_dim
        layer_switch = (
            config._sliding_window_pattern if hasattr(config, "_sliding_window_pattern") else 2
        )  # 2 is for BC
        is_sliding = torch.tensor(
            [bool((i + 1) % layer_switch) for i in range(config.num_hidden_layers)], dtype=torch.bool
        )
        global_cache_shape = [batch_size, n_heads, seq_len, d_head]
        if hasattr(config, "sliding_window"):
            sliding_cache_shape = [batch_size, n_heads, min(config.sliding_window, seq_len), d_head]
        past_key_values = []
        cache_shape = global_cache_shape
        for i in range(config.num_hidden_layers):
            if hasattr(config, "sliding_window"):
                cache_shape = global_cache_shape if not is_sliding[i] else sliding_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=dtype)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=dtype)
            pkv = (new_layer_key_cache, new_layer_value_cache)
            past_key_values.append(pkv)
        return past_key_values

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        prefill_seq_len = kwargs.get("prefill_seq_len")
        if prefill_seq_len is None:
            prefill_seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        prefill_seq_len = int(prefill_seq_len)
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 896)
        else:
            img_size = 896

        _mm_tokens_per_image = getattr(self.config, "mm_tokens_per_image", None)
        if _mm_tokens_per_image:
            logger.warning_once("mm_tokens_per_image is deprecated and will be removed in the next release.")
            vision_size = _mm_tokens_per_image
        else:
            vision_size = 256

        # Define shapes
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, prefill_seq_len)
        inputs_shapes["vision_embeds"] = (
            1,  # constants.INTERN_NUM_PATCHES,
            vision_size,  # constants.INTERN_FEATURE_SIZE,
            self.language_model.config.hidden_size,  # 5120
        )
        inputs_shapes["position_ids"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            prefill_seq_len,
        )
        inputs_shapes["pixel_values"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.INTERN_NUM_CHANNELS,
            img_size,
            img_size,
        )
        inputs_shapes["image_idx"] = (1, 1)

        # Define inputs
        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=self.config.torch_dtype)
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=self.config.torch_dtype)
        lang_inputs["position_ids"] = (
            torch.arange(prefill_seq_len, dtype=torch.int64)
            .view(1, prefill_seq_len)
            .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        # Add data for KV
        lang_inputs["past_key_values"] = self.get_dummy_pkv_cache(
            config=self.language_model.config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=prefill_seq_len,
        )

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)
        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=self.config.torch_dtype,
                shape=("batch_size", 3, "img_size", "img_size"),
            ),
        ]
