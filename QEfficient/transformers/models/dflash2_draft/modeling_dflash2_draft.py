# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""PyTorch DFlash-2 draft model (Qwen3-based diffusion LLM draft for SpD).

Reference implementation: https://github.com/z-lab/dflash/blob/main/dflash/model.py

Relative to the DFlash-1 draft (``dflash_draft``) and the sliding-window DFlash-1 assistant
(``muse_glimmer_assistant``), DFlash-2 adds three things:

1. ``GroupedDynamicCausalConv`` -- a depthwise causal convolution wrapped around BOTH the
   attention and the MLP sublayer, whose tap weights are a static per-channel weight plus a
   per-token/per-group weight predicted from the sublayer input.
2. ``CandidateSelector`` -- a bigram rescore over per-position top-k candidates. At
   ``selector_top_k == 1`` this provably degenerates to a plain argmax (see the class docstring),
   so it is NOT part of the exported graph in that configuration.
3. ``output_multiplier`` / ``final_logit_softcapping`` on the LM head.

It also differs from the DFlash-1 assistant in attention masking: DFlash-2's sliding-attention
layers are *causal within the noise block*, while its full-attention layers are bidirectional.
See ``_create_mask``.
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import onnx
import torch
import torch.utils.checkpoint
import yaml
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RMSNorm,
    repeat_kv,
    rotate_half,
)

from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

# Names given to the convolution submodules on each decoder layer. NOTE: the ONNX exporter does
# not put these names into node names -- matching on them finds ZERO nodes in both the
# Muse-Glimmer and Qwen DFlash-2 exports. The convolutions are located structurally instead, by
# `_overflow_prone_nodes`. Kept only because a future exporter may restore module scopes.
CONV_MODULE_NAMES = ("attention_conv", "mlp_conv")

# Decoder-layer-scope node, i.e. `/model/layers.3/Add_5` but NOT `/model/layers.3/self_attn/Add`.
LAYER_SCOPE_NODE = re.compile(r"^/model/layers\.\d+/[A-Za-z]+(_\d+)?$")
NORM_OP_TYPES = ("CustomRMSNorm",)


def _overflow_prone_nodes(nodes) -> set:
    """Locate the two unnormalised accumulators in the draft, by graph shape not by node index.

    Both carry activations far outside fp16 range while being invisible in the model's output,
    because ``model.norm`` divides the residual back down before the head sees it:

    * **the residual stream** -- a layer-scope ``Add`` whose result feeds a norm. That is exactly
      ``Add_2`` (post-attention, feeds ``post_attention_layernorm``) and ``Add_5`` (post-MLP,
      feeds the next layer's ``input_layernorm``, or ``/model/norm`` for the last layer). The
      layer's other Adds feed the convolutions instead and stay small, so they are left alone.
    * **the dynamic convolution accumulation** -- ``Mul`` (hidden x tap weights) into
      ``ReduceSum`` (sum over K taps) into ``Reshape``. ``finish`` runs on pre-residual,
      unnormalised attention/MLP outputs and its tap weights are a raw unbounded Linear output,
      so the sum over taps is precisely the overflow that silently zeroed the RMSNorm path in
      the DFlash-1 draft. The ``Reshape`` must be pinned too: it is pure data movement, but an
      fp16 Reshape of an fp32 sum re-introduces the overflow before the residual Add consumes it.

    Measured on Qwen3.8-27B-DFlash2 with real TLM hidden states, this set covers every tensor in
    the graph above 65504 (worst: ``layers.3/Add_5`` at 141767 and ``layers.4/ReduceSum_3`` at
    87040) and every tensor above a 4x safety margin, with nothing left over.
    """
    producer = {out: node for node in nodes for out in node.output if out}
    consumers = {}
    for node in nodes:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    picked = set()
    for node in nodes:
        if not LAYER_SCOPE_NODE.match(node.name):
            continue
        if node.op_type == "Add" and any(
            consumer.op_type in NORM_OP_TYPES for consumer in consumers.get(node.output[0], ())
        ):
            picked.add(node.name)  # residual stream
        elif node.op_type == "ReduceSum":
            picked.add(node.name)  # conv accumulation, plus the Mul feeding it ...
            for inp in node.input:
                feeder = producer.get(inp)
                if feeder is not None and feeder.op_type in ("Mul", "Add"):
                    picked.add(feeder.name)
            for consumer in consumers.get(node.output[0], ()):  # ... and the Reshape after it
                if consumer.op_type in ("Reshape", "Squeeze"):
                    picked.add(consumer.name)
    return picked


# ---------------------------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------------------------


def _draft_config(config) -> dict:
    """The ``dflash_config`` sub-dict, mirroring upstream ``_draft_config``."""
    return getattr(config, "dflash_config", None) or {}


def _draft_value(config, name: str, default=None):
    """A key in ``dflash_config`` wins, then a top-level config attribute, then ``default``.

    Mirrors upstream ``_draft_value`` exactly so a stock DFlash-2 checkpoint config works
    unchanged.
    """
    return _draft_config(config).get(name, getattr(config, name, default))


def _required_draft_value(config, name: str) -> int:
    value = _draft_value(config, name)
    if value is None:
        raise ValueError(
            f"DFlash-2 requires {name!r}, but it is present neither in config.dflash_config nor as a "
            f"top-level config attribute. Got dflash_config={_draft_config(config)!r}."
        )
    return value


# ---------------------------------------------------------------------------------------------
# attention mask
# ---------------------------------------------------------------------------------------------


def _create_mask(
    position_ids: torch.Tensor,
    target_length: int,
    causal: bool = False,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Build the noise-block attention mask for one layer type.

    Args:
        position_ids: ``[bsz, num_queries]`` absolute positions of the noise block.
        target_length: physical length of the K/V that ``eager_attention_forward`` will see for
            this layer type -- i.e. the per-layer cache capacity, NOT a logical sequence length.
        causal: whether this layer is causal within the noise block. Comes from
            ``DFlash2Attention.is_causal``, i.e. ``config.is_causal`` when set, else
            ``layer_type == "sliding_attention"``.
        sliding_window: ``config.sliding_window`` on sliding layers, ``None`` on full layers.

    Returns:
        ``[bsz, 1, num_queries, target_length]`` bool tensor; ``True`` means MASK OUT.

    What the two modes are, and which one the released checkpoint uses
    -----------------------------------------------------------------
    Upstream ``_attention_mask`` (dflash/model.py) is::

        if is_causal:                 visible &= key_position <= query_position
        if sliding_window is not None:
            visible &= query_position - key_position < sliding_window
            if not is_causal:         visible &= key_position - query_position < sliding_window

    and is built at all only when ``is_causal or sliding_window is not None``. So causality is a
    *config* property, not a property of the layer type: ``is_causal`` defaults to
    ``layer_type == "sliding_attention"`` but ``config.is_causal`` overrides it.

    ``z-lab/Qwen3.8-27B-DFlash2`` sets ``"is_causal": false`` with all five layers
    ``sliding_attention`` and ``sliding_window: 2048`` -- so its sliding layers are
    BIDIRECTIONAL within the block with a symmetric window, which is the ``causal=False`` path
    here. Do not hard-code either mode; both are reachable.

    The ``causal=False`` cutoff is per-ROW (the row's own max position) rather than per-query.
    Upstream needs no such term -- its cache holds only real positions -- but ours is a
    fixed-capacity padded cache, and since every query position exceeds every context position,
    one shared cutoff at the block's last position lets every query see every real KV entry and
    nothing else. That is exactly bidirectional-within-block.

    Upstream's future-window term (``key_position - query_position < sliding_window``) is omitted
    deliberately: by construction ``kv_positions <= row_max <= query_position + block_size - 1``,
    and ``block_size < sliding_window`` (8 vs 2048 in the released config), so the term can never
    mask anything. Restore it if a config ever has ``block_size >= sliding_window``.

    Ring-buffer slot mapping
    ------------------------
    ``QEffSlidingWindowCache`` stores sliding layers as a ring: ``write_only``/``update`` scatter
    to ``position_ids % sliding_window_len`` (cache_utils.py). So on a sliding layer, K/V slot
    ``j`` does NOT hold absolute position ``j`` -- it holds the most recent position ``p <= row_max``
    with ``p % target_length == j``. We reconstruct that below. When no wrap has happened yet
    (``row_max < target_length``) this reduces to ``kv_positions == j``, which is why a plain
    ``arange`` mask appears to work right up until the first wrap.
    """

    device = position_ids.device
    bsz, num_queries = position_ids.shape

    slots = torch.arange(target_length, device=device).view(1, 1, target_length)
    row_max = position_ids.max(dim=-1, keepdim=True).values.unsqueeze(-1)  # [bsz, 1, 1]

    invalid = None
    if sliding_window is None:
        kv_positions = slots  # [1, 1, target_length]
    else:
        # Most recent absolute position stored in each ring slot, <= row_max.
        wraps = torch.div(row_max - slots, target_length, rounding_mode="floor")
        kv_positions = slots + target_length * wraps  # [bsz, 1, target_length]
        # Slots never written yet (only possible before the first wrap) map to a negative position.
        invalid = kv_positions < 0

    # Future / padding cutoff.
    cutoff = position_ids.unsqueeze(-1) if causal else row_max  # [bsz, num_queries|1, 1]
    attention_mask = kv_positions > cutoff

    if invalid is not None:
        attention_mask = attention_mask | invalid

    if sliding_window is not None and sliding_window < target_length:
        # Drop KV older than the window, trailing EACH query's own position. Redundant when the
        # allocated ring is exactly `sliding_window` long (the ring already holds only the last
        # `target_length` positions), hence the guard -- it keeps the node count down in the
        # common case.
        attention_mask = attention_mask | (kv_positions <= position_ids.unsqueeze(-1) - sliding_window)

    return attention_mask.expand(bsz, num_queries, target_length).unsqueeze(1)


# ---------------------------------------------------------------------------------------------
# rotary embedding
# ---------------------------------------------------------------------------------------------


def qeff_apply_rope_two_streams(
    q_noise: torch.Tensor,
    k_ctx: torch.Tensor,
    k_noise: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_ctx: int,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply rope to the two-stream (context + noise) attention inputs.

    ``cos``/``sin`` are ``[bsz, n_ctx + q_len, head_dim]``, produced by ONE
    ``Qwen3RotaryEmbedding`` call on ``cat([position_ids_target, position_ids], -1)``. They are
    therefore NOT a table indexed by absolute position -- row ``j`` already holds the angles for
    the ``j``-th entry of that concatenation. Split them BY ROW: the context stream owns the
    first ``n_ctx`` rows, the noise stream the rest.

    Do not "simplify" this to ``cos[position_ids]``. Indexing by position *value* is only correct
    while the absolute positions happen to be < the table length; export example inputs are
    exactly that degenerate case, so it looks fine at export time and then silently clamps on
    device at real decode positions, giving every drafted token the wrong angle. That failure
    mode is documented at length in ``dflash_draft`` / ``muse_glimmer_assistant``.

    ``unsqueeze_dim=1`` inserts the head axis: q/k are ``[bsz, heads, seq, head_dim]`` while
    cos/sin are ``[bsz, seq, head_dim]``. Without it the multiply right-aligns ``heads`` against
    ``bsz`` and only works by accident at ``bsz == 1``.
    """
    cos_ctx = cos[:, :n_ctx].unsqueeze(unsqueeze_dim)
    sin_ctx = sin[:, :n_ctx].unsqueeze(unsqueeze_dim)
    cos_noise = cos[:, n_ctx:].unsqueeze(unsqueeze_dim)
    sin_noise = sin[:, n_ctx:].unsqueeze(unsqueeze_dim)

    k_ctx_embed = k_ctx * cos_ctx + rotate_half(k_ctx) * sin_ctx
    q_noise_embed = q_noise * cos_noise + rotate_half(q_noise) * sin_noise
    k_noise_embed = k_noise * cos_noise + rotate_half(k_noise) * sin_noise

    return (
        q_noise_embed.to(q_noise.dtype),
        k_ctx_embed.to(k_ctx.dtype),
        k_noise_embed.to(k_noise.dtype),
    )


# ---------------------------------------------------------------------------------------------
# grouped dynamic causal convolution
# ---------------------------------------------------------------------------------------------


def _causal_shift(hidden: torch.Tensor, offset: int) -> torch.Tensor:
    """Shift ``[bsz, length, hidden]`` right along ``length`` by ``offset``, zero-filling the front.

    ``offset`` is a Python int, so this lowers to a static Slice + Concat against a constant and
    stays shape-agnostic in ``length`` (safe across specializations). Upstream writes this as
    ``F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))`` on a 4-D view, where the 6-tuple pads
    the third-from-last axis; the explicit concat is the same thing with less to get wrong.

    Taps at or beyond ``length`` reach entirely into the zero left-context, so they contribute
    nothing. Upstream raises a shape error there (its ``F.pad`` grows the tensor instead of
    clamping); returning zeros is the arithmetically correct answer. It only matters for
    ``conv_kernel_size > block_size``, which no real config uses.
    """
    if offset == 0:
        return hidden
    if offset >= hidden.shape[1]:
        return torch.zeros_like(hidden)
    pad = hidden.new_zeros((hidden.shape[0], offset, hidden.shape[2]))
    return torch.cat([pad, hidden[:, :-offset]], dim=1)


class GroupedDynamicCausalConv(nn.Module):
    """Depthwise causal conv with static per-channel + dynamic per-group tap weights.

        out[b,t,g,c] = sum_o ( base[o, g*gs+c] + dynamic[b,t,o,g] ) * hidden[b, t-o, g*gs+c]

    Note the two alignments that are easy to get wrong:

    * ``dynamic`` is indexed by the OUTPUT position ``t``, not the source position ``t-o``.
    * ``finish`` applies weights derived from the sublayer's INPUT (they are computed once in
      ``prepare``, second half of the projection), not from its output.

    There is deliberately NO state carried across calls: left context is zero-padding only, so
    position 0 of every drafted block convolves over ``[h_0, 0, ..., 0]``. That matches upstream
    (``GroupedDynamicCausalConv`` holds only parameters -- no buffers, no reset/update) and is
    what the checkpoint was trained with. It also means this module needs no extra QPC I/O,
    no retention buffers and no block-boundary reset.
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        if hidden_size % group_size != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by conv_group_size {group_size}")
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        # Stored as ints, not derived from `hidden.shape[-1]` per call, so nothing traces a shape op.
        self.groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * self.groups, bias=False)

    def _convolve(self, hidden: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[B, L, H]``, ``dynamic`` ``[B, L, K, groups]``, ``base`` ``[K, H]``."""
        bsz, length, hidden_size = hidden.shape
        k, groups, group_size = self.kernel_size, self.groups, self.group_size

        # One broadcast multiply + one reduce over the tap axis, instead of upstream's
        # accumulate-in-a-loop. Same arithmetic, fewer nodes, and no `addcmul` (uneven opset
        # coverage). Materializes K x the activation, which is negligible at L = block_size.
        shifted = torch.stack([_causal_shift(hidden, offset) for offset in range(k)], dim=1)
        shifted = shifted.view(bsz, k, length, groups, group_size)

        weight = base.view(1, k, 1, groups, group_size).to(hidden.dtype) + dynamic.permute(0, 2, 1, 3).unsqueeze(-1)
        return (weight * shifted).sum(dim=1).view(bsz, length, hidden_size)

    def prepare(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convolve the sublayer input; also return the dynamic kernel that ``finish`` will use."""
        dynamic = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, self.kernel_size, self.groups)
        return (
            self._convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0]),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        """Convolve the sublayer output with the kernel stashed by ``prepare``."""
        return self._convolve(hidden, dynamic, self.base_kernel[1])


# ---------------------------------------------------------------------------------------------
# candidate selector
# ---------------------------------------------------------------------------------------------


class CandidateSelector(nn.Module):
    """Bigram rescore over per-position top-k candidates.

    Degenerate at ``selector_top_k == 1``: with a single candidate per position the rescored
    argmax is that candidate whatever the scores are (and with ``temperature > 0`` the softmax
    over one score is 1.0), so ``select`` reduces exactly to ``logits.argmax(-1)``.
    ``DFlash2ForCausalLM.forward`` therefore emits plain logits and lets the host pick tokens,
    exactly as the DFlash-1 draft does -- which keeps the existing SpD runner contract unchanged
    and keeps these vocab-sized codebooks out of the QPC (ONNX export prunes unused parameters).

    NOTE: ``z-lab/Qwen3.8-27B-DFlash2`` ships ``selector_top_k: 16``, so on that checkpoint the
    rescore is live and does change the drafted tokens. Running it at top-k 1 is a valid
    configuration but will not reproduce reference acceptance rates; ``select`` below is the
    reference to wire into the host loop when you move past top-k 1.

    Keep it on the host, not in the QPC. It is a ``block_size``-deep *serially dependent* chain
    (each step's argmax feeds the next step's codebook lookup), which is pathological for a static
    compiler, and the codebooks are ``2 * vocab * rank`` = 2 x 248320 x 256 ~= 127M parameters
    (~254MB in bf16) that would otherwise have to live on device. Emit
    ``candidates``/``unary``/``hidden_projection`` from the QPC instead -- doing the top-k on
    device also shrinks the per-iteration logits transfer from ``[B, block, 248320]`` fp32 to a
    few KB.
    """

    def __init__(self, config):
        super().__init__()
        rank = int(_required_draft_value(config, "selector_rank"))
        self.rank = rank
        self.top_k = int(_required_draft_value(config, "selector_top_k"))
        self.predecessor_codebook = nn.Embedding(config.vocab_size, rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, rank)
        self.hidden_projection = nn.Linear(config.hidden_size, rank, bias=False)

    @torch.no_grad()
    def select(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy bigram-rescored path. CPU reference; mirrors upstream ``select`` at temperature 0.

        ``anchor_ids`` ``[bsz]`` is the last already-accepted token (upstream passes
        ``block_output_ids[:, 0]``), seeding the predecessor chain.
        """
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        projected = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        for position in range(block_size if block_size is not None else hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * projected[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return torch.stack(path, dim=1), candidates


# ---------------------------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------------------------


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups).to(query.dtype)
    value_states = repeat_kv(value, module.num_key_value_groups).to(query.dtype)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DFlash2Attention(nn.Module):
    """Two-stream (context + noise) attention for the DFlash-2 draft.

    Queries come only from the noise block; keys/values come from the already-verified target
    context (``target_hidden``) *and* the noise block. The context stream is written into the
    cache with ``write_only`` (it must not advance the sliding-window progression), the noise
    stream with ``update`` (which also gathers the K/V actually used).
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # Qwen3 normalises q/k per head before rope.
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_types = getattr(config, "layer_types", None)
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        # Causality is config-driven, matching upstream: the layer type only supplies the default.
        # z-lab/Qwen3.8-27B-DFlash2 sets is_causal=false on all-sliding layers, i.e. bidirectional
        # within the block. See _create_mask.
        is_causal = getattr(config, "is_causal", None)
        self.is_causal = layer_type == "sliding_attention" if is_causal is None else bool(is_causal)
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids_target: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        kwargs.pop("output_attentions", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("labels", None)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k_ctx = self.k_norm(k_ctx.view(bsz, ctx_len, -1, self.head_dim)).transpose(1, 2)
        k_noise = self.k_norm(k_noise.view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        v_ctx = v_ctx.view(bsz, ctx_len, -1, self.head_dim).transpose(1, 2)
        v_noise = v_noise.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, k_ctx, k_noise = qeff_apply_rope_two_streams(query_states, k_ctx, k_noise, cos, sin, ctx_len)

        key_states, value_states = k_noise, v_noise
        if past_key_value is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids_target}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            # Seed the verified context first; write_only does not advance seen_tokens.
            past_key_value.write_only(k_ctx, v_ctx, self.layer_idx, cache_kwargs)

            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(k_noise, v_noise, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


# ---------------------------------------------------------------------------------------------
# decoder layer
# ---------------------------------------------------------------------------------------------


class DFlash2DecoderLayer(Qwen3DecoderLayer):
    """Qwen3 pre-norm layer with two-stream attention and a dynamic conv around each sublayer.

    The convolutions live INSIDE the residual branch, on both sides of each sublayer::

        residual = h
        h = input_layernorm(h)
        h, kernel = attention_conv.prepare(h)
        h = self_attn(h, target_hidden=...)
        h = attention_conv.finish(h, kernel)
        h = residual + h

    Note ``prepare`` transforms the tensor that becomes q, k_noise AND v_noise -- but NOT
    ``target_hidden``, so the context stream bypasses the convolution entirely. That is upstream
    behaviour; do not "helpfully" convolve the context stream too.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace Qwen3's single-stream attention with the two-stream variant.
        self.self_attn = DFlash2Attention(config=config, layer_idx=layer_idx)

        kernel_size = int(_required_draft_value(config, "conv_kernel_size"))
        group_size = int(_required_draft_value(config, "conv_group_size"))
        self.attention_conv = GroupedDynamicCausalConv(config.hidden_size, kernel_size, group_size)
        self.mlp_conv = GroupedDynamicCausalConv(config.hidden_size, kernel_size, group_size)
        self.residual_scale = 1.0
        self.scale_input_residual = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor = None,
        position_ids_target: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_ids_target=position_ids_target,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if attention_kernel is not None:
            hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)
        if self.scale_input_residual:
            residual = residual / self.residual_scale
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------------------------
# backbone
# ---------------------------------------------------------------------------------------------


class DFlash2Model(Qwen3Model):
    """DFlash-2 draft backbone.

    ``fc``/``hidden_norm`` (the projection of the concatenated target-layer hidden states) are
    NOT here: as in the DFlash-1 onboarding they live on the TLM graph, which emits an
    already-projected ``target_hidden``. ``candidate_selector`` is constructed so a stock
    checkpoint loads cleanly, but is unused at ``selector_top_k == 1``.
    """

    def __init__(self, config: Qwen3Config):
        # Build the Qwen3 scaffolding (embed_tokens, norm, rotary_emb, has_sliding_layers) but not
        # its decoder layers: every one of them would be replaced below, and on a multi-GB draft
        # allocating both stacks doubles peak host memory. Restored in `finally` so the config the
        # caller handed us is never left mutated.
        num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = 0
        try:
            super().__init__(config)
        finally:
            config.num_hidden_layers = num_hidden_layers

        self.layers = nn.ModuleList([DFlash2DecoderLayer(config, layer_idx) for layer_idx in range(num_hidden_layers)])
        self.candidate_selector = CandidateSelector(config)
        self.block_size = int(_draft_value(config, "block_size", 16))
        self.mask_token_id = _draft_value(config, "mask_token_id")
        self.__qeff_init__()
        self.post_init()

    def __qeff_init__(self):
        # QEffSlidingWindowCache.from_legacy_cache reads `_sliding_window_pattern` to tell which
        # legacy entry is the first full-attention layer (and therefore carries max_cache_len).
        full_layer = next(
            (index + 1 for index, layer_type in enumerate(self.config.layer_types) if layer_type == "full_attention"),
            self.config.num_hidden_layers,
        )
        self.config._sliding_window_pattern = full_layer

    def forward(
        self,
        target_hidden: torch.Tensor = None,
        noise_embeds: torch.FloatTensor = None,
        position_ids_target: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        from QEfficient.transformers.cache_utils import QEffSlidingWindowCache

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if noise_embeds is None:
            raise ValueError(
                "noise_embeds is required. Embed the mask/noise token ids at the "
                "DFlash2ForCausalLM level (which also applies input_embedding_scale) and pass the "
                "result here."
            )

        return_legacy_cache = False
        # NB: QEffSlidingWindowCache is a standalone class, NOT a transformers Cache subclass, so
        # it has to be named explicitly here -- an `isinstance(past_key_values, Cache)` test alone
        # sends an already-built sliding cache into from_legacy_cache, which then tries to subscript
        # it.
        if use_cache and not isinstance(past_key_values, (Cache, QEffSlidingWindowCache)):
            if past_key_values is None:
                raise ValueError(
                    "past_key_values must be a QEffSlidingWindowCache or a legacy list of per-layer "
                    "(key, value) tensors. The draft cannot allocate one itself: the sliding-window "
                    "and full-attention cache capacities come from the compiled specialization."
                )
            return_legacy_cache = True
            past_key_values = QEffSlidingWindowCache.from_legacy_cache(self.config, past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + noise_embeds.shape[1], device=noise_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Per-layer-type mask, each sized to ITS OWN cache capacity. Sliding and full layers have
        # different physical K/V lengths (sliding_window_len vs max_cache_len), and
        # eager_attention_forward's torch.where needs the mask to match the gathered K/V exactly --
        # one shared `max(...)` target_length would broadcast-mismatch on whichever type is shorter.
        sliding_len = getattr(past_key_values, "sliding_window_len", None)
        full_len = getattr(past_key_values, "max_cache_len", None)
        if isinstance(attention_mask, torch.Tensor):
            sliding_len = full_len = attention_mask.shape[-1]

        layer_types = getattr(self.config, "layer_types", None) or ["sliding_attention"] * len(self.layers)
        mask_mapping: Dict[str, Optional[torch.Tensor]] = {}
        for layer_type, length in (("full_attention", full_len), ("sliding_attention", sliding_len)):
            if layer_type not in layer_types:
                continue
            # Take causality from the layer itself rather than from the layer type: it is a config
            # knob (config.is_causal) and the released DFlash-2 checkpoint sets it to false on
            # all-sliding layers. Layers of the same type always agree on it.
            reference_layer = self.layers[layer_types.index(layer_type)]
            mask_mapping[layer_type] = _create_mask(
                position_ids=position_ids,
                target_length=length,
                causal=reference_layer.self_attn.is_causal,
                sliding_window=reference_layer.self_attn.sliding_window,
            )

        hidden_states = noise_embeds

        # ONE rope evaluation for both streams: rows [0, n_ctx) are the context-stream positions
        # and rows [n_ctx, n_ctx + q_len) the noise-stream positions.
        # qeff_apply_rope_two_streams splits them back apart BY ROW.
        position_ids_combined = torch.cat([position_ids_target, position_ids], dim=-1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids_combined)

        all_hidden_states = () if output_hidden_states else None

        for index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                target_hidden=target_hidden,
                position_ids_target=position_ids_target,
                attention_mask=mask_mapping[layer_types[index]],
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


# ---------------------------------------------------------------------------------------------
# causal LM wrapper
# ---------------------------------------------------------------------------------------------


class DFlash2ForCausalLM(Qwen3ForCausalLM):
    """DFlash-2 draft with an LM head.

    Emits full logits ``[bsz, block_size, vocab]`` and lets the host pick tokens, matching the
    DFlash-1 draft's output contract so the existing SpD runner works unchanged (it does
    ``dlm_outputs["logits"].argmax(axis=-1)``). At ``selector_top_k == 1`` that argmax IS the
    DFlash-2 selector -- see ``CandidateSelector``.

    Above top-k 1 the argmax is a strictly weaker approximation of the reference selector. Setting
    ``emit_selector_hidden`` (via ``qaic_config["dflash_emit_selector_hidden"]``) adds one extra
    output, ``selector_hidden`` ``[bsz, block_size, selector_rank]``, which is exactly what a
    host-side ``CandidateSelector`` needs to run the real bigram rescore. It is opt-in because it
    changes the QPC's output signature, so already-compiled logits-only DLMs keep working.
    """

    #: Set by DFlash2DLMTransform. See the class docstring / forward().
    emit_selector_hidden: bool = False

    def get_submodules_for_export(self) -> type[nn.Module]:
        """Return the repeated decoder-layer class used for ONNX subfunctions."""
        return {DFlash2DecoderLayer}

    def __init__(self, config: Qwen3Config):
        # Skip Qwen3ForCausalLM.__init__ and inline it: it would build a complete Qwen3Model that
        # we immediately discard for DFlash2Model. Everything else it does is reproduced verbatim
        # (identical in transformers 4.57 and 5.15).
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = DFlash2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.input_embedding_scale = float(_draft_value(config, "input_embedding_scale", 1.0))
        self.post_init()

    def generate_npi_file(self, onnx_path: Union[str, Path], model_name: Optional[str] = None) -> str:
        """Keep numerically sensitive draft operators in FP32 on AI 100.

        Two policies are combined. The op-type set below is the usual RMSNorm/Softmax/Sigmoid
        (plus Tanh, which only appears when the config sets ``final_logit_softcapping`` -- the
        Qwen DFlash-2 pair does not, so no Tanh node exists to pin there).

        On top of that, ``_overflow_prone_nodes`` pins the graph's two unnormalised accumulators
        -- the residual stream and the dynamic convolutions -- located by graph shape, because
        neither has a distinguishing op type and the exporter does not preserve the module names
        that ``CONV_MODULE_NAMES`` was written to match. Skipping them does not degrade accuracy
        gracefully: on Qwen3.8-27B-DFlash2 the residual reaches 141767 against an fp16 max of
        65504, goes inf at layer 1 and propagates, so every logit saturates at +-65504 and every
        slot decodes the same token. Pin the residual but not the convolution and the failure
        shrinks to a single corrupted slot rather than disappearing.
        """
        del model_name
        onnx_path = Path(onnx_path)
        model = onnx.load(str(onnx_path), load_external_data=False)
        fp32_ops = {"CustomRMSNorm", "Sigmoid", "Softmax", "Tanh"}
        nodes = [*model.graph.node, *(node for function in model.functions for node in function.node)]
        # A scaled residual stream emulates BF16's range while allowing the graph to stay in
        # FP16.  Unscaled models retain the conservative accumulator annotations.
        residual_scale = float(getattr(self.model, "residual_scale", 1.0))
        accumulators = set() if residual_scale > 1.0 else _overflow_prone_nodes(nodes)
        fp32_names = [
            output_name
            for node in nodes
            if node.op_type in fp32_ops
            or node.name in accumulators
            or any(name in node.name for name in CONV_MODULE_NAMES)
            for output_name in node.output
            if output_name
        ]
        npi_path = onnx_path.with_name(f"{onnx_path.stem}_dflash2_draft_npi.yaml")
        with open(npi_path, "w") as fp:
            yaml.safe_dump({"FP32NodeInstanceNames": list(dict.fromkeys(fp32_names))}, fp, sort_keys=False)
        return str(npi_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Build from a DFlash-2 draft checkpoint, remapping its key names on the way in.

        Bypasses ``PreTrainedModel.from_pretrained`` deliberately. The checkpoint's keys are
        ``layers.*`` / bare codebook tensors (see ``remap_dflash2_state_dict``), and the
        ``_checkpoint_conversion_mapping`` hook that would express that declaratively exists in
        transformers 4.5x but not in 5.15 -- which is the version this checkpoint was written with.
        Doing the remap here works on both.
        """
        from transformers import AutoConfig

        config = kwargs.pop("config", None)
        hub_kwargs = {key: kwargs.pop(key) for key in ("cache_dir", "revision", "token") if key in kwargs}
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        dtype = kwargs.pop("dtype", None) or kwargs.pop("torch_dtype", None)
        # Accepted for signature compatibility with the AutoModel path but not meaningful here:
        # this class does not go through HF's attention dispatch, and it is always materialised
        # eagerly because the remap needs a full state dict anyway.
        for ignored in ("attn_implementation", "attn_implementation_autoset", "low_cpu_mem_usage", "use_safetensors"):
            kwargs.pop(ignored, None)
        if kwargs:
            warnings.warn(f"DFlash2ForCausalLM.from_pretrained ignoring unsupported kwargs: {sorted(kwargs)}")

        if config is None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **hub_kwargs
            )

        model = cls(config, *args)
        state_dict = _read_checkpoint_tensors(pretrained_model_name_or_path, **hub_kwargs)
        missing, unexpected = model.load_state_dict(remap_dflash2_state_dict(state_dict), strict=False)

        if unexpected:
            warnings.warn(f"DFlash2ForCausalLM: unexpected checkpoint keys ignored: {sorted(unexpected)}")
        # embed_tokens/lm_head are legitimately absent -- the draft shares the target's, and
        # DFlash2DLMTransform copies them in from the TLM repo. Anything else missing is a bug.
        unexplained = [
            key for key in missing if not key.endswith(("embed_tokens.weight", "lm_head.weight", "lm_head.bias"))
        ]
        if unexplained:
            warnings.warn(f"DFlash2ForCausalLM: randomly initialized parameters: {sorted(unexplained)}")

        if dtype is not None:
            model.config.torch_dtype = dtype
            model = model.to(dtype)
        return model

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM head plus DFlash-2's output scaling / soft-capping.


        Done in fp32 (the head already casts) because ``logits * output_multiplier`` before the
        cap is a natural fp16 overflow, and the cap's ``tanh`` saturates badly in fp16 -- it is
        in the npi FP32 list for the same reason.
        """
        logits = self.lm_head(hidden_states).float()
        logits = logits * float(_draft_value(self.config, "output_multiplier", 1.0))
        softcap = _draft_value(self.config, "final_logit_softcapping")
        if softcap is not None and float(softcap) > 0:
            softcap = float(softcap)
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def forward(
        self,
        target_hidden: torch.Tensor = None,
        noise_embeds: torch.FloatTensor = None,
        position_ids_target: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) == (noise_embeds is None):
            raise ValueError("Specify exactly one of input_ids or noise_embeds.")
        if input_ids is not None:
            noise_embeds = self.model.embed_tokens(input_ids)
            if self.input_embedding_scale != 1.0:
                noise_embeds = noise_embeds * self.input_embedding_scale

        outputs = self.model(
            target_hidden=target_hidden,
            noise_embeds=noise_embeds,
            position_ids_target=position_ids_target,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        logits = self.compute_logits(outputs.last_hidden_state)

        hidden_states = outputs.hidden_states
        if getattr(self, "emit_selector_hidden", False):
            # The low-rank projection is the ONLY part of the bigram rescore that depends on the
            # draft's hidden states, so it is the only part that has to come out of the QPC. The
            # serially dependent argmax chain and the 2 x vocab x rank codebooks stay on the host
            # (see CandidateSelector); candidates/unary are recovered from `logits` there.
            selector = getattr(self.model, "candidate_selector", None)
            if selector is None:
                raise RuntimeError(
                    "emit_selector_hidden was requested but candidate_selector is absent from the "
                    "draft. DFlash2DLMTransform drops it at selector_top_k <= 1, where the rescore "
                    "provably degenerates to logits.argmax(-1) and the host needs no extra output."
                )
            hidden_states = selector.hidden_projection(outputs.last_hidden_state)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )


# ---------------------------------------------------------------------------------------------
# checkpoint key mapping
# ---------------------------------------------------------------------------------------------


def _read_checkpoint_tensors(checkpoint_path, **hub_kwargs) -> Dict[str, torch.Tensor]:
    """Read every tensor from a checkpoint given as a local directory or an HF repo id."""
    path = Path(checkpoint_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                str(checkpoint_path), allow_patterns=["*.safetensors", "pytorch_model*.bin"], **hub_kwargs
            )
        )

    state_dict: Dict[str, torch.Tensor] = {}
    shards = sorted(path.glob("*.safetensors"))
    if shards:
        from safetensors import safe_open

        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                state_dict.update({key: handle.get_tensor(key) for key in handle.keys()})
    else:
        for shard in sorted(path.glob("pytorch_model*.bin")):
            state_dict.update(torch.load(str(shard), map_location="cpu", weights_only=True))
    if not state_dict:
        raise FileNotFoundError(f"No .safetensors or pytorch_model*.bin weights found in {path}")
    return state_dict


def remap_dflash2_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map an upstream DFlash-2 draft checkpoint onto ``DFlash2ForCausalLM``'s key names.

    Verified against the 81 tensors in ``z-lab/Qwen3.8-27B-DFlash2/model.safetensors``.

    Three mismatches to fix:

    1. Upstream ``DFlash2DraftModel`` IS the top-level module, so its keys are ``layers.*``,
       ``norm.weight``, ``candidate_selector.*``. This class follows the HF convention and nests
       the backbone under ``model.``.
    2. The codebooks are stored as BARE tensors (``candidate_selector.predecessor_codebook``, no
       ``.weight``); upstream patches this at load time via ``key_mapping`` in
       ``DFlash2DraftModel.from_pretrained``, whereas ``nn.Embedding`` needs the suffix.
    3. ``fc.weight`` ``[5120, 25600]`` and ``hidden_norm.weight`` are dropped -- they belong on the
       TLM graph, which emits an already-projected ``target_hidden``. Load them into the target
       model instead (see ``DFlashTLMTransform`` in pytorch_transforms.py for the DFlash-1
       equivalent, including its absmax rescale of ``fc``).

    The checkpoint contains NO ``embed_tokens.weight`` and NO ``lm_head.weight``: the draft shares
    the target's, so the weight-loading transform must copy them in from the TLM repo, exactly as
    ``DFlashDLMTransform`` does for DFlash-1.
    """

    codebooks = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in ("fc.weight", "hidden_norm.weight"):
            continue
        if key in codebooks:
            key = f"{key}.weight"
        if key.startswith(("layers.", "norm.", "embed_tokens.", "candidate_selector.", "rotary_emb.")):
            key = f"model.{key}"
        remapped[key] = value
    return remapped
