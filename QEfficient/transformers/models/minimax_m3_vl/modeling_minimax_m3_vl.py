# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from functools import partial
from math import lcm
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import onnx
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLAttention,
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLDenseMLP,
    MiniMaxM3VLForCausalLM,
    MiniMaxM3VLIndexer,
    MiniMaxM3VLRotaryEmbedding,
    MiniMaxM3VLSparseMoeBlock,
    MiniMaxM3VLTextModel,
    MiniMaxM3VLTopKRouter,
    repeat_kv,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
)
from QEfficient.customop.utils import (
    compile_length_sequence_chunk,
    ctx_gather_3d,
    ctx_gather_block_range_kv_dp,
    ctx_gather_blocked_kv,
    ctx_gather_blocked_kv_dp,
    ctx_gather_paged_kv_dp,
    ctx_paged_scatter_dp,
    ctx_scatter_3d,
    m3_ctx_scatter,
)
from QEfficient.transformers.cache_utils import (
    QEffDynamicCache,
    QEffMiniMaxSparseCache,
)
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.models.minimax_m3_vl.onnx_transforms import MiniMaxM3StaticPrefillLoopTransform
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    MoEWeights,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    minimax_m3_clamped_glu_mlp,
)
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

MASKED_ATTENTION_LOGIT = -3.0e4
_FP16_MAX_VALUE = 65504.0


_MINIMAX_NPI_OUTPUT_SUFFIXES = (
    "/Add_output_0",
    "/Add_1_output_0",
    "/input_layernorm/CustomRMSNorm_output_0",
    "/post_attention_layernorm/CustomRMSNorm_output_0",
    "/self_attn/q_norm/CustomRMSNorm_output_0",
    "/self_attn/k_norm/CustomRMSNorm_output_0",
    "/norm/CustomRMSNorm_output_0",
)


def _dynamic_sequence_chunks(
    tensor: torch.Tensor,
    num_chunks: int,
    dim: int,
    compile_axis_size: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Split an evenly divisible dynamic axis without exporting SplitToSequence."""
    return tuple(
        _dynamic_sequence_chunk(tensor, num_chunks, chunk_idx, dim, compile_axis_size)
        for chunk_idx in range(num_chunks)
    )


def _dynamic_sequence_chunk(
    tensor: torch.Tensor,
    num_chunks: int,
    chunk_idx: int,
    dim: int,
    compile_axis_size: int | None = None,
) -> torch.Tensor:
    """Select one evenly sized chunk while allowing a loop-carried chunk index."""
    if compile_axis_size is not None:
        return compile_length_sequence_chunk(tensor, dim, num_chunks, chunk_idx, compile_axis_size)
    shape = tensor.shape
    chunked = tensor.reshape(*shape[:dim], num_chunks, -1, *shape[dim + 1 :])
    return chunked.select(dim, chunk_idx)


def _dynamic_sequence_nested_chunks(
    tensor: torch.Tensor,
    num_outer_chunks: int,
    num_inner_chunks: int,
    dim: int,
    compile_axis_size: Optional[int] = None,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Create fixed nested loop chunks while keeping the innermost sequence axis dynamic."""
    blocks = _dynamic_sequence_chunks(
        tensor,
        num_outer_chunks * num_inner_chunks,
        dim,
        compile_axis_size=compile_axis_size,
    )
    return tuple(
        tuple(blocks[outer_idx * num_inner_chunks + inner_idx] for inner_idx in range(num_inner_chunks))
        for outer_idx in range(num_outer_chunks)
    )


def _decoder_function_npi_names(model: onnx.ModelProto) -> list[str]:
    """Collect Add and CustomRMSNorm names inside MiniMax decoder functions."""
    functions = {(function.domain, function.name): function for function in model.functions}
    npi_names = []
    for function_call in model.graph.node:
        function = functions.get((function_call.domain, function_call.op_type))
        is_decoder_call = "/language_model/layers." in function_call.name or function_call.op_type.startswith(
            "QEffMiniMaxM3VLDecoderLayer"
        )
        if function is None or not is_decoder_call:
            continue
        npi_names.extend(
            f"{function_node.name}_output_0"
            for function_node in function.node
            if function_node.op_type in ("Add", "CustomRMSNorm") and function_node.name
        )
    return npi_names


def _generate_minimax_npi_file(onnx_path: Union[str, Path]) -> str:
    """Generate Minimax's graph-specific FP32 node placement file."""
    onnx_path = Path(onnx_path)
    npi_path = onnx_path.with_name(f"{onnx_path.stem}_minimax_npi.yaml")
    model = onnx.load(str(onnx_path), load_external_data=False)

    fp32_names = [
        output_name
        for node in model.graph.node
        for output_name in node.output
        if output_name and output_name.endswith(_MINIMAX_NPI_OUTPUT_SUFFIXES)
    ]
    fp32_names.extend(_decoder_function_npi_names(model))
    fp32_names = list(dict.fromkeys(fp32_names))
    if not fp32_names:
        raise ValueError(f"Could not find Minimax FP32 NPI nodes in ONNX graph: {onnx_path}")

    with npi_path.open("w") as fp:
        yaml.safe_dump({"FP32NodeInstanceNames": fp32_names}, fp, sort_keys=False)
    return str(npi_path)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def qeff_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotated_q = q[..., :rotary_dim]
    passthrough_q = q[..., rotary_dim:]
    rotated_k = k[..., :rotary_dim]
    passthrough_k = k[..., rotary_dim:]
    cos = cos[..., :rotary_dim]
    sin = sin[..., :rotary_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotated_q = rotated_q * cos + rotate_half(rotated_q) * sin
    rotated_k = rotated_k * cos + rotate_half(rotated_k) * sin
    return torch.cat((rotated_q, passthrough_q), dim=-1), torch.cat((rotated_k, passthrough_k), dim=-1)


def _scalar_like(reference: torch.Tensor, value: int | float) -> torch.Tensor:
    """Create a scalar constant without materializing ``reference.shape``."""
    return reference.new_full((), value)


class QEffMiniMaxM3VLRotaryEmbedding(MiniMaxM3VLRotaryEmbedding):
    """MiniMax RoPE backed by static cosine and sine lookup tables.

    Keeping the trigonometric operations out of the exported graph avoids
    cross-core placement conflicts when the shared RoPE outputs fan out to all
    decoder layers and the sparse-attention indexers.
    """

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device)
        self.__qeff_init__()

    def __qeff_init__(self):
        self._set_cos_sin_cache(
            seq_len=int(self.original_max_seq_len),
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        # Include the -1 padding sentinel as the first row.  Exported graphs
        # can then use a single offset Gather without runtime sign correction.
        positions = torch.arange(-1, seq_len, device=device, dtype=torch.int64).to(dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            (emb.cos() * self.attention_scaling).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (emb.sin() * self.attention_scaling).to(dtype),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cache_indices = position_ids + 1
        cos = self.cos_cached[cache_indices]
        sin = self.sin_cached[cache_indices]
        return cos.to(device=x.device, dtype=x.dtype), sin.to(device=x.device, dtype=x.dtype)


def update_running_softmax(
    current_max: torch.Tensor,
    attn_weights_block: torch.Tensor,
    current_denominator: torch.Tensor,
    output: torch.Tensor,
    value_block: torch.Tensor,
    skip_kv: bool = False,
    skip_future: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge one KV block into an online-softmax accumulator."""
    previous_max = current_max
    updated_max = torch.max(previous_max, attn_weights_block.max(dim=-1).values)
    delta_max = previous_max - updated_max
    current_exp = torch.exp(attn_weights_block - updated_max.unsqueeze(-1))
    curr_exp_sum = torch.einsum("bhqk->bhq", current_exp)
    updated_denominator = current_denominator * torch.exp(delta_max) + curr_exp_sum
    if value_block.ndim == current_exp.ndim + 1:
        queries_per_group = value_block.shape[-3]
        heads_per_query = current_exp.shape[-2] // queries_per_group
        current_exp_grouped = current_exp.reshape(
            *current_exp.shape[:-2],
            queries_per_group,
            heads_per_query,
            current_exp.shape[-1],
        )
        value_update = torch.matmul(current_exp_grouped, value_block).reshape(
            *current_exp.shape[:-2],
            current_exp.shape[-2],
            value_block.shape[-1],
        )
    else:
        value_update = torch.matmul(current_exp, value_block)
    updated_output = output * torch.exp(delta_max.unsqueeze(-1)) + value_update
    if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()):
        if skip_future is None:
            raise ValueError("skip_future is required when skip_kv is enabled.")
        updated_max = torch.where(skip_future, previous_max, updated_max)
        updated_denominator = torch.where(skip_future, current_denominator, updated_denominator)
        updated_output = torch.where(skip_future.unsqueeze(-1), output, updated_output)
    return updated_max, updated_denominator, updated_output


def qeff_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = torch.where(
                attention_mask,
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32, device=attn_weights.device),
                attn_weights,
            )
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _gather_paged_kv_selected_heads(pool: torch.Tensor, physical_ids: torch.Tensor) -> torch.Tensor:
    """Gather per-batch, per-head pages from a physical KV pool."""
    if physical_ids.ndim != 3:
        raise ValueError("physical_ids must have shape [B, H, pages].")
    batch, heads, num_pages = physical_ids.shape
    gathered = []
    for batch_idx in range(batch):
        block_ids = physical_ids[batch_idx].transpose(0, 1).contiguous().to(torch.int32)
        pages = ctx_gather_paged_kv_dp(pool, block_ids)
        gathered.append(pages.squeeze(0).view(heads, num_pages, pool.shape[2], pool.shape[3]))
    return torch.stack(gathered, dim=0)


def _resolve_num_logical_pages(ctx_len: int, page_block_size: int, configured_num_pages: Optional[int] = None) -> int:
    """Resolve paged-cache capacity while preserving the context-derived default."""
    num_pages = (
        (ctx_len + page_block_size - 1) // page_block_size
        if configured_num_pages is None
        else int(configured_num_pages)
    )
    if num_pages <= 0:
        raise ValueError("num_logical_pages must be greater than zero.")
    if num_pages * page_block_size < ctx_len:
        raise ValueError(
            f"num_logical_pages ({num_pages}) with page_block_size ({page_block_size}) cannot hold ctx_len ({ctx_len})."
        )
    return num_pages


def _num_page_groups(num_logical_pages: int, cp: int) -> int:
    """Return the number of physical pages needed by CP-grouped paged decode."""
    return (num_logical_pages + cp - 1) // cp


class QEffMiniMaxM3VLIndexer(MiniMaxM3VLIndexer):
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply partial RoPE while preserving the remainder of the index head."""
        rotary_dim = min(
            self.config.index_head_dim,
            int(self.config.head_dim * self.config.rope_parameters.get("partial_rotary_factor", 1.0)),
        )
        rotated = x[..., :rotary_dim]
        passthrough = x[..., rotary_dim:]
        head_axis = x.ndim - 3
        cos = cos.unsqueeze(head_axis)
        sin = sin.unsqueeze(head_axis)
        rotated = rotated * cos + rotate_half(rotated) * sin
        return torch.cat((rotated, passthrough), dim=-1)

    def _project_index_q_prefill(
        self,
        hidden_states: torch.Tensor,
        blocking_config: Optional[AttentionBlockingConfig] = None,
    ) -> torch.Tensor:
        """Project prefill index queries in bounded sequence chunks."""
        query_len = hidden_states.shape[1]
        hidden_states = hidden_states.to(dtype=self.q_proj.weight.dtype)
        configured_num_chunks = getattr(blocking_config, "indexer_q_proj_num_chunks", None)
        if configured_num_chunks is None:
            configured_num_chunks = (query_len + 511) // 512
        num_q_chunks = max(1, int(configured_num_chunks))
        if num_q_chunks == 1:
            return self.q_proj(hidden_states)
        compile_seq_len = getattr(blocking_config, "prefill_compile_seq_len", None)

        q_proj_chunks = [
            self.q_proj(q_chunk)
            for q_chunk in _dynamic_sequence_chunks(
                hidden_states, num_q_chunks, dim=1, compile_axis_size=compile_seq_len
            )
        ]
        return torch.cat(q_proj_chunks, dim=1)

    @staticmethod
    def _read_blocked_k_dp(
        index_key_cache: torch.Tensor,
        position_ids_dp: torch.Tensor,
        start_index: int,
        end_index: int,
        index_block_size: int,
        cp: int,
        hkv: int,
    ) -> torch.Tensor:
        """Gather one compact GP cache range while masking future positions."""
        batch_local, rows, _, _ = index_key_cache.shape
        block_len = end_index - start_index
        pos_max = position_ids_dp.max(dim=-1).values
        dp_lane_per_row = torch.arange(rows, device=index_key_cache.device) // (cp * hkv)
        pos_max_rows = pos_max[:, dp_lane_per_row]

        if cp == 1:
            gather_limit = pos_max_rows
        else:
            way = torch.arange(rows, device=index_key_cache.device).remainder(cp * hkv) // hkv
            cycle_size = index_block_size * cp
            offset_in_way = pos_max_rows.remainder(cycle_size) - way * index_block_size
            offset_in_way = torch.where(
                offset_in_way >= 0,
                offset_in_way.clamp_max(index_block_size - 1),
                torch.full_like(offset_in_way, -1),
            )
            gather_limit = (pos_max_rows // cycle_size) * index_block_size + offset_in_way

        ctx_indices = torch.arange(start_index, end_index, device=index_key_cache.device).view(1, 1, block_len)
        invalid_idx = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        ctx_indices = torch.where(ctx_indices > gather_limit.unsqueeze(-1), invalid_idx, ctx_indices)
        ctx_indices = ctx_indices.to(torch.int32).expand(batch_local, rows, block_len)
        return ctx_gather_blocked_kv_dp(index_key_cache, ctx_indices)

    @staticmethod
    def _read_blocked_k_flat_dp(
        index_key_cache: torch.Tensor,
        position_ids_dp: torch.Tensor,
        start_index: int,
        end_index: int,
        index_block_size: int,
        cp: int,
        hkv: int,
    ) -> torch.Tensor:
        """Gather one compact GP cache range from row-folded 3D retained state."""
        batch_local, dp, _ = position_ids_dp.shape
        flat_rows, _, head_dim = index_key_cache.shape
        rows = flat_rows // batch_local
        block_len = end_index - start_index
        pos_max = position_ids_dp.max(dim=-1).values
        dp_lane_per_row = torch.arange(rows, device=index_key_cache.device) // (cp * hkv)
        pos_max_rows = pos_max[:, dp_lane_per_row]
        way = torch.arange(rows, device=index_key_cache.device).remainder(cp * hkv) // hkv
        cycle_size = index_block_size * cp
        offset_in_way = pos_max_rows.remainder(cycle_size) - way * index_block_size
        offset_in_way = torch.where(
            offset_in_way >= 0, offset_in_way.clamp_max(index_block_size - 1), torch.full_like(offset_in_way, -1)
        )
        gather_limit = (pos_max_rows // cycle_size) * index_block_size + offset_in_way
        ctx_indices = torch.arange(start_index, end_index, device=index_key_cache.device).view(1, 1, block_len)
        ctx_indices = torch.where(ctx_indices > gather_limit.unsqueeze(-1), torch.zeros_like(ctx_indices), ctx_indices)
        ctx_indices = ctx_indices.to(torch.int32).expand(batch_local, rows, block_len)
        gathered = ctx_gather_3d(index_key_cache, ctx_indices.reshape(flat_rows, block_len))
        return gathered.reshape(batch_local, rows, block_len, head_dim)

    def _read_paged_kv_dp(
        self,
        pool: torch.Tensor,
        block_ids_per_page: torch.Tensor,
    ) -> torch.Tensor:
        DP, num_pages = block_ids_per_page.shape
        rows = pool.shape[1]
        assert block_ids_per_page.ndim == 2 and rows % DP == 0
        Hkv = rows // DP
        block_ids_by_row = (
            block_ids_per_page.transpose(0, 1).unsqueeze(2).expand(num_pages, DP, Hkv).reshape(num_pages, rows)
        )
        return ctx_gather_paged_kv_dp(pool, block_ids_by_row.to(torch.int32))

    def _read_index_paged_kv_dp(
        self,
        index_key_cache: torch.Tensor,
        block_ids_per_page: torch.Tensor,
    ) -> torch.Tensor:
        """Gather indexer pages while preserving physical [DP, CP, Hkv] rows."""
        if block_ids_per_page.ndim != 2:
            raise ValueError("Paged indexer block IDs must have shape [DP, pages].")
        dp, num_pages = block_ids_per_page.shape
        rows = index_key_cache.shape[1]
        if rows % dp:
            raise ValueError("Paged indexer pool rows must be divisible by DP.")
        rows_per_dp = rows // dp
        block_ids_by_row = (
            block_ids_per_page.transpose(0, 1).unsqueeze(2).expand(num_pages, dp, rows_per_dp).reshape(num_pages, rows)
        )
        return ctx_gather_paged_kv_dp(index_key_cache, block_ids_by_row.to(torch.int32))

    def _read_blocked_k_core(
        self,
        index_key_cache: torch.Tensor,
        position_ids_dp: torch.Tensor,
        start_index: int,
        end_index: int,
        num_cores: int,
        position_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather one compact index block split across prefill cores."""
        batch_local, rows, _, head_dim = index_key_cache.shape
        if rows != 1 or position_ids_dp.shape[1] != 1:
            raise NotImplementedError("Core-split MSA prefill requires DP=CP=1 and one indexer KV row.")
        block_len = end_index - start_index
        if block_len < num_cores or block_len % num_cores:
            raise ValueError("Core-split MSA prefill KV block must divide evenly across cores.")
        tokens_per_core = block_len // num_cores
        offsets = (
            torch.arange(num_cores, device=index_key_cache.device).view(1, num_cores, 1) * tokens_per_core
            + torch.arange(tokens_per_core, device=index_key_cache.device).view(1, 1, tokens_per_core)
            + start_index
        ).to(torch.int32)
        if position_max is None:
            position_max = position_ids_dp.max(dim=-1).values
        invalid = offsets > position_max[:, :, None]
        invalid_value = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        offsets = torch.where(invalid, _scalar_like(offsets, invalid_value), offsets)
        gathered = ctx_gather_blocked_kv(index_key_cache, offsets.reshape(batch_local, 1, block_len))
        return gathered.reshape(batch_local, num_cores, tokens_per_core, head_dim)

    def _write_msa_paged_prefill_cache(
        self,
        cache: torch.Tensor,
        updates: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter a contiguous prefill sequence into a physical page pool."""
        page_size = int(cache.shape[2])
        batch, query_len = position_ids.shape
        if page_size != self.config.index_block_size:
            raise ValueError("Paged MSA prefill requires page_block_size == index_block_size.")
        if query_len <= 0 or query_len % page_size:
            raise ValueError("Paged MSA prefill query length must be a multiple of page size.")
        if position_ids.shape[1] > 1 and not torch.equal(position_ids[:, 1:], position_ids[:, :-1] + 1):
            raise ValueError("Paged MSA prefill positions must be contiguous.")
        if (position_ids[:, 0] % page_size != 0).any():
            raise ValueError("Paged MSA prefill start positions must be page aligned.")
        if block_table.ndim != 3 or block_table.shape[0] != 1 or block_table.shape[1] != batch:
            raise ValueError("Paged MSA prefill block tables must have shape [1, B, pages].")
        rows = updates.shape[1]
        logical_page = position_ids // page_size
        physical_page = torch.gather(block_table[0].to(torch.int64), 1, logical_page).to(torch.int32)
        block_ids = physical_page.unsqueeze(1).expand(batch, rows, query_len)
        addresses = (position_ids % page_size).to(torch.int32).unsqueeze(1).expand_as(block_ids)
        return ctx_paged_scatter_dp(cache, block_ids, addresses, updates)

    def _read_msa_prefill_paged_block(
        self,
        index_key_cache: torch.Tensor,
        block_table: torch.Tensor,
        start_index: int,
        end_index: int,
        position_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather one logical index block from the physical page pool."""
        page_size = int(index_key_cache.shape[2])
        if start_index % page_size or end_index % page_size:
            raise ValueError("Paged MSA prefill blocks must be page aligned.")
        if block_table.ndim != 2:
            raise ValueError("Paged MSA prefill block_table must have shape [B, pages].")
        batch = block_table.shape[0]
        page_start, page_end = start_index // page_size, end_index // page_size
        page_ids = block_table[:, page_start:page_end].to(torch.int32)
        fallback_ids = block_table[:, :1].to(torch.int32)
        gathered = []
        for batch_idx in range(batch):
            ids = (
                torch.cat((page_ids[batch_idx], fallback_ids[batch_idx]))
                .view(-1, 1)
                .expand(page_end - page_start + 1, index_key_cache.shape[1])
            )
            pages = ctx_gather_paged_kv_dp(index_key_cache, ids).squeeze(0)
            block_len = end_index - start_index
            block = pages[:, :block_len]
            if position_max is not None:
                positions = torch.arange(start_index, end_index, device=block.device).view(1, block_len)
                invalid = positions > position_max[batch_idx]
                block = torch.where(invalid.unsqueeze(-1), pages[:, block_len : block_len + 1], block)
            gathered.append(block)
        return torch.stack(gathered, dim=0)

    def _read_msa_prefill_paged_block_core(
        self,
        index_key_cache: torch.Tensor,
        block_table: torch.Tensor,
        start_index: int,
        end_index: int,
        num_cores: int,
        position_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        block = self._read_msa_prefill_paged_block(index_key_cache, block_table, start_index, end_index, position_max)
        batch, _, block_len, head_dim = block.shape
        if block.shape[1] != 1 or block_len < num_cores or block_len % num_cores:
            raise ValueError("Paged MSA prefill KV block must divide evenly across cores.")
        return block[:, 0].reshape(batch, num_cores, block_len // num_cores, head_dim)

    def _select_blocks_prefill_par(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: "QEffMiniMaxSparseCache",
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        blocking_config: Optional[AttentionBlockingConfig],
        paged_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index_key_cache = past_key_values.index_keys[layer_idx]
        token_indices, token_valid, index_key_cache = self._select_blocks_prefill_par_impl(
            hidden_states,
            position_ids,
            index_key_cache,
            cos,
            sin,
            blocking_config,
            paged_block_table,
        )
        past_key_values.index_keys[layer_idx] = index_key_cache
        return token_indices, token_valid

    def _select_blocks_prefill_par_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        index_key_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        blocking_config: Optional[AttentionBlockingConfig],
        paged_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select MSA blocks with mixed Q-head and KV-token parallelism."""
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        num_cores = int(getattr(blocking_config, "num_cores_per_device", 1) or 1)
        num_index_heads = cfg.index_n_heads
        num_kv_heads = int(getattr(blocking_config, "indexer_n_head", None) or getattr(cfg, "indexer_n_head", 1) or 1)
        dim = cfg.index_head_dim
        num_kv_blocks = max(
            1,
            int(
                getattr(blocking_config, "indexer_num_blocks", None)
                or getattr(blocking_config, "num_kv_blocks", None)
                or 1
            ),
        )
        q_block_size = int(getattr(blocking_config, "indexer_q_size", None) or cfg.index_block_size)
        q_block_chunk = int(getattr(blocking_config, "indexer_q_chunk", None) or q_block_size)
        ctx_len = int(
            getattr(blocking_config, "ctx_len", None)
            or getattr(cfg, "ctx_len", None)
            or (
                paged_block_table.shape[1] * index_key_cache.shape[2]
                if paged_block_table is not None
                else index_key_cache.shape[-2]
            )
        )
        skip_kv = bool(getattr(blocking_config, "skip_kv", True))
        # A Q head owns C/Hq cores when Hq <= C.  If Hq > C, clamp to one
        # core per Q head; no additional KV split is possible in that case.
        cores_per_q_head = max(1, num_cores // num_index_heads)

        if getattr(blocking_config, "msa_indexer_dp", 1) not in (None, 1) or getattr(
            blocking_config, "msa_indexer_cp", 1
        ) not in (None, 1):
            raise ValueError("MSA prefill selection currently requires DP=1 and CP=1.")
        if query_len <= 1:
            raise ValueError("MSA prefill selection requires QL > 1.")
        if num_index_heads % num_kv_heads:
            raise ValueError("MSA prefill index_n_heads must be divisible by indexer_n_head.")
        if ctx_len % cfg.index_block_size or ctx_len % num_kv_blocks:
            raise ValueError("MSA prefill selection requires ctx_len divisible by index_block_size and num_kv_blocks.")
        if q_block_size <= 0 or q_block_chunk < q_block_size:
            raise ValueError("MSA prefill Q block sizes must be positive and ordered.")
        if q_block_chunk % q_block_size:
            raise ValueError("MSA prefill q_block_chunk must divide evenly by q_block_size.")
        if query_len % q_block_chunk:
            raise ValueError("MSA prefill query length must divide evenly by q_block_chunk.")

        if paged_block_table is None:
            expected_cache_shape = (batch, num_kv_heads, ctx_len, dim)
            if tuple(index_key_cache.shape) != expected_cache_shape:
                raise ValueError(
                    f"MSA prefill index_key_cache shape {tuple(index_key_cache.shape)} != {expected_cache_shape}."
                )
        else:
            if getattr(blocking_config, "msa_indexer_dp", 1) not in (None, 1) or getattr(
                blocking_config, "msa_indexer_cp", 1
            ) not in (None, 1):
                raise ValueError("Paged MSA prefill selection currently requires indexer DP=1 and CP=1.")
            if paged_block_table.ndim != 2 or paged_block_table.shape[0] != batch:
                raise ValueError("Paged MSA prefill indexer block_table must have shape [B, pages].")
            if index_key_cache.ndim != 4 or tuple(index_key_cache.shape[1:]) != (
                num_kv_heads,
                index_key_cache.shape[2],
                dim,
            ):
                raise ValueError("Paged MSA index cache shape is incompatible with prefill.")
            page_size = int(index_key_cache.shape[2])
            configured_page_size = getattr(blocking_config, "page_block_size", None)
            if configured_page_size is not None and int(configured_page_size) != page_size:
                raise ValueError("Paged MSA prefill page size does not match blocking_config.page_block_size.")
            if page_size != cfg.index_block_size:
                raise ValueError("Paged MSA prefill requires page size == index_block_size.")
            if paged_block_table.shape[1] < (ctx_len + page_size - 1) // page_size:
                raise ValueError("Paged MSA prefill indexer block_table is too short for ctx_len.")

        idx_q = self._project_index_q_prefill(hidden_states, blocking_config)[..., : num_index_heads * dim]
        idx_q = idx_q.view(batch, query_len, num_index_heads, dim).transpose(1, 2)
        idx_k = self.k_proj(hidden_states)[..., : num_kv_heads * dim]
        idx_k = idx_k.view(batch, query_len, num_kv_heads, dim).transpose(1, 2)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q = self._apply_rope(idx_q, cos, sin)
        idx_k = self._apply_rope(idx_k, cos, sin)
        if paged_block_table is None:
            index_key_cache = m3_ctx_scatter(index_key_cache, position_ids.to(torch.int32), idx_k)
        else:
            index_key_cache = self._write_msa_paged_prefill_cache(
                index_key_cache,
                idx_k,
                position_ids,
                paged_block_table.unsqueeze(0),
            )
        position_max = position_ids.unsqueeze(1).max(dim=-1).values

        kv_block_size = ctx_len // num_kv_blocks
        if kv_block_size < num_cores or kv_block_size % num_cores:
            raise ValueError("MSA parallel prefill KV block must divide evenly across all cores.")
        if kv_block_size % cores_per_q_head:
            raise ValueError(
                "MSA parallel prefill KV block must divide evenly across the cores assigned to each Q head."
            )
        index_block_size = cfg.index_block_size
        tokens_per_q_head = kv_block_size // cores_per_q_head
        if tokens_per_q_head % index_block_size:
            raise ValueError("MSA prefill Q-head KV split must contain complete index blocks.")
        blocks_per_q_head = tokens_per_q_head // index_block_size
        num_blocks = ctx_len // index_block_size
        mask_value = MASKED_ATTENTION_LOGIT
        topk = min(cfg.index_topk_blocks, num_blocks)
        topk_indices_chunks: list[torch.Tensor] = []
        topk_valid_chunks: list[torch.Tensor] = []
        num_q_chunks = int(query_len // q_block_chunk)
        num_q_blocks_per_chunk = q_block_chunk // q_block_size
        compile_seq_len = getattr(blocking_config, "prefill_compile_seq_len", None)
        q_chunks = _dynamic_sequence_chunks(idx_q, num_q_chunks, dim=2, compile_axis_size=compile_seq_len)
        position_chunks = _dynamic_sequence_chunks(position_ids, num_q_chunks, dim=1, compile_axis_size=compile_seq_len)
        nested_q_blocks = _dynamic_sequence_nested_chunks(
            idx_q,
            num_q_chunks,
            num_q_blocks_per_chunk,
            dim=2,
            compile_axis_size=compile_seq_len,
        )
        nested_position_blocks = _dynamic_sequence_nested_chunks(
            position_ids,
            num_q_chunks,
            num_q_blocks_per_chunk,
            dim=1,
            compile_axis_size=compile_seq_len,
        )
        for q_chunk_idx, (q_chunk, query_positions) in enumerate(zip(q_chunks, position_chunks)):
            query_chunk = q_chunk.shape[2]
            if query_chunk % q_block_size:
                raise ValueError(
                    "MSA prefill query length must be divisible by q_block_size within every q_block_chunk."
                )
            score_blocks: list[list[torch.Tensor]] = [[] for _ in range(num_q_blocks_per_chunk)]
            q_blocks = nested_q_blocks[q_chunk_idx]
            query_position_blocks = nested_position_blocks[q_chunk_idx]
            q_chunk_position = query_positions.max(dim=-1).values
            is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()

            for block_idx in range(num_kv_blocks):
                start = block_idx * kv_block_size
                end = start + kv_block_size
                block_skip_future_chunk = torch.tensor(start, device=hidden_states.device) > q_chunk_position
                if skip_kv and not is_export and bool(block_skip_future_chunk.all().item()):
                    # Keep one reduced score tensor per skipped indexer KV
                    # block.  The block is entirely future, so the score
                    # placeholder is sufficient for top-k while avoiding
                    # the cache gather and QK matmul in eager execution.
                    masked_block = torch.full(
                        (
                            batch,
                            num_index_heads,
                            cores_per_q_head,
                            q_block_size,
                            blocks_per_q_head,
                        ),
                        mask_value,
                        dtype=torch.float32,
                        device=hidden_states.device,
                    )
                    for q_block_idx in range(len(score_blocks)):
                        score_blocks[q_block_idx].extend(masked_block.clone() for _ in range(block_idx, num_kv_blocks))
                    break
                if paged_block_table is None:
                    key_block = self._read_blocked_k_core(
                        index_key_cache,
                        position_ids.unsqueeze(1),
                        start,
                        end,
                        num_cores,
                        position_max,
                    ).float()
                else:
                    key_block = self._read_msa_prefill_paged_block_core(
                        index_key_cache,
                        paged_block_table,
                        start,
                        end,
                        num_cores,
                        position_max,
                    ).float()
                # The gather already exposes the full-core KV split. Replicate
                # that split for every Q head and reshape it into each Q
                # head's remaining core group.
                tokens_per_core = kv_block_size // num_cores
                key_core = (
                    key_block.unsqueeze(1)
                    .expand(
                        batch,
                        num_index_heads,
                        num_cores,
                        tokens_per_core,
                        dim,
                    )
                    .reshape(
                        batch,
                        num_index_heads,
                        cores_per_q_head,
                        kv_block_size // cores_per_q_head,
                        dim,
                    )
                )
                for q_block_idx, (q_block, q_positions) in enumerate(zip(q_blocks, query_position_blocks)):
                    # Move Q heads to the core axis (one Q head per logical
                    # core group), then replicate Q over that group's cores.
                    q_core = q_block.unsqueeze(2).expand(
                        batch,
                        num_index_heads,
                        cores_per_q_head,
                        -1,
                        dim,
                    )
                    scores = torch.matmul(
                        q_core.float(),
                        key_core.transpose(-1, -2).float(),
                    )
                    block_skip_future = (
                        torch.tensor(start, device=hidden_states.device) > q_positions[:, None, None, :, None]
                    )
                    if skip_kv:
                        scores = torch.where(
                            block_skip_future,
                            _scalar_like(scores, mask_value),
                            scores,
                        )
                    if cfg.index_local_blocks == 0:
                        key_offsets = torch.arange(kv_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
                        q_positions = q_positions.view(batch, 1, 1, -1, 1)
                        scores = torch.where(
                            key_offsets > (q_positions - start),
                            _scalar_like(scores, mask_value),
                            scores,
                        )
                    # Reduce each index block while this KV block is local.
                    scores = scores.reshape(
                        batch,
                        num_index_heads,
                        cores_per_q_head,
                        -1,
                        blocks_per_q_head,
                        index_block_size,
                    )
                    scores = scores.amax(dim=-1)
                    # Keep the future-block predicate attached to the
                    # reduction result before changing its core/block layout.
                    if skip_kv:
                        scores = torch.where(
                            block_skip_future,
                            _scalar_like(scores, mask_value),
                            scores,
                        )
                    # Keep each KV super-block in the reduction-native
                    # [B, H, core, Q, block] layout.  The layout conversion
                    # is deferred until all KV super-blocks are collected.
                    score_blocks[q_block_idx].append(scores)

            for q_block_idx in range(len(score_blocks)):
                # Stack in KV-super-block order, then restore the logical
                # [B, H, Q, block] selector layout once at the end.  A plain
                # cat on the last dimension would change the core/block
                # ordering relative to the previous per-block permute.
                scores = torch.stack(score_blocks[q_block_idx], dim=2)
                scores = scores.permute(0, 1, 4, 2, 3, 5).reshape(batch, num_index_heads, -1, num_blocks)
                q_positions = query_position_blocks[q_block_idx]
                block_ids = torch.arange(num_blocks, device=hidden_states.device).view(1, 1, 1, num_blocks)
                mask_score = torch.tensor(mask_value, dtype=scores.dtype, device=scores.device)
                local_score = torch.tensor(-mask_value, dtype=scores.dtype, device=scores.device)
                block_skip_future = block_ids * index_block_size > q_positions[:, None, :, None]
                scores = torch.where(
                    block_skip_future,
                    mask_score,
                    scores,
                )
                for local_offset in range(cfg.index_local_blocks):
                    # local_block = q_positions // index_block_size - local_offset
                    local_block = torch.div(q_positions, index_block_size, rounding_mode="floor") - local_offset
                    local_block = torch.where(
                        local_block >= 0,
                        local_block,
                        torch.zeros_like(local_block),
                    )
                    scores = torch.where(
                        block_ids == local_block[:, None, :, None],
                        local_score,
                        scores,
                    )
                top_scores, top_order = torch.topk(scores, k=topk, dim=-1)
                topk_indices_chunks.append(
                    torch.gather(
                        block_ids.expand_as(scores).to(torch.int32),
                        -1,
                        top_order,
                    )
                )
                block_valid = top_scores > (mask_value / 2)
                if skip_kv:
                    top_block_skip_future = torch.gather(block_skip_future.expand_as(scores), -1, top_order)
                    block_valid = block_valid & ~top_block_skip_future
                topk_valid_chunks.append(block_valid)

        block_indices = torch.cat(topk_indices_chunks, dim=2)
        block_valid = torch.cat(topk_valid_chunks, dim=2)
        safe_indices = torch.where(block_valid, block_indices, torch.zeros_like(block_indices)).to(torch.int32)
        if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()):
            self.last_block_indices = safe_indices.detach()
        return safe_indices, block_valid, index_key_cache

    def _select_blocks(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: "QEffMiniMaxSparseCache",
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        blocking_config: Optional[AttentionBlockingConfig] = None,
        block_table: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if block_table is not None:
            return self._select_blocks_paged(
                hidden_states, position_ids, past_key_values, layer_idx, cos, sin, block_table, blocking_config
            )
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        ctx_len = (
            blocking_config.ctx_len if (blocking_config and blocking_config.ctx_len) else None
        ) or past_key_values.layers[layer_idx].keys.shape[2]
        num_blocks = (ctx_len + cfg.index_block_size - 1) // cfg.index_block_size

        hkv = blocking_config.indexer_n_head if (blocking_config and blocking_config.indexer_n_head) else 1
        num_cores = int(getattr(blocking_config, "num_cores_per_device", 1) or 1)
        if num_cores < 1:
            raise ValueError(f"num_cores_per_device must be positive, got {num_cores}.")

        # The GP selector owns non-paged blocked decode, including DP=CP=1.
        # Besides handling compact DP/CP cache layouts, it keeps selected
        # blocks compressed through the attention gather and inserts the
        # mandatory current block without scoring its partially-filled tokens.
        if blocking_config is not None:
            return self._select_blocks_dp(
                hidden_states, position_ids, past_key_values, layer_idx, cos, sin, blocking_config
            )

        idx_q = self.q_proj(hidden_states).view(batch, seq_len, cfg.index_n_heads, cfg.index_head_dim).transpose(1, 2)
        idx_k = self.k_proj(hidden_states).view(batch, seq_len, hkv, cfg.index_head_dim).transpose(1, 2)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q, idx_k = qeff_apply_rotary_pos_emb(
            idx_q,
            idx_k,
            cos[..., : cfg.index_head_dim],
            sin[..., : cfg.index_head_dim],
            int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        )
        idx_k = past_key_values.update_index_key_cache(idx_k, layer_idx, position_ids)

        scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=ctx_len)
        scores = scores.masked_fill(causal_mask, -1.0e30)

        padded_len = num_blocks * cfg.index_block_size
        if padded_len != ctx_len:
            scores = F.pad(scores, (0, padded_len - ctx_len), value=-1.0e30)
        block_scores = scores.view(batch, cfg.index_n_heads, seq_len, num_blocks, cfg.index_block_size).amax(dim=-1)

        block_ids = torch.arange(num_blocks, device=hidden_states.device).view(1, 1, 1, -1)
        q_block = position_ids // cfg.index_block_size
        for local_offset in range(cfg.index_local_blocks):
            local_block = (q_block - local_offset).clamp(min=0)
            block_scores = torch.where(
                block_ids == local_block[:, None, :, None],
                torch.full_like(block_scores, 1.0e30),
                block_scores,
            )

        topk_scores, block_indices = torch.topk(block_scores, k=min(cfg.index_topk_blocks, num_blocks), dim=-1)
        block_valid = topk_scores > -1.0e29
        offsets = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
        token_indices = block_indices.unsqueeze(-1) * cfg.index_block_size + offsets
        token_valid = block_valid.unsqueeze(-1) & (token_indices < ctx_len)
        token_valid = token_valid & (token_indices <= position_ids[:, None, :, None, None])
        token_indices = token_indices[:, :, 0]
        token_valid = token_valid[:, :, 0]
        safe_indices = torch.where(token_valid, token_indices, torch.zeros_like(token_indices)).to(torch.int32)
        return safe_indices, token_valid

    @staticmethod
    def _apply_rope_dp(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
        """Apply RoPE to x in DP layout.

        x:   [..., H, seq_len, D]  where the leading dims include a DP axis
        cos: [..., seq_len, rotary_dim]  (no head axis yet)
        sin: [..., seq_len, rotary_dim]
        """
        rotated = x[..., :rotary_dim]
        passthrough = x[..., rotary_dim:]
        cos = cos[..., :rotary_dim]
        sin = sin[..., :rotary_dim]
        head_axis = x.ndim - 3  # insert singleton for H axis
        cos = cos.unsqueeze(head_axis)
        sin = sin.unsqueeze(head_axis)
        rotated = rotated * cos + rotate_half(rotated) * sin
        return torch.cat((rotated, passthrough), dim=-1)

    @staticmethod
    def _to_dp_cache_shape(
        cache: torch.Tensor,
        dp: int,
        rows: int,
        cp: int,
    ) -> torch.Tensor:
        """Reshape index_key_cache from standard (batch, 1, ctx_len, dim) to DP layout (batch//dp, rows, ctx_len//cp, dim).

        Uses cache.shape for batch and ctx_len so the reshape is dynamic-friendly for ONNX export.
        """
        B, _, T, D = cache.shape
        return cache.reshape(B // dp, rows, T // cp, D)

    @staticmethod
    def _is_dp_cache_shape(cache: torch.Tensor, batch_local: int, rows: int, cache_slots: int) -> bool:
        return (
            cache.ndim == 4
            and cache.shape[0] == batch_local
            and cache.shape[1] == rows
            and cache.shape[2] == cache_slots
        )

    @staticmethod
    def _is_flat_dp_cache_shape(cache: torch.Tensor, flat_rows: int, cache_slots: int) -> bool:
        return cache.ndim == 3 and cache.shape[0] == flat_rows and cache.shape[1] == cache_slots

    @staticmethod
    def _from_dp_cache_shape(
        cache: torch.Tensor,
        dp: int,
        cp: int,
    ) -> torch.Tensor:
        """Revert index_key_cache from DP layout (batch//dp, rows, ctx_len//cp, dim) to standard (batch, 1, ctx_len, dim).

        Uses cache.shape for batch_local and cache_slots so the reshape is dynamic-friendly for ONNX export.
        """
        B, _, T, D = cache.shape
        return cache.reshape(B * dp, 1, T * cp, D)

    def _select_blocks_paged(
        self, hidden_states, position_ids, past_key_values, layer_idx, cos, sin, block_table, blocking_config
    ):
        """Select sparse blocks from the physical paged index-key cache."""
        cfg = self.config
        num_index_heads = cfg.index_n_heads
        dim = cfg.index_head_dim
        num_cores = (
            blocking_config.num_cores_per_device if blocking_config and blocking_config.num_cores_per_device else 1
        )
        num_kv_blocks = max(
            1,
            int(
                getattr(blocking_config, "indexer_num_blocks", None)
                or getattr(blocking_config, "num_kv_blocks", None)
                or 1
            ),
        )
        index_key_cache = past_key_values.index_keys.get(layer_idx)
        if index_key_cache is None or index_key_cache.ndim != 4:
            raise ValueError("Paged MSA index selection requires a physical index-key cache.")
        page_block_size = index_key_cache.shape[2]
        dp = blocking_config.msa_indexer_dp if blocking_config and blocking_config.msa_indexer_dp else 1
        cp = blocking_config.msa_indexer_cp if blocking_config and blocking_config.msa_indexer_cp else 1
        hkv = blocking_config.indexer_n_head if blocking_config and blocking_config.indexer_n_head else 1
        ctx_len = (
            blocking_config.ctx_len
            if blocking_config and blocking_config.ctx_len
            else block_table.shape[2] * page_block_size * cp
        )
        configured_num_pages = (
            getattr(blocking_config, "msa_indexer_num_logical_pages", None) if blocking_config is not None else None
        )
        if configured_num_pages is None and blocking_config is not None:
            configured_num_pages = getattr(blocking_config, "num_logical_pages", None)
        num_pages = _resolve_num_logical_pages(ctx_len, page_block_size, configured_num_pages)
        batch, query_len = hidden_states.shape[:2]
        if query_len != 1:
            raise ValueError("Paged MSA indexer selection is decode-only; QL must be 1.")
        if batch % dp:
            raise ValueError("MSA indexer batch size must be divisible by msa_indexer_dp.")
        if num_index_heads % hkv:
            raise ValueError("index_n_heads must be divisible by indexer_n_head.")
        if cfg.index_local_blocks != 1:
            raise ValueError("Paged MSA decode requires index_local_blocks == 1.")
        batch_local = batch // dp
        rows = dp * cp * hkv
        num_page_groups = _num_page_groups(num_pages, cp)
        expected_cache_shape = (
            dp * batch_local * num_page_groups,
            rows,
            page_block_size,
            dim,
        )
        if tuple(index_key_cache.shape) != expected_cache_shape:
            raise ValueError(
                f"Paged MSA index_key_cache shape {tuple(index_key_cache.shape)} != {expected_cache_shape}."
            )
        if block_table.ndim != 3 or tuple(block_table.shape[:2]) != (dp, batch_local):
            raise ValueError(
                "msa_indexer_block_table must have shape "
                f"[msa_indexer_dp, B_local, pages], got {tuple(block_table.shape)}."
            )
        if block_table.shape[2] < num_page_groups:
            raise ValueError("msa_indexer_block_table is too short for the paged indexer.")
        if ctx_len % page_block_size:
            raise ValueError("Paged MSA indexer requires ctx_len divisible by page_block_size.")
        if num_pages % (num_kv_blocks * cp):
            raise ValueError(
                f"Paged MSA indexer pages ({num_pages}) must be divisible by "
                f"num_kv_blocks * msa_indexer_cp ({num_kv_blocks * cp})."
            )
        page_groups_per_kv_block = num_pages // (num_kv_blocks * cp)
        if page_groups_per_kv_block % num_cores:
            raise ValueError(
                f"Paged MSA indexer page groups per KV block ({page_groups_per_kv_block}) must be divisible by "
                f"num_cores_per_device={num_cores}."
            )
        page_groups_per_core = page_groups_per_kv_block // num_cores
        tokens_per_core = page_groups_per_core * page_block_size

        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        if dp > 1:
            projection_dp = int(getattr(self, "indexer_paged_projection_dp", 1))
            if projection_dp != dp:
                raise ValueError(
                    "Paged MSA indexer DP projection buffers are missing or were built for a different DP factor."
                )
            idx_q_all = F.linear(
                hidden_states,
                self.indexer_paged_q_proj_weight,
                getattr(self, "indexer_paged_q_proj_bias", None),
            ).view(dp, batch_local, query_len, dp, num_index_heads, dim)
            idx_k_all = F.linear(
                hidden_states,
                self.indexer_paged_k_proj_weight,
                getattr(self, "indexer_paged_k_proj_bias", None),
            ).view(dp, batch_local, query_len, dp, hkv, dim)
            lane_idx = torch.arange(dp, device=hidden_states.device).view(dp, 1, 1, 1, 1, 1)
            q_lane_idx = lane_idx.expand(dp, batch_local, query_len, 1, num_index_heads, dim)
            k_lane_idx = lane_idx.expand(dp, batch_local, query_len, 1, hkv, dim)
            idx_q = torch.gather(idx_q_all, 3, q_lane_idx).squeeze(3).permute(1, 0, 3, 2, 4)
            idx_k = torch.gather(idx_k_all, 3, k_lane_idx).squeeze(3).permute(1, 0, 3, 2, 4)
        else:
            idx_q = self.q_proj(hidden_states).view(dp, batch_local, query_len, num_index_heads, dim)
            idx_q = idx_q.permute(1, 0, 3, 2, 4)
            idx_k = self.k_proj(hidden_states).view(dp, batch_local, query_len, hkv, dim)
            idx_k = idx_k.permute(1, 0, 3, 2, 4)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q = self._apply_rope(idx_q, cos_dp[..., :dim], sin_dp[..., :dim])
        idx_k = self._apply_rope(idx_k, cos_dp[..., :dim], sin_dp[..., :dim])

        k_updates = (
            idx_k.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, query_len, dim)
            .reshape(batch_local, rows, query_len, dim)
        )
        addr = (
            (position_ids_dp % page_block_size)
            .to(torch.int32)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        # logical_page = position_ids_dp // page_block_size
        logical_page = torch.div(position_ids_dp, page_block_size, rounding_mode="floor")
        # logical_page_group = logical_page // cp
        logical_page_group = torch.div(logical_page, cp, rounding_mode="floor")
        block_id_dp = (
            torch.gather(block_table.permute(1, 0, 2), 2, logical_page_group)
            .to(torch.int32)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(batch_local, dp, cp, hkv, query_len)
        )
        live_way = (logical_page % cp).to(torch.int64)
        way_idx = torch.arange(cp, device=hidden_states.device).view(1, 1, cp, 1, 1)
        row_live = way_idx == live_way.unsqueeze(2).unsqueeze(3)
        row_live = row_live.expand(batch_local, dp, cp, hkv, query_len)
        block_id = torch.where(row_live, block_id_dp, torch.iinfo(torch.int32).max).reshape(
            batch_local, rows, query_len
        )
        index_key_cache = ctx_paged_scatter_dp(index_key_cache, block_id, addr, k_updates)

        q_heads_per_kv = num_index_heads // hkv
        ql_eff = q_heads_per_kv * query_len
        q_rows = (
            idx_q.reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, dim)
            .unsqueeze(2)
            .expand(
                batch_local,
                dp,
                cp,
                hkv,
                q_heads_per_kv,
                query_len,
                dim,
            )
            .reshape(batch_local, rows, ql_eff, dim)
        )

        pos_rows_all = (
            position_ids_dp.view(batch_local, dp, 1, 1, query_len)
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, 1, query_len)
        )
        way_rows = torch.arange(rows, device=hidden_states.device).remainder(cp * hkv) // hkv
        dp_rows = torch.arange(rows, device=hidden_states.device) // (cp * hkv)
        page_offsets = torch.arange(page_block_size, device=hidden_states.device)
        group_order = (
            torch.arange(page_groups_per_kv_block, device=hidden_states.device)
            .view(page_groups_per_core, num_cores)
            .transpose(0, 1)
            .reshape(num_cores, page_groups_per_core)
        )

        candidate_score_groups: list[torch.Tensor] = []
        candidate_block_groups: list[torch.Tensor] = []
        device_topk = cfg.index_topk_blocks
        for batch_idx in range(batch_local):
            q_local = q_rows[batch_idx : batch_idx + 1]
            q_5d = q_local.unsqueeze(2).expand(1, rows, num_cores, ql_eff, dim)
            bt_local = block_table[:, batch_idx]
            table_rows = bt_local.index_select(0, dp_rows)
            device_block_score_groups: list[torch.Tensor] = []
            device_block_id_groups: list[torch.Tensor] = []

            def logical_page_groups_for_block(block_idx: int) -> torch.Tensor:
                return block_idx * page_groups_per_kv_block + group_order

            def logical_pages_for_rows(block_idx: int) -> torch.Tensor:
                logical_page_groups = logical_page_groups_for_block(block_idx)
                page_groups_by_row = logical_page_groups.reshape(1, -1).expand(rows, -1)
                return (page_groups_by_row * cp + way_rows.view(rows, 1)).view(rows, num_cores, page_groups_per_core)

            def read_page_groups(block_idx: int) -> torch.Tensor:
                logical_page_groups = logical_page_groups_for_block(block_idx)
                page_groups_by_row = logical_page_groups.reshape(1, -1).expand(rows, -1)
                physical_pages_by_row = torch.gather(table_rows, 1, page_groups_by_row)
                key_flat = ctx_gather_paged_kv_dp(
                    index_key_cache,
                    physical_pages_by_row.transpose(0, 1).contiguous().to(torch.int32),
                )
                return key_flat.view(1, rows, num_cores, tokens_per_core, dim)

            def score_page_groups(block_idx: int, key_5d: torch.Tensor) -> torch.Tensor:
                score_block = torch.matmul(q_5d.float(), key_5d.transpose(-1, -2).float())
                logical_pages_by_row = logical_pages_for_rows(block_idx)
                token_positions = (
                    logical_pages_by_row.unsqueeze(-1) * page_block_size + page_offsets.view(1, 1, 1, -1)
                ).reshape(rows, num_cores, tokens_per_core)
                causal_mask = token_positions.view(1, rows, num_cores, 1, tokens_per_core) > pos_rows_all[
                    batch_idx : batch_idx + 1
                ].unsqueeze(2)
                score_block = score_block.masked_fill(causal_mask, MASKED_ATTENTION_LOGIT)
                block_scores_core = score_block.view(
                    1,
                    rows,
                    num_cores,
                    ql_eff,
                    page_groups_per_core,
                    page_block_size,
                ).amax(dim=-1)
                return block_scores_core

            key_5d = read_page_groups(0)
            next_key_5d = read_page_groups(1) if num_kv_blocks > 1 else None
            for block_idx in range(num_kv_blocks - 1):
                block_scores_core = score_page_groups(block_idx, key_5d)
                block_ids_rows = logical_pages_for_rows(block_idx).view(1, rows, num_cores, page_groups_per_core)
                device_block_score_groups.append(block_scores_core)
                device_block_id_groups.append(block_ids_rows)
                key_5d = next_key_5d
                if block_idx + 2 < num_kv_blocks:
                    next_key_5d = read_page_groups(block_idx + 2)

            epilogue_idx = num_kv_blocks - 1
            epilogue_scores = score_page_groups(epilogue_idx, key_5d)
            epilogue_ids = logical_pages_for_rows(epilogue_idx).view(1, rows, num_cores, page_groups_per_core)
            device_block_score_groups.append(epilogue_scores)
            device_block_id_groups.append(epilogue_ids)

            device_block_scores = torch.cat(device_block_score_groups, dim=4)
            device_block_ids = (
                torch.cat(device_block_id_groups, dim=3)
                .unsqueeze(3)
                .expand(1, rows, num_cores, ql_eff, num_kv_blocks * page_groups_per_core)
            )
            # current_block_rows = (
            #     (position_ids_dp[batch_idx : batch_idx + 1, :, 0] // cfg.index_block_size)
            #     .view(1, dp, 1, 1)
            #     .expand(1, dp, cp, hkv)
            #     .reshape(1, rows)
            # )
            current_block_rows = torch.div(
                position_ids_dp[batch_idx : batch_idx + 1, :, 0], cfg.index_block_size, rounding_mode="floor"
            )
            current_block_rows = current_block_rows.view(1, dp, 1, 1).expand(1, dp, cp, hkv).reshape(1, rows)
            future_block = device_block_ids >= current_block_rows.view(1, rows, 1, 1, 1)
            device_block_scores = torch.where(
                future_block,
                torch.full_like(device_block_scores, MASKED_ATTENTION_LOGIT),
                device_block_scores,
            )
            core_topk = min(cfg.index_topk_blocks, device_block_scores.shape[-1])
            core_topk_scores, core_topk_indices = torch.topk(device_block_scores, k=core_topk, dim=-1)
            core_topk_block_ids = torch.gather(device_block_ids, -1, core_topk_indices)
            core_candidate_scores = core_topk_scores.permute(0, 1, 3, 2, 4).reshape(
                1, rows, ql_eff, num_cores * core_topk
            )
            core_candidate_block_ids = core_topk_block_ids.permute(0, 1, 3, 2, 4).reshape(
                1, rows, ql_eff, num_cores * core_topk
            )
            device_topk = min(cfg.index_topk_blocks, core_candidate_scores.shape[-1])
            device_topk_scores, device_topk_indices = torch.topk(core_candidate_scores, k=device_topk, dim=-1)
            device_topk_block_ids = torch.gather(core_candidate_block_ids, -1, device_topk_indices)
            candidate_score_groups.append(device_topk_scores)
            candidate_block_groups.append(device_topk_block_ids)

        candidate_scores = (
            torch.cat(candidate_score_groups, dim=0)
            .view(
                batch_local,
                dp,
                cp,
                hkv,
                q_heads_per_kv,
                query_len,
                device_topk,
            )
            .permute(0, 1, 3, 4, 5, 2, 6)
            .reshape(
                batch_local,
                dp,
                hkv,
                q_heads_per_kv,
                query_len,
                cp * device_topk,
            )
        )
        candidate_block_indices = (
            torch.cat(candidate_block_groups, dim=0)
            .view(
                batch_local,
                dp,
                cp,
                hkv,
                q_heads_per_kv,
                query_len,
                device_topk,
            )
            .permute(0, 1, 3, 4, 5, 2, 6)
            .reshape(
                batch_local,
                dp,
                hkv,
                q_heads_per_kv,
                query_len,
                cp * device_topk,
            )
        )
        topk_scores, candidate_topk = torch.topk(candidate_scores, k=cfg.index_topk_blocks, dim=-1)
        block_indices = torch.gather(candidate_block_indices, -1, candidate_topk)
        topk_scores = topk_scores.permute(1, 0, 2, 3, 4, 5).reshape(
            batch, num_index_heads, query_len, cfg.index_topk_blocks
        )
        block_indices = block_indices.permute(1, 0, 2, 3, 4, 5).reshape(
            batch, num_index_heads, query_len, cfg.index_topk_blocks
        )
        # current_block_ids = (
        #     (position_ids // cfg.index_block_size)
        #     .view(batch, 1, query_len, 1)
        #     .expand(batch, num_index_heads, query_len, 1)
        #     .to(block_indices.dtype)
        # )
        current_block_ids = torch.div(position_ids, cfg.index_block_size, rounding_mode="floor")
        current_block_ids = (
            current_block_ids.view(batch, 1, query_len, 1)
            .expand(batch, num_index_heads, query_len, 1)
            .to(block_indices.dtype)
        )
        block_indices = torch.cat((block_indices[..., :-1], current_block_ids), dim=-1)
        topk_scores = torch.cat((topk_scores[..., :-1], torch.zeros_like(topk_scores[..., -1:])), dim=-1)
        block_valid = topk_scores > (MASKED_ATTENTION_LOGIT / 2)
        block_indices = block_indices[:, :, 0]
        block_valid = block_valid[:, :, 0]
        safe_block_indices = torch.where(block_valid, block_indices, torch.zeros_like(block_indices)).to(torch.int32)
        if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()):
            self.last_block_indices = block_indices.detach()
        # safe_block_indices: [B, index_n_heads, selected_blocks]
        # block_valid:        [B, index_n_heads, selected_blocks]
        # index_key_cache: [physical_blocks, DP*cp*indexer_n_head, PBS, D]
        past_key_values.index_keys[layer_idx] = index_key_cache
        return safe_block_indices, block_valid

    def _select_blocks_dp(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: "QEffMiniMaxSparseCache",
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        blocking_config: Optional[AttentionBlockingConfig] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DP-optimised block selection using the GP cache custom operations.

        Mirrors the benchmark's ``_select_blocks`` DP path while preserving
        QEfficient's standard cache serialization.
        """
        cfg = self.config
        batch, query_len = hidden_states.shape[0], hidden_states.shape[1]
        ctx_len = blocking_config.ctx_len

        dp = blocking_config.msa_indexer_dp
        cp = blocking_config.msa_indexer_cp if (blocking_config and blocking_config.msa_indexer_cp) else 1
        hkv = blocking_config.indexer_n_head if (blocking_config and blocking_config.indexer_n_head) else 1
        num_kv_blocks = max(
            1,
            int(
                getattr(blocking_config, "indexer_num_blocks", None)
                or getattr(blocking_config, "num_kv_blocks", None)
                or 1
            ),
        )
        num_cores = (
            blocking_config.num_cores_per_device if (blocking_config and blocking_config.num_cores_per_device) else 1
        )
        skip_kv = bool(getattr(blocking_config, "skip_kv", True))
        num_index_heads = cfg.index_n_heads

        num_blocks = (ctx_len + cfg.index_block_size - 1) // cfg.index_block_size
        selected_blocks = min(cfg.index_topk_blocks, num_blocks)

        if query_len != 1:
            raise ValueError("MSA indexer GP selection is decode-only; QL must be 1.")
        if cfg.index_local_blocks != 1:
            raise ValueError("MSA indexer GP decode requires index_local_blocks == 1.")
        if dp < 1 or batch % dp:
            raise ValueError("MSA indexer batch size must be divisible by msa_indexer_dp.")
        if cp < 1 or ctx_len % cp:
            raise ValueError("MSA indexer ctx_len must be divisible by msa_indexer_cp.")
        batch_local = batch // dp
        rows = dp * cp * hkv
        cache_slots = ctx_len // cp
        dim = cfg.index_head_dim
        index_key_cache = past_key_values.index_keys.get(layer_idx)
        flat_rows = batch_local * rows
        index_cache_is_flat_dp_layout = False
        index_cache_is_dp_layout = False
        if index_key_cache is None:
            index_cache_is_flat_dp_layout = cp > 1
            cache_shape = (flat_rows, cache_slots, dim) if index_cache_is_flat_dp_layout else (batch, 1, ctx_len, dim)
            index_key_cache = torch.zeros(cache_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            index_cache_is_flat_dp_layout = cp > 1 and self._is_flat_dp_cache_shape(
                index_key_cache, flat_rows, cache_slots
            )
            index_cache_is_dp_layout = cp > 1 and self._is_dp_cache_shape(
                index_key_cache, batch_local, rows, cache_slots
            )
        if not index_cache_is_dp_layout and not index_cache_is_flat_dp_layout:
            index_key_cache = self._to_dp_cache_shape(index_key_cache, dp, rows, cp)

        # if tuple(index_key_cache.shape) != expected_cache_shape:
        #     raise ValueError(
        #         f"MSA index_key_cache shape {tuple(index_key_cache.shape)} != "
        #         f"{expected_cache_shape}."
        #     )

        if cache_slots % num_kv_blocks:
            raise ValueError(
                f"compact indexer cache slots ({cache_slots}) must be divisible by num_kv_blocks ({num_kv_blocks})."
            )
        cache_block_size = cache_slots // num_kv_blocks
        if cache_block_size % num_cores:
            raise ValueError(
                f"compact indexer KV block length ({cache_block_size}) must be "
                f"divisible by num_cores_per_device ({num_cores})."
            )
        tokens_per_core = cache_block_size // num_cores
        if cfg.index_block_size % cp:
            raise ValueError(f"index_block_size ({cfg.index_block_size}) must be divisible by msa_indexer_cp ({cp}).")
        if tokens_per_core % cfg.index_block_size:
            raise ValueError(
                f"compact indexer tokens per core ({tokens_per_core}) must be "
                f"divisible by index_block_size ({cfg.index_block_size})."
            )
        blocks_per_core = tokens_per_core // cfg.index_block_size

        # The external batch is DP-major: [dp0 local batches, dp1 local
        # batches, ...].  Convert once to the internal [B_local, DP, ...]
        # layout; keep the row axis [B_local, DP*cp*hkv] unchanged.
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        if cos.ndim == 3:
            cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
            sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        else:
            cos_dp, sin_dp = cos, sin
        idx_q = self.q_proj(hidden_states).view(dp, batch_local, query_len, num_index_heads, dim).permute(1, 0, 3, 2, 4)
        idx_k = self.k_proj(hidden_states).view(dp, batch_local, query_len, hkv, dim).permute(1, 0, 3, 2, 4)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q = self._apply_rope_dp(
            idx_q,
            cos_dp,
            sin_dp,
            int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        )
        idx_k = self._apply_rope_dp(
            idx_k,
            cos_dp,
            sin_dp,
            int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        )

        k_updates = (
            idx_k.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, query_len, dim)
            .reshape(batch_local, rows, query_len, dim)
        )
        logical_index_block = position_ids_dp // cfg.index_block_size
        live_way = (
            (logical_index_block % cp)[:, :, None, None, :]
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        row_idx = torch.arange(rows, device=hidden_states.device).view(1, rows, 1)
        row_way = row_idx.remainder(cp * hkv) // hkv
        row_live = row_way == live_way
        batch_id = (
            torch.arange(batch_local, device=hidden_states.device)
            .view(batch_local, 1, 1)
            .expand(batch_local, rows, query_len)
        )
        block_id = torch.where(row_live, batch_id, torch.iinfo(torch.int32).max).to(torch.int32)
        addr = (
            ((logical_index_block // cp) * cfg.index_block_size + position_ids_dp % cfg.index_block_size)
            .to(torch.int32)[:, :, None, None, :]
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        if index_cache_is_flat_dp_layout:
            row_live_flat = row_live.reshape(flat_rows, query_len)
            addr_flat = addr.reshape(flat_rows, query_len)
            safe_addr = torch.where(row_live_flat, addr_flat, torch.zeros_like(addr_flat)).to(torch.int32)
            previous_updates = ctx_gather_3d(index_key_cache, torch.zeros_like(safe_addr))
            scatter_updates = torch.where(
                row_live_flat.unsqueeze(-1), k_updates.reshape(flat_rows, query_len, dim), previous_updates
            )
            index_key_cache = ctx_scatter_3d(index_key_cache, safe_addr, scatter_updates)
        else:
            index_key_cache = ctx_paged_scatter_dp(index_key_cache, block_id, addr, k_updates)

        q_heads_per_kv = num_index_heads // hkv
        ql_eff = q_heads_per_kv * query_len
        q_rows = (
            idx_q.reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, dim)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, dim)
            .reshape(batch_local, rows, ql_eff, dim)
        )

        q_pos_rows_all = (
            position_ids_dp[:, :, 0]
            .view(batch_local, dp, 1, 1)
            .expand(batch_local, dp, cp, hkv)
            .reshape(batch_local, rows)
        )

        block_ranges: list[tuple[int, int]] = []
        for block_idx in range(num_kv_blocks):
            start = block_idx * cache_block_size
            end = min(start + cache_block_size, cache_slots)
            if start < end:
                block_ranges.append((start, end))

        candidate_score_groups: list[torch.Tensor] = []
        candidate_block_groups: list[torch.Tensor] = []
        local_batch_size = 1
        is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()

        def make_block_ids_rows(start: int, local: int) -> torch.Tensor:
            block_start = start * cp // cfg.index_block_size
            block_ids_core = (
                block_start
                + torch.arange(cp, device=hidden_states.device).view(cp, 1, 1)
                + torch.arange(num_cores, device=hidden_states.device).view(1, num_cores, 1) * (cp * blocks_per_core)
                + torch.arange(blocks_per_core, device=hidden_states.device).view(1, 1, blocks_per_core) * cp
            )
            return (
                block_ids_core.view(1, 1, cp, 1, num_cores, blocks_per_core)
                .expand(local, dp, cp, hkv, num_cores, blocks_per_core)
                .reshape(local, rows, num_cores, blocks_per_core)
            )

        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            q_local = q_rows[batch_start:batch_end]
            key_local = index_key_cache[batch_start:batch_end] if not index_cache_is_flat_dp_layout else index_key_cache
            pos_local = position_ids_dp[batch_start:batch_end]
            q_5d = q_local.unsqueeze(2).expand(local, rows, num_cores, ql_eff, dim)
            # q_5d: [local, rows, num_cores, ql_eff, dim]
            device_block_score_groups: list[torch.Tensor] = []
            device_block_id_groups: list[torch.Tensor] = []
            q_block_rows = (
                (position_ids_dp[batch_start:batch_end, :, 0] // cfg.index_block_size)
                .view(local, dp, 1, 1)
                .expand(local, dp, cp, hkv)
                .reshape(local, rows)
            )

            def read_key_block(start: int, end: int) -> torch.Tensor:
                return (
                    self._read_blocked_k_flat_dp(
                        key_local.reshape(batch_local, rows, cache_slots, dim)[batch_start:batch_end].reshape(
                            local * rows, cache_slots, dim
                        ),
                        pos_local,
                        start,
                        end,
                        cfg.index_block_size,
                        cp,
                        hkv,
                    )
                    if index_cache_is_flat_dp_layout
                    else self._read_blocked_k_dp(key_local, pos_local, start, end, cfg.index_block_size, cp, hkv)
                )

            def compute_block_scores(key_5d: torch.Tensor) -> torch.Tensor:
                score_block = torch.matmul(q_5d.float(), key_5d.transpose(-1, -2).float())
                score_block = score_block.view(
                    local,
                    rows,
                    num_cores,
                    ql_eff,
                    blocks_per_core,
                    cfg.index_block_size,
                )
                return score_block.amax(dim=-1)

            prologue_start, prologue_end = block_ranges[0]
            key_5d = read_key_block(prologue_start, prologue_end).view(local, rows, num_cores, tokens_per_core, dim)
            stopped_on_future = False
            for block_idx in range(len(block_ranges) - 1):
                start, _ = block_ranges[block_idx]
                block_ids_rows = make_block_ids_rows(start, local)
                block_skip_future = (
                    torch.tensor(start * cp, device=hidden_states.device, dtype=q_pos_rows_all.dtype)
                    > q_pos_rows_all[batch_start:batch_end]
                )
                if skip_kv and not is_export and bool(block_skip_future.all().item()):
                    masked_block = torch.full(
                        (local, rows, num_cores, ql_eff, blocks_per_core),
                        MASKED_ATTENTION_LOGIT,
                        dtype=torch.float32,
                        device=hidden_states.device,
                    )
                    for remaining_start, _ in block_ranges[block_idx:]:
                        device_block_score_groups.append(masked_block)
                        device_block_id_groups.append(make_block_ids_rows(remaining_start, local))
                    stopped_on_future = True
                    break

                block_scores_core = compute_block_scores(key_5d)
                if skip_kv:
                    block_scores_core = torch.where(
                        block_skip_future.view(local, rows, 1, 1, 1),
                        _scalar_like(block_scores_core, MASKED_ATTENTION_LOGIT),
                        block_scores_core,
                    )

                next_start, next_end = block_ranges[block_idx + 1]
                key_5d = read_key_block(next_start, next_end).view(local, rows, num_cores, tokens_per_core, dim)
                device_block_score_groups.append(block_scores_core)
                device_block_id_groups.append(block_ids_rows)

            if not stopped_on_future:
                epilogue_idx = len(block_ranges) - 1
                epilogue_start, _ = block_ranges[epilogue_idx]
                epilogue_block_ids = make_block_ids_rows(epilogue_start, local)
                epilogue_skip_future = (
                    torch.tensor(epilogue_start * cp, device=hidden_states.device, dtype=q_pos_rows_all.dtype)
                    > q_pos_rows_all[batch_start:batch_end]
                )
                epilogue_scores = compute_block_scores(key_5d)
                if skip_kv:
                    epilogue_scores = torch.where(
                        epilogue_skip_future.view(local, rows, 1, 1, 1),
                        _scalar_like(epilogue_scores, MASKED_ATTENTION_LOGIT),
                        epilogue_scores,
                    )
                device_block_score_groups.append(epilogue_scores)
                device_block_id_groups.append(epilogue_block_ids)

            if not device_block_score_groups:
                raise ValueError("MSA indexer selection produced no cache blocks.")

            device_block_scores = torch.cat(device_block_score_groups, dim=4)
            device_block_ids = (
                torch.cat(device_block_id_groups, dim=3)
                .unsqueeze(3)
                .expand(local, rows, num_cores, ql_eff, num_kv_blocks * blocks_per_core)
            )
            future_block = device_block_ids >= q_block_rows.view(local, rows, 1, 1, 1)
            device_block_scores = torch.where(
                future_block,
                _scalar_like(device_block_scores, MASKED_ATTENTION_LOGIT),
                device_block_scores,
            )
            # [local, DP*cp*indexer_n_head, ql_eff,
            #  num_kv_blocks*C*blocks_per_core]
            core_topk = min(selected_blocks, device_block_scores.shape[-1])
            core_topk_scores, core_topk_indices = torch.topk(device_block_scores, k=core_topk, dim=-1)
            core_topk_block_ids = torch.gather(device_block_ids, -1, core_topk_indices)
            core_candidate_scores = core_topk_scores.permute(0, 1, 3, 2, 4).reshape(
                local, rows, ql_eff, num_cores * core_topk
            )
            core_candidate_block_ids = core_topk_block_ids.permute(0, 1, 3, 2, 4).reshape(
                local, rows, ql_eff, num_cores * core_topk
            )
            device_topk = min(selected_blocks, core_candidate_scores.shape[-1])
            device_topk_scores, device_topk_indices = torch.topk(core_candidate_scores, k=device_topk, dim=-1)
            device_topk_block_ids = torch.gather(core_candidate_block_ids, -1, device_topk_indices)
            candidate_score_groups.append(device_topk_scores)
            candidate_block_groups.append(device_topk_block_ids)

        candidate_scores = (
            torch.cat(candidate_score_groups, dim=0)
            .view(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, device_topk)
            .permute(0, 1, 3, 4, 5, 2, 6)
            .reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, cp * device_topk)
        )
        candidate_block_indices = (
            torch.cat(candidate_block_groups, dim=0)
            .view(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, device_topk)
            .permute(0, 1, 3, 4, 5, 2, 6)
            .reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, cp * device_topk)
        )
        # candidate_*:
        # [batch_local, dp, indexer_n_head, q_heads_per_kv, QL,
        #  cp * device_topk]
        # Second Top-K: merge cp candidates on each DP/Hkv lane.
        topk_scores, candidate_topk = torch.topk(candidate_scores, k=selected_blocks, dim=-1)
        block_indices = torch.gather(candidate_block_indices, -1, candidate_topk)
        current_block_ids = (
            (position_ids.view(dp, batch_local, query_len).permute(1, 0, 2) // cfg.index_block_size)
            .view(batch_local, dp, 1, 1, query_len, 1)
            .expand(
                batch_local,
                dp,
                hkv,
                q_heads_per_kv,
                query_len,
                1,
            )
            .to(block_indices.dtype)
        )
        block_indices = torch.cat([block_indices[..., :-1], current_block_ids], dim=-1)
        topk_scores = torch.cat(
            [topk_scores[..., :-1], torch.zeros_like(topk_scores[..., -1:])],
            dim=-1,
        )
        # topk_scores/block_indices: [batch_local, dp, hkv,
        # q_heads_per_kv, query_len, top_k]
        # Internally the candidates are [B_local, DP, H, ...].  The external
        # batch contract is DP-major, so move DP in front of B_local before
        # flattening; a direct reshape would produce local-batch-major order.
        topk_scores = topk_scores.permute(1, 0, 2, 3, 4, 5).reshape(batch, num_index_heads, query_len, selected_blocks)
        block_indices = block_indices.permute(1, 0, 2, 3, 4, 5).reshape(
            batch, num_index_heads, query_len, selected_blocks
        )
        block_valid = topk_scores > (MASKED_ATTENTION_LOGIT / 2)
        block_indices = block_indices[:, :, 0]
        block_valid = block_valid[:, :, 0]
        safe_indices = torch.where(block_valid, block_indices, torch.zeros_like(block_indices)).to(torch.int32)
        if not index_cache_is_dp_layout and not index_cache_is_flat_dp_layout:
            index_key_cache = self._from_dp_cache_shape(index_key_cache, dp, cp)
        past_key_values.index_keys[layer_idx] = index_key_cache
        if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()):
            self.last_block_indices = block_indices.detach()
        # safe_indices/block_valid: [B, index_n_heads, selected_blocks]
        return safe_indices, block_valid

    def _select_blocks_prefill(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: "QEffMiniMaxSparseCache",
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        blocking_config: Optional[AttentionBlockingConfig],
        paged_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select sparse MSA blocks for every token in a prefill sequence."""
        cfg = self.config
        batch, query_len = hidden_states.shape[:2]
        index_key_cache = past_key_values.index_keys[layer_idx]
        dim = cfg.index_head_dim
        num_index_heads = cfg.index_n_heads
        indexer_n_head = getattr(blocking_config, "indexer_n_head", None) or getattr(cfg, "indexer_n_head", 1) or 1
        num_kv_blocks = max(
            1,
            int(
                getattr(blocking_config, "indexer_num_blocks", None)
                or getattr(blocking_config, "num_kv_blocks", None)
                or 1
            ),
        )
        num_cores = int(getattr(blocking_config, "num_cores_per_device", 1) or 1)
        ctx_len = int(
            getattr(blocking_config, "ctx_len", None)
            or getattr(cfg, "ctx_len", None)
            or (
                paged_block_table.shape[1] * index_key_cache.shape[2]
                if paged_block_table is not None
                else index_key_cache.shape[-2]
            )
        )
        q_block_size = getattr(blocking_config, "indexer_q_size", None) or cfg.index_block_size
        q_block_chunk = int(getattr(blocking_config, "indexer_q_chunk", None) or q_block_size)
        index_block_size = cfg.index_block_size
        if query_len <= 1:
            raise ValueError("MSA prefill selection requires QL > 1.")
        if getattr(blocking_config, "msa_indexer_dp", 1) not in (None, 1) or getattr(
            blocking_config, "msa_indexer_cp", 1
        ) not in (None, 1):
            raise ValueError("MSA prefill selection currently requires DP=1 and CP=1.")
        if num_index_heads % indexer_n_head:
            raise ValueError("MSA prefill index_n_heads must be divisible by indexer_n_head.")
        if indexer_n_head != 1:
            raise NotImplementedError("MSA prefill currently supports one indexer KV head.")
        if ctx_len % index_block_size:
            raise ValueError("MSA prefill selection requires ctx_len divisible by index_block_size.")
        if ctx_len % num_kv_blocks:
            raise ValueError("MSA prefill selection requires ctx_len divisible by num_kv_blocks.")
        if q_block_size <= 0 or q_block_chunk < q_block_size:
            raise ValueError("MSA prefill Q block sizes must be positive and ordered.")
        if q_block_chunk % q_block_size or q_block_size % num_cores:
            raise ValueError("MSA prefill Q block size must divide q_block_chunk and num_cores.")
        if query_len % q_block_chunk:
            raise ValueError("MSA prefill query length must divide evenly by q_block_chunk.")

        if paged_block_table is None:
            expected_cache_shape = (batch, indexer_n_head, ctx_len, dim)
            if tuple(index_key_cache.shape) != expected_cache_shape:
                raise ValueError(
                    f"MSA prefill index_key_cache shape {tuple(index_key_cache.shape)} != {expected_cache_shape}."
                )
        else:
            if (getattr(blocking_config, "msa_indexer_dp", 1) or 1) != 1 or (
                getattr(blocking_config, "msa_indexer_cp", 1) or 1
            ) != 1:
                raise ValueError("Paged MSA prefill selection currently requires DP=1 and CP=1.")
            if index_key_cache.ndim != 4 or tuple(index_key_cache.shape[1:]) != (
                indexer_n_head,
                index_key_cache.shape[2],
                dim,
            ):
                raise ValueError("Paged MSA index cache shape is incompatible with prefill.")
            page_size = int(index_key_cache.shape[2])
            configured_page_size = getattr(blocking_config, "page_block_size", None)
            if configured_page_size is not None and int(configured_page_size) != page_size:
                raise ValueError("Paged MSA prefill page size does not match blocking_config.page_block_size.")
            if page_size != index_block_size:
                raise ValueError("Paged MSA prefill requires page size == index_block_size.")
            if paged_block_table.shape[1] < (ctx_len + page_size - 1) // page_size:
                raise ValueError("Paged MSA prefill indexer block_table is too short for ctx_len.")

        idx_q = self._project_index_q_prefill(hidden_states, blocking_config)[..., : num_index_heads * dim]
        idx_q = idx_q.view(batch, query_len, num_index_heads, dim).transpose(1, 2)
        idx_k = self.k_proj(hidden_states)[..., : indexer_n_head * dim]
        idx_k = idx_k.view(batch, query_len, indexer_n_head, dim).transpose(1, 2)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q = self._apply_rope(idx_q, cos, sin)
        idx_k = self._apply_rope(idx_k, cos, sin)
        if paged_block_table is None:
            index_key_cache = m3_ctx_scatter(index_key_cache, position_ids.to(torch.int32), idx_k)
        else:
            index_key_cache = self._write_msa_paged_prefill_cache(
                index_key_cache,
                idx_k,
                position_ids,
                paged_block_table.unsqueeze(0),
            )

        kv_block_size = ctx_len // num_kv_blocks
        index_block_size = cfg.index_block_size
        num_blocks = ctx_len // index_block_size
        mask_value = MASKED_ATTENTION_LOGIT
        topk = min(cfg.index_topk_blocks, num_blocks)
        topk_indices_chunks: list[torch.Tensor] = []
        topk_valid_chunks: list[torch.Tensor] = []
        num_q_chunks = int(query_len // q_block_chunk)
        num_q_blocks_per_chunk = q_block_chunk // q_block_size
        compile_seq_len = getattr(blocking_config, "prefill_compile_seq_len", None)
        q_chunks = _dynamic_sequence_chunks(idx_q, num_q_chunks, dim=2, compile_axis_size=compile_seq_len)
        position_chunks = _dynamic_sequence_chunks(position_ids, num_q_chunks, dim=1, compile_axis_size=compile_seq_len)
        nested_q_blocks = _dynamic_sequence_nested_chunks(
            idx_q,
            num_q_chunks,
            num_q_blocks_per_chunk,
            dim=2,
            compile_axis_size=compile_seq_len,
        )
        nested_position_blocks = _dynamic_sequence_nested_chunks(
            position_ids,
            num_q_chunks,
            num_q_blocks_per_chunk,
            dim=1,
            compile_axis_size=compile_seq_len,
        )
        for q_chunk_idx, (q_chunk, query_positions) in enumerate(zip(q_chunks, position_chunks)):
            query_chunk = q_chunk.shape[2]
            if query_chunk % q_block_size:
                raise ValueError(
                    "MSA prefill query length must be divisible by q_block_size within every q_block_chunk."
                )
            score_blocks: list[list[torch.Tensor]] = [[] for _ in range(num_q_blocks_per_chunk)]
            q_blocks = nested_q_blocks[q_chunk_idx]
            query_position_blocks = nested_position_blocks[q_chunk_idx]
            q_chunk_position = query_positions.max(dim=-1).values
            is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()

            for block_idx in range(num_kv_blocks):
                start = block_idx * kv_block_size
                end = start + kv_block_size
                block_skip_future_chunk = (
                    torch.full_like(q_chunk_position, start, device=hidden_states.device) > q_chunk_position
                )
                if blocking_config.skip_kv and not is_export and bool(block_skip_future_chunk.all().item()):
                    # The remaining context blocks are entirely in the
                    # future for every query in this Q chunk.  Selection
                    # still needs one score tensor per block so that the
                    # later top-k and reshape keep the same static layout,
                    # but no cache gather or matmul is needed.
                    masked_block = torch.full(
                        (
                            batch,
                            num_index_heads,
                            q_block_size,
                            kv_block_size,
                        ),
                        mask_value,
                        dtype=torch.float32,
                        device=hidden_states.device,
                    )
                    for q_block_idx in range(len(score_blocks)):
                        score_blocks[q_block_idx].extend(masked_block.clone() for _ in range(block_idx, num_kv_blocks))
                    break
                if paged_block_table is None:
                    key_block = (
                        self._read_blocked_k_dp(
                            index_key_cache,
                            position_ids.unsqueeze(1),
                            start,
                            end,
                            index_block_size,
                            1,
                            indexer_n_head,
                        )
                        .squeeze(1)
                        .float()
                    )
                else:
                    key_block = (
                        self._read_msa_prefill_paged_block(
                            index_key_cache,
                            paged_block_table,
                            start,
                            end,
                            position_ids.max(dim=-1).values,
                        )
                        .squeeze(1)
                        .float()
                    )
                key_offsets = torch.arange(end - start, device=hidden_states.device).view(1, 1, 1, -1)
                for q_block_idx, (q_block, q_positions) in enumerate(zip(q_blocks, query_position_blocks)):
                    q_core = q_block.view(
                        batch,
                        num_index_heads,
                        num_cores,
                        -1,
                        dim,
                    ).permute(0, 2, 1, 3, 4)
                    key_core = key_block.unsqueeze(1).expand(batch, num_cores, kv_block_size, dim)
                    scores = (
                        torch.matmul(
                            q_core.float(),
                            key_core.transpose(-1, -2).unsqueeze(2),
                        )
                        .permute(0, 2, 1, 3, 4)
                        .reshape(batch, num_index_heads, -1, kv_block_size)
                    )
                    causal_mask = key_offsets > (q_positions[:, None, :, None] - start)
                    scores = torch.where(
                        causal_mask,
                        _scalar_like(scores, mask_value),
                        scores,
                    )
                    if blocking_config.skip_kv:
                        scores = torch.where(
                            (start > q_positions).view(batch, 1, -1, 1),
                            _scalar_like(scores, mask_value),
                            scores,
                        )
                    score_blocks[q_block_idx].append(scores)

            for q_block_idx in range(len(score_blocks)):
                scores = torch.cat(score_blocks[q_block_idx], dim=-1)
                scores = scores.view(
                    batch,
                    num_index_heads,
                    -1,
                    num_blocks,
                    index_block_size,
                ).amax(dim=-1)
                q_positions = query_position_blocks[q_block_idx]
                block_ids = torch.arange(num_blocks, device=hidden_states.device).view(1, 1, 1, num_blocks)
                if blocking_config.skip_kv:
                    scores = torch.where(
                        (block_ids * index_block_size > q_positions[:, None, :, None]),
                        _scalar_like(scores, mask_value),
                        scores,
                    )
                for local_offset in range(cfg.index_local_blocks):
                    # local_block = q_positions // index_block_size - local_offset
                    local_block = torch.div(q_positions, index_block_size, rounding_mode="floor") - local_offset
                    local_block = torch.where(
                        local_block >= 0,
                        local_block,
                        torch.zeros_like(local_block),
                    )
                    scores = torch.where(
                        block_ids == local_block[:, None, :, None],
                        _scalar_like(scores, -mask_value),
                        scores,
                    )
                top_scores, top_order = torch.topk(scores, k=topk, dim=-1)
                topk_indices_chunks.append(
                    torch.gather(
                        block_ids.expand_as(scores).to(torch.int32),
                        -1,
                        top_order,
                    )
                )
                topk_valid_chunks.append(top_scores > (mask_value / 2))

        block_indices = torch.cat(topk_indices_chunks, dim=2)
        block_valid = torch.cat(topk_valid_chunks, dim=2)
        safe_indices = torch.where(block_valid, block_indices, torch.zeros_like(block_indices)).to(torch.int32)
        if not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling()):
            self.last_block_indices = safe_indices.detach()
        past_key_values.index_keys[layer_idx] = index_key_cache
        return safe_indices, block_valid


class QEffMiniMaxM3VLAttention(MiniMaxM3VLAttention):
    def _msa_attention_prefill(
        self,
        query_states,
        hidden_states,
        cos,
        sin,
        key_cache,
        value_cache,
        token_indices,
        token_valid,
        position_ids,
        blocking_config,
        paged_block_table=None,
    ):
        cfg = self.config
        batch, _, query_len, head_dim = query_states.shape
        num_kv_heads = cfg.num_key_value_heads
        n_rep = self.num_key_value_groups
        num_cores = int(getattr(blocking_config, "num_cores_per_device", 1) or 1)
        if num_cores % num_kv_heads:
            raise ValueError("Sparse MSA prefill core count must be divisible by num_kv_heads.")
        # Fold KV heads into the hardware-core axis.  Each KV head owns a
        # disjoint group of cores, so the query split and the per-head gather
        # expansion use only this smaller group rather than all device cores.
        cores_per_kv_head = num_cores // num_kv_heads
        ql_chunk = int(getattr(blocking_config, "msa_q_chunk", None) or query_len)
        kv_num_blocks = max(
            1,
            int(
                getattr(blocking_config, "msa_num_kv_blocks", None)
                or getattr(blocking_config, "num_kv_blocks", None)
                or 1
            ),
        )
        selected_blocks = int(token_indices.shape[-1])
        block_size = cfg.index_block_size
        mask_value = float(MIN_MASKED_ATTENTION_VALUE)

        if query_len % cores_per_kv_head:
            raise ValueError("Sparse MSA prefill Q length must be divisible by the cores assigned to each KV head.")
        if query_len % ql_chunk:
            raise ValueError("Sparse MSA prefill Q length must divide evenly by msa_q_chunk.")
        if ql_chunk % cores_per_kv_head:
            raise ValueError("Sparse MSA prefill Q chunks must be divisible by the cores assigned to each KV head.")
        if kv_num_blocks > selected_blocks:
            raise ValueError("Sparse MSA prefill KV block count must not exceed the selected block count.")

        q = self.q_norm(
            self.q_proj(hidden_states).view(batch, query_len, cfg.num_attention_heads, cfg.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(batch, query_len, num_kv_heads, cfg.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, query_len, num_kv_heads, cfg.head_dim).transpose(1, 2)
        q, k = qeff_apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        )
        if paged_block_table is None:
            key_cache = m3_ctx_scatter(key_cache, position_ids.to(torch.int32), k)
            value_cache = m3_ctx_scatter(value_cache, position_ids.to(torch.int32), v)
        else:
            if (getattr(blocking_config, "msa_attn_dp", 1) or 1) != 1 or (
                getattr(blocking_config, "msa_attn_cp", 1) or 1
            ) != 1:
                raise ValueError("Paged MSA prefill attention currently requires DP=1 and CP=1.")
            if paged_block_table.ndim != 2 or paged_block_table.shape[0] != batch:
                raise ValueError("Paged MSA prefill attention block_table must have shape [B, pages].")
            page_size = int(key_cache.shape[2])
            configured_page_size = getattr(blocking_config, "page_block_size", None)
            if configured_page_size is not None and int(configured_page_size) != page_size:
                raise ValueError("Paged MSA prefill page size does not match blocking_config.page_block_size.")
            expected_paged_shape = (
                key_cache.shape[0],
                num_kv_heads,
                page_size,
                cfg.head_dim,
            )
            if tuple(key_cache.shape) != expected_paged_shape:
                raise ValueError(
                    f"Paged MSA prefill key_cache shape {tuple(key_cache.shape)} != {expected_paged_shape}."
                )
            if tuple(value_cache.shape) != expected_paged_shape:
                raise ValueError(
                    f"Paged MSA prefill value_cache shape {tuple(value_cache.shape)} != {expected_paged_shape}."
                )
            logical_page = position_ids // page_size
            physical_page = torch.gather(paged_block_table.to(torch.int64), 1, logical_page).to(torch.int32)
            block_ids = physical_page.unsqueeze(1).expand(batch, num_kv_heads, query_len)
            addresses = (position_ids % page_size).to(torch.int32).unsqueeze(1).expand_as(block_ids)
            key_cache = ctx_paged_scatter_dp(key_cache, block_ids, addresses, k)
            value_cache = ctx_paged_scatter_dp(value_cache, block_ids, addresses, v)

        q = q.reshape(batch, num_kv_heads, n_rep, query_len, cfg.head_dim)
        offsets = torch.arange(block_size, device=hidden_states.device).view(1, 1, 1, 1, 1, block_size)

        num_q_chunks = int(query_len // ql_chunk)
        compile_seq_len = getattr(blocking_config, "prefill_compile_seq_len", None)
        if torch._dynamo.is_compiling():
            export_seq_len = getattr(blocking_config, "prefill_export_seq_len", None)
            if export_seq_len is not None:
                num_q_chunks = int(export_seq_len // ql_chunk)
        output_seq_len = compile_seq_len if compile_seq_len is not None else query_len

        def _compute_q_chunk(q_chunk, block_chunk, block_valid_chunk, position_chunk):
            chunk_batch = q_chunk.shape[0]
            query_chunk = q_chunk.shape[3]
            chunk_selected_blocks = block_chunk.shape[-1]
            q_per_core = query_chunk // cores_per_kv_head

            # Q, not K, is split across the cores assigned to each KV head.
            # The [Hkv, C_per_Hkv] axes are folded into the device-core axis
            # for attention work, while the cache gather keeps Hkv explicit.
            #   q_core:     [B, Hkv*C_per_Hkv, Q/C_per_Hkv, Hrep, D]
            #   block_core: [B, Hkv, C_per_Hkv, Q/C_per_Hkv, selected_blocks]
            q_core = (
                q_chunk.reshape(
                    chunk_batch,
                    num_kv_heads,
                    n_rep,
                    cores_per_kv_head,
                    q_per_core,
                    cfg.head_dim,
                )
                .permute(0, 1, 3, 4, 2, 5)
                .reshape(
                    chunk_batch,
                    num_kv_heads * cores_per_kv_head,
                    q_per_core,
                    n_rep,
                    cfg.head_dim,
                )
                .float()
            )
            block_core = block_chunk.reshape(
                chunk_batch,
                num_kv_heads,
                cores_per_kv_head,
                q_per_core,
                chunk_selected_blocks,
            )
            block_valid_core = block_valid_chunk.reshape(
                chunk_batch,
                num_kv_heads,
                cores_per_kv_head,
                q_per_core,
                chunk_selected_blocks,
            )
            query_positions = position_chunk.reshape(chunk_batch, 1, cores_per_kv_head, q_per_core, 1)
            q_head_width = q_per_core * n_rep
            q_score_work = q_core.reshape(
                -1,
                n_rep,
                cfg.head_dim,
            )

            m_acc = torch.full(
                (
                    chunk_batch,
                    num_kv_heads * cores_per_kv_head,
                    q_head_width,
                ),
                -_FP16_MAX_VALUE,
                dtype=q_core.dtype,
                device=q_core.device,
            )
            s_acc = torch.zeros_like(m_acc)
            o_acc = torch.zeros(
                chunk_batch,
                num_kv_heads * cores_per_kv_head,
                q_head_width,
                cfg.head_dim,
                dtype=q_core.dtype,
                device=q_core.device,
            )

            for kv_block_idx in range(kv_num_blocks):
                # Split only at MSA block boundaries.  Expanding block IDs to
                # token addresses here keeps the gather working set bounded.
                selected_start = (chunk_selected_blocks * kv_block_idx) // kv_num_blocks
                selected_end = (chunk_selected_blocks * (kv_block_idx + 1)) // kv_num_blocks
                selected_block_count = selected_end - selected_start
                token_count = selected_block_count * block_size
                selected_block_ids = block_core[..., selected_start:selected_end]
                selected_positions = (selected_block_ids.unsqueeze(-1) * block_size + offsets).reshape(
                    chunk_batch,
                    num_kv_heads,
                    cores_per_kv_head,
                    q_per_core,
                    token_count,
                )
                selected_offsets = offsets.expand(
                    1,
                    1,
                    1,
                    1,
                    selected_block_count,
                    block_size,
                ).reshape(1, 1, 1, 1, token_count)
                local_query = (
                    (query_positions.unsqueeze(-2) - selected_block_ids.unsqueeze(-1) * block_size)
                    .expand(
                        chunk_batch,
                        num_kv_heads,
                        cores_per_kv_head,
                        q_per_core,
                        selected_block_count,
                        block_size,
                    )
                    .reshape(
                        chunk_batch,
                        num_kv_heads,
                        cores_per_kv_head,
                        q_per_core,
                        token_count,
                    )
                )
                valid_block = (
                    block_valid_core[..., selected_start:selected_end]
                    .unsqueeze(-1)
                    .expand(
                        chunk_batch,
                        num_kv_heads,
                        cores_per_kv_head,
                        q_per_core,
                        selected_block_count,
                        block_size,
                    )
                    .reshape(
                        chunk_batch,
                        num_kv_heads,
                        cores_per_kv_head,
                        q_per_core,
                        token_count,
                    )
                )
                valid_block = valid_block & (selected_positions < blocking_config.ctx_len)
                # Keep the absolute selected positions for the gather, but
                # express causality as a comparison against the contiguous
                # offset range inside each selected block.
                valid_block = valid_block & (selected_offsets <= local_query)
                # Prefill selectors return safe in-range block IDs (invalid
                # entries are zero); validity controls both the gather
                # sentinel and the attention contribution.
                # Keep the custom gather's established rank-3 index contract.
                # QAIC's importer derives its output size from that layout and
                # can reject a following reshape when given rank-5 indices.
                # Match the GQA blocked gather contract: export-time invalid
                # (future or selector-invalid) tokens use the INT32_MAX
                # sentinel so the compiler can drop their KV reads.  The
                # validity mask is still applied to the scores below to keep
                # eager/export numerics identical.
                invalid_idx_value = (
                    torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() else 0
                )
                safe_selected_positions = torch.where(
                    valid_block,
                    selected_positions,
                    _scalar_like(selected_positions, invalid_idx_value),
                )
                gather_positions = safe_selected_positions.reshape(
                    chunk_batch,
                    num_kv_heads,
                    cores_per_kv_head * q_per_core * token_count,
                ).to(torch.int32)

                if paged_block_table is None:
                    key_block = (
                        ctx_gather_blocked_kv(key_cache, gather_positions)
                        .reshape(
                            chunk_batch,
                            num_kv_heads,
                            cores_per_kv_head,
                            q_per_core,
                            token_count,
                            cfg.head_dim,
                        )
                        .reshape(
                            chunk_batch,
                            num_kv_heads * cores_per_kv_head,
                            q_per_core,
                            token_count,
                            cfg.head_dim,
                        )
                        .float()
                    )
                    value_block = (
                        ctx_gather_blocked_kv(value_cache, gather_positions)
                        .reshape(
                            chunk_batch,
                            num_kv_heads,
                            cores_per_kv_head,
                            q_per_core,
                            token_count,
                            cfg.head_dim,
                        )
                        .reshape(
                            chunk_batch,
                            num_kv_heads * cores_per_kv_head,
                            q_per_core,
                            token_count,
                            cfg.head_dim,
                        )
                        .float()
                    )
                else:
                    page_ids = (
                        block_chunk[..., selected_start:selected_end]
                        .to(torch.int64)
                        .reshape(chunk_batch, num_kv_heads, selected_block_count * query_chunk)
                    )
                    table = (
                        paged_block_table.to(torch.int64)
                        .unsqueeze(1)
                        .expand(chunk_batch, num_kv_heads, paged_block_table.shape[1])
                    )
                    physical_ids = torch.gather(table, 2, page_ids)
                    # Keep the KV-head row explicit in the page gather.  The
                    # previous batch gather accepted only [B, pages] and then
                    # returned every head implicitly, which hides the row/head
                    # split from the compiler.
                    key_pages = _gather_paged_kv_selected_heads(key_cache, physical_ids).view(
                        chunk_batch,
                        num_kv_heads,
                        query_chunk,
                        token_count,
                        cfg.head_dim,
                    )
                    value_pages = _gather_paged_kv_selected_heads(value_cache, physical_ids).view(
                        chunk_batch,
                        num_kv_heads,
                        query_chunk,
                        token_count,
                        cfg.head_dim,
                    )
                    key_block = (
                        key_pages.reshape(
                            chunk_batch,
                            num_kv_heads,
                            cores_per_kv_head,
                            q_per_core,
                            token_count,
                            cfg.head_dim,
                        )
                        .reshape(
                            chunk_batch,
                            num_kv_heads * cores_per_kv_head,
                            q_per_core,
                            token_count,
                            cfg.head_dim,
                        )
                        .float()
                    )
                    value_block = (
                        value_pages.reshape(
                            chunk_batch,
                            num_kv_heads,
                            cores_per_kv_head,
                            q_per_core,
                            token_count,
                            cfg.head_dim,
                        )
                        .reshape(
                            chunk_batch,
                            num_kv_heads * cores_per_kv_head,
                            q_per_core,
                            token_count,
                            cfg.head_dim,
                        )
                        .float()
                    )
                value_block = torch.where(
                    valid_block.reshape(
                        chunk_batch,
                        num_kv_heads * cores_per_kv_head,
                        q_per_core,
                        token_count,
                    ).unsqueeze(-1),
                    value_block,
                    _scalar_like(value_block, 0.0),
                )
                key_score_work = key_block.reshape(
                    -1,
                    token_count,
                    cfg.head_dim,
                ).transpose(1, 2)
                scores_block = torch.bmm(q_score_work, key_score_work).reshape(
                    chunk_batch,
                    num_kv_heads * cores_per_kv_head,
                    q_per_core,
                    n_rep,
                    token_count,
                ) * (cfg.head_dim**-0.5)
                attn_block = torch.where(
                    ~valid_block.reshape(
                        chunk_batch,
                        num_kv_heads * cores_per_kv_head,
                        q_per_core,
                        token_count,
                    ).unsqueeze(3),
                    _scalar_like(scores_block, mask_value),
                    scores_block,
                ).reshape(
                    chunk_batch,
                    num_kv_heads * cores_per_kv_head,
                    q_head_width,
                    token_count,
                )
                m_acc, s_acc, o_acc = update_running_softmax(
                    m_acc,
                    attn_block,
                    s_acc,
                    o_acc,
                    value_block,
                )

            out_chunk = (
                (o_acc / s_acc.clamp_min(1.0).unsqueeze(-1))
                .reshape(
                    chunk_batch,
                    num_kv_heads,
                    cores_per_kv_head,
                    q_per_core,
                    n_rep,
                    cfg.head_dim,
                )
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(
                    chunk_batch,
                    cfg.num_attention_heads,
                    query_chunk,
                    cfg.head_dim,
                )
            )
            return out_chunk

        if torch._dynamo.is_compiling():
            attn_output = torch.zeros(
                batch,
                cfg.num_attention_heads,
                output_seq_len,
                cfg.head_dim,
                dtype=torch.float32,
                device=q.device,
            )

            def _loop_cond(chunk_idx, output):
                return chunk_idx < num_q_chunks

            def _loop_body(chunk_idx, output):
                chunk_idx_value = chunk_idx.item()
                q_chunk = _dynamic_sequence_chunk(q, num_q_chunks, chunk_idx_value, 3, compile_seq_len)
                block_chunk = _dynamic_sequence_chunk(token_indices, num_q_chunks, chunk_idx_value, 2, compile_seq_len)
                block_valid_chunk = _dynamic_sequence_chunk(
                    token_valid, num_q_chunks, chunk_idx_value, 2, compile_seq_len
                )
                position_chunk = _dynamic_sequence_chunk(
                    position_ids, num_q_chunks, chunk_idx_value, 1, compile_seq_len
                )
                out_chunk = _compute_q_chunk(q_chunk, block_chunk, block_valid_chunk, position_chunk)
                chunk_start = chunk_idx_value * out_chunk.shape[2]
                updated_output = torch.slice_scatter(
                    output,
                    out_chunk,
                    dim=2,
                    start=chunk_start,
                    end=chunk_start + out_chunk.shape[2],
                )
                return chunk_idx + 1, updated_output

            loop_index = torch.zeros((), dtype=torch.int64, device=q.device)
            _, attn_output = torch.while_loop(_loop_cond, _loop_body, (loop_index, attn_output))
        else:
            q_chunks = _dynamic_sequence_chunks(q, num_q_chunks, dim=3, compile_axis_size=compile_seq_len)
            token_index_chunks = _dynamic_sequence_chunks(
                token_indices, num_q_chunks, dim=2, compile_axis_size=compile_seq_len
            )
            token_valid_chunks = _dynamic_sequence_chunks(
                token_valid, num_q_chunks, dim=2, compile_axis_size=compile_seq_len
            )
            position_chunks = _dynamic_sequence_chunks(
                position_ids, num_q_chunks, dim=1, compile_axis_size=compile_seq_len
            )
            output_chunks = (
                _compute_q_chunk(q_chunk, block_chunk, block_valid_chunk, position_chunk)
                for q_chunk, block_chunk, block_valid_chunk, position_chunk in zip(
                    q_chunks, token_index_chunks, token_valid_chunks, position_chunks
                )
            )
            attn_output = torch.cat(tuple(output_chunks), dim=2)

        return (
            attn_output,
            key_cache,
            value_cache,
        )

    def _baseline_attention(
        self,
        query_states: torch.Tensor,
        selected_k: torch.Tensor,
        selected_v: torch.Tensor,
        flat_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = query_states.shape[0], query_states.shape[2]
        num_heads = self.config.index_n_heads
        num_kv_groups = self.num_key_value_groups
        q = query_states.view(batch, num_heads, num_kv_groups, seq_len, self.head_dim)
        selected_k = selected_k.unsqueeze(2)
        selected_v = selected_v.unsqueeze(2)
        scores = torch.matmul(q.float(), selected_k.transpose(-1, -2).float()) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~flat_valid[:, :, None, None, :], -1.0e30)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(selected_v.dtype)
        return torch.matmul(probs, selected_v)

    def _baseline_attention_gp(
        self,
        query_states: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        dp: int,
        blocking_config: Optional[AttentionBlockingConfig],
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run selected-token attention in group-parallel layout.

        ``key_cache`` and ``value_cache`` have already been updated by the
        sparse cache writer and use ``[B_local, DP * Hkv, ctx_len, D]``. The
        selected indices may be compressed block IDs, row-local token IDs, or
        the external ``[batch, Hkv, blocks, block_size]`` token layout.
        """
        batch, _, query_len, head_dim = query_states.shape
        cfg = self.config
        hkv = self.config.num_key_value_heads
        num_kv_groups = self.num_key_value_groups
        rows = dp * hkv
        cp = blocking_config.msa_attn_cp if blocking_config and blocking_config.msa_attn_cp else 1
        if cp > 1:
            return self._baseline_attention_gp_cp(
                query_states,
                token_indices,
                token_valid,
                key_cache,
                value_cache,
                dp,
                cp,
                blocking_config,
                position_ids,
            )

        if query_len != 1:
            raise ValueError("MSA attention GP is decode-only; QL must be 1.")
        if dp < 1 or batch % dp:
            raise ValueError(f"MSA batch size {batch} must be divisible by msa_attn_dp={dp}.")

        batch_local = batch // dp
        if tuple(key_cache.shape[:2]) != (batch_local, rows):
            raise ValueError(f"GP key cache shape {tuple(key_cache.shape)} must start with ({batch_local}, {rows}).")
        if tuple(value_cache.shape) != tuple(key_cache.shape):
            raise ValueError("GP key and value cache shapes must match.")

        selected_len: int
        uses_block_indices = (
            token_indices.ndim == 3
            and tuple(token_indices.shape[:2]) == (batch, hkv)
            and token_indices.shape[-1] <= cfg.index_topk_blocks
        )
        if uses_block_indices:
            if position_ids is None:
                raise ValueError("MSA block-index attention requires position_ids.")
            if tuple(token_valid.shape) != tuple(token_indices.shape):
                raise ValueError("MSA block_indices and block_valid shapes must match.")
            if key_cache.shape[2] % cfg.index_block_size:
                raise ValueError("MSA block-index gather requires cache length divisible by index_block_size.")

            selected_blocks = token_indices.shape[-1]
            selected_len = selected_blocks * cfg.index_block_size
            block_ids = (
                token_indices.reshape(dp, batch_local, hkv, selected_blocks)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_blocks)
            )
            block_valid = (
                token_valid.reshape(dp, batch_local, hkv, selected_blocks)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_blocks)
            )
            safe_block_ids = torch.where(block_valid, block_ids, torch.zeros_like(block_ids)).to(torch.int32)
            key_blocks = key_cache.reshape(batch_local, rows, -1, cfg.index_block_size, head_dim)
            value_blocks = value_cache.reshape(batch_local, rows, -1, cfg.index_block_size, head_dim)
            selected_k = ctx_gather_block_range_kv_dp(key_blocks, safe_block_ids).reshape(
                batch_local, rows, selected_len, head_dim
            )
            selected_v = ctx_gather_block_range_kv_dp(value_blocks, safe_block_ids).reshape(
                batch_local, rows, selected_len, head_dim
            )

            offsets = torch.arange(cfg.index_block_size, device=query_states.device).view(1, 1, 1, cfg.index_block_size)
            selected_token_ids = block_ids.unsqueeze(-1) * cfg.index_block_size + offsets
            position_ids_dp = position_ids.reshape(dp, batch_local, query_len).permute(1, 0, 2)
            position_rows = (
                position_ids_dp.unsqueeze(2)
                .expand(batch_local, dp, hkv, query_len)
                .reshape(batch_local, rows, query_len)
            )
            valid_idx = block_valid.unsqueeze(-1) & (selected_token_ids < key_cache.shape[2])
            valid_idx = valid_idx & (selected_token_ids <= position_rows[:, :, None, :])
            valid_idx = valid_idx.reshape(batch_local, rows, selected_len)
        elif token_indices.ndim == 4:
            # External DP-major layout: [batch, Hkv, blocks, block_size].
            selected_len = token_indices.shape[-2] * token_indices.shape[-1]
            gather_idx = (
                token_indices.reshape(dp, batch_local, hkv, selected_len)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_len)
            )
            valid_idx = (
                token_valid.reshape(dp, batch_local, hkv, selected_len)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_len)
            )
        elif token_indices.ndim == 3:
            selected_len = token_indices.shape[-1]
            if tuple(token_indices.shape[:2]) == (batch_local, rows):
                gather_idx = token_indices
                valid_idx = token_valid
            elif tuple(token_indices.shape[:2]) == (batch, hkv):
                gather_idx = (
                    token_indices.reshape(dp, batch_local, hkv, selected_len)
                    .permute(1, 0, 2, 3)
                    .reshape(batch_local, rows, selected_len)
                )
                valid_idx = (
                    token_valid.reshape(dp, batch_local, hkv, selected_len)
                    .permute(1, 0, 2, 3)
                    .reshape(batch_local, rows, selected_len)
                )
            else:
                raise ValueError(
                    "MSA token indices must use [B_local, DP*Hkv, selected_len] or [batch, Hkv, selected_len]."
                )
        else:
            raise ValueError("MSA token indices must be rank 3 or rank 4.")

        if not uses_block_indices:
            if tuple(valid_idx.shape) != tuple(gather_idx.shape):
                raise ValueError("MSA token_indices and token_valid shapes must match.")
            selected_k = ctx_gather_blocked_kv_dp(key_cache, gather_idx.to(torch.int32))
            selected_v = ctx_gather_blocked_kv_dp(value_cache, gather_idx.to(torch.int32))

        num_cores = (
            blocking_config.num_cores_per_device
            if blocking_config is not None and blocking_config.num_cores_per_device
            else 1
        )
        if num_kv_groups % num_cores:
            raise ValueError(
                f"MSA Q-head groups ({num_kv_groups}) must be divisible by num_cores_per_device ({num_cores})."
            )
        groups_per_core = num_kv_groups // num_cores

        q = (
            query_states.reshape(dp, batch_local, hkv, num_kv_groups, query_len, head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch_local, rows, num_kv_groups, query_len, head_dim)
        )
        q = q.reshape(batch_local, rows, num_cores, groups_per_core * query_len, head_dim)
        selected_k = selected_k.unsqueeze(2).expand(-1, -1, num_cores, -1, -1)
        selected_v = selected_v.unsqueeze(2).expand(-1, -1, num_cores, -1, -1)
        valid = valid_idx.unsqueeze(2).expand(-1, -1, num_cores, -1)

        scores = torch.matmul(q.float(), selected_k.float().transpose(-1, -2)) * (head_dim**-0.5)
        scores = scores.masked_fill(~valid.unsqueeze(3), -1.0e30)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(selected_v.dtype)
        out = torch.matmul(probs, selected_v)
        out = (
            out.reshape(batch_local, rows, num_kv_groups, query_len, head_dim)
            .reshape(batch_local, dp, hkv, num_kv_groups, query_len, head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch, hkv * num_kv_groups, query_len, head_dim)
        )
        return out, key_cache, value_cache

    def _baseline_attention_gp_cp(
        self,
        query_states,
        token_indices,
        token_valid,
        key_cache,
        value_cache,
        dp,
        cp,
        blocking_config,
        position_ids=None,
    ):
        """Run retained sparse attention with contiguous context partitioning."""
        batch, _, query_len, head_dim = query_states.shape
        cfg = self.config
        hkv = self.config.num_key_value_heads
        num_kv_groups = self.num_key_value_groups
        if query_len != 1 or batch % dp:
            raise ValueError("Retained MSA attention CP is decode-only and requires DP-divisible batch.")
        batch_local = batch // dp
        rows = dp * hkv * cp
        local_ctx_len = key_cache.shape[2]
        if tuple(key_cache.shape[:2]) != (batch_local, rows) or tuple(value_cache.shape) != tuple(key_cache.shape):
            raise ValueError(f"Retained CP key/value cache shape must start with {(batch_local, rows)}.")
        ctx_len = blocking_config.ctx_len if blocking_config and blocking_config.ctx_len else local_ctx_len * cp

        if (
            token_indices.ndim == 3
            and tuple(token_indices.shape[:2]) == (batch, hkv)
            and token_indices.shape[-1] <= cfg.index_topk_blocks
        ):
            if position_ids is None:
                raise ValueError("MSA block-index CP attention requires position_ids.")
            if tuple(token_valid.shape) != tuple(token_indices.shape):
                raise ValueError("MSA block_indices and block_valid shapes must match.")
            if ctx_len % (cp * cfg.index_block_size):
                raise ValueError(
                    "MSA block-index CP gather requires ctx_len / msa_attn_cp divisible by index_block_size."
                )

            selected_blocks = token_indices.shape[-1]
            selected_len = selected_blocks * cfg.index_block_size
            blocks_per_cp = torch._shape_as_tensor(key_cache)[2].to(token_indices.dtype) // cfg.index_block_size
            block_ids_global = token_indices.reshape(dp, batch_local, hkv, selected_blocks).permute(1, 0, 2, 3)
            valid_global = token_valid.reshape(dp, batch_local, hkv, selected_blocks).permute(1, 0, 2, 3)
            block_cp = block_ids_global // blocks_per_cp
            block_cp_valid = (block_cp >= 0) & (block_cp < cp)
            gather_block_local = block_ids_global - block_cp * blocks_per_cp
            cp_idx = torch.arange(cp, device=query_states.device).view(1, 1, 1, cp, 1)
            valid_cp_blocks = (
                valid_global.unsqueeze(3) & block_cp_valid.unsqueeze(3) & (block_cp.unsqueeze(3) == cp_idx)
            )
            gather_block_ids = torch.where(
                valid_cp_blocks,
                gather_block_local.unsqueeze(3),
                torch.zeros_like(gather_block_local).unsqueeze(3),
            ).reshape(batch_local, rows, selected_blocks)

            key_blocks = key_cache.reshape(batch_local, rows, -1, cfg.index_block_size, head_dim)
            value_blocks = value_cache.reshape(batch_local, rows, -1, cfg.index_block_size, head_dim)
            selected_k = ctx_gather_block_range_kv_dp(key_blocks, gather_block_ids.to(torch.int32)).reshape(
                batch_local, rows, selected_len, head_dim
            )
            selected_v = ctx_gather_block_range_kv_dp(value_blocks, gather_block_ids.to(torch.int32)).reshape(
                batch_local, rows, selected_len, head_dim
            )

            offsets = torch.arange(cfg.index_block_size, device=query_states.device).view(
                1, 1, 1, 1, cfg.index_block_size
            )
            selected_token_ids = block_ids_global.unsqueeze(-1) * cfg.index_block_size + offsets
            position_ids_dp = position_ids.reshape(dp, batch_local, query_len).permute(1, 0, 2)
            valid_tokens = valid_global.unsqueeze(-1)
            valid_tokens = valid_tokens & (selected_token_ids < ctx_len)
            valid_tokens = valid_tokens & (selected_token_ids <= position_ids_dp[:, :, None, :, None])
            valid_idx = (valid_cp_blocks.unsqueeze(-1) & valid_tokens.unsqueeze(3)).reshape(
                batch_local, rows, selected_len
            )
        else:
            if token_indices.ndim == 4:
                selected_len = token_indices.shape[-2] * token_indices.shape[-1]
                global_idx, global_valid = token_indices.flatten(-2), token_valid.flatten(-2)
            elif token_indices.ndim == 3 and tuple(token_indices.shape[:2]) == (batch, hkv):
                selected_len = token_indices.shape[-1]
                global_idx, global_valid = token_indices, token_valid
            else:
                raise ValueError("Retained CP attention requires token indices shaped [batch, Hkv, ...].")
            global_idx = global_idx.reshape(dp, batch_local, hkv, selected_len).permute(1, 0, 2, 3)
            global_valid = global_valid.reshape(dp, batch_local, hkv, selected_len).permute(1, 0, 2, 3)
            token_cp = global_idx // local_ctx_len
            token_cp_valid = (token_cp >= 0) & (token_cp < cp)
            gather_idx_local = global_idx - token_cp * local_ctx_len
            cp_idx = torch.arange(cp, device=query_states.device).view(1, 1, 1, cp, 1)
            valid_cp = global_valid.unsqueeze(3) & token_cp_valid.unsqueeze(3) & (token_cp.unsqueeze(3) == cp_idx)
            gather_idx = (
                gather_idx_local.unsqueeze(3)
                .expand(batch_local, dp, hkv, cp, selected_len)
                .reshape(batch_local, rows, selected_len)
            )
            valid_idx = valid_cp.expand(batch_local, dp, hkv, cp, selected_len).reshape(batch_local, rows, selected_len)
            gather_idx = torch.where(valid_idx, gather_idx, torch.zeros_like(gather_idx)).to(torch.int32)
            selected_k = ctx_gather_blocked_kv_dp(key_cache, gather_idx)
            selected_v = ctx_gather_blocked_kv_dp(value_cache, gather_idx)
        num_cores = (
            blocking_config.num_cores_per_device if blocking_config and blocking_config.num_cores_per_device else 1
        )
        if num_kv_groups % num_cores:
            raise ValueError("MSA Q-head groups must be divisible by num_cores_per_device.")
        groups_per_core = num_kv_groups // num_cores
        selected_k = selected_k.unsqueeze(2).expand(-1, -1, num_cores, -1, -1)
        selected_v = selected_v.unsqueeze(2).expand(-1, -1, num_cores, -1, -1)
        q = query_states.reshape(dp, batch_local, hkv, num_kv_groups, query_len, head_dim).permute(1, 0, 2, 3, 4, 5)
        q = (
            q.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, num_kv_groups, query_len, head_dim)
            .reshape(batch_local, rows, num_kv_groups, query_len, head_dim)
        )
        q = q.reshape(batch_local, rows, num_cores, groups_per_core * query_len, head_dim)
        valid = valid_idx.unsqueeze(2).expand(-1, -1, num_cores, -1)
        scores = torch.matmul(q.float(), selected_k.float().transpose(-1, -2)) * (head_dim**-0.5)
        scores = scores.masked_fill(~valid.unsqueeze(3), -1.0e30)
        local_max = scores.max(dim=-1).values
        exp_scores = torch.exp(scores - local_max.unsqueeze(-1))
        exp_scores = torch.where(valid.unsqueeze(3), exp_scores, torch.zeros_like(exp_scores))
        local_sum = exp_scores.sum(dim=-1)
        local_out = torch.matmul(exp_scores.to(selected_v.dtype), selected_v)
        local_max = local_max.reshape(batch_local, dp, hkv, cp, num_cores, groups_per_core * query_len)
        local_sum = local_sum.reshape(batch_local, dp, hkv, cp, num_cores, groups_per_core * query_len)
        local_out = local_out.reshape(batch_local, dp, hkv, cp, num_cores, groups_per_core * query_len, head_dim)
        global_max = local_max.max(dim=3).values
        cp_weight = torch.exp(local_max - global_max.unsqueeze(3))
        global_sum = (cp_weight * local_sum).sum(dim=3)
        global_out = (cp_weight.unsqueeze(-1) * local_out).sum(dim=3)
        safe_sum = torch.where(global_sum == 0, torch.ones_like(global_sum), global_sum)
        out = global_out / safe_sum.unsqueeze(-1)
        out = torch.where(global_sum.unsqueeze(-1) == 0, torch.zeros_like(out), out).to(query_states.dtype)
        out = (
            out.reshape(batch_local, dp, hkv, num_kv_groups, query_len, head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch, hkv * num_kv_groups, query_len, head_dim)
        )
        return out, key_cache, value_cache

    def _paged_attention(
        self,
        query_states,
        token_indices,
        token_valid,
        key_cache,
        value_cache,
        position_ids,
        block_table,
        blocking_config,
    ):
        """Run selected-token attention against physical paged K/V pools."""
        batch, _, query_len, head_dim = query_states.shape
        cfg = self.config
        hkv = self.config.num_key_value_heads
        n_rep = self.num_key_value_groups
        dp = blocking_config.msa_attn_dp if blocking_config and blocking_config.msa_attn_dp else 1
        cp = blocking_config.msa_attn_cp if blocking_config and blocking_config.msa_attn_cp else 1
        rows = dp * hkv * cp
        ql_eff = n_rep * query_len
        if query_len != 1:
            raise ValueError("MSA attention GP is decode-only; QL must be 1.")
        if dp < 1 or batch % dp:
            raise ValueError("MSA batch size must be divisible by msa_attn_dp.")

        batch_local = batch // dp
        page_block_size = key_cache.shape[2]

        compressed_block_shape = (batch, hkv, cfg.index_topk_blocks)
        has_global_indices = False
        if token_indices.ndim == 3 and tuple(token_indices.shape) == compressed_block_shape:
            has_global_indices = True
            if tuple(token_valid.shape) != compressed_block_shape:
                raise ValueError(f"MSA block-valid shape {tuple(token_valid.shape)} != {compressed_block_shape}.")
            offsets = torch.arange(cfg.index_block_size, device=query_states.device).view(1, 1, 1, cfg.index_block_size)
            token_indices_expanded = token_indices.unsqueeze(-1) * cfg.index_block_size + offsets
            token_valid_expanded = token_valid.unsqueeze(-1)
            token_valid_expanded = token_valid_expanded & (
                token_indices_expanded
                < (
                    blocking_config.ctx_len
                    if blocking_config and blocking_config.ctx_len
                    else key_cache.shape[0] * page_block_size
                )
            )
            token_valid_expanded = token_valid_expanded & (token_indices_expanded <= position_ids[:, None, None, :])
            token_indices_expanded = torch.where(
                token_valid_expanded,
                token_indices_expanded,
                torch.zeros_like(token_indices_expanded),
            )
            # [B, Hkv, selected_blocks, index_block_size]
            selected_len = cfg.index_topk_blocks * cfg.index_block_size
            gather_idx_global = (
                token_indices_expanded.view(dp, batch_local, hkv, selected_len).permute(1, 0, 2, 3).to(torch.int32)
            )
            valid_idx_global = token_valid_expanded.view(dp, batch_local, hkv, selected_len).permute(1, 0, 2, 3)
        elif token_indices.ndim == 3:
            expected_index_shape = (batch_local, rows, token_indices.shape[-1])
            if tuple(token_indices.shape) != expected_index_shape:
                raise ValueError(
                    f"MSA token_indices shape {tuple(token_indices.shape)} is neither "
                    f"compressed {compressed_block_shape} nor expanded "
                    f"{expected_index_shape}."
                )
            if tuple(token_valid.shape) != expected_index_shape:
                raise ValueError(f"MSA token_valid shape {tuple(token_valid.shape)} != {expected_index_shape}.")
            selected_len = token_indices.shape[-1]
            gather_idx = token_indices.to(torch.int32)
            valid_idx = token_valid
        elif token_indices.ndim == 4:
            has_global_indices = True
            selected_len = token_indices.flatten(-2).shape[-1]
            gather_idx_global = (
                token_indices.flatten(-2).view(dp, batch_local, hkv, selected_len).permute(1, 0, 2, 3).to(torch.int32)
            )
            valid_idx_global = token_valid.flatten(-2).view(dp, batch_local, hkv, selected_len).permute(1, 0, 2, 3)
        else:
            raise ValueError(
                "MSA token_indices/token_valid must use either "
                "[B, Hkv, selected_blocks], "
                "[B_local, DP*CP*Hkv, selected_len] or "
                "[B, Hkv, selected_blocks, index_block_size]."
            )
        if selected_len % cfg.index_block_size:
            raise ValueError(
                f"MSA selected token count ({selected_len}) must be divisible "
                f"by index_block_size ({cfg.index_block_size})."
            )
        selected_blocks = selected_len // cfg.index_block_size
        if selected_blocks != cfg.index_topk_blocks:
            raise ValueError(
                f"MSA selected block count ({selected_blocks}) != configured selected_blocks ({cfg.index_topk_blocks})."
            )
        if blocking_config and blocking_config.ctx_len and blocking_config.ctx_len % cfg.index_block_size:
            raise ValueError("MSA GP block gather requires ctx_len divisible by index_block_size.")
        num_cores = blocking_config.num_cores_per_device if blocking_config else 1
        if n_rep % num_cores:
            raise ValueError(
                f"Paged MSA Q-head core parallelism requires num_kv_groups "
                f"({n_rep}) divisible by compile core count ({num_cores})."
            )
        q_heads_per_core = n_rep // num_cores
        if block_table is None:
            raise ValueError("Paged MSA attention requires msa_attn_block_table.")

        if cfg.index_block_size != page_block_size:
            raise ValueError("Paged MSA attention requires index_block_size == page_block_size.")
        if block_table.ndim != 3 or tuple(block_table.shape[:2]) != (
            dp,
            batch_local,
        ):
            raise ValueError(
                f"msa_attn_block_table must have shape [msa_attn_dp, B_local, pages], got {tuple(block_table.shape)}."
            )
        ctx_len = (
            blocking_config.ctx_len
            if blocking_config and blocking_config.ctx_len
            else block_table.shape[-1] * cp * page_block_size
        )
        configured_num_pages = (
            getattr(blocking_config, "msa_attn_num_logical_pages", None) if blocking_config is not None else None
        )
        if configured_num_pages is None and blocking_config is not None:
            configured_num_pages = getattr(blocking_config, "num_logical_pages", None)
        num_pages = _resolve_num_logical_pages(ctx_len, page_block_size, configured_num_pages)
        num_page_groups = _num_page_groups(num_pages, cp)
        if block_table.shape[-1] < num_page_groups:
            raise ValueError("msa_attn_block_table is too short for paged attention.")
        if has_global_indices:
            # logical_pages_global = gather_idx_global // page_block_size
            logical_pages_global = torch.div(gather_idx_global, page_block_size, rounding_mode="floor")
            owner_cp = logical_pages_global % cp
            cp_idx = torch.arange(cp, device=query_states.device).view(1, 1, cp, 1, 1)
            valid_idx = valid_idx_global.unsqueeze(2) & (owner_cp.unsqueeze(2) == cp_idx)
            gather_idx = gather_idx_global.unsqueeze(2).expand(-1, -1, cp, -1, -1)
            gather_idx = torch.where(valid_idx, gather_idx, torch.zeros_like(gather_idx))
            gather_idx = gather_idx.reshape(batch_local, rows, selected_len)
            valid_idx = valid_idx.reshape(batch_local, rows, selected_len)
        expected_cache_tail = (rows, page_block_size, head_dim)
        if key_cache.ndim != 4 or tuple(key_cache.shape[1:]) != expected_cache_tail:
            raise ValueError(
                f"MSA key_cache shape {tuple(key_cache.shape)} must be "
                f"[physical_pages, {rows}, {page_block_size}, {head_dim}]."
            )
        if tuple(value_cache.shape) != tuple(key_cache.shape):
            raise ValueError(
                f"MSA value_cache shape {tuple(value_cache.shape)} must match key_cache shape {tuple(key_cache.shape)}."
            )
        q_rows = (
            query_states.reshape(dp, batch_local, hkv, n_rep, query_len, head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, n_rep, query_len, head_dim)
            .reshape(batch_local, rows, ql_eff, head_dim)
        )
        local_batch_size = 1
        max_groups: list[torch.Tensor] = []
        sum_groups: list[torch.Tensor] = []
        out_groups: list[torch.Tensor] = []
        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            gather_blocks = gather_idx[batch_start:batch_end].view(local, rows, selected_blocks, cfg.index_block_size)
            valid_blocks = valid_idx[batch_start:batch_end].view(local, rows, selected_blocks, cfg.index_block_size)
            q_local = q_rows[batch_start:batch_end]
            # logical_pages = (gather_blocks[0, ..., 0] // page_block_size).long()
            logical_pages = torch.div(gather_blocks[0, ..., 0], page_block_size, rounding_mode="floor").long()
            row_dp = torch.arange(rows, device=query_states.device) // (cp * hkv)
            table_rows = block_table[:, batch_start].index_select(0, row_dp)
            # physical_block_ids = torch.gather(table_rows, 1, logical_pages // cp)
            physical_block_ids = torch.gather(table_rows, 1, torch.div(logical_pages, cp, rounding_mode="floor"))
            physical_block_ids = torch.where(
                valid_blocks[0].any(dim=-1),
                physical_block_ids,
                torch.zeros_like(physical_block_ids),
            )
            # ctx_gather_paged_kv_dp expects [selected_pages, rows].
            block_ids = physical_block_ids.transpose(0, 1).contiguous()
            selected_k = ctx_gather_paged_kv_dp(key_cache, block_ids.to(torch.int32))
            # Each core owns one Q head and receives a physical copy of the
            # complete selected KV sequence.
            # selected_k: [local, DP*CP*Hkv, C, selected_len, D]
            selected_k = (
                selected_k.view(local, rows, selected_len, cfg.head_dim).unsqueeze(2).repeat(1, 1, num_cores, 1, 1)
            )
            valid_core = valid_blocks.view(local, rows, selected_len).unsqueeze(2).repeat(1, 1, num_cores, 1)
            # q_core: [local, DP*CP*Hkv, C, Q_heads_per_core*QL, D]
            q_core = q_local.view(
                local,
                rows,
                num_cores,
                q_heads_per_core * query_len,
                cfg.head_dim,
            )
            attn = torch.matmul(q_core.float(), selected_k.transpose(-1, -2).float()) * (cfg.head_dim**-0.5)
            attn = attn.masked_fill(~valid_core.unsqueeze(3), MASKED_ATTENTION_LOGIT)
            max_core = attn.max(dim=-1).values
            exp_core = torch.exp(attn - max_core.unsqueeze(-1))
            exp_core = torch.where(valid_core.unsqueeze(3), exp_core, torch.zeros_like(exp_core))
            selected_v = (
                ctx_gather_paged_kv_dp(value_cache, block_ids.to(torch.int32))
                .view(
                    local,
                    rows,
                    selected_len,
                    cfg.head_dim,
                )
                .unsqueeze(2)
                .repeat(1, 1, num_cores, 1, 1)
            )
            selected_v = torch.where(valid_core.unsqueeze(-1), selected_v, torch.zeros_like(selected_v))
            sum_core = exp_core.sum(dim=-1)
            max_groups.append(max_core)
            sum_groups.append(sum_core)
            out_groups.append(torch.matmul(exp_core, selected_v.float()))

        max_all = torch.cat(max_groups, dim=0).view(batch_local, dp, cp, hkv, num_cores, -1)
        sum_all = torch.cat(sum_groups, dim=0).view(batch_local, dp, cp, hkv, num_cores, -1)
        out_all = torch.cat(out_groups, dim=0).view(batch_local, dp, cp, hkv, num_cores, -1, head_dim)
        global_max = max_all.max(dim=2).values
        cp_weight = torch.exp(max_all - global_max.unsqueeze(2))
        global_sum = (cp_weight * sum_all).sum(dim=2)
        global_out = (cp_weight.unsqueeze(-1) * out_all).sum(dim=2)
        safe_sum = torch.where(global_sum > 0, global_sum, torch.ones_like(global_sum))
        out_all = torch.where(
            global_sum.unsqueeze(-1) > 0,
            global_out / safe_sum.unsqueeze(-1),
            torch.zeros_like(global_out),
        )
        out = (
            out_all.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch_local * dp, hkv * n_rep, query_len, cfg.head_dim)
        )
        return out.to(dtype=query_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_states = hidden_states.to(dtype=self.q_proj.weight.dtype)
        query_shape = (*input_shape, self.config.num_attention_heads, self.head_dim)
        key_value_shape = (*input_shape, self.config.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_value_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = qeff_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            int(self.head_dim * self.config.rope_parameters.get("partial_rotary_factor", 1.0)),
        )

        cache_kwargs = {"position_ids": position_ids, "batch_index": kwargs.get("batch_index")}
        if attention_mask is not None:
            cache_kwargs["CCL"] = attention_mask.shape[-1]

        if self.indexer is not None and isinstance(past_key_values, QEffMiniMaxSparseCache):
            blocking_config = getattr(self, "attn_blocking_config", None)
            if blocking_config:
                cache_kwargs["dp"] = getattr(blocking_config, "msa_attn_dp", 1)
                cache_kwargs["hkv"] = self.config.num_key_value_heads
                cache_kwargs["cp"] = getattr(blocking_config, "msa_attn_cp", 1) or 1

            indexer_block_table = kwargs.get("msa_indexer_block_table")
            attention_block_table = kwargs.get("msa_attn_block_table")
            paged = attention_block_table is not None
            is_prefill = input_shape[1] > 1
            indexer_position_embeddings = kwargs.get("indexer_position_embeddings")
            indexer_dp = getattr(blocking_config, "msa_indexer_dp", 1) or 1
            if (is_prefill or paged or indexer_dp > 1) and indexer_position_embeddings is not None:
                indexer_cos, indexer_sin = indexer_position_embeddings
            else:
                indexer_cos, indexer_sin = cos, sin
            if is_prefill:
                selector = (
                    self.indexer._select_blocks_prefill_par
                    if getattr(blocking_config, "indexer_prefill_parallel", False)
                    else self.indexer._select_blocks_prefill
                )
                token_indices, token_valid = selector(
                    hidden_states,
                    position_ids,
                    past_key_values,
                    self.layer_idx,
                    indexer_cos,
                    indexer_sin,
                    blocking_config,
                    paged_block_table=indexer_block_table[0] if paged else None,
                )
            else:
                token_indices, token_valid = self.indexer._select_blocks(
                    hidden_states,
                    position_ids,
                    past_key_values,
                    self.layer_idx,
                    indexer_cos,
                    indexer_sin,
                    blocking_config=blocking_config,
                    block_table=indexer_block_table,
                )
            attn_dp = blocking_config.msa_attn_dp if (blocking_config and blocking_config.msa_attn_dp) else 1
            attn_cp = blocking_config.msa_attn_cp if (blocking_config and blocking_config.msa_attn_cp) else 1
            if paged and is_prefill:
                layer = past_key_values.layers[self.layer_idx]
                key_cache = layer.keys
                value_cache = layer.values
                attn_output, key_cache, value_cache = self._msa_attention_prefill(
                    query_states,
                    hidden_states,
                    cos,
                    sin,
                    key_cache,
                    value_cache,
                    token_indices,
                    token_valid,
                    position_ids,
                    blocking_config,
                    paged_block_table=attention_block_table[0],
                )
                layer.keys = key_cache
                layer.values = value_cache
                attn_output = attn_output.transpose(1, 2).reshape(
                    *input_shape, self.config.num_attention_heads * self.head_dim
                )
                return self.o_proj(attn_output.to(dtype=self.o_proj.weight.dtype).contiguous()), None
            if paged:
                layer = past_key_values.layers[self.layer_idx]
                key_cache = layer.keys
                value_cache = layer.values
                if (
                    key_cache is None
                    or key_cache.ndim != 4
                    or key_cache.shape[1] != attn_dp * attn_cp * self.config.num_key_value_heads
                ):
                    raise ValueError(
                        "Paged MiniMax attention requires physical [pages, DP*CP*Hkv, page_size, head_dim] caches."
                    )
                batch_local = input_shape[0] // attn_dp
                position_ids_dp = position_ids.view(attn_dp, batch_local, -1).permute(1, 0, 2)
                table = attention_block_table.permute(1, 0, 2)
                # logical_page = position_ids_dp // key_cache.shape[2]
                logical_page = torch.div(position_ids_dp, key_cache.shape[2], rounding_mode="floor")
                # logical_page_group = logical_page // attn_cp
                logical_page_group = torch.div(logical_page, attn_cp, rounding_mode="floor")
                owner_cp = logical_page % attn_cp
                cp_idx = torch.arange(attn_cp, device=key_cache.device).view(1, 1, attn_cp, 1, 1)
                row_live = owner_cp.unsqueeze(2).unsqueeze(3) == cp_idx
                row_live = row_live.expand(
                    batch_local, attn_dp, attn_cp, self.config.num_key_value_heads, key_states.shape[2]
                )
                physical_page = torch.gather(table, 2, logical_page_group).unsqueeze(2).unsqueeze(3)
                physical_page = physical_page.expand_as(row_live)
                invalid_page = torch.full_like(physical_page, torch.iinfo(torch.int32).max)
                block_id = torch.where(row_live, physical_page, invalid_page)
                addr = (position_ids_dp % key_cache.shape[2]).to(torch.int32).unsqueeze(2).unsqueeze(3)
                addr = addr.expand(batch_local, attn_dp, attn_cp, self.config.num_key_value_heads, key_states.shape[2])
                updates_k = (
                    key_states.view(attn_dp, batch_local, self.config.num_key_value_heads, -1, self.head_dim)
                    .permute(1, 0, 2, 3, 4)
                    .unsqueeze(2)
                    .expand(batch_local, attn_dp, attn_cp, self.config.num_key_value_heads, -1, self.head_dim)
                    .reshape(batch_local, attn_dp * attn_cp * self.config.num_key_value_heads, -1, self.head_dim)
                )
                updates_v = (
                    value_states.view(attn_dp, batch_local, self.config.num_key_value_heads, -1, self.head_dim)
                    .permute(1, 0, 2, 3, 4)
                    .unsqueeze(2)
                    .expand(batch_local, attn_dp, attn_cp, self.config.num_key_value_heads, -1, self.head_dim)
                    .reshape(batch_local, attn_dp * attn_cp * self.config.num_key_value_heads, -1, self.head_dim)
                )
                layer.keys = ctx_paged_scatter_dp(
                    key_cache,
                    block_id.reshape(batch_local, -1, key_states.shape[2]).to(torch.int32),
                    addr.reshape(batch_local, -1, key_states.shape[2]),
                    updates_k,
                )
                layer.values = ctx_paged_scatter_dp(
                    value_cache,
                    block_id.reshape(batch_local, -1, key_states.shape[2]).to(torch.int32),
                    addr.reshape(batch_local, -1, key_states.shape[2]),
                    updates_v,
                )
                attn_output = self._paged_attention(
                    query_states,
                    token_indices,
                    token_valid,
                    layer.keys,
                    layer.values,
                    position_ids,
                    attention_block_table,
                    blocking_config,
                )
                attn_output = attn_output.transpose(1, 2).reshape(
                    *input_shape, self.config.num_attention_heads * self.head_dim
                )
                return self.o_proj(attn_output.contiguous()), None
            if input_shape[1] > 1:
                key_cache = past_key_values.layers[self.layer_idx].keys
                value_cache = past_key_values.layers[self.layer_idx].values
                attn_output, key_cache, value_cache = self._msa_attention_prefill(
                    query_states,
                    hidden_states,
                    cos,
                    sin,
                    key_cache,
                    value_cache,
                    token_indices,
                    token_valid,
                    position_ids,
                    blocking_config,
                )
                attn_output = attn_output.transpose(1, 2).reshape(
                    *input_shape, self.config.num_attention_heads * self.head_dim
                )
                past_key_values.layers[self.layer_idx].keys = key_cache
                past_key_values.layers[self.layer_idx].values = value_cache
                return self.o_proj(attn_output.to(dtype=self.o_proj.weight.dtype).contiguous()), None
            # Sparse decode gathers from the retained cache, so write the current token before either path.
            past_key_values.write_only_sparse(key_states, value_states, self.layer_idx, cache_kwargs)
            if attn_dp > 1 or attn_cp > 1:
                dp = cache_kwargs["dp"]
                batch = input_shape[0]
                batch_local = batch // dp
                hkv = self.config.num_key_value_heads
                rows = dp * attn_cp * hkv
                key_cache_storage = past_key_values.layers[self.layer_idx].keys
                value_cache_storage = past_key_values.layers[self.layer_idx].values
                key_cache = key_cache_storage.reshape(batch_local, rows, -1, self.head_dim)
                value_cache = value_cache_storage.reshape(batch_local, rows, -1, self.head_dim)
                attn_output, key_cache, value_cache = self._baseline_attention_gp(
                    query_states,
                    token_indices,
                    token_valid,
                    key_cache,
                    value_cache,
                    dp=attn_dp,
                    blocking_config=blocking_config,
                    position_ids=position_ids,
                )
                past_key_values.layers[self.layer_idx].keys = key_cache.reshape(key_cache_storage.shape)
                past_key_values.layers[self.layer_idx].values = value_cache.reshape(value_cache_storage.shape)
            else:
                selected_k, selected_v, flat_valid = past_key_values.read_kv_with_block_indices(
                    self.layer_idx, token_indices, token_valid
                )
                attn_output = self._baseline_attention(query_states, selected_k, selected_v, flat_valid)
            attn_output = attn_output.reshape(*input_shape, self.config.num_attention_heads * self.head_dim)
        else:
            blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
            use_blocking = blocking_config is not None and blocking_config.mode != BlockingMode.NONE
            if use_blocking:
                past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
                attn_output, _ = generic_blocked_attention_interface(
                    module=self,
                    query=query_states,
                    key=key_states,
                    value=value_states,
                    attention_mask=attention_mask,
                    scaling=self.scaling,
                    layer_idx=self.layer_idx,
                    past_key_value=past_key_values,
                    blocking_config=blocking_config,
                    comp_ctx_lengths=kwargs.get("comp_ctx_lengths"),
                    batch_index=kwargs.get("batch_index"),
                    position_ids=position_ids,
                    past_seen_tokens=past_seen_tokens,
                    prefill_only=blocking_config.mode.is_prefill,
                )
            else:
                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
                attn_output, _ = qeff_eager_attention_forward(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                )
            attn_output = attn_output.reshape(*input_shape, self.config.num_attention_heads * self.head_dim)

        return self.o_proj(attn_output.contiguous()), None


def _qeff_minimax_clamp(hidden_states: torch.Tensor, min_value=None, max_value=None) -> torch.Tensor:
    if min_value is not None:
        min_tensor = torch.full_like(hidden_states, min_value, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = torch.maximum(hidden_states, min_tensor)
    if max_value is not None:
        max_tensor = torch.full_like(hidden_states, max_value, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = torch.minimum(hidden_states, max_tensor)
    return hidden_states


class QEffMiniMaxM3VLDenseMLP(MiniMaxM3VLDenseMLP):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = _qeff_minimax_clamp(gate, max_value=self.swiglu_limit)
        up = _qeff_minimax_clamp(up, min_value=-self.swiglu_limit, max_value=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return self.down_proj((up + 1.0) * glu)


class QEffMiniMaxM3VLTopKRouter(MiniMaxM3VLTopKRouter):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = nn.functional.linear(hidden_states.to(self.weight.dtype), self.weight)
        routing_weights = nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias.to(device=routing_weights.device)
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        denom = torch.einsum("tk->t", top_k_weights)
        top_k_weights = top_k_weights / denom[:, None]
        return router_logits, top_k_weights, top_k_index


class QEffMiniMaxM3VLSparseMoeBlock(QEffMoEBlockMixin, MiniMaxM3VLSparseMoeBlock):
    supported_moe_flavours = (
        MoEFlavour.SIMPLE_LOOP,
        MoEFlavour.DECODE_BMM,
        MoEFlavour.EXPERT_PARALLEL,
    )

    def __qeff_init__(self) -> None:
        super().__qeff_init__()
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
        delete_module_attrs(self.experts, "gate_up_proj", "down_proj")
        self.weights_transformed = True
        return self.moe_weights

    @property
    def moe_profile(self) -> MoEProfile:
        return MoEProfile(
            expert_mlp=partial(
                minimax_m3_clamped_glu_mlp,
                limit=self.experts.swiglu_limit,
                alpha=self.experts.swiglu_alpha,
            )
        )

    def route(self, x: torch.Tensor):
        router_logits, top_k_weights, top_k_index = self.gate(x)
        return (top_k_index, top_k_weights.to(x.dtype)), router_logits

    def apply_shared_experts(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return out * self.routed_scaling_factor + self.shared_experts(residual)


class QEffMiniMaxM3VLDecoderLayer(MiniMaxM3VLDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffMiniMaxM3VLTextModel(MiniMaxM3VLTextModel):
    def __qeff_init__(self):
        # Hoist pre-scaled RoPE tables onto the text model as parameters.  This
        # exports them as graph initializers, matching the established QEff
        # Llama/Qwen path, instead of Constant nodes inside rotary_emb.
        rotary_device = self.embed_tokens.weight.device
        if rotary_device.type == "meta":
            rotary_device = torch.device("cpu")
        # TODO: this is not teh correct solution  for handling meta device, need to figure out the original tensor which needs to be moved
        rotary_emb = QEffMiniMaxM3VLRotaryEmbedding(
            config=self.config,
            device=rotary_device,
        )
        self.cos_cached = nn.Parameter(rotary_emb.cos_cached.contiguous(), requires_grad=False)
        self.sin_cached = nn.Parameter(rotary_emb.sin_cached.contiguous(), requires_grad=False)
        # The exported forward gathers from the model-level tables directly.
        # Avoid retaining a duplicate full-context cache under rotary_emb.
        self.rotary_emb = nn.Identity()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        index_keys=None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = QEffMiniMaxSparseCache.from_legacy_cache(past_key_values, index_keys)
        elif use_cache and past_key_values is None:
            past_key_values = QEffMiniMaxSparseCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        target_length = past_seen_tokens + inputs_embeds.shape[1]
        if isinstance(attention_mask, torch.Tensor):
            target_length = attention_mask.shape[-1]
        elif past_key_values is not None and getattr(past_key_values, "layers", None):
            first_layer = past_key_values.layers[0]
            cached_keys = getattr(first_layer, "keys", None)
            if cached_keys is not None:
                target_length = cached_keys.shape[-2]
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_length)

        hidden_states = inputs_embeds
        rope_indices = position_ids + 1
        cos = self.cos_cached[rope_indices].to(device=hidden_states.device, dtype=hidden_states.dtype)
        sin = self.sin_cached[rope_indices].to(device=hidden_states.device, dtype=hidden_states.dtype)
        position_embeddings = (cos, sin)
        indexer_position_embeddings = None
        indexer = None
        blocking_config = None
        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            candidate = getattr(attention, "indexer", None)
            if candidate is not None:
                indexer = candidate
                blocking_config = getattr(attention, "attn_blocking_config", None)
                break
        indexer_dim = getattr(getattr(indexer, "config", None), "index_head_dim", None)
        indexer_dp = getattr(blocking_config, "msa_indexer_dp", 1) or 1
        if indexer_dim is not None:
            cos, sin = position_embeddings
            rotary_dim = min(cos.shape[-1], indexer_dim)
            indexer_cos = cos[..., :rotary_dim].clone()
            indexer_sin = sin[..., :rotary_dim].clone()
            if inputs_embeds.shape[1] == 1 and indexer_dp > 1:
                batch_local = inputs_embeds.shape[0] // indexer_dp
                indexer_cos = indexer_cos.reshape(
                    indexer_dp, batch_local, indexer_cos.shape[1], indexer_cos.shape[-1]
                ).permute(1, 0, 2, 3)
                indexer_sin = indexer_sin.reshape(
                    indexer_dp, batch_local, indexer_sin.shape[1], indexer_sin.shape[-1]
                ).permute(1, 0, 2, 3)
            indexer_position_embeddings = (indexer_cos, indexer_sin)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                indexer_position_embeddings=indexer_position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        if use_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffMiniMaxM3VLForCausalLM(MiniMaxM3VLForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffMiniMaxM3VLDecoderLayer}

    def generate_npi_file(self, onnx_path: Union[str, Path], model_name: Optional[str] = None) -> str:
        del model_name
        return _generate_minimax_npi_file(onnx_path)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        index_keys=None,
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            index_keys=index_keys,
            **kwargs,
        )

        if position_ids is None:
            hidden_states = outputs.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        else:
            logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
        )


class QEffMiniMaxM3VLEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        layers = getattr(getattr(self.model.model, "vision_tower", None), "layers", None)
        if layers:
            return {layers[0].__class__}
        return set()

    def forward(self, pixel_values, image_grid_thw):
        image_outputs = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        image_embeds = image_outputs.pooler_output if hasattr(image_outputs, "pooler_output") else image_outputs
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        image_embeds = image_embeds.to(pixel_values.device, pixel_values.dtype)
        bs = image_grid_thw.shape[0]
        split_size = image_embeds.shape[0] // bs
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(1))
        # image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(-1))
        return image_embeds


class QEffMiniMaxM3VLDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffMiniMaxM3VLDecoderLayer}

    def get_onnx_transforms(self) -> tuple[type, ...]:
        return (MiniMaxM3StaticPrefillLoopTransform,)

    def get_onnx_transform_kwargs(self, *, inputs, prefill_only: bool) -> dict:
        if not prefill_only:
            return {}

        sequence_input = inputs.get("inputs_embeds")
        if sequence_input is None:
            sequence_input = inputs.get("input_ids")
        if sequence_input is None:
            raise ValueError("MiniMax MSA prefill export requires input_ids or inputs_embeds to determine loop bounds.")
        export_seq_len = int(sequence_input.shape[1])

        trip_counts = set()
        layer_types = getattr(self.language_model.config, "layer_types", None)
        for module in self.language_model.modules():
            if not isinstance(module, QEffMiniMaxM3VLAttention):
                continue
            layer_idx = getattr(module, "layer_idx", None)
            if (
                layer_types is not None
                and layer_idx is not None
                and 0 <= layer_idx < len(layer_types)
                and layer_types[layer_idx] != "minimax_m3_sparse"
            ):
                continue
            blocking_config = getattr(module, "attn_blocking_config", None)
            msa_q_chunk = getattr(blocking_config, "msa_q_chunk", None)
            if msa_q_chunk is None:
                continue
            msa_q_chunk = int(msa_q_chunk)
            loop_seq_len = int(getattr(blocking_config, "prefill_export_seq_len", None) or export_seq_len)
            if msa_q_chunk <= 0 or loop_seq_len % msa_q_chunk:
                raise ValueError(
                    "MiniMax MSA prefill export sequence length must be divisible by the MSA query chunk size."
                )
            trip_counts.add(loop_seq_len // msa_q_chunk)

        if not trip_counts:
            return {}
        if len(trip_counts) != 1:
            raise ValueError(f"MiniMax sparse attention layers produced inconsistent Loop trip counts: {trip_counts}.")
        return {"minimax_m3_prefill_loop_trip_count": trip_counts.pop()}

    def get_onnx_past_key_value_names(self, layer_idx: int, layer_state=None) -> List[str]:
        return [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]

    def get_onnx_index_key_names(self) -> List[str]:
        layer_types = getattr(self.config.text_config, "layer_types", None) or getattr(
            self.language_model.config, "layer_types", None
        )
        if layer_types is None:
            return []
        index_key_names = []
        for i in range(self.config.text_config.num_hidden_layers):
            if layer_types[i] == "minimax_m3_sparse":
                index_key_names.append(f"index_key.{i}")
        return index_key_names

    def forward(
        self,
        input_ids=None,
        vision_embeds=None,
        position_ids=None,
        image_idx=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[List[int]] = None,
        index_keys=None,
        msa_indexer_block_table: Optional[torch.Tensor] = None,
        msa_attn_block_table: Optional[torch.Tensor] = None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be provided.")

        qaic_config = getattr(self.model, "qaic_config", None) or {}
        if qaic_config.get("paged_kv", False) and msa_indexer_block_table is None:
            raise ValueError("paged_kv requires msa_indexer_block_table input.")
        if qaic_config.get("paged_kv", False) and msa_attn_block_table is None:
            raise ValueError("paged_kv requires msa_attn_block_table input.")

        if inputs_embeds is None:
            inputs_embeds = self.model.model.language_model.embed_tokens(input_ids)
            _, _, hidden_dim = inputs_embeds.shape
            selected = input_ids == self.config.image_token_index
            indices1 = selected.to(torch.int64).cumsum(1) - 1
            indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
            image_features_expanded = vision_embeds.reshape(-1, hidden_dim)[indices1]
            image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
            scalar_device = "cpu" if torch._dynamo.is_compiling() else input_ids.device
            inputs_embeds = torch.where(
                input_ids.shape[1] == torch.tensor(1, device=scalar_device), inputs_embeds, image_input_embeds
            )
            image_idx_output = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        else:
            if image_idx is None:
                image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)
            image_idx_output = image_idx

        # Build a QEffMiniMaxSparseCache combining the KV cache (2-tuples) and the separate index keys.
        sparse_layer_indices = [int(name.split(".")[-1]) for name in self.get_onnx_index_key_names()]
        index_keys_dict = None
        if index_keys is not None:
            index_keys_dict = {sparse_layer_indices[j]: index_keys[j] for j in range(len(sparse_layer_indices))}
        cache = QEffMiniMaxSparseCache.from_legacy_cache(past_key_values, index_keys=index_keys_dict)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=cache,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            msa_indexer_block_table=msa_indexer_block_table,
            msa_attn_block_table=msa_attn_block_table,
            use_cache=True,
        )
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[
            torch.arange(position_ids.shape[0], device=position_ids.device).view(-1, 1), logit_index
        ]
        logits = self.model.lm_head(hidden_states).float()

        result_cache = outputs.past_key_values
        if isinstance(result_cache, QEffMiniMaxSparseCache):
            past_kv_out = result_cache.to_kv_only_cache()
            index_keys_out = result_cache.get_index_keys_tuple()
        else:
            past_kv_out = tuple((t[0], t[1]) for t in result_cache)
            index_keys_out = tuple(t[2] for t in result_cache if len(t) == 3)

        return logits, vision_embeds.clone(), image_idx_output, past_kv_out, index_keys_out


class QEffMiniMaxM3SparseForConditionalGeneration(MiniMaxM3SparseForConditionalGeneration):
    def generate_npi_file(self, onnx_path: Union[str, Path], model_name: Optional[str] = None) -> str:
        del model_name
        return _generate_minimax_npi_file(onnx_path)

    def __qeff_init__(self):
        self.language_model = self.model.language_model
        self.config._attn_implementation = "eager"
        self.model.language_model.config._attn_implementation = "eager"
        self.model.vision_tower.config._attn_implementation = "eager"

    def _qaic_config(self) -> dict:
        return getattr(self, "qaic_config", None) or {}

    def _msa_execution_factor(self) -> int:
        qaic_config = self._qaic_config()
        return lcm(
            int(qaic_config.get("msa_indexer_dp", 1) or 1),
            int(qaic_config.get("msa_attn_dp", 1) or 1),
        )

    def _execution_batch_size(self, requested_batch_size: int) -> int:
        msa_factor = self._msa_execution_factor()
        if requested_batch_size % msa_factor == 0:
            return requested_batch_size
        return requested_batch_size * msa_factor

    def _uses_context_partitioned_main_kv(self) -> bool:
        qaic_config = self._qaic_config()
        # Sparse decode enters the GP cache path whenever either attention DP
        # or CP is enabled. Keep export examples and dynamic cache metadata in
        # that same layout so compiler-side DP splitting does not separate the
        # cache batch axis from its DP-folded KV rows.
        attn_dp = int(qaic_config.get("msa_attn_dp", 1) or 1)
        attn_cp = int(qaic_config.get("msa_attn_cp", 1) or 1)
        return attn_dp > 1 or attn_cp > 1

    def get_qeff_vision_encoder(self):
        return QEffMiniMaxM3VLEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffMiniMaxM3VLDecoderWrapper(self)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        image_idx=None,
        past_key_values=None,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_ids is None or position_ids is None or pixel_values is None or image_grid_thw is None:
            raise ValueError("input_ids, position_ids, pixel_values, and image_grid_thw must be provided.")
        if image_idx is None:
            image_idx = torch.zeros((1, 1), dtype=torch.int64, device=input_ids.device)

        image_features = self.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        image_features = image_features.to(device=input_ids.device, dtype=self.lm_head.weight.dtype)

        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        _, _, hidden_dim = inputs_embeds.shape
        selected = input_ids == self.config.image_token_index
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.shape[0], device=selected.device).view(-1, 1)
        image_features_expanded = image_features.reshape(-1, hidden_dim).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        scalar_device = "cpu" if torch._dynamo.is_compiling() else input_ids.device
        inputs_embeds = torch.where(
            input_ids.shape[1] == torch.tensor(1, device=scalar_device), inputs_embeds, image_input_embeds
        )

        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)

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

    def _uses_context_partitioned_indexer_kv(self) -> bool:
        qaic_config = self._qaic_config()
        return int(qaic_config.get("msa_indexer_cp", 1) or 1) > 1

    def _uses_row_folded_main_kv(self) -> bool:
        if not self._uses_context_partitioned_main_kv():
            return False
        qaic_config = self._qaic_config()
        return int(qaic_config.get("msa_attn_dp", 1) or 1) == 1

    def _indexer_kv_partition_shape(self, config, batch_size: int, seq_len: int) -> tuple[int, ...]:
        if not self._uses_context_partitioned_indexer_kv():
            return (batch_size, 1, seq_len, config.index_head_dim)
        qaic_config = self._qaic_config()
        dp = int(qaic_config.get("msa_indexer_dp", 1) or 1)
        cp = int(qaic_config.get("msa_indexer_cp", 1) or 1)
        hkv = int(qaic_config.get("indexer_n_head", 1) or 1)
        batch_size = self._execution_batch_size(batch_size)
        if batch_size % dp:
            raise ValueError(f"Indexer CP cache batch size {batch_size} must be divisible by msa_indexer_dp={dp}.")
        if seq_len % cp:
            raise ValueError(f"Indexer context length {seq_len} must be divisible by msa_indexer_cp={cp}.")
        if config.index_n_heads % hkv:
            raise ValueError(f"index_n_heads must be divisible by indexer_n_head={hkv}.")
        return ((batch_size // dp) * dp * cp * hkv, seq_len // cp, config.index_head_dim)

    def _main_kv_partition_shape(self, config, batch_size: int, seq_len: int) -> tuple[int, ...]:
        kv_cache_shape = get_padding_shape_from_config(config=config, batch_size=batch_size, seq_len=seq_len)
        if not self._uses_context_partitioned_main_kv():
            return tuple(kv_cache_shape)

        qaic_config = self._qaic_config()
        dp = int(qaic_config.get("msa_attn_dp", 1) or 1)
        cp = int(qaic_config.get("msa_attn_cp", 1) or 1)
        batch_size = self._execution_batch_size(batch_size)
        if batch_size % dp:
            raise ValueError(f"Main-KV CP cache batch size {batch_size} must be divisible by msa_attn_dp={dp}.")

        _, num_kv_heads, cache_len, head_dim = kv_cache_shape
        local_cache_len = max(1, cache_len // cp)
        if self._uses_row_folded_main_kv():
            index_block_size = int(getattr(config, "index_block_size", 1) or 1)
            if local_cache_len % index_block_size:
                local_cache_len = ((local_cache_len + index_block_size - 1) // index_block_size) * index_block_size
            return ((batch_size // dp) * dp * num_kv_heads * cp, local_cache_len, head_dim)
        return (batch_size // dp, dp * num_kv_heads * cp, local_cache_len, head_dim)

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        vision_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len if prefill_seq_len else constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        ctx_len = ctx_len if ctx_len else constants.ONNX_EXPORT_CTX_LEN
        vision_batch_size = batch_size if vision_batch_size is None else vision_batch_size
        # img_size is accepted by generic VLM compile APIs, but MiniMax-M3 VLM
        # specialization derives language/vision shapes from patch settings.
        # Drop it to avoid leaking `-img-size=None` into qaic-compile flags.
        compiler_options.pop("img_size", None)
        num_image_patches = int(compiler_options.pop("num_image_patches", 4))
        num_images = int(compiler_options.pop("num_images", 1))
        vision_size = int(compiler_options.pop("vision_size", num_image_patches))

        qaic_config = compiler_options.pop("qaic_config", None) or {}
        self.qaic_config = qaic_config
        msa_attn_dp = int(qaic_config.get("msa_attn_dp", 1) or 1)
        msa_attn_cp = int(qaic_config.get("msa_attn_cp", 1) or 1)
        indexer_dp = int(qaic_config.get("msa_indexer_dp", 1) or 1)
        indexer_cp = int(qaic_config.get("msa_indexer_cp", 1) or 1)
        indexer_hkv = int(qaic_config.get("indexer_n_head", 1) or 1)
        paged_kv = bool(qaic_config.get("paged_kv", False))
        shared_num_pages = qaic_config.get("num_logical_pages")
        if prefill_seq_len > 1 and (indexer_dp != 1 or indexer_cp != 1 or msa_attn_dp != 1 or msa_attn_cp != 1):
            raise ValueError(
                "MiniMax MSA prefill requires indexer DP=CP=1 and attention DP=CP=1; "
                "pipeline parallelism must be configured with mdp_num_partitions."
            )
        export_batch_size = self._execution_batch_size(batch_size)
        use_context_kv = self._uses_context_partitioned_main_kv() and not paged_kv
        use_row_folded_main_kv = self._uses_row_folded_main_kv() and not paged_kv
        use_context_indexer_kv = self._uses_context_partitioned_indexer_kv() and not paged_kv
        if paged_kv:
            page_size = int(
                qaic_config.get("page_block_size", getattr(self.config.text_config, "index_block_size", 128))
            )
            indexer_num_pages = _resolve_num_logical_pages(
                ctx_len, page_size, qaic_config.get("msa_indexer_num_logical_pages", shared_num_pages)
            )
            attn_num_pages = _resolve_num_logical_pages(
                ctx_len, page_size, qaic_config.get("msa_attn_num_logical_pages", shared_num_pages)
            )
            indexer_page_groups = _num_page_groups(indexer_num_pages, indexer_cp)
            attn_page_groups = _num_page_groups(attn_num_pages, msa_attn_cp)
            if prefill_seq_len == 1:
                index_block_size = int(getattr(self.config.text_config, "index_block_size", page_size))
                if page_size != index_block_size:
                    raise ValueError("Paged MSA decode requires page_block_size == index_block_size.")
                if int(getattr(self.config.text_config, "index_local_blocks", 1)) != 1:
                    raise ValueError("Paged MSA decode requires index_local_blocks == 1.")
                indexer_num_blocks = int(qaic_config.get("indexer_num_blocks") or qaic_config.get("num_kv_blocks") or 1)
                indexer_page_divisor = indexer_num_blocks * indexer_cp
                if indexer_num_pages % indexer_page_divisor:
                    raise ValueError(
                        f"Paged MSA indexer pages ({indexer_num_pages}) must be divisible by "
                        f"indexer_num_blocks * msa_indexer_cp ({indexer_page_divisor})."
                    )
                num_cores_per_device = int(qaic_config.get("num_cores_per_device", 1) or 1)
                if (indexer_num_pages // indexer_page_divisor) % num_cores_per_device:
                    raise ValueError(
                        "Paged MSA indexer page groups per block must be divisible by num_cores_per_device."
                    )
        if use_context_kv and ctx_len % msa_attn_cp:
            raise ValueError(f"Main-KV context length {ctx_len} must be divisible by msa_attn_cp={msa_attn_cp}.")
        kv_batch_size = kv_cache_batch_size if (continuous_batching and kv_cache_batch_size) else export_batch_size
        if use_context_kv and kv_batch_size % msa_attn_dp:
            raise ValueError(
                f"Main-KV cache batch size {kv_batch_size} must be divisible by msa_attn_dp={msa_attn_dp}."
            )
        if use_context_indexer_kv and ctx_len % indexer_cp:
            raise ValueError(f"Indexer context length {ctx_len} must be divisible by msa_indexer_cp={indexer_cp}.")
        if use_context_indexer_kv and kv_batch_size % indexer_dp:
            raise ValueError(
                f"Indexer cache batch size {kv_batch_size} must be divisible by msa_indexer_dp={indexer_dp}."
            )
        main_kv_batch_size = kv_batch_size // msa_attn_dp if use_context_kv else None
        main_kv_rows = (
            main_kv_batch_size * msa_attn_dp * self.model.language_model.config.num_key_value_heads * msa_attn_cp
            if use_row_folded_main_kv
            else None
        )
        main_kv_ctx_len = ctx_len // msa_attn_cp if use_context_kv else None
        indexer_kv_batch_size = kv_batch_size // indexer_dp if use_context_indexer_kv else None
        indexer_kv_rows = (
            indexer_kv_batch_size * indexer_dp * indexer_cp * indexer_hkv if use_context_indexer_kv else None
        )
        dp_multiplier = lcm(indexer_dp, msa_attn_dp)
        if export_batch_size < dp_multiplier or export_batch_size % dp_multiplier:
            raise ValueError(
                f"batch_size ({export_batch_size}) must be at least and divisible by "
                f"the combined MSA DP factor ({dp_multiplier})."
            )

        def _build_spec(seq_len, comp_ctx_lengths=None):
            spec = {
                "batch_size": full_batch_size if (continuous_batching and seq_len == 1) else export_batch_size,
                "seq_len": seq_len,
                "ctx_len": ctx_len,
                "num_image_patches": num_image_patches,
                "num_images": num_images,
            }
            if use_context_kv:
                if use_row_folded_main_kv:
                    spec["main_kv_rows"] = main_kv_rows
                else:
                    spec["main_kv_batch_size"] = main_kv_batch_size
                spec["main_kv_ctx_len"] = main_kv_ctx_len
            if use_context_indexer_kv:
                spec["indexer_kv_batch_size"] = indexer_kv_batch_size
                spec["indexer_kv_rows"] = indexer_kv_rows
                spec["indexer_kv_ctx_len"] = ctx_len // indexer_cp
            if qaic_config.get("paged_kv", False):
                cache_batch_size = kv_cache_batch_size if continuous_batching else export_batch_size
                spec["page_size"] = page_size
                spec["indexer_num_pages"] = indexer_num_pages
                spec["attn_num_pages"] = attn_num_pages
                spec["indexer_page_groups"] = indexer_page_groups
                spec["attn_page_groups"] = attn_page_groups
                spec["indexer_physical_pages"] = cache_batch_size * indexer_page_groups
                spec["attn_physical_pages"] = cache_batch_size * attn_page_groups
                spec["indexer_batch_local"] = spec["batch_size"] // int(qaic_config.get("msa_indexer_dp", 1) or 1)
                spec["attn_batch_local"] = spec["batch_size"] // int(qaic_config.get("msa_attn_dp", 1) or 1)
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            return spec

        def _build_lang_spec(seq_len, comp_ctx_lengths=None):
            spec = {
                "batch_size": full_batch_size if (continuous_batching and seq_len == 1) else export_batch_size,
                "seq_len": seq_len,
                "ctx_len": ctx_len,
                "vision_size": vision_size,
                "vision_batch_size": vision_batch_size,
            }
            if use_context_kv:
                if use_row_folded_main_kv:
                    spec["main_kv_rows"] = main_kv_rows
                else:
                    spec["main_kv_batch_size"] = main_kv_batch_size
                spec["main_kv_ctx_len"] = main_kv_ctx_len
            if use_context_indexer_kv:
                spec["indexer_kv_batch_size"] = indexer_kv_batch_size
                spec["indexer_kv_rows"] = indexer_kv_rows
                spec["indexer_kv_ctx_len"] = ctx_len // indexer_cp
            if qaic_config.get("paged_kv", False):
                cache_batch_size = kv_cache_batch_size if continuous_batching else export_batch_size
                spec["page_size"] = page_size
                spec["indexer_batch_local"] = spec["batch_size"] // int(qaic_config.get("msa_indexer_dp", 1) or 1)
                spec["attn_batch_local"] = spec["batch_size"] // int(qaic_config.get("msa_attn_dp", 1) or 1)
                spec["indexer_num_pages"] = indexer_num_pages
                spec["attn_num_pages"] = attn_num_pages
                spec["indexer_page_groups"] = indexer_page_groups
                spec["attn_page_groups"] = attn_page_groups
                spec["indexer_physical_pages"] = cache_batch_size * indexer_page_groups
                spec["attn_physical_pages"] = cache_batch_size * attn_page_groups
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            if full_batch_size and seq_len != 1:
                spec["full_batch_exec_size"] = full_batch_size
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            return spec

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            specs = [_build_spec(prefill_seq_len, c) for c in comp_ctx_lengths_prefill]
            specs.extend(_build_spec(1, c) for c in comp_ctx_lengths_decode)
        else:
            specs = [_build_spec(prefill_seq_len), _build_spec(1)]

        if kv_offload:
            vision = [{"batch_size": batch_size, "num_image_patches": num_image_patches, "num_images": num_images}]
            if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
                lang = [_build_lang_spec(prefill_seq_len, c) for c in comp_ctx_lengths_prefill]
                lang.extend(_build_lang_spec(1, c) for c in comp_ctx_lengths_decode)
            else:
                lang = [_build_lang_spec(prefill_seq_len), _build_lang_spec(1)]
            return {"vision": vision, "lang": lang}, compiler_options

        return specs, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        vision_dynamic_axes = {
            "pixel_values": {0: "num_image_patches"},
            "image_grid_thw": {0: "num_images"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }

        lm_config = self.model.language_model.config
        qaic_config = self._qaic_config()
        paged_kv = bool(qaic_config.get("paged_kv", False))
        # The index-key cache has two possible export layouts:
        #   prefill / CP=1: [batch, hkv, ctx_len, head_dim]
        #   decode / CP>1:  [rows, ctx_len / cp, head_dim]
        # Keep the dynamic axes tied to the same layout choice used by
        # get_dummy_index_keys above.
        use_context_indexer_kv = int(qaic_config.get("msa_indexer_cp", 1) or 1) > 1 and not paged_kv
        use_context_main_kv = self._uses_context_partitioned_main_kv() and not paged_kv
        use_row_folded_main_kv = self._uses_row_folded_main_kv() and not paged_kv
        if use_row_folded_main_kv:
            past_batch_axis = "main_kv_rows"
        elif use_context_main_kv:
            past_batch_axis = "main_kv_batch_size"
        else:
            past_batch_axis = "full_batch_size" if continuous_batching else "batch_size"
        past_ctx_axis = "main_kv_ctx_len" if use_context_main_kv else "ctx_len"
        standard_past_batch_axis = "full_batch_size" if continuous_batching else "batch_size"
        indexer_batch_axis = "full_batch_size" if continuous_batching else "batch_size"
        indexer_ctx_axis = "ctx_len"

        def _set_retained_state_axes(name: str, axes: dict[int, str]) -> None:
            """Keep each retained-state output's split signature identical to its input."""
            lang_dynamic_axes[name] = axes
            lang_dynamic_axes[f"{name}_RetainedState"] = axes.copy()

        layer_types = getattr(lm_config, "layer_types", None) or ["full_attention"] * lm_config.num_hidden_layers
        for i in range(lm_config.num_hidden_layers):
            is_sparse_layer = i < len(layer_types) and layer_types[i] == "minimax_m3_sparse"
            # CP-partitioned KV layout is specific to sparse M3 layers. Dense
            # layers retain standard [batch, Hkv, ctx_len, dim] caches even
            # when the model's sparse attention uses CP.
            layer_batch_axis = past_batch_axis if is_sparse_layer else standard_past_batch_axis
            layer_ctx_axis = past_ctx_axis if is_sparse_layer else "ctx_len"
            layer_cache_axes = (
                {
                    0: layer_batch_axis,
                }
                if is_sparse_layer and use_row_folded_main_kv
                else {
                    0: layer_batch_axis,
                }
            )
            _set_retained_state_axes(f"past_key.{i}", layer_cache_axes)
            _set_retained_state_axes(f"past_value.{i}", layer_cache_axes)
            if is_sparse_layer:
                if use_context_indexer_kv:
                    # Compact CP index cache: [rows, ctx_len / cp, dim].
                    index_cache_axes = {0: "indexer_kv_rows", 1: "indexer_kv_ctx_len"}
                else:
                    # Reference prefill index cache: [batch, hkv, ctx_len, dim].
                    index_cache_axes = {0: indexer_batch_axis, 2: indexer_ctx_axis}
                _set_retained_state_axes(f"index_key.{i}", index_cache_axes)

        if qaic_config.get("paged_kv", False):
            for i in range(lm_config.num_hidden_layers):
                if i < len(layer_types) and layer_types[i] == "minimax_m3_sparse":
                    for cache_name in ("past_key", "past_value"):
                        _set_retained_state_axes(
                            f"{cache_name}.{i}",
                            {0: "attn_physical_pages", 2: "page_size"},
                        )
                    _set_retained_state_axes(
                        f"index_key.{i}",
                        # {0: "indexer_physical_pages", 2: "page_size"},
                        {2: "page_size"},
                    )
            lang_dynamic_axes["msa_indexer_block_table"] = {1: "indexer_batch_local"}
            lang_dynamic_axes["msa_attn_block_table"] = {1: "attn_batch_local", 2: "attn_page_groups"}
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if qaic_config.get("static_export_dims", False):
            # Fixed batch and context sizes remove symbolic-shape work from tracing, which dominates export time.
            static_names = {"batch_size", "full_batch_size", "ctx_len"}
            for name, axes in lang_dynamic_axes.items():
                lang_dynamic_axes[name] = {axis: dim for axis, dim in axes.items() if dim not in static_names}
        if kv_offload:
            lang_dynamic_axes["vision_embeds_RetainedState"] = lang_dynamic_axes["vision_embeds"].copy()
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}

        lang_dynamic_axes["pixel_values_RetainedState"] = vision_dynamic_axes["pixel_values"].copy()
        dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        dynamic_axes.pop("vision_embeds")
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        lm_config = self.model.language_model.config
        layer_types = getattr(lm_config, "layer_types", None) or ["full_attention"] * lm_config.num_hidden_layers
        vision_output_names = ["vision_embeds"]
        output_names = ["logits", "pixel_values_RetainedState", "image_idx_output"]
        for i in range(lm_config.num_hidden_layers):
            output_names.append(f"past_key.{i}_RetainedState")
            output_names.append(f"past_value.{i}_RetainedState")
        for i in range(lm_config.num_hidden_layers):
            if i < len(layer_types) and layer_types[i] == "minimax_m3_sparse":
                output_names.append(f"index_key.{i}_RetainedState")
        if kv_offload:
            lang_output_names = ["logits", "vision_embeds_RetainedState", "image_idx_output"]
            for i in range(lm_config.num_hidden_layers):
                lang_output_names.append(f"past_key.{i}_RetainedState")
                lang_output_names.append(f"past_value.{i}_RetainedState")
            for i in range(lm_config.num_hidden_layers):
                if i < len(layer_types) and layer_types[i] == "minimax_m3_sparse":
                    lang_output_names.append(f"index_key.{i}_RetainedState")
            return {"vision": vision_output_names, "lang": lang_output_names}
        return output_names

    def get_dummy_pkv_cache(
        self,
        config,
        batch_size,
        seq_len,
        dtype=None,
        *,
        paged=False,
        dp=1,
        cp=1,
        page_block_size=128,
        num_logical_pages=None,
        hkv=None,
    ):
        dtype = dtype or getattr(config, "torch_dtype", torch.float32) or torch.float32
        kv_cache_shape = self._main_kv_partition_shape(config=config, batch_size=batch_size, seq_len=seq_len)
        standard_kv_cache_shape = tuple(
            get_padding_shape_from_config(config=config, batch_size=batch_size, seq_len=seq_len)
        )
        layer_types = getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
        batch_local = batch_size // dp
        num_pages = (
            (seq_len + page_block_size - 1) // page_block_size if num_logical_pages is None else int(num_logical_pages)
        )
        physical_pages = dp * batch_local * _num_page_groups(num_pages, cp)
        past_key_values = []
        for layer_idx in range(config.num_hidden_layers):
            layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
            if paged and layer_type == "minimax_m3_sparse":
                shape = (
                    physical_pages,
                    dp * cp * config.num_key_value_heads,
                    page_block_size,
                    config.head_dim,
                )
            elif layer_type == "minimax_m3_sparse":
                shape = kv_cache_shape
            else:
                # Generic attention layers use CtxScatter and therefore keep
                # the standard [batch, Hkv, ctx_len, head_dim] layout.  Only
                # sparse M3 layers use the CP-partitioned GP cache above.
                shape = standard_kv_cache_shape
            past_key_values.append([torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)])
        return past_key_values

    def get_dummy_index_keys(
        self,
        config,
        batch_size,
        seq_len,
        dtype=None,
        *,
        paged=False,
        dp=1,
        cp=1,
        page_block_size=128,
        num_logical_pages=None,
        hkv=None,
        prefill=False,
    ):
        dtype = dtype or getattr(config, "torch_dtype", torch.float32) or torch.float32
        layer_types = getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
        batch_local = batch_size // dp
        num_pages = (
            (seq_len + page_block_size - 1) // page_block_size if num_logical_pages is None else int(num_logical_pages)
        )
        physical_pages = dp * batch_local * _num_page_groups(num_pages, cp)
        # MSA prefill uses the reference rank-4 cache layout consumed by
        # M3CtxScatter: [batch, indexer_kv_heads, ctx_len, head_dim].  The
        # compact [rows, ctx_len / cp, head_dim] layout is decode-only.
        if prefill and not paged:
            index_key_shape = (
                batch_size,
                int(hkv or getattr(config, "indexer_n_head", 1) or 1),
                seq_len,
                config.index_head_dim,
            )
        else:
            index_key_shape = self._indexer_kv_partition_shape(config, batch_size, seq_len)
        paged_shape = (
            physical_pages,
            dp * cp * (hkv or getattr(config, "indexer_n_head", 1)),
            page_block_size,
            config.index_head_dim,
        )
        index_keys = []
        for i in range(config.num_hidden_layers):
            if layer_types[i] == "minimax_m3_sparse":
                index_keys.append(torch.zeros(paged_shape if paged else index_key_shape, dtype=dtype))
        return index_keys

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        ctx_len: Optional[int] = None,
        **kwargs,
    ):
        ctx_len = constants.ONNX_EXPORT_CTX_LEN if ctx_len is None else int(ctx_len)
        prefill_seq_len = kwargs.get("prefill_seq_len")
        if prefill_seq_len is None:
            prefill_seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        prefill_seq_len = int(prefill_seq_len)
        if prefill_seq_len > 1:
            for module in self.model.language_model.modules():
                blocking_config = getattr(module, "attn_blocking_config", None)
                export_seq_len = getattr(blocking_config, "prefill_export_seq_len", None)
                if export_seq_len is not None:
                    prefill_seq_len = int(export_seq_len)
                    break
        past_seq_len = int(kwargs.get("past_seq_len", ctx_len))
        qaic_config = self._qaic_config()
        indexer_dp = int(qaic_config.get("msa_indexer_dp", 1) or 1)
        indexer_cp = int(qaic_config.get("msa_indexer_cp", 1) or 1)
        attn_dp = int(qaic_config.get("msa_attn_dp", 1) or 1)
        attn_cp = int(qaic_config.get("msa_attn_cp", 1) or 1)
        prefill = prefill_seq_len > 1
        if prefill and (indexer_dp != 1 or indexer_cp != 1 or attn_dp != 1 or attn_cp != 1):
            raise ValueError(
                "MiniMax MSA prefill export inputs require indexer DP=CP=1 and attention DP=CP=1; "
                "compact MSA caches are decode-only."
            )
        dp_multiplier = lcm(indexer_dp, attn_dp)
        requested_batch_size = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE if kwargs.get("batch_size") is None else int(kwargs["batch_size"])
        )
        # Export uses the caller-provided batch unchanged; validate DP compatibility.
        batch_size = requested_batch_size
        if batch_size < dp_multiplier or batch_size % dp_multiplier:
            raise ValueError(
                f"batch_size ({batch_size}) must be at least and divisible by "
                f"the combined MSA DP factor ({dp_multiplier})."
            )
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        if fbs < dp_multiplier or fbs % dp_multiplier:
            raise ValueError(
                f"export cache batch size ({fbs}) must be at least and divisible by "
                f"the combined MSA DP factor ({dp_multiplier})."
            )
        cache_batch_size = fbs
        dtype = getattr(self.config, "torch_dtype", torch.float32) or torch.float32
        paged_kv = bool(qaic_config.get("paged_kv", False))
        page_block_size = int(
            qaic_config.get("page_block_size", getattr(self.config.text_config, "index_block_size", 128))
        )
        shared_num_pages = qaic_config.get("num_logical_pages")
        indexer_num_pages = _resolve_num_logical_pages(
            ctx_len, page_block_size, qaic_config.get("msa_indexer_num_logical_pages", shared_num_pages)
        )
        attn_num_pages = _resolve_num_logical_pages(
            ctx_len, page_block_size, qaic_config.get("msa_attn_num_logical_pages", shared_num_pages)
        )
        indexer_page_groups = _num_page_groups(indexer_num_pages, indexer_cp)
        attn_page_groups = _num_page_groups(attn_num_pages, attn_cp)
        patch_dim = (
            self.config.vision_config.num_channels
            * self.config.vision_config.temporal_patch_size
            * self.config.vision_config.patch_size
            * self.config.vision_config.patch_size
        )
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        num_image_patches = int(torch.prod(image_grid_thw).item())

        inputs = {
            "input_ids": torch.zeros((batch_size, prefill_seq_len), dtype=torch.int64),
            "pixel_values": torch.zeros((num_image_patches, patch_dim), dtype=dtype),
            "image_grid_thw": image_grid_thw,
            "position_ids": torch.arange(prefill_seq_len, dtype=torch.int64)
            .view(1, prefill_seq_len)
            .repeat(batch_size, 1),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
        }
        inputs["input_ids"][:, 0] = self.config.image_token_index
        past_key_values = self.get_dummy_pkv_cache(
            config=self.model.language_model.config,
            batch_size=cache_batch_size if continuous_batching else batch_size,
            seq_len=past_seq_len,
            dtype=dtype,
            paged=paged_kv,
            dp=attn_dp,
            cp=attn_cp,
            page_block_size=page_block_size,
            num_logical_pages=attn_num_pages if paged_kv else None,
        )
        index_keys = self.get_dummy_index_keys(
            config=self.model.language_model.config,
            batch_size=cache_batch_size if continuous_batching else batch_size,
            seq_len=ctx_len,
            dtype=dtype,
            paged=paged_kv,
            dp=indexer_dp,
            cp=indexer_cp,
            page_block_size=page_block_size,
            num_logical_pages=indexer_num_pages if paged_kv else None,
            hkv=int(qaic_config.get("indexer_n_head", 1) or 1),
            prefill=prefill,
        )
        inputs["past_key_values"] = past_key_values
        inputs["index_keys"] = index_keys
        if paged_kv:
            if batch_size % indexer_dp or batch_size % attn_dp:
                raise ValueError("Paged MiniMax block-table DP factors must divide batch_size.")
            inputs["msa_indexer_block_table"] = (
                torch.arange(batch_size // indexer_dp * indexer_page_groups, dtype=torch.int32)
                .view(1, batch_size // indexer_dp, indexer_page_groups)
                .expand(indexer_dp, -1, -1)
                .contiguous()
            )
            inputs["msa_attn_block_table"] = (
                torch.arange(batch_size // attn_dp * attn_page_groups, dtype=torch.int32)
                .view(1, batch_size // attn_dp, attn_page_groups)
                .expand(attn_dp, -1, -1)
                .contiguous()
            )
        if continuous_batching:
            inputs["batch_index"] = torch.arange(batch_size).view(batch_size, 1)
        if comp_ctx_lengths is not None:
            inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)
        if kv_offload:
            vision_inputs = {
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"],
            }
            lang_inputs = {
                "input_ids": inputs["input_ids"],
                "vision_embeds": torch.zeros(
                    (batch_size, num_image_patches, self.model.language_model.config.hidden_size), dtype=dtype
                ),
                "position_ids": inputs["position_ids"],
                "image_idx": inputs["image_idx"],
                "past_key_values": past_key_values,
                "index_keys": index_keys,
            }
            if paged_kv:
                lang_inputs["msa_indexer_block_table"] = inputs["msa_indexer_block_table"]
                lang_inputs["msa_attn_block_table"] = inputs["msa_attn_block_table"]
            if continuous_batching:
                lang_inputs["batch_index"] = inputs["batch_index"]
            if comp_ctx_lengths is not None:
                lang_inputs["comp_ctx_lengths"] = inputs["comp_ctx_lengths"]
            return {"vision": vision_inputs, "lang": lang_inputs}
        return inputs

    def get_inputs_info(self):
        patch_dim = (
            self.config.vision_config.num_channels
            * self.config.vision_config.temporal_patch_size
            * self.config.vision_config.patch_size
            * self.config.vision_config.patch_size
        )
        inputs_info = [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=self.config.torch_dtype, shape=("num_image_patches", patch_dim)),
            IOInfo(name="image_grid_thw", datatype=torch.int64, shape=("num_images", 3)),
        ]
        qaic_config = self._qaic_config()
        if qaic_config.get("paged_kv", False):
            inputs_info.extend(
                [
                    IOInfo(
                        name="msa_indexer_block_table",
                        datatype=torch.int32,
                        shape=("msa_indexer_dp", "indexer_batch_local", "indexer_page_groups"),
                    ),
                    IOInfo(
                        name="msa_attn_block_table",
                        datatype=torch.int32,
                        shape=("msa_attn_dp", "attn_batch_local", "attn_page_groups"),
                    ),
                ]
            )
        return inputs_info
