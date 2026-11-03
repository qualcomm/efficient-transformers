"""Export, compile, validate, and profile one Qwen3.8-Flash QSA prefill block.

This is deliberately an attention-only graph: it runs layer 3's QSA with the
four retained states, without entering the decode-only GDN path. The default
workload matches ``attention/qsa/HANDOFF_FOR_QEFF_MIGRATION.md`` (PL=1024,
CL=65536, start=64512, AI100, four cores).

Example::

    HF_HUB_CACHE=/home/huggingface_hub \\
    /home/ochougul/envs/qwen38f/bin/python \\
        examples/text_generation/qwen4_exp_prefill_qsa_qpc.py \\
        --snapshot /home/huggingface_hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540 \\
        --checkpoint qwen3.8_flash_experiment/example_repro_20261013/qwen38_flash_4layer_checkpoint_v2 \\
        --compile-dir attention_benchmarks/qsa_only_prefill_cores4_20261103
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
_QEFF_PREPARE_MROPE = None

BATCH_SIZE = 1
PREFILL_SEQ_LEN = 1024
CONTEXT_LENGTH = 65_536
START_POSITION = 64_512
QSA_LAYER_INDEX = 3
NUM_CORES = 4
DEVICE_LIST = "30"
MODEL_DTYPE = torch.float32
OPSTATS_ITERATIONS = 30
OPSTATS_PROFILE_START_ITER = 20
OPSTATS_PROFILE_SAMPLES = 2
OPSTATS_STATS_LEVEL = 70
REFERENCE_LATENCY_MS = 484.58
COMPILER_OPTS_UNSUPPORTED = (
    "-aic-hoist-vtcm-loads=false -aic-op-stats-verbosity 2 -aic-userdma-async=0 -aic-hmx-async=0 -aic-sync-ts-starts"
)
QSA_CONFIG = {
    "indexer_dp": 1,
    "indexer_cp": 1,
    "attn_dp": 1,
    "attn_cp": 1,
    "num_cores_per_device": NUM_CORES,
    "indexer_num_blocks": 16,
    "indexer_ql_chunk": 512,
    "indexer_q_block_size": 128,
    "indexer_topk_blocking": 4,
    "sparse_q_block_size": 64,
    "sparse_kv_num_blocks": 8,
    "skip_kv": False,
}
HOST_INPUT_NAMES = ("hidden_states", "position_ids")
STATE_NAMES = ("qsa_key_state", "qsa_value_state", "qsa_index_state", "qsa_partial_state")
OUTPUT_NAMES = (
    "attention_output",
    "qsa_key_state_RetainedState",
    "qsa_value_state_RetainedState",
    "qsa_index_state_RetainedState",
    "qsa_partial_state_RetainedState",
)


def _load_config(snapshot: Path) -> Qwen4ExpTextConfig:
    raw = json.loads((snapshot / "config.json").read_text())["text_config"]
    raw["dtype"] = "float32"
    raw["num_hidden_layers"] = 4
    raw["layer_types"] = raw["layer_types"][:4]
    raw["ple_layer_ids"] = [layer_id for layer_id in raw.get("ple_layer_ids", []) if layer_id <= 4]
    raw["max_position_embeddings"] = CONTEXT_LENGTH
    config = Qwen4ExpTextConfig(**raw)
    config._attn_implementation = "eager"
    return config


def _ensure_checkpoint(snapshot: Path, checkpoint: Path) -> None:
    if checkpoint.is_dir():
        required = (checkpoint / "config.json", checkpoint / "model.safetensors.index.json")
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Four-layer checkpoint is incomplete: {missing}")
        return
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint target exists but is not a directory: {checkpoint}")

    create_script = Path(__file__).with_name("qwen4_exp_create_4layer_checkpoint.py")
    command = [sys.executable, str(create_script), "--snapshot", str(snapshot), "--output", str(checkpoint)]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _load_attention_pair(config: Qwen4ExpTextConfig, checkpoint: Path):
    from QEfficient.transformers.models.qwen4_exp.modeling_qwen4_exp import QEffQwen4ExpTextAttention

    weights_index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.layers.{QSA_LAYER_INDEX}.self_attn."

    def load_weights(module):
        tensors = {}
        for name in module.state_dict():
            source_name = prefix + name
            try:
                shard_name = weights_index[source_name]
            except KeyError as error:
                raise KeyError(f"Local four-layer checkpoint is missing {source_name}") from error
            with safe_open(str(checkpoint / shard_name), framework="pt", device="cpu") as shard:
                tensors[name] = shard.get_tensor(source_name).to(dtype=MODEL_DTYPE)
        module.load_state_dict(tensors, strict=True)
        module.to(dtype=MODEL_DTYPE).eval()

    hf_attention = Qwen4ExpTextAttention(config, layer_idx=QSA_LAYER_INDEX)
    qeff_attention = QEffQwen4ExpTextAttention(config, layer_idx=QSA_LAYER_INDEX)
    load_weights(hf_attention)
    load_weights(qeff_attention)
    qeff_attention.configure_qsa(
        QSA_CONFIG,
        ctx_len=CONTEXT_LENGTH,
        seq_len=PREFILL_SEQ_LEN,
        prefill_only=True,
    )
    return hf_attention, qeff_attention


class _QSAState:
    """Minimal cache surface consumed by one QEff QSA attention module."""

    def __init__(self, position_ids, states):
        self.position_ids = position_ids
        self.qsa_key_states = [None] * (QSA_LAYER_INDEX + 1)
        self.qsa_value_states = [None] * (QSA_LAYER_INDEX + 1)
        self.qsa_index_states = [None] * (QSA_LAYER_INDEX + 1)
        self.qsa_partial_states = [None] * (QSA_LAYER_INDEX + 1)
        for state_list, state in zip(
            (self.qsa_key_states, self.qsa_value_states, self.qsa_index_states, self.qsa_partial_states),
            states,
            strict=True,
        ):
            state_list[QSA_LAYER_INDEX] = state

    def update_qsa_state(self, layer_idx, key_state, value_state, index_state, partial_state):
        self.qsa_key_states[layer_idx] = key_state
        self.qsa_value_states[layer_idx] = value_state
        self.qsa_index_states[layer_idx] = index_state
        self.qsa_partial_states[layer_idx] = partial_state


class _AttentionBlock(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attn = attention


class _LayerStack(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(QSA_LAYER_INDEX)] + [_AttentionBlock(attention)]
        )


class QSAOnlyGraph(torch.nn.Module):
    """Expose only QSA as a retained-state graph, with model-compatible RoPE."""

    def __init__(self, attention, config, rotary_tables):
        super().__init__()
        self.config = config
        self.model = _LayerStack(attention)
        rotary_width, cos_cached, sin_cached, pooled_cos_cached, pooled_sin_cached = rotary_tables
        self.rotary_width = rotary_width
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])
        self.register_buffer("cos_cached", cos_cached.contiguous())
        self.register_buffer("sin_cached", sin_cached.contiguous())
        self.register_buffer("pooled_cos_cached", pooled_cos_cached.contiguous())
        self.register_buffer("pooled_sin_cached", pooled_sin_cached.contiguous())

    def forward(
        self,
        hidden_states,
        position_ids,
        qsa_key_state,
        qsa_value_state,
        qsa_index_state,
        qsa_partial_state,
    ):
        rotary_position_ids = position_ids[1:]
        cos, sin = _QEFF_PREPARE_MROPE(
            self.cos_cached,
            self.sin_cached,
            rotary_position_ids,
            self.mrope_section,
            dtype=hidden_states.dtype,
            rotary_width=self.rotary_width,
        )
        position_embeddings = (
            cos,
            sin,
            self.cos_cached,
            self.sin_cached,
            rotary_position_ids,
            self.mrope_section,
            self.pooled_cos_cached,
            self.pooled_sin_cached,
        )
        cache = _QSAState(
            rotary_position_ids,
            (qsa_key_state, qsa_value_state, qsa_index_state, qsa_partial_state),
        )
        attention_output, _ = self.model.layers[QSA_LAYER_INDEX].self_attn(
            hidden_states,
            position_embeddings,
            past_key_values=cache,
        )
        return (
            attention_output,
            cache.qsa_key_states[QSA_LAYER_INDEX],
            cache.qsa_value_states[QSA_LAYER_INDEX],
            cache.qsa_index_states[QSA_LAYER_INDEX],
            cache.qsa_partial_states[QSA_LAYER_INDEX].clone(),
        )


def _rotary_tables(config):
    from QEfficient.transformers.models.qwen4_exp.modeling_qwen4_exp import (
        QEffQwen4ExpTextRotaryEmbedding,
        _build_qwen4_exp_pooled_mrope_cache,
    )

    rotary = QEffQwen4ExpTextRotaryEmbedding(config=config, device="cpu")
    cos_cached = (rotary.cos_cached * rotary.attention_scaling).contiguous()
    sin_cached = (rotary.sin_cached * rotary.attention_scaling).contiguous()
    rotary_width = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0)) // 2
    pooled_cos, pooled_sin = _build_qwen4_exp_pooled_mrope_cache(cos_cached, sin_cached, config, rotary_width)
    return rotary_width, cos_cached, sin_cached, pooled_cos, pooled_sin


class _QSAExportAdapter:
    """Small bridge to QEFFBaseModel's ONNX transforms and QAIC compiler setup."""

    def __new__(cls, model):
        from QEfficient.base.modeling_qeff import QEFFBaseModel
        from QEfficient.base.onnx_transforms import CustomOpTransform

        class Adapter(QEFFBaseModel):
            _pytorch_transforms: ClassVar[list] = []
            _onnx_transforms: ClassVar[list] = []

            @property
            def get_model_config(self):
                return self.model.config.to_diff_dict()

            def export(self, inputs, output_names, export_dir):
                self.onnx_path = self._export(
                    inputs,
                    output_names=output_names,
                    dynamic_axes={},
                    export_dir=export_dir,
                    offload_pt_weights=False,
                    dynamo=True,
                )
                return self.onnx_path

            def compile(self, onnx_path, compile_dir, *, profiling=False):
                options = {
                    "onnx_path": str(onnx_path),
                    "compile_dir": str(compile_dir),
                    "mxint8_kv_cache": True,
                    "specializations": [
                        {
                            "batch_size": BATCH_SIZE,
                            "seq_len": PREFILL_SEQ_LEN,
                            "ctx_len": CONTEXT_LENGTH,
                        }
                    ],
                    "custom_io": _custom_io_config(),
                    "mdp_ts_num_devices": 1,
                    "prefill_only": True,
                    "offload_pt_weights": False,
                    "convert_to_fp16": True,
                    "mxfp6_matmul": True,
                    "aic_num_cores": NUM_CORES,
                    "aic_hw_version": "ai100",
                    "retained_state": True,
                    "allow_mxint8_mdp_io": True,
                    "user_tiled": True,
                    "split_model_io": True,
                }
                if profiling:
                    options.update(
                        {
                            "stats_level": OPSTATS_STATS_LEVEL,
                            "ddr_stats": True,
                            "aic_pmu_recipe": "KernelUtil",
                            "aic_perf_metrics": True,
                        }
                    )
                return Path(self._compile(**options))

        Adapter._onnx_transforms = [CustomOpTransform]
        return Adapter(model)


def _custom_io_config() -> dict[str, str]:
    result = {}
    for name in STATE_NAMES:
        precision = "float" if name == "qsa_partial_state" else "mxint8"
        result[name] = precision
        result[f"{name}_RetainedState"] = precision
    return result


def _make_inputs(config):
    torch.manual_seed(17)
    hidden_states = torch.randn((BATCH_SIZE, PREFILL_SEQ_LEN, config.hidden_size), dtype=MODEL_DTYPE) * 0.5
    positions = torch.arange(START_POSITION, START_POSITION + PREFILL_SEQ_LEN, dtype=torch.long)
    position_ids = positions.view(1, 1, -1).expand(4, BATCH_SIZE, PREFILL_SEQ_LEN).contiguous()
    states = (
        torch.zeros((BATCH_SIZE, config.num_key_value_heads, CONTEXT_LENGTH, config.head_dim), dtype=MODEL_DTYPE),
        torch.zeros((BATCH_SIZE, config.num_key_value_heads, CONTEXT_LENGTH, config.head_dim), dtype=MODEL_DTYPE),
        torch.zeros(
            (BATCH_SIZE, 1, CONTEXT_LENGTH // config.indexer_compress_ratio, config.indexer_head_dim),
            dtype=MODEL_DTYPE,
        ),
        torch.zeros((BATCH_SIZE, 1, config.indexer_head_dim), dtype=torch.float32),
    )
    return hidden_states, position_ids, states


def _seed_hf_prefix(cache, config):
    layer = cache.layers[QSA_LAYER_INDEX]
    layer.keys = torch.zeros(
        (BATCH_SIZE, config.num_key_value_heads, START_POSITION, config.head_dim), dtype=MODEL_DTYPE
    )
    layer.values = torch.zeros_like(layer.keys)
    layer.is_initialized = True
    layer.indexer_keys = torch.zeros((BATCH_SIZE, START_POSITION, config.indexer_head_dim), dtype=MODEL_DTYPE)
    layer.is_indexer_initialized = True
    cache.position_ids = torch.arange(START_POSITION, dtype=torch.long).view(1, 1, -1).expand(3, BATCH_SIZE, -1)


@torch.no_grad()
def _run_hf_sparse_reference(hf_attention, config, rotary, inputs):
    """Run HF projections/indexer and exact selected-token attention without a dense Q×CL matrix."""
    hidden_states, position_ids, _ = inputs
    cache = DynamicCache(config=config)
    _seed_hf_prefix(cache, config)
    full_positions = torch.arange(CONTEXT_LENGTH, dtype=torch.long).view(1, 1, -1).expand(3, BATCH_SIZE, -1)
    rotary_input = torch.empty((BATCH_SIZE, 1, config.hidden_size), dtype=MODEL_DTYPE)
    full_cos, full_sin = rotary(rotary_input, full_positions)
    position_embeddings = (full_cos, full_sin)
    key_positions = torch.arange(CONTEXT_LENGTH).view(1, 1, 1, CONTEXT_LENGTH)
    query_positions = position_ids[0].view(BATCH_SIZE, 1, PREFILL_SEQ_LEN, 1)
    causal_mask = key_positions <= query_positions

    selected_mask = hf_attention.indexer(hidden_states, position_embeddings, causal_mask, cache)[:, 0]
    query_gate = hf_attention.q_proj(hidden_states).view(BATCH_SIZE, PREFILL_SEQ_LEN, -1, config.head_dim * 2)
    query_states, gate = query_gate.chunk(2, dim=-1)
    gate = gate.reshape(BATCH_SIZE, PREFILL_SEQ_LEN, -1)
    query_states = hf_attention.q_norm(query_states).transpose(1, 2)
    key_states = hf_attention.k_norm(
        hf_attention.k_proj(hidden_states).view(BATCH_SIZE, PREFILL_SEQ_LEN, -1, config.head_dim)
    ).transpose(1, 2)
    value_states = (
        hf_attention.v_proj(hidden_states).view(BATCH_SIZE, PREFILL_SEQ_LEN, -1, config.head_dim).transpose(1, 2)
    )
    cos, sin = full_cos[:, -PREFILL_SEQ_LEN:, :], full_sin[:, -PREFILL_SEQ_LEN:, :]
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states, value_states = cache.update(key_states, value_states, QSA_LAYER_INDEX)

    max_selected = hf_attention.indexer.token_budget + hf_attention.indexer.compress_ratio - 1
    selected_counts, selected_indices = torch.topk(selected_mask.to(torch.int32), max_selected, dim=-1, sorted=True)
    selected_valid = selected_counts.bool()
    batch, query_len, _ = hidden_states.shape
    kv_heads = config.num_key_value_heads
    heads_per_kv = config.num_attention_heads // kv_heads
    query_work = query_states.reshape(batch, kv_heads, heads_per_kv, query_len, config.head_dim)
    output_chunks = []
    query_chunk_size = 8
    for q_start in range(0, query_len, query_chunk_size):
        q_end = min(q_start + query_chunk_size, query_len)
        indices = selected_indices[:, q_start:q_end]
        valid = selected_valid[:, q_start:q_end]
        q_count, selected_count = indices.shape[1:]
        gather_indices = indices[:, None, :, :, None].expand(batch, kv_heads, q_count, selected_count, config.head_dim)
        key_source = key_states.unsqueeze(2).expand(batch, kv_heads, q_count, CONTEXT_LENGTH, config.head_dim)
        value_source = value_states.unsqueeze(2).expand_as(key_source)
        selected_keys = torch.gather(key_source, 3, gather_indices)
        selected_values = torch.gather(value_source, 3, gather_indices)
        query_chunk = query_work[..., q_start:q_end, :]
        logits = torch.einsum("bhgqd,bhqkd->bhgqk", query_chunk, selected_keys) * hf_attention.scaling
        logits = logits.masked_fill(~valid[:, None, None, :, :], -3.0e4)
        probabilities = torch.softmax(logits.float(), dim=-1).to(query_states.dtype)
        output = torch.einsum("bhgqk,bhqkd->bhgqd", probabilities, selected_values)
        output_chunks.append(output)

    attn_output = torch.cat(output_chunks, dim=3).permute(0, 3, 1, 2, 4)
    attn_output = attn_output.reshape(batch, query_len, -1).contiguous() * gate.sigmoid()
    attn_output = hf_attention.o_proj(attn_output)
    return attn_output, cache


def _max_mean_error(reference, candidate):
    reference = reference.detach().float().cpu().numpy()
    candidate = candidate.detach().float().cpu().numpy()
    if reference.shape != candidate.shape:
        raise AssertionError(f"Output shape mismatch: HF={reference.shape}, candidate={candidate.shape}")
    difference = np.abs(reference - candidate)
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
        "rms": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
    }


def _compare(reference, candidate, *, atol, rtol, label):
    metrics = _max_mean_error(reference, candidate)
    metrics.update({"atol": atol, "rtol": rtol})
    if not np.allclose(
        reference.detach().float().cpu().numpy(), candidate.detach().float().cpu().numpy(), atol=atol, rtol=rtol
    ):
        raise AssertionError(f"{label} parity failed: {metrics}")
    return metrics


def _validate_retained_states(config, hf_cache, qeff_states, rotary_tables, qeff_attention):
    _, qeff_key, qeff_value, qeff_index, qeff_partial = qeff_states
    hf_layer = hf_cache.layers[QSA_LAYER_INDEX]
    q_len_slice = slice(START_POSITION, START_POSITION + PREFILL_SEQ_LEN)
    key_error = _max_mean_error(hf_layer.keys[:, :, q_len_slice], qeff_key[:, :, q_len_slice])
    value_error = _max_mean_error(hf_layer.values[:, :, q_len_slice], qeff_value[:, :, q_len_slice])
    torch.testing.assert_close(hf_layer.keys[:, :, q_len_slice], qeff_key[:, :, q_len_slice])
    torch.testing.assert_close(hf_layer.values[:, :, q_len_slice], qeff_value[:, :, q_len_slice])

    ratio = config.indexer_compress_ratio
    pooled_positions = torch.arange(CONTEXT_LENGTH // ratio, dtype=torch.long) * ratio
    pooled_cos = rotary_tables[3].index_select(0, pooled_positions).unsqueeze(0)
    pooled_sin = rotary_tables[4].index_select(0, pooled_positions).unsqueeze(0)
    raw_index = hf_layer.indexer_keys.reshape(BATCH_SIZE, CONTEXT_LENGTH // ratio, ratio, config.indexer_head_dim)
    expected_index = raw_index.float().mean(dim=2).to(raw_index.dtype)
    expected_index = qeff_attention.indexer.k_layernorm(expected_index)
    expected_index = apply_rotary_pos_emb(
        expected_index.unsqueeze(2), cos=pooled_cos, sin=pooled_sin, unsqueeze_dim=2
    ).squeeze(2)
    actual_index = qeff_index.reshape(BATCH_SIZE, -1, config.indexer_head_dim)
    index_error = _max_mean_error(expected_index, actual_index)
    torch.testing.assert_close(actual_index, expected_index)
    torch.testing.assert_close(qeff_partial, torch.zeros_like(qeff_partial))
    return {"key_cache": key_error, "value_cache": value_error, "index_cache": index_error, "partial": "zero"}


def _build_parity_inputs(config, hf_attention, qeff_attention, rotary_tables):
    inputs = _make_inputs(config)
    hidden_states, position_ids, states = inputs
    rotary = Qwen4ExpTextRotaryEmbedding(config)
    hf_output, hf_cache = _run_hf_sparse_reference(hf_attention, config, rotary, inputs)
    graph = QSAOnlyGraph(qeff_attention, config, rotary_tables).eval()
    with torch.no_grad():
        qeff_outputs = graph(hidden_states, position_ids, *states)
    return hf_output, hf_cache, graph, qeff_outputs, inputs


def _validate_graph(onnx_path: Path, config):
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    graph_inputs = {value.name: value for value in graph.graph.input}
    graph_outputs = {value.name: value for value in graph.graph.output}

    def shape(value_info):
        return [
            dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in value_info.type.tensor_type.shape.dim
        ]

    expected = {
        "qsa_key_state": [BATCH_SIZE, config.num_key_value_heads, CONTEXT_LENGTH, config.head_dim],
        "qsa_value_state": [BATCH_SIZE, config.num_key_value_heads, CONTEXT_LENGTH, config.head_dim],
        "qsa_index_state": [BATCH_SIZE, 1, CONTEXT_LENGTH // config.indexer_compress_ratio, config.indexer_head_dim],
        "qsa_partial_state": [BATCH_SIZE, 1, config.indexer_head_dim],
    }
    state_shapes = {}
    for name, expected_shape in expected.items():
        output_name = f"{name}_RetainedState"
        if name not in graph_inputs or output_name not in graph_outputs:
            raise ValueError(f"Missing retained-state graph binding: {name}/{output_name}")
        input_shape, output_shape = shape(graph_inputs[name]), shape(graph_outputs[output_name])
        if input_shape != expected_shape or output_shape != expected_shape:
            raise ValueError(
                f"Unexpected state shape for {name}: input={input_shape}, "
                f"output={output_shape}, expected={expected_shape}"
            )
        state_shapes[name] = input_shape
    if "attention_output" not in graph_outputs or shape(graph_outputs["attention_output"]) != [
        BATCH_SIZE,
        PREFILL_SEQ_LEN,
        config.hidden_size,
    ]:
        raise ValueError("Exported QSA graph has an unexpected attention output shape")
    nodes = list(graph.graph.node)
    for function in graph.functions:
        nodes.extend(function.node)
    op_types = [node.op_type.lower() for node in nodes]
    topk_count = sum(item == "topk" for item in op_types)
    gather_count = sum("gather" in item for item in op_types)
    if not topk_count or not gather_count:
        raise ValueError(f"Sparse TopK/gather ops missing from ONNX: TopK={topk_count}, gather={gather_count}")
    dense = []
    for info in (*graph.graph.value_info, *graph.graph.input, *graph.graph.output):
        tensor_shape = shape(info)
        if len(tensor_shape) >= 2 and tensor_shape[-2:] == [PREFILL_SEQ_LEN, CONTEXT_LENGTH]:
            dense.append((info.name, tensor_shape))
    if dense:
        raise ValueError(f"Export exposes dense [PL, CL] tensors: {dense[:5]}")
    return {
        "retained_state_shapes": state_shapes,
        "topk_node_count": topk_count,
        "gather_node_count": gather_count,
        "dense_pl_ctx_tensor_count": 0,
    }


class QAICQSAOnlyRunner:
    def __init__(self, qpc_path: Path, device_ids: list[int]):
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        self.session = QAICInferenceSession(qpc_path, device_ids=device_ids)
        retained = [
            name
            for name in (*self.session.input_names, *self.session.output_names)
            if self._basename(name).startswith("qsa_")
        ]
        self.session.skip_buffers(retained)
        inputs = {self._basename(name): name for name in self.session.input_names}
        missing = set(HOST_INPUT_NAMES) - set(inputs)
        if missing:
            raise ValueError(f"QSA QPC is missing host inputs: {sorted(missing)}")
        self.input_names = {name: inputs[name] for name in HOST_INPUT_NAMES}
        self.input_dtypes = {
            name: self.session.aic_to_np_dtype_mapping[
                self.session.bindings[self.session.binding_index_map[binding]].type
            ]
            for name, binding in self.input_names.items()
        }
        self.input_shapes = {
            name: tuple(self.session.bindings[self.session.binding_index_map[binding]].dims)
            for name, binding in self.input_names.items()
        }
        self.output_name = next(
            (name for name in self.session.output_names if self._basename(name) == "attention_output"), None
        )
        if self.output_name is None:
            raise ValueError(f"QSA QPC is missing attention_output: {self.session.output_names}")
        self.output_shape = tuple(self.session.bindings[self.session.binding_index_map[self.output_name]].dims)
        output_binding = self.session.bindings[self.session.binding_index_map[self.output_name]]
        self.output_dtype = self.session.aic_to_np_dtype_mapping[output_binding.type]
        self.output_is_bfloat16 = self.session.binding_is_bfloat16(self.output_name)
        self.session.set_buffers({self.output_name: np.zeros(self.output_shape, dtype=self.output_dtype)})

    @staticmethod
    def _basename(name):
        return name.rsplit("/", 1)[-1]

    def run(self, hidden_states, position_ids):
        arrays = {"hidden_states": hidden_states, "position_ids": position_ids}
        inputs = {
            self.input_names[name]: np.asarray(arrays[name], dtype=self.input_dtypes[name]) for name in HOST_INPUT_NAMES
        }
        for name, binding in self.input_names.items():
            if tuple(inputs[binding].shape) != self.input_shapes[name]:
                raise ValueError(f"Unexpected {name} shape {inputs[binding].shape}; expected {self.input_shapes[name]}")
        outputs = self.session.run(inputs)
        value = next((tensor for name, tensor in outputs.items() if self._basename(name) == "attention_output"), None)
        if value is None:
            raise ValueError(f"QAIC run omitted attention_output: {sorted(outputs)}")
        value = np.asarray(value)
        if self.output_is_bfloat16:
            value = (value.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            value = value.astype(np.float32, copy=False)
        return value


def _parse_device_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _run_direct_latency(qpc_path, device_ids, hidden_states, position_ids):
    runner = QAICQSAOnlyRunner(qpc_path, device_ids)
    hidden_np = hidden_states.detach().cpu().numpy()
    position_np = position_ids.detach().cpu().numpy()
    for _ in range(5):
        runner.run(hidden_np, position_np)
    samples = []
    for _ in range(10):
        start = time.perf_counter()
        runner.run(hidden_np, position_np)
        samples.append((time.perf_counter() - start) * 1000)
    return runner, {
        "warmups": 5,
        "iterations": 10,
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "samples_ms": samples,
        "reference_single_qsa_layer_mean_ms": REFERENCE_LATENCY_MS,
        "at_or_below_reference": bool(np.mean(samples) <= REFERENCE_LATENCY_MS),
    }


def _write_opstats_batch(onnx_path, work_dir, hidden_states, position_ids, runner):
    import onnx
    from onnx import TensorProto

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    dtype_map = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
    }
    graph_inputs = {value.name: value for value in onnx_model.graph.input}
    graph_outputs = {value.name: value for value in onnx_model.graph.output}
    data_dir = work_dir / "io" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_arrays = {"hidden_states": hidden_states, "position_ids": position_ids}
    entries = []
    for name, value in input_arrays.items():
        elem_type = graph_inputs[name].type.tensor_type.elem_type
        dtype = runner.input_dtypes[name] if name in runner.input_dtypes else dtype_map[elem_type]
        array = np.ascontiguousarray(value.detach().cpu().numpy().astype(dtype, copy=False))
        filename = f"{name}.raw"
        array.tofile(data_dir / filename)
        entries.append(
            {
                "path": f"data/{filename}",
                "io-direction": "in",
                "elem-size": array.itemsize,
                "map-to": name,
                "dims": list(array.shape),
            }
        )
    output = graph_outputs["attention_output"]
    out_shape = [dimension.dim_value for dimension in output.type.tensor_type.shape.dim]
    out_dtype = runner.output_dtype if hasattr(runner, "output_dtype") else dtype_map[output.type.tensor_type.elem_type]
    output_array = np.zeros(out_shape, dtype=out_dtype)
    output_name = "attention_output.raw"
    output_array.tofile(data_dir / output_name)
    entries.append(
        {
            "path": f"data/{output_name}",
            "io-direction": "out",
            "elem-size": output_array.itemsize,
            "map-to": "attention_output",
            "dims": out_shape,
        }
    )
    batch_path = work_dir / "io" / "aic_batch_io.json"
    batch_path.write_text(json.dumps({"IO-files": [entries]}, indent=2) + "\n")
    return batch_path


def _run_opstats(profile_qpc, onnx_path, work_dir, hidden_states, position_ids, runner, device_list):
    if work_dir.exists():
        raise FileExistsError(f"Refusing to overwrite opstats output: {work_dir}")
    raw_dir, opstats_dir = work_dir / "raw_device_stats", work_dir / "opstats"
    raw_dir.mkdir(parents=True)
    opstats_dir.mkdir(parents=True)
    batch_path = _write_opstats_batch(onnx_path, work_dir, hidden_states, position_ids, runner)
    qpc_bin_dir = profile_qpc if (profile_qpc / "programqpc.bin").is_file() else profile_qpc / "qpc"
    run_command = [
        "/opt/qti-aic/exec/qaic-runner",
        "-d",
        device_list,
        "-t",
        str(qpc_bin_dir),
        "-n",
        str(OPSTATS_ITERATIONS),
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        str(OPSTATS_PROFILE_START_ITER),
        "--aic-profiling-num-samples",
        str(OPSTATS_PROFILE_SAMPLES),
        "--aic-profiling-out-dir",
        str(raw_dir),
        "--aic-batch-json-input",
        str(batch_path),
    ]
    print("$", " ".join(run_command), flush=True)
    with (work_dir / "run_perf.log").open("w") as log_file:
        subprocess.run(run_command, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    opstats_command = [
        "/opt/qti-aic/exec/qaic-opstats",
        "--qpc",
        str(qpc_bin_dir / "programqpc.bin"),
        "--input-dir",
        str(raw_dir),
        "--output-dir",
        str(opstats_dir),
        "--summary",
        "--trace",
    ]
    print("$", " ".join(opstats_command), flush=True)
    subprocess.run(opstats_command, check=True)

    extractor = importlib.import_module("perf.layerwise_trace_extractor.extract_qwen38_flash_full_model")
    scope_map, discovered_types = extractor.load_scope_map(onnx_path)
    traces = sorted(opstats_dir.glob("*trace.json"))
    # QAIC's trace header says "ns", but its ts/dur fields and opSyncDurUs are
    # microseconds (the Core-0 span matches runner ExecTimeUs). Keep the event
    # analysis in microseconds instead of scaling these fields by 1e-3.
    results = [extractor.analyze_trace(path, scope_map, {QSA_LAYER_INDEX}, time_unit_override="us") for path in traces]
    rows = extractor.aggregate_trace_results(results)
    if not rows:
        raise ValueError(f"No layer {QSA_LAYER_INDEX} attention timing could be mapped from {opstats_dir}")
    module_csv = work_dir / "module_timing.csv"
    extractor.write_module_csv(rows, module_csv)
    summary = extractor.parse_opstats_summaries(opstats_dir)
    runner_log = (work_dir / "run_perf.log").read_text(errors="replace")
    exec_times = [float(value) for value in re.findall(r"ExecTimeUs(?:_Prefill)?\s*[,=:]\s*([0-9.]+)", runner_log)]
    duration_match = re.search(r"TotalDuration\s+(\d+)us", runner_log)
    runner_metrics = {
        "exec_time_us_mean": exec_times[-1] if exec_times else None,
        "total_duration_us": int(duration_match.group(1)) if duration_match else None,
    }
    for report_name, metric_name in (
        ("InfPCycles_Prefill", "inference_pcycles_mean"),
        ("DDRTrafficMB_Prefill", "ddr_traffic_mb_mean"),
        ("DDRBandwidthGBPerSec_Prefill", "ddr_bandwidth_gb_s_mean"),
    ):
        match = re.search(rf"{report_name}\s*,\s*([0-9.]+)", runner_log)
        runner_metrics[metric_name] = float(match.group(1)) if match else None
    module_timing_path = work_dir / "module_timing.json"
    module_timing = {
        "layerwise_trace_rows": rows,
        "trace_count": len(traces),
        "trace_time_unit_override": "us",
        "trace_unit_note": (
            "QAIC reports displayTimeUnit=ns, while event ts/dur are in us; the trace Core-0 duration "
            "matches runner ExecTimeUs."
        ),
        "trace_mapping": {"mapped_nodes": len(scope_map), "discovered_layer_types": discovered_types},
    }
    module_timing_path.write_text(json.dumps(module_timing, indent=2) + "\n")
    return {
        "opstats_dir": str(opstats_dir),
        "runner_log": str(work_dir / "run_perf.log"),
        "trace_count": len(traces),
        "max_core_pcycles": summary["max_core_pcycles"],
        "total_ddr_mb": summary["total_ddr_mb"],
        "opstats_summary_files": len(summary["files"]),
        "ddr_mb_per_summary_sample": summary["total_ddr_mb"] / max(len(summary["files"]), 1),
        "runner_metrics": runner_metrics,
        "layerwise_trace_rows": rows,
        "layerwise_csv": str(module_csv),
        "module_timing_json": str(module_timing_path),
        "trace_mapping": module_timing["trace_mapping"],
        "profiling_note": "Opstats profiling build; keep ExecTimeUs separate from unprofiled direct latency.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            "/home/huggingface_hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("qwen3.8_flash_experiment/example_repro_20261013/qwen38_flash_4layer_checkpoint_v2"),
    )
    parser.add_argument(
        "--compile-dir",
        type=Path,
        default=Path("attention_benchmarks/qsa_only_prefill_cores4_20261103"),
    )
    parser.add_argument("--device-list", default=DEVICE_LIST)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--skip-opstats", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_HUB_CACHE"):
        raise OSError("Set HF_HUB_CACHE to the user-approved local Hugging Face cache before loading model weights")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    snapshot, checkpoint, compile_dir = args.snapshot.resolve(), args.checkpoint.resolve(), args.compile_dir.resolve()
    for path in (snapshot / "config.json", snapshot / "model.safetensors.index.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if compile_dir.exists() and any(compile_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty compile directory: {compile_dir}")
    compile_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QEFF_HOME"] = str(compile_dir / "qeff_home")
    os.environ.setdefault("QAIC_COMPILER_OPTS_UNSUPPORTED", COMPILER_OPTS_UNSUPPORTED)
    _ensure_checkpoint(snapshot, checkpoint)

    global _QEFF_PREPARE_MROPE
    from QEfficient.transformers.models.qwen4_exp.modeling_qwen4_exp import qeff_prepare_qwen4_exp_mrope

    _QEFF_PREPARE_MROPE = qeff_prepare_qwen4_exp_mrope

    config = _load_config(snapshot)
    print(f"HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}")
    print(f"QEFF_HOME={os.environ['QEFF_HOME']}")
    print(
        f"QSA-only layer={QSA_LAYER_INDEX}, PL={PREFILL_SEQ_LEN}, CL={CONTEXT_LENGTH}, "
        f"start={START_POSITION}, cores={NUM_CORES}; config={json.dumps(QSA_CONFIG, sort_keys=True)}",
        flush=True,
    )
    hf_attention, qeff_attention = _load_attention_pair(config, checkpoint)
    rotary_tables = _rotary_tables(config)
    hf_output, hf_cache, graph, qeff_outputs, inputs = _build_parity_inputs(
        config, hf_attention, qeff_attention, rotary_tables
    )
    pytorch_metrics = _compare(hf_output, qeff_outputs[0], atol=1e-4, rtol=1e-4, label="HF/QEff PyTorch")
    retained_metrics = _validate_retained_states(config, hf_cache, qeff_outputs, rotary_tables, qeff_attention)
    hf_reference = hf_output.detach().float().cpu().numpy()
    del hf_attention, qeff_attention, hf_output, hf_cache
    gc.collect()

    hidden_states, position_ids, states = inputs
    export_inputs = {
        "hidden_states": hidden_states,
        "position_ids": position_ids,
        **dict(zip(STATE_NAMES, states, strict=True)),
    }
    adapter = _QSAExportAdapter(graph)
    print("Exporting isolated QSA module in FP32", flush=True)
    onnx_path = Path(
        adapter.export(
            export_inputs,
            list(OUTPUT_NAMES),
            compile_dir / "export",
        )
    ).resolve()
    graph_metrics = _validate_graph(onnx_path, config)
    print(f"ONNX export: {onnx_path}", flush=True)

    print("Compiling with AI100 FP16 conversion, MXFP6, MXINT8 retained states, and four cores", flush=True)
    qpc_path = adapter.compile(onnx_path, compile_dir / "compile")
    profile_qpc_path = None
    if not args.skip_opstats:
        profile_qpc_path = adapter.compile(onnx_path, compile_dir / "opstats_compile", profiling=True)
    runner = QAICQSAOnlyRunner(qpc_path, _parse_device_ids(args.device_list))
    qaic_output = runner.run(hidden_states.cpu().numpy(), position_ids.cpu().numpy())
    qaic_metrics = _compare(
        torch.from_numpy(hf_reference),
        torch.from_numpy(qaic_output),
        atol=args.atol,
        rtol=args.rtol,
        label="HF/QEff PyTorch to QAIC",
    )
    direct_runner, latency = _run_direct_latency(
        qpc_path,
        _parse_device_ids(args.device_list),
        hidden_states,
        position_ids,
    )
    opstats = None
    if profile_qpc_path is not None:
        opstats = _run_opstats(
            profile_qpc_path,
            onnx_path,
            compile_dir / "opstats_run",
            hidden_states,
            position_ids,
            direct_runner,
            args.device_list,
        )

    report = {
        "model": "Qwen3.8-Flash-Next QSA only (layer 3)",
        "snapshot": str(snapshot),
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "qpc": str(qpc_path),
        "profile_qpc": str(profile_qpc_path) if profile_qpc_path else None,
        "workload": {
            "batch_size": BATCH_SIZE,
            "prefill_seq_len": PREFILL_SEQ_LEN,
            "ctx_len": CONTEXT_LENGTH,
            "start_position": START_POSITION,
            "num_cores": NUM_CORES,
            "device_ids": _parse_device_ids(args.device_list),
            "qsa_config": QSA_CONFIG,
            "prefix_cache": "zero initialized, matching the standalone handoff benchmark",
            "model_dtype_during_load_and_export": "float32",
            "compiler_conversion": "-convert-to-fp16",
        },
        "onnx_graph_audit": graph_metrics,
        "hf_qeff_pytorch": pytorch_metrics,
        "hf_qeff_retained_states": retained_metrics,
        "hf_qaic_attention_output": qaic_metrics,
        "direct_qaic_latency": latency,
        "opstats": opstats,
    }
    report_path = compile_dir / "prefill_qsa_only_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")
    if not latency["at_or_below_reference"]:
        print(
            f"WARNING: QSA-only mean latency {latency['mean_ms']:.2f} ms exceeds "
            f"the 4-core reference {REFERENCE_LATENCY_MS:.2f} ms.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
