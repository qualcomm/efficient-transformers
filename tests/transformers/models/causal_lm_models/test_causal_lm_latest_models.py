# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MiniMax M3 AIC parity coverage for the latest causal-model test suite."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.text_generation.minimax_m3_paged_dp_decode_only_configurable_pages import (
    _build_qaic_config,
    build_block_table,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLConfig,
    MiniMaxM3VLTextConfig,
    MiniMaxM3VLVisionConfig,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from tests.unit_test.models.test_model_quickcheck import _ort_session


def _tiny_minimax_m3_text_config(dtype=torch.float32, microbench=False) -> MiniMaxM3VLTextConfig:
    config = MiniMaxM3VLTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=16, dense_intermediate_size=64,
        shared_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=32, num_local_experts=4,
        num_experts_per_tok=2, routed_scaling_factor=1.0,
        layer_types=["full_attention", "minimax_m3_sparse"],
        mlp_layer_types=["dense", "sparse"], index_n_heads=2, index_head_dim=8,
        index_block_size=4, index_topk_blocks=2, index_local_blocks=1,
    )
    if microbench:
        config.num_hidden_layers = 1
        config.layer_types = ["minimax_m3_sparse"]
        config.mlp_layer_types = ["dense"]
        config.head_dim = 32
        config.max_position_embeddings = 8
        config.index_head_dim = 16
        config.index_block_size = 8
        config.index_topk_blocks = 1
        config.rope_parameters = {
            "rope_type": "default", "rope_theta": 5_000_000.0, "partial_rotary_factor": 0.5
        }
    config.torch_dtype = dtype
    return config


def _tiny_minimax_m3_vlm_config(microbench=False) -> MiniMaxM3VLConfig:
    text_config = _tiny_minimax_m3_text_config(microbench=microbench)
    vision_config = MiniMaxM3VLVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        image_size=28, patch_size=14, temporal_patch_size=1, spatial_merge_size=1, num_channels=3,
    )
    return MiniMaxM3VLConfig(
        text_config=text_config, vision_config=vision_config, image_token_index=4,
        projector_hidden_size=text_config.hidden_size,
    )


def _tiny_minimax_m3_paged_dp_cp_config():
    args = SimpleNamespace(
        num_kv_blocks=1, attn_dp=1, attn_cp=1, indexer_num_blocks=None, msa_num_kv_blocks=None,
        skip_kv=False, non_dynamic_cache=False, msa_indexer_dp=1, msa_indexer_cp=2,
        msa_attn_dp=2, msa_attn_cp=2, indexer_n_head=1, num_cores=1, gqa_num_pages=1,
        msa_indexer_num_logical_pages=2, msa_attn_num_logical_pages=2, gqa_physical_pages=8,
        msa_indexer_physical_pages=8, msa_attn_physical_pages=8, expert_parallel_chunk_size=4,
        cores_per_expert=1, tree_reduce=True, static_export_dims=False,
    )
    qaic_config = _build_qaic_config(args, 8, 4, 4)
    for key in ('moe_config', 'indexer_num_blocks', 'msa_num_kv_blocks', 'skip_kv', 'non_dynamic_cache'):
        qaic_config.pop(key)
    return qaic_config


def _tiny_minimax_m3_paged_decode_inputs(qeff_model):
    inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True, ctx_len=8, past_seq_len=8, prefill_seq_len=1, batch_size=2
    )["lang"]
    inputs["input_ids"] = torch.tensor([[3], [5]], dtype=torch.int64)
    inputs["position_ids"] = torch.tensor([[7], [7]], dtype=torch.int64)
    table_generator = torch.Generator(device="cpu").manual_seed(11)
    inputs['gqa_block_table'] = build_block_table(1, 1, 2, 1, table_generator, 2)
    inputs['msa_indexer_block_table'] = build_block_table(1, 2, 2, 2, table_generator, 2)
    inputs['msa_attn_block_table'] = build_block_table(2, 2, 2, 2, table_generator, 2)
    return inputs


def _flatten_minimax_paged_ort_inputs(session, inputs):
    ort_inputs = {}
    input_names = set(session.input_names) if hasattr(session, "input_names") else {
        item.name for item in session.get_inputs()
    }
    for name in input_names:
        if name in inputs and torch.is_tensor(inputs[name]):
            ort_inputs[name] = inputs[name].detach().numpy()
        elif name.startswith(("past_key.", "past_value.")):
            layer_idx = int(name.split('.')[1])
            cache_idx = 0 if name.startswith('past_key.') else 1
            ort_inputs[name] = inputs['past_key_values'][layer_idx][cache_idx].detach().numpy()
        elif name.startswith('index_key.'):
            ort_inputs[name] = inputs['index_keys'][0].detach().numpy()
        elif name == "index_keys":
            ort_inputs[name] = inputs['index_keys'][0].detach().numpy()
        else:
            raise AssertionError(f"Unhandled MiniMax ORT input: {name}")
    return ort_inputs

@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.accuracy
def test_minimax_m3_paged_dp_cp_qeff_pytorch_vs_aic(tmp_path):
    torch.manual_seed(7)
    qaic_config = _tiny_minimax_m3_paged_dp_cp_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(_tiny_minimax_m3_vlm_config()).eval()
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    qeff_model.model.qaic_config = qaic_config
    qeff_model.lang_model.model.qaic_config = qaic_config

    qeff_model.transform(
        ctx_len=8,
        seq_len=1,
        bs=2,
        num_devices=1,
        num_cores=1,
        qaic_config=qaic_config,
        prefill_seq_len=1,
    )
    inputs = _tiny_minimax_m3_paged_decode_inputs(qeff_model)
    with torch.no_grad():
        qeff_logits = qeff_model.lang_model.model(**inputs)[0].detach().float().numpy()

    qpc_path = qeff_model.compile(
        compile_dir=tmp_path / "minimax-m3-paged-dp-cp-aic",
        prefill_seq_len=1,
        ctx_len=8,
        batch_size=2,
        num_cores=1,
        num_devices=1,
        skip_vision=True,
        node_precision_info=True,
        use_onnx_subfunctions=False,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
        retain_full_kv=True,
        split_model_io=True,
        qaic_config=qaic_config,
        offload_pt_weights=False,
    )["lang_decode_qpc_path"]
    session = QAICInferenceSession(qpc_path)
    try:

        def binding_value(input_name, source=None):
            binding = session.bindings[session.binding_index_map[input_name]]
            shape = tuple(binding.dims)
            dtype = session.aic_to_np_dtype_mapping[binding.type]
            if source is None:
                return np.zeros(shape, dtype=dtype)
            source = np.asarray(source, dtype=dtype)
            if source.ndim == len(shape) and source.shape[:-1] == shape[:-1] and source.shape[-1] < shape[-1]:
                padded = np.zeros(shape, dtype=dtype)
                padded[..., : source.shape[-1]] = source
                return padded
            assert source.size == np.prod(shape), (
                f"AIC binding shape mismatch for {input_name}: source {source.shape}, expected {shape}"
            )
            return source.reshape(shape)

        source_inputs = _flatten_minimax_paged_ort_inputs(session, inputs)
        aic_inputs = {name: binding_value(name, source_inputs.get(name)) for name in session.input_names}
        aic_outputs = session.run(aic_inputs)
    finally:
        session.deactivate()

    np.testing.assert_allclose(aic_outputs["logits"], qeff_logits, atol=1e-3, rtol=1e-3)


@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_minimax_m3_decode_qeff_pytorch_vs_aic(tmp_path):
    torch.manual_seed(7)
    vlm_config = _tiny_minimax_m3_vlm_config()
    text_config = vlm_config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)

    input_ids = torch.tensor([[3]], dtype=torch.int64)
    position_ids = torch.tensor([[4]], dtype=torch.int64)
    runtime_cache_len = 8
    past_key_values = tuple(
        (
            torch.zeros((1, text_config.num_key_value_heads, runtime_cache_len, text_config.head_dim)),
            torch.zeros((1, text_config.num_key_value_heads, runtime_cache_len, text_config.head_dim)),
        )
        for _ in range(text_config.num_hidden_layers)
    )
    index_keys = [
        torch.zeros((1, 1, runtime_cache_len, text_config.index_head_dim))
        for layer_type in text_config.layer_types
        if layer_type == "minimax_m3_sparse"
    ]
    dummy_vision_embeds = torch.zeros((1, 1, text_config.hidden_size), dtype=torch.float32)
    dummy_image_idx = torch.zeros((1, 1), dtype=torch.int64)

    with torch.no_grad():
        qeff_outputs = qeff_model.lang_model.model(
            input_ids=input_ids,
            vision_embeds=dummy_vision_embeds,
            position_ids=position_ids,
            image_idx=dummy_image_idx,
            past_key_values=past_key_values,
            index_keys=index_keys,
        )
    qeff_logits = qeff_outputs[0].detach().cpu().numpy()

    qpc_paths = qeff_model.compile(
        compile_dir=tmp_path / "minimax-m3-aic",
        prefill_seq_len=1,
        ctx_len=runtime_cache_len,
        batch_size=1,
        num_cores=1,
        num_devices=1,
        skip_vision=True,
        node_precision_info=True,
        qaic_config={},
        offload_pt_weights=False,
    )
    session = QAICInferenceSession(qpc_paths["lang_decode_qpc_path"])
    try:

        def binding_shape_dtype(input_name):
            binding = session.bindings[session.binding_index_map[input_name]]
            shape = tuple(binding.dims)
            assert all(dim >= 0 for dim in shape), f"Dynamic AIC binding shape for {input_name}: {shape}"
            return shape, session.aic_to_np_dtype_mapping[binding.type]

        def binding_zeros(input_name):
            shape, dtype = binding_shape_dtype(input_name)
            return np.zeros(shape, dtype=dtype)

        def binding_copy(input_name, source):
            shape, dtype = binding_shape_dtype(input_name)
            source = np.asarray(source, dtype=dtype)
            assert source.size == np.prod(shape), (
                f"AIC binding shape mismatch for {input_name}: source {source.shape}, expected {shape}"
            )
            return source.reshape(shape)

        aic_inputs = {}
        for input_name in session.input_names:
            if input_name == "input_ids":
                aic_inputs[input_name] = binding_copy(input_name, input_ids.numpy())
            elif input_name == "position_ids":
                aic_inputs[input_name] = binding_copy(input_name, position_ids.numpy())
            elif input_name == "vision_embeds":
                aic_inputs[input_name] = binding_zeros(input_name)
            elif input_name == "image_idx":
                aic_inputs[input_name] = binding_copy(input_name, dummy_image_idx.numpy())
            elif input_name.startswith(("past_key.", "past_value.")):
                aic_inputs[input_name] = binding_zeros(input_name)
            elif input_name.startswith("index_key."):
                aic_inputs[input_name] = binding_zeros(input_name)
            elif input_name == "index_keys":
                aic_inputs[input_name] = binding_zeros(input_name)
            else:
                raise AssertionError(f"Unhandled Minimax AIC input: {input_name}")
        aic_outputs = session.run(aic_inputs)
    finally:
        session.deactivate()

    np.testing.assert_allclose(aic_outputs["logits"], qeff_logits, atol=1e-2, rtol=1e-2)
@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_minimax_m3_expert_parallel_prefill_hf_pytorch_vs_aic(tmp_path):
    """Check MiniMax expert-parallel prefill logits against the original HF model."""
    torch.manual_seed(7)
    vlm_config = _tiny_minimax_m3_vlm_config()
    text_config = vlm_config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    model_hf_orig = deepcopy(model_hf)
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)

    prefill_len, cache_len = 4, 8
    input_ids = torch.tensor([[3, 5, 7, 9]], dtype=torch.int64)
    position_ids = torch.arange(prefill_len, dtype=torch.int64).view(1, -1)
    with torch.no_grad():
        hf_prefill = model_hf_orig.model.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=True,
        )
        hf_logits = model_hf_orig.lm_head(hf_prefill.last_hidden_state[:, -1:, :]).float().cpu().numpy()

    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": 2,
        "msa_indexer_dp": 1,
        "msa_indexer_cp": 1,
        "msa_attn_dp": 1,
        "indexer_n_head": 1,
        "num_cores_per_device": 2,
        "msa_q_chunk": 2,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": prefill_len,
        },
    }
    qpc_path = qeff_model.compile(
        compile_dir=tmp_path / "minimax-m3-expert-parallel-prefill-aic",
        prefill_seq_len=prefill_len,
        ctx_len=cache_len,
        batch_size=1,
        num_cores=4,
        num_devices=1,
        skip_vision=True,
        node_precision_info=True,
        prefill_only=True,
        enable_chunking=True,
        qaic_config=qaic_config,
        offload_pt_weights=False,
    )["lang_prefill_qpc_path"]
    assert qeff_model.lang_model.hash_params["moe_prefill_flavour"] == "expert_parallel"

    session = QAICInferenceSession(qpc_path)
    try:

        def binding_value(input_name, source=None):
            binding = session.bindings[session.binding_index_map[input_name]]
            shape = tuple(binding.dims)
            dtype = session.aic_to_np_dtype_mapping[binding.type]
            if source is None:
                return np.zeros(shape, dtype=dtype)
            source = np.asarray(source, dtype=dtype)
            assert source.size == np.prod(shape), (
                f"AIC binding shape mismatch for {input_name}: source {source.shape}, expected {shape}"
            )
            return source.reshape(shape)

        aic_inputs = {}
        for input_name in session.input_names:
            if input_name == "input_ids":
                aic_inputs[input_name] = binding_value(input_name, input_ids.numpy())
            elif input_name == "position_ids":
                aic_inputs[input_name] = binding_value(input_name, position_ids.numpy())
            elif input_name == "image_idx":
                aic_inputs[input_name] = binding_value(input_name, np.zeros((1, 1), dtype=np.int64))
            elif input_name == "vision_embeds" or input_name.startswith(("past_key.", "past_value.", "index_key.")):
                aic_inputs[input_name] = binding_value(input_name)
            else:
                raise AssertionError(f"Unhandled MiniMax AIC input: {input_name}")
        aic_outputs = session.run(aic_inputs)
    finally:
        session.deactivate()

    np.testing.assert_allclose(aic_outputs["logits"], hf_logits, atol=1e-2, rtol=1e-2)


@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_minimax_m3_prefill_decode_hf_pytorch_vs_aic(tmp_path):
    """Check AIC parity against the original HF MiniMax prefill-to-decode handoff."""
    torch.manual_seed(7)
    config = _tiny_minimax_m3_vlm_config()
    text_config = config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(config).eval()
    model_hf_orig = deepcopy(model_hf)
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)
    prefill_len, cache_len = 4, 8
    input_ids = torch.tensor([[3, 5, 7, 9]], dtype=torch.int64)
    prefill_positions = torch.arange(prefill_len, dtype=torch.int64).view(1, -1)
    decode_positions = torch.tensor([[prefill_len]], dtype=torch.int64)
    vision = torch.zeros((1, 1, text_config.hidden_size))
    image_idx = torch.zeros((1, 1), dtype=torch.int64)
    with torch.no_grad():
        hf_prefill = model_hf_orig.model.language_model(
            input_ids=input_ids,
            position_ids=prefill_positions,
            use_cache=True,
        )
        pt_prefill_logits = model_hf_orig.lm_head(hf_prefill.last_hidden_state[:, -1:, :]).float()
        next_ids = pt_prefill_logits.argmax(dim=-1)
        hf_decode = model_hf_orig.model.language_model(
            input_ids=next_ids,
            position_ids=decode_positions,
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        )
        pt_decode_logits = model_hf_orig.lm_head(hf_decode.last_hidden_state[:, -1:, :]).float()

    def run_stage(qpc_path, stage_input_ids, stage_positions, retained=None):
        session = QAICInferenceSession(qpc_path)
        try:

            def adapt(name, source=None):
                binding = session.bindings[session.binding_index_map[name]]
                shape = tuple(binding.dims)
                dtype = session.aic_to_np_dtype_mapping[binding.type]
                if source is None:
                    return np.zeros(shape, dtype=dtype)
                source = np.asarray(source, dtype=dtype)
                assert source.size == np.prod(shape), (
                    f"AIC binding shape mismatch for {name}: {source.shape}, expected {shape}"
                )
                return source.reshape(shape)

            inputs = {}
            for name in session.input_names:
                if name == "input_ids":
                    inputs[name] = adapt(name, stage_input_ids.numpy())
                elif name == "position_ids":
                    inputs[name] = adapt(name, stage_positions.numpy())
                elif name == "vision_embeds":
                    inputs[name] = adapt(name, vision.expand(1, stage_positions.shape[1], -1).numpy())
                elif name == "image_idx":
                    inputs[name] = adapt(name, image_idx.numpy())
                elif name.startswith(("past_key.", "past_value.", "index_key.")):
                    inputs[name] = adapt(name, retained.get(name) if retained else None)
                else:
                    raise AssertionError(f"Unhandled MiniMax AIC input: {name}")
            return session.run(inputs)
        finally:
            session.deactivate()

    def run_ort_stage(onnx_path, stage_input_ids, stage_positions, retained=None):
        session = _ort_session(Path(onnx_path))
        input_shapes = {meta.name: tuple(meta.shape) for meta in session.get_inputs()}
        input_types = {
            meta.name: np.int64 if meta.type == "tensor(int64)" else np.float32 for meta in session.get_inputs()
        }

        def adapt(name, source=None):
            shape = input_shapes[name]
            dtype = input_types[name]
            if source is None:
                assert all(isinstance(dim, int) for dim in shape), (
                    f"ORT input {name} has unresolved shape {shape} without a source tensor"
                )
                return np.zeros(shape, dtype=dtype)
            source = np.asarray(source, dtype=dtype)
            for expected, actual in zip(shape, source.shape):
                if isinstance(expected, int):
                    assert expected == actual, (
                        f"ORT binding shape mismatch for {name}: {source.shape}, expected {shape}"
                    )
            return source

        inputs = {}
        for name in input_shapes:
            if name == "input_ids":
                inputs[name] = adapt(name, stage_input_ids.numpy())
            elif name == "position_ids":
                inputs[name] = adapt(name, stage_positions.numpy())
            elif name == "vision_embeds":
                inputs[name] = adapt(name, vision.expand(1, stage_positions.shape[1], -1).numpy())
            elif name == "image_idx":
                inputs[name] = adapt(name, image_idx.numpy())
            elif name.startswith(("past_key.", "past_value.")):
                source = (
                    retained.get(name)
                    if retained
                    else np.zeros((1, text_config.num_key_value_heads, cache_len, text_config.head_dim))
                )
                inputs[name] = adapt(name, source)
            elif name.startswith("index_key."):
                source = retained.get(name) if retained else np.zeros((1, 1, cache_len, text_config.index_head_dim))
                inputs[name] = adapt(name, source)
            else:
                raise AssertionError(f"Unhandled MiniMax ORT input: {name}")
        return dict(zip((output.name for output in session.get_outputs()), session.run(None, inputs)))

    compile_args = dict(
        ctx_len=cache_len,
        batch_size=1,
        num_cores=4,
        num_devices=1,
        skip_vision=True,
        node_precision_info=True,
        qaic_config={
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": 2,
            "msa_indexer_dp": 1,
            "msa_indexer_cp": 1,
            "msa_attn_dp": 1,
            "indexer_n_head": 1,
            "num_cores_per_device": 2,
            "msa_q_chunk": 2,
        },
        offload_pt_weights=False,
    )

    prefill_qpc = qeff_model.compile(
        compile_dir=tmp_path / "minimax-prefill-aic",
        prefill_seq_len=prefill_len,
        prefill_only=True,
        enable_chunking=True,
        **compile_args,
    )["lang_prefill_qpc_path"]
    prefill_onnx = Path(qeff_model.lang_model.onnx_path)
    decode_qpc = qeff_model.compile(
        compile_dir=tmp_path / "minimax-decode-aic", prefill_seq_len=1, prefill_only=False, **compile_args
    )["lang_decode_qpc_path"]
    decode_onnx = Path(qeff_model.lang_model.onnx_path)

    ort_prefill = run_ort_stage(prefill_onnx, input_ids, prefill_positions)
    np.testing.assert_allclose(ort_prefill["logits"], pt_prefill_logits.numpy(), atol=1e-4, rtol=1e-4)
    ort_retained = {
        name.removesuffix("_RetainedState"): output
        for name, output in ort_prefill.items()
        if name.endswith("_RetainedState")
    }
    ort_decode = run_ort_stage(decode_onnx, ort_prefill["logits"].argmax(axis=-1), decode_positions, ort_retained)
    np.testing.assert_allclose(ort_decode["logits"], pt_decode_logits.numpy(), atol=1e-4, rtol=1e-4)

    aic_prefill = run_stage(prefill_qpc, input_ids, prefill_positions)
    np.testing.assert_allclose(aic_prefill["logits"], pt_prefill_logits.numpy(), atol=1e-2, rtol=1e-2)
    retained = {
        name.removesuffix("_RetainedState"): output
        for name, output in aic_prefill.items()
        if name.endswith("_RetainedState")
    }
    aic_decode = run_stage(decode_qpc, aic_prefill["logits"].argmax(axis=-1), decode_positions, retained)
    np.testing.assert_allclose(aic_decode["logits"], pt_decode_logits.numpy(), atol=1e-2, rtol=1e-2)
