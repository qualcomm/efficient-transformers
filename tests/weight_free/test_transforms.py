# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Unit tests for the dynamo-specific transforms used by the weight-free export pipeline.

These are the same transforms tested in tests/dynamo/test_transforms.py.
They are duplicated here because PruneFakeInitializersTransform is specifically
relevant to weight-free export (it removes meta/fake tensor initializers that only
appear when the model was traced on meta device).

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import onnx_ir as ir
import pytest
import torch
from onnx import TensorProto, helper
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.base.checkpoint_transforms import (
    CHECKPOINT_PREPARED_MANIFEST,
    CHECKPOINT_PREPARED_SENTINEL,
    CheckpointTransformPipeline,
)
from QEfficient.base.onnx_transforms import (
    PreserveNestedCacheRetainedStateTransform,
    PruneFakeInitializersTransform,
    RenameRepeatedSubgraphTransform,
)
from QEfficient.exporter.weight_free import checkpoint_key_resolver
from QEfficient.exporter.weight_free.checkpoint_key_resolver import find_checkpoint_key
from QEfficient.exporter.weight_free.checkpoint_transforms import (
    DtypeConversionCheckpointTransform,
    GlmMoeDsaReducedCheckpointTransform,
    GraniteMoeFusedExpertSplitCheckpointTransform,
    MoEExpertStackingCheckpointTransform,
    MoEFusedExpertSplitCheckpointTransform,
)
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import runtime_requirements
from QEfficient.utils.export_utils import _generate_export_hash, convert_dynamic_axes_to_dynamic_shapes
from QEfficient.utils.runtime_requirements import validate_runtime_requirements
from QEfficient.utils.torch_patches import temporarily_enable_nested_compile_regions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model, cfg


def _make_minimal_onnx_with_repeated_subgraphs(num_layers: int = 2, scatter_count_per_fn: int = 2):
    """
    Build a minimal ONNX ModelProto that mimics dynamo's repeated-subgraph output:
      - graph has num_layers call nodes (one per layer), each referencing repeated_subgraphN
      - each function contains scatter_count_per_fn CtxScatter nodes
      - graph outputs include past_key/value _RetainedState placeholders (dangling)
    """
    functions = []
    call_nodes = []
    graph_outputs = []
    graph_inputs = []

    for i in range(num_layers):
        fn_name = f"repeated_subgraph{i}"

        scatter_nodes = []
        fn_outputs = []
        for j in range(scatter_count_per_fn):
            kind = "key" if j == 0 else "value"
            scatter_out = f"scatter_{kind}_{i}"
            scatter_node = helper.make_node(
                "CtxScatter",
                inputs=[f"past_{kind}.{i}", f"new_{kind}_{i}", "position_ids"],
                outputs=[scatter_out],
                domain="qti.aisw",
            )
            scatter_nodes.append(scatter_node)
            fn_outputs.append(scatter_out)

        fn = helper.make_function(
            domain="",
            fname=fn_name,
            inputs=[f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"],
            outputs=fn_outputs,
            nodes=scatter_nodes,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("qti.aisw", 1)],
        )
        functions.append(fn)

        retained_key = f"past_key.{i}_RetainedState"
        retained_val = f"past_value.{i}_RetainedState"
        call_node = helper.make_node(
            fn_name,
            inputs=[f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"],
            outputs=[],
            domain="",
        )
        call_nodes.append(call_node)

        graph_outputs.append(helper.make_tensor_value_info(retained_key, TensorProto.FLOAT, None))
        graph_outputs.append(helper.make_tensor_value_info(retained_val, TensorProto.FLOAT, None))

        graph_inputs.append(helper.make_tensor_value_info(f"past_key.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"past_value.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"hidden_{i}", TensorProto.FLOAT, None))

    graph_inputs.append(helper.make_tensor_value_info("position_ids", TensorProto.INT64, None))

    graph = helper.make_graph(call_nodes, "test_graph", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for fn in functions:
        model.functions.append(fn)
    return model


def _write_safetensors_checkpoint(root, tensors):
    shard_name = "model.safetensors"
    save_file({key: tensor.contiguous() for key, tensor in tensors.items()}, str(root / shard_name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: shard_name for key in tensors}}, indent=2)
    )


def _load_prepared_tensors(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    loaded = {}
    for key, shard_name in index.items():
        with safe_open(str(root / shard_name), framework="pt") as handle:
            loaded[key] = handle.get_tensor(key)
    return loaded


# ---------------------------------------------------------------------------
# Test checkpoint layout transforms
# ---------------------------------------------------------------------------


class TestWeightFreeCheckpointTransforms:
    def test_reduced_glm_transform_requires_fp8_metadata_when_quant_config_is_cleared(self):
        weight_map = {
            "model.layers.0.self_attn.q_b_proj.weight": "model.safetensors",
            "model.layers.0.self_attn.q_b_proj.weight_scale_inv": "model.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
            "model.layers.0.mlp.up_proj.weight": "model.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model.safetensors",
            "model.layers.1.mlp.experts.0.gate_proj.weight": "model.safetensors",
        }
        config_without_source_metadata = SimpleNamespace(model_type="glm_moe_dsa", quantization_config=None)
        config_with_source_metadata = SimpleNamespace(
            model_type="glm_moe_dsa",
            quantization_config=None,
            _glm_moe_dsa_weight_block_size=(128, 64),
        )

        assert not GlmMoeDsaReducedCheckpointTransform.is_applicable(
            weight_map, model_config=config_without_source_metadata
        )
        assert GlmMoeDsaReducedCheckpointTransform.is_applicable(
            weight_map, model_config=config_with_source_metadata
        )

    def test_weight_free_compile_default_is_only_glm_multitoken_prefill(self, monkeypatch):
        from QEfficient.transformers.models.modeling_auto import (
            _allow_weight_free_prefill_compile_by_default,
        )

        glm_config = SimpleNamespace(model_type="glm_moe_dsa", architectures=None)
        other_config = SimpleNamespace(model_type="llama", architectures=None)
        monkeypatch.delenv("QEFF_ALLOW_WEIGHT_FREE_PREFILL_COMPILE", raising=False)

        assert _allow_weight_free_prefill_compile_by_default(glm_config, prefill_only=True, prefill_seq_len=128)
        assert not _allow_weight_free_prefill_compile_by_default(
            glm_config, prefill_only=False, prefill_seq_len=128
        )
        assert not _allow_weight_free_prefill_compile_by_default(glm_config, prefill_only=True, prefill_seq_len=1)
        assert not _allow_weight_free_prefill_compile_by_default(
            other_config, prefill_only=True, prefill_seq_len=128
        )

    def test_prepares_reduced_glm_moe_dsa_fp8_checkpoint(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        q_b_proj = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        q_b_proj_scale = torch.full((4, 1), 2.0)
        expert_gate = [torch.arange(8, dtype=torch.float32).reshape(2, 4) + offset for offset in (0, 10)]
        expert_up = [torch.arange(8, dtype=torch.float32).reshape(2, 4) + offset for offset in (20, 30)]
        expert_down = [torch.arange(8, dtype=torch.float32).reshape(4, 2) + offset for offset in (40, 50)]
        _write_safetensors_checkpoint(
            src,
            {
                "model.embed_tokens.weight": torch.ones(2, 2),
                "model.layers.0.self_attn.q_b_proj.weight": q_b_proj,
                "model.layers.0.self_attn.q_b_proj.weight_scale_inv": q_b_proj_scale,
                "model.layers.0.mlp.gate_proj.weight": torch.ones(2, 2),
                "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
                "model.layers.0.mlp.down_proj.weight": torch.ones(2, 2),
                "model.layers.0.input_layernorm.weight": torch.ones(2),
                "model.layers.0.self_attn.indexer.wk.weight": torch.full((2, 2), 5.0),
                "model.layers.0.self_attn.indexer.wq_b.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv": torch.full((2, 2), 6.0),
                **{
                    f"model.layers.3.mlp.experts.{expert_idx}.{kind}_proj.weight": tensor
                    for expert_idx in range(2)
                    for kind, tensor in (
                        ("gate", expert_gate[expert_idx]),
                        ("up", expert_up[expert_idx]),
                        ("down", expert_down[expert_idx]),
                    )
                },
                **{
                    f"model.layers.3.mlp.experts.{expert_idx}.{kind}_proj.weight_scale_inv": torch.full(
                        (1, 2) if kind != "down" else (2, 1), scale
                    )
                    for expert_idx in range(2)
                    for kind, scale in (("gate", 3.0), ("up", 4.0), ("down", 5.0))
                },
                "model.layers.3.mlp.gate.weight": torch.full((2, 2), 6.0),
                "model.layers.4.self_attn.q_b_proj.weight": torch.full((8, 2), 9.0),
                "model.mtp.layers.0.weight": torch.full((2, 2), 10.0),
                "model.norm.weight": torch.ones(2),
                "lm_head.weight": torch.ones(2, 2),
            },
        )
        config = SimpleNamespace(
            model_type="glm_moe_dsa",
            num_hidden_layers=4,
            num_attention_heads=2,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            n_routed_experts=2,
            quantization_config={"quant_method": "fp8", "weight_block_size": [2, 2]},
        )

        assert GlmMoeDsaReducedCheckpointTransform.is_applicable(
            json.loads((src / "model.safetensors.index.json").read_text())["weight_map"],
            model_config=config,
        )
        assert GlmMoeDsaReducedCheckpointTransform.apply(src, out, model_config=config)

        tensors = _load_prepared_tensors(out)
        expected_q_b_proj = q_b_proj * 2
        expected_q_up = expected_q_b_proj.T.view(-1, 2, 4)[..., :2].reshape(-1, 4).unsqueeze(0)
        expected_q_rope = expected_q_b_proj.T.view(-1, 2, 4)[..., 2:].reshape(-1, 4).unsqueeze(0)
        torch.testing.assert_close(tensors["model.layers.0.self_attn.q_up"], expected_q_up)
        torch.testing.assert_close(tensors["model.layers.0.self_attn.q_rope"], expected_q_rope)
        torch.testing.assert_close(
            tensors["model.layers.3.mlp.moe_weights.gate"], torch.stack(expert_gate).transpose(1, 2) * 3
        )
        torch.testing.assert_close(
            tensors["model.layers.3.mlp.moe_weights.up"], torch.stack(expert_up).transpose(1, 2) * 4
        )
        torch.testing.assert_close(
            tensors["model.layers.3.mlp.moe_weights.down"], torch.stack(expert_down).transpose(1, 2) * 5
        )
        torch.testing.assert_close(tensors["model.layers.0.self_attn.indexer.wk.weight"], torch.full((2, 2), 5.0))
        torch.testing.assert_close(
            tensors["model.layers.0.self_attn.indexer.wq_b.weight"],
            torch.arange(12, dtype=torch.float32).reshape(3, 4) * 6,
        )
        torch.testing.assert_close(tensors["model.layers.3.mlp.gate.weight"], torch.full((2, 2), 6.0))
        assert not any("layers.4" in key or ".mtp." in key or key.endswith("_scale_inv") for key in tensors)

    def test_prepares_reduced_glm_moe_dsa_checkpoint_in_expert_parallel_layout(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        tensors = {
            "model.layers.0.self_attn.q_b_proj.weight": torch.ones(8, 2),
            "model.layers.0.self_attn.q_b_proj.weight_scale_inv": torch.ones(4, 1),
            "model.layers.0.mlp.gate_proj.weight": torch.ones(2, 2),
            "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
            "model.layers.0.mlp.down_proj.weight": torch.ones(2, 2),
        }
        source_experts = {"gate": [], "up": [], "down": []}
        for expert_idx in range(4):
            for kind, shape in (("gate", (2, 4)), ("up", (2, 4)), ("down", (4, 2))):
                tensor = torch.arange(8, dtype=torch.float32).reshape(shape) + 10 * expert_idx
                source_experts[kind].append(tensor)
                tensors[f"model.layers.3.mlp.experts.{expert_idx}.{kind}_proj.weight"] = tensor
                scale_shape = (1, 2) if kind != "down" else (2, 1)
                tensors[f"model.layers.3.mlp.experts.{expert_idx}.{kind}_proj.weight_scale_inv"] = torch.ones(
                    scale_shape
                )
        _write_safetensors_checkpoint(src, tensors)

        config = SimpleNamespace(
            model_type="glm_moe_dsa",
            num_hidden_layers=4,
            num_attention_heads=2,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            n_routed_experts=4,
            mlp_layer_types=["dense", "dense", "dense", "sparse"],
            quantization_config={"quant_method": "fp8", "weight_block_size": [2, 2]},
        )
        expert_parallel_layout = {
            "flavour": "expert_parallel",
            "layout_version": 1,
            "num_pipeline_stages": 2,
            "num_parallelized_experts": 2,
        }

        assert GlmMoeDsaReducedCheckpointTransform.apply(
            src,
            out,
            model_config=config,
            expert_parallel_layout=expert_parallel_layout,
        )

        prepared = _load_prepared_tensors(out)
        for kind, canonical in (
            (
                "gate",
                torch.stack(source_experts["gate"]).transpose(1, 2),
            ),
            (
                "up",
                torch.stack(source_experts["up"]).transpose(1, 2),
            ),
            (
                "down",
                torch.stack(source_experts["down"]).transpose(1, 2),
            ),
        ):
            expected = canonical.view(2, 2, *canonical.shape[1:]).transpose(0, 1).contiguous()
            actual = prepared[f"model.layers.3.mlp.moe_weights.{kind}"]
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected)

    def test_glm_moe_dsa_checkpoint_requires_positive_layer_count(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        _write_safetensors_checkpoint(
            src,
            {
                "model.layers.0.self_attn.q_b_proj.weight": torch.ones(8, 2),
                "model.layers.0.self_attn.q_b_proj.weight_scale_inv": torch.ones(4, 1),
                "model.layers.0.mlp.gate_proj.weight": torch.ones(2, 2),
                "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
                "model.layers.0.mlp.down_proj.weight": torch.ones(2, 2),
                "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones(2, 2),
            },
        )

        with pytest.raises(ValueError, match="positive num_hidden_layers"):
            GlmMoeDsaReducedCheckpointTransform.apply(
                src,
                tmp_path / "out",
                model_config=SimpleNamespace(
                    model_type="glm_moe_dsa",
                    num_hidden_layers=0,
                    quantization_config={"quant_method": "fp8", "weight_block_size": [2, 2]},
                ),
            )

    def test_glm_moe_dsa_checkpoint_stacks_all_active_sparse_layers(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        tensors = {
            "model.layers.0.self_attn.q_b_proj.weight": torch.ones(8, 2),
            "model.layers.0.self_attn.q_b_proj.weight_scale_inv": torch.ones(4, 1),
            "model.layers.0.mlp.gate_proj.weight": torch.ones(2, 2),
            "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
            "model.layers.0.mlp.down_proj.weight": torch.ones(2, 2),
        }
        for layer_idx in (3, 4):
            for kind in ("gate", "up", "down"):
                tensors[f"model.layers.{layer_idx}.mlp.experts.0.{kind}_proj.weight"] = torch.full((2, 2), layer_idx)
                tensors[f"model.layers.{layer_idx}.mlp.experts.0.{kind}_proj.weight_scale_inv"] = torch.ones(1, 1)
        _write_safetensors_checkpoint(src, tensors)

        config = SimpleNamespace(
            model_type="glm_moe_dsa",
            num_hidden_layers=5,
            num_attention_heads=2,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            n_routed_experts=1,
            mlp_layer_types=["dense", "dense", "dense", "sparse", "sparse"],
            quantization_config={"quant_method": "fp8", "weight_block_size": [2, 2]},
        )

        assert GlmMoeDsaReducedCheckpointTransform.apply(src, out, model_config=config)
        prepared = _load_prepared_tensors(out)
        for layer_idx in (3, 4):
            for kind in ("gate", "up", "down"):
                key = f"model.layers.{layer_idx}.mlp.moe_weights.{kind}"
                torch.testing.assert_close(prepared[key], torch.full((1, 2, 2), layer_idx, dtype=torch.float32))

    def test_glm_checkpoint_transform_precedes_generic_moe_stacking(self):
        transforms = QEFFAutoModelForCausalLM._checkpoint_transforms
        assert transforms.index(GlmMoeDsaReducedCheckpointTransform) < transforms.index(
            MoEExpertStackingCheckpointTransform
        )

    def test_glm_fp8_block_size_uses_preserved_metadata_after_quant_config_is_cleared(self):
        config = SimpleNamespace(quantization_config=None, _glm_moe_dsa_weight_block_size=(128, 64))
        assert GlmMoeDsaReducedCheckpointTransform._weight_block_size(config) == (128, 64)

        with pytest.raises(ValueError, match="two-dimensional FP8 weight_block_size"):
            GlmMoeDsaReducedCheckpointTransform._weight_block_size(SimpleNamespace(quantization_config=None))

    def test_glm_partial_fp8_hf_patch_uses_configured_block_size(self, monkeypatch):
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        from examples.glm.glm53_four_layer_decode_compile_generate import install_partial_fp8_dequant_patch

        monkeypatch.setattr(Fp8Dequantize, "_dequantize_one", Fp8Dequantize._dequantize_one)
        install_partial_fp8_dequant_patch(weight_block_size=(4, 2))

        dequantizer = object.__new__(Fp8Dequantize)
        quantized = torch.ones((5, 2), dtype=torch.float32)
        scales = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
        result = dequantizer._dequantize_one(quantized, scales, output_dtype=torch.float32)

        expected = torch.tensor([[2.0, 2.0]] * 4 + [[3.0, 3.0]])
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            ("dsa_cp1", (1, 2176, 1, 32)),
            ("dsa_cp2", (1, 2176, 2, 32)),
            ("dsa_ts16_smoke", (16, 4096, 16, 32)),
        ],
    )
    def test_glm_validation_runtime_defaults_are_preset_specific(self, preset, expected):
        from examples.glm.glm53_four_layer_decode_compile_generate import resolve_runtime_dimensions

        args = SimpleNamespace(
            attention_preset=preset,
            batch_size=None,
            ctx_len=None,
            num_devices=None,
            generation_len=None,
            prompt_len=4,
        )
        assert resolve_runtime_dimensions(args) == expected

    def test_glm_validation_runtime_cli_values_override_preset_defaults(self):
        from examples.glm.glm53_four_layer_decode_compile_generate import resolve_runtime_dimensions

        args = SimpleNamespace(
            attention_preset="dsa_ts16_smoke",
            batch_size=32,
            ctx_len=8192,
            num_devices=32,
            generation_len=7,
            prompt_len=4,
        )
        assert resolve_runtime_dimensions(args) == (32, 8192, 32, 7)

    def test_glm_generation_perf_metrics_are_json_safe(self):
        from examples.glm.glm53_four_layer_decode_compile_generate import serialize_qeff_perf_metrics

        exec_info = SimpleNamespace(
            perf_metrics=SimpleNamespace(
                prefill_time=0.25,
                decode_perf=50.0,
                total_perf=40.0,
                total_time=1.5,
            )
        )

        assert serialize_qeff_perf_metrics(exec_info, batch_size=2) == {
            "prefill_time": 0.25,
            "decode_perf": 50.0,
            "total_perf": 40.0,
            "total_time": 1.5,
            "decode_token_latency_ms_per_token": 10.0,
        }

    def test_glm_weight_free_compile_uses_mxint8_for_all_cache_io(self, tmp_path):
        from examples.glm.glm53_four_layer_decode_compile_generate import compile_weight_free_decode_only

        qeff_model = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(num_hidden_layers=2),
                get_indexer_cache_layers=lambda config: (0,),
            ),
            _compile=MagicMock(return_value=tmp_path / "qpc"),
        )

        compile_weight_free_decode_only(
            qeff_model,
            tmp_path / "model.onnx",
            tmp_path / "compile",
            ctx_len=4096,
            batch_size=16,
            num_devices=16,
            num_cores=16,
            cache_io_dtype="mxint8",
            user_tiled=False,
        )

        compile_kwargs = qeff_model._compile.call_args.kwargs
        assert compile_kwargs["custom_io"] == {
            "compressed_kv.0": "mxint8",
            "compressed_kv.0_RetainedState": "mxint8",
            "k_pe.0": "mxint8",
            "k_pe.0_RetainedState": "mxint8",
            "compressed_kv.1": "mxint8",
            "compressed_kv.1_RetainedState": "mxint8",
            "k_pe.1": "mxint8",
            "k_pe.1_RetainedState": "mxint8",
            "indexer_key.0": "mxint8",
            "indexer_key.0_RetainedState": "mxint8",
        }
        assert compile_kwargs["mxint8_kv_cache"] is True
        assert compile_kwargs["allow_mxint8_mdp_io"] is True

    def test_checkpoint_pipeline_rebuilds_when_source_changes(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"weight": torch.ones(2, dtype=torch.float16)})

        pipeline = CheckpointTransformPipeline([DtypeConversionCheckpointTransform])
        prepared = pipeline.apply(src, out, target_dtype=torch.float32)

        assert prepared == out
        assert (out / CHECKPOINT_PREPARED_MANIFEST).is_file()
        torch.testing.assert_close(_load_prepared_tensors(out)["weight"], torch.ones(2, dtype=torch.float32))

        _write_safetensors_checkpoint(src, {"weight": torch.ones(3, dtype=torch.float16)})
        prepared = pipeline.apply(src, out, target_dtype=torch.float32)

        assert prepared == out
        torch.testing.assert_close(_load_prepared_tensors(out)["weight"], torch.ones(3, dtype=torch.float32))

    def test_checkpoint_pipeline_rebuilds_when_model_context_changes(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"weight": torch.ones(2, dtype=torch.float16)})
        pipeline = CheckpointTransformPipeline([DtypeConversionCheckpointTransform])

        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            model_config=SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=4),
        )
        stale_marker = out / "stale"
        stale_marker.touch()
        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            model_config=SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=78),
        )

        manifest = json.loads((out / CHECKPOINT_PREPARED_MANIFEST).read_text())
        assert not stale_marker.exists()
        assert manifest["transform_context"] == {"model_type": "glm_moe_dsa", "num_hidden_layers": 78}

    def test_checkpoint_pipeline_rebuilds_when_moe_weight_layout_changes(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"weight": torch.ones(2, dtype=torch.float16)})
        pipeline = CheckpointTransformPipeline([DtypeConversionCheckpointTransform])
        model_config = SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=4)

        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            model_config=model_config,
            checkpoint_transform_context={
                "moe_weight_layout": {
                    "flavour": "expert_parallel",
                    "num_pipeline_stages": 2,
                    "num_parallelized_experts": 2,
                }
            },
        )
        stale_marker = out / "stale"
        stale_marker.touch()
        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            model_config=model_config,
            checkpoint_transform_context={
                "moe_weight_layout": {
                    "flavour": "expert_parallel",
                    "num_pipeline_stages": 4,
                    "num_parallelized_experts": 1,
                }
            },
        )

        manifest = json.loads((out / CHECKPOINT_PREPARED_MANIFEST).read_text())
        assert not stale_marker.exists()
        assert manifest["transform_context"]["checkpoint_transform_context"]["moe_weight_layout"][
            "num_pipeline_stages"
        ] == 4

    def test_checkpoint_pipeline_rebuilds_incomplete_prepared_dir(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"weight": torch.ones(2, dtype=torch.float16)})

        pipeline = CheckpointTransformPipeline([DtypeConversionCheckpointTransform])
        prepared = pipeline.apply(src, out, target_dtype=torch.float32)

        assert prepared == out
        (out / "model.safetensors").unlink()
        assert (out / CHECKPOINT_PREPARED_MANIFEST).is_file()
        assert (out / CHECKPOINT_PREPARED_SENTINEL).is_file()

        prepared = pipeline.apply(src, out, target_dtype=torch.float32)

        assert prepared == out
        torch.testing.assert_close(_load_prepared_tensors(out)["weight"], torch.ones(2, dtype=torch.float32))

    def test_stacks_per_expert_weights_to_moe_weights(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe"

        gate_0 = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        gate_1 = gate_0 + 100
        up_0 = gate_0 + 200
        up_1 = gate_0 + 300
        down_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 400
        down_1 = down_0 + 100
        _write_safetensors_checkpoint(
            src,
            {
                f"{prefix}.experts.0.w1.weight": gate_0,
                f"{prefix}.experts.0.w3.weight": up_0,
                f"{prefix}.experts.0.w2.weight": down_0,
                f"{prefix}.experts.1.w1.weight": gate_1,
                f"{prefix}.experts.1.w3.weight": up_1,
                f"{prefix}.experts.1.w2.weight": down_1,
                "model.embed_tokens.weight": torch.ones(2, 4),
            },
        )

        changed = MoEExpertStackingCheckpointTransform.apply(
            src,
            out,
            target_dtype=torch.float32,
            max_workers_scan=1,
            max_workers_layers=1,
            max_workers_base=1,
        )

        assert changed
        tensors = _load_prepared_tensors(out)
        torch.testing.assert_close(
            tensors[f"{prefix}.moe_weights.gate"],
            torch.stack([gate_0, gate_1]).transpose(1, 2),
        )
        torch.testing.assert_close(
            tensors[f"{prefix}.moe_weights.up"],
            torch.stack([up_0, up_1]).transpose(1, 2),
        )
        torch.testing.assert_close(
            tensors[f"{prefix}.moe_weights.down"],
            torch.stack([down_0, down_1]).transpose(1, 2),
        )
        assert f"{prefix}.experts.gate_proj" not in tensors
        assert f"{prefix}.experts.down_proj_t" not in tensors

    def test_splits_dim2_fused_experts_with_bias_to_moe_weights(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.mlp.experts"
        moe_prefix = "model.layers.0.mlp.moe_weights"
        gate = torch.full((2, 3, 4), 1.0)
        up = torch.full((2, 3, 4), 2.0)
        gate_up = torch.empty(2, 3, 8)
        gate_up[..., 0::2] = gate
        gate_up[..., 1::2] = up
        down = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        gate_bias = torch.full((2, 4), 3.0)
        up_bias = torch.full((2, 4), 4.0)
        gate_up_bias = torch.empty(2, 8)
        gate_up_bias[..., 0::2] = gate_bias
        gate_up_bias[..., 1::2] = up_bias
        down_bias = torch.full((2, 3), 5.0)
        _write_safetensors_checkpoint(
            src,
            {
                f"{prefix}.gate_up_proj": gate_up,
                f"{prefix}.down_proj": down,
                f"{prefix}.gate_up_proj_bias": gate_up_bias,
                f"{prefix}.down_proj_bias": down_bias,
            },
        )

        changed = MoEFusedExpertSplitCheckpointTransform.apply(src, out, target_dtype=torch.float32)

        assert changed
        tensors = _load_prepared_tensors(out)
        torch.testing.assert_close(tensors[f"{moe_prefix}.gate"], gate)
        torch.testing.assert_close(tensors[f"{moe_prefix}.up"], up)
        torch.testing.assert_close(tensors[f"{moe_prefix}.down"], down)
        torch.testing.assert_close(tensors[f"{moe_prefix}.gate_bias"], gate_bias)
        torch.testing.assert_close(tensors[f"{moe_prefix}.up_bias"], up_bias)
        torch.testing.assert_close(tensors[f"{moe_prefix}.down_bias"], down_bias)

    def test_splits_granitemoe_fused_parallel_experts_to_moe_weights(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe"
        gate = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        up = gate + 100
        gate_up = torch.cat((gate, up), dim=0).reshape(1, 4, 4)
        down = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        _write_safetensors_checkpoint(
            src,
            {
                f"{prefix}.input_linear.weight": gate_up,
                f"{prefix}.output_linear.weight": down,
            },
        )

        changed = GraniteMoeFusedExpertSplitCheckpointTransform.apply(src, out, target_dtype=torch.float32)

        assert changed
        tensors = _load_prepared_tensors(out)
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.gate"], gate_up[:, :2, :].transpose(1, 2))
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.up"], gate_up[:, 2:, :].transpose(1, 2))
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.down"], down.transpose(1, 2))

    def test_resolver_accepts_moe_weight_aliases(self):
        checkpoint_index = {
            "model.layers.0.mlp.moe_weights.gate": "model.safetensors",
            "model.layers.1.mlp.experts.moe_weights.up": "model.safetensors",
            "model.layers.2.block_sparse_moe.experts.down_proj_t": "model.safetensors",
        }
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        assert (
            find_checkpoint_key("model.layers.0.mlp.experts.moe_weights.gate", checkpoint_index, backbone)
            == "model.layers.0.mlp.moe_weights.gate"
        )
        assert (
            find_checkpoint_key("model.layers.1.mlp.moe_weights.up", checkpoint_index, backbone)
            == "model.layers.1.mlp.experts.moe_weights.up"
        )
        assert (
            find_checkpoint_key("model.layers.2.block_sparse_moe.moe_weights.down", checkpoint_index, backbone)
            == "model.layers.2.block_sparse_moe.experts.down_proj_t"
        )

    def test_resolver_accepts_granitemoe_router_layer_alias(self):
        checkpoint_index = {"model.layers.0.block_sparse_moe.router.layer.weight": "model.safetensors"}
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        assert (
            find_checkpoint_key("model.layers.0.block_sparse_moe.router.weight", checkpoint_index, backbone)
            == "model.layers.0.block_sparse_moe.router.layer.weight"
        )

    def test_resolver_rejects_ambiguous_moe_weight_aliases(self):
        checkpoint_index = {
            "model.layers.0.mlp.experts.moe_weights.gate": "model.safetensors",
            "model.layers.0.mlp.moe_weights.gate": "model.safetensors",
        }
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        with pytest.raises(ValueError, match="Ambiguous checkpoint key"):
            find_checkpoint_key("model.layers.0.mlp.experts.moe_weights.gate", checkpoint_index, backbone)

    def test_resolver_rejects_ambiguous_prefix_fallbacks(self):
        checkpoint_index = {
            "base_model.model.embed_tokens.weight": "model.safetensors",
            "model.embed_tokens.weight": "model.safetensors",
        }
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        with pytest.raises(ValueError, match="Ambiguous checkpoint key"):
            find_checkpoint_key("base_model.model.embed_tokens.weight", checkpoint_index, backbone)

    @pytest.mark.parametrize(
        "state_kind,state_name",
        [
            ("parameter", "weight"),
            ("buffer", "running_scale"),
        ],
    )
    def test_promote_initializers_rejects_unresolved_model_state(self, tmp_path, monkeypatch, state_kind, state_name):
        src = tmp_path / "src"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"other.weight": torch.ones(2, dtype=torch.float32)})

        class MissingStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if state_kind == "parameter":
                    self.register_parameter(state_name, torch.nn.Parameter(torch.ones(2)))
                else:
                    self.register_buffer(state_name, torch.ones(2))

        initializer = SimpleNamespace(shape=(2,), dtype=ir.DataType.FLOAT)
        graph = SimpleNamespace(initializers={state_name: initializer}, inputs=[])
        onnx_program = SimpleNamespace(model=SimpleNamespace(graph=graph))
        monkeypatch.setattr(
            checkpoint_key_resolver.ir,
            "Value",
            lambda name, shape, type: SimpleNamespace(name=name, shape=shape, type=type),
        )

        with pytest.raises(ValueError, match=f"Could not resolve model initializer '{state_name}'"):
            checkpoint_key_resolver.promote_initializers_and_build_spec(
                onnx_program=onnx_program,
                model_ref=str(src),
                model_name="tiny-missing-state",
                qeff_model=SimpleNamespace(model=MissingStateModel()),
            )

    @pytest.mark.parametrize(
        "state_kind,state_name",
        [
            ("buffer", "rotary_emb.inv_freq"),
            ("buffer", "transformer.h.0.attn.embed_positions"),
            ("buffer", "model.embed_tokens.embed_scale"),
            ("parameter", "model.sin_cached"),
            ("parameter", "model.cos_cached"),
        ],
    )
    def test_promote_initializers_keeps_computed_state_embedded(self, tmp_path, monkeypatch, state_kind, state_name):
        src = tmp_path / "src"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"other.weight": torch.ones(2, dtype=torch.float32)})

        class ComputedStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                parent = self
                parts = state_name.split(".")
                for part in parts[:-1]:
                    child = torch.nn.Module()
                    setattr(parent, part, child)
                    parent = child
                if state_kind == "parameter":
                    parent.register_parameter(parts[-1], torch.nn.Parameter(torch.ones(2)))
                else:
                    parent.register_buffer(parts[-1], torch.ones(2))

        initializer = SimpleNamespace(shape=(2,), dtype=ir.DataType.FLOAT)
        graph = SimpleNamespace(initializers={state_name: initializer}, inputs=[])
        onnx_program = SimpleNamespace(model=SimpleNamespace(graph=graph))
        monkeypatch.setattr(
            checkpoint_key_resolver.ir,
            "Value",
            lambda name, shape, type: SimpleNamespace(name=name, shape=shape, type=type),
        )

        spec = checkpoint_key_resolver.promote_initializers_and_build_spec(
            onnx_program=onnx_program,
            model_ref=str(src),
            model_name="tiny-computed-state",
            qeff_model=SimpleNamespace(model=ComputedStateModel()),
        )

        assert state_name in graph.initializers
        assert graph.inputs == []
        assert spec.inputs == []

    def test_promotes_embed_tokens_for_tied_model(self, tmp_path, monkeypatch):
        """When tie_word_embeddings=True, torch.export deduplicates tied weights —
        only model.embed_tokens.weight appears as an ONNX initializer, never
        lm_head.weight. Verify the canonical name is promoted correctly."""
        src = tmp_path / "src"
        src.mkdir()
        tied_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        _write_safetensors_checkpoint(src, {"model.embed_tokens.weight": tied_weight})

        class TiedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(4, 3)
                self.lm_head = torch.nn.Linear(3, 4, bias=False)
                self.lm_head.weight = self.model.embed_tokens.weight

        initializer = SimpleNamespace(shape=tied_weight.shape, dtype=ir.DataType.FLOAT)
        # Realistic: torch.export deduplicates tied weights — only the canonical
        # name (model.embed_tokens.weight) appears as an ONNX initializer.
        graph = SimpleNamespace(initializers={"model.embed_tokens.weight": initializer}, inputs=[])
        onnx_program = SimpleNamespace(model=SimpleNamespace(graph=graph))
        monkeypatch.setattr(
            checkpoint_key_resolver.ir,
            "Value",
            lambda name, shape, type: SimpleNamespace(name=name, shape=shape, type=type),
        )

        spec = checkpoint_key_resolver.promote_initializers_and_build_spec(
            onnx_program=onnx_program,
            model_ref=str(src),
            model_name="tiny-tied",
            qeff_model=SimpleNamespace(model=TiedModel()),
        )

        assert "model.embed_tokens.weight" not in graph.initializers
        assert [v.name for v in graph.inputs] == ["model.embed_tokens.weight"]
        assert spec.inputs[0].name == "model.embed_tokens.weight"
        assert spec.inputs[0].location.key == "model.embed_tokens.weight"


def _fake_export(
    self,
    example_inputs,
    output_names,
    dynamic_axes,
    onnx_transform_kwargs=None,
    export_dir=None,
    dynamo=False,
    dynamic_shapes=None,
    **export_kwargs,
):
    pass


class TestWeightFreeExportHash:
    def test_ctx_gather_3d_fake_preserves_feature_dimension(self):
        import QEfficient.customop.dynamo_ops  # noqa: F401

        with torch._subclasses.fake_tensor.FakeTensorMode():
            data = torch.empty((2, 8, 5))
            ctx_indices = torch.empty((2, 3), dtype=torch.int32)
            output = torch.ops.qefficient.ctx_gather_3d(data, ctx_indices)

        assert tuple(output.shape) == (2, 3, 5)

    def test_dynamic_shapes_preserve_glm_nested_caches(self):
        dynamic_shapes = convert_dynamic_axes_to_dynamic_shapes(
            {
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "compressed_kv.0": {0: "batch_size", 2: "ctx_len"},
                "k_pe.0": {0: "batch_size", 2: "ctx_len"},
                "compressed_kv.1": {0: "batch_size", 2: "ctx_len"},
                "k_pe.1": {0: "batch_size", 2: "ctx_len"},
                "indexer_key.0": {0: "batch_size", 1: "ctx_len"},
                "indexer_key.2": {0: "batch_size", 1: "ctx_len"},
            },
            SimpleNamespace(model_type="glm_moe_dsa"),
        )

        assert "compressed_kvs" in dynamic_shapes
        assert isinstance(dynamic_shapes["compressed_kvs"][0], list)
        assert len(dynamic_shapes["compressed_kvs"]) == 2
        assert "indexer_key_cache" in dynamic_shapes
        assert len(dynamic_shapes["indexer_key_cache"]) == 2

    def test_dynamic_shapes_keep_folded_glm_dimensions_static(self):
        dynamic_shapes = convert_dynamic_axes_to_dynamic_shapes(
            {
                "compressed_kv.0": {0: "glm_attn_batch_local_0", 2: "ctx_len"},
                "indexer_key.0": {0: "batch_size", 2: "glm_indexer_ctx_local_0"},
            },
            SimpleNamespace(model_type="glm_moe_dsa"),
        )

        assert set(dynamic_shapes["compressed_kvs"][0][0]) == {2}
        assert set(dynamic_shapes["indexer_key_cache"][0]) == {0}

    def test_weight_free_export_hash_differs_from_regular_dynamo(self):
        config = SimpleNamespace(to_diff_dict=lambda: {"model_type": "llama"})
        common_model = SimpleNamespace(
            model=SimpleNamespace(config=config),
            hash_params={"pretrained_model_name_or_path": "tiny"},
            _use_onnx_subfunctions=False,
            _weight_free=False,
        )
        weight_free_model = SimpleNamespace(
            model=SimpleNamespace(config=config),
            hash_params={"pretrained_model_name_or_path": "tiny"},
            _use_onnx_subfunctions=False,
            _weight_free=True,
        )
        common_kwargs = {
            "example_inputs": {"input_ids": torch.ones(1, 2, dtype=torch.int64)},
            "output_names": ["logits"],
            "dynamic_axes": {"input_ids": {0: "batch_size"}},
            "dynamo": True,
        }

        regular_hash, regular_params = _generate_export_hash(
            common_model,
            (),
            dict(common_kwargs),
            _fake_export,
        )
        weight_free_hash, weight_free_params = _generate_export_hash(
            weight_free_model,
            (),
            dict(common_kwargs),
            _fake_export,
        )

        assert regular_hash != weight_free_hash
        assert "weight_free" not in regular_params
        assert weight_free_params["weight_free"] is True


class TestRuntimeRequirements:
    def test_validate_runtime_requirements_accepts_matching_requirements(self, monkeypatch):
        requirements = {"torch": "==2.13.0", "accelerate": "==1.9.0"}
        versions = {"torch": "2.13.0+cpu", "accelerate": "1.9.0"}

        monkeypatch.setattr(runtime_requirements.metadata, "version", lambda name: versions[name])

        validate_runtime_requirements(
            requirements,
            feature_name="weight_free=True",
            install_command="pip install -r requirements.txt",
        )

    def test_validate_runtime_requirements_reports_all_mismatches(self, monkeypatch):
        requirements = {"accelerate": "==1.9.0", "missing-package": "==1.0"}

        def version(name):
            if name == "missing-package":
                raise runtime_requirements.metadata.PackageNotFoundError
            return "1.14.0"

        monkeypatch.setattr(runtime_requirements.metadata, "version", version)

        with pytest.raises(AssertionError) as exc_info:
            validate_runtime_requirements(
                requirements,
                feature_name="dynamo=True",
                install_command="pip install -r requirements.txt",
            )

        message = str(exc_info.value)
        assert "dynamo=True requires the Dynamo export environment" in message
        assert "Mismatched packages:" in message
        assert "Install or repair the environment with:" in message
        assert "accelerate: installed 1.14.0, expected ==1.9.0" in message
        assert "missing-package: not installed, expected ==1.0" in message

    def test_weight_free_from_pretrained_validates_requirements_before_model_build(self, monkeypatch):
        calls = []

        def validate(feature_name):
            calls.append(feature_name)
            raise AssertionError("bad weight-free environment")

        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto.validate_dynamo_export_requirements",
            validate,
        )
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto._build_meta_model",
            lambda *args, **kwargs: pytest.fail("_build_meta_model should not be called"),
        )

        with pytest.raises(AssertionError, match="bad weight-free environment"):
            QEFFAutoModelForCausalLM.from_pretrained("dummy-model", weight_free=True)

        assert calls == ["weight_free=True"]

    def test_export_wrapper_validates_dynamo_requirements_before_export(self, monkeypatch, tmp_path):
        from QEfficient.utils.export_utils import export_wrapper

        calls = []

        def validate(feature_name):
            calls.append(feature_name)
            raise AssertionError("bad dynamo environment")

        monkeypatch.setattr("QEfficient.utils.export_utils.validate_dynamo_export_requirements", validate)

        class DummyQEff:
            @export_wrapper
            def export(
                self,
                example_inputs,
                output_names,
                dynamic_axes,
                export_dir=None,
                dynamo=False,
                dynamic_shapes=None,
            ):
                pytest.fail("export should not run")

        with (
            pytest.warns(DeprecationWarning, match="Direct \\.export\\(\\) is deprecated"),
            pytest.raises(AssertionError, match="bad dynamo environment"),
        ):
            DummyQEff().export(
                example_inputs={},
                output_names=[],
                dynamic_axes={},
                export_dir=tmp_path,
                dynamo=True,
            )

        assert calls == ["dynamo=True"]


# ---------------------------------------------------------------------------
# TestTemporarilyEnableNestedCompileRegions
# ---------------------------------------------------------------------------


class TestTemporarilyEnableNestedCompileRegions:
    def test_patches_decoder_layers_and_restores(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        inner_model = qeff_model.model

        decoder_layers = [m for m in inner_model.modules() if isinstance(m, QEffLlamaDecoderLayer)]
        assert len(decoder_layers) > 0, "No QEffLlamaDecoderLayer found in wrapped model"

        original_qualnames = [getattr(m.forward, "__qualname__", "") for m in decoder_layers]

        with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
            for m in decoder_layers:
                fwd = getattr(m, "forward", None)
                qualname = getattr(fwd, "__qualname__", "")
                assert (
                    "mark_compile_region" in qualname or "nested_compile_region" in qualname or "inner" in qualname
                ), (
                    f"Expected nested_compile_region wrapper on {m.__class__.__name__}.forward, "
                    f"got qualname: {qualname!r}"
                )

        for m, orig_qn in zip(decoder_layers, original_qualnames):
            fwd = getattr(m, "forward", None)
            qualname = getattr(fwd, "__qualname__", "")
            assert qualname == orig_qn, f"forward qualname not restored: expected {orig_qn!r}, got {qualname!r}"

    def test_noop_when_already_wrapped(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        inner_model = qeff_model.model

        decoder_layers = [m for m in inner_model.modules() if isinstance(m, QEffLlamaDecoderLayer)]

        with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
            wrapped_forwards_first = [id(m.forward) for m in decoder_layers]

            with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
                wrapped_forwards_second = [id(m.forward) for m in decoder_layers]

        assert len(wrapped_forwards_first) == len(wrapped_forwards_second)


# ---------------------------------------------------------------------------
# TestPreserveNestedCacheRetainedStateTransform
# ---------------------------------------------------------------------------


class TestPreserveNestedCacheRetainedStateTransform:
    def test_adds_retained_state_outputs_to_call_nodes(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)
        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert changed, "Transform should have modified the model (dangling _RetainedState outputs)"

        for fn in model.functions:
            assert len(fn.output) >= 2, (
                f"Function '{fn.name}' should have at least 2 outputs after transform, got {list(fn.output)}"
            )

    def test_noop_when_no_dangling_retained_states(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        for out in list(model.graph.output):
            if out.name.endswith("_RetainedState"):
                model.graph.output.remove(out)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert not changed, "Transform should be a no-op when there are no dangling _RetainedState outputs"

    def test_noop_when_scatter_count_not_two(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=1, scatter_count_per_fn=1)
        PreserveNestedCacheRetainedStateTransform.apply(model)
        fn = model.functions[0]
        assert len(fn.output) == 1, f"Function with 1 scatter should not have outputs added, got {list(fn.output)}"


# ---------------------------------------------------------------------------
# TestRenameRepeatedSubgraphTransform
# ---------------------------------------------------------------------------


class TestRenameRepeatedSubgraphTransform:
    def test_renames_repeated_subgraph_functions(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert changed

        fn_names = [fn.name for fn in model.functions]
        assert "QEffLlamaDecoderLayer" in fn_names, f"Expected 'QEffLlamaDecoderLayer' in {fn_names}"
        assert "QEffLlamaDecoderLayer_1" in fn_names, f"Expected 'QEffLlamaDecoderLayer_1' in {fn_names}"

        node_op_types = [n.op_type for n in model.graph.node]
        assert "QEffLlamaDecoderLayer" in node_op_types
        assert "QEffLlamaDecoderLayer_1" in node_op_types

    def test_noop_on_empty_classnames(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=[])
        assert not changed

    def test_noop_when_no_repeated_subgraph_functions(self):
        fn = helper.make_function(
            domain="",
            fname="SomeOtherFunction",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        graph = helper.make_graph([], "g", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(fn)

        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert not changed

    def test_handles_alternative_subgraph_pattern(self):
        fn0 = helper.make_function(
            domain="",
            fname="subgraph_0",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        fn1 = helper.make_function(
            domain="",
            fname="subgraph_1",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        call0 = helper.make_node("subgraph_0", inputs=[], outputs=[])
        call1 = helper.make_node("subgraph_1", inputs=[], outputs=[])
        graph = helper.make_graph([call0, call1], "g", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.extend([fn0, fn1])

        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["MyDecoderLayer"])
        assert changed
        fn_names = {fn.name for fn in model.functions}
        assert "MyDecoderLayer" in fn_names


# ---------------------------------------------------------------------------
# TestPruneFakeInitializersTransform
# ---------------------------------------------------------------------------


class TestPruneFakeInitializersTransform:
    def _make_mock_onnx_program(self, initializer_names, used_names, fake_initializers):
        from torch._subclasses.fake_tensor import FakeTensor

        initializers = {}
        for name in initializer_names:
            mock_init = MagicMock()
            if name in fake_initializers:
                fake_tensor = MagicMock(spec=FakeTensor)
                mock_init.const_value.raw = fake_tensor
            else:
                mock_init.const_value.raw = torch.zeros(2)
            initializers[name] = mock_init

        mock_graph = MagicMock()
        mock_graph.initializers = initializers

        mock_node = MagicMock()
        mock_node.inputs = list(used_names)
        mock_graph.__iter__ = lambda self: iter([mock_node])
        mock_graph.outputs = []

        mock_program = MagicMock()
        mock_program.model.graph = mock_graph
        return mock_program

    def test_prunes_fake_tensor_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a", "weight_b"],
            used_names=set(),
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert changed
        assert "weight_a" not in program.model.graph.initializers

    def test_preserves_used_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a"],
            used_names={"weight_a"},
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "weight_a" in program.model.graph.initializers

    def test_preserves_non_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["real_weight"],
            used_names=set(),
            fake_initializers=set(),
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "real_weight" in program.model.graph.initializers


def test_glm_mla_derived_parameters_cover_all_production_graph_weights():
    from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import derive_glm_mla_parameters

    q_b = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
    kv_b = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
    derived = derive_glm_mla_parameters(
        q_b,
        kv_b,
        num_heads=2,
        q_lora_rank=4,
        kv_lora_rank=3,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=2,
    )

    assert set(derived) == {
        "q_up",
        "q_rope",
        "k_up",
        "v_up",
        "per_head_q_up",
        "per_head_k_up",
        "per_head_k_up_normal",
        "per_head_v_up",
        "fusedqk",
    }
    torch.testing.assert_close(
        derived["fusedqk"],
        torch.matmul(derived["per_head_q_up"], derived["per_head_k_up"]),
    )
