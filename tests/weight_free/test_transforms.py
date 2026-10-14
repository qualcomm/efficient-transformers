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
import threading
import time
from pathlib import Path
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
    CheckpointPlan,
    CheckpointTask,
    CheckpointTransformPipeline,
    TaskParams,
    TensorRef,
    execute_checkpoint_plan,
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
    GptOssMxfp4ExpertDequantSplitCheckpointTransform,
    GraniteMoeFusedExpertSplitCheckpointTransform,
    MoEExpertStackingCheckpointTransform,
    MoEFusedExpertSplitCheckpointTransform,
)
from QEfficient.exporter.weight_free.ort_weight_injection import load_weight_free_ort_inputs
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.moe.weights import MoEWeights
from QEfficient.utils import runtime_requirements
from QEfficient.utils.checkpoint_utils import checkpoint_root
from QEfficient.utils.export_utils import _generate_export_hash
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


def _make_minimal_onnx_with_repeated_subgraphs(
    num_layers: int = 2,
    scatter_count_per_fn: int = 2,
    include_unrelated_scatter: bool = False,
):
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
        fn_inputs = [f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"]
        if include_unrelated_scatter:
            fn_inputs.append(f"moe_indices_{i}")
            scatter_nodes.append(
                helper.make_node(
                    "CtxScatter3DInt",
                    inputs=[f"moe_indices_{i}", "position_ids", f"moe_indices_{i}"],
                    outputs=[f"moe_scatter_{i}"],
                    domain="qti.aisw",
                )
            )
        for j in range(scatter_count_per_fn):
            kind = "key" if j == 0 else "value"
            scatter_out = f"scatter_{i}_{j}"
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
            inputs=fn_inputs,
            outputs=fn_outputs,
            nodes=scatter_nodes,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("qti.aisw", 1)],
        )
        functions.append(fn)

        retained_key = f"past_key.{i}_RetainedState"
        retained_val = f"past_value.{i}_RetainedState"
        call_node = helper.make_node(
            fn_name,
            inputs=fn_inputs,
            outputs=[],
            domain="",
        )
        call_nodes.append(call_node)

        graph_outputs.append(helper.make_tensor_value_info(retained_key, TensorProto.FLOAT, None))
        graph_outputs.append(helper.make_tensor_value_info(retained_val, TensorProto.FLOAT, None))

        graph_inputs.append(helper.make_tensor_value_info(f"past_key.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"past_value.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"hidden_{i}", TensorProto.FLOAT, None))
        if include_unrelated_scatter:
            graph_inputs.append(helper.make_tensor_value_info(f"moe_indices_{i}", TensorProto.INT32, None))

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


def _write_sharded_safetensors_checkpoint(root, shards):
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file({key: tensor.contiguous() for key, tensor in tensors.items()}, str(root / shard_name))
        weight_map.update({key: shard_name for key in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))


def _load_prepared_tensors(root):
    """Load all tensors from a prepared checkpoint directory.

    Reads from model.safetensors.index.json if present; otherwise scans
    all safetensors shards directly. The index is written by the pipeline
    Stage ⑤ finalise step. Tests that call transforms directly (without
    going through the pipeline) will not have an index file, so the shard
    scan fallback is needed there.
    """
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        index = {}
        for sf in sorted(root.glob("*.safetensors")):
            with safe_open(str(sf), framework="pt") as handle:
                for k in handle.keys():
                    index[k] = sf.name
    loaded = {}
    for key, shard_name in index.items():
        with safe_open(str(root / shard_name), framework="pt") as handle:
            loaded[key] = handle.get_tensor(key)
    return loaded


def test_checkpoint_root_symlinked_shards(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    blob = hub / "models--org--m" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x")
    local = tmp_path / "mymodel"
    local.mkdir()
    shard = local / "model.safetensors"
    shard.symlink_to(blob)

    root = checkpoint_root(str(local), [str(shard)])

    assert root == tmp_path
    assert shard.relative_to(root) == Path("mymodel/model.safetensors")


# ---------------------------------------------------------------------------
# Test checkpoint layout transforms
# ---------------------------------------------------------------------------


class TestWeightFreeCheckpointTransforms:
    def test_qwen3_original_layout_exposes_source_parameter_names(self):
        from transformers import Qwen3MoeConfig

        from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import QEffQwen3MoeExperts

        experts = QEffQwen3MoeExperts(
            Qwen3MoeConfig(num_experts=2, hidden_size=4, moe_intermediate_size=6, intermediate_size=6)
        )
        experts._qeff_use_original_checkpoint = True

        weights = experts.transform_weights()
        parameter_names = {name for name, _ in experts.named_parameters()}

        assert parameter_names == {
            f"{expert}.{projection}.weight"
            for expert in ("0", "1")
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        assert tuple(weights.gate.shape) == (2, 4, 6)
        assert tuple(weights.up.shape) == (2, 4, 6)
        assert tuple(weights.down.shape) == (2, 6, 4)

    def test_gpt_oss_original_layout_splits_fused_source_weights_in_forward(self):
        from transformers import GptOssConfig

        from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import QEffGptOssExperts

        experts = QEffGptOssExperts(GptOssConfig(num_local_experts=2, hidden_size=4, intermediate_size=6))
        source = {name: parameter.detach().clone() for name, parameter in experts.named_parameters()}
        experts._qeff_use_original_checkpoint = True

        weights = experts.transform_weights()

        torch.testing.assert_close(weights.gate, source["gate_up_proj"][..., ::2])
        torch.testing.assert_close(weights.up, source["gate_up_proj"][..., 1::2])
        torch.testing.assert_close(weights.down, source["down_proj"])
        torch.testing.assert_close(weights.gate_bias, source["gate_up_proj_bias"][..., ::2])
        torch.testing.assert_close(weights.up_bias, source["gate_up_proj_bias"][..., 1::2])
        torch.testing.assert_close(weights.down_bias, source["down_proj_bias"])
        assert set(dict(experts.named_parameters())) == set(source)

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

        (out / "base-0000.safetensors").unlink()
        pipeline.apply(src, out, target_dtype=torch.float32)
        torch.testing.assert_close(_load_prepared_tensors(out)["weight"], torch.ones(2, dtype=torch.float32))

        _write_safetensors_checkpoint(src, {"weight": torch.ones(3, dtype=torch.float16)})
        prepared = pipeline.apply(src, out, target_dtype=torch.float32)

        assert prepared == out
        torch.testing.assert_close(_load_prepared_tensors(out)["weight"], torch.ones(3, dtype=torch.float32))

    def test_checkpoint_pipeline_rebuilds_incomplete_prepared_dir(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"weight": torch.ones(2, dtype=torch.float16)})

        pipeline = CheckpointTransformPipeline([DtypeConversionCheckpointTransform])
        prepared = pipeline.apply(src, out, target_dtype=torch.float32)
        prepared_weight_map = json.loads((prepared / "model.safetensors.index.json").read_text())["weight_map"]
        prepared_shard = next(iter(set(prepared_weight_map.values())))
        (prepared / prepared_shard).unlink()

        prepared = pipeline.apply(src, out, target_dtype=torch.float32)

        assert prepared == out
        torch.testing.assert_close(_load_prepared_tensors(out)["weight"], torch.ones(2, dtype=torch.float32))

    def test_ort_weight_injection_resolves_cache_relative_spec_for_absolute_model_id(self, tmp_path):
        hf_cache = tmp_path / "hf_cache"
        prepared = hf_cache / "models--org--model" / "snapshots" / "prepared"
        export_dir = tmp_path / "export"
        prepared.mkdir(parents=True)
        export_dir.mkdir()
        save_file({"weight": torch.tensor([1.0, 2.0])}, str(prepared / "model.safetensors"))

        weight_spec_path = export_dir / "weight_spec.json"
        weight_spec_path.write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "format": "safetensors",
                            "path": "models--org--model/snapshots/prepared/model.safetensors",
                        }
                    ],
                    "inputs": [{"name": "weight", "location": {"file": 0, "key": "weight"}}],
                    "model_id": str(prepared),
                    "model_name": "tiny",
                    "version": 5,
                }
            )
        )

        ort_inputs = load_weight_free_ort_inputs(weight_spec_path, {})

        assert ort_inputs["weight"].tolist() == [1.0, 2.0]

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

        pipeline = CheckpointTransformPipeline([MoEExpertStackingCheckpointTransform])
        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            config=SimpleNamespace(num_local_experts=2),
            max_workers=1,
        )

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

    def test_pipeline_keeps_expert_parallel_experts_canonical_with_one_final_write(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe"
        tensors = {}
        gate_weights = []
        up_weights = []
        down_weights = []
        for expert_index in range(8):
            gate = torch.arange(8, dtype=torch.float32).reshape(2, 4) + expert_index * 100
            up = gate + 1_000
            down = torch.arange(8, dtype=torch.float32).reshape(4, 2) + expert_index * 100 + 2_000
            gate_weights.append(gate)
            up_weights.append(up)
            down_weights.append(down)
            tensors[f"{prefix}.experts.{expert_index}.w1.weight"] = gate
            tensors[f"{prefix}.experts.{expert_index}.w3.weight"] = up
            tensors[f"{prefix}.experts.{expert_index}.w2.weight"] = down
        _write_safetensors_checkpoint(src, tensors)

        expected_weights = MoEWeights(
            gate=torch.stack(gate_weights).transpose(1, 2),
            up=torch.stack(up_weights).transpose(1, 2),
            down=torch.stack(down_weights).transpose(1, 2),
        )

        pipeline = CheckpointTransformPipeline(
            [MoEExpertStackingCheckpointTransform, DtypeConversionCheckpointTransform]
        )
        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            config=SimpleNamespace(num_local_experts=8, model_type="mixtral"),
            hash_params={
                "moe_prefill_flavour": "expert_parallel",
                "moe_prefill_num_pipeline_stages": 2,
                "moe_prefill_num_parallelized_experts": 4,
            },
            max_workers=1,
        )

        assert sorted(path.name for path in out.glob("*.safetensors")) == ["experts-layer-00000.safetensors"]
        tensors = _load_prepared_tensors(out)
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.gate"], expected_weights.gate)
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.up"], expected_weights.up)
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.down"], expected_weights.down)

    def test_pipeline_dequantizes_gptoss_to_canonical_final_keys(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.mlp.experts"
        _write_sharded_safetensors_checkpoint(
            src,
            {
                "experts-a.safetensors": {
                    f"{prefix}.gate_up_proj_blocks": torch.zeros((2, 6, 1, 2), dtype=torch.uint8),
                    f"{prefix}.gate_up_proj_scales": torch.full((2, 6, 1), 127, dtype=torch.uint8),
                    f"{prefix}.gate_up_proj_bias": torch.arange(12, dtype=torch.float16).reshape(2, 6),
                },
                "experts-b.safetensors": {
                    f"{prefix}.down_proj_blocks": torch.zeros((2, 4, 1, 2), dtype=torch.uint8),
                    f"{prefix}.down_proj_scales": torch.full((2, 4, 1), 127, dtype=torch.uint8),
                    f"{prefix}.down_proj_bias": torch.ones((2, 4), dtype=torch.float16),
                    "model.embed_tokens.weight": torch.ones((4, 4), dtype=torch.float16),
                },
            },
        )
        pipeline = CheckpointTransformPipeline(
            [
                GptOssMxfp4ExpertDequantSplitCheckpointTransform,
                MoEExpertStackingCheckpointTransform,
                MoEFusedExpertSplitCheckpointTransform,
                DtypeConversionCheckpointTransform,
            ]
        )

        pipeline.apply(
            src,
            out,
            target_dtype=torch.bfloat16,
            config=SimpleNamespace(num_local_experts=2, model_type="gpt_oss"),
            max_workers=1,
        )

        tensors = _load_prepared_tensors(out)
        moe_prefix = "model.layers.0.mlp.moe_weights"
        assert tensors[f"{moe_prefix}.gate"].shape == (2, 4, 3)
        assert tensors[f"{moe_prefix}.up"].shape == (2, 4, 3)
        assert tensors[f"{moe_prefix}.down"].shape == (2, 4, 4)
        assert tensors[f"{moe_prefix}.gate_bias"].dtype == torch.bfloat16
        assert tensors[f"{moe_prefix}.up_bias"].dtype == torch.bfloat16
        assert tensors[f"{moe_prefix}.down_bias"].dtype == torch.bfloat16
        assert tensors["model.embed_tokens.weight"].dtype == torch.bfloat16

    @pytest.mark.parametrize("layout", ["mixtral", "granite"])
    def test_pipeline_groups_fused_experts_across_source_shards(self, tmp_path, layout):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe"
        gate = torch.arange(8, dtype=torch.float16).reshape(1, 2, 4)
        up = gate + 100
        gate_up = torch.cat((gate, up), dim=1)
        down = torch.arange(8, dtype=torch.float16).reshape(1, 4, 2)

        if layout == "mixtral":
            gate_key = f"{prefix}.experts.gate_up_proj"
            down_key = f"{prefix}.experts.down_proj"
        else:
            gate_key = f"{prefix}.input_linear.weight"
            down_key = f"{prefix}.output_linear.weight"

        _write_sharded_safetensors_checkpoint(
            src,
            {
                "gate.safetensors": {gate_key: gate_up},
                "down.safetensors": {down_key: down},
            },
        )
        pipeline = CheckpointTransformPipeline(
            [MoEExpertStackingCheckpointTransform, MoEFusedExpertSplitCheckpointTransform]
        )

        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            config=SimpleNamespace(num_local_experts=1, model_type=layout),
            max_workers=1,
        )

        assert sorted(path.name for path in out.glob("*.safetensors")) == ["fused-group-00000.safetensors"]
        tensors = _load_prepared_tensors(out)
        moe_prefix = f"{prefix}.moe_weights"
        torch.testing.assert_close(tensors[f"{moe_prefix}.gate"], gate.transpose(1, 2).float())
        torch.testing.assert_close(tensors[f"{moe_prefix}.up"], up.transpose(1, 2).float())
        torch.testing.assert_close(tensors[f"{moe_prefix}.down"], down.transpose(1, 2).float())

        index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
        assert set(index) == set(tensors)
        assert all((out / shard_name).is_file() for shard_name in index.values())

    def test_pipeline_splits_fused_experts_canonically_in_one_task(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe.experts"
        gate = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4)
        up = gate + 100
        gate_up = torch.cat((gate, up), dim=1)
        down = torch.arange(32, dtype=torch.float32).reshape(4, 4, 2)
        _write_sharded_safetensors_checkpoint(
            src,
            {
                "gate.safetensors": {f"{prefix}.gate_up_proj": gate_up},
                "down.safetensors": {f"{prefix}.down_proj": down},
            },
        )
        pipeline = CheckpointTransformPipeline(
            [
                MoEFusedExpertSplitCheckpointTransform,
                DtypeConversionCheckpointTransform,
            ]
        )

        plan, _ = pipeline.build_plan(
            src,
            torch.float32,
            config=SimpleNamespace(num_local_experts=4, model_type="mixtral"),
            hash_params={
                "moe_prefill_flavour": "expert_parallel",
                "moe_prefill_num_pipeline_stages": 2,
                "moe_prefill_num_parallelized_experts": 2,
            },
        )
        pipeline.apply(src, out, target_dtype=torch.float32, plan=plan, max_workers=1)

        assert len(plan.tasks) == 1
        task = plan.tasks[0]
        assert task.params.as_dict()["stages"] == (
            "split",
            "dtype",
            "final",
        )
        assert [stage.params.transform_id for stage in task.stages] == [
            MoEFusedExpertSplitCheckpointTransform.TRANSFORM_ID,
            DtypeConversionCheckpointTransform.TRANSFORM_ID,
        ]
        assert {ref.stage for ref in task.stages[0].output_refs} == {"canonical"}
        assert set(task.stages[1].input_refs) == set(task.stages[0].output_refs)
        assert {ref.stage for ref in task.stages[1].output_refs} == {"final"}

        tensors = _load_prepared_tensors(out)
        moe_prefix = "model.layers.0.block_sparse_moe.moe_weights"
        torch.testing.assert_close(tensors[f"{moe_prefix}.gate"], gate.transpose(1, 2))
        torch.testing.assert_close(tensors[f"{moe_prefix}.up"], up.transpose(1, 2))
        torch.testing.assert_close(tensors[f"{moe_prefix}.down"], down.transpose(1, 2))
        assert sorted(path.name for path in out.glob("*.safetensors")) == ["fused-group-00000.safetensors"]

    def test_fused_plan_rejects_missing_required_tensor_before_output(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe.experts"
        _write_safetensors_checkpoint(
            src,
            {f"{prefix}.gate_up_proj": torch.ones((1, 4, 4), dtype=torch.float16)},
        )
        pipeline = CheckpointTransformPipeline([MoEFusedExpertSplitCheckpointTransform])

        with pytest.raises(ValueError, match="missing required keys"):
            pipeline.build_plan(
                src,
                torch.float32,
                config=SimpleNamespace(num_local_experts=1, model_type="mixtral"),
            )

        assert not (tmp_path / "out").exists()

    def test_grouped_plan_rejects_missing_expert_projection_before_output(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe"
        _write_safetensors_checkpoint(
            src,
            {
                f"{prefix}.experts.0.w1.weight": torch.ones(2, 4),
                f"{prefix}.experts.0.w2.weight": torch.ones(4, 2),
            },
        )
        pipeline = CheckpointTransformPipeline(
            [MoEExpertStackingCheckpointTransform, DtypeConversionCheckpointTransform]
        )

        with pytest.raises(ValueError, match="missing one of"):
            pipeline.build_plan(
                src,
                torch.float32,
                config=SimpleNamespace(num_local_experts=1, model_type="mixtral"),
            )

        assert not (tmp_path / "out").exists()

    def test_prefill_and_decode_share_one_prepared_checkpoint(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe"
        tensors = {}
        for expert_index in range(4):
            tensors[f"{prefix}.experts.{expert_index}.w1.weight"] = torch.ones(2, 4)
            tensors[f"{prefix}.experts.{expert_index}.w3.weight"] = torch.ones(2, 4)
            tensors[f"{prefix}.experts.{expert_index}.w2.weight"] = torch.ones(4, 2)
        _write_safetensors_checkpoint(src, tensors)
        pipeline = CheckpointTransformPipeline(
            [MoEExpertStackingCheckpointTransform, DtypeConversionCheckpointTransform]
        )
        config = SimpleNamespace(num_local_experts=4, model_type="mixtral")

        from QEfficient.exporter.weight_free.export import _prepared_checkpoint_hash

        hash_params = [
            {"moe_prefill_flavour": "decode_bmm"},
            {
                "moe_prefill_flavour": "expert_parallel",
                "moe_prefill_num_pipeline_stages": 2,
                "moe_prefill_num_parallelized_experts": 2,
                "moe_prefill_expert_parallel_chunk_size": 64,
            },
            {
                "moe_prefill_flavour": "expert_parallel",
                "moe_prefill_num_pipeline_stages": 4,
                "moe_prefill_num_parallelized_experts": 1,
                "moe_prefill_expert_parallel_chunk_size": 128,
            },
        ]
        hashes = set()
        for params in hash_params:
            plan, active_group = pipeline.build_plan(src, torch.float32, config=config, hash_params=params)
            hashes.add(
                _prepared_checkpoint_hash(
                    model_ref=str(src),
                    target_dtype=torch.float32,
                    active_group_transform_id=active_group,
                    plan_payload=plan.fingerprint_payload(),
                )
            )
        assert len(hashes) == 1

    def test_spec_builder_rejects_weight_shape_that_differs_from_checkpoint(self):
        from QEfficient.exporter.weight_free.checkpoint_key_resolver import _check_stored_shape

        _check_stored_shape("w", [8, 4, 2], "w", [8, 4, 2])
        _check_stored_shape("w", ["batch", 4, 2], "w", [8, 4, 2])
        with pytest.raises(ValueError, match="stored layout"):
            _check_stored_shape("moe.gate", [4, 2, 4, 2], "moe.gate", [8, 4, 2])

    def test_safetensors_header_shapes_match_tensors(self, tmp_path):
        from QEfficient.exporter.weight_free.checkpoint_key_resolver import _safetensors_shapes

        _write_safetensors_checkpoint(tmp_path, {"a": torch.ones(3, 5), "b": torch.ones(7)})
        assert _safetensors_shapes(str(tmp_path / "model.safetensors")) == {"a": [3, 5], "b": [7]}

    @staticmethod
    def _weight_graph(source_dtype, target_dtype):
        weight = ir.Value(name="weight", shape=[2, 2], type=ir.TensorType(target_dtype))
        hidden = ir.Value(name="hidden", shape=[2, 2], type=ir.TensorType(ir.DataType.FLOAT))
        output = ir.Value(name="output", shape=[2, 2], type=ir.TensorType(ir.DataType.FLOAT))
        matmul = ir.Node(
            domain="",
            op_type="MatMul",
            inputs=[weight, hidden],
            outputs=[output],
            name="matmul",
            version=17,
        )
        graph = ir.Graph(
            inputs=[hidden],
            outputs=[output],
            nodes=[matmul],
            initializers=[weight],
            opset_imports={"": 17},
        )
        from QEfficient.exporter.weight_free.checkpoint_key_resolver import _promote_initializer

        _promote_initializer(SimpleNamespace(graph=graph), "weight", weight, source_dtype)
        return graph

    def test_original_bfloat16_weight_casts_to_fp16_graph_dtype(self):
        graph = self._weight_graph(ir.DataType.BFLOAT16, ir.DataType.FLOAT16)

        assert [(value.name, value.dtype) for value in graph.inputs] == [
            ("hidden", ir.DataType.FLOAT),
            ("weight", ir.DataType.BFLOAT16),
        ]
        assert [node.op_type for node in graph] == ["Cast", "MatMul"]
        cast, matmul = list(graph)
        assert cast.attributes["to"].value == int(ir.DataType.FLOAT16)
        assert matmul.inputs[0].name == cast.outputs[0].name
        assert not graph.initializers

    def test_original_bfloat16_weight_has_no_cast_for_bfloat16_graph_dtype(self):
        graph = self._weight_graph(ir.DataType.BFLOAT16, ir.DataType.BFLOAT16)

        assert [(value.name, value.dtype) for value in graph.inputs] == [
            ("hidden", ir.DataType.FLOAT),
            ("weight", ir.DataType.BFLOAT16),
        ]
        assert [node.op_type for node in graph] == ["MatMul"]
        assert list(graph)[0].inputs[0].name == "weight"
        assert not graph.initializers

    def test_scheduler_waits_for_staged_dependency(self, tmp_path):
        execution_order = []

        first = CheckpointTask(
            task_id="stack",
            input_refs=(TensorRef("expert", "raw"),),
            output_refs=(TensorRef("expert", "stacked"),),
            source_files=("model.safetensors",),
            output_file="unused-stacked.safetensors",
            estimated_peak_bytes=1,
            params=TaskParams("stack"),
            runner=lambda src, out, dtype: execution_order.append("stack") or {},
        )
        second = CheckpointTask(
            task_id="pack",
            input_refs=(TensorRef("expert", "stacked"),),
            output_refs=(TensorRef("expert", "final"),),
            source_files=(),
            output_file="final.safetensors",
            estimated_peak_bytes=1,
            params=TaskParams("pack"),
            runner=lambda src, out, dtype: execution_order.append("pack") or {"expert": "final.safetensors"},
        )
        plan = CheckpointPlan(
            tasks=[second, first],
            raw_refs={TensorRef("expert", "raw")},
            target_dtype=torch.float32,
            source_fingerprint=[],
            transform_ids=("stack", "pack"),
        )

        result = execute_checkpoint_plan(
            plan,
            tmp_path,
            tmp_path,
            torch.float32,
            max_ram_bytes=2,
            max_workers=2,
        )

        assert execution_order == ["stack", "pack"]
        assert result == {"expert": "final.safetensors"}

    def test_scheduler_limits_concurrent_task_memory(self, tmp_path):
        lock = threading.Lock()
        active = 0
        max_active = 0

        def make_runner(name):
            def runner(src, out, dtype):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
                return {name: f"{name}.safetensors"}

            return runner

        tasks = [
            CheckpointTask(
                task_id=name,
                input_refs=(TensorRef(name, "raw"),),
                output_refs=(TensorRef(name, "final"),),
                source_files=(f"{name}.safetensors",),
                output_file=f"{name}.safetensors",
                estimated_peak_bytes=6,
                params=TaskParams("test"),
                runner=make_runner(name),
            )
            for name in ("a", "b")
        ]
        plan = CheckpointPlan(
            tasks=tasks,
            raw_refs={TensorRef("a", "raw"), TensorRef("b", "raw")},
            target_dtype=torch.float32,
            source_fingerprint=[],
            transform_ids=("test",),
        )

        execute_checkpoint_plan(
            plan,
            tmp_path,
            tmp_path,
            torch.float32,
            max_ram_bytes=10,
            max_workers=2,
        )

        assert max_active == 1

    def test_scheduler_rejects_task_above_memory_cap(self, tmp_path):
        task = CheckpointTask(
            task_id="too-large",
            input_refs=(TensorRef("weight", "raw"),),
            output_refs=(TensorRef("weight", "final"),),
            source_files=("model.safetensors",),
            output_file="model.safetensors",
            estimated_peak_bytes=11,
            params=TaskParams("test"),
            runner=lambda src, out, dtype: {},
        )
        plan = CheckpointPlan(
            tasks=[task],
            raw_refs={TensorRef("weight", "raw")},
            target_dtype=torch.float32,
            source_fingerprint=[],
            transform_ids=("test",),
        )

        with pytest.raises(ValueError, match="requires at least"):
            execute_checkpoint_plan(
                plan,
                tmp_path,
                tmp_path,
                torch.float32,
                max_ram_bytes=10,
            )

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

        pipeline = CheckpointTransformPipeline([MoEFusedExpertSplitCheckpointTransform])
        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            config=SimpleNamespace(num_local_experts=2),
        )

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

        pipeline = CheckpointTransformPipeline([GraniteMoeFusedExpertSplitCheckpointTransform])
        pipeline.apply(
            src,
            out,
            target_dtype=torch.float32,
            config=SimpleNamespace(num_local_experts=1),
        )

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

    @pytest.mark.parametrize(
        ("onnx_name", "checkpoint_name"),
        [
            ("model.layers.0.mlp.gate.weight", "model.layers.0.mlp.router.weight"),
            ("model.layers.0.mlp.router.weight", "model.layers.0.mlp.gate.weight"),
        ],
    )
    def test_resolver_accepts_router_gate_aliases(self, onnx_name, checkpoint_name):
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        assert find_checkpoint_key(onnx_name, {checkpoint_name: "model.safetensors"}, backbone) == checkpoint_name

    def test_resolver_prefers_exact_router_gate_name(self):
        checkpoint_index = {
            "model.layers.0.mlp.gate.weight": "model.safetensors",
            "model.layers.0.mlp.router.weight": "model.safetensors",
        }
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        assert (
            find_checkpoint_key("model.layers.0.mlp.gate.weight", checkpoint_index, backbone)
            == "model.layers.0.mlp.gate.weight"
        )

    def test_resolver_combines_router_gate_alias_with_active_transform(self):
        checkpoint_name = "model.layers.0.block_sparse_moe.router.weight"
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        assert (
            find_checkpoint_key(
                "model.layers.0.mlp.gate.weight",
                {checkpoint_name: "model.safetensors"},
                backbone,
                MoEExpertStackingCheckpointTransform,
            )
            == checkpoint_name
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
        assert spec.external_data_root == str(tmp_path)


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

    def test_rejects_missing_cache_writer_without_partial_rewire(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=1, scatter_count_per_fn=1)
        with pytest.raises(ValueError, match="Could not uniquely resolve"):
            PreserveNestedCacheRetainedStateTransform.apply(model)
        fn = model.functions[0]
        assert len(fn.output) == 1, f"Function with 1 scatter should not have outputs added, got {list(fn.output)}"
        assert not model.graph.node[0].output, "Function call must not be partially rewired"

    def test_uses_cache_input_lineage_when_moe_scatter_is_present(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(
            num_layers=1,
            scatter_count_per_fn=2,
            include_unrelated_scatter=True,
        )

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)

        assert changed
        fn = model.functions[0]
        call_node = model.graph.node[0]
        assert "moe_scatter_0" not in fn.output
        assert fn.output[-2:] == ["scatter_0_0", "scatter_0_1"]
        assert call_node.output[-2:] == ["past_key.0_RetainedState", "past_value.0_RetainedState"]


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
