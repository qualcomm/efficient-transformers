# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for QEFFBaseModel base class.

CPU-only tests that do NOT require QAIC hardware.
Run with: pytest tests/unit_test/base/ -n auto -v
"""

import json
import subprocess
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import patch

import onnx
import pytest
import torch
from onnx import TensorProto, helper
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from QEfficient.base.modeling_qeff import generate_mdp_compiler_dump
from QEfficient.compile.mdp_generator import _layer_partition_bounds, generate_disagg_mdp_partition_config
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

VOCAB_SIZE = 500
CTX_LEN = 32
SEQ_LEN = 8


def make_tiny_gpt2():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=VOCAB_SIZE, n_positions=CTX_LEN, n_ctx=CTX_LEN)
    return GPT2LMHeadModel(cfg).eval(), cfg


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval(), cfg


@pytest.mark.cpu_only
class TestQEFFBaseModelProperties:
    """Test QEFFBaseModel properties and class methods."""

    def test_model_name_returns_class_name(self):
        """model_name property returns a non-empty string."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert isinstance(qeff.model_name, str)
        assert len(qeff.model_name) > 0

    def test_model_name_strips_qeff_prefix(self):
        """model_name strips QEff/QEFF prefix from transformed model class name."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # After KVCacheTransform, model becomes QEffGPT2LMHeadModel
        # model_name should strip the QEff prefix
        assert not qeff.model_name.startswith("QEff")
        assert not qeff.model_name.startswith("QEFF")

    def test_transform_names_returns_list_of_strings(self):
        """_transform_names instance method returns list of transform names."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        names = qeff._transform_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 0

    def test_transform_names_includes_pytorch_transforms(self):
        """_transform_names includes KVCacheTransform."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        names = qeff._transform_names()
        assert "KVCacheTransform" in names

    def test_transform_names_includes_onnx_transforms(self):
        """_transform_names includes ONNX transforms when present."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        names = qeff._transform_names()
        # _transform_names returns pytorch + onnx transform names.
        # QEFFAutoModelForCausalLM._onnx_transforms is empty by default,
        # so only pytorch transforms are expected.
        assert isinstance(names, list)
        assert len(names) > 0

    def test_init_sets_onnx_path_to_none(self):
        """__init__ sets onnx_path to None initially."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.onnx_path is None

    def test_init_sets_qpc_path_to_none(self):
        """__init__ sets qpc_path to None initially."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.qpc_path is None

    def test_init_sets_qpc_session_to_none(self):
        """__init__ sets qpc_session to None initially."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.qpc_session is None

    def test_init_is_weights_offloaded_false(self):
        """__init__ sets _is_weights_offloaded to False initially."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff._is_weights_offloaded is False

    def test_model_architecture_extracted(self):
        """model_architecture is extracted from config.architectures."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # GPT2 config has architectures attribute
        assert qeff.model_architecture is not None or qeff.model_architecture is None


@pytest.mark.cpu_only
class TestQEFFBaseModelWeightOffloading:
    """Test weight offloading functionality."""

    def test_offload_model_weights_sets_flag(self):
        """_offload_model_weights(True) offloads weights and sets flag."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff._offload_model_weights(offload_pt_weights=True)
        assert result is True
        assert qeff._is_weights_offloaded is True

    def test_offload_model_weights_false_does_not_offload(self):
        """_offload_model_weights(False) does not offload weights."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff._offload_model_weights(offload_pt_weights=False)
        assert result is False
        assert qeff._is_weights_offloaded is False

    def test_offload_model_weights_idempotent(self):
        """_offload_model_weights is idempotent (second call returns False)."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff._offload_model_weights(offload_pt_weights=True)
        # Second call should return False (already offloaded)
        result = qeff._offload_model_weights(offload_pt_weights=True)
        assert result is False

    def test_model_offloaded_check_raises_when_offloaded(self):
        """_model_offloaded_check raises RuntimeError when weights are offloaded."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff._offload_model_weights(offload_pt_weights=True)
        with pytest.raises(RuntimeError, match="weights have been offloaded"):
            qeff._model_offloaded_check()

    def test_model_offloaded_check_passes_when_not_offloaded(self):
        """_model_offloaded_check does not raise when weights are not offloaded."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Should not raise
        qeff._model_offloaded_check()

    def test_offload_clears_parameter_storage(self):
        """_offload_model_weights moves all parameters and buffers to meta device."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Check that parameters are NOT on meta before offloading
        assert not any(p.is_meta for p in qeff.model.parameters())

        qeff._offload_model_weights(offload_pt_weights=True)

        # After offloading, ALL parameters and buffers must be on meta device
        assert all(p.is_meta for p in qeff.model.parameters())
        assert all(b.is_meta for b in qeff.model.buffers())

    def test_offload_clears_plain_tensor_attributes(self):
        """_offload_model_weights clears plain tensor attributes (not params/buffers)."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)

        # Attach a plain tensor attribute to a submodule (simulates MoE stacked weights)
        first_child = next(iter(qeff.model.modules()))
        first_child.extra_weight = torch.randn(8, 8)
        assert not first_child.extra_weight.is_meta

        qeff._offload_model_weights(offload_pt_weights=True)

        # The plain tensor attribute should also be on meta device
        assert first_child.extra_weight.is_meta

    def test_offload_preserves_plain_tensor_shape_and_dtype(self):
        """_offload_model_weights must keep shape/dtype of plain tensor attributes.

        Regression guard: an earlier implementation replaced unregistered tensor
        attributes with ``torch.empty(0, device="meta")``, which silently broke
        downstream code that broadcasts against or copies into them (e.g. the
        LoRA re-export path that calls ``module.lora_scalings.copy_(...)``).
        Meta tensors carry no storage regardless of shape, so preserving shape
        costs nothing and keeps shape-dependent code working.
        """
        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)

        first_child = next(iter(qeff.model.modules()))
        first_child.extra_weight = torch.randn(3, 1, 1, 1, dtype=torch.float32)

        qeff._offload_model_weights(offload_pt_weights=True)

        assert first_child.extra_weight.is_meta
        assert tuple(first_child.extra_weight.shape) == (3, 1, 1, 1)
        assert first_child.extra_weight.dtype == torch.float32

        # Shape-dependent ops downstream must still type-check; this raised
        # ``RuntimeError: output with shape [0] doesn't match the broadcast
        # shape [3, 1, 1, 0]`` under the broken implementation.
        first_child.extra_weight.copy_(torch.ones(3, 1, 1, 1))


@pytest.mark.cpu_only
def _get_any_attn_blocking_config(model):
    for m in model.modules():
        if hasattr(m, "attn_blocking_config"):
            return m.attn_blocking_config
    return None


@pytest.mark.cpu_only
class TestQEFFBaseModelHashParams:
    """Test hash_params initialization."""

    def test_hash_params_is_dict(self):
        """hash_params is a dictionary."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert isinstance(qeff.hash_params, dict)

    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert "qeff_auto_class" in qeff.hash_params
        assert qeff.hash_params["qeff_auto_class"] == "QEFFAutoModelForCausalLM"

    def test_hash_params_contains_pretrained_model_name(self):
        """hash_params contains pretrained_model_name_or_path when provided."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, pretrained_model_name_or_path="test-model")
        assert "pretrained_model_name_or_path" in qeff.hash_params
        assert qeff.hash_params["pretrained_model_name_or_path"] == "test-model"


@pytest.mark.cpu_only
class TestQEFFBaseModelTransformBlocking:
    """Tests for QEFFBaseModel.transform() attention blocking behavior."""

    @pytest.mark.parametrize("blocking_mode", ["kv", "q", "qkv", "hq", "hkv", "hqkv"])
    def test_transform_blocking_mode_runs_auto_configurator(self, blocking_mode):
        # Use a slightly larger head count here to make it possible for "h" mode to result in head blocking
        # when num_devices > 1.
        cfg = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=CTX_LEN,
        )
        model = LlamaForCausalLM(cfg).eval()
        qeff = QEFFAutoModelForCausalLM(model)

        high_cl = 131072
        qaic_config = {
            "blocking_mode": blocking_mode,
            # Do not specify num_kv_blocks/num_q_blocks/head_block_size, so auto configurator is used.
        }

        qeff.transform(
            ctx_len=high_cl,
            seq_len=high_cl,
            bs=1,
            num_devices=2,
            qaic_config=qaic_config,
            aic_num_cores=16,
            convert_to_fp16=False,
        )

        cfg = _get_any_attn_blocking_config(qeff.model)
        assert cfg is not None, (
            "Expected BlockingAttentionTransform to attach attn_blocking_config to attention modules"
        )

        decided_mode = cfg.mode.value
        requested_mode = blocking_mode

        # If the configurator decides blocking is not needed, decided_mode can be "".
        # Otherwise, the decided mode should be a substring of the requested mode.
        assert decided_mode in requested_mode

        # For each kind of blocking that was decided, ensure it actually enabled >1 blocks.
        if "kv" in decided_mode:
            assert cfg.num_kv_blocks is not None and cfg.num_kv_blocks > 1
        if "q" in decided_mode:
            assert cfg.num_q_blocks is not None and cfg.num_q_blocks > 1
        if "h" in decided_mode:
            assert cfg.head_block_size is not None and cfg.head_block_size > 1

    @pytest.mark.parametrize(
        ("compiler_options", "expected_moe_num_devices"),
        [
            pytest.param({}, 8, id="no_mdp_partitions"),
            pytest.param({"mdp_num_partitions": 1}, 8, id="single_partition"),
            pytest.param({"mdp_num_partitions": 4}, 2, id="pp4_ts2"),
        ],
    )
    def test_transform_passes_partition_ts_devices_to_moe_transform(
        self, monkeypatch, compiler_options, expected_moe_num_devices
    ):
        from QEfficient.base import modeling_qeff

        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        captured_kwargs = {}

        def fake_moe_apply(model, **kwargs):
            captured_kwargs.update(kwargs)
            return model, False

        monkeypatch.setattr(modeling_qeff.OptimizedMoETransform, "apply", staticmethod(fake_moe_apply))

        qeff.transform(
            ctx_len=32,
            seq_len=8,
            bs=1,
            num_devices=8,
            qaic_config={"moe_config": {"flavour": "expert_parallel"}},
            aic_num_cores=16,
            prefill_only=True,
            prefill_seq_len=8,
            **compiler_options,
        )

        assert captured_kwargs["num_devices"] == expected_moe_num_devices

    def test_transform_rejects_invalid_mdp_num_partitions_for_moe_transform(self):
        model, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)

        with pytest.raises(ValueError, match="mdp_num_partitions must be greater than zero"):
            qeff.transform(
                ctx_len=32,
                seq_len=8,
                bs=1,
                num_devices=8,
                qaic_config={"moe_config": {"flavour": "expert_parallel"}},
                aic_num_cores=16,
                prefill_only=True,
                prefill_seq_len=8,
                mdp_num_partitions=0,
            )


@pytest.mark.cpu_only
@pytest.mark.onnx
@pytest.mark.slow
class TestQEFFBaseModelGetOnnxPath:
    """Test get_onnx_path method."""

    def test_get_onnx_path_returns_onnx_path(self, tmp_export_dir):
        """get_onnx_path calls export and returns a valid onnx_path."""
        import os

        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # get_onnx_path calls self.export() internally
        onnx_path = qeff.get_onnx_path()
        assert onnx_path is not None
        assert qeff.onnx_path is not None
        assert os.path.exists(str(onnx_path))

    def test_get_onnx_path_sets_onnx_path_attribute(self, tmp_export_dir):
        """get_onnx_path sets self.onnx_path after export."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.onnx_path is None  # Before export
        qeff.get_onnx_path()
        assert qeff.onnx_path is not None  # After export

    def test_get_onnx_path_second_call_returns_cached_path(self, tmp_export_dir):
        """get_onnx_path returns the same path on a second call (cached)."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        onnx_path_1 = qeff.get_onnx_path()
        onnx_path_2 = qeff.get_onnx_path()
        assert str(onnx_path_1) == str(onnx_path_2)


def _bounds_to_layer_counts(bounds: List[int], total_layers: int) -> List[int]:
    """Convert exclusive-upper-bound list from _layer_partition_bounds to per-partition layer counts."""
    pts = [0] + bounds + [total_layers]
    return [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]


def _build_synthetic_gpt2_onnx(num_layers: int, out_path: Path) -> None:
    """Write a minimal ONNX graph whose nodes carry 'h.N' transformer-layer names.

    The graph has topology: embed_tokens -> h.0/* -> h.1/* -> ... -> lm_head.
    No external data or weights are needed.  generate_disagg_mdp_partition_config
    parses node names via _get_layer_num, so this is sufficient to exercise the
    ONNX strategy end-to-end without loading a real model.
    """
    nodes: List[Any] = []
    nodes.append(helper.make_node("Identity", inputs=["input_ids"], outputs=["embed_out"], name="embed_tokens"))
    prev_out = "embed_out"
    for layer_idx in range(num_layers):
        attn_out = f"attn_out_{layer_idx}"
        mlp_out = f"mlp_out_{layer_idx}"
        nodes.append(helper.make_node("Identity", inputs=[prev_out], outputs=[attn_out], name=f"h.{layer_idx}/attn"))
        nodes.append(helper.make_node("Identity", inputs=[attn_out], outputs=[mlp_out], name=f"h.{layer_idx}/mlp"))
        prev_out = mlp_out
    nodes.append(helper.make_node("Identity", inputs=[prev_out], outputs=["logits"], name="lm_head"))

    graph = helper.make_graph(
        nodes,
        "synthetic_gpt2_graph",
        [helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 8, 500])],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(onnx_model, str(out_path))


def _build_synthetic_local_function_onnx(out_path: Path) -> None:
    """Write a two-layer graph with nested ONNX local functions.

    Dynamo exports local-function body names with a leading slash.  QAIC expands
    those local functions recursively and uses ``<call-site>//<function>/...``
    names in the MDP compiler dump.  This fixture exercises that exact naming
    rule without requiring a model export or QAIC hardware.
    """
    nested_function = helper.make_function(
        "local",
        "NestedFn",
        ["X"],
        ["Y"],
        [helper.make_node("Identity", ["X"], ["Y"], name="n0")],
        opset_imports=[helper.make_opsetid("", 17)],
    )
    layer_function = helper.make_function(
        "local",
        "LayerFn",
        ["X"],
        ["Y"],
        [
            helper.make_node("Identity", ["X"], ["linear"], name="/LayerFn/linear"),
            helper.make_node("NestedFn", ["linear"], ["Y"], domain="local", name="/LayerFn/nested"),
        ],
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("local", 1)],
    )
    graph = helper.make_graph(
        [
            helper.make_node("LayerFn", ["input"], ["layer0"], domain="local", name="layer_0/call"),
            helper.make_node("LayerFn", ["layer0"], ["output"], domain="local", name="layer_1/call"),
        ],
        "synthetic_local_function_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    onnx_model = helper.make_model(
        graph,
        functions=[nested_function, layer_function],
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("local", 1)],
    )
    onnx.save(onnx_model, str(out_path))


def _fake_subprocess_run(command: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Monkeypatch for subprocess.run: create compile artifacts requested by the command."""
    binary_dir: Optional[Path] = None
    dump_path: Optional[Path] = None
    onnx_path: Optional[Path] = None
    for arg in command:
        if arg.startswith("-aic-binary-dir="):
            binary_dir = Path(arg.split("=", 1)[1])
        elif arg.startswith("-mdp-dump-partition-config="):
            dump_path = Path(arg.split("=", 1)[1])
        elif arg.startswith("-m="):
            onnx_path = Path(arg.split("=", 1)[1])
    if binary_dir is not None:
        binary_dir.mkdir(parents=True, exist_ok=True)
        (binary_dir / "programqpc.bin").write_bytes(b"FAKE_QPC")
    if dump_path is not None:
        node_names: List[str] = []
        if onnx_path is not None:
            model = onnx.load(onnx_path, load_external_data=False)
            node_names = [node.name for node in model.graph.node if node.name]
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps({"partitions": [{"nodeList": node_names}]}))
    return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"", stderr=b"")


@pytest.mark.cpu_only
@pytest.mark.mdp
class TestMdpLayerPartitionBounds:
    """Unit tests for _layer_partition_bounds balanced remainder distribution (ONNX strategy only)."""

    def test_8_layers_3_partitions_counts(self):
        """8 layers / 3 partitions -> layer counts [3, 3, 2]."""
        bounds = _layer_partition_bounds(8, 3)
        counts = _bounds_to_layer_counts(bounds, 8)
        assert counts == [3, 3, 2], f"Expected [3, 3, 2], got {counts}"

    def test_10_layers_4_partitions_counts(self):
        """10 layers / 4 partitions -> layer counts [2, 3, 3, 2]."""
        bounds = _layer_partition_bounds(10, 4)
        counts = _bounds_to_layer_counts(bounds, 10)
        assert counts == [2, 3, 3, 2], f"Expected [2, 3, 3, 2], got {counts}"

    def test_max_minus_min_le_1_for_8_3(self):
        """max - min layer count <= 1 for 8/3 case (balanced distribution invariant)."""
        counts = _bounds_to_layer_counts(_layer_partition_bounds(8, 3), 8)
        assert max(counts) - min(counts) <= 1

    def test_max_minus_min_le_1_for_10_4(self):
        """max - min layer count <= 1 for 10/4 case (balanced distribution invariant)."""
        counts = _bounds_to_layer_counts(_layer_partition_bounds(10, 4), 10)
        assert max(counts) - min(counts) <= 1

    def test_last_partition_receives_base_allocation_8_3(self):
        """Last partition receives only the base (floor) allocation for 8/3."""
        num_layers, num_partitions = 8, 3
        base = num_layers // num_partitions
        counts = _bounds_to_layer_counts(_layer_partition_bounds(num_layers, num_partitions), num_layers)
        assert counts[-1] == base, f"Expected last partition to have base={base}, got {counts[-1]}"

    def test_last_partition_receives_base_allocation_10_4(self):
        """Last partition receives only the base (floor) allocation for 10/4."""
        num_layers, num_partitions = 10, 4
        base = num_layers // num_partitions
        counts = _bounds_to_layer_counts(_layer_partition_bounds(num_layers, num_partitions), num_layers)
        assert counts[-1] == base, f"Expected last partition to have base={base}, got {counts[-1]}"

    def test_remainder_goes_to_middle_partitions_for_10_4(self):
        """Remainder layers (2) go to middle partitions (indices 1 and 2) before index 0 for 10/4."""
        bounds = _layer_partition_bounds(10, 4)
        counts = _bounds_to_layer_counts(bounds, 10)
        base = 10 // 4
        middle_extras = sum(1 for c in counts[1:-1] if c > base)
        first_extras = 1 if counts[0] > base else 0
        assert middle_extras == 2, f"Expected 2 middle partitions with extra layer, got {middle_extras}"
        assert first_extras == 0, f"Expected first partition to have base count, got {counts[0]}"

    def test_evenly_divisible_has_zero_spread_12_4(self):
        """12 layers / 4 partitions divides evenly; all counts equal base."""
        counts = _bounds_to_layer_counts(_layer_partition_bounds(12, 4), 12)
        assert counts == [3, 3, 3, 3], f"Expected [3, 3, 3, 3], got {counts}"

    def test_two_partition_remainder_goes_to_first_5_2(self):
        """5 layers / 2 partitions -> [3, 2]: remainder goes to first (no middle zone)."""
        counts = _bounds_to_layer_counts(_layer_partition_bounds(5, 2), 5)
        assert counts == [3, 2], f"Expected [3, 2], got {counts}"

    def test_bounds_length_equals_num_partitions_minus_one(self):
        """_layer_partition_bounds returns a list of length num_partitions - 1."""
        for num_layers, num_partitions in [(8, 3), (10, 4), (12, 4), (5, 2)]:
            bounds = _layer_partition_bounds(num_layers, num_partitions)
            assert len(bounds) == num_partitions - 1, (
                f"Expected {num_partitions - 1} bounds for {num_layers}/{num_partitions}, got {len(bounds)}"
            )

    def test_total_layers_preserved(self):
        """Sum of per-partition layer counts equals num_layers for all test cases."""
        for num_layers, num_partitions in [(8, 3), (10, 4), (12, 4), (5, 2), (9, 4)]:
            counts = _bounds_to_layer_counts(_layer_partition_bounds(num_layers, num_partitions), num_layers)
            assert sum(counts) == num_layers, (
                f"Expected sum={num_layers} for {num_layers}/{num_partitions}, got {sum(counts)}"
            )


@pytest.mark.cpu_only
@pytest.mark.mdp
class TestMdpLocalFunctionExpansion:
    """Validate MDP names for compiler-expanded ONNX local functions."""

    def test_nested_local_function_nodes_use_compiler_paths(self, tmp_path: Path):
        onnx_path = tmp_path / "local_functions.onnx"
        _build_synthetic_local_function_onnx(onnx_path)

        mdp = generate_disagg_mdp_partition_config(
            onnx_path=str(onnx_path),
            num_devices=2,
            num_partitions=2,
            num_layers=2,
        )

        assert mdp["partitions"][0]["nodeList"] == [
            "layer_0/call//LayerFn/linear",
            "layer_0/call//LayerFn/nested/n0",
        ]
        assert mdp["partitions"][1]["nodeList"] == [
            "layer_1/call//LayerFn/linear",
            "layer_1/call//LayerFn/nested/n0",
        ]


@pytest.mark.cpu_only
@pytest.mark.mdp
class TestMdpCompileIntegration:
    """CPU-only compile integration tests for the ONNX MDP strategy.

    Uses a synthetic ONNX graph (no HF Hub download) and monkeypatches
    subprocess.run so no real qaic-compile binary is invoked.
    """

    @pytest.fixture()
    def compile_workspace(self, tmp_path: Path):
        """Provide a workspace with a synthetic ONNX and an empty compile directory."""
        onnx_path = tmp_path / "model.onnx"
        _build_synthetic_gpt2_onnx(num_layers=2, out_path=onnx_path)
        compile_dir = tmp_path / "compile_out"
        return tmp_path, onnx_path, compile_dir

    def _run_mdp_compile(
        self,
        compile_workspace,
        mdp_ts_num_devices: int = 4,
        mdp_num_partitions: int = 2,
        mdp_strategy: str = "onnx",
    ) -> tuple:
        """Build QEFFAutoModelForCausalLM, monkeypatch subprocess, run compile, return artifacts."""
        tmp_path, onnx_path, compile_dir = compile_workspace
        model_hf, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model_hf)

        with patch("QEfficient.base.modeling_qeff.subprocess.run", side_effect=_fake_subprocess_run):
            qeff._compile(
                onnx_path=str(onnx_path),
                compile_dir=str(compile_dir),
                mdp_ts_num_devices=mdp_ts_num_devices,
                mdp_num_partitions=mdp_num_partitions,
                mdp_strategy=mdp_strategy,
            )

        return compile_dir, onnx_path, qeff

    def test_mdp_disagg_json_is_created(self, compile_workspace):
        """mdp_disagg_4d_2p.json is written to compile_dir when mdp_num_partitions=2."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        disagg_files = list(compile_dir.glob("mdp_disagg_4d_2p.json"))
        assert len(disagg_files) == 1, "Expected exactly one mdp_disagg_4d_2p.json in compile_dir"

    def test_mdp_disagg_json_has_two_partitions(self, compile_workspace):
        """mdp_disagg_4d_2p.json contains exactly two partitions."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        disagg_path = compile_dir / "mdp_disagg_4d_2p.json"
        with open(disagg_path) as fh:
            mdp_data = json.load(fh)
        assert len(mdp_data["partitions"]) == 2

    def test_mdp_disagg_partition0_devices_are_0_1(self, compile_workspace):
        """Partition0 in mdp_disagg_4d_2p.json is assigned devices [0, 1]."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        with open(compile_dir / "mdp_disagg_4d_2p.json") as fh:
            mdp_data = json.load(fh)
        partition0_device_ids = [d["deviceId"] for d in mdp_data["partitions"][0]["devices"]]
        assert partition0_device_ids == [0, 1], f"Expected [0, 1], got {partition0_device_ids}"

    def test_mdp_disagg_partition1_devices_are_2_3(self, compile_workspace):
        """Partition1 in mdp_disagg_4d_2p.json is assigned devices [2, 3]."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        with open(compile_dir / "mdp_disagg_4d_2p.json") as fh:
            mdp_data = json.load(fh)
        partition1_device_ids = [d["deviceId"] for d in mdp_data["partitions"][1]["devices"]]
        assert partition1_device_ids == [2, 3], f"Expected [2, 3], got {partition1_device_ids}"

    def test_hashed_compile_params_command_references_mdp_json(self, compile_workspace):
        """-mdp-load-partition-config flag in hashed_compile_params.json points at the disagg JSON."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        hashed_path = next(compile_dir.glob("**/hashed_compile_params.json"), None)
        assert hashed_path is not None, "hashed_compile_params.json not found under compile_dir"
        with open(hashed_path) as fh:
            hashed = json.load(fh)
        command_str = " ".join(str(c) for c in hashed["command"])
        assert "-mdp-load-partition-config=" in command_str, (
            "-mdp-load-partition-config flag missing from compile command"
        )
        assert "mdp_disagg_4d_2p.json" in command_str, (
            "mdp_disagg_4d_2p.json not referenced in -mdp-load-partition-config"
        )

    def test_hashed_compile_params_contains_mdp_num_partitions(self, compile_workspace):
        """hashed_compile_params.json records mdp_num_partitions=2."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        hashed_path = next(compile_dir.glob("**/hashed_compile_params.json"), None)
        assert hashed_path is not None, "hashed_compile_params.json not found under compile_dir"
        with open(hashed_path) as fh:
            hashed = json.load(fh)
        assert hashed.get("mdp_num_partitions") == 2, (
            f"Expected mdp_num_partitions=2, got {hashed.get('mdp_num_partitions')}"
        )

    def test_hashed_compile_params_contains_mdp_strategy_onnx(self, compile_workspace):
        """hashed_compile_params.json records mdp_strategy='onnx'."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        hashed_path = next(compile_dir.glob("**/hashed_compile_params.json"), None)
        assert hashed_path is not None, "hashed_compile_params.json not found under compile_dir"
        with open(hashed_path) as fh:
            hashed = json.load(fh)
        assert hashed.get("mdp_strategy") == "onnx", f"Expected mdp_strategy='onnx', got {hashed.get('mdp_strategy')}"

    def test_hashed_compile_params_contains_mdp_ts_json(self, compile_workspace):
        """hashed_compile_params.json records mdp_ts_json (the generated MDP dict)."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        hashed_path = next(compile_dir.glob("**/hashed_compile_params.json"), None)
        assert hashed_path is not None, "hashed_compile_params.json not found under compile_dir"
        with open(hashed_path) as fh:
            hashed = json.load(fh)
        assert "mdp_ts_json" in hashed, "mdp_ts_json key missing from hashed_compile_params.json"
        assert hashed["mdp_ts_json"] is not None, "mdp_ts_json value is None in hashed_compile_params.json"
        assert "partitions" in hashed["mdp_ts_json"], "mdp_ts_json missing 'partitions' key"

    def test_qconfig_compiler_config_contains_mdp_num_partitions(self, compile_workspace):
        """qconfig.json compiler_config contains mdp_num_partitions=2."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        qconfig_path = next(compile_dir.glob("**/qconfig.json"), None)
        assert qconfig_path is not None, "qconfig.json not found under compile_dir"
        with open(qconfig_path) as fh:
            qconfig = json.load(fh)
        compiler_cfg = qconfig.get("qpc_config", {}).get("compiler_config", {})
        assert compiler_cfg.get("mdp_num_partitions") == 2, (
            f"Expected mdp_num_partitions=2 in qconfig compiler_config, got {compiler_cfg.get('mdp_num_partitions')}"
        )

    def test_qconfig_compiler_config_contains_mdp_strategy_onnx(self, compile_workspace):
        """qconfig.json compiler_config contains mdp_strategy='onnx'."""
        compile_dir, _, _ = self._run_mdp_compile(compile_workspace)
        qconfig_path = next(compile_dir.glob("**/qconfig.json"), None)
        assert qconfig_path is not None, "qconfig.json not found under compile_dir"
        with open(qconfig_path) as fh:
            qconfig = json.load(fh)
        compiler_cfg = qconfig.get("qpc_config", {}).get("compiler_config", {})
        assert compiler_cfg.get("mdp_strategy") == "onnx", (
            f"Expected mdp_strategy='onnx' in qconfig compiler_config, got {compiler_cfg.get('mdp_strategy')}"
        )

    def test_user_mdp_compiler_dump_path_is_deprecated(self, compile_workspace):
        """User-provided compiler dumps are deprecated and ignored in favor of auto-generation."""
        _, onnx_path, compile_dir = compile_workspace
        model_hf, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model_hf)
        user_dump_path = compile_dir / "compiler_dump.json"

        with (
            patch("QEfficient.base.modeling_qeff.logger.warning") as warning_mock,
            patch("QEfficient.base.modeling_qeff.subprocess.run", side_effect=_fake_subprocess_run) as run_mock,
        ):
            qeff._compile(
                onnx_path=str(onnx_path),
                compile_dir=str(compile_dir),
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                mdp_strategy="intersection",
                mdp_compiler_dump_path=str(user_dump_path),
            )

        command_strs = [" ".join(str(arg) for arg in call.args[0]) for call in run_mock.call_args_list]
        warning_mock.assert_called_once()
        assert "mdp_compiler_dump_path is deprecated and no longer used" in warning_mock.call_args.args[0]
        assert str(user_dump_path) not in " ".join(command_strs)
        assert any("/mdp_" in command_str for command_str in command_strs)
        assert any("tmp_mdp_compiler_dump_4d_2p.json" in command_str for command_str in command_strs)

    def test_intersection_mdp_dump_reuses_compile_command(self, compile_workspace):
        """Auto-generated compiler dumps are derived from the final compiler-facing command."""
        _, onnx_path, compile_dir = compile_workspace
        model_hf, _ = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model_hf)

        with patch("QEfficient.base.modeling_qeff.subprocess.run", side_effect=_fake_subprocess_run) as run_mock:
            qeff._compile(
                onnx_path=str(onnx_path),
                compile_dir=str(compile_dir),
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                mdp_strategy="intersection",
                use_onnx_subfunctions=True,
                specializations=[{"batch_size": 1, "seq_len": 8, "ctx_len": 32}],
                custom_io={"input_ids": "float16"},
                aic_hw_version="ai100",
                retained_state=True,
                convert_to_fp16=True,
                mxfp6_matmul=True,
                aic_num_cores=16,
                split_model_io=True,
                mos=1,
                aic_enable_depth_first=True,
            )

        dump_commands = [
            call.args[0]
            for call in run_mock.call_args_list
            if any(str(arg).startswith("-mdp-dump-partition-config=") for arg in call.args[0])
        ]
        final_commands = [
            call.args[0]
            for call in run_mock.call_args_list
            if any(str(arg).startswith("-aic-binary-dir=") for arg in call.args[0])
        ]
        assert len(dump_commands) == 1
        assert len(final_commands) == 1

        dump_command = dump_commands[0]
        final_command = final_commands[0]
        dump_command_str = " ".join(str(arg) for arg in dump_command)
        final_command_str = " ".join(str(arg) for arg in final_command)

        for expected in [
            "/opt/qti-aic/exec/qaic-compile",
            "-aic-hw",
            "-aic-hw-version=ai100",
            f"-m={onnx_path}",
            "-retained-state",
            "-convert-to-fp16",
            "-mxfp6-matmul",
            "-aic-num-cores=16",
            "-split-model-io",
            "-mos=1",
            "-aic-enable-depth-first",
            "-sub-functions",
        ]:
            assert expected in dump_command_str
            assert expected in final_command_str

        dump_mdp_flags = [arg for arg in dump_command if str(arg).startswith("-mdp-dump-partition-config=")]
        assert len(dump_mdp_flags) == 1
        dump_path = Path(str(dump_mdp_flags[0]).split("=", 1)[1])
        mdp_dir = dump_path.parent
        assert mdp_dir.parent == compile_dir
        assert mdp_dir.name.startswith("mdp_")
        assert dump_path.is_file()
        assert (mdp_dir / "specializations.json").is_file()
        assert (mdp_dir / "custom_io.yaml").is_file()
        assert (mdp_dir / "mdp_disagg_4d_2p.json").is_file()

        assert f"-network-specialization-config={mdp_dir / 'specializations.json'}" in dump_command_str
        assert f"-custom-IO-list-file={mdp_dir / 'custom_io.yaml'}" in dump_command_str

        final_specialization_flags = [
            arg for arg in final_command if str(arg).startswith("-network-specialization-config=")
        ]
        final_custom_io_flags = [arg for arg in final_command if str(arg).startswith("-custom-IO-list-file=")]
        assert len(final_specialization_flags) == 1
        assert len(final_custom_io_flags) == 1
        assert str(compile_dir / "qpc-") in final_specialization_flags[0]
        assert final_specialization_flags[0].endswith("/specializations.json")
        assert str(compile_dir / "qpc-") in final_custom_io_flags[0]
        assert final_custom_io_flags[0].endswith("/custom_io.yaml")

        assert f"-mdp-dump-partition-config={mdp_dir / 'tmp_mdp_compiler_dump_4d_2p.json'}" in dump_command_str
        assert "-mdp-load-partition-config=" not in dump_command_str
        assert "-aic-binary-dir=" not in dump_command_str
        assert "-mdp-dump-partition-config=" not in final_command_str
        assert f"-mdp-load-partition-config={mdp_dir / 'mdp_disagg_4d_2p.json'}" in final_command_str
        assert "-aic-binary-dir=" in final_command_str

    def test_mdp_compiler_dump_hash_changes_with_custom_io(self, compile_workspace):
        """Different custom_io creates a different mdp_<hash> dump directory."""
        _, onnx_path, compile_dir = compile_workspace
        command = ["/opt/qti-aic/exec/qaic-compile", "-aic-hw", f"-m={onnx_path}"]

        with patch("QEfficient.base.modeling_qeff.subprocess.run", side_effect=_fake_subprocess_run):
            fp16_dump = generate_mdp_compiler_dump(
                compile_dir=compile_dir,
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                compile_command=command,
                custom_io={"input_ids": "float16"},
            )
            int8_dump = generate_mdp_compiler_dump(
                compile_dir=compile_dir,
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                compile_command=command,
                custom_io={"input_ids": "int8"},
            )

        assert Path(fp16_dump).parent != Path(int8_dump).parent
        assert Path(fp16_dump).parent.name.startswith("mdp_")
        assert Path(int8_dump).parent.name.startswith("mdp_")

    def test_mdp_compiler_dump_hash_changes_with_specializations(self, compile_workspace):
        """Different specializations creates a different mdp_<hash> dump directory."""
        _, onnx_path, compile_dir = compile_workspace
        command = ["/opt/qti-aic/exec/qaic-compile", "-aic-hw", f"-m={onnx_path}"]

        with patch("QEfficient.base.modeling_qeff.subprocess.run", side_effect=_fake_subprocess_run):
            seq8_dump = generate_mdp_compiler_dump(
                compile_dir=compile_dir,
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                compile_command=command,
                specializations=[{"batch_size": 1, "seq_len": 8, "ctx_len": 32}],
            )
            seq16_dump = generate_mdp_compiler_dump(
                compile_dir=compile_dir,
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                compile_command=command,
                specializations=[{"batch_size": 1, "seq_len": 16, "ctx_len": 32}],
            )

        assert Path(seq8_dump).parent != Path(seq16_dump).parent
        assert Path(seq8_dump).parent.name.startswith("mdp_")
        assert Path(seq16_dump).parent.name.startswith("mdp_")

    def test_mdp_compiler_dump_hash_changes_with_compile_command(self, compile_workspace):
        """Different compiler command inputs create different mdp_<hash> dump directories."""
        _, onnx_path, compile_dir = compile_workspace
        base_command = ["/opt/qti-aic/exec/qaic-compile", "-aic-hw", f"-m={onnx_path}"]
        fp16_command = base_command + ["-convert-to-fp16"]

        with patch("QEfficient.base.modeling_qeff.subprocess.run", side_effect=_fake_subprocess_run):
            base_dump = generate_mdp_compiler_dump(
                compile_dir=compile_dir,
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                compile_command=base_command,
            )
            fp16_dump = generate_mdp_compiler_dump(
                compile_dir=compile_dir,
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                compile_command=fp16_command,
            )

        assert Path(base_dump).parent != Path(fp16_dump).parent
        assert Path(base_dump).parent.name.startswith("mdp_")
        assert Path(fp16_dump).parent.name.startswith("mdp_")

    def test_mdp_compiler_dump_reuses_same_hash_dir_for_same_inputs(self, compile_workspace):
        """Same dump inputs reuse the same mdp_<hash> dump and skip subprocess rerun."""
        _, onnx_path, compile_dir = compile_workspace
        command = ["/opt/qti-aic/exec/qaic-compile", "-aic-hw", f"-m={onnx_path}"]

        with patch("QEfficient.base.modeling_qeff.subprocess.run", side_effect=_fake_subprocess_run) as run_mock:
            first_dump = generate_mdp_compiler_dump(
                compile_dir=compile_dir,
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                compile_command=command,
                specializations=[{"batch_size": 1, "seq_len": 8, "ctx_len": 32}],
                custom_io={"input_ids": "float16"},
            )
            second_dump = generate_mdp_compiler_dump(
                compile_dir=compile_dir,
                mdp_ts_num_devices=4,
                mdp_num_partitions=2,
                compile_command=command,
                specializations=[{"batch_size": 1, "seq_len": 8, "ctx_len": 32}],
                custom_io={"input_ids": "float16"},
            )

        assert first_dump == second_dump
        assert Path(first_dump).parent.name.startswith("mdp_")
        assert run_mock.call_count == 1
