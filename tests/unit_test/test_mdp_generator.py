# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Focused tests for MDP partition generation and validation."""

import json
from importlib import import_module

import onnx
import pytest
from onnx import TensorProto, helper

from QEfficient.compile.mdp_generator import (
    _find_mdp_partition_order_violations,
    _repair_scoped_alias_partitioning,
    generate_disagg_mdp_intersection_config,
    precheck_and_autofix_mdp_partition_order,
)

mdp_generator_module = import_module("QEfficient.compile.mdp_generator")


def _save_dependency_graph(path, nodes, output_names=("output",), functions=()):
    graph = helper.make_graph(
        nodes,
        "mdp_dependency_graph",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in output_names],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("test.functions", 1)],
    )
    model.functions.extend(functions)
    onnx.save(model, path)
    return path


def _mdp(*node_lists):
    return {
        "connections": [{"devices": [0, 1], "type": "p2p"}],
        "partitions": [
            {
                "name": f"Partition{index}",
                "devices": [{"deviceId": index, "numCores": 16}],
                "nodeList": list(node_list),
                "metadata": f"preserve-{index}",
            }
            for index, node_list in enumerate(node_lists)
        ],
        "root_metadata": "preserve-root",
    }


@pytest.mark.cpu_only
def test_precheck_removes_gather_before_argmax_partition_violation(tmp_path):
    """Reproduce the compiler error for a Gather partitioned before its ArgMax producer."""
    onnx_path = tmp_path / "argmax_gather.onnx"
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "ArgMax",
                    inputs=["input"],
                    outputs=["argmax_indices"],
                    axis=1,
                    keepdims=0,
                    name="/ArgMax",
                ),
                helper.make_node(
                    "Gather",
                    inputs=["input", "argmax_indices"],
                    outputs=["output"],
                    axis=1,
                    name="/Gather_16",
                ),
            ],
            name="argmax_gather_graph",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)

    # Mirrors the failing 4-partition assignment: consumer in Partition1,
    # producer in Partition3.
    mdp_json = {
        "partitions": [
            {"name": "Partition0", "nodeList": []},
            {"name": "Partition1", "nodeList": ["/Gather_16"]},
            {"name": "Partition2", "nodeList": []},
            {"name": "Partition3", "nodeList": ["/ArgMax"]},
        ]
    }

    assert _find_mdp_partition_order_violations(str(onnx_path), mdp_json) == [("/Gather_16", "/ArgMax")]

    removed = precheck_and_autofix_mdp_partition_order(str(onnx_path), mdp_json)

    assert removed == 1
    assert mdp_json["partitions"][1]["nodeList"] == []
    assert mdp_json["partitions"][3]["nodeList"] == ["/ArgMax"]
    assert _find_mdp_partition_order_violations(str(onnx_path), mdp_json) == []


@pytest.mark.cpu_only
def test_intersection_precheck_does_not_reorder_protected_nodes(tmp_path):
    """The ordering pre-check does not topologically reorder protected names."""
    onnx_path = _save_dependency_graph(
        tmp_path / "stable_sort.onnx",
        [
            helper.make_node("Identity", ["input"], ["middle"], name="/Producer"),
            helper.make_node("Identity", ["middle"], ["output"], name="/Consumer"),
            helper.make_node("Identity", ["input"], ["independent"], name="/Independent"),
        ],
        output_names=("output", "independent"),
    )
    mdp_json = _mdp(["/Consumer", "/Independent", "/Producer"])
    original_node_lists = [list(partition["nodeList"]) for partition in mdp_json["partitions"]]

    assert _find_mdp_partition_order_violations(str(onnx_path), mdp_json) == [("/Consumer", "/Producer")]

    changed = precheck_and_autofix_mdp_partition_order(
        str(onnx_path), mdp_json, protected_nodes={"/Consumer", "/Independent", "/Producer"}
    )

    assert changed == 0
    assert [partition["nodeList"] for partition in mdp_json["partitions"]] == original_node_lists


@pytest.mark.cpu_only
def test_intersection_precheck_leaves_protected_cross_partition_violation(tmp_path):
    """Protected nodes are not moved or removed by the ordering pre-check."""
    onnx_path = _save_dependency_graph(
        tmp_path / "cross_partition.onnx",
        [
            helper.make_node("Identity", ["input"], ["middle"], name="/Producer"),
            helper.make_node("Identity", ["middle"], ["consumer_output"], name="/Consumer"),
            helper.make_node("Identity", ["input"], ["independent0"], name="/Independent0"),
            helper.make_node("Identity", ["input"], ["output"], name="/Independent1"),
        ],
        output_names=("consumer_output", "independent0", "output"),
    )
    mdp_json = _mdp(["/Consumer", "/Independent0"], ["/Producer", "/Independent1"])
    original = [dict(partition, nodeList=list(partition["nodeList"])) for partition in mdp_json["partitions"]]
    selected = {name for partition in mdp_json["partitions"] for name in partition["nodeList"]}

    assert _find_mdp_partition_order_violations(str(onnx_path), mdp_json) == [("/Consumer", "/Producer")]

    changed = precheck_and_autofix_mdp_partition_order(str(onnx_path), mdp_json, protected_nodes=selected)

    assert changed == 0
    assert mdp_json["partitions"] == original
    assert mdp_json["root_metadata"] == "preserve-root"


@pytest.mark.cpu_only
def test_order_precheck_does_not_trace_through_omitted_nodes(tmp_path):
    """The legacy pre-check only checks direct edges whose endpoints are in the MDP."""
    onnx_path = _save_dependency_graph(
        tmp_path / "omitted_path.onnx",
        [
            helper.make_node("Identity", ["input"], ["value0"], name="/SelectedA"),
            helper.make_node("Identity", ["value0"], ["value1"], name="/Omitted0"),
            helper.make_node("Identity", ["value1"], ["value2"], name="/SelectedB"),
            helper.make_node("Identity", ["value2"], ["value3"], name="/Omitted1"),
            helper.make_node("Identity", ["value3"], ["output"], name="/SelectedC"),
        ],
    )
    mdp_json = _mdp(["/SelectedC", "/SelectedA"], ["/SelectedB"])
    selected = {"/SelectedA", "/SelectedB", "/SelectedC"}
    original_node_lists = [list(partition["nodeList"]) for partition in mdp_json["partitions"]]

    # Every dependency in this chain has an omitted endpoint, so the current
    # main-graph pre-check finds no selected-to-selected direct edge.
    assert _find_mdp_partition_order_violations(str(onnx_path), mdp_json) == []
    assert precheck_and_autofix_mdp_partition_order(str(onnx_path), mdp_json, protected_nodes=selected) == 0

    assert [partition["nodeList"] for partition in mdp_json["partitions"]] == original_node_lists
    assert {name for partition in mdp_json["partitions"] for name in partition["nodeList"]} == selected


@pytest.mark.cpu_only
def test_order_precheck_does_not_validate_local_function_body_dependencies(tmp_path):
    """The pre-check indexes top-level nodes, not inlined function-body names."""
    local_function = helper.make_function(
        "test.functions",
        "Wrapped",
        inputs=["function_input"],
        outputs=["function_output"],
        nodes=[
            helper.make_node("Identity", ["function_input"], ["middle"], name="body_producer"),
            helper.make_node("Identity", ["middle"], ["function_output"], name="body_consumer"),
        ],
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx_path = _save_dependency_graph(
        tmp_path / "local_function.onnx",
        [
            helper.make_node("Identity", ["input"], ["call_input"], name="/Before"),
            helper.make_node("Wrapped", ["call_input"], ["call_output"], domain="test.functions", name="/Call"),
            helper.make_node("Identity", ["call_output"], ["output"], name="/After"),
        ],
        functions=(local_function,),
    )
    selected = {"/Before", "/Call/body_producer", "/Call/body_consumer", "/After"}
    mdp_json = _mdp(["/After", "/Call/body_consumer"], ["/Call/body_producer", "/Before"])
    original_node_lists = [list(partition["nodeList"]) for partition in mdp_json["partitions"]]

    assert _find_mdp_partition_order_violations(str(onnx_path), mdp_json) == []
    assert precheck_and_autofix_mdp_partition_order(str(onnx_path), mdp_json, protected_nodes=selected) == 0

    assert [partition["nodeList"] for partition in mdp_json["partitions"]] == original_node_lists


@pytest.mark.cpu_only
def test_order_precheck_does_not_validate_general_graph_cycles(tmp_path):
    """Cycle detection is outside this main-graph ordering pre-check's scope."""
    onnx_path = _save_dependency_graph(
        tmp_path / "cycle.onnx",
        [
            helper.make_node("Identity", ["value_b"], ["value_a"], name="/CycleA"),
            helper.make_node("Identity", ["value_a"], ["value_b"], name="/CycleB"),
        ],
        output_names=("value_a",),
    )
    mdp_json = _mdp(["/CycleA"], ["/CycleB"])
    original_node_lists = [list(partition["nodeList"]) for partition in mdp_json["partitions"]]

    # The scan observes CycleA -> CycleB from ONNX list order, but cannot infer
    # the reverse edge because CycleB's output has not been seen when CycleA is
    # visited. General ONNX cycle validation is not performed here.
    assert _find_mdp_partition_order_violations(str(onnx_path), mdp_json) == []
    assert (
        precheck_and_autofix_mdp_partition_order(str(onnx_path), mdp_json, protected_nodes={"/CycleA", "/CycleB"}) == 0
    )

    assert [partition["nodeList"] for partition in mdp_json["partitions"]] == original_node_lists


@pytest.mark.cpu_only
def test_order_precheck_does_not_validate_assignment_completeness_or_uniqueness(tmp_path):
    """The pre-check handles order violations, not selected-name assignment integrity."""
    onnx_path = _save_dependency_graph(
        tmp_path / "assignment_validation.onnx",
        [helper.make_node("Identity", ["input"], ["output"], name="/Selected")],
    )
    duplicate_mdp = _mdp(["/Selected"], ["/Selected"])
    duplicate_original = [list(partition["nodeList"]) for partition in duplicate_mdp["partitions"]]

    assert precheck_and_autofix_mdp_partition_order(str(onnx_path), duplicate_mdp, protected_nodes={"/Selected"}) == 0

    assert [partition["nodeList"] for partition in duplicate_mdp["partitions"]] == duplicate_original

    missing_mdp = _mdp(["/Selected"], [])
    missing_original = [list(partition["nodeList"]) for partition in missing_mdp["partitions"]]
    assert (
        precheck_and_autofix_mdp_partition_order(str(onnx_path), missing_mdp, protected_nodes={"/Selected", "/Missing"})
        == 0
    )

    assert [partition["nodeList"] for partition in missing_mdp["partitions"]] == missing_original


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "protected_nodes, expected_removed, expected_node_lists",
    [
        (None, 1, [["/language_model/Gather_5"], []]),
        (set(), 1, [["/language_model/Gather_5"], []]),
        ({"/language_model/Gather_5"}, 1, [["/language_model/Gather_5"], []]),
        ({"/Gather_5"}, 0, [["/language_model/Gather_5"], ["/Gather_5"]]),
        (
            {"/language_model/Gather_5", "/Gather_5"},
            0,
            [["/language_model/Gather_5"], ["/Gather_5"]],
        ),
    ],
    ids=("legacy-none", "empty-set", "scoped-protected", "unscoped-protected", "both-protected"),
)
def test_precheck_alias_cleanup_respects_exact_protected_name(
    tmp_path, protected_nodes, expected_removed, expected_node_lists
):
    """The legacy cleanup targets the unscoped name unless that exact name is protected."""
    onnx_path = _save_dependency_graph(
        tmp_path / "distinct_names.onnx",
        [
            helper.make_node("Identity", ["input"], ["scoped_output"], name="/language_model/Gather_5"),
            helper.make_node("Identity", ["input"], ["output"], name="/Gather_5"),
        ],
    )
    mdp_json = _mdp(["/language_model/Gather_5"], ["/Gather_5"])

    removed = precheck_and_autofix_mdp_partition_order(str(onnx_path), mdp_json, protected_nodes=protected_nodes)

    assert removed == expected_removed
    assert [partition["nodeList"] for partition in mdp_json["partitions"]] == expected_node_lists


@pytest.mark.cpu_only
def test_compiler_repair_uses_exact_top_level_name_not_scoped_peer(tmp_path):
    """Only the unscoped top-level name is relocated; its scoped peer is distinct."""
    onnx_path = _save_dependency_graph(
        tmp_path / "distinct_compiler_names.onnx",
        [
            helper.make_node("Identity", ["input"], ["scoped_output"], name="/language_model/Gather_5"),
            helper.make_node("Identity", ["input"], ["output"], name="/Gather_5"),
        ],
        output_names=("scoped_output", "output"),
    )
    mdp_json = _mdp(["/language_model/Gather_5"], ["/Gather_5"])
    compiler_dump = {
        "partitions": [
            {"nodeList": ["/Gather_5"]},
            {"nodeList": ["/language_model/Gather_5"]},
        ]
    }

    repaired = _repair_scoped_alias_partitioning(
        onnx_path=str(onnx_path),
        mdp_json=mdp_json,
        compiler_dump=compiler_dump,
    )

    assert repaired == 1
    assert [partition["nodeList"] for partition in mdp_json["partitions"]] == [
        ["/Gather_5", "/language_model/Gather_5"],
        [],
    ]


@pytest.mark.cpu_only
def test_intersection_generator_equalizes_compiler_dump_and_relocates_root_node(tmp_path, monkeypatch):
    """A 2-part compiler dump can guide placement in a 7-part MDP."""
    callsite = "/language_model/layers.3/QEffQwen3_5FullAttentionDecoderLayer"
    onnx_path = _save_dependency_graph(
        tmp_path / "compiler_partition_repair.onnx",
        [
            helper.make_node("Identity", ["input"], ["gather_output"], name="/Gather_5"),
            helper.make_node("Identity", ["gather_output"], ["output"], name=callsite),
        ],
    )
    qeff_mdp = _mdp([callsite], [], [], [], [], [], ["/Gather_5"])
    qeff_mdp["connections"] = [{"devices": list(range(7)), "type": "p2p"}]
    monkeypatch.setattr(mdp_generator_module, "generate_disagg_mdp_partition_config", lambda **kwargs: qeff_mdp)

    compiler_dump_path = tmp_path / "tmp_mdp_compiler_dump_3d_3p.json"
    compiler_dump_path.write_text(
        json.dumps(
            {
                "connections": [{"devices": [8, 9], "type": "compiler-dump"}],
                "partitions": [
                    {
                        "name": "CompilerPartition0",
                        "nodeList": ["/Gather_5", "/Other0", "/Other1", "/Other2", callsite, "/Other3", "/Other4"],
                    },
                    {"name": "CompilerPartition1", "nodeList": [f"/Other{index}" for index in range(5, 12)]},
                ],
            }
        )
    )

    mdp_json = generate_disagg_mdp_intersection_config(
        onnx_path=str(onnx_path),
        compiler_dump_path=str(compiler_dump_path),
        num_devices=7,
        num_partitions=7,
        num_layers=7,
    )

    assert mdp_json["partitions"][0]["nodeList"] == ["/Gather_5", callsite]
    assert all(not partition["nodeList"] for partition in mdp_json["partitions"][1:])
    selected_nodes = [name for partition in mdp_json["partitions"] for name in partition["nodeList"]]
    assert sorted(selected_nodes) == sorted(["/Gather_5", callsite])
    assert selected_nodes.count("/Gather_5") == 1
    assert selected_nodes.count(callsite) == 1
    assert [partition["name"] for partition in mdp_json["partitions"]] == [
        partition["name"] for partition in qeff_mdp["partitions"]
    ]
    assert [partition["devices"] for partition in mdp_json["partitions"]] == [
        partition["devices"] for partition in qeff_mdp["partitions"]
    ]
    assert mdp_json["connections"] == qeff_mdp["connections"]

    equalized_dump_path = tmp_path / "tmp_compiler_pp_updated.json"
    assert equalized_dump_path.exists()
    equalized_dump = json.loads(equalized_dump_path.read_text())
    assert [len(partition["nodeList"]) for partition in equalized_dump["partitions"]] == [2] * 7
    assert [partition["nodeList"] for partition in equalized_dump["partitions"]] == [
        ["/Gather_5", "/Other0"],
        ["/Other1", "/Other2"],
        [callsite, "/Other3"],
        ["/Other4", "/Other5"],
        ["/Other6", "/Other7"],
        ["/Other8", "/Other9"],
        ["/Other10", "/Other11"],
    ]
    assert [partition["devices"] for partition in equalized_dump["partitions"]] == [
        partition["devices"] for partition in qeff_mdp["partitions"]
    ]
    assert equalized_dump["connections"] == qeff_mdp["connections"]
    assert not (tmp_path / "tmp_updated_compiler_dump.json").exists()


@pytest.mark.cpu_only
@pytest.mark.parametrize("blocked_dependency", ["producer", "consumer"])
def test_intersection_repair_skips_moves_crossing_transitive_dependencies(tmp_path, blocked_dependency):
    """The repair guard traces through omitted main-graph nodes in both directions."""
    callsite = "/language_model/layers.3/QEffQwen3_5FullAttentionDecoderLayer"
    onnx_path = _save_dependency_graph(
        tmp_path / f"blocked_{blocked_dependency}.onnx",
        [
            helper.make_node("Identity", ["input"], ["producer_value"], name="/Producer"),
            helper.make_node("Identity", ["producer_value"], ["gather_input"], name="/OmittedProducer"),
            helper.make_node("Identity", ["gather_input"], ["gather_value"], name="/Gather_5"),
            helper.make_node("Identity", ["gather_value"], ["consumer_input"], name="/OmittedConsumer"),
            helper.make_node("Identity", ["consumer_input"], ["output"], name=callsite),
        ],
    )

    if blocked_dependency == "producer":
        mdp_json = _mdp([callsite], ["/Producer"], ["/Gather_5"])
        compiler_node_lists = [["/Gather_5", callsite], ["/Producer", "/Other"]]
    else:
        mdp_json = _mdp([callsite], [], ["/Gather_5"])
        compiler_node_lists = [[callsite, "/Other0"], ["/Gather_5", "/Other1"]]

    compiler_dump = {"partitions": [{"nodeList": nodes} for nodes in compiler_node_lists]}
    original_node_lists = [list(partition["nodeList"]) for partition in mdp_json["partitions"]]

    repaired = _repair_scoped_alias_partitioning(
        onnx_path=str(onnx_path),
        mdp_json=mdp_json,
        compiler_dump=compiler_dump,
        compiler_dump_path=str(tmp_path / "compiler_dump.json"),
    )

    assert repaired == 0
    assert [partition["nodeList"] for partition in mdp_json["partitions"]] == original_node_lists
