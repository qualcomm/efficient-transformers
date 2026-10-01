# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Safe regression reproducer for QRANIUMSW-64771.

The compiler MDP dump names Qwen retained-state graph inputs and linear-attention
initializers. Before the fix, the MDP generator enumerated only NodeProto names,
so these valid partition entries were missing from an INTERSECTION configuration.
"""

import json
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from QEfficient.compile.mdp_generator import generate_disagg_mdp_intersection_config


def build_qwen_mdp_fixture(onnx_path: Path) -> None:
    """Write a minimal graph carrying the two compiler-visible Qwen value classes."""
    graph = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                inputs=["hidden_states"],
                outputs=["layer_0_output"],
                name="/model/layers.0/decoder",
            ),
            helper.make_node(
                "Identity",
                inputs=["layer_0_output"],
                outputs=["logits"],
                name="/model/layers.1/decoder",
            ),
        ],
        "qwen_mdp_fixture",
        [
            helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("recurrent_state.0", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("recurrent_state.1", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1])],
        initializer=[
            helper.make_tensor("model.layers.0.linear_attn._ones_lower", TensorProto.FLOAT, [1], [1.0]),
            helper.make_tensor("model.layers.1.linear_attn._ones_lower", TensorProto.FLOAT, [1], [1.0]),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), onnx_path)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="qraniumsw_64771_") as workspace:
        workspace_path = Path(workspace)
        onnx_path = workspace_path / "qwen_mdp.onnx"
        compiler_dump_path = workspace_path / "compiler_dump.json"
        build_qwen_mdp_fixture(onnx_path)
        compiler_dump_path.write_text(
            json.dumps(
                {
                    "partitions": [
                        {
                            "nodeList": [
                                "recurrent_state.0",
                                "model.layers.0.linear_attn._ones_lower",
                                "/model/layers.0/decoder",
                                "recurrent_state.1",
                                "model.layers.1.linear_attn._ones_lower",
                                "/model/layers.1/decoder",
                            ]
                        }
                    ]
                }
            )
        )

        mdp = generate_disagg_mdp_intersection_config(
            onnx_path=str(onnx_path),
            compiler_dump_path=str(compiler_dump_path),
            num_devices=2,
            num_partitions=2,
            num_layers=2,
            num_cores=4,
        )

    expected = {
        "recurrent_state.0",
        "model.layers.0.linear_attn._ones_lower",
        "/model/layers.0/decoder",
        "recurrent_state.1",
        "model.layers.1.linear_attn._ones_lower",
        "/model/layers.1/decoder",
    }
    generated = {name for partition in mdp["partitions"] for name in partition["nodeList"]}
    missing = expected - generated
    if missing:
        raise RuntimeError(f"QRANIUMSW-64771 reproduced: missing MDP entries: {sorted(missing)}")
    print("QRANIUMSW-64771 regression check passed")


if __name__ == "__main__":
    main()
