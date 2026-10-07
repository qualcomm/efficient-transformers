# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from numbers import Integral

import onnx

from QEfficient.base.onnx_transforms import BaseOnnxTransform


class MiniMaxM3StaticPrefillLoopTransform(BaseOnnxTransform):
    """Make the exported MiniMax MSA query-chunk loop statically bounded for QAIC."""

    _apply_with_pipeline_kwargs = True
    _TARGET_FUNCTION_NAME = "QEffMiniMaxM3VLDecoderLayer"
    _VALUE_PREFIX = "qeff_minimax_m3_msa_prefill_loop"

    @staticmethod
    def _loop_body(node: onnx.NodeProto) -> onnx.GraphProto | None:
        return next((attr.g for attr in node.attribute if attr.name == "body" and attr.HasField("g")), None)

    @classmethod
    def _is_query_chunk_loop(cls, node: onnx.NodeProto) -> bool:
        if node.op_type != "Loop" or len(node.input) < 4:
            return False
        body = cls._loop_body(node)
        if body is None:
            return False
        body_values = [output.name for output in body.output]
        body_values.extend(output for body_node in body.node for output in body_node.output)
        return any("slice_scatter" in value for value in body_values)

    @staticmethod
    def _used_graph_values(graph: onnx.GraphProto) -> set[str]:
        used = {value.name for value in (*graph.input, *graph.output, *graph.value_info, *graph.initializer)}
        for node in graph.node:
            used.update(name for name in (*node.input, *node.output) if name)
        return used

    @staticmethod
    def _used_function_values(function: onnx.FunctionProto) -> set[str]:
        used = set(function.input) | set(function.output)
        for node in function.node:
            used.update(name for name in (*node.input, *node.output) if name)
        return used

    @staticmethod
    def _unique_value_name(base_name: str, used_names: set[str]) -> str:
        candidate = base_name
        suffix = 0
        while candidate in used_names:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        used_names.add(candidate)
        return candidate

    @classmethod
    def _rewrite_body_condition(cls, body: onnx.GraphProto, loop_index: int) -> bool:
        if not body.output:
            raise ValueError("MiniMax MSA prefill Loop body has no condition output.")
        if body.output[0].name.startswith(cls._VALUE_PREFIX):
            return False

        used_names = cls._used_graph_values(body)
        cond_name = cls._unique_value_name(f"{cls._VALUE_PREFIX}_{loop_index}_body_cond_true", used_names)
        body.initializer.append(onnx.helper.make_tensor(cond_name, onnx.TensorProto.BOOL, [], [True]))
        body.output[0].name = cond_name
        return True

    @classmethod
    def _rewrite_loop(
        cls,
        node: onnx.NodeProto,
        trip_count: int,
        loop_index: int,
        used_names: set[str],
    ) -> tuple[bool, list[onnx.NodeProto], list[onnx.TensorProto]]:
        body = cls._loop_body(node)
        if body is None:
            raise ValueError(f"MiniMax MSA prefill Loop node {node.name!r} has no body graph.")

        expected_trip_prefix = f"{cls._VALUE_PREFIX}_{trip_count}_"
        controls_are_static = node.input[0].startswith(cls._VALUE_PREFIX) and node.input[1].startswith(
            cls._VALUE_PREFIX
        )
        if controls_are_static:
            if not node.input[0].startswith(expected_trip_prefix):
                raise ValueError(
                    f"MiniMax MSA prefill Loop was already rewritten with a different trip count: {node.input[0]!r}."
                )
            return cls._rewrite_body_condition(body, loop_index), [], []

        trip_name = cls._unique_value_name(f"{cls._VALUE_PREFIX}_{trip_count}_{loop_index}_trip_count", used_names)
        cond_name = cls._unique_value_name(f"{cls._VALUE_PREFIX}_{trip_count}_{loop_index}_cond_true", used_names)
        node.input[0] = trip_name
        node.input[1] = cond_name
        cls._rewrite_body_condition(body, loop_index)

        constant_nodes = [
            onnx.helper.make_node(
                "Constant",
                [],
                [trip_name],
                value=onnx.helper.make_tensor(f"{trip_name}_value", onnx.TensorProto.INT64, [], [trip_count]),
            ),
            onnx.helper.make_node(
                "Constant",
                [],
                [cond_name],
                value=onnx.helper.make_tensor(f"{cond_name}_value", onnx.TensorProto.BOOL, [], [True]),
            ),
        ]
        graph_initializers = [
            onnx.helper.make_tensor(trip_name, onnx.TensorProto.INT64, [], [trip_count]),
            onnx.helper.make_tensor(cond_name, onnx.TensorProto.BOOL, [], [True]),
        ]
        return True, constant_nodes, graph_initializers

    @classmethod
    def _rewrite_function(cls, function: onnx.FunctionProto, trip_count: int, loop_offset: int) -> tuple[bool, int]:
        used_names = cls._used_function_values(function)
        transformed = False
        matched_loops = 0
        rewritten_nodes = []

        for node in function.node:
            if not cls._is_query_chunk_loop(node):
                rewritten_nodes.append(node)
                continue

            loop_index = loop_offset + matched_loops
            matched_loops += 1
            loop_changed, constant_nodes, _ = cls._rewrite_loop(node, trip_count, loop_index, used_names)
            rewritten_nodes.extend(constant_nodes)
            rewritten_nodes.append(node)
            transformed |= loop_changed

        if transformed:
            del function.node[:]
            function.node.extend(rewritten_nodes)
        return transformed, matched_loops

    @classmethod
    def _rewrite_graph(cls, graph: onnx.GraphProto, trip_count: int, loop_offset: int) -> tuple[bool, int]:
        used_names = cls._used_graph_values(graph)
        transformed = False
        matched_loops = 0

        for node in graph.node:
            for attr in node.attribute:
                nested_graphs = [attr.g] if attr.HasField("g") else []
                nested_graphs.extend(attr.graphs)
                for nested_graph in nested_graphs:
                    nested_changed, nested_matches = cls._rewrite_graph(
                        nested_graph, trip_count, loop_offset + matched_loops
                    )
                    transformed |= nested_changed
                    matched_loops += nested_matches

            if not cls._is_query_chunk_loop(node):
                continue

            loop_index = loop_offset + matched_loops
            matched_loops += 1
            loop_changed, _, graph_initializers = cls._rewrite_loop(node, trip_count, loop_index, used_names)
            if graph_initializers:
                graph.initializer.extend(graph_initializers)
            transformed |= loop_changed

        return transformed, matched_loops

    @classmethod
    def apply(
        cls,
        model: onnx.ModelProto,
        *,
        minimax_m3_prefill_loop_trip_count: int | None = None,
        **kwargs,
    ) -> bool:
        del kwargs
        if minimax_m3_prefill_loop_trip_count is None:
            return False
        if isinstance(minimax_m3_prefill_loop_trip_count, bool) or not isinstance(
            minimax_m3_prefill_loop_trip_count, Integral
        ):
            raise TypeError("MiniMax MSA prefill Loop trip count must be an integer.")

        trip_count = int(minimax_m3_prefill_loop_trip_count)
        if trip_count <= 0:
            raise ValueError("MiniMax MSA prefill Loop trip count must be positive.")

        target_functions = [function for function in model.functions if cls._TARGET_FUNCTION_NAME in function.name]
        transformed = False
        matched_loops = 0
        if target_functions:
            for function in target_functions:
                function_changed, function_matches = cls._rewrite_function(function, trip_count, matched_loops)
                transformed |= function_changed
                matched_loops += function_matches
        else:
            transformed, matched_loops = cls._rewrite_graph(model.graph, trip_count, 0)

        if matched_loops == 0:
            raise ValueError("Could not find the exported MiniMax MSA prefill query-chunk Loop.")
        return transformed
