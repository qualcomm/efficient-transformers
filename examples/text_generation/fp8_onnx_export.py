# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Export a pre-quantized FP8 causal language model and verify its ONNX weights.

Example::

    HF_HOME=/home/huggingface_hub \
    HF_HUB_CACHE=/home/huggingface_hub/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    python examples/text_generation/fp8_onnx_export.py \
        --model-name Qwen/Qwen3-0.6B-FP8 \
        --export-dir /tmp/qwen3_fp8_onnx

Use ``--legacy`` only with a PyTorch version that supports the requested legacy
ONNX opset.  Dynamo export is the default because blocked FP8 dequantization
uses the ONNX ``DequantizeLinear.block_size`` attribute, which requires opset 21.
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForCausalLM


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Return a quantization config as a plain dictionary when possible."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return None


def _weight_granularity(quantization_config: Any) -> str:
    """Describe the FP8 weight scale granularity from a HF quantization config."""
    config = _as_dict(quantization_config) or {}

    if config.get("weight_block_size") is not None:
        return f"blockwise {config['weight_block_size']}"

    config_groups = config.get("config_groups") or {}
    group = config_groups.get("group_0") or {}
    weights = group.get("weights") or {}
    strategy = weights.get("strategy")
    if strategy:
        return strategy

    weights = config.get("weights_quantization_scheme") or {}
    return weights.get("strategy", "unknown")


def _resolve_dtype(dtype_name: str, config: Any) -> torch.dtype:
    """Resolve the export dtype, defaulting to the model config's dtype."""
    if dtype_name == "auto":
        dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            return dtype
        raise ValueError(f"Model config does not declare a supported dtype, got {dtype!r}; pass --dtype explicitly.")

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _onnx_fp8_summary(onnx_path: str) -> None:
    """Print and validate retained FP8 initializers and dequantization nodes."""
    model = onnx.load(onnx_path, load_external_data=False)
    fp8_initializers = [
        initializer for initializer in model.graph.initializer if initializer.data_type == onnx.TensorProto.FLOAT8E4M3FN
    ]
    fp8_initializer_names = {initializer.name for initializer in fp8_initializers}
    initializers_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    dequantize_attributes = Counter()

    # Legacy export emits DequantizeLinear directly in graph.node.  Dynamo
    # export emits an ONNX local function call, whose function body contains
    # the standard DequantizeLinear node.  Verify both representations.
    local_functions = {(function.domain, function.name): function for function in model.functions}
    fp8_dequantize_nodes = []

    def verify_dequantize(node, scale_name, weight_name):
        if weight_name not in fp8_initializer_names:
            return
        fp8_dequantize_nodes.append(node)
        attributes = {attribute.name: attribute.i for attribute in node.attribute}
        if "block_size" in attributes:
            dequantize_attributes["blockwise"] += 1
            if attributes.get("axis") != -1 or attributes["block_size"] <= 0:
                raise RuntimeError("Blocked FP8 DequantizeLinear must use a valid axis=-1 and positive block_size.")
            scale = initializers_by_name.get(scale_name)
            if scale is not None and len(scale.dims) != 2:
                raise RuntimeError("Blocked FP8 DequantizeLinear scale must be rank 2 for a 2-D weight.")
        elif attributes.get("axis") == 0:
            dequantize_attributes["channelwise"] += 1
            scale = initializers_by_name.get(scale_name)
            if scale is not None and len(scale.dims) != 1:
                raise RuntimeError("Channelwise FP8 DequantizeLinear scale must be rank 1.")
        else:
            dequantize_attributes["tensorwise"] += 1
            scale = initializers_by_name.get(scale_name)
            if scale is not None and len(scale.dims) != 0:
                raise RuntimeError("Tensorwise FP8 DequantizeLinear scale must be rank 0.")

    for node in model.graph.node:
        if node.op_type == "DequantizeLinear" and node.input:
            verify_dequantize(node, node.input[1], node.input[0])
            continue

        function = local_functions.get((node.domain, node.op_type))
        if function is None or not node.input:
            continue
        function_inputs = dict(zip(function.input, node.input))
        weight_name = function_inputs.get(function.input[0])
        scale_name = function_inputs.get(function.input[1]) if len(function.input) > 1 else None
        for function_node in function.node:
            if function_node.op_type == "DequantizeLinear":
                verify_dequantize(function_node, scale_name, weight_name)

    print(f"ONNX path: {onnx_path}")
    print(f"ONNX opset: {next(opset.version for opset in model.opset_import if opset.domain == '')}")
    print(f"FP8 FLOAT8E4M3FN initializers: {len(fp8_initializers)}")
    print(f"FP8 DequantizeLinear nodes: {len(fp8_dequantize_nodes)}")
    print(f"Detected weight granularities: {dict(dequantize_attributes)}")

    if not fp8_initializers:
        raise RuntimeError("No FLOAT8E4M3FN initializers found; FP8 weights were not retained in ONNX.")
    if not fp8_dequantize_nodes:
        raise RuntimeError("No DequantizeLinear nodes found for retained FP8 weights.")

    # Pass the path so ONNX can validate models whose external data is larger
    # than the in-memory protobuf limit.
    onnx.checker.check_model(onnx_path)
    print("ONNX checker: pass")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a pre-quantized FP8 causal LM and verify retained FP8 ONNX weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model ID or local model path")
    parser.add_argument("--export-dir", required=True, help="Directory for the exported ONNX model")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Compute dtype; auto uses the model config dtype to keep FP8 scales and activations consistent",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the legacy TorchScript exporter instead of the default dynamo exporter",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when loading the config and model",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    export_dtype = _resolve_dtype(args.dtype, config)
    print(f"Quantization config: {config.quantization_config}")
    print(f"Configured FP8 weight granularity: {_weight_granularity(config.quantization_config)}")
    print(f"Export dtype: {export_dtype}")

    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=export_dtype,
        trust_remote_code=args.trust_remote_code,
        dequantize_fp8_weights=False,
    )

    onnx_path = model.export(
        export_dir=str(Path(args.export_dir)),
        dynamo=not args.legacy,
        offload_pt_weights=False,
    )
    _onnx_fp8_summary(onnx_path)


if __name__ == "__main__":
    main()
