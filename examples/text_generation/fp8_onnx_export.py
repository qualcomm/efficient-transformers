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


def _onnx_fp8_summary(onnx_path: str) -> None:
    """Print and validate retained FP8 initializers and dequantization nodes."""
    model = onnx.load(onnx_path, load_external_data=False)
    fp8_initializers = [
        initializer for initializer in model.graph.initializer if initializer.data_type == onnx.TensorProto.FLOAT8E4M3FN
    ]
    fp8_initializer_names = {initializer.name for initializer in fp8_initializers}
    dequantize_nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    fp8_dequantize_nodes = [node for node in dequantize_nodes if node.input and node.input[0] in fp8_initializer_names]
    initializers_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    dequantize_attributes = Counter()
    for node in fp8_dequantize_nodes:
        attributes = {attribute.name: attribute.i for attribute in node.attribute}
        if "block_size" in attributes:
            dequantize_attributes["blockwise"] += 1
        elif attributes.get("axis") == 0:
            dequantize_attributes["channelwise"] += 1
            scale = initializers_by_name.get(node.input[1])
            if scale is not None and len(scale.dims) != 1:
                raise RuntimeError("Channelwise FP8 DequantizeLinear scale must be rank 1.")
        else:
            dequantize_attributes["tensorwise"] += 1
            scale = initializers_by_name.get(node.input[1])
            if scale is not None and len(scale.dims) != 0:
                raise RuntimeError("Tensorwise FP8 DequantizeLinear scale must be rank 0.")

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
    print(f"Quantization config: {config.quantization_config}")
    print(f"Configured FP8 weight granularity: {_weight_granularity(config.quantization_config)}")

    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.float16,
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
