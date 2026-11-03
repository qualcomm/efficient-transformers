# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Export, compile, and run DeepSeek-V4-Flash in one-token decode mode."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import onnx
import torch
from transformers import AutoConfig, AutoTokenizer

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_HF_CACHE = "/home/huggingface_hub"
DEFAULT_ARTIFACT_ROOT = "/local/mnt/workspace/abhishek/qeff_artifacts"
PREFILL_PROMPT = "Hello, how are you?"
GATE_NON_MATMUL_OUTPUTS = (
    "mlp/gate/score_fn/Softplus_output_0",
    "mlp/gate/score_fn/Sqrt_output_0",
    "mlp/gate/Add_output_0",
    "mlp/gate/Reshape_output_0",
    "mlp/gate/GatherElements_output_0",
    "mlp/gate/Einsum_output_0",
    "mlp/gate/Unsqueeze_output_0",
    "mlp/gate/Div_output_0",
    "mlp/gate/Mul_output_0",
)
MATMUL_OP_TYPES = {"Gemm", "MatMul"}
WEIGHT_TRANSFORM_OP_TYPES = {
    "Add",
    "Cast",
    "CastLike",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "DequantizeLinear",
    "Div",
    "Expand",
    "Gather",
    "Identity",
    "Mul",
    "Neg",
    "QuantizeLinear",
    "Reshape",
    "Slice",
    "Split",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
    "Where",
}
HW_CORES_PER_DEVICE = {"ai100": 16, "ai200": 4}
MODEL_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_device_group(value: str) -> list[int]:
    return [int(device_id) for device_id in value.strip("[]").split(",") if device_id.strip()]


def generate_npi_file(onnx_path: Path, artifact_root: Path, num_hidden_layers: int) -> Path:
    """Generate NPI entries for non-MatMul outputs in the earlier gate target lists."""
    model = onnx.load(onnx_path, load_external_data=False)
    function_keys = {(function.domain, function.name) for function in model.functions}
    if any(
        (node.domain, node.op_type) in function_keys
        and "DeepseekV4DecoderLayer" in node.op_type
        for node in model.graph.node
    ):
        return generate_dynamo_npi_file(model, artifact_root, num_hidden_layers)

    gate_targets = tuple(target for target in GATE_NON_MATMUL_OUTPUTS)
    fp32_outputs = list(
        dict.fromkeys(
            output
            for node in model.graph.node
            if node.op_type not in MATMUL_OP_TYPES
            for output in node.output
            if any(output.endswith(target) for target in gate_targets)
        )
    )
    if not fp32_outputs:
        raise ValueError(f"No requested gate non-MatMul outputs were found in {onnx_path}.")

    npi_path = artifact_root / f"deepseek_v4_gate_non_matmul_fp32_{num_hidden_layers}layer.yaml"
    npi_contents = "FP32NodeInstanceNames:\n" + "".join(f"  - {output}\n" for output in fp32_outputs)
    npi_path.write_text(npi_contents, encoding="utf-8")
    return npi_path


def generate_dynamo_npi_file(model: onnx.ModelProto, artifact_root: Path, num_hidden_layers: int) -> Path:
    """Generate NPI outputs mapped from the non-MatMul gate targets in Dynamo subfunctions."""
    fp32_outputs = generate_subfunction_npi_outputs(model, num_hidden_layers)

    if not fp32_outputs:
        raise ValueError(f"Cannot generate Dynamo NPI; no gate non-MatMul outputs were found in {model.graph.name}.")

    npi_path = artifact_root / f"deepseek_v4_gate_non_matmul_fp32_{num_hidden_layers}layer.yaml"
    npi_contents = "FP32NodeInstanceNames:\n" + "".join(f"  - {output}\n" for output in fp32_outputs)
    npi_path.write_text(npi_contents, encoding="utf-8")
    return npi_path


def generate_subfunction_npi_outputs(model: onnx.ModelProto, num_hidden_layers: int) -> list[str]:
    functions = {(function.domain, function.name): function for function in model.functions}
    fp32_outputs = []
    decoder_layer_idx = 0

    for call_node in model.graph.node:
        function = functions.get((call_node.domain, call_node.op_type))
        if function is None or "DeepseekV4DecoderLayer" not in function.name:
            continue

        layer_idx = decoder_layer_idx
        decoder_layer_idx += 1
        if layer_idx >= num_hidden_layers:
            break

        output_instance_prefix = f"/model/layers.{layer_idx}/"
        layer_outputs = _map_gate_non_matmul_outputs(function, call_node, layer_idx)
        fp32_outputs.extend(f"{output_instance_prefix}{node.name}_output_0" for node in layer_outputs)

    return fp32_outputs


def _input_depends_on_weight(
    tensor_name: str,
    weight_inputs: set[str],
    producers: dict[str, onnx.NodeProto],
    visited: set[str] | None = None,
) -> bool:
    if tensor_name in weight_inputs:
        return True
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return False
    visited.add(tensor_name)
    producer = producers.get(tensor_name)
    if producer is None or producer.op_type not in WEIGHT_TRANSFORM_OP_TYPES:
        return False
    return any(_input_depends_on_weight(node_input, weight_inputs, producers, visited) for node_input in producer.input)


def _require_one_node(nodes: list[onnx.NodeProto], target_name: str, layer_idx: int) -> onnx.NodeProto:
    if len(nodes) != 1:
        raise ValueError(f"Expected one {target_name} node for layer {layer_idx}, found {len(nodes)}.")
    return nodes[0]


def _find_upstream_nodes(
    tensor_name: str,
    target_op_type: str,
    producers: dict[str, onnx.NodeProto],
) -> list[onnx.NodeProto]:
    matches = {}
    visited = set()
    pending = [(tensor_name, 0)]
    closest_distance = None
    while pending:
        current, distance = pending.pop(0)
        if closest_distance is not None and distance > closest_distance:
            break
        if current in visited:
            continue
        visited.add(current)
        producer = producers.get(current)
        if producer is None:
            continue
        if producer.op_type == target_op_type:
            matches[producer.name] = producer
            closest_distance = distance
        else:
            pending.extend((node_input, distance + 1) for node_input in producer.input)
    return list(matches.values())


def _map_gate_non_matmul_outputs(
    function: onnx.FunctionProto,
    call_node: onnx.NodeProto,
    layer_idx: int,
) -> list[onnx.NodeProto]:
    producers = {output: node for node in function.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in function.node:
        for node_input in node.input:
            consumers[node_input].append(node)

    gate_weight_inputs = {
        function_input
        for function_input, call_input in zip(function.input, call_node.input)
        if call_input == f"model.layers.{layer_idx}.mlp.gate.weight"
    }
    if not gate_weight_inputs:
        raise ValueError(f"Cannot map gate NPI for layer {layer_idx}; gate weight input is absent.")

    gate_matmuls = [
        node
        for node in function.node
        if node.op_type in MATMUL_OP_TYPES
        and len(node.input) > 1
        and _input_depends_on_weight(node.input[1], gate_weight_inputs, producers)
    ]
    gate_matmul = _require_one_node(gate_matmuls, "gate projection MatMul", layer_idx)

    softplus = _require_one_node(
        [
            node
            for output in gate_matmul.output
            for node in consumers[output]
            if node.op_type in {"Softplus", "aten_softplus"}
        ],
        "gate Softplus",
        layer_idx,
    )
    sqrt = _require_one_node(
        [node for output in softplus.output for node in consumers[output] if node.op_type == "Sqrt"],
        "gate Sqrt",
        layer_idx,
    )
    gather = _require_one_node(
        [node for node in function.node if node.op_type == "GatherElements" and node.input[0] in sqrt.output],
        "gate GatherElements",
        layer_idx,
    )
    gate_scores = _require_one_node(
        [node for output in gather.output for node in consumers[output] if node.op_type == "Einsum"],
        "gate Einsum",
        layer_idx,
    )
    normalize = _require_one_node(
        [node for output in gather.output for node in consumers[output] if node.op_type == "Div"],
        "gate Div",
        layer_idx,
    )
    unsqueeze = _require_one_node(
        [node for output in gate_scores.output for node in consumers[output] if node.op_type == "Unsqueeze"],
        "gate Unsqueeze",
        layer_idx,
    )
    denominator_add = _require_one_node(
        [
            node
            for output in unsqueeze.output
            for node in consumers[output]
            if node.op_type == "Add" and any(output in normalize.input for output in node.output)
        ],
        "gate denominator Add",
        layer_idx,
    )
    scale = _require_one_node(
        [node for output in normalize.output for node in consumers[output] if node.op_type == "Mul"],
        "gate scaling Mul",
        layer_idx,
    )

    index_reshape = _require_one_node(
        _find_upstream_nodes(gather.input[1], "Reshape", producers),
        "gate index Reshape",
        layer_idx,
    )

    return [softplus, sqrt, denominator_add, index_reshape, gather, gate_scores, unsqueeze, normalize, scale]


def parse_args(defaults: dict[str, object] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache", type=Path, default=Path(DEFAULT_HF_CACHE))
    parser.add_argument("--artifact-root", type=Path, default=Path(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument(
        "--model-dtype",
        choices=tuple(MODEL_DTYPES),
        default="float16",
        help="Model and ONNX export dtype.",
    )
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Export and compile the blocked CSA/HCA prefill graph instead of the one-token decode graph.",
    )
    parser.add_argument(
        "--use-onnx-subfunctions",
        action="store_true",
        help="Export repeated decoder blocks as ONNX subfunctions.",
    )
    parser.add_argument(
        "--mdp-num-partitions",
        "--pp-size",
        dest="mdp_num_partitions",
        type=int,
        default=1,
        help="Pipeline-parallel MDP partitions for prefill; values above one enable disaggregated ONNX MDP.",
    )
    parser.add_argument(
        "--mdp-strategy",
        choices=("onnx", "intersection"),
        default="onnx",
        help="Partition-assignment strategy used when pipeline parallelism is enabled.",
    )
    parser.add_argument("--generation-len", type=int, default=250)
    parser.add_argument("--num-hidden-layers", type=int, default=43)
    parser.add_argument("--num-cores", "--compile-num-cores", dest="num_cores", type=int, default=12)
    parser.add_argument("--device-group", type=parse_device_group, default=[i for i in range(4)])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--attn-dp", type=int, default=None, help="Folded decode attention data-parallel rows.")
    parser.add_argument("--indexer-cp", type=int, default=1, help="CSA indexer compressed-cache context parallelism.")
    parser.add_argument(
        "--csa-prefill-attn-gather-sub-qc",
        "--csa-prefill-q-block-size",
        dest="csa_prefill_attn_gather_sub_qc",
        type=int,
        default=None,
        help="CSA prefill compressed-attention query subchunk size.",
    )
    parser.add_argument(
        "--csa-prefill-indexer-kv-block-size",
        "--csa-prefill-kv-block-size",
        "--csa-block-size",
        dest="csa_prefill_indexer_kv_block_size",
        type=int,
        default=None,
        help="CSA prefill indexer score tile size.",
    )
    parser.add_argument(
        "--csa-prefill-indexer-topk-per-block",
        type=int,
        default=None,
        help="CSA prefill candidates retained from each indexer score tile.",
    )
    parser.add_argument(
        "--csa-prefill-indexer-mode",
        choices=("global", "blockwise"),
        default=None,
        help="CSA prefill indexer selection: exact global top-k or approximate blockwise candidate merge.",
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=1,
        help="CSA indexer CP-local blocks; values above one currently trigger a QAIC compiler crash.",
    )
    parser.add_argument(
        "--hca-compressed-kv-cp",
        type=int,
        default=1,
        help="HCA compressed-KV context-parallel degree.",
    )
    parser.add_argument(
        "--hca-attn-blocks",
        type=int,
        default=None,
        help="HCA dense compressed-attention tiles; defaults to the hardware attention-core count.",
    )
    parser.add_argument(
        "--hca-prefill-q-block-size",
        type=int,
        default=None,
        help="HCA prefill query block size; defaults to the model sliding-window length.",
    )
    parser.add_argument("--hw-version", choices=tuple(HW_CORES_PER_DEVICE), default="ai100")
    parser.add_argument(
        "--ffn-blocking-mode",
        choices=("default", "token", "weight", "token_weight"),
        default="default",
        help="Routed-expert FFN blocking mode. The default preserves the unblocked graph.",
    )
    parser.add_argument(
        "--ffn-token-block-size",
        type=int,
        default=None,
        help="Token tile size for token or token_weight FFN blocking.",
    )
    parser.add_argument(
        "--ffn-weight-block-size",
        type=int,
        default=None,
        help="Intermediate-dimension tile size for weight or token_weight FFN blocking.",
    )
    parser.add_argument("--prefill-prompt", default=PREFILL_PROMPT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--automation", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--enable-residual-npi",
        action="store_true",
        help="Generate and compile with BF16 NPI for graph-mapped non-MatMul gate outputs.",
    )
    parser.add_argument(
        "--no-mxfp6-matmul",
        action="store_true",
        help="Disable MXFP6 MatMul weight compression during QAIC compilation.",
    )
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args()


def build_ffn_blocking_config(args: argparse.Namespace) -> dict[str, int | str | None]:
    """Build the compile-time routed-expert FFN blocking configuration."""
    return {
        "ffn_blocking_mode": args.ffn_blocking_mode,
        "ffn_token_block_size": args.ffn_token_block_size,
        "ffn_weight_block_size": args.ffn_weight_block_size,
    }


def configure_qeff_parallel_layout(config, args: argparse.Namespace) -> None:
    """Apply the benchmark's shared folded decode controls to QEff cache layouts."""
    attention_dp = args.attn_dp or len(args.device_group)
    indexer_cp = args.indexer_cp
    num_kv_blocks = args.num_kv_blocks
    attention_cores = HW_CORES_PER_DEVICE[args.hw_version]
    hca_compressed_kv_cp = args.hca_compressed_kv_cp
    hca_attn_blocks = args.hca_attn_blocks or attention_cores
    if attention_dp < 1:
        raise ValueError("attn_dp must be at least 1.")
    if indexer_cp < 1:
        raise ValueError("indexer_cp must be at least 1.")
    if num_kv_blocks < 1:
        raise ValueError("num_kv_blocks must be at least 1.")
    if hca_compressed_kv_cp < 1:
        raise ValueError("hca_compressed_kv_cp must be at least 1.")
    if hca_attn_blocks < 1:
        raise ValueError("hca_attn_blocks must be at least 1.")
    hca_prefill_q_block_size = getattr(args, "hca_prefill_q_block_size", None)
    if hca_prefill_q_block_size is not None and hca_prefill_q_block_size < 1:
        raise ValueError("hca_prefill_q_block_size must be at least 1.")
    if args.batch_size % attention_dp:
        raise ValueError("batch_size must be divisible by attn_dp.")

    csa_capacity = None
    hca_capacity = None
    for layer_type in config.layer_types:
        if layer_type == "compressed_sparse_attention":
            csa_capacity = (args.ctx_len + config.compress_rates[layer_type] - 1) // config.compress_rates[layer_type]
            if csa_capacity % indexer_cp:
                raise ValueError("CSA compressed-cache capacity must be divisible by indexer_cp.")
            csa_slots_per_cp = csa_capacity // indexer_cp
            if csa_slots_per_cp % num_kv_blocks:
                raise ValueError("CSA CP-way slot count must be divisible by num_kv_blocks.")
            if (csa_slots_per_cp // num_kv_blocks) % attention_cores:
                raise ValueError("CSA KV-block width must be divisible by the hardware attention-core count.")
        if layer_type == "heavily_compressed_attention":
            hca_capacity = (args.ctx_len + config.compress_rates[layer_type] - 1) // config.compress_rates[layer_type]
            if hca_capacity % hca_compressed_kv_cp:
                raise ValueError("HCA compressed-cache capacity must be divisible by hca_compressed_kv_cp.")
            if (hca_capacity // hca_compressed_kv_cp) % hca_attn_blocks:
                raise ValueError("HCA CP-way slot count must be divisible by hca_attn_blocks.")
    config.qeff_csa_attention_dp = attention_dp
    config.qeff_csa_indexer_cp = indexer_cp
    config.qeff_csa_num_kv_blocks = num_kv_blocks
    config.qeff_csa_indexer_attention_cores = attention_cores
    config.qeff_csa_folded_row_cache = True
    csa_prefill_attn_gather_sub_qc = getattr(args, "csa_prefill_attn_gather_sub_qc", None)
    if csa_prefill_attn_gather_sub_qc is not None:
        config.qeff_csa_prefill_attn_gather_sub_qc = csa_prefill_attn_gather_sub_qc
    csa_prefill_indexer_kv_block_size = getattr(args, "csa_prefill_indexer_kv_block_size", None)
    if csa_prefill_indexer_kv_block_size is not None:
        config.qeff_csa_prefill_indexer_kv_block_size = csa_prefill_indexer_kv_block_size
    csa_prefill_indexer_topk_per_block = getattr(args, "csa_prefill_indexer_topk_per_block", None)
    if csa_prefill_indexer_topk_per_block is not None:
        config.qeff_csa_prefill_indexer_topk_per_block = csa_prefill_indexer_topk_per_block
    csa_prefill_indexer_mode = getattr(args, "csa_prefill_indexer_mode", None)
    if csa_prefill_indexer_mode is not None:
        config.qeff_csa_prefill_indexer_mode = csa_prefill_indexer_mode
    config.qeff_hca_attention_dp = attention_dp
    config.qeff_hca_compressed_kv_cp = hca_compressed_kv_cp
    config.qeff_hca_attn_blocks = hca_attn_blocks
    config.qeff_hca_folded_row_cache = True
    if hca_prefill_q_block_size is not None:
        config.qeff_hca_prefill_q_block_size = hca_prefill_q_block_size
    print(
        "Folded decode layout: "
        f"batch_size={args.batch_size}, attn_dp={attention_dp}, indexer_cp={indexer_cp}, "
        f"num_kv_blocks={num_kv_blocks}, "
        f"hw_version={args.hw_version}, hca_compressed_kv_cp={hca_compressed_kv_cp}, "
        f"hca_attn_blocks={hca_attn_blocks}, csa_capacity={csa_capacity}, hca_capacity={hca_capacity}"
    )


def main(defaults: dict[str, object] | None = None) -> None:
    args = parse_args(defaults)
    if args.ctx_len < 2:
        raise ValueError("ctx_len must be at least 2.")
    if not 1 <= args.prefill_seq_len <= args.ctx_len:
        raise ValueError("prefill_seq_len must be in [1, ctx_len].")
    if not 1 <= args.generation_len < args.ctx_len:
        raise ValueError("generation_len must be in [1, ctx_len).")
    if args.num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be at least 1.")
    if not args.device_group:
        raise ValueError("device_group must contain at least one QAIC device ID.")
    if not args.prefill_only and args.batch_size % len(args.device_group):
        raise ValueError("DeepSeek V4 Flash decode requires batch_size to be divisible by the device-group size.")
    if args.mdp_num_partitions < 1:
        raise ValueError("mdp_num_partitions must be at least 1.")
    if args.mdp_num_partitions > 1:
        if not args.prefill_only:
            raise ValueError("Pipeline-parallel MDP is supported only for prefill-only compilation.")
        if len(args.device_group) % args.mdp_num_partitions:
            raise ValueError("device_group size must be divisible by mdp_num_partitions.")
        if args.export_only:
            raise ValueError("Pipeline-parallel MDP export requires compilation; do not use --export-only.")
    os.environ["HF_HUB_CACHE"] = str(args.hf_cache)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ.setdefault("QEFF_HOME", str(args.artifact_root))
    os.environ.setdefault("QEFF_CHECKPOINT_HOME", str(args.artifact_root / "checkpoints"))

    from QEfficient import QEFFAutoModelForCausalLM

    export_root = args.artifact_root / "onnx"
    compile_root = args.artifact_root / "compile"
    export_root.mkdir(parents=True, exist_ok=True)
    compile_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} tokenizer and config")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.hf_cache,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(
        args.model_id,
        cache_dir=args.hf_cache,
        local_files_only=args.local_files_only,
    )
    if args.num_hidden_layers > config.num_hidden_layers:
        raise ValueError(f"num_hidden_layers cannot exceed the checkpoint's {config.num_hidden_layers} layers.")
    config.num_hidden_layers = args.num_hidden_layers
    config.layer_types = config.layer_types[: args.num_hidden_layers]
    config.mlp_layer_types = config.mlp_layer_types[: args.num_hidden_layers]
    config.qeff_swa_prefill_only = args.prefill_only
    config.qeff_csa_prefill_only = args.prefill_only
    config.qeff_hca_prefill_only = args.prefill_only
    batch_size = args.batch_size
    configure_qeff_parallel_layout(config, args)
    qaic_config = build_ffn_blocking_config(args)
    print(
        "FFN blocking: "
        f"mode={args.ffn_blocking_mode}, token_block_size={args.ffn_token_block_size}, "
        f"weight_block_size={args.ffn_weight_block_size}"
    )

    print("Building the QEfficient model in weight-free mode")
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.hf_cache,
        config=config,
        local_files_only=args.local_files_only,
        weight_free=True,
    )
    model_dtype = MODEL_DTYPES[args.model_dtype]
    qeff_model.model.to(dtype=model_dtype)
    qeff_model.model.config.torch_dtype = model_dtype
    qeff_model.model.eval()
    pipeline_parallel = args.mdp_num_partitions > 1
    if not pipeline_parallel:
        qeff_model.transform(
            ctx_len=args.ctx_len,
            seq_len=args.prefill_seq_len if args.prefill_only else 1,
            bs=batch_size,
            num_devices=len(args.device_group),
            qaic_config=qaic_config,
            prefill_only=args.prefill_only,
            num_cores=args.num_cores,
        )
    else:
        qeff_model.transform(
            ctx_len=args.ctx_len,
            seq_len=args.prefill_seq_len,
            bs=batch_size,
            num_devices=len(args.device_group),
            qaic_config=qaic_config,
            prefill_only=args.prefill_only,
            num_cores=args.num_cores,
            prefill_seq_len=args.prefill_seq_len,
            mdp_num_partitions=args.mdp_num_partitions,
        )

    graph_kind = "blocked DeepSeek V4 prefill" if args.prefill_only else "one-token decode"
    print(f"Exporting the {graph_kind} graph through qeff_model.export()")
    export_kwargs = {}
    if pipeline_parallel:
        export_kwargs = {
            "enable_chunking": True,
            "retain_full_kv": True,
            "split_model_io": True,
            "mdp_num_partitions": args.mdp_num_partitions,
            "mdp_strategy": args.mdp_strategy,
        }
    onnx_path = Path(
        qeff_model.export(
            export_dir=str(export_root),
            prefill_only=args.prefill_only,
            prefill_seq_len=args.prefill_seq_len,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
            dynamo=True,
            export_batch_size=batch_size,
            cache_ctx_len=args.ctx_len,
            qaic_config=qaic_config,
            **export_kwargs,
        )
    )
    print(f"ONNX_PATH={onnx_path}")

    npi_path = None
    if args.enable_residual_npi:
        if args.model_dtype != "bfloat16":
            raise ValueError("--enable-residual-npi is intended for --model-dtype bfloat16.")
        if onnx_path is None:
            raise ValueError("ONNX path is required to generate residual NPI.")
        npi_path = generate_npi_file(onnx_path, args.artifact_root, args.num_hidden_layers)
        print(f"NPI_PATH={npi_path}")

    if args.export_only:
        return

    print(
        f"Compiling {graph_kind} specialization: seq_len={args.prefill_seq_len if args.prefill_only else 1}, ctx_len={args.ctx_len}"
    )
    compile_kwargs = {
        "onnx_path": str(onnx_path) if onnx_path is not None else None,
        "compile_dir": str(compile_root),
        "prefill_seq_len": args.prefill_seq_len if args.prefill_only else 1,
        "ctx_len": args.ctx_len,
        "batch_size": batch_size,
        "num_cores": args.num_cores,
        "num_devices": len(args.device_group),
        "aic_hw_version": args.hw_version,
        "prefill_only": args.prefill_only,
        "use_onnx_subfunctions": args.use_onnx_subfunctions,
        "mxint8_kv_cache": False,
        "mxfp6_matmul": not args.no_mxfp6_matmul,
        "user_tiled": True,
        "qaic_config": qaic_config,
        "dynamo": True,
    }
    if args.enable_residual_npi:
        compile_kwargs["node_precision_info"] = npi_path
    if args.mdp_num_partitions > 1:
        compile_kwargs.update(
            enable_chunking=True,
            retain_full_kv=True,
            split_model_io=True,
            mdp_num_partitions=args.mdp_num_partitions,
            mdp_strategy=args.mdp_strategy,
        )
        print(
            "Pipeline-parallel prefill: "
            f"partitions={args.mdp_num_partitions}, devices_per_partition="
            f"{len(args.device_group) // args.mdp_num_partitions}, mdp_strategy={args.mdp_strategy}"
        )
    qpc_path = Path(qeff_model.compile(**compile_kwargs))
    if onnx_path is None:
        onnx_path = Path(qeff_model.onnx_path)
        print(f"ONNX_PATH={onnx_path}")
    print(f"QPC_PATH={qpc_path}")
    print(f"SPECIALIZATIONS_PATH={qpc_path.parent / 'specializations.json'}")
    print(f"CUSTOM_IO_PATH={qpc_path.parent / 'custom_io.yaml'}")

    if args.compile_only or args.prefill_only:
        return

    print("Running qeff_model.generate()")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[args.prefill_prompt] * batch_size,
        device_ids=args.device_group,
        generation_len=args.generation_len,
        automation=args.automation,
        stream=False,
    )

    result_path = args.artifact_root / "generation_result.json"
    result = {
        "model_id": args.model_id,
        "prefill_prompt": args.prefill_prompt,
        "onnx_path": str(onnx_path),
        # "npi_path": str(npi_path),
        "qpc_path": str(qpc_path),
        "generated_texts": exec_info.generated_texts,
        "generated_ids": [ids.tolist() if hasattr(ids, "tolist") else ids for ids in exec_info.generated_ids],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"RESULT_PATH={result_path}")
    print(f"GENERATED_TEXT={exec_info.generated_texts}")


if __name__ == "__main__":
    main()
