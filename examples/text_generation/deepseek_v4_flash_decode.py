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
from collections import Counter
from pathlib import Path

import onnx
import torch
import yaml
from onnx import TensorProto, helper
from transformers import AutoConfig, AutoTokenizer

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_HF_CACHE = "/home/huggingface_hub"
DEFAULT_ARTIFACT_ROOT = "/home/abhishek/.cache/qeff_artifacts"
PREFILL_PROMPT = "Hello, how are you?"
DEEPSEEK_V4_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<｜User｜>{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}<｜Assistant｜>{{ message['content'] }}<｜end▁of▁sentence｜>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<｜Assistant｜><think>{% endif %}"
)
HW_CORES_PER_DEVICE = {"ai100": 16, "ai200": 4}
MODEL_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DECODER_FUNCTION_DOMAIN = "pkg.torch.__subgraph__"
DECODER_FUNCTION_PREFIX = "QEffDeepseekV4DecoderLayer"
Container = onnx.GraphProto | onnx.FunctionProto


def parse_device_group(value: str) -> list[int]:
    return [int(device_id) for device_id in value.strip("[]").split(",") if device_id.strip()]


def tensor_types(container: Container) -> dict[str, int]:
    values = list(container.value_info)
    if isinstance(container, onnx.GraphProto):
        values.extend(container.input)
        values.extend(container.output)
    types = {
        value_info.name: value_info.type.tensor_type.elem_type
        for value_info in values
        if value_info.type.tensor_type.HasField("elem_type")
    }
    if isinstance(container, onnx.GraphProto):
        types.update({initializer.name: initializer.data_type for initializer in container.initializer})
    return types


def is_matrix_operation(node: onnx.NodeProto) -> bool:
    """Return whether retaining this operation in FP32 would make matrix math expensive."""
    if node.op_type in {"MatMul", "Gemm"}:
        return True
    if node.op_type != "Einsum":
        return False
    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    equation = attributes.get("equation", b"")
    if isinstance(equation, bytes):
        equation = equation.decode()
    return "," in equation.partition("->")[0]


def selected_npi_nodes(container: Container) -> list[onnx.NodeProto]:
    types = tensor_types(container)
    return [
        node
        for node in container.node
        if node.output and not is_matrix_operation(node) and types.get(node.output[0]) == TensorProto.FLOAT
    ]


def canonical_output_name(node: onnx.NodeProto) -> str:
    if "__node_" not in node.name:
        return f"{node.name}_output_0"
    module_path, fx_name = node.name.split("__node_", 1)
    module_path = module_path.rstrip("_")
    module_leaf = module_path.rsplit("/", 1)[-1]
    if fx_name == module_leaf:
        node_path = module_path
    elif fx_name.startswith(f"{module_leaf}__"):
        node_path = f"{module_path}/{fx_name[len(module_leaf) + 2 :]}"
    else:
        node_path = f"{module_path}/{fx_name}"
    return f"{node_path}_output_0"


def build_output_aliases(
    container: Container, selected: list[onnx.NodeProto], public_outputs: set[str]
) -> dict[str, str]:
    aliases = {
        node.output[0]: node.output[0] if node.output[0] in public_outputs else canonical_output_name(node)
        for node in selected
    }
    if len(set(aliases.values())) != len(aliases):
        collisions = [name for name, count in Counter(aliases.values()).items() if count > 1]
        raise ValueError(f"NPI output aliases collide: {collisions[:10]}")
    reserved_outputs = {output for node in container.node for output in node.output} - set(aliases)
    collisions = sorted(set(aliases.values()) & reserved_outputs)
    if collisions:
        raise ValueError(f"NPI aliases collide with existing values: {collisions[:10]}")
    return aliases


def rename_values(container: Container, aliases: dict[str, str]) -> None:
    for node in container.node:
        for index, input_name in enumerate(node.input):
            node.input[index] = aliases.get(input_name, input_name)
        for index, output_name in enumerate(node.output):
            node.output[index] = aliases.get(output_name, output_name)
    for value_info in container.value_info:
        value_info.name = aliases.get(value_info.name, value_info.name)
    if isinstance(container, onnx.FunctionProto):
        for index, output_name in enumerate(container.output):
            container.output[index] = aliases.get(output_name, output_name)


def decoder_layer_instances(model: onnx.ModelProto) -> dict[str, list[int]]:
    instances: dict[str, list[int]] = {}
    for node in model.graph.node:
        if node.domain != DECODER_FUNCTION_DOMAIN or not node.op_type.startswith(DECODER_FUNCTION_PREFIX):
            continue
        if "/model/layers." not in node.name:
            raise ValueError(f"Cannot determine the decoder layer index from ONNX node {node.name!r}.")
        layer_text = node.name.split("/model/layers.", 1)[1].split("/", 1)[0]
        instances.setdefault(node.op_type, []).append(int(layer_text))
    return instances


def npi_category(node: onnx.NodeProto) -> str:
    name = node.name
    if "/attn_hc/" in name:
        return "attention hyperconnection"
    if "/ffn_hc/" in name:
        return "FFN hyperconnection"
    if "/hc_head/" in name:
        return "final hyperconnection head"
    if "/self_attn/compressor/indexer/" in name:
        return "CSA indexer/scorer"
    if "/self_attn/compressor/" in name:
        return "attention compressor"
    if "/self_attn/" in name:
        return "attention/rotary/softmax"
    if "/mlp/gate/" in name:
        return "MoE router"
    if "/mlp/experts/" in name:
        return "routed experts"
    if "/mlp/shared_experts/" in name:
        return "shared experts"
    if "norm" in name:
        return "normalization"
    if "/lm_head/" in name or "logits" in node.output:
        return "LM head/logits"
    return "other FP32 chain"


def summarize_npi_container(
    name: str,
    selected: list[onnx.NodeProto],
    aliases: dict[str, str],
    layers: list[int],
) -> dict[str, object]:
    multiplier = max(len(layers), 1)
    return {
        "container": name,
        "layers": layers,
        "unique_npi_entries": len(selected),
        "instantiated_node_count": len(selected) * multiplier,
        "category_counts": dict(Counter(npi_category(node) for node in selected)),
        "operation_counts": dict(Counter(node.op_type for node in selected)),
        "entries": [aliases[node.output[0]] for node in selected],
    }


def generate_npi_artifacts(onnx_path: Path) -> tuple[Path, Path, Path]:
    """Create an exhaustive non-matrix FP32 NPI and a name-compatible ONNX copy."""
    model = onnx.load(onnx_path, load_external_data=False)
    function_instances = decoder_layer_instances(model)
    container_summaries = []
    npi_names = []

    graph_selected = selected_npi_nodes(model.graph)
    graph_aliases = build_output_aliases(model.graph, graph_selected, {output.name for output in model.graph.output})
    container_summaries.append(summarize_npi_container("main graph", graph_selected, graph_aliases, []))
    npi_names.extend(graph_aliases[node.output[0]] for node in graph_selected)
    rename_values(model.graph, graph_aliases)

    for function in model.functions:
        if function.domain != DECODER_FUNCTION_DOMAIN or not function.name.startswith(DECODER_FUNCTION_PREFIX):
            continue
        selected = selected_npi_nodes(function)
        aliases = build_output_aliases(function, selected, set())
        container_summaries.append(
            summarize_npi_container(function.name, selected, aliases, function_instances.get(function.name, []))
        )
        npi_names.extend(aliases[node.output[0]] for node in selected)
        rename_values(function, aliases)

    npi_names = list(dict.fromkeys(npi_names))
    if not npi_names:
        raise RuntimeError("No FP32 non-matrix nodes were found for NPI generation.")

    output_onnx_path = onnx_path.with_name(f"{onnx_path.stem}.exhaustive-npi{onnx_path.suffix}")
    npi_path = onnx_path.with_name(f"{onnx_path.stem}.exhaustive-npi.yaml")
    summary_path = onnx_path.with_name(f"{onnx_path.stem}.exhaustive-npi.summary.json")
    onnx.checker.check_model(model)
    onnx.save_model(model, output_onnx_path)
    npi_path.write_text(yaml.safe_dump({"FP32NodeInstanceNames": npi_names}, sort_keys=False), encoding="utf-8")

    summary = {
        "source_onnx": str(onnx_path),
        "npi_compatible_onnx": str(output_onnx_path),
        "npi": str(npi_path),
        "scope": "All ONNX FP32-output chains; matrix operations excluded",
        "excluded_operations": ["MatMul", "Gemm", "multi-input Einsum contractions"],
        "retained_einsum_kind": "single-input reductions only",
        "unique_npi_entries": len(npi_names),
        "instantiated_node_count": sum(item["instantiated_node_count"] for item in container_summaries),
        "containers": container_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_onnx_path, npi_path, summary_path


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
        type=int,
        default=None,
        help="CSA prefill compressed-attention query subchunk size.",
    )
    parser.add_argument(
        "--csa-prefill-indexer-kv-block-size",
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
    parser.add_argument(
        "-enable-npi",
        "--enable-npi",
        "--generate-npi",
        dest="enable_npi",
        action="store_true",
        help="Generate an exhaustive non-matrix FP32 NPI from the exported ONNX and compile with it.",
    )
    parser.add_argument(
        "--mxfp6-matmul",
        action="store_true",
        help="Enable MXFP6 MatMul compilation. Disabled by default and independent of NPI generation.",
    )
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
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
    config.qeff_hca_attention_dp = attention_dp
    config.qeff_hca_compressed_kv_cp = hca_compressed_kv_cp
    config.qeff_hca_attn_blocks = hca_attn_blocks
    config.qeff_hca_folded_row_cache = True
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
    transform_kwargs = {
        "ctx_len": args.ctx_len,
        "seq_len": args.prefill_seq_len if args.prefill_only else 1,
        "bs": batch_size,
        "num_devices": len(args.device_group),
        "qaic_config": qaic_config,
        "prefill_only": args.prefill_only,
        "num_cores": args.num_cores,
    }
    if pipeline_parallel:
        transform_kwargs.update(
            seq_len=args.prefill_seq_len,
            prefill_seq_len=args.prefill_seq_len,
            mdp_num_partitions=args.mdp_num_partitions,
        )
    qeff_model.transform(**transform_kwargs)

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
    npi_summary_path = None
    if args.enable_npi:
        onnx_path, npi_path, npi_summary_path = generate_npi_artifacts(onnx_path)
        print(f"NPI_ONNX_PATH={onnx_path}")
        print(f"NPI_PATH={npi_path}")
        print(f"NPI_SUMMARY_PATH={npi_summary_path}")

    if args.export_only:
        return

    print(
        f"Compiling {graph_kind} specialization: seq_len={args.prefill_seq_len if args.prefill_only else 1}, ctx_len={args.ctx_len}"
    )
    compile_kwargs = {
        "onnx_path": str(onnx_path),
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
        "mxfp6_matmul": args.mxfp6_matmul,
        "user_tiled": True,
        "qaic_config": qaic_config,
        "dynamo": True,
    }
    if npi_path is not None:
        compile_kwargs["node_precision_info"] = str(npi_path)
    if pipeline_parallel:
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
    print(f"QPC_PATH={qpc_path}")
    print(f"SPECIALIZATIONS_PATH={qpc_path.parent / 'specializations.json'}")
    print(f"CUSTOM_IO_PATH={qpc_path.parent / 'custom_io.yaml'}")

    if args.compile_only or args.prefill_only:
        return

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prefill_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=tokenizer.chat_template or DEEPSEEK_V4_CHAT_TEMPLATE,
        thinking_mode="thinking",
    )
    print("Running qeff_model.generate()")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[prompt] * batch_size,
        device_ids=args.device_group,
        generation_len=args.generation_len,
        automation=args.automation,
        stream=False,
    )

    result_path = args.artifact_root / "generation_result.json"
    result = {
        "model_id": args.model_id,
        "prefill_prompt": prompt,
        "onnx_path": str(onnx_path),
        "npi_path": str(npi_path) if npi_path is not None else None,
        "npi_summary_path": str(npi_summary_path) if npi_summary_path is not None else None,
        "qpc_path": str(qpc_path),
        "generated_texts": exec_info.generated_texts,
        "generated_ids": [ids.tolist() if hasattr(ids, "tolist") else ids for ids in exec_info.generated_ids],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"RESULT_PATH={result_path}")
    print(f"GENERATED_TEXT={exec_info.generated_texts}")


if __name__ == "__main__":
    main()
