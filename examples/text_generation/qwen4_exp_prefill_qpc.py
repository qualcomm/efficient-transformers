"""Export, compile, and execute a prefill-only Qwen3.8-Flash-Next model.

The default workload follows ``attention/qsa/HANDOFF_FOR_QEFF_MIGRATION.md``:
batch 1, PL=1,024, CL=65,536, start position 64,512, one AI100, and four
cores. Use ``--layers 4`` for the leading four layers or ``--layers 48`` for
the full text model. A missing four-layer checkpoint is generated from the
local Hugging Face snapshot.

Example::

    HF_HUB_CACHE=/home/huggingface_hub \\
    /home/ochougul/envs/qwen38f/bin/python \\
        examples/text_generation/qwen4_exp_prefill_qpc.py \\
        --model-name-or-path Qwen/Qwen3.8-Flash-Next --layers 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_SIZE = 1
PREFILL_SEQ_LEN = 1024
CONTEXT_LENGTH = 65_536
START_POSITION = 64_512
NUM_CORES = 4
DEVICE_ID = 30
DEFAULT_LAYERS = 4
MODEL_DTYPE = torch.float32
COMPILER_OPTS_UNSUPPORTED = (
    "-aic-hoist-vtcm-loads=false -aic-op-stats-verbosity 2 -aic-userdma-async=0 -aic-hmx-async=0 -aic-sync-ts-starts"
)
QSA_CONFIG = {
    "indexer_dp": 1,
    "indexer_cp": 1,
    "attn_dp": 1,
    "attn_cp": 1,
    "num_cores_per_device": NUM_CORES,
    "indexer_num_blocks": 16,
    "indexer_ql_chunk": 512,
    "indexer_q_block_size": 128,
    "indexer_topk_blocking": 4,
    "sparse_q_block_size": 64,
    "sparse_kv_num_blocks": 8,
    "skip_kv": False,
}
HOST_INPUT_NAMES = ("input_ids", "ngram_embeddings", "position_ids")


def _resolve_snapshot(model_name_or_path: str) -> Path:
    source = Path(model_name_or_path).expanduser()
    if source.is_dir():
        return source.resolve()

    cache_dir = os.environ.get("HF_HUB_CACHE")
    if not cache_dir:
        raise OSError("Set HF_HUB_CACHE before resolving a Hugging Face model ID")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_name_or_path, cache_dir=cache_dir)).resolve()


def _window_config(snapshot: Path, layers: int):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    raw = json.loads((snapshot / "config.json").read_text())["text_config"]
    raw["dtype"] = "float32"
    raw["num_hidden_layers"] = layers
    raw["layer_types"] = raw["layer_types"][:layers]
    raw["ple_layer_ids"] = [layer_id for layer_id in raw.get("ple_layer_ids", []) if layer_id <= layers]
    raw["max_position_embeddings"] = CONTEXT_LENGTH
    config = Qwen4ExpTextConfig(**raw)
    config._attn_implementation = "eager"
    return config


def _ensure_four_layer_checkpoint(snapshot: Path, checkpoint: Path) -> None:
    if checkpoint.is_dir():
        required = (checkpoint / "config.json", checkpoint / "model.safetensors.index.json")
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Four-layer checkpoint is incomplete: {missing}")
        return
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint target exists but is not a directory: {checkpoint}")

    creator = Path(__file__).with_name("qwen4_exp_create_4layer_checkpoint.py")
    command = [sys.executable, str(creator), "--snapshot", str(snapshot), "--output", str(checkpoint)]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _build_model(snapshot: Path, checkpoint: Path | None, layers: int):
    sys.path.insert(0, str(REPO_ROOT))
    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

    config = _window_config(snapshot, layers)
    model_source = checkpoint if checkpoint is not None else snapshot
    model = QEFFAutoModelForCausalLM.from_pretrained(
        str(model_source),
        config=config,
        qaic_config=QSA_CONFIG,
        torch_dtype=MODEL_DTYPE,
    )
    for layer in model.model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if hasattr(attention, "configure_qsa"):
            attention.configure_qsa(
                QSA_CONFIG,
                ctx_len=CONTEXT_LENGTH,
                seq_len=PREFILL_SEQ_LEN,
                prefill_only=True,
            )
    model.model.eval()
    return model


def _make_position_ids(batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    positions = np.arange(START_POSITION, START_POSITION + PREFILL_SEQ_LEN, dtype=np.int64)
    return np.broadcast_to(positions[None, None, :], (4, batch_size, PREFILL_SEQ_LEN)).copy()


class Qwen4ExpPrefillRunner:
    """Run one prefill-only QPC invocation with zero-initialized inputs."""

    def __init__(self, qpc_path: Path, device_id: int):
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        self.session = QAICInferenceSession(qpc_path, device_ids=[device_id])
        retained = [
            name
            for name in (*self.session.input_names, *self.session.output_names)
            if self._basename(name).startswith(("gdn_", "ple_", "qsa_"))
        ]
        self.session.skip_buffers(retained)
        inputs_by_name = {self._basename(name): name for name in self.session.input_names}
        missing = set(HOST_INPUT_NAMES) - set(inputs_by_name)
        if missing:
            raise ValueError(f"Prefill QPC is missing host inputs: {sorted(missing)}")
        self.input_names = {name: inputs_by_name[name] for name in HOST_INPUT_NAMES}
        self.input_shapes = {name: self._binding_shape(binding) for name, binding in self.input_names.items()}
        self.input_dtypes = {name: self._binding_dtype(binding) for name, binding in self.input_names.items()}
        self.input_batch_size = self.input_shapes["input_ids"][0]

        self.logits_name = next((name for name in self.session.output_names if self._basename(name) == "logits"), None)
        if self.logits_name is None:
            raise ValueError(f"Prefill QPC is missing logits output: {self.session.output_names}")
        self.logits_shape = self._binding_shape(self.logits_name)
        self.logits_dtype = self._binding_dtype(self.logits_name)
        self.logits_is_bfloat16 = self.session.binding_is_bfloat16(self.logits_name)
        self.session.set_buffers({self.logits_name: np.empty(self.logits_shape, dtype=self.logits_dtype)})

    @staticmethod
    def _basename(name: str) -> str:
        return name.rsplit("/", 1)[-1]

    def _binding_shape(self, name: str) -> tuple[int, ...]:
        binding = self.session.bindings[self.session.binding_index_map[name]]
        return tuple(binding.dims)

    def _binding_dtype(self, name: str):
        binding = self.session.bindings[self.session.binding_index_map[name]]
        return self.session.aic_to_np_dtype_mapping[binding.type]

    def run(self, ple_embed_dim: int, expected_batch_size: int) -> np.ndarray:
        input_ids_shape = self.input_shapes["input_ids"]
        if len(input_ids_shape) != 2 or input_ids_shape[1] != PREFILL_SEQ_LEN:
            raise ValueError(f"Unexpected input_ids binding shape: {input_ids_shape}")
        batch_size = input_ids_shape[0]
        if batch_size != expected_batch_size:
            raise ValueError(
                f"QPC input batch {batch_size} does not match requested compile batch {expected_batch_size}"
            )
        inputs = {
            "input_ids": np.zeros(input_ids_shape, dtype=np.int64),
            "ngram_embeddings": np.zeros((batch_size, PREFILL_SEQ_LEN, ple_embed_dim), dtype=np.float32),
            "position_ids": _make_position_ids(batch_size),
        }
        bound_inputs = {}
        for name, value in inputs.items():
            binding = self.input_names[name]
            if tuple(value.shape) != self.input_shapes[name]:
                raise ValueError(f"{name} binding shape is {self.input_shapes[name]}, expected {value.shape}")
            bound_inputs[binding] = value.astype(self.input_dtypes[name], copy=False)

        outputs = self.session.run(bound_inputs)
        logits = next((value for name, value in outputs.items() if self._basename(name) == "logits"), None)
        if logits is None:
            raise ValueError(f"QAIC run did not return logits: {sorted(outputs)}")
        logits = np.asarray(logits)
        if self.logits_is_bfloat16:
            logits = (logits.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            logits = logits.astype(np.float32, copy=False)
        if logits.shape != self.logits_shape or not np.isfinite(logits).all():
            raise ValueError(f"Invalid prefill logits shape/values: shape={logits.shape}")
        return logits


def _validate_export(onnx_path: Path, retained_state_names: list[str], expected_batch_size: int) -> dict[str, int]:
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    graph_input_values = {value.name.rsplit("/", 1)[-1]: value for value in graph.graph.input}
    graph_inputs = set(graph_input_values)
    graph_outputs = {value.name.rsplit("/", 1)[-1] for value in graph.graph.output}
    required_inputs = set(HOST_INPUT_NAMES) | set(retained_state_names)
    required_outputs = {"logits", *(f"{name}_RetainedState" for name in retained_state_names)}
    missing_inputs = required_inputs - graph_inputs
    missing_outputs = required_outputs - graph_outputs
    if missing_inputs or missing_outputs:
        raise ValueError(
            f"Exported retained-state ABI is incomplete: missing inputs={sorted(missing_inputs)}, "
            f"missing outputs={sorted(missing_outputs)}"
        )

    input_ids_shape = [
        dim.dim_value if dim.HasField("dim_value") else None
        for dim in graph_input_values["input_ids"].type.tensor_type.shape.dim
    ]
    if len(input_ids_shape) != 2 or input_ids_shape != [expected_batch_size, PREFILL_SEQ_LEN]:
        raise ValueError(
            f"Exported input_ids shape must match batch={expected_batch_size}, PL={PREFILL_SEQ_LEN}: {input_ids_shape}"
        )

    nodes = list(graph.graph.node)
    for function in graph.functions:
        nodes.extend(function.node)
    topk_count = sum(node.op_type.lower() == "topk" for node in nodes)
    if not topk_count:
        raise ValueError("Exported prefill graph contains no TopK node for sparse QSA selection")
    return {"retained_state_count": len(retained_state_names), "topk_node_count": topk_count}


def _compile_and_run(args: argparse.Namespace, snapshot: Path, checkpoint: Path | None) -> None:
    compile_dir = args.compile_dir.resolve()
    if compile_dir.exists() and any(compile_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty artifact directory: {compile_dir}")
    compile_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QEFF_HOME", str(compile_dir / "qeff_home"))
    os.environ.setdefault("QAIC_COMPILER_OPTS_UNSUPPORTED", COMPILER_OPTS_UNSUPPORTED)
    print(f"QEFF_HOME={os.environ['QEFF_HOME']}")
    print(f"QAIC_COMPILER_OPTS_UNSUPPORTED={os.environ['QAIC_COMPILER_OPTS_UNSUPPORTED']}")

    model = _build_model(snapshot, checkpoint, args.layers)
    qpc_path = Path(
        model.compile(
            compile_dir=str(compile_dir),
            batch_size=args.batch_size,
            ctx_len=CONTEXT_LENGTH,
            prefill_seq_len=PREFILL_SEQ_LEN,
            prefill_only=True,
            num_devices=1,
            num_cores=NUM_CORES,
            mxfp6_matmul=True,
            mxint8_kv_cache=True,
            offload_pt_weights=False,
            dynamo=True,
            use_onnx_subfunctions=False,
            aic_hw_version="ai100",
            allow_mxint8_mdp_io=True,
            user_tiled=True,
            split_model_io=True,
            qaic_config=QSA_CONFIG,
        )
    )
    qpc_binary = qpc_path if (qpc_path / "programqpc.bin").is_file() else qpc_path / "qpc"
    if not (qpc_binary / "programqpc.bin").is_file():
        raise FileNotFoundError(f"Compiled QPC binary not found under {qpc_path}")

    export_metrics = _validate_export(Path(model.onnx_path), model.model.get_retained_state_names(), args.batch_size)
    runner = Qwen4ExpPrefillRunner(qpc_path, args.device_id)
    logits = runner.run(model.model.config.ple_embed_dim, args.batch_size)
    print(f"prefill_qpc={qpc_path}")
    print(f"onnx_path={model.onnx_path}")
    print(
        f"layers={args.layers} batch_size_specialization={args.batch_size} "
        f"qpc_input_batch_size={runner.input_batch_size} seq_len={PREFILL_SEQ_LEN} ctx_len={CONTEXT_LENGTH}"
    )
    print(
        f"retained_state_count={export_metrics['retained_state_count']} topk_node_count={export_metrics['topk_node_count']}"
    )
    print(f"logits_shape={logits.shape} logits_finite=true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--checkpoint", type=Path, help="optional checkpoint directory for the selected layer count")
    parser.add_argument("--layers", choices=(4, 48), type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--four-layer-checkpoint",
        type=Path,
        default=Path("qwen3.8_flash_experiment/example_repro_20261013/qwen38_flash_4layer_checkpoint_v2"),
        help="checkpoint generated from the snapshot when --layers 4 and --checkpoint is omitted",
    )
    parser.add_argument(
        "--compile-dir",
        type=Path,
        help="fresh artifact directory (defaults to a timestamped directory under attention_benchmarks/)",
    )
    parser.add_argument("--device-id", type=int, default=DEVICE_ID, help="physical QAIC device id")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.compile_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.compile_dir = Path(f"attention_benchmarks/qwen38_flash_prefill_{args.layers}layer_cores4_{stamp}")
    args.compile_dir = args.compile_dir.resolve()
    snapshot = _resolve_snapshot(args.model_name_or_path)

    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if args.layers == 4 and checkpoint is None:
        checkpoint = args.four_layer_checkpoint.resolve()
        _ensure_four_layer_checkpoint(snapshot, checkpoint)
    if checkpoint is not None and not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    _compile_and_run(args, snapshot, checkpoint)


if __name__ == "__main__":
    main()
