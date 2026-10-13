#!/usr/bin/env python3
"""Build and run one decode step for a four-layer Qwen3.8-Flash-Next window.

This is a decode-only DP16/CP1 flow.  It prunes the local checkpoint to the
leading four layers, preserving the GDN, PLE, and QSA layer schedule.

Example::

    /home/ochougul/envs/qwen38f/bin/python \
        examples/text_generation/qwen4_exp_dp16_decode_qpc.py \
        --snapshot /home/huggingface_hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/<revision> \

Use ``--checkpoint`` to reuse an existing four-layer checkpoint, or
``--qpc`` to skip export/compile and replay an existing QPC.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


NUM_DEVICES = 16
NUM_CORES = 10
BATCH_SIZE = 16
LAYERS = 4
CONTEXT_LENGTH = 65_536


def build_model(snapshot: Path, checkpoint: Path, compile_dir: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen4_exp_decode_parity import (
        _prune_weight_free_checkpoint,
        _window_config,
    )
    qaic_config = {
        "indexer_dp": 16,
        "indexer_cp": 1,
        "attn_dp": 16,
        "attn_cp": 1,
        "indexer_num_blocks": 4,
        "indexer_ql_chunk": 512,
        "indexer_q_block_size": 512,
        "indexer_topk_blocking": 4,
        "sparse_q_block_size": 16,
        "sparse_kv_num_blocks": 8,
        "num_cores_per_device": NUM_CORES,
    }
    config = _window_config(snapshot, LAYERS, CONTEXT_LENGTH)
    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

    return QEFFAutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        config=config,
        qaic_config=qaic_config,
        weight_free=True,
        torch_dtype=torch.float32,
    )


def is_retained_state(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith(("gdn_", "ple_", "qsa_"))


def run_one_decode_step(qpc_path: Path) -> None:
    from QEfficient.generation.cloud_infer import QAICInferenceSession

    session = QAICInferenceSession(qpc_path, device_ids=None)
    retained = [
        name
        for name in (*session.input_names, *session.output_names)
        if is_retained_state(name) or name.rsplit("/", 1)[-1].endswith("_RetainedState")
    ]
    session.skip_buffers(retained)

    basename_to_name = {name.rsplit("/", 1)[-1]: name for name in session.input_names}
    required = {"input_ids", "ngram_embeddings", "position_ids"}
    missing = required - basename_to_name.keys()
    if missing:
        raise RuntimeError(f"QPC is missing host inputs: {sorted(missing)}")

    def binding_shape(basename: str) -> tuple[int, ...]:
        binding_name = basename_to_name[basename]
        return tuple(session.bindings[session.binding_index_map[binding_name]].dims)

    input_ids = np.zeros(binding_shape("input_ids"), dtype=np.int64)
    ngram_embeddings = np.zeros(binding_shape("ngram_embeddings"), dtype=np.float32)
    position_ids = np.zeros(binding_shape("position_ids"), dtype=np.int64)
    host_inputs = {
        basename_to_name["input_ids"]: input_ids,
        basename_to_name["ngram_embeddings"]: ngram_embeddings,
        basename_to_name["position_ids"]: position_ids,
    }

    logits_name = next(
        (name for name in session.output_names if name.rsplit("/", 1)[-1] == "logits"),
        None,
    )
    if logits_name is None:
        raise RuntimeError(f"QPC is missing logits output: {session.output_names}")
    logits_binding = session.bindings[session.binding_index_map[logits_name]]
    session.set_buffers({logits_name: np.zeros(tuple(logits_binding.dims), dtype=np.float32)})
    outputs = session.run(host_inputs)
    logits = next(value for name, value in outputs.items() if name.rsplit("/", 1)[-1] == "logits")
    print(f"executed one decode step; logits shape={logits.shape}; next token={int(np.argmax(logits))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="local Qwen3.8-Flash-Next snapshot directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("qwen3.8_flash_experiment/example_repro_20261013/qwen38_flash_4layer_checkpoint_v2"),
        help="local four-layer checkpoint; avoids rebuilding it from the full snapshot",
    )
    parser.add_argument(
        "--compile-dir",
        type=Path,
        default=Path("attention_benchmarks/qsa_4layer_dp16"),
        help="directory for exported ONNX and QPC artifacts",
    )
    parser.add_argument("--qpc", type=Path, help="reuse an existing QPC and skip model export/compile")
    args = parser.parse_args()
    args.snapshot = args.snapshot.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.compile_dir = args.compile_dir.resolve()

    if args.qpc is None:
        os.environ.setdefault("QEFF_HOME", str(args.compile_dir))
        if not args.checkpoint.is_dir():
            raise FileNotFoundError(f"Four-layer checkpoint directory not found: {args.checkpoint}")
        qeff_model = build_model(args.snapshot, args.checkpoint, args.compile_dir)
        qpc_path = Path(
            qeff_model.compile(
                compile_dir=str(args.compile_dir),
                batch_size=BATCH_SIZE,
                ctx_len=CONTEXT_LENGTH,
                prefill_seq_len=1,
                num_devices=NUM_DEVICES,
                num_cores=NUM_CORES,
                mxfp6_matmul=True,
                offload_pt_weights=False,
                dynamo=True,
                use_onnx_subfunctions=False,
                aic_hw_version="ai100",
                mxint8_kv_cache=True,
                allow_mxint8_mdp_io=True,
            )
        )
    else:
        qpc_path = args.qpc

    if not (qpc_path / "programqpc.bin").is_file():
        raise FileNotFoundError(f"QPC binary not found: {qpc_path / 'programqpc.bin'}")
    print(f"running QPC: {qpc_path}")
    run_one_decode_step(qpc_path)


if __name__ == "__main__":
    main()
