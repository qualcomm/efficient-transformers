"""Compare the prefill-only QEff QSA block with its HF sparse-attention reference.

The check uses the local four-layer Qwen3.8-Flash-Next checkpoint, layer 3,
and the aligned PL=1024 / CL=65536 workload from the QSA handoff. It does not
export or compile a graph and does not enter the GDN path.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qwen4_exp_prefill_qsa_qpc as qsa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            "/home/huggingface_hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("qwen3.8_flash_experiment/example_repro_20261013/qwen38_flash_4layer_checkpoint_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("attention_benchmarks/qsa_prefill_parity_cores4_20261103"),
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    if not os.environ.get("HF_HUB_CACHE"):
        raise OSError("Set HF_HUB_CACHE to the approved local Hugging Face cache before loading model weights")
    snapshot = args.snapshot.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    for path in (snapshot / "config.json", checkpoint / "model.safetensors.index.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty parity output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QEFF_HOME"] = str(output_dir / "qeff_home")

    from QEfficient.transformers.models.qwen4_exp.modeling_qwen4_exp import qeff_prepare_qwen4_exp_mrope

    qsa._QEFF_PREPARE_MROPE = qeff_prepare_qwen4_exp_mrope
    config = qsa._load_config(snapshot)
    hf_attention, qeff_attention = qsa._load_attention_pair(config, checkpoint)
    rotary_tables = qsa._rotary_tables(config)
    hf_output, hf_cache, graph, qeff_outputs, inputs = qsa._build_parity_inputs(
        config,
        hf_attention,
        qeff_attention,
        rotary_tables,
    )

    output_metrics = qsa._compare(
        hf_output,
        qeff_outputs[0],
        atol=args.atol,
        rtol=args.rtol,
        label="HF/QEff prefill-only QSA",
    )
    retained_metrics = qsa._validate_retained_states(
        config,
        hf_cache,
        qeff_outputs,
        rotary_tables,
        qeff_attention,
    )
    report = {
        "result": "pass",
        "model": "Qwen3.8-Flash-Next QSA only (layer 3)",
        "snapshot": str(snapshot),
        "checkpoint": str(checkpoint),
        "workload": {
            "batch_size": qsa.BATCH_SIZE,
            "prefill_seq_len": qsa.PREFILL_SEQ_LEN,
            "ctx_len": qsa.CONTEXT_LENGTH,
            "start_position": qsa.START_POSITION,
            "qsa_config": qsa.QSA_CONFIG,
            "model_dtype": "float32",
            "prefix_cache": "zero initialized",
        },
        "attention_output": output_metrics,
        "retained_states": retained_metrics,
        "reference_note": (
            "HF projections, rotary embeddings, and HF indexer selection are used. "
            "Selected-token gathers replace HF's dense masked QxCL attention matrix so the full workload fits in memory."
        ),
    }
    report_path = output_dir / "prefill_qsa_parity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")

    del hf_attention, qeff_attention, hf_output, hf_cache, graph, qeff_outputs, inputs
    gc.collect()


if __name__ == "__main__":
    main()
