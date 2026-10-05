#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generate QAIC per-op traces for a four-layer GLM-5.3 prefill graph."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = SCRIPT_DIR / "glm53_4layer_prefill_trace_artifacts"
PERF_RUNNER = "/opt/qti-aic/exec/qaic-runner"
PERF_OPSTATS = "/opt/qti-aic/exec/qaic-opstats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hf-cache", type=Path, default=Path("/home/huggingface_hub"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=512)
    parser.add_argument("--ctx-len", type=int, default=65536)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--hw-version", choices=("ai100", "ai200"), default="ai100")
    parser.add_argument("--device-list", default=None, help="qaic-runner device mapping, for example 1 or 0:1.")
    parser.add_argument("--attention-preset", default="dsa_prefill_cp1")
    parser.add_argument("--attention-qaic-json", default=None)
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Write profiling scripts without running them.")
    parser.add_argument("--perf-num-iters", type=int, default=30)
    parser.add_argument("--perf-profile-start-iter", type=int, default=20)
    parser.add_argument("--perf-num-samples", type=int, default=2)
    parser.add_argument("--perf-stats-level", type=int, default=70)
    return parser.parse_args()


@contextmanager
def intercept_qeff_compile_subprocess():
    """Capture QEfficient's production compiler command without executing it."""
    import QEfficient.base.modeling_qeff as modeling_qeff

    captured: list[list[str]] = []
    real_run = modeling_qeff.subprocess.run

    def fake_run(command, *args, **kwargs):
        captured.append([str(part) for part in command])
        return subprocess.CompletedProcess(command, returncode=0, stdout=b"", stderr=b"")

    modeling_qeff.subprocess.run = fake_run
    try:
        yield captured
    finally:
        modeling_qeff.subprocess.run = real_run


def retained_state_names(qeff_model) -> set[str]:
    config = qeff_model.model.config
    names: set[str] = set()
    for layer_idx in range(config.num_hidden_layers):
        for cache_name in (f"compressed_kv.{layer_idx}", f"k_pe.{layer_idx}"):
            names.add(cache_name)
            names.add(f"{cache_name}_RetainedState")
            names.add(f"{cache_name}_InternalRetainedState")
    for cache_idx, _ in enumerate(qeff_model.model.get_indexer_cache_layers(config)):
        cache_name = f"indexer_key.{cache_idx}"
        names.add(cache_name)
        names.add(f"{cache_name}_RetainedState")
        names.add(f"{cache_name}_InternalRetainedState")
    return names


def _onnx_value_info(model, name: str):
    values = [*model.graph.input, *model.graph.output, *model.graph.value_info]
    for value in values:
        if value.name == name or value.name.rsplit("/", 1)[-1] == name:
            return value
    raise KeyError(f"ONNX value {name!r} was not found.")


def _onnx_numpy_dtype(value_info) -> np.dtype:
    import onnx

    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(value_info.type.tensor_type.elem_type))


def build_synthetic_io(
    onnx_path: Path,
    *,
    batch_size: int,
    prefill_seq_len: int,
    vocab_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    input_ids_info = _onnx_value_info(model, "input_ids")
    position_ids_info = _onnx_value_info(model, "position_ids")
    logits_info = _onnx_value_info(model, "logits")

    inputs = {
        "input_ids": np.zeros(
            (batch_size, prefill_seq_len),
            dtype=_onnx_numpy_dtype(input_ids_info),
        ),
        "position_ids": np.broadcast_to(
            np.arange(prefill_seq_len, dtype=_onnx_numpy_dtype(position_ids_info)),
            (batch_size, prefill_seq_len),
        ).copy(),
    }
    outputs = {
        "logits": np.zeros(
            (batch_size, prefill_seq_len, vocab_size),
            dtype=_onnx_numpy_dtype(logits_info),
        )
    }
    return inputs, outputs


def dump_qaic_io(
    inputs: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    io_dir: Path,
    *,
    skip_names: set[str],
) -> Path:
    data_dir = io_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    io_list: list[dict[str, Any]] = []
    for direction, tensors in (("in", inputs), ("out", outputs)):
        for name, array in tensors.items():
            if name in skip_names:
                continue
            raw_path = data_dir / f"{name}.raw"
            array.tofile(raw_path)
            io_list.append(
                {
                    "path": f"data/{name}.raw",
                    "io-direction": direction,
                    "elem-size": array.itemsize,
                    "map-to": name,
                    "dims": list(array.shape),
                }
            )
    json_path = io_dir / "aic_batch_io.json"
    json_path.write_text(json.dumps({"IO-files": [io_list]}, indent=2), encoding="utf-8")
    return json_path


def write_perf_scripts(
    artifact_dir: Path,
    base_compile_cmd: list[str],
    io_json_path: Path,
    *,
    perf_num_iters: int,
    perf_profile_start_iter: int,
    perf_num_samples: int,
    perf_stats_level: int,
    device_list: str | None,
) -> Path:
    perf_dir = artifact_dir / "perf_dump"
    qpc_dir = artifact_dir / "qpc"
    stats_dir = perf_dir / "raw_device_stats"
    opstats_dir = perf_dir / "opstats"
    for directory in (perf_dir, stats_dir, opstats_dir):
        directory.mkdir(parents=True, exist_ok=True)

    compile_cmd = list(base_compile_cmd) + [
        f"-stats-level={perf_stats_level}",
        "-ddr-stats",
        "-aic-pmu-recipe=KernelUtil",
        "-aic-perf-metrics",
    ]
    runner_cmd = [
        PERF_RUNNER,
        "-t",
        str(qpc_dir),
        "-n",
        str(perf_num_iters),
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        str(perf_profile_start_iter),
        "--aic-profiling-num-samples",
        str(perf_num_samples),
        "--aic-profiling-out-dir",
        str(stats_dir),
        "--aic-batch-json-input",
        str(io_json_path),
    ]
    if device_list is not None:
        runner_cmd.extend(["-D", device_list])
    opstats_cmd = [
        PERF_OPSTATS,
        "--qpc",
        str(qpc_dir / "programqpc.bin"),
        "--input-dir",
        str(stats_dir),
        "--output-dir",
        str(opstats_dir),
        "--summary",
        "--trace",
    ]

    scripts = {
        "compile_perf.sh": [f"rm -rf {shlex.quote(str(qpc_dir))}", shlex.join(compile_cmd)],
        "run_perf.sh": [
            f"rm -rf {shlex.quote(str(stats_dir))}",
            f"rm -rf {shlex.quote(str(opstats_dir))}",
            f"mkdir -p {shlex.quote(str(stats_dir))} {shlex.quote(str(opstats_dir))}",
            shlex.join(runner_cmd),
        ],
        "prefill_perf.sh": [shlex.join(opstats_cmd)],
    }
    for name, commands in scripts.items():
        script_path = perf_dir / name
        script_path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n",
            encoding="utf-8",
        )
        script_path.chmod(0o755)
    return perf_dir


def run_script(script_path: Path) -> None:
    log_path = script_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            [str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL),
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed: {script_path}\nSee log: {log_path}\n\n{log_path.read_text()}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_id).expanduser()
    if model_path.exists():
        args.model_id = str(model_path.resolve())
    args.hf_cache = args.hf_cache.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    if args.num_hidden_layers != 4:
        raise ValueError("This reduced GLM-5.3 profiling flow supports exactly four layers.")
    if args.prefill_seq_len <= 1 or args.prefill_seq_len > args.ctx_len:
        raise ValueError("prefill_seq_len must be greater than 1 and no larger than ctx_len.")
    if args.perf_profile_start_iter + args.perf_num_samples > args.perf_num_iters:
        raise ValueError("profile start iteration plus sample count must not exceed total iterations.")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    temp_root = args.artifact_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(args.hf_cache)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["QEFF_HOME"] = str(args.artifact_root)
    os.environ["TMPDIR"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from QEfficient import QEFFAutoModelForCausalLM
    from examples.glm.glm53_four_layer_prefill_compile import (
        PREFILL_ATTENTION_PRESETS,
        install_partial_fp8_dequant_patch,
        validate_prefill_dimensions,
    )

    if args.attention_preset not in PREFILL_ATTENTION_PRESETS:
        raise ValueError(
            f"Unknown attention preset {args.attention_preset!r}; "
            f"choose one of {sorted(PREFILL_ATTENTION_PRESETS)}."
        )
    qaic_config = copy.deepcopy(PREFILL_ATTENTION_PRESETS[args.attention_preset])
    if args.attention_qaic_json:
        qaic_config.update(json.loads(args.attention_qaic_json))
    validate_prefill_dimensions(
        qaic_config,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
    )

    config = AutoConfig.from_pretrained(
        args.model_id,
        cache_dir=str(args.hf_cache),
        local_files_only=args.local_files_only,
    )
    config.num_hidden_layers = args.num_hidden_layers
    if args.attention_preset.startswith("dense"):
        config.layer_types = ["full_attention"] * args.num_hidden_layers
    else:
        config.layer_types = list(config.layer_types[: args.num_hidden_layers])
        config.indexer_types = list(config.indexer_types[: args.num_hidden_layers])
    config.use_cache = True
    config.torch_dtype = torch.float32
    config.dtype = torch.float32

    install_partial_fp8_dequant_patch()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        cache_dir=str(args.hf_cache),
        local_files_only=args.local_files_only,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).eval()
    qeff_model = QEFFAutoModelForCausalLM(
        hf_model,
        pretrained_model_name_or_path=args.model_id,
        qaic_config=qaic_config,
    )
    qeff_model.transform(
        ctx_len=args.ctx_len,
        seq_len=args.prefill_seq_len,
        bs=args.batch_size,
        num_devices=args.num_devices,
        qaic_config=qaic_config,
        num_cores=args.num_cores,
        prefill_only=True,
    )

    export_root = args.artifact_root / "onnx"
    compile_root = args.artifact_root / "compile"
    print("Exporting the multi-token prefill graph")
    onnx_path = Path(
        qeff_model.export(
            export_dir=str(export_root),
            prefill_only=True,
            prefill_seq_len=args.prefill_seq_len,
            offload_pt_weights=False,
            dynamo=args.dynamo,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
        )
    )
    print(f"ONNX_PATH={onnx_path}")

    print("Capturing the production prefill compiler command")
    with intercept_qeff_compile_subprocess() as captured:
        qpc_dir = Path(
            qeff_model.compile(
                onnx_path=str(onnx_path),
                compile_dir=str(compile_root),
                prefill_seq_len=args.prefill_seq_len,
                ctx_len=args.ctx_len,
                batch_size=args.batch_size,
                num_devices=args.num_devices,
                num_cores=args.num_cores,
                aic_hw_version=args.hw_version,
                prefill_only=True,
                use_onnx_subfunctions=args.use_onnx_subfunctions,
                offload_pt_weights=False,
            )
        )
    if len(captured) != 1:
        raise RuntimeError(
            f"Expected one intercepted qaic-compile invocation, got {len(captured)}. "
            "Use a fresh --artifact-root if the hashed QPC already exists."
        )

    inputs, outputs = build_synthetic_io(
        onnx_path,
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        vocab_size=config.vocab_size,
    )
    io_json_path = dump_qaic_io(
        inputs,
        outputs,
        qpc_dir.parent / "io",
        skip_names=retained_state_names(qeff_model),
    )
    perf_dir = write_perf_scripts(
        qpc_dir.parent,
        captured[0],
        io_json_path,
        perf_num_iters=args.perf_num_iters,
        perf_profile_start_iter=args.perf_profile_start_iter,
        perf_num_samples=args.perf_num_samples,
        perf_stats_level=args.perf_stats_level,
        device_list=args.device_list,
    )
    print(f"QPC_DIR={qpc_dir}")
    print(f"IO_JSON={io_json_path}")
    print(f"PERF_DIR={perf_dir}")

    if args.prepare_only:
        return
    for script_name in ("compile_perf.sh", "run_perf.sh", "prefill_perf.sh"):
        script_path = perf_dir / script_name
        print(f"Running {script_path}")
        run_script(script_path)
    print(f"Traces: {perf_dir}/opstats/*.trace.json")
    print(f"Summaries: {perf_dir}/opstats/*.summary.txt")


if __name__ == "__main__":
    main()
