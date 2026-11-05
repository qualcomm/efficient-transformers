#!/usr/bin/env python3
"""Run an existing four-layer Qwen4-Exp QPC with QAIC opstats enabled.

The input batch is generated from the exported ONNX graph.  Retained-state
bindings are deliberately omitted from the batch JSON: QAIC initializes and
keeps those buffers on device between iterations.

Example::

    ~/envs/main/bin/python \
        examples/text_generation/qwen4_exp_dp16_decode_qpc_opstats.py \
        --qpc attention_benchmarks/qsa_4layer_dp16/qpc-c513bf97d82c79fe/qpc \
        --onnx attention_benchmarks/qsa_4layer_dp16/Qwen4ExpForCausalLM/ \
            Qwen4ExpForCausalLM-e3dbcc19b08cefc8/Qwen4ExpForCausalLM.onnx \
        --device-list 30..45
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import yaml

ONNX_DTYPES = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.BOOL: np.bool_,
}


def _shape(value_info: onnx.ValueInfoProto, symbols: dict[str, int]) -> tuple[int, ...]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.dim_param in symbols:
            shape.append(symbols[dim.dim_param])
        else:
            raise ValueError(f"No concrete value supplied for ONNX dimension {dim.dim_param!r} in {value_info.name}")
    return tuple(shape)


def _custom_io_dtypes(custom_io_path: Path | None) -> dict[str, np.dtype]:
    if custom_io_path is None or not custom_io_path.is_file():
        return {}
    precision_to_dtype = {
        "float": np.float32,
        "float16": np.float16,
        "float32": np.float32,
        "int32": np.int32,
        "int64": np.int64,
    }
    entries = yaml.safe_load(custom_io_path.read_text()) or []
    return {
        entry["IOName"]: np.dtype(precision_to_dtype[entry["Precision"]])
        for entry in entries
        if entry.get("IOName") and entry.get("Precision") in precision_to_dtype
    }


def _write_batch(
    onnx_path: Path,
    custom_io_path: Path | None,
    io_dir: Path,
    batch_size: int,
    ctx_len: int,
    pooled_ctx_len: int,
) -> Path:
    model = onnx.load(str(onnx_path), load_external_data=False)
    custom_dtypes = _custom_io_dtypes(custom_io_path)
    symbols = {
        "batch_size": batch_size,
        "ctx_len": ctx_len,
        "pooled_ctx_len": pooled_ctx_len,
    }
    data_dir = io_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    host_input_names = {"input_ids", "ngram_embeddings", "position_ids"}
    retained = {output.name.removesuffix("_RetainedState") for output in model.graph.output}
    entries = []
    for value_info in model.graph.input:
        if (
            value_info.name not in host_input_names
            or value_info.name in initializer_names
            or value_info.name in retained
            or value_info.name.endswith("_RetainedState")
        ):
            continue
        dtype = custom_dtypes.get(value_info.name) or ONNX_DTYPES.get(value_info.type.tensor_type.elem_type)
        if dtype is None:
            raise ValueError(f"Unsupported ONNX dtype for {value_info.name}: {value_info.type.tensor_type.elem_type}")
        shape = _shape(value_info, symbols)
        path = data_dir / f"{value_info.name.replace('/', '_')}.raw"
        np.zeros(shape, dtype=dtype).tofile(path)
        entries.append(
            {
                "path": str(path.relative_to(io_dir)),
                "io-direction": "in",
                "elem-size": np.dtype(dtype).itemsize,
                "map-to": value_info.name,
                "dims": list(shape),
            }
        )

    for value_info in model.graph.output:
        if value_info.name.endswith("_RetainedState"):
            continue
        dtype = custom_dtypes.get(value_info.name) or ONNX_DTYPES.get(value_info.type.tensor_type.elem_type)
        if dtype is None:
            raise ValueError(f"Unsupported ONNX dtype for {value_info.name}: {value_info.type.tensor_type.elem_type}")
        shape = _shape(value_info, symbols)
        path = data_dir / f"{value_info.name.replace('/', '_')}.raw"
        np.zeros(shape, dtype=dtype).tofile(path)
        entries.append(
            {
                "path": str(path.relative_to(io_dir)),
                "io-direction": "out",
                "elem-size": np.dtype(dtype).itemsize,
                "map-to": value_info.name,
                "dims": list(shape),
            }
        )

    batch_json = io_dir / "aic_batch_io.json"
    batch_json.write_text(json.dumps({"IO-files": [entries]}, indent=2) + "\n")
    return batch_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qpc", type=Path, required=True, help="QPC directory containing programqpc.bin")
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX graph used to derive dummy IO")
    parser.add_argument(
        "--compile-qpc",
        type=Path,
        default=None,
        help="Compile this profiling QPC directly from --onnx before running it",
    )
    parser.add_argument("--custom-io", type=Path, default=None, help="custom_io.yaml used to override ONNX dtypes")
    parser.add_argument("--specializations", type=Path, default=None)
    parser.add_argument("--mdp-config", type=Path, default=None)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--work-dir", type=Path, default=None, help="Output directory for IO, raw stats, and opstats")
    parser.add_argument("--device-list", default="30..45", help="qaic-runner device list, e.g. 30..45")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile-start-iter", type=int, default=20)
    parser.add_argument("--profile-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ctx-len", type=int, default=16)
    parser.add_argument("--pooled-ctx-len", type=int, default=8)
    args = parser.parse_args()

    qpc = args.qpc.resolve()
    onnx_path = args.onnx.resolve()
    custom_io = (args.custom_io or qpc.parent / "custom_io.yaml").resolve()
    specializations = (args.specializations or qpc.parent / "specializations.json").resolve()
    mdp_config = (args.mdp_config or qpc.parent / "mdp_ts_16.json").resolve()
    compile_qpc = args.compile_qpc.resolve() if args.compile_qpc else None
    if compile_qpc is not None:
        if compile_qpc.exists():
            raise FileExistsError(f"Profiling QPC output already exists: {compile_qpc}")
        for path in (custom_io, specializations, mdp_config):
            if not path.is_file():
                raise FileNotFoundError(path)
        compile_qpc.parent.mkdir(parents=True, exist_ok=True)
        mdp_for_compile = mdp_config
        mdp_data = json.loads(mdp_config.read_text())
        configured_cores = {
            device["numCores"] for partition in mdp_data["partitions"] for device in partition["devices"]
        }
        if configured_cores != {args.num_cores}:
            for partition in mdp_data["partitions"]:
                for device in partition["devices"]:
                    device["numCores"] = args.num_cores
            mdp_for_compile = compile_qpc.parent / f"mdp_ts_{args.num_cores}.json"
            mdp_for_compile.write_text(json.dumps(mdp_data, indent=2) + "\n")
        compile_cmd = [
            "/opt/qti-aic/exec/qaic-compile",
            "-aic-hw",
            "-aic-hw-version=ai100",
            f"-m={onnx_path}",
            "-retained-state",
            "-convert-to-fp16",
            f"-aic-num-cores={args.num_cores}",
            "-allow-mxint8-mdp-io",
            f"-mdp-load-partition-config={mdp_for_compile}",
            f"-network-specialization-config={specializations}",
            f"-custom-IO-list-file={custom_io}",
            f"-aic-binary-dir={compile_qpc}",
            "-stats-level=70",
            "-ddr-stats",
            "-aic-pmu-recipe=KernelUtil",
            "-aic-perf-metrics",
        ]
        compile_env = os.environ.copy()
        compile_env.setdefault(
            "QAIC_COMPILER_OPTS_UNSUPPORTED",
            "-aic-hoist-vtcm-loads=false -aic-op-stats-verbosity 2 "
            "-aic-userdma-async=0 -aic-hmx-async=0 -aic-sync-ts-starts",
        )
        print("$", " ".join(compile_cmd), flush=True)
        subprocess.run(compile_cmd, check=True, env=compile_env)
        # Profile the QPC just compiled with stats enabled, not the reference
        # QPC supplied for its compile-time artifacts.
        qpc = compile_qpc
    if not (qpc / "programqpc.bin").is_file():
        raise FileNotFoundError(qpc / "programqpc.bin")
    if args.work_dir is None:
        work_dir = qpc.parent / "opstats_run"
    else:
        work_dir = args.work_dir.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    io_dir = work_dir / "io"
    raw_stats = work_dir / "raw_device_stats"
    opstats = work_dir / "opstats"
    raw_stats.mkdir(parents=True, exist_ok=True)
    opstats.mkdir(parents=True, exist_ok=True)
    batch_json = _write_batch(onnx_path, custom_io, io_dir, args.batch_size, args.ctx_len, args.pooled_ctx_len)

    runner = "/opt/qti-aic/exec/qaic-runner"
    opstats_tool = "/opt/qti-aic/exec/qaic-opstats"
    runner_cmd = [
        runner,
        "-D",
        args.device_list,
        "-t",
        str(qpc),
        "-n",
        str(args.iterations),
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        str(args.profile_start_iter),
        "--aic-profiling-num-samples",
        str(args.profile_samples),
        "--aic-profiling-out-dir",
        str(raw_stats),
        "--aic-batch-json-input",
        str(batch_json),
    ]
    print("$", " ".join(runner_cmd), flush=True)
    subprocess.run(runner_cmd, check=True)

    opstats_cmd = [
        opstats_tool,
        "--qpc",
        str(qpc / "programqpc.bin"),
        "--input-dir",
        str(raw_stats),
        "--output-dir",
        str(opstats),
        "--summary",
        "--trace",
    ]
    print("$", " ".join(opstats_cmd), flush=True)
    subprocess.run(opstats_cmd, check=True)
    print(f"opstats written to {opstats}")


if __name__ == "__main__":
    main()
