#!/usr/bin/env python3
"""Compile, profile, and extract layerwise opstats for four-layer Qwen3.8 Flash.

The model used by this workflow is the four-layer Qwen4-Exp/Qwen3.8-Flash
configuration whose layer pattern is three ``linear_attention`` layers followed
by one ``indexed_attention`` layer.  The script deliberately keeps the QAIC
steps explicit so that the generated commands can be copied to a QAIC host or
audited after a run:

* visible inputs are ``input_ids``, FP32 ``ngram_embeddings``, and
  ``position_ids``;
* retained states are omitted from the qaic-runner batch input;
* profiling uses the QSA reference's 30 iterations, profiling from iteration
  20, two samples, stats level 70, DDR stats, and KernelUtil PMU recipe;
* each opstats trace is mapped back to ONNX node namespace metadata and then
  aggregated by layer and Qwen component.

No model download or Python model import is required.  An ONNX graph and a
compiled QPC are sufficient for post-processing; ``--compile`` and the device
execution flags are optional.  The generated ``report.json`` records commands,
environment, observed runtime output, opstats summaries, and any execution
blocker without treating a failed or missing device run as a performance
measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable

COMPILER = "/opt/qti-aic/exec/qaic-compile"
RUNNER = "/opt/qti-aic/exec/qaic-runner"
OPSTATS = "/opt/qti-aic/exec/qaic-opstats"
COMPILER_OPTS_UNSUPPORTED = (
    "-aic-hoist-vtcm-loads=false -aic-op-stats-verbosity 2 "
    "-aic-userdma-async=0 -aic-hmx-async=0 -aic-sync-ts-starts"
)
LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "indexed_attention")
RETAINED_NAMES = (
    "gdn_conv_state.0",
    "gdn_recurrent_state.0",
    "gdn_conv_state.1",
    "gdn_recurrent_state.1",
    "ple_conv_state.1",
    "gdn_conv_state.2",
    "gdn_recurrent_state.2",
    "qsa_key_state.3",
    "qsa_value_state.3",
    "qsa_index_state.3",
    "qsa_partial_state.3",
)
COMPONENTS = (
    "linear_attn",
    "self_attn",
    "mlp",
    "attn_hyper_connection",
    "mlp_hyper_connection",
    "ple",
    "other",
    "unclassified",
)
LAYER_SCOPE_RE = re.compile(r"(?:^|[/:'\"\s])model\.layers\.(\d+)(?:\.([A-Za-z_][A-Za-z0-9_]*))?")
NODE_TOKEN_RE = re.compile(r"(?:^|[/\s])(node_[A-Za-z0-9_]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=None, help="Existing ONNX graph; omit to create a fresh export.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/qwen38_flash_weightfree/qwen38_flash_4layer_checkpoint_v2"),
        help="Local four-layer weight-free checkpoint used when --onnx is omitted.",
    )
    parser.add_argument(
        "--config-source",
        type=Path,
        default=Path("qwen3.8_flash_experiment/export_256k_v4-4b4dfffa754684cd/hashed_export_params.json"),
        help="Metadata containing the config for the local checkpoint (which has no config.json).",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Output directory for compile/perf assets.")
    parser.add_argument(
        "--opstats-dir",
        type=Path,
        default=None,
        help="Existing qaic-opstats directory; defaults to artifact-dir/perf_dump/opstats or artifact-dir/opstats.",
    )
    parser.add_argument("--qpc-dir", type=Path, default=None, help="Existing QPC directory; defaults to artifact/qpc.")
    parser.add_argument("--compile", action="store_true", help="Run the emitted compile.sh.")
    parser.add_argument("--run-runtime", action="store_true", help="Run unprofiled qaic-runner timing.")
    parser.add_argument(
        "--run-perf", action="store_true", help="Compile a perf QPC, run qaic-runner, and decode opstats."
    )
    parser.add_argument(
        "--overwrite-qpc", action="store_true", help="Allow compile.sh to replace an existing QPC directory."
    )
    parser.add_argument("--trace-glob", default="*-merged.qaic-opstats.trace.json")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Module-level CSV output; defaults to <artifact-dir>/module_timing.csv when analysis runs.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--io-precision", choices=("float16", "mxint8"), default="float16")
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--pooled-ctx-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument(
        "--device-group",
        default=None,
        help="Physical QIDs; omit to let qaic-runner/device picker select the devices.",
    )
    parser.add_argument("--hw-version", choices=("ai100", "ai200"), default="ai100")
    parser.add_argument(
        "--compile-num-cores",
        type=int,
        default=None,
        help="Defaults to 16 for this DP16 QSA configuration; set explicitly for a measured configuration.",
    )
    parser.add_argument(
        "--compile-profile",
        choices=("qeff", "qsa"),
        default="qsa",
        help="qeff matches the existing four-layer QPC; qsa adds -mxfp6-matmul, -user-tiled, and -split-model-io.",
    )
    parser.add_argument("--runtime-iters", type=int, default=50)
    parser.add_argument("--perf-num-iters", type=int, default=30)
    parser.add_argument("--perf-profile-start-iter", type=int, default=20)
    parser.add_argument("--perf-num-samples", type=int, default=2)
    parser.add_argument("--perf-stats-level", type=int, default=70)
    parser.add_argument(
        "--slice-index", type=int, default=None, help="Analyze only one trace slice; default analyzes all slices."
    )
    parser.add_argument(
        "--skip-analysis", action="store_true", help="Emit/run assets without reading ONNX trace metadata."
    )
    return parser.parse_args()


def shell_join(command: Iterable[Any]) -> str:
    return shlex.join(str(part) for part in command)


def parse_device_ids(device_group: str | None, num_devices: int) -> list[int]:
    if device_group is None:
        return []
    ids: list[int] = []
    for part in device_group.strip().strip("[]").split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            start, end = part.split("..", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    if len(ids) != num_devices:
        raise ValueError(f"--device-group has {len(ids)} devices but --num-devices is {num_devices}")
    return ids


def runner_device_args(device_ids: list[int]) -> list[str]:
    if not device_ids:
        return []
    if len(device_ids) == 1:
        return ["-d", str(device_ids[0])]
    if device_ids == list(range(device_ids[0], device_ids[-1] + 1)):
        return ["-D", f"{device_ids[0]}..{device_ids[-1]}"]
    return ["-D", ",".join(str(device_id) for device_id in device_ids)]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body.rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def compile_command(
    args: argparse.Namespace, qpc_dir: Path, specialization: Path, mdp: Path, custom_io: Path
) -> list[str]:
    command = [
        COMPILER,
        "-aic-hw",
        f"-aic-hw-version={args.hw_version}",
        f"-m={args.onnx.resolve()}",
        "-retained-state",
        "-convert-to-fp16",
        f"-aic-num-cores={args.compile_num_cores}",
        f"-mdp-load-partition-config={mdp}",
        f"-network-specialization-config={specialization}",
        f"-custom-IO-list-file={custom_io}",
        f"-aic-binary-dir={qpc_dir}",
    ]
    if args.compile_profile == "qsa":
        command.insert(6, "-mxfp6-matmul")
        command.extend(("-user-tiled", "-split-model-io"))
    return command


def emit_assets(args: argparse.Namespace, device_ids: list[int]) -> dict[str, Any]:
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    qpc_dir = (args.qpc_dir or out / "qpc").resolve()
    perf_dir = out / "perf_dump"
    perf_qpc_dir = perf_dir / "qpc"
    raw_dir = perf_dir / "raw_device_stats"
    opstats_dir = perf_dir / "opstats"
    for directory in (perf_dir, raw_dir, opstats_dir):
        directory.mkdir(parents=True, exist_ok=True)

    specialization = out / "specializations.json"
    write_json(
        specialization,
        {
            "specializations": [
                {
                    "name": "Decode",
                    "symbols": {
                        "batch_size": str(args.batch_size),
                        "ctx_len": str(args.ctx_len),
                        "pooled_ctx_len": str(args.pooled_ctx_len),
                        "seq_len": str(args.seq_len),
                    },
                }
            ]
        },
    )
    mdp = out / f"mdp_ts_{args.num_devices}.json"
    write_json(
        mdp,
        {
            "connections": [{"devices": list(range(args.num_devices)), "type": "p2p"}],
            "partitions": [
                {
                    "name": "Partition0",
                    "devices": [
                        {"deviceId": index, "numCores": args.compile_num_cores}
                        for index in range(args.num_devices)
                    ],
                }
            ],
        },
    )
    custom_io = out / "custom_io.yaml"
    def retained_precision(name: str) -> str:
        if args.io_precision == "mxint8" and name.startswith("qsa_"):
            return "mxint8"
        return "float16"

    custom_io.write_text(
        "".join(
            f" - IOName: {name}\n   Precision: {retained_precision(name)}\n\n"
            f" - IOName: {name}_RetainedState\n   Precision: {retained_precision(name)}\n\n"
            for name in RETAINED_NAMES
        ),
        encoding="utf-8",
    )

    io_dir = out / "io"
    data_dir = io_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    visible_inputs = {
        "input_ids": np.zeros((args.batch_size, args.seq_len), dtype=np.int64),
        # QSA/Qwen handoff contract: application-side ngram embeddings are FP32.
        "ngram_embeddings": np.zeros((args.batch_size, args.seq_len, args.hidden_size), dtype=np.float32),
        "position_ids": np.zeros((4, args.batch_size, args.seq_len), dtype=np.int64),
    }
    visible_outputs = {
        "logits": np.zeros((args.batch_size, args.seq_len, args.vocab_size), dtype=np.float32),
    }
    entries = []
    for direction, arrays in (("in", visible_inputs), ("out", visible_outputs)):
        for name, array in arrays.items():
            filename = name.replace(".", "_") + ".raw"
            np.ascontiguousarray(array).tofile(data_dir / filename)
            entries.append(
                {
                    "path": f"data/{filename}",
                    "io-direction": direction,
                    "elem-size": array.itemsize,
                    "map-to": name,
                    "dims": list(array.shape),
                }
            )
    io_json = io_dir / "aic_batch_io.json"
    write_json(io_json, {"IO-files": [entries]})

    base_command = compile_command(args, qpc_dir, specialization, mdp, custom_io)
    env = f"export QAIC_COMPILER_OPTS_UNSUPPORTED={shlex.quote(COMPILER_OPTS_UNSUPPORTED)}\n"
    clean_guard = ""
    if not args.overwrite_qpc:
        clean_guard = (
            f'if [ -e {shlex.quote(str(qpc_dir))} ]; then '
            'echo "QPC exists; pass --overwrite-qpc" >&2; exit 2; fi\n'
        )
    write_executable(out / "compile.sh", env + clean_guard + shell_join(base_command))

    perf_command = [
        f"-aic-binary-dir={perf_qpc_dir}" if part.startswith("-aic-binary-dir=") else part for part in base_command
    ]
    perf_command.extend(
        (f"-stats-level={args.perf_stats_level}", "-ddr-stats", "-aic-pmu-recipe=KernelUtil", "-aic-perf-metrics")
    )
    write_executable(
        perf_dir / "compile_perf.sh",
        env + f"rm -rf {shlex.quote(str(perf_qpc_dir))}\n" + shell_join(perf_command),
    )
    runner_base = [
        RUNNER,
        *runner_device_args(device_ids),
        "-t",
        qpc_dir,
        "-n",
        args.runtime_iters,
        "--aic-batch-json-input",
        io_json,
    ]
    write_executable(out / "run_runtime.sh", shell_join(runner_base))
    perf_runner = [
        RUNNER,
        *runner_device_args(device_ids),
        "-t",
        perf_qpc_dir,
        "-n",
        args.perf_num_iters,
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        args.perf_profile_start_iter,
        "--aic-profiling-num-samples",
        args.perf_num_samples,
        "--aic-profiling-out-dir",
        raw_dir,
        "--aic-batch-json-input",
        io_json,
    ]
    opstats_command = [
        OPSTATS,
        "--qpc",
        perf_qpc_dir / "programqpc.bin",
        "--input-dir",
        raw_dir,
        "--output-dir",
        opstats_dir,
        "--summary",
        "--trace",
    ]
    write_executable(
        perf_dir / "run_perf.sh",
        f"rm -rf {shlex.quote(str(raw_dir))}/* {shlex.quote(str(opstats_dir))}/*\n" + shell_join(perf_runner),
    )
    write_executable(perf_dir / "decode_perf.sh", shell_join(opstats_command))
    return {
        "qpc_dir": str(qpc_dir),
        "perf_qpc_dir": str(perf_qpc_dir),
        "specializations": str(specialization),
        "mdp_config": str(mdp),
        "custom_io": str(custom_io),
        "io_json": str(io_json),
        "compile_command": base_command,
        "perf_compile_command": perf_command,
        "runtime_command": runner_base,
        "perf_runner_command": perf_runner,
        "opstats_command": opstats_command,
    }


def run_script(path: Path, log_path: Path) -> dict[str, Any]:
    completed = subprocess.run(["bash", str(path)], capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    return {"command": str(path), "returncode": completed.returncode, "output": output}


def runtime_metrics(output: str) -> dict[str, Any]:
    duration = re.search(r"TotalDuration\s+(\d+)us", output)
    throughput = re.search(r"Inf/Sec\s+([0-9.]+)", output)
    return {
        "total_duration_us": int(duration.group(1)) if duration else None,
        "inferences_per_second": float(throughput.group(1)) if throughput else None,
    }


def environment_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "QAIC_COMPILER_OPTS_UNSUPPORTED": os.environ.get("QAIC_COMPILER_OPTS_UNSUPPORTED", COMPILER_OPTS_UNSUPPORTED),
    }
    tool_commands = (
        ("qaic_compile", [COMPILER, "--version"]),
        ("qaic_runner", [RUNNER, "--help"]),
        ("qaic_opstats", [OPSTATS, "--help"]),
    )
    for name, command in tool_commands:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = {"returncode": completed.returncode, "output": (completed.stdout + completed.stderr)[:1000]}
        except (OSError, subprocess.SubprocessError) as error:
            result[name] = {"error": str(error)}
    return result


def load_scope_map(onnx_path: Path) -> tuple[dict[str, tuple[int, str]], dict[int, str]]:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX analysis requires the onnx package in the selected Python environment") from error
    model = onnx.load(str(onnx_path), load_external_data=False)
    scope_map: dict[str, tuple[int, str]] = {}
    components_by_layer: dict[int, set[str]] = defaultdict(set)
    for node in model.graph.node:
        properties = {item.key: item.value for item in node.metadata_props}
        scope = properties.get("namespace") or properties.get("pkg.torch.onnx.name_scopes", "")
        matches = list(LAYER_SCOPE_RE.finditer(scope))
        if not matches:
            continue
        layer_index = int(matches[0].group(1))
        component = next((match.group(2) for match in reversed(matches) if match.group(2)), "other")
        scope_map[node.name] = (layer_index, component)
    if not scope_map:
        raise ValueError(f"No model.layers.<N> namespace metadata found in {onnx_path}")
    for layer_index, component in scope_map.values():
        components_by_layer[layer_index].add(component)
    discovered_types = {
        layer_index: "indexed_attention" if "self_attn" in components else "linear_attention"
        for layer_index, components in components_by_layer.items()
    }
    return scope_map, discovered_types


def node_name_from_op(op_name: str, known_nodes: set[str]) -> str | None:
    for token in NODE_TOKEN_RE.findall(op_name):
        if token in known_nodes:
            return token
    for token in known_nodes:
        if f"/{token}/" in op_name or op_name.endswith("/" + token):
            return token
    return None


def iter_trace_events(trace_path: Path) -> Iterable[dict[str, Any]]:
    """Yield events from a line-oriented QAIC Chrome trace without loading it all."""
    with trace_path.open(encoding="utf-8") as trace_file:
        for line in trace_file:
            stripped = line.strip().rstrip(",")
            if not stripped or stripped in {"{", "}", "[", "]"}:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and "traceEvents" not in event:
                yield event


def trace_display_time_unit(trace_path: Path) -> str:
    """Read the Chrome trace display unit without retaining the trace payload."""
    display_unit_re = re.compile(r'"displayTimeUnit"\s*:\s*"([^"]+)"')
    with trace_path.open(encoding="utf-8") as trace_file:
        for line in trace_file:
            match = display_unit_re.search(line)
            if match is not None:
                return match.group(1)
    return "us"


def component_group(component: str) -> str:
    return component if component in COMPONENTS else "other"


def event_duration_us(event: dict[str, Any], time_unit: str) -> float:
    args = event.get("args", {})
    if str(event.get("name", "")).startswith("sync ") and "opSyncDurUs" in args:
        return float(args["opSyncDurUs"])
    duration = float(event.get("dur", 0.0))
    scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3}.get(time_unit)
    if scale is None:
        raise ValueError(f"Unsupported trace display time unit: {time_unit}")
    return duration * scale


def union_length(spans: list[tuple[float, float]]) -> float:
    if not spans:
        return 0.0
    total = 0.0
    start, end = sorted(spans)[0]
    for next_start, next_end in sorted(spans)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def add_interval(accumulator: dict[str, Any], span: tuple[float, float]) -> None:
    """Merge mostly time-ordered trace intervals without retaining every event."""
    intervals = accumulator["spans"]
    if not intervals:
        intervals.append(span)
        return
    start, end = span
    last_start, last_end = intervals[-1]
    if start < last_start:
        accumulator["late_spans"].append(span)
    elif start <= last_end:
        intervals[-1] = (last_start, max(last_end, end))
    else:
        intervals.append(span)


def accumulator_union_length(accumulator: dict[str, Any]) -> float:
    late_spans = accumulator["late_spans"]
    if not late_spans:
        return union_length(accumulator["spans"])
    return union_length(accumulator["spans"] + late_spans)


def analyze_trace(
    trace_path: Path,
    scope_map: dict[str, tuple[int, str]],
    selected_layers: set[int],
    *,
    time_unit_override: str | None = None,
) -> dict[str, Any]:
    time_unit = time_unit_override or trace_display_time_unit(trace_path)
    time_scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3}.get(time_unit)
    if time_scale is None:
        raise ValueError(f"Unsupported trace display time unit: {time_unit}")
    known_nodes = set(scope_map)
    accumulators: dict[int, dict[str, Any]] = {
        layer: {
            "spans": [],
            "late_spans": [],
            "busy": 0.0,
            "groups": {
                group: {"spans": [], "late_spans": [], "busy": 0.0} for group in COMPONENTS
            },
        }
        for layer in selected_layers
    }
    all_events = {"spans": [], "late_spans": []}
    unresolved = 0
    mapped = 0
    for event in iter_trace_events(trace_path):
        if event.get("ph") != "X":
            continue
        timestamp = float(event.get("ts", 0.0))
        duration = event_duration_us(event, time_unit)
        timestamp_us = timestamp * time_scale
        span = (timestamp_us, timestamp_us + duration)
        add_interval(all_events, span)
        node_name = node_name_from_op(str(event.get("args", {}).get("opName", "")), known_nodes)
        if node_name is None:
            unresolved += 1
            continue
        layer, component = scope_map[node_name]
        if layer not in selected_layers:
            continue
        group = component_group(component)
        accumulator = accumulators[layer]
        add_interval(accumulator, span)
        accumulator["busy"] += duration
        group_accumulator = accumulator["groups"][group]
        add_interval(group_accumulator, span)
        group_accumulator["busy"] += duration
        mapped += 1
    rows = []
    for layer in sorted(accumulators):
        accumulator = accumulators[layer]
        layer_wallclock = accumulator_union_length(accumulator)
        all_layer_spans = [
            span
            for item in accumulators.values()
            for span in item["spans"] + item["late_spans"]
        ]
        row: dict[str, Any] = {
            "layer_index": layer,
            "layer_type": LAYER_TYPES[layer] if layer < len(LAYER_TYPES) else "unknown",
            "layer_wallclock_us": layer_wallclock,
            "layer_busy_us": accumulator["busy"],
            "pct_of_mapped_layers": (
                100.0
                * layer_wallclock
                / union_length(all_layer_spans)
                if layer_wallclock
                else 0.0
            ),
        }
        for group, values in accumulator["groups"].items():
            wallclock = accumulator_union_length(values)
            row[f"{group}_wallclock_us"] = wallclock
            row[f"{group}_busy_us"] = values["busy"]
            row[f"{group}_pct_of_layer"] = 100.0 * wallclock / layer_wallclock if layer_wallclock else 0.0
        rows.append(row)
    return {
        "trace_path": str(trace_path),
        "display_time_unit": time_unit,
        "all_event_wallclock_us": accumulator_union_length(all_events),
        "mapped_event_wallclock_us": union_length(
            [
                span
                for item in accumulators.values()
                for span in item["spans"] + item["late_spans"]
            ]
        ),
        "mapped_event_count": mapped,
        "unresolved_event_count": unresolved,
        "rows": rows,
    }


def aggregate_trace_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for row in result["rows"]:
            grouped[row["layer_index"]].append(row)
    rows = []
    for layer in sorted(grouped):
        values = grouped[layer]
        numeric_keys = [
            key
            for key, value in values[0].items()
            if isinstance(value, (int, float)) and key != "layer_index"
        ]
        row: dict[str, Any] = {"layer_index": layer, "layer_type": values[0]["layer_type"], "samples": len(values)}
        for key in numeric_keys:
            series = [float(value[key]) for value in values]
            row[f"{key}_mean"] = mean(series)
            row[f"{key}_median"] = median(series)
            row[f"{key}_stdev"] = stdev(series) if len(series) > 1 else 0.0
            row[f"{key}_min"] = min(series)
            row[f"{key}_max"] = max(series)
        rows.append(row)
    return rows


def write_module_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a compact, spreadsheet-friendly layer/module timing report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write module CSV without trace results")
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_opstats_dir(artifact_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        opstats_dir = explicit.resolve()
        if not opstats_dir.is_dir():
            raise FileNotFoundError(f"Opstats directory does not exist: {opstats_dir}")
        return opstats_dir
    candidates = (artifact_dir / "perf_dump" / "opstats", artifact_dir / "opstats")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No opstats directory found under {artifact_dir}; pass --opstats-dir explicitly."
    )


def parse_opstats_summaries(opstats_dir: Path) -> dict[str, Any]:
    summaries = []
    for path in sorted(opstats_dir.glob("*.summary.txt")):
        text = path.read_text(encoding="utf-8")
        pcycles = [int(value) for value in re.findall(r"Total inference pcycle is (\d+)", text)]
        ddr = [float(value) for value in re.findall(r"DDR traffic count ([\d.]+) MB", text)]
        summaries.append({"path": str(path), "max_core_pcycles": max(pcycles, default=0), "ddr_mb": sum(ddr)})
    return {
        "files": summaries,
        "max_core_pcycles": max((item["max_core_pcycles"] for item in summaries), default=0),
        "total_ddr_mb": sum(item["ddr_mb"] for item in summaries),
    }


def export_and_prepare_compile(args: argparse.Namespace) -> tuple[Path, Path]:
    """Create a fresh Qwen4-Exp export and QEff compiler replay bundle."""
    import torch
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

    checkpoint = args.checkpoint.resolve()
    config_source = args.config_source.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Local checkpoint not found: {checkpoint}")
    metadata = json.loads(config_source.read_text(encoding="utf-8"))
    config_values = metadata.get("config")
    if not config_values:
        raise ValueError(f"Expected a 'config' object in {config_source}")
    config_values = dict(config_values)
    config_values["num_hidden_layers"] = 4
    config_values["layer_types"] = list(config_values["layer_types"][:4])
    config_values["ple_layer_ids"] = [layer for layer in config_values.get("ple_layer_ids", []) if layer < 4]
    config_values["max_position_embeddings"] = args.ctx_len
    config_values["dtype"] = "float32"
    config = Qwen4ExpTextConfig(**config_values)

    qsa_config = {
        "indexer_dp": 16,
        "indexer_cp": 1,
        "attn_dp": 16,
        "attn_cp": 1,
        "num_cores_per_device": args.compile_num_cores,
        "ctx_len": args.ctx_len,
        "ffn_blocking_mode": "token",
        "ffn_token_block_size": 4,
    }
    model = QEFFAutoModelForCausalLM.from_pretrained(
        str(checkpoint), config=config, weight_free=True, torch_dtype=torch.float32
    )
    model.model.eval()
    compile_dir = args.artifact_dir / "qeff_compile"
    compile_artifacts = model.compile(
        compile_dir=str(compile_dir),
        prefill_seq_len=1,
        prefill_only=False,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        num_devices=args.num_devices,
        num_cores=args.compile_num_cores,
        mxfp6_matmul=False,
        user_tiled=True,
        split_model_io=True,
        qaic_config=qsa_config,
        dynamo=True,
        use_onnx_subfunctions=False,
        offload_pt_weights=False,
        artifacts=True,
    )
    onnx_path = Path(model.onnx_path).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Fresh export did not produce an ONNX file: {onnx_path}")
    return onnx_path, Path(compile_artifacts).resolve()


def main() -> int:
    args = parse_args()
    if args.compile_num_cores is None:
        args.compile_num_cores = 16
    args.artifact_dir = args.artifact_dir.resolve()
    qeff_compile_artifacts = None
    if args.onnx is None:
        args.onnx, qeff_compile_artifacts = export_and_prepare_compile(args)
    else:
        args.onnx = args.onnx.resolve()
    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    device_ids = parse_device_ids(args.device_group, args.num_devices)
    run_device_workflow = args.compile or args.run_runtime or args.run_perf or qeff_compile_artifacts is not None
    assets = emit_assets(args, device_ids) if run_device_workflow else {}
    report: dict[str, Any] = {
        "model": {
            "onnx": str(args.onnx.resolve()),
            "layer_types": list(LAYER_TYPES),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "ctx_len": args.ctx_len,
            "pooled_ctx_len": args.pooled_ctx_len,
            "num_hidden_layers": 4,
        },
        "hardware": {
            "hw_version": args.hw_version,
            "device_ids": device_ids or "auto-selected",
            "compile_num_cores_per_device": args.compile_num_cores,
        },
        "profiling_policy": {
            "mode": "decode-only",
            "prefill_seq_len": 1,
            "runtime_iters": args.runtime_iters,
            "perf_num_iters": args.perf_num_iters,
            "perf_profile_start_iter": args.perf_profile_start_iter,
            "perf_num_samples": args.perf_num_samples,
            "perf_stats_level": args.perf_stats_level,
            "timing_statistic": (
                "qaic-runner TotalDuration/iterations when available; opstats layer "
                "wallclock mean/median/stdev across trace files"
            ),
        },
        "environment": environment_snapshot(),
        "assets": assets,
        "execution": {},
    }
    if qeff_compile_artifacts is not None:
        report["assets"]["qeff_compile_artifacts"] = str(qeff_compile_artifacts)
    report_path = args.artifact_dir / "report.json"
    if args.compile:
        result = run_script(args.artifact_dir / "compile.sh", args.artifact_dir / "compile.log")
        report["execution"]["compile"] = result
        if result["returncode"] != 0:
            report["blocker"] = "qaic-compile failed; no runtime or opstats measurement was claimed."
    if args.run_runtime:
        result = run_script(args.artifact_dir / "run_runtime.sh", args.artifact_dir / "runtime.log")
        result["metrics"] = runtime_metrics(result["output"])
        report["execution"]["runtime"] = result
        if result["returncode"] != 0:
            report["blocker"] = "qaic-runner runtime timing failed; no runtime measurement was claimed."
    if args.run_perf:
        perf_steps = (
            ("compile_perf.sh", "compile_perf.log"),
            ("run_perf.sh", "run_perf.log"),
            ("decode_perf.sh", "decode_perf.log"),
        )
        for name, log_name in perf_steps:
            result = run_script(args.artifact_dir / "perf_dump" / name, args.artifact_dir / "perf_dump" / log_name)
            report["execution"][name] = result
            if result["returncode"] != 0:
                report["blocker"] = f"{name} failed; opstats/runtime data is incomplete."
                break
    if not args.skip_analysis:
        scope_map, discovered_types = load_scope_map(args.onnx)
        opstats_dir = resolve_opstats_dir(args.artifact_dir, args.opstats_dir)
        trace_paths = sorted(opstats_dir.glob(args.trace_glob))
        if args.slice_index is not None:
            trace_paths = [path for path in trace_paths if f"slice{args.slice_index:02d}" in path.name]
        trace_results = [analyze_trace(path, scope_map, set(args.layers)) for path in trace_paths]
        aggregate_rows = aggregate_trace_results(trace_results)
        output_csv = args.output_csv or args.artifact_dir / "module_timing.csv"
        report["onnx_analysis"] = {
            "mapped_nodes": len(scope_map),
            "discovered_layer_types": discovered_types,
            "opstats_dir": str(opstats_dir),
            "trace_count": len(trace_results),
            "traces": trace_results,
            "aggregate_layer_rows": aggregate_rows,
            "module_csv": str(output_csv),
            "opstats_summary": parse_opstats_summaries(opstats_dir),
        }
        if aggregate_rows:
            write_module_csv(aggregate_rows, output_csv)
        if not trace_paths and not report.get("blocker"):
            report["blocker"] = (
                "No opstats trace JSON was found; provide a compiled/perf-run QPC or "
                "run --run-perf on QAIC hardware."
            )
    write_json(report_path, report)
    print(report_path)
    if report.get("blocker"):
        print(f"BLOCKER: {report['blocker']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
