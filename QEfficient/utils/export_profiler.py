# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Opt-in export profiling: stage timers, graph sizes, and loop-unroll budgets.

Enable with the environment variable ``QEFF_EXPORT_PROFILE``:

* ``QEFF_EXPORT_PROFILE=1``         stage timings, peak RSS, FX/ONNX node counts,
                                    torch.export strategy attempts, dynamo phase times,
                                    per-layer loop-unroll budget.
* ``QEFF_EXPORT_PROFILE=cprofile``  everything above plus a cProfile of the
                                    ``torch.onnx.export`` call (``export_cprofile.prof``
                                    and ``export_cprofile_top.txt`` next to the ONNX).

Every helper here is best effort: if a private PyTorch API moved, profiling degrades
silently instead of failing the export. Nothing here runs inside the traced region.
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import resource
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ENV_VAR = "QEFF_EXPORT_PROFILE"
_PREFIX = "[export-profile]"


def profile_mode() -> str:
    return os.environ.get(ENV_VAR, "").strip().lower()


def enabled() -> bool:
    return profile_mode() not in ("", "0", "false", "off", "no")


def cprofile_enabled() -> bool:
    return profile_mode() == "cprofile"


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _emit(message: str) -> None:
    print(f"{_PREFIX} {message}", flush=True)


class ExportProfiler:
    """Process-wide collector. Flat records with a depth so nesting stays readable."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.notes: dict[str, Any] = {}
        self._depth = 0

    @contextmanager
    def stage(self, name: str, **extra: Any) -> Iterator[dict[str, Any]]:
        """Time a block. Yields a dict the caller may add fields to."""
        if not enabled():
            yield {}
            return
        record: dict[str, Any] = {"stage": name, "depth": self._depth, **extra}
        self.records.append(record)
        self._depth += 1
        _emit(f"{'  ' * record['depth']}> {name}")
        start = time.perf_counter()
        ok = False
        try:
            yield record
            ok = True
        finally:
            self._depth -= 1
            record["seconds"] = round(time.perf_counter() - start, 3)
            record["peak_rss_mb"] = round(_peak_rss_mb(), 1)
            record["ok"] = ok
            fields = {k: v for k, v in record.items() if k not in ("stage", "depth", "seconds", "ok", "peak_rss_mb")}
            suffix = f" {fields}" if fields else ""
            status = "" if ok else " FAILED"
            _emit(
                f"{'  ' * record['depth']}< {name}: {record['seconds']:.2f}s"
                f" (peak RSS {record['peak_rss_mb']:.0f} MB){status}{suffix}"
            )

    def note(self, key: str, value: Any) -> None:
        if not enabled():
            return
        self.notes[key] = value
        _emit(f"{key}: {value}")

    def dump(self, path: Path) -> Path | None:
        if not enabled():
            return None
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as fp:
                json.dump({"stages": self.records, "notes": self.notes}, fp, indent=2, default=str)
            _emit(f"profile written to {path}")
            return path
        except OSError as exc:  # pragma: no cover - diagnostics only
            _emit(f"could not write profile to {path}: {exc}")
            return None


PROFILER = ExportProfiler()


# --------------------------------------------------------------------------- graph statistics


def fx_graph_stats(graph_module: Any, top_k: int = 15) -> dict[str, Any]:
    """Count FX nodes in a GraphModule and every nested GraphModule (while_loop bodies,
    invoke_subgraph / repeated subgraphs). Shared submodules are counted once."""
    import torch

    seen: set = set()
    per_graph: dict[str, int] = {}
    ops: Counter = Counter()

    def visit(module: Any, name: str) -> None:
        if id(module) in seen:
            return
        seen.add(id(module))
        graph = getattr(module, "graph", None)
        if graph is not None:
            count = 0
            for node in graph.nodes:
                count += 1
                if node.op == "call_function":
                    ops[getattr(node.target, "__name__", str(node.target))] += 1
            per_graph[name] = count
        for child_name, child in module.named_children():
            if isinstance(child, torch.fx.GraphModule):
                visit(child, f"{name}.{child_name}")

    visit(graph_module, "root")
    largest = sorted(per_graph.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "total_nodes": sum(per_graph.values()),
        "num_graphs": len(per_graph),
        "largest_graphs": largest,
        "top_ops": ops.most_common(top_k),
    }


def onnx_graph_stats(model: Any, top_k: int = 15) -> dict[str, Any]:
    """Count ONNX nodes in the main graph, every function, and nested Loop/If bodies."""
    ops: Counter = Counter()

    def count_graph(graph: Any) -> int:
        total = 0
        for node in graph.node:
            total += 1
            ops[node.op_type] += 1
            for attr in node.attribute:
                if attr.HasField("g"):
                    total += count_graph(attr.g)
                for sub in attr.graphs:
                    total += count_graph(sub)
        return total

    main_nodes = count_graph(model.graph)
    function_nodes: dict[str, int] = {}
    for function in model.functions:
        total = 0
        for node in function.node:
            total += 1
            ops[node.op_type] += 1
            for attr in node.attribute:
                if attr.HasField("g"):
                    total += count_graph(attr.g)
        function_nodes[function.name] = total
    return {
        "main_graph_nodes": main_nodes,
        "num_functions": len(function_nodes),
        "function_nodes": sorted(function_nodes.items(), key=lambda kv: kv[1], reverse=True)[:10],
        "total_nodes_incl_functions": main_nodes + sum(function_nodes.values()),
        "loop_nodes": ops.get("Loop", 0),
        "top_ops": ops.most_common(top_k),
    }


# --------------------------------------------------------------------------- loop budget


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, None)
    return default if value is None else value


def loop_unroll_budget(model: Any) -> dict[str, Any]:
    """Report how many times each Python loop in the prefill path unrolls per layer.

    Computed from the blocking config outside the trace, mirroring the formulas in
    ``modeling_minimax_m3_vl.py`` / ``moe/flavours.py``. Iterations here are a direct
    proxy for traced FX nodes, which is what export time scales with.
    """
    layers: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        cfg = getattr(module, "attn_blocking_config", None)
        if cfg is not None and getattr(cfg, "msa_q_chunk", None):
            export_seq = _attr(cfg, "prefill_export_seq_len") or _attr(cfg, "prefill_compile_seq_len")
            entry: dict[str, Any] = {"module": name, "export_seq_len": export_seq}
            if export_seq:
                msa_q_chunk = int(cfg.msa_q_chunk)
                entry["msa_q_chunks"] = int(export_seq) // msa_q_chunk if msa_q_chunk else None
                entry["msa_kv_blocks_in_body"] = _attr(cfg, "msa_num_kv_blocks", _attr(cfg, "num_kv_blocks", 1))
                q_size = _attr(cfg, "indexer_q_size")
                q_chunk = _attr(cfg, "indexer_q_chunk", q_size)
                idx_kv = int(_attr(cfg, "indexer_num_blocks", _attr(cfg, "num_kv_blocks", 1)))
                if q_size and q_chunk:
                    q_chunks = int(export_seq) // int(q_chunk)
                    q_blocks = int(q_chunk) // int(q_size)
                    entry["indexer_unrolled_iterations"] = q_chunks * idx_kv * q_blocks
                    entry["indexer_shape"] = f"{q_chunks} q-chunks x {idx_kv} kv-blocks x {q_blocks} q-blocks"
            layers.append(entry)
        if hasattr(module, "expert_parallel_num_packed_chunks") or hasattr(module, "num_pipeline_stages"):
            stages = int(_attr(module, "num_pipeline_stages", 1))
            chunks = int(_attr(module, "expert_parallel_num_packed_chunks", 1))
            layers.append(
                {
                    "module": name,
                    "moe_unrolled_iterations": stages * chunks,
                    "moe_shape": f"{stages} pipeline stages x {chunks} packed chunks",
                }
            )
    return {"entries": layers[:12], "num_entries": len(layers)}


# --------------------------------------------------------------------------- torch internals


@contextmanager
def torch_export_internals_timed() -> Iterator[None]:
    """Time torch.onnx's internal phases without restructuring the export call.

    Wraps (when present): each torch.export capture-strategy attempt, ExportedProgram
    decomposition, and the FX -> ONNX IR translation. A failed capture strategy that
    falls back to the next one re-traces the whole model, so it shows up here as two
    long "capture" stages.
    """
    if not enabled():
        yield
        return

    restore: list[tuple] = []

    def wrap(owner: Any, attr: str, stage_name: str, on_result=None) -> None:
        original = getattr(owner, attr, None)
        if original is None:
            return

        def wrapped(*args, **kwargs):
            label = stage_name
            if args and stage_name == "capture" and hasattr(args[0], "__class__"):
                label = f"capture[{type(args[0]).__name__}]"
            with PROFILER.stage(label) as record:
                result = original(*args, **kwargs)
                if on_result is not None:
                    try:
                        on_result(result, record)
                    except Exception as exc:  # pragma: no cover - diagnostics only
                        record["stats_error"] = repr(exc)
                return result

        setattr(owner, attr, wrapped)
        restore.append((owner, attr, original))

    def capture_stats(result: Any, record: dict[str, Any]) -> None:
        program = getattr(result, "exported_program", None)
        exc = getattr(result, "exception", None)
        record["success"] = program is not None
        if exc is not None:
            record["error"] = repr(exc)[:300]
        if program is not None:
            record["fx"] = fx_graph_stats(program.graph_module)

    def decomp_stats(result: Any, record: dict[str, Any]) -> None:
        record["fx_after_decomp"] = fx_graph_stats(result.graph_module)["total_nodes"]

    try:
        from torch.onnx._internal.exporter import _capture_strategies

        wrap(_capture_strategies.CaptureStrategy, "__call__", "capture", capture_stats)
    except Exception:
        pass
    try:
        import torch

        wrap(torch.export.ExportedProgram, "run_decompositions", "run_decompositions", decomp_stats)
    except Exception:
        pass
    try:
        from torch.onnx._internal.exporter import _core

        wrap(_core, "exported_program_to_ir", "fx_to_onnx_ir")
    except Exception:
        pass

    try:
        yield
    finally:
        for owner, attr, original in reversed(restore):
            setattr(owner, attr, original)


def dynamo_phase_report() -> None:
    """Log dynamo's own per-phase timers and counters (graph breaks, recompiles)."""
    if not enabled():
        return
    try:
        from torch._dynamo.utils import compile_times, counters

        times = compile_times(repr="str", aggregate=True)
        if times:
            _emit("dynamo phase times:\n" + str(times))
        interesting = {
            key: dict(counters[key]) for key in ("graph_break", "stats", "unimplemented") if counters.get(key)
        }
        if interesting:
            PROFILER.note("dynamo_counters", interesting)
    except Exception:
        pass


@contextmanager
def maybe_cprofile(output_dir: Path, top_n: int = 60) -> Iterator[None]:
    if not cprofile_enabled():
        yield
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        output_dir = Path(output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            prof_path = output_dir / "export_cprofile.prof"
            profiler.dump_stats(str(prof_path))
            buffer = io.StringIO()
            stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
            stats.print_stats(top_n)
            buffer.write("\n\n==== sorted by tottime ====\n")
            stats.sort_stats("tottime").print_stats(top_n)
            top_path = output_dir / "export_cprofile_top.txt"
            top_path.write_text(buffer.getvalue())
            _emit(f"cProfile written to {prof_path} (summary: {top_path})")
        except OSError as exc:  # pragma: no cover - diagnostics only
            _emit(f"could not write cProfile output: {exc}")
