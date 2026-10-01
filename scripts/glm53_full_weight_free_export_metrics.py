#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Measure full GLM-5.3 weight-free export without compilation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

MODEL_ID = "zai-org/GLM-5.3"
DEFAULT_HF_CACHE = "/home/huggingface_hub"
DEFAULT_CHECKPOINT_HOME = "/home/huggingface_hub/qeff_glm53_full_metrics"
DEFAULT_OUTPUT_DIR = "/home/ochougul/efficient-transformers/artifacts/glm53_full_weight_free_metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--checkpoint-home", default=DEFAULT_CHECKPOINT_HOME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    return parser.parse_args()


def current_rss_bytes() -> int:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS was not found in /proc/self/status")


class PeakRssMonitor:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self.baseline_bytes = current_rss_bytes()
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, current_rss_bytes())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.peak_bytes = max(self.peak_bytes, current_rss_bytes())
        self._stop.set()
        self._thread.join()


def measure_phase(name: str, operation: Callable[[], Any], sample_interval: float) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    with PeakRssMonitor(sample_interval) as monitor:
        result = operation()
    metrics = {
        "name": name,
        "duration_seconds": time.perf_counter() - start,
        "baseline_rss_bytes": monitor.baseline_bytes,
        "peak_rss_bytes": monitor.peak_bytes,
        "peak_rss_delta_bytes": monitor.peak_bytes - monitor.baseline_bytes,
    }
    print(json.dumps({"event": "phase_complete", **metrics}), flush=True)
    return result, metrics


def directory_size_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def main() -> None:
    args = parse_args()
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive.")

    os.environ["HF_HUB_CACHE"] = args.hf_cache
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["QEFF_WF_HOME"] = args.checkpoint_home
    os.environ["QEFF_HOME"] = args.output_dir

    from QEfficient import QEFFAutoModelForCausalLM

    logging.getLogger("QEfficient").setLevel(logging.INFO)

    dtype = getattr(torch, args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_id, cache_dir=args.hf_cache)
    config.use_cache = True
    config.torch_dtype = dtype
    config.dtype = dtype

    qeff_model, initialization_metrics = measure_phase(
        "model_initialization",
        lambda: QEFFAutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            cache_dir=args.hf_cache,
            torch_dtype=dtype,
            weight_free=True,
            qaic_config={"mla_absorption": {"cache_compressed": True}},
        ),
        args.sample_interval,
    )
    qeff_model.model.eval()

    onnx_path, export_metrics = measure_phase(
        "qeff_export",
        lambda: qeff_model.export(
            export_dir=str(output_dir / "export"),
            prefill_only=False,
            offload_pt_weights=False,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
        ),
        args.sample_interval,
    )
    onnx_path = Path(onnx_path)
    weight_spec_path = onnx_path.with_name("weight_spec.json")
    weight_free_metrics = qeff_model._weight_free_export_metrics
    prepared_dirs = sorted(Path(args.checkpoint_home).glob("*-qeff-prepared-*"))
    if len(prepared_dirs) != 1:
        raise RuntimeError(f"Expected one prepared checkpoint in {args.checkpoint_home}, found {prepared_dirs}")
    prepared_path = prepared_dirs[0]

    metrics = {
        "model_id": args.model_id,
        "num_hidden_layers": config.num_hidden_layers,
        "dtype": str(dtype),
        "initialization": initialization_metrics,
        "torch_onnx_export_seconds": weight_free_metrics["torch_onnx_export_seconds"],
        "one_time_weight_preparation_seconds": weight_free_metrics["checkpoint_preparation_seconds"],
        "qeff_export": export_metrics,
        "overall_peak_rss_bytes": max(
            initialization_metrics["peak_rss_bytes"],
            export_metrics["peak_rss_bytes"],
        ),
        "prepared_checkpoint_path": str(prepared_path),
        "prepared_checkpoint_size_bytes": directory_size_bytes(prepared_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "weight_spec_path": str(weight_spec_path),
        "weight_spec_size_bytes": weight_spec_path.stat().st_size,
        "compile_triggered": False,
    }
    metrics_path = output_dir / "full_export_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"event": "full_export_metrics", "metrics_path": str(metrics_path), **metrics}), flush=True)


if __name__ == "__main__":
    main()
