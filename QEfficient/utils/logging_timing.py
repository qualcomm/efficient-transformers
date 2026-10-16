# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from tabulate import tabulate


@dataclass
class RunState:
    run_id: int
    model: str
    milestones: Dict[str, float] = field(default_factory=dict)
    table_written: bool = False


def build_timing_table(state: Optional[RunState], milestones: Dict[str, str]) -> Optional[str]:
    if state is None:
        return None

    times = state.milestones
    now = time.time()
    t_start = times.get(milestones["load_start"], min(times.values(), default=now))
    t_load_done = max(times.get(milestones["load_complete"], t_start), t_start)
    t_export_done = max(
        times.get(milestones["export_complete"], times.get(milestones["export_skipped"], t_load_done)),
        t_load_done,
    )
    t_compile_done = max(
        times.get(milestones["compile_complete"], times.get("compile_failed", t_export_done)),
        t_export_done,
    )

    loading = max(0.0, t_load_done - t_start)
    exporting = 0.0 if milestones["export_skipped"] in times else max(0.0, t_export_done - t_load_done)
    compiling = 0.0 if milestones["compile_skipped"] in times else max(0.0, t_compile_done - t_export_done)

    generation_complete = milestones["generation_complete"]
    if times.get("generate_start") and times.get(generation_complete):
        generation = max(0.0, times[generation_complete] - max(times["generate_start"], t_compile_done))
    elif times.get(generation_complete):
        generation = max(0.0, times[generation_complete] - t_compile_done)
    else:
        generation = 0.0

    total = loading + exporting + compiling + generation
    return tabulate(
        [
            ["Model Loading", loading],
            ["Model Exporting", exporting],
            ["Model Compilation", compiling],
            ["Text Generation", generation],
            ["Total Time", total],
        ],
        headers=["Step", "Time (s)"],
        tablefmt="github",
        floatfmt=".3f",
    )
