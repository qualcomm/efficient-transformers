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


def record_timing_milestone(state: RunState, milestone: str, timestamp: float) -> None:
    """Store a milestone, keeping the original run start if it is repeated."""
    if milestone == "load_start":
        state.milestones.setdefault(milestone, timestamp)
    else:
        state.milestones[milestone] = timestamp


def _timestamp(times: Dict[str, float], milestones: tuple[str, ...], fallback: float) -> float:
    """Return the first available event in stage order."""
    return next((times[name] for name in milestones if name in times), fallback)


def build_timing_table(state: Optional[RunState], milestones: Dict[str, str]) -> Optional[str]:
    if state is None:
        return None

    times = state.milestones
    now = time.time()
    load_start = times.get(milestones["load_start"], min(times.values(), default=now))
    export_start = times.get(milestones["export_start"])
    compile_start = times.get(milestones["compile_start"])
    generation_start = times.get(milestones["generation_start"])

    # A stage ends when the next one starts; its completion event is a fallback
    # if the run stops before reaching the next stage.
    load_end = _timestamp(
        times,
        (
            milestones["export_start"],
            milestones["compile_start"],
            milestones["generation_start"],
            milestones["load_complete"],
        ),
        now,
    )
    loading = max(0.0, load_end - load_start)

    export_end = _timestamp(
        times,
        (milestones["compile_start"], milestones["generation_start"], milestones["export_complete"], "export_failed"),
        now,
    )
    exporting = (
        max(0.0, export_end - export_start)
        if export_start is not None and milestones["export_skipped"] not in times
        else 0.0
    )

    compile_complete = _timestamp(
        times,
        (milestones["compile_complete"], "compile_failed", milestones["compile_skipped"]),
        generation_start if generation_start is not None else now,
    )
    compile_end = max(generation_start, compile_complete) if generation_start is not None else compile_complete
    compiling = (
        max(0.0, compile_end - compile_start)
        if compile_start is not None and milestones["compile_skipped"] not in times
        else 0.0
    )

    mdp_start = times.get(milestones["mdp_generation_start"])
    mdp_end = times.get(milestones["mdp_generation_complete"], now) if mdp_start is not None else None
    mdp_generation = max(0.0, mdp_end - mdp_start) if mdp_start is not None else 0.0

    # MDP config generation may fall before the compiler invocation or inside
    # it. Keep it as its own row and remove it from the containing phase.
    if mdp_start is not None and mdp_end is not None:
        if export_start is not None and export_start <= mdp_start <= mdp_end <= export_end:
            exporting = max(0.0, exporting - mdp_generation)
        elif compile_start is not None and compile_start <= mdp_start <= mdp_end <= compile_end:
            compiling = max(0.0, compiling - mdp_generation)

    if generation_start is not None:
        generation_start = max(generation_start, compile_complete)
        generation = max(0.0, times.get(milestones["generation_complete"], now) - generation_start)
    elif milestones["generation_complete"] in times:
        compile_done = _timestamp(
            times,
            (milestones["compile_complete"], "compile_failed", milestones["compile_skipped"]),
            load_end,
        )
        generation = max(0.0, times[milestones["generation_complete"]] - compile_done)
    else:
        generation = 0.0

    rows = [["Model Loading", loading], ["Model Exporting", exporting]]
    if mdp_start is not None or milestones["mdp_generation_complete"] in times:
        rows.append(["MDP Generation", mdp_generation])
    rows.extend(
        [
            ["Model Compilation", compiling],
            ["Text Generation", generation],
            ["Total Time", loading + exporting + mdp_generation + compiling + generation],
        ]
    )
    return tabulate(rows, headers=["Step", "Time (s)"], tablefmt="github", floatfmt=".3f")
