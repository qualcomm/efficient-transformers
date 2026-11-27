# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from tabulate import tabulate


@dataclass
class RunState:
    run_id: int
    model: str
    milestones: Dict[str, float] = field(default_factory=dict)
    table_written: bool = False
    milestone_events: list[tuple[str, float, Optional[str]]] = field(default_factory=list)


def record_timing_milestone(state: RunState, milestone: str, timestamp: float, namespace: Optional[str] = None) -> None:
    """Store each event and update the legacy milestone snapshot."""
    state.milestone_events.append((milestone, timestamp, namespace))
    if milestone == "load_start":
        state.milestones.setdefault(milestone, timestamp)
    else:
        state.milestones[milestone] = timestamp


def _timestamp(times: Dict[str, float], milestones: tuple[str, ...], fallback: float) -> float:
    """Return the first available event in stage order."""
    return next((times[name] for name in milestones if name in times), fallback)


def _event_timestamp(
    events: list[tuple[str, float, Optional[str]]], milestones: tuple[str, ...], fallback: float
) -> float:
    """Return the earliest occurrence of the first available milestone."""
    for milestone in milestones:
        occurrences = [timestamp for event, timestamp, _ in events if event == milestone]
        if occurrences:
            return min(occurrences)
    return fallback


def _stage_attempt_intervals(
    events: list[tuple[str, float, Optional[str]]],
    start_milestone: str,
    completion_milestones: tuple[str, ...],
    skipped_milestone: str,
    fallback_milestones: tuple[str, ...],
    fallback: float,
) -> tuple[list[tuple[float, float]], int, bool]:
    """Pair repeated stage starts with their outcomes; skipped attempts have no duration."""
    starts_by_namespace = {}
    intervals = []
    start_count = 0
    saw_skip = False
    terminal_milestones = set(completion_milestones) | {skipped_milestone}

    for milestone, timestamp, namespace in sorted(events, key=lambda event: event[1]):
        if milestone == start_milestone:
            starts_by_namespace.setdefault(namespace, deque()).append(timestamp)
            start_count += 1
        elif milestone in terminal_milestones:
            saw_skip |= milestone == skipped_milestone
            starts = starts_by_namespace.get(namespace)
            if starts:
                start = starts.popleft()
                if milestone != skipped_milestone and timestamp >= start:
                    intervals.append((start, timestamp))

    for starts in starts_by_namespace.values():
        for start in starts:
            later_events = [
                (milestone, timestamp, namespace) for milestone, timestamp, namespace in events if timestamp >= start
            ]
            end = _event_timestamp(later_events, fallback_milestones, fallback)
            if end >= start:
                intervals.append((start, end))

    return intervals, start_count, saw_skip


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    """Return elapsed wall time covered by intervals, without double-counting overlap."""
    if not intervals:
        return 0.0

    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def build_timing_table(state: Optional[RunState], milestones: Dict[str, str]) -> Optional[str]:
    if state is None:
        return None

    times = state.milestones
    events = state.milestone_events or [(name, timestamp, None) for name, timestamp in times.items()]
    now = time.time()
    load_start = times.get(milestones["load_start"], min(times.values(), default=now))
    export_start = times.get(milestones["export_start"])
    compile_start = times.get(milestones["compile_start"])
    generation_start = times.get(milestones["generation_start"])

    # A stage ends when the next one starts; its completion event is a fallback
    # if the run stops before reaching the next stage.
    load_end = _event_timestamp(
        events,
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
    export_intervals, export_start_count, export_saw_skip = _stage_attempt_intervals(
        events,
        milestones["export_start"],
        (milestones["export_complete"], "export_failed"),
        milestones["export_skipped"],
        (milestones["compile_start"], milestones["generation_start"]),
        now,
    )
    if export_start_count > 1 or export_saw_skip:
        exporting = _union_duration(export_intervals)
    else:
        exporting = max(0.0, export_end - export_start) if export_start is not None else 0.0

    compile_complete = _timestamp(
        times,
        (milestones["compile_complete"], "compile_failed", milestones["compile_skipped"]),
        generation_start if generation_start is not None else now,
    )
    compile_end = max(generation_start, compile_complete) if generation_start is not None else compile_complete
    compile_intervals, compile_start_count, compile_saw_skip = _stage_attempt_intervals(
        events,
        milestones["compile_start"],
        (milestones["compile_complete"], "compile_failed"),
        milestones["compile_skipped"],
        (milestones["generation_start"],),
        now,
    )
    if compile_start_count > 1 or compile_saw_skip:
        compiling = _union_duration(compile_intervals)
    else:
        compiling = max(0.0, compile_end - compile_start) if compile_start is not None else 0.0

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
