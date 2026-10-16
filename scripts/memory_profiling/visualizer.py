# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEfficient Memory Visualizer - Production Quality Enhanced Visualization

This module provides production-quality visualization with detailed segment analysis,
clear operation boundaries, and comprehensive memory metrics.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .profiler import QEffMemoryProfiler

from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


class QEffMemoryVisualizer:
    """Production-quality memory visualization with enhanced segment analysis."""

    def __init__(self, profiler: "QEffMemoryProfiler"):
        """Initialize visualizer with profiler data."""
        self.profiler = profiler
        self._setup_matplotlib_style()

    def _setup_matplotlib_style(self) -> None:
        """Configure matplotlib for professional styling."""
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.size": 10,
                "font.family": ["DejaVu Sans", "sans-serif"],
                "axes.linewidth": 1.2,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "grid.alpha": 0.3,
                "lines.linewidth": 2.0,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#333333",
                "text.color": "#333333",
                "axes.labelcolor": "#333333",
                "xtick.color": "#333333",
                "ytick.color": "#333333",
            }
        )

    def generate_professional_graph(self, filename: str) -> None:
        """Generate enhanced multi-panel memory profile with synchronized visualization."""
        if not self.profiler.samples:
            logger.warning("No data to plot")
            return

        # Get synchronized data
        sync_data = self.profiler.get_synchronized_data()

        # Create figure with professional layout - Fixed spacing to prevent title overlap
        fig = plt.figure(figsize=(20, 12), facecolor="white")
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[2.5, 1.8, 1.2],
            width_ratios=[1, 1],
            hspace=0.35,
            wspace=0.2,
            left=0.05,
            right=0.98,
            top=0.90,
            bottom=0.08,
        )

        # Create subplots
        ax_memory = fig.add_subplot(gs[0, :])  # Memory usage (full width)
        ax_cpu = fig.add_subplot(gs[1, :])  # CPU usage (full width)
        ax_disk = fig.add_subplot(gs[2, 0])  # Disk I/O (left)
        ax_timing = fig.add_subplot(gs[2, 1])  # Phase Duration (right)

        # Prepare data
        relative_times = sync_data["timestamps"]
        max_rss = max(sync_data["rss_memory"]) if sync_data["rss_memory"] else 0
        use_gb = max_rss > 2048
        scale = 1024 if use_gb else 1
        unit = "GB" if use_gb else "MB"
        rss_scaled = [x / scale for x in sync_data["rss_memory"]]

        # Normalize CPU usage to prevent > 100% values (multi-core issue)
        normalized_cpu = [min(cpu, 100.0) for cpu in sync_data["cpu_usage"]]

        # Setup plots
        self._setup_memory_plot(ax_memory, relative_times, rss_scaled, scale, unit)
        self._setup_cpu_plot(ax_cpu, relative_times, normalized_cpu)
        self._setup_disk_io_plot(ax_disk, sync_data)
        self._setup_timing_plot(ax_timing)

        # Add main title with proper spacing
        fig.suptitle(
            "QEfficient Enhanced Memory & Performance Analysis - Synchronized View",
            fontsize=18,
            fontweight="bold",
            color="#2E86AB",
            y=0.95,
        )

        # Save with high quality
        plt.savefig(
            filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", format="png", pad_inches=0.2
        )
        plt.close()

        logger.info(f"Enhanced synchronized memory profile saved: {filename}")

    def _setup_memory_plot(
        self, ax, relative_times: List[float], rss_scaled: List[float], scale: float, unit: str
    ) -> None:
        """Setup the main memory usage plot with enhanced visualization."""
        if not relative_times or not rss_scaled:
            ax.text(
                0.5,
                0.5,
                "No memory data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#666666",
            )
            return

        start_time = self.profiler.samples[0].timestamp

        # Draw segment backgrounds
        self._draw_segment_backgrounds(ax, relative_times, rss_scaled, start_time)

        # Main memory line
        ax.plot(
            relative_times, rss_scaled, color="#2E86AB", linewidth=3.5, label="Memory Usage (RSS)", alpha=0.9, zorder=5
        )
        ax.fill_between(relative_times, rss_scaled, alpha=0.15, color="#2E86AB", zorder=1)

        # Add segment boundaries and annotations
        self._draw_segment_boundaries(ax, start_time, max(rss_scaled))
        self._mark_peak_memory(ax, start_time, scale, unit)

        # Format axes
        ax.set_xlabel("Time (seconds)", fontsize=13, fontweight="bold")
        ax.set_ylabel(f"Memory Usage ({unit})", fontsize=13, fontweight="bold")
        ax.set_xlim(0, max(relative_times) * 1.02)
        ax.set_ylim(0, max(rss_scaled) * 1.15)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color="#CCCCCC")
        ax.set_axisbelow(True)

        # Enhanced title
        total_duration = relative_times[-1] if relative_times else 0
        peak_memory = max(rss_scaled) if rss_scaled else 0
        ax.set_title(
            f"Memory Usage Over Time | Peak: {peak_memory:.1f} {unit} | Duration: {total_duration:.1f}s",
            fontsize=14,
            fontweight="bold",
            color="#2E86AB",
            pad=15,
        )

        # Add legend
        self._add_segment_legend(ax)

    def _setup_cpu_plot(self, ax, relative_times: List[float], cpu_usage: List[float]) -> None:
        """Setup CPU plot with perfect synchronization to memory plot."""
        if not relative_times or not cpu_usage or len(cpu_usage) != len(relative_times):
            ax.text(
                0.5,
                0.5,
                "CPU data not available or not synchronized",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#666666",
            )
            ax.set_title("CPU Usage Over Time", fontsize=14, fontweight="bold")
            if relative_times:
                ax.set_xlim(0, max(relative_times) * 1.02)
            return

        start_time = self.profiler.samples[0].timestamp

        # Draw segment backgrounds for consistency
        self._draw_segment_backgrounds(ax, relative_times, cpu_usage, start_time, max_val=100)

        # Main CPU line
        ax.plot(relative_times, cpu_usage, color="#FF6B35", linewidth=3, label="CPU Usage", alpha=0.9, zorder=5)
        ax.fill_between(relative_times, cpu_usage, alpha=0.2, color="#FF6B35", zorder=1)

        # Add segment boundaries
        self._draw_segment_boundaries(ax, start_time, max(cpu_usage) if cpu_usage else 100)

        # Add average line
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        ax.axhline(
            y=avg_cpu,
            color="#E74C3C",
            linestyle="-",
            alpha=0.8,
            linewidth=2.5,
            label=f"Average: {avg_cpu:.1f}%",
            zorder=4,
        )

        # Add performance zones
        ax.axhspan(0, 25, alpha=0.08, color="#4CAF50", zorder=0)
        ax.axhspan(25, 50, alpha=0.08, color="#FFC107", zorder=0)
        ax.axhspan(50, 75, alpha=0.08, color="#FF9800", zorder=0)
        ax.axhspan(75, 100, alpha=0.08, color="#F44336", zorder=0)

        # Format axes
        ax.set_ylabel("CPU Usage (%)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(relative_times) * 1.02)
        ax.set_ylim(0, max(cpu_usage) * 1.1 if cpu_usage else 100)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color="#CCCCCC")
        ax.set_axisbelow(True)

        # Enhanced title
        max_cpu = max(cpu_usage)
        ax.set_title(
            f"CPU Usage Over Time | Peak: {max_cpu:.1f}% | Average: {avg_cpu:.1f}%",
            fontsize=14,
            fontweight="bold",
            color="#FF6B35",
            pad=15,
        )

        # Compact legend
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    def _setup_disk_io_plot(self, ax, sync_data: Dict[str, List[float]]) -> None:
        """Setup enhanced disk I/O plot showing phase-based analysis."""
        if not self.profiler.operations or len(self.profiler.operations) < 2:
            ax.text(
                0.5,
                0.5,
                "No operation phases available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#666666",
            )
            ax.set_title("Disk I/O per Phase", fontsize=14, fontweight="bold")
            return

        # Calculate I/O per phase
        operations, read_totals, write_totals = self._calculate_io_per_phase(sync_data)

        if not operations:
            ax.text(
                0.5,
                0.5,
                "No significant disk I/O detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#666666",
            )
            ax.set_title("Disk I/O per Phase", fontsize=14, fontweight="bold")
            return

        # Create enhanced bar chart
        x_pos = np.arange(len(operations))
        bar_width = 0.35

        bars_read = ax.bar(
            x_pos - bar_width / 2,
            read_totals,
            bar_width,
            label="Read (MB)",
            color="#2196F3",
            alpha=0.8,
            edgecolor="white",
            linewidth=1.5,
        )
        bars_write = ax.bar(
            x_pos + bar_width / 2,
            write_totals,
            bar_width,
            label="Write (MB)",
            color="#FF5722",
            alpha=0.8,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels
        self._add_bar_labels(ax, bars_read, bars_write, read_totals, write_totals)

        # Format axes
        ax.set_ylabel("Total I/O (MB)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Operation Phase", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(operations, rotation=45, ha="right", fontsize=10)

        max_val = max(max(read_totals) if read_totals else [0], max(write_totals) if write_totals else [0])
        ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#CCCCCC", axis="y")
        ax.set_title("Disk I/O per Operation Phase", fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

        # Summary statistics
        total_read = sum(read_totals)
        total_write = sum(write_totals)
        ax.text(
            0.02,
            0.98,
            f"Total I/O: {total_read:.1f} MB read, {total_write:.1f} MB write",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=1),
        )

    def _setup_timing_plot(self, ax) -> None:
        """Setup enhanced timing analysis plot."""
        operations, durations, colors = self._get_timing_data()

        if not operations:
            ax.text(
                0.5,
                0.5,
                "No timing data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#666666",
            )
            ax.set_title("Phase Duration Analysis", fontsize=14, fontweight="bold")
            return

        # Enhanced horizontal bar chart
        y_pos = np.arange(len(operations))
        bars = ax.barh(y_pos, durations, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5, height=0.6)

        # Add duration labels
        self._add_duration_labels(ax, bars, durations)

        # Format axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(operations, fontsize=11)
        ax.set_xlabel("Duration (seconds)", fontsize=12, fontweight="bold")
        ax.set_title("Phase Duration Analysis", fontsize=14, fontweight="bold", pad=15)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#CCCCCC", axis="x")
        ax.set_xlim(0, max(durations) * 1.2)

        # Add total duration summary
        total_duration = sum(durations)
        ax.text(
            0.98,
            0.02,
            f"Total: {total_duration:.1f}s",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=1),
        )

    def _draw_segment_backgrounds(
        self,
        ax,
        relative_times: List[float],
        data_values: List[float],
        start_time: datetime,
        max_val: Optional[float] = None,
    ) -> None:
        """Draw colored background segments for each operation."""
        if len(self.profiler.operations) < 2:
            return

        max_value = max_val or (max(data_values) * 1.1 if data_values else 100)

        for i in range(len(self.profiler.operations) - 1):
            op_start_time = (self.profiler.operations[i][0] - start_time).total_seconds()
            op_end_time = (self.profiler.operations[i + 1][0] - start_time).total_seconds()
            op_name = self.profiler.operations[i][1]

            color = self.profiler.SEGMENT_COLORS.get(op_name, "#F0F0F0")

            rect = patches.Rectangle(
                (op_start_time, 0),
                op_end_time - op_start_time,
                max_value,
                linewidth=0,
                facecolor=color,
                alpha=0.15,
                zorder=0,
            )
            ax.add_patch(rect)

    def _draw_segment_boundaries(self, ax, start_time: datetime, max_value: float) -> None:
        """Draw vertical lines at segment boundaries."""
        for i, (op_time, op_name) in enumerate(self.profiler.operations):
            if i == 0:
                continue

            boundary_time = (op_time - start_time).total_seconds()
            ax.axvline(x=boundary_time, color="#666666", linestyle="--", alpha=0.6, linewidth=2, zorder=3)

    def _mark_peak_memory(self, ax, start_time: datetime, scale: float, unit: str) -> None:
        """Mark peak memory with enhanced annotation."""
        if not self.profiler.peak_rss_time:
            return

        peak_time_rel = (self.profiler.peak_rss_time - start_time).total_seconds()
        peak_rss_scaled = self.profiler.peak_rss / scale

        # Enhanced peak marker
        ax.plot(
            peak_time_rel,
            peak_rss_scaled,
            "o",
            color="#E74C3C",
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=3,
            zorder=10,
            label="Peak Memory",
        )

        # Enhanced annotation
        peak_text = f"Peak: {peak_rss_scaled:.1f} {unit}\nPhase: {self.profiler.peak_operation}"
        ax.annotate(
            peak_text,
            xy=(peak_time_rel, peak_rss_scaled),
            xytext=(25, 25),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#E74C3C", alpha=0.95, edgecolor="white", linewidth=2),
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2.5),
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="left",
            va="bottom",
            zorder=15,
        )

    def _add_segment_legend(self, ax) -> None:
        """Add enhanced segment legend with better styling."""
        if not self.profiler.operations:
            return

        unique_ops = []
        seen_ops = set()
        for _, op_name in self.profiler.operations:
            if op_name not in seen_ops and op_name not in ["Initialization", "Completion"]:
                unique_ops.append(op_name)
                seen_ops.add(op_name)

        if not unique_ops:
            return

        legend_elements = []
        for op_name in unique_ops:
            color = self.profiler.SEGMENT_COLORS.get(op_name, "#666666")
            duration = self.profiler.operation_durations.get(op_name, 0)

            label = f"{op_name} ({duration:.1f}s)" if duration > 0 else op_name
            legend_elements.append(patches.Patch(color=color, alpha=0.8, label=label))

        legend = ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=11,
            title="QEfficient Phases",
            title_fontsize=12,
            framealpha=0.95,
            edgecolor="#2E86AB",
            fancybox=True,
        )
        legend.get_frame().set_facecolor("#F8F9FA")

    def _calculate_io_per_phase(self, sync_data: Dict[str, List[float]]) -> Tuple[List[str], List[float], List[float]]:
        """Calculate I/O totals per operation phase."""
        operations = []
        read_totals = []
        write_totals = []

        valid_operations = [
            (op_time, op_name)
            for op_time, op_name in self.profiler.operations
            if op_name not in ["Initialization", "Completion"]
        ]

        if not valid_operations:
            return operations, read_totals, write_totals

        relative_times = sync_data["timestamps"]
        start_time = self.profiler.samples[0].timestamp

        for i, (op_time, op_name) in enumerate(valid_operations):
            op_start_time = (op_time - start_time).total_seconds()

            if i + 1 < len(valid_operations):
                op_end_time = (valid_operations[i + 1][0] - start_time).total_seconds()
            else:
                op_end_time = max(relative_times) if relative_times else op_start_time + 1

            # Find data indices
            start_idx = next((j for j, t in enumerate(relative_times) if t >= op_start_time), 0)
            end_idx = next((j for j, t in enumerate(relative_times) if t >= op_end_time), len(relative_times) - 1)

            if start_idx < len(sync_data["disk_read_bytes"]) and end_idx < len(sync_data["disk_read_bytes"]):
                read_total = sync_data["disk_read_bytes"][end_idx] - sync_data["disk_read_bytes"][start_idx]
                write_total = sync_data["disk_write_bytes"][end_idx] - sync_data["disk_write_bytes"][start_idx]

                if read_total > 0.01 or write_total > 0.01:
                    operations.append(op_name)
                    read_totals.append(max(0, read_total))
                    write_totals.append(max(0, write_total))

        return operations, read_totals, write_totals

    def _get_timing_data(self) -> Tuple[List[str], List[float], List[str]]:
        """Get timing data for operations."""
        operations = []
        durations = []
        colors = []

        for op_time, op_name in self.profiler.operations:
            if op_name in ["Initialization", "Completion"]:
                continue
            duration = self.profiler.operation_durations.get(op_name, 0)
            if duration > 0:
                operations.append(op_name)
                durations.append(duration)
                colors.append(self.profiler.SEGMENT_COLORS.get(op_name, "#666666"))

        return operations, durations, colors

    def _add_bar_labels(self, ax, bars_read, bars_write, read_totals: List[float], write_totals: List[float]) -> None:
        """Add value labels on bars."""
        max_val = max(max(read_totals) if read_totals else [0], max(write_totals) if write_totals else [0])

        for i, (read_bar, write_bar, read_val, write_val) in enumerate(
            zip(bars_read, bars_write, read_totals, write_totals)
        ):
            if read_val > 0.01:
                ax.text(
                    read_bar.get_x() + read_bar.get_width() / 2,
                    read_bar.get_height() + max_val * 0.02,
                    f"{read_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#2196F3",
                )

            if write_val > 0.01:
                ax.text(
                    write_bar.get_x() + write_bar.get_width() / 2,
                    write_bar.get_height() + max_val * 0.02,
                    f"{write_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#FF5722",
                )

    def _add_duration_labels(self, ax, bars, durations: List[float]) -> None:
        """Add duration labels on timing bars."""
        max_duration = max(durations)

        for i, (bar, duration) in enumerate(zip(bars, durations)):
            width = bar.get_width()
            minutes = int(duration // 60)
            seconds = duration % 60

            if minutes > 0:
                duration_text = f"{minutes}m {seconds:.1f}s"
            else:
                duration_text = f"{seconds:.1f}s"

            ax.text(
                width + max_duration * 0.02,
                bar.get_y() + bar.get_height() / 2,
                duration_text,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
