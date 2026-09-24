#!/usr/bin/env python3
"""Create RAPID manuscript Figure 2 and its derived-value audit files.

Outputs PNG and CSV/TXT validation artifacts only. No PDF/SVG is produced.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, ScalarFormatter


RAPID = Path(__file__).resolve().parents[1]
ROOT = RAPID / "results" / "annotate_precision" / "stead_iso_2026-08-13"
OUT = RAPID / "docs" / "figures" / "paper_png"

MODELS = ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
SHORT = {
    "EQCCT": "EQCCT",
    "PhaseNet": "PN",
    "PhaseNetLight": "PNL",
    "EQTransformer": "EQT",
    "EQT-NC": "EQT-NC",
}
MODEL_COLORS = {
    "EQCCT": "#24557A",
    "PhaseNet": "#2A8C82",
    "PhaseNetLight": "#4C8BB8",
    "EQTransformer": "#B88A22",
    "EQT-NC": "#D97745",
}
MODEL_MARKERS = {
    "EQCCT": "o",
    "PhaseNet": "s",
    "PhaseNetLight": "^",
    "EQTransformer": "D",
    "EQT-NC": "P",
}

INK = "#17212B"
MUTED = "#65727E"
GRID = "#D8E0E6"
FP32 = "#596671"
BF16 = "#2A8C82"
FP16 = "#D97745"

BATCHES = [64, 128, 256, 512]
CPUS = [5, 10, 15, 20]
NETWORKS = [250, 580]
DEVICES = ["cpu", "gpu"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "semibold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.65,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def result_path(method: str, model: str, n_stations: int, device: str, n_cpus: int, batch: int) -> Path:
    modern = (
        ROOT
        / method
        / "stead"
        / f"{n_stations}st"
        / model
        / device
        / f"cpus{n_cpus}"
        / f"thr{n_cpus}"
        / f"bs{batch}"
        / "ann_prec"
        / "result.json"
    )
    if modern.is_file():
        return modern
    legacy = (
        ROOT
        / method
        / "stead"
        / f"{n_stations}st"
        / model
        / device
        / f"cpus{n_cpus}"
        / f"thr{n_cpus}"
        / "ann_prec"
        / "result.json"
    )
    return legacy if batch == 256 and legacy.is_file() else modern


def mean_runtime(method: str, model: str, n_stations: int, device: str, n_cpus: int, batch: int):
    path = result_path(method, model, n_stations, device, n_cpus, batch)
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    value = (result.get("timing") or {}).get("inference_s_mean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def derive_rows():
    rows = []
    for model in MODELS:
        for n_stations in NETWORKS:
            for device in DEVICES:
                for n_cpus in CPUS:
                    for batch in BATCHES:
                        fp32 = mean_runtime(
                            "annotate_fp32", model, n_stations, device, n_cpus, batch
                        )
                        bf16 = mean_runtime(
                            "annotate_bf16", model, n_stations, device, n_cpus, batch
                        )
                        fp16 = mean_runtime(
                            "annotate_fp16", model, n_stations, device, n_cpus, batch
                        )
                        if fp32 is None or bf16 is None:
                            continue
                        rows.append(
                            {
                                "model": model,
                                "n_stations": n_stations,
                                "device": device,
                                "n_cpus": n_cpus,
                                "batch_size": batch,
                                "fp32_s": fp32,
                                "bf16_s": bf16,
                                "bf16_speedup": fp32 / bf16,
                                "fp16_s": fp16,
                                "fp16_speedup": None if fp16 is None else fp32 / fp16,
                                "fp16_status": "supported" if fp16 is not None else "unsupported",
                            }
                        )

    keyed = {
        (r["model"], r["n_stations"], r["device"], r["n_cpus"], r["batch_size"]): r
        for r in rows
    }
    for row in rows:
        base_key = (
            row["model"],
            row["n_stations"],
            row["device"],
            row["n_cpus"],
            64,
        )
        base = keyed[base_key]
        row["fp32_relative_to_bs64"] = row["fp32_s"] / base["fp32_s"]
        row["bf16_relative_to_bs64"] = row["bf16_s"] / base["bf16_s"]
    return rows


def summary_record(panel, metric, device, group, values):
    values = [float(v) for v in values]
    return {
        "panel": panel,
        "metric": metric,
        "device": device,
        "group": group,
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "q25": percentile(values, 25),
        "q75": percentile(values, 75),
        "min": min(values),
        "max": max(values),
        "status": "supported",
    }


def derive_summary(rows):
    summary = []
    all_bf16 = [r["bf16_speedup"] for r in rows]
    summary.append(summary_record("A/B", "FP32/BF16 speedup", "all", "all models", all_bf16))

    for device in DEVICES:
        device_rows = [r for r in rows if r["device"] == device]
        summary.append(
            summary_record(
                "A",
                "FP32 runtime (s)",
                device,
                "all models",
                [r["fp32_s"] for r in device_rows],
            )
        )
        summary.append(
            summary_record(
                "A",
                "BF16 runtime (s)",
                device,
                "all models",
                [r["bf16_s"] for r in device_rows],
            )
        )
        for model in MODELS:
            model_rows = [r for r in device_rows if r["model"] == model]
            summary.append(
                summary_record(
                    "B",
                    "FP32/BF16 speedup",
                    device,
                    model,
                    [r["bf16_speedup"] for r in model_rows],
                )
            )

        for precision, field in (
            ("FP32", "fp32_s"),
            ("BF16", "bf16_s"),
        ):
            for model in MODELS:
                for batch in BATCHES:
                    values = [
                        r[field]
                        for r in device_rows
                        if r["model"] == model and r["batch_size"] == batch
                    ]
                    summary.append(
                        summary_record(
                            "C",
                            f"{precision} runtime (s)",
                            device,
                            f"{model}; batch={batch}",
                            values,
                        )
                    )

        for model in MODELS:
            values = [
                r["fp16_speedup"]
                for r in device_rows
                if r["model"] == model and r["fp16_speedup"] is not None
            ]
            if values:
                summary.append(
                    summary_record(
                        "D", "FP32/FP16 speedup", device, model, values
                    )
                )
            else:
                summary.append(
                    {
                        "panel": "D",
                        "metric": "FP32/FP16 speedup",
                        "device": device,
                        "group": model,
                        "n": 0,
                        "mean": "",
                        "median": "",
                        "q25": "",
                        "q75": "",
                        "min": "",
                        "max": "",
                        "status": "unsupported",
                    }
                )
    return summary


def validate(rows, summary) -> str:
    bf16 = [r["bf16_speedup"] for r in rows]
    fp16 = [r["fp16_speedup"] for r in rows if r["fp16_speedup"] is not None]
    fp16_wins = sum(value > 1 for value in fp16)
    unsupported = {
        r["model"]
        for r in rows
        if r["fp16_status"] == "unsupported"
    }

    assert len(rows) == 320, f"Expected 320 FP32/BF16 matches, found {len(rows)}"
    assert math.isclose(statistics.mean(bf16), 1.1666840739299065, rel_tol=1e-10)
    assert math.isclose(statistics.median(bf16), 1.080081705448793, rel_tol=1e-10)
    assert len(fp16) == 192, f"Expected 192 successful FP16 matches, found {len(fp16)}"
    assert unsupported == {"EQTransformer", "EQT-NC"}, unsupported

    # Every Panel C trajectory must contain the same four batch sizes.
    complete = defaultdict(set)
    for row in rows:
        key = (row["model"], row["n_stations"], row["device"], row["n_cpus"])
        complete[key].add(row["batch_size"])
    assert len(complete) == 80
    assert all(batches == set(BATCHES) for batches in complete.values())

    batch_overall = {}
    batch_actual = {}
    for batch in BATCHES:
        values, actual_values = [], []
        for row in rows:
            if row["batch_size"] != batch:
                continue
            values.extend(
                [
                    row["fp32_relative_to_bs64"],
                    row["bf16_relative_to_bs64"],
                ]
            )
            actual_values.extend([row["fp32_s"], row["bf16_s"]])
        batch_overall[batch] = statistics.mean(values)
        batch_actual[batch] = statistics.mean(actual_values)
    assert min(batch_overall, key=batch_overall.get) == 512
    assert min(batch_actual, key=batch_actual.get) == 512

    lines = [
        "Figure 2 / Supplementary Figure S1 validation",
        f"FP32/BF16 matched cells: n={len(rows)}",
        f"BF16 speedup mean: {statistics.mean(bf16):.12f}",
        f"BF16 speedup median: {statistics.median(bf16):.12f}",
        f"Successful FP16/FP32 matched cells: n={len(fp16)}",
        f"FP16 faster-than-FP32 cells: n={fp16_wins}/{len(fp16)}",
        "FP16 unsupported models: EQTransformer, EQT-NC",
        f"Complete batch trajectories: n={len(complete)} configuration series per precision",
        "Overall normalized runtime by batch: "
        + ", ".join(f"{batch}={batch_overall[batch]:.12f}" for batch in BATCHES),
        "Overall mean runtime (s) by batch: "
        + ", ".join(f"{batch}={batch_actual[batch]:.12f}" for batch in BATCHES),
        "Selected overall batch: 512",
        f"Summary rows written: n={len(summary)}",
    ]
    return "\n".join(lines) + "\n"


def write_derived_values(rows, summary, validation):
    OUT.mkdir(parents=True, exist_ok=True)
    values_path = OUT / "figure02_panel_values.csv"
    with values_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUT / "figure02_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    validation_path = OUT / "figure02_validation.txt"
    validation_path.write_text(validation)
    return values_path, summary_path, validation_path


def clean_axis(ax, *, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)


def panel_label(ax, label):
    ax.text(
        -0.18,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )


def model_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[m],
            marker=MODEL_MARKERS[m],
            linewidth=0,
            markersize=5,
            label=SHORT[m],
        )
        for m in MODELS
    ]


def scatter_panel(ax, rows, device):
    selected = [r for r in rows if r["device"] == device]
    for model in MODELS:
        model_rows = [r for r in selected if r["model"] == model]
        ax.scatter(
            [r["fp32_s"] for r in model_rows],
            [r["bf16_s"] for r in model_rows],
            s=9,
            marker=MODEL_MARKERS[model],
            facecolor=MODEL_COLORS[model],
            edgecolor="none",
            alpha=0.56,
            rasterized=True,
        )
    values = [r["fp32_s"] for r in selected] + [r["bf16_s"] for r in selected]
    low, high = min(values) * 0.82, max(values) * 1.22
    ax.plot([low, high], [low, high], color=INK, linewidth=0.75, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("FP32 runtime (s)")
    ax.set_ylabel("BF16 runtime (s)" if device == "cpu" else "")
    ax.set_title(device.upper())
    clean_axis(ax, grid_axis=None)


def distribution_panel(ax, rows, device, field, *, fp16=False):
    positions = np.arange(1, len(MODELS) + 1)
    data = []
    for model in MODELS:
        values = [
            r[field]
            for r in rows
            if r["device"] == device
            and r["model"] == model
            and r[field] is not None
        ]
        data.append(values)

    supported_positions = [p for p, values in zip(positions, data) if values]
    supported_data = [values for values in data if values]
    boxes = ax.boxplot(
        supported_data,
        positions=supported_positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.0},
        boxprops={"linewidth": 0.75, "edgecolor": INK},
        whiskerprops={"linewidth": 0.65, "color": MUTED},
        capprops={"linewidth": 0.65, "color": MUTED},
    )
    for box, position in zip(boxes["boxes"], supported_positions):
        model = MODELS[position - 1]
        box.set_facecolor(MODEL_COLORS[model])
        box.set_alpha(0.32)

    for position, model, values in zip(positions, MODELS, data):
        if not values:
            continue
        jitter = np.linspace(-0.18, 0.18, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=8,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            alpha=0.42,
            edgecolor="none",
            rasterized=True,
        )

    ax.axhline(1, color=INK, linewidth=0.75, linestyle="--")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT[m] for m in MODELS])
    if device == "cpu":
        ax.set_ylabel("FP16 speedup (×)" if fp16 else "BF16 speedup (×)")
    else:
        ax.set_ylabel("")
    ax.set_title(device.upper())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    if fp16:
        if device == "cpu":
            ax.set_ylim(0.075, 3.1)
            ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 3])
            unsupported_y = 0.105
        else:
            ax.set_ylim(0.88, 1.29)
            ax.set_yticks([0.9, 1.0, 1.1, 1.2])
            unsupported_y = 0.91
        for position in (4, 5):
            ax.text(
                position,
                unsupported_y,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=MUTED,
            )
    elif device == "cpu":
        ax.set_ylim(0.18, 5.8)
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
    else:
        ax.set_ylim(0.70, 1.65)
        ax.set_yticks([0.8, 1.0, 1.2, 1.5])
    clean_axis(ax)


def batch_panel(ax, rows, device):
    for precision, field, color, linestyle in (
        ("FP32", "fp32_relative_to_bs64", FP32, "--"),
        ("BF16", "bf16_relative_to_bs64", BF16, "-"),
    ):
        for model in MODELS:
            medians = []
            for batch in BATCHES:
                values = [
                    r[field]
                    for r in rows
                    if r["device"] == device
                    and r["model"] == model
                    and r["batch_size"] == batch
                ]
                medians.append(statistics.median(values))
            ax.plot(
                BATCHES,
                medians,
                color=color,
                linestyle=linestyle,
                linewidth=0.9,
                marker=MODEL_MARKERS[model],
                markersize=3.4,
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor=MODEL_COLORS[model],
                alpha=0.88,
            )
    ax.axhline(1, color=INK, linewidth=0.65, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(BATCHES)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Runtime / runtime at batch 64" if device == "cpu" else "")
    ax.set_title(device.upper())
    if device == "cpu":
        ax.set_ylim(0.40, 1.35)
        ax.set_yticks([0.5, 0.75, 1.0, 1.25])
        ax.legend(
            handles=[
                Line2D([0], [0], color=FP32, linestyle="--", linewidth=1.1, label="FP32"),
                Line2D([0], [0], color=BF16, linestyle="-", linewidth=1.1, label="BF16"),
            ],
            loc="lower left",
            ncol=2,
        )
    else:
        ax.set_ylim(0.50, 1.14)
        ax.set_yticks([0.5, 0.75, 1.0])
    clean_axis(ax)


def batch_small_multiple(ax, rows, model, device, *, row_index):
    """One model/device batch panel with raw values, median, and IQR."""
    x = np.arange(len(BATCHES), dtype=float)
    precision_specs = (
        ("FP32", "fp32_s", FP32, "--", -0.045),
        ("BF16", "bf16_s", BF16, "-", 0.045),
    )
    panel_values = [
        r[field]
        for _, field, _, _, _ in precision_specs
        for r in rows
        if r["model"] == model and r["device"] == device
    ]
    if device == "cpu":
        # Tighten the view to the central 90%; boundary triangles retain
        # visibility of observations outside the displayed range.
        display_low = percentile(panel_values, 5) * 0.95
        display_high = percentile(panel_values, 95) * 1.05
    else:
        display_low = min(panel_values) * 0.82
        display_high = max(panel_values) * 1.22

    for _, field, color, linestyle, offset in precision_specs:
        medians, q25, q75 = [], [], []
        for batch_index, batch in enumerate(BATCHES):
            values = [
                r[field]
                for r in rows
                if r["model"] == model
                and r["device"] == device
                and r["batch_size"] == batch
            ]
            assert len(values) == 8
            jitter = np.linspace(-0.055, 0.055, len(values))
            point_x = batch_index + offset + jitter
            visible = [
                display_low <= value <= display_high
                for value in values
            ]
            ax.scatter(
                point_x[visible],
                np.asarray(values)[visible],
                s=6,
                color=color,
                alpha=0.22,
                edgecolor="none",
                rasterized=True,
                zorder=2,
            )
            below = np.asarray(values) < display_low
            above = np.asarray(values) > display_high
            if np.any(below):
                ax.scatter(
                    point_x[below],
                    np.full(np.count_nonzero(below), display_low * 1.04),
                    s=7,
                    marker="v",
                    color=color,
                    alpha=0.38,
                    edgecolor="none",
                    zorder=4,
                )
            if np.any(above):
                ax.scatter(
                    point_x[above],
                    np.full(np.count_nonzero(above), display_high / 1.04),
                    s=7,
                    marker="^",
                    color=color,
                    alpha=0.38,
                    edgecolor="none",
                    zorder=4,
                )
            medians.append(statistics.median(values))
            q25.append(percentile(values, 25))
            q75.append(percentile(values, 75))

        shifted_x = x + offset
        ax.fill_between(
            shifted_x,
            q25,
            q75,
            color=color,
            alpha=0.09,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            shifted_x,
            medians,
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
            marker="o",
            markersize=2.6,
            zorder=3,
        )

    ax.set_xlim(-0.25, 3.25)
    ax.set_xticks(x)
    if row_index == len(MODELS) - 1:
        ax.set_xticklabels(BATCHES)
        ax.set_xlabel("Batch size")
    else:
        ax.set_xticklabels([])
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_ylim(display_low, display_high)
    readable_ticks = [
        value
        for value in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)
        if display_low <= value <= display_high
    ]
    ax.set_yticks(readable_ticks)
    clean_axis(ax)


def add_shared_model_legend(fig):
    fig.legend(
        handles=model_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=5,
        columnspacing=1.35,
        handletextpad=0.35,
    )


def make_main_figure(rows):
    """Main Figure 2: BF16 matched runtimes and speedup distributions."""
    fig = plt.figure(figsize=(7.25, 5.25))
    grid = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.28)
    axes = np.asarray(
        [[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(2)]
    )

    for col, device in enumerate(DEVICES):
        scatter_panel(axes[0, col], rows, device)
        distribution_panel(axes[1, col], rows, device, "bf16_speedup")

    for row, label in enumerate("AB"):
        panel_label(axes[row, 0], label)

    add_shared_model_legend(fig)
    fig.subplots_adjust(top=0.925, bottom=0.095, left=0.105, right=0.985)
    return fig


def make_supplement_figure(rows):
    """Supplementary Figure S1: batch-size and FP16 supporting results."""
    fig = plt.figure(figsize=(7.25, 9.0))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[5.15, 1.65],
        hspace=0.23,
    )
    batch_grid = outer[0].subgridspec(5, 2, hspace=0.13, wspace=0.24)
    fp16_grid = outer[1].subgridspec(1, 2, wspace=0.28)

    batch_axes = np.asarray(
        [
            [fig.add_subplot(batch_grid[row, col]) for col in range(2)]
            for row in range(5)
        ]
    )
    for row_index, model in enumerate(MODELS):
        for col, device in enumerate(DEVICES):
            ax = batch_axes[row_index, col]
            batch_small_multiple(
                ax,
                rows,
                model,
                device,
                row_index=row_index,
            )
            if row_index == 0:
                ax.set_title(device.upper())
            if col == 0:
                ax.text(
                    -0.24,
                    0.50,
                    SHORT[model],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7.5,
                    fontweight="semibold",
                    color=INK,
                )
            if row_index == 2 and col == 0:
                ax.set_ylabel("Runtime (s)")

    panel_label(batch_axes[0, 0], "A")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=FP32,
                linestyle="--",
                marker="o",
                markersize=3,
                linewidth=1.15,
                label="FP32",
            ),
            Line2D(
                [0],
                [0],
                color=BF16,
                linestyle="-",
                marker="o",
                markersize=3,
                linewidth=1.15,
                label="BF16",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.994),
        ncol=2,
        columnspacing=1.4,
        handletextpad=0.45,
    )

    fp16_axes = [fig.add_subplot(fp16_grid[0, col]) for col in range(2)]
    for col, device in enumerate(DEVICES):
        distribution_panel(
            fp16_axes[col],
            rows,
            device,
            "fp16_speedup",
            fp16=True,
        )
    panel_label(fp16_axes[0], "B")

    fig.subplots_adjust(top=0.955, bottom=0.065, left=0.125, right=0.985)
    return fig


def main() -> int:
    configure_style()
    rows = derive_rows()
    summary = derive_summary(rows)
    validation = validate(rows, summary)

    # Audit artifacts are deliberately written before the plot is exported.
    values_path, summary_path, validation_path = write_derived_values(
        rows, summary, validation
    )
    print(validation, end="")
    print(values_path)
    print(summary_path)
    print(validation_path)

    main_figure = make_main_figure(rows)
    main_path = OUT / "figure02_reduced_precision_runtime.png"
    main_figure.savefig(main_path, dpi=600)
    plt.close(main_figure)
    print(main_path)

    supplement_figure = make_supplement_figure(rows)
    supplement_path = OUT / "figureS01_reduced_precision_support.png"
    supplement_figure.savefig(supplement_path, dpi=600)
    plt.close(supplement_figure)
    print(supplement_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
