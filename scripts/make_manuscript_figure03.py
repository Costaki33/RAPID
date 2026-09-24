#!/usr/bin/env python3
"""Create RAPID manuscript Figure 3 and its validation artifacts.

Outputs a 600-dpi PNG plus processed CSV/TXT files. No PDF/SVG is produced.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


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

PRECISIONS = {
    "annotate_bf16": ("BF16", "#2A8C82"),
    "annotate_fp16": ("FP16", "#D97745"),
}
PHASE_MARKERS = {"P": "o", "S": "^"}

NETWORKS = [250, 580]
DEVICES = ["cpu", "gpu"]
CPUS = [5, 10, 15, 20]
BATCHES = [64, 128, 256, 512]
PHASES = ["P", "S"]

INK = "#17212B"
MUTED = "#65727E"
GRID = "#D8E0E6"
NA_COLOR = "#E7EAED"
HEATMAP = LinearSegmentedColormap.from_list(
    "agreement",
    ["#F2C4BE", "#F8DEDA", "#FAFAF9", "#E2EDF6", "#C4DBEC"],
)
HEATMAP.set_bad(NA_COLOR)


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


def result_path(
    method: str,
    model: str,
    n_stations: int,
    device: str,
    n_cpus: int,
    batch: int,
) -> Path:
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


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def block_metric(block, phase: str, metric: str):
    return finite((block or {}).get(f"{phase}.{metric}_mean"))


def first_repeat_phase(block, phase: str):
    repeats = (block or {}).get("repeats") or []
    return (repeats[0].get(phase) or {}) if repeats else {}


def derive_rows():
    rows = []
    for method, (precision, _) in PRECISIONS.items():
        for model in MODELS:
            for n_stations in NETWORKS:
                for device in DEVICES:
                    for n_cpus in CPUS:
                        for batch in BATCHES:
                            reduced_path = result_path(
                                method,
                                model,
                                n_stations,
                                device,
                                n_cpus,
                                batch,
                            )
                            fp32_path = result_path(
                                "annotate_fp32",
                                model,
                                n_stations,
                                device,
                                n_cpus,
                                batch,
                            )
                            if not fp32_path.is_file():
                                raise RuntimeError(f"Missing matched FP32 cell: {fp32_path}")

                            if not reduced_path.is_file():
                                if method == "annotate_fp16" and model in {
                                    "EQTransformer",
                                    "EQT-NC",
                                }:
                                    for phase in PHASES:
                                        rows.append(
                                            {
                                                "precision": precision,
                                                "method": method,
                                                "model": model,
                                                "n_stations": n_stations,
                                                "device": device,
                                                "n_cpus": n_cpus,
                                                "n_threads": n_cpus,
                                                "batch_size": batch,
                                                "packaging": "merged",
                                                "phase": phase,
                                                "status": "unsupported",
                                                "direct_precision": None,
                                                "direct_recall": None,
                                                "direct_f1": None,
                                                "direct_matched": None,
                                                "direct_missing": None,
                                                "direct_additional": None,
                                                "direct_mean_dt_samples": None,
                                                "direct_p95_abs_dt_samples": None,
                                                "fp32_catalog_f1": None,
                                                "reduced_catalog_f1": None,
                                                "catalog_delta_f1": None,
                                            }
                                        )
                                    continue
                                raise RuntimeError(f"Unexpected missing reduced cell: {reduced_path}")

                            reduced = json.loads(reduced_path.read_text())
                            fp32 = json.loads(fp32_path.read_text())
                            direct = reduced.get("pick_quality_vs_fp32") or {}
                            reduced_catalog = (
                                reduced.get("pick_quality_vs_catalog") or {}
                            )
                            fp32_catalog = fp32.get("pick_quality_vs_catalog") or {}

                            for phase in PHASES:
                                direct_f1 = block_metric(direct, phase, "f1")
                                reduced_catalog_f1 = block_metric(
                                    reduced_catalog, phase, "f1"
                                )
                                fp32_catalog_f1 = block_metric(
                                    fp32_catalog, phase, "f1"
                                )
                                first = first_repeat_phase(direct, phase)
                                n_detected = first.get("n_detected")
                                if direct_f1 is not None:
                                    status = "valid"
                                elif n_detected == 0:
                                    status = "no_picks"
                                else:
                                    status = "invalid"
                                delta = (
                                    None
                                    if reduced_catalog_f1 is None
                                    or fp32_catalog_f1 is None
                                    else reduced_catalog_f1 - fp32_catalog_f1
                                )
                                rows.append(
                                    {
                                        "precision": precision,
                                        "method": method,
                                        "model": model,
                                        "n_stations": n_stations,
                                        "device": device,
                                        "n_cpus": n_cpus,
                                        "n_threads": n_cpus,
                                        "batch_size": batch,
                                        "packaging": "merged",
                                        "phase": phase,
                                        "status": status,
                                        "direct_precision": block_metric(
                                            direct, phase, "precision"
                                        ),
                                        "direct_recall": block_metric(
                                            direct, phase, "recall"
                                        ),
                                        "direct_f1": direct_f1,
                                        "direct_matched": first.get("matched"),
                                        "direct_missing": first.get("missing"),
                                        "direct_additional": first.get("additional"),
                                        "direct_mean_dt_samples": block_metric(
                                            direct, phase, "mean_dt"
                                        ),
                                        "direct_p95_abs_dt_samples": block_metric(
                                            direct, phase, "p95_abs_dt"
                                        ),
                                        "fp32_catalog_f1": fp32_catalog_f1,
                                        "reduced_catalog_f1": reduced_catalog_f1,
                                        "catalog_delta_f1": delta,
                                    }
                                )
    return rows


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def direct_summary(rows):
    summary = []
    for precision in ("BF16", "FP16"):
        for model in MODELS:
            for phase in PHASES:
                selected = [
                    r["direct_f1"]
                    for r in rows
                    if r["precision"] == precision
                    and r["model"] == model
                    and r["phase"] == phase
                    and r["direct_f1"] is not None
                ]
                statuses = {
                    r["status"]
                    for r in rows
                    if r["precision"] == precision
                    and r["model"] == model
                    and r["phase"] == phase
                }
                if selected:
                    status = "valid"
                    median = statistics.median(selected)
                    low, high = min(selected), max(selected)
                else:
                    status = (
                        "no_picks"
                        if "no_picks" in statuses
                        else "unsupported"
                        if "unsupported" in statuses
                        else "invalid"
                    )
                    median = low = high = None
                summary.append(
                    {
                        "precision": precision,
                        "model": model,
                        "phase": phase,
                        "status": status,
                        "n": len(selected),
                        "median_direct_f1": median,
                        "min_direct_f1": low,
                        "max_direct_f1": high,
                    }
                )
    return summary


def catalog_summary(rows):
    summary = []
    for precision in ("BF16", "FP16"):
        for model in MODELS:
            for phase in PHASES:
                selected = [
                    r["catalog_delta_f1"]
                    for r in rows
                    if r["precision"] == precision
                    and r["model"] == model
                    and r["phase"] == phase
                    and r["catalog_delta_f1"] is not None
                ]
                statuses = {
                    r["status"]
                    for r in rows
                    if r["precision"] == precision
                    and r["model"] == model
                    and r["phase"] == phase
                }
                if selected:
                    record = {
                        "precision": precision,
                        "model": model,
                        "phase": phase,
                        "status": "valid",
                        "n": len(selected),
                        "median_catalog_delta_f1": statistics.median(selected),
                        "q25_catalog_delta_f1": percentile(selected, 25),
                        "q75_catalog_delta_f1": percentile(selected, 75),
                        "min_catalog_delta_f1": min(selected),
                        "max_catalog_delta_f1": max(selected),
                    }
                else:
                    status = (
                        "no_picks"
                        if "no_picks" in statuses
                        else "unsupported"
                        if "unsupported" in statuses
                        else "invalid"
                    )
                    record = {
                        "precision": precision,
                        "model": model,
                        "phase": phase,
                        "status": status,
                        "n": 0,
                        "median_catalog_delta_f1": None,
                        "q25_catalog_delta_f1": None,
                        "q75_catalog_delta_f1": None,
                        "min_catalog_delta_f1": None,
                        "max_catalog_delta_f1": None,
                    }
                summary.append(record)
    return summary


def worst_delta(rows, precision):
    selected = [
        r
        for r in rows
        if r["precision"] == precision and r["catalog_delta_f1"] is not None
    ]
    return max(selected, key=lambda row: abs(row["catalog_delta_f1"]))


def validate(rows, direct, catalog):
    bf16_valid = [
        r for r in rows if r["precision"] == "BF16" and r["direct_f1"] is not None
    ]
    fp16_valid = [
        r for r in rows if r["precision"] == "FP16" and r["direct_f1"] is not None
    ]
    assert len(bf16_valid) == 640
    assert len(fp16_valid) == 320

    bf16_min = min(bf16_valid, key=lambda row: row["direct_f1"])
    assert math.isclose(bf16_min["direct_f1"], 0.9959839357429718)
    assert bf16_min["model"] == "PhaseNetLight" and bf16_min["phase"] == "S"

    eqtnc_p = [
        r["direct_f1"]
        for r in bf16_valid
        if r["model"] == "EQT-NC" and r["phase"] == "P"
    ]
    eqtnc_s = [
        r["direct_f1"]
        for r in bf16_valid
        if r["model"] == "EQT-NC" and r["phase"] == "S"
    ]
    assert all(value == 1.0 for value in eqtnc_p)
    assert statistics.median(eqtnc_s) == 1.0 and min(eqtnc_s) < 1.0

    def median_direct(precision, model, phase):
        values = [
            r["direct_f1"]
            for r in rows
            if r["precision"] == precision
            and r["model"] == model
            and r["phase"] == phase
            and r["direct_f1"] is not None
        ]
        return statistics.median(values) if values else None

    assert math.isclose(
        median_direct("FP16", "PhaseNetLight", "P"), 0.6812300540935932
    )
    assert math.isclose(
        median_direct("FP16", "PhaseNetLight", "S"), 0.45789101203113936
    )
    assert math.isclose(
        median_direct("FP16", "EQCCT", "P"), 0.9665952427624153
    )
    assert median_direct("FP16", "EQCCT", "S") is None

    worst_bf16 = worst_delta(rows, "BF16")
    worst_fp16 = worst_delta(rows, "FP16")
    lines = [
        "Figure 3 validation",
        f"Processed configuration-phase rows: n={len(rows)}",
        f"Valid BF16 direct comparisons: n={len(bf16_valid)}",
        f"Valid FP16 direct comparisons: n={len(fp16_valid)}",
        f"Minimum BF16 direct F1: {bf16_min['direct_f1']:.12f} "
        f"({bf16_min['model']} {bf16_min['phase']}, "
        f"{bf16_min['n_stations']} stations, {bf16_min['device']}, "
        f"cores={bf16_min['n_cpus']}, batch={bf16_min['batch_size']})",
        "EQT-NC BF16 P: exactly 1.000 across all 64 cells",
        f"EQT-NC BF16 S: median={statistics.median(eqtnc_s):.12f}, "
        f"range={min(eqtnc_s):.12f}–{max(eqtnc_s):.12f}",
        "FP16 EQCCT S: no picks / N/A",
        "FP16 EQTransformer and EQT-NC: unsupported / N/A",
        f"Largest |BF16 catalog ΔF1|: {worst_bf16['catalog_delta_f1']:.12f} "
        f"({worst_bf16['model']} {worst_bf16['phase']}, "
        f"{worst_bf16['n_stations']} stations, {worst_bf16['device']}, "
        f"cores={worst_bf16['n_cpus']}, batch={worst_bf16['batch_size']})",
        f"Largest |FP16 catalog ΔF1|: {worst_fp16['catalog_delta_f1']:.12f} "
        f"({worst_fp16['model']} {worst_fp16['phase']}, "
        f"{worst_fp16['n_stations']} stations, {worst_fp16['device']}, "
        f"cores={worst_fp16['n_cpus']}, batch={worst_fp16['batch_size']})",
        f"Direct-summary rows: n={len(direct)}",
        f"Catalog-summary rows: n={len(catalog)}",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(rows, direct, catalog, validation):
    OUT.mkdir(parents=True, exist_ok=True)
    direct_rows = [
        {
            key: row[key]
            for key in (
                "precision",
                "method",
                "model",
                "n_stations",
                "device",
                "n_cpus",
                "n_threads",
                "batch_size",
                "packaging",
                "phase",
                "status",
                "direct_precision",
                "direct_recall",
                "direct_f1",
                "direct_matched",
                "direct_missing",
                "direct_additional",
                "direct_mean_dt_samples",
                "direct_p95_abs_dt_samples",
            )
        }
        for row in rows
    ]
    catalog_rows = [
        {
            key: row[key]
            for key in (
                "precision",
                "method",
                "model",
                "n_stations",
                "device",
                "n_cpus",
                "n_threads",
                "batch_size",
                "packaging",
                "phase",
                "status",
                "fp32_catalog_f1",
                "reduced_catalog_f1",
                "catalog_delta_f1",
            )
        }
        for row in rows
    ]
    paths = [
        OUT / "figure03_panelA_direct_agreement.csv",
        OUT / "figure03_panelB_catalog_delta.csv",
        OUT / "figure03_direct_summary.csv",
        OUT / "figure03_catalog_summary.csv",
    ]
    write_csv(paths[0], direct_rows)
    write_csv(paths[1], catalog_rows)
    write_csv(paths[2], direct)
    write_csv(paths[3], catalog)
    validation_path = OUT / "figure03_validation.txt"
    validation_path.write_text(validation)
    return paths + [validation_path]


def apply_model_tick_colors(ax):
    for label in ax.get_xticklabels():
        label.set_color(INK)


def panel_label(ax, label):
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)


def heatmap_panel(ax, summary, precision):
    matrix = np.full((2, len(MODELS)), np.nan)
    statuses = {}
    for phase_index, phase in enumerate(PHASES):
        for model_index, model in enumerate(MODELS):
            record = next(
                row
                for row in summary
                if row["precision"] == precision
                and row["model"] == model
                and row["phase"] == phase
            )
            statuses[(phase_index, model_index)] = record["status"]
            if record["median_direct_f1"] is not None:
                matrix[phase_index, model_index] = record["median_direct_f1"]

    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        vmin=0,
        vmax=1,
        cmap=HEATMAP,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([SHORT[model] for model in MODELS])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(PHASES)
    ax.set_title(f"{precision} vs FP32")
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    apply_model_tick_colors(ax)

    for phase_index in range(2):
        for model_index in range(len(MODELS)):
            value = matrix[phase_index, model_index]
            if np.isfinite(value):
                text = f"{value:.3f}"
                color = INK
            else:
                text = "N/A"
                color = MUTED
            ax.text(
                model_index,
                phase_index,
                text,
                ha="center",
                va="center",
                fontsize=8.0,
                color=color,
                fontweight="semibold" if not np.isfinite(value) else "normal",
            )
    return image


def catalog_panel(ax, rows, precision):
    model_x = np.arange(len(MODELS), dtype=float)
    phase_offsets = {"P": -0.12, "S": 0.12}

    all_valid = []
    for model_index, model in enumerate(MODELS):
        color = MODEL_COLORS[model]
        for phase in PHASES:
            values = [
                row["catalog_delta_f1"]
                for row in rows
                if row["precision"] == precision
                and row["model"] == model
                and row["phase"] == phase
                and row["catalog_delta_f1"] is not None
            ]
            x_center = model_index + phase_offsets[phase]
            if not values:
                continue
            all_valid.extend(values)
            jitter = np.linspace(-0.065, 0.065, len(values))
            ax.scatter(
                x_center + jitter,
                values,
                s=7,
                marker=PHASE_MARKERS[phase],
                color=color,
                alpha=0.30,
                edgecolor="none",
                rasterized=True,
                zorder=2,
            )
            median = statistics.median(values)
            q25, q75 = percentile(values, 25), percentile(values, 75)
            ax.errorbar(
                x_center,
                median,
                yerr=[[median - q25], [q75 - median]],
                fmt=PHASE_MARKERS[phase],
                markersize=4.5,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.45,
                color=color,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.2,
                zorder=4,
            )

    ax.axhline(0, color=INK, linewidth=0.75, linestyle="--")
    ax.set_xticks(model_x)
    ax.set_xticklabels([SHORT[model] for model in MODELS])
    apply_model_tick_colors(ax)
    ax.set_title(precision)
    ax.set_ylabel("Catalog ΔF1" if precision == "BF16" else "")
    if precision == "BF16":
        low, high = min(all_valid), max(all_valid)
        pad = (high - low) * 0.18
        ax.set_ylim(low - pad, high + pad)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
    else:
        ax.set_ylim(-0.55, 0.045)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.text(
            phase_offsets["S"],
            -0.065,
            "S N/A",
            ha="center",
            va="center",
            fontsize=6.5,
            color=MUTED,
        )
        for model_index in (3, 4):
            ax.text(
                model_index,
                -0.065,
                "N/A",
                ha="center",
                va="center",
                fontsize=6.5,
                color=MUTED,
            )
    clean_axis(ax)


def make_figure(rows, direct):
    fig = plt.figure(figsize=(7.25, 5.4))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.78, 1.35],
        hspace=0.50,
        wspace=0.30,
    )
    axes = np.asarray(
        [[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(2)]
    )

    heatmap_panel(axes[0, 0], direct, "BF16")
    heatmap_panel(axes[0, 1], direct, "FP16")
    panel_label(axes[0, 0], "A")

    catalog_panel(axes[1, 0], rows, "BF16")
    catalog_panel(axes[1, 1], rows, "FP16")
    panel_label(axes[1, 0], "B")

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=INK,
                markerfacecolor="white",
                markeredgecolor=INK,
                markersize=4.5,
                label="P",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="none",
                color=INK,
                markerfacecolor="white",
                markeredgecolor=INK,
                markersize=4.8,
                label="S",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.52, 0.012),
        ncol=2,
        columnspacing=1.25,
        handletextpad=0.4,
    )
    fig.subplots_adjust(top=0.965, bottom=0.135, left=0.10, right=0.985)
    return fig


def main() -> int:
    configure_style()
    rows = derive_rows()
    direct = direct_summary(rows)
    catalog = catalog_summary(rows)
    validation = validate(rows, direct, catalog)

    # Write all derived values and validation before exporting the figure.
    paths = write_artifacts(rows, direct, catalog, validation)
    print(validation, end="")
    for path in paths:
        print(path)

    figure = make_figure(rows, direct)
    figure_path = OUT / "figure03_phase_pick_agreement.png"
    revised_figure_path = OUT / "figure03_phase_pick_agreement_revised.png"
    final_figure_path = OUT / "figure03_phase_pick_agreement_final.png"
    scatter_figure_path = OUT / "figure03_phase_pick_agreement_scatter.png"
    figure.savefig(figure_path, dpi=600)
    figure.savefig(revised_figure_path, dpi=600)
    figure.savefig(final_figure_path, dpi=600)
    figure.savefig(scatter_figure_path, dpi=600)
    plt.close(figure)
    print(figure_path)
    print(revised_figure_path)
    print(final_figure_path)
    print(scatter_figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
