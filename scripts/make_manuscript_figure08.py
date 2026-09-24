#!/usr/bin/env python3
"""Create RAPID manuscript Figure 8: memory and cross-hardware transfer."""

from __future__ import annotations

import csv
import json
import math
import statistics
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


RAPID = Path(__file__).resolve().parents[1]
ORCH_ROOT = RAPID / "results" / "orch_annotate" / "stead_2026-08-14"
NATIVE_ROOT = RAPID / "results" / "annotate_precision" / "stead_iso_2026-08-13"
TRANSFER_ROOT = (
    RAPID
    / "handoff"
    / "locked_recipe_transfer_BEGE-TEXA75535L"
    / "raw_results"
)
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

K_CPU = [1, 2, 4, 5, 10, 20]
K_GPU = [1, 2, 4]
K_COMMON = [5, 10, 20]
NATIVE_CPUS = [5, 10, 15, 20]

INK = "#17212B"
MUTED = "#65727E"
GRID = "#D8E0E6"
WORKSTATION = "#3A3A3A"
LAPTOP = "#7A5195"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "axes.titleweight": "semibold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def orch_path(
    composition: str,
    model: str,
    device: str,
    k_ma: int,
    k_ripper: int,
    arrival: str = "playback",
    fill: str = "static",
) -> Path:
    return (
        ORCH_ROOT
        / composition
        / "annotate_bf16"
        / "stead"
        / "580st"
        / model
        / device
        / f"kma{k_ma}_krp{k_ripper}"
        / "sg"
        / arrival
        / fill
        / "bs512"
        / "orch_ann"
        / "result.json"
    )


def native_path(model: str, device: str, n_cpus: int) -> Path:
    return (
        NATIVE_ROOT
        / "annotate_bf16"
        / "stead"
        / "580st"
        / model
        / device
        / f"cpus{n_cpus}"
        / f"thr{n_cpus}"
        / "bs512"
        / "ann_prec"
        / "result.json"
    )


def transfer_path(k: int, arrival: str, fill: str) -> Path:
    return (
        TRANSFER_ROOT
        / "ma"
        / "annotate_bf16"
        / "stead"
        / "580st"
        / "EQCCT"
        / "cpu"
        / f"kma{k}_krp0"
        / "sg"
        / arrival
        / fill
        / "bs512"
        / "xfer"
        / "result.json"
    )


def repeat_summary(record: dict, key: str) -> tuple[float, float, list[float]]:
    repeats = (record.get("timing") or {}).get("repeats") or []
    if len(repeats) != 5 or not all(item.get("success") is True for item in repeats):
        raise RuntimeError("Expected five successful repeats")
    values = [float(item[key]) for item in repeats]
    return statistics.fmean(values), statistics.stdev(values), values


def memory_summary(record: dict, key: str) -> tuple[float, float, list[float]]:
    mean, std, values = repeat_summary(record, key)
    summary_key = f"{key}_mean"
    stored = (record.get("memory") or {}).get(summary_key)
    if stored is None or not math.isclose(
        mean, float(stored), rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError(f"Memory mean disagrees for {summary_key}")
    return mean / 1000.0, std / 1000.0, [value / 1000.0 for value in values]


@lru_cache(maxsize=None)
def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing result: {path}")
    return json.loads(path.read_text())


def fastest_native(model: str, device: str) -> tuple[int, dict]:
    candidates = []
    for n_cpus in NATIVE_CPUS:
        path = native_path(model, device, n_cpus)
        record = load_json(path)
        mean, _, _ = repeat_summary(record, "inference_s")
        candidates.append((mean, n_cpus, record))
    _, n_cpus, record = min(candidates, key=lambda item: item[0])
    return n_cpus, record


def pooled_p95_from_result(path: Path) -> float:
    result = load_json(path)
    directory = path.parent
    network_size = int((result.get("meta") or {}).get("n_stations"))
    latencies = []
    for repeat_index in range(5):
        repeat_path = directory / "repeats" / f"repeat_{repeat_index}.json"
        repeat = load_json(repeat_path)
        if repeat.get("success") is not True:
            raise RuntimeError(f"Unsuccessful repeat: {repeat_path}")
        events = repeat.get("station_events") or []
        if len(events) != network_size:
            raise RuntimeError(f"Unexpected station count in {repeat_path}")
        latencies.extend(
            float(event["returned_s"]) - float(event["ready_s"])
            for event in events
        )
    p95 = float(np.percentile(np.asarray(latencies, dtype=float), 95))
    stored = (
        (((result.get("latency") or {}).get("pooled_across_repeats") or {})
        .get("e2e_finish_minus_ready")
        or {})
        .get("p95")
    )
    if stored is None or not math.isclose(
        p95, float(stored), rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(f"Pooled p95 disagrees in {path}")
    return p95


def derive_rows() -> list[dict]:
    rows = []
    for device, panel, memory_key, ks in (
        ("cpu", "A", "peak_pss_mb", K_CPU),
        ("gpu", "B", "peak_vram_mb", K_GPU),
    ):
        for model in MODELS:
            native_cpus, native_record = fastest_native(model, device)
            native_mean, native_std, native_values = memory_summary(
                native_record, memory_key
            )
            rows.append(
                {
                    "panel": panel,
                    "row_type": "memory",
                    "host": "Workstation",
                    "metric": "PSS" if device == "cpu" else "VRAM",
                    "model": model,
                    "device": device,
                    "condition": "Native",
                    "k": 0,
                    "mean_gb": native_mean,
                    "std_gb": native_std,
                    "repeat_values_gb": ";".join(
                        f"{value:.12f}" for value in native_values
                    ),
                    "native_n_cpus": native_cpus,
                    "raw_value_s": "",
                    "normalized_value": "",
                    "source_file": str(
                        native_path(model, device, native_cpus).relative_to(RAPID)
                    ),
                }
            )
            for k in ks:
                path = orch_path("ma", model, device, k, 0)
                record = load_json(path)
                mean, std, values = memory_summary(record, memory_key)
                rows.append(
                    {
                        "panel": panel,
                        "row_type": "memory",
                        "host": "Workstation",
                        "metric": "PSS" if device == "cpu" else "VRAM",
                        "model": model,
                        "device": device,
                        "condition": "Model-Actor",
                        "k": k,
                        "mean_gb": mean,
                        "std_gb": std,
                        "repeat_values_gb": ";".join(
                            f"{value:.12f}" for value in values
                        ),
                        "native_n_cpus": "",
                        "raw_value_s": "",
                        "normalized_value": "",
                        "source_file": str(path.relative_to(RAPID)),
                    }
                )

    transfer_values = {}
    for host in ("Workstation", "Laptop"):
        for metric in ("All-ready", "Streaming"):
            values = []
            sources = []
            for k in K_COMMON:
                if host == "Workstation":
                    if metric == "All-ready":
                        path = orch_path("ma", "EQCCT", "cpu", k, 0)
                        record = load_json(path)
                        raw, _, _ = repeat_summary(record, "makespan_s")
                    else:
                        path = orch_path(
                            "ma",
                            "EQCCT",
                            "cpu",
                            k,
                            0,
                            arrival="staggered",
                            fill="eager",
                        )
                        raw = pooled_p95_from_result(path)
                else:
                    if metric == "All-ready":
                        path = transfer_path(k, "playback", "static")
                        record = load_json(path)
                        raw, _, _ = repeat_summary(record, "makespan_s")
                    else:
                        path = transfer_path(k, "staggered", "eager")
                        raw = pooled_p95_from_result(path)
                values.append(raw)
                sources.append(path)
            minimum = min(values)
            transfer_values[(host, metric)] = values
            for k, raw, path in zip(K_COMMON, values, sources):
                rows.append(
                    {
                        "panel": "C",
                        "row_type": "transfer",
                        "host": host,
                        "metric": metric,
                        "model": "EQCCT",
                        "device": "cpu",
                        "condition": metric,
                        "k": k,
                        "mean_gb": "",
                        "std_gb": "",
                        "repeat_values_gb": "",
                        "native_n_cpus": "",
                        "raw_value_s": raw,
                        "normalized_value": raw / minimum,
                        "source_file": str(path.relative_to(RAPID)),
                    }
                )
    return rows


def validate_ripper_pss() -> list[dict]:
    rows = []
    for device, ks in (("cpu", [1, 2, 4, 5]), ("gpu", [1, 2])):
        for model in MODELS:
            for k in ks:
                ma = load_json(orch_path("ma", model, device, k, 0))
                ripper = load_json(orch_path("ripper", model, device, 0, k))
                ma_pss = float((ma.get("memory") or {})["peak_pss_mb_mean"])
                ripper_pss = float(
                    (ripper.get("memory") or {})["peak_pss_mb_mean"]
                )
                rows.append(
                    {
                        "device": device,
                        "model": model,
                        "k": k,
                        "ratio": ripper_pss / ma_pss,
                    }
                )
    return rows


def validate(rows: list[dict]) -> str:
    lines = ["Figure 8 validation", ""]
    for model in MODELS:
        cpu_rows = [
            row
            for row in rows
            if row["panel"] == "A" and row["model"] == model
        ]
        native = next(row for row in cpu_rows if row["condition"] == "Native")
        actors = sorted(
            (row for row in cpu_rows if row["condition"] == "Model-Actor"),
            key=lambda row: int(row["k"]),
        )
        k1 = next(row for row in actors if row["k"] == 1)
        k20 = next(row for row in actors if row["k"] == 20)
        lines.append(
            f"Panel A {model}: native={float(native['mean_gb']):.6f} GB; "
            + ", ".join(
                f"K={row['k']} {float(row['mean_gb']):.6f} GB"
                for row in actors
            )
            + f"; K20/K1={float(k20['mean_gb']) / float(k1['mean_gb']):.6f}"
        )
    k1_values = [
        float(row["mean_gb"])
        for row in rows
        if row["panel"] == "A" and row["k"] == 1
    ]
    k20_values = [
        float(row["mean_gb"])
        for row in rows
        if row["panel"] == "A" and row["k"] == 20
    ]
    if not (5.00 <= min(k1_values) and max(k1_values) <= 5.68):
        raise RuntimeError("CPU K=1 PSS range did not reproduce")
    if not (12.7 <= min(k20_values) and max(k20_values) <= 14.1):
        raise RuntimeError("CPU K=20 PSS range did not reproduce")

    ripper = validate_ripper_pss()
    ratios = [row["ratio"] for row in ripper]
    outlier = max(ripper, key=lambda row: abs(row["ratio"] - 1))
    lines.extend(
        [
            "",
            f"Matched Ripper/Model-Actor PSS comparisons: n={len(ratios)}",
            f"Median Ripper/Model-Actor PSS ratio: "
            f"{statistics.median(ratios):.12f}",
            f"Ratio range: {min(ratios):.12f}–{max(ratios):.12f}",
            f"Largest absolute difference: "
            f"{abs(outlier['ratio'] - 1) * 100:.3f}% "
            f"({outlier['model']}, {outlier['device']}, K={outlier['k']})",
            "The requested ~6% bound is not universal; the PhaseNet CPU K=5 "
            "Ripper value is 9.67% higher.",
            "",
        ]
    )

    for model in MODELS:
        gpu_rows = [
            row
            for row in rows
            if row["panel"] == "B" and row["model"] == model
        ]
        native = next(row for row in gpu_rows if row["condition"] == "Native")
        actors = sorted(
            (row for row in gpu_rows if row["condition"] == "Model-Actor"),
            key=lambda row: int(row["k"]),
        )
        lines.append(
            f"Panel B {model}: native={float(native['mean_gb']):.6f} GB; "
            + ", ".join(
                f"K={row['k']} {float(row['mean_gb']):.6f} GB"
                for row in actors
            )
        )
    gpu_k4 = {
        row["model"]: float(row["mean_gb"])
        for row in rows
        if row["panel"] == "B" and row["k"] == 4
    }
    if not math.isclose(gpu_k4["EQCCT"], 3.9762, abs_tol=0.01):
        raise RuntimeError("EQCCT K=4 VRAM did not reproduce")
    other_gpu = [gpu_k4[model] for model in MODELS if model != "EQCCT"]
    if not (2.71 <= min(other_gpu) and max(other_gpu) <= 2.91):
        raise RuntimeError("Non-EQCCT K=4 VRAM range did not reproduce")

    lines.append("")
    for host in ("Workstation", "Laptop"):
        for metric in ("All-ready", "Streaming"):
            transfer = sorted(
                (
                    row
                    for row in rows
                    if row["panel"] == "C"
                    and row["host"] == host
                    and row["metric"] == metric
                ),
                key=lambda row: int(row["k"]),
            )
            best = min(transfer, key=lambda row: float(row["raw_value_s"]))
            lines.append(
                f"Panel C {host} {metric}: "
                + ", ".join(
                    f"K={row['k']} raw={float(row['raw_value_s']):.9f} s "
                    f"normalized={float(row['normalized_value']):.9f}"
                    for row in transfer
                )
                + f"; minimum K={best['k']}"
            )
            expected = 20 if host == "Workstation" else 10
            if best["k"] != expected:
                raise RuntimeError(
                    f"Unexpected {host} {metric} minimum K={best['k']}"
                )
    lines.extend(
        [
            "",
            "Laptop GPU VRAM is not plotted because Ray-worker VRAM was not "
            "directly measured on the second host.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(rows: list[dict], validation: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "figure08_memory_transfer_derived.csv"
    fields = [
        "panel",
        "row_type",
        "host",
        "metric",
        "model",
        "device",
        "condition",
        "k",
        "mean_gb",
        "std_gb",
        "repeat_values_gb",
        "native_n_cpus",
        "raw_value_s",
        "normalized_value",
        "source_file",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    validation_path = OUT / "figure08_memory_transfer_validation.txt"
    validation_path.write_text(validation)
    return csv_path, validation_path


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter: str) -> None:
    ax.text(
        0.0,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )


def memory_axis(ax, rows: list[dict], panel: str, device: str, letter: str) -> None:
    relevant = [row for row in rows if row["panel"] == panel]
    ks = K_CPU if device == "cpu" else K_GPU
    categories = ["Native"] + [str(k) for k in ks]
    positions = np.arange(len(categories))
    for model in MODELS:
        model_rows = [row for row in relevant if row["model"] == model]
        native = next(row for row in model_rows if row["condition"] == "Native")
        actor_rows = sorted(
            (row for row in model_rows if row["condition"] == "Model-Actor"),
            key=lambda row: int(row["k"]),
        )
        ax.scatter(
            0,
            float(native["mean_gb"]),
            s=28,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            zorder=4,
        )
        ax.plot(
            positions[1:],
            [float(row["mean_gb"]) for row in actor_rows],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=4.3,
            linewidth=1.15,
            zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Inference instances, K")
    ax.set_ylabel(
        "Peak process-tree RAM (GB)"
        if device == "cpu"
        else "Peak process-tree VRAM (GB)"
    )
    ax.set_ylim(0, 15.2 if device == "cpu" else 4.3)
    ax.set_title(device.upper(), pad=5)
    panel_label(ax, letter)
    clean_axis(ax)


def transfer_axis(
    ax, rows: list[dict], metric: str, subtitle: str, show_ylabel: bool
) -> None:
    for host, color, marker in (
        ("Workstation", WORKSTATION, "o"),
        ("Laptop", LAPTOP, "s"),
    ):
        host_rows = sorted(
            (
                row
                for row in rows
                if row["panel"] == "C"
                and row["host"] == host
                and row["metric"] == metric
            ),
            key=lambda row: int(row["k"]),
        )
        ax.plot(
            [int(row["k"]) for row in host_rows],
            [float(row["normalized_value"]) for row in host_rows],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.25,
            label=host,
        )
    ax.axhline(1, color=INK, linewidth=0.75, linestyle="--")
    ax.set_xticks(K_COMMON)
    ax.set_xlabel("Inference instances, K")
    if show_ylabel:
        ax.set_ylabel("Runtime relative to host minimum")
    ax.set_ylim(0.9, 3.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_title(subtitle, pad=5)
    clean_axis(ax)


def make_figure(rows: list[dict]) -> plt.Figure:
    figure = plt.figure(figsize=(9.0, 7.0))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        top=0.95,
        bottom=0.13,
        wspace=0.24,
        hspace=0.38,
        height_ratios=[1.0, 0.95],
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    memory_axis(ax_a, rows, "A", "cpu", "A")
    memory_axis(ax_b, rows, "B", "gpu", "B")

    panel_c = grid[1, :].subgridspec(
        2, 2, height_ratios=[0.10, 1], hspace=0.05, wspace=0.18
    )
    header = figure.add_subplot(panel_c[0, :])
    header.axis("off")
    header.text(
        0.0,
        0.65,
        "C",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    ax_c1 = figure.add_subplot(panel_c[1, 0])
    ax_c2 = figure.add_subplot(panel_c[1, 1])
    transfer_axis(
        ax_c1,
        rows,
        "All-ready",
        "All-ready simulation",
        True,
    )
    transfer_axis(
        ax_c2,
        rows,
        "Streaming",
        "Real-time streaming simulation",
        False,
    )
    ax_c1.legend(loc="upper right")

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS[model],
                linestyle="-",
                color=MODEL_COLORS[model],
                markerfacecolor=MODEL_COLORS[model],
                markersize=4.8,
                linewidth=1.1,
                label=SHORT[model],
            )
            for model in MODELS
        ],
        loc="lower center",
        bbox_to_anchor=(0.52, 0.025),
        ncol=5,
        columnspacing=1.4,
        handletextpad=0.45,
    )
    return figure


def main() -> int:
    configure_style()
    rows = derive_rows()
    validation = validate(rows)
    csv_path, validation_path = write_artifacts(rows, validation)
    print(validation, end="")

    figure = make_figure(rows)
    base = OUT / "figure08_memory_transfer"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    relabeled_png_path = OUT / "figure08_memory_transfer_relabeled.png"
    host_colors_png_path = OUT / "figure08_memory_transfer_host_colors.png"
    figure.savefig(png_path, dpi=600)
    figure.savefig(pdf_path)
    figure.savefig(svg_path)
    figure.savefig(relabeled_png_path, dpi=600)
    figure.savefig(host_colors_png_path, dpi=600)
    plt.close(figure)

    print(csv_path)
    print(validation_path)
    print(png_path)
    print(pdf_path)
    print(svg_path)
    print(relabeled_png_path)
    print(host_colors_png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
