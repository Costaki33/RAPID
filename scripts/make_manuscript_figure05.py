#!/usr/bin/env python3
"""Create RAPID manuscript Figure 5: all-ready orchestration performance.

The script validates five-repeat means directly from the raw orchestration
records, writes every plotted value to CSV, and exports a 600-dpi PNG.
"""

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
from matplotlib.ticker import FuncFormatter, NullFormatter


RAPID = Path(__file__).resolve().parents[1]
ROOT = RAPID / "results" / "orch_annotate" / "stead_2026-08-14"
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

NETWORKS = [250, 580]
METHODS = ["annotate_fp32", "annotate_bf16"]
PRECISION = {"annotate_fp32": "FP32", "annotate_bf16": "BF16"}
K_MA = {"cpu": [1, 2, 4, 5, 10, 20], "gpu": [1, 2, 4]}
K_SHARED = {"cpu": [1, 2, 4, 5], "gpu": [1, 2]}

INK = "#17212B"
MUTED = "#65727E"
GRID = "#D8E0E6"
MA_COLOR = "#2A8C82"
RIPPER_COLOR = "#D97745"


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
        }
    )


def result_path(
    strategy: str,
    method: str,
    network_size: int,
    model: str,
    device: str,
    k: int,
    packaging: str,
) -> Path:
    composition = "ma" if strategy == "Model-Actor" else "ripper"
    k_ma = k if strategy == "Model-Actor" else 0
    k_ripper = 0 if strategy == "Model-Actor" else k
    return (
        ROOT
        / composition
        / method
        / "stead"
        / f"{network_size}st"
        / model
        / device
        / f"kma{k_ma}_krp{k_ripper}"
        / packaging.lower()
        / "playback"
        / "static"
        / "bs512"
        / "orch_ann"
        / "result.json"
    )


@lru_cache(maxsize=None)
def load_cell(
    strategy: str,
    method: str,
    network_size: int,
    model: str,
    device: str,
    k: int,
    packaging: str,
) -> dict:
    path = result_path(
        strategy, method, network_size, model, device, k, packaging
    )
    if not path.is_file():
        return {
            "status": "missing",
            "makespan_s_mean": None,
            "source_file": str(path.relative_to(RAPID)),
        }

    record = json.loads(path.read_text())
    meta = record.get("meta") or {}
    expected = {
        "n_stations": network_size,
        "model": model,
        "device": device,
        "packaging": packaging.lower(),
        "arrival": "playback",
        "fill": "static",
        "repeats": 5,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"Metadata mismatch in {path}: {key}={meta.get(key)!r}, "
                f"expected {value!r}"
            )

    repeats = (record.get("timing") or {}).get("repeats") or []
    if len(repeats) != 5:
        raise RuntimeError(f"Expected five repeats in {path}, found {len(repeats)}")
    values = []
    for repeat in repeats:
        if repeat.get("success") is not True:
            raise RuntimeError(f"Unsuccessful repeat in {path}")
        value = repeat.get("makespan_s")
        if value is None:
            raise RuntimeError(f"Missing makespan_s in repeat from {path}")
        values.append(float(value))

    mean = statistics.fmean(values)
    stored_mean = (record.get("orch") or {}).get("makespan_s_mean")
    if stored_mean is None or not math.isclose(
        mean, float(stored_mean), rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(
            f"Five-repeat mean disagrees with summary in {path}: "
            f"{mean} vs {stored_mean}"
        )

    return {
        "status": "valid",
        "makespan_s_mean": mean,
        "source_file": str(path.relative_to(RAPID)),
    }


def panel_a_rows() -> list[dict]:
    rows = []
    for device in ("cpu", "gpu"):
        for method in METHODS:
            for network_size in NETWORKS:
                for model in MODELS:
                    for k in K_MA[device]:
                        sg = load_cell(
                            "Model-Actor",
                            method,
                            network_size,
                            model,
                            device,
                            k,
                            "SG",
                        )
                        s1 = load_cell(
                            "Model-Actor",
                            method,
                            network_size,
                            model,
                            device,
                            k,
                            "S1",
                        )
                        if sg["status"] != "valid" or s1["status"] != "valid":
                            raise RuntimeError(
                                "Missing Model-Actor SG/S1 matched cell: "
                                f"{method}, {network_size}, {model}, "
                                f"{device}, K={k}"
                            )
                        rows.append(
                            {
                            "panel": "A",
                            "comparison": "S1_to_SG",
                            "precision": PRECISION[method],
                            "model": model,
                            "device": device,
                            "network_size": network_size,
                            "k": k,
                            "strategy": "Model-Actor",
                            "packaging": "SG_vs_S1",
                            "status": "valid",
                            "makespan_s_mean": "",
                            "numerator_s": s1["makespan_s_mean"],
                            "denominator_s": sg["makespan_s_mean"],
                            "speedup": s1["makespan_s_mean"]
                            / sg["makespan_s_mean"],
                            "source_file": "",
                            "numerator_source": s1["source_file"],
                            "denominator_source": sg["source_file"],
                            }
                        )
    return rows


def panel_b_rows() -> list[dict]:
    rows = []
    for device in ("cpu", "gpu"):
        for method in METHODS:
            for network_size in NETWORKS:
                for model in MODELS:
                    for k in K_SHARED[device]:
                        model_actor = load_cell(
                            "Model-Actor",
                            method,
                            network_size,
                            model,
                            device,
                            k,
                            "SG",
                        )
                        ripper = load_cell(
                            "Ripper",
                            method,
                            network_size,
                            model,
                            device,
                            k,
                            "SG",
                        )
                        if (
                            model_actor["status"] != "valid"
                            or ripper["status"] != "valid"
                        ):
                            raise RuntimeError(
                                "Missing matched SG persistence cell: "
                                f"{method}, {network_size}, {model}, "
                                f"{device}, K={k}"
                            )
                        rows.append(
                            {
                            "panel": "B",
                            "comparison": "Ripper_to_ModelActor",
                            "precision": PRECISION[method],
                            "model": model,
                            "device": device,
                            "network_size": network_size,
                            "k": k,
                            "strategy": "Ripper_vs_Model-Actor",
                            "packaging": "SG",
                            "status": "valid",
                            "makespan_s_mean": "",
                            "numerator_s": ripper["makespan_s_mean"],
                            "denominator_s": model_actor["makespan_s_mean"],
                            "speedup": ripper["makespan_s_mean"]
                            / model_actor["makespan_s_mean"],
                            "source_file": "",
                            "numerator_source": ripper["source_file"],
                            "denominator_source": model_actor["source_file"],
                            }
                        )
    return rows


def interaction_rows() -> list[dict]:
    rows = []
    for panel, device, k in (("C", "cpu", 5), ("D", "gpu", 2)):
        for model in MODELS:
            for strategy in ("Model-Actor", "Ripper"):
                for packaging in ("SG", "S1"):
                    cell = load_cell(
                        strategy,
                        "annotate_bf16",
                        580,
                        model,
                        device,
                        k,
                        packaging,
                    )
                    rows.append(
                        {
                            "panel": panel,
                            "comparison": "combined_strategies",
                            "precision": "BF16",
                            "model": model,
                            "device": device,
                            "network_size": 580,
                            "k": k,
                            "strategy": strategy,
                            "packaging": packaging,
                            "status": cell["status"],
                            "makespan_s_mean": (
                                cell["makespan_s_mean"]
                                if cell["makespan_s_mean"] is not None
                                else ""
                            ),
                            "numerator_s": "",
                            "denominator_s": "",
                            "speedup": "",
                            "source_file": cell["source_file"],
                            "numerator_source": "",
                            "denominator_source": "",
                        }
                    )
    return rows


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_speedups(rows: list[dict], panel: str) -> list[str]:
    relevant = [row for row in rows if row["panel"] == panel]
    values = [float(row["speedup"]) for row in relevant]
    lines = [
        f"Panel {panel} matched configurations: n={len(values)}",
        f"Panel {panel} wins (>1×): n={sum(value > 1 for value in values)}",
        f"Panel {panel} median speedup overall: {statistics.median(values):.12f}",
    ]
    for device in ("cpu", "gpu"):
        device_values = [
            float(row["speedup"]) for row in relevant if row["device"] == device
        ]
        lines.extend(
            [
                f"Panel {panel} {device.upper()} comparisons: n={len(device_values)}",
                f"Panel {panel} {device.upper()} wins: "
                f"n={sum(value > 1 for value in device_values)}",
                f"Panel {panel} median speedup {device.upper()}: "
                f"{statistics.median(device_values):.12f}",
            ]
        )
    largest = max(relevant, key=lambda row: float(row["speedup"]))
    lines.append(
        f"Panel {panel} largest speedup: {float(largest['speedup']):.12f} "
        f"({largest['precision']}, {largest['model']}, "
        f"{largest['network_size']} stations, "
        f"{largest['device']}, K={largest['k']})"
    )
    return lines


def validate(rows: list[dict]) -> str:
    panel_a = [row for row in rows if row["panel"] == "A"]
    panel_b = [row for row in rows if row["panel"] == "B"]
    if len(panel_a) != 180 or sum(float(row["speedup"]) > 1 for row in panel_a) != 166:
        raise RuntimeError("Panel A did not reproduce 166 wins in 180 comparisons")
    if len(panel_b) != 120 or sum(float(row["speedup"]) > 1 for row in panel_b) != 116:
        raise RuntimeError("Panel B did not reproduce 116 wins in 120 comparisons")

    lines = [
        "Figure 5 validation",
        "All available cells use five successful isolated repeats.",
        "Panels A/B include matched FP32 and BF16 orchestration cells.",
        "Panels C/D show Annotate-BF16 means.",
        "",
        *summarize_speedups(rows, "A"),
        "",
        *summarize_speedups(rows, "B"),
        "",
        "Panels C/D Annotate-BF16 mean makespan values (580 stations):",
    ]
    for panel in ("C", "D"):
        panel_rows = [row for row in rows if row["panel"] == panel]
        device = panel_rows[0]["device"].upper()
        k = panel_rows[0]["k"]
        lines.append(f"Panel {panel}: {device}, K={k}")
        for model in MODELS:
            for strategy in ("Model-Actor", "Ripper"):
                for packaging in ("SG", "S1"):
                    row = next(
                        item
                        for item in panel_rows
                        if item["model"] == model
                        and item["strategy"] == strategy
                        and item["packaging"] == packaging
                    )
                    value = (
                        f"{float(row['makespan_s_mean']):.12f} s"
                        if row["status"] == "valid"
                        else "N/A (not measured)"
                    )
                    lines.append(
                        f"  {model} | {strategy} | {packaging}: {value}"
                    )
    gpu_ripper_s1 = [
        row
        for row in rows
        if row["panel"] == "D"
        and row["strategy"] == "Ripper"
        and row["packaging"] == "S1"
        and row["status"] == "valid"
    ]
    lines.extend(
        [
            "",
            "Available 580-station GPU Ripper-S1 values:",
            *[
                f"  {row['model']}: {float(row['makespan_s_mean']):.6f} s "
                f"({float(row['makespan_s_mean']) / 60:.3f} min)"
                for row in gpu_ripper_s1
            ],
            "Ripper-S1 was not measured for PNL, EQT, or EQT-NC and is plotted "
            "as N/A.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(rows: list[dict], validation: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "figure05_all_ready_derived.csv"
    fields = [
        "panel",
        "comparison",
        "precision",
        "model",
        "device",
        "network_size",
        "k",
        "strategy",
        "packaging",
        "status",
        "makespan_s_mean",
        "numerator_s",
        "denominator_s",
        "speedup",
        "source_file",
        "numerator_source",
        "denominator_source",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    validation_path = OUT / "figure05_all_ready_validation.txt"
    validation_path.write_text(validation)
    return csv_path, validation_path


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def plain_log_axis(ax, ticks: list[float]) -> None:
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def header_axis(fig, spec, letter: str, title: str):
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.text(
        0.0,
        0.70,
        letter,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    if title:
        ax.text(
            0.065,
            0.70,
            title,
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="semibold",
            color=INK,
        )
    return ax


def effect_axis(
    ax,
    rows: list[dict],
    panel: str,
    device: str,
    ylabel: str,
    ylim: tuple[float, float],
    ticks: list[float],
) -> None:
    relevant = [
        row for row in rows if row["panel"] == panel and row["device"] == device
    ]
    rng = np.random.default_rng(42 if device == "cpu" else 43)
    for model_index, model in enumerate(MODELS):
        values = [
            float(row["speedup"]) for row in relevant if row["model"] == model
        ]
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(
            model_index + jitter,
            values,
            s=11,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            alpha=0.30,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        q25 = percentile(values, 25)
        median = statistics.median(values)
        q75 = percentile(values, 75)
        ax.errorbar(
            model_index,
            median,
            yerr=[[median - q25], [q75 - median]],
            fmt=MODEL_MARKERS[model],
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=MODEL_COLORS[model],
            markeredgewidth=1.0,
            ecolor=MODEL_COLORS[model],
            elinewidth=1.15,
            capsize=2.2,
            zorder=4,
        )

    ax.axhline(1, color=INK, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    plain_log_axis(ax, ticks)
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([SHORT[model] for model in MODELS])
    ax.set_title(device.upper(), pad=4)
    if ylabel:
        ax.set_ylabel(ylabel)
    clean_axis(ax)


def interaction_axis(
    ax,
    rows: list[dict],
    panel: str,
    device: str,
    k: int,
    letter: str,
    ylim: tuple[float, float],
    ticks: list[float],
) -> None:
    relevant = [row for row in rows if row["panel"] == panel]
    styles = [
        ("Model-Actor", "SG", -0.27, "o", MA_COLOR, True),
        ("Model-Actor", "S1", -0.09, "o", MA_COLOR, False),
        ("Ripper", "SG", 0.09, "s", RIPPER_COLOR, True),
        ("Ripper", "S1", 0.27, "s", RIPPER_COLOR, False),
    ]
    for strategy, packaging, offset, marker, color, filled in styles:
        for model_index, model in enumerate(MODELS):
            row = next(
                item
                for item in relevant
                if item["model"] == model
                and item["strategy"] == strategy
                and item["packaging"] == packaging
            )
            x = model_index + offset
            if row["status"] != "valid":
                ax.text(
                    x,
                    0.93,
                    "N/A",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=MUTED,
                )
                continue
            ax.plot(
                x,
                float(row["makespan_s_mean"]),
                marker=marker,
                markersize=6.0,
                linestyle="none",
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                zorder=4,
            )

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    plain_log_axis(ax, ticks)
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([SHORT[model] for model in MODELS])
    ax.set_ylabel("Network makespan (s)")
    ax.set_title(letter, loc="left", pad=7)
    clean_axis(ax)


def make_figure(rows: list[dict]) -> plt.Figure:
    figure = plt.figure(figsize=(10.0, 7.0))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.985,
        top=0.965,
        bottom=0.12,
        wspace=0.22,
        hspace=0.34,
        height_ratios=[1.0, 1.03],
    )

    top_a = grid[0, 0].subgridspec(
        2, 2, height_ratios=[0.12, 1.0], hspace=0.05, wspace=0.22
    )
    header_axis(figure, top_a[0, :], "A", "")
    ax_a_cpu = figure.add_subplot(top_a[1, 0])
    ax_a_gpu = figure.add_subplot(top_a[1, 1])
    effect_axis(
        ax_a_cpu,
        rows,
        "A",
        "cpu",
        "SG speedup over S1 (×)",
        (0.35, 30),
        [0.5, 1, 2, 5, 10, 20],
    )
    effect_axis(
        ax_a_gpu,
        rows,
        "A",
        "gpu",
        "",
        (0.35, 30),
        [0.5, 1, 2, 5, 10, 20],
    )

    top_b = grid[0, 1].subgridspec(
        2, 2, height_ratios=[0.12, 1.0], hspace=0.05, wspace=0.22
    )
    header_axis(figure, top_b[0, :], "B", "")
    ax_b_cpu = figure.add_subplot(top_b[1, 0])
    ax_b_gpu = figure.add_subplot(top_b[1, 1])
    effect_axis(
        ax_b_cpu,
        rows,
        "B",
        "cpu",
        "Model-Actor speedup over Ripper (×)",
        (0.8, 25),
        [1, 2, 5, 10, 20],
    )
    effect_axis(
        ax_b_gpu,
        rows,
        "B",
        "gpu",
        "",
        (0.8, 25),
        [1, 2, 5, 10, 20],
    )

    ax_c = figure.add_subplot(grid[1, 0])
    interaction_axis(
        ax_c,
        rows,
        "C",
        "cpu",
        5,
        "C",
        (0.18, 200),
        [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
    )
    ax_d = figure.add_subplot(grid[1, 1])
    interaction_axis(
        ax_d,
        rows,
        "D",
        "gpu",
        2,
        "D",
        (0.18, 1800),
        [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=MA_COLOR,
            markeredgecolor=MA_COLOR,
            markersize=5.5,
            label="Model-Actor",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=RIPPER_COLOR,
            markeredgecolor=RIPPER_COLOR,
            markersize=5.5,
            label="Ripper",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=INK,
            markeredgecolor=INK,
            markersize=5.5,
            label="SG (filled)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markersize=5.5,
            label="S1 (open)",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.025),
        ncol=4,
        columnspacing=1.5,
        handletextpad=0.45,
    )
    return figure


def main() -> int:
    configure_style()
    rows = panel_a_rows() + panel_b_rows() + interaction_rows()
    validation = validate(rows)
    csv_path, validation_path = write_artifacts(rows, validation)
    print(validation, end="")

    figure = make_figure(rows)
    figure_path = OUT / "figure05_all_ready_orchestration.png"
    corrected_figure_path = OUT / "figure05_all_ready_orchestration_corrected.png"
    retitled_figure_path = OUT / "figure05_all_ready_orchestration_retitled.png"
    letters_figure_path = OUT / "figure05_all_ready_orchestration_letters_only.png"
    figure.savefig(figure_path, dpi=600)
    figure.savefig(corrected_figure_path, dpi=600)
    figure.savefig(retitled_figure_path, dpi=600)
    figure.savefig(letters_figure_path, dpi=600)
    plt.close(figure)

    print(csv_path)
    print(validation_path)
    print(figure_path)
    print(corrected_figure_path)
    print(retitled_figure_path)
    print(letters_figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
