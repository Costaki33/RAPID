#!/usr/bin/env python3
"""Create RAPID manuscript Figure 7: scaling and native comparison."""

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
ORCH_ROOT = RAPID / "results" / "orch_annotate" / "stead_2026-08-14"
NATIVE_ROOT = RAPID / "results" / "annotate_precision" / "stead_iso_2026-08-13"
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

K_VALUES = {"cpu": [1, 2, 4, 5, 10, 20], "gpu": [1, 2, 4]}
EXPECTED_SELECTED = {"cpu": 20, "gpu": 4}
NATIVE_CPUS = [5, 10, 15, 20]

INK = "#17212B"
MUTED = "#65727E"
GRID = "#D8E0E6"


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


def orch_path(model: str, device: str, k: int) -> Path:
    return (
        ORCH_ROOT
        / "ma"
        / "annotate_bf16"
        / "stead"
        / "580st"
        / model
        / device
        / f"kma{k}_krp0"
        / "sg"
        / "playback"
        / "static"
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


@lru_cache(maxsize=None)
def load_orch(model: str, device: str, k: int) -> dict:
    path = orch_path(model, device, k)
    if not path.is_file():
        raise RuntimeError(f"Missing orchestration cell: {path}")
    record = json.loads(path.read_text())
    meta = record.get("meta") or {}
    expected = {
        "composition": "ma",
        "method": "annotate_bf16",
        "n_stations": 580,
        "model": model,
        "device": device,
        "k_ma": k,
        "k_rp": 0,
        "packaging": "sg",
        "arrival": "playback",
        "fill": "static",
        "batch_size": 512,
        "repeats": 5,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"Metadata mismatch in {path}: {key}={meta.get(key)!r}, "
                f"expected {value!r}"
            )
    repeats = (record.get("timing") or {}).get("repeats") or []
    if len(repeats) != 5 or not all(item.get("success") is True for item in repeats):
        raise RuntimeError(f"Expected five successful repeats in {path}")
    values = [float(item["makespan_s"]) for item in repeats]
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    orch = record.get("orch") or {}
    if not math.isclose(
        mean, float(orch.get("makespan_s_mean")), rel_tol=0.0, abs_tol=1e-10
    ) or not math.isclose(
        std, float(orch.get("makespan_s_std")), rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(f"Repeat statistics disagree with summary in {path}")
    return {
        "mean_s": mean,
        "std_s": std,
        "repeat_values_s": values,
        "source_file": str(path.relative_to(RAPID)),
    }


@lru_cache(maxsize=None)
def load_native(model: str, device: str, n_cpus: int) -> dict:
    path = native_path(model, device, n_cpus)
    if not path.is_file():
        raise RuntimeError(f"Missing native cell: {path}")
    record = json.loads(path.read_text())
    meta = record.get("meta") or {}
    expected = {
        "method": "annotate_bf16",
        "n_stations": 580,
        "model": model,
        "device": device,
        "n_cpus": n_cpus,
        "torch_threads": n_cpus,
        "batch_size": 512,
        "repeats": 5,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"Metadata mismatch in {path}: {key}={meta.get(key)!r}, "
                f"expected {value!r}"
            )
    repeats = (record.get("timing") or {}).get("repeats") or []
    if len(repeats) != 5 or not all(item.get("success") is True for item in repeats):
        raise RuntimeError(f"Expected five successful repeats in {path}")
    values = [float(item["inference_s"]) for item in repeats]
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    timing = record.get("timing") or {}
    if not math.isclose(
        mean, float(timing.get("inference_s_mean")), rel_tol=0.0, abs_tol=1e-10
    ) or not math.isclose(
        std, float(timing.get("inference_s_std")), rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(f"Repeat statistics disagree with summary in {path}")
    return {
        "mean_s": mean,
        "std_s": std,
        "repeat_values_s": values,
        "n_cpus": n_cpus,
        "threads": n_cpus,
        "source_file": str(path.relative_to(RAPID)),
    }


def derive_rows() -> list[dict]:
    rows = []
    selected_by_model_device = {}
    for device in ("cpu", "gpu"):
        for model in MODELS:
            cells = [(k, load_orch(model, device, k)) for k in K_VALUES[device]]
            minimum = min(cell["mean_s"] for _, cell in cells)
            selected_k = next(
                k for k, cell in cells if cell["mean_s"] <= 1.10 * minimum
            )
            selected_by_model_device[(model, device)] = selected_k
            for k, cell in cells:
                rows.append(
                    {
                        "panel": "A" if device == "cpu" else "B",
                        "row_type": "scaling",
                        "model": model,
                        "device": device,
                        "k": k,
                        "mean_s": cell["mean_s"],
                        "std_s": cell["std_s"],
                        "repeat_values_s": ";".join(
                            f"{value:.12f}" for value in cell["repeat_values_s"]
                        ),
                        "minimum_s": minimum,
                        "selected_k": selected_k,
                        "is_selected": k == selected_k,
                        "native_n_cpus": "",
                        "native_threads": "",
                        "native_mean_s": "",
                        "native_std_s": "",
                        "rapid_mean_s": "",
                        "rapid_std_s": "",
                        "speedup": "",
                        "source_file": cell["source_file"],
                        "native_source": "",
                        "rapid_source": "",
                    }
                )

    for device in ("cpu", "gpu"):
        for model in MODELS:
            candidates = [
                (n_cpus, load_native(model, device, n_cpus))
                for n_cpus in NATIVE_CPUS
            ]
            native_cpus, native = min(
                candidates, key=lambda item: item[1]["mean_s"]
            )
            selected_k = selected_by_model_device[(model, device)]
            rapid = load_orch(model, device, selected_k)
            rows.append(
                {
                    "panel": "C" if device == "cpu" else "D",
                    "row_type": "native_comparison",
                    "model": model,
                    "device": device,
                    "k": selected_k,
                    "mean_s": "",
                    "std_s": "",
                    "repeat_values_s": "",
                    "minimum_s": "",
                    "selected_k": selected_k,
                    "is_selected": True,
                    "native_n_cpus": native_cpus,
                    "native_threads": native["threads"],
                    "native_mean_s": native["mean_s"],
                    "native_std_s": native["std_s"],
                    "rapid_mean_s": rapid["mean_s"],
                    "rapid_std_s": rapid["std_s"],
                    "speedup": native["mean_s"] / rapid["mean_s"],
                    "source_file": "",
                    "native_source": native["source_file"],
                    "rapid_source": rapid["source_file"],
                }
            )
    return rows


def validate(rows: list[dict]) -> str:
    lines = [
        "Figure 7 validation",
        "Scaling and native values are recalculated from five successful repeats.",
        "",
    ]
    for device in ("cpu", "gpu"):
        lines.append(f"{device.upper()} scaling")
        expected_k = EXPECTED_SELECTED[device]
        for model in MODELS:
            model_rows = [
                row
                for row in rows
                if row["row_type"] == "scaling"
                and row["device"] == device
                and row["model"] == model
            ]
            model_rows.sort(key=lambda row: int(row["k"]))
            selected = int(model_rows[0]["selected_k"])
            if selected != expected_k:
                raise RuntimeError(
                    f"Unexpected selected K for {model} {device}: {selected}"
                )
            values = ", ".join(
                f"K={row['k']}: {float(row['mean_s']):.9f} "
                f"± {float(row['std_s']):.9f} s"
                for row in model_rows
            )
            lines.extend(
                [
                    f"  {model}: {values}",
                    f"    minimum={min(float(row['mean_s']) for row in model_rows):.9f} s; "
                    f"selected K={selected}",
                ]
            )
        lines.append("")

    for device in ("cpu", "gpu"):
        lines.append(f"{device.upper()} native comparison")
        comparison_rows = [
            row
            for row in rows
            if row["row_type"] == "native_comparison"
            and row["device"] == device
        ]
        for row in comparison_rows:
            speedup = float(row["speedup"])
            lines.append(
                f"  {row['model']}: native={float(row['native_mean_s']):.9f} s "
                f"(CPUs/threads={row['native_n_cpus']}), "
                f"RAPID={float(row['rapid_mean_s']):.9f} s "
                f"(K={row['selected_k']}), speedup={speedup:.9f}"
            )
        if device == "cpu" and not all(
            float(row["speedup"]) > 1 for row in comparison_rows
        ):
            raise RuntimeError("Not all CPU native-to-RAPID speedups exceed 1")
        if device == "gpu":
            expected = {
                "EQCCT": True,
                "PhaseNet": True,
                "PhaseNetLight": True,
                "EQTransformer": False,
                "EQT-NC": False,
            }
            for row in comparison_rows:
                observed = float(row["speedup"]) > 1
                if observed != expected[row["model"]]:
                    raise RuntimeError(
                        f"Unexpected GPU comparison direction for {row['model']}"
                    )
        lines.append("")
    return "\n".join(lines)


def write_artifacts(rows: list[dict], validation: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "figure07_scaling_native_derived.csv"
    fields = [
        "panel",
        "row_type",
        "model",
        "device",
        "k",
        "mean_s",
        "std_s",
        "repeat_values_s",
        "minimum_s",
        "selected_k",
        "is_selected",
        "native_n_cpus",
        "native_threads",
        "native_mean_s",
        "native_std_s",
        "rapid_mean_s",
        "rapid_std_s",
        "speedup",
        "source_file",
        "native_source",
        "rapid_source",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    validation_path = OUT / "figure07_scaling_native_validation.txt"
    validation_path.write_text(validation + "\n")
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


def plain_log_axis(ax, ticks: list[float]) -> None:
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def scaling_axis(ax, rows: list[dict], device: str, letter: str) -> None:
    relevant = [
        row
        for row in rows
        if row["row_type"] == "scaling" and row["device"] == device
    ]
    model_offsets = dict(
        zip(MODELS, np.linspace(-0.16, 0.16, len(MODELS)))
    )
    for model in MODELS:
        model_rows = [row for row in relevant if row["model"] == model]
        model_rows.sort(key=lambda row: int(row["k"]))
        ks = [int(row["k"]) for row in model_rows]
        means = [float(row["mean_s"]) for row in model_rows]
        ax.plot(
            ks,
            means,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=4.3,
            linewidth=1.15,
            alpha=0.95,
            zorder=4,
        )
        for row in model_rows:
            repeat_values = [
                float(value)
                for value in str(row["repeat_values_s"]).split(";")
                if value
            ]
            x_values = np.full(len(repeat_values), float(row["k"]))
            x_values += model_offsets[model]
            x_values += np.linspace(-0.018, 0.018, len(repeat_values))
            ax.scatter(
                x_values,
                repeat_values,
                s=8,
                marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                alpha=0.24,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
    ax.set_yscale("log")
    if device == "cpu":
        ax.set_ylim(0.12, 12)
        plain_log_axis(ax, [0.2, 0.5, 1, 2, 5, 10])
    else:
        ax.set_ylim(0.12, 2.1)
        plain_log_axis(ax, [0.2, 0.5, 1, 2])
    ax.set_xticks(K_VALUES[device])
    ax.set_xlabel("Inference instances, K")
    ax.set_ylabel("Network makespan (s)")
    ax.set_title(device.upper(), pad=5)
    panel_label(ax, letter)
    clean_axis(ax)


def comparison_axis(ax, rows: list[dict], device: str, letter: str) -> None:
    relevant = [
        row
        for row in rows
        if row["row_type"] == "native_comparison" and row["device"] == device
    ]
    x = np.arange(len(MODELS))
    for index, model in enumerate(MODELS):
        row = next(item for item in relevant if item["model"] == model)
        speedup = float(row["speedup"])
        ax.scatter(
            index,
            speedup,
            s=46,
            marker=MODEL_MARKERS[model],
            facecolor=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        offset = 0.075 if speedup >= 1 else -0.075
        ax.text(
            index,
            speedup + offset,
            f"{speedup:.2f}×",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=6.4,
            color=MODEL_COLORS[model],
        )
    ax.axhline(1, color=INK, linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.45, len(MODELS) - 0.55)
    ax.set_ylim(0.65, 2.75)
    ax.set_yticks([0.75, 1, 1.5, 2, 2.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[model] for model in MODELS])
    ax.set_ylabel("Model-Actor-BF16 speedup over Annotate-BF16 (×)")
    ax.set_title(device.upper(), pad=5)
    panel_label(ax, letter)
    clean_axis(ax)


def make_figure(rows: list[dict]) -> plt.Figure:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 6.0),
        gridspec_kw={"hspace": 0.38, "wspace": 0.24},
    )
    scaling_axis(axes[0, 0], rows, "cpu", "A")
    scaling_axis(axes[0, 1], rows, "gpu", "B")
    comparison_axis(axes[1, 0], rows, "cpu", "C")
    comparison_axis(axes[1, 1], rows, "gpu", "D")
    figure.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.15)
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
        bbox_to_anchor=(0.52, 0.03),
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
    print(validation)

    figure = make_figure(rows)
    base = OUT / "figure07_scaling_native_comparison"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    revised_png_path = OUT / "figure07_scaling_native_comparison_revised.png"
    repeats_png_path = OUT / "figure07_scaling_native_comparison_repeat_points.png"
    figure.savefig(png_path, dpi=600)
    figure.savefig(pdf_path)
    figure.savefig(svg_path)
    figure.savefig(revised_png_path, dpi=600)
    figure.savefig(repeats_png_path, dpi=600)
    plt.close(figure)

    print(csv_path)
    print(validation_path)
    print(png_path)
    print(pdf_path)
    print(svg_path)
    print(revised_png_path)
    print(repeats_png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
