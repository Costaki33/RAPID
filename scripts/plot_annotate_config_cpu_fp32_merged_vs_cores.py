#!/usr/bin/env python3
"""First annotate-config figure: CPU fp32 merged 580st, inference vs n_cpus.

Locks: device=cpu, dtype=fp32, packaging=merged, n_stations=580,
batch_size=512, thread_mult=1.0 (T = n_cpus). Y is per-repeat inference_s
(3 isolated subprocesses per cell) plus the 3-repeat mean line.
"""
from __future__ import annotations

import csv
import json
import math
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAPID = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAPID))

ROOT = RAPID / "results" / "annotate_config" / "stead_iso_2026-09-08"
OUT_DIR = ROOT / "figures"

_TRIAL = runpy.run_path(str(RAPID / "benchmarks" / "fair" / "run_annotate_precision_trial.py"))
config_result_relpath = _TRIAL["config_result_relpath"]
TAG = _TRIAL["CONFIG_TAG"]
MODELS = list(_TRIAL["MODELS"])

N_STATIONS = 580
DEVICE = "cpu"
METHOD = "annotate_fp32"
PACKAGING = "merged"
BATCH_SIZE = 512
THREAD_MULT = 1.0
CPUS = (5, 10, 15, 20)

# Okabe–Ito from plot_merged_annotate_vs_cores.py, plus EQCCTP vermillion.
MODEL_COLOR = {
    "EQCCT": "#0072B2",
    "PhaseNet": "#E69F00",
    "PhaseNetLight": "#009E73",
    "EQTransformer": "#CC79A7",
    "EQT-NC": "#56B4E9",
    "EQCCTP": "#D55E00",
}

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


def _threads(n_cpus: int) -> int:
    from rapid.benchmark.fairness import threads_from_multiplier

    return threads_from_multiplier(n_cpus, THREAD_MULT)


def _cell(model: str, n_cpus: int) -> dict:
    return {
        "method": METHOD,
        "model": model,
        "n_stations": N_STATIONS,
        "packaging": PACKAGING,
        "device": DEVICE,
        "n_cpus": n_cpus,
        "thread_mult": THREAD_MULT,
        "torch_threads": _threads(n_cpus),
        "batch_size": BATCH_SIZE,
        "interop_threads": 1,
    }


def _is_usable(result: dict) -> bool:
    timing = result.get("timing") or {}
    meta = result.get("meta") or {}
    if timing.get("timed_out") or meta.get("timed_out") or meta.get("timed_out_by"):
        return False
    try:
        return float(timing.get("success_rate") or 0) >= 1.0
    except (TypeError, ValueError):
        return False


def load_rows() -> list[dict]:
    rows: list[dict] = []
    missing: list[str] = []
    skipped: list[str] = []
    for model in MODELS:
        for n_cpus in CPUS:
            cell = _cell(model, n_cpus)
            path = ROOT / config_result_relpath(cell, tag=TAG)
            label = f"{model} cpus={n_cpus} thr={cell['torch_threads']}"
            if not path.is_file():
                missing.append(label)
                continue
            result = json.loads(path.read_text())
            if not _is_usable(result):
                skipped.append(label)
                continue
            repeats = (result.get("timing") or {}).get("repeats") or []
            n_ok = 0
            for rec in repeats:
                if not rec.get("success"):
                    continue
                inf = rec.get("inference_s")
                try:
                    inf_f = float(inf)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(inf_f) or inf_f < 0:
                    continue
                rows.append(
                    {
                        "model": model,
                        "n_cpus": n_cpus,
                        "torch_threads": cell["torch_threads"],
                        "thread_mult": THREAD_MULT,
                        "repeat_index": int(rec.get("repeat_index", n_ok)),
                        "inference_s": inf_f,
                    }
                )
                n_ok += 1
            if n_ok == 0:
                skipped.append(label)
    if missing:
        print(f"missing result.json ({len(missing)}):")
        for item in missing:
            print(f"  {item}")
    if skipped:
        print(f"skipped timed-out/failed ({len(skipped)}):")
        for item in skipped:
            print(f"  {item}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "n_cpus", "torch_threads", "thread_mult", "repeat_index", "inference_s"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], png_path: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    n_models = len(MODELS)
    model_off = np.linspace(-0.45, 0.45, n_models)
    repeat_off = (-0.08, 0.0, 0.08)

    ymax = 0.0
    for mi, model in enumerate(MODELS):
        color = MODEL_COLOR[model]
        xs_mean: list[float] = []
        ys_mean: list[float] = []
        by_cpu: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            if row["model"] == model:
                by_cpu[int(row["n_cpus"])].append(row)
        for n_cpus in CPUS:
            pts = sorted(by_cpu.get(n_cpus, []), key=lambda r: r["repeat_index"])
            if not pts:
                continue
            xs_mean.append(float(n_cpus))
            ys_mean.append(float(np.mean([p["inference_s"] for p in pts])))
            for i, pt in enumerate(pts):
                x = n_cpus + model_off[mi] + repeat_off[i % len(repeat_off)]
                y = pt["inference_s"]
                ymax = max(ymax, y)
                ax.scatter(
                    x,
                    y,
                    s=18,
                    color=color,
                    alpha=0.45,
                    linewidths=0,
                    zorder=3,
                )
        if xs_mean:
            ax.plot(
                xs_mean,
                ys_mean,
                color=color,
                linestyle="-",
                linewidth=1.6,
                marker="o",
                markersize=5.5,
                markerfacecolor=color,
                markeredgecolor=color,
                zorder=4,
                label=model,
            )

    ax.set_xticks(list(CPUS))
    ax.set_xlim(CPUS[0] - 1.4, CPUS[-1] + 1.4)
    ax.set_ylim(0, ymax * 1.12 if ymax else 1.0)
    ax.set_xlabel("n_cpus")
    ax.set_ylabel("inference time (s)")
    ax.set_title(
        "CPU · fp32 · merged · 580 st · batch=512 · thread_mult=1.0 (T = n_cpus)"
    )
    ax.grid(True, axis="y", linewidth=0.4, color=GRID, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", title="model")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    n_expected = len(MODELS) * len(CPUS) * 3
    print(f"loaded {len(rows)} repeat points (expected {n_expected})")
    by_model = defaultdict(int)
    for row in rows:
        by_model[row["model"]] += 1
    for model in MODELS:
        print(f"  {model:14s} {by_model[model]}")
    if len(rows) != n_expected:
        print("WARNING: point count does not match 24 cells × 3 repeats")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "cpu_fp32_merged_580st_bs512_mult1_vs_cores.csv"
    png_path = OUT_DIR / "cpu_fp32_merged_580st_bs512_mult1_vs_cores.png"
    write_csv(csv_path, rows)
    plot(rows, png_path)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
