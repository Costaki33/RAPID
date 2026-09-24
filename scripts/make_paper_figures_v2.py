#!/usr/bin/env python3
"""Clean, journal-style figures for the RAPID Annotate manuscript.

The first figure set is intentionally preserved. This script writes a redesigned
set to ``docs/figures/paper_v2`` as vector PDF and 400-dpi PNG.
"""
from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

RAPID = Path(__file__).resolve().parents[1]
OUT = RAPID / "docs" / "figures" / "paper_v2"
BASE_SCRIPT = RAPID / "scripts" / "make_paper_figures.py"

spec = importlib.util.spec_from_file_location("rapid_figures_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

MODELS = base.MODELS
SHORT = base.SHORT
K_CPU = base.K_CPU
K_GPU = base.K_GPU

# Restrained, colorblind-safe palette.
INK = "#17212B"
MUTED = "#637381"
GRID = "#DDE3E8"
LIGHT = "#F4F6F8"
NAVY = "#24557A"
BLUE = "#4C8BB8"
TEAL = "#2A8C82"
ORANGE = "#D97745"
RED = "#B84A45"
GOLD = "#C39A32"
PURPLE = "#7668A6"
MODEL_COLORS = {
    "EQCCT": NAVY,
    "PhaseNet": TEAL,
    "PhaseNetLight": BLUE,
    "EQTransformer": GOLD,
    "EQT-NC": ORANGE,
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax, *, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid == "y":
        ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    elif grid == "x":
        ax.grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, label: str, title: str) -> None:
    ax.set_title(title, loc="left", pad=9)
    ax.text(
        -0.04,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        ha="right",
        va="bottom",
    )


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=400)
    plt.close(fig)


def rounded_box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    face="white",
    edge=GRID,
    lw=1.0,
    fontsize=8,
    weight="normal",
    color=INK,
    radius=0.03,
):
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.25,
    )
    return box


def arrow(ax, start, end, *, color=MUTED, lw=1.1):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10),
    )


def load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def all_precision_cells():
    rows = []
    for path in base.PREC.rglob("result.json"):
        r = load_json(path)
        if not r:
            continue
        m = r.get("meta") or {}
        method = m.get("method")
        if method not in {"annotate_fp32", "annotate_bf16", "annotate_fp16"}:
            continue
        value = base.inf_mean(r)
        if value is None or not math.isfinite(value):
            continue
        rows.append(
            {
                "method": method,
                "model": m.get("model"),
                "nst": int(m.get("n_stations")),
                "device": m.get("device"),
                "cpus": int(m.get("n_cpus")),
                "batch": int(m.get("batch_size")),
                "runtime": value,
                "result": r,
            }
        )
    return rows


PRECISION_ROWS = all_precision_cells()


def matched_precision_speedups(method: str, *, device=None, model=None):
    by_key = defaultdict(dict)
    for row in PRECISION_ROWS:
        if device and row["device"] != device:
            continue
        if model and row["model"] != model:
            continue
        key = (row["model"], row["nst"], row["device"], row["cpus"], row["batch"])
        by_key[key][row["method"]] = row["runtime"]
    values = []
    for methods in by_key.values():
        if "annotate_fp32" in methods and method in methods:
            values.append(methods["annotate_fp32"] / methods[method])
    return values


def precision_f1(method: str, model: str, phase: str):
    r = load_json(base.prec_path(method, model, 580, "cpu", 20, 512))
    return base.f1_summary(r, "pick_quality_vs_fp32", phase)


def orch_value(
    comp,
    nst,
    model,
    device,
    kma,
    krp,
    pack,
    arrival,
    fill,
    *,
    latency=False,
):
    r = load_json(
        base.orch_path(
            comp,
            "annotate_bf16",
            nst,
            model,
            device,
            kma,
            krp,
            pack,
            arrival,
            fill,
        )
    )
    return base.p95(r) if latency else base.mk(r)


XFER = RAPID / "handoff" / "locked_recipe_transfer_BEGE-TEXA75535L" / "raw_results"


def mem_mean(r, key: str):
    if not r:
        return None
    v = (r.get("memory") or {}).get(key)
    return float(v) if v is not None else None


def pss_gb(r):
    v = mem_mean(r, "peak_pss_mb_mean")
    return None if v is None else v / 1000.0


def vram_gb(r):
    v = mem_mean(r, "peak_vram_mb_mean")
    return None if v is None or v <= 0 else v / 1000.0


def native_prec(method, model, device, nst=580, cpus=20, bs=512):
    return load_json(base.prec_path(method, model, nst, device, cpus, bs))


def xfer_native(model, device, nst=580, cpus=20):
    return load_json(
        XFER
        / "annotate_bf16"
        / "stead"
        / f"{nst}st"
        / model
        / device
        / f"cpus{cpus}"
        / f"thr{cpus}"
        / "bs512"
        / "merged"
        / "xfer"
        / "result.json"
    )


def xfer_orch(model, device, k, nst=580, arrival="playback", fill="static"):
    return load_json(
        XFER
        / "ma"
        / "annotate_bf16"
        / "stead"
        / f"{nst}st"
        / model
        / device
        / f"kma{k}_krp0"
        / "sg"
        / arrival
        / fill
        / "bs512"
        / "xfer"
        / "result.json"
    )


def figure1() -> None:
    """Minimal precision pipeline diagram."""
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    y, h = 2.45, 1.15
    rounded_box(ax, (0.15, y), 1.55, h, "Network\nStream", face=LIGHT, edge=GRID, weight="semibold")
    stages = [
        (2.15, "Window"),
        (3.75, "Batch"),
        (5.35, "Forward\npass"),
        (6.95, "Reconstruct"),
    ]
    for i, (x, label) in enumerate(stages):
        selected = label.startswith("Forward")
        rounded_box(
            ax,
            (x, y),
            1.25,
            h,
            label,
            face="#EAF1F6" if selected else "white",
            edge=NAVY if selected else GRID,
            lw=1.5 if selected else 1,
            weight="semibold" if selected else "normal",
            color=NAVY if selected else INK,
        )
        if i:
            arrow(ax, (stages[i - 1][0] + 1.25, y + h / 2), (x, y + h / 2))
    arrow(ax, (1.70, y + h / 2), (2.15, y + h / 2))
    rounded_box(ax, (8.75, y), 1.25, h, "Probability\ntraces", face=LIGHT, edge=GRID)
    arrow(ax, (8.20, y + h / 2), (8.75, y + h / 2))
    rounded_box(ax, (10.45, y), 1.35, h, "Discrete\nP/S picks", face="#F7F3EA", edge=GRID)
    arrow(ax, (10.0, y + h / 2), (10.45, y + h / 2))

    ax.text(5.98, 4.28, "Only this stage changes", ha="center", fontsize=8, color=NAVY, fontweight="semibold")
    arrow(ax, (5.98, 4.08), (5.98, 3.66), color=NAVY)
    ax.text(5.98, 1.70, "tensor dtype", ha="center", fontsize=7.5, color=MUTED)
    variants = [
        (4.30, "FP32", "reference", LIGHT, MUTED),
        (5.42, "BF16", "selected", "#E5F1F0", TEAL),
        (6.54, "FP16", "limited", "#F9ECE6", ORANGE),
    ]
    for x, dtype, sub, face, edge in variants:
        rounded_box(
            ax,
            (x, 0.70),
            1.0,
            0.72,
            f"{dtype}\n{sub}",
            face=face,
            edge=edge,
            lw=1.3,
            fontsize=7.5,
            weight="semibold" if dtype == "BF16" else "normal",
            color=edge if dtype != "FP32" else INK,
        )
    ax.text(
        11.15,
        1.55,
        "Classify aggregation was\nperformed offline for quality",
        ha="center",
        va="top",
        fontsize=7,
        color=MUTED,
    )
    save(fig, "fig01_annotate_pipeline")


def figure2() -> None:
    """Precision speedups and batch-size summary."""
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.55, wspace=0.34)

    for col, device in enumerate(("cpu", "gpu")):
        ax = fig.add_subplot(gs[0, col])
        y = np.arange(len(MODELS))
        for j, (method, label, color, marker, offset) in enumerate(
            [
                ("annotate_bf16", "BF16", TEAL, "o", -0.12),
                ("annotate_fp16", "FP16", ORANGE, "D", 0.12),
            ]
        ):
            medians, lows, highs = [], [], []
            for model in MODELS:
                vals = matched_precision_speedups(method, device=device, model=model)
                if not vals:
                    medians.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)
                else:
                    medians.append(np.median(vals))
                    lows.append(np.percentile(vals, 25))
                    highs.append(np.percentile(vals, 75))
            med = np.asarray(medians)
            lo = np.asarray(lows)
            hi = np.asarray(highs)
            valid = np.isfinite(med)
            ax.errorbar(
                med[valid],
                y[valid] + offset,
                xerr=np.vstack([med[valid] - lo[valid], hi[valid] - med[valid]]),
                fmt=marker,
                color=color,
                markerfacecolor="white" if method == "annotate_fp16" else color,
                markeredgewidth=1.2,
                markersize=5,
                linewidth=1.1,
                capsize=2,
                label=label,
                zorder=3,
            )
        ax.axvline(1, color=INK, linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[m] for m in MODELS])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(0.08, 2.6)
        ax.set_xticks([0.1, 0.25, 0.5, 1, 2])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Speedup versus FP32  (>1 is faster)")
        panel(ax, "AB"[col], f"{device.upper()} matched cells")
        clean_axis(ax, grid="x")
        if col == 0:
            ax.legend(loc="upper right")
        if col == 1:
            for idx, model in enumerate(MODELS):
                if model in {"EQTransformer", "EQT-NC"}:
                    ax.text(0.09, idx, "FP16 incompatible", color=ORANGE, fontsize=6.5, va="center")

    ax = fig.add_subplot(gs[1, :])
    batches = [64, 128, 256, 512]
    # Normalize each model/network/device/core series by its own fastest batch.
    values = defaultdict(list)
    grouped = defaultdict(dict)
    for row in PRECISION_ROWS:
        if row["method"] not in {"annotate_fp32", "annotate_bf16"}:
            continue
        key = (row["method"], row["model"], row["nst"], row["device"], row["cpus"])
        grouped[key][row["batch"]] = row["runtime"]
    for (method, *_), series in grouped.items():
        if len(series) < 2:
            continue
        best = min(series.values())
        for batch, runtime in series.items():
            values[(method, batch)].append(runtime / best)
    for method, label, color, marker in (
        ("annotate_fp32", "FP32", MUTED, "s"),
        ("annotate_bf16", "BF16", TEAL, "o"),
    ):
        med = [np.median(values[(method, b)]) for b in batches]
        q1 = [np.percentile(values[(method, b)], 25) for b in batches]
        q3 = [np.percentile(values[(method, b)], 75) for b in batches]
        ax.plot(batches, med, marker=marker, color=color, label=label, lw=1.5, ms=5)
        ax.fill_between(batches, q1, q3, color=color, alpha=0.12, linewidth=0)
    ax.axvline(512, color=TEAL, linewidth=0.9, linestyle="--")
    ax.text(505, 1.36, "selected", color=TEAL, ha="right", va="top", fontsize=7)
    ax.set_xticks(batches)
    ax.set_xlabel("Annotate batch size")
    ax.set_ylabel("Relative runtime")
    ax.set_ylim(0.96, 1.40)
    panel(ax, "C", "Batch-size sensitivity (median and interquartile range)")
    clean_axis(ax)
    ax.legend(ncol=2, loc="upper center")
    save(fig, "fig02_precision_performance")


def figure3() -> None:
    """Pick agreement as a compact dot plot."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    y = np.arange(len(MODELS))
    for ax, phase, letter in zip(axes, ("P", "S"), ("A", "B")):
        for method, label, color, marker, offset in (
            ("annotate_bf16", "BF16", TEAL, "o", -0.12),
            ("annotate_fp16", "FP16", ORANGE, "D", 0.12),
        ):
            vals = [precision_f1(method, model, phase) for model in MODELS]
            valid_y, valid_x = [], []
            for i, value in enumerate(vals):
                if value is not None and math.isfinite(value):
                    valid_y.append(i + offset)
                    valid_x.append(value)
            ax.scatter(
                valid_x,
                valid_y,
                marker=marker,
                s=38,
                color=color,
                facecolor="white" if method == "annotate_fp16" else color,
                linewidth=1.2,
                label=label,
                zorder=3,
            )
        ax.axvline(1, color=INK, linewidth=0.9)
        ax.set_xlim(0.4, 1.015)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[m] for m in MODELS])
        ax.set_xlabel("F1 versus FP32")
        panel(ax, letter, f"{phase}-phase agreement")
        clean_axis(ax, grid="x")
        for i, model in enumerate(MODELS):
            if model in {"EQTransformer", "EQT-NC"}:
                ax.text(0.42, i + 0.12, "FP16 not evaluated", fontsize=6.5, color=ORANGE, va="center")
    axes[0].set_ylim(len(MODELS) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig03_pick_agreement")


def figure4() -> None:
    """RAPID architecture without dense prose."""
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Top: decomposition.
    ax.text(0.2, 9.65, "A", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.62, 9.65, "Network decomposition", fontsize=10, fontweight="semibold", color=INK)
    rounded_box(ax, (0.3, 8.05), 1.45, 1.05, "N station\nStreams", face=LIGHT, edge=GRID, weight="semibold")
    arrow(ax, (1.78, 8.57), (2.45, 8.57))
    rounded_box(
        ax,
        (2.5, 8.05),
        2.0,
        1.05,
        "Dispatcher\nG = ceil(N / K)",
        face="#EAF1F6",
        edge=NAVY,
        lw=1.3,
        weight="semibold",
        color=NAVY,
    )
    arrow(ax, (4.55, 8.57), (5.20, 8.57))
    for i, label in enumerate(("1", "2", "…", "K")):
        x = 5.28 + i * 1.35
        rounded_box(
            ax,
            (x, 8.05),
            1.0,
            1.05,
            f"Group {label}\n≈ G stations" if label != "…" else "…",
            face="white",
            edge=GRID,
            fontsize=7.2,
        )

    # Middle: orthogonal axes.
    ax.text(0.2, 7.15, "B", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.62, 7.15, "Two independent orchestration choices", fontsize=10, fontweight="semibold", color=INK)
    ax.text(0.55, 6.25, "Packaging", fontsize=8, color=MUTED, fontweight="semibold")
    rounded_box(ax, (2.05, 5.65), 2.35, 1.15, "SG\ngrouped call", face="#E7F3F1", edge=TEAL, lw=1.3, weight="semibold", color=TEAL)
    ax.text(4.72, 6.23, "or", color=MUTED, ha="center", va="center")
    rounded_box(ax, (5.05, 5.65), 2.35, 1.15, "S1\nsingle-station call", face="white", edge=GRID, weight="semibold")

    ax.text(0.55, 4.60, "Lifecycle", fontsize=8, color=MUTED, fontweight="semibold")
    rounded_box(ax, (2.05, 3.75), 2.35, 1.35, "Model-Actor\nload once\nremain resident", face="#EAF1F6", edge=NAVY, lw=1.3, weight="semibold", color=NAVY)
    ax.text(4.72, 4.42, "or", color=MUTED, ha="center", va="center")
    rounded_box(ax, (5.05, 3.75), 2.35, 1.35, "Ripper\nload → infer\nterminate", face="#F9ECE6", edge=ORANGE, lw=1.3, weight="semibold", color=ORANGE)
    ax.plot([8.05, 8.05], [3.65, 6.9], color=GRID, lw=1)
    ax.text(8.45, 6.45, "Selected combination", fontsize=8, color=MUTED)
    rounded_box(ax, (8.45, 4.75), 3.0, 1.25, "Model-Actor + SG", face="#E5F1F0", edge=TEAL, lw=1.5, fontsize=10, weight="semibold", color=INK)
    ax.text(9.95, 4.25, "persistent models + batched groups", fontsize=7, color=MUTED, ha="center")

    # Bottom: ready queue.
    ax.text(0.2, 2.85, "C", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.62, 2.85, "Streaming dispatch", fontsize=10, fontweight="semibold", color=INK)
    arrivals = [(0.6, "t=0"), (1.75, "t=15"), (2.9, "t=30"), (4.05, "t=65"), (5.2, "t=90")]
    for x, label in arrivals:
        ax.scatter(x, 1.85, s=34, color=BLUE, zorder=3)
        ax.text(x, 1.47, label, ha="center", fontsize=7, color=MUTED)
    ax.plot([0.6, 5.2], [1.85, 1.85], color=GRID, lw=1, zorder=1)
    arrow(ax, (5.45, 1.85), (6.25, 1.85))
    rounded_box(ax, (6.3, 1.20), 1.45, 1.3, "Ready\nqueue", face="#F7F3EA", edge=GOLD, lw=1.2, weight="semibold")
    arrow(ax, (7.8, 1.85), (8.55, 1.85))
    rounded_box(ax, (8.6, 1.55), 1.25, 0.9, "Actor 1", face="#EAF1F6", edge=NAVY)
    rounded_box(ax, (10.1, 1.55), 1.25, 0.9, "Actor K", face="#EAF1F6", edge=NAVY)
    ax.text(9.98, 1.0, "next free actor takes the next ready group", fontsize=7.3, color=MUTED, ha="center")
    ax.text(9.98, 0.50, "Eager: dispatch immediately, even when group size < G", fontsize=7.6, color=TEAL, ha="center", fontweight="semibold")
    save(fig, "fig04_rapid_architecture")


def matched_orch_ratios(kind: str, device: str):
    values = []
    for nst in (250, 580):
        for model in MODELS:
            if kind == "grouping":
                ks = K_CPU if device == "cpu" else K_GPU
                for k in ks:
                    sg = orch_value("ma", nst, model, device, k, 0, "sg", "playback", "static")
                    s1 = orch_value("ma", nst, model, device, k, 0, "s1", "playback", "static")
                    if sg and s1:
                        values.append(s1 / sg)
            else:
                ks = [1, 2, 4, 5] if device == "cpu" else [1, 2]
                for k in ks:
                    ma = orch_value("ma", nst, model, device, k, 0, "sg", "playback", "static")
                    rp = orch_value("ripper", nst, model, device, 0, k, "sg", "playback", "static")
                    if ma and rp:
                        values.append(rp / ma)
    return values


def effect_distribution(ax, groups, *, title, letter, color):
    rng = np.random.default_rng(42)
    positions = np.arange(len(groups))
    for pos, (name, vals) in enumerate(groups):
        vals = np.asarray(vals)
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=9,
            color=color,
            alpha=0.25,
            edgecolors="none",
            zorder=2,
        )
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([pos - 0.19, pos + 0.19], [med, med], color=INK, lw=2.2, zorder=4)
        ax.plot([pos, pos], [q1, q3], color=INK, lw=1.1, zorder=4)
        ax.scatter([pos], [med], s=34, color="white", edgecolor=INK, linewidth=1, zorder=5)
        ax.text(pos, max(vals) * 1.10, f"{med:.1f}× median", ha="center", fontsize=7, color=INK)
    ax.axhline(1, color=MUTED, linewidth=0.9, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([name for name, _ in groups])
    ax.set_ylabel("Paired-cell speedup")
    panel(ax, letter, title)
    clean_axis(ax)


def figure5() -> None:
    """Matched-cell effects; no giant outlier bars."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    effect_distribution(
        axes[0],
        [
            ("CPU", matched_orch_ratios("grouping", "cpu")),
            ("GPU", matched_orch_ratios("grouping", "gpu")),
        ],
        title="Grouping benefit: S1 runtime / SG runtime",
        letter="A",
        color=TEAL,
    )
    axes[0].text(
        0.02,
        0.03,
        "166 / 180 matched cells favor SG",
        transform=axes[0].transAxes,
        fontsize=7.5,
        color=MUTED,
    )
    effect_distribution(
        axes[1],
        [
            ("CPU", matched_orch_ratios("persistence", "cpu")),
            ("GPU", matched_orch_ratios("persistence", "gpu")),
        ],
        title="Persistence benefit: Ripper / Model-Actor",
        letter="B",
        color=NAVY,
    )
    axes[1].text(
        0.02,
        0.03,
        "116 / 120 matched SG cells favor Model-Actor",
        transform=axes[1].transAxes,
        fontsize=7.5,
        color=MUTED,
    )
    save(fig, "fig05_grouping_persistence")


def best_hybrid(model, device):
    splits = [(10, 2), (5, 5), (4, 1)] if device == "cpu" else [(2, 1), (1, 1)]
    vals = []
    for comp in ("ma_ontime_rp_delayed", "rp_ontime_ma_delayed"):
        for kma, krp in splits:
            value = orch_value(
                comp,
                580,
                model,
                device,
                kma,
                krp,
                "sg",
                "staggered",
                "eager",
                latency=True,
            )
            if value:
                vals.append(value)
    return min(vals) if vals else np.nan


def figure6() -> None:
    """Streaming latency with concise metric explanation."""
    fig = plt.figure(figsize=(7.2, 4.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.46, 1], hspace=0.38, wspace=0.30)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    ax_top.axvspan(60, 100, color=LIGHT, zorder=0)
    ax_top.plot([0, 94], [0.60, 0.60], color=MUTED, lw=1)
    for t in (0, 5, 15, 30, 60, 65, 90):
        ax_top.scatter(t, 0.60, s=28, color=BLUE, zorder=2)
        ax_top.text(t, 0.35, str(t), ha="center", va="top", fontsize=7, color=MUTED)
    ax_top.text(29, 0.88, "window 0", fontsize=7, color=MUTED)
    ax_top.text(76, 0.88, "window 1", fontsize=7, color=MUTED)
    ax_top.text(97, 0.60, "time (s)", fontsize=7, color=MUTED, va="center")
    ax_top.text(
        50,
        0.06,
        "Network makespan ≈ 90 s reflects the latest arrival.  Streaming is ranked by p95 finish−ready.",
        ha="center",
        fontsize=8,
        color=INK,
        fontweight="semibold",
    )

    strategies = ["Eager SG", "S1", "Wait-5", "Wait-10", "Best hybrid"]
    colors = [TEAL, PURPLE, GOLD, ORANGE, MUTED]
    for col, (device, k, letter) in enumerate((("cpu", 20, "A"), ("gpu", 4, "B"))):
        ax = fig.add_subplot(gs[1, col])
        y = np.arange(len(strategies))
        for model in MODELS:
            vals = [
                orch_value("ma", 580, model, device, k, 0, "sg", "staggered", "eager", latency=True),
                orch_value("ma", 580, model, device, k, 0, "s1", "staggered", "eager", latency=True),
                orch_value("ma", 580, model, device, k, 0, "sg", "staggered", "w5", latency=True),
                orch_value("ma", 580, model, device, k, 0, "sg", "staggered", "w10", latency=True),
                best_hybrid(model, device),
            ]
            ax.plot(vals, y, color=GRID, lw=0.8, zorder=1)
            for yi, (value, color) in enumerate(zip(vals, colors)):
                if value and math.isfinite(value):
                    ax.scatter(value, yi, s=26, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        for yi in range(len(strategies)):
            collected = []
            for model in MODELS:
                if yi == 0:
                    v = orch_value("ma", 580, model, device, k, 0, "sg", "staggered", "eager", latency=True)
                elif yi == 1:
                    v = orch_value("ma", 580, model, device, k, 0, "s1", "staggered", "eager", latency=True)
                elif yi == 2:
                    v = orch_value("ma", 580, model, device, k, 0, "sg", "staggered", "w5", latency=True)
                elif yi == 3:
                    v = orch_value("ma", 580, model, device, k, 0, "sg", "staggered", "w10", latency=True)
                else:
                    v = best_hybrid(model, device)
                if v:
                    collected.append(v)
            if collected:
                med = np.median(collected)
                ax.scatter(med, yi, s=65, facecolor="none", edgecolor=INK, linewidth=1.1, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(0.07, 16)
        ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_yticks(y)
        ax.set_yticklabels(strategies)
        ax.invert_yaxis()
        ax.set_xlabel("p95 finish−ready (s, log scale)")
        panel(ax, letter, f"{device.upper()}  ·  K = {k}")
        clean_axis(ax, grid="x")
        if col == 1:
            ax.text(0.98, 0.03, "○ median across models", transform=ax.transAxes, ha="right", fontsize=6.8, color=MUTED)
    save(fig, "fig06_streaming_latency")


def figure7() -> None:
    """K scaling across both station counts."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.75), sharex="col")
    for row, nst in enumerate((250, 580)):
        for col, (device, ks) in enumerate((("cpu", K_CPU), ("gpu", K_GPU))):
            ax = axes[row, col]
            for model in MODELS:
                vals = [
                    orch_value("ma", nst, model, device, k, 0, "sg", "playback", "static")
                    for k in ks
                ]
                ax.plot(
                    ks,
                    vals,
                    marker="o",
                    ms=3.6,
                    lw=1.25,
                    color=MODEL_COLORS[model],
                    label=SHORT[model],
                )
                finite = [(k, v) for k, v in zip(ks, vals) if v and math.isfinite(v)]
                best = min(v for _, v in finite)
                elbow = next(k for k, v in finite if v <= 1.10 * best)
                ax.scatter(
                    elbow,
                    dict(finite)[elbow],
                    s=45,
                    facecolors="white",
                    edgecolors=MODEL_COLORS[model],
                    linewidth=1.25,
                    zorder=4,
                )
            ax.set_yscale("log")
            ax.set_xticks(ks)
            ax.set_ylabel("Makespan (s, log)")
            if row == 1:
                ax.set_xlabel("Inference instances  K")
            letter = chr(ord("A") + row * 2 + col)
            panel(ax, letter, f"{nst} stations  ·  {device.upper()}")
            clean_axis(ax)
            if row == 0 and col == 0:
                ax.legend(ncol=1, loc="upper right")
            if row == 1 and col == 1:
                ax.text(0.98, 0.05, "○ smallest K within 10% of minimum", transform=ax.transAxes, ha="right", fontsize=6.8, color=MUTED)
    save(fig, "fig07_instance_scaling")


def native_values(model, device):
    return {
        "sequential": base.best_seq_inf(model, device),
        "merged": base.best_merged_inf("annotate_bf16", model, device),
        "rapid": orch_value(
            "ma",
            580,
            model,
            device,
            20 if device == "cpu" else 4,
            0,
            "sg",
            "playback",
            "static",
        ),
    }


def figure8() -> None:
    """Raw runtime and speedup versus native merged."""
    fig = plt.figure(figsize=(7.2, 5.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.78], hspace=0.48, wspace=0.30)
    method_style = {
        "sequential": ("Native sequential", MUTED, "s"),
        "merged": ("Native merged", INK, "o"),
        "rapid": ("RAPID MA–SG", TEAL, "D"),
    }
    x = np.arange(len(MODELS))
    for col, device in enumerate(("cpu", "gpu")):
        ax = fig.add_subplot(gs[0, col])
        for method, (label, color, marker) in method_style.items():
            vals = [native_values(model, device)[method] for model in MODELS]
            ax.plot(x, vals, marker=marker, color=color, lw=1.15, ms=5, label=label)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_ylabel("Runtime (s, log)")
        panel(ax, "AB"[col], f"{device.upper()} raw runtime")
        clean_axis(ax)
        if col == 0:
            ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[1, :])
    y = np.arange(len(MODELS))
    for device, color, marker, offset in (
        ("cpu", NAVY, "o", -0.12),
        ("gpu", ORANGE, "D", 0.12),
    ):
        vals = []
        for model in MODELS:
            v = native_values(model, device)
            vals.append(v["merged"] / v["rapid"])
        ax.scatter(vals, y + offset, s=42, color=color, marker=marker, label=device.upper(), zorder=3)
        for value, yi in zip(vals, y + offset):
            ax.plot([1, value], [yi, yi], color=color, alpha=0.35, lw=1.1, zorder=1)
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.75, 3.1)
    ax.set_xticks([0.8, 1, 1.5, 2, 3])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlabel("Speedup versus native merged Annotate  (>1 favors RAPID)")
    panel(ax, "C", "RAPID speedup versus native merged Annotate")
    clean_axis(ax, grid="x")
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.015))
    save(fig, "fig08_native_comparison")


def figure7_8_combined() -> None:
    """Seven-figure alternative: compact scaling + payoff."""
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], hspace=0.43, wspace=0.30)
    for col, (device, ks) in enumerate((("cpu", K_CPU), ("gpu", K_GPU))):
        ax = fig.add_subplot(gs[0, col])
        for model in MODELS:
            vals = [orch_value("ma", 580, model, device, k, 0, "sg", "playback", "static") for k in ks]
            ax.plot(ks, vals, marker="o", ms=3.8, lw=1.25, color=MODEL_COLORS[model], label=SHORT[model])
            best = min(vals)
            elbow = next(k for k, v in zip(ks, vals) if v <= 1.10 * best)
            ax.scatter(elbow, vals[ks.index(elbow)], s=44, facecolors="white", edgecolors=MODEL_COLORS[model], lw=1.2, zorder=4)
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xlabel("Inference instances  K")
        ax.set_ylabel("Makespan (s, log)")
        panel(ax, "AB"[col], f"580-station {device.upper()} scaling")
        clean_axis(ax)
        if col == 0:
            ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[1, :])
    y = np.arange(len(MODELS))
    for device, color, marker, offset in (("cpu", NAVY, "o", -0.12), ("gpu", ORANGE, "D", 0.12)):
        vals = []
        for model in MODELS:
            v = native_values(model, device)
            vals.append(v["merged"] / v["rapid"])
        ax.scatter(vals, y + offset, s=43, color=color, marker=marker, label=device.upper(), zorder=3)
        for value, yi in zip(vals, y + offset):
            ax.plot([1, value], [yi, yi], color=color, alpha=0.35, lw=1.1)
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.75, 3.1)
    ax.set_xticks([0.8, 1, 1.5, 2, 3])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlabel("Speedup versus native merged Annotate  (>1 favors RAPID)")
    panel(ax, "C", "Operating-point payoff")
    clean_axis(ax, grid="x")
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.015))
    save(fig, "fig07_combined_scaling_native")


def figure9() -> None:
    """Host PSS and GPU VRAM for native Annotate and RAPID orchestration."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    x = np.arange(len(MODELS))
    dtypes = [
        ("annotate_fp32", "FP32", MUTED, "s"),
        ("annotate_bf16", "BF16", NAVY, "o"),
        ("annotate_fp16", "FP16", ORANGE, "D"),
    ]

    ax = axes[0, 0]
    for method, label, color, marker in dtypes:
        xs, vals = [], []
        for i, model in enumerate(MODELS):
            v = pss_gb(native_prec(method, model, "cpu"))
            if v is None:
                continue
            xs.append(x[i])
            vals.append(v)
        ax.plot(xs, vals, marker=marker, color=color, lw=1.15, ms=5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS])
    ax.set_ylabel("Peak PSS (GB)")
    panel(ax, "A", "Native merged CPU  ·  580 stations")
    clean_axis(ax)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    for method, label, color, marker in dtypes:
        xs, vals = [], []
        for i, model in enumerate(MODELS):
            v = vram_gb(native_prec(method, model, "gpu"))
            if v is None:
                continue
            xs.append(x[i])
            vals.append(v)
        ax.plot(xs, vals, marker=marker, color=color, lw=1.15, ms=5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS])
    ax.set_ylabel("Peak VRAM (GB)")
    panel(ax, "B", "Native merged GPU  ·  580 stations")
    clean_axis(ax)

    ax = axes[1, 0]
    for model in MODELS:
        ma = []
        for k in K_CPU:
            r = load_json(base.orch_path("ma", "annotate_bf16", 580, model, "cpu", k, 0, "sg", "playback", "static"))
            ma.append(pss_gb(r))
        ax.plot(K_CPU, ma, marker="o", ms=3.6, lw=1.25, color=MODEL_COLORS[model], label=SHORT[model])
        rp_k = [1, 2, 4, 5]
        rp = []
        for k in rp_k:
            r = load_json(base.orch_path("ripper", "annotate_bf16", 580, model, "cpu", 0, k, "sg", "playback", "static"))
            rp.append(pss_gb(r))
        ax.plot(rp_k, rp, marker="x", ms=4.2, lw=0.0, color=MODEL_COLORS[model], zorder=4)
    ax.set_xticks(K_CPU)
    ax.set_xlabel("Inference instances  K")
    ax.set_ylabel("Peak PSS (GB)")
    panel(ax, "C", "CPU RAPID SG  ·  Model-Actor vs Ripper")
    clean_axis(ax)
    ax.legend(ncol=2, loc="upper left")
    ax.text(0.98, 0.04, "○ Model-Actor   × Ripper", transform=ax.transAxes, ha="right", fontsize=6.8, color=MUTED)

    ax = axes[1, 1]
    for model in MODELS:
        ma = []
        for k in K_GPU:
            r = load_json(base.orch_path("ma", "annotate_bf16", 580, model, "gpu", k, 0, "sg", "playback", "static"))
            ma.append(vram_gb(r))
        ax.plot(K_GPU, ma, marker="o", ms=3.6, lw=1.25, color=MODEL_COLORS[model], label=SHORT[model])
        rp_k = [1, 2]
        rp = []
        for k in rp_k:
            r = load_json(base.orch_path("ripper", "annotate_bf16", 580, model, "gpu", 0, k, "sg", "playback", "static"))
            rp.append(vram_gb(r))
        ax.plot(rp_k, rp, marker="x", ms=4.2, lw=0.0, color=MODEL_COLORS[model], zorder=4)
    ax.set_xticks(K_GPU)
    ax.set_xlabel("Inference instances  K")
    ax.set_ylabel("Peak VRAM (GB)")
    panel(ax, "D", "GPU RAPID SG  ·  Model-Actor vs Ripper")
    clean_axis(ax)
    save(fig, "fig09_memory")


def figure10() -> None:
    """Locked-recipe transfer from workstation to laptop."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    x = np.arange(len(MODELS))
    width = 0.36

    ax = axes[0, 0]
    ws = [base.best_merged_inf("annotate_bf16", m, "cpu") for m in MODELS]
    lt = []
    for m in MODELS:
        r = xfer_native(m, "cpu")
        lt.append(None if not r else (r.get("timing") or {}).get("inference_s_mean"))
    ax.bar(x - width / 2, ws, width, color=NAVY, label="Workstation")
    ax.bar(x + width / 2, lt, width, color=ORANGE, label="Laptop")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS])
    ax.set_ylabel("Inference time (s, log)")
    panel(ax, "A", "Native merged CPU  ·  580 stations")
    clean_axis(ax)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ws = [base.best_merged_inf("annotate_bf16", m, "gpu") for m in MODELS]
    lt = []
    for m in MODELS:
        r = xfer_native(m, "gpu")
        lt.append(None if not r else (r.get("timing") or {}).get("inference_s_mean"))
    ax.bar(x - width / 2, ws, width, color=NAVY, label="Workstation")
    ax.bar(x + width / 2, lt, width, color=ORANGE, label="Laptop")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS])
    ax.set_ylabel("Inference time (s)")
    panel(ax, "B", "Native merged GPU  ·  580 stations")
    clean_axis(ax)

    ax = axes[1, 0]
    for model in MODELS:
        ws_vals, lt_vals, ks = [], [], []
        for k in (5, 10, 20):
            wr = load_json(base.orch_path("ma", "annotate_bf16", 580, model, "cpu", k, 0, "sg", "playback", "static"))
            lr = xfer_orch(model, "cpu", k)
            if wr is None or lr is None:
                continue
            ks.append(k)
            ws_vals.append(base.mk(wr))
            lt_vals.append(base.mk(lr))
        ax.plot(ks, ws_vals, marker="o", ms=3.6, lw=1.2, color=MODEL_COLORS[model], label=SHORT[model])
        ax.plot(ks, lt_vals, marker="s", ms=3.4, lw=1.0, ls="--", color=MODEL_COLORS[model])
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 20])
    ax.set_xlabel("CPU inference instances  K")
    ax.set_ylabel("Playback makespan (s, log)")
    panel(ax, "C", "Model-Actor SG CPU  ·  580 stations")
    clean_axis(ax)
    ax.legend(ncol=2, loc="upper right")
    ax.text(0.02, 0.04, "solid workstation   dashed laptop", transform=ax.transAxes, fontsize=6.8, color=MUTED)

    ax = axes[1, 1]
    for model in MODELS:
        ws_vals, lt_vals, ks = [], [], []
        for k in (5, 10, 20):
            wr = load_json(base.orch_path("ma", "annotate_bf16", 580, model, "cpu", k, 0, "sg", "playback", "static"))
            lr = xfer_orch(model, "cpu", k)
            if wr is None or lr is None:
                continue
            ks.append(k)
            ws_vals.append(pss_gb(wr))
            lt_vals.append(pss_gb(lr))
        ax.plot(ks, ws_vals, marker="o", ms=3.6, lw=1.2, color=MODEL_COLORS[model], label=SHORT[model])
        ax.plot(ks, lt_vals, marker="s", ms=3.4, lw=1.0, ls="--", color=MODEL_COLORS[model])
    ax.set_xticks([5, 10, 20])
    ax.set_xlabel("CPU inference instances  K")
    ax.set_ylabel("Peak PSS (GB)")
    panel(ax, "D", "Host memory  ·  Model-Actor SG CPU")
    clean_axis(ax)
    save(fig, "fig10_transfer")


def write_captions() -> None:
    text = """# Redesigned paper figure captions

Use the PDF files for publication and PNG files for drafting. If seven
main-text figures are required, replace Figures 7 and 8 with
`fig07_combined_scaling_native`.

**Figure 1.** SeisBench Annotate workflow and reduced-precision variants.
Windowing, batching, and reconstruction are unchanged; only the forward-pass
dtype differs. Discrete picks were aggregated offline and excluded from timed
Annotate inference.

**Figure 2.** Reduced-precision performance. (A, B) Median matched-cell speedup
relative to FP32, with interquartile ranges, by model and device. Values above
one favor reduced precision. EQTransformer FP16 is incompatible with its
padding sentinel. (C) Batch-size sensitivity normalized within each matched
series; lines show medians and bands show interquartile ranges.

**Figure 3.** P- and S-phase pick agreement versus FP32 for the 580-station CPU
merged workload at batch 512. BF16 is effectively identical to FP32. FP16 is
model dependent and was not evaluated for EQTransformer models.

**Figure 4.** RAPID architecture. The network is decomposed into K groups of
approximately G = ceil(N/K) stations. Packaging (SG or S1) and worker lifecycle
(persistent Model-Actor or ephemeral Ripper) are independent choices. Streaming
uses a ready queue and eager dispatch to the next free actor.

**Figure 5.** Matched-cell effects of grouping and persistence under all-ready
arrival. Small points are matched configurations; vertical intervals span the
interquartile range and open circles mark medians. SG beats S1 in 166/180
Model-Actor cells; Model-Actor beats Ripper in 116/120 matched SG cells.

**Figure 6.** Streaming performance on 580 stations. The ~90 s makespan reflects
the latest simulated arrival; methods are ranked by pooled p95 finish-minus-ready.
Points are phase-picking models and open circles are medians. Eager SG remains
sub-second, while waiting and hybrid Ripper assignment add latency.

**Figure 7.** Model-Actor SG makespan versus K under a fixed 20-core allocation
or one GPU. Open circles mark the smallest K within 10% of minimum runtime.
Both 250- and 580-station workloads are shown.

**Figure 8.** Selected RAPID Model-Actor SG versus native SeisBench on 580
stations. (A, B) Raw runtime for native sequential, native merged, and RAPID.
(C) Speedup against native merged Annotate. Values above one favor RAPID.

**Figure 9.** Memory cost of native Annotate and RAPID orchestration on the
workstation, 580-station network. Peak proportional set size (PSS) is used for
host memory so shared pages among Ray workers are not counted once per process.
(A) Native merged CPU PSS by numerical precision. (B) Native merged GPU VRAM.
(C) CPU RAPID SG PSS versus K; circles are Model-Actor and crosses are Ripper.
(D) GPU VRAM versus K for the same comparison. Ripper peak occupancy matches
Model-Actor at overlapping K because K in-flight workers all hold models.

**Figure 10.** Locked-recipe transfer from the workstation (Threadripper PRO
7985WX, RTX 6000 Ada) to a consumer laptop (Intel Core i7-13700H, RTX 4050
Laptop 6 GB, WSL2 capped at 24 GB). (A, B) Native merged Annotate-BF16 on 580
stations. Absolute times are not comparable across hosts; GPU remains the faster
path on both. (C) Model-Actor SG CPU playback makespan versus K. On the laptop,
K=20 is often slower than K=10. (D) Corresponding peak PSS. Laptop K=20 remains
within the 24 GB WSL memory cap.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CAPTIONS.md").write_text(text)


def main() -> int:
    set_style()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    figure7_8_combined()
    figure9()
    figure10()
    write_captions()
    print(f"Wrote redesigned figures to {OUT}")
    for path in sorted(OUT.glob("*.png")):
        print(" ", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
