#!/usr/bin/env python3
"""Publication figures for the RAPID Annotate-BF16 orchestration manuscript.

Writes PDF (vector) + PNG (300 dpi) under docs/figures/paper/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

RAPID = Path(__file__).resolve().parents[1]
PREC = RAPID / "results" / "annotate_precision" / "stead_iso_2026-08-13"
SEQ = RAPID / "results" / "annotate_precision" / "stead_seq_bf16_2026-08-13"
ORCH = RAPID / "results" / "orch_annotate" / "stead_2026-08-14"
OUT = RAPID / "docs" / "figures" / "paper"

MODELS = ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
SHORT = {
    "EQCCT": "EQCCT",
    "PhaseNet": "PN",
    "PhaseNetLight": "PNL",
    "EQTransformer": "EQT",
    "EQT-NC": "EQT-NC",
}
MCOL = {
    "EQCCT": "#0072B2",
    "PhaseNet": "#009E73",
    "PhaseNetLight": "#56B4E9",
    "EQTransformer": "#E69F00",
    "EQT-NC": "#D55E00",
}
EQT = {"EQTransformer", "EQT-NC"}
K_CPU = [1, 2, 4, 5, 10, 20]
K_GPU = [1, 2, 4]

C_FP32, C_BF16, C_FP16 = "#4D4D4D", "#0072B2", "#D55E00"
C_MA, C_RP = "#0072B2", "#D55E00"
C_SG, C_S1 = "#009E73", "#CC79A7"
C_W5, C_W10, C_HY = "#E69F00", "#56B4E9", "#7A7A7A"
C_NM, C_NS = "#4D4D4D", "#B0B0B0"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "semibold",
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def load(p: Path):
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def inf_mean(r) -> float | None:
    if not r:
        return None
    v = (r.get("timing") or {}).get("inference_s_mean")
    return float(v) if v is not None else None


def best_merged_inf(method, model, device, nst=580, bs=512) -> float | None:
    """Lowest mean inference among 5/10/15/20-core merged cells (skip thread explosions)."""
    best = None
    for c in (5, 10, 15, 20):
        v = inf_mean(load(prec_path(method, model, nst, device, c, bs)))
        if v is None or v > 100:
            continue
        best = v if best is None else min(best, v)
    return best


def best_seq_inf(model, device, nst=580) -> float | None:
    best = None
    for c in (5, 20):
        for bs in (256, 512):
            v = inf_mean(load(seq_path(model, nst, device, c, bs)))
            if v is None or v > 200:
                continue
            best = v if best is None else min(best, v)
    return best


def mk(r) -> float | None:
    if not r:
        return None
    v = (r.get("orch") or {}).get("makespan_s_mean")
    return float(v) if v is not None else None


def p95(r) -> float | None:
    if not r:
        return None
    v = (
        ((r.get("latency") or {}).get("pooled_across_repeats") or {})
        .get("e2e_finish_minus_ready")
        or {}
    ).get("p95")
    return float(v) if v is not None else None


def f1_summary(r, key: str, phase: str) -> float | None:
    if not r:
        return None
    block = r.get(key) or {}
    v = block.get(f"{phase}.f1_mean")
    if v is not None:
        return float(v)
    reps = block.get("repeats") or []
    xs = []
    for rep in reps:
        ph = rep.get(phase) or {}
        if ph.get("f1") is not None:
            xs.append(float(ph["f1"]))
    return float(np.mean(xs)) if xs else None


def prec_path(method, model, nst, device, cpus, bs, tag="ann_prec"):
    return (
        PREC
        / method
        / "stead"
        / f"{nst}st"
        / model
        / device
        / f"cpus{cpus}"
        / f"thr{cpus}"
        / f"bs{bs}"
        / tag
        / "result.json"
    )


def seq_path(model, nst, device, cpus, bs=512):
    return (
        SEQ
        / "annotate_bf16"
        / "stead"
        / f"{nst}st"
        / model
        / device
        / f"cpus{cpus}"
        / f"thr{cpus}"
        / f"bs{bs}"
        / "sequential"
        / "ann_seq"
        / "result.json"
    )


def orch_path(comp, method, nst, model, device, kma, krp, pack, arrival, fill):
    return (
        ORCH
        / comp
        / method
        / "stead"
        / f"{nst}st"
        / model
        / device
        / f"kma{kma}_krp{krp}"
        / pack
        / arrival
        / fill
        / "bs512"
        / "orch_ann"
        / "result.json"
    )


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


def _box(ax, xy, w, h, text, fc="#F4F4F4", ec="#333333", fontsize=7.5, bold=False):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=0.9,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="semibold" if bold else "normal",
        color="#1A1A1A",
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9, mutation_scale=9),
    )


def figure1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    _box(ax, (0.12, 1.45), 1.55, 1.3, "Network-scale\nObsPy Stream\n(N stations)", fc="#EEF6FB")
    _box(ax, (1.95, 0.95), 4.35, 2.45, "", fc="#F7F7F7", ec="#0072B2")
    ax.text(4.12, 3.18, "SeisBench Annotate", ha="center", fontsize=9, fontweight="semibold", color="#0072B2")
    _box(ax, (2.1, 2.15), 0.95, 0.72, "Window", fc="#FFFFFF")
    _box(ax, (3.2, 2.15), 0.95, 0.72, "Batch", fc="#FFFFFF")
    _box(ax, (4.3, 2.15), 0.85, 0.72, "Forward\npass", fc="#FFFFFF")
    _box(ax, (5.3, 2.15), 0.85, 0.72, "Reconstruct", fc="#FFFFFF")
    _arrow(ax, 3.05, 2.51, 3.2, 2.51)
    _arrow(ax, 4.15, 2.51, 4.3, 2.51)
    _arrow(ax, 5.15, 2.51, 5.3, 2.51)
    _arrow(ax, 1.67, 2.1, 1.95, 2.1)
    _box(ax, (6.55, 1.45), 1.4, 1.3, "Probability\ntraces\n(P, S, …)", fc="#EEF6FB")
    _arrow(ax, 6.3, 2.1, 6.55, 2.1)
    _box(ax, (8.15, 1.45), 1.7, 1.3, "Classify\n(offline here)\nDiscrete P/S", fc="#F4F0E8")
    _arrow(ax, 7.95, 2.1, 8.15, 2.1)

    # precision variants
    y0 = 0.18
    ax.text(0.15, 0.72, "Numerical precision (forward pass only)", fontsize=7.5, color="#333")
    _box(ax, (0.15, y0), 2.9, 0.45, "Annotate-FP32  (reference)", fc="#E8E8E8", ec=C_FP32, fontsize=7)
    _box(ax, (3.2, y0), 3.2, 0.45, "Annotate-BF16  (selected)", fc="#D6EAF8", ec=C_BF16, fontsize=7)
    _box(ax, (6.55, y0), 3.25, 0.45, "Annotate-FP16  (not general)", fc="#F8E0D2", ec=C_FP16, fontsize=7)
    ax.text(
        5.0,
        3.95,
        "Windowing, batching, and trace reconstruction are unchanged; only tensor dtype differs.",
        ha="center",
        fontsize=7,
        color="#444",
        style="italic",
    )
    save(fig, "fig01_annotate_pipeline")


def figure4() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(5, 9.7, "RAPID workload decomposition and model-instance management", ha="center", fontsize=10, fontweight="semibold")

    # N stations
    _box(ax, (0.25, 8.35), 2.2, 0.95, "N station\nStreams", fc="#EEF6FB")
    ax.text(3.55, 9.05, "G = ⌈N / K⌉", ha="center", fontsize=8.5, fontweight="semibold")
    _arrow(ax, 2.45, 8.85, 2.85, 8.85)
    _box(ax, (2.85, 8.35), 2.35, 0.95, "Round-robin\ninto K groups", fc="#F7F7F7")
    _arrow(ax, 5.2, 8.85, 5.65, 8.85)
    for i, lab in enumerate(["Instance 1", "Instance 2", "Instance K"]):
        _box(ax, (5.7 + i * 1.4, 8.4), 1.3, 0.85, f"{lab}\n~G stations", fc="#EAF4EA", fontsize=6.5)

    # SG vs S1
    ax.text(0.25, 7.95, "A. Packaging", fontsize=8.5, fontweight="semibold")
    _box(ax, (0.25, 6.15), 4.55, 1.65, "", fc="#F4FBF6", ec=C_SG)
    ax.text(2.52, 7.55, "SG  — grouped Annotate-BF16", ha="center", fontsize=8, fontweight="semibold", color="#00664A")
    ax.text(2.52, 6.85, "Merge assigned stations into one\nObsPy Stream → one Annotate call\n(batch size 512)", ha="center", fontsize=7)
    _box(ax, (5.2, 6.15), 4.55, 1.65, "", fc="#FBF4F8", ec=C_S1)
    ax.text(7.48, 7.55, "S1  — one station per call", ha="center", fontsize=8, fontweight="semibold", color="#8A3A70")
    ax.text(7.48, 6.85, "Each station is an independent\nAnnotate call (G = 1).\nGPU kernels are typically too small.", ha="center", fontsize=7)

    # Ripper vs MA
    ax.text(0.25, 5.75, "B. Persistence", fontsize=8.5, fontweight="semibold")
    _box(ax, (0.25, 3.55), 4.55, 2.05, "", fc="#FDF2EC", ec=C_RP)
    ax.text(2.52, 5.35, "Ripper  (ephemeral)", ha="center", fontsize=8, fontweight="semibold", color="#9C3D00")
    ax.text(2.52, 4.4, "load model → Annotate → terminate\nInitialization is inside makespan.\nEach additional worker repeats load.", ha="center", fontsize=7)
    _box(ax, (5.2, 3.55), 4.55, 2.05, "", fc="#EAF3FA", ec=C_MA)
    ax.text(7.48, 5.35, "Model-Actor  (persistent)", ha="center", fontsize=8, fontweight="semibold", color="#004C8A")
    ax.text(7.48, 4.4, "load + warm once (not in makespan)\n→ remain resident\n→ accept later groups from dispatcher", ha="center", fontsize=7)

    # Ready queue
    ax.text(0.25, 3.15, "C. Streaming dispatcher", fontsize=8.5, fontweight="semibold")
    _box(ax, (0.25, 0.25), 9.5, 2.75, "", fc="#FAFAFA", ec="#666")
    ax.text(5.0, 2.7, "All-ready: static round-robin.  Streaming: ready queue (next free instance takes next ready group).", ha="center", fontsize=7.5)
    xs = [1.1, 2.4, 3.7, 5.0, 6.3]
    labs = ["t = 0", "t = 15 s", "t = 30 s", "t = 65 s", "t = 90 s"]
    for x, lab in zip(xs, labs):
        _box(ax, (x - 0.45, 1.55), 0.95, 0.7, lab, fc="#EEF6FB", fontsize=6.2)
    _box(ax, (7.55, 1.45), 1.85, 0.9, "Ready\nqueue", fc="#FFF6D6", fontsize=7.5)
    _box(ax, (1.3, 0.4), 2.4, 0.7, "Actor A  (idle → take next)", fc="#D6EAF8", fontsize=6.5)
    _box(ax, (4.0, 0.4), 2.4, 0.7, "Actor B  (busy)", fc="#D6EAF8", fontsize=6.5)
    _box(ax, (6.7, 0.4), 2.4, 0.7, "Eager: send even if < G", fc="#EAF4EA", fontsize=6.5)
    save(fig, "fig04_rapid_architecture")


def figure2() -> None:
    nst, cpus, bs_sel = 580, 20, 512
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=False)

    x = np.arange(len(MODELS))
    w = 0.26
    for ax, device, title in ((axes[0], "cpu", "CPU  (fastest of 5–20 cores)"), (axes[1], "gpu", "GPU + 20 host cores")):
        for j, (method, color, lab, off) in enumerate(
            [
                ("annotate_fp32", C_FP32, "FP32", -w),
                ("annotate_bf16", C_BF16, "BF16", 0.0),
                ("annotate_fp16", C_FP16, "FP16", w),
            ]
        ):
            ys, hatch = [], []
            for m in MODELS:
                if method == "annotate_fp16" and m in EQT:
                    ys.append(np.nan)
                    hatch.append(True)
                    continue
                if device == "cpu":
                    v = best_merged_inf(method, m, device, nst, bs_sel)
                else:
                    v = inf_mean(load(prec_path(method, m, nst, device, cpus, bs_sel)))
                ys.append(v if v is not None else np.nan)
                hatch.append(False)
            bars = ax.bar(
                x + off,
                [0 if math.isnan(v) else v for v in ys],
                w,
                color=color,
                label=lab,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
            for i, m in enumerate(MODELS):
                if method == "annotate_fp16" and m in EQT:
                    ax.scatter(x[i] + off, 0.08, marker="x", s=28, color=C_FP16, zorder=4, linewidths=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_title(title)
        ax.set_ylabel("Mean Annotate inference (s)" if device == "cpu" else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCCCCC", zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].text(0.98, 0.95, "×  EQT/EQT-NC FP16 incompatible", transform=axes[1].transAxes, ha="right", va="top", fontsize=6.5, color=C_FP16)
    fig.suptitle("Merged-network Annotate, 580 stations, batch 512. CPU: fastest among 5–20 cores.", fontsize=8.5, y=1.02)
    save(fig, "fig02_precision_runtime")

    # batch-size companion panel as fig02b
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    batches = [64, 128, 256, 512]
    for ax, device, title in ((axes[0], "cpu", "CPU BF16"), (axes[1], "gpu", "GPU BF16")):
        for m in MODELS:
            ys = []
            for b in batches:
                if device == "cpu":
                    ys.append(best_merged_inf("annotate_bf16", m, device, 580, b))
                else:
                    ys.append(inf_mean(load(prec_path("annotate_bf16", m, 580, device, 20, b))))
            ax.plot(batches, ys, "-o", color=MCOL[m], label=SHORT[m], ms=4, lw=1.2)
        ax.axvline(512, color="#888", ls="--", lw=0.7, zorder=0)
        ax.set_xticks(batches)
        ax.set_xlabel("Batch size")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCCCCC")
        if device == "cpu":
            ax.set_ylabel("Mean inference (s)")
            ax.legend(frameon=False, ncol=1, loc="upper right")
    fig.suptitle("BF16 runtime versus Annotate batch size (580 stations, 20 cores)", fontsize=9, y=1.02)
    save(fig, "fig02b_batch_size")


def figure3() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    x = np.arange(len(MODELS))
    w = 0.34
    for ax, phase, title in ((axes[0], "P", "P-phase F1 versus FP32"), (axes[1], "S", "S-phase F1 versus FP32")):
        for method, color, lab, off in (
            ("annotate_bf16", C_BF16, "BF16", -w / 2),
            ("annotate_fp16", C_FP16, "FP16", w / 2),
        ):
            ys = []
            for m in MODELS:
                if method == "annotate_fp16" and m in EQT:
                    ys.append(np.nan)
                    continue
                r = load(prec_path(method, m, 580, "cpu", 20, 512))
                ys.append(f1_summary(r, "pick_quality_vs_fp32", phase) or np.nan)
            ax.bar(
                x + off,
                [0 if (isinstance(v, float) and math.isnan(v)) else v for v in ys],
                w,
                color=color,
                label=lab,
                edgecolor="white",
                linewidth=0.3,
            )
            for i, m in enumerate(MODELS):
                if method == "annotate_fp16" and m in EQT:
                    ax.scatter(x[i] + off, 0.05, marker="x", s=32, color=C_FP16, zorder=5, linewidths=1.2)
        ax.axhline(1.0, color="#888", ls=":", lw=0.7)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCCCCC")
        ax.set_axisbelow(True)
        if phase == "P":
            ax.set_ylabel("F1 versus merged-network FP32")
            ax.legend(frameon=False, loc="lower left")
    axes[1].text(0.98, 0.08, "×  FP16 not evaluated\n(EQT pad sentinel)", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=6.5, color=C_FP16)
    fig.suptitle("Pick agreement, 580-station CPU merged Annotate, batch 512", fontsize=9, y=1.02)
    save(fig, "fig03_pick_agreement")


def figure5() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    x = np.arange(len(MODELS))
    w = 0.2
    series = [
        ("ma", 20, 0, "sg", C_MA, "MA–SG  K=20", 0),
        ("ma", 20, 0, "s1", C_S1, "MA–S1  K=20", 1),
        ("ripper", 0, 5, "sg", C_RP, "Ripper–SG  K=5", 2),
        ("ripper", 0, 5, "s1", "#7A1F1F", "Ripper–S1  K=5", 3),
    ]
    gpu_series = [
        ("ma", 4, 0, "sg", C_MA, "MA–SG  K=4", 0),
        ("ma", 4, 0, "s1", C_S1, "MA–S1  K=4", 1),
        ("ripper", 0, 2, "sg", C_RP, "Ripper–SG  K=2", 2),
        ("ripper", 0, 1, "s1", "#7A1F1F", "Ripper–S1  K=1", 3),
    ]
    for ax, device, title, spec in (
        (axes[0], "cpu", "CPU, 580 stations", series),
        (axes[1], "gpu", "GPU, 580 stations", gpu_series),
    ):
        for comp, kma, krp, pack, color, lab, j in spec:
            ys = []
            for m in MODELS:
                r = load(orch_path(comp, "annotate_bf16", 580, m, device, kma, krp, pack, "playback", "static"))
                v = mk(r) if r else np.nan
                ys.append(v if v is not None else np.nan)
            offs = (j - 1.5) * w
            plotv = []
            for v in ys:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    plotv.append(np.nan)
                else:
                    plotv.append(v)
            ax.bar(x + offs, plotv, w, color=color, label=lab, edgecolor="white", lw=0.3, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_title(title)
        ax.set_yscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCCCCC", zorder=0)
        ax.set_axisbelow(True)
        if device == "cpu":
            ax.set_ylabel("Makespan (s, log)")
            ax.legend(frameon=False, fontsize=6.2, loc="upper right")
        ax.set_ylim(0.08, 4000)
    fig.suptitle("All-ready Annotate-BF16. Missing Ripper–S1 bars were not run (pathological control only for EQCCT/PN).", fontsize=8, y=1.03)
    save(fig, "fig05_allready_orchestration")


def _best_hybrid(model, device) -> float | None:
    splits = [(10, 2), (5, 5), (4, 1)] if device == "cpu" else [(2, 1), (1, 1)]
    best = None
    for pol in ("ma_ontime_rp_delayed", "rp_ontime_ma_delayed"):
        for kma, krp in splits:
            r = load(orch_path(pol, "annotate_bf16", 580, model, device, kma, krp, "sg", "staggered", "eager"))
            v = p95(r)
            if v is None:
                continue
            best = v if best is None else min(best, v)
    return best


def figure6() -> None:
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.35], hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_xlim(0, 100)
    ax0.set_ylim(0, 3.2)
    ax0.axis("off")
    ax0.text(0, 3.05, "A  Staggered arrivals and ranking metric", fontsize=9, fontweight="semibold")
    ax0.annotate("", xy=(92, 2.35), xytext=(2, 2.35), arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.9))
    ax0.text(92, 2.55, "t (s)", fontsize=7)
    for t, lab, y in ((0, "on time", 2.35), (15, "", 2.35), (30, "", 2.35), (65, "next\nminute", 2.35), (90, "delay\nceiling", 2.35)):
        ax0.plot(t, 2.35, "o", color="#0072B2", ms=6)
        ax0.text(t, 1.85 if t < 60 else 1.55, f"{t}s", ha="center", fontsize=6.5, color="#333")
    ax0.axvspan(60, 100, color="#F0F0F0", zorder=0)
    ax0.text(80, 2.9, "chunk 1", fontsize=6.5, color="#666")
    ax0.text(25, 2.9, "chunk 0  (60 s window)", fontsize=6.5, color="#666")
    ax0.text(
        50,
        0.55,
        "Makespan ≈ 90 s is the latest simulated arrival, not RAPID.\nRank streaming on p95 finish−ready (time from station-ready to picks).",
        ha="center",
        fontsize=7.5,
        color="#333",
    )

    x = np.arange(len(MODELS))
    w = 0.16
    specs = [
        ("Eager SG", C_MA, "ma", "sg", "eager"),
        ("S1", C_S1, "ma", "s1", "eager"),
        ("Wait-5", C_W5, "ma", "sg", "w5"),
        ("Wait-10", C_W10, "ma", "sg", "w10"),
        ("Hybrid (best)", C_HY, None, None, None),
    ]
    for ax, device, k, title in (
        (fig.add_subplot(gs[1, 0]), "cpu", 20, "B  CPU  K = 20"),
        (fig.add_subplot(gs[1, 1]), "gpu", 4, "C  GPU  K = 4"),
    ):
        for j, (lab, color, comp, pack, fill) in enumerate(specs):
            ys = []
            for m in MODELS:
                if lab.startswith("Hybrid"):
                    ys.append(_best_hybrid(m, device) or np.nan)
                else:
                    r = load(orch_path("ma", "annotate_bf16", 580, m, device, k, 0, pack, "staggered", fill))
                    ys.append(p95(r) if r else np.nan)
            offs = (j - 2) * w
            ax.bar(x + offs, [0 if (isinstance(v, float) and math.isnan(v)) else v for v in ys], w, color=color, label=lab, edgecolor="white", lw=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=7)
        ax.set_title(title, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCCCCC")
        ax.set_axisbelow(True)
        ax.set_ylabel("p95 finish−ready (s)")
        if device == "cpu":
            ax.legend(frameon=False, fontsize=6, loc="upper right")
            ax.text(0.02, 0.97, "Wait-10 ≈ 5 s: groups fill before the 10 s cap", transform=ax.transAxes, fontsize=6, va="top", color="#555")
    fig.suptitle("Streaming, 580 stations, Annotate-BF16, Model-Actor unless noted", fontsize=9, y=0.99)
    save(fig, "fig06_streaming")


def figure7() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for ax, device, ks, title in (
        (axes[0], "cpu", K_CPU, "CPU  (fixed 20-core slot)"),
        (axes[1], "gpu", K_GPU, "GPU  (one device, 20 host cores)"),
    ):
        for m in MODELS:
            ys = []
            for k in ks:
                r = load(orch_path("ma", "annotate_bf16", 580, m, device, k, 0, "sg", "playback", "static"))
                ys.append(mk(r) if r else np.nan)
            ax.plot(ks, ys, "-o", color=MCOL[m], label=SHORT[m], ms=4.5, lw=1.25)
            finite = [(k, y) for k, y in zip(ks, ys) if y is not None and not math.isnan(y)]
            if not finite:
                continue
            best = min(y for _, y in finite)
            elbow = next(k for k, y in finite if y <= 1.10 * best)
            ye = dict(finite)[elbow]
            ax.scatter([elbow], [ye], s=42, facecolors="none", edgecolors=MCOL[m], linewidths=1.3, zorder=5)
        ax.set_xlabel("Inference instances  K")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCCCCC")
        if device == "cpu":
            ax.set_ylabel("All-ready makespan (s)")
            ax.legend(frameon=False, fontsize=6.5, loc="upper right")
            ax.set_xticks(K_CPU)
        else:
            ax.set_xticks(K_GPU)
        ax.text(0.03, 0.04, "○  smallest K within 10% of minimum", transform=ax.transAxes, fontsize=6.5, color="#444")
    fig.suptitle("Model-Actor SG, Annotate-BF16, 580 stations. K=20 is 20 one-thread actors, not 20 threads.", fontsize=8.5, y=1.03)
    save(fig, "fig07_instance_scaling")


def figure8() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    x = np.arange(len(MODELS))
    w = 0.26
    native_seq = {}
    native_mrg = {}
    rapid = {}
    for m in MODELS:
        for device in ("cpu", "gpu"):
            native_seq[(m, device)] = best_seq_inf(m, device)
            native_mrg[(m, device)] = best_merged_inf("annotate_bf16", m, device)
            k = 20 if device == "cpu" else 4
            rapid[(m, device)] = mk(load(orch_path("ma", "annotate_bf16", 580, m, device, k, 0, "sg", "playback", "static")))

    for ax, device, title in ((axes[0], "cpu", "CPU"), (axes[1], "gpu", "GPU")):
        for j, (lab, color, src) in enumerate(
            [
                ("Native sequential", C_NS, native_seq),
                ("Native merged", C_NM, native_mrg),
                ("RAPID MA–SG", C_MA, rapid),
            ]
        ):
            ys = [src[(m, device)] or np.nan for m in MODELS]
            ax.bar(x + (j - 1) * w, [0 if (isinstance(v, float) and math.isnan(v)) else v for v in ys], w, color=color, label=lab, edgecolor="white", lw=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_title(title + ("  RAPID K=20" if device == "cpu" else "  RAPID K=4"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCCCCC")
        ax.set_axisbelow(True)
        ax.set_yscale("log")
        if device == "cpu":
            ax.set_ylabel("Runtime (s, log)")
            ax.legend(frameon=False, loc="upper right")
        ax.set_ylim(0.08, 80)
    fig.suptitle("580-station Annotate-BF16. Native merged = fastest 5–20 core cell. Sequential = best 5/20-core cell.", fontsize=8, y=1.03)
    save(fig, "fig08_vs_native")


def figure7_8_combined() -> None:
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    # reuse scaling
    for col, device, ks, title in (
        (0, "cpu", K_CPU, "A  CPU scaling (fixed 20 cores)"),
        (1, "gpu", K_GPU, "B  GPU scaling (one device)"),
    ):
        ax = fig.add_subplot(gs[0, col])
        for m in MODELS:
            ys = []
            for k in ks:
                r = load(orch_path("ma", "annotate_bf16", 580, m, device, k, 0, "sg", "playback", "static"))
                ys.append(mk(r) if r else np.nan)
            ax.plot(ks, ys, "-o", color=MCOL[m], label=SHORT[m], ms=4, lw=1.2)
            finite = [(k, y) for k, y in zip(ks, ys) if y is not None and not math.isnan(y)]
            if finite:
                best = min(y for _, y in finite)
                elbow = next(k for k, y in finite if y <= 1.10 * best)
                ax.scatter([elbow], [dict(finite)[elbow]], s=38, facecolors="none", edgecolors=MCOL[m], linewidths=1.2, zorder=5)
        ax.set_xlabel("K")
        ax.set_title(title, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCC")
        ax.set_xticks(ks)
        if col == 0:
            ax.set_ylabel("Makespan (s)")
            ax.legend(frameon=False, fontsize=6, loc="upper right")

    x = np.arange(len(MODELS))
    w = 0.26
    native_seq, native_mrg, rapid = {}, {}, {}
    for m in MODELS:
        for device in ("cpu", "gpu"):
            native_seq[(m, device)] = best_seq_inf(m, device)
            native_mrg[(m, device)] = best_merged_inf("annotate_bf16", m, device)
            k = 20 if device == "cpu" else 4
            rapid[(m, device)] = mk(load(orch_path("ma", "annotate_bf16", 580, m, device, k, 0, "sg", "playback", "static")))
    for col, device, title in ((0, "cpu", "C  vs native CPU"), (1, "gpu", "D  vs native GPU")):
        ax = fig.add_subplot(gs[1, col])
        for j, (lab, color, src) in enumerate(
            [("Native sequential", C_NS, native_seq), ("Native merged", C_NM, native_mrg), ("RAPID MA–SG", C_MA, rapid)]
        ):
            ys = [src[(m, device)] or np.nan for m in MODELS]
            ax.bar(x + (j - 1) * w, [0 if (isinstance(v, float) and math.isnan(v)) else v for v in ys], w, color=color, label=lab, edgecolor="white", lw=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_title(title, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.5, color="#CCC")
        ax.set_yscale("log")
        ax.set_ylim(0.08, 80)
        if col == 0:
            ax.set_ylabel("Runtime (s, log)")
            ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Selected RAPID configuration: Model-Actor SG, Annotate-BF16, 580 stations", fontsize=9, y=0.99)
    save(fig, "fig07_08_combined")


def main() -> int:
    style()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    figure7_8_combined()
    print(f"wrote figures in {OUT}")
    for p in sorted(OUT.glob("fig*.png")):
        print(" ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
