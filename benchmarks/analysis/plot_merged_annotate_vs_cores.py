#!/usr/bin/env python3
"""Paper figures for merged-network Annotate (bf16 winner; fp32/fp16 optional).

The old overlay (5 models × 2 devices × 4 batches on one axis) was unreadable.
These figures split CPU and GPU, and separate the two questions:

  vs_cores  — batch locked at 512 (orch config). X = host cores.
  vs_batch  — cores locked at 5. X = batch size.

Layout for both: 2×2, rows = 250 / 580 stations, cols = CPU / GPU.
Color = model (fixed Okabe–Ito map). CPU = filled circle + solid line;
GPU = open triangle + dashed line. Y-scale is shared down each column
(250 vs 580 comparable; CPU and GPU keep separate scales so GPU is readable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/annotate_precision/stead_iso_2026-08-13/merged_plot_compact.json"
OUT_DIR = ROOT / "results/annotate_precision/figures"

MODEL_COLOR = {
    "EQCCT": "#0072B2",
    "PhaseNet": "#E69F00",
    "PhaseNetLight": "#009E73",
    "EQTransformer": "#CC79A7",
    "EQT-NC": "#56B4E9",
}
MODELS = list(MODEL_COLOR)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _series_style(device: str, color: str) -> dict:
    if device == "gpu":
        return dict(
            color=color, linestyle="--", marker="^", markersize=6.5,
            linewidth=1.6, markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.3,
        )
    return dict(
        color=color, linestyle="-", marker="o", markersize=5.5,
        linewidth=1.6, markerfacecolor=color, markeredgecolor=color,
    )


def _column_ymax(block: dict, device: str, key_fn) -> float:
    ymax = 0.0
    for st in ("250", "580"):
        panel = block.get(st) or {}
        for model in MODELS:
            ys = key_fn((panel.get(model) or {}).get(device) or {})
            if not ys:
                continue
            ymax = max(ymax, max(v for v in ys if v is not None))
    return ymax * 1.12 if ymax else 1.0


def _draw_panel(ax, xs, panel, device: str, key_fn, xlabel: str, title: str, ylim: float) -> None:
    for model in MODELS:
        if model not in panel:
            continue
        ys_raw = key_fn(panel[model].get(device) or {})
        if not ys_raw or all(v is None for v in ys_raw):
            continue
        y = [float("nan") if v is None else float(v) for v in ys_raw]
        ax.plot(xs, y, **_series_style(device, MODEL_COLOR[model]))
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - (xs[1] - xs[0]) * 0.35, xs[-1] + (xs[1] - xs[0]) * 0.35)
    ax.set_ylim(0, ylim)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.set_xlabel(xlabel)


def _legend(fig) -> None:
    model_handles = [
        mlines.Line2D([], [], color=MODEL_COLOR[m], marker="o", linewidth=1.6, label=m)
        for m in MODELS
    ]
    device_handles = [
        mlines.Line2D([], [], color="0.3", marker="o", linestyle="-",
                      markerfacecolor="0.3", markersize=7, label="CPU"),
        mlines.Line2D([], [], color="0.3", marker="^", linestyle="--",
                      markerfacecolor="white", markeredgecolor="0.3",
                      markersize=7, label="GPU"),
    ]
    fig.legend(
        handles=model_handles + device_handles,
        loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.02),
    )


def plot_cores(block: dict, dtype: str) -> Path:
    xs = [5, 10, 15, 20]
    key_fn = lambda d: d.get("512")
    ylims = {dev: _column_ymax(block, dev, key_fn) for dev in ("cpu", "gpu")}
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), sharex=True, sharey="col")
    for r, st in enumerate(("250", "580")):
        panel = block[st]
        for c, device in enumerate(("cpu", "gpu")):
            ax = axes[r][c]
            _draw_panel(
                ax, xs, panel, device, key_fn,
                xlabel="CPU cores allocated to the trial",
                title=f"{st} stations  ·  {device.upper()}  ·  batch 512",
                ylim=ylims[device],
            )
            if c == 0:
                ax.set_ylabel("Inference time (s)")
    _legend(fig)
    fig.text(
        0.01, 0.01,
        f"annotate_{dtype}  ·  merged-network SeisBench annotate() mean of 5 repeats  ·  "
        "batch locked at 512  ·  color = model  ·  CPU = circle/solid, GPU = triangle/dashed  ·  "
        "y-scale shared down each column (250 vs 580).  Source: stead_iso_2026-08-13.",
        fontsize=7, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"merged_annotate_{dtype}_vs_cores.pdf"
    png = OUT_DIR / f"merged_annotate_{dtype}_vs_cores.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def plot_batch(block: dict, dtype: str) -> Path:
    xs = [64, 128, 256, 512]
    key_fn = lambda d, _xs=xs: [d.get(str(b), [None, None, None, None])[0] for b in _xs]
    ylims = {dev: _column_ymax(block, dev, key_fn) for dev in ("cpu", "gpu")}
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), sharex=True, sharey="col")
    for r, st in enumerate(("250", "580")):
        panel = block[st]
        for c, device in enumerate(("cpu", "gpu")):
            ax = axes[r][c]
            _draw_panel(
                ax, xs, panel, device, key_fn,
                xlabel="Batch size",
                title=f"{st} stations  ·  {device.upper()}  ·  5 cores",
                ylim=ylims[device],
            )
            if c == 0:
                ax.set_ylabel("Inference time (s)")
    _legend(fig)
    fig.text(
        0.01, 0.01,
        f"annotate_{dtype}  ·  merged-network SeisBench annotate() mean of 5 repeats  ·  "
        "cores locked at 5  ·  color = model  ·  CPU = circle/solid, GPU = triangle/dashed  ·  "
        "y-scale shared down each column (250 vs 580).  Source: stead_iso_2026-08-13.",
        fontsize=7, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"merged_annotate_{dtype}_vs_batch.pdf"
    png = OUT_DIR / f"merged_annotate_{dtype}_vs_batch.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args()
    compact = json.loads(DATA.read_text())
    block = compact[args.dtype]
    _style()
    plot_cores(block, args.dtype)
    plot_batch(block, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
