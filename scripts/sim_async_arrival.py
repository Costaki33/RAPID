#!/usr/bin/env python3
"""Async-arrival / partial-availability model: why annotate() cannot meet a
real-time budget even though its batch wall time fits in the window.

This is a first-order *model* grounded in MEASURED component times from the fair
benchmark (annotate batch wall time T_ann, and Model-Actor warm per-window
latency W). It is not a substitute for the empirical paced-feed test (designed
in the paper's Future Work) -- it isolates the structural argument:

  In real time, a network's stations do not all arrive at window close; data
  latency spreads arrivals over [0, L]. annotate() is a BATCH call -- to include
  a station you must wait for it and (re)run the whole batch. Model-Actor is a
  persistent warm pool -- each station is picked shortly after it ARRIVES, so
  compute overlaps arrival.

  * annotate, wait-for-all:  last pick ready at  L + T_ann   (latency cost)
  * annotate, budget-cutoff: run the batch by the deadline on whatever arrived;
                             coverage = fraction arrived by (deadline - T_ann)
  * Model-Actor (warm):      each station picked ~immediately on arrival;
                             last pick ready at ~L; coverage = fraction by deadline

The batch-compute T_ann is a latency annotate() pays AFTER waiting and CANNOT
overlap with arrival; Model-Actor overlaps it away. The gap is exactly T_ann.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures_v3"
OUT.mkdir(parents=True, exist_ok=True)
BUDGET = 30.0   # real-time budget (s)

# Measured component times (s), STEAD 580 stations, from the fair benchmark.
# T_ann = annotate batch wall time; W = Model-Actor warm per-window latency.
MEAS = {
    #            T_ann_cpu  T_ann_gpu  W_cpu
    "PhaseNet":      (4.5,   4.3,   0.77),
    "PhaseNetLight": (7.5,   4.6,   0.73),
    "EQTransformer": (14.7,  4.6,   2.20),
    "EQT-NC":        (18.4,  4.6,   1.67),
}
N = 580


def end_to_end_latency(L, T_ann, W):
    """Time from window-close to the LAST pick being ready, vs arrival spread L."""
    # annotate must wait for the last station (t=L), then run the full batch.
    ann = L + T_ann
    # Model-Actor (warm): stations picked ~immediately on arrival; the last one
    # arrives at L and is picked within one per-station time (W/N, negligible vs L);
    # if the whole window arrived as a burst it would take W, so use max.
    ma = np.maximum(L, W) + W / N
    return ann, ma


def coverage_within_budget(L, T_ann, W):
    """Fraction of stations whose pick is ready within the 30 s budget."""
    L = np.asarray(L, dtype=float)
    # annotate, budget-cutoff: to finish the batch by the deadline it must START
    # by (BUDGET - T_ann); it can only include stations arrived by then.
    start_by = max(0.0, BUDGET - T_ann)
    ann = np.clip(start_by / np.maximum(L, 1e-9), 0, 1)
    # Model-Actor: a station arriving at a is done at ~a; covered if a <= BUDGET.
    ma = np.clip(BUDGET / np.maximum(L, 1e-9), 0, 1)
    return ann, ma


def main():
    Ls = np.linspace(0, 45, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: end-to-end latency vs arrival spread (EQTransformer CPU, the heavy case).
    ax = axes[0]
    for model, style in (("EQTransformer", "-"), ("PhaseNet", "--")):
        T_ann, _, W = MEAS[model]
        ann, ma = end_to_end_latency(Ls, T_ann, W)
        ax.plot(Ls, ann, style, color="#c0392b", label=f"annotate() — {model}")
        ax.plot(Ls, ma, style, color="#2e86c1", label=f"Model-Actor — {model}")
    ax.axhline(BUDGET, color="k", ls=":", lw=1)
    ax.text(1, BUDGET + 1, "30 s budget", fontsize=8)
    ax.set_xlabel("data-latency spread L (s) — stations arrive over [0, L]")
    ax.set_ylabel("time to last pick (s)")
    ax.set_title("End-to-end latency under async arrival\n(annotate pays the batch AFTER waiting; MA overlaps it)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 60)

    # Panel B: coverage achievable within the budget (EQTransformer CPU).
    ax = axes[1]
    for model, style in (("EQTransformer", "-"), ("PhaseNet", "--")):
        T_ann, _, W = MEAS[model]
        ann, ma = coverage_within_budget(Ls, T_ann, W)
        ax.plot(Ls, 100 * ann, style, color="#c0392b", label=f"annotate() — {model}")
        ax.plot(Ls, 100 * ma, style, color="#2e86c1", label=f"Model-Actor — {model}")
    ax.set_xlabel("data-latency spread L (s)")
    ax.set_ylabel("% stations picked within 30 s budget")
    ax.set_title("Coverage within budget\n(annotate must drop late stations to stay in time)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 105)

    fig.suptitle("Why annotate() cannot stream: latency-vs-throughput under realistic async arrival "
                 "(model grounded in measured T_ann + warm latency, STEAD 580 st, CPU)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_async_arrival.png", bbox_inches="tight")
    plt.close(fig)

    # Printed summary: at what arrival spread does each method bust the budget?
    print("Async-arrival model (STEAD 580 st, CPU):")
    print(f"{'model':14s} {'T_ann':>6s} {'W':>5s}  annotate busts 30s at L=   MA busts at L=   annotate cover@L=30s")
    for model in MEAS:
        T_ann, _, W = MEAS[model]
        ann_bust = max(0.0, BUDGET - T_ann)        # L beyond which L+T_ann > 30
        ma_bust = BUDGET - W / N                    # ~30
        cov_30 = min(1.0, max(0.0, BUDGET - T_ann) / 30.0)
        print(f"{model:14s} {T_ann:6.1f} {W:5.2f}        {ann_bust:5.1f} s              {ma_bust:5.1f} s          {100*cov_30:5.0f}%")
    print(f"\nWrote {OUT/'fig_async_arrival.png'}")


if __name__ == "__main__":
    main()
