"""Generate a small set of publication-style figures from ``results/matrix.jsonl``.

Run from ``RAPID/``::

    export PYTHONPATH=\"$PWD:$PWD/..:$PYTHONPATH\"
    python scripts/paper_figures.py --jsonl results/matrix.jsonl --out-dir figures/paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


def main() -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from rapid.analysis import load_results, speedup_vs_baseline

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/matrix.jsonl")
    ap.add_argument("--out-dir", default="figures/paper")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_results(args.jsonl, drop_env=True)
    skip = df.get("is_skipped_incompatible", False)
    valid = df[
        ~df["is_error"]
        & ~skip
        & df["wall_time_s"].notna()
        & (df["wall_time_s"] > 0)
    ].copy()

    models = sorted(valid["model_label"].unique())
    n_list = sorted(valid["n_stations"].unique())

    # ------------------------------------------------------------------
    # Fig 1 — Best speedup vs 1-GPU baseline at largest workload (580 stn)
    # ------------------------------------------------------------------
    sp = speedup_vs_baseline(
        valid,
        match_on=("model_label", "n_stations", "device"),
        metric="wall_time_s",
    )
    sp580 = sp[(sp["device"] == "cuda:0") & (sp["n_stations"] == 580)].copy()
    idx = sp580.groupby("model_label")["speedup_median"].idxmax()
    best580 = sp580.loc[idx].sort_values("speedup_median", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(best580))
    ax.bar(x, best580["speedup_median"].values, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(best580["model_label"].values, rotation=15, ha="right")
    ax.set_ylabel("Speedup vs SeisBench annotate() (1 GPU)\n(median wall time, 3 repeats)")
    ax.set_title("Peak RAPID speedup at 580 stations — cuda:0 (best variant per model)")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="parity")
    for i, (_, r) in enumerate(best580.iterrows()):
        ax.annotate(
            r["variant"].replace("lean_pytorch/", "")[:28] + ("…" if len(r["variant"]) > 32 else ""),
            xy=(i, r["speedup_median"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            rotation=35,
        )
    ax.legend(loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    p1 = out / "fig1_speedup_580_stations.png"
    fig.savefig(p1, dpi=200)
    fig.savefig(p1.with_suffix(".svg"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2 — Speedup vs n_stations (PhaseNet: baseline vs best worker)
    # ------------------------------------------------------------------
    sp0 = sp[sp["device"] == "cuda:0"].copy()
    pn = sp0[sp0["model_label"] == "PhaseNet"].copy()
    idx = pn.groupby(["n_stations"])["speedup_median"].idxmax()
    curve = pn.loc[idx].sort_values("n_stations")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["n_stations"], curve["speedup_median"], "o-", lw=2, ms=8, label="Best pooled preprocess (PhaseNet)")
    ax.axhline(1.0, color="gray", ls="--", label="annotate() parity")
    ax.set_xlabel("Number of stations")
    ax.set_ylabel("Speedup vs 1-GPU baseline (median)")
    ax.set_title("PhaseNet — pooled CPU preprocess + lean GPU scales with workload")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    p2 = out / "fig2_phase_net_speedup_vs_n_stations.png"
    fig.savefig(p2, dpi=200)
    fig.savefig(p2.with_suffix(".svg"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 3 — Dual-GPU pipelined vs 2-GPU baseline (580 stn)
    # ------------------------------------------------------------------
    sp2 = speedup_vs_baseline(
        valid,
        match_on=("model_label", "n_stations", "device"),
        metric="wall_time_s",
    )
    sp2 = sp2[sp2["device"].astype(str).str.contains("+", regex=False)]
    sp2 = sp2[(sp2["n_stations"] == 580) & (sp2["kind"] == "dual_gpu")]
    sp2 = sp2[sp2["variant"].str.contains("2gpu_cpu", na=False)]
    sp2 = sp2[~sp2["variant"].str.contains("2gpu_baseline", na=False)]
    idx = sp2.groupby("model_label")["speedup_median"].idxmax()
    best2 = sp2.loc[idx].sort_values("speedup_median", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(best2))
    ax.bar(x, best2["speedup_median"].values, color="darkseagreen", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(best2["model_label"].values, rotation=15, ha="right")
    ax.set_ylabel("Speedup vs 2×GPU SeisBench annotate()")
    ax.set_title("Pipelined dual-GPU lean (best n_cpu_workers_per_gpu) at 580 stations")
    ax.axhline(1.0, color="gray", ls="--")
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    p3 = out / "fig3_dual_gpu_speedup_580.png"
    fig.savefig(p3, dpi=200)
    fig.savefig(p3.with_suffix(".svg"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 4 — Wall time: baseline vs best worker (all models, 580 stn)
    # ------------------------------------------------------------------
    sub = valid[
        (valid["n_stations"] == 580)
        & (valid["device"] == "cuda:0")
        & (valid["kind"].isin(["baseline", "cpu_worker_sweep"]))
    ]
    rows = []
    for m in models:
        b = sub[(sub["model_label"] == m) & (sub["kind"] == "baseline")]
        w = sub[(sub["model_label"] == m) & (sub["kind"] == "cpu_worker_sweep")]
        if b.empty or w.empty:
            continue
        b_med = b["wall_time_s"].median()
        best = w.groupby("variant")["wall_time_s"].median().sort_values().iloc[0]
        best_v = w.groupby("variant")["wall_time_s"].median().sort_values().index[0]
        rows.append({"model": m, "baseline_s": b_med, "rapid_s": best, "variant": best_v})
    comp = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp))
    w = 0.35
    ax.bar(x - w / 2, comp["baseline_s"], w, label="SeisBench annotate()", color="coral")
    ax.bar(x + w / 2, comp["rapid_s"], w, label="RAPID (best pooled)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["model"].values, rotation=15, ha="right")
    ax.set_ylabel("Median wall time (s)")
    ax.set_title("580 stations — 1 GPU: baseline vs best CPU-pooled lean path")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    p4 = out / "fig4_walltime_baseline_vs_rapid_580.png"
    fig.savefig(p4, dpi=200)
    fig.savefig(p4.with_suffix(".svg"))
    plt.close(fig)

    print("Wrote:", p1, p2, p3, p4, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
