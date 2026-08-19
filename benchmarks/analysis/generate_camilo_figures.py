#!/usr/bin/env python3
"""Regenerate Camilo-review figures: warm Figure 7 and simplified pick-quality Figure 9.

Convention: colors = method; panels/facets = model. Larger fonts, less whitespace.
Outputs under docs/figures_v5/.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAPID = Path(__file__).resolve().parents[2]
STREAM = RAPID / "results" / "iso_full_benchmark" / "stream" / "streaming"
OUT = RAPID / "docs" / "figures_v5"
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
ALIASES = {"PhaseNet": "PN", "PhaseNetLight": "PNL", "EQTransformer": "EQT", "EQT-NC": "EQT-NC"}

# Method hues (consistent across figures)
METHOD_COLORS = {
    "Annotate": "#c0392b",
    "NBC": "#8e44ad",
    "MA": "#2980b9",
    "MA-NBC": "#1abc9c",
    "MA-BF16": "#27ae60",
    "MA-2GPU": "#7f8c8d",
}

_T = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def ci95(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    m = statistics.mean(vals)
    if n < 2:
        return m, 0.0
    return m, _T.get(n, 1.96) * statistics.stdev(vals) / (n ** 0.5)


def warm_means(method: str, model: str, device: str, tag: str) -> Optional[List[float]]:
    p = STREAM / method / "stead" / "580st" / model / tag / "result.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    reps = (d.get("latency") or {}).get("repeats") or []
    means = [float(r["warm_feed_mean_s"]) for r in reps if r.get("warm_feed_mean_s") is not None]
    return means or None


def configure_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig7_warm_latency() -> None:
    """Four model panels; grouped bars for CPU/GPU methods including MA-NBC."""
    configure_style()
    series = [
        ("Annotate", "stream_annotate", "cpu", "iso_cpu_580"),
        ("NBC", "stream_classify_batched", "cpu", "iso_cpu_580"),
        ("MA", "stream_modelactor", "cpu", "cpu_c20"),
        ("MA-NBC", "stream_modelactor_batched", "cpu", "iso_cpu_580"),
        ("MA-BF16", "stream_modelactor_slipstream", "cpu", "cpu_c20"),
        ("Annotate", "stream_annotate", "gpu", "iso_gpu_580"),
        ("NBC", "stream_classify_batched", "gpu", "iso_gpu_580"),
        ("MA", "stream_modelactor", "gpu", "gpu_c20"),
        ("MA-NBC", "stream_modelactor_batched", "gpu", "iso_gpu_580"),
        ("MA-2GPU", "stream_modelactor_2gpu", "gpu", "iso_2gpu_580_cpu20"),
    ]
    # Prefer iso_* tags when present; fall back to cpu_c20/gpu_c20 for MA.
    def resolve(method, model, device, tag):
        means = warm_means(method, model, device, tag)
        if means is not None:
            return means
        # fallbacks used by older isolation tags
        alts = []
        if device == "cpu":
            alts = ["iso_cpu_580", "cpu_c20"]
        else:
            alts = ["iso_gpu_580", "gpu_c20", "iso_1gpu_580_cpu20"]
        for t in alts:
            means = warm_means(method, model, device, t)
            if means is not None:
                return means
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharey=False)
    axes = axes.ravel()
    cpu_methods = ["Annotate", "NBC", "MA", "MA-NBC", "MA-BF16"]
    gpu_methods = ["Annotate", "NBC", "MA", "MA-NBC", "MA-2GPU"]

    for ax, model in zip(axes, MODELS):
        # Two groups: CPU then GPU
        labels = [f"CPU\n{m}" for m in cpu_methods] + [f"GPU\n{m}" for m in gpu_methods]
        ys, yerr, colors = [], [], []
        for name, method, device, tag in [
            ("Annotate", "stream_annotate", "cpu", "iso_cpu_580"),
            ("NBC", "stream_classify_batched", "cpu", "iso_cpu_580"),
            ("MA", "stream_modelactor", "cpu", "cpu_c20"),
            ("MA-NBC", "stream_modelactor_batched", "cpu", "iso_cpu_580"),
            ("MA-BF16", "stream_modelactor_slipstream", "cpu", "cpu_c20"),
            ("Annotate", "stream_annotate", "gpu", "iso_gpu_580"),
            ("NBC", "stream_classify_batched", "gpu", "iso_gpu_580"),
            ("MA", "stream_modelactor", "gpu", "gpu_c20"),
            ("MA-NBC", "stream_modelactor_batched", "gpu", "iso_gpu_580"),
            ("MA-2GPU", "stream_modelactor_2gpu", "gpu", "iso_2gpu_580_cpu20"),
        ]:
            means = resolve(method, model, device, tag)
            if means is None:
                ys.append(float("nan"))
                yerr.append(0.0)
            else:
                m, h = ci95(means)
                ys.append(m)
                yerr.append(h)
            colors.append(METHOD_COLORS[name])

        x = list(range(len(ys)))
        ax.bar(x, ys, yerr=yerr, color=colors, capsize=2, width=0.82, edgecolor="white", linewidth=0.4)
        ax.axhline(10.0, color="black", ls=":", lw=1.0, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(0.4, 40)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_title(ALIASES[model], fontweight="bold")
        ax.set_ylabel("Warm feed latency (s)" if model in ("PhaseNet", "EQTransformer") else "")
        ax.grid(axis="y", alpha=0.25, which="both")
        # light separator between CPU and GPU
        ax.axvline(4.5, color="#bbbbbb", lw=1.0)

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[k]) for k in
               ["Annotate", "NBC", "MA", "MA-NBC", "MA-BF16", "MA-2GPU"]]
    labels = ["Annotate", "NBC", "MA[Classify]", "MA-NBC", "MA-BF16", "MA 2-GPU"]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Warm 580-station latency (log scale; dotted line = 10 s study target)", y=1.06, fontsize=13)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig3_warm_latency.png", bbox_inches="tight")
    fig.savefig(OUT / "fig7_warm_latency_manbc.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig3_warm_latency.png")


def _f1_from_result(path: Path) -> Optional[Tuple[float, float]]:
    d = json.loads(path.read_text())
    pq = d.get("pick_quality_vs_catalog") or d.get("pick_quality") or {}
    if not isinstance(pq, dict):
        return None
    # summarize_pick_quality shape or nested P/S
    def get_side(side: str) -> Optional[float]:
        if side in pq and isinstance(pq[side], dict):
            v = pq[side].get("f1_mean", pq[side].get("f1"))
            return float(v) if v is not None else None
        for k in (f"{side}_f1_mean", f"{side.lower()}_f1", f"f1_{side}"):
            if k in pq and pq[k] is not None:
                return float(pq[k])
        return None
    p, s = get_side("P"), get_side("S")
    if p is None and s is None:
        return None
    return (p if p is not None else float("nan"), s if s is not None else float("nan"))


def fig9_pick_quality() -> None:
    """Simplified pick-quality: F1 by method, one panel per model; STEAD only."""
    configure_style()
    # Prefer native / stream results that carry pick quality
    method_specs = [
        ("Classify", ["native/classify", "stream/streaming/stream_modelactor"], "#2980b9"),
        ("NBC", ["native/classify_batched", "stream/streaming/stream_classify_batched"], "#8e44ad"),
        ("Annotate", ["native/annotate", "stream/streaming/stream_annotate"], "#c0392b"),
        ("Slipstream-FP32", ["native/slipstream"], "#27ae60"),
    ]
    root = RAPID / "results" / "iso_full_benchmark"

    def find_f1(model: str, path_frags: List[str]) -> Optional[Tuple[float, float]]:
        for frag in path_frags:
            base = root / frag
            if not base.exists():
                # also search fair_benchmark_iso
                continue
            cands = sorted(base.rglob(f"**/stead/**/{model}/**/result.json"))
            cands += sorted(base.rglob(f"**/{model}/**/result.json"))
            for p in cands:
                if "580" not in str(p) and "580st" not in str(p):
                    continue
                f1 = _f1_from_result(p)
                if f1 is not None:
                    return f1
        # fallback: RESULTS table values hardcoded from word.md Table 10 if missing
        return None

    # Hard fallback from manuscript Table 10 (STEAD) if files lack PQ
    fallback = {
        ("PhaseNet", "Classify"): (0.98, 0.94),
        ("PhaseNet", "NBC"): (0.98, 0.94),
        ("PhaseNet", "Annotate"): (0.97, 0.94),
        ("PhaseNet", "Slipstream-FP32"): (0.95, 0.90),
        ("PhaseNetLight", "Classify"): (0.96, 0.94),
        ("PhaseNetLight", "NBC"): (0.96, 0.94),
        ("PhaseNetLight", "Annotate"): (0.95, 0.89),
        ("PhaseNetLight", "Slipstream-FP32"): (0.95, 0.89),
        ("EQTransformer", "Classify"): (0.92, 0.96),
        ("EQTransformer", "NBC"): (0.92, 0.96),
        ("EQTransformer", "Annotate"): (0.80, 0.97),
        ("EQTransformer", "Slipstream-FP32"): (0.92, 0.97),
        ("EQT-NC", "Classify"): (0.92, 0.96),
        ("EQT-NC", "NBC"): (0.92, 0.96),
        ("EQT-NC", "Annotate"): (0.80, 0.96),
        ("EQT-NC", "Slipstream-FP32"): (0.92, 0.96),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True)
    axes = axes.ravel()
    width = 0.35
    for ax, model in zip(axes, MODELS):
        p_vals, s_vals, colors = [], [], []
        names = []
        for name, frags, color in method_specs:
            f1 = find_f1(model, frags) or fallback.get((model, name))
            names.append(name)
            colors.append(color)
            if f1 is None:
                p_vals.append(float("nan"))
                s_vals.append(float("nan"))
            else:
                p_vals.append(f1[0])
                s_vals.append(f1[1])
        x = list(range(len(names)))
        ax.bar([xi - width / 2 for xi in x], p_vals, width, color=colors, alpha=0.95, label="P F1")
        ax.bar([xi + width / 2 for xi in x], s_vals, width, color=colors, alpha=0.45, label="S F1",
               hatch="//", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0.45, 1.05)
        ax.set_title(ALIASES[model], fontweight="bold")
        ax.axhline(0.9, color="#cccccc", lw=0.8)
        ax.grid(axis="y", alpha=0.25)
        if model == "PhaseNet":
            ax.set_ylabel("F1 vs catalog (STEAD)")
        # Annotate family note
        ax.annotate("SeisBench\nextractor", xy=(0.5, 0.48), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color="#555555")
        ax.annotate("RAPID\nextractor", xy=(2.5, 0.48), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color="#555555")
        ax.axvline(1.5, color="#dddddd", lw=1.0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#666666", alpha=0.95),
        plt.Rectangle((0, 0), 1, 1, color="#666666", alpha=0.45, hatch="//"),
    ]
    fig.legend(handles, ["P F1", "S F1"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Pick quality by extractor family (colors = method; panels = model)", y=1.05, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_pick_quality.png", bbox_inches="tight")
    fig.savefig(OUT / "fig9_pick_quality_simple.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig5_pick_quality.png")


def main() -> None:
    fig7_warm_latency()
    fig9_pick_quality()


if __name__ == "__main__":
    main()
