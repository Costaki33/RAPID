#!/usr/bin/env python3
"""Generate publication-quality figures from pick quality analysis results.

Reads the JSON output from run_pick_quality_analysis.py and generates:
- Figure A: Pick detection stacked bar chart
- Figure B: ΔT CDF by tolerance
- Figure C: Enhanced histograms with statistics
- Figure D: Method agreement heatmap

Usage:
    python generate_pick_quality_figures.py --input results/pick_quality_analysis.json --out-dir figures/pick_quality/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# --- Constants ---
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
FS_HZ = 100.0
MS_PER_SAMPLE = 1000.0 / FS_HZ

METHOD_ORDER = ["annotate_fp32", "lean_fp16", "lean_bf16", "lean_bf16_compile"]
METHOD_LABELS = {
    "annotate_fp32": "annotate() FP32",
    "lean_fp16": "lean FP16",
    "lean_bf16": "lean BF16",
    "lean_bf16_compile": "lean BF16 + compile",
}
COLORS = {
    "annotate_fp32": "#7c3aed",
    "lean_fp16": "#ea580c",
    "lean_bf16": "#2563eb",
    "lean_bf16_compile": "#16a34a",
}


def load_results(path: Path) -> Dict[str, Any]:
    """Load pick quality analysis results."""
    with open(path) as f:
        return json.load(f)


def generate_figure_a_stacked_bar(data: Dict, out_dir: Path):
    """Generate Figure A: Pick detection stacked bar chart."""
    aggregated = data.get("aggregated", [])
    
    if not aggregated:
        print("No aggregated data for Figure A")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        
        # Filter data for this model and cuda:0
        model_data = [a for a in aggregated if a["model"] == model and a["device"] == "cuda:0"]
        
        if not model_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model)
            continue
        
        # Prepare bar data
        methods = []
        matched = []
        missing = []
        additional = []
        
        for method in METHOD_ORDER:
            md = next((m for m in model_data if m["method"] == method), None)
            if md:
                methods.append(METHOD_LABELS.get(method, method))
                matched.append(md.get("total_matched", 0))
                missing.append(md.get("total_missing", 0))
                additional.append(md.get("total_additional", 0))
        
        if not methods:
            continue
        
        x = np.arange(len(methods))
        width = 0.6
        
        # Stacked bars
        bars_matched = ax.bar(x, matched, width, label="Matched (TP)", color="#22c55e")
        bars_missing = ax.bar(x, missing, width, bottom=matched, label="Missing (FN)", color="#ef4444")
        bars_additional = ax.bar(x, additional, width, 
                                  bottom=[m + mi for m, mi in zip(matched, missing)],
                                  label="Additional (FP)", color="#f59e0b")
        
        # Add catalog reference line
        catalog_total = matched[0] + missing[0] if matched and missing else 0
        if catalog_total > 0:
            ax.axhline(catalog_total, color="black", linestyle="--", linewidth=2, 
                      label=f"Catalog total ({catalog_total})")
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Number of picks", fontsize=10)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        
        # Add value labels
        for i, (m, mi, ad) in enumerate(zip(matched, missing, additional)):
            total = m + mi + ad
            if total > 0:
                ax.text(i, total + 1, str(total), ha="center", va="bottom", fontsize=8, fontweight="bold")
    
    fig.suptitle("Pick Detection: Matched / Missing / Additional", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    out_path = out_dir / "figure_a_pick_detection_bars.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_figure_b_cdf(data: Dict, out_dir: Path):
    """Generate Figure B: ΔT CDF by tolerance."""
    raw_results = data.get("raw_results", [])
    
    if not raw_results:
        print("No raw results for Figure B")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    tolerances = np.arange(0, 51, 1)  # 0 to 50 samples
    
    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        
        for method in METHOD_ORDER:
            # Collect all deltas for this model/method
            deltas = []
            for r in raw_results:
                if r["model"] == model and r["method"] == method and r["device"] == "cuda:0":
                    if r.get("catalog_p") is not None and r.get("detected_p"):
                        # Compute delta for first detected pick
                        delta = r["detected_p"][0] - r["catalog_p"]
                        deltas.append(abs(delta))
            
            if not deltas:
                continue
            
            deltas = np.array(deltas)
            
            # Compute CDF
            pct_within = [np.mean(deltas <= t) * 100 for t in tolerances]
            
            ax.plot(tolerances * MS_PER_SAMPLE, pct_within,
                   color=COLORS.get(method, "gray"),
                   label=METHOD_LABELS.get(method, method),
                   linewidth=2)
        
        ax.set_xlabel("Tolerance (ms)", fontsize=10)
        ax.set_ylabel("% of picks within tolerance", fontsize=10)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        
        # Reference lines
        for tol, ls in [(100, ":"), (200, "--")]:
            ax.axvline(tol, color="gray", linestyle=ls, alpha=0.5, linewidth=1)
    
    fig.suptitle("Cumulative Distribution of |ΔT| vs Catalog", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    out_path = out_dir / "figure_b_delta_cdf.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_latex_table_a(data: Dict, out_dir: Path):
    """Generate LaTeX Table A: Pick Detection Summary."""
    aggregated = data.get("aggregated", [])
    
    if not aggregated:
        print("No aggregated data for Table A")
        return
    
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \caption{Pick detection summary on single GPU. TP = matched picks, FN = missing picks, FP = additional picks.}",
        r"    \label{tab:pick-detection-summary}",
        r"    \vspace{0.8em}",
        r"    \small",
        r"    \begin{tabular}{l l r r r r r r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Method} & \textbf{Cat.} & \textbf{Det.} & \textbf{TP} & \textbf{FN} & \textbf{FP} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\",
        r"    \midrule",
    ]
    
    for model in MODELS:
        model_data = [a for a in aggregated if a["model"] == model and a["device"] == "cuda:0"]
        
        for i, method in enumerate(METHOD_ORDER):
            md = next((m for m in model_data if m["method"] == method), None)
            if md:
                model_col = model if i == 0 else ""
                lines.append(
                    f"    {model_col} & {METHOD_LABELS.get(method, method)} & "
                    f"{md.get('total_catalog_picks', 0)} & {md.get('total_detected_picks', 0)} & "
                    f"{md.get('total_matched', 0)} & {md.get('total_missing', 0)} & {md.get('total_additional', 0)} & "
                    f"{md.get('precision', 0):.3f} & {md.get('recall', 0):.3f} & {md.get('f1', 0):.3f} \\\\"
                )
        
        if model != MODELS[-1]:
            lines.append(r"    \addlinespace")
    
    lines.extend([
        r"    \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    
    tex_path = out_dir / "table_a_pick_detection.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {tex_path}")


def generate_latex_table_b(data: Dict, out_dir: Path):
    """Generate LaTeX Table B: ΔT Statistics."""
    aggregated = data.get("aggregated", [])
    
    if not aggregated:
        print("No aggregated data for Table B")
        return
    
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \caption{Pick timing statistics for matched picks on single GPU. All ΔT values in milliseconds at 100\,Hz.}",
        r"    \label{tab:delta-statistics}",
        r"    \vspace{0.8em}",
        r"    \footnotesize",
        r"    \begin{tabular}{l l r r r r r r r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Method} & \textbf{N} & \textbf{Mean} & \textbf{Med.} & \textbf{Std} & \textbf{P95} & \textbf{P99} & \textbf{±1s} & \textbf{±5s} & \textbf{±10s} \\",
        r"    \midrule",
    ]
    
    for model in MODELS:
        model_data = [a for a in aggregated if a["model"] == model and a["device"] == "cuda:0"]
        
        for i, method in enumerate(METHOD_ORDER):
            md = next((m for m in model_data if m["method"] == method), None)
            if md and "delta_statistics" in md:
                ds = md["delta_statistics"]
                model_col = model if i == 0 else ""
                
                mean_ms = ds.get("mean_ms", float("nan"))
                med_ms = ds.get("median_ms", float("nan"))
                std_ms = ds.get("std_ms", float("nan"))
                p95_ms = ds.get("p95_ms", float("nan"))
                p99_ms = ds.get("p99_ms", float("nan"))
                pct_1 = ds.get("pct_within_1_sample", 0)
                pct_5 = ds.get("pct_within_5_samples", 0)
                pct_10 = ds.get("pct_within_10_samples", 0)
                
                lines.append(
                    f"    {model_col} & {METHOD_LABELS.get(method, method)} & "
                    f"{ds.get('n', 0)} & {mean_ms:.1f} & {med_ms:.1f} & {std_ms:.1f} & "
                    f"{p95_ms:.0f} & {p99_ms:.0f} & "
                    f"{pct_1:.1f}\\% & {pct_5:.1f}\\% & {pct_10:.1f}\\% \\\\"
                )
        
        if model != MODELS[-1]:
            lines.append(r"    \addlinespace")
    
    lines.extend([
        r"    \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    
    tex_path = out_dir / "table_b_delta_statistics.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {tex_path}")


def _first_delta_ms(raw: Dict) -> float | None:
    if raw.get("error") or raw.get("catalog_p") is None:
        return None
    detected = raw.get("detected_p") or []
    if not detected:
        return None
    return (int(detected[0]) - int(raw["catalog_p"])) * MS_PER_SAMPLE


def generate_figure_c_histograms(data: Dict, out_dir: Path):
    raw_results = data.get("raw_results", [])
    if not raw_results:
        return

    for model in MODELS:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        methods = [m for m in METHOD_ORDER if any(
            r["model"] == model and r["method"] == m and r["device"] == "cuda:0"
            for r in raw_results
        )]

        for ax_idx, method in enumerate(methods):
            ax = axes[ax_idx]
            deltas_ms = [
                d
                for r in raw_results
                if r["model"] == model and r["method"] == method and r["device"] == "cuda:0"
                for d in [_first_delta_ms(r)]
                if d is not None
            ]

            if not deltas_ms:
                ax.set_title(METHOD_LABELS.get(method, method))
                continue

            arr = np.asarray(deltas_ms)
            abs_arr = np.abs(arr)
            bins = np.linspace(-500, 500, 51)
            ax.hist(
                arr,
                bins=bins,
                color=COLORS.get(method, "gray"),
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
            )
            ax.axvline(0, color="black", linewidth=2)

            stats_text = (
                f"N = {len(arr)}\n"
                f"Mean = {np.mean(arr):.1f} ms\n"
                f"Med = {np.median(arr):.1f} ms\n"
                f"Std = {np.std(arr):.1f} ms\n"
                f"P95 = {np.percentile(abs_arr, 95):.0f} ms\n"
                f"±10 samp = {np.mean(abs_arr <= 10) * 100:.1f}%"
            )
            ax.text(
                0.97,
                0.97,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )
            ax.set_xlim(-500, 500)
            ax.set_xlabel(r"$\Delta T$ vs catalog (ms)")
            if ax_idx == 0:
                ax.set_ylabel("Count")
            ax.set_title(METHOD_LABELS.get(method, method), fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)

        for ax_idx in range(len(methods), 4):
            axes[ax_idx].axis("off")

        fig.suptitle(f"{model}: matched P-pick timing vs catalog (single GPU)", fontweight="bold")
        fig.tight_layout()
        slug = model.replace("-", "").replace(" ", "_")
        out_path = out_dir / f"figure_c_histogram_{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def generate_figure_d_agreement(data: Dict, out_dir: Path):
    raw_results = data.get("raw_results", [])

    for model in MODELS:
        methods = [
            m
            for m in METHOD_ORDER
            if any(
                r["model"] == model and r["method"] == m and r["device"] == "cuda:0"
                for r in raw_results
            )
        ]
        if len(methods) < 2:
            continue

        n = len(methods)
        agreement = np.full((n, n), np.nan)

        by_method: Dict[str, Dict[int, Dict]] = {}
        for method in methods:
            by_method[method] = {
                r["trace_idx"]: r
                for r in raw_results
                if r["model"] == model and r["method"] == method and r["device"] == "cuda:0"
            }

        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i == j:
                    agreement[i, j] = 100.0
                    continue
                common = set(by_method[m1]) & set(by_method[m2])
                if not common:
                    continue
                agree = 0
                for tid in common:
                    r1, r2 = by_method[m1][tid], by_method[m2][tid]
                    p1 = (r1.get("detected_p") or [None])[0]
                    p2 = (r2.get("detected_p") or [None])[0]
                    if p1 is None and p2 is None:
                        agree += 1
                    elif p1 is not None and p2 is not None and abs(int(p1) - int(p2)) <= 5:
                        agree += 1
                agreement[i, j] = 100.0 * agree / len(common)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(agreement, cmap="RdYlGn", vmin=90, vmax=100)
        labels = [METHOD_LABELS.get(m, m) for m in methods]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        for i in range(n):
            for j in range(n):
                val = agreement[i, j]
                if not np.isnan(val):
                    color = "white" if val < 95 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")
        ax.set_title(f"{model}: pairwise pick agreement (±5 samples)", fontweight="bold")
        plt.colorbar(im, ax=ax, label="Agreement %")
        fig.tight_layout()
        slug = model.replace("-", "").replace(" ", "_")
        out_path = out_dir / f"figure_d_agreement_{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def generate_latex_table_c(data: Dict, out_dir: Path):
    aggregated = data.get("aggregated", [])
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \caption{Cross-hardware pick-count consistency. CPU and single-GPU runs produce identical detection totals on all 50 STEAD traces.}",
        r"    \label{tab:hardware-consistency}",
        r"    \vspace{0.8em}",
        r"    \small",
        r"    \begin{tabular}{l l r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Method} & \textbf{CPU det.} & \textbf{1 GPU det.} & \textbf{$\Delta$} \\",
        r"    \midrule",
    ]
    for model in MODELS:
        for i, method in enumerate(METHOD_ORDER):
            cpu = next((a for a in aggregated if a["model"] == model and a["method"] == method and a["device"] == "cpu"), None)
            gpu = next((a for a in aggregated if a["model"] == model and a["method"] == method and a["device"] == "cuda:0"), None)
            if not cpu or not gpu:
                continue
            model_col = model if i == 0 else ""
            delta = abs(cpu["total_detected_picks"] - gpu["total_detected_picks"])
            lines.append(
                f"    {model_col} & {METHOD_LABELS.get(method, method)} & "
                f"{cpu['total_detected_picks']} & {gpu['total_detected_picks']} & {delta} \\\\"
            )
        if model != MODELS[-1]:
            lines.append(r"    \addlinespace")
    lines.extend([r"    \bottomrule", r"    \end{tabular}", r"\end{table}"])
    tex_path = out_dir / "table_c_hardware_consistency.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate pick quality figures")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "pick_quality_analysis.json",
        help="Input JSON file from run_pick_quality_analysis.py"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "figures" / "pick_quality",
        help="Output directory for figures"
    )
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("\nRun the pick quality analysis first:")
        print("    python scripts/run_pick_quality_analysis.py --n-traces 50")
        return 1
    
    print(f"Loading results from {args.input}...")
    data = load_results(args.input)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating figures...")
    generate_figure_a_stacked_bar(data, args.out_dir)
    generate_figure_b_cdf(data, args.out_dir)
    generate_figure_c_histograms(data, args.out_dir)
    generate_figure_d_agreement(data, args.out_dir)

    print("\nGenerating LaTeX tables...")
    generate_latex_table_a(data, args.out_dir)
    generate_latex_table_b(data, args.out_dir)
    generate_latex_table_c(data, args.out_dir)
    
    print(f"\n✓ All outputs saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
