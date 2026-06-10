#!/usr/bin/env python3
"""Analyze pick quality from existing JSONL benchmark results.

Generates:
- Table A: Pick Detection Summary (counts + precision/recall/F1)
- Table B: ΔT Statistics (mean, median, std, percentiles, tolerances)
- Table C: Cross-Hardware Consistency
- Figure A: Stacked bar chart of matched/missing/additional picks
- Figure B: ΔT CDF (cumulative distribution by tolerance)
- Figure C: Enhanced histograms with statistics insets
- Figure D: Method agreement heatmap

Usage:
    python analyze_pick_quality.py --jsonl-dir results/ --out-dir figures/pick_quality/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Constants ---
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
N_STATION_TIERS = (64, 256, 580)
FS_HZ = 100.0
MS_PER_SAMPLE = 1000.0 / FS_HZ

DEVICE_LABELS = {
    "cpu": "CPU",
    "cuda:0": "1 GPU",
    "cuda:0+cuda:1": "2 GPU",
}

METHOD_LABELS = {
    "annotate": "annotate() FP32",
    "fp16": "lean FP16",
    "bf16": "lean BF16",
    "bf16_compile": "lean BF16 + compile",
}

COLORS = {
    "annotate": "#7c3aed",
    "bf16": "#2563eb",
    "fp16": "#ea580c",
    "bf16_compile": "#16a34a",
}


def load_jsonl_files(jsonl_dir: Path) -> List[Dict]:
    """Load all JSONL benchmark files."""
    rows = []
    for p in sorted(jsonl_dir.glob("seisbench_matrix_lean*.jsonl")):
        if p.suffix == ".bak":
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "env":
                    continue
                rows.append(r)
    return rows


def classify_method(row: Dict) -> Optional[str]:
    """Classify row into method category."""
    runner = row.get("runner", "")
    dtype = row.get("dtype", "")
    compile_flag = bool((row.get("backend_extra") or {}).get("compile"))
    
    if runner in ("baseline_annotate", "baseline_annotate_dual"):
        return "annotate"
    elif runner in ("lean_pytorch", "lean_pytorch_dual_pipelined"):
        if dtype == "bf16":
            return "bf16_compile" if compile_flag else "bf16"
        elif dtype == "fp16":
            return "fp16" if not compile_flag else None
    return None


def get_device_category(device: str) -> str:
    """Categorize device string."""
    if device == "cpu":
        return "cpu"
    elif "cuda:0+cuda:1" in device or device == "cuda:0+cuda:1":
        return "cuda:0+cuda:1"
    elif "cuda" in device:
        return "cuda:0"
    return device


def extract_pick_data(rows: List[Dict]) -> pd.DataFrame:
    """Extract pick quality data from JSONL rows."""
    records = []
    
    for r in rows:
        if r.get("benchmark_status") == "skipped_incompatible":
            continue
        
        pq = r.get("pick_quality") or {}
        delta_p = pq.get("onset_delta_p_vs_catalog")
        onset_p = pq.get("onset_p")
        
        if delta_p is None:
            continue
        
        method = classify_method(r)
        if method is None:
            continue
        
        device = get_device_category(r.get("device", ""))
        model = r.get("model_label", "")
        n_stations = int(r.get("n_stations") or 0)
        
        records.append({
            "model": model,
            "method": method,
            "device": device,
            "n_stations": n_stations,
            "delta_p_samples": int(delta_p),
            "delta_p_ms": float(delta_p) * MS_PER_SAMPLE,
            "onset_p": onset_p,
            "catalog_p": r.get("p_catalog_in_window"),
            "trace_row": r.get("sb_trace_row"),
            "trial_uid": r.get("trial_uid"),
            "has_pick": onset_p is not None,
        })
    
    return pd.DataFrame(records)


def compute_delta_statistics(deltas: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive ΔT statistics."""
    if len(deltas) == 0:
        return {k: float("nan") for k in [
            "n", "mean", "median", "std", "p50", "p95", "p99",
            "mean_ms", "median_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms",
            "pct_1", "pct_5", "pct_10"
        ]}
    
    abs_deltas = np.abs(deltas)
    
    return {
        "n": len(deltas),
        "mean": float(np.mean(deltas)),
        "median": float(np.median(deltas)),
        "std": float(np.std(deltas)),
        "p50": float(np.percentile(abs_deltas, 50)),
        "p95": float(np.percentile(abs_deltas, 95)),
        "p99": float(np.percentile(abs_deltas, 99)),
        "mean_ms": float(np.mean(deltas) * MS_PER_SAMPLE),
        "median_ms": float(np.median(deltas) * MS_PER_SAMPLE),
        "std_ms": float(np.std(deltas) * MS_PER_SAMPLE),
        "p50_ms": float(np.percentile(abs_deltas, 50) * MS_PER_SAMPLE),
        "p95_ms": float(np.percentile(abs_deltas, 95) * MS_PER_SAMPLE),
        "p99_ms": float(np.percentile(abs_deltas, 99) * MS_PER_SAMPLE),
        "pct_1": float(np.mean(abs_deltas <= 1) * 100),
        "pct_5": float(np.mean(abs_deltas <= 5) * 100),
        "pct_10": float(np.mean(abs_deltas <= 10) * 100),
    }


def generate_table_b_delta_statistics(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Generate Table B: ΔT Statistics for all model/method combinations."""
    results = []
    
    for model in MODELS:
        for method in ["annotate", "fp16", "bf16", "bf16_compile"]:
            mask = (df["model"] == model) & (df["method"] == method) & (df["device"] == "cuda:0")
            subset = df[mask]
            
            if len(subset) == 0:
                continue
            
            deltas = subset["delta_p_samples"].values
            stats = compute_delta_statistics(deltas)
            
            results.append({
                "Model": model,
                "Method": METHOD_LABELS.get(method, method),
                "N": int(stats["n"]),
                "Mean ΔT (ms)": f"{stats['mean_ms']:.1f}",
                "Median ΔT (ms)": f"{stats['median_ms']:.1f}",
                "Std (ms)": f"{stats['std_ms']:.1f}",
                "P50 (ms)": f"{stats['p50_ms']:.0f}",
                "P95 (ms)": f"{stats['p95_ms']:.0f}",
                "P99 (ms)": f"{stats['p99_ms']:.0f}",
                "±1 samp": f"{stats['pct_1']:.1f}%",
                "±5 samp": f"{stats['pct_5']:.1f}%",
                "±10 samp": f"{stats['pct_10']:.1f}%",
            })
    
    result_df = pd.DataFrame(results)
    
    # Save as CSV
    csv_path = out_dir / "table_b_delta_statistics.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
    # Save as LaTeX
    tex_path = out_dir / "table_b_delta_statistics.tex"
    result_df.to_latex(tex_path, index=False, escape=False)
    print(f"Saved {tex_path}")
    
    return result_df


def generate_table_c_hardware_consistency(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Generate Table C: Cross-hardware consistency."""
    results = []
    
    for model in MODELS:
        for method in ["annotate", "bf16"]:
            counts = {}
            for device in ["cpu", "cuda:0", "cuda:0+cuda:1"]:
                mask = (df["model"] == model) & (df["method"] == method) & (df["device"] == device)
                counts[device] = len(df[mask])
            
            if sum(counts.values()) == 0:
                continue
            
            # Compute consistency (here we just show counts since we don't have full pick lists)
            count_list = [c for c in counts.values() if c > 0]
            max_diff = max(count_list) - min(count_list) if count_list else 0
            
            results.append({
                "Model": model,
                "Method": METHOD_LABELS.get(method, method),
                "CPU": counts.get("cpu", 0),
                "1 GPU": counts.get("cuda:0", 0),
                "2 GPU": counts.get("cuda:0+cuda:1", 0),
                "Max Δ Trials": max_diff,
            })
    
    result_df = pd.DataFrame(results)
    csv_path = out_dir / "table_c_hardware_consistency.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
    return result_df


def generate_figure_b_cdf(df: pd.DataFrame, out_dir: Path):
    """Generate Figure B: ΔT CDF (cumulative distribution by tolerance)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    tolerances = np.arange(0, 51, 1)  # 0 to 50 samples
    
    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        
        for method, color in COLORS.items():
            mask = (df["model"] == model) & (df["method"] == method) & (df["device"] == "cuda:0")
            deltas = df[mask]["delta_p_samples"].abs().values
            
            if len(deltas) == 0:
                continue
            
            # Compute CDF
            pct_within = [np.mean(deltas <= t) * 100 for t in tolerances]
            
            ax.plot(tolerances * MS_PER_SAMPLE, pct_within, 
                   color=color, label=METHOD_LABELS.get(method, method), linewidth=2)
        
        ax.set_xlabel("Tolerance (ms)", fontsize=10)
        ax.set_ylabel("% of picks within tolerance", fontsize=10)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        
        # Add vertical lines at key tolerances
        for tol, ls in [(10, ":"), (50, "--"), (100, "-.")]:
            ax.axvline(tol, color="gray", linestyle=ls, alpha=0.5, linewidth=1)
    
    fig.tight_layout()
    
    out_path = out_dir / "figure_b_delta_cdf.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_figure_c_enhanced_histograms(df: pd.DataFrame, out_dir: Path):
    """Generate Figure C: Enhanced histograms with statistics insets."""
    
    for model in MODELS:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        
        methods = ["annotate", "fp16", "bf16", "bf16_compile"]
        
        for ax_idx, method in enumerate(methods):
            ax = axes[ax_idx]
            
            mask = (df["model"] == model) & (df["method"] == method) & (df["device"] == "cuda:0")
            deltas_ms = df[mask]["delta_p_ms"].values
            
            if len(deltas_ms) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(METHOD_LABELS.get(method, method))
                continue
            
            # Compute statistics
            stats = compute_delta_statistics(deltas_ms / MS_PER_SAMPLE)  # Convert back to samples
            
            # Plot histogram
            bins = np.linspace(-500, 500, 51)
            ax.hist(deltas_ms, bins=bins, color=COLORS.get(method, "gray"), 
                   alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.axvline(0, color="black", linestyle="-", linewidth=2, label="Catalog P")
            
            # Add statistics inset
            stats_text = (
                f"N = {stats['n']:.0f}\n"
                f"Mean = {stats['mean_ms']:.1f} ms\n"
                f"Median = {stats['median_ms']:.1f} ms\n"
                f"Std = {stats['std_ms']:.1f} ms\n"
                f"P95 = {stats['p95_ms']:.0f} ms\n"
                f"±10 samp = {stats['pct_10']:.1f}%"
            )
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment="top", horizontalalignment="right",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
            
            ax.set_xlim(-500, 500)
            ax.set_xlabel("ΔT vs catalog (ms)", fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel("Count", fontsize=10)
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=11, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
        
        fig.suptitle(f"{model}: Pick Time Distribution vs Catalog", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        
        model_slug = model.replace("-", "").replace(" ", "_")
        out_path = out_dir / f"figure_c_histogram_{model_slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def generate_figure_d_agreement_heatmap(df: pd.DataFrame, out_dir: Path):
    """Generate Figure D: Method agreement heatmap."""
    
    for model in MODELS:
        methods = ["annotate", "fp16", "bf16", "bf16_compile"]
        method_names = [METHOD_LABELS.get(m, m) for m in methods]
        
        # For each pair of methods, compute agreement on same traces
        agreement_matrix = np.zeros((len(methods), len(methods)))
        
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i == j:
                    agreement_matrix[i, j] = 100.0
                    continue
                
                # Find common traces
                mask1 = (df["model"] == model) & (df["method"] == m1) & (df["device"] == "cuda:0")
                mask2 = (df["model"] == model) & (df["method"] == m2) & (df["device"] == "cuda:0")
                
                df1 = df[mask1].set_index("trace_row")
                df2 = df[mask2].set_index("trace_row")
                
                common_traces = df1.index.intersection(df2.index)
                
                if len(common_traces) == 0:
                    agreement_matrix[i, j] = float("nan")
                    continue
                
                # Compare deltas (within ±5 samples = "agreement")
                deltas1 = df1.loc[common_traces, "delta_p_samples"]
                deltas2 = df2.loc[common_traces, "delta_p_samples"]
                
                diff = np.abs(deltas1 - deltas2)
                agreement = np.mean(diff <= 5) * 100  # Within 5 samples
                agreement_matrix[i, j] = agreement
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(agreement_matrix, cmap="RdYlGn", vmin=90, vmax=100)
        
        ax.set_xticks(range(len(methods)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(method_names, fontsize=9)
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(methods)):
                val = agreement_matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 95 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", 
                           color=text_color, fontsize=10, fontweight="bold")
        
        ax.set_title(f"{model}: Pick Agreement (% within ±5 samples)", fontsize=12, fontweight="bold")
        
        cbar = plt.colorbar(im, ax=ax, label="Agreement %")
        
        fig.tight_layout()
        
        model_slug = model.replace("-", "").replace(" ", "_")
        out_path = out_dir / f"figure_d_agreement_{model_slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("PICK QUALITY SUMMARY (1 GPU, all station counts combined)")
    print("=" * 80)
    
    print(f"\n{'Model':<15} {'Method':<20} {'N':>6} {'Mean ΔT':>10} {'Med ΔT':>10} {'P95':>8} {'±10 samp':>10}")
    print("-" * 85)
    
    for model in MODELS:
        for method in ["annotate", "fp16", "bf16", "bf16_compile"]:
            mask = (df["model"] == model) & (df["method"] == method) & (df["device"] == "cuda:0")
            subset = df[mask]
            
            if len(subset) == 0:
                continue
            
            deltas = subset["delta_p_samples"].values
            stats = compute_delta_statistics(deltas)
            
            print(f"{model:<15} {METHOD_LABELS.get(method, method):<20} "
                  f"{stats['n']:>6.0f} "
                  f"{stats['mean_ms']:>9.1f}ms "
                  f"{stats['median_ms']:>9.1f}ms "
                  f"{stats['p95_ms']:>7.0f}ms "
                  f"{stats['pct_10']:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze pick quality from JSONL results")
    parser.add_argument(
        "--jsonl-dir", 
        type=Path, 
        default=Path(__file__).resolve().parent.parent / "results",
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "figures" / "pick_quality",
        help="Output directory for figures and tables"
    )
    args = parser.parse_args()
    
    print(f"Loading JSONL files from {args.jsonl_dir}...")
    rows = load_jsonl_files(args.jsonl_dir)
    print(f"Loaded {len(rows)} rows")
    
    print("\nExtracting pick quality data...")
    df = extract_pick_data(rows)
    print(f"Extracted {len(df)} pick records")
    
    if len(df) == 0:
        print("ERROR: No pick quality data found!")
        return 1
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tables
    print("\nGenerating tables...")
    generate_table_b_delta_statistics(df, args.out_dir)
    generate_table_c_hardware_consistency(df, args.out_dir)
    
    # Generate figures
    print("\nGenerating figures...")
    generate_figure_b_cdf(df, args.out_dir)
    generate_figure_c_enhanced_histograms(df, args.out_dir)
    generate_figure_d_agreement_heatmap(df, args.out_dir)
    
    # Print summary
    print_summary(df)
    
    print(f"\n✓ All outputs saved to {args.out_dir}")
    
    # Print limitation notice
    print("\n" + "=" * 80)
    print("LIMITATION NOTICE")
    print("=" * 80)
    print("""
The current benchmark data has limitations:
1. Only ONE pick per trial is stored (first station index)
2. Synthetic trace duplication means all "stations" have the SAME catalog pick
3. Full pick lists are not available, preventing true precision/recall calculation

To compute proper precision/recall/F1 with matched/missing/additional picks,
run the new pick quality benchmark:

    python scripts/run_pick_quality_analysis.py --n-traces 50 --output results/pick_quality_full.json

This will run inference on real traces and store all detected picks.
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
