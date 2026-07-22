#!/usr/bin/env python3
"""Aggregate repeated pick-quality runs (mean +/- std across runs) per dataset.

Reads the per-run JSON files produced by run_pick_quality_analysis.py
(results/pick_quality_<dataset>_run<k>_seed<seed>.json), computes the mean and
sample standard deviation of each metric across the runs, prints a console
summary, and emits publication LaTeX tables (detection, timing, cross-hardware).

Usage:
    python benchmarks/analysis/aggregate_pick_quality_runs.py \
        --dataset txed \
        --glob 'results/pick_quality_txed_run*_seed*.json' \
        --out-dir figures/pick_quality/
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
METHOD_ORDER = ["annotate_fp32", "lean_fp16", "lean_bf16", "lean_bf16_compile"]
METHOD_LABELS = {
    "annotate_fp32": r"\texttt{annotate()} FP32",
    "lean_fp16": "Slipstream FP16",
    "lean_bf16": "Slipstream BF16",
    "lean_bf16_compile": "Slipstream BF16 + compile",
}
GPU_DEVICE = "cuda:0"


def _mean_std(values: List[float]) -> Tuple[float, float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var)


def load_runs(paths: List[Path]) -> List[Dict[str, Any]]:
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def collect(runs: List[Dict[str, Any]], device: str) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    """Map (model, method) -> metric -> list of per-run values for one device."""
    acc: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for agg in run.get("aggregated", []):
            if agg.get("device") != device:
                continue
            key = (agg["model"], agg["method"])
            acc[key]["precision"].append(agg["precision"])
            acc[key]["recall"].append(agg["recall"])
            acc[key]["f1"].append(agg["f1"])
            acc[key]["detected"].append(agg["total_detected_picks"])
            acc[key]["matched"].append(agg["total_matched"])
            acc[key]["missing"].append(agg["total_missing"])
            acc[key]["additional"].append(agg["total_additional"])
            acc[key]["duplicates"].append(agg["total_duplicates"])
            acc[key]["catalog"].append(agg["total_catalog_picks"])
            ds = agg.get("delta_statistics", {})
            for stat in ("n", "mean_ms", "median_ms", "std_ms", "p95_ms", "p99_ms",
                         "pct_within_1_sample", "pct_within_5_samples", "pct_within_10_samples"):
                acc[key][f"dt_{stat}"].append(ds.get(stat, float("nan")))
    return acc


def fmt(mean: float, std: float, dec: int) -> str:
    if math.isnan(mean):
        return "--"
    return f"{mean:.{dec}f}$\\pm${std:.{dec}f}"


def methods_for(model: str, acc: Dict[Tuple[str, str], Any]) -> List[str]:
    return [m for m in METHOD_ORDER if (model, m) in acc]


def detection_table(acc, dataset: str, n_runs: int) -> str:
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        rf"    \caption{{Pick-detection accuracy on the {dataset.upper()} dataset, single GPU, "
        rf"reported as mean$\pm$std over {n_runs} independent runs of 50 catalog traces each "
        r"(seeds 42--46). Cat.\ = catalog P picks (50 per run); Det.\ = picks returned; "
        r"TP/FN/FP = true positives / false negatives / false positives "
        r"(match tolerance $\pm$50 samples). Prec.\ $=$ TP/(TP+FP); Rec.\ $=$ TP/(TP+FN); "
        r"F1 is their harmonic mean.}",
        rf"    \label{{tab:pick-detection-{dataset}}}",
        r"    \vspace{0.8em}",
        r"    \footnotesize",
        r"    \begin{tabular}{l l r r r r r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Method} & \textbf{Det.} & \textbf{TP} & \textbf{FN} & "
        r"\textbf{FP} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\",
        r"    \midrule",
    ]
    for mi, model in enumerate(MODELS):
        ms = methods_for(model, acc)
        for i, method in enumerate(ms):
            d = acc[(model, method)]
            model_col = model if i == 0 else ""
            det = fmt(*_mean_std(d["detected"]), 1)
            tp = fmt(*_mean_std(d["matched"]), 1)
            fn = fmt(*_mean_std(d["missing"]), 1)
            fp = fmt(*_mean_std(d["additional"]), 1)
            prec = fmt(*_mean_std(d["precision"]), 3)
            rec = fmt(*_mean_std(d["recall"]), 3)
            f1 = fmt(*_mean_std(d["f1"]), 3)
            lines.append(
                f"    {model_col} & {METHOD_LABELS[method]} & {det} & {tp} & {fn} & {fp} & "
                f"{prec} & {rec} & {f1} \\\\"
            )
        if mi != len(MODELS) - 1:
            lines.append(r"    \addlinespace")
    lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def timing_table(acc, dataset: str, n_runs: int) -> str:
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        rf"    \caption{{Pick-timing accuracy for matched picks on the {dataset.upper()} dataset, "
        rf"single GPU, mean over {n_runs} runs. $\Delta T =$ detected minus catalog pick time in "
        r"milliseconds (1 sample $=$ 10\,ms at 100\,Hz); negative means early. N is matched picks; "
        r"P95/P99 are percentiles of $|\Delta T|$; $\pm$5s/$\pm$10s give the fraction of matched "
        r"picks within 5 and 10 samples of the catalog.}",
        rf"    \label{{tab:delta-statistics-{dataset}}}",
        r"    \vspace{0.8em}",
        r"    \footnotesize",
        r"    \begin{tabular}{l l r r r r r r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Method} & \textbf{N} & \textbf{Mean} & \textbf{Med.} & "
        r"\textbf{Std} & \textbf{P95} & \textbf{P99} & \textbf{$\pm$5s} & \textbf{$\pm$10s} \\",
        r"    \midrule",
    ]
    for mi, model in enumerate(MODELS):
        ms = methods_for(model, acc)
        for i, method in enumerate(ms):
            d = acc[(model, method)]
            model_col = model if i == 0 else ""
            n = _mean_std(d["dt_n"])[0]
            mean = _mean_std(d["dt_mean_ms"])[0]
            med = _mean_std(d["dt_median_ms"])[0]
            std = _mean_std(d["dt_std_ms"])[0]
            p95 = _mean_std(d["dt_p95_ms"])[0]
            p99 = _mean_std(d["dt_p99_ms"])[0]
            w5 = _mean_std(d["dt_pct_within_5_samples"])[0]
            w10 = _mean_std(d["dt_pct_within_10_samples"])[0]
            lines.append(
                f"    {model_col} & {METHOD_LABELS[method]} & {n:.0f} & {mean:.1f} & {med:.1f} & "
                f"{std:.1f} & {p95:.0f} & {p99:.0f} & {w5:.1f}\\% & {w10:.1f}\\% \\\\"
            )
        if mi != len(MODELS) - 1:
            lines.append(r"    \addlinespace")
    lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def hardware_table(acc_cpu, acc_gpu, dataset: str, n_runs: int) -> str:
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        rf"    \caption{{Cross-hardware consistency on the {dataset.upper()} dataset, mean$\pm$std "
        rf"over {n_runs} runs. CPU det.\ and 1 GPU det.\ are the number of detected picks on each "
        r"device over the same traces; $|\Delta|$ is the mean absolute difference between devices.}",
        rf"    \label{{tab:hardware-consistency-{dataset}}}",
        r"    \vspace{0.8em}",
        r"    \small",
        r"    \begin{tabular}{l l r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Method} & \textbf{CPU det.} & \textbf{1 GPU det.} & "
        r"\textbf{$|\Delta|$} \\",
        r"    \midrule",
    ]
    for mi, model in enumerate(MODELS):
        ms = [m for m in METHOD_ORDER if (model, m) in acc_gpu and (model, m) in acc_cpu]
        for i, method in enumerate(ms):
            cpu = acc_cpu[(model, method)]["detected"]
            gpu = acc_gpu[(model, method)]["detected"]
            deltas = [abs(c - g) for c, g in zip(cpu, gpu)]
            model_col = model if i == 0 else ""
            cpu_s = fmt(*_mean_std(cpu), 1)
            gpu_s = fmt(*_mean_std(gpu), 1)
            dl_s = fmt(*_mean_std(deltas), 2)
            lines.append(f"    {model_col} & {METHOD_LABELS[method]} & {cpu_s} & {gpu_s} & {dl_s} \\\\")
        if mi != len(MODELS) - 1:
            lines.append(r"    \addlinespace")
    lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def print_summary(acc, dataset: str):
    print(f"\n=== {dataset.upper()} (single GPU) mean+/-std over runs ===")
    print(f"{'Model':<14}{'Method':<26}{'Prec':>14}{'Rec':>14}{'F1':>14}")
    for model in MODELS:
        for method in methods_for(model, acc):
            d = acc[(model, method)]
            p = _mean_std(d["precision"]); r = _mean_std(d["recall"]); f = _mean_std(d["f1"])
            print(f"{model:<14}{method:<26}"
                  f"{p[0]:.3f}+/-{p[1]:.3f} {r[0]:.3f}+/-{r[1]:.3f} {f[0]:.3f}+/-{f[1]:.3f}")


def write_pooled_json(runs: List[Dict[str, Any]], dataset: str, out_path: Path) -> None:
    """Pool raw_results across runs and recompute aggregated, for figure generation."""
    import sys as _sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    from run_pick_quality_analysis import PickResult, aggregate_results, METHODS, method_allowed  # type: ignore
    from rapid.seisbench_precision_eval import DEFAULT_MODELS  # type: ignore

    raw: List[PickResult] = []
    for run in runs:
        for r in run.get("raw_results", []):
            raw.append(PickResult(**r))

    models = sorted({r.model for r in raw})
    methods = [m["name"] for m in METHODS]
    devices = sorted({r.device for r in raw})

    aggregated: List[Dict[str, Any]] = []
    for model in models:
        for method in methods:
            for device in devices:
                agg = aggregate_results(raw, model, method, device)
                if "error" not in agg:
                    aggregated.append(agg)

    pooled = {
        "metadata": {"dataset": dataset, "pooled_runs": len(runs)},
        "raw_results": [r.__dict__ for r in raw],
        "aggregated": aggregated,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pooled, f)
    print(f"Wrote pooled JSON ({len(raw)} records) to {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate repeated pick-quality runs")
    ap.add_argument("--dataset", required=True, choices=["stead", "txed"])
    ap.add_argument("--glob", default=None, help="Glob for run files (default: results/pick_quality_<dataset>_run*_seed*.json)")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/pick_quality"))
    ap.add_argument("--pool-output", type=Path, default=None, help="If set, write a pooled JSON (all runs) for figure generation")
    args = ap.parse_args()

    pattern = args.glob or f"results/pick_quality_{args.dataset}_run*_seed*.json"
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        print(f"No run files match {pattern}")
        return 1
    print(f"Loading {len(paths)} runs for {args.dataset}:")
    for p in paths:
        print(f"  {p}")
    runs = load_runs(paths)
    n_runs = len(runs)

    acc_gpu = collect(runs, GPU_DEVICE)
    acc_cpu = collect(runs, "cpu")

    print_summary(acc_gpu, args.dataset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"table_detection_{args.dataset}.tex").write_text(
        detection_table(acc_gpu, args.dataset, n_runs), encoding="utf-8")
    (args.out_dir / f"table_timing_{args.dataset}.tex").write_text(
        timing_table(acc_gpu, args.dataset, n_runs), encoding="utf-8")
    (args.out_dir / f"table_hardware_{args.dataset}.tex").write_text(
        hardware_table(acc_cpu, acc_gpu, args.dataset, n_runs), encoding="utf-8")
    print(f"\nSaved LaTeX tables to {args.out_dir}/table_*_{args.dataset}.tex")

    if args.pool_output is not None:
        write_pooled_json(runs, args.dataset, args.pool_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
