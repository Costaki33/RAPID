"""Plots and tables for RAPID benchmark results.

Expects a JSONL file produced by :func:`rapid.matrix.run_matrix`. Provides:

- ``throughput_vs_batch(...)`` — throughput (stations/s) vs batch_size, one
  line per backend/dtype. Highlights the plateau in the curve.
- ``speedup_vs_baseline(...)`` — per-backend speedup factor over the baseline
  annotate() call, bars grouped by model.
- ``stage_breakdown(...)`` — stacked bar of preprocess / window / forward /
  post for the best config per backend.
- ``dual_gpu_scaling(...)`` — 1-GPU vs 2-GPU wall time per model.
- ``cpu_worker_sweep_plot(...)`` — wall time and GPU utilization vs #CPU workers.
- ``quality_vs_speed(...)`` — pick-time drift vs speedup scatter.

All plots write PNG + SVG to a target directory and return the path list.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


LOG = logging.getLogger("rapid.visualize")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(jsonl_path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                LOG.warning("Skipping malformed JSONL row: %s", e)
    return rows


def _filter(rows: Iterable[Dict[str, Any]], **kw) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _backend_label(r: Dict[str, Any]) -> str:
    extra = r.get("backend_extra") or {}
    sfx = "_compile" if extra.get("compile") else ""
    return f"{r['backend']}_{r.get('dtype', 'fp32')}{sfx}"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _ensure_dir(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def throughput_vs_batch(
    rows: List[Dict[str, Any]],
    *,
    model_label: str,
    device: str,
    n_stations: int,
    out_dir: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    single = [
        r for r in rows
        if r.get("kind") == "single"
        and r.get("model_label") == model_label
        and r.get("device") == device
        and r.get("n_stations") == n_stations
    ]
    if not single:
        raise ValueError("No 'single' rows match filter.")

    by_bk: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in single:
        if r.get("overlap_samples", 0) != 0:
            continue  # Prefer overlap=0 curve; see separate overlap plot for parity.
        by_bk[_backend_label(r)][r["batch_size"]].append(
            n_stations / r["total_s"] if r["total_s"] > 0 else 0.0
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for bk, bs_map in sorted(by_bk.items()):
        bss = sorted(bs_map)
        ys = [_mean(bs_map[b]) for b in bss]
        ax.plot(bss, ys, marker="o", label=bk)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (stations/s)")
    ax.set_title(f"Throughput vs batch — {model_label} · {device} · {n_stations} stations")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = _ensure_dir(out_dir) / f"throughput_vs_batch_{model_label}_{device.replace(':','')}_{n_stations}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def speedup_vs_baseline(
    rows: List[Dict[str, Any]],
    *,
    device: str,
    n_stations: int,
    out_dir: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    baselines: Dict[str, List[float]] = defaultdict(list)
    for r in _filter(rows, kind="baseline", device=device, n_stations=n_stations):
        baselines[r["model_label"]].append(r["total_s"])
    if not baselines:
        raise ValueError("No baseline rows found for filter.")
    baseline_mean = {k: _mean(v) for k, v in baselines.items()}

    # Best single-backend time per (model, backend) at any batch size.
    best: Dict[Tuple[str, str], float] = {}
    for r in _filter(rows, kind="single", device=device, n_stations=n_stations):
        key = (r["model_label"], _backend_label(r))
        best[key] = min(best.get(key, float("inf")), r["total_s"])

    # Group bars per model.
    models = sorted({m for m, _ in best})
    backends = sorted({b for _, b in best})
    x = np.arange(len(models))
    width = 0.8 / max(1, len(backends))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, bk in enumerate(backends):
        ys = []
        for m in models:
            t = best.get((m, bk))
            base = baseline_mean.get(m)
            if t is None or base is None or t <= 0:
                ys.append(float("nan"))
            else:
                ys.append(base / t)
        ax.bar(x + i * width, ys, width=width, label=bk)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Speedup vs baseline annotate()  (higher is better)")
    ax.set_title(f"Speedup vs baseline · {device} · {n_stations} stations")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = _ensure_dir(out_dir) / f"speedup_vs_baseline_{device.replace(':','')}_{n_stations}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def stage_breakdown(
    rows: List[Dict[str, Any]],
    *,
    model_label: str,
    device: str,
    n_stations: int,
    out_dir: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    single = _filter(rows, kind="single", device=device, n_stations=n_stations, model_label=model_label)
    best_per_bk: Dict[str, Dict[str, Any]] = {}
    for r in single:
        bk = _backend_label(r)
        if bk not in best_per_bk or r["total_s"] < best_per_bk[bk]["total_s"]:
            best_per_bk[bk] = r

    # Include baseline as end-to-end (single stage).
    for r in _filter(rows, kind="baseline", device=device, n_stations=n_stations, model_label=model_label):
        best_per_bk.setdefault("baseline_annotate", r)

    backends = sorted(best_per_bk)
    stages_order = ["merge_streams", "preprocess", "window_cut_and_stack", "forward", "annotate_end_to_end"]

    matrix = np.zeros((len(stages_order), len(backends)))
    for j, bk in enumerate(backends):
        st = best_per_bk[bk].get("stage_times_s", {})
        for i, stg in enumerate(stages_order):
            matrix[i, j] = float(st.get(stg, 0.0))

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(backends))
    for i, stg in enumerate(stages_order):
        ax.bar(backends, matrix[i], bottom=bottom, label=stg)
        bottom += matrix[i]
    ax.set_ylabel("Wall time (s, best config per backend)")
    ax.set_title(f"Stage breakdown — {model_label} · {device} · {n_stations} stations")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()

    out = _ensure_dir(out_dir) / f"stage_breakdown_{model_label}_{device.replace(':','')}_{n_stations}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def dual_gpu_scaling(
    rows: List[Dict[str, Any]],
    *,
    n_stations: int,
    out_dir: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    dual = _filter(rows, kind="dual_gpu", n_stations=n_stations)
    single = _filter(rows, kind="single", device="cuda:0", n_stations=n_stations)

    best_single: Dict[Tuple[str, str], float] = {}
    for r in single:
        k = (r["model_label"], _backend_label(r))
        best_single[k] = min(best_single.get(k, float("inf")), r["total_s"])

    best_dual: Dict[Tuple[str, str], float] = {}
    for r in dual:
        k = (r["model_label"], f"{r['backend']}_{r.get('dtype','fp32')}")
        best_dual[k] = min(best_dual.get(k, float("inf")), r.get("wall_time_s", float("inf")))

    keys = sorted(set(best_single) & set(best_dual))
    if not keys:
        raise ValueError("No matching (model, backend) pairs between single and dual runs.")

    labels = [f"{m}\n{bk}" for m, bk in keys]
    x = np.arange(len(keys))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, [best_single[k] for k in keys], width=w, label="1 GPU")
    ax.bar(x + w/2, [best_dual[k] for k in keys], width=w, label="2 GPUs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, fontsize=8, ha="right")
    ax.set_ylabel("Total wall time (s)")
    ax.set_title(f"1-GPU vs 2-GPU wall time · {n_stations} stations")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()

    out = _ensure_dir(out_dir) / f"dual_gpu_scaling_{n_stations}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def cpu_worker_sweep_plot(
    rows: List[Dict[str, Any]],
    *,
    model_label: str,
    n_stations: int,
    out_dir: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    cpu = _filter(rows, kind="cpu_worker_sweep", model_label=model_label, n_stations=n_stations)
    if not cpu:
        raise ValueError("No cpu_worker_sweep rows match filter.")

    by_bk: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in cpu:
        by_bk[f"{r['backend']}_{r.get('dtype','fp32')}"][r["n_cpu_workers"]].append(r)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for bk, m in sorted(by_bk.items()):
        xs = sorted(m)
        wall = [_mean([x["wall_time_s"] for x in m[n]]) for n in xs]
        util = [_mean([x["gpu_utilization_pct"] for x in m[n]]) for n in xs]
        ax1.plot(xs, wall, marker="o", label=bk)
        ax2.plot(xs, util, marker="o", label=bk)
    ax1.set_ylabel("Wall time (s)")
    ax1.set_title(f"CPU worker sweep — {model_label} · {n_stations} stations")
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend(fontsize=8)
    ax2.set_ylabel("GPU utilization (%)")
    ax2.set_xlabel("# CPU preprocessing workers")
    ax2.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()

    out = _ensure_dir(out_dir) / f"cpu_worker_sweep_{model_label}_{n_stations}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def render_all(
    jsonl_path: str | Path,
    out_dir: str | Path,
    *,
    devices: Optional[List[str]] = None,
) -> List[Path]:
    """Render every plot defined above for every matching slice in ``jsonl_path``."""
    rows = load_results(jsonl_path)
    if not rows:
        LOG.warning("No rows found.")
        return []

    outputs: List[Path] = []
    models = sorted({r.get("model_label") for r in rows if r.get("model_label")})
    n_list = sorted({r.get("n_stations") for r in rows if isinstance(r.get("n_stations"), int)})
    dev_list = devices or sorted({
        r.get("device") for r in rows
        if r.get("device") and not r.get("device", "").startswith("cuda:0+")
    })

    for device in dev_list:
        for n in n_list:
            try:
                outputs.append(speedup_vs_baseline(rows, device=device, n_stations=n, out_dir=out_dir))
            except Exception as e:
                LOG.info("speedup_vs_baseline skipped: %s", e)
            for m in models:
                try:
                    outputs.append(throughput_vs_batch(
                        rows, model_label=m, device=device, n_stations=n, out_dir=out_dir
                    ))
                except Exception as e:
                    LOG.info("throughput_vs_batch skipped: %s", e)
                try:
                    outputs.append(stage_breakdown(
                        rows, model_label=m, device=device, n_stations=n, out_dir=out_dir
                    ))
                except Exception as e:
                    LOG.info("stage_breakdown skipped: %s", e)

    for n in n_list:
        try:
            outputs.append(dual_gpu_scaling(rows, n_stations=n, out_dir=out_dir))
        except Exception as e:
            LOG.info("dual_gpu_scaling skipped: %s", e)

    for m in models:
        for n in n_list:
            try:
                outputs.append(cpu_worker_sweep_plot(rows, model_label=m, n_stations=n, out_dir=out_dir))
            except Exception as e:
                LOG.info("cpu_worker_sweep_plot skipped: %s", e)

    return outputs
