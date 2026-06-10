#!/usr/bin/env python3
"""Native SeisBench ``classify()`` and ``annotate()`` sweep (250 / 580 stations).

Benchmarks unmodified SeisBench APIs on the same synthetic STEAD/TXED station
networks as ``run_seisbench_sweep.py`` (PhaseNet, PhaseNetLight, EQTransformer,
EQT-NC — no EQCCT). Dataset build and timechunk layout are **not** timed.

**classify** — per-station ``model.classify()`` after ``annotate_stream_pre``,
parallelized with a fixed-size process pool (``--cpus`` workers on CPU;
``--cpus 1`` on GPU).

**annotate** — one merged-stream ``model.annotate()`` call (offline batch mode,
same semantics as RAPID ``baseline_annotate``).

CPU grid default: 5, 8, 11, 14, 17, 20 (pinned via ``sched_setaffinity``).
GPU block: **1 GPU only** (``cuda:0``).

Example (single trial)::

    python scripts/run_native_seisbench_sweep.py \\
        --dataset stead --n-stations 250 --model PhaseNet \\
        --method classify --cpus 5

Full sweep::

    python scripts/run_native_seisbench_sweep.py --run-all
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RAPID_ROOT = Path(__file__).resolve().parents[1]
EQCCTPRO_ROOT = RAPID_ROOT  # eqcctpro package is vendored inside RAPID
for p in (str(RAPID_ROOT), str(EQCCTPRO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rapid.data import load_all_streams, load_station_stream, preprocess_for_model, select_stations  # noqa: E402
from rapid.runners.single_gpu import run_baseline_single  # noqa: E402
from rapid.backends.baseline import BaselineAnnotate  # noqa: E402

from scripts.build_seisbench_network import build_network  # noqa: E402
from scripts.run_seisbench_sweep import (  # noqa: E402
    DATASETS,
    DEFAULT_CPU_GRID,
    DEFAULT_MAX_CPUS,
    MODELS,
    STATION_COUNTS,
    _ensure_network,
    _materialize_network_layout,
    _net_dir,
    _parse_cpu_grid,
    _prepare_all_networks,
)

METHODS = ("classify", "annotate")

NATIVE_CSV_COLUMNS = [
    "Trial Number",
    "Method",
    "Model Used",
    "Number of Stations Used",
    "Number of CPUs Allocated",
    "GPUs Used",
    "Parallel Workers",
    "Total Wall Time (s)",
    # annotate path
    "Merge Streams Time (s)",
    "Annotate End-to-End Time (s)",
    # classify path (per-station means; sums for serial-equivalent work)
    "Avg Preprocess Time (s)",
    "Avg Classify Time (s)",
    "Sum Station Work Time (s)",
    "Scheduling Residual (s)",
    "Trial Success",
    "Error Message",
    "Comments",
]


def _method_dir(method: str) -> str:
    return "classify" if method == "classify" else "annotate"


def _result_base(
    args, dataset: str, n_stations: int, model_key: str, method: str, device_tag: str
) -> Path:
    return (
        args.results_root
        / dataset.lower()
        / f"{n_stations}st"
        / _method_dir(method)
        / model_key
        / device_tag
    )


@contextlib.contextmanager
def _cpu_affinity(n_cpus: int, max_cpus: int):
    """Pin the current process to cores ``0 .. n_cpus-1`` (capped by max_cpus)."""
    n = max(1, min(int(n_cpus), int(max_cpus)))
    mask = set(range(n))
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, mask)
    try:
        yield n
    finally:
        os.sched_setaffinity(0, old)


def _set_thread_env(n_cpus: int, *, per_worker: bool = False) -> Dict[str, str]:
    """Limit BLAS/OpenMP threads; return previous values for restore."""
    if per_worker:
        n = "1"
    else:
        n = str(max(1, int(n_cpus)))
    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    prev = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = n
    return prev


def _restore_thread_env(prev: Dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    trial_no = 1
    if not write_header:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                try:
                    trial_no = int(rows[-1]["Trial Number"]) + 1
                except (KeyError, ValueError, TypeError):
                    trial_no = len(rows) + 1
    row = dict(row)
    row["Trial Number"] = trial_no
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NATIVE_CSV_COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in NATIVE_CSV_COLUMNS})


# ---------------------------------------------------------------------------
# classify() workers (one model per process, loaded in pool initializer)
# ---------------------------------------------------------------------------

_G_MODEL = None
_G_DEVICE = "cpu"
_G_P_TH = 0.3
_G_S_TH = 0.3
_G_DET_TH = 0.3


def _classify_pool_init(
    parent: str,
    child: str,
    device: str,
    p_threshold: float,
    s_threshold: float,
    detection_threshold: float,
) -> None:
    global _G_MODEL, _G_DEVICE, _G_P_TH, _G_S_TH, _G_DET_TH
    _set_thread_env(1, per_worker=True)
    import seisbench.models as sbm
    import torch

    _G_DEVICE = device
    _G_P_TH = p_threshold
    _G_S_TH = s_threshold
    _G_DET_TH = detection_threshold
    cls = getattr(sbm, parent)
    _G_MODEL = cls.from_pretrained(child)
    _G_MODEL.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        _G_MODEL.to(torch.device(device))


def _classify_one_station(task: Tuple[str, str]) -> Dict[str, float]:
    """Load one station, preprocess, classify; return phase timings in seconds."""
    station, dataset_dir = task
    import torch

    t0 = time.perf_counter()
    st = load_station_stream(dataset_dir, station)
    if len(st) == 0:
        raise RuntimeError(f"empty stream for {station}")
    t_pre0 = time.perf_counter()
    st_pre = preprocess_for_model(_G_MODEL, st)
    preprocess_s = time.perf_counter() - t_pre0
    t_inf0 = time.perf_counter()
    _G_MODEL.classify(
        st_pre,
        P_threshold=_G_P_TH,
        S_threshold=_G_S_TH,
        Detection_threshold=_G_DET_TH,
    )
    if _G_DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(_G_DEVICE)
    classify_s = time.perf_counter() - t_inf0
    total_s = time.perf_counter() - t0
    return {
        "preprocess_s": preprocess_s,
        "classify_s": classify_s,
        "total_s": total_s,
    }


def _run_classify_trial(
    *,
    dataset_dir: Path,
    stations: List[str],
    parent: str,
    child: str,
    device: str,
    n_cpus: int,
    n_workers: int,
    p_threshold: float,
    s_threshold: float,
    detection_threshold: float,
) -> Dict[str, Any]:
    tasks = [(sta, str(dataset_dir)) for sta in stations]
    t_wall0 = time.perf_counter()
    per_station: List[Dict[str, float]] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_classify_pool_init,
        initargs=(parent, child, device, p_threshold, s_threshold, detection_threshold),
    ) as pool:
        futures = {pool.submit(_classify_one_station, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            per_station.append(fut.result())
    wall_s = time.perf_counter() - t_wall0

    pre = [x["preprocess_s"] for x in per_station]
    clf = [x["classify_s"] for x in per_station]
    tot = [x["total_s"] for x in per_station]
    sum_work = float(sum(tot))
    avg_pre = float(statistics.mean(pre)) if pre else 0.0
    avg_clf = float(statistics.mean(clf)) if clf else 0.0
    ideal_parallel = sum_work / max(1, n_workers)
    sched_residual = max(0.0, wall_s - ideal_parallel)

    return dict(
        wall_s=wall_s,
        avg_preprocess_s=avg_pre,
        avg_classify_s=avg_clf,
        sum_station_work_s=sum_work,
        scheduling_residual_s=sched_residual,
        n_workers=n_workers,
    )


def _run_annotate_trial(
    *,
    streams: List[Tuple[str, Any]],
    parent: str,
    child: str,
    device: str,
    n_cpus: int,
) -> Dict[str, Any]:
    prev = _set_thread_env(n_cpus, per_worker=False)
    try:
        backend = BaselineAnnotate(parent_model=parent, child_model=child, device=device)
        backend.load()
        try:
            t0 = time.perf_counter()
            result = run_baseline_single(backend, streams, merge_into_one_stream=True)
            wall_s = time.perf_counter() - t0
            stages = result.stage_times
            return dict(
                wall_s=wall_s,
                merge_streams_s=float(stages.get("merge_streams", 0.0)),
                annotate_e2e_s=float(stages.get("annotate_end_to_end", 0.0)),
            )
        finally:
            backend.close()
    finally:
        _restore_thread_env(prev)


def _run_one_trial(
    args,
    *,
    dataset: str,
    n_stations: int,
    model_key: str,
    method: str,
    n_cpus: int,
    device: str,
    gpu_ids: List[int],
) -> None:
    m = MODELS[model_key]
    parent, child = m["parent"], m["child"]
    model_used = f"{parent}/{child}"
    net_dir = _net_dir(args.net_root, dataset, n_stations)
    stations = select_stations(net_dir, n_stations)
    device_tag = "cpu" if not gpu_ids else f"gpu{gpu_ids[0]}"
    csv_path = (
        _result_base(args, dataset, n_stations, model_key, method, device_tag)
        / "timing"
        / "native_seisbench_results.csv"
    )

    if device.startswith("cuda"):
        n_workers = 1
        eff_cpus = 1
    else:
        n_workers = max(1, min(int(n_cpus), len(stations)))
        eff_cpus = int(n_cpus)

    tag = f"{dataset}_{n_stations}st_{model_key}_{method}_{device_tag}_cpus{eff_cpus}"
    print(f"\n=== NATIVE {method.upper()}: {tag} ===")

    if args.dry_run:
        print(f"  would write {csv_path}")
        return

    row: Dict[str, Any] = {
        "Method": method,
        "Model Used": model_used,
        "Number of Stations Used": len(stations),
        "Number of CPUs Allocated": eff_cpus,
        "GPUs Used": json.dumps(gpu_ids) if gpu_ids else "[]",
        "Trial Success": "1",
        "Error Message": "",
        "Comments": (
            f"Native SeisBench {method}(); "
            f"{'process pool' if method == 'classify' else 'merged-stream annotate'}; "
            f"affinity cores 0-{eff_cpus - 1}"
        ),
    }

    try:
        with _cpu_affinity(eff_cpus, args.max_cpus):
            if method == "classify":
                metrics = _run_classify_trial(
                    dataset_dir=net_dir,
                    stations=stations,
                    parent=parent,
                    child=child,
                    device=device,
                    n_cpus=eff_cpus,
                    n_workers=n_workers,
                    p_threshold=args.p_threshold,
                    s_threshold=args.s_threshold,
                    detection_threshold=args.detection_threshold,
                )
                row.update(
                    {
                        "Parallel Workers": metrics["n_workers"],
                        "Total Wall Time (s)": round(metrics["wall_s"], 6),
                        "Merge Streams Time (s)": "",
                        "Annotate End-to-End Time (s)": "",
                        "Avg Preprocess Time (s)": round(metrics["avg_preprocess_s"], 6),
                        "Avg Classify Time (s)": round(metrics["avg_classify_s"], 6),
                        "Sum Station Work Time (s)": round(metrics["sum_station_work_s"], 6),
                        "Scheduling Residual (s)": round(metrics["scheduling_residual_s"], 6),
                    },
                )
            else:
                print(f"  loading {len(stations)} station streams ...")
                t_load = time.perf_counter()
                streams = load_all_streams(net_dir, stations)
                print(f"  loaded in {time.perf_counter() - t_load:.2f}s")
                metrics = _run_annotate_trial(
                    streams=streams,
                    parent=parent,
                    child=child,
                    device=device,
                    n_cpus=eff_cpus,
                )
                row.update(
                    {
                        "Parallel Workers": 1,
                        "Total Wall Time (s)": round(metrics["wall_s"], 6),
                        "Merge Streams Time (s)": round(metrics["merge_streams_s"], 6),
                        "Annotate End-to-End Time (s)": round(metrics["annotate_e2e_s"], 6),
                        "Avg Preprocess Time (s)": "",
                        "Avg Classify Time (s)": "",
                        "Sum Station Work Time (s)": "",
                        "Scheduling Residual (s)": "",
                    },
                )
    except Exception as exc:
        import traceback

        row["Trial Success"] = "0"
        row["Error Message"] = str(exc)
        row["Comments"] = traceback.format_exc()
        print(f"  FAILED: {exc}")

    _append_csv_row(csv_path, row)
    print(f"  wrote {csv_path}")


def _methods_for_args(args) -> List[str]:
    if args.method == "both":
        return ["classify", "annotate"]
    return [args.method]


def _do(args, *, dataset, n_stations, model_key, method, n_cpus, gpu_ids, device_tag):
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
    saved_method = args.method
    args.method = method
    try:
        _run_one_trial(
            args,
            dataset=dataset,
            n_stations=n_stations,
            model_key=model_key,
            method=method,
            n_cpus=n_cpus,
            device=device,
            gpu_ids=gpu_ids,
        )
    finally:
        args.method = saved_method


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-root", type=Path, default=EQCCTPRO_ROOT / "data" / "seisbench_networks")
    ap.add_argument(
        "--results-root",
        type=Path,
        default=RAPID_ROOT / "results" / "native_seisbench_sweep",
    )
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--n-stations", type=int, choices=STATION_COUNTS)
    ap.add_argument("--model", choices=list(MODELS.keys()))
    ap.add_argument("--method", choices=["classify", "annotate", "both"], default="both")
    ap.add_argument("--cpus", type=int, default=20, help="CPU count / worker pool size (single trial, CPU)")
    ap.add_argument(
        "--gpus",
        default="",
        help="GPU id for single trial (e.g. 0); empty = CPU only",
    )
    ap.add_argument("--max-cpus", type=int, default=DEFAULT_MAX_CPUS, help="Affinity pool 0..N-1")
    ap.add_argument(
        "--cpu-grid",
        default=",".join(str(c) for c in DEFAULT_CPU_GRID),
        help="CPU counts when --run-all (comma-separated)",
    )
    ap.add_argument("--gpu-id", type=int, default=0, help="Single GPU for GPU block (default cuda:0)")
    ap.add_argument(
        "--sweep-cpu-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--sweep-with-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one GPU trial per (dataset, N, model, method) on --gpu-id",
    )
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--detection-threshold", type=float, default=0.3)
    ap.add_argument("--n-unique", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-s", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--prepare-networks-only", action="store_true")
    ap.add_argument("--skip-network-prep", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.cpu_grid = _parse_cpu_grid(args.cpu_grid)
    args.gpus = [int(x) for x in str(args.gpus).split(",") if x.strip()] if str(getattr(args, "gpus", "")).strip() else []

    if args.prepare_networks_only:
        if not args.dry_run:
            _prepare_all_networks(args)
        else:
            print("DRY-RUN: would prepare networks for", DATASETS, STATION_COUNTS)
        return 0

    if args.run_all:
        print(
            f"RUN-ALL native SeisBench: stations={STATION_COUNTS} methods=classify+annotate "
            f"models={list(MODELS.keys())}\n"
            f"  cpu_grid={args.cpu_grid} max_cpus={args.max_cpus} "
            f"sweep_cpu={args.sweep_cpu_only} sweep_gpu={args.sweep_with_gpu} "
            f"gpu_id={args.gpu_id}\n"
            f"  results under {args.results_root}"
        )
        if not args.dry_run and not args.skip_network_prep:
            _prepare_all_networks(args)
        for dataset in DATASETS:
            for n_stations in STATION_COUNTS:
                meta = _ensure_network(
                    args, dataset, n_stations, build=not args.skip_network_prep, materialize=False
                )
                if not args.dry_run and not args.skip_network_prep:
                    _materialize_network_layout(args, dataset, n_stations, meta)
                for model_key in MODELS:
                    for method in METHODS:
                        if args.sweep_cpu_only:
                            for n_cpus in args.cpu_grid:
                                _do(
                                    args,
                                    dataset=dataset,
                                    n_stations=n_stations,
                                    model_key=model_key,
                                    method=method,
                                    n_cpus=n_cpus,
                                    gpu_ids=[],
                                    device_tag="cpu",
                                )
                        if args.sweep_with_gpu:
                            _do(
                                args,
                                dataset=dataset,
                                n_stations=n_stations,
                                model_key=model_key,
                                method=method,
                                n_cpus=1,
                                gpu_ids=[args.gpu_id],
                                device_tag=f"gpu{args.gpu_id}",
                            )
        return 0

    missing = [k for k in ("dataset", "n_stations", "model") if getattr(args, k) is None]
    if missing:
        ap.error(f"Specify {missing} or use --run-all")

    meta = _ensure_network(args, args.dataset, args.n_stations, build=True, materialize=True)
    _ = meta
    gpu_ids = list(args.gpus)
    device_tag = f"gpu{gpu_ids[0]}" if gpu_ids else "cpu"
    for method in _methods_for_args(args):
        _do(
            args,
            dataset=args.dataset,
            n_stations=args.n_stations,
            model_key=args.model,
            method=method,
            n_cpus=args.cpus,
            gpu_ids=gpu_ids,
            device_tag=device_tag,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
