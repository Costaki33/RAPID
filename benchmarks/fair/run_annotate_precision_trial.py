#!/usr/bin/env python3
"""Annotate precision comparison trial (FP32 / BF16 / FP16).

Times SeisBench ``annotate()`` only. Discrete picks are produced offline via
``classify_aggregate`` (not included in runtime) and scored against the
network catalog. BF16/FP16 picks are also compared to the FP32 picks for the
same (model, network, device, cores, threads) cell when that FP32 trial has
already written ``picks.json``.

EQCCT (SeisBench main) is the dual-branch pair ``EQCCTP`` + ``EQCCTS``: both
branches are annotated sequentially; wall time is the sum of both annotate
calls; picks are the union of each branch's ``classify_aggregate`` output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

RAPID_ROOT = Path(__file__).resolve().parents[2]
if str(RAPID_ROOT) not in sys.path:
    sys.path.insert(0, str(RAPID_ROOT))

from rapid.benchmark import fairness  # noqa: E402
from rapid.benchmark.fairness import StageTimes, build_result, pin_threads  # noqa: E402
from rapid.benchmark.pick_quality import (  # noqa: E402
    catalog_from_manifest_stations,
    compare_pick_sets,
    load_manifest_catalog,
)

# Logical study models. EQCCT expands to EQCCTP + EQCCTS at load time.
MODELS: Dict[str, Dict[str, Any]] = {
    "PhaseNet": {"parent": "PhaseNet", "child": "original", "branches": None},
    "PhaseNetLight": {"parent": "PhaseNetLight", "child": "stead", "branches": None},
    "EQTransformer": {"parent": "EQTransformer", "child": "original", "branches": None},
    "EQT-NC": {
        "parent": "EQTransformer",
        "child": "original_nonconservative",
        "branches": None,
    },
    "EQCCT": {
        "parent": "EQCCT",
        "child": "original",
        "branches": [
            {"parent": "EQCCTP", "child": "original"},
            {"parent": "EQCCTS", "child": "original"},
        ],
    },
}

METHODS = ("annotate_fp32", "annotate_bf16", "annotate_fp16")
EQT_MODELS = {"EQTransformer", "EQT-NC"}
DTYPE_OF = {
    "annotate_fp32": "fp32",
    "annotate_bf16": "bf16",
    "annotate_fp16": "fp16",
}


def _self_rss_mb() -> float:
    import psutil

    try:
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def _net_dir(net_root: Path, n_stations: int) -> Path:
    return net_root / f"stead_{n_stations}st"


def _set_affinity(core_list: Optional[List[int]]) -> int:
    if not core_list:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1
    mask = set(int(c) for c in core_list)
    try:
        os.sched_setaffinity(0, mask)
    except (OSError, AttributeError):
        pass
    return len(mask)


def _window_spec(model: str) -> Tuple[int, int]:
    """(in_samples, overlap_samples) for each model on the 6000-sample STEAD net."""
    if model in ("PhaseNet", "PhaseNetLight"):
        return 3001, 0  # two windows / station
    return 6000, 0  # EQT family + EQCCT: one window / station


def _out_dir(args) -> Path:
    thr = args.torch_threads if args.torch_threads is not None else args.n_cpus
    bs = int(getattr(args, "batch_size", 256) or 256)
    packaging = str(getattr(args, "packaging", "merged") or "merged")
    return (
        args.results_root
        / args.method
        / "stead"
        / f"{args.n_stations}st"
        / args.model
        / args.device
        / f"cpus{args.n_cpus}"
        / f"thr{thr}"
        / f"bs{bs}"
        / packaging
        / args.tag
    )


def _legacy_out_dir_bs256(args) -> Path:
    """Pre-batch-sweep / pre-packaging path; only used for resume of merged bs256."""
    thr = args.torch_threads if args.torch_threads is not None else args.n_cpus
    return (
        args.results_root
        / args.method
        / "stead"
        / f"{args.n_stations}st"
        / args.model
        / args.device
        / f"cpus{args.n_cpus}"
        / f"thr{thr}"
        / args.tag
    )


def _legacy_out_dir_merged_bs(args) -> Path:
    """Path used by the merged matrix after bs was added but before packaging/."""
    thr = args.torch_threads if args.torch_threads is not None else args.n_cpus
    bs = int(getattr(args, "batch_size", 256) or 256)
    return (
        args.results_root
        / args.method
        / "stead"
        / f"{args.n_stations}st"
        / args.model
        / args.device
        / f"cpus{args.n_cpus}"
        / f"thr{thr}"
        / f"bs{bs}"
        / args.tag
    )


def _load_branch(parent: str, child: str, device: str, dtype: str, gpu: bool):
    import seisbench.models as sbm
    import torch
    from rapid.api import EQT_FP16_MESSAGE, _wrap_forward_cast

    if parent == "EQTransformer" and dtype == "fp16":
        raise ValueError(EQT_FP16_MESSAGE)

    model = getattr(sbm, parent).from_pretrained(child)
    model.eval()
    if gpu and torch.cuda.is_available():
        model.to(torch.device(device))
    if dtype in ("bf16", "fp16"):
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
        model.to(torch_dtype)
        model = _wrap_forward_cast(model, torch_dtype)
    return model


def _picks_from_classify_output(
    out: Any,
    stations: Sequence[str],
    orig_starts: Dict[str, Any],
    sr: float = 100.0,
) -> Dict[str, Dict[str, List[float]]]:
    picks: Dict[str, Dict[str, List[float]]] = {
        sta: {"p": [], "s": []} for sta in stations
    }
    for pick in getattr(out, "picks", None) or []:
        pt = (
            getattr(pick, "peak_time", None)
            or getattr(pick, "start_time", None)
            or getattr(pick, "time", None)
        )
        ph = str(getattr(pick, "phase", "") or "").upper()
        sta = str(getattr(pick, "trace_id", "") or "")
        if "." in sta:
            parts = sta.split(".")
            sta = parts[1] if len(parts) > 1 else parts[0]
        if not sta or sta not in orig_starts or pt is None:
            continue
        samp = float(pt - orig_starts[sta]) * sr
        bucket = picks.setdefault(sta, {"p": [], "s": []})
        if ph == "P":
            bucket["p"].append(samp)
        elif ph == "S":
            bucket["s"].append(samp)
    return picks


def _merge_pick_dicts(
    a: Dict[str, Dict[str, List[float]]],
    b: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for sta in set(a) | set(b):
        out[sta] = {
            "p": list(a.get(sta, {}).get("p", [])) + list(b.get(sta, {}).get("p", [])),
            "s": list(a.get(sta, {}).get("s", [])) + list(b.get(sta, {}).get("s", [])),
        }
    return out


def _run_annotate_precision_repeat(
    *,
    net_dir: Path,
    stations: List[str],
    model_name: str,
    device: str,
    n_cpus: int,
    torch_threads: Optional[int],
    in_samples: int,
    overlap_samples: int,
    batch_size: int,
    p_threshold: float,
    s_threshold: float,
    gpu: bool,
    dtype: str,
    picks_path: Path,
    packaging: str = "merged",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[float]]]]:
    """Time annotate only; classify_aggregate runs after the timed stages.

    ``packaging``:
      * ``merged`` — one ``annotate()`` on the full-network Stream
      * ``sequential`` — one ``annotate()`` per station, strictly in order
    """
    from obspy import Stream
    import torch
    from rapid.api import classify_from_annotations
    from rapid.data import load_all_streams
    from rapid.orchestration.support.timing_util import SeisBenchStageProbes

    packaging = str(packaging or "merged").lower().strip()
    if packaging not in ("merged", "sequential"):
        raise ValueError(f"packaging must be merged|sequential, got {packaging!r}")

    spec = MODELS[model_name]
    branches = spec["branches"] or [{"parent": spec["parent"], "child": spec["child"]}]

    st = StageTimes()
    with st.stage("framework_init"):
        pin_threads(n_cpus, torch_threads=torch_threads)
        if gpu:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    models = []
    with st.stage("model_load"):
        for br in branches:
            models.append(_load_branch(br["parent"], br["child"], device, dtype, gpu))

    with st.stage("waveform_access"):
        streams = load_all_streams(net_dir, stations)
        merged = Stream()
        orig_starts: Dict[str, Any] = {}
        for sta, s in streams:
            merged += s
            if len(s):
                orig_starts[sta] = min(tr.stats.starttime for tr in s)

    ann_kw = dict(
        batch_size=int(batch_size),
        overlap=int(overlap_samples),
        strict=False,
        flexible_horizontal_components=True,
    )

    # Warmup (counted): first station through every branch.
    if streams:
        with st.stage("warmup"):
            for model in models:
                _ = model.annotate(streams[0][1], **ann_kw)
            if gpu:
                torch.cuda.synchronize()

    annotations = []
    preprocess_s = 0.0
    inference_s = 0.0
    for model in models:
        probes = SeisBenchStageProbes(model)
        probes.reset()
        t0 = time.perf_counter()
        if packaging == "merged":
            ann = model.annotate(merged, **ann_kw)
        else:
            ann = Stream()
            for _sta, s in streams:
                ann += model.annotate(s, **ann_kw)
        if gpu:
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        preprocess_s += probes.preprocess_s
        inference_s += max(0.0, wall - probes.preprocess_s)
        annotations.append(ann)
    st.add("preprocess", preprocess_s)
    st.add("inference", inference_s)

    # OFFLINE (untimed): classify_aggregate → discrete picks.
    t_pick0 = time.perf_counter()
    picks: Dict[str, Dict[str, List[float]]] = {
        sta: {"p": [], "s": []} for sta in stations
    }
    for model, ann in zip(models, annotations):
        out = classify_from_annotations(
            model,
            ann,
            P_threshold=p_threshold,
            S_threshold=s_threshold,
        )
        part = _picks_from_classify_output(out, stations, orig_starts)
        picks = _merge_pick_dicts(picks, part)
    offline_pick_s = time.perf_counter() - t_pick0
    picks_path.write_text(json.dumps(picks))

    n_per = len(fairness.window_starts(6000, in_samples, overlap_samples))
    repeat = st.as_repeat(
        success=True,
        extra={
            "n_windows": n_per * len(stations) * len(models),
            "batch_size": int(batch_size),
            "windows_per_station": float(n_per),
            "seisbench_native": True,
            "dtype": dtype,
            "n_branches": len(models),
            "packaging": packaging,
            "pick_extractor": "seisbench_classify_aggregate_offline",
            "offline_classify_aggregate_s": round(offline_pick_s, 4),
            "runtime_excludes_classify_aggregate": True,
        },
    )
    return repeat, picks


def _physical_gpu_id(args) -> int:
    return int(getattr(args, "gpu_id", 0) or 0)


def _cudnn_lib_dir() -> Optional[str]:
    """Prefer the pip nvidia-cudnn wheel over a conflicting system cuDNN."""
    try:
        import nvidia.cudnn  # type: ignore

        lib = Path(nvidia.cudnn.__file__).resolve().parent / "lib"
        if lib.is_dir():
            return str(lib)
    except Exception:
        pass
    return None


def _worker_env(args, gpu: bool) -> Dict[str, str]:
    env = dict(os.environ)
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(_physical_gpu_id(args))
        # System /lib64 often ships an older cuDNN (e.g. 9.1.1) that breaks
        # torch 2.11 (compiled against 9.10.2). Put the wheel first.
        cudnn = _cudnn_lib_dir()
        if cudnn:
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = cudnn + ((":" + prev) if prev else "")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    tt = getattr(args, "torch_threads", None)
    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    if tt == 0:
        for k in keys:
            env.pop(k, None)
    else:
        val = str(tt if tt is not None else args.n_cpus)
        for k in keys:
            env[k] = val
    return env


def _taskset_prefix(core_list: str) -> List[str]:
    """Pin the worker from process start (before torch spawns pools)."""
    cores = ",".join(c.strip() for c in str(core_list).split(",") if c.strip() != "")
    if not cores:
        return []
    return ["taskset", "-c", cores]


def _worker_argv(args, repeat_index: int) -> List[str]:
    argv = [
        "--method",
        args.method,
        "--n-stations",
        str(args.n_stations),
        "--model",
        args.model,
        "--device",
        args.device,
        "--n-cpus",
        str(args.n_cpus),
        "--gpu-id",
        str(args.gpu_id),
        "--batch-size",
        str(args.batch_size),
        "--repeats",
        str(args.repeats),
        "--p-threshold",
        str(args.p_threshold),
        "--s-threshold",
        str(args.s_threshold),
        "--tag",
        args.tag,
        "--net-root",
        str(args.net_root),
        "--results-root",
        str(args.results_root),
        "--core-list",
        args.core_list,
        "--packaging",
        getattr(args, "packaging", "merged"),
        "--repeat-index",
        str(repeat_index),
    ]
    if args.torch_threads is not None:
        argv += ["--torch-threads", str(args.torch_threads)]
    return argv


def _pick_quality(manifest_path: Path, picks: Dict[str, Dict[str, List[float]]], label: str) -> Dict[str, Any]:
    _t0, stations = load_manifest_catalog(manifest_path)
    catalog = catalog_from_manifest_stations(stations)
    return compare_pick_sets(
        catalog_by_station=catalog,
        detected_by_station=picks,
        label=label,
        reference_label="catalog",
    )


def _fp32_picks_path(args) -> Path:
    """Sibling FP32 trial picks for the same cell (method swapped to annotate_fp32)."""
    thr = args.torch_threads if args.torch_threads is not None else args.n_cpus
    bs = int(getattr(args, "batch_size", 256) or 256)
    packaging = str(getattr(args, "packaging", "merged") or "merged")
    modern = (
        args.results_root
        / "annotate_fp32"
        / "stead"
        / f"{args.n_stations}st"
        / args.model
        / args.device
        / f"cpus{args.n_cpus}"
        / f"thr{thr}"
        / f"bs{bs}"
        / packaging
        / args.tag
        / "picks.json"
    )
    if modern.is_file():
        return modern
    # Merged-matrix path (bs segment, no packaging).
    mid = (
        args.results_root
        / "annotate_fp32"
        / "stead"
        / f"{args.n_stations}st"
        / args.model
        / args.device
        / f"cpus{args.n_cpus}"
        / f"thr{thr}"
        / f"bs{bs}"
        / args.tag
        / "picks.json"
    )
    if mid.is_file():
        return mid
    return (
        args.results_root
        / "annotate_fp32"
        / "stead"
        / f"{args.n_stations}st"
        / args.model
        / args.device
        / f"cpus{args.n_cpus}"
        / f"thr{thr}"
        / args.tag
        / "picks.json"
    )


def run_one_repeat(args) -> int:
    baseline = _self_rss_mb()
    gpu = args.device == "gpu"
    device = "cuda:0" if gpu else "cpu"
    core_list = (
        [int(c) for c in str(args.core_list).split(",") if c.strip() != ""]
        if args.core_list
        else None
    )
    n_eff = _set_affinity(core_list)
    dtype = DTYPE_OF[args.method]
    in_samples, overlap_samples = _window_spec(args.model)
    net_dir = _net_dir(args.net_root, args.n_stations)

    from rapid.data import select_stations
    from rapid.orchestration.support.tools import ProcessTreeMemorySampler, process_tree_rss_mb

    stations = select_stations(net_dir, args.n_stations)
    rep_dir = _out_dir(args) / "repeats"
    rep_dir.mkdir(parents=True, exist_ok=True)
    picks_path = rep_dir / f"picks_{args.repeat_index}.json"

    sampler = ProcessTreeMemorySampler(interval_s=0.25)
    sampler.start()
    vram_sampler = None
    if gpu:
        from rapid.benchmark.fairness import GpuVramSampler

        vram_sampler = GpuVramSampler(
            process=sampler.process, gpu_index=_physical_gpu_id(args), interval_s=0.1
        )
        vram_sampler.start()
    from rapid.benchmark.fairness import ResourceUsageSampler

    res_sampler = ResourceUsageSampler(
        process=sampler.process,
        gpu_index=(_physical_gpu_id(args) if gpu else None),
        n_cores=n_eff,
        interval_s=0.25,
    ).start()

    ok = True
    err = ""
    repeat: Dict[str, Any] = {}
    try:
        repeat, _picks = _run_annotate_precision_repeat(
            net_dir=net_dir,
            stations=stations,
            model_name=args.model,
            device=device,
            n_cpus=n_eff,
            torch_threads=args.torch_threads,
            in_samples=in_samples,
            overlap_samples=overlap_samples,
            batch_size=args.batch_size,
            p_threshold=args.p_threshold,
            s_threshold=args.s_threshold,
            gpu=gpu,
            dtype=dtype,
            picks_path=picks_path,
            packaging=getattr(args, "packaging", "merged"),
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        ok = False
        err = str(exc)
        traceback.print_exc()
    finally:
        sampler.stop()
        if vram_sampler is not None:
            vram_sampler.stop()
        resources = res_sampler.stop()

    peak = sampler.peak_mb
    end = process_tree_rss_mb(sampler.process)
    mem = {
        "baseline_ram_mb": round(baseline, 2),
        "peak_ram_mb": round(peak, 2),
        "process_tree_ram_mb": round(end, 2),
        "ram_growth_mb": round(max(0.0, peak - baseline), 2),
        "baseline_pss_mb": round(sampler.baseline_pss_mb, 2),
        "peak_pss_mb": round(sampler.peak_pss_mb, 2),
        "process_tree_pss_mb": round(sampler.end_pss_mb, 2),
        "pss_growth_mb": round(max(0.0, sampler.peak_pss_mb - sampler.baseline_pss_mb), 2),
    }
    if vram_sampler is not None:
        mem["baseline_vram_mb"] = round(vram_sampler.baseline_mb, 2)
        mem["peak_vram_mb"] = round(vram_sampler.peak_mb, 2)
        mem["process_tree_vram_mb"] = round(vram_sampler.end_mb, 2)
        mem["vram_growth_mb"] = round(
            max(0.0, vram_sampler.peak_mb - vram_sampler.baseline_mb), 2
        )

    if not ok:
        rec = {"repeat_index": args.repeat_index, "success": False, "error": err, **mem, **resources}
    else:
        rec = {**repeat, "repeat_index": args.repeat_index, **mem, **resources}
    (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
    print(
        f"  [repeat {args.repeat_index}] "
        f"{'ok total=%.2fs' % rec.get('total_s', 0.0) if ok else 'FAILED: ' + err}"
    )
    return 0 if ok else 1


def run_driver(args) -> int:
    if args.model in EQT_MODELS and args.method == "annotate_fp16":
        print(
            f"SKIP {args.model} + fp16 (EQTransformer pad sentinel overflows fp16)",
            file=sys.stderr,
        )
        return 0

    gpu = args.device == "gpu"
    out_dir = _out_dir(args)
    rep_dir = out_dir / "repeats"
    result_path = out_dir / "result.json"
    net_dir = _net_dir(args.net_root, args.n_stations)
    manifest_path = net_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: missing manifest {manifest_path}", file=sys.stderr)
        return 2

    def _result_ok(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            r = json.loads(path.read_text())
            return float((r.get("timing") or {}).get("success_rate") or 0) >= 1.0
        except Exception:
            return False

    def _completed(repeats_dir: Path) -> int:
        if not repeats_dir.is_dir():
            return 0
        n = 0
        for i in range(args.repeats):
            f = repeats_dir / f"repeat_{i}.json"
            if f.is_file():
                try:
                    if json.loads(f.read_text()).get("success"):
                        n += 1
                except Exception:
                    pass
        return n

    # Resume: modern path, or legacy merged paths when packaging=merged.
    if args.resume and _result_ok(result_path) and _completed(rep_dir) >= args.repeats:
        print(f"[resume] complete -> {result_path}")
        return 0

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    lock_path = out_dir / ".owner_pid"
    if args.resume and lock_path.is_file():
        try:
            owner = int(lock_path.read_text().strip().split()[0])
        except Exception:
            owner = 0
        if owner and owner != os.getpid() and _pid_alive(owner):
            print(f"[lock] cell owned by pid {owner}, skipping duplicate start")
            return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(f"{os.getpid()}\n")
    packaging = str(getattr(args, "packaging", "merged") or "merged")
    if args.resume and packaging == "merged":
        for legacy in (
            _legacy_out_dir_merged_bs(args),
            _legacy_out_dir_bs256(args) if int(args.batch_size) == 256 else None,
        ):
            if legacy is None:
                continue
            if _result_ok(legacy / "result.json"):
                print(f"[resume] complete (legacy merged) -> {legacy / 'result.json'}")
                return 0

    for i in range(args.repeats):
        f = rep_dir / f"repeat_{i}.json"
        if args.resume and f.is_file():
            try:
                if json.loads(f.read_text()).get("success"):
                    continue
            except Exception:
                pass
        # Reuse successful legacy merged repeats into the modern tree when applicable.
        if args.resume and packaging == "merged":
            for leg_root in (
                _legacy_out_dir_merged_bs(args),
                _legacy_out_dir_bs256(args) if int(args.batch_size) == 256 else None,
            ):
                if leg_root is None:
                    continue
                leg = leg_root / "repeats" / f"repeat_{i}.json"
                if leg.is_file():
                    try:
                        if json.loads(leg.read_text()).get("success"):
                            import shutil

                            rep_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(leg, f)
                            lp = leg.parent / f"picks_{i}.json"
                            if lp.is_file():
                                shutil.copy2(lp, rep_dir / f"picks_{i}.json")
                            break
                    except Exception:
                        pass
        cmd = (
            _taskset_prefix(args.core_list)
            + [sys.executable, str(Path(__file__).resolve())]
            + _worker_argv(args, i)
        )
        subprocess.run(cmd, cwd=str(RAPID_ROOT), env=_worker_env(args, gpu))

    from rapid.benchmark.fairness import MEMORY_KEYS, RESOURCE_KEYS, summarize_pick_quality
    from rapid.benchmark.pick_export import write_picks_json

    timing_repeats: List[Dict[str, Any]] = []
    memory_repeats: List[Dict[str, Any]] = []
    resource_repeats: List[Dict[str, Any]] = []
    pq_repeats: List[Dict[str, Any]] = []
    vs_fp32_repeats: List[Dict[str, Any]] = []
    last_picks: Dict[str, Dict[str, List[float]]] = {}
    fp32_picks = None
    fp32_path = _fp32_picks_path(args)
    if args.method != "annotate_fp32" and fp32_path.is_file():
        try:
            fp32_picks = json.loads(fp32_path.read_text())
        except Exception:
            fp32_picks = None

    for i in range(args.repeats):
        f = rep_dir / f"repeat_{i}.json"
        if not f.is_file():
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        timing_repeats.append(rec)
        if not rec.get("success"):
            continue
        memory_repeats.append({k: rec[k] for k in MEMORY_KEYS if k in rec} | {"repeat_index": i})
        resource_repeats.append(
            {k: rec[k] for k in RESOURCE_KEYS if k in rec} | {"repeat_index": i}
        )
        pf = rep_dir / f"picks_{i}.json"
        if not pf.is_file():
            continue
        try:
            picks_i = json.loads(pf.read_text())
        except Exception:
            continue
        last_picks = picks_i
        try:
            pq_i = _pick_quality(manifest_path, picks_i, label=f"{args.method}_{args.tag}_r{i}")
            pq_i["repeat_index"] = i
            pq_repeats.append(pq_i)
        except Exception:
            pass
        if fp32_picks is not None:
            try:
                vs = compare_pick_sets(
                    catalog_by_station=fp32_picks,
                    detected_by_station=picks_i,
                    label=f"{args.method}_vs_fp32_r{i}",
                    reference_label="annotate_fp32",
                )
                vs["repeat_index"] = i
                vs_fp32_repeats.append(vs)
            except Exception:
                pass

    in_samples, overlap_samples = _window_spec(args.model)
    n_windows = (
        len(fairness.window_starts(6000, in_samples, overlap_samples)) * args.n_stations
    )
    n_branches = 2 if args.model == "EQCCT" else 1
    thr = args.torch_threads if args.torch_threads is not None else args.n_cpus
    meta = dict(
        method=args.method,
        family="annotate_precision",
        dataset="stead",
        n_stations=args.n_stations,
        model=args.model,
        parent=MODELS[args.model]["parent"],
        child=MODELS[args.model]["child"],
        device=args.device,
        n_cpus=args.n_cpus,
        torch_threads=thr,
        gpu_id=(args.gpu_id if gpu else None),
        in_samples=in_samples,
        overlap_samples=overlap_samples,
        dtype=DTYPE_OF[args.method],
        batch_size=args.batch_size,
        packaging=str(getattr(args, "packaging", "merged") or "merged"),
        n_windows=n_windows * n_branches,
        n_branches=n_branches,
        repeats=args.repeats,
        tag=args.tag,
        pick_extractor="seisbench_classify_aggregate_offline",
        runtime_excludes_classify_aggregate=True,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
    )
    pq = summarize_pick_quality(pq_repeats) if pq_repeats else None
    vs_fp32 = summarize_pick_quality(vs_fp32_repeats) if vs_fp32_repeats else None
    if last_picks:
        write_picks_json(out_dir / "picks.json", last_picks)

    result = build_result(
        meta=meta,
        timing_repeats=timing_repeats,
        memory_repeats=memory_repeats,
        pick_quality=pq,
        resource_repeats=resource_repeats,
    )
    if vs_fp32 is not None:
        result["pick_quality_vs_fp32"] = vs_fp32
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {result_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--n-stations", type=int, required=True, choices=[250, 580])
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-cpus", type=int, default=20)
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Torch/OpenMP thread budget. Default None = n_cpus. "
        "Pass 0 for untuned SeisBench/torch defaults.",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--packaging",
        default="merged",
        choices=["merged", "sequential"],
        help="merged = one annotate() on the full network Stream; "
        "sequential = one annotate() per station (orch one-station-per-actor twin).",
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--tag", default="ann_prec")
    ap.add_argument(
        "--net-root",
        type=Path,
        default=RAPID_ROOT / "data" / "seisbench_networks",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=RAPID_ROOT / "results" / "annotate_precision",
    )
    ap.add_argument("--core-list", default="")
    ap.add_argument(
        "--repeat-index",
        type=int,
        default=-1,
        help="If >=0, run exactly this one repeat (worker). Else driver.",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.repeat_index >= 0:
        return run_one_repeat(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
