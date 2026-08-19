#!/usr/bin/env python3
"""Orchestration trial: Model-Actor / Ripper / hybrid × annotate-fp32/bf16.

No fp16 in this suite. Batch size is locked to the merged-study winners
(fp32/bf16 = 512).

See ``benchmarks/isolation/README_ORCH_ANNOTATE.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

RAPID_ROOT = Path(__file__).resolve().parents[2]
if str(RAPID_ROOT) not in sys.path:
    sys.path.insert(0, str(RAPID_ROOT))

from rapid.benchmark.arrival import (  # noqa: E402
    CHUNK_S,
    DELAY_CHOICES_S,
    delay_summary,
    group_size,
    make_ready_times,
    percentiles,
)
from rapid.benchmark.fairness import StageTimes, build_result, pin_threads  # noqa: E402
from rapid.benchmark.orch_dispatch import WorkerPool, dispatch as orch_dispatch  # noqa: E402
from rapid.benchmark.pick_quality import (  # noqa: E402
    catalog_from_manifest_stations,
    compare_pick_sets,
    load_manifest_catalog,
)

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

METHODS = ("annotate_fp32", "annotate_bf16")
DTYPE_OF = {
    "annotate_fp32": "fp32",
    "annotate_bf16": "bf16",
}
# Merged-network annotate study (stead_iso_2026-08-13): lowest mean inference.
BEST_BATCH = {"fp32": 512, "bf16": 512}
COMPOSITIONS = (
    "ma",
    "ripper",
    "ma_ontime_rp_delayed",
    "rp_ontime_ma_delayed",
)
PACKAGINGS = ("s1", "sg")
ARRIVALS = ("playback", "staggered")
FILLS = ("static", "eager", "w5", "w10")
ORCHS = ("modelactor", "ripper")  # backward-compatible aliases for --orch


def composition_of(args) -> str:
    c = str(getattr(args, "composition", "") or "").strip()
    if c:
        return c
    orch = str(getattr(args, "orch", "") or "").strip()
    if orch == "modelactor":
        return "ma"
    if orch == "ripper":
        return "ripper"
    return "ma"


def k_split(args) -> Tuple[int, int]:
    kma = int(getattr(args, "k_ma", 0) or 0)
    krp = int(getattr(args, "k_rp", 0) or 0)
    k = int(getattr(args, "n_instances", 0) or 0)
    comp = composition_of(args)
    if kma or krp:
        return max(0, kma), max(0, krp)
    if comp == "ma":
        return max(1, k), 0
    if comp == "ripper":
        return 0, max(1, k)
    return max(0, kma), max(0, krp)


def fill_of(args) -> str:
    f = str(getattr(args, "fill", "") or "").strip()
    arrival = str(getattr(args, "arrival", "") or "")
    if arrival == "playback":
        return "static"
    if f:
        return f
    mw = float(getattr(args, "max_wait_s", 0) or 0)
    if mw <= 0:
        return "eager"
    if mw <= 5:
        return "w5"
    return "w10"


def max_wait_of(fill: str) -> float:
    return {"static": -1.0, "eager": 0.0, "w5": 5.0, "w10": 10.0}.get(fill, 0.0)


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
    if model in ("PhaseNet", "PhaseNetLight"):
        return 3001, 0
    return 6000, 0


def _batch_for(dtype: str, override: int) -> int:
    if int(override) > 0:
        return int(override)
    return int(BEST_BATCH[dtype])


def _out_dir(args) -> Path:
    dtype = DTYPE_OF[args.method]
    bs = _batch_for(dtype, args.batch_size)
    kma, krp = k_split(args)
    return (
        args.results_root
        / composition_of(args)
        / args.method
        / "stead"
        / f"{args.n_stations}st"
        / args.model
        / args.device
        / f"kma{kma}_krp{krp}"
        / args.packaging
        / args.arrival
        / fill_of(args)
        / f"bs{bs}"
        / args.tag
    )


def _picks_from_classify_output(
    out: Any,
    orig_starts: Dict[str, Any],
    sr: float = 100.0,
) -> Dict[str, Dict[str, List[float]]]:
    picks: Dict[str, Dict[str, List[float]]] = {
        sta: {"p": [], "s": []} for sta in orig_starts
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


def _merge_group(by_sta: Dict[str, Any], stas: Sequence[str]):
    from obspy import Stream

    merged = Stream()
    orig_starts: Dict[str, Any] = {}
    for sta in stas:
        stq = by_sta.get(sta)
        if stq is None or len(stq) == 0:
            continue
        merged += stq
        orig_starts[sta] = min(tr.stats.starttime for tr in stq)
    return merged, orig_starts


def _station_events_from_tasks(tasks: List[Dict[str, Any]], ready: Dict[str, float]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for t in tasks:
        queued = float(t["queued_s"])
        returned = float(t["returned_s"])
        inf = float(t.get("inference_s") or 0.0)
        n = max(1, len(t["stations"]))
        for sta in t["stations"]:
            rdy = float(ready.get(sta, 0.0))
            events.append(
                {
                    "station": sta,
                    "ready_s": round(rdy, 6),
                    "queued_s": round(queued, 6),
                    "returned_s": round(returned, 6),
                    "queue_s": round(queued - rdy, 6),  # start-ready
                    "e2e_s": round(returned - rdy, 6),  # finish-ready
                    "service_s": round(returned - queued, 6),
                    "inference_share_s": round(inf / n, 6),
                    "n_in_group": n,
                    "task_id": t["task_id"],
                    "worker": t.get("worker"),
                }
            )
    return events


def _latency_block(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    e2e = [e["e2e_s"] for e in events]
    queue = [e["queue_s"] for e in events]
    service = [e["service_s"] for e in events]
    return {
        "n_station_samples": len(events),
        "e2e_finish_minus_ready": percentiles(e2e),
        "queue_start_minus_ready": percentiles(queue),
        "service_returned_minus_queued": percentiles(service),
    }


def _skip_reason(args) -> Optional[str]:
    if args.method == "annotate_fp16":
        return "fp16 is excluded from the orchestration suite"
    comp = composition_of(args)
    kma, krp = k_split(args)
    if args.arrival == "playback" and comp in (
        "ma_ontime_rp_delayed",
        "rp_ontime_ma_delayed",
    ):
        return "hybrid compositions are staggered-only (playback has no delayed pool)"
    if comp in ("ma_ontime_rp_delayed", "rp_ontime_ma_delayed") and (kma < 1 or krp < 1):
        return "hybrid compositions need k_ma>=1 and k_rp>=1"
    if fill_of(args) != "static" and args.arrival == "playback":
        return "playback uses static partition (fill=static)"
    return None


# ---------------------------------------------------------------------------
# Dispatch (Ray submit wrappers + orch_dispatch)
# ---------------------------------------------------------------------------


def _submit_one(
    *,
    kind: str,
    actors,
    ripper_remote,
    worker: Any,
    stream,
    cls_kw: Dict[str, Any],
    gpu: bool,
    gpu_frac: float,
):
    if kind == "modelactor":
        return actors[int(worker)].classify.remote(stream, **{
            k: v for k, v in cls_kw.items() if not str(k).startswith("_")
        })
    opts: Dict[str, Any] = {"num_cpus": 0 if gpu else 1}
    if gpu:
        opts["num_gpus"] = float(gpu_frac)
    return ripper_remote.options(**opts).remote(
        cls_kw["_model_name"],
        cls_kw["_dtype"],
        gpu,
        cls_kw["_batch_size"],
        cls_kw["_overlap"],
        stream,
        cls_kw["P_threshold"],
        cls_kw["S_threshold"],
        cls_kw["Detection_threshold"],
    )



# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one_repeat(args) -> int:
    baseline = _self_rss_mb()
    gpu = args.device == "gpu"
    core_list = (
        [int(c) for c in str(args.core_list).split(",") if c.strip() != ""]
        if args.core_list
        else None
    )
    n_eff = _set_affinity(core_list)
    dtype = DTYPE_OF[args.method]
    batch_size = _batch_for(dtype, args.batch_size)
    in_samples, overlap_samples = _window_spec(args.model)
    net_dir = _net_dir(args.net_root, args.n_stations)
    kma, krp = k_split(args)
    kma = min(kma, int(args.n_stations)) if kma else 0
    krp = min(krp, int(args.n_stations)) if krp else 0
    fill = fill_of(args)
    delay_seed = int(args.seed) + int(args.repeat_index)

    from rapid.data import load_all_streams, select_stations
    from rapid.orchestration.support.tools import (
        ProcessTreeMemorySampler,
        process_tree_rss_mb,
        resolve_ray_temp_dir,
    )

    stations = select_stations(net_dir, args.n_stations)
    ready = make_ready_times(stations, mode=args.arrival, seed=delay_seed)
    rep_dir = _out_dir(args) / "repeats"
    rep_dir.mkdir(parents=True, exist_ok=True)

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

    # One thread per instance; Ray workers inherit this.
    pin_threads(n_eff, torch_threads=1)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"

    ok = True
    err = ""
    dispatch_out: Dict[str, Any] = {}
    st = StageTimes()
    try:
        with st.stage("framework_init"):
            import ray

            ray.init(
                num_cpus=n_eff,
                num_gpus=(1 if gpu else 0),
                include_dashboard=False,
                ignore_reinit_error=True,
                logging_level="ERROR",
                _temp_dir=resolve_ray_temp_dir(args.tmp_dir),
            )

        from rapid.orchestration.actors.orch_annotate import (
            OrchAnnotateActor,
            orch_ripper_classify,
        )

        k_gpu_users = max(1, kma + krp)
        gpu_frac = (1.0 / k_gpu_users) if gpu else 0.0

        def _spawn_actors(n: int):
            out = []
            for _ in range(n):
                if gpu:
                    out.append(
                        OrchAnnotateActor.options(num_gpus=gpu_frac, num_cpus=0).remote(
                            model_name=args.model,
                            annotate_dtype=dtype,
                            use_gpu=True,
                            annotate_batch_size=batch_size,
                            overlap_samples=overlap_samples,
                            gpus_to_use=([0] if gpu else False),
                        )
                    )
                else:
                    out.append(
                        OrchAnnotateActor.options(num_cpus=1).remote(
                            model_name=args.model,
                            annotate_dtype=dtype,
                            use_gpu=False,
                            annotate_batch_size=batch_size,
                            overlap_samples=overlap_samples,
                            gpus_to_use=False,
                        )
                    )
            return out

        ma_actors = []
        with st.stage("model_load"):
            if kma:
                ma_actors = _spawn_actors(kma)
                ray.get([a.ready.remote() for a in ma_actors])

        with st.stage("warmup"):
            if ma_actors:
                ray.get([a.warmup.remote() for a in ma_actors])

        with st.stage("waveform_access"):
            streams = load_all_streams(net_dir, stations)

        cls_kw = dict(
            P_threshold=args.p_threshold,
            S_threshold=args.s_threshold,
            Detection_threshold=args.p_threshold,
            _model_name=args.model,
            _dtype=dtype,
            _batch_size=batch_size,
            _overlap=overlap_samples,
        )

        comp = composition_of(args)
        n_st = len(streams)
        pools: Dict[str, WorkerPool] = {}
        if comp == "ma":
            g = group_size(n_st, max(1, kma), args.packaging)
            pools["all"] = WorkerPool("all", "modelactor", max(1, kma), ma_actors, gpu_frac, g)
        elif comp == "ripper":
            g = group_size(n_st, max(1, krp), args.packaging)
            pools["all"] = WorkerPool("all", "ripper", max(1, krp), [], gpu_frac, g)
        elif comp == "ma_ontime_rp_delayed":
            n_ontime = sum(1 for t in ready.values() if t <= 0)
            n_delayed = n_st - n_ontime
            pools["ontime"] = WorkerPool(
                "ontime", "modelactor", kma, ma_actors, gpu_frac,
                group_size(max(1, n_ontime), kma, args.packaging),
            )
            pools["delayed"] = WorkerPool(
                "delayed", "ripper", krp, [], gpu_frac,
                group_size(max(1, n_delayed), krp, args.packaging),
            )
        else:  # rp_ontime_ma_delayed
            n_ontime = sum(1 for t in ready.values() if t <= 0)
            n_delayed = n_st - n_ontime
            pools["ontime"] = WorkerPool(
                "ontime", "ripper", krp, [], gpu_frac,
                group_size(max(1, n_ontime), krp, args.packaging),
            )
            pools["delayed"] = WorkerPool(
                "delayed", "modelactor", kma, ma_actors, gpu_frac,
                group_size(max(1, n_delayed), kma, args.packaging),
            )

        def submit_fn(**kw):
            return _submit_one(ripper_remote=orch_ripper_classify, **kw)

        def wait_fn(refs, num_returns=1, timeout=None):
            return ray.wait(refs, num_returns=num_returns, timeout=timeout)

        t_disp = time.perf_counter()
        dispatch_out = orch_dispatch(
            pools=pools,
            packaging=args.packaging,
            arrival=args.arrival,
            fill=fill,
            streams=streams,
            ready=ready,
            cls_kw=cls_kw,
            gpu=gpu,
            submit_fn=submit_fn,
            wait_fn=wait_fn,
            get_fn=ray.get,
            merge_group_fn=_merge_group,
            picks_fn=_picks_from_classify_output,
            chunk_s=float(getattr(args, "chunk_s", CHUNK_S) or CHUNK_S),
            max_wait_s=max_wait_of(fill),
        )
        dispatch_out["latency"] = _latency_block(dispatch_out.get("station_events") or [])
        dispatch_out["group_size"] = next(iter(pools.values())).g
        disp_wall = time.perf_counter() - t_disp
        st.add("inference", disp_wall)
        st.add(
            "pick_generation",
            sum(float(t.get("pick_extract_s") or 0.0) for t in dispatch_out.get("tasks") or []),
        )

        try:
            ray.shutdown()
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        import traceback

        ok = False
        err = str(exc)
        traceback.print_exc()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
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

    (rep_dir / f"delays_{args.repeat_index}.json").write_text(
        json.dumps(
            {
                "seed": delay_seed,
                "arrival": args.arrival,
                "delay_choices_s": list(DELAY_CHOICES_S),
                "summary": delay_summary(ready),
                "ready_s": ready,
            },
            indent=2,
        )
    )

    if not ok:
        rec = {
            "repeat_index": args.repeat_index,
            "success": False,
            "error": err,
            **mem,
            **resources,
        }
        (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
        print(f"  [repeat {args.repeat_index}] FAILED: {err}", file=sys.stderr)
        return 1

    picks = dispatch_out.pop("picks")
    (rep_dir / f"picks_{args.repeat_index}.json").write_text(json.dumps(picks))
    rec = st.as_repeat(
        success=True,
        extra={
            "repeat_index": args.repeat_index,
            "composition": composition_of(args),
            "k_ma": kma,
            "k_rp": krp,
            "n_instances": kma + krp,
            "fill": fill,
            "group_size": dispatch_out["group_size"],
            "n_tasks": dispatch_out["n_tasks"],
            "makespan_s": dispatch_out["makespan_s"],
            "compute_span_s": dispatch_out["compute_span_s"],
            "sum_busy_s": dispatch_out["sum_busy_s"],
            "idle_frac_wall": dispatch_out["idle_frac_wall"],
            "idle_frac_compute": dispatch_out["idle_frac_compute"],
            "stations_per_chunk": dispatch_out.get("stations_per_chunk"),
            "chunk_s": dispatch_out.get("chunk_s"),
            "batch_size": batch_size,
            "dtype": dtype,
            "delay_seed": delay_seed,
            "delay_summary": delay_summary(ready),
            "latency": dispatch_out["latency"],
            "n_station_events": len(dispatch_out["station_events"]),
            "station_events": dispatch_out["station_events"],
            "n_tasks_detail": dispatch_out["tasks"],
            **mem,
            **resources,
        },
    )
    (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
    lat = dispatch_out["latency"]["e2e_finish_minus_ready"]
    print(
        f"  [repeat {args.repeat_index}] ok makespan={dispatch_out['makespan_s']:.2f}s "
        f"e2e_p95={lat.get('p95')} n_tasks={dispatch_out['n_tasks']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _physical_gpu_id(args) -> int:
    return int(getattr(args, "gpu_id", 0) or 0)


def _cudnn_lib_dir() -> Optional[str]:
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
        cudnn = _cudnn_lib_dir()
        if cudnn:
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = cudnn + ((":" + prev) if prev else "")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    return env


def _taskset_prefix(core_list: str) -> List[str]:
    cores = ",".join(c.strip() for c in str(core_list).split(",") if c.strip() != "")
    if not cores:
        return []
    return ["taskset", "-c", cores]


def _worker_argv(args, repeat_index: int) -> List[str]:
    kma, krp = k_split(args)
    argv = [
        "--composition",
        composition_of(args),
        "--method",
        args.method,
        "--model",
        args.model,
        "--n-stations",
        str(args.n_stations),
        "--device",
        args.device,
        "--n-cpus",
        str(args.n_cpus),
        "--k-ma",
        str(kma),
        "--k-rp",
        str(krp),
        "--n-instances",
        str(kma + krp),
        "--packaging",
        args.packaging,
        "--arrival",
        args.arrival,
        "--fill",
        fill_of(args),
        "--batch-size",
        str(args.batch_size),
        "--gpu-id",
        str(args.gpu_id),
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
        "--tmp-dir",
        str(args.tmp_dir),
        "--core-list",
        args.core_list,
        "--seed",
        str(args.seed),
        "--chunk-s",
        str(getattr(args, "chunk_s", CHUNK_S)),
        "--repeat-index",
        str(repeat_index),
    ]
    return argv
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


def run_driver(args) -> int:
    reason = _skip_reason(args)
    if reason:
        print(f"SKIP {reason}", file=sys.stderr)
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

    if args.resume and _result_ok(result_path):
        n_ok = 0
        for i in range(args.repeats):
            f = rep_dir / f"repeat_{i}.json"
            if f.is_file():
                try:
                    if json.loads(f.read_text()).get("success"):
                        n_ok += 1
                except Exception:
                    pass
        if n_ok >= args.repeats:
            print(f"[resume] complete -> {result_path}")
            return 0

    for i in range(args.repeats):
        f = rep_dir / f"repeat_{i}.json"
        if args.resume and f.is_file():
            try:
                if json.loads(f.read_text()).get("success"):
                    continue
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
    last_picks: Dict[str, Dict[str, List[float]]] = {}
    all_events: List[Dict[str, Any]] = []
    per_repeat_lat: List[Dict[str, Any]] = []

    for i in range(args.repeats):
        f = rep_dir / f"repeat_{i}.json"
        if not f.is_file():
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        slim = {
            k: v
            for k, v in rec.items()
            if k not in ("station_events", "n_tasks_detail")
        }
        timing_repeats.append(slim)
        if not rec.get("success"):
            continue
        memory_repeats.append({k: rec[k] for k in MEMORY_KEYS if k in rec} | {"repeat_index": i})
        resource_repeats.append(
            {k: rec[k] for k in RESOURCE_KEYS if k in rec} | {"repeat_index": i}
        )
        ev = rec.get("station_events") or []
        all_events.extend(ev)
        if rec.get("latency"):
            per_repeat_lat.append({"repeat_index": i, **rec["latency"]})
        pf = rep_dir / f"picks_{i}.json"
        if not pf.is_file():
            continue
        try:
            picks_i = json.loads(pf.read_text())
        except Exception:
            continue
        last_picks = picks_i
        try:
            pq_i = _pick_quality(
                manifest_path, picks_i, label=f"{composition_of(args)}_{args.method}_{args.tag}_r{i}"
            )
            pq_i["repeat_index"] = i
            pq_repeats.append(pq_i)
        except Exception:
            pass

    dtype = DTYPE_OF[args.method]
    bs = _batch_for(dtype, args.batch_size)
    in_samples, overlap_samples = _window_spec(args.model)
    kma, krp = k_split(args)
    n_branches = 2 if args.model == "EQCCT" else 1
    meta = dict(
        family="orch_annotate",
        composition=composition_of(args),
        method=args.method,
        dtype=dtype,
        dataset="stead",
        n_stations=args.n_stations,
        model=args.model,
        parent=MODELS[args.model]["parent"],
        child=MODELS[args.model]["child"],
        n_branches=n_branches,
        device=args.device,
        n_cpus=args.n_cpus,
        k_ma=kma,
        k_rp=krp,
        n_instances=kma + krp,
        packaging=args.packaging,
        arrival=args.arrival,
        fill=fill_of(args),
        group_size=group_size(args.n_stations, max(1, kma + krp), args.packaging),
        batch_size=bs,
        batch_size_source="merged_study_best" if args.batch_size <= 0 else "override",
        best_batch_table=BEST_BATCH,
        in_samples=in_samples,
        overlap_samples=overlap_samples,
        gpu_id=(args.gpu_id if gpu else None),
        repeats=args.repeats,
        tag=args.tag,
        seed=args.seed,
        delay_choices_s=list(DELAY_CHOICES_S),
        chunk_s=float(getattr(args, "chunk_s", CHUNK_S) or CHUNK_S),
        max_wait_s=max_wait_of(fill_of(args)),
        torch_threads=1,
        pick_extractor="seisbench_classify_aggregate_in_actor",
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
        n_station_latency_samples=len(all_events),
    )
    pq = summarize_pick_quality(pq_repeats) if pq_repeats else None
    if last_picks:
        write_picks_json(out_dir / "picks.json", last_picks)

    result = build_result(
        meta=meta,
        timing_repeats=timing_repeats,
        memory_repeats=memory_repeats,
        pick_quality=pq,
        resource_repeats=resource_repeats,
    )
    # Extra orch metrics (not in StageTimes, so summarize here).
    ok_recs = [r for r in timing_repeats if r.get("success")]
    extra_keys = (
        "makespan_s",
        "compute_span_s",
        "sum_busy_s",
        "idle_frac_wall",
        "idle_frac_compute",
        "n_tasks",
    )
    orch_stats: Dict[str, Any] = {}
    from rapid.benchmark.fairness import _agg

    for key in extra_keys:
        stats = _agg([float(r[key]) for r in ok_recs if r.get(key) is not None])
        for sk, sv in stats.items():
            orch_stats[f"{key}_{sk}"] = sv
    result["orch"] = orch_stats
    result["latency"] = {
        "pooled_across_repeats": _latency_block(all_events),
        "per_repeat": per_repeat_lat,
        "note": (
            "Pooled N_stations × n_repeats samples. e2e = finish-ready, "
            "queue = start-ready, service = returned-queued."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {result_path}")
    return 0


def iter_matrix(
    *,
    layer: str = "all",
    quick: bool = False,
    skip_ripper_s1: Optional[bool] = None,
    skip_staggered_ripper: Optional[bool] = None,
    skip_fp32_realtime: Optional[bool] = None,
):
    """Yield cell dicts. Layers: playback, staggered, hybrid, all.

    Homogeneous Ripper + S1 reloads the model per station and is pathological
    (smoke GPU cells took ~25–28 min). Full layers skip those cells by default.

    After playback, dtype is locked to bf16 and homogeneous Ripper lost on an
    all-ready network. Staggered/hybrid therefore skip fp32 and homogeneous
    Ripper by default; hybrid polarities still test Ripper on the delay pool.

    The QUICK smoke still emits Ripper S1 so the existing GPU/CPU controls stay
    on disk; ``--resume`` will not re-queue finished smoke results.
    """
    models = ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
    methods = ["annotate_fp32", "annotate_bf16"]
    stations = [250, 580]
    packs = ["s1", "sg"]
    k_cpu_ma = [1, 2, 4, 5, 10, 20]
    k_gpu_ma = [1, 2, 4]
    k_cpu_rp = [1, 2, 4, 5]
    k_gpu_rp = [1, 2]
    if quick:
        models = ["EQCCT", "PhaseNet"]
        methods = ["annotate_bf16"]
        stations = [580]
        k_cpu_ma = [5, 20]
        k_gpu_ma = [1, 2]
        k_cpu_rp = [2, 5]
        k_gpu_rp = [1, 2]
        layer = "playback" if layer == "all" else layer
    if skip_ripper_s1 is None:
        skip_ripper_s1 = not quick
    if skip_staggered_ripper is None:
        skip_staggered_ripper = not quick
    if skip_fp32_realtime is None:
        skip_fp32_realtime = not quick

    def emit(comp, method, model, n_st, device, kma, krp, pack, arrival, fill):
        if skip_ripper_s1 and comp == "ripper" and pack == "s1":
            return None
        if skip_staggered_ripper and comp == "ripper" and arrival == "staggered":
            return None
        if skip_fp32_realtime and method == "annotate_fp32" and arrival == "staggered":
            return None
        return {
            "composition": comp,
            "method": method,
            "model": model,
            "n_stations": n_st,
            "device": device,
            "k_ma": kma,
            "k_rp": krp,
            "n_instances": kma + krp,
            "packaging": pack,
            "arrival": arrival,
            "fill": fill,
        }

    want = {"playback", "staggered", "hybrid"} if layer == "all" else {layer}

    if "playback" in want:
        for n_st in stations:
            for model in models:
                for method in methods:
                    for pack in packs:
                        for device, ks in (("cpu", k_cpu_ma), ("gpu", k_gpu_ma)):
                            for k in ks:
                                cell = emit("ma", method, model, n_st, device, k, 0, pack, "playback", "static")
                                if cell:
                                    yield cell
                        for device, ks in (("cpu", k_cpu_rp), ("gpu", k_gpu_rp)):
                            for k in ks:
                                cell = emit("ripper", method, model, n_st, device, 0, k, pack, "playback", "static")
                                if cell:
                                    yield cell

    if "staggered" in want:
        # Full CPU/GPU K sweep with eager fill. fill-G (w5, w10) on SG only,
        # at the slot-sized K values (CPU 5/20, GPU 2/4).
        for n_st in stations:
            for model in models:
                for method in methods:
                    for pack in packs:
                        fills = ["eager"] if pack == "s1" else ["eager", "w5", "w10"]
                        # MA
                        for device, ks, fill_ks in (
                            ("cpu", k_cpu_ma, {5, 20}),
                            ("gpu", k_gpu_ma, {2, 4}),
                        ):
                            for k in ks:
                                use_fills = ["eager"] if k not in fill_ks else fills
                                for fill in use_fills:
                                    cell = emit("ma", method, model, n_st, device, k, 0, pack, "staggered", fill)
                                    if cell:
                                        yield cell
                        # Ripper
                        for device, ks, fill_ks in (
                            ("cpu", k_cpu_rp, {2, 5}),
                            ("gpu", k_gpu_rp, {1, 2}),
                        ):
                            for k in ks:
                                use_fills = ["eager"] if k not in fill_ks else fills
                                for fill in use_fills:
                                    cell = emit("ripper", method, model, n_st, device, 0, k, pack, "staggered", fill)
                                    if cell:
                                        yield cell

    if "hybrid" in want:
        cpu_splits = [(10, 2), (5, 5), (4, 1)]
        gpu_splits = [(2, 1), (1, 1)]
        for n_st in stations:
            for model in models:
                for method in methods:
                    for pack in packs:
                        fills = ["eager"] if pack == "s1" else ["eager", "w10"]
                        for polarity in ("ma_ontime_rp_delayed", "rp_ontime_ma_delayed"):
                            for device, splits in (("cpu", cpu_splits), ("gpu", gpu_splits)):
                                for kma, krp in splits:
                                    for fill in fills:
                                        cell = emit(
                                            polarity, method, model, n_st, device,
                                            kma, krp, pack, "staggered", fill,
                                        )
                                        if cell:
                                            yield cell


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--composition", default="", choices=("",) + COMPOSITIONS)
    ap.add_argument("--orch", default="", choices=("",) + ORCHS, help="Alias: modelactor=ma, ripper=ripper.")
    ap.add_argument("--method", default="", choices=("",) + METHODS)
    ap.add_argument("--model", default="", choices=("",) + tuple(MODELS.keys()))
    ap.add_argument("--n-stations", type=int, default=0, choices=[0, 250, 580])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-cpus", type=int, default=20, help="Isolation slot size (cores).")
    ap.add_argument("--n-instances", type=int, default=0, help="Homogeneous K (used if k-ma/k-rp omitted).")
    ap.add_argument("--k-ma", type=int, default=0, help="Model-Actor instance count.")
    ap.add_argument("--k-rp", type=int, default=0, help="Ripper in-flight count.")
    ap.add_argument("--packaging", default="", choices=("",) + PACKAGINGS)
    ap.add_argument("--arrival", default="", choices=("",) + ARRIVALS)
    ap.add_argument("--fill", default="", choices=("",) + FILLS)
    ap.add_argument("--chunk-s", type=float, default=CHUNK_S)
    ap.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print cell JSON lines and exit.",
    )
    ap.add_argument(
        "--layer",
        default="all",
        choices=["playback", "staggered", "hybrid", "all"],
        help="Matrix layer for --print-matrix / launcher.",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--skip-ripper-s1",
        action="store_true",
        default=None,
        help="Omit homogeneous Ripper+S1 cells (default on unless --quick).",
    )
    ap.add_argument(
        "--include-ripper-s1",
        action="store_true",
        help="Keep homogeneous Ripper+S1 cells in --print-matrix.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = locked best from merged study (fp32/bf16=512).",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--tag", default="orch_ann")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-wait-s", type=float, default=-1.0, help="Deprecated; use --fill.")
    ap.add_argument("--net-root", type=Path, default=RAPID_ROOT / "data" / "seisbench_networks")
    ap.add_argument("--results-root", type=Path, default=RAPID_ROOT / "results" / "orch_annotate")
    ap.add_argument("--tmp-dir", type=Path, default=RAPID_ROOT / "results" / "tmp_ray")
    ap.add_argument("--core-list", default="")
    ap.add_argument("--repeat-index", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.print_matrix:
        skip_rp_s1 = False if args.include_ripper_s1 else args.skip_ripper_s1
        cells = list(
            iter_matrix(
                layer=args.layer,
                quick=args.quick,
                skip_ripper_s1=skip_rp_s1,
            )
        )
        for cell in cells:
            print(json.dumps(cell))
        print(f"# n_cells={len(cells)} layer={args.layer}", file=sys.stderr)
        return 0

    if not args.composition and args.orch:
        args.composition = "ma" if args.orch == "modelactor" else "ripper"

    missing = [
        name
        for name, val in (
            ("--composition or --orch", args.composition or args.orch),
            ("--method", args.method),
            ("--model", args.model),
            ("--n-stations", args.n_stations),
            ("--packaging", args.packaging),
            ("--arrival", args.arrival),
        )
        if not val
    ]
    if missing:
        ap.error("missing required arguments: " + ", ".join(missing))
    kma, krp = k_split(args)
    if kma + krp < 1:
        ap.error("need --k-ma/--k-rp or --n-instances")

    if args.repeat_index >= 0:
        return run_one_repeat(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
