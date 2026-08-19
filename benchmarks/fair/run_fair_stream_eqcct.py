#!/usr/bin/env python3
"""Warm Model-Actor EQCCT (TensorFlow) head-to-head — station-group dispatch.

Keeps a persistent TF ModelActor pool warm across feeds, partitioning the
580-station STEAD inventory across actors (SCMLPick-style station groups).
Writes schema-compatible result.json under results/iso_full_benchmark/stream.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RAPID_ROOT = Path(__file__).resolve().parents[2]
EQCCTPRO_ROOT = RAPID_ROOT.parent
if str(RAPID_ROOT) not in sys.path:
    sys.path.insert(0, str(RAPID_ROOT))

from rapid.benchmark import fairness  # noqa: E402
from rapid.benchmark.fairness import StageTimes, build_result, pin_threads, summarize_pick_quality  # noqa: E402
from rapid.benchmark.pick_quality import (  # noqa: E402
    catalog_from_manifest_stations,
    compare_pick_sets,
    load_manifest_catalog,
)


def _eqcct_weights() -> tuple[Path, Path]:
    candidates = [
        EQCCTPRO_ROOT / "models" / "EQCCT",
        RAPID_ROOT / "models" / "EQCCT",
    ]
    for base in candidates:
        p = base / "test_trainer_024.h5"
        s = base / "test_trainer_021.h5"
        if p.is_file() and s.is_file():
            return p, s
    raise FileNotFoundError(
        "EQCCT weights not found under eqcctpro/models/EQCCT or RAPID/models/EQCCT"
    )


def _net_dir(net_root: Path, n_stations: int) -> Path:
    return net_root / f"stead_{n_stations}st"


def _self_rss_mb() -> float:
    import psutil

    try:
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


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


def _out_dir(args) -> Path:
    return (
        args.results_root / "streaming" / "stream_modelactor_eqcct" / "stead"
        / f"{args.n_stations}st" / "EQCCT" / args.tag
    )


def _preprocess_args(args) -> Dict[str, Any]:
    return {
        "overlap": float(args.overlap_min),
        "batch_size": int(args.batch_size),
        "normalization_mode": "std",
        "P_threshold": float(args.p_threshold),
        "S_threshold": float(args.s_threshold),
        "Detection_threshold": float(args.detection_threshold),
        # Filter defaults used by resolve_waveform_filter_params
        "use_filter": True,
        "filter_type": "bandpass",
        "freqmin": 1.0,
        "freqmax": 45.0,
        "corners": 2,
        "zerophase": True,
    }


def run_one_repeat(args) -> int:
    baseline = _self_rss_mb()
    gpu = args.device == "gpu"
    core_list = [int(c) for c in str(args.core_list).split(",") if c.strip() != ""] if args.core_list else None
    n_eff = _set_affinity(core_list)
    n_cpus = n_eff if core_list else args.n_cpus
    n_actors = args.concurrency or max(1, min(n_cpus, args.n_stations))
    pin_threads(n_cpus)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"

    from rapid.orchestration.support.tools import ProcessTreeMemorySampler, process_tree_rss_mb

    sampler = ProcessTreeMemorySampler(interval_s=0.5)
    sampler.start()
    from rapid.benchmark.fairness import ResourceUsageSampler

    res_sampler = ResourceUsageSampler(
        process=sampler.process,
        gpu_index=(int(args.gpu_id or 0) if gpu else None),
        n_cores=n_cpus,
        interval_s=0.25,
    ).start()

    rep_dir = _out_dir(args) / "repeats"
    rep_dir.mkdir(parents=True, exist_ok=True)
    net_dir = _net_dir(args.net_root, args.n_stations)
    p_model, s_model = _eqcct_weights()
    prep = _preprocess_args(args)

    ok = True
    err = ""
    st = StageTimes()
    feeds: List[Dict[str, Any]] = []
    last_picks: Dict[str, Dict[str, List[float]]] = {}

    try:
        with st.stage("framework_init"):
            import ray
            from rapid.orchestration.support.tools import resolve_ray_temp_dir

            ray.init(
                num_cpus=n_cpus,
                num_gpus=(1 if gpu else 0),
                include_dashboard=False,
                ignore_reinit_error=True,
                logging_level="ERROR",
                _temp_dir=resolve_ray_temp_dir(args.tmp_dir),
            )

        with st.stage("model_load"):
            from rapid.orchestration.actors.parallelization import ModelActor

            actors = []
            gpu_frac = (1.0 / n_actors) if gpu else 0.0
            for _ in range(n_actors):
                if gpu:
                    actors.append(
                        ModelActor.options(num_gpus=gpu_frac, num_cpus=0).remote(
                            str(p_model), str(s_model), gpus_to_use=[0], use_gpu=True,
                            intra_threads=1, inter_threads=1,
                        )
                    )
                else:
                    actors.append(
                        ModelActor.options(num_cpus=1).remote(
                            str(p_model), str(s_model), gpus_to_use=False, use_gpu=False,
                            intra_threads=1, inter_threads=1,
                        )
                    )
            ray.get([a.ready.remote() for a in actors])

        from rapid.data import load_all_streams, select_stations

        stations = select_stations(net_dir, args.n_stations)
        session_t0 = time.monotonic()
        for k in range(args.n_feeds):
            wait_s = late_s = 0.0
            if args.feed_interval_s > 0:
                target = session_t0 + k * args.feed_interval_s
                now = time.monotonic()
                wait_s = max(0.0, target - now)
                late_s = max(0.0, now - target)
                if wait_s > 0:
                    time.sleep(wait_s)

            t = time.perf_counter()
            streams = load_all_streams(net_dir, stations)
            wf_s = time.perf_counter() - t

            buckets: List[List[Any]] = [[] for _ in range(n_actors)]
            for idx, item in enumerate(streams):
                buckets[idx % n_actors].append(item)

            t = time.perf_counter()
            refs = []
            for actor_idx, bucket in enumerate(buckets):
                if not bucket:
                    continue
                refs.append(actors[actor_idx].predict_station_group.remote(bucket, prep))
            outs = ray.get(refs)
            inf_s = time.perf_counter() - t

            feed_picks: Dict[str, Dict[str, List[float]]] = {sta: {"p": [], "s": []} for sta in stations}
            for part in outs:
                for sta, picks in part.items():
                    feed_picks[sta] = picks
            last_picks = feed_picks
            pick_s = 0.0

            st.add("waveform_access", wf_s)
            st.add("inference", inf_s)
            st.add("pick_generation", pick_s)
            feeds.append({
                "feed_index": k,
                "scheduled_offset_s": k * args.feed_interval_s,
                "wait_s": round(wait_s, 3),
                "late_s": round(late_s, 3),
                "waveform_access_s": round(wf_s, 6),
                "inference_s": round(inf_s, 6),
                "pick_generation_s": round(pick_s, 6),
                "feed_total_s": round(wf_s + inf_s + pick_s, 6),
                "n_windows": len(streams),
                "rss_mb_now": round(process_tree_rss_mb(sampler.process), 2),
                "peak_ram_mb_so_far": round(sampler.peak_mb, 2),
            })
            print(f"  [repeat {args.repeat_index}] feed {k}: total={wf_s + inf_s:.2f}s "
                  f"(wf={wf_s:.2f} inf={inf_s:.2f}) wait={wait_s:.1f}s late={late_s:.1f}s")

        session_wall_s = time.monotonic() - session_t0
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        import traceback
        ok = False
        err = str(exc)
        traceback.print_exc()
        session_wall_s = 0.0
    finally:
        sampler.stop()
        resources = res_sampler.stop()

    if not ok:
        rec = st.as_repeat(success=False, extra={"error": err, "repeat_index": args.repeat_index})
        (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
        return 1

    cold = feeds[0]["feed_total_s"] if feeds else None
    warm = [f["feed_total_s"] for f in feeds[1:]] if len(feeds) > 1 else []
    wi = [f["inference_s"] for f in feeds[1:]] if len(feeds) > 1 else []
    latency = {
        "cold_feed_total_s": cold,
        "warm_feed_mean_s": statistics.mean(warm) if warm else None,
        "warm_feed_min_s": min(warm) if warm else None,
        "warm_feed_max_s": max(warm) if warm else None,
        "warm_feed_std_s": statistics.stdev(warm) if len(warm) > 1 else 0.0,
        "warm_inference_mean_s": statistics.mean(wi) if wi else None,
    }
    mem = {
        "baseline_ram_mb": round(baseline, 2),
        "peak_pss_mb": round(sampler.peak_mb, 2),
        "end_rss_mb": round(process_tree_rss_mb(sampler.process), 2),
    }
    rec = st.as_repeat(success=True, extra={
        "repeat_index": args.repeat_index,
        "n_feeds": args.n_feeds,
        "feed_interval_s": args.feed_interval_s,
        "session_wall_s": round(session_wall_s, 3),
        "n_actors": n_actors,
        "concurrency": n_actors,
        "feeds": feeds,
        **latency,
        **mem,
        **resources,
    })
    (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
    (rep_dir / f"picks_{args.repeat_index}.json").write_text(json.dumps(last_picks))
    print(f"  [repeat {args.repeat_index}] ok total={rec['total_s']:.2f}s")
    return 0


def run_driver(args) -> int:
    out_dir = _out_dir(args)
    rep_dir = out_dir / "repeats"
    result_path = out_dir / "result.json"
    net_dir = _net_dir(args.net_root, args.n_stations)
    manifest_path = net_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR missing manifest {manifest_path}", file=sys.stderr)
        return 2

    def _ok(i: int) -> bool:
        f = rep_dir / f"repeat_{i}.json"
        if not f.is_file():
            return False
        try:
            return bool(json.loads(f.read_text()).get("success"))
        except Exception:
            return False

    if args.resume and result_path.is_file() and all(_ok(i) for i in range(args.repeats)):
        print(f"[resume] complete -> {result_path}")
        return 0

    env = dict(os.environ)
    if args.device == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""

    for i in range(args.repeats):
        if args.resume and _ok(i):
            continue
        argv = [
            sys.executable, __file__,
            "--worker", "--repeat-index", str(i),
            "--device", args.device, "--n-stations", str(args.n_stations),
            "--n-cpus", str(args.n_cpus), "--concurrency", str(args.concurrency),
            "--core-list", args.core_list, "--gpu-id", str(args.gpu_id),
            "--n-feeds", str(args.n_feeds), "--feed-interval-s", str(args.feed_interval_s),
            "--repeats", str(args.repeats), "--tag", args.tag,
            "--net-root", str(args.net_root), "--results-root", str(args.results_root),
            "--tmp-dir", str(args.tmp_dir),
            "--p-threshold", str(args.p_threshold), "--s-threshold", str(args.s_threshold),
            "--detection-threshold", str(args.detection_threshold),
            "--batch-size", str(args.batch_size), "--overlap-min", str(args.overlap_min),
        ]
        print(f"=== EQCCT warm repeat {i}/{args.repeats - 1} ===")
        rc = subprocess.call(argv, env=env, cwd=str(RAPID_ROOT))
        if rc != 0:
            print(f"repeat {i} failed rc={rc}", file=sys.stderr)

    # Aggregate
    timing_repeats, latency_repeats, memory_repeats, resource_repeats, pq_repeats = [], [], [], [], []
    last_picks = {}
    MEMORY_KEYS = ("baseline_ram_mb", "peak_pss_mb", "end_rss_mb")
    LATENCY_KEYS = ("cold_feed_total_s", "warm_feed_mean_s", "warm_feed_min_s",
                    "warm_feed_max_s", "warm_feed_std_s", "warm_inference_mean_s")
    for i in range(args.repeats):
        f = rep_dir / f"repeat_{i}.json"
        if not f.is_file():
            continue
        rec = json.loads(f.read_text())
        timing_repeats.append(rec)
        if rec.get("success"):
            memory_repeats.append({k: rec[k] for k in MEMORY_KEYS if k in rec} | {"repeat_index": i})
            latency_repeats.append({k: rec[k] for k in LATENCY_KEYS if k in rec} | {"repeat_index": i})
            pf = rep_dir / f"picks_{i}.json"
            if pf.is_file():
                last_picks = json.loads(pf.read_text())
                try:
                    _t0, sta_cat = load_manifest_catalog(manifest_path)
                    catalog = catalog_from_manifest_stations(sta_cat)
                    pq_i = compare_pick_sets(
                        catalog_by_station=catalog, detected_by_station=last_picks,
                        label=f"stream_modelactor_eqcct_{args.tag}_r{i}", reference_label="catalog",
                    )
                    pq_i["repeat_index"] = i
                    pq_repeats.append(pq_i)
                except Exception:
                    pass

    cores = [int(c) for c in str(args.core_list).split(",") if c.strip() != ""]
    n_cpus = len(cores) if cores else args.n_cpus
    conc = args.concurrency or n_cpus
    meta = dict(
        method="stream_modelactor_eqcct", family="streaming", dataset="stead",
        n_stations=args.n_stations, model="EQCCT", parent="EQCCT", child="texnet",
        device=args.device, n_cpus=n_cpus, gpu_id=(args.gpu_id if args.device == "gpu" else None),
        concurrency=conc, dtype="fp32", batch_size=args.batch_size,
        n_feeds=args.n_feeds, feed_interval_s=args.feed_interval_s,
        back_to_back=bool(args.feed_interval_s <= 0),
        repeats=args.repeats, tag=args.tag,
        pick_extractor="eqcct_tf_picker",
        p_threshold=args.p_threshold, s_threshold=args.s_threshold,
    )
    from rapid.benchmark.fairness import summarize_pick_quality

    pq = summarize_pick_quality(pq_repeats) if pq_repeats else None
    result = build_result(meta=meta, timing_repeats=timing_repeats,
                          memory_repeats=memory_repeats, pick_quality=pq,
                          resource_repeats=resource_repeats)
    if latency_repeats:
        def _agg(xs):
            xs = [float(x) for x in xs if x is not None]
            if not xs:
                return {}
            out = {"mean": statistics.mean(xs), "min": min(xs), "max": max(xs)}
            if len(xs) > 1:
                out["std"] = statistics.stdev(xs)
            return out
        lat: Dict[str, Any] = {"repeats": latency_repeats}
        for key in LATENCY_KEYS:
            stats = _agg([r.get(key) for r in latency_repeats])
            for k, v in stats.items():
                lat[f"{key}_{k}"] = v
        result["latency"] = lat
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {result_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--repeat-index", type=int, default=-1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-stations", type=int, default=580)
    ap.add_argument("--n-cpus", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--core-list", default="")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-feeds", type=int, default=8)
    ap.add_argument("--feed-interval-s", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--tag", default="iso_cpu_580")
    ap.add_argument("--net-root", type=Path, default=RAPID_ROOT / "data" / "seisbench_networks")
    ap.add_argument("--results-root", type=Path, default=RAPID_ROOT / "results" / "iso_full_benchmark" / "stream")
    ap.add_argument("--tmp-dir", type=Path, default=Path.home() / "rapid_ray")
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--detection-threshold", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--overlap-min", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.worker:
        return run_one_repeat(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
