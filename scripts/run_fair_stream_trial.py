#!/usr/bin/env python3
"""Run ONE streaming (warm Model-Actor) fair-benchmark trial -> schema-v3 result.json.

Simulates a deployed real-time pipeline: Ray + the Model-Actor pool are created
ONCE, then the full synthetic network (250/580 stations) is fed to the actors
``--n-feeds`` times. Nothing is torn down between feeds, so feeds 1..N-1 show
the warm steady state while feed 0 carries any first-call overhead -- exactly
what a kept-alive deployment would see.

Two pacing modes
----------------
* ``--feed-interval-s 60`` (legacy): wall-clock paced feeds at t=0,60,120,...
  Real-time emulation, but each repeat takes >= n_feeds * interval.
* ``--feed-interval-s 0`` (back-to-back latency mode): feed k+1 is submitted
  the moment feed k completes. Warm-path latency does not depend on the idle
  gap between feeds, so this measures the same cold-feed and warm-feed
  latencies in seconds instead of minutes. ``scripts/run_latency_sweep.sh``
  drives a slim matrix in this mode.

Each repeat record carries per-feed detail plus cold/warm aggregates
(``cold_feed_total_s``, ``warm_feed_mean_s`` ...); the trial result adds a
``latency`` section aggregated across repeats. Feed 0 IS the deliberately
timed warmup of this family (the cold/warm split is the deliverable), so the
unified ``warmup`` stage stays 0 here.

Strategies
----------
* ``stream_modelactor``            -- actors run SeisBench ``model.classify()``
  end-to-end (same picking path as the cold ``modelactor`` strategy).
* ``stream_modelactor_slipstream`` -- actors run RAPID Slipstream (lean PyTorch)
  with the precision sweep (fp32 / fp16[+compile] / bf16[+compile]) and batch
  size sweep.

Stations are distributed round-robin across the actor pool (even split). The
actor count equals the marching concurrency (= CPU budget by default), each
actor pinned to one logical core's worth of threads.

Independent repeats: each repeat is ONE full session (create actors -> N paced
feeds -> teardown) in a fresh subprocess. Default 3 repeats (sessions are >= 3
minutes each).

Timing semantics
----------------
* ``framework_init``  -- ray.init
* ``model_load``      -- actor creation + model loads (ready barrier)
* ``waveform_access`` / ``inference`` / ``pick_generation`` -- summed over feeds
* ``total_s``         -- compute only (idle wait between feeds EXCLUDED)
* per-feed detail (incl. wait/late seconds and memory snapshots) is recorded in
  ``feeds`` inside each repeat record.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RAPID_ROOT = Path(__file__).resolve().parents[1]
EQCCTPRO_ROOT = RAPID_ROOT  # eqcctpro package is vendored inside RAPID
for p in (str(RAPID_ROOT), str(EQCCTPRO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rapid.benchmark import fairness  # noqa: E402
from rapid.benchmark.fairness import StageTimes, build_result, pin_threads  # noqa: E402
from rapid.benchmark.pick_quality import (  # noqa: E402
    catalog_from_manifest_stations,
    compare_pick_sets,
    load_manifest_catalog,
)

STRATEGIES = ("stream_modelactor", "stream_modelactor_slipstream", "stream_annotate",
              "stream_modelactor_2gpu")
MODELS: Dict[str, Dict[str, str]] = {
    "PhaseNet": {"parent": "PhaseNet", "child": "original"},
    "PhaseNetLight": {"parent": "PhaseNetLight", "child": "stead"},
    "EQTransformer": {"parent": "EQTransformer", "child": "original"},
    "EQT-NC": {"parent": "EQTransformer", "child": "original_nonconservative"},
}


def _net_dir(net_root: Path, dataset: str, n_stations: int, net_suffix: str) -> Path:
    return net_root / f"{dataset.lower()}_{n_stations}st{net_suffix}"


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
        args.results_root / "streaming" / args.strategy / args.dataset.lower()
        / f"{args.n_stations}st" / args.model / args.tag
    )


def _picks_from_classify_output(out, trace_start, sr: float = 100.0):
    """SeisBench/Slipstream ClassifyOutput-like -> {p:[samples], s:[samples]}."""
    p_list: List[float] = []
    s_list: List[float] = []
    for pick in (getattr(out, "picks", None) or []):
        pt = getattr(pick, "peak_time", None) or getattr(pick, "start_time", None) or getattr(pick, "time", None)
        ph = str(getattr(pick, "phase", "") or "").upper()
        if pt is None or trace_start is None:
            continue
        samp = float(pt - trace_start) * sr
        if ph == "P":
            p_list.append(samp)
        elif ph == "S":
            s_list.append(samp)
    return {"p": p_list, "s": s_list}


# ---------------------------------------------------------------------------
# Worker: ONE full streaming session (create actors -> N paced feeds)
# ---------------------------------------------------------------------------


def run_one_repeat(args) -> int:
    baseline = _self_rss_mb()  # clean floor BEFORE ray/torch/eqcctpro imports

    gpu = args.device == "gpu"
    core_list = [int(c) for c in str(args.core_list).split(",") if c.strip() != ""] if args.core_list else None
    n_eff = _set_affinity(core_list)
    n_cpus = n_eff if core_list else args.n_cpus
    n_actors = args.concurrency or max(1, min(n_cpus, args.n_stations))

    m = MODELS[args.model]
    net_dir = _net_dir(args.net_root, args.dataset, args.n_stations, args.net_suffix)
    trace_len = 3001 if args.net_suffix == "_w3001" else 6000

    # Size the driver's own torch pool to the core budget (driver does no heavy
    # inference, but keep it bounded), THEN force the thread env vars back to 1:
    # pin_threads() writes OMP/MKL/... = n_cpus into os.environ, and ray.init()
    # propagates the driver env to the raylet -> every actor. With n_actors ==
    # n_cpus, inherited n-thread budgets mean n*n OpenMP threads spin-waiting on
    # n cores (~700x slower inference, see smoke_v3 stream_mas). One compute
    # thread per actor keeps the pool's total budget == the marching CPUs.
    pin_threads(n_cpus)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"

    from eqcctpro.tools import ProcessTreeMemorySampler, process_tree_rss_mb

    sampler = ProcessTreeMemorySampler(interval_s=0.5)
    sampler.start()
    vram_sampler = None
    if gpu:
        from rapid.benchmark.fairness import GpuVramSampler

        vram_sampler = GpuVramSampler(
            process=sampler.process, gpu_index=int(args.gpu_id or 0), interval_s=0.1
        )
        vram_sampler.start()
    from rapid.benchmark.fairness import ResourceUsageSampler

    res_sampler = ResourceUsageSampler(
        process=sampler.process,
        gpu_index=(int(args.gpu_id or 0) if gpu else None),
        n_cores=n_cpus,
        interval_s=0.25,
    ).start()

    rep_dir = _out_dir(args) / "repeats"
    rep_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    err = ""
    st = StageTimes()
    feeds: List[Dict[str, Any]] = []
    last_picks: Dict[str, Dict[str, List[float]]] = {}

    # Warm-annotate baseline: keep ONE SeisBench model loaded in-process and call
    # model.annotate() on the whole merged network per feed (no Ray, no actors).
    # Feed 0 is cold (includes lazy CUDA init); feeds 1..N-1 are warm steady state
    # -- the fair "warm annotate per window" number to set against warm Model-Actor.
    ann_mode = args.strategy == "stream_annotate"
    # Spread the actor pool across BOTH physical GPUs (vs the default single-device
    # pool). Tests whether the GPU "loss" for Model-Actor is fundamental or just a
    # single-device contention artifact: with 2 GPUs visible and num_gpus=2.0/N per
    # actor, Ray packs ~N/2 actors onto each device, halving on-device contention.
    two_gpu = args.strategy == "stream_modelactor_2gpu"

    try:
        with st.stage("framework_init"):
            import torch  # noqa: F401
            if ann_mode:
                pass
            else:
                import ray
                from eqcctpro.tools import resolve_ray_temp_dir

                ray.init(
                    num_cpus=n_cpus,
                    num_gpus=((2 if two_gpu else 1) if gpu else 0),
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    logging_level="ERROR",
                    _temp_dir=resolve_ray_temp_dir(args.tmp_dir),
                )

        amodel = None
        with st.stage("model_load"):
            slip = args.strategy == "stream_modelactor_slipstream"
            if ann_mode:
                import seisbench.models as sbm
                import torch
                amodel = getattr(sbm, m["parent"]).from_pretrained(m["child"])
                amodel.eval()
                if gpu and torch.cuda.is_available():
                    amodel.to(torch.device("cuda:0"))
                actors = []
            elif slip:
                from eqcctpro.slipstream_actor import SlipstreamSeisBenchModelActor as ActorCls

                remote_kwargs = dict(
                    parent_model_name=m["parent"],
                    child_model_name=m["child"],
                    gpus_to_use=([0] if gpu else False),
                    use_gpu=gpu,
                    slipstream_dtype=args.dtype,
                    slipstream_compile=args.compile,
                    overlap_samples=int(args.overlap_samples),
                    lean_batch_size=int(args.slipstream_batch_size),
                )
            else:
                from eqcctpro.parallelization import SeisBenchModelActor as ActorCls

                remote_kwargs = dict(
                    parent_model_name=m["parent"],
                    child_model_name=m["child"],
                    gpus_to_use=([0] if gpu else False),
                    use_gpu=gpu,
                )
            if not ann_mode:
                actors = []
                # With 2 GPUs visible, requesting 2.0/N gpu per actor makes the
                # pool's total demand 2.0, which Ray packs as ~N/2 actors per device
                # (each actor still sees its assigned GPU as cuda:0).
                gpu_frac = (2.0 if two_gpu else 1.0) / n_actors
                for _ in range(n_actors):
                    if gpu:
                        actors.append(ActorCls.options(num_gpus=gpu_frac, num_cpus=0).remote(**remote_kwargs))
                    else:
                        actors.append(ActorCls.options(num_cpus=1).remote(**remote_kwargs))
                ray.get([a.ready.remote() for a in actors])

        from rapid.data import load_all_streams, select_stations

        stations = select_stations(net_dir, args.n_stations)
        if slip:
            cls_kw: Dict[str, Any] = dict(P_threshold=args.p_threshold, S_threshold=args.s_threshold)
        else:
            cls_kw = dict(
                P_threshold=args.p_threshold,
                S_threshold=args.s_threshold,
                Detection_threshold=args.detection_threshold,
                strict=False,
                flexible_horizontal_components=True,
                overlap=int(args.overlap_samples),
            )

        n_win_per = len(fairness.window_starts(trace_len, args.in_samples, args.overlap_samples))
        session_t0 = time.monotonic()

        for k in range(args.n_feeds):
            if args.feed_interval_s > 0:
                target = session_t0 + k * args.feed_interval_s
                now = time.monotonic()
                wait_s = max(0.0, target - now)
                late_s = max(0.0, now - target)
                if wait_s > 0:
                    time.sleep(wait_s)
            else:
                # Back-to-back latency mode: no pacing, no lateness concept.
                wait_s = 0.0
                late_s = 0.0

            t = time.perf_counter()
            streams = load_all_streams(net_dir, stations)
            wf_s = time.perf_counter() - t

            if ann_mode:
                # Warm batched annotate over the WHOLE merged network (the path a
                # batch-annotate deployment would call per window, kept warm).
                from obspy import Stream
                import torch
                merged = Stream()
                for sta, stq in streams:
                    merged += stq
                ann_kw = dict(batch_size=int(args.slipstream_batch_size),
                              overlap=int(args.overlap_samples), strict=False,
                              flexible_horizontal_components=True)
                t = time.perf_counter()
                _ = amodel.annotate(merged, **ann_kw)
                if gpu:
                    torch.cuda.synchronize()
                inf_s = time.perf_counter() - t
                pick_s = 0.0          # picks scored by the native annotate family; timing is the deliverable here
                feed_picks = {}
            else:
                t = time.perf_counter()
                refs = []
                metas = []
                for idx, (sta, stq) in enumerate(streams):
                    actor = actors[idx % n_actors]
                    refs.append(actor.classify.remote(stq, **cls_kw))
                    start0 = min(tr.stats.starttime for tr in stq) if len(stq) else None
                    metas.append((sta, start0))
                outs = ray.get(refs)
                inf_s = time.perf_counter() - t

                t = time.perf_counter()
                feed_picks = {}
                for (sta, start0), out in zip(metas, outs):
                    feed_picks[sta] = _picks_from_classify_output(out, start0)
                pick_s = time.perf_counter() - t

            st.add("waveform_access", wf_s)
            st.add("inference", inf_s)
            st.add("pick_generation", pick_s)
            last_picks = feed_picks

            feeds.append({
                "feed_index": k,
                "scheduled_offset_s": k * args.feed_interval_s,
                "wait_s": round(wait_s, 3),
                "late_s": round(late_s, 3),
                "waveform_access_s": round(wf_s, 6),
                "inference_s": round(inf_s, 6),
                "pick_generation_s": round(pick_s, 6),
                "feed_total_s": round(wf_s + inf_s + pick_s, 6),
                "n_windows": n_win_per * len(streams),
                "rss_mb_now": round(process_tree_rss_mb(sampler.process), 2),
                "peak_ram_mb_so_far": round(sampler.peak_mb, 2),
                **({
                    "vram_mb_now": round(vram_sampler.end_mb, 2),
                    "peak_vram_mb_so_far": round(vram_sampler.peak_mb, 2),
                } if vram_sampler is not None else {}),
            })
            print(f"  [repeat {args.repeat_index}] feed {k}: total={wf_s + inf_s + pick_s:.2f}s "
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
        # PSS trio: shared pages (Ray plasma, shared libs) counted once across
        # the tree -- the apples-to-apples metric vs single-process natives.
        "baseline_pss_mb": round(sampler.baseline_pss_mb, 2),
        "peak_pss_mb": round(sampler.peak_pss_mb, 2),
        "process_tree_pss_mb": round(sampler.end_pss_mb, 2),
        "pss_growth_mb": round(max(0.0, sampler.peak_pss_mb - sampler.baseline_pss_mb), 2),
    }
    if vram_sampler is not None:
        mem["baseline_vram_mb"] = round(vram_sampler.baseline_mb, 2)
        mem["peak_vram_mb"] = round(vram_sampler.peak_mb, 2)
        mem["process_tree_vram_mb"] = round(vram_sampler.end_mb, 2)
        mem["vram_growth_mb"] = round(max(0.0, vram_sampler.peak_mb - vram_sampler.baseline_mb), 2)

    if not ok:
        rec = {"repeat_index": args.repeat_index, "success": False, "error": err, "feeds": feeds, **mem, **resources}
        (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
        print(f"  [repeat {args.repeat_index}] FAILED: {err}", file=sys.stderr)
        return 1

    # Cold/warm latency aggregates: feed 0 carries first-call overhead (the
    # timed warmup of this family); feeds 1..N-1 are the warm steady state.
    import statistics

    latency: Dict[str, Any] = {}
    if feeds:
        latency["cold_feed_total_s"] = feeds[0]["feed_total_s"]
        latency["cold_feed_inference_s"] = feeds[0]["inference_s"]
    warm = feeds[1:]
    if warm:
        wt = [f["feed_total_s"] for f in warm]
        wi = [f["inference_s"] for f in warm]
        latency.update({
            "warm_feed_mean_s": round(statistics.mean(wt), 6),
            "warm_feed_min_s": round(min(wt), 6),
            "warm_feed_max_s": round(max(wt), 6),
            "warm_feed_std_s": round(statistics.stdev(wt), 6) if len(wt) > 1 else 0.0,
            "warm_inference_mean_s": round(statistics.mean(wi), 6),
        })

    n_win_total = sum(f["n_windows"] for f in feeds)
    rec = st.as_repeat(success=True, extra={
        "repeat_index": args.repeat_index,
        "n_feeds": args.n_feeds,
        "feed_interval_s": args.feed_interval_s,
        "session_wall_s": round(session_wall_s, 3),
        "idle_wait_s": round(sum(f["wait_s"] for f in feeds), 3),
        "n_windows": n_win_total,
        "n_actors": n_actors,
        "concurrency": n_actors,
        "feeds": feeds,
        **latency,
        **mem,
        **resources,
    })
    (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
    (rep_dir / f"picks_{args.repeat_index}.json").write_text(json.dumps(last_picks))
    print(f"  [repeat {args.repeat_index}] ok total={rec['total_s']:.2f}s (session wall {session_wall_s:.1f}s)")
    return 0


# ---------------------------------------------------------------------------
# Driver: spawn N independent session subprocesses + aggregate
# ---------------------------------------------------------------------------


def _physical_gpu_id(args) -> int:
    return int(getattr(args, "gpu_id", 0) or 0)


def _worker_env(args, gpu: bool) -> Dict[str, str]:
    env = dict(os.environ)
    if gpu and args.strategy == "stream_modelactor_2gpu":
        # Expose BOTH physical GPUs so Ray can spread the actor pool across them.
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
    elif gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(_physical_gpu_id(args))
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _worker_argv(args, repeat_index: int) -> List[str]:
    argv = [
        "--strategy", args.strategy, "--dataset", args.dataset, "--n-stations", str(args.n_stations),
        "--model", args.model, "--device", args.device, "--n-cpus", str(args.n_cpus),
        "--gpu-id", str(_physical_gpu_id(args)), "--core-list", args.core_list,
        "--concurrency", str(args.concurrency), "--dtype", args.dtype,
        "--repeats", str(args.repeats), "--in-samples", str(args.in_samples),
        "--overlap-samples", str(args.overlap_samples), "--net-suffix", args.net_suffix,
        "--slipstream-batch-size", str(args.slipstream_batch_size),
        "--n-feeds", str(args.n_feeds), "--feed-interval-s", str(args.feed_interval_s),
        "--p-threshold", str(args.p_threshold), "--s-threshold", str(args.s_threshold),
        "--detection-threshold", str(args.detection_threshold), "--tag", args.tag,
        "--net-root", str(args.net_root), "--results-root", str(args.results_root),
        "--tmp-dir", str(args.tmp_dir), "--repeat-index", str(repeat_index),
    ]
    if args.compile:
        argv.append("--compile")
    return argv


def run_driver(args) -> int:
    gpu = args.device == "gpu"
    net_dir = _net_dir(args.net_root, args.dataset, args.n_stations, args.net_suffix)
    manifest_path = net_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR missing manifest {manifest_path}", file=sys.stderr)
        return 2

    cores = [int(c) for c in str(args.core_list).split(",") if c.strip() != ""]
    n_cpus = len(cores) if cores else args.n_cpus
    conc = args.concurrency or max(1, min(n_cpus, args.n_stations))
    out_dir = _out_dir(args)
    rep_dir = out_dir / "repeats"
    result_path = out_dir / "result.json"

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

    for i in range(args.repeats):
        if args.resume and _ok(i):
            continue
        cmd = [sys.executable, str(Path(__file__).resolve())] + _worker_argv(args, i)
        subprocess.run(cmd, cwd=str(RAPID_ROOT), env=_worker_env(args, gpu))

    from rapid.benchmark.fairness import (
        MEMORY_KEYS,
        RESOURCE_KEYS,
        _agg,
        summarize_pick_quality,
    )

    LATENCY_KEYS = (
        "cold_feed_total_s", "cold_feed_inference_s",
        "warm_feed_mean_s", "warm_feed_min_s", "warm_feed_max_s",
        "warm_feed_std_s", "warm_inference_mean_s",
    )

    timing_repeats: List[Dict[str, Any]] = []
    memory_repeats: List[Dict[str, Any]] = []
    resource_repeats: List[Dict[str, Any]] = []
    pq_repeats: List[Dict[str, Any]] = []
    latency_repeats: List[Dict[str, Any]] = []
    last_picks: Dict[str, Dict[str, List[float]]] = {}
    for i in range(args.repeats):
        f = rep_dir / f"repeat_{i}.json"
        if not f.is_file():
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        timing_repeats.append(rec)
        if rec.get("success"):
            memory_repeats.append({k: rec[k] for k in MEMORY_KEYS if k in rec} | {"repeat_index": i})
            resource_repeats.append({k: rec[k] for k in RESOURCE_KEYS if k in rec} | {"repeat_index": i})
            latency_repeats.append({k: rec[k] for k in LATENCY_KEYS if k in rec} | {"repeat_index": i})
            pf = rep_dir / f"picks_{i}.json"
            if pf.is_file():
                try:
                    picks_i = json.loads(pf.read_text())
                except Exception:
                    picks_i = None
                if picks_i:
                    last_picks = picks_i
                    try:
                        _t0, sta_cat = load_manifest_catalog(manifest_path)
                        catalog = catalog_from_manifest_stations(sta_cat)
                        pq_i = compare_pick_sets(
                            catalog_by_station=catalog, detected_by_station=picks_i,
                            label=f"{args.strategy}_{args.tag}_r{i}", reference_label="catalog",
                        )
                        pq_i["repeat_index"] = i
                        pq_repeats.append(pq_i)
                    except Exception:
                        pass

    m = MODELS[args.model]
    trace_len = 3001 if args.net_suffix == "_w3001" else 6000
    n_win_feed = len(fairness.window_starts(trace_len, args.in_samples, args.overlap_samples)) * args.n_stations
    meta = dict(
        method=args.strategy, family="streaming", dataset=args.dataset.lower(),
        n_stations=args.n_stations, model=args.model, parent=m["parent"], child=m["child"],
        device=args.device, n_cpus=n_cpus, gpu_id=(_physical_gpu_id(args) if gpu else None),
        in_samples=args.in_samples, overlap_samples=args.overlap_samples,
        net_window=(args.net_suffix or "_w6000").lstrip("_"), window_samples=args.in_samples,
        dtype=args.dtype, compile=args.compile, concurrency=conc,
        batch_size=args.slipstream_batch_size,
        n_feeds=args.n_feeds, feed_interval_s=args.feed_interval_s,
        back_to_back=bool(args.feed_interval_s <= 0),
        n_windows=n_win_feed, n_windows_total=n_win_feed * args.n_feeds,
        repeats=args.repeats, tag=args.tag,
        pick_extractor=("rapid_threshold_crossing" if "slipstream" in args.strategy else "seisbench_classify"),
        p_threshold=args.p_threshold, s_threshold=args.s_threshold,
    )

    pq = summarize_pick_quality(pq_repeats) if pq_repeats else None
    result = build_result(meta=meta, timing_repeats=timing_repeats,
                          memory_repeats=memory_repeats, pick_quality=pq,
                          resource_repeats=resource_repeats)
    if latency_repeats:
        lat: Dict[str, Any] = {"repeats": latency_repeats}
        for key in LATENCY_KEYS:
            stats = _agg([float(r[key]) for r in latency_repeats if r.get(key) is not None])
            for k, v in stats.items():
                lat[f"{key}_{k}"] = v
        result["latency"] = lat
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {result_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--dataset", required=True, choices=["stead", "txed"])
    ap.add_argument("--n-stations", type=int, required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-cpus", type=int, default=20)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--core-list", default="")
    ap.add_argument("--concurrency", type=int, default=0, help="actor count; 0 = n_cpus")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--in-samples", type=int, default=6000)
    ap.add_argument("--overlap-samples", type=int, default=0)
    ap.add_argument("--net-suffix", default="", help="'' for 6000 net, '_w3001' for trimmed")
    ap.add_argument("--slipstream-batch-size", type=int, default=256)
    ap.add_argument("--n-feeds", type=int, default=4, help="feeds at t=0..(n-1)*interval")
    ap.add_argument("--feed-interval-s", type=float, default=60.0)
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--detection-threshold", type=float, default=0.3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--net-root", type=Path, default=EQCCTPRO_ROOT / "data" / "seisbench_networks")
    ap.add_argument("--results-root", type=Path, default=RAPID_ROOT / "results" / "fair_benchmark")
    # Short Ray temp dir on /home (see run_fair_orch_trial.py): the long vendored
    # path overflows Ray's 107-byte AF_UNIX socket limit and falls back to /tmp.
    ap.add_argument("--tmp-dir", type=Path, default=Path.home() / "rapid_ray")
    ap.add_argument("--repeat-index", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.repeat_index >= 0:
        return run_one_repeat(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
