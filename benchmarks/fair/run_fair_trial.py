#!/usr/bin/env python3
"""Run ONE native fair-benchmark trial and emit a unified schema-v2 result.json.

A "trial" is a single matrix point::

    method x dataset x n_stations x model x device x n_cpus x window-regime
          x dtype x compile x batch_size   (with N repeats inside)

Independent repeats (memory fairness)
-------------------------------------
Each of the N repeats runs in its OWN fresh subprocess (``--repeat-index``). A
fresh interpreter means torch/seisbench are re-imported, the model is re-loaded,
the CUDA context + allocator caches are gone, and the RSS baseline starts at the
bare-interpreter floor. So ``ram_growth_mb`` reflects the TRUE cost of that run,
not a shrinking delta caused by a model instance already resident from a prior
repeat. The driver process only orchestrates subprocesses and aggregates their
per-repeat JSON into the final ``result.json``.

Stricter memory columns
------------------------
``baseline_ram_mb`` is sampled at worker entry BEFORE any heavy import, so it is
a clean near-floor value identical in spirit across every method. ``peak_ram_mb``
is the process-tree high-water mark over the whole repeat (imports + model load +
inference + any pool workers). ``ram_growth_mb = peak - baseline``. Because every
repeat starts from the same fresh floor, these three columns are directly
comparable across methods/regimes/devices.

Native methods (SeisBench end-to-end vs RAPID Slipstream)
----------------------------------------------------------
* ``annotate``  -- SeisBench's built-in ``model.annotate()`` on the merged
  network Stream (its own preprocessing/windowing/stacking), batched via the
  ``batch_size`` argdict entry; picks thresholded from the probability traces.
* ``classify`` -- SeisBench's built-in ``model.classify()`` END-TO-END, one
  station at a time, STRICTLY SEQUENTIAL in a single process. The CPU marching
  scheme sets the torch thread budget only (no worker pool, no batching). This is
  the NAIVE per-station usage pattern (worst-case single-process classify) and is
  the exact serial twin of RAPID Model-Actor[classify], whose Ray tasks are also
  per-station.
* ``classify_batched`` -- SeisBench's built-in ``model.classify()`` on the FULL
  merged network in ONE call, so SeisBench batches windows ACROSS stations
  (classify == annotate + classify_aggregate). This is SeisBench's BEST
  single-process picking path -- the fair upper bound for the native picker.
  Picks still come from SeisBench's own ``classify_aggregate``. NB: SeisBench has
  no built-in cross-station/process parallelism (the deprecated ``parallelism``
  arg routes through synchronous ``annotate``); ``batch_size`` and torch intra-op
  threads are its only throughput levers, both of which we sweep.
* ``annotate_bf16`` / ``annotate_fp16`` -- SeisBench ``annotate()`` after casting
  weights to BF16/FP16; discrete picks via SeisBench ``classify_aggregate``.

Three-regime windowing (matches orchestration)
----------------------------------------------
* EQT / EQT-NC ("w6000"): single 6000-sample window on the 6000-sample network.
* PhaseNet / PhaseNetLight "w6000x2": 3001-sample windows, overlap 0, over the
  full 6000-sample trace -> 2 windows/station.
* PhaseNet / PhaseNetLight "w6000ov03": 3001-sample windows, 0.3 overlap (900
  samples), over the 6000-sample trace -> 3 windows/station.
* PhaseNet / PhaseNetLight "w3001": single 3001-sample window on the trimmed
  3001-sample network -> 1 window/station.

Native and orchestration read the SAME network dir per regime, so they feed
identical traces.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RAPID_ROOT = Path(__file__).resolve().parents[2]
if str(RAPID_ROOT) not in sys.path:
    sys.path.insert(0, str(RAPID_ROOT))

# Only light imports at module load (numpy via fairness/pick_quality). torch and
# seisbench are imported lazily INSIDE the worker so the memory baseline is clean.
from rapid.benchmark import fairness  # noqa: E402
from rapid.benchmark.fairness import (  # noqa: E402
    StageTimes,
    build_result,
    build_windowed_batch,
    pin_threads,
    windows_to_station_picks,
)
from rapid.benchmark.pick_quality import (  # noqa: E402
    catalog_from_manifest_stations,
    compare_pick_sets,
    load_manifest_catalog,
)

MODELS: Dict[str, Dict[str, str]] = {
    "PhaseNet": {"parent": "PhaseNet", "child": "original"},
    "PhaseNetLight": {"parent": "PhaseNetLight", "child": "stead"},
    "EQTransformer": {"parent": "EQTransformer", "child": "original"},
    "EQT-NC": {"parent": "EQTransformer", "child": "original_nonconservative"},
}

NATIVE_METHODS = ("annotate", "classify", "classify_batched", "annotate_bf16", "annotate_fp16")
# Legacy name kept only so old schedulers fail with a clear message.
_REMOVED_METHODS = ("slipstream",)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _self_rss_mb() -> float:
    import psutil

    try:
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def _net_dir(net_root: Path, dataset: str, n_stations: int, net_suffix: str) -> Path:
    return net_root / f"{dataset.lower()}_{n_stations}st{net_suffix}"


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


def _phase_indices(raw_model):
    from rapid.seisbench_precision_eval import phase_indices

    return phase_indices(raw_model)


# ---------------------------------------------------------------------------
# slipstream (RAPID lean PyTorch, single-process windowed forward)
# ---------------------------------------------------------------------------


def _run_batched_repeat(
    *,
    net_dir: Path,
    stations: List[str],
    parent: str,
    child: str,
    device: str,
    n_cpus: int,
    torch_threads: Optional[int] = None,
    in_samples: int,
    overlap_samples: int,
    dtype: str,
    compile_model: bool,
    batch_size: int,
    p_threshold: float,
    s_threshold: float,
    min_separation: int,
    gpu: bool,
    picks_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[float]]]]:
    from rapid.backends.lean_pytorch import LeanPyTorchBackend
    from rapid.data import load_all_streams

    st = StageTimes()
    with st.stage("framework_init"):
        import torch  # noqa: F401

        pin_threads(n_cpus, torch_threads=torch_threads)
        if gpu:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    with st.stage("model_load"):
        backend = LeanPyTorchBackend(parent, child, device=device, dtype=dtype, compile=compile_model)
        backend.load()

    try:
        with st.stage("waveform_access"):
            streams = load_all_streams(net_dir, stations)

        with st.stage("preprocess"):
            station_ids, windows, n_per, starts = build_windowed_batch(
                backend._raw_model, streams, in_samples, overlap_samples,
                component_order=backend.component_order,
            )

        n_windows = int(windows.shape[0])
        bs = max(1, min(int(batch_size), max(1, n_windows)))

        # Warmup: torch.compile / cuDNN autotune / lazy CUDA init, measured as
        # its own stage (counted in total) so every family pays it identically.
        if n_windows > 0:
            with st.stage("warmup"):
                _ = backend.infer_chunked(windows[: min(bs, n_windows)], bs)
                if gpu:
                    import torch
                    torch.cuda.synchronize()

        with st.stage("inference"):
            preds = backend.infer_chunked(windows, bs)
            if gpu:
                import torch
                torch.cuda.synchronize()

        with st.stage("pick_generation"):
            p_idx, s_idx = _phase_indices(backend._raw_model)
            picks = windows_to_station_picks(
                preds, station_ids, n_per, starts, p_idx=p_idx, s_idx=s_idx,
                p_threshold=p_threshold, s_threshold=s_threshold, min_separation=min_separation,
            )
            # Persist picks INSIDE the timed stage: orchestration's pick-write
            # timing includes its disk write, so the native stage must too.
            picks_path.write_text(json.dumps(picks))
    finally:
        backend.close()

    wps = (n_windows / len(station_ids)) if station_ids else 0.0
    repeat = st.as_repeat(success=True, extra={
        "n_windows": n_windows, "batch_size": bs, "windows_per_station": round(wps, 3),
    })
    return repeat, picks


# ---------------------------------------------------------------------------
# annotate (SeisBench built-in, end-to-end, batched via argdict batch_size)
# ---------------------------------------------------------------------------


def _picks_from_prob_traces(
    ann,
    orig_starts: Dict[str, Any],
    p_threshold: float,
    s_threshold: float,
    min_separation: int,
    sr: float = 100.0,
) -> Dict[str, Dict[str, List[float]]]:
    """Threshold P/S probability traces from ``model.annotate()`` into picks.

    Detected onsets are mapped back to trace-relative samples using the offset
    between the annotation trace start and the original trace start (SeisBench
    blinding can trim the edges).
    """
    from rapid.quality import extract_picks_simple

    out: Dict[str, Dict[str, List[float]]] = {}
    for tr in ann:
        ch = str(tr.stats.channel or "")
        if ch.endswith("_P"):
            phase, thr = "p", p_threshold
        elif ch.endswith("_S"):
            phase, thr = "s", s_threshold
        else:
            continue  # _N noise / _Detection traces
        sta = tr.stats.station
        start0 = orig_starts.get(sta)
        if start0 is None:
            continue
        offset = float(tr.stats.starttime - start0) * sr
        onsets = extract_picks_simple(tr.data, threshold=thr, min_separation=min_separation)
        d = out.setdefault(sta, {"p": [], "s": []})
        d[phase].extend(float(offset + o) for o in onsets)
    return out


def _run_annotate_repeat(
    *,
    net_dir: Path,
    stations: List[str],
    parent: str,
    child: str,
    device: str,
    n_cpus: int,
    torch_threads: Optional[int] = None,
    in_samples: int,
    overlap_samples: int,
    trace_len: int,
    batch_size: int,
    p_threshold: float,
    s_threshold: float,
    min_separation: int,
    gpu: bool,
    picks_path: Path,
    dtype: str = "fp32",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[float]]]]:
    """SeisBench ``model.annotate()`` end-to-end on the merged network Stream.

    When ``dtype`` is ``bf16`` or ``fp16``, weights are cast before annotate and
    discrete picks come from SeisBench ``classify_aggregate`` (Classify family).
    FP32 annotate keeps RAPID's rising-edge threshold extractor for continuity
    with the historical Annotate baseline.
    """
    from obspy import Stream
    from rapid.data import load_all_streams

    dtype = str(dtype).lower()
    use_classify_aggregate = dtype in ("bf16", "fp16")

    st = StageTimes()
    with st.stage("framework_init"):
        import torch

        pin_threads(n_cpus, torch_threads=torch_threads)

    with st.stage("model_load"):
        import seisbench.models as sbm

        if parent == "EQTransformer" and dtype == "fp16":
            raise ValueError(
                "EQTransformer cannot run in fp16 (padding sentinel overflows). Use bf16."
            )

        model = getattr(sbm, parent).from_pretrained(child)
        model.eval()
        if gpu and torch.cuda.is_available():
            model.to(torch.device(device))
        if use_classify_aggregate:
            torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
            model.to(torch_dtype)
            from rapid.api import _wrap_forward_cast

            model = _wrap_forward_cast(model, torch_dtype)

    with st.stage("waveform_access"):
        streams = load_all_streams(net_dir, stations)
        merged = Stream()
        orig_starts: Dict[str, Any] = {}
        for sta, s in streams:
            merged += s
            if len(s):
                orig_starts[sta] = min(tr.stats.starttime for tr in s)

    # Preprocess/windowing happen INSIDE annotate(). Stage probes wrap
    # SeisBench's annotate_stream_pre/annotate_batch_pre hooks, so the
    # preprocess stage below is the MEASURED busy time of those hooks and the
    # inference stage is the annotate wall minus that measured preprocess.
    from rapid.orchestration.support.timing_util import SeisBenchStageProbes

    probes = SeisBenchStageProbes(model)
    ann_kw = dict(
        batch_size=int(batch_size),
        overlap=int(overlap_samples),
        strict=False,
        flexible_horizontal_components=True,
    )

    # Warmup: lazy CUDA init / cuDNN autotune on one station, measured as its
    # own stage (counted in total) so every family pays it identically.
    if streams:
        with st.stage("warmup"):
            _ = model.annotate(streams[0][1], **ann_kw)
            if gpu:
                torch.cuda.synchronize()

    probes.reset()
    t0 = time.perf_counter()
    ann = model.annotate(merged, **ann_kw)
    if gpu:
        torch.cuda.synchronize()
    annotate_wall = time.perf_counter() - t0
    st.add("preprocess", probes.preprocess_s)
    st.add("inference", max(0.0, annotate_wall - probes.preprocess_s))

    with st.stage("pick_generation"):
        if use_classify_aggregate:
            from rapid.api import classify_from_annotations

            out = classify_from_annotations(
                model,
                ann,
                P_threshold=p_threshold,
                S_threshold=s_threshold,
            )
            sr = 100.0
            picks: Dict[str, Dict[str, List[float]]] = {
                sta: {"p": [], "s": []} for sta in stations
            }
            for pick in (getattr(out, "picks", None) or []):
                pt = (
                    getattr(pick, "peak_time", None)
                    or getattr(pick, "start_time", None)
                    or getattr(pick, "time", None)
                )
                ph = str(getattr(pick, "phase", "") or "").upper()
                # Trace station is on the pick when available.
                sta = str(getattr(pick, "trace_id", "") or "")
                # Pick.trace_id is often NET.STA.LOC.CHA — take station code.
                if "." in sta:
                    parts = sta.split(".")
                    sta = parts[1] if len(parts) > 1 else parts[0]
                if not sta or sta not in orig_starts or pt is None:
                    # Fall back: match against orig_starts by scanning? skip unknown
                    continue
                start0 = orig_starts[sta]
                samp = float(pt - start0) * sr
                if ph == "P":
                    picks.setdefault(sta, {"p": [], "s": []})["p"].append(samp)
                elif ph == "S":
                    picks.setdefault(sta, {"p": [], "s": []})["s"].append(samp)
        else:
            picks = _picks_from_prob_traces(
                ann, orig_starts, p_threshold, s_threshold, min_separation
            )
        # Persist picks INSIDE the timed stage (matches orchestration's
        # pick-write timing, which includes its disk write).
        picks_path.write_text(json.dumps(picks))

    n_per = len(fairness.window_starts(trace_len, in_samples, overlap_samples))
    repeat = st.as_repeat(success=True, extra={
        "n_windows": n_per * len(stations), "batch_size": int(batch_size),
        "windows_per_station": float(n_per), "seisbench_native": True,
        "preprocess_measured_via_hooks": True,
        "dtype": dtype,
        "pick_extractor": (
            "seisbench_classify_aggregate" if use_classify_aggregate else "rapid_threshold_crossing"
        ),
    })
    return repeat, picks


# ---------------------------------------------------------------------------
# classify (SeisBench built-in, end-to-end, strictly sequential)
# ---------------------------------------------------------------------------


def _run_classify_seq_repeat(
    *,
    net_dir: Path,
    stations: List[str],
    parent: str,
    child: str,
    device: str,
    n_cpus: int,
    torch_threads: Optional[int] = None,
    in_samples: int,
    overlap_samples: int,
    trace_len: int,
    p_threshold: float,
    s_threshold: float,
    min_separation: int,
    gpu: bool,
    picks_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[float]]]]:
    """SeisBench ``model.classify()`` end-to-end, one station at a time.

    Strictly sequential in ONE process: the CPU marching scheme only sets the
    torch thread budget. Mirrors the kwargs the orchestration classify paths
    pass (thresholds + overlap), so picks are produced by the same SeisBench
    pipeline in both families.
    """
    from rapid.data import load_station_stream

    st = StageTimes()
    with st.stage("framework_init"):
        import torch

        pin_threads(n_cpus, torch_threads=torch_threads)

    with st.stage("model_load"):
        import seisbench.models as sbm

        model = getattr(sbm, parent).from_pretrained(child)
        model.eval()
        if gpu and torch.cuda.is_available():
            model.to(torch.device(device))

    # Stage probes: classify() is monolithic; the wrapped SeisBench hooks give
    # MEASURED per-call preprocess + pick-aggregation segments, so the stage
    # columns are measured (inference = classify wall minus those segments).
    from rapid.orchestration.support.timing_util import SeisBenchStageProbes

    probes = SeisBenchStageProbes(model)
    cls_kw = dict(
        P_threshold=p_threshold,
        S_threshold=s_threshold,
        strict=False,
        flexible_horizontal_components=True,
        overlap=int(overlap_samples),
    )

    sr = 100.0
    picks_acc: Dict[str, Dict[str, List[float]]] = {}

    # Warmup: lazy CUDA init / first-call setup on one station, measured as
    # its own stage (counted in total) so every family pays it identically.
    if stations:
        with st.stage("warmup"):
            warm = load_station_stream(net_dir, stations[0])
            _ = model.classify(warm, **cls_kw)
            if gpu:
                torch.cuda.synchronize()

    for sta in stations:
        t = time.perf_counter()
        stq = load_station_stream(net_dir, sta)
        st.add("waveform_access", time.perf_counter() - t)

        probes.reset()
        t = time.perf_counter()
        out = model.classify(stq, **cls_kw)
        if gpu:
            torch.cuda.synchronize()
        cls_wall = time.perf_counter() - t
        st.add("preprocess", probes.preprocess_s)
        st.add("pick_generation", probes.pick_aggregate_s)
        st.add("inference", max(0.0, cls_wall - probes.preprocess_s - probes.pick_aggregate_s))

        t = time.perf_counter()
        start0 = min(tr.stats.starttime for tr in stq) if len(stq) else None
        p_list: List[float] = []
        s_list: List[float] = []
        for pick in (getattr(out, "picks", None) or []):
            pt = getattr(pick, "peak_time", None) or getattr(pick, "start_time", None) or getattr(pick, "time", None)
            ph = str(getattr(pick, "phase", "") or "").upper()
            if pt is None or start0 is None:
                continue
            samp = float(pt - start0) * sr
            if ph == "P":
                p_list.append(samp)
            elif ph == "S":
                s_list.append(samp)
        picks_acc[sta] = {"p": p_list, "s": s_list}
        st.add("pick_generation", time.perf_counter() - t)

    with st.stage("pick_generation"):
        # Persist picks INSIDE the timed stage (matches orchestration).
        picks_path.write_text(json.dumps(picks_acc))

    n_per = len(fairness.window_starts(trace_len, in_samples, overlap_samples))
    repeat = st.as_repeat(success=True, extra={
        "n_windows": n_per * len(stations), "concurrency": 1, "sequential": True,
        "windows_per_station": float(n_per), "seisbench_native": True,
        "preprocess_measured_via_hooks": True,
    })
    return repeat, picks_acc


# ---------------------------------------------------------------------------
# classify_batched (SeisBench built-in, full network in ONE call)
# ---------------------------------------------------------------------------


def _run_classify_batched_repeat(
    *,
    net_dir: Path,
    stations: List[str],
    parent: str,
    child: str,
    device: str,
    n_cpus: int,
    torch_threads: Optional[int] = None,
    in_samples: int,
    overlap_samples: int,
    trace_len: int,
    batch_size: int,
    p_threshold: float,
    s_threshold: float,
    min_separation: int,
    gpu: bool,
    picks_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[float]]]]:
    """SeisBench ``model.classify()`` on the FULL merged network in ONE call.

    Unlike the per-station ``classify`` baseline, the entire network Stream is
    passed to a single ``classify()`` so SeisBench batches windows ACROSS stations
    (classify == annotate + classify_aggregate). This is SeisBench's best
    single-process picking path and the fair upper bound for the native picker.
    Picks come from SeisBench's own ``classify_aggregate`` (same provenance as the
    per-station classify), mapped back to stations via each pick's ``trace_id``.
    """
    from obspy import Stream
    from rapid.data import load_all_streams

    st = StageTimes()
    with st.stage("framework_init"):
        import torch

        pin_threads(n_cpus, torch_threads=torch_threads)

    with st.stage("model_load"):
        import seisbench.models as sbm

        model = getattr(sbm, parent).from_pretrained(child)
        model.eval()
        if gpu and torch.cuda.is_available():
            model.to(torch.device(device))

    with st.stage("waveform_access"):
        streams = load_all_streams(net_dir, stations)
        merged = Stream()
        orig_starts: Dict[str, Any] = {}
        for sta, s in streams:
            merged += s
            if len(s):
                orig_starts[sta] = min(tr.stats.starttime for tr in s)

    from rapid.orchestration.support.timing_util import SeisBenchStageProbes

    probes = SeisBenchStageProbes(model)
    cls_kw = dict(
        P_threshold=p_threshold,
        S_threshold=s_threshold,
        strict=False,
        flexible_horizontal_components=True,
        overlap=int(overlap_samples),
        batch_size=int(batch_size),
    )

    # Warmup: lazy CUDA init / first-call setup on one station, measured as its
    # own stage (counted in total) so every family pays it identically.
    if streams:
        with st.stage("warmup"):
            _ = model.classify(streams[0][1], **cls_kw)
            if gpu:
                torch.cuda.synchronize()

    probes.reset()
    t0 = time.perf_counter()
    out = model.classify(merged, **cls_kw)
    if gpu:
        torch.cuda.synchronize()
    classify_wall = time.perf_counter() - t0
    st.add("preprocess", probes.preprocess_s)
    st.add("pick_generation", probes.pick_aggregate_s)
    st.add("inference", max(0.0, classify_wall - probes.preprocess_s - probes.pick_aggregate_s))

    sr = 100.0
    sta_set = set(stations)

    def _station_of(trace_id: str) -> Optional[str]:
        """Resolve a pick's station from its trace_id (NET.STA.LOC.CHA)."""
        if not trace_id:
            return None
        for tok in str(trace_id).split("."):
            if tok in sta_set:
                return tok
        for sta in sta_set:  # fallback: substring match
            if sta and sta in str(trace_id):
                return sta
        return None

    with st.stage("pick_generation"):
        picks_acc: Dict[str, Dict[str, List[float]]] = {sta: {"p": [], "s": []} for sta in stations}
        for pick in (getattr(out, "picks", None) or []):
            sta = _station_of(getattr(pick, "trace_id", "") or "")
            if sta is None:
                continue
            start0 = orig_starts.get(sta)
            pt = getattr(pick, "peak_time", None) or getattr(pick, "start_time", None) or getattr(pick, "time", None)
            ph = str(getattr(pick, "phase", "") or "").upper()
            if pt is None or start0 is None:
                continue
            samp = float(pt - start0) * sr
            if ph == "P":
                picks_acc[sta]["p"].append(samp)
            elif ph == "S":
                picks_acc[sta]["s"].append(samp)
        # Persist picks INSIDE the timed stage (matches the other native paths).
        picks_path.write_text(json.dumps(picks_acc))

    n_per = len(fairness.window_starts(trace_len, in_samples, overlap_samples))
    repeat = st.as_repeat(success=True, extra={
        "n_windows": n_per * len(stations), "batch_size": int(batch_size),
        "concurrency": 1, "sequential": True, "batched_across_stations": True,
        "windows_per_station": float(n_per), "seisbench_native": True,
        "preprocess_measured_via_hooks": True,
    })
    return repeat, picks_acc


# ---------------------------------------------------------------------------
# Worker: run exactly ONE repeat in this fresh process
# ---------------------------------------------------------------------------


def run_one_repeat(args) -> int:
    # Clean baseline BEFORE any heavy import / model load / CUDA init.
    baseline = _self_rss_mb()

    gpu = args.device == "gpu"
    device = "cuda:0" if gpu else "cpu"
    core_list = [int(c) for c in str(args.core_list).split(",") if c.strip() != ""] if args.core_list else None
    n_eff = _set_affinity(core_list)

    m = MODELS[args.model]
    parent, child = m["parent"], m["child"]
    net_dir = _net_dir(args.net_root, args.dataset, args.n_stations, args.net_suffix)

    from rapid.data import select_stations
    from rapid.orchestration.support.tools import ProcessTreeMemorySampler, process_tree_rss_mb

    stations = select_stations(net_dir, args.n_stations)
    trace_len = 3001 if args.net_suffix == "_w3001" else 6000

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
    picks: Dict[str, Dict[str, List[float]]] = {}
    try:
        if args.method == "classify":
            repeat, picks = _run_classify_seq_repeat(
                net_dir=net_dir, stations=stations, parent=parent, child=child,
                device=device, n_cpus=n_eff, torch_threads=args.torch_threads, in_samples=args.in_samples,
                overlap_samples=args.overlap_samples, trace_len=trace_len,
                p_threshold=args.p_threshold, s_threshold=args.s_threshold,
                min_separation=args.min_separation, gpu=gpu, picks_path=picks_path,
            )
        elif args.method == "annotate":
            repeat, picks = _run_annotate_repeat(
                net_dir=net_dir, stations=stations, parent=parent, child=child,
                device=device, n_cpus=n_eff, torch_threads=args.torch_threads, in_samples=args.in_samples,
                overlap_samples=args.overlap_samples, trace_len=trace_len,
                batch_size=args.batch_size, p_threshold=args.p_threshold,
                s_threshold=args.s_threshold, min_separation=args.min_separation, gpu=gpu,
                picks_path=picks_path, dtype="fp32",
            )
        elif args.method in ("annotate_bf16", "annotate_fp16"):
            dtype = "bf16" if args.method == "annotate_bf16" else "fp16"
            repeat, picks = _run_annotate_repeat(
                net_dir=net_dir, stations=stations, parent=parent, child=child,
                device=device, n_cpus=n_eff, torch_threads=args.torch_threads, in_samples=args.in_samples,
                overlap_samples=args.overlap_samples, trace_len=trace_len,
                batch_size=args.batch_size, p_threshold=args.p_threshold,
                s_threshold=args.s_threshold, min_separation=args.min_separation, gpu=gpu,
                picks_path=picks_path, dtype=dtype,
            )
        elif args.method == "classify_batched":
            repeat, picks = _run_classify_batched_repeat(
                net_dir=net_dir, stations=stations, parent=parent, child=child,
                device=device, n_cpus=n_eff, torch_threads=args.torch_threads, in_samples=args.in_samples,
                overlap_samples=args.overlap_samples, trace_len=trace_len,
                batch_size=args.batch_size, p_threshold=args.p_threshold,
                s_threshold=args.s_threshold, min_separation=args.min_separation, gpu=gpu,
                picks_path=picks_path,
            )
        else:
            raise ValueError(
                f"Unknown method {args.method!r}. slipstream was removed; "
                "use annotate_bf16 / annotate_fp16."
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
        # PSS trio: shared pages counted once across the tree -- the
        # apples-to-apples cross-family memory metric (RSS kept for history).
        "baseline_pss_mb": round(sampler.baseline_pss_mb, 2),
        "peak_pss_mb": round(sampler.peak_pss_mb, 2),
        "process_tree_pss_mb": round(sampler.end_pss_mb, 2),
        "pss_growth_mb": round(max(0.0, sampler.peak_pss_mb - sampler.baseline_pss_mb), 2),
    }
    if vram_sampler is not None:
        # Process-tree VRAM trio (PID-isolated), mirroring the RAM trio. Captures
        # classify pool workers / any GPU subprocess.
        mem["baseline_vram_mb"] = round(vram_sampler.baseline_mb, 2)
        mem["peak_vram_mb"] = round(vram_sampler.peak_mb, 2)
        mem["process_tree_vram_mb"] = round(vram_sampler.end_mb, 2)
        mem["vram_growth_mb"] = round(max(0.0, vram_sampler.peak_mb - vram_sampler.baseline_mb), 2)

    if not ok:
        rec = {"repeat_index": args.repeat_index, "success": False, "error": err, **mem, **resources}
    else:
        rec = {**repeat, "repeat_index": args.repeat_index, **mem, **resources}
    (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
    print(f"  [repeat {args.repeat_index}] {'ok total=%.2fs' % rec.get('total_s', 0.0) if ok else 'FAILED: ' + err}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Driver: spawn N independent repeat subprocesses + aggregate
# ---------------------------------------------------------------------------


def _out_dir(args) -> Path:
    return (
        args.results_root / args.method / args.dataset.lower()
        / f"{args.n_stations}st" / args.model / args.tag
    )


def _pick_quality(manifest_path: Path, picks: Dict[str, Dict[str, List[float]]], label: str) -> Dict[str, Any]:
    _t0, stations = load_manifest_catalog(manifest_path)
    catalog = catalog_from_manifest_stations(stations)
    return compare_pick_sets(
        catalog_by_station=catalog, detected_by_station=picks,
        label=label, reference_label="catalog",
    )


def run_driver(args) -> int:
    gpu = args.device == "gpu"
    out_dir = _out_dir(args)
    rep_dir = out_dir / "repeats"
    result_path = out_dir / "result.json"
    net_dir = _net_dir(args.net_root, args.dataset, args.n_stations, args.net_suffix)
    manifest_path = net_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: missing manifest {manifest_path}", file=sys.stderr)
        return 2

    # Resume: keep completed per-repeat JSONs.
    def _completed() -> int:
        if not rep_dir.is_dir():
            return 0
        n = 0
        for i in range(args.repeats):
            f = rep_dir / f"repeat_{i}.json"
            if f.is_file():
                try:
                    if json.loads(f.read_text()).get("success"):
                        n += 1
                except Exception:
                    pass
        return n

    if args.resume and result_path.is_file() and _completed() >= args.repeats:
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
        cmd = [sys.executable, str(Path(__file__).resolve())] + _worker_argv(args, i)
        subprocess.run(cmd, cwd=str(RAPID_ROOT), env=_worker_env(args, gpu))

    # Aggregate per-repeat JSONs.
    from rapid.benchmark.fairness import MEMORY_KEYS, RESOURCE_KEYS, summarize_pick_quality

    timing_repeats: List[Dict[str, Any]] = []
    memory_repeats: List[Dict[str, Any]] = []
    resource_repeats: List[Dict[str, Any]] = []
    pq_repeats: List[Dict[str, Any]] = []
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
            pf = rep_dir / f"picks_{i}.json"
            if pf.is_file():
                try:
                    picks_i = json.loads(pf.read_text())
                except Exception:
                    picks_i = None
                if picks_i:
                    last_picks = picks_i
                    # Score EVERY successful repeat so quality variance across
                    # repeats (fp16/bf16/compile nondeterminism) is captured.
                    try:
                        pq_i = _pick_quality(manifest_path, picks_i, label=f"{args.method}_{args.tag}_r{i}")
                        pq_i["repeat_index"] = i
                        pq_repeats.append(pq_i)
                    except Exception:
                        pass

    trace_len = 3001 if args.net_suffix == "_w3001" else 6000
    n_windows = len(fairness.window_starts(trace_len, args.in_samples, args.overlap_samples)) * args.n_stations
    meta = dict(
        method=args.method, family="native", dataset=args.dataset.lower(),
        n_stations=args.n_stations, model=args.model, parent=MODELS[args.model]["parent"],
        child=MODELS[args.model]["child"], device=args.device, n_cpus=args.n_cpus,
        torch_threads=(args.torch_threads if args.torch_threads is not None else args.n_cpus),
        gpu_id=(args.gpu_id if gpu else None), in_samples=args.in_samples,
        overlap_samples=args.overlap_samples, net_window=(args.net_suffix or "_w6000").lstrip("_"),
        window_samples=args.in_samples, dtype=args.dtype, compile=args.compile,
        batch_size=args.batch_size, n_windows=n_windows, n_stations_windows=args.n_stations,
        repeats=args.repeats, tag=args.tag,
        # Pick provenance: classify* and annotate_bf16/fp16 use SeisBench
        # classify_aggregate; FP32 annotate uses RAPID's rising-edge extractor.
        pick_extractor=(
            "seisbench_classify"
            if args.method in ("classify", "classify_batched")
            else (
                "seisbench_classify_aggregate"
                if args.method in ("annotate_bf16", "annotate_fp16")
                else "rapid_threshold_crossing"
            )
        ),
        min_separation=args.min_separation,
        p_threshold=args.p_threshold, s_threshold=args.s_threshold,
    )
    pq = summarize_pick_quality(pq_repeats) if pq_repeats else None
    if last_picks:
        from rapid.benchmark.pick_export import write_picks_json

        write_picks_json(out_dir / "picks.json", last_picks)

    result = build_result(meta=meta, timing_repeats=timing_repeats, memory_repeats=memory_repeats,
                          pick_quality=pq, resource_repeats=resource_repeats)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {result_path}")
    return 0


def _worker_argv(args, repeat_index: int) -> List[str]:
    argv = [
        "--method", args.method, "--dataset", args.dataset, "--n-stations", str(args.n_stations),
        "--model", args.model, "--device", args.device, "--n-cpus", str(args.n_cpus),
        "--gpu-id", str(args.gpu_id), "--in-samples", str(args.in_samples),
        "--overlap-samples", str(args.overlap_samples), "--net-suffix", args.net_suffix,
        "--dtype", args.dtype, "--batch-size", str(args.batch_size), "--repeats", str(args.repeats),
        "--p-threshold", str(args.p_threshold), "--s-threshold", str(args.s_threshold),
        "--min-separation", str(args.min_separation), "--tag", args.tag,
        "--net-root", str(args.net_root), "--results-root", str(args.results_root),
        "--core-list", args.core_list, "--repeat-index", str(repeat_index),
    ]
    if args.torch_threads is not None:
        argv += ["--torch-threads", str(args.torch_threads)]   # <-- was dropped: worker ran at n_cpus threads
    if args.compile:
        argv.append("--compile")
    return argv


def _physical_gpu_id(args) -> int:
    """Physical CUDA device index (scheduler dual-GPU slots use 0 and 1)."""
    return int(getattr(args, "gpu_id", 0) or 0)


def _worker_env(args, gpu: bool) -> Dict[str, str]:
    env = dict(os.environ)
    if gpu:
        # Pin to the physical GPU assigned by the scheduler. PyTorch still uses cuda:0
        # inside the worker because only one device is visible.
        env["CUDA_VISIBLE_DEVICES"] = str(_physical_gpu_id(args))
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    # CRITICAL: set the compute-thread env in the PARENT, before the worker imports
    # torch. torch fixes its intra-op pool size from OMP_NUM_THREADS at import; a
    # later torch.set_num_threads() in pin_threads does NOT shrink an already-built
    # pool, so without this the thread sweep silently ran everything at the default
    # (~physical-core count). torch_threads==0 => leave unset (true SeisBench/torch
    # out-of-the-box default); None => the core budget; else the swept value.
    tt = getattr(args, "torch_threads", None)
    THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    if tt == 0:
        for k in THREAD_KEYS:
            env.pop(k, None)
    else:
        val = str(tt if tt is not None else args.n_cpus)
        for k in THREAD_KEYS:
            env[k] = val
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=NATIVE_METHODS)
    ap.add_argument("--dataset", required=True, choices=["stead", "txed"])
    ap.add_argument("--n-stations", type=int, required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-cpus", type=int, default=20)
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="Override torch intra/inter-op thread count (decoupled from the core "
                         "budget) for the thread-sensitivity sweep. Default None = use n_cpus.")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--in-samples", type=int, default=6000)
    ap.add_argument("--overlap-samples", type=int, default=0)
    ap.add_argument("--net-suffix", default="", help="'' for 6000 network, '_w3001' for trimmed")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--min-separation", type=int, default=50)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--net-root", type=Path, default=RAPID_ROOT / "data" / "seisbench_networks")
    ap.add_argument("--results-root", type=Path, default=RAPID_ROOT / "results" / "fair_benchmark")
    ap.add_argument("--core-list", default="")
    ap.add_argument("--repeat-index", type=int, default=-1,
                    help="If >=0, run exactly this one repeat (worker mode). Else driver mode.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.repeat_index >= 0:
        return run_one_repeat(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
