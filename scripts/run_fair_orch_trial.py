#!/usr/bin/env python3
"""Run ONE orchestration fair-benchmark trial -> unified schema-v2 result.json.

Strategies (eqcctpro / Ray): ``ripper`` (Ripper), ``modelactor`` (Model-Actor),
``modelactor_slipstream`` (Model-Actor + Slipstream FP16/BF16/compile).

Same fairness contract as the native runner (``run_fair_trial.py``)
==================================================================
* Same network per regime, so orchestration feeds BYTE-IDENTICAL windows:
  - EQT / EQT-NC: 6000-sample net, in_samples 6000 -> 1 window/station.
  - PhaseNet/PhaseNetLight regime B ("w3001"): trimmed 3001-sample net,
    in_samples 3001 -> 1 window/station (``n_windows == n_stations``).
  - PhaseNet/PhaseNetLight regime A ("w6000ov03"): 6000-sample net, 3001-sample
    windows at overlap 0.3 (900 samples). We force this overlap in BOTH paths:
    the slipstream actor via ``slipstream_overlap_samples`` and the
    Model-Actor/Ripper SeisBench ``classify`` via ``seisbench_overlap_samples``,
    so window stepping matches the native runner.

Independent repeats (memory fairness)
-------------------------------------
Each repeat runs in its OWN fresh subprocess (``--repeat-index``): Ray, torch and
TF are imported, used, then torn down with the process. ``baseline_ram_mb`` is
sampled at worker entry BEFORE any heavy import (clean near-floor), ``peak_ram_mb``
is the process-tree high-water (captures Ray workers), and
``ram_growth_mb = peak - baseline`` is the true per-run cost -- directly
comparable with the native columns.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

RAPID_ROOT = Path(__file__).resolve().parents[1]
EQCCTPRO_ROOT = RAPID_ROOT  # eqcctpro package is vendored inside RAPID
for p in (str(RAPID_ROOT), str(EQCCTPRO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rapid.benchmark.fairness import STAGES, build_result, window_starts  # noqa: E402

STRATEGIES = ("ripper", "ripper_slipstream", "modelactor", "modelactor_slipstream")
MODELS = {
    "PhaseNet": {"parent": "PhaseNet", "child": "original"},
    "PhaseNetLight": {"parent": "PhaseNetLight", "child": "stead"},
    "EQTransformer": {"parent": "EQTransformer", "child": "original"},
    "EQT-NC": {"parent": "EQTransformer", "child": "original_nonconservative"},
}


def _net_dir(net_root: Path, dataset: str, n_stations: int, net_suffix: str) -> Path:
    return net_root / f"{dataset.lower()}_{n_stations}st{net_suffix}"


def _trace_len(net_suffix: str) -> int:
    return 3001 if net_suffix == "_w3001" else 6000


def _expected_n_windows(net_suffix: str, n_stations: int, in_samples: int, overlap_samples: int) -> int:
    starts = window_starts(_trace_len(net_suffix), in_samples, overlap_samples)
    return len(starts) * n_stations


def _self_rss_mb() -> float:
    import psutil

    try:
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def _strategy_flags(strategy: str, dtype: str, compile_model: bool) -> Dict[str, Any]:
    if strategy == "ripper":
        return dict(ripper=True, slipstream_inference=False, slipstream_dtype="fp32", slipstream_compile=False)
    if strategy == "ripper_slipstream":
        return dict(ripper=True, slipstream_inference=True, slipstream_dtype=dtype, slipstream_compile=compile_model)
    if strategy == "modelactor":
        return dict(ripper=False, slipstream_inference=False, slipstream_dtype="fp32", slipstream_compile=False)
    if strategy == "modelactor_slipstream":
        return dict(ripper=False, slipstream_inference=True, slipstream_dtype=dtype, slipstream_compile=compile_model)
    raise ValueError(strategy)


def _f(row: Dict[str, Any], key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _trial_to_stages(rec: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Map an eqcctpro trial record onto unified stages summing to total_trial_time_s.

    Everything is built from MEASURED quantities:

    * wall-clock segments measured at the driver -- setup (``framework_init``),
      actor-pool creation (``model_load``), pool warmup (``warmup``), and the
      station-processing loop wall (``picker``);
    * per-stage busy-second SUMS measured inside every station task (mSEED
      read, model-input preprocess, forward, pick extraction + write -- with
      in-actor segments measured by probes inside classify()/the lean path).

    Stages inside the pipelined picker loop overlap in wall time, so per-stage
    wall clock is ill-defined there; the unified columns scale the picker wall
    by the measured busy-time shares (the raw sums are returned alongside in
    ``measured`` and stored on the repeat record). Ripper mode loads/warms the
    model inside each task, so those stages come from the task sums too.
    """
    total = _f(rec, "total_trial_time_s")
    picker = min(_f(rec, "picker_wall_s"), total)
    actor_creation = _f(rec, "actor_creation_s")
    warmup_wall = _f(rec, "warmup_wall_s")

    busy = {
        "model_load": _f(rec, "sum_model_load_s"),
        "warmup": _f(rec, "sum_warmup_s"),
        "waveform_access": _f(rec, "sum_waveform_load_s"),
        "preprocess": _f(rec, "sum_preprocess_s"),
        "inference": _f(rec, "sum_inference_s"),
        "pick_generation": _f(rec, "sum_pick_write_s"),
    }

    stages = {s: 0.0 for s in STAGES}
    stages["framework_init"] = max(0.0, total - picker - actor_creation - warmup_wall)
    stages["model_load"] = actor_creation
    stages["warmup"] = warmup_wall

    bsum = sum(busy.values())
    if bsum > 0:
        for k, w in busy.items():
            stages[k] += picker * (w / bsum)
    else:
        stages["inference"] += picker

    measured = {
        "picker_wall_s": round(picker, 6),
        "actor_creation_wall_s": round(actor_creation, 6),
        "warmup_wall_s": round(warmup_wall, 6),
        "stage_busy_sums_s": {k: round(v, 6) for k, v in busy.items()},
        "stage_busy_total_s": round(bsum, 6),
        "n_tasks": int(_f(rec, "successful_inference_tasks")),
        "pipelined_stage_normalization": "picker_wall x measured_busy_share" if bsum > 0 else "picker_wall_as_inference",
    }
    return stages, measured


def _inference_succeeded(rec: Optional[Dict[str, Any]]) -> tuple[bool, str]:
    """True when at least one station task completed inference."""
    if rec is None:
        return False, "no trial record (trial_results.json)"
    success_tasks = int(_f(rec, "successful_inference_tasks"))
    total_tasks = int(_f(rec, "total_station_tasks"))
    if success_tasks > 0:
        return True, ""
    if total_tasks > 0:
        return False, "zero successful inference tasks"
    if _f(rec, "avg_inference_s") > 0:
        return True, ""
    return False, "zero successful inference tasks (no station tasks recorded)"


def _read_last_trial_record(json_path: Path) -> Optional[Dict[str, Any]]:
    if not json_path.is_file():
        return None
    try:
        trials = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None
    return trials[-1] if isinstance(trials, list) and trials else None


def _out_dir(args) -> Path:
    return (
        args.results_root / "orchestration" / args.strategy / args.dataset.lower()
        / f"{args.n_stations}st" / args.model / args.tag
    )


# ---------------------------------------------------------------------------
# Worker: one EvaluateSystem point in a fresh process
# ---------------------------------------------------------------------------


def run_one_repeat(args) -> int:
    baseline = _self_rss_mb()  # clean floor BEFORE eqcctpro/ray/torch import

    m = MODELS[args.model]
    net_dir = _net_dir(args.net_root, args.dataset, args.n_stations, args.net_suffix)
    manifest_path = net_dir / "manifest.json"
    meta = json.loads(manifest_path.read_text())["meta"]

    gpu = args.device == "gpu"
    cores = [int(c) for c in str(args.core_list).split(",") if c.strip() != ""]
    if cores:
        try:
            os.sched_setaffinity(0, set(cores))
        except (OSError, AttributeError):
            pass
    n_cpus = len(cores) if cores else args.n_cpus
    conc = args.concurrency or max(1, min(n_cpus, args.n_stations))

    out_dir = _out_dir(args)
    rep_dir = out_dir / "repeats"
    rep_dir.mkdir(parents=True, exist_ok=True)
    work_dir = rep_dir / f"work_{args.repeat_index}"
    pick_out = work_dir / "output"
    work_dir.mkdir(parents=True, exist_ok=True)
    # eqcctpro tracks orchestration results in trial_results.json (snake_case
    # fields); the legacy CSV in the same dir is internal plumbing and ignored.
    csv_name = "gpu_test_results.csv" if gpu else "cpu_test_results.csv"
    csv_path = work_dir / csv_name
    trial_json_path = work_dir / "trial_results.json"

    from obspy import UTCDateTime
    from eqcctpro.tools import materialize_input_into_timechunk_layout

    t0 = UTCDateTime(meta["start_time"])
    try:
        materialize_input_into_timechunk_layout(str(net_dir), [[t0, UTCDateTime(meta["end_time"])]], logger=None)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] materialize: {exc}")

    from eqcctpro import EvaluateSystem
    from eqcctpro.tools import ProcessTreeMemorySampler, process_tree_rss_mb
    import eqcctpro.functionality as _func

    # Keep picks for scoring (EvaluateSystem would otherwise delete *_outputs).
    _func.remove_output_subdirs = lambda *a, **k: None

    flags = _strategy_flags(args.strategy, args.dtype, args.compile)
    physical_gpu = int(getattr(args, "gpu_id", 0) or 0)
    # Use the PHYSICAL GPU the scheduler assigned (via --gpu-id). eqcctpro sets
    # CUDA_VISIBLE_DEVICES from selected_gpus, so passing [0] unconditionally
    # forced EVERY orchestration GPU trial onto physical GPU 0 (GPU1 idle, and
    # two concurrent trials sharing GPU0). Passing the assigned id sends each
    # trial to its own GPU, so the two scheduler GPU slots use both devices.
    gpus = [physical_gpu] if gpu else None
    # Force the per-window overlap so orchestration windows == native windows.
    # For single-window regimes (trace == in_samples) overlap is a no-op (1 window).
    sb_overlap = int(args.overlap_samples)

    sampler = ProcessTreeMemorySampler(interval_s=0.5)
    sampler.start()
    vram_sampler = None
    if gpu:
        from rapid.benchmark.fairness import GpuVramSampler

        vram_sampler = GpuVramSampler(
            process=sampler.process, gpu_index=physical_gpu, interval_s=0.1
        )
        vram_sampler.start()
    from rapid.benchmark.fairness import ResourceUsageSampler

    res_sampler = ResourceUsageSampler(
        process=sampler.process,
        gpu_index=(physical_gpu if gpu else None),
        n_cores=n_cpus,
        interval_s=0.25,
    ).start()
    ok = True
    err = ""

    def _evaluate_once() -> None:
        ev = EvaluateSystem(
            eval_mode="gpu" if gpu else "cpu",
            input_dir=str(net_dir),
            output_dir=str(pick_out),
            log_filepath=str(work_dir / "eqcctpro.log"),
            csv_dir=str(work_dir),
            model_type="seisbench",
            seisbench_parent_model=m["parent"],
            seisbench_child_model=m["child"],
            selected_gpus=gpus,
            start_time=meta["start_time"],
            end_time=meta["end_time"],
            timechunk_dt=meta["timechunk_dt"],
            waveform_overlap=0,
            tmp_dir=str(args.tmp_dir),
            Detection_threshold=args.detection_threshold,
            P_threshold=args.p_threshold,
            S_threshold=args.s_threshold,
            slipstream_overlap_samples=sb_overlap,
            slipstream_batch_size=args.slipstream_batch_size,
            # When --concurrency is explicitly set (oversubscription sweep),
            # let in-flight slipstream CPU tasks exceed the core budget; the
            # RAM/VRAM caps remain the only limit.
            slipstream_cap_tasks_to_cpus=(args.concurrency == 0),
            seisbench_overlap_samples=sb_overlap,
            pick_output_format="ascii",
            ascii_station_pick_format="csv",
            overwrite=True,
            exact_resume_match=False,
            min_gpu_amount=1,
            cpu_id_list=cores if cores else list(range(n_cpus)),
            min_cpu_amount=n_cpus,
            cpu_test_step_size=max(1, n_cpus),
            stations2use=args.n_stations,
            starting_amount_of_stations=args.n_stations,
            station_list_step_size=args.n_stations,
            min_conc_stations=conc,
            conc_station_tasks_step_size=args.n_stations,
            conc_station_tasks_max_only=False,
            concurrency_on_max_actors=False,
            **flags,
        )
        ev.evaluate()

    try:
        _evaluate_once()
        rec_probe = _read_last_trial_record(trial_json_path)
        inf_ok, inf_err = _inference_succeeded(rec_probe)
        if not inf_ok:
            print(
                f"  [repeat {args.repeat_index}] {inf_err}; restarting evaluate once",
                file=sys.stderr,
            )
            import shutil

            try:
                import ray

                ray.shutdown()
            except Exception:
                pass
            for stale in (trial_json_path, csv_path):
                if stale.is_file():
                    stale.unlink()
            shutil.rmtree(pick_out, ignore_errors=True)
            pick_out.mkdir(parents=True, exist_ok=True)
            _evaluate_once()
    except Exception as exc:  # noqa: BLE001
        ok = False
        err = str(exc)
        import traceback

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
        # PSS trio: shared pages (Ray plasma, shared libs) counted once across
        # the tree -- the apples-to-apples metric vs single-process natives.
        "baseline_pss_mb": round(sampler.baseline_pss_mb, 2),
        "peak_pss_mb": round(sampler.peak_pss_mb, 2),
        "process_tree_pss_mb": round(sampler.end_pss_mb, 2),
        "pss_growth_mb": round(max(0.0, sampler.peak_pss_mb - sampler.baseline_pss_mb), 2),
    }
    trial = _read_last_trial_record(trial_json_path) if ok else None
    # Process-tree VRAM trio (PID-isolated) -- the SAME measurement the native
    # runner uses, so VRAM is directly comparable. Ray actors are children of this
    # worker, so their on-device VRAM is captured.
    if vram_sampler is not None:
        mem["baseline_vram_mb"] = round(vram_sampler.baseline_mb, 2)
        mem["peak_vram_mb"] = round(vram_sampler.peak_mb, 2)
        mem["process_tree_vram_mb"] = round(vram_sampler.end_mb, 2)
        mem["vram_growth_mb"] = round(max(0.0, vram_sampler.peak_mb - vram_sampler.baseline_mb), 2)
    if trial is None:
        msg = err or "no trial record (trial_results.json)"
        rec = {"repeat_index": args.repeat_index, "success": False, "error": msg, **mem, **resources}
        (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
        print(f"  [repeat {args.repeat_index}] FAILED: {msg}", file=sys.stderr)
        return 1

    inf_ok, inf_err = _inference_succeeded(trial)
    if not inf_ok:
        rec = {
            "repeat_index": args.repeat_index,
            "success": False,
            "error": inf_err,
            "successful_inference_tasks": int(_f(trial, "successful_inference_tasks")),
            "total_station_tasks": int(_f(trial, "total_station_tasks")),
            **mem,
            **resources,
        }
        (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))
        print(f"  [repeat {args.repeat_index}] FAILED: {inf_err}", file=sys.stderr)
        return 1

    stages, measured = _trial_to_stages(trial)
    rec = {f"{s}_s": round(stages[s], 6) for s in STAGES}
    rec["total_s"] = round(sum(stages.values()), 6)
    rec["success"] = True
    rec["repeat_index"] = args.repeat_index
    rec["n_windows"] = _expected_n_windows(args.net_suffix, args.n_stations, args.in_samples, args.overlap_samples)
    rec["concurrency"] = conc
    rec["n_modelactors"] = _f(trial, "n_modelactors")
    # Orchestration metadata straight from the eqcctpro trial record.
    for key in (
        "orchestration_strategy",
        "concurrent_tasks",
        "actual_ripper_concurrent_tasks",
        "batch_size",
        "successful_inference_tasks",
        "total_station_tasks",
    ):
        if trial.get(key) is not None:
            rec[key] = trial[key]
    rec.update(measured)
    rec.update(mem)
    rec.update(resources)
    (rep_dir / f"repeat_{args.repeat_index}.json").write_text(json.dumps(rec, default=str))

    # Score picks vs catalog into a per-repeat file.
    try:
        from scripts.compare_orchestrated_picks import compare_network_picks

        pq = compare_network_picks(
            manifest_path=manifest_path, picks_dir=pick_out, out_json=None,
            label=f"{args.strategy}_{args.tag}", p_threshold=args.p_threshold, s_threshold=args.s_threshold,
        )
        (rep_dir / f"pq_{args.repeat_index}.json").write_text(json.dumps(pq, default=str))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] pick scoring: {exc}")
    print(f"  [repeat {args.repeat_index}] ok total={rec['total_s']:.2f}s")
    return 0


# ---------------------------------------------------------------------------
# Driver: spawn N independent repeat subprocesses + aggregate
# ---------------------------------------------------------------------------


def _vram_capped_redundant(args, conc: int) -> Optional[Dict[str, Any]]:
    """Detect whether this oversub trial duplicates an already-capped sibling.

    The achieved actor pool is ``min(requested, memory_cap)`` and the memory cap
    (VRAM on GPU, RAM on CPU) is fixed for a given (model, device, precision) and
    does NOT depend on the requested concurrency or the host-core budget. So once
    a sibling at the SAME host-core budget and precision has been VRAM/RAM-capped
    to ``M`` actors (achieved ``M`` < its requested), every higher request also
    yields exactly ``M`` actors -- an identical pool on identical hardware, hence
    an identical trial. We skip those and point at the representative sibling.

    Returns the skip-info dict (sibling tag + capped actor count) or None. Only
    the higher-request trial skips; the smallest request that first hits the cap
    still runs and establishes ``M``.
    """
    out_dir = _out_dir(args)
    model_dir = out_dir.parent  # .../oversub/orchestration/<strategy>/<ds>/<Nst>/<model>/
    if not model_dir.is_dir():
        return None
    for sib in model_dir.glob("*/result.json"):
        if sib.parent == out_dir:
            continue
        try:
            data = json.loads(sib.read_text())
        except Exception:
            continue
        if data.get("skipped"):
            continue
        meta = data.get("meta", {})
        # Same experimental condition modulo the multiplier: same host-core
        # budget, device, and precision. (Different n_cpus = different host
        # preprocessing parallelism = a genuinely different condition.)
        if meta.get("n_cpus") != args.n_cpus or meta.get("device") != args.device:
            continue
        if (meta.get("dtype") or "fp32") != args.dtype:
            continue
        sib_req = meta.get("concurrency")
        if not sib_req:
            continue
        reps = [r for r in data.get("timing", {}).get("repeats", []) if r.get("success")]
        achieved = max((int(r.get("n_modelactors") or 0) for r in reps), default=0)
        if achieved <= 0:
            continue
        capped = achieved < sib_req            # the sibling itself hit the memory cap
        if capped and conc >= achieved and conc > sib_req:
            return {"redundant_with": meta.get("tag"), "capped_actors": achieved,
                    "sibling_requested": sib_req}
    return None


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

    if args.resume and result_path.is_file():
        try:
            prior = json.loads(result_path.read_text())
        except Exception:
            prior = {}
        if prior.get("skipped") or all(_ok(i) for i in range(args.repeats)):
            print(f"[resume] complete -> {result_path}")
            return 0

    # Oversub dedup: skip if an already-capped sibling at the same host-core
    # budget + precision proves this request yields the identical actor pool.
    if getattr(args, "dedup_vram_capped", False):
        red = _vram_capped_redundant(args, conc)
        if red is not None:
            m = MODELS[args.model]
            skip_doc = {
                "schema_version": 3,
                "skipped": True,
                "skip_reason": (
                    f"VRAM/RAM-capped redundant: requested concurrency {conc} would yield the "
                    f"same {red['capped_actors']}-actor pool as sibling '{red['redundant_with']}' "
                    f"(requested {red['sibling_requested']}, capped to {red['capped_actors']}) on "
                    f"identical hardware ({args.n_cpus} host CPUs, {args.device}, {args.dtype}). "
                    f"Not run to avoid recomputing an already-measured configuration."
                ),
                "redundant_with": red["redundant_with"],
                "capped_actors": red["capped_actors"],
                "meta": dict(
                    method=args.strategy, family="orchestration", dataset=args.dataset.lower(),
                    n_stations=args.n_stations, model=args.model, parent=m["parent"], child=m["child"],
                    device=args.device, n_cpus=n_cpus, dtype=args.dtype, compile=args.compile,
                    concurrency=conc, tag=args.tag,
                ),
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(skip_doc, indent=2, default=str))
            print(f"[skip-redundant] {args.tag}: conc={conc} -> "
                  f"same {red['capped_actors']} actors as {red['redundant_with']}; wrote {result_path}")
            return 0

    for i in range(args.repeats):
        if args.resume and _ok(i):
            continue
        cmd = [sys.executable, str(Path(__file__).resolve())] + _worker_argv(args, i)
        subprocess.run(cmd, cwd=str(RAPID_ROOT), env=_worker_env(args, gpu))

    from rapid.benchmark.fairness import MEMORY_KEYS, RESOURCE_KEYS, summarize_pick_quality

    timing_repeats: List[Dict[str, Any]] = []
    memory_repeats: List[Dict[str, Any]] = []
    resource_repeats: List[Dict[str, Any]] = []
    pq_repeats: List[Dict[str, Any]] = []
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
            pqf = rep_dir / f"pq_{i}.json"
            if pqf.is_file():
                try:
                    pq_i = json.loads(pqf.read_text())
                    pq_i["repeat_index"] = i
                    pq_repeats.append(pq_i)
                except Exception:
                    pass

    m = MODELS[args.model]
    meta_out = dict(
        method=args.strategy, family="orchestration", dataset=args.dataset.lower(),
        n_stations=args.n_stations, model=args.model, parent=m["parent"], child=m["child"],
        device=args.device, n_cpus=n_cpus, gpu_id=(_physical_gpu_id(args) if gpu else None),
        in_samples=args.in_samples, overlap_samples=args.overlap_samples,
        net_window=(args.net_suffix or "_w6000").lstrip("_"), window_samples=args.in_samples,
        dtype=args.dtype, compile=args.compile, concurrency=conc,
        n_windows=_expected_n_windows(args.net_suffix, args.n_stations, args.in_samples, args.overlap_samples),
        repeats=args.repeats, tag=args.tag,
        # Pick provenance (mirrors the native runner's meta): the classify
        # strategies use SeisBench's internal picker; slipstream strategies use
        # RAPID's threshold-crossing extractor inside the actor/task.
        pick_extractor=("rapid_threshold_crossing" if "slipstream" in args.strategy else "seisbench_classify"),
        p_threshold=args.p_threshold, s_threshold=args.s_threshold,
    )
    pq = summarize_pick_quality(pq_repeats) if pq_repeats else None
    result = build_result(meta=meta_out, timing_repeats=timing_repeats,
                          memory_repeats=memory_repeats, pick_quality=pq,
                          resource_repeats=resource_repeats)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {result_path}")
    return 0


def _physical_gpu_id(args) -> int:
    return int(getattr(args, "gpu_id", 0) or 0)


def _worker_env(args, gpu: bool) -> Dict[str, str]:
    env = dict(os.environ)
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(_physical_gpu_id(args))
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _worker_argv(args, repeat_index: int) -> List[str]:
    argv = [
        "--strategy", args.strategy, "--dataset", args.dataset, "--n-stations", str(args.n_stations),
        "--model", args.model, "--device", args.device, "--n-cpus", str(args.n_cpus),
        "--gpu-id", str(_physical_gpu_id(args)),
        "--core-list", args.core_list, "--concurrency", str(args.concurrency),
        "--dtype", args.dtype, "--repeats", str(args.repeats),
        "--in-samples", str(args.in_samples), "--overlap-samples", str(args.overlap_samples),
        "--net-suffix", args.net_suffix, "--slipstream-batch-size", str(args.slipstream_batch_size),
        "--p-threshold", str(args.p_threshold), "--s-threshold", str(args.s_threshold),
        "--detection-threshold", str(args.detection_threshold), "--tag", args.tag,
        "--net-root", str(args.net_root), "--results-root", str(args.results_root),
        "--tmp-dir", str(args.tmp_dir), "--repeat-index", str(repeat_index),
    ]
    if args.compile:
        argv.append("--compile")
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--dataset", required=True, choices=["stead", "txed"])
    ap.add_argument("--n-stations", type=int, required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-cpus", type=int, default=20)
    ap.add_argument("--gpu-id", type=int, default=0,
                    help="Physical CUDA device index (0 or 1) set by the scheduler.")
    ap.add_argument("--core-list", default="")
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--in-samples", type=int, default=6000)
    ap.add_argument("--overlap-samples", type=int, default=0)
    ap.add_argument("--net-suffix", default="", help="'' for 6000 net, '_w3001' for trimmed")
    ap.add_argument("--slipstream-batch-size", type=int, default=256)
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--detection-threshold", type=float, default=0.3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--net-root", type=Path, default=EQCCTPRO_ROOT / "data" / "seisbench_networks")
    ap.add_argument("--results-root", type=Path, default=RAPID_ROOT / "results" / "fair_benchmark")
    # Ray temp dir. MUST be short: Ray's AF_UNIX socket paths cap at 107 bytes,
    # and the vendored repo path is long enough that eqcctpro would silently fall
    # back to /tmp/eqcctpro_ray (small root fs) and fill it. A short ~/rapid_ray
    # keeps Ray sessions on /home (large fs).
    ap.add_argument("--tmp-dir", type=Path, default=Path.home() / "rapid_ray")
    ap.add_argument("--repeat-index", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dedup-vram-capped", action="store_true",
                    help="Skip this trial (writing a noted skip result.json) if an already-capped "
                         "sibling at the same host-core budget + precision proves the requested "
                         "concurrency yields an identical actor pool. Used by the oversub sweep.")
    args = ap.parse_args()
    if args.repeat_index >= 0:
        return run_one_repeat(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
