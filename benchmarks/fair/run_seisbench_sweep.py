#!/usr/bin/env python3
"""STEAD/TXED orchestration sweep for RAPID (250 & 580 stations).

This replaces the earlier 228-station TexNet sweep. It builds synthetic mSEED
station networks from labeled STEAD/TXED catalog traces (see
``build_seisbench_network.py``) and benchmarks the three strategies the paper
now reports:

  1. ``ripper``            - Ripper, native SeisBench ``classify()`` per task (baseline).
  2. ``modelactor``        - Model-Actor, native ``classify()`` in each persistent actor.
  3. ``modelactor_slipstream`` - Model-Actor whose actors run RAPID Slipstream
                             (lean PyTorch) at FP16 / BF16 / BF16+compile.

Two phases:

  * ``timing`` - an ``EvaluateSystem`` CPU/concurrency sweep per strategy that
    produces the scaling CSVs (the orchestration tables). Timing CSV rows measure
    orchestration + inference only (Ray init, actor pool, waveform processing);
    **network synthesis** (``build_seisbench_network``) and **timechunk layout
    materialization** run in a separate prepare phase and are not included.
  * ``picks``  - a single ``RunEQCCTPro`` run per applicable strategy that *saves*
    per-station picks, then compares them against the catalog labels in the
    network manifest (precision/recall/F1, ΔT statistics, tolerances) via
    ``compare_orchestrated_picks.py``. Numerical precision only changes in
    Slipstream, so this is where BF16/FP16 pick quality is validated.

Single trial example (Model-Actor + Slipstream BF16, STEAD, 580 stations, 1 GPU)::

    python benchmarks/fair/run_seisbench_sweep.py \\
        --dataset stead --n-stations 580 \\
        --strategy modelactor_slipstream --precision bf16 \\
        --model PhaseNet --cpus 20 --gpus 0 --conc-stations 22 \\
        --phase both

Full orchestration timing sweep (CPU grid 5--20, concurrency 20%%-100%%,
250 and 580 stations, CPU-only then CPU+1 GPU)::

    python benchmarks/fair/run_seisbench_sweep.py --run-all --phase timing

Pick-quality runs (single hardware profile, saves picks + catalog JSON)::

    python benchmarks/fair/run_seisbench_sweep.py --run-all --phase picks --cpus 20 --gpus 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

RAPID_ROOT = Path(__file__).resolve().parents[2]
EQCCTPRO_ROOT = RAPID_ROOT  # eqcctpro package is vendored inside RAPID
for p in (str(RAPID_ROOT), str(EQCCTPRO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eqcctpro import EvaluateSystem, RunEQCCTPro  # noqa: E402
from eqcctpro.tools import resolve_ray_temp_dir  # noqa: E402

from scripts.build_seisbench_network import build_network  # noqa: E402
from scripts.compare_orchestrated_picks import compare_network_picks  # noqa: E402

DATASETS = ["stead", "txed"]
STATION_COUNTS = [250, 580]
DEFAULT_CPU_GRID = [5, 8, 11, 14, 17, 20]
DEFAULT_MAX_CPUS = 20

MODELS: Dict[str, Dict[str, Any]] = {
    "PhaseNet": {"model_type": "seisbench", "parent": "PhaseNet", "child": "original"},
    "PhaseNetLight": {"model_type": "seisbench", "parent": "PhaseNetLight", "child": "stead"},
    "EQTransformer": {"model_type": "seisbench", "parent": "EQTransformer", "child": "original"},
    "EQT-NC": {"model_type": "seisbench", "parent": "EQTransformer", "child": "original_nonconservative"},
}

EQT_PARENTS = {"EQTransformer", "EQT-NC"}

# Slipstream precisions per model. EQTransformer family overflows FP16 attention
# masks, so only BF16 (and BF16+compile) are valid there.
def _precisions_for(model_key: str) -> List[str]:
    if model_key in EQT_PARENTS:
        return ["bf16", "bf16_compile"]
    return ["fp16", "bf16", "bf16_compile"]


PRECISION_SPECS = {
    "fp16": dict(slipstream_dtype="fp16", slipstream_compile=False),
    "bf16": dict(slipstream_dtype="bf16", slipstream_compile=False),
    "bf16_compile": dict(slipstream_dtype="bf16", slipstream_compile=True),
}


def _net_dir(out_root: Path, dataset: str, n_stations: int) -> Path:
    return out_root / f"{dataset.lower()}_{n_stations}st"


def _times_list_from_meta(meta: Dict[str, Any], waveform_overlap: int = 0) -> list:
    """Match EvaluateSystem.chunk_time() for a single-window STEAD/TXED network."""
    from obspy import UTCDateTime

    starttime = UTCDateTime(meta["start_time"]) - (waveform_overlap * 60)
    endtime = UTCDateTime(meta["end_time"])
    timechunk_dt = int(meta["timechunk_dt"])
    times_list = []
    start = starttime
    end = start + (waveform_overlap * 60) + (timechunk_dt * 60)
    while start <= endtime:
        if end >= endtime:
            end = endtime
            times_list.append([start, end])
            break
        times_list.append([start, end])
        start = end - (waveform_overlap * 60)
        end = start + (waveform_overlap * 60) + (timechunk_dt * 60)
    return times_list


def _materialize_network_layout(
    args,
    dataset: str,
    n_stations: int,
    meta: Dict[str, Any],
    *,
    waveform_overlap: int = 0,
) -> None:
    """Expand ``<station>/*.mseed`` into timechunk dirs (EQCCT-ready layout).

    Done once before timing/picks so EvaluateSystem's per-run materialize is a
    no-op and layout work is not charged to timing CSV rows.
    """
    from eqcctpro.tools import materialize_input_into_timechunk_layout

    net_dir = _net_dir(args.net_root, dataset, n_stations)
    times_list = _times_list_from_meta(meta, waveform_overlap=waveform_overlap)
    materialize_input_into_timechunk_layout(str(net_dir), times_list, logger=None)


def _ensure_network(
    args,
    dataset: str,
    n_stations: int,
    *,
    build: bool = True,
    materialize: bool = True,
) -> Dict[str, Any]:
    net_dir = _net_dir(args.net_root, dataset, n_stations)
    manifest_path = net_dir / "manifest.json"
    if build and (args.rebuild or not manifest_path.is_file()):
        print(
            f"PREPARE (not timed): building {dataset}_{n_stations}st network "
            f"under {net_dir} ..."
        )
        build_network(
            dataset=dataset,
            n_stations=n_stations,
            out_root=args.net_root,
            n_unique=args.n_unique,
            seed=args.seed,
            require_s=args.require_s,
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing network manifest {manifest_path}. "
            f"Run with --prepare-networks-only or allow build (default)."
        )
    meta = json.loads(manifest_path.read_text())["meta"]
    if materialize and not args.dry_run:
        print(
            f"PREPARE (not timed): materializing timechunk layout for "
            f"{dataset}_{n_stations}st ..."
        )
        _materialize_network_layout(args, dataset, n_stations, meta)
    return meta


def _prepare_all_networks(args) -> None:
    """Build + materialize every STEAD/TXED network before any timing or picks."""
    print(
        "PREPARE: synthesizing datasets and EQCCT layout (excluded from timing CSVs) ..."
    )
    for dataset in DATASETS:
        for n_stations in STATION_COUNTS:
            _ensure_network(args, dataset, n_stations, build=True, materialize=True)
    print("PREPARE: all networks ready.\n")


def _strategy_flags(strategy: str, precision: Optional[str]) -> Dict[str, Any]:
    if strategy == "ripper":
        return dict(ripper=True, slipstream_inference=False,
                    slipstream_dtype="fp32", slipstream_compile=False)
    if strategy == "modelactor":
        return dict(ripper=False, slipstream_inference=False,
                    slipstream_dtype="fp32", slipstream_compile=False)
    if strategy == "modelactor_slipstream":
        spec = PRECISION_SPECS[precision]
        return dict(ripper=False, slipstream_inference=True, **spec)
    raise ValueError(f"Unknown strategy {strategy}")


def _config_tag(dataset, n_stations, model_key, strategy, precision, device_tag: str = ""):
    tag = f"{dataset}_{n_stations}st_{model_key}_{strategy}"
    if strategy == "modelactor_slipstream":
        tag += f"_{precision}"
    if device_tag:
        tag += f"_{device_tag}"
    return tag


# Datasets each strategy is benchmarked against.
#   Ripper / Model-Actor are orchestration baselines: timing only, STEAD only
#   (their picks are numerically classify()-identical, so no cross-dataset
#   pick comparison is needed).
#   Model-Actor Slipstream changes numerics (FP16/BF16), so it runs on STEAD and
#   TXED for both timing and scored picks.
STRATEGY_DATASETS = {
    "ripper": ["stead"],
    "modelactor": ["stead"],
    "modelactor_slipstream": ["stead", "txed"],
}

METHOD_DIR = {
    "ripper": "Ripper",
    "modelactor": "Model-Actor",
    "modelactor_slipstream": "Model-Actor-Slipstream",
}


def _nested_subpath(dataset, n_stations, model_key, strategy, precision, device_tag: str = "") -> Path:
    """Dataset / <N>st / Method / [precision] / Model / [device] tree."""
    parts = [dataset.lower(), f"{n_stations}st", METHOD_DIR[strategy]]
    if strategy == "modelactor_slipstream" and precision:
        parts.append(precision)
    parts.append(model_key)
    if device_tag:
        parts.append(device_tag)
    return Path(*parts)


def _result_base(args, dataset, n_stations, model_key, strategy, precision, device_tag: str = "") -> Path:
    return args.results_root / _nested_subpath(
        dataset, n_stations, model_key, strategy, precision, device_tag
    )


def _parse_cpu_grid(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _cpu_sweep_step(grid: List[int]) -> int:
    """Step size so EvaluateSystem's ``range(min, max_cpus+1, step)`` hits every grid point."""
    if len(grid) < 2:
        return 1
    steps = [b - a for a, b in zip(grid, grid[1:])]
    g = steps[0]
    for d in steps[1:]:
        while d:
            g, d = d, g % d
    return max(1, g)


def _timing_sweep_kwargs(args, n_stations: int) -> Dict[str, Any]:
    """EvaluateSystem knobs for multi-CPU + 20%-100% concurrency at fixed N stations."""
    grid = args.cpu_grid
    max_cpus = args.max_cpus
    if max_cpus < max(grid):
        raise ValueError(f"--max-cpus {max_cpus} is smaller than max --cpu-grid value {max(grid)}")
    step = _cpu_sweep_step(grid)
    offset = int(getattr(args, "cpu_offset", 0) or 0)
    return dict(
        cpu_id_list=list(range(offset, offset + max_cpus)),
        min_cpu_amount=min(grid),
        cpu_test_step_size=step,
        stations2use=n_stations,
        starting_amount_of_stations=n_stations,
        station_list_step_size=n_stations,
        min_conc_stations=1,
        conc_station_tasks_step_size=0,
        conc_station_tasks_max_only=False,
        concurrency_on_max_actors=True,
    )


def _single_trial_timing_kwargs(args, n_stations: int) -> Dict[str, Any]:
    """One CPU count and a single fixed concurrency point (legacy / pick runs).

    With ``--conc-sweep`` we instead march 20%-100% on the memory-capped max
    actors. Otherwise we benchmark exactly one concurrency level equal to
    ``--conc-stations`` (default ``cpus // 2``). We deliberately avoid
    ``conc_station_tasks_max_only`` here: that flag forces the full station
    count (capped to the memory max), which silently ignores ``--conc-stations``
    and spawns far more actors than requested. Using a step size of ``n_stations``
    makes ``range(conc, n_stations+1, step)`` yield the single value ``[conc]``.
    """
    conc = max(1, min(args.conc_stations or max(1, args.cpus // 2), n_stations))
    offset = int(getattr(args, "cpu_offset", 0) or 0)
    return dict(
        cpu_id_list=list(range(offset, offset + args.cpus)),
        min_cpu_amount=args.cpus,
        cpu_test_step_size=args.cpus,
        stations2use=n_stations,
        starting_amount_of_stations=n_stations,
        station_list_step_size=n_stations,
        min_conc_stations=conc,
        conc_station_tasks_step_size=0 if args.conc_sweep else n_stations,
        conc_station_tasks_max_only=False,
        concurrency_on_max_actors=args.conc_sweep,
    )


def _run_timing(
    args,
    *,
    dataset,
    n_stations,
    model_key,
    strategy,
    precision,
    meta,
    gpus: List[int],
    device_tag: str,
) -> None:
    m = MODELS[model_key]
    flags = _strategy_flags(strategy, precision)
    if flags.get("slipstream_compile") and len(gpus) > 1:
        print(
            f"SKIP timing {_config_tag(dataset, n_stations, model_key, strategy, precision, device_tag)}: "
            "compile needs single GPU"
        )
        return

    tag = _config_tag(dataset, n_stations, model_key, strategy, precision, device_tag)
    base = _result_base(args, dataset, n_stations, model_key, strategy, precision, device_tag)
    csv_dir = base / "timing"
    csv_dir.mkdir(parents=True, exist_ok=True)

    sweep = args.timing_sweep
    timing_kw = _timing_sweep_kwargs(args, n_stations) if sweep else _single_trial_timing_kwargs(args, n_stations)

    ev = EvaluateSystem(
        eval_mode="gpu" if gpus else "cpu",
        input_dir=str(_net_dir(args.net_root, dataset, n_stations)),
        output_dir=str(csv_dir / "output"),
        log_filepath=str(csv_dir / "eqcctpro.log"),
        csv_dir=str(csv_dir),
        model_type="seisbench",
        seisbench_parent_model=m["parent"],
        seisbench_child_model=m["child"],
        selected_gpus=gpus if gpus else None,
        max_vram_mb=args.max_vram_mb,
        ram_safety_cap=args.ram_safety_cap,
        cudnn_headroom=args.cudnn_headroom,
        start_time=meta["start_time"],
        end_time=meta["end_time"],
        timechunk_dt=meta["timechunk_dt"],
        waveform_overlap=0,
        tmp_dir=str(args.tmp_dir),
        Detection_threshold=args.detection_threshold,
        P_threshold=args.p_threshold,
        S_threshold=args.s_threshold,
        slipstream_overlap_samples=args.slipstream_overlap_samples,
        slipstream_batch_size=args.slipstream_batch_size,
        pick_output_format="ascii",
        ascii_station_pick_format="csv",
        overwrite=args.overwrite,
        concurrency_march_fraction=args.conc_march_frac,
        concurrency_values=args.conc_values,
        exact_resume_match=args.exact_resume,
        min_gpu_amount=int(getattr(args, "min_gpus", 1) or 1),
        **timing_kw,
        **flags,
    )
    mode = "SWEEP" if sweep else "SINGLE"
    if args.conc_values:
        conc_desc = f"explicit{args.conc_values}"
    elif sweep or args.conc_sweep:
        conc_desc = f"{int(round(args.conc_march_frac * 100))}% march"
    else:
        conc_desc = "max-only"
    print(
        f"\n=== TIMING [{mode}]: {tag} device={device_tag or 'cpu'} "
        f"max_cpus={len(timing_kw['cpu_id_list'])} min_cpu={timing_kw['min_cpu_amount']} "
        f"cpu_step={timing_kw['cpu_test_step_size']} gpus={gpus or 'none'} "
        f"N={n_stations} conc={conc_desc} ==="
    )
    if args.dry_run:
        return
    ev.evaluate()


def _run_picks(
    args,
    *,
    dataset,
    n_stations,
    model_key,
    strategy,
    precision,
    meta,
    gpus: Optional[List[int]] = None,
    device_tag: str = "",
) -> None:
    m = MODELS[model_key]
    flags = _strategy_flags(strategy, precision)
    gpus = list(gpus) if gpus is not None else list(args.gpus)
    if flags.get("slipstream_compile") and len(gpus) > 1:
        print("SKIP picks: compile needs single GPU")
        return

    tag = _config_tag(dataset, n_stations, model_key, strategy, precision, device_tag)
    base = _result_base(args, dataset, n_stations, model_key, strategy, precision, device_tag)
    pick_dir = base / "picks"
    pick_dir.mkdir(parents=True, exist_ok=True)
    pick_cpus = args.pick_cpus or args.cpus
    conc = args.conc_stations or max(1, pick_cpus // 2)

    print(f"\n=== PICKS: {tag} cpus={pick_cpus} gpus={gpus or 'none'} conc<={conc} N={n_stations} ===")
    if args.dry_run:
        return

    def _build(c: int):
        return RunEQCCTPro(
            use_gpu=bool(gpus),
            input_dir=str(_net_dir(args.net_root, dataset, n_stations)),
            output_dir=str(pick_dir),
            log_filepath=str(pick_dir / "eqcctpro.log"),
            model_type="seisbench",
            seisbench_parent_model=m["parent"],
            seisbench_child_model=m["child"],
            number_of_concurrent_station_predictions=c,
            selected_gpus=gpus if gpus else None,
            stations2use=n_stations,
            vram_mb=None,  # auto per-worker budget = total safe VRAM / concurrent workers
            cpu_id_list=list(range(pick_cpus)),
            start_time=meta["start_time"],
            end_time=meta["end_time"],
            timechunk_dt=meta["timechunk_dt"],
            waveform_overlap=0,
            tmp_dir=str(args.tmp_dir),
            Detection_threshold=args.detection_threshold,
            P_threshold=args.p_threshold,
            S_threshold=args.s_threshold,
            slipstream_overlap_samples=args.slipstream_overlap_samples,
            slipstream_batch_size=args.slipstream_batch_size,
            pick_output_format="ascii",
            ascii_station_pick_format="csv",
            overwrite=True,
            **flags,
        )

    # The upfront per-GPU VRAM admission reserves the model footprint on top of
    # the auto per-worker budget, which can reject the requested concurrency by a
    # worker or two. Picks only need correctness, not max throughput, so back off
    # concurrency until the run is admitted.
    run = None
    while conc >= 1:
        try:
            run = _build(conc)
            break
        except RuntimeError as exc:
            if "insufficient" in str(exc) and conc > 1:
                new_conc = max(1, conc - 1)
                print(f"  VRAM admission rejected conc={conc}; retrying conc={new_conc}")
                conc = new_conc
                continue
            raise
    if run is None:
        run = _build(1)
    run.run_eqcctpro()

    manifest_path = _net_dir(args.net_root, dataset, n_stations) / "manifest.json"
    out_json = base / "pick_quality.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = compare_network_picks(
        manifest_path=manifest_path,
        picks_dir=pick_dir,
        out_json=out_json,
        label=tag,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
    )
    if summary:
        print(f"  pick-quality: P F1={summary['P'].get('f1'):.3f} "
              f"R={summary['P'].get('recall'):.3f} | metrics -> {out_json}")


def _applicable(strategy: str, model_key: str, precision: Optional[str]) -> bool:
    if strategy == "modelactor_slipstream":
        return precision in _precisions_for(model_key)
    return True


def _timing_devices_for_run_all(args, *, compile_config: bool) -> List[tuple[List[int], str]]:
    """Timing device blocks for --run-all.

    The GPU block passes the full ``--gpu-ids`` set; EvaluateSystem's GPU loop
    then sweeps 1-GPU, 2-GPU, ... internally (``selected_gpus[:k+1]``), so one
    call covers both single- and dual-GPU timing. Compiled Slipstream forces a
    single GPU because multiple compiled CUDA graphs in one process can corrupt.
    """
    modes: List[tuple[List[int], str]] = []
    if args.sweep_cpu_only:
        modes.append(([], "cpu"))
    if args.sweep_with_gpu and args.gpu_ids:
        gpu_set = [args.gpu_ids[0]] if compile_config else list(args.gpu_ids)
        modes.append((gpu_set, "gpu"))
    if not modes:
        raise ValueError("No device modes selected; use --sweep-cpu-only and/or --sweep-with-gpu")
    return modes


def _pick_devices_for_run_all(args) -> List[tuple[List[int], str]]:
    """Hardware profiles for saved+scored pick runs (cross-hardware consistency).

    Pick numerics depend on device + precision, not on actor-pool size or GPU
    count, so picks run once per profile (CPU and a single GPU) rather than over
    the whole CPU/concurrency grid.
    """
    modes: List[tuple[List[int], str]] = []
    for token in args.pick_devices:
        token = token.strip().lower()
        if token == "cpu":
            modes.append(([], "cpu"))
        elif token.startswith("gpu"):
            gid = int(token[3:]) if token[3:] else args.gpu_ids[0]
            modes.append(([gid], f"gpu{gid}"))
    return modes


def _do_timing(args, *, dataset, n_stations, model_key, strategy, precision, meta, compile_config):
    for gpus, device_tag in _timing_devices_for_run_all(args, compile_config=compile_config):
        _run_timing(
            args,
            dataset=dataset,
            n_stations=n_stations,
            model_key=model_key,
            strategy=strategy,
            precision=precision,
            meta=meta,
            gpus=gpus,
            device_tag=device_tag,
        )


def _do_picks(args, *, dataset, n_stations, model_key, strategy, precision, meta):
    # Pick quality (and pick-time) only matters where numerics can change, i.e.
    # Model-Actor Slipstream (FP16/BF16). Ripper and Model-Actor use the native
    # classify() backend and are scored only for timing, so they skip the pick
    # phase entirely. Slipstream picks are scored against the catalog ground
    # truth in the network manifest (true reference picks for STEAD and TXED).
    if strategy != "modelactor_slipstream":
        return
    for pick_gpus, pick_tag in _pick_devices_for_run_all(args):
        _run_picks(
            args,
            dataset=dataset,
            n_stations=n_stations,
            model_key=model_key,
            strategy=strategy,
            precision=precision,
            meta=meta,
            gpus=pick_gpus,
            device_tag=pick_tag,
        )


def _do(args, *, dataset, n_stations, model_key, strategy, precision, meta):
    compile_config = bool(_strategy_flags(strategy, precision).get("slipstream_compile"))
    if args.phase in ("timing", "both"):
        _do_timing(args, dataset=dataset, n_stations=n_stations, model_key=model_key,
                   strategy=strategy, precision=precision, meta=meta, compile_config=compile_config)
    if args.phase in ("picks", "both"):
        _do_picks(args, dataset=dataset, n_stations=n_stations, model_key=model_key,
                  strategy=strategy, precision=precision, meta=meta)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-root", type=Path, default=EQCCTPRO_ROOT / "data" / "seisbench_networks")
    ap.add_argument(
        "--results-root",
        type=Path,
        default=RAPID_ROOT / "results" / "orchestration_sweep",
        help="Root for all trial data; nested as <dataset>/<N>st/<Method>/[precision]/<Model>/<device>/",
    )
    ap.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Ray/temp root (default: short path under $TMPDIR or /tmp/eqcctpro_ray; "
        "avoid deep repo paths — AF_UNIX socket limit is 107 bytes)",
    )

    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--n-stations", type=int, choices=STATION_COUNTS)
    ap.add_argument("--model", choices=list(MODELS.keys()))
    ap.add_argument("--strategy", choices=["ripper", "modelactor", "modelactor_slipstream"])
    ap.add_argument("--precision", choices=list(PRECISION_SPECS.keys()))
    ap.add_argument(
        "--phase",
        choices=["timing", "picks", "both"],
        default="both",
        help="timing = EvaluateSystem scaling sweep; picks = saved+scored RunEQCCTPro; both = timing then picks",
    )

    ap.add_argument("--cpus", type=int, default=20, help="CPU count for single-config / single-trial runs")
    ap.add_argument(
        "--gpus",
        default="0",
        help="GPU id(s) for single-config runs (comma-separated); empty = CPU-only",
    )
    ap.add_argument("--conc-stations", type=int, default=None, help="Fixed actor/task pool (single-trial / picks)")
    ap.add_argument(
        "--timing-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Multi-CPU grid + 20%%-100%% concurrency via EvaluateSystem (default on; use --no-timing-sweep for one trial)",
    )
    ap.add_argument(
        "--conc-sweep",
        action="store_true",
        help="With --no-timing-sweep, still use 20%%-100%% concurrency steps",
    )
    ap.add_argument("--max-cpus", type=int, default=DEFAULT_MAX_CPUS, help="Affinity pool size (N cores)")
    ap.add_argument(
        "--cpu-offset",
        type=int,
        default=0,
        help="Base core index for CPU affinity; trials pin to cores "
        "[offset .. offset+max_cpus-1]. Lets strategies run concurrently on disjoint "
        "core ranges (e.g. ripper 0-19, modelactor 20-39, slipstream 40-59).",
    )
    ap.add_argument(
        "--only-strategies",
        default="",
        help="Restrict --run-all to these strategies (comma-separated): "
        "ripper, modelactor, modelactor_slipstream. Empty = all.",
    )
    ap.add_argument(
        "--cpu-grid",
        default=",".join(str(c) for c in DEFAULT_CPU_GRID),
        help="CPU counts to test when --timing-sweep (comma-separated)",
    )
    ap.add_argument(
        "--conc-march-frac",
        type=float,
        default=0.2,
        help="Concurrency march step as a fraction of the feasible max actors/tasks "
        "(0.2=20%% default; use 0.1 for a finer 10%% march with more low-end points)",
    )
    ap.add_argument(
        "--conc-values",
        default="",
        help="Explicit actor/task concurrency counts to benchmark (comma-separated, e.g. "
        "'1,2,5,10,20,43,86'); overrides --conc-march-frac. Each is capped to the feasible max.",
    )
    ap.add_argument(
        "--exact-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume/skip using ONLY exact trial matches (no +/-5%% fuzzy tolerance) so a "
        "finer march runs new points near existing ones while reusing identical rows. "
        "Use --no-exact-resume for the legacy fuzzy de-dup.",
    )
    ap.add_argument(
        "--gpu-ids",
        default="0,1",
        help="GPU set for the --run-all GPU timing block; EvaluateSystem sweeps "
        "min_gpus..len GPUs internally (e.g. '0' = 1 GPU on device 0; '0,1' = 1- and 2-GPU)",
    )
    ap.add_argument(
        "--min-gpus",
        type=int,
        default=1,
        help="Smallest GPU count to benchmark in the GPU block. Use 2 with --gpu-ids 0,1 "
        "for a dual-GPU-only phase that skips re-running 1-GPU trials.",
    )
    ap.add_argument(
        "--sweep-cpu-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --run-all, run CPU-only timing blocks",
    )
    ap.add_argument(
        "--sweep-with-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --run-all, run the GPU timing block (1 GPU and 2 GPUs via --gpu-ids)",
    )
    ap.add_argument(
        "--pick-devices",
        default="cpu,gpu0",
        help="Hardware profiles for saved+scored pick runs (comma): e.g. cpu,gpu0 (cross-hardware consistency)",
    )
    ap.add_argument("--pick-cpus", type=int, default=None, help="CPU count for pick runs (default --cpus / --max-cpus)")

    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--detection-threshold", type=float, default=0.3)
    ap.add_argument("--ram-safety-cap", type=float, default=0.95)
    ap.add_argument("--cudnn-headroom", type=float, default=0.20)
    ap.add_argument("--max-vram-mb", type=float, default=40000)
    ap.add_argument("--slipstream-overlap-samples", type=int, default=0)
    ap.add_argument("--slipstream-batch-size", type=int, default=256)

    ap.add_argument("--n-unique", type=int, default=None, help="Distinct catalog traces before tiling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-s", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="Rebuild networks even if manifest exists")

    ap.add_argument("--run-all", action="store_true")
    ap.add_argument(
        "--prepare-networks-only",
        action="store_true",
        help="Only build STEAD/TXED networks + timechunk layout, then exit (no timing/picks)",
    )
    ap.add_argument(
        "--skip-network-prep",
        action="store_true",
        help="With --run-all, assume networks already exist; do not rebuild or rematerialize",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.gpus = [int(x) for x in args.gpus.split(",") if x.strip()] if args.gpus.strip() else []
    args.tmp_dir = Path(resolve_ray_temp_dir(args.tmp_dir))
    args.cpu_grid = _parse_cpu_grid(args.cpu_grid)
    args.gpu_ids = [int(x) for x in str(args.gpu_ids).split(",") if x.strip()] if str(args.gpu_ids).strip() else []
    args.pick_devices = [t for t in str(args.pick_devices).split(",") if t.strip()]
    args.conc_values = (
        [int(x) for x in str(args.conc_values).split(",") if x.strip()]
        if str(args.conc_values).strip()
        else None
    )
    _valid_strats = {"ripper", "modelactor", "modelactor_slipstream"}
    args.only_strategies = [s.strip() for s in str(args.only_strategies).split(",") if s.strip()]
    bad = [s for s in args.only_strategies if s not in _valid_strats]
    if bad:
        ap.error(f"--only-strategies has invalid value(s) {bad}; choose from {sorted(_valid_strats)}")

    if args.prepare_networks_only:
        if not args.dry_run:
            _prepare_all_networks(args)
        else:
            print("DRY-RUN: would prepare networks for", DATASETS, STATION_COUNTS)
        return 0

    if args.run_all:
        if args.phase in ("timing", "both"):
            args.timing_sweep = True
        timing_tags = [t for _, t in _timing_devices_for_run_all(args, compile_config=False)]
        gpu_block = next((g for g, t in _timing_devices_for_run_all(args, compile_config=False) if t == "gpu"), [])
        print(
            f"RUN-ALL: stations={STATION_COUNTS} phase={args.phase}\n"
            f"  datasets per strategy: ripper={STRATEGY_DATASETS['ripper']} "
            f"modelactor={STRATEGY_DATASETS['modelactor']} "
            f"modelactor_slipstream={STRATEGY_DATASETS['modelactor_slipstream']}\n"
            f"  timing: cpu_grid={args.cpu_grid} max_cpus={args.max_cpus} "
            f"concurrency=20% march on memory-capped max actors\n"
            f"  timing CSVs: processing only (excludes network build + layout prep)\n"
            f"  timing devices={timing_tags} (GPU block sweeps 1..{len(gpu_block)} GPUs from {gpu_block})\n"
            f"  picks (scored, slipstream only): devices={args.pick_devices}"
        )
        if not args.dry_run and not args.skip_network_prep:
            _prepare_all_networks(args)
        network_cache: Dict[tuple[str, int], Dict[str, Any]] = {}
        for dataset in DATASETS:
            for n_stations in STATION_COUNTS:
                key = (dataset, n_stations)
                if key not in network_cache:
                    network_cache[key] = _ensure_network(
                        args, dataset, n_stations, build=False, materialize=False
                    )
                meta = network_cache[key]
                for model_key in MODELS:
                    for strategy in ("ripper", "modelactor", "modelactor_slipstream"):
                        if args.only_strategies and strategy not in args.only_strategies:
                            continue
                        if dataset not in STRATEGY_DATASETS[strategy]:
                            # Ripper / Model-Actor are STEAD-only (timing baselines).
                            continue
                        if strategy == "modelactor_slipstream":
                            for precision in _precisions_for(model_key):
                                _do(args, dataset=dataset, n_stations=n_stations,
                                    model_key=model_key, strategy=strategy,
                                    precision=precision, meta=meta)
                        else:
                            _do(args, dataset=dataset, n_stations=n_stations,
                                model_key=model_key, strategy=strategy,
                                precision=None, meta=meta)
        return 0

    missing = [k for k in ("dataset", "n_stations", "model", "strategy") if getattr(args, k) is None]
    if missing:
        ap.error(f"Specify {missing} (or use --run-all)")
    if args.strategy == "modelactor_slipstream":
        if not args.precision:
            ap.error("--precision required for modelactor_slipstream")
        if args.precision not in _precisions_for(args.model):
            ap.error(f"{args.precision} invalid for {args.model}; allowed {_precisions_for(args.model)}")

    # Single-config: build + materialize once here, then time only processing below.
    prep = not args.skip_network_prep
    meta = _ensure_network(args, args.dataset, args.n_stations, build=prep, materialize=prep)
    gpus = args.gpus
    device_tag = f"gpu{gpus[0]}" if gpus else "cpu"
    compile_config = bool(_strategy_flags(args.strategy, args.precision).get("slipstream_compile"))
    if args.phase in ("timing", "both"):
        _run_timing(args, dataset=args.dataset, n_stations=args.n_stations, model_key=args.model,
                    strategy=args.strategy, precision=args.precision, meta=meta,
                    gpus=gpus, device_tag=device_tag)
    if args.phase in ("picks", "both") and args.strategy == "modelactor_slipstream":
        _run_picks(args, dataset=args.dataset, n_stations=args.n_stations, model_key=args.model,
                   strategy=args.strategy, precision=args.precision, meta=meta,
                   gpus=gpus, device_tag=device_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
