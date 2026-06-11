#!/usr/bin/env python3
"""Single global FCFS scheduler for the unified fair benchmark.

Replaces the old two-layer scheduling (bash slot pool + per-sweep waves). One
queue holds every trial (native + orchestration). The scheduler owns the core
map and dispatches each trial onto a DISJOINT set of logical cores -- never
overlapping ranges. GPU trials each claim one of ``--num-gpus`` dedicated host-
core blocks (default: cores 0-19 for GPU0, 20-39 for GPU1) and the matching
``CUDA_VISIBLE_DEVICES`` index, so up to two GPU trials can run concurrently
without sharing a GPU or host-core budget. CPU trials draw from the remaining
pool. Slots refill the instant a trial finishes (backfill in submission order).

Each trial is one subprocess:
* native      -> ``run_fair_trial.py`` (annotate / classify / slipstream)
* orchestration -> ``run_fair_orch_trial.py`` (Ripper / Model-Actor / MAS)

Both emit the unified schema-v2 ``result.json`` and support ``--resume``; the
scheduler also pre-skips trials whose result.json is already complete.

Examples::

    # Dry-run the whole matrix (prints every trial + resource needs)
    python scripts/run_fair_scheduler.py --dry-run

    # Native-only, CPU grid + GPU, 5 repeats, 120-core machine
    python scripts/run_fair_scheduler.py --family native --total-cpus 120
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

RAPID_ROOT = Path(__file__).resolve().parents[1]
EQCCTPRO_ROOT = RAPID_ROOT  # eqcctpro package is vendored inside RAPID

DATASETS = ["stead", "txed"]
STATION_COUNTS = [250, 580]
DEFAULT_CPU_GRID = [5, 8, 11, 14, 17, 20]
DEFAULT_BATCH_SIZES = [64, 128, 256, 512]
DEFAULT_REPEATS = 5

# Slipstream precisions per model (fp16 overflows EQT attention masks).
EQT_MODELS = {"EQTransformer", "EQT-NC"}
ALL_MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]

# Three-regime window matrix (applied IDENTICALLY to native + orchestration +
# streaming so all families are directly comparable). Each regime is
# (tag, in_samples, overlap_samples, net_suffix):
#   * EQT / EQT-NC ("w6000"): one 6000-sample window on the 6000-sample net.
#   * PhaseNet / PhaseNetLight "w6000x2": two 3001-sample windows over the full
#     6000-sample trace -- [0:3001] and the tail [2999:6000] (overlap 0).
#   * PhaseNet / PhaseNetLight "w6000ov03": 3001-sample windows at 0.3 overlap
#     (900 samples) over the 6000-sample trace -> 3 windows/station.
#   * PhaseNet / PhaseNetLight "w3001": one 3001-sample window on the trimmed
#     3001-sample net -> 1 window/station.
def _window_regimes(model: str):
    if model in EQT_MODELS:
        return [("w6000", 6000, 0, "")]
    return [
        ("w6000x2", 3001, 0, ""),
        ("w6000ov03", 3001, 900, ""),
        ("w3001", 3001, 0, "_w3001"),
    ]


def _slipstream_specs(model: str):
    """(dtype, compile, tag-fragment) for each slipstream precision.

    fp32 baseline for ALL models. EQT family: bf16 +/- compile only (fp16 is
    blocked by the -1e10 pad sentinel). PN/PNL: fp16 and bf16, each +/- compile.
    cast_weights=True for any reduced precision is enforced inside the backend.
    """
    precs = [("fp32", False, "fp32")]
    if model not in EQT_MODELS:
        precs += [("fp16", False, "fp16"), ("fp16", True, "fp16_compile")]
    precs += [("bf16", False, "bf16"), ("bf16", True, "bf16_compile")]
    return precs

# Orchestration strategies (eqcctpro). The *_slipstream strategies sweep
# precision x batch; ripper/modelactor run SeisBench classify() end-to-end at
# fp32 (no batch dimension).
ORCH_STRATEGIES = ["ripper", "ripper_slipstream", "modelactor", "modelactor_slipstream"]
ORCH_SLIPSTREAM_STRATEGIES = {"ripper_slipstream", "modelactor_slipstream"}

# Streaming (warm Model-Actor) strategies: 4 paced feeds, actors stay alive.
STREAM_STRATEGIES = ["stream_modelactor", "stream_modelactor_slipstream"]


# ---------------------------------------------------------------------------
# Trial spec
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    trial_id: str
    cmd: List[str]
    n_cpus: int
    needs_gpu: bool
    result_path: Path
    repeats: int
    # runtime state
    cores: List[int] = field(default_factory=list)
    gpu_index: Optional[int] = None
    proc: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None


def _is_complete(result_path: Path, repeats: int) -> bool:
    if not result_path.is_file():
        return False
    try:
        data = json.loads(result_path.read_text())
        reps = data.get("timing", {}).get("repeats", [])
        done = sum(1 for r in reps if r.get("success"))
        return done >= repeats
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------


def _devices(cpu_grid: List[int], sweep_gpu: bool):
    # GPU trials march the SAME host-CPU sweep as CPU trials (5,8,11,14,17,20):
    # inference runs on the GPU while the host-core budget (affinity + torch/TF
    # threads, and classify pool workers) varies, so we can see how host CPU
    # resources affect the GPU pipeline. Each GPU trial claims one GPU slot.
    devs = [("cpu", c) for c in cpu_grid]
    if sweep_gpu:
        devs += [("gpu", c) for c in cpu_grid]
    return devs


def build_native_trials(args) -> List[Trial]:
    trials: List[Trial] = []
    py = sys.executable
    runner = str(RAPID_ROOT / "scripts" / "run_fair_trial.py")
    devices = _devices(args.cpu_grid, args.sweep_gpu)
    models = args.models or ALL_MODELS

    for model in models:
        for wtag, in_samples, overlap, net_suffix in _window_regimes(model):
            for dataset in (args.datasets or DATASETS):
                for n_st in (args.stations or STATION_COUNTS):
                    for dev_kind, n_cpus in devices:
                        gpu = dev_kind == "gpu"
                        ncpu_eff = n_cpus
                        dtag = f"gpu_cpu{n_cpus}" if gpu else f"cpu{n_cpus}"

                        def mk(method: str, extra_tag: str, dtype: str, compile_flag: bool, batch: int):
                            tag = f"{method}_{wtag}_{extra_tag}_{dtag}"
                            base = (
                                args.results_root / method / dataset.lower()
                                / f"{n_st}st" / model / tag
                            )
                            cmd = [
                                py, runner,
                                "--method", method,
                                "--dataset", dataset,
                                "--n-stations", str(n_st),
                                "--model", model,
                                "--device", dev_kind,
                                "--n-cpus", str(ncpu_eff),
                                "--in-samples", str(in_samples),
                                "--overlap-samples", str(overlap),
                                "--net-suffix", net_suffix,
                                "--dtype", dtype,
                                "--batch-size", str(batch),
                                "--repeats", str(args.repeats),
                                "--tag", tag,
                                "--net-root", str(args.net_root),
                                "--results-root", str(args.results_root),
                                "--p-threshold", str(args.p_threshold),
                                "--s-threshold", str(args.s_threshold),
                                "--resume",
                            ]
                            if compile_flag:
                                cmd.append("--compile")
                            return Trial(
                                trial_id=f"{dataset}/{n_st}st/{model}/{tag}",
                                cmd=cmd, n_cpus=ncpu_eff, needs_gpu=gpu,
                                result_path=base / "result.json", repeats=args.repeats,
                            )

                        if "annotate" in args.methods:
                            for bs in args.batch_sizes:
                                trials.append(mk("annotate", f"bs{bs}", "fp32", False, bs))
                        if "classify" in args.methods:
                            trials.append(mk("classify", "single", "fp32", False, 1))
                        if "slipstream" in args.methods:
                            for dtype, comp, ptag in _slipstream_specs(model):
                                for bs in args.batch_sizes:
                                    trials.append(mk("slipstream", f"{ptag}_bs{bs}", dtype, comp, bs))
    return trials


def build_orch_trials(args) -> List[Trial]:
    """Orchestration trials (Ripper / Model-Actor / MAS) via run_fair_orch_trial.py."""
    trials: List[Trial] = []
    runner = RAPID_ROOT / "scripts" / "run_fair_orch_trial.py"
    if not runner.is_file():
        return trials  # wrapper not present yet
    py = sys.executable
    devices = _devices(args.cpu_grid, args.sweep_gpu)
    models = args.models or ALL_MODELS

    for model in models:
        # Identical window regimes to native so the two are directly comparable.
        for wtag, in_samples, overlap, net_suffix in _window_regimes(model):
            for strategy in (args.orch_strategies or ORCH_STRATEGIES):
                slip = strategy in ORCH_SLIPSTREAM_STRATEGIES
                precs = _slipstream_specs(model) if slip else [("fp32", False, "fp32")]
                # Ripper re-loads (and would re-compile) the model inside EVERY
                # station task, so torch.compile costs are paid per task -- a
                # configuration nobody would deploy. Dropped 2026-06-11 to cut
                # ~2,700 of the slowest trials; Model-Actor keeps its compile
                # variants (compiled once per persistent actor).
                if strategy == "ripper_slipstream":
                    precs = [p for p in precs if not p[1]]
                batches = args.batch_sizes if slip else [256]
                for dataset in (args.datasets or DATASETS):
                    for n_st in (args.stations or STATION_COUNTS):
                        for dev_kind, n_cpus in devices:
                            gpu = dev_kind == "gpu"
                            ncpu_eff = n_cpus
                            dtag = f"gpu_cpu{n_cpus}" if gpu else f"cpu{n_cpus}"
                            for dtype, comp, ptag in precs:
                                for bs in batches:
                                    btag = f"_bs{bs}" if slip else ""
                                    tag = f"{strategy}_{wtag}_{ptag}{btag}_{dtag}"
                                    base = args.results_root / "orchestration" / strategy / dataset.lower() / f"{n_st}st" / model / tag
                                    cmd = [
                                        py, str(runner),
                                        "--strategy", strategy,
                                        "--dataset", dataset,
                                        "--n-stations", str(n_st),
                                        "--model", model,
                                        "--device", dev_kind,
                                        "--n-cpus", str(ncpu_eff),
                                        "--in-samples", str(in_samples),
                                        "--overlap-samples", str(overlap),
                                        "--net-suffix", net_suffix,
                                        "--dtype", dtype,
                                        "--slipstream-batch-size", str(bs),
                                        "--repeats", str(args.orch_repeats),
                                        "--tag", tag,
                                        "--net-root", str(args.net_root),
                                        "--results-root", str(args.results_root),
                                        "--resume",
                                    ]
                                    if comp:
                                        cmd.append("--compile")
                                    trials.append(Trial(
                                        trial_id=f"orch/{strategy}/{dataset}/{n_st}st/{model}/{tag}",
                                        cmd=cmd, n_cpus=ncpu_eff, needs_gpu=gpu,
                                        result_path=base / "result.json", repeats=args.orch_repeats,
                                    ))
    return trials


def build_stream_trials(args) -> List[Trial]:
    """Streaming (warm Model-Actor) trials via run_fair_stream_trial.py.

    One trial = N paced feeds (default 4 @ 60s) with the actor pool kept alive.
    ``stream_modelactor`` runs SeisBench classify() in the actors (fp32, no
    batch); ``stream_modelactor_slipstream`` sweeps precision x batch.
    """
    trials: List[Trial] = []
    runner = RAPID_ROOT / "scripts" / "run_fair_stream_trial.py"
    if not runner.is_file():
        return trials
    py = sys.executable
    devices = _devices(args.cpu_grid, args.sweep_gpu)
    models = args.models or ALL_MODELS

    for model in models:
        for wtag, in_samples, overlap, net_suffix in _window_regimes(model):
            for strategy in (args.stream_strategies or STREAM_STRATEGIES):
                slip = strategy == "stream_modelactor_slipstream"
                precs = _slipstream_specs(model) if slip else [("fp32", False, "fp32")]
                batches = args.batch_sizes if slip else [256]
                for dataset in (args.datasets or DATASETS):
                    for n_st in (args.stations or STATION_COUNTS):
                        for dev_kind, n_cpus in devices:
                            gpu = dev_kind == "gpu"
                            dtag = f"gpu_cpu{n_cpus}" if gpu else f"cpu{n_cpus}"
                            for dtype, comp, ptag in precs:
                                for bs in batches:
                                    btag = f"_bs{bs}" if slip else ""
                                    tag = f"{strategy}_{wtag}_{ptag}{btag}_{dtag}"
                                    base = args.results_root / "streaming" / strategy / dataset.lower() / f"{n_st}st" / model / tag
                                    cmd = [
                                        py, str(runner),
                                        "--strategy", strategy,
                                        "--dataset", dataset,
                                        "--n-stations", str(n_st),
                                        "--model", model,
                                        "--device", dev_kind,
                                        "--n-cpus", str(n_cpus),
                                        "--in-samples", str(in_samples),
                                        "--overlap-samples", str(overlap),
                                        "--net-suffix", net_suffix,
                                        "--dtype", dtype,
                                        "--slipstream-batch-size", str(bs),
                                        "--n-feeds", str(args.stream_feeds),
                                        "--feed-interval-s", str(args.stream_interval_s),
                                        "--repeats", str(args.stream_repeats),
                                        "--tag", tag,
                                        "--net-root", str(args.net_root),
                                        "--results-root", str(args.results_root),
                                        "--resume",
                                    ]
                                    if comp:
                                        cmd.append("--compile")
                                    trials.append(Trial(
                                        trial_id=f"stream/{strategy}/{dataset}/{n_st}st/{model}/{tag}",
                                        cmd=cmd, n_cpus=n_cpus, needs_gpu=gpu,
                                        result_path=base / "result.json", repeats=args.stream_repeats,
                                    ))
    return trials


def build_oversub_trials(args) -> List[Trial]:
    """Oversubscription sweep: concurrency = multiplier x cores at fixed core budgets.

    Probes eqcctpro's memory-bound concurrency philosophy: actors/tasks are NOT
    bound to CPUs (Ray ``num_cpus=0``; only the trial's affinity mask limits
    which cores run), so in-flight tasks may exceed the core budget until the
    RAM cap (CPU mode) or VRAM cap (GPU mode) binds. Each trial requests
    ``concurrency = mult * n_cpus`` on a fixed pinned core block; the repeat
    records carry requested ``concurrency`` vs achieved ``n_modelactors`` so
    capping is visible in the data.

    Kept about ONE variable: a single canonical window regime per model
    (w6000ov03 for the PhaseNet family, w6000 for EQT) and fp32/bs256 for the
    slipstream strategies. Results land under ``<results-root>/oversub/`` so
    they never mix with the main matrix.
    """
    trials: List[Trial] = []
    runner = RAPID_ROOT / "scripts" / "run_fair_orch_trial.py"
    if not runner.is_file():
        return trials
    py = sys.executable
    devices = _devices(args.oversub_cpu_grid, args.sweep_gpu)
    models = args.models or ALL_MODELS
    results_root = args.results_root / "oversub"

    for model in models:
        regimes = _window_regimes(model)
        # Canonical regime: index 1 = w6000ov03 for the PhaseNet family;
        # EQT models only have w6000.
        wtag, in_samples, overlap, net_suffix = regimes[min(1, len(regimes) - 1)]
        for strategy in ORCH_STRATEGIES:
            slip = strategy in ORCH_SLIPSTREAM_STRATEGIES
            btag = "_bs256" if slip else ""
            for dataset in (args.datasets or DATASETS):
                for n_st in (args.stations or STATION_COUNTS):
                    for dev_kind, n_cpus in devices:
                        gpu = dev_kind == "gpu"
                        dtag = f"gpu_cpu{n_cpus}" if gpu else f"cpu{n_cpus}"
                        for mult in args.oversub_multipliers:
                            conc = min(int(mult) * n_cpus, n_st)
                            tag = f"oversub_{strategy}_{wtag}_fp32{btag}_c{mult}x_{dtag}"
                            base = (results_root / "orchestration" / strategy / dataset.lower()
                                    / f"{n_st}st" / model / tag)
                            cmd = [
                                py, str(runner),
                                "--strategy", strategy,
                                "--dataset", dataset,
                                "--n-stations", str(n_st),
                                "--model", model,
                                "--device", dev_kind,
                                "--n-cpus", str(n_cpus),
                                "--concurrency", str(conc),
                                "--in-samples", str(in_samples),
                                "--overlap-samples", str(overlap),
                                "--net-suffix", net_suffix,
                                "--dtype", "fp32",
                                "--slipstream-batch-size", "256",
                                "--repeats", str(args.oversub_repeats),
                                "--tag", tag,
                                "--net-root", str(args.net_root),
                                "--results-root", str(results_root),
                                "--resume",
                            ]
                            trials.append(Trial(
                                trial_id=f"oversub/{strategy}/{dataset}/{n_st}st/{model}/{tag}",
                                cmd=cmd, n_cpus=n_cpus, needs_gpu=gpu,
                                result_path=base / "result.json", repeats=args.oversub_repeats,
                            ))
    return trials


# ---------------------------------------------------------------------------
# FCFS scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    def __init__(
        self,
        total_cpus: int,
        log_dir: Path,
        poll_s: float = 1.0,
        gpu_core_block: int = 20,
        num_gpus: int = 2,
    ):
        # Each GPU gets a dedicated host-core block of size gpu_core_block.
        # GPU i uses cores [i*block, (i+1)*block) and CUDA_VISIBLE_DEVICES=i.
        # CPU-only trials draw from the pool above all GPU blocks.
        self.total_cpus = total_cpus
        gpu_core_block = max(1, min(gpu_core_block, total_cpus))
        num_gpus = max(1, num_gpus)
        gpu_reserved = min(total_cpus, gpu_core_block * num_gpus)
        self.gpu_slots: List[Dict[str, object]] = []
        for gi in range(num_gpus):
            start = gi * gpu_core_block
            end = min(start + gpu_core_block, gpu_reserved)
            if start >= end:
                break
            self.gpu_slots.append({
                "cuda_id": gi,
                "cores": list(range(start, end)),
                "free": True,
            })
        self.cpu_free: List[int] = list(range(gpu_reserved, total_cpus))
        self.log_dir = log_dir
        self.poll_s = poll_s
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _gpu_slot_for(self, t: Trial) -> Optional[Dict[str, object]]:
        for slot in self.gpu_slots:
            cores = slot["cores"]  # type: ignore[assignment]
            if slot["free"] and len(cores) >= t.n_cpus:  # type: ignore[arg-type]
                return slot
        return None

    def _can_dispatch(self, t: Trial) -> bool:
        if t.needs_gpu:
            return self._gpu_slot_for(t) is not None
        return len(self.cpu_free) >= t.n_cpus

    def _alloc(self, t: Trial) -> None:
        if t.needs_gpu:
            slot = self._gpu_slot_for(t)
            if slot is None:
                raise RuntimeError(f"no GPU slot for {t.trial_id}")
            cores = slot["cores"]  # type: ignore[assignment]
            t.cores = cores[: t.n_cpus]  # type: ignore[index]
            t.gpu_index = int(slot["cuda_id"])  # type: ignore[arg-type]
            slot["free"] = False
        else:
            t.cores = self.cpu_free[: t.n_cpus]
            self.cpu_free = self.cpu_free[t.n_cpus :]
            t.gpu_index = None

    def _release(self, t: Trial) -> None:
        if t.needs_gpu and t.gpu_index is not None:
            for slot in self.gpu_slots:
                if int(slot["cuda_id"]) == int(t.gpu_index):
                    slot["free"] = True
                    break
        else:
            self.cpu_free = sorted(self.cpu_free + t.cores)
        t.cores = []
        t.gpu_index = None

    def _launch(self, t: Trial) -> None:
        cmd = list(t.cmd) + ["--core-list", ",".join(str(c) for c in t.cores)]
        if t.needs_gpu and t.gpu_index is not None:
            cmd += ["--gpu-id", str(t.gpu_index)]
        env = dict(os.environ)
        if t.needs_gpu and t.gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(t.gpu_index)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        safe = t.trial_id.replace("/", "__")
        t.log_path = self.log_dir / f"{safe}.log"
        fh = open(t.log_path, "w")
        t.proc = subprocess.Popen(cmd, cwd=str(RAPID_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        t._fh = fh  # type: ignore[attr-defined]
        if t.needs_gpu and t.gpu_index is not None:
            gpu_s = f" [GPU{t.gpu_index}]"
        elif t.needs_gpu:
            gpu_s = " [GPU]"
        else:
            gpu_s = ""
        print(
            f"[dispatch] {t.trial_id}{gpu_s} cores={t.cores[0]}..{t.cores[-1]}({t.n_cpus})",
            flush=True,
        )

    def run(self, queue: List[Trial]) -> None:
        pending = list(queue)
        running: List[Trial] = []
        total = len(pending)
        done = 0
        while pending or running:
            # Backfill in submission order: launch the first fitting trials.
            progressed = True
            while progressed:
                progressed = False
                for t in list(pending):
                    if self._can_dispatch(t):
                        self._alloc(t)
                        self._launch(t)
                        pending.remove(t)
                        running.append(t)
                        progressed = True
            # Wait for any to finish.
            time.sleep(self.poll_s)
            for t in list(running):
                if t.proc is not None and t.proc.poll() is not None:
                    rc = t.proc.returncode
                    try:
                        t._fh.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self._release(t)
                    running.remove(t)
                    done += 1
                    status = "ok" if rc == 0 else f"rc={rc}"
                    n_gpu_free = sum(1 for s in self.gpu_slots if s["free"])
                    print(
                        f"[done {done}/{total}] {t.trial_id} {status} "
                        f"(cpu_free={len(self.cpu_free)} gpu_slots_free={n_gpu_free}/{len(self.gpu_slots)})",
                        flush=True,
                    )
            # Deadlock guard: nothing running and head can never fit.
            if not running and pending:
                if not any(self._can_dispatch(t) for t in pending):
                    max_gpu_cores = max((len(s["cores"]) for s in self.gpu_slots), default=0)  # type: ignore[arg-type]
                    too_big = [
                        t for t in pending
                        if (t.needs_gpu and t.n_cpus > max_gpu_cores)
                        or (not t.needs_gpu and t.n_cpus > len(self.cpu_free) and t.n_cpus > self.total_cpus)
                    ]
                    for t in too_big:
                        cap = max_gpu_cores if t.needs_gpu else self.total_cpus
                        print(f"[skip] {t.trial_id}: needs {t.n_cpus} cpus > {cap}", flush=True)
                        pending.remove(t)
                    if not too_big:
                        print("[error] no dispatchable trials and none running; aborting", flush=True)
                        break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["native", "orchestration", "streaming", "oversub", "all"], default="all")
    ap.add_argument("--methods", default="annotate,classify,slipstream",
                    help="Native methods to include")
    ap.add_argument("--orch-strategies", default=None, help="Comma list; default all 4")
    ap.add_argument("--stream-strategies", default=None, help="Comma list; default both")
    ap.add_argument("--stream-repeats", type=int, default=3,
                    help="Independent sessions per streaming trial (each >= n_feeds * interval).")
    ap.add_argument("--stream-feeds", type=int, default=4)
    ap.add_argument("--stream-interval-s", type=float, default=60.0)
    ap.add_argument("--oversub-cpu-grid", default="5,10,15,20",
                    help="Core budgets for the oversubscription sweep (--family oversub).")
    ap.add_argument("--oversub-multipliers", default="1,2,3,4",
                    help="Requested concurrency = multiplier x cores (--family oversub).")
    ap.add_argument("--oversub-repeats", type=int, default=3)
    ap.add_argument("--models", default=None, help="Comma list; default all 4")
    ap.add_argument("--datasets", default=None, help="Comma list; default stead,txed")
    ap.add_argument("--stations", default=None, help="Comma list; default 250,580")
    ap.add_argument("--cpu-grid", default=",".join(str(c) for c in DEFAULT_CPU_GRID))
    ap.add_argument("--batch-sizes", default=",".join(str(b) for b in DEFAULT_BATCH_SIZES))
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--orch-repeats", type=int, default=1,
                    help="Repeats for orchestration trials. The native family keeps "
                         "--repeats for variance; orchestration wraps the same picker, "
                         "so one repeat per configuration suffices (decided 2026-06-10).")
    ap.add_argument("--sweep-gpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gpu-core-block", type=int, default=20,
                    help="Host cores per GPU block (GPU0: [0,N), GPU1: [N,2N), ...).")
    ap.add_argument("--num-gpus", type=int, default=2,
                    help="Concurrent GPU trial slots (each gets its own core block + CUDA index).")
    ap.add_argument("--gpu-host-cpus", type=int, default=20,
                    help="(Deprecated) GPU trials now sweep the same --cpu-grid for host cores.")
    ap.add_argument("--total-cpus", type=int, default=min(os.cpu_count() or 8, 120))
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--net-root", type=Path, default=EQCCTPRO_ROOT / "data" / "seisbench_networks")
    ap.add_argument("--results-root", type=Path, default=RAPID_ROOT / "results" / "fair_benchmark")
    ap.add_argument("--log-dir", type=Path, default=RAPID_ROOT / "results" / "fair_benchmark" / "_logs")
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Only schedule first N trials (debug)")
    args = ap.parse_args()

    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.cpu_grid = [int(c) for c in args.cpu_grid.split(",") if c.strip()]
    args.batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    args.oversub_cpu_grid = [int(c) for c in args.oversub_cpu_grid.split(",") if c.strip()]
    args.oversub_multipliers = [int(m) for m in args.oversub_multipliers.split(",") if m.strip()]
    if args.models:
        args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.datasets:
        args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.stations:
        args.stations = [int(s) for s in args.stations.split(",") if s.strip()]
    if args.orch_strategies:
        args.orch_strategies = [s.strip() for s in args.orch_strategies.split(",") if s.strip()]
    if args.stream_strategies:
        args.stream_strategies = [s.strip() for s in args.stream_strategies.split(",") if s.strip()]

    trials: List[Trial] = []
    if args.family in ("native", "all"):
        trials += build_native_trials(args)
    if args.family in ("orchestration", "all"):
        trials += build_orch_trials(args)
    # Streaming (paced real-time feeds) is excluded from "all": the paced
    # 60s feeds make the family take weeks. Run it explicitly with
    # --family streaming if ever needed.
    if args.family == "streaming":
        trials += build_stream_trials(args)
    # Oversubscription sweep is an auxiliary study, also excluded from "all";
    # run via --family oversub (see scripts/run_oversub_sweep.sh).
    if args.family == "oversub":
        trials += build_oversub_trials(args)

    if args.limit:
        trials = trials[: args.limit]

    # Resume pre-skip.
    todo = [t for t in trials if not _is_complete(t.result_path, t.repeats)]
    skipped = len(trials) - len(todo)
    print(f"Matrix: {len(trials)} trials total; {skipped} already complete; {len(todo)} to run.", flush=True)
    gpu_reserved_end = min(args.total_cpus, args.gpu_core_block * args.num_gpus)
    gpu_blocks = ", ".join(
        f"gpu{gi}=[{gi * args.gpu_core_block},{min((gi + 1) * args.gpu_core_block, gpu_reserved_end)})"
        for gi in range(args.num_gpus)
        if gi * args.gpu_core_block < gpu_reserved_end
    )
    print(
        f"Resources: total_cpus={args.total_cpus} cpu_pool=[{gpu_reserved_end},{args.total_cpus}) "
        f"gpu_blocks={gpu_blocks or 'n/a'} gpu={'yes' if args.sweep_gpu else 'no'} "
        f"gpu_host_cpu_sweep={args.cpu_grid if args.sweep_gpu else 'n/a'}",
        flush=True,
    )

    if args.dry_run:
        for t in todo:
            g = " GPU" if t.needs_gpu else ""
            print(f"  {t.trial_id}  cpus={t.n_cpus}{g}")
        return 0

    if not todo:
        print("Nothing to run.")
        return 0

    sched = Scheduler(
        total_cpus=args.total_cpus,
        log_dir=args.log_dir,
        poll_s=args.poll_s,
        gpu_core_block=args.gpu_core_block,
        num_gpus=args.num_gpus,
    )
    sched.run(todo)
    print("All trials finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
