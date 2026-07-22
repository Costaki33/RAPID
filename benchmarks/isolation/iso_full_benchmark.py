#!/usr/bin/env python3
"""Generator + runner for the FULL isolated benchmark (results/iso_full_benchmark/).

Enumerates the trial grid for the iso-only paper and runs each cell STRICTLY
SEQUENTIALLY (one trial at a time, idle box) by shelling out to the existing
trial drivers with --resume. Every trial records timing, memory (peak
process-tree PSS), and pick quality vs catalog.

By default the grid is the COST-REDUCED scope (see prunes below), which produces
every number the paper reports in ~1 day of sequential compute instead of ~15.
Pass --full to enumerate the complete ideal grid.

Common axes (4 models x {250,580} stations; STEAD timing, TXED pick-quality only):
  native    classify/classify_batched/annotate/slipstream, single process, cores
            {5,10,15,20}, torch threads swept as round(mult*cores), mult
            {0.25,0.5,1,2,3,4}. classify_batched (full-network classify in one
            call = SeisBench's best single-process picker) runs CPU + 1 GPU.
  orch      cold start, CPU + 1-GPU, concurrency = cores in {5,10,15,20}.
  oversub   actors-per-core swept as round(mult*cores), cores {5,10,15,20}, CPU + 1-GPU.
  stream    warm, CPU + 1-GPU, concurrency = cores in {5,10,15,20}, 8 feeds x 5 sessions.

Per-model precision policy ("with respect to the model"): FP16 is unsafe for
EQT/EQT-NC -> those run {fp32,bf16}; PN/PNL run {fp32,fp16,bf16}. Each dtype runs
with and without torch.compile.

Default (cost-reduced) prunes -- see README.md for the rationale and timings:
  * TXED carries pick quality only, which is already supplied by the consolidated
    txed_native re-run, so the generated grid is STEAD-only.
  * native per-station classify is NOT re-swept (its thread sensitivity is already
    captured by the consolidated 20-core thread sweep); annotate + slipstream are
    emitted at 1 thread (both thread-insensitive). classify_batched (SeisBench's
    best single-process picker) IS emitted, CPU + 1 GPU, at 1 thread.
  * Ripper is dropped from the orch sweep (it is already shown unviable; the
    20-core control is consolidated).
  * oversub is 580-only (250 is a one-off generalization spot-check elsewhere).
  * torch.compile variants are kept ONLY in the warm streaming family (the only
    place steady-state forward speed could benefit); native/orch/oversub run
    eager only.
  * Repeats are 3 everywhere (oversub included) for cross-trial fairness.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

RAPID_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = RAPID_ROOT / "results" / "iso_full_benchmark"

MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
STATIONS = [250, 580]
CORES = [5, 10, 15, 20]
MULTS = [0.25, 0.5, 1, 2, 3, 4]          # ascending (required for oversub dedup)
REPEATS = 3
STREAM_SESSIONS = 5
STREAM_FEEDS = 8
MAX_THREADS = 64
PY = sys.executable

Trial = Dict[str, object]


@dataclass
class Cfg:
    full: bool = False

    @property
    def datasets(self) -> List[str]:
        # TXED pick quality comes from the consolidated txed_native re-run; the
        # generated grid is STEAD-only unless --full.
        return ["stead", "txed"] if self.full else ["stead"]

    @property
    def drop_native_classify(self) -> bool:
        return not self.full

    @property
    def drop_orch_ripper(self) -> bool:
        return not self.full

    @property
    def flat_threads(self) -> bool:
        return not self.full

    @property
    def oversub_stations(self) -> List[int]:
        return STATIONS if self.full else [580]

    def compile_allowed(self, family: str) -> bool:
        return self.full or family == "stream"


def insamples(model: str) -> int:
    return 3001 if model in ("PhaseNet", "PhaseNetLight") else 6000


def core_list(cores: int) -> str:
    return ",".join(str(c) for c in range(cores))


def mult_to_int(mult: float, base: int, lo: int = 1, hi: int | None = None) -> int:
    v = max(lo, int(math.floor(mult * base + 0.5)))
    return min(v, hi) if hi is not None else v


def precisions(model: str, family: str, cfg: Cfg) -> List[Tuple[str, bool]]:
    dtypes = ["fp32", "fp16", "bf16"] if model in ("PhaseNet", "PhaseNetLight") else ["fp32", "bf16"]
    out: List[Tuple[str, bool]] = []
    for dt in dtypes:
        out.append((dt, False))
        if cfg.compile_allowed(family):
            out.append((dt, True))
    return out


def _pcsuffix(dt: str, cmp: bool) -> str:
    return f"_{dt}{'_compile' if cmp else ''}"


def _mtag(mult: float) -> str:
    return str(int(mult)) if float(mult).is_integer() else f"{mult:g}".replace(".", "p")


def native_trials(cfg: Cfg) -> List[Trial]:
    out: List[Trial] = []
    root = RESULTS_ROOT / "native"
    for ds in cfg.datasets:
        for st in STATIONS:
            for model in MODELS:
                ins = insamples(model)
                for cores in CORES:
                    full_threads = sorted({mult_to_int(m, cores, hi=MAX_THREADS) for m in MULTS})
                    # annotate/slipstream are thread-insensitive -> 1 thread only when
                    # pruned (swept across the core budget to show core-independence);
                    # classify keeps the full thread sweep when not dropped.
                    flat_threads = [1] if cfg.flat_threads else full_threads
                    cl = core_list(cores)

                    def _base(thr):
                        return [
                            PY, "benchmarks/fair/run_fair_trial.py",
                            "--dataset", ds, "--n-stations", str(st), "--model", model,
                            "--device", "cpu", "--n-cpus", str(cores), "--core-list", cl,
                            "--torch-threads", str(thr), "--in-samples", str(ins),
                            "--overlap-samples", "0", "--batch-size", "256",
                            "--repeats", str(REPEATS), "--results-root", str(root), "--resume",
                        ]

                    if not cfg.drop_native_classify:
                        for thr in full_threads:
                            out.append({"desc": f"native classify {ds} {st}st {model} c{cores} thr{thr}",
                                        "argv": _base(thr) + ["--method", "classify", "--dtype", "fp32",
                                                              "--tag", f"c{cores}_thr{thr}"]})
                    # classify_batched: SeisBench's BEST single-process picker (full
                    # network in ONE classify() call -> batches across stations).
                    # The fair upper bound vs per-station classify and vs
                    # Model-Actor[classify]. 1 thread (measured optimum), CPU + 1 GPU.
                    for dev in ("cpu", "gpu"):
                        cb = [
                            PY, "benchmarks/fair/run_fair_trial.py",
                            "--dataset", ds, "--n-stations", str(st), "--model", model,
                            "--device", dev, "--gpu-id", "0", "--n-cpus", str(cores),
                            "--core-list", cl, "--torch-threads", "1",
                            "--in-samples", str(ins), "--overlap-samples", "0",
                            "--batch-size", "256", "--repeats", str(REPEATS),
                            "--results-root", str(root), "--resume",
                            "--method", "classify_batched", "--dtype", "fp32",
                            "--tag", f"{dev}_c{cores}_thr1",
                        ]
                        out.append({"desc": f"native classify_batched {dev} {ds} {st}st {model} c{cores}",
                                    "argv": cb})
                    for thr in flat_threads:
                        out.append({"desc": f"native annotate {ds} {st}st {model} c{cores} thr{thr}",
                                    "argv": _base(thr) + ["--method", "annotate", "--dtype", "fp32",
                                                          "--tag", f"c{cores}_thr{thr}"]})
                        for dt, cmp in precisions(model, "native", cfg):
                            argv = _base(thr) + ["--method", "slipstream", "--dtype", dt,
                                                 "--tag", f"c{cores}_thr{thr}{_pcsuffix(dt, cmp)}"]
                            if cmp:
                                argv.append("--compile")
                            out.append({"desc": f"native slipstream {ds} {st}st {model} c{cores} thr{thr} {dt}{'+c' if cmp else ''}",
                                        "argv": argv})
    return out


def orch_trials(cfg: Cfg) -> List[Trial]:
    out: List[Trial] = []
    root = RESULTS_ROOT / "orch"
    for ds in cfg.datasets:
        for st in STATIONS:
            for model in MODELS:
                ins = insamples(model)
                for cores in CORES:
                    cl = core_list(cores)
                    for dev in ("cpu", "gpu"):
                        base = [
                            PY, "benchmarks/fair/run_fair_orch_trial.py",
                            "--dataset", ds, "--n-stations", str(st), "--model", model,
                            "--device", dev, "--gpu-id", "0", "--n-cpus", str(cores),
                            "--core-list", cl, "--concurrency", str(cores),
                            "--in-samples", str(ins), "--overlap-samples", "0",
                            "--repeats", str(REPEATS), "--results-root", str(root), "--resume",
                        ]
                        strats = ["modelactor"] if cfg.drop_orch_ripper else ["ripper", "modelactor"]
                        for strat in strats:
                            out.append({"desc": f"orch {strat} {dev} {ds} {st}st {model} c{cores}",
                                        "argv": base + ["--strategy", strat, "--dtype", "fp32",
                                                        "--tag", f"{dev}_c{cores}"]})
                        for dt, cmp in precisions(model, "orch", cfg):
                            argv = base + ["--strategy", "modelactor_slipstream", "--dtype", dt,
                                           "--slipstream-batch-size", "256",
                                           "--tag", f"{dev}_c{cores}{_pcsuffix(dt, cmp)}"]
                            if cmp:
                                argv.append("--compile")
                            out.append({"desc": f"orch modelactor_slipstream {dev} {ds} {st}st {model} c{cores} {dt}{'+c' if cmp else ''}",
                                        "argv": argv})
    return out


def oversub_trials(cfg: Cfg) -> List[Trial]:
    out: List[Trial] = []
    root = RESULTS_ROOT / "oversub"
    for ds in cfg.datasets:
        for st in cfg.oversub_stations:
            for model in MODELS:
                ins = insamples(model)
                for cores in CORES:
                    cl = core_list(cores)
                    for dev in ("cpu", "gpu"):
                        for mult in MULTS:
                            conc = mult_to_int(mult, cores, lo=1, hi=st)
                            base = [
                                PY, "benchmarks/fair/run_fair_orch_trial.py",
                                "--dataset", ds, "--n-stations", str(st), "--model", model,
                                "--device", dev, "--gpu-id", "0", "--n-cpus", str(cores),
                                "--core-list", cl, "--concurrency", str(conc),
                                "--in-samples", str(ins), "--overlap-samples", "0",
                                "--repeats", str(REPEATS), "--results-root", str(root),
                                "--resume", "--dedup-vram-capped",
                            ]
                            out.append({"desc": f"oversub modelactor {dev} {ds} {st}st {model} c{cores} x{mult}->{conc}",
                                        "argv": base + ["--strategy", "modelactor", "--dtype", "fp32",
                                                        "--tag", f"{dev}_c{cores}_x{_mtag(mult)}"]})
                            for dt, cmp in precisions(model, "oversub", cfg):
                                argv = base + ["--strategy", "modelactor_slipstream", "--dtype", dt,
                                               "--slipstream-batch-size", "256",
                                               "--tag", f"{dev}_c{cores}_x{_mtag(mult)}{_pcsuffix(dt, cmp)}"]
                                if cmp:
                                    argv.append("--compile")
                                out.append({"desc": f"oversub modelactor_slipstream {dev} {ds} {st}st {model} c{cores} x{mult} {dt}{'+c' if cmp else ''}",
                                            "argv": argv})
    return out


def stream_trials(cfg: Cfg) -> List[Trial]:
    out: List[Trial] = []
    root = RESULTS_ROOT / "stream"
    for ds in cfg.datasets:
        for st in STATIONS:
            for model in MODELS:
                ins = insamples(model)
                for cores in CORES:
                    cl = core_list(cores)
                    for dev in ("cpu", "gpu"):
                        base = [
                            PY, "benchmarks/fair/run_fair_stream_trial.py",
                            "--dataset", ds, "--n-stations", str(st), "--model", model,
                            "--device", dev, "--gpu-id", "0", "--n-cpus", str(cores),
                            "--core-list", cl, "--concurrency", str(cores),
                            "--in-samples", str(ins), "--overlap-samples", "0",
                            "--repeats", str(STREAM_SESSIONS), "--n-feeds", str(STREAM_FEEDS),
                            "--feed-interval-s", "0", "--results-root", str(root), "--resume",
                        ]
                        out.append({"desc": f"stream annotate {dev} {ds} {st}st {model} c{cores}",
                                    "argv": base + ["--strategy", "stream_annotate", "--dtype", "fp32",
                                                    "--tag", f"{dev}_c{cores}"]})
                        out.append({"desc": f"stream modelactor {dev} {ds} {st}st {model} c{cores}",
                                    "argv": base + ["--strategy", "stream_modelactor", "--dtype", "fp32",
                                                    "--tag", f"{dev}_c{cores}"]})
                        for dt, cmp in precisions(model, "stream", cfg):
                            argv = base + ["--strategy", "stream_modelactor_slipstream", "--dtype", dt,
                                           "--slipstream-batch-size", "256",
                                           "--tag", f"{dev}_c{cores}{_pcsuffix(dt, cmp)}"]
                            if cmp:
                                argv.append("--compile")
                            out.append({"desc": f"stream modelactor_slipstream {dev} {ds} {st}st {model} c{cores} {dt}{'+c' if cmp else ''}",
                                        "argv": argv})
                        # Two-GPU Model-Actor (classify forward) spans BOTH physical
                        # devices; GPU-only, emitted once per (ds,st,model,cores).
                        if dev == "gpu":
                            out.append({"desc": f"stream modelactor_2gpu {ds} {st}st {model} c{cores}",
                                        "argv": base + ["--strategy", "stream_modelactor_2gpu",
                                                        "--dtype", "fp32", "--tag", f"gpu2_c{cores}"]})
    return out


FAMILIES = {"native": native_trials, "orch": orch_trials,
            "oversub": oversub_trials, "stream": stream_trials}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="all", choices=["all", *FAMILIES.keys()])
    ap.add_argument("--full", action="store_true", help="enumerate the complete ideal grid (no cost prunes)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and counts; run nothing")
    args = ap.parse_args()

    cfg = Cfg(full=args.full)
    fams = list(FAMILIES) if args.family == "all" else [args.family]
    plan: List[Tuple[str, Trial]] = [(fam, t) for fam in fams for t in FAMILIES[fam](cfg)]

    counts: Dict[str, int] = {}
    for fam, _ in plan:
        counts[fam] = counts.get(fam, 0) + 1
    scope = "FULL ideal grid" if args.full else "cost-reduced grid"
    print(f"=== {scope}: trial configs (each = {REPEATS} repeats; stream = {STREAM_SESSIONS} sessions) ===")
    for fam in fams:
        print(f"  {fam:9s}: {counts.get(fam, 0):6d}")
    print(f"  {'TOTAL':9s}: {len(plan):6d}")

    if args.dry_run:
        print("\n(dry run) first 4 commands:")
        for _, t in plan[:4]:
            print("  " + " ".join(str(x) for x in t["argv"]))
        return 0

    for i, (fam, t) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {fam}: {t['desc']}", flush=True)
        subprocess.run([str(x) for x in t["argv"]], cwd=str(RAPID_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
