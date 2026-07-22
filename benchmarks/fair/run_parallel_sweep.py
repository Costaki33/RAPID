#!/usr/bin/env python3
"""Parallel orchestrator for the STEAD/TXED timing sweep.

Runs ``run_seisbench_sweep.py`` strategies concurrently so the full grid finishes
faster, by giving each strategy a **disjoint CPU-core range** (and, on GPU, a
dedicated GPU). Networks are prepared once up front; every worker is launched with
``--skip-network-prep`` and its own Ray temp dir.

Phases
------
``cpu``   - one subprocess per strategy, all at once, pinned to disjoint cores:
              ripper                -> cores [0 .. C-1]
              modelactor            -> cores [C .. 2C-1]
              modelactor_slipstream -> cores [2C .. 3C-1]            (C = --max-cpus)

``gpu``   - 1 GPU per strategy, rotated across the available GPUs (default 0,1) so
            up to ``len(--gpus)`` strategies run at once. Each GPU slot also gets a
            disjoint CPU range for its preprocessing workers. (single-GPU timing)

``dual-gpu`` - each strategy uses BOTH GPUs (``--gpu-ids 0,1 --min-gpus 2``), run
            sequentially (both GPUs busy per job). Skips re-running 1-GPU points.
            Use this AFTER the single-GPU ``gpu`` phase.

Examples
--------
Prepare once, then CPU + single-GPU phases::

    python benchmarks/fair/run_parallel_sweep.py --phase cpu+gpu

Just the CPU phase, finer 10%% march::

    python benchmarks/fair/run_parallel_sweep.py --phase cpu --conc-march-frac 0.1

Dual-GPU phase later::

    python benchmarks/fair/run_parallel_sweep.py --phase dual-gpu
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_RAPID_ROOT = _HERE.parent
SWEEP_SCRIPT = _HERE / "run_seisbench_sweep.py"

# Stable ordering -> deterministic CPU-range assignment.
STRATEGIES = ["ripper", "modelactor", "modelactor_slipstream"]
STRATEGY_SHORT = {
    "ripper": "rip",
    "modelactor": "ma",
    "modelactor_slipstream": "mas",
}


def _job_env(n_cpus: int, ram_budget_mb: Optional[float]) -> Dict[str, str]:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = str(max(1, n_cpus))
    # Hand this worker its own RAM slice so concurrent strategies can't collectively
    # oversubscribe system RAM. EvaluateSystem/parallelization read this absolute budget.
    if ram_budget_mb is not None and ram_budget_mb > 0:
        env["EQCCTPRO_RAM_BUDGET_MB"] = f"{ram_budget_mb:.0f}"
    return env


def _tmp_dir_for(tag: str) -> str:
    # Keep short: Ray's AF_UNIX socket path limit is 107 bytes.
    return f"/tmp/eqr_{tag}"


def _base_passthrough(args) -> List[str]:
    extra: List[str] = [
        "--max-cpus", str(args.max_cpus),
        "--conc-march-frac", str(args.conc_march_frac),
    ]
    if args.conc_values:
        extra += ["--conc-values", args.conc_values]
    extra += ["--exact-resume"] if args.exact_resume else ["--no-exact-resume"]
    if args.overwrite:
        extra += ["--overwrite"]
    if args.dry_run:
        extra += ["--dry-run"]
    return extra


def _build_cmd(
    args,
    *,
    strategy: str,
    cpu_offset: int,
    device: str,           # "cpu" | "gpu" | "dual-gpu"
    gpu_ids: Optional[List[int]],
    min_gpus: int,
    tag: str,
) -> List[str]:
    cmd: List[str] = []
    if args.use_taskset and shutil.which("taskset"):
        cmd += ["taskset", "-c", f"{cpu_offset}-{cpu_offset + args.max_cpus - 1}"]
    cmd += [
        sys.executable, str(SWEEP_SCRIPT),
        "--run-all",
        "--phase", "timing",
        "--only-strategies", strategy,
        "--skip-network-prep",
        "--cpu-offset", str(cpu_offset),
        "--tmp-dir", _tmp_dir_for(tag),
    ]
    if device == "cpu":
        cmd += ["--sweep-cpu-only", "--no-sweep-with-gpu"]
    else:
        cmd += ["--no-sweep-cpu-only", "--sweep-with-gpu",
                "--gpu-ids", ",".join(str(g) for g in (gpu_ids or [])),
                "--min-gpus", str(min_gpus)]
    cmd += _base_passthrough(args)
    return cmd


class _Job:
    def __init__(self, tag: str, cmd: List[str], log_path: Path, env: Dict[str, str],
                 gpu_ids: Optional[List[int]], cpu_offset: int):
        self.tag = tag
        self.cmd = cmd
        self.log_path = log_path
        self.env = env
        self.gpu_ids = gpu_ids
        self.cpu_offset = cpu_offset
        self.proc: Optional[subprocess.Popen] = None
        self.log_fh = None
        self.t0 = 0.0

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_fh = self.log_path.open("w", encoding="utf-8")
        self.t0 = time.time()
        print(f"[orchestrator] START {self.tag}: {shlex.join(self.cmd)}")
        print(f"[orchestrator]   log -> {self.log_path}")
        self.proc = subprocess.Popen(
            self.cmd, stdout=self.log_fh, stderr=subprocess.STDOUT,
            env=self.env, cwd=str(_RAPID_ROOT),
        )

    def poll(self) -> Optional[int]:
        if self.proc is None:
            return None
        rc = self.proc.poll()
        if rc is not None and self.log_fh is not None:
            self.log_fh.flush()
            self.log_fh.close()
            self.log_fh = None
            dur = time.time() - self.t0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[orchestrator] DONE  {self.tag}: {status} in {dur:.1f}s")
        return rc


def _prepare_networks(args) -> int:
    if args.skip_network_prep:
        print("[orchestrator] Skipping network prep (--skip-network-prep).")
        return 0
    cmd = [sys.executable, str(SWEEP_SCRIPT), "--prepare-networks-only"]
    if args.dry_run:
        cmd += ["--dry-run"]
    print(f"[orchestrator] PREPARE networks: {shlex.join(cmd)}")
    return subprocess.run(cmd, cwd=str(_RAPID_ROOT)).returncode


def _run_cpu_phase(args, strategies: List[str]) -> int:
    """Run strategies in waves of --max-parallel, each pinned to its own core range
    and given an equal slice of the usable RAM budget so a wave can't OOM the box.

    --max-parallel 1 => fully sequential, and each strategy gets the FULL budget
    (so it can spin up max actors). >1 => that many strategies run at once, each with
    budget/wave_size.
    """
    failures = 0
    wave_size = max(1, min(args.max_parallel, len(strategies)))
    for w in range(0, len(strategies), wave_size):
        wave = strategies[w:w + wave_size]
        per_job_ram = args.usable_ram_mb / len(wave)
        print(
            f"[orchestrator] CPU wave {w // wave_size + 1}: {wave} "
            f"(RAM/job ~{per_job_ram / 1024:.0f} GB of {args.usable_ram_mb / 1024:.0f} GB usable)"
        )
        jobs: List[_Job] = []
        for idx, strat in enumerate(wave):
            offset = idx * args.max_cpus
            tag = f"cpu_{STRATEGY_SHORT[strat]}"
            cmd = _build_cmd(args, strategy=strat, cpu_offset=offset,
                             device="cpu", gpu_ids=None, min_gpus=1, tag=tag)
            jobs.append(_Job(tag, cmd, args.log_dir / f"{tag}.log",
                             _job_env(args.max_cpus, per_job_ram), None, offset))
        failures |= _run_jobs_concurrent(jobs, args)
    return failures


def _run_gpu_phase(args, strategies: List[str], *, dual: bool) -> int:
    """Single-GPU: 1 GPU per strategy, rotated across --gpus (concurrency = #GPUs).

    Dual-GPU: each strategy uses ALL GPUs; jobs run one-at-a-time.
    """
    gpus = args.gpus
    if dual:
        # One slot; the single job uses every GPU. Disjoint CPU range not needed.
        slots: List[Tuple[List[int], int]] = [(list(gpus), 0)]
        min_gpus = 2
        device = "dual-gpu"
    else:
        # One slot per GPU; each slot gets a dedicated GPU + CPU range.
        slots = [([g], i * args.max_cpus) for i, g in enumerate(gpus)]
        min_gpus = 1
        device = "gpu"

    if args.dry_run:
        # Slots never free in dry-run; show the planned assignment round-robin.
        for i, strat in enumerate(strategies):
            gpu_ids, offset = slots[i % len(slots)]
            gtag = "".join(str(g) for g in gpu_ids)
            tag = f"{'dual' if dual else 'gpu'}{gtag}_{STRATEGY_SHORT[strat]}"
            cmd = _build_cmd(args, strategy=strat, cpu_offset=offset,
                             device=device, gpu_ids=gpu_ids, min_gpus=min_gpus, tag=tag)
            print(f"[orchestrator] DRY-RUN {tag}: {shlex.join(cmd)}")
        return 0

    # At most len(slots) jobs run at once, so each gets an equal RAM slice. The CPU side
    # of GPU trials (preprocessing) still uses RAM, so this guards against OOM too.
    per_job_ram = args.usable_ram_mb / max(1, len(slots))
    pending: List[str] = list(strategies)
    free_slots: List[Tuple[List[int], int]] = list(slots)
    running: List[Tuple[_Job, Tuple[List[int], int]]] = []
    failures = 0

    while pending or running:
        while pending and free_slots:
            slot = free_slots.pop(0)
            gpu_ids, offset = slot
            strat = pending.pop(0)
            gtag = "".join(str(g) for g in gpu_ids)
            tag = f"{'dual' if dual else 'gpu'}{gtag}_{STRATEGY_SHORT[strat]}"
            cmd = _build_cmd(args, strategy=strat, cpu_offset=offset,
                             device=device, gpu_ids=gpu_ids, min_gpus=min_gpus, tag=tag)
            job = _Job(tag, cmd, args.log_dir / f"{tag}.log",
                       _job_env(args.max_cpus, per_job_ram), gpu_ids, offset)
            job.start()
            running.append((job, slot))
        time.sleep(2.0)
        still: List[Tuple[_Job, Tuple[List[int], int]]] = []
        for job, slot in running:
            rc = job.poll()
            if rc is None:
                still.append((job, slot))
            else:
                if rc != 0:
                    failures += 1
                free_slots.append(slot)
        running = still
    return 1 if failures else 0


def _run_jobs_concurrent(jobs: List[_Job], args) -> int:
    if args.dry_run:
        for j in jobs:
            print(f"[orchestrator] DRY-RUN {j.tag}: {shlex.join(j.cmd)}")
        return 0
    for j in jobs:
        j.start()
    failures = 0
    remaining = list(jobs)
    while remaining:
        time.sleep(2.0)
        still = []
        for j in remaining:
            rc = j.poll()
            if rc is None:
                still.append(j)
            elif rc != 0:
                failures += 1
        remaining = still
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--phase",
        choices=["cpu", "gpu", "dual-gpu", "cpu+gpu", "all"],
        default="cpu+gpu",
        help="cpu = parallel strategies on disjoint cores; gpu = 1 GPU/strategy rotated; "
        "dual-gpu = both GPUs per strategy (sequential); cpu+gpu = cpu then gpu; all = +dual-gpu",
    )
    ap.add_argument("--max-cpus", type=int, default=20, help="Cores per strategy (range width)")
    ap.add_argument("--gpus", default="0,1", help="Available GPU ids (comma-separated)")
    ap.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="Strategies to run (comma-separated subset of: " + ", ".join(STRATEGIES) + ")",
    )
    ap.add_argument("--conc-march-frac", type=float, default=0.2)
    ap.add_argument("--conc-values", default="")
    ap.add_argument("--exact-resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Max strategies running at once (CPU phase). 0 = all selected strategies. "
        "Use 1 for fully sequential (each strategy gets the FULL RAM budget). The RAM "
        "budget is split evenly across the strategies in a wave so they can't OOM.",
    )
    ap.add_argument(
        "--ram-budget-gb",
        type=float,
        default=0.0,
        help="Total RAM (GB) the orchestrator may plan against. 0 = autodetect installed RAM.",
    )
    ap.add_argument(
        "--ram-reserve-gb",
        type=float,
        default=40.0,
        help="RAM (GB) held back for the OS/driver/page cache before slicing per strategy.",
    )
    ap.add_argument("--use-taskset", action=argparse.BooleanOptionalAction, default=True,
                    help="Wrap each worker in taskset to hard-pin its core range (default on)")
    ap.add_argument("--log-dir", type=Path, default=_RAPID_ROOT / "results" / "logs" / "parallel")
    ap.add_argument("--skip-network-prep", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.gpus = [int(x) for x in str(args.gpus).split(",") if x.strip()]
    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    bad = [s for s in strategies if s not in STRATEGIES]
    if bad:
        ap.error(f"--strategies has invalid value(s) {bad}; choose from {STRATEGIES}")

    if not args.max_parallel or args.max_parallel < 1:
        args.max_parallel = len(strategies)

    # Compute the usable RAM budget that gets sliced across concurrent strategies.
    if args.ram_budget_gb and args.ram_budget_gb > 0:
        total_ram_mb = args.ram_budget_gb * 1024.0
    else:
        try:
            import psutil
            total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
        except Exception:
            total_ram_mb = 500.0 * 1024.0  # conservative fallback
    args.usable_ram_mb = max(1024.0, total_ram_mb - args.ram_reserve_gb * 1024.0)
    print(
        f"[orchestrator] RAM: total~{total_ram_mb/1024:.0f} GB, reserve {args.ram_reserve_gb:.0f} GB, "
        f"usable {args.usable_ram_mb/1024:.0f} GB; max_parallel={args.max_parallel} "
        f"=> ~{args.usable_ram_mb/1024/max(1,min(args.max_parallel,len(strategies))):.0f} GB/strategy in a wave"
    )

    wave_size = max(1, min(args.max_parallel, len(strategies)))
    n_cores_needed = wave_size * args.max_cpus
    avail = len(os.sched_getaffinity(0))
    print(
        f"[orchestrator] phase={args.phase} strategies={strategies} "
        f"max_cpus={args.max_cpus} gpus={args.gpus}\n"
        f"[orchestrator] CPU phase runs {wave_size} at once, needs {n_cores_needed} cores "
        f"(have {avail} available); ranges within a wave: " + ", ".join(
            f"slot{i} {i*args.max_cpus}-{(i+1)*args.max_cpus-1}"
            for i in range(wave_size)
        )
    )
    if args.phase in ("cpu", "cpu+gpu", "all") and n_cores_needed > avail:
        print(
            f"[orchestrator] WARNING: requested {n_cores_needed} cores but only {avail} "
            f"available — strategies will oversubscribe. Lower --max-cpus or --strategies."
        )

    rc = _prepare_networks(args)
    if rc != 0:
        print(f"[orchestrator] Network prep failed (rc={rc}); aborting.")
        return rc

    overall = 0
    if args.phase in ("cpu", "cpu+gpu", "all"):
        print("\n[orchestrator] ===== CPU PHASE =====")
        overall |= _run_cpu_phase(args, strategies)
    if args.phase in ("gpu", "cpu+gpu", "all"):
        print("\n[orchestrator] ===== SINGLE-GPU PHASE (1 GPU/strategy, rotated) =====")
        overall |= _run_gpu_phase(args, strategies, dual=False)
    if args.phase in ("dual-gpu", "all"):
        print("\n[orchestrator] ===== DUAL-GPU PHASE (both GPUs/strategy, sequential) =====")
        overall |= _run_gpu_phase(args, strategies, dual=True)

    print(f"\n[orchestrator] complete (exit={overall}).")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
