"""Autonomously run the SeisBench dtype matrix configs across a CPU-affinity sweep.

Background
----------
The existing matrix driver only honours ``cfg.cpus`` for the dual-GPU pipelined
runner. Every other row (CPU-only, single-GPU, dual-GPU baseline, dual-GPU
``lean_pytorch_dual_serial``) runs unbound, which on this 128-thread host means
the BLAS / OpenMP layer is free to use all 128 logical CPUs even when we want
to characterise the model under a fixed core budget.

This orchestrator wraps each config in an outer ``taskset`` so the entire
matrix subprocess (and every kernel it launches) is constrained to a fixed
affinity mask. It also forces the BLAS / OpenMP environment to the same core
count so the math libraries do not silently spawn extra threads. Outputs are
tagged with the affinity, e.g. ``seisbench_matrix_lean_cpu_aff12.jsonl``, so
runs at different core budgets do not collide.

The SeisBench matrix is launched via ``scripts/run_seisbench_matrix.py``
(which consumes ``SeisBenchMatrixConfig``); the generic ``scripts/run_matrix.py``
is for the miniSEED network sweep and does not understand SeisBench config
fields like ``seisbench_datasets``.

For each (config, n_cpus) pair the script:
  1. Loads the source JSON config.
  2. Rewrites it in memory with:
       - ``output_jsonl``: ``..._aff{N}.jsonl``
       - ``cpus``: ``[N]``  (single element so the dual pipelined runner pins
         to the same N as the outer mask, instead of sweeping a different N)
       - ``resume_include_jsonl``: ``[]`` (we want fresh affinity-bound data,
         not to inherit "completed" cells from the unbound runs)
       - ``resume``: ``true``  (so partial completions can recover)
       - ``_affinity_n_cpus``: tag for traceability
  3. Writes the modified config to ``RAPID/configs/_affinity/<stem>_aff{N}.json``.
  4. Launches ``scripts/run_seisbench_matrix.py --config <tmp>`` via
     ``taskset -c 0-{N-1}`` with ``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = N``.
  5. Streams stdout / stderr to ``results/logs/<stem>_aff{N}.log``.
  6. Continues to the next pair on success; on failure, logs the return code
     and either stops (default) or keeps going (``--keep-going``).

Usage
-----
Run the full default sweep (CPU + 1-GPU + 2-GPU configs at 12 / 16 / 20 cores)::

    cd RAPID
    python scripts/run_all_affinity.py

Restrict to a subset::

    python scripts/run_all_affinity.py \
        --configs configs/seisbench_dtype_matrix_cpu_only.json \
        --cpus 12 16

Dry-run (prints commands, writes temp configs, runs nothing)::

    python scripts/run_all_affinity.py --dry-run

Resume after an interruption (each subprocess already supports resume; the
orchestrator just relaunches the same pair, which picks up where it stopped)::

    python scripts/run_all_affinity.py
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
_RAPID_ROOT = _HERE.parent

DEFAULT_CONFIGS = [
    "configs/seisbench_dtype_matrix_cpu_only.json",
    "configs/seisbench_dtype_matrix.json",
]
DEFAULT_CPUS = [12, 16, 20]


def _resolve_under_rapid(p: str) -> Path:
    """Return ``p`` resolved either as absolute, CWD-relative, or RAPID-relative."""
    pth = Path(p)
    if pth.is_absolute():
        return pth
    cwd_candidate = Path.cwd() / pth
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    rapid_candidate = _RAPID_ROOT / pth
    return rapid_candidate.resolve()


def _affinity_output_path(orig: str, n_cpus: int) -> str:
    p = Path(orig)
    return str(p.with_name(f"{p.stem}_aff{n_cpus}{p.suffix}"))


def _build_temp_config(
    src_cfg_path: Path,
    n_cpus: int,
    tmp_dir: Path,
) -> Tuple[Path, str]:
    """Materialise the affinity-flavoured config and return (path, output_jsonl)."""
    cfg = json.loads(src_cfg_path.read_text())
    src_output = cfg.get("output_jsonl") or f"results/{src_cfg_path.stem}.jsonl"
    new_output = _affinity_output_path(src_output, n_cpus)

    cfg["output_jsonl"] = new_output
    # Fresh affinity-bound run: do NOT inherit completed-cell keys from the
    # earlier unbound JSONLs. Internal resume on the new path is still on, so
    # an interrupted run can recover from its own partial output.
    cfg["resume_include_jsonl"] = []
    cfg["resume"] = True
    # Single-element so the dual-GPU pipelined runner re-pins to the same N as
    # our outer taskset mask (no nested / conflicting affinity sweep).
    cfg["cpus"] = [int(n_cpus)]
    cfg["_affinity_n_cpus"] = int(n_cpus)
    cfg["_affinity_source_config"] = str(src_cfg_path)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{src_cfg_path.stem}_aff{n_cpus}.json"
    tmp_path.write_text(json.dumps(cfg, indent=2))
    return tmp_path, new_output


def _run_one(
    src_cfg_path: Path,
    n_cpus: int,
    *,
    repo_root: Path,
    run_matrix_script: Path,
    log_dir: Path,
    tmp_dir: Path,
    dry_run: bool,
    offset: int,
) -> int:
    if shutil.which("taskset") is None:
        print("[orchestrator] ERROR: taskset is not on PATH; cannot pin CPUs.")
        return 127

    tmp_cfg_path, out_jsonl = _build_temp_config(src_cfg_path, n_cpus, tmp_dir)
    cpu_list = f"{offset}-{offset + n_cpus - 1}"

    cmd: List[str] = [
        "taskset",
        "-c",
        cpu_list,
        sys.executable,
        str(run_matrix_script),
        "--config",
        str(tmp_cfg_path),
    ]

    env = os.environ.copy()
    # Force every BLAS / threading layer to honour the affinity budget. Without
    # this, OpenMP / MKL would still spin up 128 threads onto the 12 cores we
    # pinned, which both distorts timing and adds context-switch overhead.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[var] = str(n_cpus)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{src_cfg_path.stem}_aff{n_cpus}.log"

    rendered = " ".join(shlex.quote(c) for c in cmd)
    print(
        f"[orchestrator] {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"LAUNCH {src_cfg_path.name} aff={n_cpus} "
        f"-> {Path(out_jsonl).name} (log: {log_path.name})"
    )
    print(f"[orchestrator]   cmd: {rendered}")

    if dry_run:
        print("[orchestrator]   DRY RUN: not executing.")
        return 0

    t0 = time.time()
    with open(log_path, "ab", buffering=0) as logf:
        header = (
            f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} START aff={n_cpus} "
            f"src={src_cfg_path.name} cwd={repo_root} ===\n"
            f"cmd: {rendered}\n"
            f"env: OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS="
            f"NUMEXPR_NUM_THREADS={n_cpus}\n"
            f"output_jsonl: {out_jsonl}\n"
        ).encode()
        logf.write(header)
        try:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(repo_root),
                check=False,
            )
            rc = int(proc.returncode)
        except KeyboardInterrupt:
            logf.write(b"\n=== INTERRUPTED BY USER (KeyboardInterrupt) ===\n")
            raise
        finally:
            dt = time.time() - t0
            footer = (
                f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} END "
                f"elapsed={dt:.1f}s ===\n"
            ).encode()
            logf.write(footer)

    print(
        f"[orchestrator] {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"END    {src_cfg_path.name} aff={n_cpus} rc={rc} elapsed={dt:.1f}s"
    )
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run SeisBench matrix configs across a CPU-affinity sweep. "
            "Each (config, n_cpus) pair runs taskset -c 0-{N-1} python "
            "scripts/run_matrix.py --config <flavoured tmp config>."
        ),
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help=(
            "Source matrix config files (paths relative to RAPID/ or absolute). "
            f"Default: {DEFAULT_CONFIGS}"
        ),
    )
    ap.add_argument(
        "--cpu-offset",
        type=int,
        default=0,
        help="Offset to apply to the CPU list. Default: 0",
    )
    ap.add_argument(
        "--cpus",
        nargs="+",
        type=int,
        default=DEFAULT_CPUS,
        help=f"CPU-affinity values to sweep. Default: {DEFAULT_CPUS}",
    )
    ap.add_argument(
        "--repo-root",
        default=str(_RAPID_ROOT),
        help="Working directory used for the subprocess. Default: RAPID/",
    )
    ap.add_argument(
        "--run-matrix-script",
        default=str(_RAPID_ROOT / "scripts" / "run_seisbench_matrix.py"),
        help=(
            "Path to the SeisBench matrix entry point. Default: "
            "scripts/run_seisbench_matrix.py (uses SeisBenchMatrixConfig). "
            "scripts/run_matrix.py uses MatrixConfig and will reject SeisBench "
            "configs containing 'seisbench_datasets' etc."
        ),
    )
    ap.add_argument(
        "--log-dir",
        default=None,
        help="Directory for per-pair log files. Default: <repo-root>/results/logs/affinity",
    )
    ap.add_argument(
        "--tmp-config-dir",
        default=None,
        help="Directory for materialised affinity configs. Default: <repo-root>/configs/_affinity",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write temp configs but do not execute.",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining (config, n_cpus) pairs after a failure.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_matrix_script = Path(args.run_matrix_script).resolve()
    log_dir = (
        Path(args.log_dir).resolve()
        if args.log_dir
        else (repo_root / "results" / "logs" / "affinity").resolve()
    )
    tmp_dir = (
        Path(args.tmp_config_dir).resolve()
        if args.tmp_config_dir
        else (repo_root / "configs" / "_affinity").resolve()
    )

    if not run_matrix_script.exists():
        print(f"[orchestrator] ERROR: run_matrix.py not found at {run_matrix_script}")
        return 2

    offset = args.cpu_offset
    src_cfg_paths: List[Path] = []
    for c in args.configs:
        p = _resolve_under_rapid(c)
        if not p.exists():
            print(f"[orchestrator] ERROR: config not found: {c} (resolved {p})")
            return 2
        src_cfg_paths.append(p)

    pairs = [(p, n) for p in src_cfg_paths for n in args.cpus]

    print("[orchestrator] Plan:")
    for cfg_p, n in pairs:
        print(f"  - {cfg_p.relative_to(repo_root) if str(cfg_p).startswith(str(repo_root)) else cfg_p}  @ {n} CPUs")
    print(f"[orchestrator] log_dir   = {log_dir}")
    print(f"[orchestrator] tmp_dir   = {tmp_dir}")
    print(f"[orchestrator] repo_root = {repo_root}")
    print()

    failures: List[Tuple[Path, int, int]] = []
    t_start = time.time()
    for cfg_p, n in pairs:
        try:
            rc = _run_one(
                cfg_p,
                n,
                repo_root=repo_root,
                run_matrix_script=run_matrix_script,
                log_dir=log_dir,
                tmp_dir=tmp_dir,
                dry_run=args.dry_run,
                offset=offset,
            )
        except KeyboardInterrupt:
            print("\n[orchestrator] Interrupted by user. Stopping.")
            return 130
        if rc != 0:
            print(f"[orchestrator] FAILURE rc={rc} for {cfg_p.name} aff={n}")
            failures.append((cfg_p, n, rc))
            if not args.keep_going:
                print(
                    "[orchestrator] Stopping (use --keep-going to continue past failures)."
                )
                break

    total_dt = time.time() - t_start
    print()
    print(f"[orchestrator] Done. elapsed={total_dt:.1f}s pairs={len(pairs)} failures={len(failures)}")
    for cfg_p, n, rc in failures:
        print(f"  FAILED: {cfg_p.name} aff={n} rc={rc}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
