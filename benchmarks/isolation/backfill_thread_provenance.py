#!/usr/bin/env python3
"""Backfill per-worker compute-thread provenance into existing result JSONs.

The original benchmark coupled the core budget to the thread count: single-process
methods ran ``pin_threads(n_cpus)`` (so torch intra-op threads == n_cpus), while
actor-based methods run ONE compute thread per actor (parallelism comes from N
processes, not threads). That distinction was never recorded. This stamps each
existing result with:

* ``threads_per_worker`` -- compute threads in each inference process
* ``n_compute_workers``  -- number of such processes (1 for single-process,
  ``concurrency`` for actor pools)
* ``torch_threads``      -- alias for threads_per_worker (sweep axis)

so the data is self-describing and the thread sweep can skip configs already
covered. Idempotent: never overwrites a field that is already present.

    python3 benchmarks/isolation/backfill_thread_provenance.py [--apply]   (default: dry-run)
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRS = ["results/fair_benchmark", "results/fair_benchmark_h2h",
        "results/fair_benchmark_h2h_v2", "results/fair_benchmark_h2h_2gpu",
        "results/fair_benchmark_threadsweep"]
# methods whose compute is one process using n_cpus threads
SINGLE_PROC = {"annotate", "classify", "slipstream", "stream_annotate"}
# actor-pool methods: 1 thread per actor, concurrency actors
ACTOR = {"modelactor", "ripper", "modelactor_slipstream",
         "stream_modelactor", "stream_modelactor_slipstream", "stream_modelactor_2gpu"}


def main():
    apply = "--apply" in sys.argv
    n_files = n_stamped = 0
    by_kind = {}
    for d in DIRS:
        for p in glob.glob(str(ROOT / d / "**" / "result.json"), recursive=True):
            try:
                r = json.loads(Path(p).read_text()); m = r.get("meta") or {}
            except Exception:
                continue
            n_files += 1
            meth = m.get("method"); n_cpus = m.get("n_cpus")
            if meth in SINGLE_PROC:
                tpw, nw = n_cpus, 1
            elif meth in ACTOR:
                tpw, nw = 1, m.get("concurrency")
            else:
                continue
            changed = False
            for key, val in (("threads_per_worker", tpw), ("n_compute_workers", nw),
                             ("torch_threads", tpw)):
                if key not in m and val is not None:
                    m[key] = val; changed = True
            if changed:
                n_stamped += 1
                by_kind[meth] = by_kind.get(meth, 0) + 1
                if apply:
                    r["meta"] = m
                    Path(p).write_text(json.dumps(r, default=str))
    mode = "APPLIED" if apply else "DRY-RUN (use --apply to write)"
    print(f"{mode}: scanned {n_files} result.json, stamped {n_stamped}")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
