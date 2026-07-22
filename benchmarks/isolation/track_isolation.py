#!/usr/bin/env python3
"""Progress tracker for the isolated re-measurement run.

Run it anytime to see how far along the clean (isolated) trials are:

    python3 benchmarks/isolation/track_isolation.py          # one snapshot
    watch -n 30 python3 benchmarks/isolation/track_isolation.py   # live, refresh every 30s

It reads the result.json files under results/fair_benchmark_iso/, buckets them
into the five phases, and shows done / target per phase plus what's running now.
"""
from __future__ import annotations
import glob, json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISO = ROOT / "results" / "fair_benchmark_iso"


def metas(subdir):
    out = []
    for p in glob.glob(str(ISO / subdir / "**" / "result.json"), recursive=True):
        try:
            r = json.loads(Path(p).read_text())
            out.append((r.get("meta", {}), r))
        except Exception:
            pass
    return out


def main():
    h2h = metas("h2h")
    # bucket h2h by tag prefix
    p1 = sum(1 for m, _ in h2h if m.get("method") in ("stream_annotate", "stream_modelactor")
             and str(m.get("tag", "")).startswith(("iso_cpu_", "iso_gpu_")))
    p2_2g = sum(1 for m, _ in h2h if str(m.get("tag", "")).startswith("iso_2gpu_"))
    p2_1g = sum(1 for m, _ in h2h if str(m.get("tag", "")).startswith("iso_1gpu_"))
    native = metas("native")
    orch = metas("orch")
    over = metas("oversub")
    over_skip = sum(1 for _, r in over if r.get("skipped"))

    rows = [
        ("1  Head-to-head (annotate vs Model-Actor, CPU+GPU, 580&250)", p1, 32),
        ("2  GPU sweep  2-GPU split across cpu{5,10,15,20}", p2_2g, 32),
        ("2  GPU sweep  1-GPU companion across cpu{5,10,15}", p2_1g, 24),
        ("3  Native thread sweep (annotate/classify/slipstream)", len(native), 84),
        ("4  Orchestration cold-start (Model-Actor, Ripper)", len(orch), 16),
        ("5  Oversubscription (actors-per-core sweep)", len(over), 480),
    ]
    done = sum(r[1] for r in rows)
    target = sum(r[2] for r in rows)

    # ---- LIVE ISOLATION CHECK ----
    # Two independent signals, both robust to the microsecond fork->exec window
    # where a freshly-spawned repeat-worker momentarily still carries the driver's
    # argv (before exec swaps in --repeat-index), which can transiently look like
    # a 2nd "driver". We (a) count active isolation RUNNER scripts (stable; the
    # real structural guarantee = 1), and (b) double-sample the worker count and
    # take the SUSTAINED value, so a fork-exec blip can't raise a false alarm.
    def count(cmd):
        return int(subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip() or 0)
    RUNNERS = ("ps -eo args | grep -E 'run_isolation(_oversub)?\\.sh' | grep -v grep "
               "| grep -v 'kill -0' | grep -c bash")
    WORKERS = "ps -eo args | grep -- '--repeat-index' | grep -v grep | wc -l"
    n_runner = count(RUNNERS)
    w1 = count(WORKERS); time.sleep(0.4); w2 = count(WORKERS)
    nd = min(w1, w2)                      # sustained computing-worker count
    if n_runner > 1:
        iso = f"** WARNING: {n_runner} ISOLATION RUNNERS ACTIVE — STOP ALL BUT ONE **"
    elif nd > 1:
        iso = f"** WARNING: {nd} WORKERS COMPUTING AT ONCE — ISOLATION BROKEN **"
    elif nd == 1:
        iso = "OK  (exactly 1 trial computing — isolated)"
    else:
        iso = "IDLE (between trials / finished)"

    bar = lambda d, t: ("#" * int(20 * d / t)).ljust(20) if t else " " * 20
    print("=" * 74)
    print("  ISOLATED RE-MEASUREMENT  (strictly one trial at a time)")
    print(f"  ISOLATION: {iso}   [runners={n_runner}, computing workers={nd}]")
    print("=" * 74)
    print("  (bars below = cumulative COMPLETED cells per category, NOT concurrent runs)")
    for label, d, t in rows:
        print(f"  [{bar(d,t)}] {d:4d}/{t:<4d}  {label}")
    if over_skip:
        print(f"  (oversub: {over_skip} cells correctly skipped as VRAM/RAM-cap-redundant)")
    print("-" * 74)
    print(f"  TOTAL  {done}/{target}  ({100*done//max(target,1)}%)")

    # what's running + elapsed
    cur = subprocess.run(["bash", "-c",
        "ps -eo args | grep -E 'run_fair_(stream_trial|trial|orch_trial)' | grep -v grep | head -1"],
        capture_output=True, text=True).stdout
    def opt(flag):
        if flag in cur:
            return cur.split(flag, 1)[1].split()[0]
        return "?"
    alive = subprocess.run(["bash", "-c", "ps -eo args | grep run_isolation.sh | grep -v grep | grep -c bash"],
                           capture_output=True, text=True).stdout.strip()
    print("-" * 74)
    if alive != "0" and cur.strip():
        meth = opt("--strategy") if "--strategy" in cur else opt("--method")
        print(f"  RUNNING NOW: {meth}  model={opt('--model')}  dev={opt('--device')}  "
              f"stations={opt('--n-stations')}")
    elif alive != "0":
        print("  runner alive (between trials)")
    else:
        oc = subprocess.run(["bash", "-c", "ps -eo args | grep run_isolation_oversub.sh | grep -v grep | grep -c bash"],
                            capture_output=True, text=True).stdout.strip()
        print("  main runner NOT active" + ("  (oversub phase running)" if oc != "0" else "  (finished or stopped)"))
    print("=" * 74)


if __name__ == "__main__":
    main()
