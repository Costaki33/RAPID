#!/usr/bin/env python3
"""Single-screen status dashboard for the fair benchmark -- built for `watch`.

    watch -n 60 'python3 scripts/fair_dashboard.py'

Shows, in one view:
  * health line (schedulers alive, disk, both-GPU usage)
  * completion table: trials done / target per phase + strategy
  * headline results on the canonical workload (STEAD 580 st): best wall time,
    inference-stage time, peak memory (PSS), and P-wave F1 per method, on CPU
    and GPU -- so timing, memory, and pick quality are all on one screen.

Reads results/fair_benchmark/**/result.json fresh each run (~1-3 s on the full
matrix), so it's safe to re-run on a short watch interval.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "fair_benchmark"
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]

# Per-strategy targets (stable; see scripts/run_fair_scheduler.py builders).
# ripper_slipstream dropped 2026-06-16 (see run_fair_scheduler.py).
TARGETS = {
    ("matrix", "annotate"): 1536, ("matrix", "classify"): 384, ("matrix", "slipstream"): 6912,
    ("matrix", "ripper"): 384, ("matrix", "modelactor"): 384,
    ("matrix", "modelactor_slipstream"): 6912,
    ("latency", "stream_modelactor"): 128, ("latency", "stream_modelactor_slipstream"): 576,
    ("oversub", "ripper"): 192, ("oversub", "modelactor"): 192,
    ("oversub", "modelactor_slipstream"): 480,
}
PHASE_LABEL = {"matrix": "Main matrix", "latency": "Latency sweep", "oversub": "Oversub sweep"}
ORDER = (
    [("matrix", m) for m in ("annotate", "classify", "slipstream", "ripper", "modelactor",
                             "modelactor_slipstream")]
    + [("latency", m) for m in ("stream_modelactor", "stream_modelactor_slipstream")]
    + [("oversub", m) for m in ("ripper", "modelactor", "modelactor_slipstream")]
)


def canon(model):
    return "w6000" if model in ("EQTransformer", "EQT-NC") else "w6000ov03"


def bucket(path):
    return "oversub" if "/oversub/" in path else ("latency" if "/streaming/" in path else "matrix")


def _parse(p):
    try:
        r = json.loads(Path(p).read_text())
    except Exception:
        return None
    m = r.get("meta", {})
    b = bucket(p)
    skipped = bool(r.get("skipped"))
    reps = [x for x in r.get("timing", {}).get("repeats", []) if x.get("success")]
    ok = skipped or bool(reps)
    row = None
    if reps:
        t = r.get("timing", {})
        mem = r.get("memory") or {}
        pq = r.get("pick_quality_vs_catalog") or {}
        row = dict(
            bucket=b, method=m.get("method"), model=m.get("model"), ds=m.get("dataset"),
            st=m.get("n_stations"), dev=m.get("device"), ncpu=m.get("n_cpus"),
            dtype=m.get("dtype"), comp=bool(m.get("compile")), tag=m.get("tag", ""),
            total=t.get("total_s_mean"), infer=t.get("inference_s_mean"),
            pss=mem.get("peak_pss_mb_mean"), f1=pq.get("P.f1_mean"),
        )
    return (b, m.get("method"), ok, row)


CACHE = RES / ".dashboard_cache.json"


def _find_result_jsons():
    """Locate trial-level result.json files without descending into the heavy
    per-repeat ``repeats/work_*/`` subtrees (logs, picks, CSVs) -- those hold
    hundreds of thousands of files and never contain the aggregated result.json.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(RES):
        # Prune the per-repeat subtree from traversal.
        if "repeats" in dirnames:
            dirnames.remove("repeats")
        if "result.json" in filenames:
            found.append(os.path.join(dirpath, "result.json"))
    return found


def load():
    """Parse result.json files, caching by mtime so `watch` refreshes are fast.

    First run parses everything (~1 min on the full matrix); later runs only
    re-parse files whose mtime changed (newly-completed trials), so a refresh
    is typically well under a second.
    """
    paths = _find_result_jsons()
    try:
        cache = json.loads(CACHE.read_text())
    except Exception:
        cache = {}
    new_cache = {}
    rows = []
    done = defaultdict(int)
    rate = defaultdict(int)   # completions in the last hour, by (bucket, method)
    now = time.time()
    for p in paths:
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        ent = cache.get(p)
        if ent and ent.get("mt") == mt:
            b, meth, ok, row = ent["b"], ent["meth"], ent["ok"], ent["row"]
        else:
            res = _parse(p)
            if res is None:
                continue
            b, meth, ok, row = res
        new_cache[p] = {"mt": mt, "b": b, "meth": meth, "ok": ok, "row": row}
        if ok:
            done[(b, meth)] += 1
            if mt > now - 3600:   # result.json mtime ~ completion time
                rate[(b, meth)] += 1
        if row:
            rows.append(row)
    try:
        tmp = str(CACHE) + ".tmp"
        Path(tmp).write_text(json.dumps(new_cache))
        os.replace(tmp, CACHE)
    except Exception:
        pass
    return rows, done, rate


def running_now():
    """Parse `ps` for active per-repeat trial workers -> list of (gpu/cpu, strategy, model)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    active = []
    for line in out.splitlines():
        if "--repeat-index" not in line or "run_fair_" not in line:
            continue
        toks = line.split()
        def val(flag):
            return toks[toks.index(flag) + 1] if flag in toks else None
        strat = val("--strategy") or val("--method") or "?"
        model = val("--model") or "?"
        gid = val("--gpu-id")
        dev = val("--device") or ""
        slot = f"GPU{gid}" if (gid is not None and dev == "gpu") else "CPU"
        active.append((slot, strat, model))
    return sorted(active)


def best(rows, **kw):
    key = kw.pop("key", "total")
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items()) and r.get(key) is not None]
    return min(out, key=lambda r: r[key]) if out else None


def health():
    scheds = subprocess.run(["pgrep", "-fc", "run_fair_scheduler.py"], capture_output=True, text=True).stdout.strip()
    line = f"schedulers={scheds}"
    for fs in ("/", "/home"):
        try:
            u = shutil.disk_usage(fs)
            line += f"  {fs}={100*u.used//u.total}%"
        except Exception:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        gpus = len(set(x.strip() for x in out.splitlines() if x.strip()))
        line += f"  GPUs-in-use={gpus}/2"
    except Exception:
        pass
    return line


def main():
    rows, done, rate = load()
    print(f"FAIR BENCHMARK  {time.strftime('%a %Y-%m-%d %H:%M:%S')}   {health()}")

    # --- Running now (which strategy/model on each slot) ---
    act = running_now()
    if act:
        summary = "  ".join(f"{slot}:{strat}/{model}" for slot, strat, model in act)
    else:
        summary = "(no trial workers active)"
    print(f"RUNNING NOW ({len(act)}): {summary}")
    print("=" * 92)

    # --- Completion table ---
    print(f"{'Phase':13s} {'Strategy':26s} {'Done':>6s} {'Target':>7s} {'%':>5s} {'/hr':>4s}  {'progress':<18s}")
    gd = gt = grate = 0
    last = None
    for k in ORDER:
        tgt = TARGETS.get(k, 0)
        d = min(done.get(k, 0), tgt)
        rt = rate.get(k, 0)
        gd += d; gt += tgt; grate += rt
        pct = (100 * d // tgt) if tgt else 0
        bar = "#" * (pct * 18 // 100) + "." * (18 - pct * 18 // 100)
        ph = PHASE_LABEL[k[0]] if k[0] != last else ""
        last = k[0]
        rstr = str(rt) if rt else ("." if d < tgt else "")
        print(f"{ph:13s} {k[1]:26s} {d:6d} {tgt:7d} {pct:4d}% {rstr:>4s}  [{bar}]")
    pct = 100 * gd // gt if gt else 0
    pct = 100 * gd // gt if gt else 0
    print("-" * 92)
    eta = ""
    if grate > 0 and gt > gd:
        hrs = (gt - gd) / grate
        eta = f"   ~{hrs:.0f}h left at {grate}/hr" if hrs < 72 else f"   ~{hrs/24:.1f}d left at {grate}/hr"
    print(f"{'TOTAL':13s} {'':26s} {gd:6d} {gt:7d} {pct:4d}% {grate:>4d}{eta}")
    print("(/hr = trials completed in the last hour; '.' = pending, none finished this hour -- "
          "queued behind another strategy, not stuck)")
    print()

    # --- Headline results: STEAD 580 st, per method, CPU + GPU ---
    print("HEADLINE RESULTS  (STEAD, 580 stations, canonical regime; best config shown)")
    print(f"{'method':22s} {'dev':4s} {'total s':>8s} {'infer s':>8s} {'peak GB':>8s} {'P-F1':>6s}")
    METHS = [
        ("annotate", dict(method="annotate")),
        ("classify", dict(method="classify")),
        ("slipstream bf16", dict(method="slipstream", dtype="bf16")),
        ("ripper", dict(method="ripper")),
        ("modelactor", dict(method="modelactor")),
        ("MA+slip bf16", dict(method="modelactor_slipstream", dtype="bf16")),
    ]
    for dev in ("cpu", "gpu"):
        for label, flt in METHS:
            # pick the model set matching this method; average headline over models
            cells = []
            tot = inf = pss = f1 = n = 0
            for model in MODELS:
                cand = [r for r in rows if r["bucket"] == "matrix" and r["ds"] == "stead"
                        and r["st"] == 580 and r["dev"] == dev and not r["comp"]
                        and (canon(model) in r["tag"] or model in ("EQTransformer", "EQT-NC"))
                        and all(r.get(k) == v for k, v in flt.items())
                        and (r["ncpu"] == 20 if dev == "cpu" else True)]
                b = min(cand, key=lambda r: r["total"]) if cand else None
                if b:
                    n += 1
                    tot += b["total"] or 0
                    inf += b["infer"] or 0
                    pss += (b["pss"] or 0) / 1000.0
                    f1 += b["f1"] or 0
            if n:
                print(f"{label:22s} {dev:4s} {tot/n:8.1f} {inf/n:8.2f} {pss/n:8.1f} {f1/n:6.3f}  (n={n})")
            else:
                print(f"{label:22s} {dev:4s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>6s}  (re-running)")
        print()
    print("total/infer/PSS/F1 are MEANS over the 4 models (best config each). 'infer' = forward-")
    print("stage only (slipstream's real speedup); 'total' = full cold start (fixed load dominates).")


if __name__ == "__main__":
    main()
