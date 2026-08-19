#!/usr/bin/env bash
# Live progress for the annotate-precision isolation matrix.
#
# Usage:
#   bash benchmarks/isolation/watch_annotate_precision_progress.sh
#   bash benchmarks/isolation/watch_annotate_precision_progress.sh --once
#   RESULTS_ROOT=results/annotate_precision/stead_iso_2026-08-13 \
#     bash benchmarks/isolation/watch_annotate_precision_progress.sh
set -u
cd "$(dirname "$0")/../.."

ROOT="${RESULTS_ROOT:-results/annotate_precision/stead_iso_2026-08-13}"
PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

report() {
  clear 2>/dev/null || true
  echo "Annotate-precision progress  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Root: $ROOT"
  echo

  "$PY" - "$ROOT" <<'PY'
import json, sys, time
from pathlib import Path
from collections import Counter, defaultdict

root = Path(sys.argv[1])
METHODS = ["annotate_fp32", "annotate_bf16", "annotate_fp16"]
MODELS = ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
EQT = {"EQTransformer", "EQT-NC"}
CPUS = [5, 10, 15, 20]
BATCHES = [64, 128, 256, 512]
STATIONS = [250, 580]

expected = []
for st in STATIONS:
    for method in METHODS:
        for model in MODELS:
            if method == "annotate_fp16" and model in EQT:
                continue
            for bs in BATCHES:
                for c in CPUS:
                    for device in ("cpu", "gpu"):
                        expected.append((method, model, st, device, c, bs))

def result_path(method, model, st, device, c, bs):
    thr = c
    modern = (
        root / method / "stead" / f"{st}st" / model / device
        / f"cpus{c}" / f"thr{thr}" / f"bs{bs}" / "ann_prec" / "result.json"
    )
    legacy = (
        root / method / "stead" / f"{st}st" / model / device
        / f"cpus{c}" / f"thr{thr}" / "ann_prec" / "result.json"
    )
    if modern.is_file():
        return modern
    if bs == 256 and legacy.is_file():
        return legacy
    return None

done = {}
failed = []
for cell in expected:
    p = result_path(*cell)
    if p is None:
        continue
    try:
        r = json.loads(p.read_text())
        sr = float((r.get("timing") or {}).get("success_rate") or 0)
        if sr >= 1.0:
            done[cell] = r
        else:
            failed.append(cell)
    except Exception:
        failed.append(cell)

n_exp = len(expected)
n_done = len(done)
n_fail = len(failed)
n_left = n_exp - n_done - n_fail
pct = 100.0 * n_done / n_exp if n_exp else 0.0
bar_w = 40
filled = int(bar_w * n_done / n_exp) if n_exp else 0
bar = "#" * filled + "-" * (bar_w - filled)
print(f"Overall  [{bar}]  {n_done}/{n_exp}  ({pct:.1f}%)")
print(f"         remaining={n_left}  failed/partial={n_fail}")
print()

# By method × device
print("By method × device:")
print(f"  {'method':14s} {'cpu':>8s} {'gpu':>8s}")
for method in METHODS:
    for device in ("cpu", "gpu"):
        pass
for method in METHODS:
    parts = []
    for device in ("cpu", "gpu"):
        e = sum(1 for c in expected if c[0] == method and c[3] == device)
        d = sum(1 for c in done if c[0] == method and c[3] == device)
        parts.append(f"{d}/{e}")
    print(f"  {method:14s} {parts[0]:>8s} {parts[1]:>8s}")

print()
print("By batch size:")
print(f"  {'bs':>5s}  {'done':>10s}  {'left':>8s}")
for bs in BATCHES:
    e = sum(1 for c in expected if c[5] == bs)
    d = sum(1 for c in done if c[5] == bs)
    print(f"  {bs:5d}  {d:4d}/{e:<4d}   {e-d:5d}")

print()
print("By station:")
for st in STATIONS:
    e = sum(1 for c in expected if c[2] == st)
    d = sum(1 for c in done if c[2] == st)
    print(f"  {st}st  {d}/{e}")

# Running procs
print()
print("Live trials (from process list):")
import subprocess, re
try:
    out = subprocess.check_output(
        ["ps", "-eo", "pid,etime,cmd"], text=True, stderr=subprocess.DEVNULL
    )
except Exception:
    out = ""
alive = []
for line in out.splitlines():
    if "run_annotate_precision_trial.py" not in line:
        continue
    if "awk" in line or "watch_annotate" in line:
        continue
    kind = "worker" if "--repeat-index" in line else "driver"
    def grab(flag):
        m = re.search(rf"{flag}\s+(\S+)", line)
        return m.group(1) if m else "?"
    method = grab("--method").replace("annotate_", "")
    model = grab("--model")
    st = grab("--n-stations")
    device = grab("--device")
    ncpus = grab("--n-cpus")
    batch = grab("--batch-size")
    gid = grab("--gpu-id")
    et = line.split(None, 2)[1] if len(line.split(None, 2)) > 1 else "?"
    if method == "?" and model == "?":
        continue
    alive.append(
        f"  [{kind:6s}] {method} {model} {st}st {device} c{ncpus} bs{batch} gpu{gid}  etime={et}"
    )
if alive:
    # unique-ish lines
    for a in sorted(set(alive)):
        print(a)
else:
    print("  (none)")

# Recent finishes
log = root / "iso_annotate_precision_parallel.log"
print()
print("Recent log:")
if log.is_file():
    lines = log.read_text(errors="replace").strip().splitlines()
    for ln in lines[-8:]:
        print(" ", ln[:120])
else:
    print("  (no parallel log yet)")
PY
}

if [[ "$ONCE" == "1" ]]; then
  report
  exit 0
fi

echo "Refreshing every 30s  (Ctrl-C to stop).  One-shot: add --once"
echo
while true; do
  report
  sleep 30
done
