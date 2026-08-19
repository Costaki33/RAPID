#!/usr/bin/env bash
# Live progress for sequential per-station annotate-bf16 baseline.
#
# Usage:
#   bash benchmarks/isolation/watch_annotate_sequential_progress.sh
#   bash benchmarks/isolation/watch_annotate_sequential_progress.sh --once
set -u
cd "$(dirname "$0")/../.."

ROOT="${RESULTS_ROOT:-results/annotate_precision/stead_seq_bf16_2026-08-13}"
PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

report() {
  clear 2>/dev/null || true
  echo "Sequential per-station annotate-bf16  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Root: $ROOT"
  echo

  "$PY" - "$ROOT" <<'PY'
import json, re, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1])
METHOD = "annotate_bf16"
MODELS = ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
STATIONS = [250, 580]
CPUS = [5, 20]
BATCHES = [256, 512]
PACKAGING = "sequential"
TAG = "ann_seq"

expected = []
for st in STATIONS:
    for model in MODELS:
        for bs in BATCHES:
            for c in CPUS:
                for device in ("cpu", "gpu"):
                    expected.append((model, st, device, c, bs))

def result_path(model, st, device, c, bs):
    return (
        root / METHOD / "stead" / f"{st}st" / model / device
        / f"cpus{c}" / f"thr{c}" / f"bs{bs}" / PACKAGING / TAG / "result.json"
    )

done, failed = {}, []
for cell in expected:
    p = result_path(*cell)
    if not p.is_file():
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

n_exp, n_done, n_fail = len(expected), len(done), len(failed)
n_left = n_exp - n_done - n_fail
pct = 100.0 * n_done / n_exp if n_exp else 0.0
bar_w = 40
filled = int(bar_w * n_done / n_exp) if n_exp else 0
bar = "#" * filled + "-" * (bar_w - filled)
print(f"Overall  [{bar}]  {n_done}/{n_exp}  ({pct:.1f}%)")
print(f"         remaining={n_left}  failed/partial={n_fail}")
print()

print("By device:")
for device in ("cpu", "gpu"):
    e = sum(1 for c in expected if c[2] == device)
    d = sum(1 for c in done if c[2] == device)
    print(f"  {device:3s}  {d}/{e}")

print()
print("By model:")
for model in MODELS:
    e = sum(1 for c in expected if c[0] == model)
    d = sum(1 for c in done if c[0] == model)
    print(f"  {model:14s}  {d}/{e}")

print()
print("By station / batch / cores:")
for st in STATIONS:
    e = sum(1 for c in expected if c[1] == st)
    d = sum(1 for c in done if c[1] == st)
    print(f"  {st}st  {d}/{e}")
for bs in BATCHES:
    e = sum(1 for c in expected if c[4] == bs)
    d = sum(1 for c in done if c[4] == bs)
    print(f"  bs{bs}  {d}/{e}")
for cpus in CPUS:
    e = sum(1 for c in expected if c[3] == cpus)
    d = sum(1 for c in done if c[3] == cpus)
    print(f"  c{cpus}  {d}/{e}")

print()
print("Finished cells (inference_s mean):")
print(f"  {'model':14s} {'st':>4s} {'dev':3s} {'c':>2s} {'bs':>3s} {'inf_s':>8s} {'P_f1':>6s}")
rows = []
for cell, r in done.items():
    model, st, device, cpus, bs = cell
    inf = (r.get("timing") or {}).get("inference_s_mean")
    pf1 = (r.get("pick_quality_vs_catalog") or {}).get("P.f1_mean")
    rows.append((model, st, device, cpus, bs, inf, pf1))
rows.sort(key=lambda x: (x[1], x[0], x[2], x[3], x[4]))
for model, st, device, cpus, bs, inf, pf1 in rows[-12:]:
    inf_s = f"{inf:.3f}" if inf is not None else "—"
    pf = f"{pf1:.3f}" if isinstance(pf1, (int, float)) else "—"
    print(f"  {model:14s} {st:4d} {device:3s} {cpus:2d} {bs:3d} {inf_s:>8s} {pf:>6s}")
if not rows:
    print("  (none yet)")

print()
print("Live trials:")
try:
    out = subprocess.check_output(["ps", "-eo", "pid,etime,cmd"], text=True)
except Exception:
    out = ""
alive = []
for line in out.splitlines():
    if "run_annotate_precision_trial.py" not in line:
        continue
    if "sequential" not in line and "ann_seq" not in line:
        continue
    kind = "worker" if "--repeat-index" in line else "driver"
    def grab(flag):
        m = re.search(rf"{flag}\s+(\S+)", line)
        return m.group(1) if m else "?"
    et = line.split(None, 2)[1] if len(line.split(None, 2)) > 1 else "?"
    alive.append(
        f"  [{kind:6s}] {grab('--model')} {grab('--n-stations')}st "
        f"{grab('--device')} c{grab('--n-cpus')} bs{grab('--batch-size')} "
        f"gpu{grab('--gpu-id')}  etime={et}"
    )
if alive:
    for a in sorted(set(alive)):
        print(a)
else:
    print("  (none)")

log = root / "iso_sequential.log"
print()
print("Recent log:")
if log.is_file():
    for ln in log.read_text(errors="replace").strip().splitlines()[-8:]:
        print(" ", ln[:120])
else:
    print("  (no log yet)")
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
