#!/usr/bin/env bash
# Progress for the 5/10/15 CPU orch sweep.
#
# Usage:
#   LAYER=all bash benchmarks/isolation/watch_orch_annotate_core_budget.sh --once
#   LAYER=all bash benchmarks/isolation/watch_orch_annotate_core_budget.sh
set -u
cd "$(dirname "$0")/../.."

ROOT="${RESULTS_ROOT:-results/orch_annotate/stead_2026-08-21_cpus5_10_15}"
PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
LAYER="${LAYER:-all}"
NCPU_GRID="${NCPU_GRID:-5,10,15}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

report() {
  clear 2>/dev/null || true
  echo "Orchestration annotate core-budget  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Root: $ROOT"
  echo "Layer: $LAYER  NCPU_GRID=$NCPU_GRID  QUICK=${QUICK:-0}"
  echo

  if [[ -z "${SKIP_RIPPER_S1:-}" ]]; then
    if [[ "${QUICK:-0}" == "1" ]]; then _SKIP_RP_S1=0; else _SKIP_RP_S1=1; fi
  else
    _SKIP_RP_S1="$SKIP_RIPPER_S1"
  fi
  if [[ -z "${SKIP_STAGGERED_RIPPER:-}" ]]; then
    if [[ "${QUICK:-0}" == "1" ]]; then _SKIP_RP_ST=0; else _SKIP_RP_ST=1; fi
  else
    _SKIP_RP_ST="$SKIP_STAGGERED_RIPPER"
  fi
  if [[ -z "${SKIP_FP32_REALTIME:-}" ]]; then
    if [[ "${QUICK:-0}" == "1" ]]; then _SKIP_FP32=0; else _SKIP_FP32=1; fi
  else
    _SKIP_FP32="$SKIP_FP32_REALTIME"
  fi
  "$PY" - "$ROOT" "$LAYER" "${QUICK:-0}" "$_SKIP_RP_S1" "$_SKIP_RP_ST" "$_SKIP_FP32" "$NCPU_GRID" <<'PY'
import json, re, subprocess, sys
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1])
layer = sys.argv[2]
quick = sys.argv[3] == "1"
skip_ripper_s1 = sys.argv[4] == "1"
skip_staggered_ripper = sys.argv[5] == "1"
skip_fp32_realtime = sys.argv[6] == "1"
ncpu_grid = [int(x) for x in sys.argv[7].split(",") if x.strip()]
TAG = "orch_ann"

import runpy
ns = runpy.run_path("benchmarks/fair/run_orch_annotate_trial.py")
BEST_BATCH, DTYPE_OF = ns["BEST_BATCH"], ns["DTYPE_OF"]
iter_core_budget_matrix, orch_cell_dir = ns["iter_core_budget_matrix"], ns["orch_cell_dir"]

expected = []
for n_cpus in ncpu_grid:
    expected.extend(list(iter_core_budget_matrix(
        n_cpus,
        layer=layer,
        quick=quick,
        skip_ripper_s1=skip_ripper_s1,
        skip_staggered_ripper=skip_staggered_ripper,
        skip_fp32_realtime=skip_fp32_realtime,
    )))

done, failed = {}, []
for c in expected:
    key = (
        c["n_cpus"], c["composition"], c["method"], c["model"], c["n_stations"],
        c["device"], c["k_ma"], c["k_rp"], c["packaging"], c["arrival"], c["fill"],
    )
    p = orch_cell_dir(root, c, n_cpus=c["n_cpus"], tag=TAG, batch_size=BEST_BATCH[DTYPE_OF[c["method"]]]) / "result.json"
    if not p.is_file():
        continue
    try:
        r = json.loads(p.read_text())
        sr = float((r.get("timing") or {}).get("success_rate") or 0)
        if sr >= 1.0:
            done[key] = r
        else:
            failed.append(key)
    except Exception:
        failed.append(key)

n_exp, n_done, n_fail = len(expected), len(done), len(failed)
n_left = n_exp - n_done - n_fail
pct = 100.0 * n_done / n_exp if n_exp else 0.0
bar_w = 40
filled = int(bar_w * n_done / n_exp) if n_exp else 0
bar = "#" * filled + "-" * (bar_w - filled)
print(f"Overall  [{bar}]  {n_done}/{n_exp}  ({pct:.1f}%)")
print(f"         remaining={n_left}  failed/partial={n_fail}")
print()

def show(title, getter):
    print(title)
    ctr_e, ctr_d = Counter(), Counter()
    for c in expected:
        ctr_e[getter(c)] += 1
    for key in done:
        n_cpus, comp, method, model, st, device, kma, krp, pack, arrival, fill = key
        cell = {
            "n_cpus": n_cpus, "composition": comp, "method": method, "model": model,
            "n_stations": st, "device": device, "k_ma": kma, "k_rp": krp,
            "packaging": pack, "arrival": arrival, "fill": fill,
        }
        ctr_d[getter(cell)] += 1
    for k, e in ctr_e.items():
        print(f"  {str(k):36s}  {ctr_d[k]}/{e}")
    print()

show("By n_cpus:", lambda c: c["n_cpus"])
show("By composition:", lambda c: c["composition"])
show("By device:", lambda c: c["device"])
show("By arrival:", lambda c: c["arrival"])
show("By packaging:", lambda c: c["packaging"])

print("Live trials:")
try:
    out = subprocess.check_output(["ps", "-eo", "pid,etime,cmd"], text=True)
except Exception:
    out = ""
alive = []
for line in out.splitlines():
    if "run_orch_annotate_trial.py" not in line:
        continue
    if "--print-matrix" in line:
        continue
    kind = "worker" if "--repeat-index" in line else "driver"
    def grab(flag):
        m = re.search(rf"{flag}\s+(\S+)", line)
        return m.group(1) if m else "?"
    et = line.split(None, 2)[1] if len(line.split(None, 2)) > 1 else "?"
    alive.append(
        f"  [{kind:6s}] cpus={grab('--n-cpus')} {grab('--composition')} {grab('--method')} "
        f"{grab('--model')} {grab('--n-stations')}st {grab('--device')} "
        f"kma={grab('--k-ma')} krp={grab('--k-rp')} {grab('--packaging')} "
        f"{grab('--arrival')} {grab('--fill')}  etime={et}"
    )
if alive:
    for a in sorted(set(alive)):
        print(a)
else:
    print("  (none)")

log = root / "iso_orch_annotate.log"
print()
print("Recent log:")
if log.is_file():
    for ln in log.read_text(errors="replace").strip().splitlines()[-8:]:
        print(" ", ln[:160])
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
