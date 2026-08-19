#!/usr/bin/env bash
# Live progress for orchestration annotate trials (no fp16).
#
# Usage:
#   LAYER=playback bash benchmarks/isolation/watch_orch_annotate_progress.sh
#   LAYER=all bash benchmarks/isolation/watch_orch_annotate_progress.sh --once
#   QUICK=1 LAYER=playback RESULTS_ROOT=... bash benchmarks/isolation/watch_orch_annotate_progress.sh --once
set -u
cd "$(dirname "$0")/../.."

ROOT="${RESULTS_ROOT:-results/orch_annotate/stead_2026-08-14}"
PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
LAYER="${LAYER:-playback}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

report() {
  clear 2>/dev/null || true
  echo "Orchestration annotate  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Root: $ROOT"
  echo "Layer: $LAYER  QUICK=${QUICK:-0}  SKIP_RIPPER_S1=${SKIP_RIPPER_S1:-}  SKIP_STAGGERED_RIPPER=${SKIP_STAGGERED_RIPPER:-}  SKIP_FP32_REALTIME=${SKIP_FP32_REALTIME:-}"
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
  "$PY" - "$ROOT" "$LAYER" "${QUICK:-0}" "$_SKIP_RP_S1" "$_SKIP_RP_ST" "$_SKIP_FP32" <<'PY'
import json, re, subprocess, sys
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1])
layer = sys.argv[2]
quick = sys.argv[3] == "1"
skip_ripper_s1 = sys.argv[4] == "1"
skip_staggered_ripper = sys.argv[5] == "1"
skip_fp32_realtime = sys.argv[6] == "1"
TAG = "orch_ann"

import runpy
ns = runpy.run_path("benchmarks/fair/run_orch_annotate_trial.py")
BEST_BATCH, DTYPE_OF, iter_matrix = ns["BEST_BATCH"], ns["DTYPE_OF"], ns["iter_matrix"]
expected = list(iter_matrix(
    layer=layer,
    quick=quick,
    skip_ripper_s1=skip_ripper_s1,
    skip_staggered_ripper=skip_staggered_ripper,
    skip_fp32_realtime=skip_fp32_realtime,
))

def result_path(c):
    dtype = DTYPE_OF[c["method"]]
    bs = BEST_BATCH[dtype]
    return (
        root / c["composition"] / c["method"] / "stead" / f"{c['n_stations']}st"
        / c["model"] / c["device"] / f"kma{c['k_ma']}_krp{c['k_rp']}" / c["packaging"]
        / c["arrival"] / c["fill"] / f"bs{bs}" / TAG / "result.json"
    )

done, failed = {}, []
for c in expected:
    key = (
        c["composition"], c["method"], c["model"], c["n_stations"], c["device"],
        c["k_ma"], c["k_rp"], c["packaging"], c["arrival"], c["fill"],
    )
    p = result_path(c)
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

def show(title, keys):
    print(title)
    ctr_e, ctr_d = Counter(), Counter()
    names = keys if isinstance(keys, tuple) else (keys,)
    for c in expected:
        k = tuple(c[x] for x in names) if len(names) > 1 else c[names[0]]
        ctr_e[k] += 1
    mapping_i = {
        "composition": 0, "method": 1, "model": 2, "n_stations": 3,
        "device": 4, "k_ma": 5, "k_rp": 6, "packaging": 7, "arrival": 8, "fill": 9,
    }
    for key in done:
        k = tuple(key[mapping_i[x]] for x in names) if len(names) > 1 else key[mapping_i[names[0]]]
        ctr_d[k] += 1
    for k, e in ctr_e.items():
        print(f"  {str(k):36s}  {ctr_d[k]}/{e}")
    print()

show("By composition:", "composition")
show("By method:", "method")
show("By device:", "device")
show("By packaging:", "packaging")
show("By arrival:", "arrival")
show("By fill:", "fill")
show("By stations:", "n_stations")
show("By model:", "model")

print("Recent finished (makespan mean, e2e p95 pooled):")
print(
    f"  {'comp':22s} {'method':14s} {'model':14s} {'dev':3s} "
    f"{'kma':>3s} {'krp':>3s} {'pk':2s} {'arr':10s} {'fill':6s} {'mk_s':>8s} {'p95':>8s}"
)
rows = []
for key, r in done.items():
    comp, method, model, st, device, kma, krp, pack, arrival, fill = key
    mk = (r.get("orch") or {}).get("makespan_s_mean")
    p95 = (
        ((r.get("latency") or {}).get("pooled_across_repeats") or {})
        .get("e2e_finish_minus_ready", {})
        .get("p95")
    )
    rows.append((comp, method, model, device, kma, krp, pack, arrival, fill, mk, p95))
rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
for row in rows[-12:]:
    comp, method, model, device, kma, krp, pack, arrival, fill, mk, p95 = row
    mk_s = f"{mk:.2f}" if isinstance(mk, (int, float)) else "—"
    p95_s = f"{p95:.2f}" if isinstance(p95, (int, float)) else "—"
    print(
        f"  {comp:22s} {method:14s} {model:14s} {device:3s} "
        f"{kma:3d} {krp:3d} {pack:2s} {arrival:10s} {fill:6s} {mk_s:>8s} {p95_s:>8s}"
    )
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
        f"  [{kind:6s}] {grab('--composition')} {grab('--method')} {grab('--model')} "
        f"{grab('--n-stations')}st {grab('--device')} kma={grab('--k-ma')} krp={grab('--k-rp')} "
        f"{grab('--packaging')} {grab('--arrival')} {grab('--fill')}  etime={et}"
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
        print(" ", ln[:140])
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
