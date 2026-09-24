#!/usr/bin/env bash
# Live progress for the native annotate configuration sweep.
#
# Usage:
#   bash benchmarks/isolation/watch_annotate_config_progress.sh
#   bash benchmarks/isolation/watch_annotate_config_progress.sh --once
#   RESULTS_ROOT=results/annotate_config/stead_iso_YYYY-MM-DD \
#     bash benchmarks/isolation/watch_annotate_config_progress.sh --once
set -u
cd "$(dirname "$0")/../.."

ROOT="${RESULTS_ROOT:-}"
if [[ -z "$ROOT" && -f results/annotate_config/RESULTS_ROOT.txt ]]; then
  ROOT=$(cat results/annotate_config/RESULTS_ROOT.txt)
fi
if [[ -z "$ROOT" ]]; then
  # newest dated tree
  ROOT=$(ls -d results/annotate_config/stead_iso_* 2>/dev/null | tail -1 || true)
fi
ROOT="${ROOT:-results/annotate_config/stead_iso_$(date +%Y-%m-%d)}"

PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

report() {
  clear 2>/dev/null || true
  echo "Annotate-config progress  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Root: $ROOT"
  echo

  PYTHONPATH="." "$PY" - "$ROOT" <<'PY'
import json, re, subprocess, sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
ns = __import__("runpy").run_path("benchmarks/fair/run_annotate_precision_trial.py")
iter_config_matrix = ns["iter_config_matrix"]
config_result_relpath = ns["config_result_relpath"]
TAG = ns["CONFIG_TAG"]

expected = list(iter_config_matrix())
n_exp = len(expected)

def load_result(cell):
    p = root / config_result_relpath(cell, tag=TAG)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"timing": {"success_rate": 0.0}, "_parse_error": True}

done, timed_out, failed, pending = [], [], [], []
for cell in expected:
    r = load_result(cell)
    if r is None:
        pending.append(cell)
        continue
    timing = r.get("timing") or {}
    if (
        timing.get("timed_out")
        or r.get("meta", {}).get("timed_out")
        or r.get("meta", {}).get("timed_out_by")
    ):
        timed_out.append(cell)
    elif float(timing.get("success_rate") or 0) >= 1.0:
        done.append(cell)
    else:
        failed.append(cell)

n_done, n_to, n_fail, n_left = len(done), len(timed_out), len(failed), len(pending)
pct = 100.0 * n_done / n_exp if n_exp else 0.0
bar_w = 40
filled = int(bar_w * n_done / n_exp) if n_exp else 0
bar = "#" * filled + "-" * (bar_w - filled)
print(f"Overall  [{bar}]  {n_done}/{n_exp}  ({pct:.1f}%)")
print(f"         remaining={n_left}  timed_out={n_to}  failed/partial={n_fail}")
print()

def key_count(rows, fn):
    c = Counter(fn(x) for x in rows)
    return c

print("By packaging × device:")
print(f"  {'packaging':12s} {'cpu done':>10s} {'gpu done':>10s}")
for packaging in ns["CONFIG_PACKAGINGS"]:
    parts = []
    for device in ns["CONFIG_DEVICES"]:
        e = sum(1 for c in expected if c["packaging"] == packaging and c["device"] == device)
        d = sum(1 for c in done if c["packaging"] == packaging and c["device"] == device)
        parts.append(f"{d}/{e}")
    print(f"  {packaging:12s} {parts[0]:>10s} {parts[1]:>10s}")

print()
print("By method:")
for method in ns["METHODS"]:
    e = sum(1 for c in expected if c["method"] == method)
    d = sum(1 for c in done if c["method"] == method)
    print(f"  {method:14s} {d:5d}/{e:<5d}")

print()
print("By model:")
for model in ns["MODELS"]:
    e = sum(1 for c in expected if c["model"] == model)
    d = sum(1 for c in done if c["model"] == model)
    print(f"  {model:14s} {d:5d}/{e:<5d}")

print()
print("By thread multiplier:")
for mult in ns["CONFIG_THREAD_MULTS"]:
    e = sum(1 for c in expected if c["thread_mult"] == mult)
    d = sum(1 for c in done if c["thread_mult"] == mult)
    print(f"  {mult:4.2f}x  {d:5d}/{e:<5d}")

print()
print("By station:")
for st in ns["CONFIG_STATIONS"]:
    e = sum(1 for c in expected if c["n_stations"] == st)
    d = sum(1 for c in done if c["n_stations"] == st)
    print(f"  {st}st  {d}/{e}")

print()
print("Live trials (from process list):")
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
    pkg = grab("--packaging")
    thr = grab("--torch-threads")
    gid = grab("--gpu-id")
    et = line.split(None, 2)[1] if len(line.split(None, 2)) > 1 else "?"
    if method == "?" and model == "?":
        continue
    alive.append(
        f"  [{kind:6s}] {method} {model} {st}st {device} {pkg} c{ncpus} thr{thr} "
        f"bs{batch} gpu{gid}  etime={et}"
    )
if alive:
    for a in sorted(set(alive)):
        print(a)
else:
    print("  (none)")

print()
print("Isolation occupancy:")
try:
    pgrep = subprocess.check_output(
        ["pgrep", "-af", "run_iso_annotate_config|run_annotate_precision_trial"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    n_live = sum(
        1
        for ln in pgrep.splitlines()
        if "run_annotate_precision_trial.py" in ln or "run_iso_annotate_config" in ln
    )
    print(f"  matching procs: {n_live}")
except Exception:
    print("  (pgrep unavailable)")

log = root / "iso_annotate_config.log"
print()
print("Recent log:")
if log.is_file():
    lines = log.read_text(errors="replace").strip().splitlines()
    for ln in lines[-8:]:
        print(" ", ln[:140])
else:
    print("  (no config log yet)")
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
