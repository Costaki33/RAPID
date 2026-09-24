#!/usr/bin/env bash
# Progress for the native-winner Model-Actor / Ripper campaign.
#
# Usage:
#   bash benchmarks/isolation/watch_orch_native_winners.sh
#   bash benchmarks/isolation/watch_orch_native_winners.sh --once
#   STAGES=A,B RESULTS_ROOT=... bash benchmarks/isolation/watch_orch_native_winners.sh --once
set -u
cd "$(dirname "$0")/../.."

ROOT="${RESULTS_ROOT:-}"
if [[ -z "$ROOT" && -f results/orch_annotate/NATIVE_WINNERS_ROOT.txt ]]; then
  ROOT=$(cat results/orch_annotate/NATIVE_WINNERS_ROOT.txt)
fi
ROOT="${ROOT:-results/orch_annotate/stead_iso_$(date +%Y-%m-%d)_native_winners}"
NATIVE_ROOT="${NATIVE_ROOT:-results/annotate_config/stead_iso_2026-09-08}"
WINNERS="${WINNERS:-$NATIVE_ROOT/winners_merged_quality_safe.json}"
STAGES="${STAGES:-A,B,C}"
PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

report() {
  clear 2>/dev/null || true
  echo "Native-winner orch  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Root: $ROOT"
  echo "Stages: $STAGES"
  echo
  PYTHONPATH="benchmarks/fair:." "$PY" - "$ROOT" "$WINNERS" "$STAGES" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
winners_path = Path(sys.argv[2])
stages = [s.strip().upper() for s in sys.argv[3].split(",") if s.strip()]
from native_winner_orch import iter_stage, load_winners, winner_cell_dir, TAG

winners = load_winners(winners_path)

def status(cell):
    p = winner_cell_dir(root, cell, tag=TAG) / "result.json"
    if not p.is_file():
        return "pending"
    try:
        r = json.loads(p.read_text())
    except Exception:
        return "failed"
    timing = r.get("timing") or {}
    if timing.get("timed_out") or (r.get("meta") or {}).get("timed_out"):
        return "timed_out"
    if float(timing.get("success_rate") or 0) >= 1.0:
        return "done"
    return "failed"

for stage in stages:
    try:
        cells = list(iter_stage(stage, winners, results_root=root, tag=TAG))
    except Exception as exc:
        print(f"Stage {stage}: cannot enumerate ({exc})")
        print()
        continue
    counts = Counter(status(c) for c in cells)
    n = len(cells)
    n_done = counts["done"]
    pct = 100.0 * n_done / n if n else 0.0
    bar_w = 40
    filled = int(bar_w * n_done / n) if n else 0
    bar = "#" * filled + "-" * (bar_w - filled)
    print(f"Stage {stage}  [{bar}]  {n_done}/{n}  ({pct:.1f}%)")
    print(
        f"         pending={counts['pending']}  failed={counts['failed']}  "
        f"timed_out={counts['timed_out']}"
    )
    by = Counter()
    for c in cells:
        if status(c) == "done":
            by[(c["device"], c["n_stations"], c["composition"])] += 1
    if by:
        print("         done by device/st/comp:", dict(by))
    print()

log = root / "iso_orch_native_winners.log"
if log.is_file():
    lines = log.read_text(errors="replace").strip().splitlines()
    print("Last log lines:")
    for line in lines[-8:]:
        print(" ", line)
PY
}

if [[ "$ONCE" == "1" ]]; then
  report
  exit 0
fi
while true; do
  report
  sleep 30
done
