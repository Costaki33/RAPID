#!/usr/bin/env bash
# Progress for the locked-recipe transfer suite.
#
#   RESULTS_ROOT=results/locked_recipe_transfer/<run> \
#     bash benchmarks/isolation/watch_locked_recipe_transfer.sh
#   ... bash benchmarks/isolation/watch_locked_recipe_transfer.sh --once
set -u
cd "$(dirname "$0")/../.."

PY="${RAPID_PYTHON:-$(command -v python3)}"
if [[ -x /home/skevofilaxc/miniconda3/envs/rapid/bin/python && -z "${RAPID_PYTHON:-}" ]]; then
  PY=/home/skevofilaxc/miniconda3/envs/rapid/bin/python
fi
ROOT="${RESULTS_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  echo "Set RESULTS_ROOT to the transfer run directory." >&2
  exit 1
fi
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

report() {
  clear 2>/dev/null || true
  echo "Locked-recipe transfer  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Root: $ROOT"
  echo "LAYER=${LAYER:-all}  CORE_GRID=${CORE_GRID:-5,10,15,20}  SKIP_GPU=${SKIP_GPU:-0}"
  echo
  "$PY" - "$ROOT" <<'PY'
import json, os, sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, "benchmarks/fair")
from locked_recipe_transfer_matrix import cell_ok, env_cells, result_path

cells = env_cells()
n = len(cells)
done = fail = 0
by_layer = Counter()
by_layer_done = Counter()
live_fail = []
for c in cells:
    by_layer[c["layer"]] += 1
    p = result_path(root, c)
    if cell_ok(p):
        done += 1
        by_layer_done[c["layer"]] += 1
        continue
    if p.is_file():
        fail += 1
        live_fail.append(c)

remain = n - done - fail
width = 40
filled = int(width * done / n) if n else width
bar = "#" * filled + "-" * (width - filled)
print(f"Overall  [{bar}]  {done}/{n}  ({100.0 * done / n if n else 0:.1f}%)")
print(f"         remaining={remain}  failed/partial={fail}")
print()
print("By layer:")
for layer in ("native", "playback", "staggered"):
    if by_layer[layer] == 0:
        continue
    print(f"  {layer:12} {by_layer_done[layer]}/{by_layer[layer]}")
if live_fail:
    print()
    print("Failed/partial (first 8):")
    for c in live_fail[:8]:
        print(
            f"  {c['layer']} {c['model']} {c['n_stations']}st {c['device']} "
            f"cpus={c['n_cpus']} k={c['k_ma']}"
        )

log = root / "locked_recipe_transfer.log"
if log.is_file():
    lines = log.read_text(errors="replace").splitlines()
    print()
    print("Log tail:")
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
  sleep 15
done
