#!/usr/bin/env bash
# Wait for SeisBench STEAD waveform download (or existing cache), then
# build real networks and run Phase B pilot.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"

LOG="$HOME/RAPID/results/xps_validation/stead_wait_pilot.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Ins) waiting for STEAD cache ==="
# SeisBench stores STEAD under ~/.seisbench/datasets/stead/
STEAD_DIR="$HOME/.seisbench/datasets/stead"
while true; do
  # Heuristic: waveforms directory present and no active download progress file growth stall
  # Prefer checking that a prior bootstrap isn't mid-download by looking for incomplete marker.
  if python - <<'PY'
from pathlib import Path
import seisbench.data as sbd
try:
    ds = sbd.STEAD(download_kwargs={"progress_bar": False})
    # Accessing metadata length proves local cache is usable without re-download.
    n = len(ds)
    print(f"STEAD ready n={n}")
    raise SystemExit(0)
except Exception as e:
    print(f"STEAD not ready: {e}")
    raise SystemExit(1)
PY
  then
    break
  fi
  echo "$(date -Ins) STEAD still downloading; sleeping 120s"
  sleep 120
done

echo "=== $(date -Ins) building STEAD networks ==="
bash /mnt/c/Users/cgs2528/Projects/RAPID/scripts/wsl_bootstrap_data.sh

echo "=== $(date -Ins) starting pilot ==="
export CORES="$(cat "$HOME/RAPID/.xps_cores")"
export GPU_ACTOR_CAP=1
export CPU_ACTOR_CAP=10
export PHASE=pilot
bash benchmarks/isolation/run_xps_fastest.sh

echo "=== $(date -Ins) pilot complete ==="
# Mirror results to Windows workspace
rsync -a "$HOME/RAPID/results/xps_validation/" /mnt/c/Users/cgs2528/Projects/RAPID/results/xps_validation/
