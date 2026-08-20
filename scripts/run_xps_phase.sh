#!/usr/bin/env bash
# Launch XPS runbook phases inside WSL with machine-local defaults.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"

PHASE="${1:-smoke}"
export CORES="${CORES:-$(cat "$HOME/RAPID/.xps_cores")}"
# RTX 4050 Laptop = 6 GB; runner also auto-detects, but set explicitly for provenance.
export GPU_ACTOR_CAP="${GPU_ACTOR_CAP:-1}"
export CPU_ACTOR_CAP="${CPU_ACTOR_CAP:-10}"
export PHASE

echo "PHASE=$PHASE CORES=$CORES GPU_ACTOR_CAP=$GPU_ACTOR_CAP CPU_ACTOR_CAP=$CPU_ACTOR_CAP"
bash benchmarks/isolation/run_xps_fastest.sh
