#!/usr/bin/env bash
# Warm EQCCT Model-Actor (station-group) head-to-head on the 580-station STEAD net.
# Uses the `rapid` conda env (working TensorFlow); base env TF is broken.
set -euo pipefail
cd "$(dirname "$0")/../.."

CORES=$(seq -s, 0 19)
RESULTS_ROOT=results/iso_full_benchmark/stream
LOG=results/iso_full_benchmark/warm_eqcct.log
PY="${PY:-$HOME/miniconda3/envs/rapid/bin/python}"
mkdir -p "$(dirname "$LOG")"

run_one() {
  local device="$1" conc="$2" tag="$3"
  echo "=== $(date -Ins) warm EQCCT Model-Actor: ${device} conc=${conc} ===" | tee -a "$LOG"
  "$PY" benchmarks/fair/run_fair_stream_eqcct.py \
    --device "$device" \
    --n-stations 580 \
    --n-cpus 20 \
    --core-list "$CORES" \
    --concurrency "$conc" \
    --gpu-id 0 \
    --n-feeds 8 \
    --feed-interval-s 0 \
    --repeats 10 \
    --tag "$tag" \
    --results-root "$RESULTS_ROOT" \
    --resume 2>&1 | tee -a "$LOG"
}

run_one cpu 20 iso_cpu_580
run_one gpu 2 iso_gpu_580
