#!/usr/bin/env bash
# Two-GPU Model-Actor head-to-head: spread the actor pool across BOTH RTX 6000
# Adas and measure warm per-window latency, to test whether the single-GPU
# Model-Actor "loss" (T5) is fundamental or just single-device contention.
#
# Each trial uses BOTH GPUs, so trials MUST run sequentially (no GPU-slot sharing).
# Mirrors the h2h_v2 protocol: STEAD, 20 host cores, 20 actors, 8 feeds, 10 repeats.
set -u
cd "$(dirname "$0")/.."
CORES=$(seq -s, 0 19)
ROOT=results/fair_benchmark_h2h_2gpu
LOG=$ROOT/run_2gpu.log
mkdir -p "$ROOT"

run() {  # model  stations  tag
  echo "=== $(date +%H:%M:%S)  $1  ${2}st  ($3) ===" | tee -a "$LOG"
  python3 scripts/run_fair_stream_trial.py \
    --strategy stream_modelactor_2gpu --dataset stead --n-stations "$2" --model "$1" \
    --device gpu --n-cpus 20 --concurrency 20 --core-list "$CORES" \
    --repeats 10 --n-feeds 8 --feed-interval-s 0 \
    --tag "$3" --results-root "$ROOT" --resume >> "$LOG" 2>&1
}

for ST in 580 250; do
  run PhaseNet      "$ST" "2gpu_${ST}"
  run PhaseNetLight "$ST" "2gpu_${ST}"
  run EQTransformer "$ST" "2gpu_${ST}"
  run EQT-NC        "$ST" "2gpu_${ST}"
done
echo "=== ALL 2-GPU TRIALS DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"
