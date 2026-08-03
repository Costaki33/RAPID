#!/usr/bin/env bash
# Missing kept-warm native baseline: one SeisBench classify() call on the full
# merged 580-station network, using the measured one-thread optimum.
set -euo pipefail

cd "$(dirname "$0")/../.."

CORES=$(seq -s, 0 19)
RESULTS_ROOT=results/iso_full_benchmark/stream
LOG=results/iso_full_benchmark/warm_batched.log
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)

in_samples() {
  case "$1" in
    PhaseNet|PhaseNetLight) echo 3001 ;;
    *) echo 6000 ;;
  esac
}

mkdir -p "$(dirname "$LOG")"

for model in "${MODELS[@]}"; do
  samples=$(in_samples "$model")
  for device in cpu gpu; do
    tag="iso_${device}_580"
    echo "=== $(date -Ins) warm Network-Batched Classify: ${model} ${device} ===" | tee -a "$LOG"
    python benchmarks/fair/run_fair_stream_trial.py \
      --strategy stream_classify_batched \
      --dataset stead \
      --n-stations 580 \
      --model "$model" \
      --device "$device" \
      --n-cpus 20 \
      --torch-threads 1 \
      --gpu-id 0 \
      --core-list "$CORES" \
      --concurrency 1 \
      --in-samples "$samples" \
      --overlap-samples 0 \
      --dtype fp32 \
      --slipstream-batch-size 256 \
      --repeats 10 \
      --n-feeds 8 \
      --feed-interval-s 0 \
      --tag "$tag" \
      --results-root "$RESULTS_ROOT" \
      --resume 2>&1 | tee -a "$LOG"
  done
done
