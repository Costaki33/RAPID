#!/usr/bin/env bash
# Warm Model-Actor + Network-Batched Classify: persistent actors each run
# classify() on a multi-station merged share (not one station per call).
#
# Matched to the paper warm head-to-head: 580 stations, 20 cores, 10x8 feeds,
# batch 256, thr=1 per actor. CPU uses 20 actors (same as stream_modelactor
# cpu_c20). GPU runs both matched concurrency=20 and a low-contention
# concurrency=2 layout (larger per-actor batches).
set -euo pipefail

cd "$(dirname "$0")/../.."

CORES=$(seq -s, 0 19)
RESULTS_ROOT=results/iso_full_benchmark/stream
LOG=results/iso_full_benchmark/warm_ma_batched.log
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)

in_samples() {
  case "$1" in
    PhaseNet|PhaseNetLight) echo 3001 ;;
    *) echo 6000 ;;
  esac
}

mkdir -p "$(dirname "$LOG")"

run_one() {
  local model="$1" device="$2" conc="$3" tag="$4"
  local samples
  samples=$(in_samples "$model")
  echo "=== $(date -Ins) warm Model-Actor NBC: ${model} ${device} conc=${conc} tag=${tag} ===" | tee -a "$LOG"
  python benchmarks/fair/run_fair_stream_trial.py \
    --strategy stream_modelactor_batched \
    --dataset stead \
    --n-stations 580 \
    --model "$model" \
    --device "$device" \
    --n-cpus 20 \
    --gpu-id 0 \
    --core-list "$CORES" \
    --concurrency "$conc" \
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
}

for model in "${MODELS[@]}"; do
  run_one "$model" cpu 20 iso_cpu_580
  run_one "$model" gpu 20 iso_gpu_580
  # Fewer GPU actors → larger cross-station batches per classify() call.
  run_one "$model" gpu 2 iso_gpu_c2_580
done
