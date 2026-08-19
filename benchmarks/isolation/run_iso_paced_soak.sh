#!/usr/bin/env bash
# Paced continuous-feed soak: feed every 60 s for many windows (Camilo real-time ask).
# Models: PhaseNet (light) + EQTransformer (heavy). Methods: Annotate, NBC, MA-NBC.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Prefer the rapid env (SeisBench + Torch); base conda TF is unrelated here but
# rapid keeps the same interpreter as the iso warm runners.
PY="${RAPID_PYTHON:-$HOME/miniconda3/envs/rapid/bin/python}"
CORES=$(seq -s, 0 19)
RESULTS_ROOT=results/iso_full_benchmark/stream
LOG=results/iso_full_benchmark/paced_soak.log
N_FEEDS="${N_FEEDS:-20}"
REPEATS="${REPEATS:-2}"
mkdir -p "$(dirname "$LOG")"

in_samples() {
  case "$1" in
    PhaseNet|PhaseNetLight) echo 3001 ;;
    *) echo 6000 ;;
  esac
}

run_one() {
  local strategy="$1" model="$2" device="$3" conc="$4" thr="$5" tag="$6"
  local samples
  samples=$(in_samples "$model")
  echo "=== $(date -Ins) paced soak: ${strategy} ${model} ${device} feeds=${N_FEEDS} ===" | tee -a "$LOG"
  local extra=()
  if [[ -n "$thr" ]]; then
    extra+=(--torch-threads "$thr")
  fi
  "$PY" benchmarks/fair/run_fair_stream_trial.py \
    --strategy "$strategy" \
    --dataset stead \
    --n-stations 580 \
    --model "$model" \
    --device "$device" \
    --n-cpus 20 \
    --core-list "$CORES" \
    --concurrency "$conc" \
    --gpu-id 0 \
    --in-samples "$samples" \
    --overlap-samples 0 \
    --dtype fp32 \
    --slipstream-batch-size 256 \
    --repeats "$REPEATS" \
    --n-feeds "$N_FEEDS" \
    --feed-interval-s 60 \
    --tag "$tag" \
    --results-root "$RESULTS_ROOT" \
    --resume "${extra[@]}" 2>&1 | tee -a "$LOG"
}

# Light + heavy on CPU; Annotate uses 8 threads (measured optimum band).
for model in PhaseNet EQTransformer; do
  run_one stream_annotate "$model" cpu 1 8 "soak_cpu_580"
  run_one stream_classify_batched "$model" cpu 1 1 "soak_cpu_580"
  run_one stream_modelactor_batched "$model" cpu 20 "" "soak_cpu_580"
done
