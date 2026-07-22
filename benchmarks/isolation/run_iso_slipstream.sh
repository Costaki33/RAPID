#!/usr/bin/env bash
# Isolated Slipstream warm-latency add-on: the head-to-head measured annotate vs
# Model-Actor (classify); this adds the lean Slipstream-BF16 actor path on the
# SAME protocol (sequential, 10 repeats x 8 feeds, per-model native window) so its
# warm per-window latency is on equally clean footing. Writes into the h2h dir so
# analyze_iso.py / generate_iso_tables.py pick it up as a new method column.
set -u
cd "$(dirname "$0")/../.."
CORES=$(seq -s, 0 19)
ROOT=results/fair_benchmark_iso
LOG=$ROOT/isolation_slipstream.log
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

note "ISO SLIPSTREAM START"
for ST in 580 250; do
  for M in "${MODELS[@]}"; do
    for DEV in cpu gpu; do
      ins=$(insamples "$M")
      note "SLIP stream_modelactor_slipstream / $DEV / $M / ${ST}st (bf16, in=$ins)"
      python3 benchmarks/fair/run_fair_stream_trial.py \
        --strategy stream_modelactor_slipstream --dataset stead --n-stations "$ST" --model "$M" \
        --device "$DEV" --n-cpus 20 --gpu-id 0 --core-list "$CORES" --concurrency 20 \
        --in-samples "$ins" --overlap-samples 0 --dtype bf16 --slipstream-batch-size 256 \
        --repeats 10 --n-feeds 8 --feed-interval-s 0 \
        --tag "iso_${DEV}_${ST}" --results-root "$ROOT/h2h" --resume >> "$LOG" 2>&1
    done
  done
done
note "ISO SLIPSTREAM DONE"
