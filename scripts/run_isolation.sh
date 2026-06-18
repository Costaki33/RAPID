#!/usr/bin/env bash
# STRICTLY SEQUENTIAL, fully-isolated re-measurement of every timing-critical
# result. Each trial runs in the foreground to completion before the next starts,
# so NOTHING else competes for cores or memory bandwidth -- the only way to get a
# trustworthy per-window latency (we measured PhaseNetLight classify at 2-4 s
# alone vs 306 s while a GPU job shared the box).
#
# Removes BOTH confounds: thread over-subscription (native baselines run at the
# measured optimum, torch threads = 1) and concurrent-trial contention (this
# script never overlaps trials). Uniform regime: each model's native window
# (PhaseNet/PNL 3001, EQT/EQT-NC 6000), overlap 0, 6000-sample network, fp32.
#
# Ordered by priority so the central results land first. Resumable (--resume).
set -u
cd "$(dirname "$0")/.."
CORES=$(seq -s, 0 19)                       # one 20-core trial, alone on the box
ROOT=results/fair_benchmark_iso
LOG=$ROOT/isolation.log
mkdir -p "$ROOT"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }

note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

stream() {  # strategy device model stations  (warm head-to-head / 2-GPU)
  local strat=$1 dev=$2 model=$3 st=$4 ins; ins=$(insamples "$model")
  note "STREAM $strat / $dev / $model / ${st}st (in=$ins)"
  python3 scripts/run_fair_stream_trial.py \
    --strategy "$strat" --dataset stead --n-stations "$st" --model "$model" \
    --device "$dev" --n-cpus 20 --gpu-id 0 --core-list "$CORES" --concurrency 20 \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats 10 --n-feeds 8 --feed-interval-s 0 \
    --tag "iso_${dev}_${st}" --results-root "$ROOT/h2h" --resume >> "$LOG" 2>&1
}

native() {  # method model stations  (single-process baseline at optimal threads=1)
  local meth=$1 model=$2 st=$3 ins; ins=$(insamples "$model")
  note "NATIVE $meth / $model / ${st}st thr1 (in=$ins)"
  python3 scripts/run_fair_trial.py \
    --method "$meth" --dataset stead --n-stations "$st" --model "$model" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads 1 \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats 3 --tag "iso_thr1" --results-root "$ROOT/native" --resume >> "$LOG" 2>&1
}

orch() {  # strategy device model stations  (cold-start orchestration)
  local strat=$1 dev=$2 model=$3 st=$4 ins; ins=$(insamples "$model")
  note "ORCH $strat / $dev / $model / ${st}st (in=$ins)"
  python3 scripts/run_fair_orch_trial.py \
    --strategy "$strat" --dataset stead --n-stations "$st" --model "$model" \
    --device "$dev" --n-cpus 20 --gpu-id 0 --core-list "$CORES" \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats 2 --tag "iso_${dev}_${st}" --results-root "$ROOT/orch" --resume >> "$LOG" 2>&1
}

note "ISOLATION RUN START"

# ---- Phase 1: warm head-to-head (T5) -- the central result, CPU + GPU ----
for ST in 580 250; do
  for M in "${MODELS[@]}"; do
    for STRAT in stream_annotate stream_modelactor; do
      for DEV in cpu gpu; do stream "$STRAT" "$DEV" "$M" "$ST"; done
    done
  done
done

# ---- Phase 2: 2-GPU actor split (re-measure; previously contended) ----
for ST in 580 250; do
  for M in "${MODELS[@]}"; do stream stream_modelactor_2gpu gpu "$M" "$ST"; done
done

# ---- Phase 3: native single-process baselines at threads=1 (T1) ----
for ST in 580 250; do
  for M in "${MODELS[@]}"; do
    for METH in annotate slipstream classify; do native "$METH" "$M" "$ST"; done
  done
done

# ---- Phase 4: cold-start orchestration (T2), CPU + GPU (slow; last) ----
for M in "${MODELS[@]}"; do
  for STRAT in modelactor ripper; do
    for DEV in cpu gpu; do orch "$STRAT" "$DEV" "$M" 580; done
  done
done

note "ISOLATION RUN DONE"
