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
cd "$(dirname "$0")/../.."
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
  python3 benchmarks/fair/run_fair_stream_trial.py \
    --strategy "$strat" --dataset stead --n-stations "$st" --model "$model" \
    --device "$dev" --n-cpus 20 --gpu-id 0 --core-list "$CORES" --concurrency 20 \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats 10 --n-feeds 8 --feed-interval-s 0 \
    --tag "iso_${dev}_${st}" --results-root "$ROOT/h2h" --resume >> "$LOG" 2>&1
}

gpu_ma_sweep() {  # strategy model stations ncpus  (1-GPU or 2-GPU MA across CPU/concurrency)
  local strat=$1 model=$2 st=$3 c=$4 ins; ins=$(insamples "$model")
  local cores; cores=$(seq -s, 0 $((c-1)))           # exactly c cores, conc = c actors
  local lbl="1gpu"; [ "$strat" = "stream_modelactor_2gpu" ] && lbl="2gpu"
  note "GPU-SWEEP $strat / $model / ${st}st cpu$c (in=$ins)"
  python3 benchmarks/fair/run_fair_stream_trial.py \
    --strategy "$strat" --dataset stead --n-stations "$st" --model "$model" \
    --device gpu --n-cpus "$c" --gpu-id 0 --core-list "$cores" --concurrency "$c" \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats 5 --n-feeds 8 --feed-interval-s 0 \
    --tag "iso_${lbl}_${st}_cpu${c}" --results-root "$ROOT/h2h" --resume >> "$LOG" 2>&1
}

native() {  # method model stations threads repeats  (single-process thread sweep)
  local meth=$1 model=$2 st=$3 thr=$4 reps=$5 ins; ins=$(insamples "$model")
  note "NATIVE $meth / $model / ${st}st thr$thr x$reps (in=$ins)"
  python3 benchmarks/fair/run_fair_trial.py \
    --method "$meth" --dataset stead --n-stations "$st" --model "$model" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads "$thr" \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats "$reps" --tag "iso_thr${thr}" --results-root "$ROOT/native" --resume >> "$LOG" 2>&1
}

orch() {  # strategy device model stations  (cold-start orchestration)
  local strat=$1 dev=$2 model=$3 st=$4 ins; ins=$(insamples "$model")
  note "ORCH $strat / $dev / $model / ${st}st (in=$ins)"
  python3 benchmarks/fair/run_fair_orch_trial.py \
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

# ---- Phase 2: GPU Model-Actor across the CPU/concurrency sweep {5,10,15,20} ----
# 2-GPU split AND a clean single-GPU companion, so 1-GPU vs 2-GPU is comparable
# at matched concurrency (1-GPU cpu20 already covered by Phase 1, so 5,10,15 only).
for ST in 580 250; do
  for M in "${MODELS[@]}"; do
    for C in 5 10 15 20; do gpu_ma_sweep stream_modelactor_2gpu "$M" "$ST" "$C"; done
    for C in 5 10 15;    do gpu_ma_sweep stream_modelactor      "$M" "$ST" "$C"; done
  done
done

# ---- Phase 3: native single-process THREAD SWEEP (T1), lean ----
# Optimum lives near 1 thread; sweep {1,2,4} both stations. The naive baseline is
# SeisBench/torch's true out-of-the-box default (threads=0 sentinel -> no pinning,
# torch uses physical-core count), anchored at 580 only -- 3 reps for the cheap
# batched methods, 1 rep for catastrophic classify.
for ST in 580 250; do
  for M in "${MODELS[@]}"; do
    # annotate/slipstream are cheap at any thread count -> full {1,2,4} curve.
    for METH in annotate slipstream; do
      for T in 1 2 4; do native "$METH" "$M" "$ST" "$T" 3; done
    done
    # classify is catastrophic when oversubscribed -> just its optimum (thr1, the
    # reportable T1) at 3 reps; the naive-default anchor below shows the blow-up.
    native classify "$M" "$ST" 1 3
  done
done
for M in "${MODELS[@]}"; do                       # naive torch-default anchor (~64 threads), 580 only
  for METH in annotate slipstream; do native "$METH" "$M" 580 0 3; done
  native classify "$M" 580 0 1
done

# ---- Phase 4: cold-start orchestration (T2), CPU + GPU, 580 only ----
# (cold start is model-load-dominated -> ~station-count-independent; 580 suffices.)
for M in "${MODELS[@]}"; do
  for STRAT in modelactor ripper; do
    for DEV in cpu gpu; do orch "$STRAT" "$DEV" "$M" 580; done
  done
done

note "ISOLATION RUN DONE"
