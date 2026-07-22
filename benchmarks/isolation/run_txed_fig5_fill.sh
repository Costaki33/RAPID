#!/usr/bin/env bash
# Fill missing TXED pick-quality cells for Figure 5.
#
# Missing vs STEAD coverage:
#   classify_batched  x 4 models   (tag cpu_c20_thr1)
#   slipstream FP16   x PN/PNL     (tag iso_fp16)
#   slipstream BF16   x 4 models   (tag iso_bf16)
#
# Isolated / sequential: one trial at a time, torch threads = 1.
# Affinity pinned to cores 20-39 so we stay off the busy 0-19 block.
set -u
cd "$(dirname "$0")/../.."

CORES=$(seq -s, 20 39)
ROOT=results/iso_full_benchmark/native
LOG=$ROOT/../run_txed_fig5_fill.log
mkdir -p "$ROOT"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $* ===" | tee -a "$LOG"; }

run_one() {
  note "$*"
  python3 benchmarks/fair/run_fair_trial.py "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    note "FAILED (rc=$rc): $*"
  else
    note "OK"
  fi
  return $rc
}

note "TXED FIG5 FILL START (cores 20-39, sequential, thr=1)"

# 1) Batched-Classify — all 4 models
for M in "${MODELS[@]}"; do
  INS=$(insamples "$M")
  run_one \
    --method classify_batched --dataset txed --n-stations 580 --model "$M" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads 1 \
    --in-samples "$INS" --overlap-samples 0 --dtype fp32 --batch-size 256 \
    --repeats 3 --tag "cpu_c20_thr1" --results-root "$ROOT" --resume
done

# 2) Slipstream FP16 — PN / PNL only (EQT family numerically unsafe)
for M in PhaseNet PhaseNetLight; do
  INS=$(insamples "$M")
  run_one \
    --method slipstream --dataset txed --n-stations 580 --model "$M" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads 1 \
    --in-samples "$INS" --overlap-samples 0 --dtype fp16 --batch-size 256 \
    --repeats 3 --tag "iso_fp16" --results-root "$ROOT" --resume
done

# 3) Slipstream BF16 — all 4 models
for M in "${MODELS[@]}"; do
  INS=$(insamples "$M")
  run_one \
    --method slipstream --dataset txed --n-stations 580 --model "$M" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads 1 \
    --in-samples "$INS" --overlap-samples 0 --dtype bf16 --batch-size 256 \
    --repeats 3 --tag "iso_bf16" --results-root "$ROOT" --resume
done

note "TXED FIG5 FILL DONE"
