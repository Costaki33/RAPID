#!/usr/bin/env bash
# Isolated reduced-precision / compile sweep (PICK QUALITY only), STEAD.
#
# Why: the paper's precision claim is a pick-SAFETY result (FP16 silently
# collapses PhaseNetLight; BF16 tracks FP32), which is deterministic w.r.t.
# scheduling. So the iso-native source is the single-process Slipstream native
# family swept over dtype x compile, run sequentially on an idle box.
#
# Coverage mirrors the original precision matrix:
#   PN / PNL          : fp32, fp16, fp16+compile, bf16, bf16+compile
#   EQT / EQT-NC      : fp32, bf16, bf16+compile            (fp16 numerically unsafe)
# Same regime as the STEAD iso run: per-model native window, overlap 0, batch 256,
# torch threads = 1, 3 repeats (each repeat is scored, so compile/precision
# nondeterminism shows up as quality variance). DEDICATED results subtree.
set -u
cd "$(dirname "$0")/../.."
CORES=$(seq -s, 0 19)
ROOT=results/fair_benchmark_iso
LOG=$ROOT/iso_precision.log
mkdir -p "$ROOT"
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

slip() {  # model stations dtype compile(0/1)
  local model=$1 st=$2 dt=$3 cmp=$4 ins; ins=$(insamples "$model")
  local cflag="" ctag=""
  [ "$cmp" = "1" ] && { cflag="--compile"; ctag="_compile"; }
  note "PRECISION slipstream / $model / ${st}st / ${dt}${ctag} (in=$ins)"
  python3 benchmarks/fair/run_fair_trial.py \
    --method slipstream --dataset stead --n-stations "$st" --model "$model" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads 1 \
    --in-samples "$ins" --overlap-samples 0 --dtype "$dt" $cflag --batch-size 256 \
    --repeats 3 --tag "iso_${dt}${ctag}" --results-root "$ROOT/precision" --resume >> "$LOG" 2>&1
}

note "ISO PRECISION START"
for ST in 580 250; do
  # PN / PNL: full precision + compile grid (fp16 is in-scope here)
  for M in PhaseNet PhaseNetLight; do
    slip "$M" "$ST" fp32 0
    slip "$M" "$ST" fp16 0; slip "$M" "$ST" fp16 1
    slip "$M" "$ST" bf16 0; slip "$M" "$ST" bf16 1
  done
  # EQT / EQT-NC: fp16 unsafe -> fp32 + bf16(+compile) only
  for M in EQTransformer EQT-NC; do
    slip "$M" "$ST" fp32 0
    slip "$M" "$ST" bf16 0; slip "$M" "$ST" bf16 1
  done
done
note "ISO PRECISION DONE"
