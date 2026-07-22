#!/usr/bin/env bash
# Isolated fill for Figure 1 GPU panels + Batched-Classify thread sweep.
#
# 1) GPU core budget @ thr=1 for classify / annotate / slipstream (c in 5,10,15,20)
# 2) GPU thread sweep @ 20 cores for classify / annotate / slipstream (thr in 0,1,2,4,8)
# 3) Batched-Classify thread sweep @ 20 cores, CPU + GPU (thr in 0,1,2,4,8)
#
# Tags land under iso_full_benchmark/native so fig1 loaders can find them.
# --resume makes this safe to re-run. Strictly sequential.
set -u
cd "$(dirname "$0")/../.."
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate rapid

ROOT=results/iso_full_benchmark/native
LOG=results/fair_benchmark_iso/iso_gpu_native_and_batched_threads.log
mkdir -p results/fair_benchmark_iso "$ROOT"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
BUDGETS=(5 10 15 20)
THREADS=(0 1 2 4 8)
CORES20=$(seq -s, 0 19)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

gpu_core() {
  local meth=$1 model=$2 c=$3
  local ins; ins=$(insamples "$model")
  local cores; cores=$(seq -s, 0 $((c-1)))
  local tag dtype_args=()
  if [ "$meth" = "slipstream" ]; then
    tag="gpu_c${c}_thr1_fp32"
    dtype_args=(--dtype fp32)
  else
    tag="gpu_c${c}_thr1"
    dtype_args=(--dtype fp32)
  fi
  note "GPU-CORE $meth / $model / c${c} thr1"
  python3 benchmarks/fair/run_fair_trial.py \
    --method "$meth" --dataset stead --n-stations 580 --model "$model" \
    --device gpu --gpu-id 0 --n-cpus "$c" --core-list "$cores" --torch-threads 1 \
    --in-samples "$ins" --overlap-samples 0 "${dtype_args[@]}" --batch-size 256 \
    --repeats 3 --tag "$tag" --results-root "$ROOT" --resume >> "$LOG" 2>&1
}

gpu_thr() {
  local meth=$1 model=$2 thr=$3
  local ins; ins=$(insamples "$model")
  local tag dtype_args=(--dtype fp32)
  if [ "$meth" = "slipstream" ]; then
    tag="gpu_thr${thr}_fp32"
  else
    tag="gpu_thr${thr}"
  fi
  note "GPU-THR $meth / $model / thr${thr}"
  python3 benchmarks/fair/run_fair_trial.py \
    --method "$meth" --dataset stead --n-stations 580 --model "$model" \
    --device gpu --gpu-id 0 --n-cpus 20 --core-list "$CORES20" --torch-threads "$thr" \
    --in-samples "$ins" --overlap-samples 0 "${dtype_args[@]}" --batch-size 256 \
    --repeats 3 --tag "$tag" --results-root "$ROOT" --resume >> "$LOG" 2>&1
}

batched_thr() {
  local dev=$1 model=$2 thr=$3
  local ins; ins=$(insamples "$model")
  local tag="${dev}_c20_thr${thr}"
  note "BATCHED-THR $dev / $model / thr${thr}"
  python3 benchmarks/fair/run_fair_trial.py \
    --method classify_batched --dataset stead --n-stations 580 --model "$model" \
    --device "$dev" --gpu-id 0 --n-cpus 20 --core-list "$CORES20" --torch-threads "$thr" \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 --batch-size 256 \
    --repeats 3 --tag "$tag" --results-root "$ROOT" --resume >> "$LOG" 2>&1
}

note "GPU NATIVE + BATCHED THREAD FILL START"

# Fastest first: Batched-Classify thread sweep (CPU then GPU)
for M in "${MODELS[@]}"; do
  for T in "${THREADS[@]}"; do
    batched_thr cpu "$M" "$T"
  done
done
for M in "${MODELS[@]}"; do
  for T in "${THREADS[@]}"; do
    batched_thr gpu "$M" "$T"
  done
done

# GPU core budget @ thr=1 (classify already has gpu_c20_thr1; --resume skips)
for M in "${MODELS[@]}"; do
  for C in "${BUDGETS[@]}"; do
    for METH in classify annotate slipstream; do
      gpu_core "$METH" "$M" "$C"
    done
  done
done

# GPU thread sweep @ 20 cores.
# Skip thr0 (uncapped ~64) for Per-station Classify on EQT/EQT-NC: CPU already
# shows 1e3 s class blow-ups; re-running that on GPU would dominate wall time
# without changing the paper claim.
for M in "${MODELS[@]}"; do
  for T in "${THREADS[@]}"; do
    for METH in classify annotate slipstream; do
      if [ "$METH" = "classify" ] && [ "$T" = "0" ] && \
         { [ "$M" = "EQTransformer" ] || [ "$M" = "EQT-NC" ]; }; then
        note "SKIP GPU-THR classify / $M / thr0 (CPU default already documents blow-up)"
        continue
      fi
      gpu_thr "$METH" "$M" "$T"
    done
  done
done

# CPU Slipstream thr8 for EQT family (gap in Table 2 optima band)
for M in EQTransformer EQT-NC; do
  ins=$(insamples "$M")
  note "CPU-SLIP thr8 / $M"
  python3 benchmarks/fair/run_fair_trial.py \
    --method slipstream --dataset stead --n-stations 580 --model "$M" \
    --device cpu --n-cpus 20 --core-list "$CORES20" --torch-threads 8 \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 --batch-size 256 \
    --repeats 3 --tag "iso_thr8" --results-root "$ROOT" --resume >> "$LOG" 2>&1
done

note "GPU NATIVE + BATCHED THREAD FILL DONE"
