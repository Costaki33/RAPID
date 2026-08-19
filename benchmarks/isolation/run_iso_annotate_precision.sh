#!/usr/bin/env bash
# Sequential Annotate precision comparison (FP32 / BF16 / FP16).
#
# Matrix:
#   models   : EQCCT, PhaseNet, PhaseNetLight, EQTransformer, EQT-NC
#   methods  : annotate_fp32, annotate_bf16, annotate_fp16  (EQT family skips fp16)
#   dataset  : STEAD only, 250 and 580 unique-trace nets (P/S in first 3001)
#   devices  : CPU cores {5,10,15,20} + 1 GPU
#   threads  : torch/OMP = n_cpus (SeisBench default for the allocated cores)
#              optional --thread-sweep adds {1,2,4,8} at 20 CPU cores
#   repeats  : 5 isolated subprocesses per cell
#
# Runtime times annotate() only. classify_aggregate is offline (pick quality).
# BF16/FP16 cells also compare picks to the sibling annotate_fp32 picks.json.
#
# Usage:
#   bash benchmarks/isolation/run_iso_annotate_precision.sh
#   bash benchmarks/isolation/run_iso_annotate_precision.sh --thread-sweep
#   bash benchmarks/isolation/run_iso_annotate_precision.sh --dry-run
#   RESULTS_ROOT=results/annotate_precision/stead_iso_YYYY-MM-DD \
#     bash benchmarks/isolation/run_iso_annotate_precision.sh
set -u
cd "$(dirname "$0")/../.."

PY="${RAPID_PYTHON:-$(command -v python3)}"
if [[ -x /home/skevofilaxc/miniconda3/envs/rapid/bin/python ]]; then
  PY=/home/skevofilaxc/miniconda3/envs/rapid/bin/python
fi

# Default: dated, self-contained tree so prior partial runs are never mixed in.
ROOT="${RESULTS_ROOT:-results/annotate_precision/stead_iso_$(date +%Y-%m-%d)}"
LOG=$ROOT/iso_annotate_precision.log
TAG=ann_prec
REPEATS=5
BATCH=256
THREAD_SWEEP=0
DRY=0
GPU_ID="${GPU_ID:-0}"

for arg in "$@"; do
  case "$arg" in
    --thread-sweep) THREAD_SWEEP=1 ;;
    --dry-run) DRY=1 ;;
    --repeats=*) REPEATS="${arg#*=}" ;;
    --results-root=*) ROOT="${arg#*=}"; LOG=$ROOT/iso_annotate_precision.log ;;
  esac
done

mkdir -p "$ROOT"
# Machine-readable pointer + human README for this run tree.
cat >"$ROOT/README.md" <<EOF
# Annotate precision isolation run

- Started: $(date -Is)
- Results root: \`$ROOT\`
- Dataset: STEAD only (250 / 580 unique-trace nets)
- Methods: annotate_fp32, annotate_bf16, annotate_fp16
- Models: EQCCT, PhaseNet, PhaseNetLight, EQTransformer, EQT-NC
- Devices: CPU **and** GPU cores {5,10,15,20}; batch sizes {64,128,256,512}
- GPU host cores still come from the assigned GPU slot (parallel runner) or 0..N-1 (this sequential script)
- Repeats: $REPEATS isolated subprocesses / cell
- Timing: SeisBench annotate only (classify_aggregate offline)
- Layout: \`<method>/stead/<N>st/<model>/<cpu|gpu>/cpus<N>/thr<N>/bs<B>/ann_prec/result.json\`
EOF
printf '%s\n' "$ROOT" >"$ROOT/RESULTS_ROOT.txt"

note() { echo "=== $(date +%Y-%m-%dT%H:%M:%S)  $* ===" | tee -a "$LOG"; }

cores_for() {
  local n=$1
  local i s=""
  for ((i=0; i<n; i++)); do
    [[ -n "$s" ]] && s+=","
    s+="$i"
  done
  echo "$s"
}

METHODS=(annotate_fp32 annotate_bf16 annotate_fp16)
MODELS=(EQCCT PhaseNet PhaseNetLight EQTransformer EQT-NC)
STATIONS=(250 580)
CPU_GRID=(5 10 15 20)
BATCH_GRID=(64 128 256 512)

run_one() {
  local method=$1 model=$2 st=$3 device=$4 ncpus=$5 batch=$6 thr=$7
  if [[ "$method" == "annotate_fp16" && ( "$model" == "EQTransformer" || "$model" == "EQT-NC" ) ]]; then
    note "SKIP $method / $model (fp16 unsafe)"
    return 0
  fi
  local cores; cores=$(cores_for "$ncpus")
  local thr_args=()
  if [[ -n "$thr" ]]; then
    thr_args=(--torch-threads "$thr")
  fi
  note "$method / $model / ${st}st / $device / cpus=$ncpus bs=$batch thr=${thr:-$ncpus} cores=$cores"
  if [[ "$DRY" == "1" ]]; then
    echo "DRY: taskset -c $cores $PY benchmarks/fair/run_annotate_precision_trial.py --method $method --model $model --n-stations $st --device $device --n-cpus $ncpus ${thr_args[*]} --core-list $cores --batch-size $batch --repeats $REPEATS --tag $TAG --results-root $ROOT --resume"
    return 0
  fi
  taskset -c "$cores" "$PY" benchmarks/fair/run_annotate_precision_trial.py \
    --method "$method" --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$ncpus" "${thr_args[@]}" \
    --core-list "$cores" --gpu-id "${GPU_ID}" \
    --batch-size "$batch" --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$LOG" 2>&1
}

note "ISO ANNOTATE PRECISION START (py=$PY root=$ROOT gpu=$GPU_ID)"

for ST in "${STATIONS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      for BS in "${BATCH_GRID[@]}"; do
        for C in "${CPU_GRID[@]}"; do
          run_one "$METHOD" "$MODEL" "$ST" cpu "$C" "$BS" ""
          run_one "$METHOD" "$MODEL" "$ST" gpu "$C" "$BS" ""
        done
      done
    done
  done
done

if [[ "$THREAD_SWEEP" == "1" ]]; then
  note "THREAD SWEEP START (cpu 20, bs=256)"
  for ST in "${STATIONS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for MODEL in "${MODELS[@]}"; do
        for THR in 1 2 4 8; do
          run_one "$METHOD" "$MODEL" "$ST" cpu 20 256 "$THR"
        done
      done
    done
  done
  note "THREAD SWEEP DONE"
fi

note "ISO ANNOTATE PRECISION DONE"
