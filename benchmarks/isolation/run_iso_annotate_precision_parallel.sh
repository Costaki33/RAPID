#!/usr/bin/env bash
# Parallel Annotate-precision isolation runner.
#
# Core / device layout (disjoint, so CPU + both GPUs can run together):
#   CPU-only trials : cores 0-19   (trial uses first n_cpus of that block)
#   GPU 1 trials    : cores 20-39 + CUDA device 1  (first n_cpus of block)
#   GPU 0 trials    : cores 40-59 + CUDA device 0  (first n_cpus of block)
#
# Matrix dimensions:
#   methods      : annotate_fp32 / bf16 / fp16  (EQT family skips fp16)
#   models       : EQCCT, PhaseNet, PhaseNetLight, EQTransformer, EQT-NC
#   stations     : 250, 580 (STEAD)
#   n_cpus       : 5, 10, 15, 20   ← CPU and GPU both sweep this
#   batch_size   : 64, 128, 256, 512
#   repeats      : 5 isolated subprocesses / cell
#
# Usage:
#   RESULTS_ROOT=results/annotate_precision/stead_iso_2026-08-13 \
#     bash benchmarks/isolation/run_iso_annotate_precision_parallel.sh
set -u
cd "$(dirname "$0")/../.."

PY="${RAPID_PYTHON:-$(command -v python3)}"
if [[ -x /home/skevofilaxc/miniconda3/envs/rapid/bin/python ]]; then
  PY=/home/skevofilaxc/miniconda3/envs/rapid/bin/python
fi

ROOT="${RESULTS_ROOT:-results/annotate_precision/stead_iso_$(date +%Y-%m-%d)}"
TAG=ann_prec
REPEATS=5
LOG_DIR="$ROOT/parallel_logs"
mkdir -p "$ROOT" "$LOG_DIR"

METHODS=(annotate_fp32 annotate_bf16 annotate_fp16)
MODELS=(EQCCT PhaseNet PhaseNetLight EQTransformer EQT-NC)
STATIONS=(250 580)
CPU_GRID=(5 10 15 20)
BATCH_GRID=(64 128 256 512)

CPU_BASE=0
GPU1_BASE=20   # GPU_ID=1
GPU0_BASE=40   # GPU_ID=0

cores_from() {
  local base=$1 n=$2 i s=""
  for ((i=0; i<n; i++)); do
    [[ -n "$s" ]] && s+=","
    s+=$((base + i))
  done
  echo "$s"
}

is_done() {
  local method=$1 model=$2 st=$3 device=$4 ncpus=$5 batch=$6
  local thr=$ncpus
  local modern="$ROOT/$method/stead/${st}st/$model/$device/cpus${ncpus}/thr${thr}/bs${batch}/$TAG/result.json"
  local legacy="$ROOT/$method/stead/${st}st/$model/$device/cpus${ncpus}/thr${thr}/$TAG/result.json"
  local f=""
  if [[ -f "$modern" ]]; then
    f="$modern"
  elif [[ "$batch" == "256" && -f "$legacy" ]]; then
    f="$legacy"
  else
    return 1
  fi
  "$PY" - "$f" <<'PY'
import json, sys
r=json.loads(open(sys.argv[1]).read())
sr=(r.get("timing") or {}).get("success_rate") or 0
sys.exit(0 if sr >= 1.0 else 1)
PY
}

note() { echo "=== $(date +%Y-%m-%dT%H:%M:%S)  $* ===" | tee -a "$ROOT/iso_annotate_precision_parallel.log"; }

run_trial() {
  local method=$1 model=$2 st=$3 device=$4 ncpus=$5 batch=$6 core_base=$7 gpu_id=$8
  local cores; cores=$(cores_from "$core_base" "$ncpus")
  local slot_log="$LOG_DIR/${method}_${model}_${st}st_${device}_c${ncpus}_bs${batch}_gpu${gpu_id}.log"
  note "START $method / $model / ${st}st / $device / cpus=$ncpus bs=$batch cores=$cores gpu_id=$gpu_id"
  taskset -c "$cores" "$PY" benchmarks/fair/run_annotate_precision_trial.py \
    --method "$method" --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$ncpus" \
    --core-list "$cores" --gpu-id "$gpu_id" \
    --batch-size "$batch" --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  local rc=$?
  note "DONE  $method / $model / ${st}st / $device / cpus=$ncpus bs=$batch rc=$rc"
  return $rc
}

# Build remaining queues: method|model|st|device|ncpus|batch
CPU_QUEUE=()
GPU_QUEUE=()
for ST in "${STATIONS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      if [[ "$METHOD" == "annotate_fp16" && ( "$MODEL" == "EQTransformer" || "$MODEL" == "EQT-NC" ) ]]; then
        continue
      fi
      for BS in "${BATCH_GRID[@]}"; do
        for C in "${CPU_GRID[@]}"; do
          if ! is_done "$METHOD" "$MODEL" "$ST" cpu "$C" "$BS"; then
            CPU_QUEUE+=("$METHOD|$MODEL|$ST|cpu|$C|$BS")
          fi
          if ! is_done "$METHOD" "$MODEL" "$ST" gpu "$C" "$BS"; then
            GPU_QUEUE+=("$METHOD|$MODEL|$ST|gpu|$C|$BS")
          fi
        done
      done
    done
  done
done

cat >"$ROOT/README_MATRIX.md" <<EOF
# Annotate precision matrix

- Results root: \`$ROOT\`
- Methods: fp32 / bf16 / fp16 (EQT family skips fp16)
- Models: EQCCT, PhaseNet, PhaseNetLight, EQTransformer, EQT-NC
- Stations: STEAD 250 / 580
- CPU counts (CPU **and** GPU): ${CPU_GRID[*]}
- Batch sizes: ${BATCH_GRID[*]}
- Repeats: $REPEATS / cell
- Layout: CPU cores 0-19; GPU1 cores 20-39; GPU0 cores 40-59
- Path: \`<method>/stead/<N>st/<model>/<cpu|gpu>/cpus<N>/thr<N>/bs<B>/ann_prec/result.json\`
EOF

note "PARALLEL START root=$ROOT cpu_queue=${#CPU_QUEUE[@]} gpu_queue=${#GPU_QUEUE[@]} (total remaining $(( ${#CPU_QUEUE[@]} + ${#GPU_QUEUE[@]} )))"
note "Layout: CPU ${CPU_BASE}-$((CPU_BASE+19)); GPU1 ${GPU1_BASE}-$((GPU1_BASE+19)); GPU0 ${GPU0_BASE}-$((GPU0_BASE+19))"
note "Grids: cpus=(${CPU_GRID[*]})  batch=(${BATCH_GRID[*]})"

gpu_slot_for_index() {
  local i=$1
  if (( i % 2 == 0 )); then
    echo "1 $GPU1_BASE"
  else
    echo "0 $GPU0_BASE"
  fi
}

cpu_worker() {
  local item method model st device ncpus batch
  for item in "${CPU_QUEUE[@]}"; do
    IFS='|' read -r method model st device ncpus batch <<<"$item"
    run_trial "$method" "$model" "$st" "$device" "$ncpus" "$batch" "$CPU_BASE" 0 || true
  done
  note "CPU WORKER DRAINED"
}

gpu_worker() {
  local worker_id=$1
  local idx=0 item method model st device ncpus batch gpu_id core_base
  for item in "${GPU_QUEUE[@]}"; do
    if (( idx % 2 == worker_id )); then
      IFS='|' read -r method model st device ncpus batch <<<"$item"
      read -r gpu_id core_base <<<"$(gpu_slot_for_index "$idx")"
      run_trial "$method" "$model" "$st" "$device" "$ncpus" "$batch" "$core_base" "$gpu_id" || true
    fi
    idx=$((idx + 1))
  done
  note "GPU WORKER $worker_id DRAINED"
}

cpu_worker &
CPU_PID=$!
gpu_worker 0 &
GPU1_PID=$!
gpu_worker 1 &
GPU0_PID=$!

note "PIDs cpu=$CPU_PID gpu1_worker=$GPU1_PID gpu0_worker=$GPU0_PID"
wait "$CPU_PID" "$GPU1_PID" "$GPU0_PID"
note "PARALLEL DONE"
