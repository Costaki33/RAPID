#!/usr/bin/env bash
# Sequential per-station Annotate baseline — winning pipeline from the merged study.
#
# Winner: annotate_bf16
# Thin config grid (orch-relevant twin of one-station-per-actor):
#   models   : EQCCT, PhaseNet, PhaseNetLight, EQTransformer, EQT-NC
#   stations : STEAD 250 / 580
#   devices  : cpu + gpu (parallel slots)
#   n_cpus   : 5, 20
#   batch    : 256, 512
#   packaging: sequential  (one annotate() per station)
#   repeats  : 5
#
# Core layout (same as merged parallel runner):
#   CPU  : cores 0-19
#   GPU1 : cores 20-39 + CUDA 1
#   GPU0 : cores 40-59 + CUDA 0
set -u
cd "$(dirname "$0")/../.."

PY=/home/skevofilaxc/miniconda3/envs/rapid/bin/python
ROOT="${RESULTS_ROOT:-results/annotate_precision/stead_seq_bf16_2026-08-13}"
TAG=ann_seq
REPEATS=5
METHOD=annotate_bf16
PACKAGING=sequential

MODELS=(EQCCT PhaseNet PhaseNetLight EQTransformer EQT-NC)
STATIONS=(250 580)
CPU_GRID=(5 20)
BATCH_GRID=(256 512)

CPU_BASE=0
GPU1_BASE=20
GPU0_BASE=40

mkdir -p "$ROOT" "$ROOT/parallel_logs"

cores_from() {
  local base=$1 n=$2 i s=""
  for ((i=0; i<n; i++)); do
    [[ -n "$s" ]] && s+=","
    s+=$((base + i))
  done
  echo "$s"
}

is_done() {
  local model=$1 st=$2 device=$3 ncpus=$4 batch=$5
  local f="$ROOT/$METHOD/stead/${st}st/$model/$device/cpus${ncpus}/thr${ncpus}/bs${batch}/$PACKAGING/$TAG/result.json"
  [[ -f "$f" ]] || return 1
  "$PY" - "$f" <<'PY'
import json, sys
r=json.loads(open(sys.argv[1]).read())
sys.exit(0 if float((r.get("timing") or {}).get("success_rate") or 0) >= 1.0 else 1)
PY
}

note() { echo "=== $(date +%Y-%m-%dT%H:%M:%S)  $* ===" | tee -a "$ROOT/iso_sequential.log"; }

run_trial() {
  local model=$1 st=$2 device=$3 ncpus=$4 batch=$5 core_base=$6 gpu_id=$7
  local cores; cores=$(cores_from "$core_base" "$ncpus")
  local slot_log="$ROOT/parallel_logs/${METHOD}_${model}_${st}st_${device}_c${ncpus}_bs${batch}_gpu${gpu_id}.log"
  note "START $METHOD / $model / ${st}st / $device / cpus=$ncpus bs=$batch packaging=$PACKAGING cores=$cores gpu=$gpu_id"
  taskset -c "$cores" "$PY" benchmarks/fair/run_annotate_precision_trial.py \
    --method "$METHOD" --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$ncpus" --packaging "$PACKAGING" \
    --core-list "$cores" --gpu-id "$gpu_id" \
    --batch-size "$batch" --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  note "DONE  $METHOD / $model / ${st}st / $device / cpus=$ncpus bs=$batch rc=$?"
}

cat >"$ROOT/README.md" <<EOF
# Sequential per-station Annotate (winning pipeline)

- Pipeline: **annotate_bf16** (winner of merged-network precision study)
- Packaging: **sequential** — one \`annotate()\` per station
- Purpose: native twin of orchestration "one station per actor"
- Compared later against merged bf16 results in \`stead_iso_2026-08-13\`
- Grid: models×{250,580}×{cpu,gpu}×cpus{5,20}×bs{256,512}, 5 repeats
- Cores: CPU 0-19; GPU1 20-39; GPU0 40-59
EOF

CPU_QUEUE=()
GPU_QUEUE=()
for ST in "${STATIONS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for BS in "${BATCH_GRID[@]}"; do
      for C in "${CPU_GRID[@]}"; do
        if ! is_done "$MODEL" "$ST" cpu "$C" "$BS"; then
          CPU_QUEUE+=("$MODEL|$ST|cpu|$C|$BS")
        fi
        if ! is_done "$MODEL" "$ST" gpu "$C" "$BS"; then
          GPU_QUEUE+=("$MODEL|$ST|gpu|$C|$BS")
        fi
      done
    done
  done
done

note "SEQ START root=$ROOT method=$METHOD cpu_q=${#CPU_QUEUE[@]} gpu_q=${#GPU_QUEUE[@]}"

cpu_worker() {
  local item model st device ncpus batch
  for item in "${CPU_QUEUE[@]}"; do
    IFS='|' read -r model st device ncpus batch <<<"$item"
    run_trial "$model" "$st" "$device" "$ncpus" "$batch" "$CPU_BASE" 0 || true
  done
  note "CPU WORKER DRAINED"
}

gpu_worker() {
  local worker_id=$1 idx=0 item model st device ncpus batch gpu_id core_base
  for item in "${GPU_QUEUE[@]}"; do
    if (( idx % 2 == worker_id )); then
      IFS='|' read -r model st device ncpus batch <<<"$item"
      if (( idx % 2 == 0 )); then gpu_id=1; core_base=$GPU1_BASE; else gpu_id=0; core_base=$GPU0_BASE; fi
      run_trial "$model" "$st" "$device" "$ncpus" "$batch" "$core_base" "$gpu_id" || true
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
note "PIDs cpu=$CPU_PID gpu1=$GPU1_PID gpu0=$GPU0_PID"
wait "$CPU_PID" "$GPU1_PID" "$GPU0_PID"
note "SEQ DONE"
