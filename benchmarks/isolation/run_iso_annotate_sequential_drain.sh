#!/usr/bin/env bash
# Drain the remaining sequential CPU cells onto free isolated core blocks.
# Does NOT touch the live EQTransformer 580 cpu c20 bs256 job on cores 0-19.
#
# Layout (machine has 0-127; original isolation used 0-59):
#   0-19   live EQT 580 c20 bs256 (leave alone)
#   20-39  EQT    580 c20 bs512
#   40-59  EQT-NC 580 c20 bs256
#   60-79  EQT-NC 580 c20 bs512
#   80-84  EQT    580 c5  bs512
#   85-89  EQT-NC 580 c5  bs256
#   90-94  EQT-NC 580 c5  bs512
set -u
cd "$(dirname "$0")/../.."

PY=/home/skevofilaxc/miniconda3/envs/rapid/bin/python
ROOT="${RESULTS_ROOT:-results/annotate_precision/stead_seq_bf16_2026-08-13}"
TAG=ann_seq
REPEATS=5
METHOD=annotate_bf16
PACKAGING=sequential

mkdir -p "$ROOT/parallel_logs"

note() { echo "=== $(date +%Y-%m-%dT%H:%M:%S)  $* ===" | tee -a "$ROOT/iso_sequential.log"; }

cores_from() {
  local base=$1 n=$2 i s=""
  for ((i=0; i<n; i++)); do
    [[ -n "$s" ]] && s+=","
    s+=$((base + i))
  done
  echo "$s"
}

run_trial() {
  local model=$1 st=$2 ncpus=$3 batch=$4 core_base=$5
  local cores; cores=$(cores_from "$core_base" "$ncpus")
  local slot_log="$ROOT/parallel_logs/${METHOD}_${model}_${st}st_cpu_c${ncpus}_bs${batch}_drain.log"
  note "DRAIN START $METHOD / $model / ${st}st / cpu / cpus=$ncpus bs=$batch cores=$cores"
  taskset -c "$cores" "$PY" benchmarks/fair/run_annotate_precision_trial.py \
    --method "$METHOD" --model "$model" --n-stations "$st" \
    --device cpu --n-cpus "$ncpus" --packaging "$PACKAGING" \
    --core-list "$cores" --gpu-id 0 \
    --batch-size "$batch" --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  note "DRAIN DONE  $METHOD / $model / ${st}st / cpu / cpus=$ncpus bs=$batch rc=$?"
}

# Three slow c20 cells on disjoint 20-core blocks (not 0-19).
run_trial EQTransformer 580 20 512 20 &
P1=$!
run_trial EQT-NC 580 20 256 40 &
P2=$!
run_trial EQT-NC 580 20 512 60 &
P3=$!

# Three fast c5 cells on leftover cores.
run_trial EQTransformer 580 5 512 80 &
P4=$!
run_trial EQT-NC 580 5 256 85 &
P5=$!
run_trial EQT-NC 580 5 512 90 &
P6=$!

note "DRAIN PIDS c20=$P1,$P2,$P3 c5=$P4,$P5,$P6 (live EQT c20 bs256 left on 0-19)"
wait "$P1" "$P2" "$P3" "$P4" "$P5" "$P6"
note "DRAIN ALL DONE"
