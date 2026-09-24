#!/usr/bin/env bash
# Isolated Model-Actor / Ripper campaign using native merged annotate() winners.
#
# Stages:
#   A  playback MA SG K=1, native dtype + batch + T
#   B  playback MA+Ripper SG, 1 thread/worker, native dtype+batch
#      580: full K ladder (capped at n_cpus); 250: K in {1,2}
#      GPU host-CPU budgets match CPU: 5, 10, 15, 20 (first n of the GPU slot)
#   C  staggered eager at each B group's playback elbow (smallest K within 10%)
#
# Isolation (do not overlap other RAPID jobs):
#   CPU  : cores 0-19   (first n_cpus of the block)
#   GPU1 : cores 20-39 + CUDA 1
#   GPU0 : cores 40-59 + CUDA 0
#
# New tree — does NOT write to orch_annotate/stead_2026-08-14.
#
# Usage:
#   STAGES=A,B,C bash benchmarks/isolation/run_iso_orch_native_winners.sh
#   DRY_RUN=1 bash benchmarks/isolation/run_iso_orch_native_winners.sh
#   SMOKE=1 REPEATS=1 bash benchmarks/isolation/run_iso_orch_native_winners.sh
#
# Detached (survives logout / Cursor shell teardown):
#   RESULTS_ROOT=results/orch_annotate/stead_iso_YYYY-MM-DD_native_winners \
#     nohup setsid bash benchmarks/isolation/run_iso_orch_native_winners.sh \
#     </dev/null >>"$RESULTS_ROOT/iso_orch_native_winners.nohup.out" 2>&1 &
set -u
trap '' HUP
cd "$(dirname "$0")/../.."

PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
ROOT="${RESULTS_ROOT:-results/orch_annotate/stead_iso_$(date +%Y-%m-%d)_native_winners}"
NATIVE_ROOT="${NATIVE_ROOT:-results/annotate_config/stead_iso_2026-09-08}"
WINNERS="${WINNERS:-$NATIVE_ROOT/winners_merged_quality_safe.json}"
TAG=orch_ann
REPEATS="${REPEATS:-5}"
STAGES="${STAGES:-A,B,C}"
CELL_TIMEOUT_S="${CELL_TIMEOUT_S:-3600}"
TIMEOUT_BIN="${TIMEOUT_BIN:-/usr/bin/timeout}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

CPU_BASE=0
GPU1_BASE=20
GPU0_BASE=40

mkdir -p "$ROOT" "$ROOT/parallel_logs" results/orch_annotate
printf '%s\n' "$ROOT" > results/orch_annotate/NATIVE_WINNERS_ROOT.txt

if [[ ! -f "$WINNERS" ]]; then
  echo "exporting native winners -> $WINNERS"
  PYTHONPATH="." "$PY" scripts/export_annotate_config_winners.py "$NATIVE_ROOT"
fi
if [[ ! -f "$WINNERS" ]]; then
  echo "ERROR: missing $WINNERS" >&2
  exit 2
fi

cores_from() {
  local base=$1 n=$2 i s=""
  for ((i=0; i<n; i++)); do
    [[ -n "$s" ]] && s+=","
    s+=$((base + i))
  done
  echo "$s"
}

note() { echo "=== $(date +%Y-%m-%dT%H:%M:%S)  $* ===" | tee -a "$ROOT/iso_orch_native_winners.log"; }

run_trial() {
  local ncpus=$1 comp=$2 method=$3 model=$4 st=$5 device=$6 kma=$7 krp=$8 pack=$9
  local arrival=${10} fill=${11} batch=${12} thr=${13} stage=${14} core_base=${15} gpu_id=${16}
  local cores; cores=$(cores_from "$core_base" "$ncpus")
  local slot_log="$ROOT/parallel_logs/s${stage}_c${ncpus}_${comp}_${method}_${model}_${st}st_${device}_kma${kma}_krp${krp}_${pack}_${arrival}_${fill}_bs${batch}_thr${thr}_gpu${gpu_id}.log"
  note "START stage=$stage cpus=$ncpus $comp $method $model ${st}st $device kma=$kma krp=$krp $pack $arrival $fill bs=$batch thr=$thr cores=$cores gpu=$gpu_id"
  if [[ "$DRY_RUN" == "1" ]]; then
    note "DRY_RUN skip"
    return 0
  fi
  "$TIMEOUT_BIN" --signal=KILL "${CELL_TIMEOUT_S}" taskset -c "$cores" "$PY" benchmarks/fair/run_orch_annotate_trial.py \
    --composition "$comp" --method "$method" --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$ncpus" --k-ma "$kma" --k-rp "$krp" \
    --packaging "$pack" --arrival "$arrival" --fill "$fill" \
    --batch-size "$batch" --torch-threads "$thr" --winner-paths \
    --core-list "$cores" --gpu-id "$gpu_id" \
    --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  local rc=$?
  note "DONE  stage=$stage cpus=$ncpus $comp $method $model ${st}st $device kma=$kma krp=$krp $pack $arrival $fill bs=$batch thr=$thr rc=$rc"
  return $rc
}

write_queue() {
  local stage=$1 dest=$2
  PYTHONPATH="benchmarks/fair:." "$PY" - "$ROOT" "$WINNERS" "$TAG" "$stage" "$SMOKE" <<'PY' >"$dest"
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
winners_path = Path(sys.argv[2])
tag = sys.argv[3]
stage = sys.argv[4]
smoke = sys.argv[5] == "1"

from native_winner_orch import (
    iter_stage,
    load_winners,
    winner_cell_dir,
    cell_queue_line,
)

winners = load_winners(winners_path)
cells = list(iter_stage(stage, winners, results_root=root, tag=tag))
if smoke:
    keep = []
    for c in cells:
        if c["model"] != "PhaseNet" or int(c["n_stations"]) != 580:
            continue
        if c["composition"] != "ma":
            continue
        if int(c["k_ma"]) != 1:
            continue
        if c["device"] == "cpu" and int(c["n_cpus"]) == 20:
            keep.append(c)
        elif c["device"] == "gpu" and int(c["n_cpus"]) == 20:
            keep.append(c)
    cells = keep[:2]

print(f"N_CELLS {len(cells)}")
n_skip = 0
for c in cells:
    p = winner_cell_dir(root, c, tag=tag) / "result.json"
    done = False
    if p.is_file():
        try:
            r = json.loads(p.read_text())
            done = float((r.get("timing") or {}).get("success_rate") or 0) >= 1.0
        except Exception:
            done = False
    if done:
        n_skip += 1
        continue
    print(f"{c['device'].upper()} {cell_queue_line(c)}")
print(f"N_SKIP {n_skip}")
PY
}

drain_stage() {
  local stage=$1
  local queues="$ROOT/.queues.${stage}.$$"
  write_queue "$stage" "$queues"
  local CPU_QUEUE=() GPU_QUEUE=() n_cells=0 n_skip=0
  while read -r kind rest; do
    [[ -z "${kind:-}" ]] && continue
    case "$kind" in
      N_CELLS) n_cells=$rest ;;
      N_SKIP) n_skip=$rest ;;
      CPU) CPU_QUEUE+=("$rest") ;;
      GPU) GPU_QUEUE+=("$rest") ;;
    esac
  done <"$queues"
  rm -f "$queues"
  note "STAGE $stage cells=$n_cells skip_done=$n_skip remaining cpu=${#CPU_QUEUE[@]} gpu=${#GPU_QUEUE[@]} root=$ROOT repeats=$REPEATS smoke=$SMOKE"

  if [[ "$DRY_RUN" == "1" ]]; then
    note "DRY_RUN=1 — stage $stage not launching"
    return 0
  fi
  if [[ ${#CPU_QUEUE[@]} -eq 0 && ${#GPU_QUEUE[@]} -eq 0 ]]; then
    note "STAGE $stage nothing to run"
    return 0
  fi

  cpu_worker() {
    local item ncpus comp method model st device kma krp pack arrival fill batch thr stg
    for item in "${CPU_QUEUE[@]}"; do
      IFS='|' read -r ncpus comp method model st device kma krp pack arrival fill batch thr stg <<<"$item"
      run_trial "$ncpus" "$comp" "$method" "$model" "$st" "$device" "$kma" "$krp" "$pack" "$arrival" "$fill" "$batch" "$thr" "$stg" "$CPU_BASE" 0 || true
    done
    note "CPU WORKER DRAINED stage=$stage"
  }
  gpu_worker() {
    local worker_id=$1 idx=0 item ncpus comp method model st device kma krp pack arrival fill batch thr stg gpu_id core_base
    for item in "${GPU_QUEUE[@]}"; do
      if (( idx % 2 == worker_id )); then
        IFS='|' read -r ncpus comp method model st device kma krp pack arrival fill batch thr stg <<<"$item"
        if (( idx % 2 == 0 )); then gpu_id=1; core_base=$GPU1_BASE; else gpu_id=0; core_base=$GPU0_BASE; fi
        run_trial "$ncpus" "$comp" "$method" "$model" "$st" "$device" "$kma" "$krp" "$pack" "$arrival" "$fill" "$batch" "$thr" "$stg" "$core_base" "$gpu_id" || true
      fi
      idx=$((idx + 1))
    done
    note "GPU WORKER $worker_id DRAINED stage=$stage"
  }

  cpu_worker &
  local CPU_PID=$!
  gpu_worker 0 &
  local GPU1_PID=$!
  gpu_worker 1 &
  local GPU0_PID=$!
  note "PIDs stage=$stage cpu=$CPU_PID gpu1=$GPU1_PID gpu0=$GPU0_PID"
  wait "$CPU_PID" "$GPU1_PID" "$GPU0_PID"
  note "STAGE $stage DONE"
}

cat > "$ROOT/README.md" <<EOF
# Native-winner orchestration campaign

Native floors: \`$NATIVE_ROOT\`
Winners: \`$WINNERS\`
Repeats: $REPEATS
Stages: $STAGES

- A: MA SG K=1 playback, native T/B/dtype
- B: MA+Ripper SG playback, 1 thread/worker; 580 full K, 250 K=1,2; CPU and GPU cpus=5,10,15,20
- C: staggered eager at playback elbow (10%)
EOF

note "ORCH NATIVE WINNERS start stages=$STAGES root=$ROOT winners=$WINNERS repeats=$REPEATS timeout=${CELL_TIMEOUT_S}s"

IFS=',' read -r -a STAGE_ARR <<<"$STAGES"
for stage in "${STAGE_ARR[@]}"; do
  stage=$(echo "$stage" | tr -d '[:space:]' | tr '[:lower:]' '[:upper:]')
  [[ -z "$stage" ]] && continue
  drain_stage "$stage"
done

note "ORCH NATIVE WINNERS DONE stages=$STAGES"
