#!/usr/bin/env bash
# Locked-recipe transfer suite: native merged Annotate, playback MA+SG, staggered MA+SG eager.
#
# Do not add fp16, Ripper, hybrid, S1, or wait-5/wait-10.
# Full brief: RAPID/docs/RAPID_LOCKED_RECIPE_TRANSFER.md
#
# Env:
#   LAYER=all|native|playback|staggered   (default all)
#   MODELS=EQCCT,PhaseNet,...             (default all five)
#   STATIONS=250,580
#   DEVICES=cpu,gpu
#   CORE_GRID=5,10,15,20
#   SKIP_GPU=1 / SKIP_CPU=1
#   CPU_K_CAP / GPU_K_CAP                 (optional actor caps)
#   REPEATS=5
#   SMOKE=1                               (PhaseNet, 250st, 5 cores, 1 repeat)
#   DRY_RUN=1
#   PARALLEL=1                            (CPU queue + GPU queue; default is serial)
#   CPU_BASE=0  GPU_BASE=0  GPU_ID=0
#   CORES=0,1,2,...                       (explicit affinity; first n_cpus used)
#   RESULTS_ROOT=results/locked_recipe_transfer/<name>
#   RAPID_PYTHON=...
set -u
cd "$(dirname "$0")/../.."

PY="${RAPID_PYTHON:-$(command -v python3)}"
if [[ -x /home/skevofilaxc/miniconda3/envs/rapid/bin/python && -z "${RAPID_PYTHON:-}" ]]; then
  PY=/home/skevofilaxc/miniconda3/envs/rapid/bin/python
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  export LAYER="${LAYER:-all}"
  export MODELS="${MODELS:-PhaseNet}"
  export STATIONS="${STATIONS:-250}"
  export CORE_GRID="${CORE_GRID:-5}"
  REPEATS="${REPEATS:-1}"
fi

LAYER="${LAYER:-all}"
REPEATS="${REPEATS:-5}"
TAG=xfer
CPU_BASE="${CPU_BASE:-0}"
GPU_BASE="${GPU_BASE:-0}"
GPU_ID="${GPU_ID:-0}"
ROOT="${RESULTS_ROOT:-results/locked_recipe_transfer/$(hostname -s)_$(date +%Y-%m-%d)}"
LOG="$ROOT/locked_recipe_transfer.log"
mkdir -p "$ROOT/parallel_logs"

export LAYER MODELS="${MODELS:-}" STATIONS="${STATIONS:-}" DEVICES="${DEVICES:-}" \
  CORE_GRID="${CORE_GRID:-}" SKIP_GPU="${SKIP_GPU:-}" SKIP_CPU="${SKIP_CPU:-}" \
  CPU_K_CAP="${CPU_K_CAP:-}" GPU_K_CAP="${GPU_K_CAP:-}"

note() { echo "=== $(date +%Y-%m-%dT%H:%M:%S)  $* ===" | tee -a "$LOG"; }

cores_for() {
  local n=$1 base=$2
  local i s="" id
  if [[ -n "${CORES:-}" ]]; then
    IFS=',' read -ra ALL <<<"$CORES"
    if (( ${#ALL[@]} < n )); then
      echo ""
      return 1
    fi
    for ((i=0; i<n; i++)); do
      [[ -n "$s" ]] && s+=","
      s+="${ALL[$i]}"
    done
    echo "$s"
    return 0
  fi
  for ((i=0; i<n; i++)); do
    [[ -n "$s" ]] && s+=","
    s+=$((base + i))
  done
  echo "$s"
}

is_done() {
  local p=$1
  [[ -f "$p" ]] || return 1
  "$PY" - "$p" <<'PY'
import json, sys
r=json.loads(open(sys.argv[1]).read())
sr=(r.get("timing") or {}).get("success_rate") or 0
sys.exit(0 if float(sr) >= 1.0 else 1)
PY
}

native_path() {
  local model=$1 st=$2 device=$3 ncpus=$4
  echo "$ROOT/annotate_bf16/stead/${st}st/$model/$device/cpus${ncpus}/thr${ncpus}/bs512/merged/$TAG/result.json"
}

orch_path() {
  local model=$1 st=$2 device=$3 kma=$4 arrival=$5 fill=$6
  echo "$ROOT/ma/annotate_bf16/stead/${st}st/$model/$device/kma${kma}_krp0/sg/${arrival}/${fill}/bs512/$TAG/result.json"
}

run_native() {
  local model=$1 st=$2 device=$3 ncpus=$4 cores=$5 gpu_id=$6
  local p; p=$(native_path "$model" "$st" "$device" "$ncpus")
  if is_done "$p"; then
    note "SKIP native $model ${st}st $device cpus=$ncpus"
    return 0
  fi
  local slot_log="$ROOT/parallel_logs/native_${model}_${st}st_${device}_c${ncpus}.log"
  note "START native $model ${st}st $device cpus=$ncpus cores=$cores gpu=$gpu_id"
  taskset -c "$cores" "$PY" benchmarks/fair/run_annotate_precision_trial.py \
    --method annotate_bf16 --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$ncpus" --torch-threads "$ncpus" \
    --core-list "$cores" --gpu-id "$gpu_id" \
    --batch-size 512 --packaging merged --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  local rc=$?
  note "DONE  native $model ${st}st $device cpus=$ncpus rc=$rc"
  return $rc
}

run_orch() {
  local model=$1 st=$2 device=$3 ncpus=$4 kma=$5 arrival=$6 fill=$7 cores=$8 gpu_id=$9
  local p; p=$(orch_path "$model" "$st" "$device" "$kma" "$arrival" "$fill")
  if is_done "$p"; then
    note "SKIP $arrival $model ${st}st $device k=$kma cpus=$ncpus"
    return 0
  fi
  local slot_log="$ROOT/parallel_logs/${arrival}_${model}_${st}st_${device}_k${kma}_c${ncpus}.log"
  note "START $arrival $model ${st}st $device kma=$kma cpus=$ncpus cores=$cores gpu=$gpu_id fill=$fill"
  taskset -c "$cores" "$PY" benchmarks/fair/run_orch_annotate_trial.py \
    --composition ma --method annotate_bf16 --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$ncpus" --k-ma "$kma" --k-rp 0 \
    --packaging sg --arrival "$arrival" --fill "$fill" \
    --core-list "$cores" --gpu-id "$gpu_id" \
    --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  local rc=$?
  note "DONE  $arrival $model ${st}st $device kma=$kma rc=$rc"
  return $rc
}

run_cell() {
  local layer=$1 model=$2 st=$3 device=$4 ncpus=$5 kma=$6 arrival=$7 fill=$8
  local base=$CPU_BASE gpu_id=0
  if [[ "$device" == "gpu" ]]; then
    base=$GPU_BASE
    gpu_id=$GPU_ID
  fi
  local cores
  if ! cores=$(cores_for "$ncpus" "$base"); then
    note "SKIP $layer $model ${st}st $device cpus=$ncpus (not enough CORES)"
    return 0
  fi
  if [[ "$layer" == "native" ]]; then
    run_native "$model" "$st" "$device" "$ncpus" "$cores" "$gpu_id" || true
  else
    run_orch "$model" "$st" "$device" "$ncpus" "$kma" "$arrival" "$fill" "$cores" "$gpu_id" || true
  fi
}

MATRIX=$("$PY" benchmarks/fair/locked_recipe_transfer_matrix.py --print-matrix)
N_CELLS=$(printf '%s\n' "$MATRIX" | awk '/^N_CELLS/{print $2}')
CPU_ITEMS=()
GPU_ITEMS=()
while IFS='|' read -r layer model st device ncpus kma arrival fill; do
  [[ "$layer" == "N_CELLS "* ]] && continue
  [[ -z "${layer:-}" || "$layer" == N_CELLS ]] && continue
  item="$layer|$model|$st|$device|$ncpus|$kma|$arrival|$fill"
  if [[ "$device" == "cpu" ]]; then
    CPU_ITEMS+=("$item")
  else
    GPU_ITEMS+=("$item")
  fi
done < <(printf '%s\n' "$MATRIX" | awk 'NR>1 && $0 !~ /^N_CELLS/')

cp benchmarks/fair/locked_recipe_transfer_matrix.py "$ROOT/" 2>/dev/null || true
cat >"$ROOT/README.md" <<EOF
# Locked-recipe transfer run

- Started: $(date -Is)
- Host: $(hostname)
- Results: \`$ROOT\`
- Layer: $LAYER
- Repeats: $REPEATS
- Cells in this matrix: $N_CELLS
- CPU base: $CPU_BASE  GPU base: $GPU_BASE  GPU_ID: $GPU_ID
- SMOKE=${SMOKE:-0} PARALLEL=${PARALLEL:-0}

Read \`RAPID/docs/RAPID_LOCKED_RECIPE_TRANSFER.md\` before changing the matrix.
EOF

note "TRANSFER START py=$PY root=$ROOT cells=$N_CELLS cpu=${#CPU_ITEMS[@]} gpu=${#GPU_ITEMS[@]} layer=$LAYER repeats=$REPEATS smoke=${SMOKE:-0} parallel=${PARALLEL:-0}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  note "DRY_RUN=1 — not launching. First 12 cells:"
  printf '%s\n' "$MATRIX" | head -n 13 | tee -a "$LOG"
  exit 0
fi

drain() {
  local item layer model st device ncpus kma arrival fill
  for item in "$@"; do
    IFS='|' read -r layer model st device ncpus kma arrival fill <<<"$item"
    run_cell "$layer" "$model" "$st" "$device" "$ncpus" "$kma" "$arrival" "$fill"
  done
}

if [[ "${PARALLEL:-0}" == "1" ]]; then
  drain "${CPU_ITEMS[@]+"${CPU_ITEMS[@]}"}" &
  CPU_PID=$!
  drain "${GPU_ITEMS[@]+"${GPU_ITEMS[@]}"}" &
  GPU_PID=$!
  note "PIDs cpu=$CPU_PID gpu=$GPU_PID"
  wait "$CPU_PID" "$GPU_PID"
else
  drain "${CPU_ITEMS[@]+"${CPU_ITEMS[@]}"}" "${GPU_ITEMS[@]+"${GPU_ITEMS[@]}"}"
fi

note "TRANSFER DONE"
"$PY" benchmarks/fair/locked_recipe_transfer_matrix.py --status "$ROOT" | tee -a "$LOG"
