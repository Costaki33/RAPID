#!/usr/bin/env bash
# Isolated orchestration annotate-precision matrix.
#
# Do NOT start until sequential native (run_iso_annotate_sequential_bf16.sh)
# has finished — they share cores 0-19 / 20-39 / 40-59.
#
# See benchmarks/isolation/README_ORCH_ANNOTATE.md
#
# Env:
#   LAYER=playback|staggered|hybrid|all   (default playback)
#   QUICK=1
#   SKIP_RIPPER_S1=1   (default on unless QUICK=1; smoke Ripper S1 stays on disk)
#   SKIP_STAGGERED_RIPPER=1  (default on unless QUICK=1; hybrid polarities still run)
#   SKIP_FP32_REALTIME=1     (default on unless QUICK=1; playback already compared dtypes)
#   DRY_RUN=1
#   RESULTS_ROOT=...
set -u
cd "$(dirname "$0")/../.."

PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
ROOT="${RESULTS_ROOT:-results/orch_annotate/stead_2026-08-14}"
TAG=orch_ann
REPEATS=5
NCPUS=20
LAYER="${LAYER:-playback}"

CPU_BASE=0
GPU1_BASE=20
GPU0_BASE=40

if [[ -z "${SKIP_RIPPER_S1:-}" ]]; then
  if [[ "${QUICK:-0}" == "1" ]]; then SKIP_RIPPER_S1=0; else SKIP_RIPPER_S1=1; fi
fi
if [[ -z "${SKIP_STAGGERED_RIPPER:-}" ]]; then
  if [[ "${QUICK:-0}" == "1" ]]; then SKIP_STAGGERED_RIPPER=0; else SKIP_STAGGERED_RIPPER=1; fi
fi
if [[ -z "${SKIP_FP32_REALTIME:-}" ]]; then
  if [[ "${QUICK:-0}" == "1" ]]; then SKIP_FP32_REALTIME=0; else SKIP_FP32_REALTIME=1; fi
fi
MATRIX_FLAGS=(--layer "$LAYER")
[[ "${QUICK:-0}" == "1" ]] && MATRIX_FLAGS+=(--quick)
[[ "$SKIP_RIPPER_S1" == "1" ]] && MATRIX_FLAGS+=(--skip-ripper-s1)

mkdir -p "$ROOT" "$ROOT/parallel_logs"

cores_from() {
  local base=$1 n=$2 i s=""
  for ((i=0; i<n; i++)); do
    [[ -n "$s" ]] && s+=","
    s+=$((base + i))
  done
  echo "$s"
}

note() { echo "=== $(date +%Y-%m-%dT%H:%M:%S)  $* ===" | tee -a "$ROOT/iso_orch_annotate.log"; }

run_trial() {
  local comp=$1 method=$2 model=$3 st=$4 device=$5 kma=$6 krp=$7 pack=$8 arrival=$9 fill=${10} core_base=${11} gpu_id=${12}
  local cores; cores=$(cores_from "$core_base" "$NCPUS")
  local slot_log="$ROOT/parallel_logs/${comp}_${method}_${model}_${st}st_${device}_kma${kma}_krp${krp}_${pack}_${arrival}_${fill}_gpu${gpu_id}.log"
  note "START $comp $method $model ${st}st $device kma=$kma krp=$krp $pack $arrival $fill cores=$cores gpu=$gpu_id"
  taskset -c "$cores" "$PY" benchmarks/fair/run_orch_annotate_trial.py \
    --composition "$comp" --method "$method" --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$NCPUS" --k-ma "$kma" --k-rp "$krp" \
    --packaging "$pack" --arrival "$arrival" --fill "$fill" \
    --core-list "$cores" --gpu-id "$gpu_id" \
    --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  local rc=$?
  note "DONE  $comp $method $model ${st}st $device kma=$kma krp=$krp $pack $arrival $fill rc=$rc"
  return $rc
}

QUEUES="$ROOT/.queues.$$"
"$PY" - "$ROOT" "$TAG" "$LAYER" "${QUICK:-0}" "$SKIP_RIPPER_S1" "$SKIP_STAGGERED_RIPPER" "$SKIP_FP32_REALTIME" <<'PY' >"$QUEUES"
import json, sys
from pathlib import Path
import runpy

root = Path(sys.argv[1])
tag = sys.argv[2]
layer = sys.argv[3]
quick = sys.argv[4] == "1"
skip_ripper_s1 = sys.argv[5] == "1"
skip_staggered_ripper = sys.argv[6] == "1"
skip_fp32_realtime = sys.argv[7] == "1"
ns = runpy.run_path("benchmarks/fair/run_orch_annotate_trial.py")
BEST_BATCH, DTYPE_OF, iter_matrix = ns["BEST_BATCH"], ns["DTYPE_OF"], ns["iter_matrix"]
cells = list(iter_matrix(
    layer=layer,
    quick=quick,
    skip_ripper_s1=skip_ripper_s1,
    skip_staggered_ripper=skip_staggered_ripper,
    skip_fp32_realtime=skip_fp32_realtime,
))
print(f"N_CELLS {len(cells)}")
for c in cells:
    dtype = DTYPE_OF[c["method"]]
    bs = BEST_BATCH[dtype]
    p = (
        root / c["composition"] / c["method"] / "stead" / f"{c['n_stations']}st"
        / c["model"] / c["device"] / f"kma{c['k_ma']}_krp{c['k_rp']}" / c["packaging"]
        / c["arrival"] / c["fill"] / f"bs{bs}" / tag / "result.json"
    )
    done = False
    if p.is_file():
        try:
            r = json.loads(p.read_text())
            done = float((r.get("timing") or {}).get("success_rate") or 0) >= 1.0
        except Exception:
            done = False
    if done:
        continue
    item = "|".join(str(c[k]) for k in (
        "composition", "method", "model", "n_stations", "device",
        "k_ma", "k_rp", "packaging", "arrival", "fill",
    ))
    print(f"{c['device'].upper()} {item}")
PY

CPU_QUEUE=()
GPU_QUEUE=()
n_cells=0
while read -r kind rest; do
  [[ -z "${kind:-}" ]] && continue
  if [[ "$kind" == "N_CELLS" ]]; then
    n_cells=$rest
    continue
  fi
  if [[ "$kind" == "CPU" ]]; then
    CPU_QUEUE+=("$rest")
  else
    GPU_QUEUE+=("$rest")
  fi
done <"$QUEUES"
rm -f "$QUEUES"

cp benchmarks/isolation/README_ORCH_ANNOTATE.md "$ROOT/README.md"
note "ORCH MATRIX layer=$LAYER cells=$n_cells remaining cpu=${#CPU_QUEUE[@]} gpu=${#GPU_QUEUE[@]} root=$ROOT quick=${QUICK:-0} skip_ripper_s1=$SKIP_RIPPER_S1 skip_staggered_ripper=$SKIP_STAGGERED_RIPPER skip_fp32_realtime=$SKIP_FP32_REALTIME"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  note "DRY_RUN=1 — not launching"
  exit 0
fi

cpu_worker() {
  local item comp method model st device kma krp pack arrival fill
  for item in "${CPU_QUEUE[@]}"; do
    IFS='|' read -r comp method model st device kma krp pack arrival fill <<<"$item"
    run_trial "$comp" "$method" "$model" "$st" "$device" "$kma" "$krp" "$pack" "$arrival" "$fill" "$CPU_BASE" 0 || true
  done
  note "CPU WORKER DRAINED"
}

gpu_worker() {
  local worker_id=$1 idx=0 item comp method model st device kma krp pack arrival fill gpu_id core_base
  for item in "${GPU_QUEUE[@]}"; do
    if (( idx % 2 == worker_id )); then
      IFS='|' read -r comp method model st device kma krp pack arrival fill <<<"$item"
      if (( idx % 2 == 0 )); then gpu_id=1; core_base=$GPU1_BASE; else gpu_id=0; core_base=$GPU0_BASE; fi
      run_trial "$comp" "$method" "$model" "$st" "$device" "$kma" "$krp" "$pack" "$arrival" "$fill" "$core_base" "$gpu_id" || true
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
note "ORCH ANNOTATE DONE"
