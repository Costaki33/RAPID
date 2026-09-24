#!/usr/bin/env bash
# Core-budget extension of the 20-core orch suite: n_cpus in {5,10,15}.
#
# Does NOT touch results/orch_annotate/stead_2026-08-14 (the 20-core set).
# K is capped at n_cpus. GPU K ladders stay MA {1,2,4} / Ripper {1,2}.
#
# Same isolation slots as the 20-core launcher:
#   CPU  : cores 0-19   (first n_cpus of the block)
#   GPU1 : cores 20-39 + CUDA 1
#   GPU0 : cores 40-59 + CUDA 0
#
# Env:
#   LAYER=playback|staggered|hybrid|all   (default all)
#   NCPU_GRID=5,10,15
#   SKIP_RIPPER_S1=1   (default on unless QUICK=1)
#   SKIP_STAGGERED_RIPPER=1
#   SKIP_FP32_REALTIME=1
#   DRY_RUN=1
#   RESULTS_ROOT=...
set -u
cd "$(dirname "$0")/../.."

PY="${RAPID_PYTHON:-/home/skevofilaxc/miniconda3/envs/rapid/bin/python}"
ROOT="${RESULTS_ROOT:-results/orch_annotate/stead_2026-08-21_cpus5_10_15}"
TAG=orch_ann
REPEATS=5
LAYER="${LAYER:-all}"
NCPU_GRID="${NCPU_GRID:-5,10,15}"

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
  local ncpus=$1 comp=$2 method=$3 model=$4 st=$5 device=$6 kma=$7 krp=$8 pack=$9 arrival=${10} fill=${11} core_base=${12} gpu_id=${13}
  local cores; cores=$(cores_from "$core_base" "$ncpus")
  local slot_log="$ROOT/parallel_logs/c${ncpus}_${comp}_${method}_${model}_${st}st_${device}_kma${kma}_krp${krp}_${pack}_${arrival}_${fill}_gpu${gpu_id}.log"
  note "START cpus=$ncpus $comp $method $model ${st}st $device kma=$kma krp=$krp $pack $arrival $fill cores=$cores gpu=$gpu_id"
  taskset -c "$cores" "$PY" benchmarks/fair/run_orch_annotate_trial.py \
    --composition "$comp" --method "$method" --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$ncpus" --k-ma "$kma" --k-rp "$krp" \
    --packaging "$pack" --arrival "$arrival" --fill "$fill" \
    --core-list "$cores" --gpu-id "$gpu_id" \
    --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  local rc=$?
  note "DONE  cpus=$ncpus $comp $method $model ${st}st $device kma=$kma krp=$krp $pack $arrival $fill rc=$rc"
  return $rc
}

QUEUES="$ROOT/.queues.$$"
"$PY" - "$ROOT" "$TAG" "$LAYER" "${QUICK:-0}" "$SKIP_RIPPER_S1" "$SKIP_STAGGERED_RIPPER" "$SKIP_FP32_REALTIME" "$NCPU_GRID" <<'PY' >"$QUEUES"
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
ncpu_grid = [int(x) for x in sys.argv[8].split(",") if x.strip()]
ns = runpy.run_path("benchmarks/fair/run_orch_annotate_trial.py")
BEST_BATCH, DTYPE_OF = ns["BEST_BATCH"], ns["DTYPE_OF"]
iter_core_budget_matrix, orch_cell_dir = ns["iter_core_budget_matrix"], ns["orch_cell_dir"]

cells = []
for n_cpus in ncpu_grid:
    for c in iter_core_budget_matrix(
        n_cpus,
        layer=layer,
        quick=quick,
        skip_ripper_s1=skip_ripper_s1,
        skip_staggered_ripper=skip_staggered_ripper,
        skip_fp32_realtime=skip_fp32_realtime,
    ):
        cells.append(c)

print(f"N_CELLS {len(cells)}")
for c in cells:
    dtype = DTYPE_OF[c["method"]]
    bs = BEST_BATCH[dtype]
    p = orch_cell_dir(root, c, n_cpus=c["n_cpus"], tag=tag, batch_size=bs) / "result.json"
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
        "n_cpus", "composition", "method", "model", "n_stations", "device",
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

cat > "$ROOT/README.md" <<EOF
# Orch annotate core-budget sweep (5 / 10 / 15 CPUs)

20-core results stay in \`results/orch_annotate/stead_2026-08-14/\`.
This root only has n_cpus in {5,10,15}. Paths include \`cpusN\` after the device.
K never exceeds n_cpus. GPU K ladders are unchanged (MA 1/2/4, Ripper 1/2).
Same skip flags as the 20-core suite (no Ripper+S1, no homogeneous staggered Ripper, no fp32 realtime).
EOF

note "ORCH CORE-BUDGET layer=$LAYER cells=$n_cells remaining cpu=${#CPU_QUEUE[@]} gpu=${#GPU_QUEUE[@]} root=$ROOT ncpu_grid=$NCPU_GRID quick=${QUICK:-0} skip_ripper_s1=$SKIP_RIPPER_S1 skip_staggered_ripper=$SKIP_STAGGERED_RIPPER skip_fp32_realtime=$SKIP_FP32_REALTIME"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  note "DRY_RUN=1 — not launching"
  exit 0
fi

cpu_worker() {
  local item ncpus comp method model st device kma krp pack arrival fill
  for item in "${CPU_QUEUE[@]}"; do
    IFS='|' read -r ncpus comp method model st device kma krp pack arrival fill <<<"$item"
    run_trial "$ncpus" "$comp" "$method" "$model" "$st" "$device" "$kma" "$krp" "$pack" "$arrival" "$fill" "$CPU_BASE" 0 || true
  done
  note "CPU WORKER DRAINED"
}

gpu_worker() {
  local worker_id=$1 idx=0 item ncpus comp method model st device kma krp pack arrival fill gpu_id core_base
  for item in "${GPU_QUEUE[@]}"; do
    if (( idx % 2 == worker_id )); then
      IFS='|' read -r ncpus comp method model st device kma krp pack arrival fill <<<"$item"
      if (( idx % 2 == 0 )); then gpu_id=1; core_base=$GPU1_BASE; else gpu_id=0; core_base=$GPU0_BASE; fi
      run_trial "$ncpus" "$comp" "$method" "$model" "$st" "$device" "$kma" "$krp" "$pack" "$arrival" "$fill" "$core_base" "$gpu_id" || true
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
note "ORCH ANNOTATE CORE-BUDGET DONE"
