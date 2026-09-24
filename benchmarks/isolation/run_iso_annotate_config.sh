#!/usr/bin/env bash
# Parallel native Annotate configuration sweep (no orchestration).
#
# Question: per model, which dtype + core budget + intra-op thread multiplier
# + batch + device minimizes annotate() time on a 250/580-station STEAD net,
# for merged (one call) and sequential (one call per station) packaging.
#
# 20-core slots on this 128-logical-CPU workstation:
#   CPU home     : cores 0-19
#   GPU1         : cores 20-39 + CUDA 1; when that GPU queue is empty, the
#                  SAME 20-core block runs leftover CPU trials (CUDA hidden)
#   GPU0         : cores 40-59 + CUDA 0; same steal onto 40-59
#   extra CPU    : 60-79, 80-99, 100-119  (leftover 120-127 unused, <20)
#
# Each GPU trial still sees one physical GPU. CPU trials set
# CUDA_VISIBLE_DEVICES empty even when they run on a former GPU slot.
#
# Matrix:
#   models       : PhaseNet, PhaseNetLight, EQTransformer, EQT-NC, EQCCT, EQCCTP
#   methods      : annotate_fp32 / bf16 / fp16  (EQT family skips fp16)
#   stations     : 250, 580
#   packaging    : merged, sequential
#   n_cpus       : 5, 10, 15, 20
#   thread_mult  : 0.25, 0.5, 1.0, 1.5, 2.0
#   inter-op     : 1
#   batch_size   : 64, 128, 256, 512
#   repeats      : 3
#   cell timeout : 30 min — recorded as timed out by us (RAPID), not retried
#
# Usage:
#   bash benchmarks/isolation/run_iso_annotate_config.sh
#   RESULTS_ROOT=results/annotate_config/stead_iso_YYYY-MM-DD \
#     bash benchmarks/isolation/run_iso_annotate_config.sh
#   DRY=1 bash benchmarks/isolation/run_iso_annotate_config.sh
#
# Rerun RAPID-timeout cells to completion (no cell timeout; keep successful repeats):
#   RESULTS_ROOT=results/annotate_config/stead_iso_YYYY-MM-DD \
#     RETRY_TIMEOUTS=1 CELL_TIMEOUT_S=0 \
#     bash benchmarks/isolation/run_iso_annotate_config.sh
#
# Launch detached (survives logout / Cursor shell teardown):
#   RESULTS_ROOT=results/annotate_config/stead_iso_YYYY-MM-DD \
#     nohup setsid bash benchmarks/isolation/run_iso_annotate_config.sh \
#     </dev/null >>"$RESULTS_ROOT/iso_annotate_config.nohup.out" 2>&1 &
set -u
# SIGHUP killed the 2026-09-08 run when the Cursor background shell was
# reaped: last START never got DONE, workers vanished, isolation went idle.
trap '' HUP
cd "$(dirname "$0")/../.."

PY="${RAPID_PYTHON:-$(command -v python3)}"
if [[ -x /home/skevofilaxc/miniconda3/envs/rapid/bin/python ]]; then
  PY=/home/skevofilaxc/miniconda3/envs/rapid/bin/python
fi

ROOT="${RESULTS_ROOT:-results/annotate_config/stead_iso_$(date +%Y-%m-%d)}"
TAG=ann_cfg
REPEATS="${REPEATS:-3}"
CELL_TIMEOUT_S="${CELL_TIMEOUT_S:-1800}"
RETRY_TIMEOUTS="${RETRY_TIMEOUTS:-0}"
TIMEOUT_BIN="${TIMEOUT_BIN:-/usr/bin/timeout}"
DRY="${DRY:-0}"
CHUNK=20
LOG_DIR="$ROOT/parallel_logs"
CPU_WORK="$ROOT/cpu_work_queue.txt"
CPU_LOCK="$ROOT/cpu_work_queue.lock"
mkdir -p "$ROOT" "$LOG_DIR"
mkdir -p results/annotate_config
printf '%s\n' "$ROOT" > results/annotate_config/RESULTS_ROOT.txt

CPU_HOME_BASE=0
GPU1_BASE=20
GPU0_BASE=40
GPU1_ID=1
GPU0_ID=0

cores_from() {
  local base=$1 n=$2 i s=""
  for ((i=0; i<n; i++)); do
    [[ -n "$s" ]] && s+=","
    s+=$((base + i))
  done
  echo "$s"
}

threads_for() {
  local ncpus=$1 mult=$2
  PYTHONPATH="." "$PY" -c "from rapid.benchmark.fairness import threads_from_multiplier; print(threads_from_multiplier(int('$ncpus'), float('$mult')))"
}

cell_result_path() {
  local method=$1 model=$2 st=$3 device=$4 ncpus=$5 thr=$6 batch=$7 packaging=$8
  echo "$ROOT/$method/stead/${st}st/$model/$device/cpus${ncpus}/thr${thr}/bs${batch}/$packaging/$TAG/result.json"
}

is_done() {
  local f=$1
  [[ -f "$f" ]] || return 1
  PYTHONPATH="." "$PY" - "$f" <<'PY'
import json, sys
r = json.loads(open(sys.argv[1]).read())
timing = r.get("timing") or {}
meta = r.get("meta") or {}
if timing.get("timed_out") or meta.get("timed_out") or meta.get("timed_out_by"):
    sys.exit(0)
sr = float(timing.get("success_rate") or 0)
sys.exit(0 if sr >= 1.0 else 1)
PY
}

write_timeout_result() {
  local f=$1 method=$2 model=$3 st=$4 device=$5 ncpus=$6 thr=$7 batch=$8 packaging=$9 mult=${10} gpu_id=${11}
  mkdir -p "$(dirname "$f")"
  PYTHONPATH="." "$PY" - "$f" "$method" "$model" "$st" "$device" "$ncpus" "$thr" "$batch" "$packaging" "$mult" "$gpu_id" "$CELL_TIMEOUT_S" "$REPEATS" "$TAG" <<'PY'
import json, sys, time
path, method, model, st, device, ncpus, thr, batch, packaging, mult, gpu_id, timeout_s, repeats, tag = sys.argv[1:]
dtype = {"annotate_fp32": "fp32", "annotate_bf16": "bf16", "annotate_fp16": "fp16"}[method]
msg = (
    f"Timed out by us (RAPID isolation launcher) after {timeout_s}s. "
    "The cell was stopped by CELL_TIMEOUT_S, not by an OS crash, and will not be retried."
)
out = {
    "schema_version": 3,
    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "meta": {
        "method": method,
        "family": "annotate_config",
        "dataset": "stead",
        "n_stations": int(st),
        "model": model,
        "device": device,
        "n_cpus": int(ncpus),
        "torch_threads": int(thr),
        "thread_mult": float(mult),
        "interop_threads": 1,
        "gpu_id": (int(gpu_id) if device == "gpu" else None),
        "dtype": dtype,
        "batch_size": int(batch),
        "packaging": packaging,
        "repeats": int(repeats),
        "tag": tag,
        "timed_out": True,
        "timed_out_by": "rapid_iso_annotate_config",
        "cell_timeout_s": int(timeout_s),
    },
    "timing": {
        "success_rate": 0.0,
        "timed_out": True,
        "timed_out_by": "rapid_iso_annotate_config",
        "n_repeats": 0,
        "repeats": [],
    },
    "error": msg,
}
open(path, "w").write(json.dumps(out, indent=2) + "\n")
PY
}

note() { echo "=== $(date +%Y-%m-%dT%H:%M:%S)  $* ===" | tee -a "$ROOT/iso_annotate_config.log"; }

run_trial() {
  local method=$1 model=$2 st=$3 device=$4 ncpus=$5 batch=$6 packaging=$7 mult=$8 core_base=$9 gpu_id=${10}
  local thr cores f slot_log rc
  thr=$(threads_for "$ncpus" "$mult")
  cores=$(cores_from "$core_base" "$ncpus")
  f=$(cell_result_path "$method" "$model" "$st" "$device" "$ncpus" "$thr" "$batch" "$packaging")
  slot_log="$LOG_DIR/${method}_${model}_${st}st_${device}_c${ncpus}_thr${thr}_m${mult}_bs${batch}_${packaging}_slot${core_base}.log"
  if is_done "$f"; then
    note "SKIP done $method / $model / ${st}st / $device / cpus=$ncpus thr=$thr ($mult) bs=$batch $packaging"
    return 0
  fi
  note "START $method / $model / ${st}st / $device / cpus=$ncpus thr=$thr (mult=$mult) bs=$batch $packaging cores=$cores gpu_id=$gpu_id"
  if [[ "$DRY" == "1" ]]; then
    echo "DRY: timeout ${CELL_TIMEOUT_S}s taskset -c $cores ... --packaging $packaging --interop-threads 1"
    return 0
  fi
  "$TIMEOUT_BIN" --signal=KILL "${CELL_TIMEOUT_S}" taskset -c "$cores" "$PY" benchmarks/fair/run_annotate_precision_trial.py \
    --method "$method" --model "$model" --n-stations "$st" \
    --device "$device" --n-cpus "$ncpus" --torch-threads "$thr" \
    --thread-mult "$mult" --interop-threads 1 \
    --core-list "$cores" --gpu-id "$gpu_id" \
    --batch-size "$batch" --packaging "$packaging" \
    --repeats "$REPEATS" --tag "$TAG" \
    --results-root "$ROOT" --resume >>"$slot_log" 2>&1
  rc=$?
  if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
    note "TIMED OUT BY US $method / $model / ${st}st / $device / cpus=$ncpus thr=$thr bs=$batch $packaging after ${CELL_TIMEOUT_S}s"
    write_timeout_result "$f" "$method" "$model" "$st" "$device" "$ncpus" "$thr" "$batch" "$packaging" "$mult" "$gpu_id"
    return 0
  fi
  note "DONE  $method / $model / ${st}st / $device / cpus=$ncpus thr=$thr bs=$batch $packaging rc=$rc"
  return "$rc"
}

pop_cpu_cell() {
  # One line from the shared CPU queue. Empty string means drained.
  flock "$CPU_LOCK" bash -c '
    f="'"$CPU_WORK"'"
    if [[ ! -s "$f" ]]; then
      exit 0
    fi
    IFS= read -r line < "$f" || exit 0
    tail -n +2 "$f" > "$f.tmp" || true
    mv "$f.tmp" "$f"
    printf "%s\n" "$line"
  '
}

cpu_drain_on_slot() {
  local slot_name=$1 core_base=$2
  local item method model st device ncpus batch packaging mult
  while true; do
    item=$(pop_cpu_cell || true)
    [[ -n "${item:-}" ]] || break
    IFS='|' read -r method model st device ncpus batch packaging mult <<<"$item"
    note "CPU-STEAL slot=$slot_name cores=${core_base}-$((core_base+CHUNK-1)) $method / $model / ${st}st"
    run_trial "$method" "$model" "$st" "$device" "$ncpus" "$batch" "$packaging" "$mult" "$core_base" 0 || true
  done
  note "CPU QUEUE EMPTY on slot=$slot_name"
}

# Remaining cells from the shared Python iterator.
mapfile -t ALL_CELLS < <(
  PYTHONPATH="." "$PY" - <<'PY'
import runpy
ns = runpy.run_path("benchmarks/fair/run_annotate_precision_trial.py")
for c in ns["iter_config_matrix"]():
    print(
        "|".join(
            [
                c["method"],
                c["model"],
                str(c["n_stations"]),
                c["device"],
                str(c["n_cpus"]),
                str(c["batch_size"]),
                c["packaging"],
                str(c["thread_mult"]),
                str(c["torch_threads"]),
            ]
        )
    )
PY
)

GPU_QUEUE=()
: > "$CPU_WORK"
n_expected=${#ALL_CELLS[@]}
if [[ "$n_expected" -eq 0 ]]; then
  echo "ERROR: iter_config_matrix produced 0 cells (python failed?)" >&2
  exit 2
fi
n_skip=0
n_cpu_remain=0
n_gpu_remain=0
for item in "${ALL_CELLS[@]}"; do
  IFS='|' read -r method model st device ncpus batch packaging mult thr <<<"$item"
  f=$(cell_result_path "$method" "$model" "$st" "$device" "$ncpus" "$thr" "$batch" "$packaging")
  if is_done "$f"; then
    n_skip=$((n_skip + 1))
    continue
  fi
  payload="$method|$model|$st|$device|$ncpus|$batch|$packaging|$mult"
  if [[ "$device" == "cpu" ]]; then
    printf '%s\n' "$payload" >> "$CPU_WORK"
    n_cpu_remain=$((n_cpu_remain + 1))
  else
    GPU_QUEUE+=("$payload")
    n_gpu_remain=$((n_gpu_remain + 1))
  fi
done

NPROC=$(nproc)
N_EXTRA_CPU=0
EXTRA_BASES=()
base=$((GPU0_BASE + CHUNK))
while (( base + CHUNK <= NPROC )); do
  EXTRA_BASES+=("$base")
  N_EXTRA_CPU=$((N_EXTRA_CPU + 1))
  base=$((base + CHUNK))
done

cat >"$ROOT/README_MATRIX.md" <<EOF
# Native annotate configuration sweep

- Started: $(date -Is)
- Results root: \`$ROOT\`
- Tag: \`$TAG\`
- Models: PhaseNet, PhaseNetLight, EQTransformer, EQT-NC, EQCCT, EQCCTP
- Methods: fp32 / bf16 / fp16 (EQTransformer and EQT-NC skip fp16)
- Stations: STEAD 250 / 580
- Packaging: merged, sequential
- CPU counts (CPU **and** GPU host cores): 5 10 15 20
- Thread multipliers: 0.25 0.5 1.0 1.5 2.0  (intra-op = round-half-up(mult × cores); inter-op = 1)
- Batch sizes: 64 128 256 512
- Repeats: $REPEATS isolated subprocesses / cell
- Cell timeout: ${CELL_TIMEOUT_S}s, **timed out by us** (RAPID launcher); not retried
- Slots (20 cores each): CPU home 0-19; GPU1 20-39 (CUDA 1, then leftover CPU on 20-39); GPU0 40-59 (CUDA 0, then leftover CPU on 40-59); extra CPU ${EXTRA_BASES[*]:-none}
- Expected cells: $n_expected
- Path: \`<method>/stead/<N>st/<model>/<cpu|gpu>/cpus<N>/thr<T>/bs<B>/<merged|sequential>/$TAG/result.json\`
- Orchestration is out of scope. See \`RAPID/docs/RAPID_ANNOTATE_CONFIG_SWEEP.md\`.
EOF
printf '%s\n' "$ROOT" >"$ROOT/RESULTS_ROOT.txt"

note "CONFIG SWEEP START py=$PY root=$ROOT expected=$n_expected already_done=$n_skip cpu_remain=$n_cpu_remain gpu_remain=$n_gpu_remain"
note "Slots: CPU home ${CPU_HOME_BASE}-$((CPU_HOME_BASE+CHUNK-1)); GPU1 ${GPU1_BASE}-$((GPU1_BASE+CHUNK-1)) cuda=$GPU1_ID then steal CPU; GPU0 ${GPU0_BASE}-$((GPU0_BASE+CHUNK-1)) cuda=$GPU0_ID then steal CPU; extra CPU bases=${EXTRA_BASES[*]:-none}"
note "Repeats=$REPEATS timeout=${CELL_TIMEOUT_S}s timed-out-by-us interop=1"

if [[ "$DRY" == "1" ]]; then
  note "DRY RUN: would run $((n_cpu_remain + n_gpu_remain)) remaining cells"
  exit 0
fi

cpu_home_worker() {
  note "CPU HOME worker on cores ${CPU_HOME_BASE}-$((CPU_HOME_BASE+CHUNK-1))"
  cpu_drain_on_slot "cpu_home" "$CPU_HOME_BASE"
}

extra_cpu_worker() {
  local core_base=$1
  note "EXTRA CPU worker on cores ${core_base}-$((core_base+CHUNK-1))"
  cpu_drain_on_slot "cpu_${core_base}" "$core_base"
}

gpu_worker() {
  local worker_id=$1 gpu_id=$2 core_base=$3
  local idx=0 item method model st device ncpus batch packaging mult
  note "GPU worker $worker_id cuda=$gpu_id cores=${core_base}-$((core_base+CHUNK-1))"
  for item in "${GPU_QUEUE[@]}"; do
    if (( idx % 2 == worker_id )); then
      IFS='|' read -r method model st device ncpus batch packaging mult <<<"$item"
      run_trial "$method" "$model" "$st" "$device" "$ncpus" "$batch" "$packaging" "$mult" "$core_base" "$gpu_id" || true
    fi
    idx=$((idx + 1))
  done
  note "GPU worker $worker_id CUDA queue drained; filling cores ${core_base}-$((core_base+CHUNK-1)) with leftover CPU trials"
  cpu_drain_on_slot "gpu${gpu_id}_then_cpu" "$core_base"
}

PIDS=()
cpu_home_worker &
PIDS+=($!)
gpu_worker 0 "$GPU1_ID" "$GPU1_BASE" &
PIDS+=($!)
gpu_worker 1 "$GPU0_ID" "$GPU0_BASE" &
PIDS+=($!)
for b in "${EXTRA_BASES[@]+"${EXTRA_BASES[@]}"}"; do
  extra_cpu_worker "$b" &
  PIDS+=($!)
done

note "PIDs ${PIDS[*]}"
echo "$$" >"$ROOT/launcher.pid"
printf '%s\n' "${PIDS[@]}" >"$ROOT/worker.pids"
wait "${PIDS[@]}"
note "CONFIG SWEEP DONE"
rm -f "$ROOT/launcher.pid" "$ROOT/worker.pids"
