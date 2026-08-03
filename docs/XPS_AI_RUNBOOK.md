# XPS AI Runbook — Fastest Solutions Only

Instructions for another AI (or operator) to validate RAPID’s fastest warm methods on the Dell XPS 15 9530. Do **not** invent a different matrix. Follow this document.

## Goal

Answer: on a consumer performance laptop with **one GPU**, do the paper’s fastest warm methods still meet the study 10-second phase-picking target for a 580-station STEAD window?

Primary comparison is **within-machine method ordering**, not absolute workstation vs laptop runtime.

## Hardware under test

- Dell XPS 15 9530
- Windows 11 Enterprise (host)
- Intel Core i9-13900H: 6 P-cores (+SMT) + 8 E-cores = 14 physical / 20 logical
- 32 GB RAM
- 1× NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 expected)
- **No second GPU** — never run `stream_modelactor_2gpu`

## Platform requirement

Use **WSL2 Ubuntu** or bare-metal Linux.

Do **not** use native Windows for this benchmark suite:

- `sched_setaffinity` is required for core isolation
- process-tree PSS is Linux-only
- Ray / TensorFlow / Bash isolation scripts assume Linux

Recommended WSL2 config in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=20
swap=8GB
```

Then:

```powershell
wsl --shutdown
```

## What to run / what to skip

### Run (fastest / paper-relevant)

| Strategy | Why |
|---|---|
| `stream_classify_batched` | Missing warm native discrete-pick baseline for the paper |
| `stream_annotate` | Warm native probability-trace baseline |
| `stream_modelactor` | Per-station Classify in persistent actors |
| `stream_modelactor_batched` | Persistent actors + Network-Batched Classify per station share |
| `stream_modelactor_slipstream` with `bf16` | Secondary RAPID precision path |

### Skip

- Ripper / cold ephemeral workers
- 2-GPU Model-Actor
- Per-Station Classify
- Oversubscription sweeps
- FP16 EQTransformer (sentinel invalid)
- Full cold-start orchestration grids
- Soak tests unless primary finishes and user asks

## Core isolation (required)

RAPID’s `--n-cpus` and `--core-list` count **logical CPUs**. On this hybrid CPU they are **not** equivalent to Threadripper physical cores.

### 1. Inspect topology inside WSL/Linux

```bash
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ
for f in /sys/devices/system/cpu/cpu[0-9]*/topology/core_type; do
  printf '%s=' "$f"; cat "$f" 2>/dev/null || echo missing
done
```

### 2. Build the affinity list

Prefer **one hardware thread per P-core**. Example procedure:

1. Identify P-cores from `core_type` or from the highest-MAXMHZ cores.
2. For each chosen physical core, pick **exactly one** logical CPU ID.
3. Avoid pairing SMT siblings in the same budget.
4. Prefer excluding E-cores from the primary validation list.

If `core_type` is unavailable, select five highest-frequency unique physical `CORE` IDs and one logical CPU from each. Label results as “selected logical CPUs,” not “physical cores,” unless topology confirms P-cores.

### 3. Export CORES before every run

```bash
# Example only — replace with the IDs discovered on this laptop:
export CORES=0,2,4,6,8
```

The runner refuses to start if `CORES` is empty.

### 4. Verify pinning after a smoke trial

```bash
# While a trial is running:
ps -eo pid,psr,comm,args | rg 'run_fair_stream_trial|ray::|python' | head
taskset -pc <pid>
```

If workers are free-floating across all 20 logical CPUs, stop and fix affinity before continuing.

## Memory / VRAM caps built into the runner

| Resource | Cap | Reason |
|---|---|---|
| CPU actors | `CPU_ACTOR_CAP=10` default | 20-actor workstation pools used ~11–12 GB warm / ~22 GB cold; XPS+WSL cannot safely reproduce 20 actors |
| GPU actors | `GPU_ACTOR_CAP=2` default | 8 GB laptop GPU; EQT actors need ~2 GB each plus headroom |
| Host CPU grid | `5 10 15 20` | Same shape as paper sweep; budgets larger than available isolated IDs are skipped |

If GPU actor creation OOMs:

```bash
export GPU_ACTOR_CAP=1
export BATCH_SIZE=64   # only if batch 256 OOMs; record the change
```

Changing batch size invalidates direct timing comparison with workstation batch-256 cells. Prefer reducing concurrency first.

## Phases

Runner: `benchmarks/isolation/run_xps_fastest.sh`

### Phase A — smoke (must pass before anything else)

```bash
cd /path/to/RAPID
export CORES='REPLACE_WITH_REAL_IDS'   # >=5 IDs
export PHASE=smoke
bash benchmarks/isolation/run_xps_fastest.sh
```

Checks: imports, manifests, CUDA visibility, affinity, one PhaseNet 250-station cell.

### Phase B — pilot

```bash
export PHASE=pilot
bash benchmarks/isolation/run_xps_fastest.sh
```

Models: PhaseNet + EQTransformer, 250 stations, 1 repeat × 2 feeds, host budget 5.

Abort and report if:

- WSL OOM / thrashing
- NVML cannot see the Python PID for GPU VRAM
- GPU Model-Actor cannot create even 1 actor
- any method exceeds several minutes per 250-station feed

### Phase C — primary (paper-facing laptop table)

```bash
export PHASE=primary
# optional overrides:
# export CPU_GRID='5 10 15 20'
# export REPEATS=5
bash benchmarks/isolation/run_xps_fastest.sh
```

Default primary matrix:

- Models: PN, PNL, EQT, EQT-NC
- Stations: 580
- Host CPU budgets: 5, 10, 15, 20 (capped by isolated ID count)
- Devices: CPU and 1 GPU
- Strategies listed above
- 5 repeats × 8 feeds, feed 0 discarded as cold

### Phase D — extension (only if primary succeeds and user wants workstation-matched sample size)

```bash
export PHASE=extension   # 10 repeats
bash benchmarks/isolation/run_xps_fastest.sh
```

## Exact method settings

| Method | Device | concurrency | torch threads | dtype |
|---|---|---|---|---|
| Network-Batched Classify | CPU/GPU | 1 | 1 | fp32 |
| Annotate | CPU/GPU | 1 | `min(8, n_cpus)` | fp32 |
| Model-Actor | CPU | `min(n_cpus, CPU_ACTOR_CAP)` | 1 per actor | fp32 |
| Model-Actor Slipstream | CPU | same as Model-Actor | 1 per actor | bf16 |
| Model-Actor | GPU | `min(n_cpus, GPU_ACTOR_CAP)` | 1 per actor | fp32 |

Network suffixes:

- PhaseNet / PhaseNetLight: `--net-suffix _w3001`, `--in-samples 3001`
- EQTransformer / EQT-NC: empty suffix, `--in-samples 6000`

## Provenance (mandatory)

Before Phase C:

```bash
mkdir -p results/xps_validation/provenance
{
  date -Ins
  uname -a
  cat /proc/version || true
  lscpu
  free -h
  echo "CORES=$CORES"
  nvidia-smi -q
  python - <<'PY'
import torch, ray, seisbench, numpy, obspy, sys
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
    print("bf16", torch.cuda.is_bf16_supported())
print("ray", ray.__version__)
print("seisbench", seisbench.__version__)
print("numpy", numpy.__version__)
print("obspy", obspy.__version__)
PY
  git rev-parse HEAD || true
  sha256sum data/seisbench_networks/stead_580st/manifest.json \
            data/seisbench_networks/stead_580st_w3001/manifest.json
} > results/xps_validation/provenance/environment.txt
```

Also record in the same folder (plain text is fine):

- Windows build, WSL version, VBS on/off
- AC power connected (required)
- Dell thermal / Windows power mode
- Start/end GPU temperature
- Achieved actor count vs requested
- Any OOM or batch-size change

## After the run

1. Confirm every expected `result.json` exists under `results/xps_validation/`.
2. Summarize warm means (feeds ≥1) for each method × device × model × host budget.
3. Report method ordering vs the 10 s target.
4. Do **not** claim absolute XPS times equal workstation times.
5. Valid claim shape: “On the XPS under budget C, warm CPU Model-Actor was X% faster/slower than warm Network-Batched Classify / Annotate.”

## Analysis one-liner

```bash
python - <<'PY'
import json, statistics
from pathlib import Path
root = Path("results/xps_validation")
rows = []
for p in root.rglob("result.json"):
    m = json.loads(p.read_text())["meta"]
    means = []
    for rf in sorted((p.parent / "repeats").glob("repeat_*.json")):
        rr = json.loads(rf.read_text())
        if not rr.get("success"):
            continue
        warm = [f["feed_total_s"] for f in rr.get("feeds", []) if f.get("feed_index", 0) >= 1]
        if warm:
            means.append(statistics.mean(warm))
    if means:
        rows.append((m["model"], m["method"], m["device"], m.get("n_cpus"), m.get("concurrency"),
                     round(statistics.mean(means), 3), len(means)))
for r in sorted(rows):
    print(r)
PY
```

## Failure policy

| Symptom | Action |
|---|---|
| OOM at CPU actors ≥10 | lower `CPU_ACTOR_CAP` to 5 and continue; note the change |
| GPU OOM at actors=2 | set `GPU_ACTOR_CAP=1`; keep batch 256 if possible |
| batch 256 GPU OOM | try 128 then 64; mark those cells non-comparable to workstation batch-256 |
| affinity not sticking | stop; fix CORES / WSL before more trials |
| swapping in `free -h` / `vmstat` | stop; reduce actors or free host RAM |

## Estimated runtime (primary)

Rough guide after pilot:

- ~4 models × 4 host budgets × ~7 method/device cells × 5 repeats × ~10–40 s/session
- Expect multiple hours; use `--resume` (enabled) and keep the laptop plugged in.

Do not start overnight runs until smoke + pilot pass.
