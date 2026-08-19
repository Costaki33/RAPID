# Locked-recipe transfer suite — handoff for another model

**Read this first.** Do not invent a new dtype, K, Ripper, hybrid, S1, or wait-5/wait-10 sweep. This is a **portability check** of three already-locked configurations on **another computer**.

Parent scientific brief: `RAPID/docs/RAPID_ANNOTATE_TEST_STORY.md`  
Session rules (isolation, no commit unless asked): `RAPID/docs/RAPID_ANNOTATE_SESSION_HANDOFF.md`

---

## What you are answering

On **this machine**, with the **same 5/10/15/20-core × CPU/GPU sweep** used in the merged Annotate precision study, how do these three **locked** recipes behave for the five pickers?

| Layer | Meaning | Locked recipe |
|---|---|---|
| **native** | Best single-process Annotate (all streams in one call) | `annotate_bf16`, batch **512**, packaging **merged**, torch/OMP threads = `n_cpus` |
| **playback** | Best all-stations-ready orchestration | Model-Actor, **SG**, fill **static**, **1 thread per actor** |
| **staggered** | Best delayed-station (realtime) orchestration | Model-Actor, **SG**, fill **eager**, **1 thread per actor** |

Models: **EQCCT, PhaseNet, PhaseNetLight, EQTransformer, EQT-NC**.  
Networks: STEAD **250** and **580** (`data/seisbench_networks/stead_250st`, `stead_580st`).

**Not in this suite:** fp16, fp32, sequential S1, homogeneous Ripper, hybrid MA+Ripper, wait-5 / wait-10, batch sweep, K sweep.

EQCCT is SeisBench **`EQCCTP` + `EQCCTS`**. Do not treat it as one WaveformModel.

---

## Why these three (do not reopen)

Workstation (Threadripper + RTX 6000 Ada) four-step suite, Aug 2026:

1. **Precision:** bf16 won 16/20 slices vs fp32/fp16. Batch 512. FP16 dropped: EQTransformer / EQT-NC overflow (pad sentinel \(-\,10^{10}\)); PhaseNetLight P F1 ~0.65 vs fp32; EQCCT CPU fp16 9–33 s.
2. **Playback:** Model-Actor + SG beat S1 and Ripper when everyone is ready. GPU **K=4** (PhaseNet @ 250: **K=2**). CPU **K=20** at 580 (K=10 enough for smaller models at 250).
3. **Staggered:** rank **p95 finish−ready**, not makespan (~90 s delay ceiling). Eager beat wait-5/10. Hybrid lost. Playback K still holds.

This transfer run asks whether **ordering and ballpark latency** survive a different CPU/GPU, not whether to change the recipe.

---

## Worker count K (already decided; only capped here)

```
CPU 580:           want K=20
CPU 250 EQCCT:     want K=20
CPU 250 others:    want K=10
GPU:               want K=4  (PhaseNet @ 250: K=2)
then K = min(want, n_cpus, CPU_K_CAP or GPU_K_CAP)
```

Do not sweep K. Caps exist for **small RAM/VRAM** (laptop), not for fishing a new elbow.

---

## How to rank

| Layer | Metric | JSON field |
|---|---|---|
| native | mean **inference** seconds | `timing.inference_s_mean` |
| playback | mean **makespan** | `orch.makespan_s_mean` |
| staggered | pooled **p95 finish−ready** | `latency.pooled_across_repeats.e2e_finish_minus_ready.p95` |

Staggered **makespan will be ~90 s** if anyone draws a 90 s delay. That is not a scheduler ranking. Idle gaps are **simulated** in `rapid/benchmark/orch_dispatch.py` (do not restore real `time.sleep`).

Picks: `pick_quality_vs_catalog` (and native also `pick_quality_vs_fp32` only if a sibling fp32 cell exists — it will not in this suite). Tolerance 50 samples (0.5 s at 100 Hz). Extractor is SeisBench `classify_aggregate`, thresholds 0.3.

---

## Workstation reference (580 stations, matched cells)

Use these as **order-of-magnitude** checks, not pass/fail gates. Another machine will be slower. **Within-machine ordering** should still be: grouped Annotate ≫ sequential; playback MA SG ≪ Ripper; staggered eager p95 ≪ wait-5 (~5 s).

**Native merged bf16, quality-safe best** (mean inference, s) — often not the 20-core cell:

| Model | CPU | GPU |
|---|---:|---:|
| EQCCT | 1.41 | 0.55 |
| PhaseNet | 0.38 | 0.32 |
| PhaseNetLight | 0.50 (fp32 was faster on CPU; this suite still runs bf16) | 0.32 |
| EQTransformer | 0.99 | 0.43 |
| EQT-NC | 0.95 | 0.37 |

**Playback MA SG bf16** (makespan, s) at locked K:

| Model | CPU K=20 | GPU K=4 |
|---|---:|---:|
| EQCCT | 0.58 | 0.219 |
| PhaseNet | 0.167 | 0.184 |
| PhaseNetLight | 0.282 | 0.169 |
| EQTransformer | 0.438 | 0.509 |
| EQT-NC | 0.408 | 0.449 |

**Staggered MA SG eager p95 finish−ready** (s) at the same K:

| Model | CPU K=20 | GPU K=4 |
|---|---:|---:|
| EQCCT | 0.518 | 0.199 |
| PhaseNet | 0.165 | 0.176 |
| PhaseNetLight | 0.214 | 0.281 |
| EQTransformer | 0.344 | 0.471 |
| EQT-NC | 0.346 | 0.443 |

---

## Prerequisites on the other computer

- Linux (or WSL2). Not native Windows (`taskset`, `sched_setaffinity`, Ray).
- Python env with RAPID + SeisBench + PyTorch (+ CUDA if GPU). Prefer `RAPID_PYTHON=/path/to/python`.
- STEAD networks present:

```
RAPID/data/seisbench_networks/stead_250st/manifest.json
RAPID/data/seisbench_networks/stead_580st/manifest.json
```

Copy the whole `stead_*st` trees if missing. Do not rebuild with a different seed if you want pick F1 comparable to the workstation.

- Enough RAM/VRAM: workstation CPU MA used ~11–13 GB PSS at 20 actors; EQCCT GPU is two branches. If actors OOM, set `CPU_K_CAP` / `GPU_K_CAP` (e.g. laptop `GPU_K_CAP=2`) and record that in the run README. Do not silently drop EQCCT.

---

## Commands

Always `cd` into the RAPID root (`.../eqcctpro/RAPID`).

### 0. Count cells (default = 240)

```bash
python benchmarks/fair/locked_recipe_transfer_matrix.py --print-matrix | head
```

Default: 5 models × 2 networks × 2 devices × 4 core budgets × 3 layers = **240** cells × **5** repeats.

### 1. Smoke (required first)

```bash
export RAPID_PYTHON=$(command -v python)   # or the conda env
export RESULTS_ROOT=results/locked_recipe_transfer/$(hostname -s)_smoke
SMOKE=1 bash benchmarks/isolation/run_iso_locked_recipe_transfer.sh
```

Smoke = PhaseNet, 250 stations, 5 cores, 1 repeat, all three layers, CPU and GPU (unless `SKIP_GPU=1`). Confirm `result.json` `timing.success_rate == 1.0` before the full grid.

### 2. Full transfer (serial, safest)

```bash
export RESULTS_ROOT=results/locked_recipe_transfer/$(hostname -s)_$(date +%Y-%m-%d)
# Optional pinning (laptop hybrid CPU: one logical ID per P-core):
# export CORES=0,2,4,6,8
# export GPU_K_CAP=2
# export SKIP_GPU=1          # CPU-only machine
bash benchmarks/isolation/run_iso_locked_recipe_transfer.sh
```

Resume-safe. Default is **one cell at a time**. Pinning: CPU trials use `CPU_BASE` (default 0); GPU trials use `GPU_BASE` (default 0) and `GPU_ID` (default 0).

Threadripper-style disjoint GPU slot (only if those cores exist and are free):

```bash
CPU_BASE=0 GPU_BASE=20 GPU_ID=1 PARALLEL=1 \
  bash benchmarks/isolation/run_iso_locked_recipe_transfer.sh
```

Do **not** overlap this with other RAPID jobs.

### 3. Watch

```bash
RESULTS_ROOT=results/locked_recipe_transfer/<run> \
  bash benchmarks/isolation/watch_locked_recipe_transfer.sh --once
```

Omit `--once` to refresh every 15 s.

### Useful env

| Variable | Default | Role |
|---|---|---|
| `LAYER` | `all` | `native`, `playback`, `staggered`, or comma list |
| `MODELS` | all five | comma list |
| `STATIONS` | `250,580` | |
| `CORE_GRID` | `5,10,15,20` | same sweep as the precision study |
| `DEVICES` | `cpu,gpu` | |
| `SKIP_GPU` / `SKIP_CPU` | unset | |
| `CPU_K_CAP` / `GPU_K_CAP` | unset | actor caps |
| `REPEATS` | `5` | smoke uses `1` |
| `CORES` | unset | explicit affinity list; skip cell if fewer IDs than `n_cpus` |
| `DRY_RUN` | `0` | print matrix, do not run |

---

## Result paths

Native:

```
$RESULTS_ROOT/annotate_bf16/stead/{N}st/{model}/{cpu|gpu}/cpus{C}/thr{C}/bs512/merged/xfer/result.json
```

Playback / staggered:

```
$RESULTS_ROOT/ma/annotate_bf16/stead/{N}st/{model}/{cpu|gpu}/kma{K}_krp0/sg/{playback|staggered}/{static|eager}/bs512/xfer/result.json
```

---

## What to report back

1. Host: CPU model, logical CPU count, GPU name / VRAM, `RAPID_PYTHON`, `CORES` / caps used.
2. Smoke pass/fail.
3. `done / expected` and any failed cells (log under `$RESULTS_ROOT/parallel_logs/`).
4. For **580 × 20-core CPU** and **580 × GPU at K=4** (or the capped K you actually ran): native inference, playback makespan, staggered p95, catalog P/S F1 — one row per model.
5. Whether **within-machine order** matches the workstation (bf16 merged works; MA SG playback is much faster than native sequential would be; staggered eager p95 is << 5 s, not ~90 s makespan).

Do **not** change the locked recipe because this machine is slower in absolute seconds.

---

## Files

| Path | Role |
|---|---|
| `benchmarks/fair/locked_recipe_transfer_matrix.py` | Cell list + K policy + `--print-matrix` / `--status` |
| `benchmarks/isolation/run_iso_locked_recipe_transfer.sh` | Isolated launcher (resume-safe) |
| `benchmarks/isolation/watch_locked_recipe_transfer.sh` | Progress |
| `benchmarks/fair/run_annotate_precision_trial.py` | Native merged trial |
| `benchmarks/fair/run_orch_annotate_trial.py` | Playback / staggered trial |

---

## Do not

- Re-run the 832-cell dtype matrix or the 1040-cell orch matrix on the workstation “to compare.”
- Set `QUICK=1` on the old orch launcher; that re-enables skipped Ripper/fp32 cells.
- Give each actor 20 torch threads.
- Sleep real 90 s delays (virtual idle is already in dispatch).
- Commit or push unless the user asks.
