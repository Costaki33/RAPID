# Locked-recipe transfer handoff — BEGE-TEXA75535L

**Audience:** another model / coauthor preparing the RAPID paper.
**Purpose:** this laptop is being returned; this folder is the portable record of the locked-recipe transfer run.

Parent protocol: `docs/RAPID_LOCKED_RECIPE_TRANSFER.md` (do not invent new recipes or K sweeps).

---

## 1. Machine under test

| Item | Value |
|---|---|
| Hostname | BEGE-TEXA75535L |
| Host OS | Windows 11 + **WSL2 Ubuntu 26.04** (benchmarks require Linux: `taskset` / `sched_setaffinity` / Ray) |
| CPU | 13th Gen Intel Core **i7-13700H** (host: 14C/20T; WSL exposed 20 logical CPUs) |
| RAM | ~32 GB host; WSL capped **24 GB** via `%UserProfile%\.wslconfig` |
| GPU | NVIDIA GeForce **RTX 4050 Laptop**, **6141 MiB** (~6 GB) |
| NVIDIA driver (Windows) | see `provenance/host_snapshot.txt` / nvidia-smi |
| Power | Keep AC connected for long runs; sleep/standby suspends WSL and can interrupt trials |

### WSL config used

```ini
[wsl2]
memory=24GB
processors=20
swap=8GB
```

### Software stack (conda env `rapid`)

```
/home/cgs2528/miniconda3/envs/rapid/lib/python3.10/site-packages/obspy/core/util/base.py:26: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
python 3.10.20
torch 2.5.1+cu124 cuda 12.4 avail True
gpu_name NVIDIA GeForce RTX 4050 Laptop GPU
vram_bytes 6438846464
bf16 True
seisbench 0.12.5
ray 2.42.1
numpy 1.26.4
obspy 1.4.1
```

| Note | Detail |
|---|---|
| SeisBench | Upgraded mid-campaign from **0.10.2 → 0.12.5** so `EQCCTP`/`EQCCTS` exist; early EQCCT native failures were from missing attributes, not model physics |
| RAPID git | see `provenance/host_snapshot.txt` |
| Networks | Repo-provided `data/seisbench_networks/stead_250st` and `stead_580st` (same trees as workstation transfer; do not rebuild with a new seed if F1 must match) |

---

## 2. What was run (locked recipes)

### Layers (unchanged from workstation lock)

| Layer | Meaning | Locked recipe |
|---|---|---|
| **native** | single-process Annotate | `annotate_bf16`, batch **512**, packaging **merged**, torch/OMP threads = `n_cpus` |
| **playback** | all-stations-ready orch | Model-Actor, **SG**, fill **static**, 1 thread/actor |
| **staggered** | delayed-station orch | Model-Actor, **SG**, fill **eager**, 1 thread/actor |

Models: **EQCCT, PhaseNet, PhaseNetLight, EQTransformer, EQT-NC**.
Networks: STEAD **250** and **580**.
Devices: **CPU** and **1× GPU**.
Core grid: **5, 10, 15, 20**.
Repeats: **5** (smoke used 1 earlier; final package is the full run).

### Locked worker counts K (workstation) vs achieved on this laptop

```
K = min(locked_want, n_cpus, CPU_K_CAP or GPU_K_CAP)
```

| Setting | Locked want | This laptop |
|---|---|---|
| GPU (most models) | **K=4** | **K=4** after GPU re-run (`GPU_K_CAP=4`) |
| GPU PhaseNet @ 250 | **K=2** | **K=2** (matches lock) |
| CPU @ 580 / EQCCT | **K=20** | **K=20** after uncapped re-run (was K≤10 with `CPU_K_CAP=10`) |
| CPU @ 250 others | **K=10** | **K=5** at cpus=5; **K=10** at cpus≥10 |

**Affinity:** `CORES=0,1,2,...,19` (all 20 logical CPUs) so 15/20 core-grid cells were not skipped for missing IDs.

**Important about SKIP lines in the log:** orchestration result paths are keyed by **K** (`kma{K}/...`), not by `n_cpus`.
Once a successful `kma4` (or `kma2` / `kma20`) file exists, matrix rows for other `n_cpus` with the same K are logged as SKIP (duplicate path).
Those are **not** missing locked configs.

**CPU K=20 follow-up:** `scripts/rerun_cpu_orch_k20.sh` unsets `CPU_K_CAP` and runs `CORE_GRID=20` for playback+staggered CPU. Disclose RAM/WSL risk; OOM fallback is `CPU_K_CAP=15` only.

### Caps / conditions the paper must disclose

1. **`CPU_K_CAP`** — initial campaign used **10**; K=20 re-run unsets the cap for locked comparison (disclose RAM risk / any OOM fallback).
2. **`GPU_K_CAP=4`** on the successful GPU orch re-run — matches locked GPU K (6 GB VRAM was enough; an earlier attempt used K=2 then results were cleared and re-run at K=4).
3. **Absolute times will be slower** than Threadripper + RTX 6000 Ada; claim **within-machine ordering**, not absolute equality.
4. Staggered **makespan ~90 s** is expected (simulated delay ceiling); rank **p95 finish−ready**, not makespan.
5. EQCCT is SeisBench **`EQCCTP` + `EQCCTS`**, not a single WaveformModel / not TF EQCCTPro for this suite.
6. Do **not** treat SKIP-duplicate orch rows as failures or as a reason to change the locked recipe.

---

## 3. Result inventory

| Artifact | Path in this handoff |
|---|---|
| Tidy table | `data/transfer_summary.csv` |
| Interactive plots | `figures/transfer_canvas.html` |
| Full raw `result.json` tree | on the machine under `~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/` (too large to duplicate here; CSV is the analysis extract) |
| Run log | `provenance/locked_recipe_transfer.log` |
| Caps note | `provenance/LOCKED_VS_ACHIEVED_K.md` / this document |

### Successful cells in CSV

```
method     device
native     cpu       40
           gpu       40
playback   cpu       26
           gpu       10
staggered  cpu       26
           gpu       10
total_rows=152
```

Metrics encoded in `runtime_s` / `metric`:

| method | metric column meaning |
|---|---|
| native | `timing.inference_s_mean` |
| playback | `orch.makespan_s_mean` |
| staggered | `latency.pooled_across_repeats.e2e_finish_minus_ready.p95` |

---

## 4. Paper-facing numbers (this laptop)

### 4a. Native merged bf16 — mean inference (s)

Prefer the fastest core cell that is quality-safe; table below lists **cpus=20** (full grid available) and GPU at cpus=20.

| Model | Stations | CPU cpus20 | GPU |
|---|---:|---:|---:|
| EQCCT | 250 | 22.792 | 0.733 |
| EQCCT | 580 | 47.215 | 1.650 |
| PhaseNet | 250 | 0.988 | 0.376 |
| PhaseNet | 580 | 2.380 | 0.944 |
| PhaseNetLight | 250 | 0.419 | 0.340 |
| PhaseNetLight | 580 | 1.032 | 0.826 |
| EQTransformer | 250 | 3.724 | 0.621 |
| EQTransformer | 580 | 8.384 | 1.312 |
| EQT-NC | 250 | 3.650 | 0.488 |
| EQT-NC | 580 | 8.771 | 1.063 |

### 4b. Playback MA SG — makespan (s) at achieved K

| Model | Stations | CPU K=5 | CPU K=10 | GPU K | GPU makespan |
|---|---:|---:|---:|---:|---:|
| EQCCT | 250 | 15.743 | 10.445 | 4 | 0.651 |
| EQCCT | 580 | 36.921 | 24.489 | 4 | 1.262 |
| PhaseNet | 250 | 1.681 | 0.997 | 2 | 0.347 |
| PhaseNet | 580 | 3.357 | 2.377 | 4 | 0.442 |
| PhaseNetLight | 250 | 0.379 | 0.271 | 4 | 0.241 |
| PhaseNetLight | 580 | 0.794 | 0.559 | 4 | 0.389 |
| EQTransformer | 250 | 4.872 | 3.236 | 4 | 0.989 |
| EQTransformer | 580 | 11.482 | 7.659 | 4 | 1.436 |
| EQT-NC | 250 | 4.897 | 3.121 | 4 | 0.693 |
| EQT-NC | 580 | 11.751 | 7.493 | 4 | 1.178 |

### 4c. Staggered MA SG eager — p95 finish−ready (s)

| Model | Stations | CPU K=5 | CPU K=10 | GPU K | GPU p95 |
|---|---:|---:|---:|---:|---:|
| EQCCT | 250 | 14.440 | 9.407 | 4 | 0.681 |
| EQCCT | 580 | 36.729 | 24.442 | 4 | 1.230 |
| PhaseNet | 250 | 1.608 | 1.096 | 2 | 0.337 |
| PhaseNet | 580 | 3.140 | 2.308 | 4 | 0.431 |
| PhaseNetLight | 250 | 0.410 | 0.289 | 4 | 0.226 |
| PhaseNetLight | 580 | 0.895 | 0.558 | 4 | 0.379 |
| EQTransformer | 250 | 5.099 | 3.587 | 4 | 1.032 |
| EQTransformer | 580 | 11.940 | 7.156 | 4 | 1.387 |
| EQT-NC | 250 | 4.404 | 2.924 | 4 | 0.700 |
| EQT-NC | 580 | 11.525 | 6.790 | 4 | 1.129 |

### 4d. Workstation reference (580) — order-of-magnitude only

From `docs/RAPID_LOCKED_RECIPE_TRANSFER.md` (Threadripper + RTX 6000 Ada). **Do not** claim XPS/laptop times equal these.

| Model | WS native CPU | WS native GPU | WS playback CPU K20 | WS playback GPU K4 | WS stag p95 CPU | WS stag p95 GPU |
|---|---:|---:|---:|---:|---:|---:|
| EQCCT | 1.41 | 0.55 | 0.58 | 0.219 | 0.518 | 0.199 |
| PhaseNet | 0.38 | 0.32 | 0.167 | 0.184 | 0.165 | 0.176 |
| PhaseNetLight | 0.50 | 0.32 | 0.282 | 0.169 | 0.214 | 0.281 |
| EQTransformer | 0.99 | 0.43 | 0.438 | 0.509 | 0.344 | 0.471 |
| EQT-NC | 0.95 | 0.37 | 0.408 | 0.449 | 0.346 | 0.443 |

---

## 5. Within-machine observations (for the paper narrative)

Use these as qualitative checks against the handoff brief:

1. **GPU ≪ CPU** for native EQCCT/EQT-family on this laptop (large CPU native times for EQCCT ~47 s at 580/20 vs ~1.65 s GPU).
2. **Playback / staggered GPU** at locked K=4 stays sub-second to ~1.4 s makespan/p95 for 580 — still far below wait-5 (~5 s) territory for staggered p95.
3. **CPU orch K=10 beats K=5** consistently (actor parallelism helps under the cap).
4. **PhaseNetLight** remains the fastest native/CPU-orch path among the five on this host.
5. Absolute gap vs workstation is large (consumer GPU + hybrid CPU + WSL); keep claims comparative within this host and vs locked recipe *shape*.

---

## 6. Campaign chronology / pitfalls (so the next model does not rediscover them)

1. Initial zip install had no git; later synced to `origin/main` (networks committed in-repo).
2. First XPS-style `CORES=0,2,4,...,18` (10 IDs) skipped 15/20 cells; fixed to full `0..19`.
3. SeisBench 0.10.2 lacked `EQCCTP`; upgraded to **0.12.5** before EQCCT succeeded.
4. First GPU orch pass used `GPU_K_CAP=2`; those trees were cleared and **re-run at `GPU_K_CAP=4`** (locked).
5. Long runs must avoid Windows sleep; WSL suspends with the host.
6. Full raw results live on this disk under WSL `~/RAPID/...`; copy off-machine before returning the laptop if raw JSON is needed.

---

## 7. How to regenerate plots

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rapid
cd ~/RAPID
python scripts/plot_locked_transfer_results.py
```

---

## 8. Recommended paper wording

> We transferred the locked Annotate bf16 / Model-Actor SG recipes to a consumer laptop
> (Intel i7-13700H, RTX 4050 6 GB, WSL2). Native cells used the 5–20 logical-CPU grid.
> GPU Model-Actor used the locked worker count K=4 (K=2 for PhaseNet at 250 stations).
> CPU Model-Actor initially used K≤10 (`CPU_K_CAP=10`); a follow-up unset the cap to compare at
> workstation K=20 for 580 stations and EQCCT@250 (disclose RAM/WSL risk if any cell OOM’d).
> We report within-machine method ordering and do not equate absolute latencies to the workstation.

---

Assembled: 2026-08-20T17:04:20.398643+00:00
