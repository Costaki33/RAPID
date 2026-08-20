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
| SeisBench (this laptop) | **0.12.5** (upgraded mid-campaign from 0.10.2 so `EQCCTP`/`EQCCTS` exist) |
| SeisBench (workstation / original-data lock) | **0.11.8** — version differ; catalog F1 still matches workstation picks |
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
Native core grid: **5, 10, 15, 20** (80 native cells complete).
Orch unique K coverage (this package): **CPU K=5 / 10 / 20**; **GPU K=4** (+ PhaseNet@250 **K=2**).
**Intentionally skipped:** CPU orch **K=15** (bracketed by K=10 and K=20; do not run).
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

**Affinity:** `CORES=0,1,2,...,19` (all 20 logical CPUs) so native 15/20 core-grid cells were not skipped for missing IDs.

**Important about SKIP lines in the log:** orchestration result paths are keyed by **K** (`kma{K}/...`), not by `n_cpus`.
Once a successful `kma4` (or `kma2` / `kma20`) file exists, matrix rows for other `n_cpus` with the same K are logged as SKIP (duplicate path).
Those are **not** missing locked configs — skip duplicate `n_cpus` rows that share the same K path.

**CPU K=20 follow-up:** first matrix used `CPU_K_CAP=10`; then `scripts/rerun_cpu_orch_k20.sh` unset the cap for locked comparison at K=20.
Peak process set size (PSS) on this box was ~**9–11 GB** at K=20 — a laptop / hybrid-CPU / WSL result.
On this host, CPU **K=20 is often worse than K=10** for staggered p95; that is **not** a reason to change the workstation lock.

### Caps / conditions the paper must disclose

1. **SeisBench 0.12.5** on this laptop vs workstation **0.11.8**.
2. **WSL memory cap 24 GB**; GPU is **RTX 4050 Laptop 6 GB**.
3. **Caps story:** first matrix with **`CPU_K_CAP=10`**, then uncapped **K=20** follow-up for locked comparison; **`GPU_K_CAP=4`** (PhaseNet@250 → K=2).
4. **Do not** recommend changing the workstation lock because of K=20 laptop results.
5. Unique orch coverage: CPU K=5/10/20, GPU K=4 (+ PhaseNet 250 K=2); SKIP duplicate `n_cpus` rows that share the same K path; intentionally skip CPU K=15.
6. **Absolute times will be slower** than Threadripper + RTX 6000 Ada; claim **within-machine ordering**, not absolute equality (e.g. native EQCCT CPU ~47 s here vs ~1.41 s on workstation).
7. Staggered **makespan ~90 s** is expected (simulated delay ceiling); rank **p95 finish−ready** (GPU staggered p95 is **0.2–1.4 s** here — not makespan, not wait-5).
8. EQCCT is SeisBench **`EQCCTP` + `EQCCTS`**, not a single WaveformModel / not TF EQCCTPro for this suite.
9. This laptop package is enough for the transfer question: locked recipes still work, GPU stays the fast path, catalog picks match workstation.

---

## 3. Result inventory

| Artifact | Path in this handoff |
|---|---|
| Tidy table (with catalog P/S F1) | `data/transfer_summary.csv` |
| Interactive plots | `figures/transfer_canvas.html` |
| Full raw `result.json` tree | `raw_results/` (152 successful cells; also under WSL `~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/`) |
| Run log | `provenance/locked_recipe_transfer.log` |
| Caps note | `provenance/LOCKED_VS_ACHIEVED_K.md` / this document |
| Index | `README.md` |

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
success_rate_1.0_rows=152
failures=0
```

**152** successful `result.json` rows in this package; **all** have `success_rate=1.0`; **zero** failures; native **80** cells complete.

Metrics encoded in `runtime_s` / `metric`:

| method | metric column meaning |
|---|---|
| native | `timing.inference_s_mean` |
| playback | `orch.makespan_s_mean` |
| staggered | `latency.pooled_across_repeats.e2e_finish_minus_ready.p95` |

Catalog quality columns: `p_f1` / `s_f1` from `pick_quality_vs_catalog` (`P.f1_mean` / `S.f1_mean`).
Expected workstation catalog picks (examples): EQCCT ~**0.905 / 0.946**, PhaseNet ~**0.974 / 0.937**.

---

## 4. Paper-facing numbers (this laptop)

### 4a. Native merged bf16 — mean inference (s)

Prefer the fastest core cell that is quality-safe; table below lists **cpus=20** (full grid available) and GPU at cpus=20.

| Model | Stations | CPU cpus20 | GPU | P F1 | S F1 |
|---|---:|---:|---:|---:|---:|
| EQCCT | 250 | 22.792 | 0.733 | 0.8979 | 0.9424 |
| EQCCT | 580 | 47.215 | 1.650 | 0.9052 | 0.9464 |
| PhaseNet | 250 | 0.988 | 0.376 | 0.9641 | 0.9336 |
| PhaseNet | 580 | 2.380 | 0.944 | 0.9743 | 0.9365 |
| PhaseNetLight | 250 | 0.419 | 0.340 | 0.9409 | 0.9438 |
| PhaseNetLight | 580 | 1.032 | 0.826 | 0.9582 | 0.9445 |
| EQTransformer | 250 | 3.724 | 0.621 | 0.9064 | 0.9608 |
| EQTransformer | 580 | 8.384 | 1.312 | 0.9143 | 0.9656 |
| EQT-NC | 250 | 3.650 | 0.488 | 0.9083 | 0.9630 |
| EQT-NC | 580 | 8.771 | 1.063 | 0.9143 | 0.9639 |

### 4b. Playback MA SG — makespan (s) at achieved K

| Model | Stations | CPU K=5 | CPU K=10 | CPU K=20 | GPU K | GPU makespan |
|---|---:|---:|---:|---:|---:|---:|
| EQCCT | 250 | 15.743 | 10.445 | 11.196 | 4 | 0.651 |
| EQCCT | 580 | 36.921 | 24.489 | 26.204 | 4 | 1.262 |
| PhaseNet | 250 | 1.681 | 0.997 | — | 2 | 0.347 |
| PhaseNet | 580 | 3.357 | 2.377 | 2.289 | 4 | 0.442 |
| PhaseNetLight | 250 | 0.379 | 0.271 | — | 4 | 0.241 |
| PhaseNetLight | 580 | 0.794 | 0.559 | 0.571 | 4 | 0.389 |
| EQTransformer | 250 | 4.872 | 3.236 | — | 4 | 0.989 |
| EQTransformer | 580 | 11.482 | 7.659 | 6.803 | 4 | 1.436 |
| EQT-NC | 250 | 4.897 | 3.121 | — | 4 | 0.693 |
| EQT-NC | 580 | 11.751 | 7.493 | 7.133 | 4 | 1.178 |

### 4c. Staggered MA SG eager — p95 finish−ready (s)

| Model | Stations | CPU K=5 | CPU K=10 | CPU K=20 | GPU K | GPU p95 |
|---|---:|---:|---:|---:|---:|---:|
| EQCCT | 250 | 14.440 | 9.407 | 8.617 | 4 | 0.681 |
| EQCCT | 580 | 36.729 | 24.442 | 34.430 | 4 | 1.230 |
| PhaseNet | 250 | 1.608 | 1.096 | — | 2 | 0.337 |
| PhaseNet | 580 | 3.140 | 2.308 | 2.345 | 4 | 0.431 |
| PhaseNetLight | 250 | 0.410 | 0.289 | — | 4 | 0.226 |
| PhaseNetLight | 580 | 0.895 | 0.558 | 0.730 | 4 | 0.379 |
| EQTransformer | 250 | 5.099 | 3.587 | — | 4 | 1.032 |
| EQTransformer | 580 | 11.940 | 7.156 | 9.550 | 4 | 1.387 |
| EQT-NC | 250 | 4.404 | 2.924 | — | 4 | 0.700 |
| EQT-NC | 580 | 11.525 | 6.790 | 8.476 | 4 | 1.129 |

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

1. **GPU ≪ CPU** for heavy models on this laptop (e.g. native EQCCT 580/cpus20 ~**47 s** CPU vs ~**1.65 s** GPU; workstation native EQCCT CPU is ~**1.41 s** — report **ordering**, not equality).
2. **Staggered GPU p95** at locked K is **0.2–1.4 s** — that is finish−ready latency, **not** makespan (~90 s) and **not** wait-5.
3. **CPU orch K=10 beats K=5**; on this box **K=20 is often worse than K=10** for staggered — laptop/hybrid-CPU result; **do not** change the workstation lock.
4. **PhaseNetLight** remains the fastest native/CPU-orch path among the five on this host.
5. Catalog P/S F1 matches workstation picks (EQCCT ~0.905/0.946, PhaseNet ~0.974/0.937).
6. This package answers the transfer question: locked recipes still work; GPU stays the fast path.

---

## 6. Campaign chronology / pitfalls (so the next model does not rediscover them)

1. Initial zip install had no git; later synced to `origin/main` (networks committed in-repo).
2. First XPS-style `CORES=0,2,4,...,18` (10 IDs) skipped 15/20 cells; fixed to full `0..19`.
3. SeisBench 0.10.2 lacked `EQCCTP`; upgraded to **0.12.5** before EQCCT succeeded (workstation lock used **0.11.8**).
4. First GPU orch pass used `GPU_K_CAP=2`; those trees were cleared and **re-run at `GPU_K_CAP=4`** (locked).
5. First CPU orch matrix used `CPU_K_CAP=10`; K=20 follow-up unset the cap (peak PSS ~9–11 GB). Intentionally **no** CPU K=15 orch.
6. Long runs must avoid Windows sleep; WSL suspends with the host.
7. This handoff folder is self-contained (`data/`, `figures/`, `raw_results/`, `provenance/`); WSL source also under `~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/`.

---

## 7. How to regenerate plots / handoff

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rapid
cd ~/RAPID   # or /mnt/c/Users/cgs2528/Projects/RAPID
python scripts/plot_locked_transfer_results.py
python scripts/assemble_locked_transfer_handoff.py
```

Plots write under `~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/`; assemble copies CSV + `figures/transfer_canvas.html` into this handoff folder.

---

## 8. Recommended paper wording

> We transferred the locked Annotate bf16 / Model-Actor SG recipes to a consumer laptop
> (Intel i7-13700H, RTX 4050 Laptop 6 GB, WSL2 capped at 24 GB; SeisBench 0.12.5 vs workstation 0.11.8).
> Native cells used the 5–20 logical-CPU grid (80 cells). GPU Model-Actor used locked K=4
> (K=2 for PhaseNet at 250 stations). CPU Model-Actor first used K≤10 (`CPU_K_CAP=10`), then an
> uncapped K=20 follow-up for locked comparison (peak PSS ~9–11 GB; K=20 often slower than K=10
> for staggered on this hybrid CPU — we do not change the workstation lock). Unique orch coverage
> is CPU K=5/10/20 and GPU K=4; CPU K=15 was intentionally skipped. Catalog P/S F1 matched
> workstation picks. We report within-machine method ordering (GPU ≪ CPU for heavy models;
> staggered GPU p95 0.2–1.4 s) and do not equate absolute latencies to the workstation.

---

Assembled: 2026-08-20T17:24:46.800388+00:00
