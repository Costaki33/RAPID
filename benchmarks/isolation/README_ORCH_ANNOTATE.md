# Orchestration annotate tests — what we are measuring

This suite asks a practical question:

**On one isolated machine slot (20 CPU cores, or 20 cores plus one GPU), how should RAPID schedule SeisBench Annotate so a network of 250 or 580 stations finishes a minute of waveforms as fast as possible — including when some stations show up late?**

We do **not** test fp16 here. Every cell uses **annotate_fp32** or **annotate_bf16**, with batch size locked to the merged-network study winner (**512**).

---

## Two pictures of “the network is ready”

**Playback (all-ready).** Every station’s waveforms exist at t=0. This is the clean lab case: replay a recorded network. Stations are handed out round-robin and stay assigned to the same worker. Use this to compare against the native merged and sequential baselines, and to find how many workers (K) you actually want.

**Realtime / staggered (60-second chunks).** Not every station is on time. We randomly pick how many of the N stations are delayed, then each delayed station draws a delay from

`0, 5, 10, 15, 20, 25, 30, 60, 65, 90` seconds.

Delays under 60 s still land in the first minute. 60 / 65 / 90 s land in the **next** minute, so we can test spill.

Time is cut into **60-second waveform chunks** `[0,60)`, `[60,120)`, … like a real-time feed:

- At t=0 you might have 500 of 580 stations. Process those. Do not wait for the rest.
- At t=30, 40 more become ready. Process those too.
- The last 40 might not arrive until t=65. That is fine. They are processed in minute 1, together with whatever else becomes ready in that minute.

A station that misses the current chunk is not a failure. It is next-minute work.

---

## How work is packaged

**S1 — one station per call.** Closest to “one station per actor.” The native sequential bf16 trial is the serial floor. With even work, orchestration S1 should beat that by about K, until you oversubscribe threads or VRAM.

**SG — a merged group of stations per call.** `G = ceil(N / K)` (or ceil of the stations assigned to that pool, for hybrids). This is how merged-network Annotate got its speed: SeisBench batches windows across stations. Too many tiny groups lose that. Too few groups leave cores idle.

---

## How many workers (K)

Do not sweep K = 1…N. Use a log-ish ladder on the pinned slot, then pick the **elbow**: the smallest K whose makespan stops dropping much.

| Slot | Model-Actor K | Ripper K |
|---|---|---|
| CPU (cores 0–19) | 1, 2, 4, 5, 10, 20 | 1, 2, 4, 5 (reload is expensive) |
| GPU (20 host cores + 1 GPU) | 1, 2, 4 | 1, 2 |

VRAM will cap GPU K, especially EQCCT (two model branches).

---

## Ready-queue (realtime only)

Static round-robin is the wrong policy when stations arrive late: worker 3 can own 29 late stations while worker 0 sits idle.

Realtime uses a **ready queue** (work-stealing): the next free instance takes the next ready work.

Two ways to fill a group:

| Fill | Meaning |
|---|---|
| **eager** | If only the next station is ready, process it. Partial SG groups are allowed. Never idle waiting for a full group. |
| **w5 / w10** | Wait up to 5 or 10 seconds to collect G stations, but **always flush at the 60 s chunk boundary**. |

S1 is always eager (G = 1). Playback is always **static** (no queue).

---

## Who owns on-time vs delayed stations

**Model-Actor (MA):** the model stays loaded in K persistent Ray actors. Good when the same workers keep getting work.

**Ripper:** each task loads the model, runs, and the worker is torn down (`max_calls=1`). Good when you do not want models sitting warm through a 30–65 s gap — you pay reload instead.

Four compositions:

1. **MA only** — one actor pool does everything.
2. **Ripper only** — one Ripper pool does everything.
3. **MA on-time + Ripper delayed** — actors chew the t=0 mass; Ripper absorbs late stations.
4. **Ripper on-time + MA delayed** — one-shots burn the known burst; warm actors wait for the trickle.

Hybrids are realtime-only. Playback has no delayed pool.

On-time means `ready_t = 0`. Delayed means `ready_t > 0` (including 60/65/90 s spill into minute 1).

---

## What we record

Each cell is 5 isolated repeats. Per station we store finish−ready and start−ready time, so p90 / p95 / p99 have `N × 5` samples (1,250 or 2,900).

Also: makespan, idle fraction, how many stations fell in chunk 0 vs chunk 1, pick F1 vs the catalog, RAM / VRAM.

**Winner rule:** smallest K (or smallest hybrid split) at the makespan elbow; break ties on p95 finish−ready, then on pick quality.

---

## How to run (after sequential native is done)

Isolation slots are unchanged: CPU 0–19, GPU1 20–39, GPU0 40–59.

Ripper+S1 is omitted by default (`SKIP_RIPPER_S1=1`) except in `QUICK=1` smoke.
The eight smoke Ripper S1 controls (EQCCT/PhaseNet × CPU/GPU × two K values) stay on disk.

After playback, staggered/hybrid also skip fp32 (`SKIP_FP32_REALTIME=1`) and homogeneous Ripper (`SKIP_STAGGERED_RIPPER=1`). Hybrid polarities still put Ripper on one of the two pools. Idle delay gaps are simulated, not slept on the wall clock.

```bash
# Layer A — all-ready MA + Ripper SG, 250 and 580, CPU + GPU K ladders
# (Ripper S1 skipped; smoke controls already at this results root)
LAYER=playback bash benchmarks/isolation/run_iso_orch_annotate.sh

# Layer C — realtime ready-queue (eager / w5 / w10)
LAYER=staggered bash benchmarks/isolation/run_iso_orch_annotate.sh

# Layer D — hybrid polarities
LAYER=hybrid bash benchmarks/isolation/run_iso_orch_annotate.sh

# Everything (large)
LAYER=all bash benchmarks/isolation/run_iso_orch_annotate.sh

# Small smoke
QUICK=1 LAYER=playback bash benchmarks/isolation/run_iso_orch_annotate.sh
```

Watch:

```bash
LAYER=playback bash benchmarks/isolation/watch_orch_annotate_progress.sh
```

Results: `results/orch_annotate/stead_2026-08-14/`

Path shape:

`{composition}/{method}/stead/{N}st/{model}/{cpu|gpu}/kma{A}_krp{B}/{s1|sg}/{playback|staggered}/{static|eager|w5|w10}/bs512/orch_ann/result.json`

---

## Native floors these tests sit on

| Native trial | Orch twin |
|---|---|
| Merged-network annotate (this study’s 832-cell matrix) | SG playback |
| Sequential per-station annotate-bf16 | S1 playback |

If orch S1 is not ~K faster than sequential, the actors are not paying for themselves. If orch SG is much slower than merged native, grouping got too small.
