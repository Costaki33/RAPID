# Locked-recipe transfer — what was tested vs what was locked

Host: BEGE-TEXA75535L (i7-13700H, RTX 4050 Laptop 6 GB, WSL2)
Run: `results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19`
Date: 2026-08-19 (GPU K=4 re-run 2026-08-20; CPU K=20 re-run 2026-08-20)

## Locked recipes (from workstation; do not change)

| Layer | Recipe |
|---|---|
| native | annotate_bf16, batch 512, packaging merged, threads = n_cpus |
| playback | Model-Actor, SG, fill static, 1 thread/actor |
| staggered | Model-Actor, SG, fill eager, 1 thread/actor |

Locked worker counts **K** (before machine caps):

| Device | Locked want |
|---|---|
| GPU most models | K=4 |
| GPU PhaseNet @ 250 stations | K=2 |
| CPU @ 580 (and EQCCT @ 250) | K=20 |
| CPU @ 250 other models | K=10 |

Then: `K = min(want, n_cpus, CPU_K_CAP or GPU_K_CAP)`.

## Caps used on this laptop (must note)

| Cap | Value | Why |
|---|---|---|
| `GPU_K_CAP` | **4** (after re-run; was 2) | Match locked GPU K; 6 GB VRAM was enough |
| `CPU_K_CAP` | **unset for K=20 re-run** (was **10**) | Initial campaign capped at K=10 for RAM; follow-up lifts cap to reach workstation K=20 |
| `CORES` | 0–19 (20 logical) | full CORE_GRID 5/10/15/20; K=20 needs all 20 IDs |

## What that means for reporting

**Native:** tested as locked (bf16 / batch 512 / merged) across the core grid. OK.

**CPU orchestration:** locked *shape* (MA+SG static/eager).
- Initial: at `n_cpus≥10` → **K=10** (`CPU_K_CAP=10`), not workstation K=20.
- Follow-up (`scripts/rerun_cpu_orch_k20.sh`): `CPU_K_CAP` unset, `CORE_GRID=20`, `DEVICES=cpu` → target **K=20** for 580 all models + 250 EQCCT (12 unique `kma20_*` paths). **RAM risk** on laptop/WSL; if OOM, try `CPU_K_CAP=15` and note it.

**GPU orchestration:** locked *shape*; after re-run with `GPU_K_CAP=4`:
- PhaseNet @ 250: **K=2** (matches lock)
- All other GPU orch cells: **K=4** (matches lock)

SKIP lines for orch rows that share an existing K path are expected duplicates. They are not missing locked configs.

## Valid claim shape

- “On this laptop we transferred the locked bf16 / MA-SG recipes.”
- “GPU MA used locked K=4 (K=2 for PhaseNet@250) after `GPU_K_CAP=4` re-run.”
- “CPU MA initially used K≤10 (`CPU_K_CAP=10`); a follow-up unset the cap to compare at workstation K=20 (disclose RAM risk / any OOM fallback).”
- Do **not** claim the laptop proved a new best K; do **not** invent new recipes.

## Follow-up commands

```bash
# CPU K=20 (tmux-friendly)
bash ~/RAPID/scripts/rerun_cpu_orch_k20.sh

# Watch
tail -f ~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/locked_recipe_transfer.log
python ~/RAPID/scripts/audit_locked_k20_missing.py

# OOM fallback only
CPU_K_CAP=15 bash ~/RAPID/scripts/rerun_cpu_orch_k20.sh
```
