Host transfer run — 2026-08-19T17:02:17,516261760-05:00
hostname: BEGE-TEXA75535L
CPU: Intel Core i7-13700H (WSL: 10 physical CORE IDs / 20 logical)
GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB
RAPID_PYTHON=/home/cgs2528/miniconda3/envs/rapid/bin/python
CORES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
GPU_K_CAP=2
CPU_K_CAP=10
Note: CORE_GRID cells with n_cpus > 10 isolated IDs are skipped.
Do not change locked recipes (bf16/batch512/MA-SG/eager).

GPU orch re-run 2026-08-20T10:46:02-05:00
LAYER=playback,staggered DEVICES=gpu GPU_K_CAP=4 CPU_K_CAP=10 CORES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
Intent: match locked GPU K=4; if OOM lower GPU_K_CAP and note VRAM.

GPU orch re-run 2026-08-20T10:46:23-05:00
LAYER=playback,staggered DEVICES=gpu GPU_K_CAP=4 CPU_K_CAP=10 CORES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
Intent: match locked GPU K=4; if OOM lower GPU_K_CAP and note VRAM.

CPU orch K=20 re-run 2026-08-20T11:31:20-05:00
LAYER=playback,staggered DEVICES=cpu CORE_GRID=20 GPU_K_CAP=4 CPU_K_CAP=<unset>
CORES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
Intent: workstation locked CPU K=20 (580 all + 250 EQCCT). RAM risk on laptop/WSL.
Fallback if OOM: CPU_K_CAP=15 then note failure; do not invent new recipes.
