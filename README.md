# RAPID — faster-than-SeisBench's `annotate()` benchmarking toolkit

This repository is part of a larger project focused on enabling real-time seismic phase picking for seismic event detection using deep learning models. 

The preliminary work, [EQCCTPro/RAPID](https://github.com/ut-beg-texnet/eqcct/tree/main/eqcctpro), enabled sub-11s 3-C waveform processing using persistant model actors to handle 228 stations of 1-minute seismic data for production applications with the Texas Seismological Network (TexNet). This architecture was integrated into [SCMLPick](https://github.com/ut-beg-texnet/scmlpick), a SeisComP module that integrates deep learning models into the SeisComP interface for real-time seismic phase picking, serving as the backbone of the processing approach currently operational in producation at TexNet. 

Further work is focused on improving processing speeds beyond the persistent actor approach by combing different levels of numerical precision with batching. Batching has been applied in SeisBench's `annotate()`, and preliminary trials show that we can achieve faster processing than `annotate()` through these techniques. Prelimary results can be found [here](RAPID_Seisbench_speedup.pdf), with final trials are being finalized for publication in the near future.

## Models and backends
 
**Models**: PhaseNet, PhaseNetLight (3001-sample window), EQTransformer, EQT-NC (6000-sample window). EQCCT is a planned addition once it's integrated into SeisBench.
 
**Backends**:
- `baseline_annotate` — unmodified SeisBench
- `lean_pytorch` — FP32 / FP16 / BF16, with optional `torch.compile`
- `onnx` — ONNX Runtime (optional; only registered if the package imports)
- `tensorrt` — prebuilt `.plan` engines (optional; same)

## Setup: Conda environment

**1. Create and activate an env**
Make sure to match the Python version to the PyTorch CUDA wheels you will install.

```bash
conda create -n rapid python=3.11 -y
conda activate rapid
```

**2. Install the env library packages using environment.yml**


**3. Install optional backend dependencies(ONNX, ONNX Runtime GPU, and related helpers; see Optional backends**:

```bash
cd RAPID
pip install -r requirements-extra.txt
```

This assumes the core stack is already installed. Swap `onnxruntime-gpu` for `onnxruntime` in that file if you only need CPU inference. TensorRT comes from NVIDIA for your specific CUDA toolkit version. See the comments at the bottom of `requirements-extra.txt` for more info.

## Quick start

```bash
cd RAPID
 
# Single config sanity check, runs in ~a minute on one GPU
python scripts/run_benchmark.py \
    --dataset-dir /path/to/data/20241215T120000Z_20241215T120100Z \
    --model PhaseNet --child original \
    --backend lean_pytorch --dtype fp16 \
    --device cuda:0 --n-stations 228 --batch-size 256 --repeats 3
 
# Pipelined single-GPU (the fast path: parallel CPU preprocess
# feeding megabatched GPU forward with CPU<->GPU overlap)
python scripts/run_pipelined.py \
    --dataset-dir "$DATA_DIR" --model PhaseNet --child original \
    --n-stations 580 --batch-size 256 --dtype fp16 \
    --mode single_gpu --n-cpu-workers 16 --repeats 3
 
# Fair dual-GPU baseline: SeisBench annotate() on 2 GPUs, stations split 50/50
python scripts/run_pipelined.py \
    --dataset-dir "$DATA_DIR" --model PhaseNet --child original \
    --n-stations 580 --mode baseline_dual_gpu --repeats 3
 
# Pipelined dual-GPU: each GPU shard runs its own CPU preprocess pool
python scripts/run_pipelined.py \
    --dataset-dir "$DATA_DIR" --model PhaseNet --child original \
    --n-stations 580 --batch-size 512 --dtype bf16 \
    --mode dual_gpu --n-cpu-workers 8 --repeats 3
 
# Full matrix (all 4 models x 4 station counts x 5 backends x 9 batch sizes x 3 repeats)
python scripts/run_matrix.py --config configs/full_matrix.json
 
# Generate plots from the outputted JSONL file
python scripts/make_plots.py --jsonl results/matrix.jsonl --out-dir figures
```

## ONNX / TensorRT - Optional backends
 
After installing the extra dependencies, you can export pretrained weights to ONNX
 
```bash
# ONNX only
python scripts/export_models.py --onnx-dir models_exported/onnx --skip-trt
 
# ONNX + TRT engines (pick the opt batch for your most common shape)
python scripts/export_models.py \
    --onnx-dir models_exported/onnx \
    --trt-dir  models_exported/trt \
    --opt-batch 228 --max-batch 1024
```
 
Then add the exported paths to `configs/full_matrix.json`:
 
```json
{ "name": "onnx",     "dtype": "fp32", "onnx_path": "models_exported/onnx/PhaseNet_original.onnx" },
{ "name": "tensorrt", "dtype": "fp16", "engine_path": "models_exported/trt/PhaseNet_original_fp16.plan", "max_batch_size": 1024 }
```

## Pick quality
 
Pick quality is evaluated against catalog ground truth on the SeisBench evaluation traces. Every trial in the dtype / timing matrix appends `pick_quality`, including median absolute onset offset vs catalog P and S (`onset_delta_*_vs_catalog` in samples at model sampling rate).
 
```bash
cd RAPID
python scripts/run_seisbench_matrix.py --config configs/seisbench_dtype_matrix.json
```
 
Use `traces_per_dataset` in the JSON config to control how many traces are drawn per dataset (100 is standard for the publication matrix).
 
For a quick same-waveform FP16 vs FP32 comparison on any local miniSEED chunk (no catalog needed):
 
```bash
python scripts/compare_fp16_fp32.py \
    --dataset-dir /path/to/timechunk \
    --model PhaseNet --child original \
    --device cuda:0 --n-stations 228 \
    --out-json results/fp16_vs_fp32_PhaseNet.json
```
 
Reports probability trace drift (MAE, max absolute error, RMSE, Pearson correlation), pick-time delta at threshold (median, p95, max — in samples at model sr), and FP16 speedup over FP32.
 
For a broader sweep of probability and pick drift vs FP32 on miniSEED workloads, there's also `scripts/run_quality_matrix.py`.


## What each timed stage means

| Stage                  | What happens                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `merge_streams`        | (baseline only) concatenating all station ObsPy Streams for `model.annotate()`.    |
| `annotate_end_to_end`  | (baseline only) all of SeisBench's internal pipeline, end-to-end.                  |
| `preprocess`           | SeisBench's `annotate_stream_pre` (filter, resample) run once per station.         |
| `window_cut_and_stack` | Build a single `(N_total_windows, 3, in_samples)` numpy array across all stations. |
| `forward`              | Backend's `infer_chunked` — the model forward pass (CUDA-synchronized).            |


Baseline collapses the lean stages into `annotate_end_to_end`; the lean backends
expose them separately so we can see *where* the speedup comes from.

### EQCCTPro orchestration sweep (`cpu_test_results.csv`)

Runs driven by `scripts/run_seisbench_sweep.py` (Ripper, Model-Actor, Slipstream)
write timing rows to `results/orchestration_sweep/.../timing/cpu_test_results.csv`.
Those columns use a different schema from the lean benchmark stages above
(actor creation, per-task preprocess/inference/write, sum compute, scheduling residual).

See **[docs/cpu_test_results_timing.md](docs/cpu_test_results_timing.md)** for
column definitions and how to interpret `Sum Task Compute Time` vs wall-clock metrics.

### Parallel orchestrator (faster sweeps via CPU/GPU isolation)

`scripts/run_parallel_sweep.py` runs the three strategies **concurrently** to cut
wall-clock time, giving each one an isolated slice of hardware so they don't fight
over cores or a GPU:

- **CPU phase** — all strategies at once on disjoint core ranges (with `--max-cpus 20`):
  Ripper -> cores `0-19`, Model-Actor -> `20-39`, Slipstream -> `40-59`. Each worker
  is `taskset`-pinned and gets its own Ray temp dir; results still land in the same
  per-strategy `cpu_test_results.csv` files (no contention).
- **Single-GPU phase** — 1 GPU per strategy, rotated across the available GPUs
  (`--gpus 0,1`), so up to two strategies run at once (e.g. Ripper on GPU 0,
  Model-Actor on GPU 1, then Slipstream rotates back to GPU 0). Each GPU slot also
  gets a disjoint CPU range for its preprocessing workers.
- **Dual-GPU phase** (run *after* the single-GPU phase) — each strategy uses **both**
  GPUs (`--gpu-ids 0,1 --min-gpus 2`), run sequentially since both GPUs are busy. The
  new `--min-gpus 2` skips re-running the 1-GPU points already recorded.

**RAM is the real limiter** (each actor ≈ 2.3 GB). Running three strategies at once
could OOM a 500 GB box if each one independently capped against the whole machine. The
orchestrator prevents this by giving every concurrently-running strategy a **disjoint
RAM slice** via the `EQCCTPRO_RAM_BUDGET_MB` env var, which `EvaluateSystem` and the Ray
workers honor when capping actors. With `--max-parallel 3` on ~500 GB, each strategy gets
~153 GB (≈ 66 actors); the three together stay under budget.

- `--ram-budget-gb` total RAM to plan against (default: autodetect installed RAM).
- `--ram-reserve-gb` held back for OS/driver/page cache (default 40).
- `--max-parallel` strategies running at once. `1` = fully **sequential**, and each
  strategy then gets the *full* budget (max actors) — slowest wall-clock but maximal
  per-strategy concurrency. The default runs all selected strategies at once.

```bash
# Prepare networks once, then CPU + single-GPU phases (all 3 in parallel, RAM split):
python scripts/run_parallel_sweep.py --phase cpu+gpu --ram-budget-gb 500

# Safer middle ground: two at a time, each gets half the budget:
python scripts/run_parallel_sweep.py --phase cpu --ram-budget-gb 500 --max-parallel 2

# Fully sequential (each strategy uses the whole budget / max actors):
python scripts/run_parallel_sweep.py --phase cpu --ram-budget-gb 500 --max-parallel 1

# Dual-GPU phase later (2-GPU points only):
python scripts/run_parallel_sweep.py --phase dual-gpu
```

The underlying `scripts/run_seisbench_sweep.py` gained `--cpu-offset` (affinity base),
`--only-strategies` (filter `--run-all`), and `--min-gpus` (skip smaller GPU counts).
Per-strategy logs go to `results/logs/parallel/`.

### Native SeisBench ``classify()`` / ``annotate()`` sweep

Direct SeisBench API baselines (no Ray orchestration) on the same 250/580 station
networks: ``scripts/run_native_seisbench_sweep.py``. Results under
``results/native_seisbench_sweep/``. See the script docstring for ``--run-all``.

### The method families (evolution of speedups)

Every row in `results/matrix.jsonl` falls into one of these families. They're
recorded as distinct `kind` + `variant` combinations so analysis scripts can
tell them apart and plot the evolution side by side.


| #   | Kind               | Variant suffix                        | What it is                                                                                                                                                         |
| --- | ------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `baseline`         | (none)                                | SeisBench's `model.annotate()` on one device (CPU or CUDA).                                                                                                        |
| 2   | `dual_gpu`         | `2gpu_baseline`                       | SeisBench's `model.annotate()` run in parallel on 2 GPUs, stations split 50/50.                                                                                    |
| 3   | `single`           | (none)                                | Lean path, 1 GPU, single-threaded preprocess.                                                                                                                      |
| 4   | `cpu_worker_sweep` | `cpuN` (device `cuda:0`)              | Lean path, 1 GPU, N parallel CPU preprocess workers feeding one GPU inference actor.                                                                               |
| 5   | `dual_gpu_serial`  | `2gpu_serial`                         | Lean path, 2 GPUs, single-threaded preprocess per shard (no CPU pool). Kept for the evolution comparison; roughly equivalent to #3 on half the stations per shard. |
| 6   | `dual_gpu`         | `2gpu_cpuN`                           | Lean path, 2 GPUs, each shard runs its own N-worker CPU preprocess pool (pipelined).                                                                               |
| 7   | `cpu_worker_sweep` | `cpu_infer_poolN[_tT]` (device `cpu`) | Lean path, CPU inference, N parallel CPU preprocess workers feeding one CPU inference actor pinned to `T` BLAS threads (or auto-split when `T` is absent).         |