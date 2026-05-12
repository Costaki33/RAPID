#!/usr/bin/env python3
"""Quick test of process-based dual-GPU runner."""
import numpy as np
import torch.multiprocessing as mp

# Must be set before importing torch
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    from rapid.data import WindowSpec, build_megabatch
    from rapid.runners.dual_gpu_process import run_dual_gpu_process

    # Generate synthetic data
    N_STATIONS = 64
    IN_SAMPLES = 3001  # PhaseNet
    rng = np.random.default_rng(42)

    arrays = [
        (f"STA{i:04d}", rng.standard_normal((3, IN_SAMPLES + 1000)).astype(np.float32))
        for i in range(N_STATIONS)
    ]

    spec = WindowSpec(in_samples=IN_SAMPLES, overlap_samples=0)
    megabatch = build_megabatch(arrays, spec)

    print(f"Testing with {megabatch.total_windows} windows from {N_STATIONS} stations")

    # Test PhaseNet BF16 without compile
    print("\n1. PhaseNet BF16 (no compile)...")
    result = run_dual_gpu_process(
        parent_model="PhaseNet",
        child_model="stead",
        dtype="bf16",
        megabatch=megabatch,
        batch_size=256,
        compile_model=False,
    )
    print(f"   Wall time: {result.wall_time_s:.3f}s")
    print(f"   GPU0 time: {result.gpu0_time_s:.3f}s")
    print(f"   GPU1 time: {result.gpu1_time_s:.3f}s")

    # Test PhaseNet BF16 with compile
    print("\n2. PhaseNet BF16 (with compile)...")
    result = run_dual_gpu_process(
        parent_model="PhaseNet",
        child_model="stead",
        dtype="bf16",
        megabatch=megabatch,
        batch_size=256,
        compile_model=True,
    )
    print(f"   Wall time: {result.wall_time_s:.3f}s")
    print(f"   GPU0 time: {result.gpu0_time_s:.3f}s")
    print(f"   GPU1 time: {result.gpu1_time_s:.3f}s")

    # Test EQTransformer BF16
    IN_SAMPLES_EQT = 6000
    arrays_eqt = [
        (f"STA{i:04d}", rng.standard_normal((3, IN_SAMPLES_EQT + 1000)).astype(np.float32))
        for i in range(N_STATIONS)
    ]
    spec_eqt = WindowSpec(in_samples=IN_SAMPLES_EQT, overlap_samples=0)
    megabatch_eqt = build_megabatch(arrays_eqt, spec_eqt)

    print(f"\n3. EQTransformer BF16 (no compile) - {megabatch_eqt.total_windows} windows...")
    result = run_dual_gpu_process(
        parent_model="EQTransformer",
        child_model="original",
        dtype="bf16",
        megabatch=megabatch_eqt,
        batch_size=256,
        compile_model=False,
    )
    print(f"   Wall time: {result.wall_time_s:.3f}s")
    print(f"   GPU0 time: {result.gpu0_time_s:.3f}s")
    print(f"   GPU1 time: {result.gpu1_time_s:.3f}s")

    print("\nAll tests passed!")
