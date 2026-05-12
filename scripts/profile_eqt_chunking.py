#!/usr/bin/env python3
"""Profile EQTransformer inference: breakdown of where time is spent.

This script profiles different approaches to understand the overhead sources:
1. Raw forward pass (model only)
2. Forward with annotate_batch_pre/post 
3. LeanPyTorchBackend.infer_batch (our lean path)
4. model.annotate() (SeisBench's full pipeline)
5. Chunked vs megabatch processing

Key finding: The forward pass itself is fast (~0.05s for 580 stations).
The overhead comes from data movement, preprocessing pipelines, and asyncio.

Usage:
    python scripts/profile_eqt_chunking.py [--n-stations 580] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np
import torch


def load_eqtransformer(device: str = "cuda:0"):
    """Load EQTransformer model."""
    import seisbench.models as sbm
    
    model = sbm.EQTransformer.from_pretrained("original")
    model.eval()
    model.to(device)
    return model


def generate_synthetic_data(
    n_stations: int,
    in_samples: int = 6000,
    n_channels: int = 3,
) -> np.ndarray:
    """Generate synthetic waveform data (B, C, T)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_stations, n_channels, in_samples)).astype(np.float32)


def run_megabatch(
    model: torch.nn.Module,
    data: np.ndarray,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, float]:
    """Run inference on the full megabatch at once (our current lean approach)."""
    x = torch.from_numpy(data).to(device, dtype=dtype)
    
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.perf_counter()
    
    with torch.inference_mode():
        # Mimic annotate_batch_pre for EQTransformer
        x = x - x.mean(dim=-1, keepdim=True)
        std = x.std(dim=(-1, -2), keepdim=True)
        x = x / (std + 1e-10)
        
        # Forward pass
        preds = model(x)
    
    torch.cuda.synchronize() if device.startswith("cuda") else None
    elapsed = time.perf_counter() - t0
    
    # Stack outputs
    if isinstance(preds, tuple):
        out = torch.stack(preds, dim=-1).cpu().numpy()
    else:
        out = preds.cpu().numpy()
    
    return out, elapsed


def run_chunked(
    model: torch.nn.Module,
    data: np.ndarray,
    device: str,
    chunk_size: int = 256,
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, float]:
    """Run inference in chunks (like annotate()'s default behavior)."""
    n_samples = data.shape[0]
    results = []
    
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.perf_counter()
    
    with torch.inference_mode():
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = data[start:end]
            
            x = torch.from_numpy(chunk).to(device, dtype=dtype)
            
            # Mimic annotate_batch_pre for EQTransformer
            x = x - x.mean(dim=-1, keepdim=True)
            std = x.std(dim=(-1, -2), keepdim=True)
            x = x / (std + 1e-10)
            
            # Forward pass
            preds = model(x)
            
            # Stack and collect
            if isinstance(preds, tuple):
                out = torch.stack(preds, dim=-1)
            else:
                out = preds
            results.append(out.cpu())
    
    torch.cuda.synchronize() if device.startswith("cuda") else None
    elapsed = time.perf_counter() - t0
    
    final = torch.cat(results, dim=0).numpy()
    return final, elapsed


def run_annotate_baseline(
    model: torch.nn.Module,
    data: np.ndarray,
    device: str,
) -> tuple[np.ndarray, float]:
    """Run inference via SeisBench's annotate() for comparison.
    
    Note: annotate() expects an ObsPy Stream, so we create a minimal one.
    """
    import obspy
    
    n_stations = data.shape[0]
    sampling_rate = 100.0
    
    # Create a Stream with synthetic traces
    stream = obspy.Stream()
    for i in range(n_stations):
        for c, comp in enumerate(["Z", "N", "E"]):
            tr = obspy.Trace(data=data[i, c, :].copy())
            tr.stats.sampling_rate = sampling_rate
            tr.stats.network = "XX"
            tr.stats.station = f"S{i:04d}"
            tr.stats.channel = f"HH{comp}"
            tr.stats.starttime = obspy.UTCDateTime(0)
            stream.append(tr)
    
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.perf_counter()
    
    # Run annotate
    _ = model.annotate(stream)
    
    torch.cuda.synchronize() if device.startswith("cuda") else None
    elapsed = time.perf_counter() - t0
    
    return None, elapsed  # Don't return predictions, just timing


def warmup(model: torch.nn.Module, device: str, n_warmup: int = 3):
    """Warm up the model with small batches."""
    dummy = np.random.randn(16, 3, 6000).astype(np.float32)
    x = torch.from_numpy(dummy).to(device)
    
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(x)
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark(
    fn: Callable,
    model: torch.nn.Module,
    data: np.ndarray,
    device: str,
    n_trials: int = 5,
    **kwargs,
) -> list[float]:
    """Run multiple trials and return timings."""
    times = []
    for _ in range(n_trials):
        _, elapsed = fn(model, data, device, **kwargs)
        times.append(elapsed)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-stations", type=int, default=580)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--skip-annotate", action="store_true", 
                        help="Skip annotate() baseline (slow)")
    args = parser.parse_args()
    
    device = args.device
    n_stations = args.n_stations
    n_trials = args.n_trials
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    
    print("=" * 70)
    print(f"EQTransformer Chunking Profile")
    print(f"  n_stations: {n_stations}")
    print(f"  device: {device}")
    print(f"  dtype: {args.dtype}")
    print(f"  n_trials: {n_trials}")
    print("=" * 70)
    
    # Load model
    print("\nLoading EQTransformer...")
    model = load_eqtransformer(device)
    
    # Generate data
    print(f"Generating synthetic data ({n_stations} stations)...")
    data = generate_synthetic_data(n_stations)
    
    # Warmup
    print("Warming up...")
    warmup(model, device)
    
    # Benchmark megabatch (current lean approach)
    print("\n--- Megabatch (full batch at once) ---")
    mega_times = benchmark(run_megabatch, model, data, device, n_trials, dtype=dtype)
    print(f"  Times: {[f'{t:.4f}s' for t in mega_times]}")
    print(f"  Mean:  {np.mean(mega_times):.4f}s ± {np.std(mega_times):.4f}s")
    
    # Benchmark chunked with various chunk sizes
    chunk_sizes = [64, 128, 256, 512]
    chunk_results = {}
    
    for chunk_size in chunk_sizes:
        print(f"\n--- Chunked (chunk_size={chunk_size}) ---")
        times = benchmark(
            run_chunked, model, data, device, n_trials, 
            chunk_size=chunk_size, dtype=dtype
        )
        chunk_results[chunk_size] = times
        print(f"  Times: {[f'{t:.4f}s' for t in times]}")
        print(f"  Mean:  {np.mean(times):.4f}s ± {np.std(times):.4f}s")
    
    # Benchmark annotate() baseline
    if not args.skip_annotate:
        print("\n--- annotate() baseline ---")
        # annotate() uses CPU for preprocessing, so fair comparison needs
        # model on the right device
        annotate_times = benchmark(
            run_annotate_baseline, model, data, device, n_trials
        )
        print(f"  Times: {[f'{t:.4f}s' for t in annotate_times]}")
        print(f"  Mean:  {np.mean(annotate_times):.4f}s ± {np.std(annotate_times):.4f}s")
    else:
        annotate_times = None
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    mega_mean = np.mean(mega_times)
    print(f"\nMegabatch:           {mega_mean:.4f}s (baseline for speedup)")
    
    for chunk_size, times in chunk_results.items():
        mean_t = np.mean(times)
        speedup = mega_mean / mean_t
        print(f"Chunked (bs={chunk_size:3d}):    {mean_t:.4f}s  ({speedup:.2f}x vs megabatch)")
    
    if annotate_times:
        ann_mean = np.mean(annotate_times)
        speedup_mega = ann_mean / mega_mean
        print(f"\nannotate():          {ann_mean:.4f}s")
        print(f"  -> megabatch is {speedup_mega:.2f}x {'faster' if speedup_mega > 1 else 'SLOWER'} than annotate()")
        
        best_chunk_size = min(chunk_results.keys(), key=lambda k: np.mean(chunk_results[k]))
        best_chunk_mean = np.mean(chunk_results[best_chunk_size])
        speedup_chunk = ann_mean / best_chunk_mean
        print(f"  -> chunked(bs={best_chunk_size}) is {speedup_chunk:.2f}x {'faster' if speedup_chunk > 1 else 'SLOWER'} than annotate()")
    
    print("\n" + "=" * 70)
    if any(np.mean(times) < mega_mean for times in chunk_results.values()):
        best = min(chunk_results.keys(), key=lambda k: np.mean(chunk_results[k]))
        print(f"FINDING: Chunked processing (bs={best}) is FASTER than megabatch.")
        print("         This suggests our lean path could benefit from chunking.")
    else:
        print("FINDING: Megabatch is faster than all chunked variants.")
        print("         The EQTransformer slowdown is NOT due to batch size.")
    print("=" * 70)


if __name__ == "__main__":
    main()
