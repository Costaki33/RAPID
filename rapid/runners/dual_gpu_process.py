"""Process-based dual-GPU runner.

Unlike the threaded dual-GPU runner, this uses separate processes for each GPU,
avoiding CUDA context corruption and enabling torch.compile with CUDA graphs.

Each worker process gets its own Python interpreter and CUDA context, so:
1. No GIL contention during Python execution
2. No CUDA graph state corruption between devices
3. torch.compile works safely with mode="reduce-overhead"

The tradeoff is higher startup cost (process spawn + model load per GPU).

Uses shared memory for efficient data transfer between processes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from ..backends.lean_pytorch import LeanPyTorchBackend
from ..data import Megabatch, WindowSpec, build_megabatch


@dataclass
class DualGPUProcessResult:
    backend_name: str
    model: str
    dtype: str
    n_stations: int
    n_windows: int
    batch_size: int
    wall_time_s: float
    gpu0_time_s: float
    gpu1_time_s: float
    predictions: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _worker_fn(
    rank: int,
    device: str,
    parent_model: str,
    child_model: str,
    dtype: str,
    compile_model: bool,
    shm_name: str,
    shm_shape: Tuple[int, ...],
    shm_dtype: str,
    start_idx: int,
    end_idx: int,
    batch_size: int,
    result_queue: mp.Queue,
    warmup_iters: int = 2,
):
    """Worker function that runs in a separate process.
    
    Uses shared memory to access the data without pickling overhead.
    """
    try:
        # Set CUDA device for this process
        torch.cuda.set_device(device)
        
        # Access shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        full_windows = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
        windows = full_windows[start_idx:end_idx]
        
        # Load backend
        backend = LeanPyTorchBackend(
            parent_model=parent_model,
            child_model=child_model,
            device=device,
            dtype=dtype,
            compile=compile_model,
            compile_mode="reduce-overhead" if compile_model else "default",
        )
        backend.load()
        
        # Warmup
        if warmup_iters > 0 and len(windows) > 0:
            dummy = np.zeros(
                (min(batch_size, len(windows)), windows.shape[1], windows.shape[2]),
                dtype=np.float32,
            )
            for _ in range(warmup_iters):
                backend.infer_batch(dummy)
            torch.cuda.synchronize()
        
        # Time the inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        preds = backend.infer_chunked(windows, batch_size=batch_size)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        # Put results in queue (don't pass predictions back, too large)
        result_queue.put({
            "rank": rank,
            "device": device,
            "elapsed": elapsed,
            "n_windows": len(windows),
            "preds_shape": preds.shape if preds is not None else None,
            "error": None,
        })
        
        backend.close()
        shm.close()
        
    except Exception as e:
        import traceback
        result_queue.put({
            "rank": rank,
            "device": device,
            "elapsed": None,
            "n_windows": 0,
            "preds_shape": None,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        })


def run_dual_gpu_process(
    parent_model: str,
    child_model: str,
    dtype: str,
    megabatch: Megabatch,
    batch_size: int,
    compile_model: bool = False,
    devices: Tuple[str, str] = ("cuda:0", "cuda:1"),
    warmup_iters: int = 2,
) -> DualGPUProcessResult:
    """Run inference on two GPUs using separate processes.
    
    Parameters
    ----------
    parent_model : str
        Model family (e.g., "PhaseNet", "EQTransformer")
    child_model : str
        Pretrained weights name (e.g., "original", "stead")
    dtype : str
        Data type: "fp32", "fp16", or "bf16"
    megabatch : Megabatch
        Pre-built megabatch of windows
    batch_size : int
        Sub-batch size for infer_chunked
    compile_model : bool
        Whether to use torch.compile (safe in process-based runner)
    devices : tuple
        GPU device strings
    warmup_iters : int
        Number of warmup iterations per worker
        
    Returns
    -------
    DualGPUProcessResult
        Timing and prediction results
    """
    windows = megabatch.windows
    n_total = len(windows)
    
    if n_total == 0:
        return DualGPUProcessResult(
            backend_name="lean_pytorch_dual_process",
            model=f"{parent_model}/{child_model}",
            dtype=dtype,
            n_stations=0,
            n_windows=0,
            batch_size=batch_size,
            wall_time_s=0.0,
            gpu0_time_s=0.0,
            gpu1_time_s=0.0,
        )
    
    # Create shared memory for windows array
    shm = shared_memory.SharedMemory(create=True, size=windows.nbytes)
    shm_array = np.ndarray(windows.shape, dtype=windows.dtype, buffer=shm.buf)
    np.copyto(shm_array, windows)
    
    # Split indices evenly between GPUs
    mid = n_total // 2
    
    try:
        # Use spawn method for clean CUDA contexts
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        
        # Create worker processes
        p0 = ctx.Process(
            target=_worker_fn,
            args=(
                0, devices[0], parent_model, child_model, dtype, compile_model,
                shm.name, windows.shape, str(windows.dtype),
                0, mid, batch_size, result_queue, warmup_iters
            ),
        )
        p1 = ctx.Process(
            target=_worker_fn,
            args=(
                1, devices[1], parent_model, child_model, dtype, compile_model,
                shm.name, windows.shape, str(windows.dtype),
                mid, n_total, batch_size, result_queue, warmup_iters
            ),
        )
        
        # Start timing (includes process startup)
        wall_t0 = time.perf_counter()
        
        p0.start()
        p1.start()
        
        # Wait for both to complete
        p0.join()
        p1.join()
        
        wall_elapsed = time.perf_counter() - wall_t0
        
        # Collect results
        results = {}
        for _ in range(2):
            r = result_queue.get()
            results[r["rank"]] = r
        
        # Check for errors
        for rank, r in results.items():
            if r["error"] is not None:
                raise RuntimeError(f"Worker {rank} failed: {r['error']}")
        
    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()
    
    return DualGPUProcessResult(
        backend_name="lean_pytorch_dual_process",
        model=f"{parent_model}/{child_model}",
        dtype=dtype,
        n_stations=megabatch.n_stations if hasattr(megabatch, 'n_stations') else -1,
        n_windows=n_total,
        batch_size=batch_size,
        wall_time_s=wall_elapsed,
        gpu0_time_s=results[0]["elapsed"],
        gpu1_time_s=results[1]["elapsed"],
        predictions=None,  # Not returning predictions to avoid memory issues
        extra={
            "compile": compile_model,
            "devices": devices,
            "shard0_windows": mid,
            "shard1_windows": n_total - mid,
        },
    )


def run_dual_gpu_process_from_arrays(
    parent_model: str,
    child_model: str,
    dtype: str,
    arrays: List[Tuple[str, np.ndarray]],
    in_samples: int,
    batch_size: int,
    overlap_samples: int = 0,
    compile_model: bool = False,
    devices: Tuple[str, str] = ("cuda:0", "cuda:1"),
    warmup_iters: int = 2,
) -> DualGPUProcessResult:
    """Convenience wrapper that builds megabatch from arrays.
    
    Parameters
    ----------
    arrays : list of (station_id, ndarray) tuples
        Each array has shape (C, T) for C channels and T samples
    in_samples : int
        Window size (model's expected input length)
    overlap_samples : int
        Overlap between consecutive windows
    """
    spec = WindowSpec(in_samples=in_samples, overlap_samples=overlap_samples)
    megabatch = build_megabatch(arrays, spec)
    
    return run_dual_gpu_process(
        parent_model=parent_model,
        child_model=child_model,
        dtype=dtype,
        megabatch=megabatch,
        batch_size=batch_size,
        compile_model=compile_model,
        devices=devices,
        warmup_iters=warmup_iters,
    )
