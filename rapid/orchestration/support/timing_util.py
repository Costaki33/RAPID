"""
Shared high-resolution timing for EQCCTPro trials and benchmarks.

Use :func:`monotonic_s` for all duration measurements (wall elapsed time) so results
are not affected by system clock adjustments (NTP). This matches common practice for
benchmarking and aligns Ripper, Model-Actor, and driver-side serial baselines.

CUDA: PyTorch kernels are often asynchronous. :func:`cuda_synchronize_best_effort`
blocks until the default CUDA device finishes queued work so stopwatches reflect
completed GPU inference/transfer when timing GPU SeisBench paths.
"""
from __future__ import annotations

import time


def monotonic_s() -> float:
    """Monotonic, high-resolution seconds (``time.perf_counter``)."""
    return time.perf_counter()


def cuda_synchronize_best_effort() -> None:
    """``torch.cuda.synchronize()`` when PyTorch CUDA is available; no-op otherwise."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class SeisBenchStageProbes:
    """Measure preprocess / pick-aggregation time INSIDE SeisBench end-to-end calls.

    ``model.annotate()`` / ``model.classify()`` are monolithic, so callers can
    only time the whole call. This installs instance-level wrappers around the
    documented SeisBench overridable subfunctions:

    * ``annotate_stream_pre`` + ``annotate_batch_pre`` -> ``preprocess_s``
      (stream resample/filter + per-batch normalisation)
    * ``classify_aggregate``                          -> ``pick_aggregate_s``
      (probability traces -> picks, classify() only)

    The accumulated busy seconds let callers report MEASURED preprocess and
    pick-generation components: ``inference = call_wall - preprocess_s -
    pick_aggregate_s``. SeisBench's annotate pipeline is cooperative asyncio in
    one thread, so the accumulators are simple sums of non-overlapping segments.
    Install once per loaded model instance; call :meth:`reset` per measurement.
    """

    _HOOKS = ("annotate_stream_pre", "annotate_batch_pre", "classify_aggregate")

    def __init__(self, model):
        self.preprocess_s = 0.0
        self.pick_aggregate_s = 0.0
        self._install(model)

    def _install(self, model) -> None:
        probes = self

        def _wrap(name: str, bucket: str):
            orig = getattr(model, name)

            def timed(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return orig(*args, **kwargs)
                finally:
                    setattr(probes, bucket, getattr(probes, bucket) + (time.perf_counter() - t0))

            setattr(model, name, timed)

        for hook in ("annotate_stream_pre", "annotate_batch_pre"):
            if hasattr(model, hook):
                _wrap(hook, "preprocess_s")
        if hasattr(model, "classify_aggregate"):
            _wrap("classify_aggregate", "pick_aggregate_s")

    def reset(self) -> None:
        self.preprocess_s = 0.0
        self.pick_aggregate_s = 0.0
