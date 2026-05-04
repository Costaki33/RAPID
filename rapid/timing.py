"""Stage-level timers with optional CUDA synchronization.

Every meaningful wall-time chunk in a RAPID run is captured as a named stage.
On GPU backends we synchronize around timed regions so forward-pass time is not
absorbed by the asynchronous CUDA queue.

Usage::

    t = Timer(device="cuda")
    with t.stage("preprocess"):
        run_preprocess(...)
    with t.stage("forward"):
        run_forward(...)
    stats = t.report()   # {"preprocess": 0.12, "forward": 0.08, ...}
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _maybe_cuda_sync(device: Optional[str]) -> None:
    if not device or "cuda" not in str(device):
        return
    try:
        import torch  # local import: timing is usable even without torch.

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class Timer:
    """Stage-accumulating timer.

    ``device`` controls whether a CUDA synchronization is performed around the
    timed region. Pass ``"cuda"`` (or any string containing ``cuda``) to force
    synchronization; any other value is a no-op sync.
    """

    device: Optional[str] = None
    _samples: Dict[str, List[float]] = field(default_factory=dict)

    @contextlib.contextmanager
    def stage(self, name: str):
        _maybe_cuda_sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _maybe_cuda_sync(self.device)
            dt = time.perf_counter() - t0
            self._samples.setdefault(name, []).append(dt)

    def record(self, name: str, seconds: float) -> None:
        self._samples.setdefault(name, []).append(float(seconds))

    def report(self) -> Dict[str, float]:
        """Sum of all recorded samples per stage."""
        return {k: float(sum(v)) for k, v in self._samples.items()}

    def report_mean(self) -> Dict[str, float]:
        return {k: float(statistics.mean(v)) for k, v in self._samples.items()}

    def report_stats(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, v in self._samples.items():
            out[k] = {
                "sum": float(sum(v)),
                "mean": float(statistics.mean(v)) if v else 0.0,
                "median": float(statistics.median(v)) if v else 0.0,
                "min": float(min(v)) if v else 0.0,
                "max": float(max(v)) if v else 0.0,
                "n": len(v),
            }
        return out

    def reset(self) -> None:
        self._samples.clear()
