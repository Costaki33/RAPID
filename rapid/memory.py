"""Memory tracking helpers for benchmark trials.

Two primitives:

- :class:`RSSPoller` — background thread samples the calling process's
  resident-set size every ``interval_s`` and records the peak. Cheap
  enough (tens of microseconds per sample) to run continuously for the
  full trial without affecting timing.
- :func:`gpu_mem_reset` / :func:`gpu_mem_peak` — thin wrappers around
  ``torch.cuda.reset_peak_memory_stats`` / ``max_memory_allocated`` so
  Ray actors can report their own GPU peak without routing through the
  driver's CUDA context (which never sees memory allocated in actor
  worker processes).

Callers use the :func:`track_memory` context manager to wrap any block
of work — it handles both CPU RSS and (optionally) per-device GPU peak,
and returns a :class:`MemStats` summary on exit.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

try:
    import psutil  # type: ignore
    _HAVE_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAVE_PSUTIL = False


@dataclass
class MemStats:
    """Summary of memory usage over one tracked block.

    Fields that couldn't be measured are left at ``0``/``None`` so the
    JSON row stays compact — a missing ``psutil`` gives all zero RSS,
    for example, without raising.
    """

    peak_rss_bytes: int = 0
    start_rss_bytes: int = 0
    delta_rss_bytes: int = 0
    peak_gpu_mem_bytes: Dict[str, int] = field(default_factory=dict)


class RSSPoller:
    """Background-thread RSS poller.

    Sampling is lock-free: the poller thread writes ``_peak`` and the
    caller reads it only after joining the thread, so no synchronisation
    is needed.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = float(interval_s)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._peak = 0
        self._start = 0
        self._proc = None

    def start(self) -> None:
        if not _HAVE_PSUTIL:
            return
        self._proc = psutil.Process(os.getpid())
        try:
            self._start = int(self._proc.memory_info().rss)
        except Exception:
            self._start = 0
        self._peak = self._start
        self._stop.clear()

        def _loop(proc, stop_evt, interval):
            # Local refs keep the poll tight.
            peak = self._peak
            while not stop_evt.is_set():
                try:
                    rss = int(proc.memory_info().rss)
                    if rss > peak:
                        peak = rss
                        self._peak = peak
                except Exception:
                    pass
                stop_evt.wait(interval)

        self._thread = threading.Thread(
            target=_loop,
            args=(self._proc, self._stop, self.interval_s),
            daemon=True,
            name="rapid-rss-poller",
        )
        self._thread.start()

    def stop(self) -> MemStats:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        # One last sample in case the peak landed between the final tick
        # and our stop signal.
        if self._proc is not None:
            try:
                rss = int(self._proc.memory_info().rss)
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
        return MemStats(
            peak_rss_bytes=int(self._peak),
            start_rss_bytes=int(self._start),
            delta_rss_bytes=int(self._peak - self._start),
        )


def gpu_mem_reset(device: Optional[str] = None) -> None:
    """Reset the CUDA peak-memory counter. No-op on CPU or without CUDA."""
    if device is None or not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)
    except Exception:
        pass


def gpu_mem_peak(device: Optional[str] = None) -> Optional[int]:
    """Return ``torch.cuda.max_memory_allocated(device)`` as int, or None."""
    if device is None or not str(device).startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated(device=device))
    except Exception:
        return None


def gpu_mem_reset_all() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device=i)
    except Exception:
        pass


def gpu_mem_peak_all() -> Dict[str, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            f"cuda:{i}": int(torch.cuda.max_memory_allocated(device=i))
            for i in range(torch.cuda.device_count())
        }
    except Exception:
        return {}


@contextmanager
def track_memory(
    *,
    gpu_devices: Optional[Iterable[str]] = None,
    rss_interval_s: float = 0.05,
):
    """Context manager: RSS poller + GPU peak reset/read for listed devices.

    Usage::

        with track_memory(gpu_devices=["cuda:0"]) as mem:
            do_work()
        # mem.stats is a MemStats with peak_rss_bytes and peak_gpu_mem_bytes

    On exit, ``mem.stats`` is populated. If a sub-process (Ray actor)
    held the tensors, driver-side CUDA peak will be near zero — use the
    actor's own ``gpu_mem_peak()`` report in that case.
    """
    poller = RSSPoller(interval_s=rss_interval_s)
    poller.start()

    gpu_devices = list(gpu_devices or [])
    for d in gpu_devices:
        gpu_mem_reset(d)

    class _MemHandle:
        stats: MemStats = MemStats()

    handle = _MemHandle()
    try:
        yield handle
    finally:
        stats = poller.stop()
        peaks: Dict[str, int] = {}
        for d in gpu_devices:
            p = gpu_mem_peak(d)
            if p is not None:
                peaks[d] = p
        stats.peak_gpu_mem_bytes = peaks
        handle.stats = stats
