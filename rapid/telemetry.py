"""GPU telemetry: NVML-backed background poller for utilization, power,
temperature, and memory use.

Motivation
----------
Our software-derived ``gpu_utilization_pct = forward_s / wall_s`` answers
"how much of wall-time did the inference actor spend in forward()?". It
does *not* answer "how busy was the hardware?" — a poorly-batched kernel
launch can leave the SMs idle for 90 % of the forward pass while still
counting as 100 % software utilization. NVML's ``utilization.gpu`` is
the hardware counter and gives the truth.

Energy (joules) is computed as the time-integral of power draw, so even
a 200 ms sampling cadence is accurate to within a few percent for runs
longer than a second.

The poller is a lightweight background thread and uses only NVML
(``nvidia-ml-py`` / the legacy ``pynvml``); no subprocess spawns. If
NVML isn't available (CPU-only box, or pynvml not installed), the
poller silently returns empty stats so the rest of the pipeline keeps
working.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_NVML_AVAILABLE: Optional[bool] = None  # lazy one-shot probe


def _nvml():
    """Import NVML once and cache the module (or None).

    ``nvidia-ml-py`` exposes the same ``pynvml`` namespace as the older
    package, so a single import works for both installs.
    """
    global _NVML_AVAILABLE
    if _NVML_AVAILABLE is False:
        return None
    # ``nvidia-ml-py`` ships the ``pynvml`` module too, but pynvml the
    # standalone package triggers a FutureWarning. Silence it locally —
    # we can't always control which is installed on the target box.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            import pynvml  # type: ignore
        except ImportError:
            _NVML_AVAILABLE = False
            return None
    try:
        pynvml.nvmlInit()
        _NVML_AVAILABLE = True
        return pynvml
    except Exception:
        _NVML_AVAILABLE = False
        return None


@dataclass
class GPUTelemetry:
    """Per-run GPU telemetry summary.

    One dict-keyed-by-device-index entry per metric. Values are per-GPU
    so a 2-GPU run yields, e.g., ``{0: 92.1, 1: 88.5}`` for mean util.
    """

    n_samples: int = 0
    sample_interval_s: float = 0.0
    elapsed_s: float = 0.0
    mean_util_pct: Dict[int, float] = field(default_factory=dict)
    peak_util_pct: Dict[int, float] = field(default_factory=dict)
    mean_power_w: Dict[int, float] = field(default_factory=dict)
    peak_power_w: Dict[int, float] = field(default_factory=dict)
    energy_j: Dict[int, float] = field(default_factory=dict)
    mean_mem_used_bytes: Dict[int, float] = field(default_factory=dict)
    peak_mem_used_bytes: Dict[int, int] = field(default_factory=dict)
    peak_temp_c: Dict[int, float] = field(default_factory=dict)

    def as_row_fields(self) -> Dict[str, object]:
        """Flatten to JSON-row-friendly fields (keyed by device index)."""
        if not self.n_samples:
            return {}

        def _flatten(d: Dict[int, float]) -> Dict[str, float]:
            return {f"cuda:{k}": float(v) for k, v in d.items()}

        return {
            "nvml_samples": self.n_samples,
            "nvml_sample_interval_s": self.sample_interval_s,
            "nvml_elapsed_s": self.elapsed_s,
            "nvml_mean_util_pct": _flatten(self.mean_util_pct),
            "nvml_peak_util_pct": _flatten(self.peak_util_pct),
            "nvml_mean_power_w": _flatten(self.mean_power_w),
            "nvml_peak_power_w": _flatten(self.peak_power_w),
            "nvml_energy_j": _flatten(self.energy_j),
            "nvml_mean_mem_used_bytes": _flatten(self.mean_mem_used_bytes),
            "nvml_peak_mem_used_bytes": {
                f"cuda:{k}": int(v) for k, v in self.peak_mem_used_bytes.items()
            },
            "nvml_peak_temp_c": _flatten(self.peak_temp_c),
        }


class GPUWatcher:
    """Background-thread NVML sampler.

    The thread samples all devices in ``device_indices`` every
    ``interval_s`` and accumulates sums/peaks. ``stop()`` returns a
    :class:`GPUTelemetry` summary. Energy is a trapezoidal integral of
    the power samples.

    Safe to construct on CPU-only boxes (the thread simply never records
    anything). Safe to call ``stop()`` without a prior ``start()``.
    """

    def __init__(
        self,
        device_indices: Optional[List[int]] = None,
        interval_s: float = 0.2,
    ):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handles: Dict[int, object] = {}
        self._device_indices = device_indices
        # Running aggregates
        self._t_start = 0.0
        self._t_end = 0.0
        self._n = 0
        self._util_sum: Dict[int, float] = {}
        self._util_peak: Dict[int, float] = {}
        self._power_sum: Dict[int, float] = {}
        self._power_peak: Dict[int, float] = {}
        self._power_last: Dict[int, float] = {}
        self._power_last_t: float = 0.0
        self._energy: Dict[int, float] = {}
        self._mem_sum: Dict[int, float] = {}
        self._mem_peak: Dict[int, int] = {}
        self._temp_peak: Dict[int, float] = {}

    def start(self) -> None:
        nvml = _nvml()
        if nvml is None:
            return
        try:
            n_dev = nvml.nvmlDeviceGetCount()
        except Exception:
            return
        if self._device_indices is None:
            indices = list(range(n_dev))
        else:
            indices = [i for i in self._device_indices if 0 <= i < n_dev]
        for i in indices:
            try:
                self._handles[i] = nvml.nvmlDeviceGetHandleByIndex(i)
                self._util_sum[i] = 0.0
                self._util_peak[i] = 0.0
                self._power_sum[i] = 0.0
                self._power_peak[i] = 0.0
                self._energy[i] = 0.0
                self._mem_sum[i] = 0.0
                self._mem_peak[i] = 0
                self._temp_peak[i] = 0.0
            except Exception:
                continue
        if not self._handles:
            return

        self._t_start = time.perf_counter()
        self._power_last_t = self._t_start
        self._stop.clear()

        def _loop():
            # Local refs for tight loop.
            while not self._stop.is_set():
                now = time.perf_counter()
                dt = now - self._power_last_t
                self._power_last_t = now
                sampled_any = False
                for idx, h in self._handles.items():
                    try:
                        util = nvml.nvmlDeviceGetUtilizationRates(h).gpu  # %
                    except Exception:
                        util = 0.0
                    try:
                        # ``nvmlDeviceGetPowerUsage`` returns milliwatts.
                        power_w = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    except Exception:
                        power_w = 0.0
                    try:
                        mem = nvml.nvmlDeviceGetMemoryInfo(h).used
                    except Exception:
                        mem = 0
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(
                            h, nvml.NVML_TEMPERATURE_GPU
                        )
                    except Exception:
                        temp = 0.0

                    self._util_sum[idx] += float(util)
                    if util > self._util_peak[idx]:
                        self._util_peak[idx] = float(util)
                    self._power_sum[idx] += float(power_w)
                    if power_w > self._power_peak[idx]:
                        self._power_peak[idx] = float(power_w)
                    # Trapezoid integral between the previous and current
                    # sample. Using the midpoint is equivalent for a
                    # constant-rate sampler and simpler.
                    last = self._power_last.get(idx, float(power_w))
                    self._energy[idx] += 0.5 * (last + float(power_w)) * dt
                    self._power_last[idx] = float(power_w)
                    self._mem_sum[idx] += float(mem)
                    if mem > self._mem_peak[idx]:
                        self._mem_peak[idx] = int(mem)
                    if temp > self._temp_peak[idx]:
                        self._temp_peak[idx] = float(temp)
                    sampled_any = True
                if sampled_any:
                    self._n += 1
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(
            target=_loop, daemon=True, name="rapid-gpu-watcher",
        )
        self._thread.start()

    def stop(self) -> GPUTelemetry:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        self._t_end = time.perf_counter()
        n = max(1, self._n)
        tel = GPUTelemetry(
            n_samples=self._n,
            sample_interval_s=self.interval_s,
            elapsed_s=max(0.0, self._t_end - self._t_start),
        )
        for idx in self._handles:
            tel.mean_util_pct[idx] = self._util_sum[idx] / n
            tel.peak_util_pct[idx] = self._util_peak[idx]
            tel.mean_power_w[idx] = self._power_sum[idx] / n
            tel.peak_power_w[idx] = self._power_peak[idx]
            tel.energy_j[idx] = self._energy[idx]
            tel.mean_mem_used_bytes[idx] = self._mem_sum[idx] / n
            tel.peak_mem_used_bytes[idx] = self._mem_peak[idx]
            tel.peak_temp_c[idx] = self._temp_peak[idx]
        return tel
