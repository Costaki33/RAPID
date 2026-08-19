"""Station arrival schedules for orchestration benchmarks.

Playback: every station ready at t=0.
Staggered: a random subset is delayed; time is processed in 60 s waveform
chunks so stations that miss the current minute spill into the next one.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

# In-minute jitter plus a few values that land in the *next* 60 s chunk
# (60, 65, 90) so realtime spill is actually tested.
DELAY_CHOICES_S = (0, 5, 10, 15, 20, 25, 30, 60, 65, 90)
CHUNK_S = 60.0


def make_ready_times(
    stations: Sequence[str],
    *,
    mode: str,
    seed: int,
) -> Dict[str, float]:
    """Return ``{station: delay_s}`` relative to trial t=0.

    * ``playback`` — every station ready at 0.
    * ``staggered`` — draw how many stations are delayed (Uniform 0..N),
      pick that many at random, assign each a delay from ``DELAY_CHOICES_S``.
    """
    names = list(stations)
    n = len(names)
    if mode == "playback" or n == 0:
        return {s: 0.0 for s in names}
    if mode != "staggered":
        raise ValueError(f"mode must be playback|staggered, got {mode!r}")

    rng = random.Random(int(seed))
    n_delayed = rng.randint(0, n)
    delayed = rng.sample(names, n_delayed) if n_delayed else []
    ready = {s: 0.0 for s in names}
    for s in delayed:
        ready[s] = float(rng.choice(DELAY_CHOICES_S))
    return ready


def chunk_index(t_s: float, chunk_s: float = CHUNK_S) -> int:
    return int(max(0.0, float(t_s)) // float(chunk_s))


def chunk_end_s(t_s: float, chunk_s: float = CHUNK_S) -> float:
    return (chunk_index(t_s, chunk_s) + 1) * float(chunk_s)


def split_ontime_delayed(ready: Dict[str, float]) -> Tuple[List[str], List[str]]:
    ontime = [s for s, t in ready.items() if float(t) <= 0.0]
    delayed = [s for s, t in ready.items() if float(t) > 0.0]
    return ontime, delayed


def delay_summary(ready: Dict[str, float], chunk_s: float = CHUNK_S) -> Dict[str, float]:
    vals = list(ready.values())
    n = len(vals)
    n_pos = sum(1 for v in vals if v > 0)
    n_spill = sum(1 for v in vals if v >= chunk_s)
    return {
        "n_stations": n,
        "n_delayed_gt0": n_pos,
        "n_ready_at_0": n - n_pos,
        "n_spill_next_chunk": n_spill,
        "delay_min_s": float(min(vals) if vals else 0.0),
        "delay_max_s": float(max(vals) if vals else 0.0),
        "delay_mean_s": float(sum(vals) / n) if n else 0.0,
        "chunk_s": float(chunk_s),
        "n_chunks_touched": (1 + chunk_index(max(vals) if vals else 0.0, chunk_s)),
    }


def percentiles(xs: List[float], ps=(50, 90, 95, 99)) -> Dict[str, float]:
    if not xs:
        return {f"p{p}": float("nan") for p in ps}
    s = sorted(float(x) for x in xs)
    n = len(s)
    out: Dict[str, float] = {
        "n": float(n),
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
    }
    for p in ps:
        if n == 1:
            out[f"p{p}"] = s[0]
            continue
        k = (p / 100.0) * (n - 1)
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        out[f"p{p}"] = s[lo] * (1.0 - frac) + s[hi] * frac
    return out


def group_size(n_stations: int, k: int, packaging: str) -> int:
    if packaging == "s1":
        return 1
    return int(math.ceil(n_stations / max(1, k)))
