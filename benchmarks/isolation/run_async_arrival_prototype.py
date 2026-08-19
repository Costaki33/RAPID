#!/usr/bin/env python3
"""Async station-arrival dispatcher prototype (Camilo comment 6).

Compares two policies on a synthetic 580-station ready-time schedule:
  1) playback: all stations ready at t=0; fire full groups immediately
  2) async:    each station gets a jittered ready time; an actor fires when
               its group reaches ``group_size`` ready stations OR ``max_wait_s``
               elapses after the first ready station in that group

Compute work is simulated with a configurable per-station cost so the
experiment isolates dispatcher behavior without loading models.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Event:
    ready_t: float
    station: int


def simulate(
    *,
    n_stations: int,
    n_actors: int,
    group_size: int,
    max_wait_s: float,
    per_station_s: float,
    jitter_s: float,
    seed: int,
    playback: bool,
) -> dict:
    rng = random.Random(seed)
    ready = [0.0 if playback else rng.uniform(0.0, jitter_s) for _ in range(n_stations)]
    # Round-robin station -> actor assignment (fixed partition)
    buckets: List[List[Event]] = [[] for _ in range(n_actors)]
    for sta, t in enumerate(ready):
        buckets[sta % n_actors].append(Event(ready_t=t, station=sta))
    for b in buckets:
        b.sort(key=lambda e: e.ready_t)

    fire_times: List[float] = []
    finish_times: List[float] = []
    idle_gaps: List[float] = []

    for bucket in buckets:
        if not bucket:
            continue
        i = 0
        actor_free = 0.0
        while i < len(bucket):
            first = bucket[i]
            # gather until group_size or timeout after first ready
            j = i
            deadline = first.ready_t + max_wait_s
            while j < len(bucket) and (j - i) < group_size:
                if bucket[j].ready_t > deadline and (j - i) > 0:
                    break
                # wait until this station is ready if still within deadline
                if bucket[j].ready_t > deadline:
                    break
                j += 1
                # if we have not filled group and next would exceed deadline, stop after at least 1
                if j < len(bucket) and (j - i) >= 1:
                    if bucket[j].ready_t > deadline and (j - i) >= 1:
                        break
            # ensure at least one station
            if j == i:
                j = i + 1
            group = bucket[i:j]
            ready_group = max(e.ready_t for e in group)
            start = max(actor_free, ready_group)
            if start > actor_free and actor_free > 0:
                idle_gaps.append(start - actor_free)
            cost = per_station_s * len(group)
            end = start + cost
            fire_times.append(start)
            finish_times.append(end)
            actor_free = end
            i = j

    makespan = max(finish_times) if finish_times else 0.0
    return {
        "playback": playback,
        "n_stations": n_stations,
        "n_actors": n_actors,
        "group_size": group_size,
        "max_wait_s": max_wait_s,
        "jitter_s": jitter_s,
        "makespan_s": makespan,
        "n_firings": len(fire_times),
        "mean_idle_gap_s": statistics.mean(idle_gaps) if idle_gaps else 0.0,
        "max_idle_gap_s": max(idle_gaps) if idle_gaps else 0.0,
        "last_ready_s": max(ready) if ready else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-stations", type=int, default=580)
    ap.add_argument("--n-actors", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=29, help="stations per firing (~580/20)")
    ap.add_argument("--max-wait-s", type=float, default=2.0)
    ap.add_argument("--per-station-s", type=float, default=0.02)
    ap.add_argument("--jitter-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    playback = simulate(
        n_stations=args.n_stations, n_actors=args.n_actors, group_size=args.group_size,
        max_wait_s=args.max_wait_s, per_station_s=args.per_station_s, jitter_s=args.jitter_s,
        seed=args.seed, playback=True,
    )
    async_ = simulate(
        n_stations=args.n_stations, n_actors=args.n_actors, group_size=args.group_size,
        max_wait_s=args.max_wait_s, per_station_s=args.per_station_s, jitter_s=args.jitter_s,
        seed=args.seed, playback=False,
    )
    out = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "playback": playback,
        "async_arrival": async_,
        "delta_makespan_s": async_["makespan_s"] - playback["makespan_s"],
        "note": (
            "Playback assumes all stations ready at t=0 (paper warm protocol). "
            "Async assigns uniform ready jitter in [0, jitter_s] and fires an actor "
            "group when group_size is reached or max_wait_s elapses after the first "
            "ready station in that group."
        ),
    }
    text = json.dumps(out, indent=2)
    print(text)
    out_path = args.out or (
        Path(__file__).resolve().parents[2]
        / "results" / "iso_full_benchmark" / "async_arrival_prototype.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
