"""Ready-queue / static dispatch for annotate-precision orchestration trials.

Pools
-----
A trial has one or two worker pools:

* homogeneous: one pool (Model-Actor *or* Ripper) that takes every station.
* hybrid: an **ontime** pool owns ``ready_t == 0`` stations; a **delayed**
  pool owns ``ready_t > 0``. Polarities are MA-on-time + Ripper-delayed, or
  the reverse.

Playback uses static round-robin on a single pool.
Staggered uses a ready-queue (work-stealing) inside each pool, in 60 s
waveform chunks. Fill policy:

* eager (max_wait_s=0): fire as soon as any station is ready (partial SG OK).
* w5 / w10: wait up to 5 or 10 s to fill G, but always flush at chunk end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rapid.benchmark.arrival import CHUNK_S, chunk_end_s, chunk_index, group_size


@dataclass
class WorkerPool:
    name: str  # "all" | "ontime" | "delayed"
    kind: str  # "modelactor" | "ripper"
    k: int
    actors: list = field(default_factory=list)
    gpu_frac: float = 0.0
    g: int = 1


def round_robin_buckets(stations: Sequence[str], k: int) -> List[List[str]]:
    buckets: List[List[str]] = [[] for _ in range(max(1, k))]
    for i, s in enumerate(stations):
        buckets[i % len(buckets)].append(s)
    return [b for b in buckets if b]


def pool_for_station(
    sta: str,
    ready: Dict[str, float],
    pools: Dict[str, WorkerPool],
) -> str:
    if "all" in pools:
        return "all"
    if float(ready.get(sta, 0.0)) <= 0.0:
        return "ontime" if "ontime" in pools else next(iter(pools))
    return "delayed" if "delayed" in pools else next(iter(pools))


def dispatch(
    *,
    pools: Dict[str, WorkerPool],
    packaging: str,
    arrival: str,
    fill: str,
    streams: List[Tuple[str, Any]],
    ready: Dict[str, float],
    cls_kw: Dict[str, Any],
    gpu: bool,
    submit_fn: Callable[..., Any],
    wait_fn: Callable[..., Tuple[list, list]],
    get_fn: Callable[[Any], Any],
    merge_group_fn: Callable,
    picks_fn: Callable,
    chunk_s: float = CHUNK_S,
    max_wait_s: float = 0.0,
) -> Dict[str, Any]:
    """Run one arrival scenario. ``submit_fn`` / ``wait_fn`` / ``get_fn`` wrap Ray."""
    by_sta = {sta: stq for sta, stq in streams}
    stations = [sta for sta, _ in streams]
    t_origin = time.monotonic()
    # Idle gaps (late-station delays, w5/w10 fill, chunk flush) are simulated.
    # Sleeping wall-clock 90 s × 5 repeats was dominating staggered trials.
    sim_ahead = 0.0

    def now() -> float:
        return time.monotonic() - t_origin + sim_ahead

    def skip_to(t_target: float) -> None:
        nonlocal sim_ahead
        gap = float(t_target) - now()
        if gap > 0:
            sim_ahead += gap

    tasks: List[Dict[str, Any]] = []
    in_flight: Dict[Any, Dict[str, Any]] = {}
    task_id = 0
    picks: Dict[str, Dict[str, List[float]]] = {sta: {"p": [], "s": []} for sta in stations}

    # Per-pool free MA actor ids (ripper uses in-flight count vs k).
    free: Dict[str, List[int]] = {
        name: list(range(len(p.actors))) if p.kind == "modelactor" else []
        for name, p in pools.items()
    }
    in_flight_by_pool: Dict[str, int] = {name: 0 for name in pools}

    def can_take(name: str) -> bool:
        p = pools[name]
        if p.kind == "modelactor":
            return bool(free[name])
        return in_flight_by_pool[name] < p.k

    def take_worker(name: str) -> Any:
        p = pools[name]
        if p.kind == "modelactor":
            return free[name].pop(0)
        return None

    def launch(name: str, stas: List[str], worker: Any = None) -> None:
        nonlocal task_id
        p = pools[name]
        merged, orig_starts = merge_group_fn(by_sta, stas)
        if len(merged) == 0:
            return
        pinned = worker is not None
        if worker is None:
            worker = take_worker(name)
        queued = now()
        ref = submit_fn(
            kind=p.kind,
            actors=p.actors,
            worker=worker,
            stream=merged,
            cls_kw=cls_kw,
            gpu=gpu,
            gpu_frac=p.gpu_frac,
        )
        meta = {
            "task_id": task_id,
            "stations": list(stas),
            "orig_starts": orig_starts,
            "queued_s": queued,
            "worker": worker,
            "pool": name,
            "kind": p.kind,
            "chunk": chunk_index(queued, chunk_s),
            "pinned": pinned,
        }
        task_id += 1
        in_flight[ref] = meta
        in_flight_by_pool[name] += 1

    def harvest(done_refs) -> None:
        for ref in done_refs:
            meta = in_flight.pop(ref)
            out = get_fn(ref)
            returned = now()
            stg = getattr(out, "stage_timing", {}) or {}
            rec = {
                k: meta[k]
                for k in ("task_id", "stations", "queued_s", "worker", "pool", "kind", "chunk")
            }
            rec["returned_s"] = returned
            rec["inference_s"] = float(stg.get("inference_s") or 0.0)
            rec["pick_extract_s"] = float(stg.get("pick_extract_s") or 0.0)
            rec["orig_starts"] = meta["orig_starts"]
            tasks.append(rec)
            in_flight_by_pool[meta["pool"]] -= 1
            p = pools[meta["pool"]]
            if (
                p.kind == "modelactor"
                and meta["worker"] is not None
                and not meta.get("pinned")
            ):
                free[meta["pool"]].append(int(meta["worker"]))
            part = picks_fn(out, meta["orig_starts"])
            for sta, ph in part.items():
                bucket = picks.setdefault(sta, {"p": [], "s": []})
                bucket["p"].extend(ph.get("p") or [])
                bucket["s"].extend(ph.get("s") or [])

    if arrival == "playback":
        # Single-pool static round-robin. Hybrid+playback is rejected upstream.
        name = next(iter(pools))
        p = pools[name]
        buckets = round_robin_buckets(stations, p.k)
        if packaging == "sg":
            for i, grp in enumerate(buckets):
                w = i if p.kind == "modelactor" else None
                launch(name, grp, worker=w)
        else:
            for i, bucket in enumerate(buckets):
                w = i if p.kind == "modelactor" else None
                for sta in bucket:
                    launch(name, [sta], worker=w)
        while in_flight:
            done, _ = wait_fn(list(in_flight), num_returns=1, timeout=None)
            harvest(done)
    else:
        not_ready = set(stations)
        queued: Dict[str, List[str]] = {name: [] for name in pools}
        first_ready: Dict[str, Optional[float]] = {name: None for name in pools}

        def consider_launch(name: str, t: float) -> None:
            p = pools[name]
            q = queued[name]
            g = max(1, p.g)
            eager = max_wait_s <= 0 or packaging == "s1"
            while can_take(name) and q:
                oldest = min(ready.get(s, 0.0) for s in q)
                chunk_flush = t >= chunk_end_s(oldest, chunk_s) - 1e-9
                waited = (t - oldest) >= max_wait_s if max_wait_s > 0 else True
                drain_pool = not any(
                    pool_for_station(s, ready, pools) == name for s in not_ready
                )
                if len(q) >= g:
                    take_n = g
                elif eager or chunk_flush or waited or drain_pool:
                    take_n = len(q)
                else:
                    break
                grp = q[:take_n]
                queued[name] = q = q[take_n:]
                launch(name, grp)
                first_ready[name] = (
                    min(ready.get(s, 0.0) for s in q) if q else None
                )

        while not_ready or any(queued.values()) or in_flight:
            t = now()
            newly = [s for s in list(not_ready) if ready.get(s, 0.0) <= t]
            for s in newly:
                not_ready.discard(s)
                name = pool_for_station(s, ready, pools)
                queued[name].append(s)
                if first_ready[name] is None:
                    first_ready[name] = ready.get(s, 0.0)
            for name in pools:
                consider_launch(name, t)

            if in_flight:
                timeouts = []
                if not_ready:
                    timeouts.append(max(0.0, min(ready[s] for s in not_ready) - now()))
                for name, q in queued.items():
                    if not q or len(q) >= pools[name].g:
                        continue
                    oldest = min(ready.get(s, 0.0) for s in q)
                    if max_wait_s > 0 and packaging != "s1":
                        timeouts.append(max(0.0, oldest + max_wait_s - now()))
                    timeouts.append(max(0.0, chunk_end_s(oldest, chunk_s) - now()))
                timeout = min(timeouts) if timeouts else None
                done, _ = wait_fn(list(in_flight), num_returns=1, timeout=timeout)
                if done:
                    harvest(done)
            elif not_ready or any(queued.values()):
                targets: List[float] = []
                if not_ready:
                    targets.append(min(ready[s] for s in not_ready))
                for name, q in queued.items():
                    if not q:
                        continue
                    oldest = min(ready.get(s, 0.0) for s in q)
                    if max_wait_s > 0 and packaging != "s1" and len(q) < pools[name].g:
                        targets.append(oldest + max_wait_s)
                    targets.append(chunk_end_s(oldest, chunk_s))
                if targets:
                    before = now()
                    skip_to(min(targets))
                    if now() <= before:
                        skip_to(before + 1e-4)
                else:
                    break
            else:
                break

    makespan = now()
    slim_tasks = [{k: v for k, v in t.items() if k != "orig_starts"} for t in tasks]
    events = []
    for t in slim_tasks:
        queued_s = float(t["queued_s"])
        returned = float(t["returned_s"])
        inf = float(t.get("inference_s") or 0.0)
        n = max(1, len(t["stations"]))
        for sta in t["stations"]:
            rdy = float(ready.get(sta, 0.0))
            events.append(
                {
                    "station": sta,
                    "ready_s": round(rdy, 6),
                    "queued_s": round(queued_s, 6),
                    "returned_s": round(returned, 6),
                    "queue_s": round(queued_s - rdy, 6),
                    "e2e_s": round(returned - rdy, 6),
                    "service_s": round(returned - queued_s, 6),
                    "inference_share_s": round(inf / n, 6),
                    "n_in_group": n,
                    "task_id": t["task_id"],
                    "worker": t.get("worker"),
                    "pool": t.get("pool"),
                    "kind": t.get("kind"),
                    "chunk": chunk_index(rdy, chunk_s),
                }
            )
    busy = sum(float(t["returned_s"] - t["queued_s"]) for t in slim_tasks)
    first_q = min((t["queued_s"] for t in slim_tasks), default=0.0)
    last_r = max((t["returned_s"] for t in slim_tasks), default=0.0)
    compute_span = max(0.0, last_r - first_q)
    k_total = sum(p.k for p in pools.values())
    chunk_counts: Dict[int, int] = {}
    for e in events:
        chunk_counts[int(e["chunk"])] = chunk_counts.get(int(e["chunk"]), 0) + 1
    return {
        "n_tasks": len(slim_tasks),
        "makespan_s": round(makespan, 6),
        "compute_span_s": round(compute_span, 6),
        "sum_busy_s": round(busy, 6),
        "idle_frac_wall": round(1.0 - busy / (k_total * makespan), 6) if makespan > 0 else 0.0,
        "idle_frac_compute": (
            round(1.0 - busy / (k_total * compute_span), 6) if compute_span > 0 else 0.0
        ),
        "k_total": k_total,
        "chunk_s": chunk_s,
        "stations_per_chunk": {str(k): v for k, v in sorted(chunk_counts.items())},
        "tasks": slim_tasks,
        "station_events": events,
        "picks": picks,
    }
