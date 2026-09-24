"""Map native merged annotate() winners onto Model-Actor / Ripper SG cells.

Stages (see isolation launcher ``run_iso_orch_native_winners.sh``):

* **A** — playback MA SG K=1 with the native winner's dtype, batch, and T.
  This is the only cell that runs the annotate() call already timed.
* **B** — playback MA + Ripper SG, 1 thread/worker, native dtype+batch.
  580 stations: full K ladder (capped at n_cpus). 250: K in {1,2} only.
  CPU and GPU use the same host-CPU budgets {5, 10, 15, 20}.
* **C** — staggered eager on the playback elbow of each B group
  (smallest K within 10% of that group's min makespan).

Do not mix these results with ``orch_annotate/stead_2026-08-14``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RAPID_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT_DEFAULT = RAPID_ROOT / "results" / "annotate_config" / "stead_iso_2026-09-08"
WINNERS_NAME = "winners_merged_quality_safe.json"
QS_THRESH = 0.99
ELBOW_REL = 0.10
TAG = "orch_ann"

MODELS = (
    "PhaseNet",
    "PhaseNetLight",
    "EQTransformer",
    "EQT-NC",
    "EQCCT",
    "EQCCTP",
)
CPU_NCPUS = (5, 10, 15, 20)
GPU_NCPUS = (5, 10, 15, 20)
K_CPU_MA = (1, 2, 4, 5, 10, 20)
K_CPU_RP = (1, 2, 4, 5)
K_GPU_MA = (1, 2, 4)
K_GPU_RP = (1, 2)
DTYPE_TO_METHOD = {"fp32": "annotate_fp32", "bf16": "annotate_bf16", "fp16": "annotate_fp16"}


def winner_key(model: str, n_stations: int, device: str, n_cpus: int) -> str:
    return f"{model}|{int(n_stations)}|{device}|{int(n_cpus)}"


def load_winners(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    p = Path(path) if path else NATIVE_ROOT_DEFAULT / WINNERS_NAME
    data = json.loads(p.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for w in data["winners"]:
        out[winner_key(w["model"], w["n_stations"], w["device"], w["n_cpus"])] = w
    return out


def lookup_winner(
    winners: Dict[str, Dict[str, Any]],
    *,
    model: str,
    n_stations: int,
    device: str,
    n_cpus: int,
) -> Dict[str, Any]:
    k = winner_key(model, n_stations, device, n_cpus)
    if k not in winners:
        raise KeyError(f"no native merged winner for {k}")
    return winners[k]


def ncpus_grid(device: str) -> Sequence[int]:
    return GPU_NCPUS if device == "gpu" else CPU_NCPUS


def k_ladder(stage: str, *, n_stations: int, device: str, composition: str, n_cpus: int) -> List[int]:
    n_cpus = int(n_cpus)
    if stage == "A":
        return [1]
    if int(n_stations) == 250:
        ladder = [1, 2]
    elif composition == "ma":
        ladder = list(K_GPU_MA if device == "gpu" else K_CPU_MA)
    else:
        ladder = list(K_GPU_RP if device == "gpu" else K_CPU_RP)
    return [k for k in ladder if k <= n_cpus]


def torch_threads_for(stage: str, k: int, winner: Dict[str, Any]) -> int:
    if stage == "A" and int(k) == 1:
        return int(winner["torch_threads"])
    return 1


def winner_cell_dir(
    root: Path,
    cell: Dict[str, Any],
    *,
    tag: str = TAG,
) -> Path:
    """Always includes cpusN and thrT so A (native T) and B (T=1) do not collide."""
    return (
        Path(root)
        / cell["composition"]
        / cell["method"]
        / "stead"
        / f"{cell['n_stations']}st"
        / cell["model"]
        / cell["device"]
        / f"cpus{int(cell['n_cpus'])}"
        / f"kma{cell['k_ma']}_krp{cell['k_rp']}"
        / cell["packaging"]
        / cell["arrival"]
        / cell["fill"]
        / f"bs{int(cell['batch_size'])}"
        / f"thr{int(cell['torch_threads'])}"
        / tag
    )


def _emit(
    *,
    stage: str,
    composition: str,
    winner: Dict[str, Any],
    k: int,
    arrival: str,
    fill: str,
) -> Dict[str, Any]:
    method = DTYPE_TO_METHOD[winner["dtype"]]
    if composition == "ma":
        kma, krp = int(k), 0
    else:
        kma, krp = 0, int(k)
    return {
        "stage": stage,
        "composition": composition,
        "method": method,
        "model": winner["model"],
        "n_stations": int(winner["n_stations"]),
        "device": winner["device"],
        "n_cpus": int(winner["n_cpus"]),
        "k_ma": kma,
        "k_rp": krp,
        "n_instances": kma + krp,
        "packaging": "sg",
        "arrival": arrival,
        "fill": fill,
        "batch_size": int(winner["batch_size"]),
        "torch_threads": torch_threads_for(stage, k, winner),
        "native_inference_s_mean": winner.get("inference_s_mean"),
        "native_thread_mult": winner.get("thread_mult"),
        "native_dtype": winner["dtype"],
    }


def iter_stage_ab(
    winners: Dict[str, Dict[str, Any]],
    stage: str,
) -> Iterable[Dict[str, Any]]:
    stage = stage.upper()
    if stage not in ("A", "B"):
        raise ValueError(f"iter_stage_ab expects A or B, got {stage}")
    compositions = ("ma",) if stage == "A" else ("ma", "ripper")
    arrival, fill = "playback", "static"
    for model in MODELS:
        for n_st in (250, 580):
            for device in ("cpu", "gpu"):
                for n_cpus in ncpus_grid(device):
                    w = lookup_winner(
                        winners, model=model, n_stations=n_st, device=device, n_cpus=n_cpus
                    )
                    for composition in compositions:
                        for k in k_ladder(
                            stage,
                            n_stations=n_st,
                            device=device,
                            composition=composition,
                            n_cpus=n_cpus,
                        ):
                            yield _emit(
                                stage=stage,
                                composition=composition,
                                winner=w,
                                k=k,
                                arrival=arrival,
                                fill=fill,
                            )


def _makespan(result: Dict[str, Any]) -> Optional[float]:
    orch = result.get("orch") or {}
    v = orch.get("makespan_s_mean")
    if v is None:
        return None
    return float(v)


def _result_ok(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        r = json.loads(path.read_text())
    except Exception:
        return None
    timing = r.get("timing") or {}
    if timing.get("timed_out") or (r.get("meta") or {}).get("timed_out"):
        return None
    if float(timing.get("success_rate") or 0) < 1.0:
        return None
    return r


def iter_stage_c_from_playback(
    results_root: Path,
    winners: Dict[str, Dict[str, Any]],
    *,
    tag: str = TAG,
    elbow_rel: float = ELBOW_REL,
) -> Iterable[Dict[str, Any]]:
    """Promote each B playback group to one staggered-eager cell at the elbow K."""
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for cell in iter_stage_ab(winners, "B"):
        key = (
            cell["composition"],
            cell["model"],
            cell["n_stations"],
            cell["device"],
            cell["n_cpus"],
        )
        groups.setdefault(key, []).append(cell)

    for _key, cells in groups.items():
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for cell in cells:
            r = _result_ok(winner_cell_dir(results_root, cell, tag=tag) / "result.json")
            if r is None:
                continue
            ms = _makespan(r)
            if ms is None:
                continue
            k = int(cell["k_ma"] or cell["k_rp"])
            scored.append((ms, k, cell))
        if not scored:
            continue
        best = min(s[0] for s in scored)
        elbow = [s for s in scored if s[0] <= best * (1.0 + float(elbow_rel))]
        _ms, _k, winner_cell = min(elbow, key=lambda s: (s[1], s[0]))
        k = int(winner_cell["k_ma"] or winner_cell["k_rp"])
        w = lookup_winner(
            winners,
            model=winner_cell["model"],
            n_stations=winner_cell["n_stations"],
            device=winner_cell["device"],
            n_cpus=winner_cell["n_cpus"],
        )
        out = _emit(
            stage="C",
            composition=winner_cell["composition"],
            winner=w,
            k=k,
            arrival="staggered",
            fill="eager",
        )
        out["promoted_from_k"] = k
        out["playback_makespan_s"] = best
        yield out


def iter_stage(
    stage: str,
    winners: Dict[str, Dict[str, Any]],
    *,
    results_root: Optional[Path] = None,
    tag: str = TAG,
) -> Iterable[Dict[str, Any]]:
    stage = stage.upper()
    if stage in ("A", "B"):
        yield from iter_stage_ab(winners, stage)
        return
    if stage == "C":
        if results_root is None:
            raise ValueError("stage C needs results_root (playback elbows)")
        yield from iter_stage_c_from_playback(results_root, winners, tag=tag)
        return
    raise ValueError(f"unknown stage {stage}")


def cell_queue_line(cell: Dict[str, Any]) -> str:
    keys = (
        "n_cpus",
        "composition",
        "method",
        "model",
        "n_stations",
        "device",
        "k_ma",
        "k_rp",
        "packaging",
        "arrival",
        "fill",
        "batch_size",
        "torch_threads",
        "stage",
    )
    return "|".join(str(cell[k]) for k in keys)
