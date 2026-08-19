#!/usr/bin/env python3
"""Cell list for the locked-recipe transfer suite (another machine).

Three layers, same 5 pickers and the 5/10/15/20 × CPU/GPU sweep used in the
merged Annotate precision study. Do not add fp16, Ripper, hybrid, S1, or
wait-5/wait-10.

See RAPID/docs/RAPID_LOCKED_RECIPE_TRANSFER.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MODELS = (
    "EQCCT",
    "PhaseNet",
    "PhaseNetLight",
    "EQTransformer",
    "EQT-NC",
)
STATIONS = (250, 580)
DEVICES = ("cpu", "gpu")
CORE_GRID = (5, 10, 15, 20)
LAYERS = ("native", "playback", "staggered")
METHOD = "annotate_bf16"
BATCH = 512
TAG = "xfer"


def _csv_ints(raw: str, default: Sequence[int]) -> List[int]:
    if not str(raw or "").strip():
        return [int(x) for x in default]
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _csv_str(raw: str, default: Sequence[str]) -> List[str]:
    if not str(raw or "").strip():
        return [str(x) for x in default]
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def locked_k(
    *,
    device: str,
    n_stations: int,
    model: str,
    n_cpus: int,
    cpu_k_cap: Optional[int] = None,
    gpu_k_cap: Optional[int] = None,
) -> int:
    """Workstation elbow, then cap to this slot and optional machine caps.

    CPU 580: K=20. CPU 250: K=20 for EQCCT, K=10 otherwise.
    GPU: K=4 except PhaseNet at 250 stations (K=2).
    """
    if device == "cpu":
        want = 20 if (int(n_stations) == 580 or model == "EQCCT") else 10
        cap = cpu_k_cap
    else:
        want = 2 if (int(n_stations) == 250 and model == "PhaseNet") else 4
        cap = gpu_k_cap
    k = min(int(want), int(n_cpus))
    if cap is not None:
        k = min(k, int(cap))
    return max(1, k)


def iter_cells(
    *,
    layers: Sequence[str] = LAYERS,
    models: Sequence[str] = MODELS,
    stations: Sequence[int] = STATIONS,
    devices: Sequence[str] = DEVICES,
    core_grid: Sequence[int] = CORE_GRID,
    cpu_k_cap: Optional[int] = None,
    gpu_k_cap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for layer in layers:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}; expected one of {LAYERS}")
        for model in models:
            for nst in stations:
                for device in devices:
                    for n_cpus in core_grid:
                        cell: Dict[str, Any] = {
                            "layer": layer,
                            "method": METHOD,
                            "model": model,
                            "n_stations": int(nst),
                            "device": device,
                            "n_cpus": int(n_cpus),
                            "batch_size": BATCH,
                            "tag": TAG,
                        }
                        if layer == "native":
                            cell.update(
                                {
                                    "packaging": "merged",
                                    "torch_threads": int(n_cpus),
                                    "composition": "",
                                    "k_ma": 0,
                                    "k_rp": 0,
                                    "arrival": "",
                                    "fill": "",
                                }
                            )
                        else:
                            k = locked_k(
                                device=device,
                                n_stations=int(nst),
                                model=model,
                                n_cpus=int(n_cpus),
                                cpu_k_cap=cpu_k_cap,
                                gpu_k_cap=gpu_k_cap,
                            )
                            cell.update(
                                {
                                    "packaging": "sg",
                                    "torch_threads": 1,
                                    "composition": "ma",
                                    "k_ma": k,
                                    "k_rp": 0,
                                    "arrival": "playback" if layer == "playback" else "staggered",
                                    "fill": "static" if layer == "playback" else "eager",
                                }
                            )
                        out.append(cell)
    return out


def result_path(root: Path, cell: Dict[str, Any]) -> Path:
    tag = cell.get("tag") or TAG
    if cell["layer"] == "native":
        thr = cell["torch_threads"]
        return (
            root
            / cell["method"]
            / "stead"
            / f"{cell['n_stations']}st"
            / cell["model"]
            / cell["device"]
            / f"cpus{cell['n_cpus']}"
            / f"thr{thr}"
            / f"bs{cell['batch_size']}"
            / "merged"
            / tag
            / "result.json"
        )
    return (
        root
        / cell["composition"]
        / cell["method"]
        / "stead"
        / f"{cell['n_stations']}st"
        / cell["model"]
        / cell["device"]
        / f"kma{cell['k_ma']}_krp{cell['k_rp']}"
        / cell["packaging"]
        / cell["arrival"]
        / cell["fill"]
        / f"bs{cell['batch_size']}"
        / tag
        / "result.json"
    )


def cell_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        r = json.loads(path.read_text())
    except Exception:
        return False
    sr = (r.get("timing") or {}).get("success_rate")
    return float(sr or 0) >= 1.0


def env_cells() -> List[Dict[str, Any]]:
    layers = _csv_str(os.environ.get("LAYER", "all"), LAYERS)
    if layers == ["all"]:
        layers = list(LAYERS)
    models = _csv_str(os.environ.get("MODELS", ""), MODELS)
    stations = _csv_ints(os.environ.get("STATIONS", ""), STATIONS)
    devices = _csv_str(os.environ.get("DEVICES", ""), DEVICES)
    if os.environ.get("SKIP_GPU", "") == "1":
        devices = [d for d in devices if d != "gpu"]
    if os.environ.get("SKIP_CPU", "") == "1":
        devices = [d for d in devices if d != "cpu"]
    core_grid = _csv_ints(os.environ.get("CORE_GRID", ""), CORE_GRID)
    cpu_k_cap = os.environ.get("CPU_K_CAP")
    gpu_k_cap = os.environ.get("GPU_K_CAP")
    return iter_cells(
        layers=layers,
        models=models,
        stations=stations,
        devices=devices,
        core_grid=core_grid,
        cpu_k_cap=int(cpu_k_cap) if cpu_k_cap else None,
        gpu_k_cap=int(gpu_k_cap) if gpu_k_cap else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print-matrix", action="store_true")
    ap.add_argument("--status", type=Path, default=None, help="Results root to score.")
    args = ap.parse_args()
    cells = env_cells()
    if args.print_matrix:
        print(f"N_CELLS {len(cells)}")
        for c in cells:
            print(
                "|".join(
                    str(c[k])
                    for k in (
                        "layer",
                        "model",
                        "n_stations",
                        "device",
                        "n_cpus",
                        "k_ma",
                        "arrival",
                        "fill",
                    )
                )
            )
        return 0
    if args.status is not None:
        done = fail = 0
        for c in cells:
            p = result_path(args.status, c)
            if cell_ok(p):
                done += 1
            elif p.is_file():
                fail += 1
        print(
            f"expected={len(cells)} done={done} failed_or_partial={fail} "
            f"remaining={len(cells) - done - fail}"
        )
        return 0
    print(f"n_cells={len(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
