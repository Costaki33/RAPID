#!/usr/bin/env python3
"""Sweep Model-Actor orchestration with optional Slipstream inference inside each actor.

The published 228-station TexNet Model-Actor tables (§3.2) used
``SeisBenchModelActor`` → ``model.classify()``. This script runs the same
``EvaluateSystem`` sweep with ``slipstream_inference=True`` so each persistent
actor loads RAPID's ``LeanPyTorchBackend`` instead.

Baseline (classify inside Model-Actor) for comparison::

    python benchmarks/fair/run_modelactor_slipstream_sweep.py \\
        --inference-path classify \\
        --model PhaseNet \\
        --cpus 20 --stations 228 --gpus 1 \\
        --conc-stations 22

Slipstream BF16 on one GPU::

    python benchmarks/fair/run_modelactor_slipstream_sweep.py \\
        --inference-path slipstream_bf16 \\
        --model EQTransformer \\
        --cpus 20 --stations 228 --gpus 1 \\
        --conc-stations 22

Full Cartesian sweep (all models × inference paths × CPU/station/GPU grids)::

    python benchmarks/fair/run_modelactor_slipstream_sweep.py --run-all

Outputs one CSV per trial under ``--csv-root`` (same schema as other EvaluateSystem runs).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo roots
RAPID_ROOT = Path(__file__).resolve().parents[2]
if str(RAPID_ROOT) not in sys.path:
    sys.path.insert(0, str(RAPID_ROOT))

from rapid.orchestration import EvaluateSystem  # noqa: E402

# Paper models (SeisBench) + EQCCT (classify/Model-Actor only; no Slipstream)
MODELS: Dict[str, Dict[str, Any]] = {
    "PhaseNet": {
        "model_type": "seisbench",
        "parent": "PhaseNet",
        "child": "original",
    },
    "PhaseNetLight": {
        "model_type": "seisbench",
        "parent": "PhaseNetLight",
        "child": "stead",
    },
    "EQTransformer": {
        "model_type": "seisbench",
        "parent": "EQTransformer",
        "child": "original",
    },
    "EQT-NC": {
        "model_type": "seisbench",
        "parent": "EQTransformer",
        "child": "original_nonconservative",
    },
    "EQCCT": {
        "model_type": "eqcct",
        "parent": None,
        "child": None,
        "p_model": "models/EQCCT/test_trainer_024.h5",
        "s_model": "models/EQCCT/test_trainer_021.h5",
    },
}

# Slipstream precision modes (EQTransformer family: no fp16)
INFERENCE_PATHS: Dict[str, Dict[str, Any]] = {
    "classify": {
        "slipstream_inference": False,
        "slipstream_dtype": "fp32",
        "slipstream_compile": False,
    },
    "slipstream_fp16": {
        "slipstream_inference": True,
        "slipstream_dtype": "fp16",
        "slipstream_compile": False,
    },
    "slipstream_bf16": {
        "slipstream_inference": True,
        "slipstream_dtype": "bf16",
        "slipstream_compile": False,
    },
    "slipstream_bf16_compile": {
        "slipstream_inference": True,
        "slipstream_dtype": "bf16",
        "slipstream_compile": True,
    },
}

EQT_PARENTS = {"EQTransformer", "EQT-NC"}


def _allowed_paths(model_key: str) -> List[str]:
    if model_key == "EQCCT":
        return ["classify"]
    if model_key in EQT_PARENTS or MODELS[model_key]["parent"] == "EQTransformer":
        return ["classify", "slipstream_bf16", "slipstream_bf16_compile"]
    return list(INFERENCE_PATHS.keys())


def _cpu_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _run_one_trial(
    *,
    model_key: str,
    path_key: str,
    cpus: int,
    stations: int,
    gpus: List[int],
    conc_stations: int,
    args: argparse.Namespace,
) -> None:
    m = MODELS[model_key]
    path = INFERENCE_PATHS[path_key]

    if path["slipstream_compile"] and len(gpus) > 1:
        print(f"SKIP {model_key}/{path_key}: compile requires single GPU")
        return

    csv_dir = (
        Path(args.csv_root)
        / f"model_{model_key.replace('/', '_')}"
        / f"path_{path_key}"
        / f"cpu{cpus}_st{stations}_gpu{len(gpus)}_conc{conc_stations}"
    )
    csv_dir.mkdir(parents=True, exist_ok=True)

    cpu_id_list = list(range(cpus))
    eval_mode = "gpu" if gpus else "cpu"

    common = dict(
        eval_mode=eval_mode,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        log_filepath=str(csv_dir / "eqcctpro.log"),
        csv_dir=str(csv_dir),
        cpu_id_list=cpu_id_list,
        min_cpu_amount=cpus,
        cpu_test_step_size=cpus,
        stations2use=stations,
        starting_amount_of_stations=stations,
        station_list_step_size=stations,
        min_conc_stations=conc_stations,
        conc_station_tasks_step_size=0,
        conc_station_tasks_max_only=True,
        ram_safety_cap=args.ram_safety_cap,
        cudnn_headroom=args.cudnn_headroom,
        ripper=False,
        start_time=args.start_time,
        end_time=args.end_time,
        timechunk_dt=args.timechunk_dt,
        waveform_overlap=0,
        tmp_dir=args.tmp_dir,
        Detection_threshold=args.detection_threshold,
        P_threshold=args.p_threshold,
        S_threshold=args.s_threshold,
        slipstream_inference=path["slipstream_inference"],
        slipstream_dtype=path["slipstream_dtype"],
        slipstream_compile=path["slipstream_compile"],
        slipstream_overlap_samples=args.slipstream_overlap_samples,
        slipstream_batch_size=args.slipstream_batch_size,
        overwrite=args.overwrite,
        pick_output_format="ascii",
    )

    if m["model_type"] == "seisbench":
        ev = EvaluateSystem(
            model_type="seisbench",
            seisbench_parent_model=m["parent"],
            seisbench_child_model=m["child"],
            selected_gpus=gpus if gpus else None,
            max_vram_mb=args.max_vram_mb,
            **common,
        )
    else:
        p_path = str(RAPID_ROOT / m["p_model"])
        s_path = str(RAPID_ROOT / m["s_model"])
        ev = EvaluateSystem(
            model_type="eqcct",
            p_model_filepath=p_path,
            s_model_filepath=s_path,
            selected_gpus=gpus if gpus else None,
            max_vram_mb=args.max_vram_mb,
            **common,
        )

    tag = f"{model_key} path={path_key} cpus={cpus} st={stations} gpus={gpus} conc={conc_stations}"
    print(f"\n=== TRIAL: {tag} ===")
    print(f"CSV dir: {csv_dir}")
    if args.dry_run:
        return
    ev.evaluate()


def main() -> int:
    ap = argparse.ArgumentParser(description="Model-Actor + Slipstream orchestration sweep")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=RAPID_ROOT / "data/230_stations_1_min_dt/20241215T120000Z_20241215T120100Z",
        help="TexNet timechunk directory (flat or per-station layout)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=RAPID_ROOT / "results/modelactor_slipstream/picks",
    )
    ap.add_argument(
        "--csv-root",
        type=Path,
        default=RAPID_ROOT / "results/modelactor_slipstream/csv",
    )
    ap.add_argument("--tmp-dir", type=Path, default=Path("/lambda1a/skevofilaxc/tmp"))
    ap.add_argument("--start-time", default="2024-12-15 12:00:00")
    ap.add_argument("--end-time", default="2024-12-15 12:01:00")
    ap.add_argument("--timechunk-dt", type=int, default=1)
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--detection-threshold", type=float, default=0.3)
    ap.add_argument("--ram-safety-cap", type=float, default=0.95)
    ap.add_argument("--cudnn-headroom", type=float, default=0.20)
    ap.add_argument("--max-vram-mb", type=float, default=40000)
    ap.add_argument("--slipstream-overlap-samples", type=int, default=0)
    ap.add_argument("--slipstream-batch-size", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--model", choices=list(MODELS.keys()))
    ap.add_argument(
        "--inference-path",
        choices=list(INFERENCE_PATHS.keys()),
        help="classify = SeisBench inside Model-Actor; slipstream_* = LeanPyTorchBackend in actor",
    )
    ap.add_argument("--cpus", type=int, default=20)
    ap.add_argument("--stations", type=int, default=228)
    ap.add_argument("--gpus", default="0", help="Comma GPU ids, or empty for CPU-only")
    ap.add_argument("--conc-stations", type=int, default=None, help="Model-Actor pool size (concurrent stations)")

    ap.add_argument("--run-all", action="store_true", help="Full sweep over default grids")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = _cpu_list(args.gpus) if args.gpus.strip() else []

    if args.run_all:
        cpu_grid = [5, 8, 11, 14, 17, 20]
        station_grid = [5, 10, 20, 50, 100, 150, 200, 228]
        gpu_grid: List[List[int]] = [[], [0], [0, 1]]
        for model_key in MODELS:
            for path_key in _allowed_paths(model_key):
                for cpus in cpu_grid:
                    for stations in station_grid:
                        for g in gpu_grid:
                            if path_key.endswith("_compile") and len(g) > 1:
                                continue
                            conc = args.conc_stations or max(1, cpus // 2)
                            _run_one_trial(
                                model_key=model_key,
                                path_key=path_key,
                                cpus=cpus,
                                stations=stations,
                                gpus=g,
                                conc_stations=conc,
                                args=args,
                            )
        return 0

    if not args.model or not args.inference_path:
        ap.error("Specify --model and --inference-path, or use --run-all")

    if args.inference_path not in _allowed_paths(args.model):
        ap.error(f"{args.inference_path} not valid for {args.model}; allowed: {_allowed_paths(args.model)}")

    conc = args.conc_stations or max(1, args.cpus // 2)
    _run_one_trial(
        model_key=args.model,
        path_key=args.inference_path,
        cpus=args.cpus,
        stations=args.stations,
        gpus=gpus,
        conc_stations=conc,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
