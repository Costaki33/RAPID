#!/usr/bin/env python3
"""Fix annotate pick_quality offset bug in existing JSONL files.

The bug: `_annotate_stream_to_window_pred` extracted windows without accounting
for the time offset between input stream and annotate output. This caused all
`baseline_annotate` pick indices (onset_p, argmax_p, etc.) to be shifted earlier
by the model-specific offset.

Model offsets (at 100 Hz):
  - PhaseNet: 250 samples (2.5s)
  - PhaseNetLight: 0 samples (no correction needed)
  - EQTransformer: 500 samples (5s)
  - EQT-NC: 500 samples (5s)

This script corrects the existing JSONL by adding the offset to pick indices
and recalculating delta-vs-catalog fields for `baseline_annotate` rows.

Usage:
  python fix_annotate_offset.py --input results/seisbench_matrix_lean_aff16.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Model-specific offsets in samples (at 100 Hz, 6000-sample input)
MODEL_OFFSETS: Dict[str, int] = {
    "PhaseNet": 250,
    "PhaseNetLight": 0,
    "EQTransformer": 500,
    "EQT-NC": 500,
}


def fix_pick_quality(
    pq: Dict[str, Any],
    offset: int,
    p_catalog_in_window: Optional[int],
    s_catalog_in_window: Optional[int],
) -> Dict[str, Any]:
    """Apply offset correction to pick_quality fields."""
    if not pq or offset == 0:
        return pq

    out = dict(pq)

    # Correct onset_p
    if out.get("onset_p") is not None:
        out["onset_p"] = int(out["onset_p"]) + offset
        if p_catalog_in_window is not None:
            out["onset_delta_p_vs_catalog"] = int(out["onset_p"]) - int(p_catalog_in_window)

    # Correct argmax_p
    if out.get("argmax_p") is not None:
        out["argmax_p"] = int(out["argmax_p"]) + offset
        if p_catalog_in_window is not None:
            out["delta_p_vs_catalog"] = int(out["argmax_p"]) - int(p_catalog_in_window)

    # Correct onset_s
    if out.get("onset_s") is not None:
        out["onset_s"] = int(out["onset_s"]) + offset
        if s_catalog_in_window is not None:
            out["onset_delta_s_vs_catalog"] = int(out["onset_s"]) - int(s_catalog_in_window)

    # Correct argmax_s
    if out.get("argmax_s") is not None:
        out["argmax_s"] = int(out["argmax_s"]) + offset
        if s_catalog_in_window is not None:
            out["delta_s_vs_catalog"] = int(out["argmax_s"]) - int(s_catalog_in_window)

    return out


def process_jsonl(input_path: Path, output_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """Process JSONL and apply corrections. Returns stats."""
    stats = {
        "total_rows": 0,
        "annotate_rows": 0,
        "corrected_rows": 0,
        "skipped_no_offset": 0,
        "skipped_no_pq": 0,
        "unknown_models": set(),
    }

    lines_out = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            stats["total_rows"] += 1
            line = line.rstrip("\n")
            if not line.strip():
                lines_out.append(line)
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                lines_out.append(line)
                continue

            # Only process baseline_annotate rows
            if row.get("runner") not in ("baseline_annotate", "baseline_annotate_dual"):
                lines_out.append(line)
                continue

            stats["annotate_rows"] += 1

            # Get model label and offset
            model_label = row.get("model_label", "")
            offset = MODEL_OFFSETS.get(model_label)

            if offset is None:
                stats["unknown_models"].add(model_label)
                lines_out.append(line)
                continue

            if offset == 0:
                stats["skipped_no_offset"] += 1
                lines_out.append(line)
                continue

            pq = row.get("pick_quality")
            if not pq:
                stats["skipped_no_pq"] += 1
                lines_out.append(line)
                continue

            # Apply correction
            p_cat = row.get("p_catalog_in_window")
            s_cat = row.get("s_catalog_in_window")
            row["pick_quality"] = fix_pick_quality(pq, offset, p_cat, s_cat)
            row["_offset_correction_applied"] = offset

            stats["corrected_rows"] += 1
            lines_out.append(json.dumps(row, separators=(",", ":")))

    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for line in lines_out:
                f.write(line + "\n")

    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="Fix annotate pick offset bug in JSONL")
    p.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    p.add_argument("--output", type=Path, default=None, help="Output JSONL (default: overwrite input)")
    p.add_argument("--backup", action="store_true", help="Create .bak backup of input")
    p.add_argument("--dry-run", action="store_true", help="Don't write, just show stats")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    output_path = args.output or args.input

    if args.backup and not args.dry_run and output_path == args.input:
        backup_path = args.input.with_suffix(args.input.suffix + ".bak")
        shutil.copy2(args.input, backup_path)
        print(f"Backup: {backup_path}")

    stats = process_jsonl(args.input, output_path, dry_run=args.dry_run)

    print(f"\nStats for {args.input}:")
    print(f"  Total rows: {stats['total_rows']}")
    print(f"  Annotate rows: {stats['annotate_rows']}")
    print(f"  Corrected: {stats['corrected_rows']}")
    print(f"  Skipped (offset=0): {stats['skipped_no_offset']}")
    print(f"  Skipped (no pick_quality): {stats['skipped_no_pq']}")
    if stats["unknown_models"]:
        print(f"  Unknown models: {stats['unknown_models']}")

    if args.dry_run:
        print("\n(dry run - no changes written)")
    else:
        print(f"\nWrote: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
