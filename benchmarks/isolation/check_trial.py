import json
import sys
from collections import defaultdict
from typing import Dict, Set


def count_unique_trace_rows_by_dataset(path: str) -> Dict[str, int]:
    """Return a mapping sb_dataset -> count of unique sb_trace_row.

    The input file is expected to contain one JSON object per line
    (JSONL format), like the records you provided.
    """
    unique_rows: Dict[str, Set[int]] = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_no}: invalid JSON ({e})", file=sys.stderr)
                continue

            sb_dataset = rec.get("sb_dataset")
            sb_trace_row = rec.get("sb_trace_row")

            # Only count entries that have both fields
            if sb_dataset is None or sb_trace_row is None:
                continue

            unique_rows[sb_dataset].add(sb_trace_row)

    return {ds: len(rows) for ds, rows in unique_rows.items()}


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1:
        print(f"Usage: python {sys.argv[0]} <path_to_jsonl_file>")
        sys.exit(1)

    path = argv[0]
    counts = count_unique_trace_rows_by_dataset(path)

    if not counts:
        print("No valid records with 'sb_dataset' and 'sb_trace_row' found.")
        return

    print("sb_dataset\tunique_sb_trace_row_count")
    for ds, cnt in sorted(counts.items()):
        print(f"{ds}\t{cnt}")


if __name__ == "__main__":
    main()
