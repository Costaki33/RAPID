"""Run the full RAPID sweep matrix defined in a JSON or YAML config.

Example::

    python scripts/run_matrix.py --config configs/full_matrix.json
"""

from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


def load_config(path: str) -> dict:
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore[import]

        return yaml.safe_load(p.read_text())
    return json.loads(p.read_text())


@contextmanager
def _single_instance_lock(jsonl_path: str, force: bool = False):
    """Guard against two run_matrix processes writing to the same JSONL.

    Creates a sibling ``<jsonl>.lock`` file containing our PID. If it already
    exists and the owning PID is alive, we bail out (unless ``force=True``).
    Stale locks (owner dead) are reclaimed automatically.
    """

    lock_path = Path(str(jsonl_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _try_claim() -> bool:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            return False
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()}\n")
        return True

    if not _try_claim():
        existing_pid: int | None = None
        try:
            existing_pid = int(lock_path.read_text().strip() or "0") or None
        except Exception:
            existing_pid = None
        if existing_pid is not None and _pid_alive(existing_pid) and not force:
            raise SystemExit(
                f"ERROR: another run_matrix process (PID {existing_pid}) is already "
                f"writing to {jsonl_path}.\n"
                f"  - If that run is actually dead, remove the stale lock: rm {lock_path}\n"
                f"  - Or re-run this command with --force-lock to override."
            )
        # Stale or forced: reclaim the lock.
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        if not _try_claim():
            raise SystemExit(
                f"ERROR: could not claim lock at {lock_path} even after clearing it."
            )

    try:
        yield lock_path
    finally:
        try:
            if lock_path.exists() and lock_path.read_text().strip() == str(os.getpid()):
                lock_path.unlink()
        except Exception:
            pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Matrix config file (.json or .yaml).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing JSONL output and re-run every cell.",
    )
    ap.add_argument(
        "--no-retry-errors", action="store_true",
        help="Treat prior error rows as done (skip them) instead of retrying.",
    )
    ap.add_argument(
        "--output-jsonl", default=None,
        help="Override cfg.output_jsonl. Resume reads from + appends to this path.",
    )
    ap.add_argument(
        "--force-lock", action="store_true",
        help="Override a stale/active lock on the output JSONL. Only use if you "
             "are sure no other run_matrix process is writing to it.",
    )
    args = ap.parse_args()

    from rapid.matrix import MatrixConfig, run_matrix

    cfg = MatrixConfig.from_dict(load_config(args.config))
    if args.no_resume:
        cfg.resume = False
    if args.no_retry_errors:
        cfg.retry_errors = False
    if args.output_jsonl:
        cfg.output_jsonl = args.output_jsonl

    lock_target = args.output_jsonl or cfg.output_jsonl
    with _single_instance_lock(lock_target, force=args.force_lock):
        out = run_matrix(cfg, dry_run=args.dry_run)
    print(f"Matrix complete. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
