# Benchmarks

Runnable evaluation entry points, grouped so the published tree stays readable.

| Folder | Purpose |
| --- | --- |
| `fair/` | Fair deployment matrix, single-trial runners, latency / oversubscription sweeps, pick comparators, model export |
| `isolation/` | Strictly sequential re-measurements (no concurrent-trial contention) |
| `analysis/` | Post-run analysis, table generation, pick-quality aggregation |

All scripts assume the RAPID repo root as the working directory (the shell
wrappers `cd` there for you). Install with `pip install -e ".[orchestration]"`
before running Model-Actor / Ripper trials.

For day-to-day picking, prefer `examples/pick_network.py` or `rapid.pick`
instead of these suites.
