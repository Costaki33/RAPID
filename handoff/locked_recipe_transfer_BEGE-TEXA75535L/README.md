# Handoff package index

Start here: **[HANDOFF.md](HANDOFF.md)** — full paper handoff for the locked-recipe transfer on this laptop.

| Path | Contents |
|---|---|
| [`data/transfer_summary.csv`](data/transfer_summary.csv) | One row per successful cell: method, model, stations, device, n_cpus, n_gpus, K, runtime_s, **p_f1**, **s_f1** (**152** rows) |
| [`figures/transfer_canvas.html`](figures/transfer_canvas.html) | Interactive Plotly canvas |
| [`raw_results/`](raw_results/) | Full raw tree (`annotate_bf16/`, `ma/`, log) — **152** `result.json` with `timing.success_rate == 1.0` |
| [`provenance/`](provenance/) | Machine README, caps notes, software versions, host snapshot, full run log |

WSL source of truth (mirrored under `raw_results/`):

`~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/`

Do **not** run more benchmark matrices for this package. Skip CPU K=15; do not rerun K=20.
