# CAIRO

Research repository for the CAIRO paper: calibrated two-stage rank-then-regress modeling for zero-inflated and heavy-tailed regression.

## What Is Canonical

- `src/cairo/`: reusable Python code
- `scripts/`: runnable experiment entry points
- `data/`: raw and processed datasets
- `results/`: generated benchmark outputs and figures
- `paper/`: local paper drafts
- `notebooks/archive/`: exploratory notebook history kept for provenance, not as the source of truth

## Current Entry Points

The first cleaned benchmark extracted from the workbench notebook is:

```bash
python3 scripts/run_zero_inflated_benchmark.py
```

This benchmark implements the zero-inflated Tweedie synthetic experiment using a stronger, reproducible protocol:

- separate training, tuning, calibration, and test splits
- train-only relevance binning
- LightGBM LambdaMART ranker plus isotonic calibration
- multi-seed summaries saved under `results/benchmarks/`

For the real-data side, the current canonical inventory check is:

```bash
python scripts/inspect_real_datasets.py
```

That script loads the benchmark datasets from `data/raw/benchmarks/` and writes a standardized dataset summary to `results/benchmarks/real_data_inventory/`.

## Repo Layout

- `data/raw/benchmarks/`: one folder per benchmark dataset
- `data/processed/`: cleaned intermediate artifacts
- `docs/`: repo map and experiment upgrade notes
- `notebooks/archive/`: mixed prototype notebook snapshots
- `paper/drafts/`: local PDF drafts
- `results/figures/`: paper figures and analysis plots

## Working Rules

- Keep exploratory work in notebooks, but promote anything used in the paper into `src/` and `scripts/`.
- Save benchmark outputs to `results/benchmarks/` with config files and per-run CSVs.
- Treat `notebooks/archive/paper_workbench.ipynb` as historical context, not the implementation target.
