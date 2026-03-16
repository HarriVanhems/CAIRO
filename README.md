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

The main paper-aligned synthetic benchmark is now:

```bash
python scripts/run_paper_synthetic_benchmark.py --mode debug
```

This runner uses the CAIRO synthetic setup that is closer to the current draft:

- RankNet / weighted RankNet / GiniNet stage-1 models
- isotonic stage-2 calibration
- robust neural and tabular baselines
- contamination and heavy-tail stress sweeps
- pair-sampling ablations
- saved NeurIPS-style plots under `results/benchmarks/paper_synthetic/figures/`

The earlier cleaned benchmark extracted from the workbench notebook remains available as a secondary synthetic runner:

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
