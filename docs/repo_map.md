# Repo Map

## Paper-to-Code Mapping

- Two-stage calibrated ranking pipeline:
  implemented progressively in `src/cairo/` and runnable from `scripts/`
- Synthetic zero-inflated Tweedie benchmark:
  `src/cairo/synthetic/zero_inflated_tweedie.py`
- Historical notebook provenance:
  `notebooks/archive/paper_workbench.ipynb`
- Figures and legacy plot outputs:
  `results/figures/`
- Benchmark outputs:
  `results/benchmarks/`

## Current State

The archived notebook contains duplicated definitions and mixes multiple experiment families in a single file. The repo has been reorganized so the notebook can now be treated as source material to extract from, rather than as the runnable artifact itself.

## Immediate Extraction Priorities

1. Real-data dataset loaders and a canonical benchmark runner.
2. RankNet and weighted RankNet training code promoted out of the notebook.
3. Robustness-to-contamination experiment promoted into its own script.
4. A single metrics module shared across all experiment families.
