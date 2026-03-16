# Notebooks

`notebooks/archive/paper_workbench.ipynb` is the preserved exploratory notebook that currently mixes:

- synthetic experiments
- robustness experiments
- real-data work
- later LambdaMART insurance-style experiments

Keep it for reference, but avoid making it the source of truth. Any experiment that matters for the paper should be promoted into versioned Python modules under `src/` plus a small runner in `scripts/`.

`notebooks/archive/paper_synthetic_workbench.ipynb` is the newer synthetic benchmark notebook copy that motivated the current paper-aligned runner in `scripts/run_paper_synthetic_benchmark.py`.
