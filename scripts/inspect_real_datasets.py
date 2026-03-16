from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cairo.data.uci import dataset_summary_table, load_real_datasets  # noqa: E402
from cairo.paths import RESULTS_DIR, ensure_dir  # noqa: E402


def main() -> None:
    datasets = load_real_datasets()
    summary = dataset_summary_table(datasets)
    output_dir = ensure_dir(RESULTS_DIR / "benchmarks" / "real_data_inventory")
    output_path = output_dir / "dataset_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved dataset summary to {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
