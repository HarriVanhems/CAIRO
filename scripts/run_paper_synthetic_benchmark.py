from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cairo.synthetic.paper_benchmark import RuntimeConfig, run_full_synthetic_benchmark  # noqa: E402


def _scenario_filter(raw: str) -> set[str] | None:
    if not raw.strip():
        return None
    return {chunk.strip() for chunk in raw.split(",") if chunk.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paper-aligned CAIRO synthetic benchmark with plots."
    )
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "benchmarks" / "paper_synthetic",
    )
    parser.add_argument(
        "--scenario-filter",
        type=str,
        default="",
        help="Comma-separated scenario names to limit the main benchmark run.",
    )
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = RuntimeConfig.from_mode(args.mode, output_dir=args.output_dir, repeats=args.repeats)
    results = run_full_synthetic_benchmark(
        cfg,
        scenario_filter=_scenario_filter(args.scenario_filter),
        skip_sweeps=args.skip_sweeps,
        skip_plots=args.skip_plots,
    )
    print(f"Saved outputs to {cfg.output_dir}")
    print(results["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
