from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cairo.synthetic.zero_inflated_tweedie import (  # noqa: E402
    ZeroInflatedBenchmarkConfig,
    default_output_dir,
    run_experiment,
    save_experiment,
)


def parse_q_values(raw: str) -> tuple[float, ...]:
    return tuple(float(chunk.strip()) for chunk in raw.split(",") if chunk.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CAIRO zero-inflated Tweedie LambdaMART benchmark."
    )
    parser.add_argument(
        "--q-values",
        default="1.0,0.85,0.75,0.5,0.25,0.1",
        help="Comma-separated zero-inflation survival probabilities.",
    )
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-tune", type=int, default=500)
    parser.add_argument("--n-calibration", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ZeroInflatedBenchmarkConfig(
        q_values=parse_q_values(args.q_values),
        replications=args.replications,
        n_train=args.n_train,
        n_tune=args.n_tune,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        n_jobs=args.n_jobs,
    )
    per_run, summary = run_experiment(config)
    output_dir = save_experiment(per_run, summary, config, output_dir=args.output_dir)
    print(f"Saved per-run results to {output_dir / 'per_run.csv'}")
    print(f"Saved summary results to {output_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
