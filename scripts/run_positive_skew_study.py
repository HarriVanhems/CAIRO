from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cairo.synthetic.positive_skew_study import (  # noqa: E402
    PositiveSkewStudyConfig,
    run_positive_skew_study,
    save_positive_skew_study,
)


def _parse_float_list(raw: str) -> tuple[float, ...] | None:
    if not raw.strip():
        return None
    values = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(float(item))
    return tuple(values) if values else None


def _parse_model_list(raw: str) -> tuple[str, ...] | None:
    if not raw.strip():
        return None
    models = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        models.append(item)
    return tuple(models) if models else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the positive-skew severity study for CAIRO as a standalone experiment."
    )
    parser.add_argument("--mode", choices=["smoke", "debug", "full"], default="smoke")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "benchmarks" / "positive_skew_study",
    )
    parser.add_argument(
        "--severity-levels",
        type=str,
        default="",
        help="Comma-separated severity values. Defaults depend on mode.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names to run. Defaults to the study model set.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = PositiveSkewStudyConfig.from_mode(
        args.mode,
        output_dir=args.output_dir,
        repeats=args.repeats,
        severity_levels=_parse_float_list(args.severity_levels),
    )
    models = _parse_model_list(args.models)
    if models is not None:
        cfg = PositiveSkewStudyConfig(
            runtime=cfg.runtime,
            severity_levels=cfg.severity_levels,
            models=models,
            output_dir=cfg.output_dir,
        )
    results = run_positive_skew_study(cfg)
    save_positive_skew_study(cfg, results, skip_plots=args.skip_plots)
    print(f"Saved outputs to {cfg.output_dir}")
    print("\nProfile summary:\n")
    print(results["profile_summary"].to_string(index=False))
    print("\nPerformance summary:\n")
    print(results["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
