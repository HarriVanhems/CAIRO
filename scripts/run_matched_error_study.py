from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cairo.synthetic.matched_error_study import (  # noqa: E402
    MatchedErrorStudyConfig,
    run_matched_error_study,
    save_matched_error_study,
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
        description="Run the matched additive-vs-multiplicative CAIRO study."
    )
    parser.add_argument("--mode", choices=["smoke", "debug", "full"], default="smoke")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "benchmarks" / "matched_error_study",
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
    parser.add_argument(
        "--visualization-severity",
        type=float,
        default=0.75,
        help="Severity level used for the 1D visualization figure.",
    )
    parser.add_argument(
        "--visualization-samples",
        type=int,
        default=1200,
        help="Number of samples used to fit the 1D visualization experiment.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = MatchedErrorStudyConfig.from_mode(
        args.mode,
        output_dir=args.output_dir,
        repeats=args.repeats,
        severity_levels=_parse_float_list(args.severity_levels),
        visualization_severity=args.visualization_severity,
        visualization_samples=args.visualization_samples,
    )
    models = _parse_model_list(args.models)
    if models is not None:
        cfg = MatchedErrorStudyConfig(
            runtime=cfg.runtime,
            severity_levels=cfg.severity_levels,
            models=models,
            output_dir=cfg.output_dir,
            visualization_severity=cfg.visualization_severity,
            visualization_samples=cfg.visualization_samples,
        )
    results = run_matched_error_study(cfg)
    save_matched_error_study(cfg, results, skip_plots=args.skip_plots)
    print(f"Saved outputs to {cfg.output_dir}")
    print("\nPerformance summary:\n")
    print(results["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
