from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from dataclasses import asdict
from io import StringIO
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cairo.synthetic import paper_benchmark as pb  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the monotone-distortion study with per-repeat isolation."
    )
    parser.add_argument("--mode", choices=["smoke", "debug", "full"], default="smoke")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "benchmarks" / "monotone_distortion_study",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--child-strength", type=float, default=None)
    parser.add_argument("--child-repeat", type=int, default=None)
    parser.add_argument("--child-seed", type=int, default=None)
    return parser


def _scenario_for_strength(strength: float) -> pb.ScenarioSpec:
    return pb.ScenarioSpec(
        name=f"Monotone distortion (α={strength:.1f})",
        kind="monotone_distortion",
        generator=lambda seed, local_cfg, value=strength: pb.gen_monotone_distortion(
            seed, local_cfg, strength=value
        ),
    )


def _run_child(mode: str, repeat_override: int | None, strength: float, repeat: int, run_seed: int) -> None:
    cfg = pb.RuntimeConfig.from_mode(mode, repeats=repeat_override)
    scenario = _scenario_for_strength(strength)
    with contextlib.redirect_stdout(sys.stderr):
        rows, _ = pb.run_one_repeat(
            scenario,
            cfg,
            repeat,
            run_seed,
            selected_models=set(pb.FAIR_NEURAL_STAGE2_MODELS),
            include_uncalibrated_cairo=True,
        )
    for row in rows:
        row["DistortionStrength"] = strength
    print(pd.DataFrame(rows).to_json(orient="records"))


def _run_parent(mode: str, repeats: int | None, output_dir: Path, resume: bool) -> None:
    cfg = pb.RuntimeConfig.from_mode(mode, output_dir=output_dir, repeats=repeats)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    for strength in cfg.monotone_strengths:
        for repeat in range(cfg.n_repeats):
            chunk_path = chunks_dir / f"strength_{strength:.1f}_repeat_{repeat + 1}.csv"
            if resume and chunk_path.exists():
                frame = pd.read_csv(chunk_path)
                all_frames.append(frame)
                print(f"[resume] {chunk_path.name}", flush=True)
                continue

            run_seed = cfg.seed + 11000 * repeat + int(strength * 100)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                mode,
                "--output-dir",
                str(output_dir),
                "--child-strength",
                str(strength),
                "--child-repeat",
                str(repeat),
                "--child-seed",
                str(run_seed),
            ]
            if repeats is not None:
                cmd.extend(["--repeats", str(repeats)])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stderr:
                print(result.stderr, end="")
            if result.returncode != 0:
                if result.stdout:
                    print(result.stdout, end="")
                raise RuntimeError(f"chunk failed: strength={strength} repeat={repeat + 1}")

            frame = pd.read_json(StringIO(result.stdout))
            frame.to_csv(chunk_path, index=False)
            all_frames.append(frame)
            partial = pd.concat(all_frames, ignore_index=True)
            partial.to_csv(output_dir / "monotone_distortion_runs.partial.csv", index=False)
            print(f"[saved] {chunk_path.name}", flush=True)

    runs = pd.concat(all_frames, ignore_index=True)
    summary = pb.summarize_with_ci(runs, ["DistortionStrength", "Model"])
    pb.ensure_dir(cfg.output_dir)
    pb.write_json(cfg.output_dir / "config.json", {**asdict(cfg), "output_dir": str(cfg.output_dir)})
    pb.write_json(cfg.output_dir / "availability.json", pb.available_components())
    pb.write_csv(cfg.output_dir / "monotone_distortion_runs.csv", runs)
    pb.write_csv(cfg.output_dir / "monotone_distortion_summary.csv", summary)
    pb.plot_sweep(summary, cfg.output_dir, "monotone_distortion_sweep", "DistortionStrength")
    pb.plot_sweep(
        summary,
        cfg.output_dir,
        "monotone_stage2_comparison",
        "DistortionStrength",
        models=pb.FAIR_NEURAL_STAGE2_MODELS,
    )
    pb.plot_monotone_story(runs, summary, cfg.output_dir)
    print(f"Saved outputs to {cfg.output_dir}")


def main() -> None:
    args = build_parser().parse_args()
    if args.child_strength is not None:
        if args.child_repeat is None or args.child_seed is None:
            raise ValueError("Child mode requires repeat and seed")
        _run_child(args.mode, args.repeats, args.child_strength, args.child_repeat, args.child_seed)
        return

    _run_parent(args.mode, args.repeats, args.output_dir, resume=not args.no_resume)


if __name__ == "__main__":
    main()
