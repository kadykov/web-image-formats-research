#!/usr/bin/env python3
"""Run the merged encode+measure pipeline.

This script loads a study configuration, then encodes and measures
quality for each image in the dataset within an optional time budget.
It produces a quality-results JSON file (same format as the separate
``measure_quality.py`` script) that can be fed directly into analysis
and report generation.

Usage:
    # Run with a time budget
    python3 scripts/run_pipeline.py format-comparison --time-budget 1h

    # Run all images (no budget), save encoded files
    python3 scripts/run_pipeline.py avif-quality-sweep --save-artifacts

    # Dry-run: preview what the pipeline would do
    python3 scripts/run_pipeline.py format-comparison --dry-run

    # List available studies
    python3 scripts/run_pipeline.py --list
"""

import argparse
import json
import sys
from pathlib import Path

from src.pipeline import PipelineRunner, parse_time_budget
from src.study import StudyConfig


def _resolve_study_path(study_arg: str, studies_dir: Path, project_root: Path) -> Path | None:
    """Resolve a study argument to a file path.

    Accepts bare names (``avif-quality-sweep``), names with extension,
    relative paths, or absolute paths.
    """
    study_path = Path(study_arg)

    if study_path.is_absolute():
        return study_path if study_path.exists() else None

    candidate = project_root / study_path
    if candidate.exists():
        return candidate

    if study_path.suffix == "":
        candidate = studies_dir / f"{study_path}.json"
    else:
        candidate = studies_dir / study_path
    if candidate.exists():
        return candidate

    return None


def list_studies(studies_dir: Path) -> None:
    """List available study configurations."""
    print("=" * 70)
    print("Available Studies")
    print("=" * 70)
    print()

    if not studies_dir.exists():
        print(f"Studies directory not found: {studies_dir}")
        return

    study_files = sorted(studies_dir.glob("*.json"))
    if not study_files:
        print("No study configurations found.")
        return

    for study_file in study_files:
        try:
            with open(study_file) as f:
                data = json.load(f)
            sid = data.get("id", study_file.stem)
            name = data.get("name", sid)
            desc = data.get("description", "")
            budget = data.get("time_budget")
            budget_str = f" | budget: {budget}s" if budget else ""
            print(f"  {sid:30s} {name}{budget_str}")
            if desc:
                print(f"  {'':30s} {desc[:70]}")
            print()
        except Exception as e:
            print(f"  {study_file.stem:30s} (error: {e})")


def dry_run(config: StudyConfig) -> None:
    """Preview what the pipeline would do without running."""
    print("=" * 70)
    print(f"DRY RUN: {config.name}")
    print("=" * 70)
    print()
    print(f"Study ID: {config.id}")
    print(f"Dataset: {config.dataset_id}")
    if config.max_images:
        print(f"Max images: {config.max_images}")
    if config.time_budget:
        print(f"Time budget: {config.time_budget}s")
    if config.description:
        print(f"Description: {config.description}")
    print()

    resolutions: list[int | None] = [None]
    if config.resize:
        resolutions = list(config.resize)

    total_per_image = 0
    for enc in config.encoders:
        n_quality = len(enc.quality)
        n_chroma = len(enc.chroma_subsampling) if enc.chroma_subsampling else 1
        n_speed = len(enc.speed) if enc.speed else 1
        n_effort = len(enc.effort) if enc.effort else 1
        n_method = len(enc.method) if enc.method else 1
        n = n_quality * n_chroma * n_speed * n_effort * n_method
        total_per_image += n
        print(
            f"  {enc.format.upper():6s}: {n_quality} quality × "
            f"{n_chroma} chroma × {n_speed} speed × "
            f"{n_effort} effort × {n_method} method = {n} tasks"
        )

    total_per_image *= len(resolutions)
    print()
    print(f"Resolutions: {len(resolutions)}")
    print(f"Tasks per image: {total_per_image}")
    if config.max_images:
        print(f"Max total tasks: {total_per_image * config.max_images}")


def main() -> int:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    studies_dir = project_root / "config" / "studies"

    parser = argparse.ArgumentParser(
        description="Run merged encode+measure pipeline from a study configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s format-comparison --time-budget 1h
  %(prog)s avif-quality-sweep --time-budget 30m --save-artifacts
  %(prog)s format-comparison                    # unlimited (all images)
  %(prog)s --list
""",
    )
    parser.add_argument(
        "study",
        nargs="?",
        help="Study name or path to study JSON config",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available study configurations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what the pipeline would do without running",
    )
    parser.add_argument(
        "--time-budget",
        type=str,
        help="Maximum time to spend (e.g. 3600, 1h, 30m, 1h30m). "
        "Overrides study config time_budget.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save encoded image files to data/encoded/<study>/",
    )
    parser.add_argument(
        "--save-worst-image",
        action="store_true",
        help="Save encoded files for the worst-quality image only "
        "(for use with visual comparison tool)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output path for quality results JSON "
        "(default: data/metrics/<study_id>/quality.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

    if args.list:
        list_studies(studies_dir)
        return 0

    if args.study is None:
        parser.print_help()
        print("\nError: Please provide a study config or use --list")
        return 1

    study_path = _resolve_study_path(args.study, studies_dir, project_root)
    if study_path is None:
        print(f"Error: Study not found: {args.study}")
        print(f"  Looked in: {studies_dir}")
        print("  Use --list to see available studies")
        return 1

    try:
        config = StudyConfig.from_file(study_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading study config: {e}")
        return 1

    if args.dry_run:
        dry_run(config)
        return 0

    # Determine time budget: CLI takes precedence over config
    time_budget: float | None = None
    if args.time_budget is not None:
        try:
            time_budget = parse_time_budget(args.time_budget)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif config.time_budget is not None:
        time_budget = config.time_budget

    # Run pipeline
    runner = PipelineRunner(project_root)

    try:
        results = runner.run(
            config,
            time_budget=time_budget,
            save_artifacts=args.save_artifacts,
            save_worst_image=args.save_worst_image,
            num_workers=args.workers,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    # Save results
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        output_path = project_root / "data" / "metrics" / config.id / "quality.json"

    results.save(output_path)
    print(f"\nResults saved to {output_path}")

    # Summary stats
    successful = [m for m in results.measurements if m.measurement_error is None]
    if successful:
        ssim2_vals = [m.ssimulacra2 for m in successful if m.ssimulacra2 is not None]
        if ssim2_vals:
            print(
                f"  SSIMULACRA2: min={min(ssim2_vals):.2f}, "
                f"max={max(ssim2_vals):.2f}, "
                f"avg={sum(ssim2_vals) / len(ssim2_vals):.2f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
