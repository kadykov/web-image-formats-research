#!/usr/bin/env python3
"""Script to run an encoding study.

This script loads a study configuration from a JSON file,
executes all encoding tasks (with optional preprocessing),
and writes a results JSON file for downstream quality measurement.

Usage:
    python3 scripts/encode_images.py avif-quality-sweep
    python3 scripts/encode_images.py format-comparison --dry-run
    python3 scripts/encode_images.py --list
"""

import argparse
import sys
from pathlib import Path

from src.study import StudyConfig, StudyRunner


def _resolve_study_path(study_arg: str, studies_dir: Path, project_root: Path) -> Path | None:
    """Resolve a study argument to an actual file path.

    Accepts any of these forms:
    - A bare name: ``avif-quality-sweep`` → ``config/studies/avif-quality-sweep.json``
    - A name with extension: ``avif-quality-sweep.json``
    - A relative path: ``config/studies/avif-quality-sweep.json``
    - An absolute path: ``/full/path/to/study.json``

    Args:
        study_arg: The study argument from the CLI
        studies_dir: Default directory containing study JSON files
        project_root: Project root directory

    Returns:
        Resolved Path if found, None otherwise
    """
    study_path = Path(study_arg)

    # Absolute path — use as-is
    if study_path.is_absolute():
        return study_path if study_path.exists() else None

    # Try as a relative path from project root (e.g. config/studies/foo.json)
    candidate = project_root / study_path
    if candidate.exists():
        return candidate

    # Try inside studies_dir by name (e.g. "avif-quality-sweep" or "avif-quality-sweep.json")
    if study_path.suffix == "":
        candidate = studies_dir / f"{study_path}.json"
    else:
        candidate = studies_dir / study_path
    if candidate.exists():
        return candidate

    return None


def list_studies(studies_dir: Path) -> None:
    """List all study configuration files.

    Args:
        studies_dir: Directory containing study JSON files
    """
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
            config = StudyConfig.from_file(study_file)
            print(f"  {study_file.name}")
            print(f"    ID: {config.id}")
            print(f"    Name: {config.name}")
            if config.description:
                print(f"    Description: {config.description}")
            print(f"    Dataset: {config.dataset_id}", end="")
            if config.max_images:
                print(f" (max {config.max_images} images)", end="")
            print()

            encoder_summary = []
            for enc in config.encoders:
                parts = [enc.format.upper()]
                parts.append(f"{len(enc.quality)} quality levels")
                if enc.chroma_subsampling:
                    parts.append(f"{len(enc.chroma_subsampling)} subsampling modes")
                if enc.speed:
                    parts.append(f"{len(enc.speed)} speed settings")
                encoder_summary.append(", ".join(parts))
            print(f"    Encoders: {'; '.join(encoder_summary)}")

            if config.resize:
                print(f"    Resolutions: {config.resize}")
            print()
        except Exception as e:
            print(f"  {study_file.name}: ERROR - {e}")
            print()


def dry_run(config: StudyConfig) -> None:
    """Show what the study would do without executing.

    Args:
        config: Study configuration
    """
    print("=" * 70)
    print(f"DRY RUN: {config.name}")
    print("=" * 70)
    print()
    print(f"Study ID: {config.id}")
    print(f"Dataset: {config.dataset_id}")
    if config.max_images:
        print(f"Max images: {config.max_images}")
    if config.description:
        print(f"Description: {config.description}")
    print()

    resolutions: list[int | None] = [None]
    if config.resize:
        resolutions = [*config.resize]  # type: ignore[list-item]

    total_tasks = 0
    for resolution in resolutions:
        res_label = f"r{resolution}" if resolution else "original"
        print(f"Resolution: {res_label}")

        for enc in config.encoders:
            chroma_count = len(enc.chroma_subsampling) if enc.chroma_subsampling else 1
            speed_count = len(enc.speed) if enc.speed else 1
            quality_count = len(enc.quality)
            n_per_image = quality_count * chroma_count * speed_count
            print(f"  {enc.format.upper()}:")
            print(f"    Quality levels: {enc.quality}")
            if enc.chroma_subsampling:
                print(f"    Chroma subsampling: {enc.chroma_subsampling}")
            if enc.speed:
                print(f"    Speed settings: {enc.speed}")
            print(f"    → {n_per_image} encodings per image")
            total_tasks += n_per_image

        print()

    if config.max_images:
        estimate = total_tasks * config.max_images
        print(
            f"Total: ~{total_tasks} encodings/image × "
            f"{config.max_images} images × "
            f"{len(resolutions)} resolution(s) = "
            f"~{estimate * len(resolutions)} encoding tasks"
        )
    else:
        print(
            f"Total: ~{total_tasks} encodings/image × "
            f"(all images) × {len(resolutions)} resolution(s)"
        )


def run_study(study_path: Path, project_root: Path, output_path: Path | None) -> int:
    """Run an encoding study.

    Args:
        study_path: Path to the study JSON file
        project_root: Project root directory
        output_path: Optional override for results output path

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        config = StudyConfig.from_file(study_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading study config: {e}")
        return 1

    runner = StudyRunner(project_root)

    try:
        results = runner.run(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error running study: {e}")
        return 1

    # Save results
    if output_path is None:
        output_path = project_root / "data" / "encoded" / config.id / "results.json"

    results.save(output_path)
    print(f"\nStudy '{config.name}' complete.")
    print(f"  Encodings: {len(results.records)}")
    print(f"  Results: {output_path}")
    return 0


def main() -> int:
    """Main entry point for the encoding script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    project_root = Path(__file__).parent.parent
    studies_dir = project_root / "config" / "studies"

    parser = argparse.ArgumentParser(
        description="Run an encoding study from a JSON configuration file.",
    )
    parser.add_argument(
        "study",
        nargs="?",
        help="Path to study JSON configuration file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available study configurations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what the study would do without encoding",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output path for results JSON",
    )

    args = parser.parse_args()

    if args.list:
        list_studies(studies_dir)
        return 0

    if args.study is None:
        parser.print_help()
        print("\nError: Please provide a study config file or use --list")
        return 1

    study_path = _resolve_study_path(args.study, studies_dir, project_root)

    if study_path is None:
        print(f"Error: Study config not found: {args.study}")
        print(f"  Looked in: {studies_dir}")
        print("  Use --list to see available studies")
        return 1

    if args.dry_run:
        try:
            config = StudyConfig.from_file(study_path)
            dry_run(config)
            return 0
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return 1

    return run_study(study_path, project_root, args.output)


if __name__ == "__main__":
    sys.exit(main())
