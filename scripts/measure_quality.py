#!/usr/bin/env python3
"""Script to measure quality metrics for encoded images.

This script loads encoding results from a JSON file produced by the
encoding stage and measures quality metrics (SSIMULACRA2, Butteraugli,
PSNR, SSIM) for all encoded images against their original references.

Usage:
    python3 scripts/measure_quality.py data/encoded/avif-quality-sweep/results.json
    python3 scripts/measure_quality.py data/encoded/format-comparison/results.json --workers 8
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quality import EncodingResults, QualityMeasurementRunner  # noqa: E402


def main() -> int:
    """Main entry point for the quality measurement script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Measure quality metrics for encoded images.",
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to encoding results JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output path for quality results JSON (default: data/metrics/<study_id>/quality.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

    # Resolve results file path (support both absolute and relative paths)
    results_path = args.results_file
    if not results_path.is_absolute():
        results_path = project_root / results_path

    if not results_path.exists():
        print(f"Error: Encoding results file not found: {results_path}")
        return 1

    # Load encoding results
    try:
        encoding_results = EncodingResults.from_file(results_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading encoding results: {e}")
        return 1

    # Run quality measurements
    runner = QualityMeasurementRunner(project_root)

    try:
        quality_results = runner.run(encoding_results, num_workers=args.workers)
    except Exception as e:
        print(f"Error running quality measurements: {e}")
        return 1

    # Save results
    if args.output is None:
        output_path = project_root / "data" / "metrics" / encoding_results.study_id / "quality.json"
    else:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_root / output_path

    quality_results.save(output_path)
    print("\nQuality measurements complete.")
    print(f"  Study: {quality_results.study_name}")
    print(f"  Measurements: {len(quality_results.measurements)}")
    print(f"  Results: {output_path}")

    # Show summary statistics
    successful = [m for m in quality_results.measurements if m.measurement_error is None]
    if successful:
        print("\nSummary:")
        ssim2_values = [m.ssimulacra2 for m in successful if m.ssimulacra2 is not None]
        if ssim2_values:
            print(
                f"  SSIMULACRA2: min={min(ssim2_values):.2f}, "
                f"max={max(ssim2_values):.2f}, "
                f"avg={sum(ssim2_values) / len(ssim2_values):.2f}"
            )

        psnr_values = [m.psnr for m in successful if m.psnr is not None]
        if psnr_values:
            print(
                f"  PSNR: min={min(psnr_values):.2f} dB, "
                f"max={max(psnr_values):.2f} dB, "
                f"avg={sum(psnr_values) / len(psnr_values):.2f} dB"
            )

        ssim_values = [m.ssim for m in successful if m.ssim is not None]
        if ssim_values:
            print(
                f"  SSIM: min={min(ssim_values):.4f}, "
                f"max={max(ssim_values):.4f}, "
                f"avg={sum(ssim_values) / len(ssim_values):.4f}"
            )

        butteraugli_values = [m.butteraugli for m in successful if m.butteraugli is not None]
        if butteraugli_values:
            print(
                f"  Butteraugli: min={min(butteraugli_values):.4f}, "
                f"max={max(butteraugli_values):.4f}, "
                f"avg={sum(butteraugli_values) / len(butteraugli_values):.4f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
