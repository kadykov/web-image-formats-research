"""Analysis and visualization module.

This module handles data analysis and visualization of compression
efficiency and quality metrics from study quality measurements.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

# Use non-interactive backend for plotting in environments without display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_quality_results(quality_json_path: Path) -> dict[str, Any]:
    """Load quality measurement results from JSON file.

    Args:
        quality_json_path: Path to quality.json file

    Returns:
        Quality results dictionary
    """
    with open(quality_json_path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def create_analysis_dataframe(quality_results: dict[str, Any]) -> pd.DataFrame:
    """Create analysis DataFrame from quality measurements.

    Args:
        quality_results: Quality results dictionary

    Returns:
        DataFrame with all measurements and derived metrics
    """
    df = pd.DataFrame(quality_results["measurements"])

    # Calculate derived metrics
    df["bytes_per_pixel"] = df["file_size"] / (df["width"] * df["height"])
    df["compression_ratio"] = df["source_file_size"] / df["file_size"]
    df["size_reduction_pct"] = (
        (df["source_file_size"] - df["file_size"]) / df["source_file_size"]
    ) * 100

    # Encoder efficiency metrics (bytes per quality score per pixel)
    # Lower is better - fewer bytes needed per pixel to achieve quality
    for metric in ["ssimulacra2", "psnr", "ssim", "butteraugli"]:
        if metric in df.columns:
            # Avoid division by zero or null values
            # For butteraugli (lower is better), we invert: bytes * metric
            if metric == "butteraugli":
                df[f"bytes_per_{metric}_per_pixel"] = np.where(
                    df[metric].notna() & (df[metric] > 0),
                    df["bytes_per_pixel"] * df[metric],
                    np.nan,
                )
            else:
                df[f"bytes_per_{metric}_per_pixel"] = np.where(
                    df[metric].notna() & (df[metric] > 0),
                    df["bytes_per_pixel"] / df[metric],
                    np.nan,
                )

    return df


def compute_statistics(df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """Compute statistical aggregates for quality and efficiency metrics.

    Args:
        df: DataFrame with measurements
        group_by: Columns to group by (e.g., ['format', 'quality', 'chroma_subsampling'])

    Returns:
        DataFrame with mean, percentiles (5, 25, 50, 75, 95) for each metric
    """
    # Metrics to aggregate
    metrics = [
        "ssimulacra2",
        "psnr",
        "ssim",
        "butteraugli",
        "file_size",
        "bytes_per_pixel",
        "compression_ratio",
        "bytes_per_ssimulacra2_per_pixel",
        "bytes_per_psnr_per_pixel",
        "bytes_per_ssim_per_pixel",
        "bytes_per_butteraugli_per_pixel",
    ]

    # Filter to only existing columns
    metrics = [m for m in metrics if m in df.columns]

    # Aggregate functions
    agg_dict = {}
    for metric in metrics:
        agg_dict[metric] = [
            ("mean", "mean"),
            ("p05", lambda x: x.quantile(0.05)),
            ("p25", lambda x: x.quantile(0.25)),
            ("p50", "median"),
            ("p75", lambda x: x.quantile(0.75)),
            ("p95", lambda x: x.quantile(0.95)),
            ("min", "min"),
            ("max", "max"),
        ]

    # Group and aggregate
    stats = df.groupby(group_by, dropna=False).agg(agg_dict)

    # Flatten multi-level columns
    stats.columns = ["_".join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()

    return stats


def determine_varying_parameters(df: pd.DataFrame) -> list[str]:
    """Determine which parameters vary in the dataframe.

    Args:
        df: DataFrame with measurements

    Returns:
        List of parameter columns that have more than one unique value
    """
    candidates = [
        "format",
        "quality",
        "chroma_subsampling",
        "speed",
        "effort",
        "method",
        "resolution",
    ]
    varying = []
    for col in candidates:
        if col in df.columns and df[col].nunique() > 1:
            varying.append(col)
    return varying


def determine_sweep_parameter(df: pd.DataFrame) -> str:
    """Determine which parameter has the longest sweep range for primary x-axis.

    Args:
        df: DataFrame with measurements

    Returns:
        Column name of the parameter with most unique values
    """
    varying = determine_varying_parameters(df)

    if not varying:
        # Default to quality if nothing varies
        return "quality" if "quality" in df.columns else df.columns[0]

    # Count unique values for each varying parameter
    counts = {col: df[col].nunique() for col in varying}

    # Return the parameter with most unique values
    return max(counts, key=lambda k: counts[k])


def determine_secondary_sweep_parameter(df: pd.DataFrame, primary: str) -> str | None:
    """Determine the second longest sweep parameter for grouping.

    This is used for rate-distortion plots where we want to connect points
    along the second-most varied parameter.

    Args:
        df: DataFrame with measurements
        primary: The primary sweep parameter (to exclude)

    Returns:
        Column name of the parameter with second-most unique values, or None
    """
    varying = determine_varying_parameters(df)

    # Remove the primary parameter
    varying = [v for v in varying if v != primary]

    if not varying:
        return None

    # Count unique values for remaining parameters
    counts = {col: df[col].nunique() for col in varying}

    # Return the parameter with most unique values
    return max(counts, key=lambda k: counts[k])


# Metric direction: True = higher is better, False = lower is better
METRIC_DIRECTIONS = {
    "ssimulacra2": True,
    "psnr": True,
    "ssim": True,
    "butteraugli": False,
    "file_size": False,
    "bytes_per_pixel": False,
    "compression_ratio": True,
    "bytes_per_ssimulacra2_per_pixel": False,
    "bytes_per_psnr_per_pixel": False,
    "bytes_per_ssim_per_pixel": False,
    "bytes_per_butteraugli_per_pixel": False,
}

# Marker styles to cycle through
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


def get_worst_percentile_col(metric: str) -> str:
    """Get the appropriate percentile column for worst-case values.

    For "higher is better" metrics, worst = p05 (lowest 5%)
    For "lower is better" metrics, worst = p95 (highest 5%)

    Args:
        metric: Metric name

    Returns:
        Column suffix for the worst percentile
    """
    higher_is_better = METRIC_DIRECTIONS.get(metric, True)
    return "p05" if higher_is_better else "p95"


def plot_quality_metrics(
    stats: pd.DataFrame,
    x_param: str,
    metric: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot mean and worst percentile quality metrics.

    Args:
        stats: Statistics DataFrame
        x_param: Parameter to use as x-axis (can be grouping column or _mean statistic)
        metric: Metric to plot (e.g., 'ssimulacra2')
        output_path: Path to save plot (WebP format)
        title: Optional custom title
    """
    mean_col = f"{metric}_mean"
    worst_col = f"{metric}_{get_worst_percentile_col(metric)}"

    if mean_col not in stats.columns or worst_col not in stats.columns:
        return

    # Check if x_param is a statistic column (ends with _mean) or a grouping column
    x_is_stat = x_param.endswith("_mean")
    if not x_is_stat and x_param not in stats.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by other parameters if they exist
    # Exclude x_param and also any statistics columns
    group_cols = [
        col
        for col in [
            "format",
            "quality",
            "chroma_subsampling",
            "speed",
            "effort",
            "method",
            "resolution",
        ]
        if col in stats.columns and col != x_param and stats[col].nunique() > 1
    ]

    # Determine direction label
    higher_is_better = METRIC_DIRECTIONS.get(metric, True)
    direction_label = "higher is better" if higher_is_better else "lower is better"

    if group_cols:
        groups = stats.groupby(group_cols, dropna=False)

        for idx, (name, group) in enumerate(groups):
            # Create label from group name
            if isinstance(name, tuple):
                label = "_".join(str(n) for n in name if n is not None)
            else:
                label = str(name) if name is not None else "default"

            # Sort by x parameter
            group = group.sort_values(x_param)

            # Select marker and color
            marker = MARKERS[idx % len(MARKERS)]
            color = f"C{idx}"

            # Plot mean (solid line, filled markers)
            ax.plot(
                group[x_param],
                group[mean_col],
                marker=marker,
                linestyle="-",
                color=color,
                label=f"{label} (mean)",
                markersize=7,
            )

            # Plot worst percentile (dashed line, same marker and color, unfilled)
            ax.plot(
                group[x_param],
                group[worst_col],
                marker=marker,
                linestyle="--",
                color=color,
                fillstyle="none",
                label=f"{label} (5% worst)",
                markersize=7,
            )
    else:
        # No grouping, just plot the data
        stats_sorted = stats.sort_values(x_param)
        ax.plot(
            stats_sorted[x_param],
            stats_sorted[mean_col],
            marker="o",
            linestyle="-",
            label="Mean",
            markersize=7,
        )
        ax.plot(
            stats_sorted[x_param],
            stats_sorted[worst_col],
            marker="o",
            linestyle="--",
            fillstyle="none",
            label="5% worst",
            markersize=7,
        )

    # Format x-axis label
    x_label = x_param.replace("_mean", "").replace("_", " ").title()
    if x_is_stat:
        # Add direction for statistics used as x-axis
        x_higher_is_better = METRIC_DIRECTIONS.get(x_param.replace("_mean", ""), True)
        x_direction = "higher is better" if x_higher_is_better else "lower is better"
        x_label = f"{x_label} ({x_direction})"

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(f"{metric.upper()} ({direction_label})", fontsize=11)
    ax.set_title(title or f"{metric.upper()} vs {x_label}", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_rate_distortion(
    stats: pd.DataFrame,
    metric: str,
    grouping_param: str | None,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot quality metric vs bytes_per_pixel (rate-distortion curve).

    Args:
        stats: Statistics DataFrame
        metric: Quality metric to plot (e.g., 'ssimulacra2')
        grouping_param: Parameter to group lines by (e.g., 'format', 'chroma_subsampling')
        output_path: Path to save plot (WebP format)
        title: Optional custom title
    """
    mean_col = f"{metric}_mean"
    worst_col = f"{metric}_{get_worst_percentile_col(metric)}"
    bpp_col = "bytes_per_pixel_mean"

    if (
        mean_col not in stats.columns
        or worst_col not in stats.columns
        or bpp_col not in stats.columns
    ):
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine direction label
    higher_is_better = METRIC_DIRECTIONS.get(metric, True)
    direction_label = "higher is better" if higher_is_better else "lower is better"

    if grouping_param and grouping_param in stats.columns and stats[grouping_param].nunique() > 1:
        # Group by the secondary sweep parameter
        groups = stats.groupby(grouping_param, dropna=False)

        for idx, (name, group) in enumerate(groups):
            # Create label from group name
            label = str(name) if name is not None else "default"

            # Sort by bytes per pixel for proper line connection
            group = group.sort_values(bpp_col)

            # Select marker and color
            marker = MARKERS[idx % len(MARKERS)]
            color = f"C{idx}"

            # Plot mean (solid line, filled markers)
            ax.plot(
                group[bpp_col],
                group[mean_col],
                marker=marker,
                linestyle="-",
                color=color,
                label=f"{label} (mean)",
                markersize=7,
            )

            # Plot worst percentile (dashed line, same marker and color, unfilled)
            ax.plot(
                group[bpp_col],
                group[worst_col],
                marker=marker,
                linestyle="--",
                color=color,
                fillstyle="none",
                label=f"{label} (5% worst)",
                markersize=7,
            )
    else:
        # No grouping, just plot all data as one series
        stats_sorted = stats.sort_values(bpp_col)
        ax.plot(
            stats_sorted[bpp_col],
            stats_sorted[mean_col],
            marker="o",
            linestyle="-",
            label="Mean",
            markersize=7,
        )
        ax.plot(
            stats_sorted[bpp_col],
            stats_sorted[worst_col],
            marker="o",
            linestyle="--",
            fillstyle="none",
            label="5% worst",
            markersize=7,
        )

    ax.set_xlabel("Bytes per Pixel (lower is better)", fontsize=11)
    ax.set_ylabel(f"{metric.upper()} ({direction_label})", fontsize=11)
    ax.set_title(title or f"{metric.upper()} vs Bytes per Pixel", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_efficiency_metrics(
    stats: pd.DataFrame,
    x_param: str,
    efficiency_metric: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot encoder efficiency metrics (bytes per quality per pixel).

    Args:
        stats: Statistics DataFrame
        x_param: Parameter to use as x-axis
        efficiency_metric: Efficiency metric (e.g., 'bytes_per_ssimulacra2_per_pixel')
        output_path: Path to save plot (WebP format)
        title: Optional custom title
    """
    mean_col = f"{efficiency_metric}_mean"
    worst_col = f"{efficiency_metric}_{get_worst_percentile_col(efficiency_metric)}"

    if mean_col not in stats.columns or worst_col not in stats.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by other parameters
    group_cols = [
        col
        for col in [
            "format",
            "quality",
            "chroma_subsampling",
            "speed",
            "effort",
            "method",
            "resolution",
        ]
        if col in stats.columns and col != x_param and stats[col].nunique() > 1
    ]

    # Determine direction label
    higher_is_better = METRIC_DIRECTIONS.get(efficiency_metric, False)
    direction_label = "higher is better" if higher_is_better else "lower is better"

    if group_cols:
        groups = stats.groupby(group_cols, dropna=False)

        for idx, (name, group) in enumerate(groups):
            if isinstance(name, tuple):
                label = "_".join(str(n) for n in name if n is not None)
            else:
                label = str(name) if name is not None else "default"

            group = group.sort_values(x_param)

            # Select marker and color
            marker = MARKERS[idx % len(MARKERS)]
            color = f"C{idx}"

            # Plot mean (solid line, filled markers)
            ax.plot(
                group[x_param],
                group[mean_col],
                marker=marker,
                linestyle="-",
                color=color,
                label=f"{label} (mean)",
                markersize=7,
            )

            # Plot worst percentile (dashed line, same marker and color, unfilled)
            ax.plot(
                group[x_param],
                group[worst_col],
                marker=marker,
                linestyle="--",
                color=color,
                fillstyle="none",
                label=f"{label} (5% worst)",
                markersize=7,
            )
    else:
        stats = stats.sort_values(x_param)
        ax.plot(
            stats[x_param], stats[mean_col], marker="o", linestyle="-", label="Mean", markersize=7
        )
        ax.plot(
            stats[x_param],
            stats[worst_col],
            marker="o",
            linestyle="--",
            fillstyle="none",
            label="5% worst",
            markersize=7,
        )

    ax.set_xlabel(x_param.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(f"{efficiency_metric.replace('_', ' ').title()} ({direction_label})", fontsize=11)
    ax.set_title(
        title
        or f"{efficiency_metric.replace('_', ' ').title()} vs {x_param.replace('_', ' ').title()}",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_bytes_per_pixel(
    stats: pd.DataFrame,
    x_param: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot bytes per pixel with mean, 5th and 95th percentiles.

    Args:
        stats: Statistics DataFrame
        x_param: Parameter to use as x-axis
        output_path: Path to save plot (WebP format)
        title: Optional custom title
    """
    mean_col = "bytes_per_pixel_mean"
    p05_col = "bytes_per_pixel_p05"
    p95_col = "bytes_per_pixel_p95"

    if mean_col not in stats.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by other parameters
    group_cols = [
        col
        for col in [
            "format",
            "quality",
            "chroma_subsampling",
            "speed",
            "effort",
            "method",
            "resolution",
        ]
        if col in stats.columns and col != x_param and stats[col].nunique() > 1
    ]

    if group_cols:
        groups = stats.groupby(group_cols, dropna=False)

        for idx, (name, group) in enumerate(groups):
            if isinstance(name, tuple):
                label = "_".join(str(n) for n in name if n is not None)
            else:
                label = str(name) if name is not None else "default"

            group = group.sort_values(x_param)

            # Select marker and color
            marker = MARKERS[idx % len(MARKERS)]
            color = f"C{idx}"

            # Plot mean (solid line, filled markers)
            ax.plot(
                group[x_param],
                group[mean_col],
                marker=marker,
                linestyle="-",
                color=color,
                label=f"{label} (mean)",
                markersize=7,
            )

            # Plot 5th percentile (dashed line, unfilled)
            if p05_col in stats.columns:
                ax.plot(
                    group[x_param],
                    group[p05_col],
                    marker=marker,
                    linestyle="--",
                    color=color,
                    fillstyle="none",
                    label=f"{label} (5% smallest)",
                    markersize=7,
                )

            # Plot 95th percentile (dotted line, unfilled)
            if p95_col in stats.columns:
                ax.plot(
                    group[x_param],
                    group[p95_col],
                    marker=marker,
                    linestyle=":",
                    color=color,
                    fillstyle="none",
                    label=f"{label} (95% largest)",
                    markersize=7,
                )
    else:
        stats = stats.sort_values(x_param)
        ax.plot(
            stats[x_param], stats[mean_col], marker="o", linestyle="-", label="Mean", markersize=7
        )
        if p05_col in stats.columns:
            ax.plot(
                stats[x_param],
                stats[p05_col],
                marker="o",
                linestyle="--",
                fillstyle="none",
                label="5% smallest",
                markersize=7,
            )
        if p95_col in stats.columns:
            ax.plot(
                stats[x_param],
                stats[p95_col],
                marker="o",
                linestyle=":",
                fillstyle="none",
                label="95% largest",
                markersize=7,
            )

    ax.set_xlabel(x_param.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel("Bytes per Pixel (lower is better)", fontsize=11)
    ax.set_title(
        title or f"Bytes per Pixel vs {x_param.replace('_', ' ').title()}",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def analyze_study(quality_json_path: Path, output_dir: Path) -> None:
    """Run complete analysis for a study.

    Generates:
    - CSV with statistics
    - Quality metric plots vs sweep parameter (mean + 5% worst)
    - Quality metric plots vs bytes_per_pixel (rate-distortion curves)
    - Bytes per pixel plots (mean + 5% + 95%)
    - Efficiency metric plots vs sweep parameter (mean + 5% worst)

    Args:
        quality_json_path: Path to quality.json file
        output_dir: Directory to save analysis outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    quality_results = load_quality_results(quality_json_path)
    df = create_analysis_dataframe(quality_results)

    # Determine grouping columns (parameters that vary)
    varying = determine_varying_parameters(df)

    if not varying:
        varying = ["format"]  # Default fallback

    # Compute statistics
    stats = compute_statistics(df, varying)

    # Save statistics to CSV
    study_id = quality_results.get("study_id", "unknown")
    stats_path = output_dir / f"{study_id}_statistics.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Statistics saved to: {stats_path}")

    # Determine sweep parameter for plots
    x_param = determine_sweep_parameter(df)
    print(f"Using '{x_param}' as primary x-axis for plots")

    # Determine secondary sweep parameter for rate-distortion plots
    secondary_param = determine_secondary_sweep_parameter(df, x_param)
    if secondary_param:
        print(f"Using '{secondary_param}' as secondary axis for rate-distortion plots")

    # Generate quality metric plots vs sweep parameter
    quality_metrics = ["ssimulacra2", "psnr", "ssim", "butteraugli"]
    for metric in quality_metrics:
        if f"{metric}_mean" in stats.columns:
            plot_path = output_dir / f"{study_id}_{metric}_vs_{x_param}.svg"
            plot_quality_metrics(stats, x_param, metric, plot_path)
            print(f"Quality plot saved: {plot_path}")

    # Generate quality metric plots vs bytes_per_pixel (rate-distortion)
    if "bytes_per_pixel_mean" in stats.columns:
        for metric in quality_metrics:
            if f"{metric}_mean" in stats.columns:
                plot_path = output_dir / f"{study_id}_{metric}_vs_bytes_per_pixel.svg"
                plot_rate_distortion(stats, metric, secondary_param, plot_path)
                print(f"Rate-distortion plot saved: {plot_path}")

    # Generate bytes per pixel plot (with p05 and p95)
    if "bytes_per_pixel_mean" in stats.columns:
        plot_path = output_dir / f"{study_id}_bytes_per_pixel_vs_{x_param}.svg"
        plot_bytes_per_pixel(stats, x_param, plot_path)
        print(f"Bytes per pixel plot saved: {plot_path}")

    # Generate efficiency metric plots vs sweep parameter
    efficiency_metrics = [
        "bytes_per_ssimulacra2_per_pixel",
        "bytes_per_psnr_per_pixel",
        "bytes_per_ssim_per_pixel",
        "bytes_per_butteraugli_per_pixel",
    ]
    for metric in efficiency_metrics:
        if f"{metric}_mean" in stats.columns:
            plot_path = output_dir / f"{study_id}_{metric}_vs_{x_param}.svg"
            plot_efficiency_metrics(stats, x_param, metric, plot_path)
            print(f"Efficiency plot saved: {plot_path}")
