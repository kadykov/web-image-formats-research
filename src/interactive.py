"""Interactive visualization module using Plotly.

This module generates interactive HTML figures from study quality
measurements, suitable for embedding in static web reports.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.analysis import (
    METRIC_DIRECTIONS,
    compute_statistics,
    create_analysis_dataframe,
    determine_secondary_sweep_parameter,
    determine_sweep_parameter,
    determine_varying_parameters,
    get_worst_percentile_col,
    load_quality_results,
)

# Consistent color palette for formats
FORMAT_COLORS: dict[str, str] = {
    "jpeg": "#e377c2",
    "webp": "#2ca02c",
    "avif": "#1f77b4",
    "jxl": "#ff7f0e",
}

# Human-readable metric labels
METRIC_LABELS: dict[str, str] = {
    "ssimulacra2": "SSIMULACRA2",
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "butteraugli": "Butteraugli",
    "file_size": "File Size (bytes)",
    "bytes_per_pixel": "Bytes per Pixel",
    "encoding_time": "Encoding Time (seconds)",
    "encoding_time_per_pixel": "Encoding Time per Pixel (seconds)",
    "compression_ratio": "Compression Ratio",
    "bytes_per_ssimulacra2_per_pixel": "Bytes per SSIMULACRA2 per Pixel",
    "bytes_per_psnr_per_pixel": "Bytes per PSNR per Pixel",
    "bytes_per_ssim_per_pixel": "Bytes per SSIM per Pixel",
    "bytes_per_butteraugli_per_pixel": "Bytes per Butteraugli per Pixel",
}

# Marker symbols to cycle through (Plotly marker names)
PLOTLY_MARKERS: list[str] = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "cross",
    "x",
    "star",
    "hexagon",
    "pentagon",
]


def _metric_label(metric: str) -> str:
    """Get human-readable label for a metric."""
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _direction_label(metric: str) -> str:
    """Get direction indicator for a metric."""
    higher_is_better = METRIC_DIRECTIONS.get(metric, True)
    return "higher is better" if higher_is_better else "lower is better"


def _axis_label(metric: str) -> str:
    """Get full axis label with direction."""
    return f"{_metric_label(metric)} ({_direction_label(metric)})"


def _group_color(group_name: str, idx: int) -> str:
    """Get color for a group, using format colors if applicable."""
    name_lower = str(group_name).lower()
    if name_lower in FORMAT_COLORS:
        return FORMAT_COLORS[name_lower]
    # Default plotly color cycle
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return colors[idx % len(colors)]


def _build_hover_text(
    group: pd.DataFrame,
    x_param: str,
    y_metric: str,
    stat_suffix: str,
    extra_params: list[str] | None = None,
) -> list[str]:
    """Build custom hover text for each point in a trace.

    Shows the encoding parameters and metric values for each data point.

    Args:
        group: DataFrame subset for one group
        x_param: X-axis parameter name
        y_metric: Y-axis metric name
        stat_suffix: Statistics suffix (e.g., 'mean', 'p05')
        extra_params: Additional parameter columns to include in hover

    Returns:
        List of hover text strings, one per data point
    """
    texts = []
    y_col = f"{y_metric}_{stat_suffix}"

    for _, row in group.iterrows():
        parts = []

        # Show all encoding parameters that exist in the data
        param_cols = [
            "format",
            "quality",
            "chroma_subsampling",
            "speed",
            "effort",
            "method",
            "resolution",
        ]
        for col in param_cols:
            if col in group.columns and pd.notna(row.get(col)):
                label = col.replace("_", " ").title()
                parts.append(f"{label}: {row[col]}")

        # Show the metric values
        parts.append("---")  # separator

        # X-axis value (if it's a stat column)
        if x_param.endswith("_mean"):
            base_metric = x_param.replace("_mean", "")
            x_label = _metric_label(base_metric)
            parts.append(f"{x_label} (mean): {row[x_param]:.4g}")
        else:
            parts.append(f"{x_param.replace('_', ' ').title()}: {row[x_param]}")

        # Y-axis value
        y_label = _metric_label(y_metric)
        parts.append(f"{y_label} ({stat_suffix}): {row[y_col]:.4g}")

        # Show additional useful stats if available
        if "bytes_per_pixel_mean" in group.columns and x_param != "bytes_per_pixel_mean":
            parts.append(f"Bytes/Pixel (mean): {row['bytes_per_pixel_mean']:.4g}")
        if "file_size_mean" in group.columns:
            size_kb = row["file_size_mean"] / 1024
            parts.append(f"File Size (mean): {size_kb:.1f} KB")
        if "compression_ratio_mean" in group.columns:
            parts.append(f"Compression Ratio (mean): {row['compression_ratio_mean']:.2f}x")

        if extra_params:
            for param in extra_params:
                if param in group.columns and pd.notna(row.get(param)):
                    parts.append(f"{param.replace('_', ' ').title()}: {row[param]}")

        texts.append("<br>".join(parts))

    return texts


def _create_figure_layout(
    title: str,
    x_label: str,
    y_label: str,
) -> dict[str, Any]:
    """Create standard Plotly layout settings.

    Args:
        title: Figure title
        x_label: X-axis label
        y_label: Y-axis label

    Returns:
        Layout dictionary for go.Figure
    """
    return {
        "title": {"text": title, "font": {"size": 16}},
        "xaxis": {
            "title": {"text": x_label, "font": {"size": 13}},
            "gridcolor": "rgba(128, 128, 128, 0.2)",
        },
        "yaxis": {
            "title": {"text": y_label, "font": {"size": 13}},
            "gridcolor": "rgba(128, 128, 128, 0.2)",
        },
        "legend": {
            "font": {"size": 11},
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        "hovermode": "closest",
        "plot_bgcolor": "white",
        "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
    }


def plot_quality_vs_param(
    stats: pd.DataFrame,
    x_param: str,
    metric: str,
    title: str | None = None,
) -> go.Figure:
    """Create interactive quality metric vs parameter plot.

    Args:
        stats: Statistics DataFrame from compute_statistics
        x_param: Parameter for x-axis (grouping column or _mean statistic)
        metric: Quality metric to plot (e.g., 'ssimulacra2')
        title: Optional custom title

    Returns:
        Plotly Figure object
    """
    mean_col = f"{metric}_mean"
    worst_col = f"{metric}_{get_worst_percentile_col(metric)}"

    if mean_col not in stats.columns or worst_col not in stats.columns:
        return go.Figure()

    x_is_stat = x_param.endswith("_mean")
    if not x_is_stat and x_param not in stats.columns:
        return go.Figure()

    # Determine x-axis label
    if x_is_stat:
        x_label = _axis_label(x_param.replace("_mean", ""))
    else:
        x_label = x_param.replace("_", " ").title()

    y_label = _axis_label(metric)
    fig_title = title or f"{_metric_label(metric)} vs {x_label.split(' (')[0]}"
    fig = go.Figure(layout=_create_figure_layout(fig_title, x_label, y_label))

    # Find grouping columns
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

    worst_suffix = get_worst_percentile_col(metric)

    if group_cols:
        groups = stats.groupby(group_cols, dropna=False)
        for idx, (name, group) in enumerate(groups):
            label = (
                "_".join(str(n) for n in name if n is not None)
                if isinstance(name, tuple)
                else str(name)
                if name is not None
                else "default"
            )
            group = group.sort_values(x_param)
            color = _group_color(label, idx)
            marker = PLOTLY_MARKERS[idx % len(PLOTLY_MARKERS)]

            # Mean trace
            fig.add_trace(
                go.Scatter(
                    x=group[x_param],
                    y=group[mean_col],
                    mode="lines+markers",
                    name=f"{label} (mean)",
                    marker={"symbol": marker, "size": 8, "color": color},
                    line={"color": color, "width": 2},
                    hovertext=_build_hover_text(group, x_param, metric, "mean"),
                    hoverinfo="text",
                )
            )

            # Worst percentile trace
            fig.add_trace(
                go.Scatter(
                    x=group[x_param],
                    y=group[worst_col],
                    mode="lines+markers",
                    name=f"{label} (5% worst)",
                    marker={
                        "symbol": f"{marker}-open",
                        "size": 8,
                        "color": color,
                    },
                    line={"color": color, "width": 1.5, "dash": "dash"},
                    hovertext=_build_hover_text(group, x_param, metric, worst_suffix),
                    hoverinfo="text",
                )
            )
    else:
        stats_sorted = stats.sort_values(x_param)

        fig.add_trace(
            go.Scatter(
                x=stats_sorted[x_param],
                y=stats_sorted[mean_col],
                mode="lines+markers",
                name="Mean",
                marker={"symbol": "circle", "size": 8},
                line={"width": 2},
                hovertext=_build_hover_text(stats_sorted, x_param, metric, "mean"),
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=stats_sorted[x_param],
                y=stats_sorted[worst_col],
                mode="lines+markers",
                name="5% worst",
                marker={"symbol": "circle-open", "size": 8},
                line={"width": 1.5, "dash": "dash"},
                hovertext=_build_hover_text(stats_sorted, x_param, metric, worst_suffix),
                hoverinfo="text",
            )
        )

    return fig


def plot_rate_distortion(
    stats: pd.DataFrame,
    metric: str,
    grouping_param: str | None,
    title: str | None = None,
    primary_param: str | None = None,
) -> go.Figure:
    """Create interactive rate-distortion plot (quality vs bytes per pixel).

    Args:
        stats: Statistics DataFrame from compute_statistics
        metric: Quality metric (e.g., 'ssimulacra2')
        grouping_param: Parameter to group lines by (e.g., 'format')
        title: Optional custom title
        primary_param: Primary sweep parameter to sort points by (e.g., 'quality', 'speed').
            When provided, points within each group are connected in the order of this
            parameter rather than by bytes_per_pixel, giving a meaningful line for
            non-monotonic sweeps such as speed or effort settings.

    Returns:
        Plotly Figure object
    """
    mean_col = f"{metric}_mean"
    worst_col = f"{metric}_{get_worst_percentile_col(metric)}"
    bpp_col = "bytes_per_pixel_mean"

    if mean_col not in stats.columns or worst_col not in stats.columns:
        return go.Figure()
    if bpp_col not in stats.columns:
        return go.Figure()

    x_label = "Bytes per Pixel (lower is better)"
    y_label = _axis_label(metric)
    fig_title = title or f"{_metric_label(metric)} vs Bytes per Pixel"
    fig = go.Figure(layout=_create_figure_layout(fig_title, x_label, y_label))

    worst_suffix = get_worst_percentile_col(metric)

    if grouping_param and grouping_param in stats.columns and stats[grouping_param].nunique() > 1:
        groups = stats.groupby(grouping_param, dropna=False)
        for idx, (name, group) in enumerate(groups):
            label = str(name) if name is not None else "default"
            sort_col = primary_param if (primary_param and primary_param in group.columns) else bpp_col
            group = group.sort_values(sort_col)
            color = _group_color(label, idx)
            marker = PLOTLY_MARKERS[idx % len(PLOTLY_MARKERS)]

            fig.add_trace(
                go.Scatter(
                    x=group[bpp_col],
                    y=group[mean_col],
                    mode="lines+markers",
                    name=f"{label} (mean)",
                    marker={"symbol": marker, "size": 8, "color": color},
                    line={"color": color, "width": 2},
                    hovertext=_build_hover_text(group, bpp_col, metric, "mean"),
                    hoverinfo="text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=group[bpp_col],
                    y=group[worst_col],
                    mode="lines+markers",
                    name=f"{label} (5% worst)",
                    marker={
                        "symbol": f"{marker}-open",
                        "size": 8,
                        "color": color,
                    },
                    line={"color": color, "width": 1.5, "dash": "dash"},
                    hovertext=_build_hover_text(group, bpp_col, metric, worst_suffix),
                    hoverinfo="text",
                )
            )
    else:
        sort_col = primary_param if (primary_param and primary_param in stats.columns) else bpp_col
        stats_sorted = stats.sort_values(sort_col)
        fig.add_trace(
            go.Scatter(
                x=stats_sorted[bpp_col],
                y=stats_sorted[mean_col],
                mode="lines+markers",
                name="Mean",
                marker={"symbol": "circle", "size": 8},
                line={"width": 2},
                hovertext=_build_hover_text(stats_sorted, bpp_col, metric, "mean"),
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=stats_sorted[bpp_col],
                y=stats_sorted[worst_col],
                mode="lines+markers",
                name="5% worst",
                marker={"symbol": "circle-open", "size": 8},
                line={"width": 1.5, "dash": "dash"},
                hovertext=_build_hover_text(stats_sorted, bpp_col, metric, worst_suffix),
                hoverinfo="text",
            )
        )

    return fig


def plot_bytes_per_pixel(
    stats: pd.DataFrame,
    x_param: str,
    title: str | None = None,
) -> go.Figure:
    """Create interactive bytes per pixel plot with percentile bands.

    Args:
        stats: Statistics DataFrame
        x_param: Parameter for x-axis
        title: Optional custom title

    Returns:
        Plotly Figure object
    """
    mean_col = "bytes_per_pixel_mean"
    p05_col = "bytes_per_pixel_p05"
    p95_col = "bytes_per_pixel_p95"

    if mean_col not in stats.columns:
        return go.Figure()

    x_label = x_param.replace("_", " ").title()
    y_label = "Bytes per Pixel (lower is better)"
    fig_title = title or f"Bytes per Pixel vs {x_label}"
    fig = go.Figure(layout=_create_figure_layout(fig_title, x_label, y_label))

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
            label = (
                "_".join(str(n) for n in name if n is not None)
                if isinstance(name, tuple)
                else str(name)
                if name is not None
                else "default"
            )
            group = group.sort_values(x_param)
            color = _group_color(label, idx)
            marker = PLOTLY_MARKERS[idx % len(PLOTLY_MARKERS)]

            fig.add_trace(
                go.Scatter(
                    x=group[x_param],
                    y=group[mean_col],
                    mode="lines+markers",
                    name=f"{label} (mean)",
                    marker={"symbol": marker, "size": 8, "color": color},
                    line={"color": color, "width": 2},
                    hovertext=_build_hover_text(group, x_param, "bytes_per_pixel", "mean"),
                    hoverinfo="text",
                )
            )
            if p05_col in stats.columns:
                fig.add_trace(
                    go.Scatter(
                        x=group[x_param],
                        y=group[p05_col],
                        mode="lines+markers",
                        name=f"{label} (5% smallest)",
                        marker={
                            "symbol": f"{marker}-open",
                            "size": 8,
                            "color": color,
                        },
                        line={"color": color, "width": 1, "dash": "dash"},
                        hovertext=_build_hover_text(group, x_param, "bytes_per_pixel", "p05"),
                        hoverinfo="text",
                    )
                )
            if p95_col in stats.columns:
                fig.add_trace(
                    go.Scatter(
                        x=group[x_param],
                        y=group[p95_col],
                        mode="lines+markers",
                        name=f"{label} (95% largest)",
                        marker={
                            "symbol": f"{marker}-open",
                            "size": 8,
                            "color": color,
                        },
                        line={"color": color, "width": 1, "dash": "dot"},
                        hovertext=_build_hover_text(group, x_param, "bytes_per_pixel", "p95"),
                        hoverinfo="text",
                    )
                )
    else:
        stats_sorted = stats.sort_values(x_param)
        fig.add_trace(
            go.Scatter(
                x=stats_sorted[x_param],
                y=stats_sorted[mean_col],
                mode="lines+markers",
                name="Mean",
                marker={"symbol": "circle", "size": 8},
                line={"width": 2},
                hovertext=_build_hover_text(stats_sorted, x_param, "bytes_per_pixel", "mean"),
                hoverinfo="text",
            )
        )
        if p05_col in stats.columns:
            fig.add_trace(
                go.Scatter(
                    x=stats_sorted[x_param],
                    y=stats_sorted[p05_col],
                    mode="lines+markers",
                    name="5% smallest",
                    marker={"symbol": "circle-open", "size": 8},
                    line={"width": 1, "dash": "dash"},
                    hovertext=_build_hover_text(stats_sorted, x_param, "bytes_per_pixel", "p05"),
                    hoverinfo="text",
                )
            )
        if p95_col in stats.columns:
            fig.add_trace(
                go.Scatter(
                    x=stats_sorted[x_param],
                    y=stats_sorted[p95_col],
                    mode="lines+markers",
                    name="95% largest",
                    marker={"symbol": "circle-open", "size": 8},
                    line={"width": 1, "dash": "dot"},
                    hovertext=_build_hover_text(stats_sorted, x_param, "bytes_per_pixel", "p95"),
                    hoverinfo="text",
                )
            )

    return fig


def _should_use_log_scale(stats: pd.DataFrame, metric_base: str, threshold: float = 10.0) -> bool:
    """Determine if logarithmic scale should be used based on dynamic range.

    Checks the ratio between maximum and minimum values across mean, p05, and p95
    columns for the given metric. If any ratio exceeds the threshold, log scale
    is recommended.

    Args:
        stats: Statistics DataFrame
        metric_base: Base metric name (e.g., 'encoding_time_per_pixel')
        threshold: Ratio threshold for using log scale (default: 10x)

    Returns:
        True if log scale should be used, False otherwise
    """
    cols_to_check = [f"{metric_base}_mean", f"{metric_base}_p05", f"{metric_base}_p95"]
    cols_present = [col for col in cols_to_check if col in stats.columns]

    if not cols_present:
        return False

    # Collect all non-null, positive values from relevant columns
    all_values = []
    for col in cols_present:
        values = stats[col].dropna()
        values = values[values > 0]  # Only positive values for log scale consideration
        all_values.extend(values.tolist())

    if len(all_values) < 2:
        return False

    min_val: float = float(min(all_values))
    max_val: float = float(max(all_values))

    if min_val <= 0:
        return False

    ratio: float = max_val / min_val
    return ratio >= threshold


def plot_encoding_time_per_pixel(
    stats: pd.DataFrame,
    x_param: str,
    title: str | None = None,
) -> go.Figure:
    """Create interactive encoding time per pixel plot with percentile bands.

    Automatically uses logarithmic scale if the dynamic range is large (>10x).

    Args:
        stats: Statistics DataFrame
        x_param: Parameter for x-axis
        title: Optional custom title

    Returns:
        Plotly Figure object
    """
    mean_col = "encoding_time_per_pixel_mean"
    p05_col = "encoding_time_per_pixel_p05"
    p95_col = "encoding_time_per_pixel_p95"

    if mean_col not in stats.columns:
        return go.Figure()

    x_label = x_param.replace("_", " ").title()
    y_label = "Encoding Time per Pixel (seconds, lower is better)"
    fig_title = title or f"Encoding Time per Pixel vs {x_label}"

    # Determine if log scale should be used
    use_log_scale = _should_use_log_scale(stats, "encoding_time_per_pixel")

    layout = _create_figure_layout(fig_title, x_label, y_label)
    if use_log_scale:
        layout["yaxis"]["type"] = "log"
        layout["yaxis"]["title"]["text"] = y_label + " [log scale]"

    fig = go.Figure(layout=layout)

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
            label = (
                "_".join(str(n) for n in name if n is not None)
                if isinstance(name, tuple)
                else str(name)
                if name is not None
                else "default"
            )
            group = group.sort_values(x_param)
            color = _group_color(label, idx)
            marker = PLOTLY_MARKERS[idx % len(PLOTLY_MARKERS)]

            fig.add_trace(
                go.Scatter(
                    x=group[x_param],
                    y=group[mean_col],
                    mode="lines+markers",
                    name=f"{label} (mean)",
                    marker={"symbol": marker, "size": 8, "color": color},
                    line={"color": color, "width": 2},
                    hovertext=_build_hover_text(group, x_param, "encoding_time_per_pixel", "mean"),
                    hoverinfo="text",
                )
            )
            if p05_col in stats.columns:
                fig.add_trace(
                    go.Scatter(
                        x=group[x_param],
                        y=group[p05_col],
                        mode="lines+markers",
                        name=f"{label} (5% fastest)",
                        marker={
                            "symbol": f"{marker}-open",
                            "size": 8,
                            "color": color,
                        },
                        line={"color": color, "width": 1, "dash": "dash"},
                        hovertext=_build_hover_text(
                            group, x_param, "encoding_time_per_pixel", "p05"
                        ),
                        hoverinfo="text",
                    )
                )
            if p95_col in stats.columns:
                fig.add_trace(
                    go.Scatter(
                        x=group[x_param],
                        y=group[p95_col],
                        mode="lines+markers",
                        name=f"{label} (95% slowest)",
                        marker={
                            "symbol": f"{marker}-open",
                            "size": 8,
                            "color": color,
                        },
                        line={"color": color, "width": 1, "dash": "dot"},
                        hovertext=_build_hover_text(
                            group, x_param, "encoding_time_per_pixel", "p95"
                        ),
                        hoverinfo="text",
                    )
                )
    else:
        stats_sorted = stats.sort_values(x_param)
        fig.add_trace(
            go.Scatter(
                x=stats_sorted[x_param],
                y=stats_sorted[mean_col],
                mode="lines+markers",
                name="Mean",
                marker={"symbol": "circle", "size": 8},
                line={"width": 2},
                hovertext=_build_hover_text(
                    stats_sorted, x_param, "encoding_time_per_pixel", "mean"
                ),
                hoverinfo="text",
            )
        )
        if p05_col in stats.columns:
            fig.add_trace(
                go.Scatter(
                    x=stats_sorted[x_param],
                    y=stats_sorted[p05_col],
                    mode="lines+markers",
                    name="5% fastest",
                    marker={"symbol": "circle-open", "size": 8},
                    line={"width": 1, "dash": "dash"},
                    hovertext=_build_hover_text(
                        stats_sorted, x_param, "encoding_time_per_pixel", "p05"
                    ),
                    hoverinfo="text",
                )
            )
        if p95_col in stats.columns:
            fig.add_trace(
                go.Scatter(
                    x=stats_sorted[x_param],
                    y=stats_sorted[p95_col],
                    mode="lines+markers",
                    name="95% slowest",
                    marker={"symbol": "circle-open", "size": 8},
                    line={"width": 1, "dash": "dot"},
                    hovertext=_build_hover_text(
                        stats_sorted, x_param, "encoding_time_per_pixel", "p95"
                    ),
                    hoverinfo="text",
                )
            )

    return fig


def plot_efficiency(
    stats: pd.DataFrame,
    x_param: str,
    efficiency_metric: str,
    title: str | None = None,
) -> go.Figure:
    """Create interactive efficiency metric plot.

    Args:
        stats: Statistics DataFrame
        x_param: Parameter for x-axis
        efficiency_metric: Efficiency metric name
        title: Optional custom title

    Returns:
        Plotly Figure object
    """
    mean_col = f"{efficiency_metric}_mean"
    worst_col = f"{efficiency_metric}_{get_worst_percentile_col(efficiency_metric)}"

    if mean_col not in stats.columns or worst_col not in stats.columns:
        return go.Figure()

    x_label = x_param.replace("_", " ").title()
    y_label = _axis_label(efficiency_metric)
    fig_title = title or f"{_metric_label(efficiency_metric)} vs {x_label}"
    fig = go.Figure(layout=_create_figure_layout(fig_title, x_label, y_label))

    worst_suffix = get_worst_percentile_col(efficiency_metric)

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
            label = (
                "_".join(str(n) for n in name if n is not None)
                if isinstance(name, tuple)
                else str(name)
                if name is not None
                else "default"
            )
            group = group.sort_values(x_param)
            color = _group_color(label, idx)
            marker = PLOTLY_MARKERS[idx % len(PLOTLY_MARKERS)]

            fig.add_trace(
                go.Scatter(
                    x=group[x_param],
                    y=group[mean_col],
                    mode="lines+markers",
                    name=f"{label} (mean)",
                    marker={"symbol": marker, "size": 8, "color": color},
                    line={"color": color, "width": 2},
                    hovertext=_build_hover_text(group, x_param, efficiency_metric, "mean"),
                    hoverinfo="text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=group[x_param],
                    y=group[worst_col],
                    mode="lines+markers",
                    name=f"{label} (5% worst)",
                    marker={
                        "symbol": f"{marker}-open",
                        "size": 8,
                        "color": color,
                    },
                    line={"color": color, "width": 1.5, "dash": "dash"},
                    hovertext=_build_hover_text(group, x_param, efficiency_metric, worst_suffix),
                    hoverinfo="text",
                )
            )
    else:
        stats_sorted = stats.sort_values(x_param)
        fig.add_trace(
            go.Scatter(
                x=stats_sorted[x_param],
                y=stats_sorted[mean_col],
                mode="lines+markers",
                name="Mean",
                marker={"symbol": "circle", "size": 8},
                line={"width": 2},
                hovertext=_build_hover_text(stats_sorted, x_param, efficiency_metric, "mean"),
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=stats_sorted[x_param],
                y=stats_sorted[worst_col],
                mode="lines+markers",
                name="5% worst",
                marker={"symbol": "circle-open", "size": 8},
                line={"width": 1.5, "dash": "dash"},
                hovertext=_build_hover_text(stats_sorted, x_param, efficiency_metric, worst_suffix),
                hoverinfo="text",
            )
        )

    return fig


def figure_to_html_fragment(fig: go.Figure) -> str:
    """Convert a Plotly figure to an HTML fragment (no plotly.js included).

    The fragment contains the <div> and <script> needed to render the figure.
    The page template must load plotly.js separately.

    Args:
        fig: Plotly Figure object

    Returns:
        HTML string fragment
    """
    html: str = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config={
            "responsive": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        },
        default_height="500px",
        default_width="100%",
    )
    return html


def generate_study_figures(
    quality_json_path: str | Path,
) -> dict[str, go.Figure]:
    """Generate all interactive figures for a study.

    This is the main entry point that mirrors analyze_study() but produces
    Plotly figures instead of saving matplotlib images.

    Args:
        quality_json_path: Path to quality.json file

    Returns:
        Dictionary mapping figure names to Plotly Figure objects
    """
    quality_json_path = Path(quality_json_path)

    # Load and prepare data (reusing analysis module functions)
    quality_results = load_quality_results(quality_json_path)
    df = create_analysis_dataframe(quality_results)

    varying = determine_varying_parameters(df)
    if not varying:
        varying = ["format"] if "format" in df.columns else [df.columns[0]]

    stats = compute_statistics(df, varying)
    study_id = quality_results.get("study_id", "unknown")
    x_param = determine_sweep_parameter(df)
    secondary_param = determine_secondary_sweep_parameter(df, x_param)

    figures: dict[str, go.Figure] = {}

    # Priority order: perceptual metrics first (SSIMULACRA2, Butteraugli), then traditional
    quality_metrics = ["ssimulacra2", "butteraugli", "psnr", "ssim"]

    # 1. Rate-distortion plots (quality vs bytes_per_pixel) - MOST INFORMATIVE
    if "bytes_per_pixel_mean" in stats.columns:
        for metric in quality_metrics:
            if f"{metric}_mean" in stats.columns:
                key = f"{study_id}_{metric}_vs_bytes_per_pixel"
                figures[key] = plot_rate_distortion(stats, metric, secondary_param, primary_param=x_param)

    # 2. Quality metric plots vs sweep parameter
    for metric in quality_metrics:
        if f"{metric}_mean" in stats.columns:
            key = f"{study_id}_{metric}_vs_{x_param}"
            figures[key] = plot_quality_vs_param(stats, x_param, metric)

    # 3. Efficiency metric plots (perceptual metrics prioritized)
    efficiency_metrics = [
        "bytes_per_ssimulacra2_per_pixel",
        "bytes_per_butteraugli_per_pixel",
        "bytes_per_psnr_per_pixel",
        "bytes_per_ssim_per_pixel",
    ]
    for metric in efficiency_metrics:
        if f"{metric}_mean" in stats.columns:
            key = f"{study_id}_{metric}_vs_{x_param}"
            figures[key] = plot_efficiency(stats, x_param, metric)

    # 4. Bytes per pixel plot
    if "bytes_per_pixel_mean" in stats.columns:
        key = f"{study_id}_bytes_per_pixel_vs_{x_param}"
        figures[key] = plot_bytes_per_pixel(stats, x_param)

    # 5. Encoding time per pixel plot
    if "encoding_time_per_pixel_mean" in stats.columns:
        key = f"{study_id}_encoding_time_per_pixel_vs_{x_param}"
        figures[key] = plot_encoding_time_per_pixel(stats, x_param)

    return figures
