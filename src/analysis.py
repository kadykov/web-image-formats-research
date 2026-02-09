"""Analysis and visualization module.

This module handles data analysis and visualization of compression
efficiency and quality metrics.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CompressionAnalyzer:
    """Analyzes compression results and generates visualizations."""

    def __init__(self, results_dir: Path) -> None:
        """Initialize the compression analyzer.

        Args:
            results_dir: Directory containing compression results
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_dataframe(self, results: list[dict[str, Any]]) -> pd.DataFrame:
        """Create a pandas DataFrame from encoding results.

        Args:
            results: List of result dictionaries

        Returns:
            DataFrame with compression results
        """
        return pd.DataFrame(results)

    def calculate_compression_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate compression ratios.

        Args:
            df: DataFrame with 'original_size' and 'compressed_size' columns

        Returns:
            DataFrame with added 'compression_ratio' column
        """
        df = df.copy()
        df["compression_ratio"] = df["original_size"] / df["compressed_size"]
        df["size_reduction_pct"] = (
            (df["original_size"] - df["compressed_size"]) / df["original_size"]
        ) * 100
        return df

    def plot_quality_vs_size(
        self,
        df: pd.DataFrame,
        quality_metric: str = "ssimulacra2",
        output_path: Path | None = None,
    ) -> None:
        """Plot quality vs file size for different formats.

        Args:
            df: DataFrame with compression results
            quality_metric: Name of the quality metric column to use
            output_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        formats = df["format"].unique()
        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(formats)))

        for format_name, color in zip(formats, colors, strict=True):
            format_df = df[df["format"] == format_name]
            ax.scatter(
                format_df["compressed_size"] / 1024,  # Convert to KB
                format_df[quality_metric],
                label=format_name.upper(),
                alpha=0.7,
                s=100,
                color=color,
            )
            # Add trend line
            if len(format_df) > 1:
                z = np.polyfit(format_df["compressed_size"] / 1024, format_df[quality_metric], 2)
                p = np.poly1d(z)
                x_line = np.linspace(
                    format_df["compressed_size"].min() / 1024,
                    format_df["compressed_size"].max() / 1024,
                    100,
                )
                ax.plot(x_line, p(x_line), "--", color=color, alpha=0.5)

        ax.set_xlabel("File Size (KB)", fontsize=12)
        ax.set_ylabel(f"{quality_metric.upper()} Score", fontsize=12)
        ax.set_title("Image Quality vs File Size Comparison", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.results_dir / f"quality_vs_size_{quality_metric}.png", dpi=300)

        plt.close()

    def plot_compression_efficiency(
        self, df: pd.DataFrame, output_path: Path | None = None
    ) -> None:
        """Plot compression ratio across different formats and quality settings.

        Args:
            df: DataFrame with compression results
            output_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        formats = df["format"].unique()
        x = np.arange(len(formats))
        width = 0.35

        # Calculate average compression ratio per format
        avg_compression = df.groupby("format")["compression_ratio"].mean()
        avg_quality = df.groupby("format")["ssimulacra2"].mean()

        ax.bar(x - width / 2, avg_compression, width, label="Compression Ratio")
        ax.bar(x + width / 2, avg_quality / 10, width, label="Avg Quality / 10")

        ax.set_xlabel("Format", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title("Average Compression Efficiency by Format", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f.upper() for f in formats])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.results_dir / "compression_efficiency.png", dpi=300)

        plt.close()

    def generate_summary_report(self, df: pd.DataFrame, output_path: Path | None = None) -> str:
        """Generate a text summary report of the analysis.

        Args:
            df: DataFrame with compression results
            output_path: Optional path to save the report

        Returns:
            Summary report as a string
        """
        report_lines = ["# Compression Analysis Summary\n"]

        # Overall statistics
        report_lines.append("## Overall Statistics\n")
        report_lines.append(f"Total images analyzed: {len(df['image_name'].unique())}")
        report_lines.append(f"Total encodings: {len(df)}")
        report_lines.append(f"Formats tested: {', '.join(df['format'].unique())}\n")

        # Per-format statistics
        report_lines.append("## Per-Format Statistics\n")
        for format_name in df["format"].unique():
            format_df = df[df["format"] == format_name]
            report_lines.append(f"### {format_name.upper()}\n")
            report_lines.append(
                f"- Average compression ratio: {format_df['compression_ratio'].mean():.2f}x"
            )
            report_lines.append(f"- Average SSIMULACRA2: {format_df['ssimulacra2'].mean():.2f}")
            report_lines.append(
                f"- Average file size: {format_df['compressed_size'].mean() / 1024:.2f} KB"
            )
            report_lines.append(
                f"- Size reduction: {format_df['size_reduction_pct'].mean():.1f}%\n"
            )

        report = "\n".join(report_lines)

        if output_path:
            output_path.write_text(report)
        else:
            (self.results_dir / "summary_report.md").write_text(report)

        return report
