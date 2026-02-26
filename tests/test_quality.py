"""Tests for quality measurement module."""

from src.quality import QualityMetrics


def test_quality_metrics_dataclass() -> None:
    """Test QualityMetrics dataclass."""
    metrics = QualityMetrics(ssimulacra2=85.5, psnr=42.3, ssim=0.98)
    assert metrics.ssimulacra2 == 85.5
    assert metrics.psnr == 42.3
    assert metrics.ssim == 0.98
    assert metrics.butteraugli is None
