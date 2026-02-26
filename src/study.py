"""Study configuration and parsing module.

This module handles loading study configurations from JSON files and
expanding parameter sweeps into encoder specifications.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EncoderConfig:
    """Configuration for a single encoder sweep."""

    format: str
    quality: list[int]
    chroma_subsampling: list[str] | None = None
    speed: list[int] | None = None
    effort: list[int] | None = None
    method: list[int] | None = None
    resolution: list[int] | None = None
    extra_args: dict[str, str | int | bool] | None = None


@dataclass
class StudyConfig:
    """Configuration for a complete encoding study."""

    id: str
    name: str
    dataset_id: str
    max_images: int | None
    encoders: list[EncoderConfig]
    description: str | None = None
    time_budget: float | None = None

    @classmethod
    def from_file(cls, config_path: Path) -> "StudyConfig":
        """Load a study configuration from a JSON file.

        Args:
            config_path: Path to the study JSON file

        Returns:
            StudyConfig instance

        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config file has invalid content
        """
        if not config_path.exists():
            msg = f"Study config not found: {config_path}"
            raise FileNotFoundError(msg)

        with open(config_path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StudyConfig":
        """Create StudyConfig from a dictionary.

        Args:
            data: Dictionary matching the study JSON schema

        Returns:
            StudyConfig instance

        Raises:
            ValueError: If required fields are missing
        """
        if "id" not in data or "dataset" not in data or "encoders" not in data:
            msg = "Study config must have 'id', 'dataset', and 'encoders' fields"
            raise ValueError(msg)

        encoders = [_parse_encoder_config(enc) for enc in data["encoders"]]

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            dataset_id=data["dataset"]["id"],
            max_images=data["dataset"].get("max_images"),
            encoders=encoders,
            description=data.get("description"),
            time_budget=data.get("time_budget"),
        )


def _parse_quality(quality: int | list[int] | dict[str, int]) -> list[int]:
    """Parse quality parameter into a list of integers.

    Supports:
    - Single integer: ``75`` → ``[75]``
    - Explicit list: ``[60, 75, 90]`` → ``[60, 75, 90]``
    - Range object: ``{"start": 30, "stop": 90, "step": 10}`` → ``[30, 40, ..., 90]``

    Args:
        quality: Quality specification from the study config

    Returns:
        List of quality values
    """
    if isinstance(quality, int):
        return [quality]
    if isinstance(quality, list):
        return quality
    if isinstance(quality, dict):
        return list(range(quality["start"], quality["stop"] + 1, quality["step"]))
    msg = f"Invalid quality specification: {quality}"
    raise ValueError(msg)


def _parse_encoder_config(data: dict[str, Any]) -> EncoderConfig:
    """Parse an encoder configuration dictionary into an EncoderConfig.

    Args:
        data: Encoder dictionary from the study JSON

    Returns:
        EncoderConfig instance
    """
    quality = _parse_quality(data["quality"])

    speed_raw = data.get("speed")
    speed: list[int] | None = None
    if speed_raw is not None:
        speed = [speed_raw] if isinstance(speed_raw, int) else speed_raw

    effort_raw = data.get("effort")
    effort: list[int] | None = None
    if effort_raw is not None:
        effort = [effort_raw] if isinstance(effort_raw, int) else effort_raw

    method_raw = data.get("method")
    method: list[int] | None = None
    if method_raw is not None:
        method = [method_raw] if isinstance(method_raw, int) else method_raw

    chroma = data.get("chroma_subsampling")

    resolution_raw = data.get("resolution")
    resolution: list[int] | None = None
    if resolution_raw is not None:
        resolution = [resolution_raw] if isinstance(resolution_raw, int) else resolution_raw

    return EncoderConfig(
        format=data["format"],
        quality=quality,
        chroma_subsampling=chroma,
        speed=speed,
        effort=effort,
        method=method,
        resolution=resolution,
        extra_args=data.get("extra_args"),
    )

