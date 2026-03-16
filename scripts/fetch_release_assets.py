#!/usr/bin/env python3
"""Fetch study results from a GitHub Release and unpack them locally.

Downloads the release assets (statistics CSVs and per-study ZIP archives)
so that the paper can be rendered from local data without running the
full encoding pipeline.

Usage:
    # Fetch the latest release
    python3 scripts/fetch_release_assets.py

    # Fetch a specific tagged release
    python3 scripts/fetch_release_assets.py --tag study-20250601-120000

    # Custom output directories
    python3 scripts/fetch_release_assets.py --analysis-dir data/analysis --release-dir release-assets
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.site_config import get_site_config

# GitHub API base for the project repository
_API_BASE = "https://api.github.com"

# Default timeout for network requests (in seconds)
DEFAULT_TIMEOUT = 30


def _repo_slug() -> str:
    """Derive the owner/repo slug from the configured repository URL."""
    url = get_site_config().repository_url.rstrip("/")
    # https://github.com/kadykov/web-image-formats-research → kadykov/web-image-formats-research
    parts = url.split("/")
    return f"{parts[-2]}/{parts[-1]}"


def _api_get(path: str) -> dict | list:
    """Perform an unauthenticated GitHub API GET request."""
    url = f"{_API_BASE}{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "web-image-formats-research/1.0",
    }
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:  # noqa: S310
            result: dict | list = json.loads(resp.read())
            return result
    except HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code} fetching {url}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error fetching {url}: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from {url}: {e}") from e


def _download(url: str) -> bytes:
    """Download a file from *url* and return its contents."""
    headers = {"User-Agent": "web-image-formats-research/1.0"}
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:  # noqa: S310
            data: bytes = resp.read()
            return data
    except HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code} downloading {url}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error downloading {url}: {e.reason}") from e


def _get_release(tag: str | None) -> dict:
    """Fetch release metadata — latest or for a specific tag."""
    slug = _repo_slug()
    if tag is None:
        result = _api_get(f"/repos/{slug}/releases/latest")
    else:
        result = _api_get(f"/repos/{slug}/releases/tags/{tag}")

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected response type from GitHub API: {type(result).__name__}")
    if "tag_name" not in result:
        raise RuntimeError("Invalid release response: missing 'tag_name' field")
    return result


def fetch_release_assets(
    *,
    tag: str | None = None,
    analysis_dir: Path = Path("data/analysis"),
    release_dir: Path = Path("release-assets"),
) -> list[Path]:
    """Download release assets and unpack study ZIPs into local directories.

    CSV statistics files are saved to *release_dir*.  ZIP archives are
    also saved there and their ``analysis/`` contents are extracted into
    *analysis_dir/<study-id>/*, mirroring the layout produced by the
    local pipeline.

    Args:
        tag: GitHub release tag name, or ``None`` for the latest release.
        analysis_dir: Where to unpack study analysis data.
        release_dir: Where to save raw release asset files.

    Returns:
        List of created file paths.
    """
    release = _get_release(tag)
    tag_name: str = release["tag_name"]
    print(f"Fetching release: {release['name']} ({tag_name})")

    assets: list[dict] = release.get("assets", [])
    if not assets:
        print("No assets found in this release.", file=sys.stderr)
        return []

    release_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for asset in assets:
        name: str = asset["name"]
        download_url: str = asset["browser_download_url"]
        size_mb = asset["size"] / (1024 * 1024)
        print(f"  Downloading {name} ({size_mb:.1f} MB)...", end=" ", flush=True)

        data = _download(download_url)

        # Save the raw asset
        dest = release_dir / name
        dest.write_bytes(data)
        created.append(dest)
        print("ok")

        # Unpack ZIP archives into the analysis tree
        if name.endswith(".zip"):
            study_id = name.removesuffix(".zip")
            study_dest = analysis_dir / study_id
            study_dest.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for member in zf.namelist():
                    # Extract only analysis/ entries, stripping the prefix
                    if member.startswith("analysis/") and not member.endswith("/"):
                        relative = member.removeprefix("analysis/")
                        out_path = study_dest / relative
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_bytes(zf.read(member))
                        created.append(out_path)

    print(f"\nDone — {len(created)} file(s) from release {tag_name}")
    return created


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch study results from a GitHub Release.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Release tag to fetch (default: latest release)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Directory to unpack study analysis data into (default: data/analysis)",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=Path("release-assets"),
        help="Directory to save raw release assets (default: release-assets)",
    )
    args = parser.parse_args()

    try:
        assets = fetch_release_assets(
            tag=args.tag,
            analysis_dir=args.analysis_dir,
            release_dir=args.release_dir,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not assets:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
