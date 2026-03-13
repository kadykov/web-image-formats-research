"""Shared website configuration and asset helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import minify_html


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_CONFIG_PATH = PROJECT_ROOT / "config" / "site.json"
SITE_ASSETS_DIR = PROJECT_ROOT / "pages" / "assets"


@dataclass(frozen=True)
class SiteConfig:
    """Typed access to shared site configuration."""

    site_name: str
    site_description: str
    site_origin: str
    base_path: str
    custom_domain: str
    repository_url: str
    report_subpath: str
    docs_subpath: str
    brand: dict[str, str]
    deployable_assets: tuple[str, ...]
    source_only_assets: tuple[str, ...]

    @property
    def canonical_base_url(self) -> str:
        """Return the public site URL including the repository subpath."""
        return f"{self.site_origin.rstrip('/')}{self.base_path}"


@cache
def get_site_config() -> SiteConfig:
    """Load site config once per process."""
    data = json.loads(SITE_CONFIG_PATH.read_text(encoding="utf-8"))
    return SiteConfig(
        site_name=data["site_name"],
        site_description=data["site_description"],
        site_origin=data["site_origin"],
        base_path=data["base_path"],
        custom_domain=data["custom_domain"],
        repository_url=data["repository_url"],
        report_subpath=data["report_subpath"],
        docs_subpath=data["docs_subpath"],
        brand=dict(data["brand"]),
        deployable_assets=tuple(data["deployable_assets"]),
        source_only_assets=tuple(data["source_only_assets"]),
    )


def canonical_url(path: str = "") -> str:
    """Build a canonical absolute URL within the published site."""
    config = get_site_config()
    if not path:
        return f"{config.canonical_base_url}/"
    return f"{config.canonical_base_url}/{path.lstrip('/')}"


def copy_deployable_assets(target_dir: Path) -> None:
    """Copy the deployable website assets into a published directory root."""
    config = get_site_config()
    target_dir.mkdir(parents=True, exist_ok=True)
    for asset_name in config.deployable_assets:
        src = SITE_ASSETS_DIR / asset_name
        if not src.exists():
            raise FileNotFoundError(f"Missing deployable asset: {src}")
        dst = target_dir / asset_name
        dst.write_bytes(src.read_bytes())


def asset_paths() -> dict[str, str]:
    """Return relative asset paths for a page living at a site-area root."""
    return {
        "favicon_ico": "assets/favicon.ico",
        "icon_svg": "assets/icon.svg",
        "apple_touch_icon": "assets/apple-touch-icon.png",
        "manifest": "assets/manifest.webmanifest",
        "logo_svg": "assets/logo.svg",
        "opengraph_image": "assets/opengraph.png",
    }


def minify_html_document(html: str) -> str:
    """Minify HTML while preserving inline CSS and JS semantics."""
    return minify_html.minify(
        html,
        keep_comments=False,
        keep_html_and_head_opening_tags=True,
        minify_css=True,
        minify_js=True,
    )


def sitemap_entries(root_dir: Path) -> list[dict[str, Any]]:
    """Collect sitemap entries from the published site tree."""
    entries: list[dict[str, Any]] = []
    for html_file in sorted(root_dir.rglob("*.html")):
        if html_file.name == "404.html":
            continue
        content = html_file.read_text(encoding="utf-8")
        if 'name="robots" content="noindex' in content.lower():
            continue

        relative = html_file.relative_to(root_dir).as_posix()
        if relative == "index.html":
            loc = canonical_url("")
        elif relative.endswith("/index.html"):
            loc = canonical_url(relative[: -len("index.html")])
        else:
            loc = canonical_url(relative)
        entries.append({"loc": loc})
    return entries