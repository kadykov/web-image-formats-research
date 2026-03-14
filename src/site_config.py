"""Shared website configuration and asset helpers."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path

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
    )


def canonical_url(path: str = "") -> str:
    """Build a canonical absolute URL within the published site."""
    config = get_site_config()
    if not path:
        return f"{config.canonical_base_url}/"
    return f"{config.canonical_base_url}/{path.lstrip('/')}"


def copy_deployable_assets(target_dir: Path) -> None:
    """Copy the website assets into a published directory root."""
    shutil.copytree(SITE_ASSETS_DIR, target_dir, dirs_exist_ok=True)


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
