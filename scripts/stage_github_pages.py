#!/usr/bin/env python3
"""Stage the GitHub Pages site root (landing page + shared assets)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from src.site_config import (
    canonical_url,
    copy_deployable_assets,
    get_site_config,
    minify_html_document,
)


def _write_robots(root: Path) -> None:
    robots = f"User-agent: *\nAllow: /\n\nSitemap: {canonical_url('sitemap-index.xml')}\n"
    (root / "robots.txt").write_text(robots, encoding="utf-8")


def _write_sitemap_index(root: Path) -> None:
    site_config = get_site_config()
    sitemapindex = Element("sitemapindex", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for loc in [
        canonical_url(f"{site_config.docs_subpath}/sitemap-index.xml"),
        canonical_url(f"{site_config.report_subpath}/sitemap.xml"),
    ]:
        SubElement(SubElement(sitemapindex, "sitemap"), "loc").text = loc
    ElementTree(sitemapindex).write(
        root / "sitemap-index.xml", encoding="utf-8", xml_declaration=True
    )


def stage_site(
    output_root: Path,
    landing_root: Path,
) -> None:
    """Stage root-level GitHub Pages files (landing page, assets, robots, sitemap-index)."""
    site_config = get_site_config()

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    landing_html = (landing_root / "index.html").read_text(encoding="utf-8")
    (output_root / "index.html").write_text(minify_html_document(landing_html), encoding="utf-8")

    not_found_html = (landing_root / "404.html").read_text(encoding="utf-8")
    (output_root / "404.html").write_text(minify_html_document(not_found_html), encoding="utf-8")

    copy_deployable_assets(output_root)

    (output_root / ".nojekyll").write_text("", encoding="utf-8")
    (output_root / "CNAME").write_text(f"{site_config.custom_domain}\n", encoding="utf-8")

    _write_robots(output_root)
    _write_sitemap_index(output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage root-level GitHub Pages files")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--landing-root", type=Path, required=True)
    args = parser.parse_args()

    stage_site(
        output_root=args.output_root,
        landing_root=args.landing_root,
    )
    print(f"Staged GitHub Pages site root at {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
