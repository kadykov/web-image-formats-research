#!/usr/bin/env python3
"""Stage the complete GitHub Pages site root for deployment."""

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
    sitemap_entries,
)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", ".github"),
    )


def _replace_subtree(dst_root: Path, name: str, src: Path | None) -> None:
    target = dst_root / name
    if src is None or not src.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target)


def _write_robots(root: Path) -> None:
    robots = f"User-agent: *\nAllow: /\n\nSitemap: {canonical_url('sitemap.xml')}\n"
    (root / "robots.txt").write_text(robots, encoding="utf-8")


def _write_sitemap(root: Path) -> None:
    urlset = Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for entry in sitemap_entries(root):
        url = SubElement(urlset, "url")
        loc = SubElement(url, "loc")
        loc.text = entry["loc"]
    ElementTree(urlset).write(root / "sitemap.xml", encoding="utf-8", xml_declaration=True)


def _ensure_noindex_404(root: Path) -> None:
    for html_file in root.rglob("404.html"):
        content = html_file.read_text(encoding="utf-8")
        if 'name="robots"' in content:
            continue
        content = content.replace(
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>',
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
            '<meta name="robots" content="noindex,follow"/>',
            1,
        )
        html_file.write_text(content, encoding="utf-8")


def stage_site(
    output_root: Path,
    current_root: Path | None,
    landing_root: Path,
    docs_root: Path | None,
    report_root: Path | None,
) -> None:
    """Build the final publishable site tree."""
    site_config = get_site_config()

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if current_root and current_root.exists():
        _copy_tree(current_root, output_root)

    landing_html = (landing_root / "index.html").read_text(encoding="utf-8")
    (output_root / "index.html").write_text(minify_html_document(landing_html), encoding="utf-8")

    assets_root = output_root / "assets"
    if assets_root.exists():
        shutil.rmtree(assets_root)
    copy_deployable_assets(assets_root)

    _replace_subtree(output_root, site_config.docs_subpath, docs_root)
    _replace_subtree(output_root, site_config.report_subpath, report_root)

    (output_root / ".nojekyll").write_text("", encoding="utf-8")
    (output_root / "CNAME").write_text(f"{site_config.custom_domain}\n", encoding="utf-8")

    _ensure_noindex_404(output_root)
    _write_robots(output_root)
    _write_sitemap(output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the GitHub Pages site tree")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--landing-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path)
    parser.add_argument("--docs-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    args = parser.parse_args()

    stage_site(
        output_root=args.output_root,
        current_root=args.current_root,
        landing_root=args.landing_root,
        docs_root=args.docs_root,
        report_root=args.report_root,
    )
    print(f"Staged GitHub Pages site at {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
