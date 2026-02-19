#!/usr/bin/env python3
"""Generate documentation for Astro Starlight.

This script:
1. Generates API reference documentation from Python docstrings using lazydocs
2. Copies existing documentation from docs/ to docs-site/src/content/docs/
3. Processes and organizes all markdown files for Astro Starlight
"""

import shutil
import subprocess
import sys
from pathlib import Path


def generate_api_docs(src_dir: Path, output_dir: Path) -> None:
    """Generate API reference documentation from Python source files.

    Args:
        src_dir: Path to source code directory
        output_dir: Path where API docs should be generated
    """
    print("📚 Generating API reference documentation...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run lazydocs to generate markdown from Python docstrings
    # Generate docs for each Python module in src/
    python_files = [
        f for f in src_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_")
    ]

    if not python_files:
        print("⚠️  No Python files found to document")
        return

    # Use lazydocs to generate documentation
    # Find lazydocs in the virtual environment
    project_root = Path(__file__).parent.parent
    lazydocs_path = project_root / ".venv" / "bin" / "lazydocs"

    lazydocs_cmd = "lazydocs" if not lazydocs_path.exists() else str(lazydocs_path)

    cmd = [
        lazydocs_cmd,
        "--output-path",
        str(output_dir),
        "--src-base-url",
        "https://github.com/kadykov/web-image-formats-research/blob/main/",
        "--no-watermark",
        *[str(f) for f in python_files],
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error generating API docs: {result.stderr}")
        sys.exit(1)

    print(f"✅ Generated API documentation for {len(python_files)} modules")


def copy_docs(docs_dir: Path, content_dir: Path) -> None:
    """Copy existing documentation to Astro content directory.

    Args:
        docs_dir: Path to existing docs directory
        content_dir: Path to Astro content/docs directory
    """
    print("\n📋 Copying existing documentation...")

    # Create an index.md from README.md
    readme_file = docs_dir.parent / "README.md"
    if readme_file.exists():
        index_target = content_dir / "index.mdx"
        content = readme_file.read_text()

        # Fix links for Starlight context
        # README.md has links like 'docs/tutorials/getting-started'
        # which need to be '/tutorials/getting-started/' for Starlight
        content = content.replace("(docs/tutorials/", "(/tutorials/")
        content = content.replace("(docs/how-to/", "(/how-to/")
        content = content.replace("(docs/reference/", "(/reference/")
        content = content.replace("(docs/explanation/", "(/explanation/")

        # Add Starlight frontmatter
        starlight_content = f"""---
title: Welcome
description: Research project for determining optimal modern image formats for the web
---

{content}
"""
        index_target.write_text(starlight_content)
        print("  ✓ Created index.mdx from README.md")

    # Copy all documentation directories
    for section_dir in ["tutorials", "how-to", "explanation", "reference"]:
        src_section = docs_dir / section_dir
        if not src_section.exists():
            continue

        dest_section = content_dir / section_dir

        # Copy the directory
        if dest_section.exists():
            shutil.rmtree(dest_section)
        shutil.copytree(src_section, dest_section)
        print(f"  ✓ Copied {section_dir}/")

    print("✅ Documentation copied successfully")


def add_frontmatter_to_docs(content_dir: Path) -> None:
    """Add Starlight frontmatter to markdown files that don't have it.

    Args:
        content_dir: Path to Astro content/docs directory
    """
    print("\n🔧 Processing markdown files...")

    for md_file in content_dir.rglob("*.md"):
        if md_file.name.startswith("."):
            continue

        content = md_file.read_text()

        # Skip if already has frontmatter
        if content.startswith("---"):
            continue

        # Extract title from first heading
        lines = content.split("\n")
        title = "Documentation"
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        # Add frontmatter with proper YAML quoting for titles with special chars
        # Quote title if it contains special characters like colons
        if ":" in title or '"' in title or "'" in title or "\n" in title:
            # Escape quotes and use double quotes
            title_escaped = title.replace('"', '\\"')
            title_yaml = f'"{title_escaped}"'
        else:
            title_yaml = title

        frontmatter = f"""---
title: {title_yaml}
---

"""
        new_content = frontmatter + content
        md_file.write_text(new_content)

    print("✅ Processed markdown files")


def create_api_index(api_dir: Path) -> None:
    """Create an index.md for the API reference section.

    Args:
        api_dir: Path to API reference directory
    """
    index_file = api_dir / "index.md"
    if index_file.exists():
        return

    content = """---
title: API Reference
description: Automatically generated API documentation from Python docstrings
---

# API Reference

This section contains automatically generated documentation from the Python source code docstrings.

## Modules

"""

    # List all generated markdown files
    for md_file in sorted(api_dir.glob("*.md")):
        if md_file.name == "index.md":
            continue
        module_name = md_file.stem
        # Clean up the module name (remove .py extension if present in filename)
        display_name = module_name.replace(".py", "")
        content += f"- [{display_name}](./{md_file.name})\n"

    index_file.write_text(content)
    print("  ✓ Created API reference index")


def add_frontmatter_to_api_docs(api_dir: Path) -> None:
    """Add frontmatter to API documentation files generated by lazydocs.

    Args:
        api_dir: Path to API documentation directory
    """
    if not api_dir.exists():
        return

    for md_file in api_dir.glob("*.md"):
        if md_file.name == "index.md":
            continue

        content = md_file.read_text()

        # Skip if already has frontmatter
        if content.startswith("---"):
            continue

        # Extract module name from filename
        module_name = md_file.stem.replace(".py", "")

        # Add frontmatter
        frontmatter = f"""---
title: "{module_name}"
description: API reference for {module_name} module
---

"""
        new_content = frontmatter + content
        md_file.write_text(new_content)


def main() -> None:
    """Main entry point."""
    # Define paths
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    docs_dir = project_root / "docs"
    docs_site_dir = project_root / "docs-site"
    content_dir = docs_site_dir / "src" / "content" / "docs"
    api_dir = content_dir / "reference" / "api"

    # Check if docs-site exists
    if not docs_site_dir.exists():
        print("❌ docs-site directory not found. Run 'just docs-init' first.")
        sys.exit(1)

    # Clean existing content (except example files)
    if content_dir.exists():
        for item in content_dir.iterdir():
            if item.name not in [".gitkeep"]:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    # Copy existing documentation first
    copy_docs(docs_dir, content_dir)

    # Note: We no longer patch frontmatter for existing docs.
    # Frontmatter should be added directly to source files in docs/

    # Generate API documentation after copying
    generate_api_docs(src_dir, api_dir)

    # Add frontmatter to API docs (auto-generated files don't have it)
    add_frontmatter_to_api_docs(api_dir)

    # Create API index
    create_api_index(api_dir)

    print("\n🎉 Documentation generation complete!")
    print(f"   Content directory: {content_dir}")
    print("\n💡 Next steps:")
    print("   • Run 'just docs-dev' to preview locally")
    print("   • Run 'just docs-build' to build for production")


if __name__ == "__main__":
    main()
