#!/usr/bin/env python3
"""Validate the root tutorial catalog and active-folder hygiene."""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def catalog_links() -> set[Path]:
    """Return local tutorial directories linked from the root README."""
    text = (ROOT / "README.md").read_text()
    links = set()
    for raw in re.findall(r"\]\(([^)#]+/)\)", text):
        if raw.startswith(("http://", "https://")):
            continue
        path = (ROOT / raw).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        if path.is_dir():
            links.add(path)
    return links


def active_notebooks() -> list[Path]:
    """Find notebooks that still live outside legacy storage."""
    bad = []
    for path in ROOT.rglob("*.ipynb"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "_legacy" in rel.parts:
            continue
        bad.append(rel)
    return sorted(bad)


def active_checkpoints() -> list[Path]:
    """Find Jupyter checkpoint directories outside legacy storage."""
    bad = []
    for path in ROOT.rglob(".ipynb_checkpoints"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "_legacy" in rel.parts:
            continue
        bad.append(rel)
    return sorted(bad)


def tutorial_dirs() -> set[Path]:
    """Return active tutorial directories with a run.py entrypoint."""
    dirs = set()
    for path in ROOT.rglob("run.py"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "_legacy" in rel.parts:
            continue
        dirs.add(path.parent.resolve())
    return dirs


def validate() -> int:
    errors = []
    links = catalog_links()
    tutorials = tutorial_dirs()

    if not links:
        errors.append("No tutorial links found in README.md")

    for directory in sorted(links):
        rel = directory.relative_to(ROOT)
        for required in ["run.py", "README.md", "figures/thumb.png"]:
            if not (directory / required).exists():
                errors.append(f"{rel} is missing {required}")

    for directory in sorted(tutorials - links):
        errors.append(f"Tutorial missing from root README catalog: {directory.relative_to(ROOT)}")

    for rel in active_notebooks():
        errors.append(f"Active notebook outside _legacy: {rel}")

    for rel in active_checkpoints():
        errors.append(f"Active checkpoint directory outside _legacy: {rel}")

    if errors:
        print("Catalog validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Catalog validation passed for {len(links)} tutorials.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
