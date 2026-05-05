#!/usr/bin/env python3
"""Validate the root tutorial catalog and active-folder hygiene."""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DISPLAY_MATH_BAD_PREFIX = re.compile(r"^\s*[+*-]\s+")
TABLE_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
FRAGILE_MATH_DELIMITERS = (
    "\\left" + "{",
    "\\left" + "\\{",
    "\\right" + "}",
    "\\right" + "\\}",
)
UNBRACED_STAR_SCRIPT = re.compile(r"(?<!\\)(\^|_)\*")
BRACED_LITERAL_STAR_SCRIPT = re.compile(r"(?<!\\)(\^|_)\{\*\}")
EMPTY_SCRIPT_TARGET = re.compile(r"(?<!\\)(\^|_)(?:\s|$|[,$.;:)\]}]|[\^_])")


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


def active_text_files() -> list[Path]:
    """Return active Markdown and Python source files to lint lightly."""
    files = []
    for pattern in ("*.md", "*.py"):
        for path in ROOT.rglob(pattern):
            rel = path.relative_to(ROOT)
            if ".git" in rel.parts or "_legacy" in rel.parts:
                continue
            files.append(path)
    return sorted(files)


def display_math_errors() -> list[str]:
    """Find display-math blocks that are likely to be misparsed by Markdown."""
    errors = []
    for path in active_text_files():
        rel = path.relative_to(ROOT)
        in_math = False
        start_line: int | None = None
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if line.strip() == "$$":
                in_math = not in_math
                start_line = lineno if in_math else None
                continue
            if in_math and DISPLAY_MATH_BAD_PREFIX.match(line):
                errors.append(
                    f"{rel}:{lineno} display math line starts with a Markdown list marker/operator"
                )
        if in_math and start_line is not None:
            errors.append(f"{rel}:{start_line} unclosed display math block")
    return errors


def count_unescaped_pipes(line: str) -> int:
    """Count Markdown table separators, ignoring escaped literal pipes."""
    return len(re.findall(r"(?<!\\)\|", line))


def is_markdown_table_separator(line: str) -> bool:
    """Return whether a line looks like a Markdown table separator row."""
    if "|" not in line:
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in cells)


def markdown_table_errors() -> list[str]:
    """Find table rows whose unescaped pipes would break GitHub rendering."""
    errors = []
    for path in active_text_files():
        if path.suffix != ".md":
            continue
        rel = path.relative_to(ROOT)
        lines = path.read_text(errors="replace").splitlines()
        in_fence = False
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("```"):
                in_fence = not in_fence
                i += 1
                continue
            if not in_fence and i + 1 < len(lines) and is_markdown_table_separator(lines[i + 1]):
                expected = count_unescaped_pipes(lines[i])
                j = i
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    actual = count_unescaped_pipes(lines[j])
                    if actual != expected:
                        errors.append(
                            f"{rel}:{j + 1} Markdown table row has {actual} unescaped pipes; expected {expected}"
                        )
                    j += 1
                i = j
                continue
            i += 1
    return errors


def fragile_math_delimiter_errors() -> list[str]:
    """Reject LaTeX delimiter forms that GitHub Markdown often misrenders."""
    errors = []
    for path in active_text_files():
        rel = path.relative_to(ROOT)
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            for delimiter in FRAGILE_MATH_DELIMITERS:
                if delimiter in line:
                    errors.append(
                        f"{rel}:{lineno} uses fragile math delimiter {delimiter}; use bracket delimiters instead"
                    )
    return errors


def math_script_errors() -> list[str]:
    """Reject math scripts that are fragile after Markdown preprocessing."""
    errors = []
    for path in active_text_files():
        if path.suffix == ".py" and path.name != "run.py":
            continue
        rel = path.relative_to(ROOT)
        in_display_math = False
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            stripped = line.strip()
            has_math_marker = "$" in line or in_display_math

            if has_math_marker:
                if UNBRACED_STAR_SCRIPT.search(line):
                    errors.append(
                        f"{rel}:{lineno} uses an unbraced star script in math; write ^{{\\ast}} or _{{\\ast}}"
                    )
                if path.suffix == ".md" and BRACED_LITERAL_STAR_SCRIPT.search(line):
                    errors.append(
                        f"{rel}:{lineno} uses a literal star script in rendered math; write ^{{\\ast}} or _{{\\ast}}"
                    )
                if EMPTY_SCRIPT_TARGET.search(line):
                    errors.append(
                        f"{rel}:{lineno} has a superscript/subscript marker with no target"
                    )

            if stripped.count("$$") % 2 == 1:
                in_display_math = not in_display_math
    return errors


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

    errors.extend(display_math_errors())
    errors.extend(markdown_table_errors())
    errors.extend(fragile_math_delimiter_errors())
    errors.extend(math_script_errors())

    if errors:
        print("Catalog validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Catalog validation passed for {len(links)} tutorials.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
