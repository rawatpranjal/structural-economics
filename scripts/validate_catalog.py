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
FRAGILE_MATH_SIZE_COMMANDS = tuple(
    "\\" + command for command in ("bigl", "bigr", "Bigl", "Bigr")
)
FRAGILE_MATH_SPACING_COMMANDS = ("\\;", "\\!", "\\,")
UNSUPPORTED_MATH_COMMANDS = ("\\operatorname",)
# Inline display math: `$$` is not the only thing on its line.
INLINE_DISPLAY_MATH = re.compile(r"^(?!\s*\$\$\s*$).*\$\$")
# Math-y tokens inside code fences (LaTeX braces or `$`).
CODE_FENCE_MATH_TOKEN = re.compile(r"\$|_\{|\^\{")
# Prose-language code fences where pseudocode lives (no real-code lang tag).
PROSE_CODE_FENCE_LANGS = {"", "text", "pseudo", "pseudocode", "algorithm", "plain"}
# Greek-letter spellings that should not stand alone as a column header.
GREEK_NAMES = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
}
# Short acronyms that should not stand alone as a column header.
BARE_ACRONYMS = {
    "OLS", "IV", "2SLS", "GMM", "MLE", "KL", "HHI", "MSE", "RMSE", "FOC",
    "MPC", "DGP", "PDF", "CDF", "AR", "MA", "VAR", "DSGE", "BLP",
}
# Header cell deemed "code-style": lowercase identifier with underscore/digit.
CODE_STYLE_HEADER = re.compile(r"^[a-z][a-z0-9]*(?:[_/][a-z0-9]+)+$")
UNBRACED_MATHBB = re.compile(r"\\mathbb\s+[A-Za-z]")
UNBRACED_STAR_SCRIPT = re.compile(r"(?<!\\)(\^|_)\*")
BRACED_LITERAL_STAR_SCRIPT = re.compile(r"(?<!\\)(\^|_)\{\*\}")
EMPTY_SCRIPT_TARGET = re.compile(r"(?<!\\)(\^|_)(?:\s|$|[,$.;:)\]}]|[\^_])")


def is_python_string_close(line: str) -> bool:
    """Return whether a Python line only closes a multiline string/call."""
    stripped = line.strip()
    return stripped in {'"""', "'''", '""")', "''')"}


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
            # Skip audit/QC reports: they quote broken claim-source math by
            # design and are not catalog content.
            if path.name.startswith("bullshit-detector_"):
                continue
            if "qc-reports" in rel.parts:
                continue
            files.append(path)
    return sorted(files)


def display_math_errors() -> list[str]:
    """Find display-math blocks that are likely to be misparsed by Markdown."""
    errors = []
    for path in active_text_files():
        rel = path.relative_to(ROOT)
        lines = path.read_text(errors="replace").splitlines()
        in_math = False
        start_line: int | None = None
        for lineno, line in enumerate(lines, start=1):
            if line.strip() == "$$":
                if in_math:
                    next_line = lines[lineno] if lineno < len(lines) else ""
                    if (
                        next_line.strip()
                        and not (path.suffix == ".py" and is_python_string_close(next_line))
                    ):
                        errors.append(
                            f"{rel}:{lineno} display math block must be followed by a blank line"
                        )
                    in_math = False
                    start_line = None
                else:
                    if lineno > 1 and lines[lineno - 2].strip():
                        errors.append(
                            f"{rel}:{lineno} display math block must be preceded by a blank line"
                        )
                    in_math = True
                    start_line = lineno
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
            for command in FRAGILE_MATH_SIZE_COMMANDS:
                if command in line:
                    errors.append(
                        f"{rel}:{lineno} uses GitHub-fragile math delimiter command {command}; use plain delimiters instead"
                    )
    return errors


def fragile_math_command_errors() -> list[str]:
    """Reject math commands that have failed in the target Markdown renderer."""
    errors = []
    for path in active_text_files():
        if path.suffix == ".py" and path.name != "run.py":
            continue
        rel = path.relative_to(ROOT)
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            bare = re.sub(r"`[^`]*`", "", line) if path.suffix == ".md" else line
            for command in FRAGILE_MATH_SPACING_COMMANDS:
                if command in bare:
                    errors.append(
                        f"{rel}:{lineno} uses renderer-fragile math spacing command {command}; remove it"
                    )
            for command in UNSUPPORTED_MATH_COMMANDS:
                if command in line:
                    errors.append(
                        f"{rel}:{lineno} uses unsupported math command {command}; use \\mathrm{{...}}"
                    )
            if UNBRACED_MATHBB.search(line):
                errors.append(
                    f"{rel}:{lineno} uses unbraced \\mathbb; write \\mathbb{{E}} or \\mathbb{{R}}"
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


def inline_display_math_errors() -> list[str]:
    """Reject `$$...$$` blocks that share a line with prose.

    GitHub markdown often falls back to text when `$$` opens or closes inline
    with surrounding prose. The fallback then runs through markdown's
    italic/underscore processing and corrupts subscripts.
    """
    errors = []
    for path in active_text_files():
        if path.suffix != ".md":
            continue
        rel = path.relative_to(ROOT)
        in_fence = False
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            # Strip inline code spans so backtick-quoted `$$...$$` references
            # in prose (e.g. in CLAUDE.md) don't trip the check.
            bare = re.sub(r"`[^`]*`", "", line)
            if "$$" not in bare:
                continue
            if bare.strip() == "$$":
                continue
            if bare.count("$$") >= 2:
                m = re.match(r"^\s*\$\$.*\$\$\s*$", bare)
                if m:
                    continue
            errors.append(
                f"{rel}:{lineno} `$$` must be on its own line; prose or punctuation must not share the line"
            )
    return errors


def pseudocode_math_errors() -> list[str]:
    """Reject LaTeX-style math inside prose-language code fences.

    Pseudocode blocks in tutorial READMEs must read as plain text. LaTeX
    macros, `$`, and brace-grouped subscripts/superscripts belong in
    `$$...$$` blocks, not in `\`\`\`text` fences.
    """
    errors = []
    for path in active_text_files():
        if path.suffix != ".md":
            continue
        rel = path.relative_to(ROOT)
        in_fence = False
        fence_lang = ""
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_fence:
                    in_fence = False
                    fence_lang = ""
                else:
                    in_fence = True
                    fence_lang = stripped[3:].strip().lower()
                continue
            if not in_fence:
                continue
            if fence_lang not in PROSE_CODE_FENCE_LANGS:
                continue
            if CODE_FENCE_MATH_TOKEN.search(line):
                errors.append(
                    f"{rel}:{lineno} pseudocode block contains math notation (`$`, `_{{...}}`, or `^{{...}}`); rewrite as plain prose"
                )
    return errors


def split_markdown_table_cells(row: str) -> list[str]:
    """Split a Markdown table row into trimmed cell contents."""
    body = row.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def header_cell_is_bad(cell: str) -> bool:
    """Return whether a header cell looks like a raw code/symbol identifier."""
    if not cell:
        return False
    # Strip a wrapping pair of backticks or `$...$` so we evaluate the content.
    plain = cell
    if plain.startswith("`") and plain.endswith("`") and len(plain) >= 2:
        plain = plain[1:-1].strip()
    # Header containing English words alongside symbols/acronyms is fine.
    if re.search(r"[A-Za-z]{4,}", plain) and " " in plain:
        return False
    # Pure-math header like `$\alpha$` or `$\beta_{\text{sugar}}$` is fine.
    if plain.startswith("$") and plain.endswith("$"):
        return False
    lower = plain.lower()
    if lower in GREEK_NAMES:
        return True
    if plain in BARE_ACRONYMS:
        return True
    if CODE_STYLE_HEADER.match(plain):
        return True
    return False


def table_header_errors() -> list[str]:
    """Reject Markdown table headers that read as code identifiers."""
    errors = []
    for path in active_text_files():
        if path.suffix != ".md":
            continue
        rel = path.relative_to(ROOT)
        lines = path.read_text(errors="replace").splitlines()
        in_fence = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if i + 1 >= len(lines):
                continue
            if not is_markdown_table_separator(lines[i + 1]):
                continue
            if "|" not in line:
                continue
            for cell in split_markdown_table_cells(line):
                if header_cell_is_bad(cell):
                    errors.append(
                        f"{rel}:{i + 1} table header cell `{cell}` reads as a code identifier; use English with the symbol in parentheses (e.g. `Price coefficient ($\\alpha$)`)"
                    )
    return errors


def _run_self_tests() -> None:
    """Smoke fixtures for each new check. Raises on regression."""
    assert INLINE_DISPLAY_MATH.search("quality: $$x = 1$$.")
    assert not INLINE_DISPLAY_MATH.search("$$")
    assert CODE_FENCE_MATH_TOKEN.search("p_{rho, gamma}(j)")
    assert CODE_FENCE_MATH_TOKEN.search("$x$")
    assert not CODE_FENCE_MATH_TOKEN.search("theta_hat <- argmax f(theta)")
    assert header_cell_is_bad("alpha")
    assert header_cell_is_bad("beta_sugar")
    assert header_cell_is_bad("gamma2/gamma1")
    assert header_cell_is_bad("OLS")
    assert not header_cell_is_bad("Price coefficient ($\\alpha$)")
    assert not header_cell_is_bad("$\\alpha$")
    assert not header_cell_is_bad("Iterations")
    assert not header_cell_is_bad("OLS estimate")


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
    errors.extend(fragile_math_command_errors())
    errors.extend(math_script_errors())
    errors.extend(inline_display_math_errors())
    errors.extend(pseudocode_math_errors())
    errors.extend(table_header_errors())

    if errors:
        print("Catalog validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Catalog validation passed for {len(links)} tutorials.")
    return 0


if __name__ == "__main__":
    _run_self_tests()
    sys.exit(validate())
