#!/usr/bin/env python3
"""Validate the root tutorial catalog and active-folder hygiene.

The catalog uses GitHub code-fence math syntax exclusively:

- Inline math:  ``$`expr`$``
- Display math: ```` ```math ```` ... ```` ``` ````

This bypasses GitHub's markdown sanitizer (which corrupts bare `$...$`
and `$$...$$` math) at the source. Bare `$...$` or `$$...$$` math
outside fenced code blocks is hard-rejected by `bare_dollar_math_errors`.

The KaTeX-genuine rules (math-font macros, star scripts, empty script
targets) still apply inside fenced math and are enforced.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
PROSE_CODE_FENCE_LANGS = {"", "text", "pseudo", "pseudocode", "algorithm", "plain"}
CODE_FENCE_MATH_TOKEN = re.compile(r"\$|_\{|\^\{")
GREEK_NAMES = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
}
BARE_ACRONYMS = {
    "OLS", "IV", "2SLS", "GMM", "MLE", "KL", "HHI", "MSE", "RMSE", "FOC",
    "MPC", "DGP", "PDF", "CDF", "AR", "MA", "VAR", "DSGE", "BLP",
}
CODE_STYLE_HEADER = re.compile(r"^[a-z][a-z0-9]*(?:[_/][a-z0-9]+)+$")
UNBRACED_MATH_FONT = re.compile(
    r"\\(mathbb|mathbf|mathcal|mathrm|mathfrak|mathit|mathsf|mathtt)\s+[A-Za-z0-9]"
)
UNBRACED_STAR_SCRIPT = re.compile(r"(?<!\\)(\^|_)\*")
BRACED_LITERAL_STAR_SCRIPT = re.compile(r"(?<!\\)(\^|_)\{\*\}")
EMPTY_SCRIPT_TARGET = re.compile(r"(?<!\\)(\^|_)(?:\s|$|[,$.;:)\]}]|[\^_])")
# Match the canonical inline code-fence math form `$`expr`$`.
INLINE_CODE_FENCE_MATH = re.compile(r"\$`[^`]+`\$")


def is_python_string_close(line: str) -> bool:
    stripped = line.strip()
    return stripped in {'"""', "'''", '""")', "''')"}


def catalog_links() -> set[Path]:
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
    bad = []
    for path in ROOT.rglob("*.ipynb"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "_legacy" in rel.parts:
            continue
        bad.append(rel)
    return sorted(bad)


def active_checkpoints() -> list[Path]:
    bad = []
    for path in ROOT.rglob(".ipynb_checkpoints"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "_legacy" in rel.parts:
            continue
        bad.append(rel)
    return sorted(bad)


def tutorial_dirs() -> set[Path]:
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
            if path.name.startswith("bullshit-detector_"):
                continue
            if "qc-reports" in rel.parts:
                continue
            files.append(path)
    return sorted(files)


def active_markdown_files() -> list[Path]:
    return [p for p in active_text_files() if p.suffix == ".md"]


def count_unescaped_pipes(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def is_markdown_table_separator(line: str) -> bool:
    if "|" not in line:
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in cells)


def markdown_table_errors() -> list[str]:
    errors = []
    for path in active_markdown_files():
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


def bare_dollar_math_errors() -> list[str]:
    """Hard-reject bare `$...$` or `$$...$$` math outside fenced code blocks.

    The catalog uses GitHub code-fence math syntax exclusively:
    ``$`expr`$`` inline and ```` ```math ```` for display. Bare-dollar
    math is corrupted by GitHub's pre-extraction markdown sanitizer and
    is not allowed anywhere in active content.

    Legitimate raw `$` survives:
    - Escaped (`\\$5`) — currency or shell literal
    - Inside an inline code span (`` `$x$` ``)
    - Inside a fenced code block (including ```` ```math ```` blocks)
    - The canonical ``$`expr`$`` inline-math form itself
    """
    errors = []
    for path in active_markdown_files():
        rel = path.relative_to(ROOT)
        lines = path.read_text(errors="replace").splitlines()
        in_fence = False
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            # Strip legitimate forms before checking for bare `$`.
            bare = INLINE_CODE_FENCE_MATH.sub("", line)
            bare = re.sub(r"`[^`]*`", "", bare)
            bare = bare.replace("\\$", "")
            if "$" in bare:
                errors.append(
                    f"{rel}:{lineno} bare `$` math outside code fence; use `$`expr`$` for inline math or a ```math fence for display"
                )
    return errors


def math_script_errors() -> list[str]:
    """Reject KaTeX-illegal script forms (unbraced star, empty target)."""
    errors = []
    for path in active_markdown_files():
        rel = path.relative_to(ROOT)
        lines = path.read_text(errors="replace").splitlines()
        in_fence = False
        fence_lang = ""
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_fence:
                    in_fence = False
                    fence_lang = ""
                else:
                    in_fence = True
                    fence_lang = stripped[3:].strip().lower()
                continue
            in_math_fence = in_fence and fence_lang == "math"
            # Inline math sites: `$`...`$`. Pull contents for checking.
            inline_bodies = [m.group(0) for m in INLINE_CODE_FENCE_MATH.finditer(line)]
            check_targets = []
            if in_math_fence:
                check_targets.append(line)
            check_targets.extend(inline_bodies)
            for target in check_targets:
                if UNBRACED_STAR_SCRIPT.search(target):
                    errors.append(
                        f"{rel}:{lineno} uses an unbraced star script in math; write ^{{\\ast}} or _{{\\ast}}"
                    )
                if BRACED_LITERAL_STAR_SCRIPT.search(target):
                    errors.append(
                        f"{rel}:{lineno} uses a literal star script in math; write ^{{\\ast}} or _{{\\ast}}"
                    )
                if EMPTY_SCRIPT_TARGET.search(target):
                    errors.append(
                        f"{rel}:{lineno} has a superscript/subscript marker with no target"
                    )
                if UNBRACED_MATH_FONT.search(target):
                    m = UNBRACED_MATH_FONT.search(target)
                    errors.append(
                        f"{rel}:{lineno} uses unbraced \\{m.group(1)}; brace the argument (e.g. \\{m.group(1)}{{X}})"
                    )
    return errors


def pseudocode_math_errors() -> list[str]:
    """Reject LaTeX-style math inside prose-language code fences."""
    errors = []
    for path in active_markdown_files():
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
    body = row.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def header_cell_is_bad(cell: str) -> bool:
    if not cell:
        return False
    plain = cell
    if plain.startswith("`") and plain.endswith("`") and len(plain) >= 2:
        plain = plain[1:-1].strip()
    if re.search(r"[A-Za-z]{4,}", plain) and " " in plain:
        return False
    # Code-fence inline math form `$`expr`$` is fine as a header.
    if INLINE_CODE_FENCE_MATH.fullmatch(plain):
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
    errors = []
    for path in active_markdown_files():
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
                        f"{rel}:{i + 1} table header cell `{cell}` reads as a code identifier; use English with the symbol in parentheses (e.g. `Price coefficient ($`\\alpha`$)`)"
                    )
    return errors


def _run_self_tests() -> None:
    """Smoke fixtures for each new check. Raises on regression."""
    assert CODE_FENCE_MATH_TOKEN.search("p_{rho, gamma}(j)")
    assert CODE_FENCE_MATH_TOKEN.search("$x$")
    assert not CODE_FENCE_MATH_TOKEN.search("theta_hat <- argmax f(theta)")
    assert header_cell_is_bad("alpha")
    assert header_cell_is_bad("beta_sugar")
    assert header_cell_is_bad("gamma2/gamma1")
    assert header_cell_is_bad("OLS")
    assert not header_cell_is_bad("Price coefficient ($`\\alpha`$)")
    assert not header_cell_is_bad("$`\\alpha`$")
    assert not header_cell_is_bad("Iterations")
    assert not header_cell_is_bad("OLS estimate")
    assert UNBRACED_MATH_FONT.search(r"\mathbf A")
    assert UNBRACED_MATH_FONT.search(r"\mathcal D")
    assert not UNBRACED_MATH_FONT.search(r"\mathbf{A}")
    # Bare-dollar smoke: an isolated `$x$` outside a code span trips the
    # rule; a wrapped `$`x`$` does not.
    assert INLINE_CODE_FENCE_MATH.search("$`x`$")
    assert not INLINE_CODE_FENCE_MATH.search("$x$")


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

    errors.extend(markdown_table_errors())
    errors.extend(bare_dollar_math_errors())
    errors.extend(math_script_errors())
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
