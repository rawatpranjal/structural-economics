#!/usr/bin/env python3
"""Mechanical fixer for catalog Markdown math.

Pipeline:

1. `reformat_dollar_blocks` -- canonicalize `$$...$$` so each `$$` sits
   alone on its line with surrounding blank lines. Required before the
   migration pass can see the canonical opening/closing form.
2. `migrate_to_code_fence` -- migrate every `$expr$` inline math to
   ``$`expr`$`` and every `$$...$$` display block to a ```` ```math ````
   fence. Code-fence math syntax bypasses GitHub's markdown sanitizer,
   which is the root cause of every recurring render bug previously
   patched in this script. The pass is idempotent: already-migrated
   sites are detected by the `$\\`` opening and the `\\`$` closing and
   passed through verbatim.
3. `brace_math_font_macros` -- wrap single-token arguments to math-font
   macros (`\\mathbb A` -> `\\mathbb{A}`). KaTeX requires braces even
   inside fenced math.
4. `strip_pseudocode_latex` -- strip LaTeX subscripts/superscripts and
   stray `$` from prose-language code fences (independent of math
   migration, kept for pseudocode hygiene).

The earlier sanitizer-survival passes (`strip_thin_space`,
`fix_escaped_curly_in_math`, `fix_glued_greek`,
`drop_single_token_subscript_braces`) were deleted with the migration.
They patched damage caused by GitHub's pre-extraction sanitizer; once
math lives inside code-fence syntax the sanitizer cannot touch it, and
running those rewrites on fenced math would corrupt legitimate LaTeX
(e.g. `\\,` is a real KaTeX thin space inside ```math fences).

Only touches `.md` files under the repo root, skipping `.git/` and
`_legacy/`. `.py` files are intentionally excluded: their `$...$`
strings drive matplotlib's mathtext renderer and must stay raw.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def active_markdown_files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "_legacy" in rel.parts:
            continue
        out.append(path)
    return sorted(out)


def reformat_dollar_blocks(text: str) -> str:
    """Rewrite every `$$...$$` block so `$$` sits alone on its line."""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence or "$$" not in line:
            out.append(line)
            i += 1
            continue
        bare = re.sub(r"`[^`]*`", "", line)
        if "$$" not in bare:
            out.append(line)
            i += 1
            continue
        first = line.find("$$")
        before = line[:first]
        body_parts: list[str] = []
        after_first = line[first + 2 :]
        rest = after_first
        line_idx = i
        end_line_idx = None
        end_after = ""
        while True:
            close_idx = rest.find("$$")
            if close_idx != -1:
                body_parts.append(rest[:close_idx])
                end_after = rest[close_idx + 2 :]
                end_line_idx = line_idx
                break
            body_parts.append(rest)
            line_idx += 1
            if line_idx >= len(lines):
                out.extend(lines[i:])
                return "\n".join(out)
            rest = lines[line_idx]
        body = "\n".join(body_parts).strip("\n")
        before_stripped = before.rstrip()
        if before_stripped:
            out.append(before_stripped)
        if out and out[-1].strip():
            out.append("")
        out.append("$$")
        for bl in body.split("\n"):
            out.append(bl)
        out.append("$$")
        end_after_stripped = end_after.lstrip()
        i = end_line_idx + 1
        if end_after_stripped:
            out.append("")
            out.append(end_after_stripped)
        else:
            if i < len(lines) and lines[i].strip():
                out.append("")
    cleaned: list[str] = []
    blank_run = 0
    for ln in out:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned.append(ln)
        else:
            blank_run = 0
            cleaned.append(ln)
    return "\n".join(cleaned)


def migrate_to_code_fence(text: str) -> str:
    """Migrate bare `$expr$` to ``$`expr`$`` and `$$...$$` blocks to
    ```math fences. Idempotent on already-migrated content."""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    in_display = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not in_display and stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if in_fence:
            out.append(line)
            i += 1
            continue

        if stripped == "$$":
            indent = line[: len(line) - len(line.lstrip())]
            if not in_display:
                out.append(f"{indent}```math")
                in_display = True
            else:
                out.append(f"{indent}```")
                in_display = False
            i += 1
            continue

        if in_display:
            out.append(line)
            i += 1
            continue

        single = re.match(r"^(\s*)\$\$(.+?)\$\$\s*$", line)
        if single:
            indent, expr = single.groups()
            out.append(f"{indent}```math")
            out.append(f"{indent}{expr.strip()}")
            out.append(f"{indent}```")
            i += 1
            continue

        out.append(_migrate_inline(line))
        i += 1
    return "\n".join(out)


def _migrate_inline(line: str) -> str:
    """Rewrite `$expr$` -> ``$`expr`$`` outside backtick code spans.

    Idempotent: a `$` followed immediately by a backtick (`$\\``) is
    recognised as the opening of an already-migrated inline math block;
    the walker skips to the matching `\\`$` and emits the whole span
    verbatim.
    """
    result: list[str] = []
    n = len(line)
    i = 0
    while i < n:
        ch = line[i]

        if ch == "`":
            run = i
            while run < n and line[run] == "`":
                run += 1
            tick_count = run - i
            ticks = "`" * tick_count
            search_from = run
            close = -1
            while search_from < n:
                k = line.find(ticks, search_from)
                if k == -1:
                    break
                end = k + tick_count
                if end == n or line[end] != "`":
                    close = k
                    break
                search_from = k + 1
                while search_from < n and line[search_from] == "`":
                    search_from += 1
            if close == -1:
                result.append(line[i:])
                return "".join(result)
            result.append(line[i : close + tick_count])
            i = close + tick_count
            continue

        if ch == "\\" and i + 1 < n and line[i + 1] == "$":
            result.append(line[i : i + 2])
            i += 2
            continue

        if ch == "$":
            # Already-migrated form: `$\`expr\`$`. Skip to closing `\`$`.
            if i + 1 < n and line[i + 1] == "`":
                k = i + 2
                close_tick = -1
                while k < n - 1:
                    if line[k] == "`" and line[k + 1] == "$":
                        close_tick = k
                        break
                    k += 1
                if close_tick != -1:
                    result.append(line[i : close_tick + 2])
                    i = close_tick + 2
                    continue

            j = i + 1
            close = -1
            while j < n:
                cj = line[j]
                if cj == "`":
                    break
                if cj == "\\" and j + 1 < n and line[j + 1] == "$":
                    j += 2
                    continue
                if cj == "$":
                    close = j
                    break
                j += 1
            if close == -1:
                result.append(ch)
                i += 1
                continue
            expr = line[i + 1 : close]
            if not expr:
                result.append("$$")
                i = close + 1
                continue
            if "`" in expr:
                result.append(line[i : close + 1])
                i = close + 1
                continue
            result.append(f"$`{expr}`$")
            i = close + 1
            continue

        result.append(ch)
        i += 1
    return "".join(result)


PROSE_FENCE_LANGS = {"", "text", "pseudo", "pseudocode", "algorithm", "plain"}


def strip_pseudocode_latex(text: str) -> str:
    """In prose-language code fences, replace LaTeX subscripts/superscripts
    with plain bracket form and strip stray `$` signs."""
    out: list[str] = []
    in_fence = False
    fence_lang = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = False
                fence_lang = ""
            else:
                in_fence = True
                fence_lang = stripped[3:].strip().lower()
            out.append(line)
            continue
        if in_fence and fence_lang in PROSE_FENCE_LANGS:
            new = line
            for _ in range(4):
                prev = new
                new = re.sub(r"_\{([^{}]*)\}", r"[\1]", new)
                new = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", new)
                if new == prev:
                    break
            new = new.replace("$", "")
            out.append(new)
        else:
            out.append(line)
    return "\n".join(out)


def brace_math_font_macros(text: str) -> str:
    """Brace single-token arguments to math-font macros across the file."""
    return re.sub(
        r"\\(mathbb|mathbf|mathcal|mathrm|mathfrak|mathit|mathsf|mathtt)\s+([A-Za-z0-9])",
        r"\\\1{\2}",
        text,
    )


def fix_file(path: Path) -> bool:
    original = path.read_text()
    text = original
    text = reformat_dollar_blocks(text)
    text = migrate_to_code_fence(text)
    text = brace_math_font_macros(text)
    text = strip_pseudocode_latex(text)
    if text != original:
        path.write_text(text)
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args:
        paths = [Path(a).resolve() for a in args]
        for p in paths:
            if not p.exists():
                print(f"error: not found: {p}", file=sys.stderr)
                return 2
    else:
        paths = active_markdown_files()
    changed = []
    for path in paths:
        if fix_file(path):
            try:
                rel = path.relative_to(ROOT)
            except ValueError:
                rel = path
            changed.append(rel)
    for rel in changed:
        print(f"fixed: {rel}")
    print(f"\nTotal files changed: {len(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
