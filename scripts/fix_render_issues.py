#!/usr/bin/env python3
"""One-shot mechanical fixer for two recurring rendering bugs.

1. Removes all `\\,` (thin space) macros from active Markdown.
2. Reformats inline `$$...$$` so `$$` always sits alone on its line,
   with blank lines before and after the block.

Only touches `.md` files under the repo root, skipping `.git/` and `_legacy/`.
Run once, inspect diff, then commit.
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


def strip_thin_space(text: str) -> str:
    """Delete every `\\,` in the file. CLAUDE.md forbids it outright now."""
    return text.replace("\\,", "")


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
        # Ignore `$$` that lives inside an inline code span.
        bare = re.sub(r"`[^`]*`", "", line)
        if "$$" not in bare:
            out.append(line)
            i += 1
            continue
        # Collect a full $$...$$ block, possibly multi-line.
        # Find positions of $$ in this line.
        first = line.find("$$")
        before = line[:first]
        # Walk forward to find closing $$ (may be on a later line).
        body_parts: list[str] = []
        after_first = line[first + 2 :]
        rest = after_first
        line_idx = i
        end_line_idx = None
        end_after = ""
        # search for closing $$ starting from after the opening
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
                # Unterminated; bail out, emit original
                out.extend(lines[i:])
                return "\n".join(out)
            rest = lines[line_idx]
        # Now body_parts is the math content as joined newline-separated chunks.
        body = "\n".join(body_parts).strip("\n")
        # Emit: (before, blank, $$, body, $$, blank, end_after)
        before_stripped = before.rstrip()
        if before_stripped:
            out.append(before_stripped)
        # Ensure blank line before $$
        if out and out[-1].strip():
            out.append("")
        out.append("$$")
        for bl in body.split("\n"):
            out.append(bl)
        out.append("$$")
        # Ensure blank line after $$ (will be enforced by checking next emit).
        end_after_stripped = end_after.lstrip()
        # Move past the consumed lines
        i = end_line_idx + 1
        if end_after_stripped:
            # If trailing prose remains, drop a blank then emit it.
            out.append("")
            out.append(end_after_stripped)
        else:
            # Ensure following line is blank
            if i < len(lines) and lines[i].strip():
                out.append("")
    # Collapse runs of 3+ blank lines to 2
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
            # Iteratively flatten nested braces.
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


def fix_escaped_curly_in_math(text: str) -> str:
    """Replace `\\{` -> `\\lbrace` and `\\}` -> `\\rbrace` inside math regions
    only. Also rewrites `\\big\\{` family to `\\big\\lbrace` family."""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    in_display = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        new_line_parts: list[str] = []
        i = 0
        # Track inline $ state per character to scope substitutions.
        inline = False
        local_display = in_display
        n = len(line)
        buf_start = 0
        while i < n:
            two = line[i:i + 2]
            ch = line[i]
            # Skip code spans.
            if ch == "`":
                # Find closing backtick.
                end = line.find("`", i + 1)
                if end == -1:
                    end = n
                else:
                    end += 1
                # Flush prior buffer with substitution if in math.
                segment = line[buf_start:i]
                new_line_parts.append(
                    _curly_sub(segment) if (inline or local_display) else segment
                )
                new_line_parts.append(line[i:end])
                i = end
                buf_start = i
                continue
            if two == "$$":
                segment = line[buf_start:i]
                new_line_parts.append(
                    _curly_sub(segment) if (inline or local_display) else segment
                )
                new_line_parts.append("$$")
                local_display = not local_display
                i += 2
                buf_start = i
                continue
            if ch == "$" and (i == 0 or line[i - 1] != "\\"):
                segment = line[buf_start:i]
                new_line_parts.append(
                    _curly_sub(segment) if (inline or local_display) else segment
                )
                new_line_parts.append("$")
                inline = not inline
                i += 1
                buf_start = i
                continue
            i += 1
        # Flush remainder.
        segment = line[buf_start:]
        new_line_parts.append(
            _curly_sub(segment) if (inline or local_display) else segment
        )
        out.append("".join(new_line_parts))
        in_display = local_display
    return "\n".join(out)


def _curly_sub(s: str) -> str:
    """Apply `\\{` -> `\\lbrace`, `\\}` -> `\\rbrace`, and the `\\big`
    family of sized delimiters wrapping curly braces. Insert a space
    after `\\lbrace`/`\\rbrace` when followed by a letter so KaTeX
    does not parse `\\lbracef` as one command."""
    s = re.sub(r"\\(big|Big|bigg|Bigg)\\\{", r"\\\1\\lbrace", s)
    s = re.sub(r"\\(big|Big|bigg|Bigg)\\\}", r"\\\1\\rbrace", s)
    s = s.replace("\\{", "\\lbrace").replace("\\}", "\\rbrace")
    s = re.sub(r"\\(lbrace|rbrace)(?=[A-Za-z])", r"\\\1 ", s)
    return s


GREEK_GLUED_FIX = re.compile(
    r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|"
    r"lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
    r"Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|"
    r"Lambda|Mu|Nu|Xi|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega|"
    r"varepsilon|vartheta|varpi|varrho|varsigma|varphi)"
    r"([a-zA-Z])"
)


def fix_glued_greek(text: str) -> str:
    """Insert a space between a Greek macro and a following letter (round-1
    `\\,` removal sometimes glued `\\alpha\\,s` into `\\alphas`)."""
    return GREEK_GLUED_FIX.sub(r"\\\1 \2", text)


SINGLE_TOKEN_BRACE_SUB = re.compile(r"_\{([A-Za-z0-9]|\\[A-Za-z]+)\}")


def drop_single_token_subscript_braces(text: str) -> str:
    """Inside inline `$...$` math, rewrite `_{x}` -> `_x` when the
    subscript is a single token (letter, digit, or LaTeX macro)."""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    in_display = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        new_parts: list[str] = []
        i = 0
        n = len(line)
        inline = False
        local_display = in_display
        buf_start = 0
        while i < n:
            ch = line[i]
            two = line[i:i + 2]
            if ch == "`":
                end = line.find("`", i + 1)
                if end == -1:
                    end = n
                else:
                    end += 1
                seg = line[buf_start:i]
                new_parts.append(
                    SINGLE_TOKEN_BRACE_SUB.sub(r"_\1", seg) if (inline and not local_display) else seg
                )
                new_parts.append(line[i:end])
                i = end
                buf_start = i
                continue
            if two == "$$":
                seg = line[buf_start:i]
                new_parts.append(
                    SINGLE_TOKEN_BRACE_SUB.sub(r"_\1", seg) if (inline and not local_display) else seg
                )
                new_parts.append("$$")
                local_display = not local_display
                i += 2
                buf_start = i
                continue
            if ch == "$" and (i == 0 or line[i - 1] != "\\"):
                seg = line[buf_start:i]
                new_parts.append(
                    SINGLE_TOKEN_BRACE_SUB.sub(r"_\1", seg) if (inline and not local_display) else seg
                )
                new_parts.append("$")
                inline = not inline
                i += 1
                buf_start = i
                continue
            i += 1
        seg = line[buf_start:]
        new_parts.append(
            SINGLE_TOKEN_BRACE_SUB.sub(r"_\1", seg) if (inline and not local_display) else seg
        )
        out.append("".join(new_parts))
        in_display = local_display
    return "\n".join(out)


def fix_file(path: Path) -> bool:
    original = path.read_text()
    text = original
    text = strip_thin_space(text)
    text = reformat_dollar_blocks(text)
    text = strip_pseudocode_latex(text)
    text = brace_math_font_macros(text)
    text = fix_escaped_curly_in_math(text)
    text = fix_glued_greek(text)
    text = drop_single_token_subscript_braces(text)
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
