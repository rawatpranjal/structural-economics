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


def fix_file(path: Path) -> bool:
    original = path.read_text()
    text = original
    text = strip_thin_space(text)
    text = reformat_dollar_blocks(text)
    text = strip_pseudocode_latex(text)
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> int:
    changed = []
    for path in active_markdown_files():
        if fix_file(path):
            changed.append(path.relative_to(ROOT))
    for rel in changed:
        print(f"fixed: {rel}")
    print(f"\nTotal files changed: {len(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
