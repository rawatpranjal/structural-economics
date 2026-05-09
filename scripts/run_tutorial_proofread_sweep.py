#!/usr/bin/env python3
"""Sequentially proofread each tutorial via a Claude (Sonnet) subagent.

Each tutorial gets a non-interactive `claude --print` run that reads the
tutorial, fetches each cited reference (WebSearch / WebFetch), and writes a
structured report to `docs/qc-reports/proofread/{slug}.md`. The runner does
not commit or edit any tutorial; it only writes reports and a manifest.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "qc-reports" / "proofread"
MANIFEST = ROOT / "docs" / "qc-reports" / "proofread-manifest.json"
PASS_NAME = "tutorial-proofread-sweep-v1"
CATALOG_LINK_RE = re.compile(r"\]\(([^)#]+/)\)")
DEFAULT_MODEL = "claude-sonnet-4-6"
MIN_OUTPUT_BYTES = 500


def utc_now() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def catalog_tutorials() -> list[Path]:
    """Return active tutorial directories in root README catalog order."""
    text = (ROOT / "README.md").read_text()
    tutorials: list[Path] = []
    seen: set[Path] = set()

    for raw in CATALOG_LINK_RE.findall(text):
        if raw.startswith(("http://", "https://")):
            continue
        path = (ROOT / raw).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        if "_legacy" in path.relative_to(ROOT).parts:
            continue
        if path in seen or not (path / "run.py").exists():
            continue
        seen.add(path)
        tutorials.append(path)

    return tutorials


def rel_tutorial(path: Path) -> str:
    """Return a POSIX tutorial path with the trailing slash used in the catalog."""
    return path.relative_to(ROOT).as_posix() + "/"


def slug_for(tutorial: Path) -> str:
    """Filename-safe slug derived from the tutorial path."""
    return rel_tutorial(tutorial).rstrip("/").replace("/", "-")


def output_path_for(tutorial: Path) -> Path:
    return OUTPUT_DIR / f"{slug_for(tutorial)}.md"


def load_manifest() -> dict[str, Any]:
    """Load or initialize the proofread sweep manifest."""
    if not MANIFEST.exists():
        return {
            "pass_name": PASS_NAME,
            "updated_at": utc_now(),
            "entries": [],
        }
    data = json.loads(MANIFEST.read_text())
    if data.get("pass_name") != PASS_NAME:
        raise SystemExit(
            f"Manifest pass_name is {data.get('pass_name')!r}; expected {PASS_NAME!r}"
        )
    if "entries" not in data or not isinstance(data["entries"], list):
        raise SystemExit("Manifest must contain an entries list")
    return data


def write_manifest(data: dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = utc_now()
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")


def manifest_entry(data: dict[str, Any], tutorial_path: str) -> dict[str, Any] | None:
    for entry in data["entries"]:
        if entry.get("tutorial_path") == tutorial_path:
            return entry
    return None


def is_complete(data: dict[str, Any], tutorial: Path) -> bool:
    """Tutorial counts as complete if manifest says so and output file is healthy."""
    entry = manifest_entry(data, rel_tutorial(tutorial))
    if not entry or entry.get("status") != "complete":
        return False
    out = output_path_for(tutorial)
    if not out.exists():
        return False
    try:
        return out.stat().st_size >= MIN_OUTPUT_BYTES
    except OSError:
        return False


def is_failed(data: dict[str, Any], tutorial: Path) -> bool:
    entry = manifest_entry(data, rel_tutorial(tutorial))
    return bool(entry and entry.get("status") == "failed")


def queued_tutorials(start_index: int, limit: int | None, retry_failed: bool) -> list[Path]:
    if start_index < 1:
        raise SystemExit("--start-index must be >= 1")
    data = load_manifest()
    queue = catalog_tutorials()[start_index - 1 :]
    queue = [
        path
        for path in queue
        if not is_complete(data, path) and (retry_failed or not is_failed(data, path))
    ]
    if limit is not None:
        if limit < 0:
            raise SystemExit("--limit must be nonnegative")
        queue = queue[:limit]
    return queue


def record_complete(tutorial: Path, output_path: Path, model: str) -> None:
    data = load_manifest()
    rel = rel_tutorial(tutorial)
    entries = [e for e in data["entries"] if e.get("tutorial_path") != rel]
    entries.append(
        {
            "tutorial_path": rel,
            "status": "complete",
            "output_path": output_path.relative_to(ROOT).as_posix(),
            "output_bytes": output_path.stat().st_size,
            "model": model,
            "updated_at": utc_now(),
        }
    )
    data["entries"] = entries
    write_manifest(data)


def record_failed(tutorial: Path, error: str, exit_code: int, model: str) -> None:
    data = load_manifest()
    rel = rel_tutorial(tutorial)
    entries = [e for e in data["entries"] if e.get("tutorial_path") != rel]
    entries.append(
        {
            "tutorial_path": rel,
            "status": "failed",
            "error": error,
            "exit_code": exit_code,
            "model": model,
            "updated_at": utc_now(),
        }
    )
    data["entries"] = entries
    write_manifest(data)


def build_prompt(tutorial: Path, model: str) -> str:
    rel = rel_tutorial(tutorial)
    slug = slug_for(tutorial)
    output_rel = output_path_for(tutorial).relative_to(ROOT).as_posix()

    return dedent(
        f"""\
        You are working in {ROOT}.
        Public project name: Computational Economics.

        You are a proofreading subagent. Your single job is to produce a
        structured proofreading report for ONE tutorial and write it to disk.

        Target tutorial: {rel}
        Output file (write exactly this one file): {output_rel}

        You MAY use:
          - Read (any file in this repo)
          - WebSearch and WebFetch (to look up cited papers)
          - Write (only for the output file above)
          - Bash for read-only commands like `grep`, `wc`, `ls`

        You MUST NOT:
          - Edit any tutorial file. The README is auto-generated from run.py;
            hand edits would be wiped on the next regeneration.
          - Run `python run.py` or any tutorial code.
          - Edit any file other than {output_rel}.
          - Create any new file or directory anywhere in the repo. The only
            file you may write is {output_rel}.
          - Run any git command. Do not stage, commit, push, or modify branches.

        Files to read first:
          - {rel}README.md  (the tutorial prose)
          - {rel}run.py     (so you can verify the README's claims match the code)

        ====================================================================
        TASK 1 - PAPER / SOURCE VERIFICATION
        ====================================================================
        Find the "## References" block in {rel}README.md. For EACH cited reference:

          (a) WebSearch by title + first author + year. Prefer publisher /
              arXiv / NBER / SSRN / journal pages over secondary citations.
          (b) WebFetch the most authoritative hit. Two or three searches per
              paper is enough; do not chase down every ranking.
          (c) Compare the tutorial's description of the paper to the source.
              Look for: wrong dates, wrong authors, misattributed equations,
              claims the paper does not actually make, the wrong paper being
              cited for a result.
          (d) If you cannot locate the paper after a reasonable search,
              record "NOT FOUND". Do NOT fabricate a URL or a verdict.

        Verdicts: OK | MINOR | MAJOR | NOT FOUND
          - OK: tutorial's description matches the source.
          - MINOR: cosmetic mismatch (wrong page range, wrong issue number).
          - MAJOR: substantive misclaim (wrong result, wrong attribution).
          - NOT FOUND: paper could not be located.

        ====================================================================
        TASK 2 - MAIN MESSAGE AUDIT
        ====================================================================
        Identify the README's stated takeaway / key insight (usually in the
        Overview and Takeaway sections). Quote it. Then check whether each
        clause is actually supported by the README's Equations, Solution
        Method, Model Setup, or Results sections.

        Verdicts: OK | OVERREACH | UNSUPPORTED
          - OK: the claim follows from what the README shows.
          - OVERREACH: true in spirit but the tutorial does not demonstrate it.
          - UNSUPPORTED: the README implies something its own equations or
            results do not show.

        ====================================================================
        TASK 3 - NOTATION COMPLETENESS
        ====================================================================
        Enumerate every notation symbol in the README:
          - Greek letters (beta, sigma, etc.)
          - Latin variables (state, control, parameters)
          - function names (V, c*, g, u, T, ...)
          - non-obvious operators

        For each, locate its first appearance and where it is defined. Flag:
          - undefined symbols
          - symbols defined AFTER first use
          - the same symbol used for two different objects
          - notation that drifts between sections (e.g. R in Equations vs
            Setup vs Results meaning different things)

        Internal completeness only. Do NOT compare against external papers
        or "standard conventions" beyond the README itself.

        Do NOT flag LaTeX formatting style as a notation issue. The repo's
        math style is intentional. Spacing commands (`\;`, `\,`, `\!`,
        `\quad`), operator-style commands (`\operatorname` vs `\mathrm`),
        delimiter style (`\left(...\right)` vs plain), `\frac` vs `\dfrac`,
        line-break / alignment choices in display math, and similar
        cosmetic choices are out of scope. Flag a math issue ONLY when:
          - a symbol is undefined, late-defined, overloaded, or drifts;
          - a LaTeX token is literally broken (e.g. `\pprox` produced by a
            Python escape bug, an unclosed `$$` block).

        ====================================================================
        OUTPUT FORMAT - use this structure verbatim
        ====================================================================

        # Proofread: {rel}

        _Model: {model}. Generated: <ISO 8601 UTC timestamp>._

        ## Paper / Source Verification

        For each cited reference, one H3 subsection:

        ### <Citation as it appears in the README>

        - **Located:** <URL or "NOT FOUND">
        - **Tutorial claims:** <one or two sentences quoting / paraphrasing the README>
        - **Source says:** <one or two sentences from the source>
        - **Verdict:** OK | MINOR | MAJOR | NOT FOUND
        - **Note:** <one line>

        ## Main Message Audit

        > <quoted main message from the README>

        | Clause | Supported by | Verdict |
        |--------|--------------|---------|
        | ...    | Equations / Method / Results | OK / OVERREACH / UNSUPPORTED |

        Issues:
        - <list any OVERREACH / UNSUPPORTED clauses with one-line explanation>

        ## Notation Completeness

        | Symbol | First appearance | Defined? | Notes |
        |--------|------------------|----------|-------|
        | ...    | ...              | ...      | ...   |

        Flagged issues:
        - <undefined / late-defined / overloaded / drifting symbols>

        ## Summary

        <one paragraph: overall verdict, count of issues by severity (e.g.
        "1 MAJOR, 2 MINOR, 0 NOT FOUND, 1 OVERREACH"), and the single most
        important fix.>

        ====================================================================
        ====================================================================
        ADDITIONAL HARD RULES
        ====================================================================

        Strictness ladder for verdicts:
          - When in doubt between OK and MINOR, choose OK.
          - When in doubt between MINOR and MAJOR, choose MINOR.
          - Strictness wastes downstream work. The fix sweep skips MINOR
            cosmetic items, so over-flagging just clutters the report.

        Do not recommend rewrites:
          - Do NOT recommend prose rewrites, restructuring, paragraph
            reorganization, or stylistic edits. Your job is to identify
            problems, not to propose how the prose should read.
          - The "Note" field may state a CONCRETE replacement (e.g. "the
            correct DOI is X", "the symbol should be a' to match the
            Bellman"). It must NOT contain editorial advice ("would read
            better if...", "consider rephrasing...").

        Paper verification discipline:
          - At most 3 web searches per cited reference. If you cannot
            locate the paper after that, record "NOT FOUND" and move on.
          - Do NOT fabricate URLs, DOIs, page ranges, or any bibliographic
            field. If you do not find an authoritative source, the field
            is "NOT FOUND" or "unverified".
          - Cosmetic mismatches (issue number off by one, page range off
            by one, edition year off by one when the DOI still resolves
            correctly) are MINOR. Do not classify them MAJOR.

        Notation discipline:
          - Flag a symbol ONLY when it is undefined, defined after first
            use, used for two distinct objects, or drifts between
            sections. Do not flag standard abbreviations (i.i.d., AR(1),
            CRRA, MPC, GE) or common notation that the audience would
            know.
          - If a symbol is defined in a table within 50 lines of its
            first use in prose, classify as "Partial" or "Acceptable",
            not as a flagged issue.

        Output discipline:
          - Stick to the four prescribed sections. Do not add any other
            section, preamble, appendix, or footer.
          - Do not write executive summaries, recommendations,
            conclusions, or "next steps" outside the prescribed Summary
            paragraph.
          - The Summary paragraph is one paragraph. Not multiple.

        Markdown rendering hygiene (the report must pass
        scripts/validate_catalog.py, which lints every .md in the repo):
          - When transcribing math from the tutorial, write star scripts
            in braced form: `$a^{{\\ast}}(s)$`, `$g^{{\\ast}}(k)$`, `$x_{{\\ast}}$`.
            NEVER write unbraced `^*` or `_*` in any inline-math span;
            the validator rejects them.
          - When quoting a tutorial parameter-table row inside a Notation
            table cell (e.g. `"Discount factor $\\beta$ | 0.95"`), escape
            the inline pipe as `\\|` so the markdown table parser does not
            split the cell. Example: `"Discount factor $\\beta$ \\| 0.95"`.
          - Do not introduce literal `*` or `_` characters as math
            scripts in any other form. If unsure, use `\\ast`.

        After you finish writing the file, stop. Do not summarize in chat
        output. Do not edit the tutorial. Do not run python. Do not commit.
        """
    )


def run_claude(tutorial: Path, claude_bin: str, model: str) -> int:
    cmd = [
        claude_bin,
        "--print",
        "--model",
        model,
        "--dangerously-skip-permissions",
        "--permission-mode",
        "bypassPermissions",
    ]
    print(f"\n=== Proofreading {rel_tutorial(tutorial)} (model={model}) ===", flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        input=build_prompt(tutorial, model),
        text=True,
    ).returncode


def dry_run(limit: int | None, start_index: int, retry_failed: bool) -> int:
    tutorials = catalog_tutorials()
    data = load_manifest()
    selected = set(queued_tutorials(start_index, limit, retry_failed))
    completed_count = sum(1 for t in tutorials if is_complete(data, t))
    failed_count = sum(1 for t in tutorials if is_failed(data, t))

    print(f"{len(tutorials)} active tutorial(s) in root README order.")
    print(f"{completed_count} complete in {MANIFEST.relative_to(ROOT)}.")
    print(f"{failed_count} previously failed.")
    print(f"{len(selected)} tutorial(s) selected for the next run.")
    for index, tutorial in enumerate(tutorials, start=1):
        rel = rel_tutorial(tutorial)
        if is_complete(data, tutorial):
            status = "complete"
        elif tutorial in selected:
            status = "queued"
        elif is_failed(data, tutorial):
            status = "failed"
        else:
            status = "not selected"
        print(f"  {index:02d}. [{status}] {rel}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially proofread each tutorial via Claude (Sonnet).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print catalog order and queue without invoking Claude.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tutorials to process this run.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based catalog position to start from before skipping completed entries.",
    )
    parser.add_argument(
        "--claude-bin",
        default="claude",
        help="Claude executable to invoke.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-process entries previously recorded as failed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run:
        return dry_run(args.limit, args.start_index, args.retry_failed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queue = queued_tutorials(args.start_index, args.limit, args.retry_failed)
    if not queue:
        print("No tutorials queued for the proofread sweep.")
        return 0

    for index, tutorial in enumerate(queue, start=1):
        rel = rel_tutorial(tutorial)
        out = output_path_for(tutorial)
        print(f"[{index}/{len(queue)}] {rel}", flush=True)

        code = run_claude(tutorial, args.claude_bin, args.model)
        if code != 0:
            err = f"claude exited with code {code}"
            print(f"  FAILED: {err}", file=sys.stderr)
            record_failed(tutorial, err, code, args.model)
            continue

        if not out.exists():
            err = f"output file missing: {out.relative_to(ROOT)}"
            print(f"  FAILED: {err}", file=sys.stderr)
            record_failed(tutorial, err, 0, args.model)
            continue

        size = out.stat().st_size
        if size < MIN_OUTPUT_BYTES:
            err = f"output file too small ({size} bytes < {MIN_OUTPUT_BYTES})"
            print(f"  FAILED: {err}", file=sys.stderr)
            record_failed(tutorial, err, 0, args.model)
            continue

        record_complete(tutorial, out, args.model)
        remaining = len(queue) - index
        print(f"  ok ({size} bytes); {remaining} remain.")
        if remaining:
            print(f"  next: {rel_tutorial(queue[index])}")

    print("Proofread sweep finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
