#!/usr/bin/env python3
"""Apply egregious-only fixes from each tutorial's proofread report.

Reads `docs/qc-reports/proofread/{slug}.md`, dispatches a Claude (Sonnet)
subagent that edits `{tutorial}/run.py` to fix wrong DOIs / URLs / paper
metadata and missing or drifting notation. The runner then regenerates the
README via `python run.py` and validates via `scripts/validate_catalog.py`.
No git operations.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROOFREAD_DIR = ROOT / "docs" / "qc-reports" / "proofread"
MANIFEST = ROOT / "docs" / "qc-reports" / "fix-manifest.json"
PASS_NAME = "tutorial-fix-sweep-v1"
CATALOG_LINK_RE = re.compile(r"\]\(([^)#]+/)\)")
DEFAULT_MODEL = "claude-sonnet-4-6"
MIN_REPORT_BYTES = 500


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def catalog_tutorials() -> list[Path]:
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
    return path.relative_to(ROOT).as_posix() + "/"


def slug_for(tutorial: Path) -> str:
    return rel_tutorial(tutorial).rstrip("/").replace("/", "-")


def proofread_report_for(tutorial: Path) -> Path:
    return PROOFREAD_DIR / f"{slug_for(tutorial)}.md"


def has_proofread_report(tutorial: Path) -> bool:
    rep = proofread_report_for(tutorial)
    if not rep.exists():
        return False
    try:
        return rep.stat().st_size >= MIN_REPORT_BYTES
    except OSError:
        return False


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {"pass_name": PASS_NAME, "updated_at": utc_now(), "entries": []}
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
    entry = manifest_entry(data, rel_tutorial(tutorial))
    return bool(entry and entry.get("status") == "complete")


def is_failed(data: dict[str, Any], tutorial: Path) -> bool:
    entry = manifest_entry(data, rel_tutorial(tutorial))
    return bool(entry and entry.get("status") == "failed")


def queued_tutorials(start_index: int, limit: int | None, retry_failed: bool) -> list[Path]:
    if start_index < 1:
        raise SystemExit("--start-index must be >= 1")
    data = load_manifest()
    queue = catalog_tutorials()[start_index - 1 :]
    queue = [
        t
        for t in queue
        if has_proofread_report(t)
        and not is_complete(data, t)
        and (retry_failed or not is_failed(data, t))
    ]
    if limit is not None:
        if limit < 0:
            raise SystemExit("--limit must be nonnegative")
        queue = queue[:limit]
    return queue


def record_complete(
    tutorial: Path,
    model: str,
    regen_seconds: float,
    validate_ok: bool,
    summary_line: str,
) -> None:
    data = load_manifest()
    rel = rel_tutorial(tutorial)
    entries = [e for e in data["entries"] if e.get("tutorial_path") != rel]
    entries.append(
        {
            "tutorial_path": rel,
            "status": "complete",
            "model": model,
            "regen_seconds": round(regen_seconds, 2),
            "validate_ok": validate_ok,
            "summary": summary_line,
            "updated_at": utc_now(),
        }
    )
    data["entries"] = entries
    write_manifest(data)


def record_failed(
    tutorial: Path,
    model: str,
    stage: str,
    error: str,
    exit_code: int,
) -> None:
    data = load_manifest()
    rel = rel_tutorial(tutorial)
    entries = [e for e in data["entries"] if e.get("tutorial_path") != rel]
    entries.append(
        {
            "tutorial_path": rel,
            "status": "failed",
            "model": model,
            "stage": stage,
            "error": error,
            "exit_code": exit_code,
            "updated_at": utc_now(),
        }
    )
    data["entries"] = entries
    write_manifest(data)


def build_prompt(tutorial: Path, model: str) -> str:
    rel = rel_tutorial(tutorial)
    report_rel = proofread_report_for(tutorial).relative_to(ROOT).as_posix()

    return dedent(
        f"""\
        You are working in {ROOT}.
        Public project name: Computational Economics.

        You are a surgical-fix subagent. Your job is to apply ONLY the most
        egregious and obvious fixes from a proofread report to ONE tutorial.

        Target tutorial: {rel}
        Source of truth (the fix list): {report_rel}

        Read first, in this order:
          - CLAUDE.md
          - STYLE_GUIDE.md
          - GLOSSARY.md
          - {report_rel}        (the proofread report - your fix list)
          - {rel}run.py         (this is what you edit)
          - {rel}README.md      (current state; DO NOT edit, it is auto-regenerated)

        ====================================================================
        STRICT FIX SCOPE - apply ONLY these
        ====================================================================

        From the report's "## Paper / Source Verification" section:
          - FIX a citation when the URL/DOI resolves to a different paper than
            the citation describes (the smoke-test exemplar: a DOI that
            resolves to an unrelated article).
          - FIX a citation when the author, year, or title is wrong.
          - SKIP MINOR entries whose only complaint is wrong page range, wrong
            issue number, or other cosmetic metadata when the DOI still
            resolves to the right paper.
          - SKIP NOT FOUND entries. Do NOT fabricate a URL.

        From the report's "## Notation Completeness" section:
          - FIX every flagged issue: undefined symbol, late-defined symbol,
            same symbol used for two different objects, notation drift between
            sections.
          - For a missing definition, add a one-line definition near the
            symbol's first use (typically inside the Equations or Model Setup
            string built in run.py).
          - For drift between sections, pick one symbol and replace the other
            consistently across run.py.
          - You MAY fix a genuinely broken LaTeX token that the report flags
            (e.g. `\pprox` from a Python `\a` BEL escape, `\=` from an
            unrecognized escape). These are real bugs, not formatting.

        SKIP ENTIRELY:
          - The report's "## Main Message Audit" section. Do NOT edit prose
            to address OVERREACH or UNSUPPORTED clauses.
          - Anything not flagged by the report.

        If the report has no in-scope issues, do nothing and write a one-line
        "no in-scope issues" summary.

        ====================================================================
        DO NOT TOUCH LATEX FORMATTING
        ====================================================================

        Do NOT make any cosmetic LaTeX / math change. The repo's math style is
        intentional and is not in scope for this pass. In particular, do NOT:

          - Change spacing commands. Leave `\;`, `\,`, `\!`, `\quad`,
            `\qquad` exactly as they are. Do NOT convert `\;=\;` to `=`,
            do NOT add or remove `\,` between symbols.
          - Change operator-style commands. Leave `\operatorname{{...}}`,
            `\mathrm{{...}}`, `\mathbf{{...}}`, `\mathbb{{...}}`, `\mathcal{{...}}`
            exactly as written. Do NOT swap `\operatorname` for `\mathrm` or
            vice versa.
          - Change delimiter style. Leave existing left/right, angle-bracket,
            and sized-delimiter commands alone.
          - Change alignment, line breaks, or `&` placement inside `align`,
            `aligned`, `cases`, or any multi-line math environment.
          - Reflow display equations across lines, add or remove `\\`,
            re-bracket fractions, or change `^{{...}}` / `_{{...}}` braces.
          - Change `\frac` to `\dfrac` or vice versa.
          - "Improve" symbol choice when the report did not flag it
            (e.g. don't change `R` to `\mathcal{{R}}` because it looks nicer).

        Allowed math edits are limited to: (a) adding a one-line text
        definition near a symbol's first use, in the surrounding prose, NOT
        inside the equation; (b) renaming a symbol to resolve drift the
        report explicitly flagged; (c) fixing a literally broken LaTeX token
        the report flagged as broken.

        ====================================================================
        EDITING RULES
        ====================================================================

        - Edit ONLY {rel}run.py and tutorial-local helper modules under {rel}.
        - Do NOT create any new file anywhere in the repo. No new helper
          modules, no new tutorials, no new tests, no notes, nothing. If a
          fix would require a new file, skip the fix.
        - Do NOT create any new directory anywhere in the repo.
        - Do NOT edit any file outside {rel}. This is a hard constraint
          enforced by a post-run scope guard that reverts violations and
          marks the run failed.
        - Do NOT hand-edit {rel}README.md. The runner regenerates it.
        - Do NOT edit the proofread report or any manifest.
        - Do NOT run `python run.py`. The runner will run it.
        - Do NOT run any git command.
        - Do NOT run any other tutorial.
        - Be conservative. When in doubt, leave it alone.
        - Keep edits surgical: a corrected DOI string, a few new symbol
          definitions inserted into the existing Equations / Setup prose. No
          rewrites of Overview, Solution Method, Results, or Takeaway prose.

        Self-check before you finish: list every file you edited. If any
        path is outside {rel}, your work will be rejected and reverted. If
        you created any new file or directory, your work will be rejected.

        ====================================================================
        ADDITIONAL HARD RULES (applied on top of everything above)
        ====================================================================

        Code & calibration are frozen:
          - Do NOT touch numeric values, parameter dictionaries, calibration
            constants, grid sizes, tolerances, simulation horizons, or any
            other model parameter. The model output must be identical
            before and after the fix.
          - Do NOT touch imports, function definitions, function bodies,
            numerical kernels, plotting code (matplotlib calls, axis labels,
            colors, line styles, figure sizes, DPI), table-construction
            code, or any executable Python outside of the prose-building
            string concatenations passed to ModelReport.
          - Do NOT change the order in which sections are added to the
            report (`add_overview`, `add_equations`, ... ). Section
            structure is fixed.

        References are immutable except for flagged fixes:
          - Do NOT add a new reference. Even if you find a great paper
            during verification, do not add it.
          - Do NOT remove a reference.
          - Do NOT reorder references.
          - The ONLY allowed reference edit is rewriting the URL / DOI /
            citation text of a SPECIFIC reference the report flagged with
            verdict MAJOR or "wrong DOI/URL/author/year".

        Prose minimalism:
          - Do NOT reword existing sentences. The ONLY allowed prose
            change is INSERTING a new sentence (or a parenthetical clause
            attached to an existing sentence) that defines a flagged
            symbol or names a flagged term.
          - Do NOT delete existing prose, even if it seems redundant.
          - Do NOT split or merge paragraphs.
          - Do NOT change punctuation, capitalization, or wording outside
            the inserted sentence.

        Diff-size cap:
          - Each flagged item should land as 1-3 added lines. Any fix that
            requires more than ~5 added/changed lines per flagged item is
            out of scope for this pass - skip that item.
          - If your total diff exceeds ~25 lines across run.py, you are
            doing too much. Stop, undo half of it, and only keep the most
            unambiguous fixes.

        Tools:
          - You MAY use Read, Edit, and Bash for read-only commands (e.g.
            `grep`, `wc`).
          - You MUST NOT use the Write tool. Write creates or overwrites
            files, both of which are forbidden. Use Edit for existing
            files only.
          - You MUST NOT install packages, run tests, run linters, or run
            any tutorial code.

        References in run.py are typically markdown links of the form
        `[Citation text](https://doi.org/...)` inside a list of strings passed
        to ModelReport. To fix a wrong DOI, edit just the URL inside that
        link. To fix a wrong year/author/title in the citation text, edit
        only the citation text and keep the surrounding code untouched.

        ====================================================================
        OUTPUT
        ====================================================================

        After applying fixes, print exactly one line to stdout summarizing
        what you did, in this format:

          fix-summary: <N_ref> ref fix(es), <N_notation> notation fix(es)

        Or if nothing was in scope:

          fix-summary: no in-scope issues

        Then stop. Do not write any other artifact.
        """
    )


def git_porcelain_paths() -> set[str]:
    """Return the set of paths reported by `git status --porcelain`."""
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    paths: set[str] = set()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if path.endswith("/"):
            path = path.rstrip("/")
        paths.add(path)
    return paths


def in_scope(path: str, tutorial: Path) -> bool:
    """Whether a path is inside the tutorial folder or is the root README catalog row."""
    rel = rel_tutorial(tutorial)
    return path == "README.md" or path.startswith(rel) or path.startswith(rel.rstrip("/"))


def revert_out_of_scope(paths: list[str]) -> list[str]:
    """Revert tracked modifications at given paths.

    Untracked files/dirs are NOT deleted, because they may belong to a
    concurrent session working on a different tutorial. The earlier behaviour
    of unlink()/rmtree() on untracked paths could clobber another agent's
    in-flight work and is no longer safe.
    """
    if not paths:
        return []
    subprocess.run(["git", "restore", "--", *paths], cwd=ROOT, check=False)
    return paths


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
    print(f"\n=== Fix sweep: {rel_tutorial(tutorial)} (model={model}) ===", flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        input=build_prompt(tutorial, model),
        text=True,
    ).returncode


def regen_readme(tutorial: Path) -> tuple[int, float, str]:
    """Run `python3 run.py` from inside the tutorial folder. Return (rc, secs, tail)."""
    print(f"  regenerating {rel_tutorial(tutorial)}README.md ...", flush=True)
    start = time.monotonic()
    proc = subprocess.run(
        ["python3", "run.py"],
        cwd=tutorial,
        text=True,
        capture_output=True,
    )
    elapsed = time.monotonic() - start
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
    return proc.returncode, elapsed, tail[0]


def run_validate() -> tuple[int, str]:
    print("  validating catalog ...", flush=True)
    proc = subprocess.run(
        ["python3", "scripts/validate_catalog.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
    return proc.returncode, tail[0]


def dry_run(limit: int | None, start_index: int, retry_failed: bool) -> int:
    tutorials = catalog_tutorials()
    data = load_manifest()
    selected = set(queued_tutorials(start_index, limit, retry_failed))
    completed = sum(1 for t in tutorials if is_complete(data, t))
    failed = sum(1 for t in tutorials if is_failed(data, t))
    have_report = sum(1 for t in tutorials if has_proofread_report(t))

    print(f"{len(tutorials)} active tutorial(s).")
    print(f"{have_report} have a proofread report under {PROOFREAD_DIR.relative_to(ROOT)}.")
    print(f"{completed} complete in {MANIFEST.relative_to(ROOT)}.")
    print(f"{failed} previously failed.")
    print(f"{len(selected)} selected for the next run.")
    for index, tutorial in enumerate(tutorials, start=1):
        rel = rel_tutorial(tutorial)
        if not has_proofread_report(tutorial):
            status = "no-report"
        elif is_complete(data, tutorial):
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
        description="Apply egregious-only fixes from proofread reports via Claude (Sonnet).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--skip-regen",
        action="store_true",
        help="Skip the python run.py regeneration step (debug only).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run:
        return dry_run(args.limit, args.start_index, args.retry_failed)

    queue = queued_tutorials(args.start_index, args.limit, args.retry_failed)
    if not queue:
        print("No tutorials queued for the fix sweep.")
        return 0

    for index, tutorial in enumerate(queue, start=1):
        rel = rel_tutorial(tutorial)
        print(f"[{index}/{len(queue)}] {rel}", flush=True)

        before = git_porcelain_paths()
        code = run_claude(tutorial, args.claude_bin, args.model)
        if code != 0:
            err = f"claude exited with code {code}"
            print(f"  FAILED at claude: {err}", file=sys.stderr)
            record_failed(tutorial, args.model, "claude", err, code)
            continue

        after = git_porcelain_paths()
        new_paths = sorted(after - before)
        out_of_scope = [p for p in new_paths if not in_scope(p, tutorial)]
        if out_of_scope:
            print(
                f"  SCOPE VIOLATION: subagent touched {len(out_of_scope)} path(s) outside {rel}:",
                file=sys.stderr,
            )
            for p in out_of_scope:
                print(f"    - {p}", file=sys.stderr)
            reverted = revert_out_of_scope(out_of_scope)
            err = f"reverted {len(reverted)} out-of-scope path(s): {reverted[:5]}"
            print(f"  REVERTED. Marking failed: {err}", file=sys.stderr)
            record_failed(tutorial, args.model, "scope_violation", err, 0)
            continue

        regen_secs = 0.0
        if not args.skip_regen:
            rc, regen_secs, tail = regen_readme(tutorial)
            if rc != 0:
                err = f"python run.py exited {rc}: {tail}"
                print(f"  FAILED at regen: {err}", file=sys.stderr)
                record_failed(tutorial, args.model, "regen", err, rc)
                continue

        rc, tail = run_validate()
        validate_ok = rc == 0
        if not validate_ok:
            err = f"validate_catalog exited {rc}: {tail}"
            print(f"  FAILED at validate: {err}", file=sys.stderr)
            record_failed(tutorial, args.model, "validate", err, rc)
            continue

        record_complete(
            tutorial,
            args.model,
            regen_secs,
            validate_ok,
            summary_line="(see claude stdout for fix-summary line)",
        )
        remaining = len(queue) - index
        print(f"  ok (regen {regen_secs:.1f}s); {remaining} remain.")
        if remaining:
            print(f"  next: {rel_tutorial(queue[index])}")

    print("Fix sweep finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
