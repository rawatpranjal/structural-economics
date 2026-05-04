#!/usr/bin/env python3
"""Run Codex sequentially over unfinished tutorial rewrites."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
CATALOG_LINK_RE = re.compile(r"\]\(([^)#]+/)\)")

BASE_PROMPT = """\
alrigth, now need to reivew each tutorial one at a time

  - the current write up is a bit robotic
  - try to make it more human readable adn not a "dumip" of code/resutls
  - better captions , more tailored
  - economics-first, then method
  - audience: phd level econ grad students --> no handholding
  - limit heavy jargon; use technical terms only when they are actually needed
  - structur eis good but a bit robotic, you have liberties to change up the structure of each torpic
  - imrpove the writing of the content relating to that tutoril in the amin readme
  - better referncing within the repo to other tutorials (if needed ofc ourse)
  - full notation, key clarifications
  - more self contained tutorial, but not too long.
 - also remove Reproduce
    python run.py --> not needed
- Remove image captions, instead just talk/exposite in the text (it should be apparent which figure)
- Must add  mathpseudo code /algo pseudo code for the key solution methods/algos (as applied to the problem).
- If ground truth is available / approximatable via finer grid/more data please do add to the graph so we can compare solution method /approximation to the ground truth. Not everywhere though/ if tedious or adds bloat to graph.
- Align with style guide, glossary and clade md
- Commit run and git push at the end, no loose ends
-
  lets start with what is not done (we are going topic by topic in sequence) look what was completed last and pick the next.
 . run it one by one? i think in that way it will updat eh tturials. we did 3 so far?
"""


def catalog_tutorials() -> list[Path]:
    """Return tutorial directories in the same order as the root README catalog."""
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
        if path in seen or not (path / "run.py").exists():
            continue
        seen.add(path)
        tutorials.append(path)

    return tutorials


def tutorial_is_done(path: Path) -> bool:
    """Return whether a tutorial already uses the current report-style flags."""
    run_text = (path / "run.py").read_text()
    return (
        "include_reproduce=False" in run_text
        and "show_figure_captions=False" in run_text
    )


def queued_tutorials() -> list[Path]:
    """Return catalog tutorials that have not yet been revised."""
    return [path for path in catalog_tutorials() if not tutorial_is_done(path)]


def git_status_porcelain() -> str:
    """Return the short machine-readable git status."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def require_clean_worktree() -> None:
    """Abort if the repository has uncommitted changes."""
    status = git_status_porcelain()
    if status:
        print("Refusing to start with a dirty worktree:", file=sys.stderr)
        print(status, file=sys.stderr)
        raise SystemExit(1)


def build_prompt(tutorial: Path) -> str:
    """Build the prompt sent to the per-tutorial Codex job."""
    rel = tutorial.relative_to(ROOT)
    return dedent(
        f"""\
        You are working in {ROOT}.
        Public project name: Computational Economics.

        Target tutorial for this run: {rel}

        Process exactly this tutorial and then stop. Do not revise later tutorials in
        the catalog. You may edit:
        - {rel}/run.py
        - generated files under {rel}/
        - the root README.md catalog row/text for {rel}

        Use python3 for local commands on this machine. Read CLAUDE.md,
        STYLE_GUIDE.md, GLOSSARY.md, and any section-level CLAUDE.md that applies to
        this tutorial before editing.

        Original user prompt:

        {BASE_PROMPT}

        Concrete finishing requirements for this single tutorial:
        - Regenerate the tutorial from its folder.
        - Run python3 scripts/validate_catalog.py from the repo root.
        - Commit only this tutorial's changes and the relevant root README.md edit.
        - git push before finishing.
        - Leave no uncommitted changes.
        """
    )


def run_codex(tutorial: Path, codex_bin: str) -> int:
    """Run one non-interactive Codex job for a tutorial."""
    cmd = [
        codex_bin,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(ROOT),
        "-",
    ]
    print(f"\n=== Running Codex for {tutorial.relative_to(ROOT)} ===", flush=True)
    return subprocess.run(cmd, cwd=ROOT, input=build_prompt(tutorial), text=True).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Codex sequentially over unfinished tutorial rewrites.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queued tutorials without launching Codex.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of queued tutorials to process.",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex executable to invoke.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = queued_tutorials()
    if args.limit is not None:
        if args.limit < 0:
            print("--limit must be nonnegative", file=sys.stderr)
            return 2
        queue = queue[: args.limit]

    if args.dry_run:
        print(f"{len(queue)} tutorial(s) queued:")
        for path in queue:
            print(f"  - {path.relative_to(ROOT)}")
        return 0

    if not queue:
        print("No unfinished tutorials found.")
        return 0

    require_clean_worktree()

    for index, tutorial in enumerate(queue, start=1):
        print(f"[{index}/{len(queue)}] {tutorial.relative_to(ROOT)}", flush=True)
        code = run_codex(tutorial, args.codex_bin)
        if code != 0:
            print(
                f"Codex failed for {tutorial.relative_to(ROOT)} with exit code {code}.",
                file=sys.stderr,
            )
            return code

        status = git_status_porcelain()
        if status:
            print(
                f"Codex left uncommitted changes after {tutorial.relative_to(ROOT)}:",
                file=sys.stderr,
            )
            print(status, file=sys.stderr)
            return 1

        remaining = len(queue) - index
        print(f"Completed {tutorial.relative_to(ROOT)}; {remaining} queued tutorial(s) remain.")
        if remaining:
            print(f"Next: {queue[index].relative_to(ROOT)}")

    print("All queued tutorials completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
