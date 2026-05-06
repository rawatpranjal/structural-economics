#!/usr/bin/env python3
"""Run Claude sequentially over the active tutorial catalog."""
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
MANIFEST = ROOT / "docs" / "qc-reports" / "tutorial-claude-sweep-manifest.json"
PASS_NAME = "claude-full-tutorial-sweep-v1"
CATALOG_LINK_RE = re.compile(r"\]\(([^)#]+/)\)")


USER_INSTRUCTIONS_VERBATIM = """\
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


def catalog_row_for(tutorial: Path) -> str:
    """Return the root README catalog row for a tutorial, if it is easy to identify."""
    rel = rel_tutorial(tutorial)
    for line in (ROOT / "README.md").read_text().splitlines():
        if f"]({rel})" in line:
            return line
    return "(Root catalog row not found by the runner; inspect README.md directly.)"


def load_manifest() -> dict[str, Any]:
    """Load or initialize the Claude sweep manifest."""
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


def completed_tutorials() -> set[str]:
    """Return tutorial paths already completed by this Claude sweep."""
    data = load_manifest()
    return {
        entry.get("tutorial_path", "")
        for entry in data["entries"]
        if entry.get("status") == "complete"
    }


def queued_tutorials(start_index: int, limit: int | None) -> list[Path]:
    """Return the remaining tutorials for this sweep."""
    if start_index < 1:
        raise SystemExit("--start-index must be >= 1")
    queue = catalog_tutorials()[start_index - 1 :]
    done = completed_tutorials()
    queue = [path for path in queue if rel_tutorial(path) not in done]
    if limit is not None:
        if limit < 0:
            raise SystemExit("--limit must be nonnegative")
        queue = queue[:limit]
    return queue


def run_git(args: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command from the repository root."""
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=check,
    )


def git_head() -> str:
    """Return the current HEAD commit."""
    return run_git(["rev-parse", "HEAD"]).stdout.strip()


def git_status_porcelain() -> str:
    """Return the machine-readable git status."""
    return run_git(["status", "--porcelain"]).stdout.strip()


def require_clean_worktree() -> None:
    """Abort if the working tree has uncommitted changes."""
    status = git_status_porcelain()
    if status:
        print("Refusing to continue with a dirty worktree:", file=sys.stderr)
        print(status, file=sys.stderr)
        raise SystemExit(1)


def upstream_ref() -> str | None:
    """Return the configured upstream ref, if there is one."""
    result = run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def require_no_unpushed_commits() -> None:
    """Abort if local commits have not been pushed to the configured upstream."""
    upstream = upstream_ref()
    if not upstream:
        return
    result = run_git(["rev-list", "--count", f"{upstream}..HEAD"])
    ahead = int(result.stdout.strip())
    if ahead:
        raise SystemExit(f"Refusing to continue: HEAD is {ahead} commit(s) ahead of {upstream}.")


def changed_paths_between(before: str, after: str) -> list[str]:
    """Return paths changed between two commits."""
    result = run_git(["diff", "--name-only", f"{before}..{after}"])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def path_allowed_for_tutorial(path: str, tutorial: Path) -> bool:
    """Return whether a changed path belongs to the single-tutorial edit set."""
    rel = rel_tutorial(tutorial)
    return path == "README.md" or path.startswith(rel)


def validate_child_commit(before: str, after: str, tutorial: Path) -> list[str]:
    """Validate that Claude made a scoped non-empty commit for one tutorial."""
    tutorial_path = rel_tutorial(tutorial)
    if before == after:
        raise SystemExit(f"Claude did not create a commit for {tutorial_path}.")

    changed = changed_paths_between(before, after)
    if not changed:
        raise SystemExit(f"Claude created only empty commit(s) for {tutorial_path}.")

    bad = [path for path in changed if not path_allowed_for_tutorial(path, tutorial)]
    if bad:
        print(f"Claude changed paths outside {tutorial_path} and root README.md:", file=sys.stderr)
        for path in bad:
            print(f"  - {path}", file=sys.stderr)
        raise SystemExit(1)

    return changed


def run_catalog_validation() -> None:
    """Run the repository catalog validator."""
    subprocess.run(
        ["python3", "scripts/validate_catalog.py"],
        cwd=ROOT,
        text=True,
        check=True,
    )


def record_manifest_completion(tutorial: Path, commit: str, changed_paths: list[str]) -> None:
    """Record one completed tutorial in the manifest."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data = load_manifest()
    tutorial_path = rel_tutorial(tutorial)
    entries = [entry for entry in data["entries"] if entry.get("tutorial_path") != tutorial_path]
    entries.append(
        {
            "tutorial_path": tutorial_path,
            "status": "complete",
            "commit": commit,
            "changed_paths": changed_paths,
            "updated_at": utc_now(),
        }
    )
    data["entries"] = entries
    data["updated_at"] = utc_now()
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")


def commit_manifest(tutorial: Path) -> None:
    """Commit the manifest progress update."""
    run_git(["add", str(MANIFEST.relative_to(ROOT))], capture=False)
    status = git_status_porcelain()
    if str(MANIFEST.relative_to(ROOT)) not in status:
        raise SystemExit(f"Manifest was not staged for {rel_tutorial(tutorial)}.")
    message = f"Record Claude sweep progress for {rel_tutorial(tutorial).rstrip('/')}"
    run_git(["commit", "-m", message], capture=False)


def git_push() -> None:
    """Push the current branch."""
    run_git(["push"], capture=False)


def build_prompt(tutorial: Path) -> str:
    """Build the prompt for one non-interactive Claude run."""
    rel = rel_tutorial(tutorial)
    changed_paths = "\n".join(
        [
            f"- {rel}run.py",
            f"- generated files under {rel}",
            f"- the single root README.md catalog row for {rel}",
        ]
    )

    return dedent(
        f"""\
        You are working in {ROOT}.
        Public project name: Computational Economics.

        Target tutorial for this run: {rel}

        Process exactly this one tutorial and then stop. Do not inspect, revise,
        regenerate, or commit later tutorials in the catalog. You may edit only:
        {changed_paths}

        The runner owns docs/qc-reports/tutorial-claude-sweep-manifest.json.
        Do not edit any docs/qc-reports files.

        Read all of these before editing:
        - CLAUDE.md
        - STYLE_GUIDE.md
        - GLOSSARY.md
        - the relevant section-level CLAUDE.md, if present
        - {rel}run.py
        - {rel}README.md
        - the root README.md catalog row for this tutorial

        Current root catalog row:
        {catalog_row_for(tutorial)}

        Original user instructions, preserved verbatim:

        {USER_INSTRUCTIONS_VERBATIM}

        Hard constraints for this single-tutorial pass:
        - Keep the prose economics-first for PhD-level economics readers; avoid
          code-dump style.
        - Improve method explanation using math notation, examples or
          counterexamples where useful, hyperparameters/tradeoffs, and what the
          method enables economically.
        - Add or improve applied mathematical pseudocode when the key solution,
          simulation, equilibrium, filtering, or estimation method is missing it.
        - Remove visible Reproduce sections and visible image captions where they
          still appear; keep useful alt text and surrounding exposition.
        - Fix math rendering issues, especially GitHub-hostile forms such as
          ^{{*}} or _{{*}}; use ^{{\\ast}} or _{{\\ast}} instead.
        - Regenerate the tutorial by running python3 run.py from inside {rel}.
        - Run python3 scripts/validate_catalog.py from the repository root.
        - Commit only this tutorial's files and the root README.md row. Do not use
          an empty commit.
        - Push the commit before finishing.
        - Leave a clean worktree.
        """
    )


def run_claude(tutorial: Path, claude_bin: str) -> int:
    """Run one non-interactive Claude job for a tutorial."""
    cmd = [
        claude_bin,
        "--print",
        "--dangerously-skip-permissions",
        "--permission-mode",
        "bypassPermissions",
    ]
    print(f"\n=== Running Claude for {rel_tutorial(tutorial)} ===", flush=True)
    return subprocess.run(cmd, cwd=ROOT, input=build_prompt(tutorial), text=True).returncode


def dry_run(limit: int | None, start_index: int) -> int:
    """Print the active catalog and queue status without launching Claude."""
    tutorials = catalog_tutorials()
    done = completed_tutorials()
    selected = set(queued_tutorials(start_index, limit))
    queue_count = len(selected)

    print(f"{len(tutorials)} active tutorial(s) in root README order.")
    print(f"{len(done)} complete in {MANIFEST.relative_to(ROOT)}.")
    print(f"{queue_count} tutorial(s) selected for the next run.")
    for index, tutorial in enumerate(tutorials, start=1):
        rel = rel_tutorial(tutorial)
        if rel in done:
            status = "complete"
        elif tutorial in selected:
            status = "queued"
        else:
            status = "not selected"
        print(f"  {index:02d}. [{status}] {rel}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Claude sequentially over active tutorial rewrites.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print catalog order and queued tutorials without launching Claude.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of remaining tutorials to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based catalog position to start from before skipping manifest-complete entries.",
    )
    parser.add_argument(
        "--claude-bin",
        default="claude",
        help="Claude executable to invoke.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run:
        return dry_run(args.limit, args.start_index)

    queue = queued_tutorials(args.start_index, args.limit)
    if not queue:
        print("No tutorials queued for the Claude sweep.")
        return 0

    require_clean_worktree()
    require_no_unpushed_commits()

    for index, tutorial in enumerate(queue, start=1):
        tutorial_path = rel_tutorial(tutorial)
        print(f"[{index}/{len(queue)}] {tutorial_path}", flush=True)
        require_clean_worktree()
        require_no_unpushed_commits()

        before = git_head()
        code = run_claude(tutorial, args.claude_bin)
        if code != 0:
            print(f"Claude failed for {tutorial_path} with exit code {code}.", file=sys.stderr)
            return code

        require_clean_worktree()
        after = git_head()
        changed_paths = validate_child_commit(before, after, tutorial)

        run_catalog_validation()
        require_clean_worktree()

        record_manifest_completion(tutorial, after, changed_paths)
        commit_manifest(tutorial)
        git_push()
        require_clean_worktree()
        require_no_unpushed_commits()

        remaining = len(queue) - index
        print(f"Completed {tutorial_path}; {remaining} queued tutorial(s) remain.")
        if remaining:
            print(f"Next: {rel_tutorial(queue[index])}")

    print("All selected Claude sweep tutorials completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
