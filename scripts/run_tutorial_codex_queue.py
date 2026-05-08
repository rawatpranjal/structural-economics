#!/usr/bin/env python3
"""Run Codex over one active tutorial, with manifest-backed recovery."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent, indent
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "qc-reports" / "tutorial-codex-sweep-manifest.json"
MANIFEST = DEFAULT_MANIFEST
CLAUDE_MANIFEST = ROOT / "docs" / "qc-reports" / "tutorial-claude-sweep-manifest.json"
DEFAULT_PASS_NAME = "codex-full-tutorial-sweep-v1"
PASS_NAME = DEFAULT_PASS_NAME
PROMPT_MODE = "standard"
CATALOG_LINK_RE = re.compile(r"\]\(([^)#]+/)\)")
BOOTSTRAP_COMPLETED_TUTORIALS = ("dynamic-programming/shock-discretization/",)
DEFAULT_MIN_FREE_MIB = 1024
TRANSIENT_CACHE_PATHS = (
    Path.home() / ".npm" / "_npx",
)


def utc_now() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_bytes(size: int) -> str:
    """Return a compact binary size for log messages."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024


def directory_size(path: Path) -> int:
    """Return a best-effort recursive size for a path."""
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.lstat().st_size
        except OSError:
            return 0

    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.lstat().st_size
        except OSError:
            continue
    return total


def disk_free_bytes() -> int:
    """Return free bytes on the repository volume."""
    return shutil.disk_usage(ROOT).free


def prune_transient_caches() -> list[str]:
    """Remove safe transient cache directories that can grow during Codex runs."""
    removed: list[str] = []
    for path in TRANSIENT_CACHE_PATHS:
        if not path.exists():
            continue
        size = directory_size(path)
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise SystemExit(f"Failed to remove transient cache {path}: {exc}") from exc
        removed.append(f"{path} ({format_bytes(size)})")
    return removed


def require_disk_headroom(min_free_mib: int, *, prune: bool) -> None:
    """Abort live runs when the repo volume is too full."""
    if min_free_mib < 0:
        raise SystemExit("--min-free-mib must be nonnegative")

    minimum = min_free_mib * 1024 * 1024
    before = disk_free_bytes()
    removed = prune_transient_caches() if prune else []
    after = disk_free_bytes()

    if removed:
        print("Removed transient cache path(s) before launching Codex:")
        for item in removed:
            print(f"  - {item}")

    if after < minimum:
        message = (
            f"Only {format_bytes(after)} free on the repository volume; "
            f"need at least {format_bytes(minimum)} before launching Codex."
        )
        if prune and before == after and not removed:
            message += " No configured transient cache paths were present to prune."
        raise SystemExit(message)

    print(
        f"Disk headroom OK: {format_bytes(after)} free "
        f"(minimum {format_bytes(minimum)}).",
        flush=True,
    )


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
            rel = path.relative_to(ROOT)
        except ValueError:
            continue
        if "_legacy" in rel.parts:
            continue
        if path in seen or not (path / "run.py").exists():
            continue
        seen.add(path)
        tutorials.append(path)

    return tutorials


def rel_tutorial(path: Path) -> str:
    """Return a POSIX tutorial path with the trailing slash used in the catalog."""
    return path.relative_to(ROOT).as_posix() + "/"


def display_path(path: Path) -> str:
    """Return a repository-relative display path when possible."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def catalog_row_for(tutorial: Path) -> str:
    """Return the root README catalog row for a tutorial, if present."""
    rel = rel_tutorial(tutorial)
    for line in (ROOT / "README.md").read_text().splitlines():
        if f"]({rel})" in line:
            return line
    return "(Root catalog row not found by the runner; inspect README.md directly.)"


def load_manifest() -> dict[str, Any]:
    """Load or initialize the Codex sweep manifest."""
    if not MANIFEST.exists():
        return {
            "pass_name": PASS_NAME,
            "updated_at": utc_now(),
            "entries": initial_manifest_entries(),
        }

    data = json.loads(MANIFEST.read_text())
    if data.get("pass_name") != PASS_NAME:
        raise SystemExit(
            f"Manifest pass_name is {data.get('pass_name')!r}; expected {PASS_NAME!r}"
        )
    if "entries" not in data or not isinstance(data["entries"], list):
        raise SystemExit("Manifest must contain an entries list")
    return data


def initial_manifest_entries() -> list[dict[str, Any]]:
    """Return initial entries for a missing manifest."""
    if MANIFEST == DEFAULT_MANIFEST and PASS_NAME == DEFAULT_PASS_NAME:
        return bootstrap_completed_entries()
    return []


def write_manifest(data: dict[str, Any]) -> None:
    """Write the manifest in stable JSON format."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")


def sort_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort manifest entries by current root README order."""
    order = {rel_tutorial(path): index for index, path in enumerate(catalog_tutorials())}
    return sorted(entries, key=lambda entry: order.get(entry.get("tutorial_path", ""), len(order)))


def bootstrap_completed_entries() -> list[dict[str, Any]]:
    """Return manually completed entries used only before a Codex manifest exists."""
    entries: list[dict[str, Any]] = []
    for tutorial_path in BOOTSTRAP_COMPLETED_TUTORIALS:
        tutorial_dir = tutorial_path.rstrip("/")
        commit_result = run_git(
            ["log", "-1", "--format=%H", "--", tutorial_dir],
            check=False,
        )
        commit = commit_result.stdout.strip() if commit_result.returncode == 0 else ""
        changed_paths = [
            "README.md",
            f"{tutorial_dir}/README.md",
            f"{tutorial_dir}/run.py",
        ]
        entries.append(
            {
                "tutorial_path": tutorial_path,
                "status": "complete",
                "commit": commit,
                "changed_paths": changed_paths,
                "updated_at": utc_now(),
            }
        )
    return entries


def seed_manifest_from_claude() -> int:
    """Seed missing Codex completions from the existing Claude sweep manifest."""
    if not CLAUDE_MANIFEST.exists():
        raise SystemExit(f"Claude manifest not found: {CLAUDE_MANIFEST.relative_to(ROOT)}")

    source = json.loads(CLAUDE_MANIFEST.read_text())
    source_entries = source.get("entries")
    if not isinstance(source_entries, list):
        raise SystemExit("Claude manifest must contain an entries list")

    data = load_manifest()
    by_path = {
        entry.get("tutorial_path"): entry
        for entry in data["entries"]
        if isinstance(entry, dict) and entry.get("tutorial_path")
    }

    added = 0
    for entry in source_entries:
        if entry.get("status") != "complete":
            continue
        tutorial_path = entry.get("tutorial_path")
        if not tutorial_path or tutorial_path in by_path:
            continue
        by_path[tutorial_path] = {
            "tutorial_path": tutorial_path,
            "status": "complete",
            "commit": entry.get("commit", ""),
            "changed_paths": entry.get("changed_paths", []),
            "updated_at": entry.get("updated_at", utc_now()),
        }
        added += 1

    data["entries"] = sort_entries(list(by_path.values()))
    data["updated_at"] = utc_now()
    write_manifest(data)
    return added


def completed_tutorials() -> set[str]:
    """Return tutorial paths already completed by this Codex sweep."""
    data = load_manifest()
    return {
        entry.get("tutorial_path", "")
        for entry in data["entries"]
        if entry.get("status") == "complete"
    }


def normalized_limit(limit: int | None) -> int:
    """Return the runner batch size, enforcing one tutorial per invocation."""
    if limit is None:
        return 1
    if limit < 0:
        raise SystemExit("--limit must be nonnegative")
    if limit > 1:
        raise SystemExit("This runner processes exactly one tutorial; --limit cannot exceed 1.")
    return limit


def queued_tutorials(start_index: int, limit: int | None) -> list[Path]:
    """Return the remaining tutorials for this one-run Codex sweep."""
    if start_index < 1:
        raise SystemExit("--start-index must be >= 1")

    queue = catalog_tutorials()[start_index - 1 :]
    done = completed_tutorials()
    queue = [path for path in queue if rel_tutorial(path) not in done]
    return queue[: normalized_limit(limit)]


def run_git(
    args: list[str], *, capture: bool = True, check: bool = True
) -> subprocess.CompletedProcess[str]:
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


def blob_lines(commit: str, path: str) -> list[str]:
    """Return a tracked file at a commit as a list of lines."""
    result = run_git(["show", f"{commit}:{path}"])
    return result.stdout.splitlines()


def validate_root_readme_changes(before: str, after: str, tutorial: Path) -> None:
    """Ensure root README changes are confined to the target tutorial catalog row."""
    rel = rel_tutorial(tutorial)
    before_lines = blob_lines(before, "README.md")
    after_lines = blob_lines(after, "README.md")
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    bad_lines: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed = before_lines[i1:i2] + after_lines[j1:j2]
        for line in changed:
            if line.strip() and rel not in line:
                bad_lines.append(line)

    if bad_lines:
        print(
            "Root README.md changes were not confined to the target catalog row:",
            file=sys.stderr,
        )
        for line in bad_lines[:20]:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)


def path_allowed_for_tutorial(path: str, tutorial: Path) -> bool:
    """Return whether a changed path belongs to the single-tutorial edit set."""
    rel = rel_tutorial(tutorial)
    return path == "README.md" or path.startswith(rel)


def validate_child_commit(before: str, after: str, tutorial: Path) -> list[str]:
    """Validate that Codex made a scoped non-empty commit for one tutorial."""
    tutorial_path = rel_tutorial(tutorial)
    if before == after:
        raise SystemExit(f"Codex did not create a commit for {tutorial_path}.")

    changed = changed_paths_between(before, after)
    if not changed:
        raise SystemExit(f"Codex created only empty commit(s) for {tutorial_path}.")

    bad = [path for path in changed if not path_allowed_for_tutorial(path, tutorial)]
    if bad:
        print(f"Codex changed paths outside {tutorial_path} and root README.md:", file=sys.stderr)
        for path in bad:
            print(f"  - {path}", file=sys.stderr)
        raise SystemExit(1)

    if "README.md" in changed:
        validate_root_readme_changes(before, after, tutorial)

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
    data["entries"] = sort_entries(entries)
    data["updated_at"] = utc_now()
    write_manifest(data)


def commit_manifest(tutorial: Path) -> None:
    """Commit the manifest progress update."""
    manifest_rel = MANIFEST.relative_to(ROOT).as_posix()
    run_git(["add", manifest_rel], capture=False)
    staged = run_git(["diff", "--cached", "--name-only"]).stdout.splitlines()
    if staged != [manifest_rel]:
        raise SystemExit(f"Expected only {manifest_rel} to be staged; got {staged!r}.")
    message = f"Record Codex sweep progress for {rel_tutorial(tutorial).rstrip('/')}"
    run_git(["commit", "-m", message], capture=False)


def git_push() -> None:
    """Push the current branch."""
    run_git(["push"], capture=False)


def standard_prompt_instructions(rel: str) -> str:
    """Return the original tutorial rewrite instructions."""
    return dedent(
        f"""\
        Instructions for this single-tutorial pass:
        - Make the tutorial economics-first for RA/pre-PhD through early PhD
          economics readers.
        - Keep the economic question distinct from the computational question.
          Use that distinction to plan the rewrite, but do not write formulaic
          paragraphs that begin "The economic question is", "The computational
          question is", or "The core method is". The prose should read like a
          person explaining the tutorial, not like a checklist.
        - The economic question is the model object or empirical/equilibrium use
          case the reader cares about. The computational question is the numerical
          obstacle or algorithmic object needed to study it. Do not collapse these
          into "the economic question is how to run the algorithm."
        - Write the opening in a conversational but precise style: start with the
          economic situation, then explain why a computation is needed.
        - Make clear the core computational method and how it serves the economic
          question.
        - Start from the economic problem, give a concrete example, introduce the
          mathematical object, explain the algorithm with compact pseudocode where
          useful, then interpret results.
        - Write in active voice where it makes the sentence clearer. Use concrete
          examples and varied sentence rhythm. Avoid forced three-part lists,
          "not just ... but ..." contrasts, generic signposting, and inflated AI
          tells such as delve, crucial, pivotal, seamless, robust, landscape,
          testament, underscores, highlights, serves as, and stands as.
        - Do not use em dashes in generated or edited prose. Use commas, colons,
          parentheses, or shorter sentences instead.
        - Avoid solver-first prose, compressed survey prose, long caveat chains,
          package trivia, and code-dump narration.
        - Keep visible Reproduce sections and image captions omitted unless needed;
          keep useful alt text.
        - Improve the root catalog row only to match the new framing. The title
          should be informative, and the description should stay high-level,
          roughly two or three short wrapped lines in the table. Put the economic
          object first and the computational method second. Do not cram
          diagnostics, exact residuals, or too many numbers into the outer README
          row.
        - Before committing, scan the edited tutorial README, run.py, and root
          catalog row for the AI tells above, formulaic question labels, and em
          dashes. Revise any matches that are part of the generated prose.
        - Edit {rel}run.py first, then regenerate from inside the tutorial folder
          with python3 run.py.
        - Run python3 scripts/validate_catalog.py from the repository root.
        - Commit only this tutorial's files and the matching root README.md row.
          Do not use an empty commit.
        - Push the commit before finishing.
        - Leave a clean worktree.
        """
    )


def cut_prompt_instructions(rel: str) -> str:
    """Return the cut-not-compress editorial instructions."""
    return dedent(
        f"""\
        Instructions for this cut-not-compress pass:
        - This is an editorial deletion pass, not a general rewrite. The goal is
          to make the tutorial easier to read by cutting material that is useful
          but not necessary for this tutorial's core lesson.
        - Before editing, identify the tutorial's core lesson in one sentence.
          Keep that sentence as your private decision rule. Do not add a visible
          "core lesson" label to the tutorial unless the local style already uses
          that label.
        - Build a keep list with 3 to 5 necessary ideas. Build a delete list
          before rewriting. Delete useful-but-not-necessary material. Do not
          reintroduce deleted content in compressed form.
        - Overview may only introduce the economic situation, the object, and the
          computational need. Put those in separate paragraphs. Do not use the
          opening to map neighboring tutorials or adjacent methods.
        - Keep one paragraph to one concept. Keep one sentence to one thought.
          Prose sentences must be 22 words or fewer. Prose paragraphs must be 70
          words or fewer.
        - Allowed jargon must be tutorial-specific and short. Use only terms the
          reader needs for this tutorial's object or method, such as the model
          name, estimator, algorithm, state variable, policy rule, equilibrium
          object, likelihood, residual, or operator being taught. Delete generic
          filler and unnecessary meta-language.
        - Avoid forced three-part lists, "not just ... but ..." contrasts,
          generic signposting, inflated AI tells such as delve, crucial, pivotal,
          seamless, robust, landscape, testament, underscores, highlights, serves
          as, and stands as, and all em dashes in generated prose.
        - Preserve correct equations, generated figures, and code needed to run
          the tutorial. Edit {rel}run.py first, then regenerate from inside the
          tutorial folder with python3 run.py.
        - Keep visible Reproduce sections and image captions omitted unless
          needed; keep useful alt text.
        - Improve the root catalog row only if needed to match the cut tutorial.
          Keep it short. Put the economic object first and the computational
          method second.

        Required child self-audit before committing:
        - After regenerating, audit the edited tutorial README, run.py generated
          prose, and root catalog row. Use temporary shell one-liners if useful,
          but do not leave helper files behind.
        - Report the 15 longest prose sentences and their word counts in your
          terminal-visible reasoning or final child message. Revise any prose
          sentence over 22 words unless it is code, a heading, a table row, math,
          a URL, or unavoidable bibliography/provenance text.
        - Report every prose paragraph over 70 words. Revise any violation unless
          it is code, math, a table, or generated metadata that should not be
          rewritten.
        - Assign a short job label to each prose paragraph, such as situation,
          object, computational need, equation, algorithm, diagnostic, or
          interpretation.
        - Report paragraphs with more than one job. Split or cut them before
          committing.
        - Report the non-allowed jargon list. Delete or replace each item before
          committing.
        - Explicitly confirm that the final cuts are deletions, not compression.
          If you moved deleted material into shorter sentences, keep cutting.
        - Run python3 scripts/validate_catalog.py from the repository root.
        - Commit only this tutorial's files and the matching root README.md row.
          Do not use an empty commit.
        - Push the commit before finishing.
        - Leave a clean worktree.
        """
    )


def build_prompt(tutorial: Path) -> str:
    """Build the prompt for one non-interactive Codex run."""
    rel = rel_tutorial(tutorial)
    changed_paths = indent(
        "\n".join(
            [
                f"- {rel}run.py",
                f"- generated files under {rel}",
                f"- the single root README.md catalog row for {rel}",
            ]
        ),
        "        ",
    )
    instructions = (
        cut_prompt_instructions(rel)
        if PROMPT_MODE == "cut"
        else standard_prompt_instructions(rel)
    ).rstrip()

    prefix = dedent(
        f"""\
        You are working in {ROOT}.
        Public project name: Computational Economics.

        Target tutorial for this run: {rel}
        Prompt mode: {PROMPT_MODE}

        Process exactly this one tutorial and stop. Do not inspect, revise,
        regenerate, or commit later tutorials in the catalog. You may edit only:
{changed_paths}

        The parent runner owns {display_path(MANIFEST)}.
        Do not edit docs/qc-reports files.

        Before editing, read:
        - CLAUDE.md
        - STYLE_GUIDE.md
        - GLOSSARY.md
        - the relevant section-level CLAUDE.md, if present
        - {rel}run.py
        - {rel}README.md
        - the root README.md catalog row for this tutorial

        Current root catalog row:
        {catalog_row_for(tutorial)}

        """
    )
    return f"{prefix}{instructions}\n"


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
    print(
        f"\n=== Running Codex for {rel_tutorial(tutorial)} ({PROMPT_MODE} mode) ===",
        flush=True,
    )
    return subprocess.run(cmd, cwd=ROOT, input=build_prompt(tutorial), text=True).returncode


def dry_run(limit: int | None, start_index: int) -> int:
    """Print the active catalog and queue status without launching Codex."""
    tutorials = catalog_tutorials()
    done = completed_tutorials()
    selected = {rel_tutorial(path) for path in queued_tutorials(start_index, limit)}
    queue_count = len(selected)
    complete_source = (
        display_path(MANIFEST)
        if MANIFEST.exists()
        else f"initial defaults for missing {display_path(MANIFEST)}"
    )

    print(f"Manifest: {display_path(MANIFEST)}")
    print(f"Pass name: {PASS_NAME}")
    print(f"Prompt mode: {PROMPT_MODE}")
    print(f"{len(tutorials)} active tutorial(s) in root README order.")
    print(f"{len(done)} complete from {complete_source}.")
    print(f"{queue_count} tutorial(s) selected for the next run.")
    for index, tutorial in enumerate(tutorials, start=1):
        rel = rel_tutorial(tutorial)
        if rel in done:
            status = "complete"
        elif rel in selected:
            status = "queued"
        else:
            status = "not selected"
        print(f"  {index:02d}. [{status}] {rel}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Codex over exactly one active tutorial rewrite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print catalog order and queued tutorial without launching Codex.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of remaining tutorials to process. Must be 0 or 1.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based catalog position to start from before skipping manifest-complete entries.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST.relative_to(ROOT).as_posix(),
        help="Manifest JSON path for this pass. Defaults to the completed v1 Codex manifest.",
    )
    parser.add_argument(
        "--pass-name",
        default=DEFAULT_PASS_NAME,
        help="Expected pass_name to read or write in the manifest.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("standard", "cut"),
        default="standard",
        help="Prompt template to send to the child Codex run.",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex executable to invoke.",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=DEFAULT_MIN_FREE_MIB,
        help="Minimum free space required on the repo volume before a live Codex run.",
    )
    parser.add_argument(
        "--skip-cache-prune",
        action="store_true",
        help="Do not remove configured transient cache paths before a live Codex run.",
    )
    parser.add_argument(
        "--seed-from-claude",
        action="store_true",
        help="Copy missing completed entries from the Claude sweep manifest into the Codex manifest.",
    )
    return parser.parse_args()


def configure_run(args: argparse.Namespace) -> None:
    """Apply CLI configuration to the module-level runner state."""
    global MANIFEST, PASS_NAME, PROMPT_MODE

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    manifest = manifest.resolve()
    try:
        manifest.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit("--manifest must be inside the repository") from exc

    if not args.pass_name.strip():
        raise SystemExit("--pass-name must be nonempty")

    MANIFEST = manifest
    PASS_NAME = args.pass_name
    PROMPT_MODE = args.prompt_mode


def main() -> int:
    args = parse_args()
    configure_run(args)

    if args.seed_from_claude:
        added = seed_manifest_from_claude()
        print(f"Seeded {added} completion(s) from {display_path(CLAUDE_MANIFEST)}.")
        if added and not args.dry_run:
            print("Seeded the Codex manifest; commit and push it before a live tutorial run.")
            return 0

    if args.dry_run:
        return dry_run(args.limit, args.start_index)

    queue = queued_tutorials(args.start_index, args.limit)
    if not queue:
        print("No tutorials queued for the Codex sweep.")
        return 0

    require_clean_worktree()
    require_no_unpushed_commits()
    require_disk_headroom(args.min_free_mib, prune=not args.skip_cache_prune)

    tutorial = queue[0]
    tutorial_path = rel_tutorial(tutorial)
    print(f"[1/1] {tutorial_path}", flush=True)

    before = git_head()
    code = run_codex(tutorial, args.codex_bin)
    if code != 0:
        print(f"Codex failed for {tutorial_path} with exit code {code}.", file=sys.stderr)
        return code

    require_clean_worktree()
    after = git_head()
    changed_paths = validate_child_commit(before, after, tutorial)
    require_no_unpushed_commits()

    run_catalog_validation()
    require_clean_worktree()

    record_manifest_completion(tutorial, after, changed_paths)
    commit_manifest(tutorial)
    git_push()
    require_clean_worktree()
    require_no_unpushed_commits()

    print(f"Completed {tutorial_path}; stopping after one tutorial.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
