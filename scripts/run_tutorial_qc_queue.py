#!/usr/bin/env python3
"""Run Codex sequentially over the later-half tutorial QC audit queue."""
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
MANIFEST = ROOT / "docs" / "qc-reports" / "tutorial-qc-manifest.json"
REPORT_DIR = ROOT / "docs" / "qc-reports" / "tutorials"
PASS_NAME = "quality-control-v1"
DEFAULT_START_INDEX = 34
DEFAULT_END_INDEX = 65
CATALOG_LINK_RE = re.compile(r"\]\(([^)#]+/)\)")


PROMPT_TEMPLATE = """\
You are working in {ROOT}.
Public project name: Computational Economics.

Target tutorial for this audit: {tutorial_path}

This is an AUDIT-ONLY quality-control pass. Do not rewrite tutorial prose, code,
figures, tables, or the root README row. Your only tracked edits should be:
- docs/qc-reports/tutorials/{report_name}.md
- docs/qc-reports/tutorial-qc-manifest.json

Read CLAUDE.md, STYLE_GUIDE.md, GLOSSARY.md, the target tutorial's run.py,
README.md, tables/*.csv if present, and the target tutorial row in the root
README.md.

Evaluate the tutorial on these dimensions:

1. Crux and intuition
   - Does the tutorial explain the critical economic concept?
   - Is it economics-first, with computation serving the economic object?
   - Is it written for PhD-level economics readers without handholding?

2. Pseudocode and method clarity
   - Is there pseudocode or algorithmic math for the key computational method?
   - Is it symbolic and problem-specific rather than a Python dump?
   - Does notation match the Equations section?

3. Results and writeup coherence
   - Do prose claims match figures, tables, and run.py calculations?
   - Spot-check at least two numerical claims when available.
   - Check whether captions are absent or non-distracting, with exposition in text.

4. Reproducibility
   - Run:
     python3 scripts/qc_subject.py {subject} --only {slug} --json --out /tmp/qc-static-{slug}.json
     python3 scripts/qc_repro.py {subject} --only {slug} --timeout 300 --restore --out /tmp/qc-repro-{slug}.json
     python3 scripts/validate_catalog.py
   - If execution fails, report it as a blocker.
   - If execution changes artifacts, report the classifications from qc_repro.

5. Root catalog row
   - Does the root README description help a reader choose the tutorial?
   - Does the thumbnail open a useful full figure?
   - Does the title still link to the tutorial folder?

Write a concise Markdown report with:
- Verdict: pass / minor issues / major issues / blocker
- Scorecard table: crux, pseudocode, coherence, reproducibility, catalog row
- Evidence: exact claims checked and whether they match
- Findings ordered by severity
- Recommended follow-up edits, but do not apply them
- Commands run and their exit status

After writing the report, update the manifest entry for this tutorial, commit only
the report and manifest, push, and leave a clean worktree.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def catalog_tutorials() -> list[Path]:
    """Return tutorial directories in root README catalog order."""
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


def rel_tutorial(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() + "/"


def report_name_for(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return rel.replace("/", "__")


def report_path_for(path: Path) -> str:
    return f"docs/qc-reports/tutorials/{report_name_for(path)}.md"


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


def completed_tutorials() -> set[str]:
    data = load_manifest()
    return {
        entry.get("tutorial_path", "")
        for entry in data["entries"]
        if entry.get("status") == "complete"
    }


def manifest_entry_for(tutorial_path: str) -> dict[str, Any] | None:
    data = load_manifest()
    for entry in data["entries"]:
        if entry.get("tutorial_path") == tutorial_path:
            return entry
    return None


def git_status_porcelain() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def require_clean_worktree() -> None:
    status = git_status_porcelain()
    if status:
        print("Refusing to start with a dirty worktree:", file=sys.stderr)
        print(status, file=sys.stderr)
        raise SystemExit(1)


def selected_tutorials(start_index: int, limit: int | None) -> list[Path]:
    tutorials = catalog_tutorials()
    if start_index < 1:
        raise SystemExit("--start-index must be >= 1")
    if len(tutorials) < DEFAULT_END_INDEX:
        raise SystemExit(
            f"Catalog has {len(tutorials)} tutorials; expected at least {DEFAULT_END_INDEX}"
        )

    selected = tutorials[start_index - 1 : DEFAULT_END_INDEX]
    done = completed_tutorials()
    selected = [path for path in selected if rel_tutorial(path) not in done]

    if limit is not None:
        if limit < 0:
            raise SystemExit("--limit must be nonnegative")
        selected = selected[:limit]

    return selected


def build_prompt(tutorial: Path) -> str:
    rel = rel_tutorial(tutorial)
    subject = tutorial.relative_to(ROOT).parts[0]
    slug = tutorial.name
    report_name = report_name_for(tutorial)
    return dedent(
        PROMPT_TEMPLATE.format(
            ROOT=ROOT,
            tutorial_path=rel,
            report_name=report_name,
            subject=subject,
            slug=slug,
        )
    )


def run_codex(tutorial: Path, codex_bin: str) -> int:
    cmd = [
        codex_bin,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(ROOT),
        "-",
    ]
    print(f"\n=== Running QC audit for {rel_tutorial(tutorial)} ===", flush=True)
    return subprocess.run(cmd, cwd=ROOT, input=build_prompt(tutorial), text=True).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Codex sequentially over tutorial QC audit reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print queued tutorials without launching Codex.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of queued tutorials to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help="1-based catalog position to start from; defaults to 33.",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex executable to invoke.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = selected_tutorials(args.start_index, args.limit)

    if args.dry_run:
        print(f"{len(queue)} tutorial(s) queued:")
        positions = {path: i for i, path in enumerate(catalog_tutorials(), start=1)}
        for path in queue:
            print(f"  {positions[path]:02d}. {rel_tutorial(path)} -> {report_path_for(path)}")
        return 0

    if not queue:
        print("No QC tutorials queued.")
        return 0

    require_clean_worktree()

    for index, tutorial in enumerate(queue, start=1):
        tutorial_path = rel_tutorial(tutorial)
        print(f"[{index}/{len(queue)}] {tutorial_path}", flush=True)
        code = run_codex(tutorial, args.codex_bin)
        if code != 0:
            print(f"Codex failed for {tutorial_path} with exit code {code}.", file=sys.stderr)
            return code

        status = git_status_porcelain()
        if status:
            print(f"Codex left uncommitted changes after {tutorial_path}:", file=sys.stderr)
            print(status, file=sys.stderr)
            return 1

        entry = manifest_entry_for(tutorial_path)
        if not entry or entry.get("status") != "complete":
            print(f"Manifest was not marked complete for {tutorial_path}.", file=sys.stderr)
            return 1

        remaining = len(queue) - index
        print(f"Completed {tutorial_path}; {remaining} queued tutorial(s) remain.")
        if remaining:
            print(f"Next: {rel_tutorial(queue[index])}")

    print("All queued QC audits completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
