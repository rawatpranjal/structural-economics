#!/usr/bin/env python3
"""Per-subject QC pass: section presence, pseudocode detection, asset references.

Usage:
    python scripts/qc_subject.py <subject>          # e.g. dynamic-programming
    python scripts/qc_subject.py <subject> --json   # machine-readable output

Emits a per-tutorial scorecard. Companion to scripts/validate_catalog.py;
this script is QC-flavored, not a hard validator. Findings feed the QC report.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SECTIONS = (
    "Overview",
    "Equations",
    "Model Setup",
    "Solution Method",
    "Results",
    "Takeaway",
    "References",
)

PSEUDOCODE_HINTS = re.compile(
    r"(?im)^\s*(?:algorithm[: ]|input[: ]|output[: ]|repeat\b|for\s+each\b|until\s+|"
    r"while\s+|initialize\b|step\s+\d|\d+\.\s+\w)"
)
PYTHONISH = re.compile(r"\b(import|def\s+\w+\s*\(|return\b|self\.|np\.|jnp\.|plt\.)")


@dataclass
class TutorialReport:
    tutorial: str
    sections_present: list[str] = field(default_factory=list)
    sections_missing: list[str] = field(default_factory=list)
    pseudocode: str = "missing"
    pseudocode_evidence: str = ""
    figure_refs: list[str] = field(default_factory=list)
    figure_missing: list[str] = field(default_factory=list)
    table_refs: list[str] = field(default_factory=list)
    table_missing: list[str] = field(default_factory=list)
    word_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def split_sections(md: str) -> dict[str, str]:
    """Return {heading: body_text} for level-2 sections."""
    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = m.group(1).strip()
            current_lines = []
        else:
            if current_heading is not None:
                current_lines.append(line)
    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines).strip()
    return sections


def detect_pseudocode(solution_text: str) -> tuple[str, str]:
    """Classify pseudocode quality inside the Solution Method body.

    Returns (verdict, evidence_snippet). Verdicts: present / inadequate / missing.
    """
    if not solution_text:
        return "missing", "no Solution Method body"

    code_blocks = re.findall(r"```(\w*)\n(.*?)```", solution_text, flags=re.DOTALL)
    has_pseudo_block = False
    has_python_block = False
    evidence = ""

    for lang, body in code_blocks:
        is_python_lang = lang.lower() in {"python", "py"}
        looks_pythonish = bool(PYTHONISH.search(body))
        looks_pseudo = bool(PSEUDOCODE_HINTS.search(body))
        if is_python_lang or looks_pythonish:
            has_python_block = True
            continue
        if looks_pseudo or lang.lower() in {"text", "pseudocode", ""} and looks_pseudo:
            has_pseudo_block = True
            evidence = body.strip().splitlines()[0][:120] if body.strip() else ""
            break

    if has_pseudo_block:
        return "present", evidence

    # check for numbered-step lists outside fences
    if re.search(r"(?m)^\s*1\.\s+\S+", solution_text) and re.search(
        r"(?m)^\s*2\.\s+\S+", solution_text
    ):
        return "present", "numbered step list (no code fence)"

    if has_python_block and not has_pseudo_block:
        return "inadequate", "only Python code blocks; no pseudocode"

    return "missing", ""


def find_figure_refs(md: str) -> list[str]:
    refs = re.findall(r'<img[^>]+src="([^"]+)"', md)
    refs += re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    return refs


def find_table_path_refs(md: str) -> list[str]:
    """Markdown links pointing into a tables/ folder, if any."""
    refs = re.findall(r"\]\((tables/[^)]+)\)", md)
    return refs


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def check_tutorial(folder: Path) -> TutorialReport:
    rel = folder.relative_to(ROOT).as_posix()
    report = TutorialReport(tutorial=rel)
    readme = folder / "README.md"
    if not readme.exists():
        report.notes.append("README.md missing")
        report.sections_missing = list(REQUIRED_SECTIONS)
        return report

    md = readme.read_text()
    sections = split_sections(md)
    present = [s for s in REQUIRED_SECTIONS if s in sections]
    missing = [s for s in REQUIRED_SECTIONS if s not in sections]
    report.sections_present = present
    report.sections_missing = missing

    for name in present:
        report.word_counts[name] = word_count(sections[name])

    sm = sections.get("Solution Method", "")
    verdict, evidence = detect_pseudocode(sm)
    report.pseudocode = verdict
    report.pseudocode_evidence = evidence

    figs = find_figure_refs(md)
    report.figure_refs = figs
    for ref in figs:
        if ref.startswith(("http://", "https://")):
            continue
        path = (folder / ref).resolve()
        if not path.exists():
            report.figure_missing.append(ref)

    tbl_refs = find_table_path_refs(md)
    report.table_refs = tbl_refs
    for ref in tbl_refs:
        path = (folder / ref).resolve()
        if not path.exists():
            report.table_missing.append(ref)

    if "Overview" in report.word_counts and report.word_counts["Overview"] < 60:
        report.notes.append(f"thin Overview ({report.word_counts['Overview']} words)")
    if "Takeaway" in report.word_counts and report.word_counts["Takeaway"] < 40:
        report.notes.append(f"thin Takeaway ({report.word_counts['Takeaway']} words)")

    return report


def list_tutorials(subject: str, only: set[str] | None = None) -> list[Path]:
    subject_dir = ROOT / subject
    if not subject_dir.is_dir():
        sys.exit(f"Subject not found: {subject}")
    tutorials = sorted(p for p in subject_dir.iterdir() if (p / "run.py").exists())
    if only:
        tutorials = [p for p in tutorials if p.name in only]
        found = {p.name for p in tutorials}
        missing = sorted(only - found)
        if missing:
            sys.exit(f"Tutorial slug(s) not found under {subject}: {', '.join(missing)}")
    return tutorials


def render_text(reports: list[TutorialReport]) -> str:
    lines = []
    for r in reports:
        lines.append(f"### {r.tutorial}")
        lines.append(f"  sections present: {', '.join(r.sections_present) or '(none)'}")
        if r.sections_missing:
            lines.append(f"  sections MISSING: {', '.join(r.sections_missing)}")
        lines.append(f"  pseudocode: {r.pseudocode}" + (f"  ({r.pseudocode_evidence})" if r.pseudocode_evidence else ""))
        if r.figure_missing:
            lines.append(f"  BROKEN figure refs: {', '.join(r.figure_missing)}")
        if r.table_missing:
            lines.append(f"  BROKEN table refs: {', '.join(r.table_missing)}")
        if r.notes:
            lines.append(f"  notes: {'; '.join(r.notes)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", help="subject folder name, e.g. dynamic-programming")
    ap.add_argument("--only", default="", help="comma-separated tutorial slugs to limit to")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--out", help="write output to file as well as stdout")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    tutorials = list_tutorials(args.subject, only=only or None)
    reports = [check_tutorial(t) for t in tutorials]

    if args.json:
        payload = json.dumps([asdict(r) for r in reports], indent=2)
    else:
        payload = render_text(reports)

    print(payload)
    if args.out:
        Path(args.out).write_text(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
