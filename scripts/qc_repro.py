#!/usr/bin/env python3
"""Reproducibility sweep for a subject's tutorials.

For each tutorial folder under <subject>/, snapshots README.md, figures/, and
tables/ (sha256), runs `python3 run.py` with cwd=<tutorial>, then re-snapshots
and reports diffs.

Usage:
    python3 scripts/qc_repro.py <subject> [--timeout 300] [--out path.json]
    python3 scripts/qc_repro.py <subject> --only cake-eating,rbc
    python3 scripts/qc_repro.py <subject> --only cake-eating --restore

Output JSON is a list of per-tutorial records:
    {
      "tutorial": "dynamic-programming/cake-eating",
      "exit_code": 0,
      "wall_seconds": 12.3,
      "stdout_tail": "...",
      "stderr_tail": "...",
      "changed": [{"path": "figures/policy-function.png", "kind": "png",
                   "before_sha": "...", "after_sha": "...",
                   "size_before": 12345, "size_after": 12350,
                   "classification": "cosmetic|substantive|unknown",
                   "detail": "..."}],
      "restore_requested": true,
      "restored": true,
      "restore_errors": [],
      "timed_out": false
    }

Run sequentially to avoid JAX/BLAS thrash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FileChange:
    path: str
    kind: str
    before_sha: str | None
    after_sha: str | None
    size_before: int | None
    size_after: int | None
    classification: str = "unknown"
    detail: str = ""


@dataclass
class TutorialRepro:
    tutorial: str
    exit_code: int | None = None
    wall_seconds: float = 0.0
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    changed: list[FileChange] = field(default_factory=list)
    restore_requested: bool = False
    restored: bool = False
    restore_errors: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(folder: Path) -> dict[str, tuple[str, int]]:
    """Return {relpath: (sha256, size_bytes)} for tracked artifacts."""
    snap: dict[str, tuple[str, int]] = {}
    for sub in ("README.md",):
        p = folder / sub
        if p.exists() and p.is_file():
            snap[sub] = (sha256_of(p), p.stat().st_size)
    for sub in ("figures", "tables"):
        d = folder / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(folder).as_posix()
            snap[rel] = (sha256_of(p), p.stat().st_size)
    return snap


def kind_of(rel: str) -> str:
    if rel.endswith(".png"):
        return "png"
    if rel.endswith(".csv"):
        return "csv"
    if rel.endswith(".md"):
        return "md"
    return "other"


def csv_numeric_close(a: Path, b: Path, atol: float = 1e-6, rtol: float = 1e-4) -> tuple[bool, str]:
    """Return (close, detail). Compares two CSVs numerically with tolerances."""
    try:
        with a.open() as fa, b.open() as fb:
            ra = list(csv.reader(fa))
            rb = list(csv.reader(fb))
    except Exception as e:  # noqa: BLE001
        return False, f"csv read error: {e}"
    if len(ra) != len(rb):
        return False, f"row count {len(ra)}->{len(rb)}"
    max_diff = 0.0
    for i, (rowa, rowb) in enumerate(zip(ra, rb)):
        if len(rowa) != len(rowb):
            return False, f"row {i} cols {len(rowa)}->{len(rowb)}"
        for j, (xa, xb) in enumerate(zip(rowa, rowb)):
            if xa == xb:
                continue
            try:
                fa_v = float(xa)
                fb_v = float(xb)
            except ValueError:
                return False, f"text mismatch at row {i} col {j}: {xa!r} vs {xb!r}"
            diff = abs(fa_v - fb_v)
            scale = max(abs(fa_v), abs(fb_v), 1.0)
            if diff > atol + rtol * scale:
                return False, f"row {i} col {j} diff {diff:.3e} (a={fa_v}, b={fb_v})"
            max_diff = max(max_diff, diff)
    return True, f"max numeric diff {max_diff:.3e}"


def png_close(a: Path, b: Path) -> tuple[bool, str]:
    """Best-effort PNG cosmetic-vs-substantive check.

    Without Pillow guaranteed available, classify by size delta only:
      - both files same size -> "cosmetic" (very likely matplotlib metadata noise)
      - size differs by < 1% -> "cosmetic" (PIL pixel rounding)
      - size differs more   -> "unknown" (let reviewer decide)
    """
    sa = a.stat().st_size
    sb = b.stat().st_size
    if sa == sb:
        return True, "byte-size identical"
    if max(sa, sb) == 0:
        return False, "empty"
    pct = abs(sa - sb) / max(sa, sb)
    if pct < 0.01:
        return True, f"size delta {pct:.2%} (<1%)"
    return False, f"size delta {pct:.2%}"


def md_text_close(a: Path, b: Path) -> tuple[bool, str]:
    """Treat README diffs as substantive unless only whitespace differs."""
    ta = a.read_text(errors="replace")
    tb = b.read_text(errors="replace")
    if ta == tb:
        return True, "identical"
    if " ".join(ta.split()) == " ".join(tb.split()):
        return True, "whitespace-only"
    return False, "text differs"


def classify(folder_after: Path, before_blob: dict[str, bytes], rel: str) -> tuple[str, str]:
    """Classify a changed file as cosmetic or substantive."""
    after = folder_after / rel
    if not after.exists():
        return "substantive", "file removed by run"
    if rel not in before_blob:
        return "substantive", "new file"
    # write before content to a temp path to compare
    tmp = after.parent / (after.name + ".__before__")
    tmp.write_bytes(before_blob[rel])
    try:
        k = kind_of(rel)
        if k == "csv":
            ok, detail = csv_numeric_close(tmp, after)
            return ("cosmetic" if ok else "substantive"), detail
        if k == "png":
            ok, detail = png_close(tmp, after)
            return ("cosmetic" if ok else "unknown"), detail
        if k == "md":
            ok, detail = md_text_close(tmp, after)
            return ("cosmetic" if ok else "substantive"), detail
        return "unknown", "unrecognized file type"
    finally:
        tmp.unlink(missing_ok=True)


def stream_tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")[-2000:]
    return value[-2000:]


def restore_snapshot(
    folder: Path,
    before: dict[str, tuple[str, int]],
    before_blob: dict[str, bytes],
) -> list[str]:
    """Restore README/figure/table artifacts to their pre-run snapshot."""
    errors: list[str] = []
    after = snapshot(folder)
    for rel in sorted(set(before) | set(after)):
        target = folder / rel
        try:
            if rel in before_blob:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(before_blob[rel])
            elif target.exists():
                target.unlink()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{rel}: restore failed: {e}")

    restored = snapshot(folder)
    for rel in sorted(set(before) | set(restored)):
        if before.get(rel) != restored.get(rel):
            errors.append(f"{rel}: still differs after restore")
    return errors


def run_tutorial(folder: Path, timeout: int, restore: bool = False) -> TutorialRepro:
    rel = folder.relative_to(ROOT).as_posix()
    rep = TutorialRepro(tutorial=rel)
    rep.restore_requested = restore
    if not (folder / "run.py").exists():
        rep.skipped = True
        rep.skip_reason = "no run.py"
        return rep

    before = snapshot(folder)
    before_blob = {
        relp: (folder / relp).read_bytes() for relp in before if (folder / relp).exists()
    }

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("PYTHONUNBUFFERED", "1")

    t0 = time.time()
    try:
        proc = subprocess.run(
            ["python3", "run.py"],
            cwd=folder,
            env=env,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        rep.exit_code = proc.returncode
        rep.stdout_tail = proc.stdout[-2000:]
        rep.stderr_tail = proc.stderr[-2000:]
    except subprocess.TimeoutExpired as e:
        rep.timed_out = True
        rep.exit_code = None
        rep.stdout_tail = stream_tail(e.stdout)
        rep.stderr_tail = stream_tail(e.stderr)
    rep.wall_seconds = time.time() - t0

    after = snapshot(folder)
    all_paths = sorted(set(before) | set(after))
    for rp in all_paths:
        b = before.get(rp)
        a = after.get(rp)
        if b == a:
            continue
        cls, detail = classify(folder, before_blob, rp)
        rep.changed.append(
            FileChange(
                path=rp,
                kind=kind_of(rp),
                before_sha=b[0] if b else None,
                after_sha=a[0] if a else None,
                size_before=b[1] if b else None,
                size_after=a[1] if a else None,
                classification=cls,
                detail=detail,
            )
        )

    if restore:
        rep.restore_errors = restore_snapshot(folder, before, before_blob)
        rep.restored = not rep.restore_errors

    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", help="subject folder, e.g. dynamic-programming")
    ap.add_argument("--timeout", type=int, default=300, help="per-tutorial timeout seconds")
    ap.add_argument("--only", default="", help="comma-separated tutorial slugs to limit to")
    ap.add_argument(
        "--restore",
        action="store_true",
        help="restore README/figure/table artifacts after classifying regeneration diffs",
    )
    ap.add_argument("--out", default="", help="write JSON output to this path")
    args = ap.parse_args()

    subj = ROOT / args.subject
    if not subj.is_dir():
        sys.exit(f"subject not found: {args.subject}")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    tutorials = sorted(p for p in subj.iterdir() if (p / "run.py").exists())
    if only:
        tutorials = [p for p in tutorials if p.name in only]

    reports: list[TutorialRepro] = []
    for t in tutorials:
        print(f"[qc-repro] {t.relative_to(ROOT)} ...", flush=True)
        rep = run_tutorial(t, args.timeout, restore=args.restore)
        status = "TIMEOUT" if rep.timed_out else (f"exit={rep.exit_code}")
        print(
            f"[qc-repro] {t.name}: {status} t={rep.wall_seconds:.1f}s "
            f"changed={len(rep.changed)} restored={rep.restored if args.restore else 'n/a'}",
            flush=True,
        )
        reports.append(rep)

    payload = json.dumps([asdict(r) for r in reports], indent=2)
    if args.out:
        Path(args.out).write_text(payload)
        print(f"[qc-repro] wrote {args.out}", flush=True)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
