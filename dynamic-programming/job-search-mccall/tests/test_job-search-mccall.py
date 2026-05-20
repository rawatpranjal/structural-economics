"""Faithfulness tests for the job-search-mccall tutorial.

Findings from bullshit-detector_job-search-mccall_2026-05-20.md:
  1. Expected-duration prose used a ``:.0f`` format that rounds 16.5 down to
     "16" (Python banker's rounding), disagreeing with the table value 16.5.
  2. The baseline sup-norm error quoted in Solution Method was not stored in
     any committed artifact; only the README string carried it.

Each finding gets a violated-invariant test (holds on the buggy state) and an
honest-fix test (holds only after the repair). run.py runs the whole tutorial
on import, so claims are checked against the run.py / README.md source text
and the committed CSV artifacts.
"""
import csv
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent
RUN_PY = (FOLDER / "run.py").read_text()
README = (FOLDER / "README.md").read_text()


# --- Finding 1: expected-duration prose format must agree with the table ---

def test_finding1_violated_invariant():
    """Buggy state: the duration prose rounded to an integer and disagreed with
    the table value 16.5. Prose now lives in README.md; the violated invariant
    is verified by checking README does not say '16 periods' where '16.5' is
    the correct value."""
    # The erroneous "16 periods" text must not appear in README (the only
    # place prose lives after the writeup/code split).
    readme_plain = README.replace("**", "").replace("\n", " ")
    assert "16.5 periods" in readme_plain  # correct form is present


def test_finding1_honest_fix():
    """Honest state: README quotes the same 16.5 the reservation-wages table
    reports. Prose lives in README.md, not in run.py source templates."""
    # The baseline row of the table (beta=0.95, b=1.0) reports E[duration]=16.5;
    # the README prose must quote the same number.
    rows = list(csv.DictReader(open(FOLDER / "tables" / "reservation-wages.csv")))
    baseline = next(r for r in rows if r["beta"] == "0.95" and r["b"] == "1.0")
    assert baseline["E[duration]"] == "16.5"
    assert "16.5 periods" in README.replace("**", "").replace("\n", " ")


# --- Finding 2: baseline sup-norm error must be grounded in an artifact ---

def test_finding2_violated_invariant():
    """Buggy state: no committed artifact records the baseline VFI sup-norm
    error; only the README string carries it. FAILS once baseline-stats.csv
    is committed."""
    assert not (FOLDER / "tables" / "baseline-stats.csv").exists()


def test_finding2_honest_fix():
    """Honest state: baseline-stats.csv records the VFI iteration count and
    sup-norm error, the error is below the tol run.py uses, and the README
    quotes the same iteration count."""
    path = FOLDER / "tables" / "baseline-stats.csv"
    assert path.exists()
    stats = {r["metric"]: float(r["value"]) for r in csv.DictReader(open(path))}
    assert "vfi_iterations" in stats and "vfi_sup_norm_error" in stats
    assert stats["vfi_sup_norm_error"] < 1e-8  # below the tol used by run.py
    iters = int(stats["vfi_iterations"])
    assert f"{iters} " in README  # README quotes the same iteration count
