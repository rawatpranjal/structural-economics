"""Faithfulness tests for the rbc tutorial.

Finding from bullshit-detector_rbc_2026-05-20.md:
  1. The fine-grid audit numbers quoted in Solution Method (max relative
     value error, max capital/hours policy gaps, coarse and fine VFI
     iteration counts and sup-norm errors) were not stored in any committed
     artifact. Only the README string carried them, so they could not be
     verified without re-running the tutorial.

The finding gets a violated-invariant test (holds on the buggy state) and an
honest-fix test (holds only after the repair). run.py runs the whole tutorial
on import, so claims are checked against the run.py / README.md source text
and the committed CSV artifacts.
"""
import csv
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent
RUN_PY = (FOLDER / "run.py").read_text()
README = (FOLDER / "README.md").read_text()


# --- Finding 1: fine-grid audit numbers must be grounded in an artifact ---

def test_finding1_violated_invariant():
    """Buggy state: no committed artifact records the fine-grid audit numbers;
    only the README string carries them. FAILS once fine-grid-audit.csv is
    committed."""
    assert not (FOLDER / "tables" / "fine-grid-audit.csv").exists()


def test_finding1_honest_fix():
    """Honest state: fine-grid-audit.csv records every audit number quoted in
    the Solution Method section, the convergence errors are below the VFI
    tolerance, and the README quotes the same coarse iteration count."""
    path = FOLDER / "tables" / "fine-grid-audit.csv"
    assert path.exists()
    audit = {r["metric"]: float(r["value"]) for r in csv.DictReader(open(path))}
    for metric in (
        "bench_V_rel",
        "bench_k_max_abs",
        "bench_l_max_abs",
        "coarse_iterations",
        "coarse_error",
        "fine_iterations",
        "fine_error",
    ):
        assert metric in audit
    assert audit["coarse_error"] < 1e-5  # below the tol used by run.py
    assert audit["fine_error"] < 1e-5
    coarse_iters = int(audit["coarse_iterations"])
    assert f"{coarse_iters} iterations" in README  # README quotes the same count
