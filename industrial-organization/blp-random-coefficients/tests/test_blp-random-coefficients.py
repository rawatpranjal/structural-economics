"""Faithfulness tests for the BLP random-coefficients tutorial.

The bullshit-detector audit (2026-05-20) flagged four DATA DRIFT findings:
runtime scalar values (contraction iterations, max delta recovery error,
GMM objective evaluations, max own-elasticity error) committed to README
prose with no backing CSV artifact. The honest fix adds a committed
`tables/convergence-diagnostics.csv` so every such number can be
cross-checked against an on-disk artifact.

Tests read the tutorial source / artifacts as text; they do not execute
``run.py`` (which would re-run the whole estimation).
"""
import re
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
README = FOLDER / "README.md"
RUN_PY = FOLDER / "run.py"
DIAG_CSV = FOLDER / "tables" / "convergence-diagnostics.csv"


def _readme() -> str:
    return README.read_text()


def _run_src() -> str:
    return RUN_PY.read_text()


# --- Finding 1: contraction iteration count -------------------------------

def test_violated_contraction_count_unbacked():
    """Violated invariant: the iteration count sits in README prose with no
    backing diagnostics CSV. PASSES on the buggy state, FAILS once the
    diagnostics CSV exists and carries the iteration count."""
    text = _readme()
    has_count = re.search(r"\bin \*\*\d+ iterations\*\*", text) is not None
    no_diag_csv = not DIAG_CSV.exists()
    assert has_count and no_diag_csv


def test_fixed_contraction_count_backed_by_csv():
    """Honest fix: the contraction iteration count is written to a committed
    diagnostics CSV. FAILS on the buggy state, PASSES after the fix."""
    assert DIAG_CSV.exists()
    csv_text = DIAG_CSV.read_text()
    assert "contraction_iters" in csv_text
    m = re.search(r"\bin \*\*(\d+) iterations\*\*", _readme())
    assert m is not None
    iters = m.group(1)
    rows = dict(
        line.split(",", 1) for line in csv_text.strip().splitlines()[1:]
    )
    assert rows["contraction_iters"].strip() == iters


# --- Finding 2: max delta recovery error ----------------------------------

def test_violated_delta_error_unbacked():
    """Violated invariant: the recovery error appears only in README prose."""
    has_err = "delta^{\\mathrm{recovered}}" in _readme()
    assert has_err and not DIAG_CSV.exists()


def test_fixed_delta_error_backed_by_csv():
    """Honest fix: max delta recovery error is a row in the diagnostics CSV."""
    assert DIAG_CSV.exists()
    assert "max_delta_error" in DIAG_CSV.read_text()


# --- Finding 3: GMM objective evaluation count ----------------------------

def test_violated_gmm_evals_unbacked_and_grid_undisclosed():
    """Violated invariant: README states the GMM eval count with no backing
    CSV, and does not disclose that the starting grid is a separate set of
    evaluations."""
    text = _readme()
    has_count = "evaluated the objective" in text
    grid_undisclosed = "grid evaluated the objective" not in text
    assert has_count and grid_undisclosed and not DIAG_CSV.exists()


def test_fixed_gmm_evals_backed_and_grid_disclosed():
    """Honest fix: GMM eval count is in the diagnostics CSV and the README
    discloses the starting-grid evaluations are separate."""
    assert DIAG_CSV.exists()
    assert "gmm_nfev" in DIAG_CSV.read_text()
    assert "grid evaluated the objective" in _readme()


# --- Finding 4: max own-elasticity error ----------------------------------

def test_violated_own_elast_error_unbacked():
    """Violated invariant: own-elasticity error is README-only prose."""
    has_err = "largest own-elasticity error" in _readme()
    assert has_err and not DIAG_CSV.exists()


def test_fixed_own_elast_error_backed_by_csv():
    """Honest fix: max own-elasticity error is a row in the diagnostics CSV."""
    assert DIAG_CSV.exists()
    assert "max_own_elast_error" in DIAG_CSV.read_text()
