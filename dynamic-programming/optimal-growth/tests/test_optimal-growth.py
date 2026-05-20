"""Faithfulness tests for the optimal-growth tutorial.

Each finding from bullshit-detector_optimal-growth_2026-05-20.md gets two
checks: a violated-invariant test that holds on the buggy artifact state and
an honest-fix test that holds only once the finding is repaired. After the
fix the violated-invariant test must FAIL and the honest-fix test must PASS.

run.py executes the whole tutorial on import, so structural claims are tested
against the run.py / README.md source text and the committed CSV artifacts.
"""
import csv
from pathlib import Path

import pytest

FOLDER = Path(__file__).resolve().parent.parent
RUN_PY = (FOLDER / "run.py").read_text()
README = (FOLDER / "README.md").read_text()


# --- Finding 1: pseudocode omits the load-bearing 0.9999 upper-bound factor ---

def test_finding1_violated_invariant():
    """Buggy state: the README pseudocode shows kp_max with no 0.9999 factor,
    so a reader transcribing it builds a solver that hits log(0). FAILS once
    the pseudocode is fixed to disclose the factor."""
    assert "0.9999" not in README


def test_finding1_honest_fix():
    """Honest state: the pseudocode discloses the 0.9999 factor that run.py
    uses on line `kp_max = min(output * 0.9999, k_max)`."""
    assert "0.9999" in README
    assert "kp_max = min(output * 0.9999, k_max)" in RUN_PY


# --- Findings 2-3: max value/policy errors not grounded in a committed table ---

def _comparison_rows():
    with open(FOLDER / "tables" / "comparison.csv") as fh:
        return list(csv.DictReader(fh))


def test_finding2_3_violated_invariant():
    """Buggy state: the 8-row comparison.csv cannot reproduce the
    'largest gap outside the bottom decile' claims; its row maxima are
    strictly below the README's reported max errors."""
    rows = _comparison_rows()
    max_v = max(abs(float(r["V error"])) for r in rows)
    max_kp = max(abs(float(r["k' error"])) for r in rows)
    # comparison.csv samples 8 points; the true 500-point maxima exceed these.
    assert max_v < 1.9e-05
    assert max_kp < 2.8e-02


def test_finding2_honest_fix():
    """Honest state: a committed full-errors.csv covers all 500 grid points,
    so the reported max value gap is grounded in an artifact."""
    path = FOLDER / "tables" / "full-errors.csv"
    assert path.exists()
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 500
    valid = rows[50:]  # bottom decile excluded, matches valid_start = n_grid // 10
    max_v = max(abs(float(r["V error"])) for r in valid)
    log = {r["metric"]: float(r["value"])
           for r in csv.DictReader(open(FOLDER / "tables" / "convergence-log.csv"))}
    assert max_v == pytest.approx(log["max_value_error_above_bottom_decile"], rel=1e-6)


def test_finding3_honest_fix():
    """Honest state: the reported max policy gap is grounded in full-errors.csv."""
    with open(FOLDER / "tables" / "full-errors.csv") as fh:
        rows = list(csv.DictReader(fh))
    valid = rows[50:]
    max_kp = max(abs(float(r["k' error"])) for r in valid)
    log = {r["metric"]: float(r["value"])
           for r in csv.DictReader(open(FOLDER / "tables" / "convergence-log.csv"))}
    assert max_kp == pytest.approx(log["max_policy_error_above_bottom_decile"], rel=1e-6)


# --- Finding 4: convergence iteration count / residual not in any artifact ---

def test_finding4_violated_invariant():
    """Buggy state: no committed artifact records the VFI iteration count or
    residual; only the README string carries them. FAILS once
    convergence-log.csv is committed."""
    assert not (FOLDER / "tables" / "convergence-log.csv").exists()


def test_finding4_honest_fix():
    """Honest state: convergence-log.csv records the VFI iteration count and
    residual, and the README quotes a count consistent with that artifact."""
    path = FOLDER / "tables" / "convergence-log.csv"
    assert path.exists()
    log = {r["metric"]: float(r["value"]) for r in csv.DictReader(open(path))}
    assert "vfi_iterations" in log and "vfi_sup_norm_error" in log
    assert log["vfi_sup_norm_error"] < 1e-6  # below the tol used by run.py
    iters = int(log["vfi_iterations"])
    assert f"{iters} " in README  # README quotes the same iteration count
