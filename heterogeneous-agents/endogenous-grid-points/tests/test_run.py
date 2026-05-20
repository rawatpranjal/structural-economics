"""Faithfulness tests for the endogenous-grid-points tutorial.

These tests guard the three findings of the 2026-05-20 bullshit-detector audit:

  Finding 1 (FALSE):      perfect-foresight MPC limit formula.
  Finding 2 (DILUTED):    Carroll (1997) patience condition statement.
  Finding 3 (MISLABELED): "Average local MPC" label / disclosure.

Importing ``run.py`` would execute the entire tutorial, so the tests inspect
the ``run.py`` source text and the generated ``README.md`` instead, and they
re-derive the Carroll closed form numerically from the tutorial's calibration.
"""

from pathlib import Path

import pytest

# Tutorial calibration, mirrored from run.py lines 203-207.
GAMMA = 2.0
BETA = 0.95
R = 1.03
BETA_R = BETA * R  # 0.9785

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
RUN_PY = TUTORIAL_DIR / "run.py"
README = TUTORIAL_DIR / "README.md"


def run_source() -> str:
    return RUN_PY.read_text()


def readme_text() -> str:
    return README.read_text() if README.exists() else ""


def carroll_pf_mpc() -> float:
    """Carroll (1997/2006) perfect-foresight MPC for infinite-horizon CRRA.

    kappa* = 1 - (beta*R)^(1/gamma) / R = 1 - G_c / R.
    """
    g_c = BETA_R ** (1.0 / GAMMA)
    return 1.0 - g_c / R


def wrong_pf_mpc() -> float:
    """The pre-fix formula: R*(beta*R)^(-1/gamma) - 1 = R/G_c - 1."""
    return R * (BETA_R ** (-1.0 / GAMMA)) - 1.0


# ---------------------------------------------------------------------------
# Finding 1: perfect-foresight MPC limit formula
# ---------------------------------------------------------------------------

def test_finding1_violated_invariant():
    """Violated-invariant: the buggy formula differs from Carroll by ~0.0016.

    PASSES on buggy code, FAILS once the code uses the Carroll closed form.
    """
    src = run_source()
    assert "gross_return * (beta_r ** (-1.0 / gamma)) - 1.0" in src
    assert abs(wrong_pf_mpc() - carroll_pf_mpc()) > 1e-4


def test_finding1_honest_fix():
    """Honest-fix: mpclim in run.py must equal the Carroll closed form.

    FAILS on buggy code, PASSES once run.py uses 1 - (beta_r**(1/gamma))/gross_return.
    """
    src = run_source()
    assert "1.0 - (beta_r ** (1.0 / gamma)) / gross_return" in src
    # The Carroll formula evaluates to ~0.0396 at this calibration.
    assert abs(carroll_pf_mpc() - 0.0396) < 1e-3
    # The README must report 0.0396, not the wrong 0.0413.
    assert "0.0413" not in readme_text()
    assert "0.0396" in readme_text()


# ---------------------------------------------------------------------------
# Finding 2: Carroll (1997) patience condition
# ---------------------------------------------------------------------------

def test_finding2_violated_invariant():
    """Violated-invariant: README never states the growth-impatience condition.

    PASSES on buggy README, FAILS after the honest fix names G_c < R.
    """
    text = readme_text()
    assert "rules out the unbounded-saving target of Carroll" in text
    assert "G_c < R" not in text
    assert "growth-impatience" not in text


def test_finding2_honest_fix():
    """Honest-fix: README states Carroll's growth-impatience condition G_c < R.

    FAILS on buggy README, PASSES after the description names G_c < R.
    """
    text = readme_text()
    assert "G_c < R" in text or "growth-impatience" in text
    # The condition Carroll actually uses, restated correctly.
    g_c = BETA_R ** (1.0 / GAMMA)
    assert g_c < R


# ---------------------------------------------------------------------------
# Finding 3: "Average local MPC" label disclosure
# ---------------------------------------------------------------------------

def test_finding3_violated_invariant():
    """Violated-invariant: README never discloses the dc/da numerical derivative.

    PASSES on buggy README, FAILS after the honest fix discloses it.
    """
    text = readme_text()
    assert "1e-6" not in text
    assert "numerical derivative" not in text
    assert "dc/da" not in text


def test_finding3_honest_fix():
    """Honest-fix: README discloses the local-slope numerical derivative.

    FAILS on buggy README, PASSES after the label/description explains dc/da.
    """
    text = readme_text()
    assert "1e-6" in text or "numerical derivative" in text or "dc/da" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
