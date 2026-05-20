"""Faithfulness tests for the dynamic-discrete-choice tutorial.

The bullshit-detector audit (2026-05-20) flagged one DILUTED/MED finding:
the title and prose advertise an MCE-IRL estimator and assert a *numerical*
equivalence ("returns the same theta to within solver tolerance",
pseudocode "theta_IRL == theta_NFXP up to solver tolerance"), but no
MCE-IRL estimator is implemented and no theta_IRL artifact exists. The
honest, scope-respecting fix keeps the algebraic-equivalence statement
(a mathematical fact) but removes the unverified numerical phrasing and
retitles the topic as an interpretation rather than a demonstrated
numerical equivalence.

Tests read the tutorial source / README as text; they do not execute
``run.py``.
"""
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
README = FOLDER / "README.md"
RUN_PY = FOLDER / "run.py"


def _readme() -> str:
    return README.read_text()


def _run_src() -> str:
    return RUN_PY.read_text()


# --- Finding 1: MCE-IRL numerical equivalence asserted but never run ------

def test_violated_numerical_equivalence_claimed_without_estimator():
    """Violated invariant: the README asserts a numerical MCE-IRL result
    (estimator "returns the same theta to within solver tolerance" and
    pseudocode output "theta_IRL == theta_NFXP up to solver tolerance")
    while no MCE-IRL estimator function is implemented. PASSES on the
    buggy state, FAILS once the unverified numerical phrasing is removed."""
    text = _readme()
    src = _run_src()
    asserts_numerical = (
        "returns the same $\\theta$ to within solver tolerance" in text
        or "returns the same \\theta to within solver tolerance" in text
        or "theta_IRL == theta_NFXP up to solver tolerance" in text
    )
    no_estimator = "def estimate_mce_irl" not in src
    assert asserts_numerical and no_estimator


def test_fixed_no_unverified_numerical_equivalence_claim():
    """Honest fix: the README no longer asserts an unverified numerical
    MCE-IRL result. The algebraic-equivalence statement may remain; the
    "to within solver tolerance" / "theta_IRL == theta_NFXP" numerical
    phrasing must be gone. FAILS on the buggy state, PASSES after the fix."""
    text = _readme()
    assert "to within solver tolerance" not in text
    assert "theta_IRL == theta_NFXP" not in text


def test_fixed_title_does_not_advertise_demonstrated_equivalence():
    """Honest fix: the title no longer lists "MCE-IRL Equivalence" as a
    fourth coequal estimator alongside NFXP, CCP, MPEC, since no MCE-IRL
    run is performed. FAILS on the buggy state, PASSES after the fix."""
    assert "MCE-IRL Equivalence" not in _readme()
