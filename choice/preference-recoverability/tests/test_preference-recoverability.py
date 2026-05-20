"""Faithfulness tests for choice/preference-recoverability/.

Audit: bullshit-detector_preference-recoverability_2026-05-20.md
Finding 1 (DILUTED): the Overview said the linear program "finds utility
scores and supporting slopes", but the LP optimizes only u_t; lambda_t are
pre-fixed to 1/expenditure before the LP runs.
Finding 2 (DILUTED): pseudocode step 4 said "average_t u_t = 1", but the
literal LP equality constraint is sum_t u_t = T (b_eq = [n_obs]).
"""
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent.parent
README = TUTORIAL_DIR / "README.md"


def test_finding1_violated_invariant_overview_claims_lp_finds_slopes():
    """Violated invariant: Overview claimed the LP finds supporting slopes.

    PASSED on the buggy README; FAILS once the Overview is corrected.
    """
    text = README.read_text()
    assert "finds utility scores and supporting slopes" in text


def test_finding1_honest_fix_overview_says_slopes_prefixed():
    """Honest fix: Overview states slopes are fixed before the LP runs.

    FAILED on the buggy README; PASSES once the Overview is corrected.
    """
    text = README.read_text()
    assert "finds utility scores and supporting slopes" not in text
    assert "fixed to one over expenditure before the program runs" in text


def test_finding2_violated_invariant_pseudocode_says_average():
    """Violated invariant: pseudocode step 4 stated "average_t u_t = 1".

    PASSED on the buggy README; FAILS once the constraint is restated.
    """
    assert "average_t u_t = 1" in README.read_text()


def test_finding2_honest_fix_pseudocode_states_sum_constraint():
    """Honest fix: pseudocode states the literal LP constraint sum_t u_t = T.

    FAILED on the buggy README; PASSES once the wording matches the code.
    """
    text = README.read_text()
    assert "average_t u_t = 1" not in text
    assert "sum_t u_t = T" in text
