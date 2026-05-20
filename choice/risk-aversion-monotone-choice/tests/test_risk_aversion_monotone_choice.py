"""Faithfulness test for the risk-aversion-monotone-choice tutorial.

Covers the single non-HOLDS finding from
bullshit-detector_risk-aversion-monotone-choice_2026-05-20.md:

  F1 (DATA DRIFT, LOW) -- the "Monotonicity violations" column is fed by
      three estimator helpers that applied two different sign thresholds
      (-1e-12 in estimate_unconstrained_logits and estimate_fixed_scale_crra,
      -1e-10 in estimate_monotone_logits). One summary column should use one
      measurement definition.

Violated-invariant test: passes while the source still carries the mixed
literal thresholds. Honest-fix test: passes once a single named constant
defines the threshold for every violation count.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_PY = (HERE.parent / "run.py").read_text()


def test_f1_violated_invariant_mixed_literal_thresholds():
    """Violated invariant: passes while the buggy code carries both a
    `< -1e-12` literal and a `< -1e-10` literal in violation counts.

    Must FAIL after the honest fix routes every violation count through one
    named constant (no `< -1e-12` literal remains)."""
    assert "< -1e-12" in RUN_PY and "< -1e-10" in RUN_PY


def test_f1_honest_fix_single_named_threshold():
    """Honest fix: every monotonicity-violation count uses one shared named
    constant, so the summary column has a single measurement definition."""
    assert "MONOTONICITY_TOL" in RUN_PY
    # No raw violation-threshold literals remain in the diff() comparisons.
    assert "np.diff(shares) < -1e-12" not in RUN_PY
    assert "np.diff(probabilities) < -1e-12" not in RUN_PY
    assert "np.diff(probabilities) < -1e-10" not in RUN_PY
    # All three violation counts share the constant.
    assert RUN_PY.count("< -MONOTONICITY_TOL") == 3
