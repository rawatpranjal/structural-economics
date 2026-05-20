"""Faithfulness tests for the solow-growth tutorial.

These tests pin the single finding in
``bullshit-detector_solow-growth_2026-05-20.md``:

- Finding 1 (DILUTED, MED): the Results prose said "Any gap comes from finite
  horizon truncation. The geometric residual is about 2.21e-04", presenting
  the linearization residual as the explanation for the terminal gap. The
  actual gap shown in the table immediately above (and in
  ``tables/steady-state-comparison.csv``) is 2.73e-04, which is 23% larger.
  The geometric residual ``|k0 - k_star| * lambda^(T-1)`` is the
  linearization's PREDICTION of the remaining gap, not the actual gap; it
  underestimates because k0=1.0 starts deep in the nonlinear region. The
  honest fix rewrites the prose to distinguish the linear-approximation
  prediction from the actual gap.

The README is the hand-maintained canonical file. The run.py contains only
computation; prose lives in README.md.
"""
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
README = (TUTORIAL_DIR / "README.md").read_text()


def _solow_quantities():
    """Recompute steady state, linearization factor, geometric residual, and
    actual terminal gap from the tutorial's primitives."""
    alpha, s, delta, n, g = 0.33, 0.24, 0.06, 0.01, 0.02
    gross_dilution = (1 + g) * (1 + n)
    eff_dep = gross_dilution - 1 + delta
    k_star = (s / eff_dep) ** (1 / (1 - alpha))
    lam = ((1 - delta) + s * alpha * k_star ** (alpha - 1)) / gross_dilution
    k0, periods = 1.0, 160
    geom_res = abs(k0 - k_star) * lam ** (periods - 1)
    k = k0
    for _ in range(periods - 1):
        k = ((1 - delta) * k + s * k**alpha) / gross_dilution
    actual_gap = abs(k - k_star)
    return geom_res, actual_gap


# ---------------------------------------------------------------------------
# Finding 1: geometric residual conflated with the actual terminal gap.
# ---------------------------------------------------------------------------
def test_finding1_violated_invariant_geometric_residual_underestimates_gap():
    """Violated invariant: the geometric residual does NOT equal the actual
    terminal gap; it underestimates it by more than 5%.

    This encodes the discrepancy the buggy prose papered over. It holds on
    the current code and stays true regardless of prose wording, proving the
    two numbers are genuinely different quantities.
    """
    geom_res, actual_gap = _solow_quantities()
    # geometric residual is the linearization prediction, ~2.21e-04
    assert abs(geom_res - 2.21e-4) / 2.21e-4 < 0.05
    # actual terminal gap is ~2.73e-04
    assert abs(actual_gap - 2.73e-4) / 2.73e-4 < 0.05
    # they differ by more than 5%
    assert abs(geom_res - actual_gap) / actual_gap > 0.05


def test_finding1_honest_fix_readme_shows_both_numbers_distinctly():
    """Honest fix: the README states the actual gap (2.73e-04) and
    describes the geometric residual (2.21e-04) as a linear-approximation
    prediction, not as "the gap".
    """
    assert "2.73e-04" in README
    assert "2.21e-04" in README
    assert "linear approximation" in README
