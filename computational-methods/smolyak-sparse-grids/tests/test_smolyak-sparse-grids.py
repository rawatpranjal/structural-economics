"""Faithfulness tests for smolyak-sparse-grids tutorial.

Audit: bullshit-detector_smolyak-sparse-grids_2026-05-20.md
Findings F1 (MISLABELED), F2/F3/F4 (DATA DRIFT).

run.py executes the whole tutorial on import, so claims are tested against
the run.py source text via file reads rather than by importing.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_PY = (HERE.parent / "run.py").read_text()

ALPHA = 0.36
A = np.array([1.0, 0.9, 1.1, 0.8])
W = A ** (1.0 / (1.0 - ALPHA))
Z = W.sum()
OMEGA = W / Z


# --- Finding 1: "absolute Euler error" mislabels a relative metric ---------

def test_f1_violated_invariant_metric_is_a_ratio():
    # Code computes a dimensionless ratio |lhs/rhs - 1|, not a raw difference.
    assert "lhs / rhs - 1" in RUN_PY


def test_f1_honest_fix_readme_prose_says_relative():
    readme = (HERE.parent / "README.md").read_text().lower()
    assert "relative euler error" in readme
    assert "absolute euler error" not in readme


# --- Finding 2: worked A^(1/(1-alpha)) values ------------------------------

def test_f2_violated_invariant_buggy_value_wrong():
    # Audit-claimed 0.846 for A[1]; true value 0.848 -> buggy claim fails.
    assert not abs(W[1] - 0.846) < 5e-4


def test_f2_honest_fix_corrected_value():
    assert abs(W[1] - 0.848) < 5e-4
    # README must carry the corrected triple, not the wrong one.
    assert "0.846" not in RUN_PY
    assert "1.000, 0.848, 1.161, 0.706" in RUN_PY


# --- Finding 3: Z approx 3.710 -> 3.714 ------------------------------------

def test_f3_violated_invariant_buggy_Z_wrong():
    assert not abs(Z - 3.710) < 1e-3


def test_f3_honest_fix_corrected_Z():
    assert abs(Z - 3.714) < 1e-3
    assert "Z \\approx 3.710" not in RUN_PY
    assert "Z \\approx 3.714" in RUN_PY


# --- Finding 4: omega[2] 0.313 (Equations) vs 0.312 (Setup) ----------------

def test_f4_violated_invariant_buggy_omega_wrong():
    assert not abs(OMEGA[2] - 0.313) < 5e-4


def test_f4_honest_fix_corrected_omega():
    assert abs(OMEGA[2] - 0.312) < 5e-4
    assert "0.269, 0.228, 0.313, 0.190" not in RUN_PY
    assert "0.269, 0.228, 0.312, 0.190" in RUN_PY
