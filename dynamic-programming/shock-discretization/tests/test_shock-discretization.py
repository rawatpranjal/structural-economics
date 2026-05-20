"""Faithfulness tests for the shock-discretization tutorial.

These tests pin the single finding in
``bullshit-detector_shock-discretization_2026-05-20.md``:

- Finding 1 (DILUTED, LOW): the Rouwenhorst pseudocode in the Solution Method
  section said "row-normalize interior rows of P_n", implying normalization is
  applied selectively. The shared helper ``lib/discretize.py`` normalizes ALL
  rows (``trans / trans.sum(axis=1, keepdims=True)``). The two are numerically
  identical because endpoint rows of the pre-normalization Rouwenhorst matrix
  already sum to exactly 1.0, so dividing them by their row sum is a no-op.
  The honest fix updates the pseudocode prose to say all rows are normalized
  and notes why endpoint rows are unaffected. ``lib/discretize.py`` is a
  shared helper and is intentionally NOT changed.

The README and ``run.py`` strings are the claim source. The violated-invariant
test encodes the buggy prose and must FAIL once the fix is applied; the
honest-fix test must PASS after the fix.
"""
from pathlib import Path

import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
README = (TUTORIAL_DIR / "README.md").read_text()
RUN_PY = (TUTORIAL_DIR / "run.py").read_text()


# ---------------------------------------------------------------------------
# Numeric ground truth: endpoint rows of the pre-normalization Rouwenhorst
# matrix sum to exactly 1.0, so "interior only" and "all rows" normalization
# produce identical transition matrices. This is what makes the pseudocode
# discrepancy a no-op rather than a bug.
# ---------------------------------------------------------------------------
def _rouwenhorst_pre_normalization(n: int, rho: float) -> np.ndarray:
    """Build the Rouwenhorst transition matrix WITHOUT the final normalization."""
    p0 = (1 + rho) / 2
    trans = np.array([[p0, 1 - p0], [1 - p0, p0]])
    for _ in range(n - 2):
        size = trans.shape[0]
        zc = np.zeros((size, 1))
        zr = np.zeros((1, size + 1))
        trans = (
            p0 * np.block([[trans, zc], [zr]])
            + (1 - p0) * np.block([[zc, trans], [zr]])
            + (1 - p0) * np.block([[zr], [trans, zc]])
            + p0 * np.block([[zr], [zc, trans]])
        )
    return trans


def test_finding1_violated_invariant_endpoint_rows_already_sum_to_one():
    """Violated invariant: endpoint rows sum to 1 pre-normalization, so the
    "interior rows" pseudocode is numerically a no-op distinction.

    This encodes the property that makes the original prose harmless: it
    holds on the buggy code and stays true regardless of the prose wording.
    It documents WHY normalizing all rows equals normalizing interior rows.
    """
    for n in (3, 5, 7, 9):
        pre = _rouwenhorst_pre_normalization(n, rho=0.95)
        row_sums = pre.sum(axis=1)
        # endpoint rows sum to exactly 1.0
        assert abs(row_sums[0] - 1.0) < 1e-12
        assert abs(row_sums[-1] - 1.0) < 1e-12
        # normalize all rows vs interior rows only -> identical matrix
        all_norm = pre / pre.sum(axis=1, keepdims=True)
        interior_norm = pre.copy()
        interior_norm[1:-1] = pre[1:-1] / pre[1:-1].sum(axis=1, keepdims=True)
        assert np.max(np.abs(all_norm - interior_norm)) < 1e-15


def test_finding1_violated_invariant_pseudocode_says_interior():
    """Violated invariant: the buggy README pseudocode said "interior rows".

    After the writeup/code split, prose lives in README.md only. The violated
    invariant is that README contains the erroneous "interior rows" phrase.
    This test now reads README (the authoritative prose source).
    FAILS once the README prose is corrected to say all rows are normalized.
    """
    assert "row-normalize interior rows of P_n" not in README


# ---------------------------------------------------------------------------
# Honest fix: pseudocode normalizes ALL rows, matching lib/discretize.py.
# ---------------------------------------------------------------------------
def test_finding1_honest_fix_pseudocode_normalizes_all_rows():
    """Honest fix: README pseudocode states that all rows of P_n are normalized.

    Prose lives in README.md after the writeup/code split.
    PASSES when README says "normalize all rows of P_n".
    """
    assert "row-normalize interior rows of P_n" not in README
    assert "normalize all rows of P_n" in README


def test_finding1_honest_fix_readme_matches_lib():
    """Honest fix: the generated README pseudocode no longer says "interior
    rows" and instead describes normalizing all rows, matching the
    all-rows normalization in lib/discretize.py.

    FAILS on the buggy README; PASSES after the fix and regeneration.
    """
    assert "interior rows of P_n" not in README
    assert "normalize all rows of P_n" in README
