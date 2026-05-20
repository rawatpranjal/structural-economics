"""Faithfulness tests for the kalman-filter tutorial.

Each audit finding gets two tests:
  * test_*_violated_invariant -- passes on the original buggy README.
  * test_*_honest_fix        -- passes only once the README discloses the step.

The README is generated from run.py, so the honest-fix tests assert on the
generated README and the run.py source.
"""

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
README = HERE / "README.md"
RUN_PY = HERE / "run.py"


def readme_text() -> str:
    return README.read_text()


# --- Finding 1: initial covariance P_{0|0} = 0 undisclosed -------------------

def test_f1_violated_invariant_run_py_uses_zero_initial_covariance():
    """The implementation defaults the initial covariance to a zero matrix."""
    src = RUN_PY.read_text()
    assert "np.zeros((state_dim, state_dim), dtype=float)" in src


def test_f1_honest_fix_readme_discloses_initial_covariance():
    """README must disclose the P_{0|0} = 0 initialization choice."""
    text = readme_text()
    assert "P_{0|0}" in text


# --- Finding 2: symmetrization step undocumented -----------------------------

def test_f2_violated_invariant_run_py_symmetrizes_covariance():
    """The covariance update symmetrizes the matrix after each step."""
    src = RUN_PY.read_text()
    assert "0.5 * (cov + cov.T)" in src


def test_f2_honest_fix_readme_documents_symmetrization():
    """README must mention the numerical symmetrization step."""
    text = readme_text()
    assert "symmetr" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
