"""Faithfulness tests for the deep-learning optimal-growth tutorial.

Covers the three bullshit-detector findings (2026-05-20):
  Finding 1 (MED): the loss minimized is mean(r^2) + 1e-3 * stability guard,
    but Equations / pseudocode describe the pure empirical risk Xi_n. The
    stability guard must be disclosed in the README prose / pseudocode.
  Finding 2 (LOW): the README claims the loss "drops by several orders of
    magnitude", but the committed CSV records only the final loss. The
    initial loss must be committed so the claim is grounded.
  Finding 3 (LOW): the README states the neural mean saving share is 0.3420,
    a runtime stochastic value not stored in any committed artifact. It must
    be committed to the summary CSV.

run.py imports JAX and trains a network on import, so prose / code tests use
file reads on run.py source and README.md; data tests read the committed CSV.
"""
from pathlib import Path

import pandas as pd
import pytest

TUT = Path(__file__).resolve().parents[1]
README = TUT / "README.md"
RUN_PY = TUT / "run.py"
CSV = TUT / "tables" / "training-summary.csv"


def readme_text() -> str:
    return README.read_text()


def run_py_text() -> str:
    return RUN_PY.read_text()


def loss_fn_source() -> str:
    """Source of loss_fn, located by header text."""
    text = run_py_text()
    start = text.index("def loss_fn(")
    rest = text[start:]
    end = rest.index("\n@jax.jit")
    return rest[:end]


# --- Finding 1: stability guard disclosure ------------------------------------

def test_loss_fn_contains_stability_guard():
    """Code fact: loss_fn adds a relu stability guard to the Euler-residual
    mean. Anchors the finding; PASSES on the current (correct) code.
    """
    src = loss_fn_source()
    assert "lower_guard" in src and "upper_guard" in src
    assert "1e-3" in src


def test_finding1_violated_invariant_guard_undisclosed():
    """Violated invariant: README never mentions the stability guard.

    PASSES on the buggy README; FAILS once the guard is disclosed in prose
    or pseudocode.
    """
    text = readme_text().lower()
    assert "stability guard" not in text and "lower_guard" not in text


def test_finding1_honest_fix_guard_disclosed():
    """Honest fix: README discloses the stability guard term.

    FAILS on the buggy README; PASSES once the guard is documented.
    """
    text = readme_text().lower()
    assert "stability guard" in text or "lower_guard" in text


# --- Finding 2: initial loss committed ----------------------------------------

def test_finding2_violated_invariant_initial_loss_absent():
    """Violated invariant: the summary CSV has no initial-loss column.

    PASSES on the buggy CSV; FAILS once the column is added.
    """
    assert "Initial loss" not in pd.read_csv(CSV).columns


def test_finding2_honest_fix_initial_loss_committed():
    """Honest fix: the summary CSV records the initial loss, and the ratio
    initial/final confirms the 'several orders of magnitude' claim.

    FAILS on the buggy CSV; PASSES once the column is added.
    """
    df = pd.read_csv(CSV)
    assert "Initial loss" in df.columns
    assert df["Initial loss"].iloc[0] / df["Final loss"].iloc[0] > 100


# --- Finding 3: mean saving share committed -----------------------------------

def test_finding3_violated_invariant_mean_share_absent():
    """Violated invariant: the summary CSV has no mean-saving-share column.

    PASSES on the buggy CSV; FAILS once the column is added.
    """
    assert "Mean saving share" not in pd.read_csv(CSV).columns


def test_finding3_honest_fix_mean_share_committed():
    """Honest fix: the summary CSV records the neural mean saving share, and
    it matches the README claim (near alpha*beta = 0.342).

    FAILS on the buggy CSV; PASSES once the column is added.
    """
    df = pd.read_csv(CSV)
    assert "Mean saving share" in df.columns
    assert abs(df["Mean saving share"].iloc[0] - 0.342) < 0.01


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
