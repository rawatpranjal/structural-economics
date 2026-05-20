"""Faithfulness tests for choice/bayesian-learning/.

Audit: bullshit-detector_bayesian-learning_2026-05-20.md
Finding 1 (DILUTED): Model Setup table advertised only the belief-simulation
horizon (T=50) but the stopping-boundary figure runs on a silent second
horizon (T_stop=30).
"""
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent.parent
README = TUTORIAL_DIR / "README.md"


def test_finding1_violated_invariant_stopping_horizon_disclosed():
    """Violated invariant: README never mentioned the stopping horizon 30.

    PASSED on the buggy README; FAILS once the honest fix surfaces T_stop.
    """
    assert "30" not in README.read_text()


def test_finding1_honest_fix_stopping_horizon_row_present():
    """Honest fix: Model Setup table names the backward-induction horizon 30.

    FAILED on the buggy README; PASSES once the Stopping horizon row exists.
    """
    text = README.read_text()
    assert "Stopping horizon" in text
    assert "| 30 |" in text
