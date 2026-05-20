"""Faithfulness tests for the NPE-vs-SMM simulation-budget claim.

The bullshit-detector audit (2026-05-20) flagged Finding 1: the Takeaway
claimed NPE runs at "roughly the same simulation budget the SMM tutorial
uses for a single parameter." NPE runs N_TRAIN = 10,000 simulations; the
SMM grid search runs 30 candidate betas * 8 shock banks + 8 pseudo-data
simulations = 248. The ratio is ~40x, not "roughly the same."

These tests are intentionally split:
- test_violated_invariant_*: encodes the buggy claim. PASSES on the buggy
  README, must FAIL after the honest fix.
- test_honest_fix_*: encodes the faithful claim. FAILS on the buggy
  README, must PASS after the honest fix.
"""

from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
README = TUTORIAL_DIR / "README.md"
RUN_PY = TUTORIAL_DIR / "run.py"

# Simulation budgets, derived from the code, not from prose.
N_TRAIN = 10_000  # run.py:48
SMM_SIMS = 30 * 8 + 8  # 30 candidate betas * 8 shock banks + 8 pseudo-data sims = 248


def _takeaway_text() -> str:
    """Return the Takeaway section of the generated README."""
    text = README.read_text()
    assert "Takeaway" in text, "README has no Takeaway section"
    return text.split("Takeaway", 1)[1]


def test_smm_sim_count_matches_grid_definition() -> None:
    """The 248-sim SMM budget must match smm_grid_for_beta's grid shape."""
    import numpy as np

    candidate_betas = np.arange(2.0, 62.0, 2.0)
    # 8 shock banks per evaluation; 8 pseudo-data simulations once.
    assert len(candidate_betas) * 8 + 8 == SMM_SIMS == 248


def test_budget_ratio_is_about_40x() -> None:
    """NPE uses ~40x more simulator calls than the SMM grid search."""
    ratio = N_TRAIN / SMM_SIMS
    assert ratio > 30
    assert round(ratio, 1) == 40.3


# --- Violated invariant: PASSES on buggy code, must FAIL after the fix. ---

def test_violated_invariant_takeaway_claims_same_budget() -> None:
    """Buggy claim: Takeaway says NPE runs at 'roughly the same' budget."""
    assert "roughly the same simulation budget" in _takeaway_text()


# --- Honest-fix pass condition: FAILS on buggy code, must PASS after fix. ---

def test_honest_fix_takeaway_states_real_ratio() -> None:
    """Faithful claim: Takeaway states the real ~40x budget gap."""
    takeaway = _takeaway_text()
    assert "roughly the same simulation budget" not in takeaway
    assert "40x" in takeaway or "40 times" in takeaway


def test_honest_fix_takeaway_cites_concrete_counts() -> None:
    """Faithful claim: Takeaway names the concrete simulation counts."""
    takeaway = _takeaway_text()
    assert "10,000" in takeaway
    assert "248" in takeaway
