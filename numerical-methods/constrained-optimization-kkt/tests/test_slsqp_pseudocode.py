"""Finding 3: the SLSQP pseudocode says "recover mu by complementary slackness
on inactive bounds". The code recover_multipliers actually computes the
binding-bound multiplier (mu_3 = 1.5) from the stationarity equation
mu_j = lam - (a_j - (Bx)_j). Complementary slackness alone only delivers the
trivially-zero multipliers on non-binding bounds. The pseudocode must name
stationarity, matching the Equations section.
"""
from pathlib import Path

TUTORIAL = Path(__file__).resolve().parents[1]
README = TUTORIAL / "README.md"


def _line_after_recover_mu(text: str) -> str:
    """The pseudocode line that starts the mu-recovery description."""
    return text.split("recover mu")[1].split("\n")[0]


def test_violated_invariant_pseudocode_says_complementary_slackness():
    """PASSES on buggy code: the recover-mu pseudocode line names
    complementary slackness, which only covers the zero-multiplier case.

    After the honest fix this test FAILS.
    """
    line = _line_after_recover_mu(README.read_text())
    assert "complementary slackness" in line


def test_honest_fix_pseudocode_names_stationarity():
    """FAILS on buggy code: the recover-mu pseudocode line never mentions
    stationarity.

    After the honest fix this test PASSES.
    """
    line = _line_after_recover_mu(README.read_text())
    assert "stationarity" in line
