"""Faithfulness tests for the scalar-optimization-monopoly-pricing tutorial.

Finding from bullshit-detector_scalar-optimization-monopoly-pricing_2026-05-20.md:
the Equations section stated the random-search rate as 1/N while the rest of
the document and the code's figure labels use 1/sqrt(N).

run.py builds figures on import, so claims are tested against README.md text.
"""

from pathlib import Path

TUT = Path(__file__).resolve().parents[1]
README = (TUT / "README.md").read_text()


# --- Finding 1: Equations section rate claim contradicts the rest of the README ---

def test_f1_violated_invariant_equations_says_one_over_n():
    """Buggy: the Equations section says random-search error is 1/N, 'same order as the grid'."""
    assert "scales as $1/N$ in one dimension, the same order as the grid" in README


def test_f1_honest_fix_equations_rate_consistent():
    """Fixed: the 1/N 'same order as the grid' claim is gone and 1/sqrt(N) is used throughout."""
    assert "scales as $1/N$ in one dimension, the same order as the grid" not in README
    assert README.count(r"1/\sqrt{N}") >= 5
