"""Finding 1: emax-fit table prose claims later ages have fewer states.

The data artifact shows the opposite: state counts grow with age. This test
guards the prose description in README.md against the inverted claim.
"""

from pathlib import Path

import pandas as pd

TUTORIAL_DIR = Path(__file__).resolve().parents[1]

FIT_CSV = TUTORIAL_DIR / "tables" / "emax-fit.csv"
README = (TUTORIAL_DIR / "README.md").read_text().lower()


def test_violated_invariant_later_ages_have_more_states():
    """Data shows age 29 has strictly more sampled states than age 16."""
    fit_df = pd.read_csv(FIT_CSV)
    age29 = fit_df.loc[fit_df["Age"] == 29, "Sampled states"].values[0]
    age16 = fit_df.loc[fit_df["Age"] == 16, "Sampled states"].values[0]
    assert age29 > age16


def test_honest_fix_description_says_early_ages_fewer():
    """README must say early ages have fewer states, not later ages."""
    assert "early ages" in README and "fewer" in README
    assert "later ages have fewer continuation states" not in README
