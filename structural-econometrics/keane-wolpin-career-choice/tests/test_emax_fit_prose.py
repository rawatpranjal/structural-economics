"""Finding 1: emax-fit table prose claims later ages have fewer states.

The data artifact shows the opposite: state counts grow with age. This test
guards the prose description string in run.py against the inverted claim.
"""

import sys
from pathlib import Path

import pandas as pd

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TUTORIAL_DIR))

import run  # noqa: E402

FIT_CSV = TUTORIAL_DIR / "tables" / "emax-fit.csv"


def _fit_table_description() -> str:
    """Pull the description string passed to add_table for emax-fit.csv."""
    src = Path(run.__file__).read_text()
    marker = '"tables/emax-fit.csv"'
    idx = src.index(marker)
    end = src.index(")", src.index("description=(", idx))
    return src[idx:end]


def test_violated_invariant_later_ages_have_more_states():
    """Data shows age 29 has strictly more sampled states than age 16."""
    fit_df = pd.read_csv(FIT_CSV)
    age29 = fit_df.loc[fit_df["Age"] == 29, "Sampled states"].values[0]
    age16 = fit_df.loc[fit_df["Age"] == 16, "Sampled states"].values[0]
    assert age29 > age16


def test_honest_fix_description_says_early_ages_fewer():
    """Description must say early ages have fewer states, not later ages."""
    description_text = _fit_table_description().lower()
    assert "early ages" in description_text and "fewer" in description_text
    assert "later ages have fewer continuation states" not in description_text
