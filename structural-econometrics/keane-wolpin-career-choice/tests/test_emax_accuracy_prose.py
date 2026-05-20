"""Finding 2: emax-accuracy figure prose says errors are largest at young ages.

That claim holds for the normalized RMSE but not the absolute RMSE the figure
plots. This test guards the figure description in README.md against the
unqualified directional claim.
"""

from pathlib import Path

import pandas as pd

TUTORIAL_DIR = Path(__file__).resolve().parents[1]

DIAG_CSV = TUTORIAL_DIR / "tables" / "emax-diagnostics.csv"
README = (TUTORIAL_DIR / "README.md").read_text()


def test_violated_invariant_absolute_rmse_grows_with_age():
    """Absolute RMSE in the figure grows with age, not with youth.

    The series is not strictly monotone (a small dip at age 18), so the
    invariant is the overall trend: the oldest age has the largest absolute
    RMSE and the youngest the smallest.
    """
    diag_df = pd.read_csv(DIAG_CSV).sort_values("Age")
    rmse = diag_df["Emax RMSE"]
    assert rmse.iloc[-1] == rmse.max()
    assert rmse.iloc[0] == rmse.min()


def test_honest_fix_description_specifies_normalized_metric():
    """README prose must qualify the directional claim as the normalized metric."""
    assert (
        "normalized" in README
        or "young ages" not in README
    )
