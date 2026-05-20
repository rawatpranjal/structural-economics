"""Finding 2: emax-accuracy figure prose says errors are largest at young ages.

That claim holds for the normalized RMSE but not the absolute RMSE the figure
plots. This test guards the figure description string against the unqualified
directional claim.
"""

import sys
from pathlib import Path

import pandas as pd

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TUTORIAL_DIR))

import run  # noqa: E402

DIAG_CSV = TUTORIAL_DIR / "tables" / "emax-diagnostics.csv"


def _accuracy_figure_description() -> str:
    """Pull the description string passed to add_figure for emax-accuracy.png."""
    src = Path(run.__file__).read_text()
    marker = '"figures/emax-accuracy.png"'
    idx = src.index(marker)
    end = src.index('"""', idx) if '"""' in src[idx:idx + 4000] else len(src)
    block = src[idx:idx + 4000]
    start = block.index("description=(")
    return block[start:block.index("),", start)]


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
    """Prose must qualify the directional claim as the normalized metric."""
    figure_description_text = _accuracy_figure_description()
    assert (
        "normalized" in figure_description_text
        or "young ages" not in figure_description_text
    )
