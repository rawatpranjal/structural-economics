"""Faithfulness tests for the Allen-Arkolakis spatial-equilibrium tutorial.

Covers bullshit-detector findings 1 and 2 (2026-05-20 audit):
  - Finding 1 (DATA DRIFT): path_gap was computed at runtime and string-
    formatted into the README with no durable CSV artifact. The honest fix
    archives it to tables/convergence-path-dependence.csv.
  - Finding 2 (DILUTED): the Results prose said "center share" while the
    code uses labor.max() (largest share). The honest fix relabels the
    prose to "largest labor share" for consistency with the code.

Each finding gets a violated-invariant test (passed on the pre-fix code)
and an honest-fix test (failed on the pre-fix code, passes after the fix).
The violated-invariant tests are expected to FAIL after the fix is applied.
"""
import sys
from pathlib import Path

import pytest

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TUTORIAL_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

RUN_SRC = (TUTORIAL_DIR / "run.py").read_text()
README = (TUTORIAL_DIR / "README.md").read_text()
PATH_GAP_CSV = TUTORIAL_DIR / "tables" / "convergence-path-dependence.csv"


# --- Finding 1: path_gap not archived ---------------------------------------

def test_path_gap_csv_absent_violated_invariant():
    """Violated invariant: no durable CSV archives path_gap.

    Passed on the pre-fix code (the number was only string-formatted into
    the README). FAILS after the fix, which writes the CSV.
    """
    assert not PATH_GAP_CSV.exists()


def test_path_gap_archived_to_csv_honest_fix():
    """Honest fix: path_gap is archived to a durable CSV with a path_gap column."""
    import pandas as pd

    assert PATH_GAP_CSV.exists()
    value = float(pd.read_csv(PATH_GAP_CSV)["path_gap"].iloc[0])
    assert value == pytest.approx(0.304, abs=0.05)


# --- Finding 2: "center share" label vs labor.max() code --------------------

def test_results_prose_says_center_share_violated_invariant():
    """Violated invariant: the dispersion Results sentence calls labor.max()
    a "center share".

    Passed on the pre-fix code (the add_results string read
    "dispersion-dominant center share is {disp_eq.labor.max()...}"). FAILS
    after the fix, which relabels that sentence to "largest labor share".
    """
    assert "dispersion-dominant center share" in RUN_SRC


def test_results_prose_label_matches_labor_max_honest_fix():
    """Honest fix: the dispersion-scenario share is labelled "largest labor share".

    The code computes disp_eq.labor.max(), the largest share, not a position-
    indexed center share. The prose must name what the code computes.
    """
    assert "largest labor share" in RUN_SRC
    assert "dispersion-dominant center share" not in README
    assert "dispersion-dominant largest labor share" in README
