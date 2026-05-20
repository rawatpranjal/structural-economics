"""Faithfulness tests for the adversarial-estimation tutorial.

Covers bullshit-detector Finding 1 (2026-05-20): the README Results prose
claimed "All Monte Carlo numbers are slightly below the asymptotic
prediction", but the neural-net row in tables/standard-errors.csv has a
Monte Carlo standard deviation (2.562) that exceeds the asymptotic value
(2.449). The prose, not the table, was wrong.
"""

from pathlib import Path

import pandas as pd

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
SE_CSV = TUTORIAL_DIR / "tables" / "standard-errors.csv"
RUN_PY = TUTORIAL_DIR / "run.py"

MC_COL = r"Monte Carlo sd $\times \sqrt{n}$"
ASYMP_COL = r"Asymptotic sd $\times \sqrt{n}$"


def _neural_row() -> pd.Series:
    df = pd.read_csv(SE_CSV)
    return df[df["Estimator"] == "Neural net disc."].iloc[0]


def test_neural_mc_sd_exceeds_asymptotic():
    """Violated invariant.

    The data contradicts an "all Monte Carlo numbers below asymptotic"
    claim: the neural-net Monte Carlo sd is strictly above its asymptotic
    value. This test PASSES on the committed data, proving the original
    prose claim was false.
    """
    row = _neural_row()
    mc_sd = float(row[MC_COL])
    asymp_sd = float(row[ASYMP_COL])
    assert mc_sd > asymp_sd


def test_prose_does_not_claim_all_mc_below_asymptotic():
    """Honest-fix pass condition.

    The run.py Results prose must not assert that all Monte Carlo numbers
    are below the asymptotic prediction, because the neural-net row breaks
    that direction. FAILS on the buggy prose, PASSES once the prose is
    corrected to acknowledge the neural-net exception.
    """
    src = RUN_PY.read_text()
    assert "All Monte Carlo numbers are slightly below the asymptotic" not in src
