"""Faithfulness tests for the minnesota-svar tutorial.

Generated from bullshit-detector_minnesota-svar_2026-05-20.md.

Findings 1 and 2 (DATA DRIFT): the Takeaway quotes a stability radius
(0.88 -> 0.76) and a shrinkage ratio (0.70) that were computed in run.py
but never persisted to any tables/*.csv artifact, so a reader could not
verify them. The honest fix writes tables/stability-metrics.csv.
"""

from pathlib import Path

import pandas as pd

TUTORIAL_DIR = Path(__file__).resolve().parents[1]


# --- Finding 1 + 2: violated-invariant tests ---------------------------------
# These describe the buggy pre-fix state. They FAIL once the honest fix
# (a committed tables/stability-metrics.csv) is in place.

def test_violated_invariant_no_stability_artifact():
    """Pre-fix: no committed CSV stores the Takeaway stability metrics."""
    tables = {f.name for f in (TUTORIAL_DIR / "tables").iterdir()}
    assert "stability-metrics.csv" not in tables


def test_violated_invariant_forecast_rmse_has_no_stability_column():
    """Pre-fix: stability/shrinkage numbers are not anywhere in a table."""
    rmse = pd.read_csv(TUTORIAL_DIR / "tables" / "forecast-rmse.csv")
    cols = " ".join(rmse.columns).lower()
    assert "stability" not in cols and "shrinkage" not in cols


# --- Finding 1 + 2: honest-fix tests -----------------------------------------
# These PASS once tables/stability-metrics.csv is committed.

def test_honest_fix_stability_metrics_csv_exists():
    assert (TUTORIAL_DIR / "tables" / "stability-metrics.csv").is_file()


def test_honest_fix_stability_metrics_csv_holds_three_metrics():
    metrics = pd.read_csv(TUTORIAL_DIR / "tables" / "stability-metrics.csv")
    joined = " ".join(metrics["Metric"].astype(str)).lower()
    assert "stability radius" in joined
    assert "ols" in joined and "bvar" in joined
    assert "norm" in joined
    assert len(metrics) == 3
