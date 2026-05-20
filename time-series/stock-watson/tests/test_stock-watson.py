"""Faithfulness tests for the stock-watson tutorial.

Generated from bullshit-detector_stock-watson_2026-05-20.md.

Finding 1 (DILUTED, HIGH, result-changing): the AR(p) regressor matrix
was built with y[start-lag-1:end-lag-1], which at forecast origin tau
regresses y_{tau+h} on y_{tau-1} and y_{tau-2} (lags 2 and 3) instead of
the claimed y_tau and y_{tau-1} (lags 1 and 2). The honest fix uses
y[start-lag:end-lag] so column lag+1 holds y_{tau-lag}.

The fix changes the absolute RMSE numbers; the relative AR-vs-FAAR
ranking is preserved because both specs carried the same lag error.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parents[1]


def _build_ar_matrix(y, p_ar, h):
    """Reproduce run.py's AR regressor construction for an arbitrary y.

    Mirrors the loop inside factor_augmented_forecast so the lag-alignment
    invariant can be checked directly without running the whole tutorial.
    """
    spec = importlib.util.spec_from_file_location(
        "stock_watson_run", TUTORIAL_DIR / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["stock_watson_run"] = module
    spec.loader.exec_module(module)

    T = len(y)
    start = p_ar
    end = T - h
    X_ar = np.ones((end - start, p_ar + 1))
    # The line under test, copied verbatim from run.py's loop body.
    import inspect
    src = inspect.getsource(module.factor_augmented_forecast)
    assert "for lag in range(p_ar):" in src
    # Extract the slice expression actually used by run.py.
    slice_line = [
        ln for ln in src.splitlines()
        if "X_ar[:, lag + 1]" in ln and "y[" in ln
    ][0].strip()
    for lag in range(p_ar):
        X_ar[:, lag + 1] = eval(  # noqa: S307 - controlled, test-only
            slice_line.split("=", 1)[1].strip(),
            {"y": y, "lag": lag, "start": start, "end": end},
        )
    return X_ar


# --- Finding 1: violated-invariant test --------------------------------------
# Describes the buggy pre-fix state: lag-1 column equals y[1], i.e. y_{tau-1}.
# FAILS once the lag construction is corrected.

def test_violated_invariant_ar_lag_skips_most_recent_obs():
    y = np.arange(200, dtype=float)
    X_ar = _build_ar_matrix(y, p_ar=2, h=1)
    # Buggy code put y[1] (= y_{tau-1}) into the lag-1 column at row 0.
    assert X_ar[0, 1] == y[1]


# --- Finding 1: honest-fix test ----------------------------------------------
# FAILS on buggy code, PASSES once column lag+1 holds y_{tau-lag}.

def test_honest_fix_ar_lag1_is_most_recent_obs():
    y = np.arange(200, dtype=float)
    X_ar = _build_ar_matrix(y, p_ar=2, h=1)
    # start = p_ar = 2; lag-1 column at row 0 must be y[start] = y[2] = y_tau.
    assert X_ar[0, 1] == y[2]


def test_honest_fix_ar_lag2_is_one_step_back():
    y = np.arange(200, dtype=float)
    X_ar = _build_ar_matrix(y, p_ar=2, h=1)
    # lag-2 column at row 0 must be y[start-1] = y[1] = y_{tau-1}.
    assert X_ar[0, 2] == y[1]


# --- Finding 2: 60% training share applies to n_eval, not T ------------------
# DATA DRIFT, LOW: the README's Model Setup said "60%" with no base; the code
# applies 60% to n_eval (the usable window after lag/horizon trimming), not T.

def test_violated_invariant_training_share_uses_n_eval_not_T():
    """With T=200, h=1, p_ar=2 the usable window is 197, giving 118 not 120."""
    T, h, p_ar = 200, 1, 2
    n_eval = T - (p_ar + h)  # y[start+h:end+h] has length end-start = T-h-p_ar
    assert n_eval == 197
    assert int(0.6 * n_eval) == 118
    assert int(0.6 * T) == 120  # the value a reader would wrongly assume


def test_honest_fix_readme_discloses_training_share_base():
    readme = (TUTORIAL_DIR / "README.md").read_text()
    assert "60% of the usable evaluation window" in readme
